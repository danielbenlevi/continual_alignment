"""
EWC-DR: Elastic weight consolidation with diagonal regularization for LLM safety alignment.

Port of continual_alignment/finetune_ewcdr.py with updated imports.
Safety-EWC-DR defaults: safety_task1_upweight=True, safety_lamda_multiplier=0.5.
Entry point: python scripts_training/finetune_ewcdr.py [args] (or via fire.Fire).
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gc
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fire
import torch
import torch.nn.functional as F
import torch.distributed as dist
import transformers
from datasets import Dataset
from peft import get_peft_model, set_peft_model_state_dict
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed

from scripts_utils.data_helpers import load_advbench_harmful
from scripts_utils.ddp_runtime import (
    DDPRuntime,
    ddp_barrier,
    ddp_is_initialized,
    get_ddp_runtime,
    setup_ddp_device,
)
from scripts_training.finetune_forever import (
    TASK_REGISTRY,
    _eval_suite,
    _fmt_eval_row,
    _load_alignment_dataset,
    _load_capability_dataset,
    _normalize_chat_template_mode,
    _parse_performance_tasks_arg,
    _prepare_tokenizer_and_model,
    _resolve_report_to,
    _save_pretrained_main_process,
    _should_use_chat_template,
    _tokenize_example_builder,
)


class EWCTrainer(transformers.Trainer):
    def __init__(
        self,
        lamda: float,
        omega: Optional[Dict[str, torch.Tensor]],
        theta_ref: Optional[Dict[str, torch.Tensor]],
        safety_task1_upweight: bool = False,
        safety_lamda_multiplier: float = 0.5,
        safety_omega: Optional[Dict[str, torch.Tensor]] = None,
        safety_theta_ref: Optional[Dict[str, torch.Tensor]] = None,
        *args,
        **kwargs,
    ):
        self.lamda = float(lamda)
        self.omega = omega
        self.theta_ref = theta_ref
        self.safety_task1_upweight = bool(safety_task1_upweight)
        self.safety_lamda_multiplier = float(safety_lamda_multiplier)
        self.safety_omega = safety_omega
        self.safety_theta_ref = safety_theta_ref
        self.last_base_loss = 0.0
        self.last_ewc_loss = 0.0
        self.last_safety_loss = 0.0
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        base_out = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        if isinstance(base_out, tuple):
            base_loss, outputs = base_out
        else:
            base_loss, outputs = base_out, None

        if isinstance(base_loss, torch.Tensor) and base_loss.ndim > 0:
            base_loss = base_loss.mean()

        ewc_loss = None
        if self.lamda > 0.0 and self.omega is not None and self.theta_ref is not None:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in self.omega or name not in self.theta_ref:
                    continue
                omega_t = self.omega[name].to(device=param.device, dtype=param.dtype)
                ref_t = self.theta_ref[name].to(device=param.device, dtype=param.dtype)
                reg = (omega_t * (param - ref_t).pow(2)).sum() / 2.0
                ewc_loss = reg if ewc_loss is None else (ewc_loss + reg)

        self.last_base_loss = float(base_loss.detach().item())
        self.last_ewc_loss = float(ewc_loss.detach().item()) if ewc_loss is not None else 0.0

        total_loss = base_loss
        if ewc_loss is not None:
            total_loss = total_loss + self.lamda * ewc_loss
        self.last_safety_loss = 0.0
        if (
            self.lamda > 0.0
            and self.safety_task1_upweight
            and self.safety_omega is not None
            and self.safety_theta_ref is not None
        ):
            safety_loss = None
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in self.safety_omega or name not in self.safety_theta_ref:
                    continue
                omega_t = self.safety_omega[name].to(device=param.device, dtype=param.dtype)
                ref_t = self.safety_theta_ref[name].to(device=param.device, dtype=param.dtype)
                reg = (omega_t * (param - ref_t).pow(2)).sum() / 2.0
                safety_loss = reg if safety_loss is None else (safety_loss + reg)
            if safety_loss is not None:
                self.last_safety_loss = float(safety_loss.detach().item())
                total_loss = total_loss + (self.lamda * self.safety_lamda_multiplier * safety_loss)

        if return_outputs:
            return total_loss, outputs
        return total_loss


def _clone_trainable_state(model) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().clone().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _merge_omega(
    old_omega: Dict[str, torch.Tensor],
    new_omega: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    merged = {}
    for name, new_t in new_omega.items():
        if name not in old_omega:
            merged[name] = new_t
            continue
        old_t = old_omega[name]
        merged[name] = alpha * old_t + (1.0 - alpha) * new_t
    return merged


def _first_model_device(model) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_importance_lr(
    model,
    dataloader,
    device: torch.device,
    omegamax: float,
) -> Dict[str, torch.Tensor]:
    omega = {
        name: torch.zeros_like(param.detach(), dtype=torch.float32, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    model_was_training = model.training
    model.train()
    n_batches = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            use_cache=False,
        )
        logits = outputs.logits
        labels = batch["labels"]
        if logits.size(1) < 2:
            continue

        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        loss.backward()

        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None or name not in omega:
                continue
            omega[name] += param.grad.detach().float().pow(2)

        n_batches += 1

    if n_batches <= 0:
        n_batches = 1
    for name in omega:
        omega[name] = (omega[name] / float(n_batches)).clamp(max=float(omegamax)).detach().cpu()

    if not model_was_training:
        model.eval()
    model.zero_grad(set_to_none=True)
    return omega


def _average_omega_across_ranks(omega: Dict[str, torch.Tensor], runtime: DDPRuntime) -> Dict[str, torch.Tensor]:
    if not runtime.is_ddp or not ddp_is_initialized(runtime):
        return omega
    reduce_device = torch.device(f"cuda:{runtime.local_rank}" if torch.cuda.is_available() else "cpu")
    for name, tensor in omega.items():
        t = tensor.to(device=reduce_device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / float(runtime.world_size)
        omega[name] = t.cpu()
    return omega


def _build_tokenized_dataloader(tokenized_dataset, tokenizer, micro_batch_size: int):
    keep_columns = {"input_ids", "attention_mask", "labels"}
    column_names = getattr(tokenized_dataset, "column_names", None)
    if column_names:
        drop_columns = [c for c in column_names if c not in keep_columns]
        if drop_columns:
            tokenized_dataset = tokenized_dataset.remove_columns(drop_columns)

    return torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=max(1, int(micro_batch_size)),
        shuffle=False,
        collate_fn=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
    )


def train(
    base_model: str = "Qwen/Qwen3-0.6B",
    output_path: str = "./checkpoints",
    cache_dir: str = "./dataset_cache",
    batch_size: int = 8,
    micro_batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    max_input_length: int = 512,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ("q_proj", "v_proj"),
    train_on_inputs: bool = False,
    add_eos_token: bool = True,
    seed: int = 42,
    chat_template_mode: str = "always",
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: Optional[str] = None,
    prompt_template_name: str = "alpaca",
    lamda: float = 10000.0,
    omegamax: float = 1e-4,
    first_task_epochs: Optional[int] = None,
    performance_tasks: str = "gsm8k,sst2,mbpp,xsum,sciq,samsum",
    align_n: int = 10000,
    alignment_source: str = "wildjailbreak_chat",
    min_alignment_examples: int = 500,
    gsm8k_train_n: int = 2000,
    sst2_train_n: int = 2000,
    mbpp_train_n: int = -1,
    xsum_train_n: int = 2000,
    sciq_train_n: int = 2000,
    samsum_train_n: int = 2000,
    gsm8k_test_n: int = 500,
    sst2_test_n: int = 500,
    mbpp_test_n: int = 500,
    xsum_test_n: int = 500,
    sciq_test_n: int = 500,
    samsum_test_n: int = 500,
    advbench_n: Optional[int] = None,
    eval_batch_size: int = 128,
    results_json: str = "",
    safety_task1_upweight: bool = False,
    safety_lamda_multiplier: float = 0.5,
):
    runtime = get_ddp_runtime()
    setup_ddp_device(runtime)
    set_seed(int(seed))
    chat_template_mode = _normalize_chat_template_mode(chat_template_mode)
    use_chat_template = _should_use_chat_template(base_model, chat_template_mode)

    report_to_target = _resolve_report_to(wandb_project)

    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(output_path, exist_ok=True)

    perf_order = _parse_performance_tasks_arg(performance_tasks)
    if not perf_order:
        raise ValueError("performance_tasks must contain at least one task.")
    if len(perf_order) != len(set(perf_order)):
        raise ValueError("performance_tasks contains duplicate tasks.")
    invalid = [t for t in perf_order if t not in TASK_REGISTRY]
    if invalid:
        raise ValueError(f"Unsupported tasks in performance_tasks: {invalid}")

    safety_ds, safety_source = _load_alignment_dataset(
        align_n=align_n,
        alignment_source=alignment_source,
        min_alignment_examples=min_alignment_examples,
    )

    train_n_map = {
        "gsm8k": gsm8k_train_n,
        "sst2": sst2_train_n,
        "mbpp": mbpp_train_n,
        "xsum": xsum_train_n,
        "sciq": sciq_train_n,
        "samsum": samsum_train_n,
    }
    test_n_map = {
        "gsm8k": gsm8k_test_n,
        "sst2": sst2_test_n,
        "mbpp": mbpp_test_n,
        "xsum": xsum_test_n,
        "sciq": sciq_test_n,
        "samsum": samsum_test_n,
    }
    test_split_map = {
        "gsm8k": "test",
        "sst2": "validation",
        "mbpp": "test",
        "xsum": "test",
        "sciq": "test",
        "samsum": "test",
    }

    train_sets: Dict[str, Dataset] = {"safety": safety_ds}
    for task in perf_order:
        train_sets[task] = _load_capability_dataset(task, split="train", n_samples=train_n_map[task])

    eval_sets: Dict[str, Dataset] = {
        task: _load_capability_dataset(task, split=test_split_map[task], n_samples=test_n_map[task])
        for task in perf_order
    }

    if advbench_n is not None and advbench_n <= 0:
        advbench_n = None
    harmful_prompts = load_advbench_harmful(n_samples=advbench_n)

    task_order = ["safety"] + perf_order
    if torch.cuda.is_available():
        eval_device = torch.device(f"cuda:{runtime.local_rank}" if runtime.is_ddp else "cuda")
    else:
        eval_device = torch.device("cpu")

    print(
        f"[sequence] task_order={task_order} | alignment_source={safety_source} "
        f"| advbench_n={len(harmful_prompts)}"
    )

    world_size = int(runtime.world_size)
    ddp = runtime.is_ddp
    if ddp:
        device_map = {"": int(runtime.local_rank)}
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    _prepare_tokenizer_and_model(tokenizer, model)
    print(f"[prompting] use_chat_template={use_chat_template} (mode={chat_template_mode})")

    generate_and_tokenize_prompt = _tokenize_example_builder(
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        train_on_inputs=train_on_inputs,
        add_eos_token=add_eos_token,
        use_chat_template=use_chat_template,
    )

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=list(lora_target_modules),
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if ddp and dist.is_initialized():
        ddp_barrier(runtime)
        for tensor in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(tensor.data, src=0)
        ddp_barrier(runtime)

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)

    if runtime.is_main_process:
        model.print_trainable_parameters()

    model_name = base_model.split("/")[-1] + "lora"
    omega: Optional[Dict[str, torch.Tensor]] = None
    theta_ref: Optional[Dict[str, torch.Tensor]] = None
    safety_task1_omega: Optional[Dict[str, torch.Tensor]] = None
    safety_task1_theta_ref: Optional[Dict[str, torch.Tensor]] = None
    seen_examples = 0

    rows: List[Tuple[str, Dict[str, float], str]] = []

    for task_id, task_name in enumerate(task_order):
        output_dir = os.path.join(
            output_path,
            model_name + "_ewcdr",
            str(task_id) + "-" + task_name,
        )
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[task] task_id={task_id}, task_name={task_name}, output_dir={output_dir}")

        current_data_raw = train_sets[task_name]
        current_data = current_data_raw.shuffle(seed=seed + task_id).map(generate_and_tokenize_prompt)
        if len(current_data) == 0:
            print("No samples in current_data, skip task.")
            continue

        effective_epochs = first_task_epochs if (first_task_epochs is not None and task_id == 0) else num_epochs
        grad_accum = batch_size // micro_batch_size
        if ddp:
            grad_accum = grad_accum // world_size
        grad_accum = max(1, grad_accum)
        total_steps_current = math.ceil(len(current_data_raw) / batch_size * effective_epochs)
        warmup_steps = max(1, int(round(0.05 * float(max(1, total_steps_current)))))

        train_args = transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=effective_epochs,
            warmup_steps=int(warmup_steps),
            lr_scheduler_type="linear",
            learning_rate=learning_rate,
            weight_decay=0.0,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            save_strategy="no",
            output_dir=output_dir,
            report_to=report_to_target,
            run_name=f"{wandb_run_name}_task_{task_id}" if (len(wandb_run_name) > 0) else None,
        )

        lamda_eff = float(lamda) if (omega is not None and theta_ref is not None and task_id > 0) else 0.0
        print(f"[ewc] lamda_eff={lamda_eff}")

        trainer = EWCTrainer(
            lamda=lamda_eff,
            omega=omega,
            theta_ref=theta_ref,
            safety_task1_upweight=(safety_task1_upweight and task_id > 0),
            safety_lamda_multiplier=safety_lamda_multiplier,
            safety_omega=safety_task1_omega,
            safety_theta_ref=safety_task1_theta_ref,
            model=model,
            train_dataset=current_data,
            eval_dataset=None,
            args=train_args,
            callbacks=[],
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True,
            ),
        )
        trainer.train()
        _save_pretrained_main_process(model, output_dir, runtime)

        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if ddp and dist.is_initialized():
            ddp_barrier(runtime)
            for tensor in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(tensor.data, src=0)
            ddp_barrier(runtime)

        imp_loader = _build_tokenized_dataloader(current_data, tokenizer, micro_batch_size)
        imp_device = _first_model_device(model)
        omega_new = _compute_importance_lr(model, imp_loader, imp_device, omegamax)
        omega_new = _average_omega_across_ranks(omega_new, runtime)
        print(f"[ewc] computed omega_new over {len(current_data_raw)} examples")

        if omega is None:
            omega = omega_new
        else:
            alpha = float(seen_examples) / float(seen_examples + len(current_data_raw))
            omega = _merge_omega(omega, omega_new, alpha)
            print(f"[ewc] merged omega with alpha={alpha:.6f}")

        seen_examples += len(current_data_raw)
        theta_ref = _clone_trainable_state(model)
        if task_id == 0:
            safety_task1_omega = {k: v.clone() for k, v in omega_new.items()}
            safety_task1_theta_ref = {k: v.clone() for k, v in theta_ref.items()}

        row = _eval_suite(
            model=model,
            tokenizer=tokenizer,
            harmful_prompts=harmful_prompts,
            eval_datasets=eval_sets,
            device=eval_device,
            eval_batch_size=eval_batch_size,
            use_chat_template=use_chat_template,
            save_dir=output_dir,
            runtime=runtime,
        )
        if runtime.is_main_process:
            stage_label = f"After T{task_id + 1}_{task_name}"
            rows.append((stage_label, row, output_dir))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if runtime.is_main_process:
        header_tasks = " | ".join(t.upper() for t in perf_order)
        print(f"\n| Stage | ASR (↓) | {header_tasks} |")
        print(f"|---|---:|{'|'.join('---:' for _ in perf_order)}|")
        for label, r, _ in rows:
            print(_fmt_eval_row(label, r, perf_order))

    if results_json and runtime.is_main_process:
        payload = {
            "base_model": base_model,
            "seed": int(seed),
            "chat_template_mode": chat_template_mode,
            "use_chat_template": bool(use_chat_template),
            "task_order": task_order,
            "alignment_source": safety_source,
            "lamda": lamda,
            "omegamax": omegamax,
            "stages": [
                {
                    "label": label,
                    "checkpoint": ckpt,
                    **metrics,
                }
                for (label, metrics, ckpt) in rows
            ],
        }
        out_path = Path(results_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[sequence] results written to {out_path}")
    ddp_barrier(runtime)


def train_safety(*args, **kwargs):
    """Safety-EWC-DR: EWC-DR with safety task upweighting enabled."""
    kwargs.setdefault("safety_task1_upweight", True)
    kwargs.setdefault("safety_lamda_multiplier", 0.5)
    return train(*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(train)
