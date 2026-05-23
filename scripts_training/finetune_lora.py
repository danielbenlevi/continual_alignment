"""
Sequential LoRA baseline: plain LoRA fine-tuning on each task in order,
carrying the adapter forward from one task to the next without any
continual-learning regularization.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gc
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fire
import torch
import torch.distributed as dist
from datasets import Dataset

from scripts_utils.trainer import Trainer, load_model_and_tokenizer
from scripts_utils.data_helpers import load_advbench_harmful
from scripts_utils.ddp_runtime import (
    ddp_barrier,
    get_ddp_runtime,
    setup_ddp_device,
)
from scripts_training.finetune_forever import (
    TASK_REGISTRY,
    _eval_suite,
    _normalize_chat_template_mode,
    _is_base_model_name,
    _load_alignment_dataset,
    _load_capability_dataset,
    _should_use_chat_template,
)


def _maybe_init_process_group(runtime) -> None:
    if runtime.is_ddp and dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)


def train(
    base_model: str = "Qwen/Qwen3-0.6B-Base",
    output_path: str = "./checkpoints",
    results_json: str = "",
    seed: int = 0,
    chat_template_mode: str = "always",
    performance_tasks: str = "gsm8k,sst2,mbpp,xsum,sciq,samsum",
    align_n: int = 10000,
    alignment_source: str = "wildjailbreak_chat",
    batch_size: int = 8,
    eval_batch_size: int = 64,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    max_seq_len: int = 512,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ("q_proj", "v_proj"),
    gsm8k_train_n: int = 2000,
    sst2_train_n: int = 2000,
    mbpp_train_n: Optional[int] = None,
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
    mbpp_eval_mode: str = "pass_at_1",
    mbpp_eval_timeout_sec: float = 3.0,
    wandb_run_name: str = "",
):
    runtime = get_ddp_runtime()
    setup_ddp_device(runtime)
    _maybe_init_process_group(runtime)

    _ = wandb_run_name
    chat_template_mode = _normalize_chat_template_mode(chat_template_mode)
    if _is_base_model_name(base_model):
        chat_template_mode = "never"
    use_chat_template = _should_use_chat_template(base_model, chat_template_mode)

    if isinstance(performance_tasks, str):
        perf_order = [t.strip().lower() for t in performance_tasks.split(",") if t.strip()]
    else:
        perf_order = [str(t).strip().lower() for t in performance_tasks]

    invalid = [t for t in perf_order if t not in TASK_REGISTRY]
    if invalid:
        raise ValueError(f"Unsupported tasks in performance_tasks: {invalid}")

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

    safety_ds, _alignment_source_used = _load_alignment_dataset(
        align_n=int(align_n),
        alignment_source=alignment_source,
        min_alignment_examples=500,
    )
    train_sets: Dict[str, Dataset] = {"safety": safety_ds}
    eval_sets: Dict[str, Dataset] = {}
    for task in perf_order:
        n = train_n_map[task]
        if n is not None and int(n) <= 0:
            n = None
        train_sets[task] = _load_capability_dataset(
            task,
            split="train",
            n_samples=n,
            base_model=base_model,
        )
        eval_sets[task] = _load_capability_dataset(
            task,
            split=test_split_map[task],
            n_samples=test_n_map[task],
            base_model=base_model,
        )

    if advbench_n is not None and int(advbench_n) <= 0:
        advbench_n = None
    harmful_prompts = load_advbench_harmful(n_samples=advbench_n)
    task_order = ["safety"] + perf_order

    out_root = Path(os.path.abspath(os.path.expanduser(output_path)))
    method_root = out_root / f"{base_model.split('/')[-1]}_lora_baseline"
    method_root.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{runtime.local_rank}" if runtime.is_ddp else "cuda")
    else:
        device = torch.device("cpu")
    rows: List[Tuple[str, Dict[str, float], str]] = []

    for task_id, task_name in enumerate(task_order):
        stage_dir = method_root / f"{task_id}-{task_name}"
        if task_id == 0:
            prev_model = base_model
        else:
            prev_model = str(method_root / f"{task_id - 1}-{task_order[task_id - 1]}" / f"epoch_{num_epochs}")

        trainer_cfg = {
            "model_name": prev_model,
            "mode": "lora",
            "rank": int(lora_r),
            "alpha": int(lora_alpha),
            "lora_dropout": float(lora_dropout),
            "lora_target_modules": list(lora_target_modules),
            "lr": float(learning_rate),
            "epochs": int(num_epochs),
            "batch_size": int(batch_size),
            "max_seq_len": int(max_seq_len),
            "seed": int(seed + task_id),
            "use_chat_template": bool(use_chat_template),
        }
        trainer = Trainer(trainer_cfg)
        save_path = trainer.train(
            train_dataset=train_sets[task_name],
            save_dir=str(stage_dir),
        )
        final_ckpt = Path(save_path) / f"epoch_{num_epochs}"
        ddp_barrier(runtime)

        eval_model, eval_tok = load_model_and_tokenizer(str(final_ckpt), device=device)
        metrics = _eval_suite(
            model=eval_model,
            tokenizer=eval_tok,
            harmful_prompts=harmful_prompts,
            eval_datasets=eval_sets,
            device=device,
            eval_batch_size=eval_batch_size,
            use_chat_template=use_chat_template,
            save_dir=str(final_ckpt),
            runtime=runtime,
            mbpp_eval_mode=mbpp_eval_mode,
            mbpp_eval_timeout_sec=mbpp_eval_timeout_sec,
        )
        if runtime.is_main_process:
            rows.append((f"After T{task_id + 1}_{task_name}", metrics, str(final_ckpt)))

        del eval_model, eval_tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ddp_barrier(runtime)

    if results_json and runtime.is_main_process:
        payload = {
            "base_model": base_model,
            "seed": int(seed),
            "mode": "lora_baseline",
            "chat_template_mode": chat_template_mode,
            "use_chat_template": bool(use_chat_template),
            "task_order": task_order,
            "alignment_source": _alignment_source_used,
            "align_n": int(align_n),
            "mbpp_eval_mode": mbpp_eval_mode,
            "mbpp_eval_timeout_sec": float(mbpp_eval_timeout_sec),
            "stages": [
                {
                    "label": label,
                    "checkpoint": ckpt,
                    **m,
                }
                for (label, m, ckpt) in rows
            ],
        }
        out_path = Path(results_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ddp_barrier(runtime)


if __name__ == "__main__":
    fire.Fire(train)
