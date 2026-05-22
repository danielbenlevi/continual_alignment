"""
Safety-FOREVER: Safety-informed FOREVER with early-layer regularization and token KL loss.

Port of continual_alignment/finetune_safety_forever.py with updated imports.
Extends the base FOREVER method with:
  - Dedicated safety memory buffer (sampled from alignment data, replayed 2× more often)
  - Early-layer regularization boost: higher λ for shallow layers that encode safety
  - Token KL loss anchoring the first N response tokens to the post-alignment distribution
  - T1 safety reference model: preserves post-alignment params as the regularization anchor

Entry point: python our_scripts/scripts_training/finetune_safety_forever.py [args] (or via fire.Fire).
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gc
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import fire
import torch
import torch.nn.functional as F
import transformers
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import set_seed

from our_scripts.scripts_utils.data_helpers import load_advbench_harmful, load_alignment_sft_dataset
from our_scripts.scripts_utils.data_helpers import load_task_dataset
from our_scripts.scripts_utils.ddp_runtime import (
    DDPRuntime,
    ddp_barrier,
    gather_records_sorted_by_idx,
    get_ddp_runtime,
    per_rank_eval_batch_size,
    setup_ddp_device,
    shard_with_global_indices,
)

# Import all shared utilities from the base FOREVER module to avoid duplication.
from our_scripts.scripts_training.finetune_forever import (
    DriftRecorderCallback,
    TauTriggerStopCallback,
    MyTrainer,
    TASK_REGISTRY,
    GENERATION_TASKS,
    _resolve_report_to,
    _normalize_chat_template_mode,
    _should_use_chat_template,
    _prepare_tokenizer_and_model,
    _save_pretrained_main_process,
    _batch_slices,
    _decode_batch_responses,
    _evaluate_safety_batched,
    _evaluate_task_performance_batched,
    _evaluate_generation_task_batched,
    _load_alignment_dataset,
    _load_capability_dataset,
    _parse_performance_tasks_arg,
    _sample_ratio,
    _build_memory_buffer,
    _tokenize_example_builder,
    _eval_suite,
    _fmt_eval_row,
)

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _is_early_layer_param(name: str, early_layer_count: int) -> bool:
    if early_layer_count <= 0:
        return False
    m = _LAYER_RE.search(name)
    if not m:
        return False
    try:
        idx = int(m.group(1))
    except Exception:
        return False
    return idx < early_layer_count


class ReplayTrainer(MyTrainer):
    """Replay loss with optional early-layer regularization boost and first-token KL anchoring."""

    def __init__(
        self,
        beta_t: float,
        theta_star: dict,
        importance: dict,
        kl_theta_star: Optional[dict] = None,
        early_layer_boost_factor: float = 1.0,
        early_layer_count: int = 0,
        use_token_kl_loss: bool = False,
        token_kl_weight: float = 0.0,
        token_kl_first_n: int = 5,
        token_kl_warmup_frac: float = 0.2,
        *args,
        **kwargs,
    ):
        self.beta_t = beta_t
        self.theta_star = theta_star
        self.importance = importance
        self.kl_theta_star = kl_theta_star
        self.early_layer_boost_factor = float(early_layer_boost_factor)
        self.early_layer_count = int(early_layer_count)
        self.use_token_kl_loss = bool(use_token_kl_loss)
        self.token_kl_weight = float(token_kl_weight)
        self.token_kl_first_n = int(token_kl_first_n)
        self.token_kl_warmup_frac = float(token_kl_warmup_frac)
        self.last_base_loss = None
        self.last_loss_reg = None
        self.last_token_kl = 0.0
        self.last_token_kl_weight = 0.0
        super().__init__(*args, **kwargs)

    def _reference_logits(self, model, inputs) -> torch.Tensor:
        trainable_params = dict(model.named_parameters())
        ref_params = self.kl_theta_star if self.kl_theta_star is not None else self.theta_star
        backups: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            try:
                for name, ref_param in ref_params.items():
                    param = trainable_params.get(name)
                    if param is None:
                        continue
                    backups[name] = param.data.detach().clone()
                    ref_cast = ref_param.to(device=param.device, dtype=param.dtype)
                    param.data.copy_(ref_cast)
                ref_out = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    use_cache=False,
                )
                return ref_out.logits.detach()
            finally:
                for name, backup in backups.items():
                    param = trainable_params.get(name)
                    if param is not None:
                        param.data.copy_(backup)

    def _current_token_kl_weight(self) -> float:
        if (not self.use_token_kl_loss) or self.token_kl_weight <= 0.0:
            return 0.0
        if self.token_kl_warmup_frac <= 0.0:
            return self.token_kl_weight
        max_steps = int(getattr(self.state, "max_steps", 0) or 0)
        if max_steps <= 0:
            return self.token_kl_weight
        warmup_steps = max(1, int(round(self.token_kl_warmup_frac * float(max_steps))))
        step = int(getattr(self.state, "global_step", 0)) + 1
        scale = min(1.0, float(step) / float(warmup_steps))
        return self.token_kl_weight * scale

    def _token_kl_loss(self, model, inputs, student_logits: torch.Tensor) -> Optional[torch.Tensor]:
        if (not self.use_token_kl_loss) or self.token_kl_first_n <= 0:
            return None

        labels = inputs["labels"][:, 1:].contiguous()
        student = student_logits[:, :-1, :].contiguous()
        ref_logits = self._reference_logits(model, inputs)
        ref = ref_logits[:, :-1, :].contiguous()

        kl_terms = []
        for b in range(labels.size(0)):
            valid_pos = (labels[b] != -100).nonzero(as_tuple=False).flatten()
            if valid_pos.numel() == 0:
                continue
            pos = valid_pos[: self.token_kl_first_n]
            s_logprob = F.log_softmax(student[b, pos, :], dim=-1)
            r_prob = F.softmax(ref[b, pos, :], dim=-1)
            token_kl = F.kl_div(s_logprob, r_prob, reduction="none").sum(dim=-1)
            kl_terms.append(token_kl)

        if not kl_terms:
            return None
        return torch.cat(kl_terms).mean()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        base_out = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        if isinstance(base_out, tuple):
            base_loss, outputs = base_out
        else:
            base_loss, outputs = base_out, None
        if isinstance(base_loss, torch.Tensor) and base_loss.ndim > 0:
            base_loss = base_loss.mean()

        loss_reg = None
        for name, param in model.named_parameters():
            if name not in self.theta_star:
                continue
            ref_param = self.theta_star[name].to(param.device)
            w = self.importance.get(name, 1.0)
            diff = param - ref_param
            layer_boost = 1.0
            if (
                self.early_layer_boost_factor > 1.0
                and _is_early_layer_param(name, self.early_layer_count)
            ):
                layer_boost = self.early_layer_boost_factor
            reg = layer_boost * w * diff.pow(2).sum()
            loss_reg = reg if loss_reg is None else (loss_reg + reg)

        self.last_base_loss = float(base_loss.detach().item())
        self.last_loss_reg = float(loss_reg.detach().item()) if loss_reg is not None else 0.0
        self.last_token_kl = 0.0
        self.last_token_kl_weight = 0.0

        token_kl_term = None
        if outputs is not None and hasattr(outputs, "logits"):
            token_kl = self._token_kl_loss(model, inputs, outputs.logits)
            if token_kl is not None:
                curr_kl_weight = self._current_token_kl_weight()
                self.last_token_kl = float(token_kl.detach().item())
                self.last_token_kl_weight = float(curr_kl_weight)
                if curr_kl_weight > 0.0:
                    token_kl_term = curr_kl_weight * token_kl

        if loss_reg is not None:
            total_loss = base_loss + 0.001 * self.beta_t * loss_reg
        else:
            total_loss = base_loss
        if token_kl_term is not None:
            total_loss = total_loss + token_kl_term

        if return_outputs:
            return total_loss, outputs
        return total_loss


def _snapshot_trainable_params(model) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().clone().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _get_t1_reference_path(output_path: str, model_name: str) -> Path:
    return Path(output_path) / f"{model_name}_clmm_safety_forever" / "t1_safety_reference.pt"


def _save_t1_reference(
    path: Path, theta_star: Dict[str, torch.Tensor], importance: Dict[str, float]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"theta_star": theta_star, "importance": importance}, str(path))


def _load_t1_reference(
    path: Path,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, float]]]:
    if not path.exists():
        return None, None
    payload = torch.load(str(path), map_location="cpu")
    theta_star = payload.get("theta_star")
    importance = payload.get("importance")
    if not isinstance(theta_star, dict) or not isinstance(importance, dict):
        return None, None
    return theta_star, importance


def _run_task_training(
    *,
    runtime: DDPRuntime,
    base_model: str,
    task_id: int,
    task_name: str,
    task_order: Sequence[str],
    train_dataset_raw: Dataset,
    history_train_sets: Sequence[Dataset],
    safety_memory_raw: Optional[Dataset],
    output_path: str,
    memory_data_ratio: int,
    safety_memory_epochs: int,
    memory_epochs: int,
    steps_per_day: int,
    batch_size: int,
    micro_batch_size: int,
    num_epochs: int,
    first_task_epochs: Optional[int],
    learning_rate: float,
    max_input_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Sequence[str],
    train_on_inputs: bool,
    add_eos_token: bool,
    seed: int,
    chat_template_mode: str,
    wandb_project: str,
    wandb_run_name: str,
    resume_from_checkpoint: Optional[str],
    prompt_template_name: str,
    enable_safety_layer_reg_boost: bool,
    safety_layer_reg_boost_factor: float,
    safety_layer_count: int,
    enable_safety_token_kl: bool,
    use_safety_reference_model: bool,
    regularize_at_standard: bool,
    safety_token_weight: float,
    safety_token_count: int,
    safety_token_warmup_frac: float,
):
    if runtime.is_main_process:
        print(f"Training CLMM-SAFETY-FOREVER Task-{task_id}: {task_name}")
    report_to_target = _resolve_report_to(wandb_project)

    effective_epochs = (
        first_task_epochs if (first_task_epochs is not None and task_id == 0) else num_epochs
    )

    model_name = base_model.split("/")[-1] + "lora"
    output_dir = os.path.join(
        output_path,
        model_name + "_clmm_safety_forever",
        str(task_id) + "-" + task_name,
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"output_dir: {output_dir}")

    resume_this_task = False
    if resume_from_checkpoint:
        ckpt_path = Path(resume_from_checkpoint).resolve()
        out_path = Path(output_dir).resolve()
        resume_this_task = (ckpt_path == out_path) or (out_path in ckpt_path.parents)
        if not resume_this_task and runtime.is_main_process:
            print(
                f"[resume] Ignoring resume_from_checkpoint={resume_from_checkpoint} for task {task_id} "
                f"because it is outside current task dir {output_dir}."
            )

    if not resume_this_task:
        if runtime.is_main_process:
            for stale_log in ("drift_log.csv", "replay_strength_log.csv", "replay_loss_log.csv"):
                stale_path = os.path.join(output_dir, stale_log)
                if os.path.exists(stale_path):
                    os.remove(stale_path)
        ddp_barrier(runtime)

    if task_id == 0:
        lora_weights = ""
    else:
        last_task_name = task_order[task_id - 1]
        lora_weights = os.path.join(
            output_path,
            model_name + "_clmm_safety_forever",
            str(task_id - 1) + "-" + last_task_name,
        )
        if not os.path.exists(lora_weights):
            raise FileNotFoundError(f"Missing previous task adapter: {lora_weights}")

    device_map = "auto"
    world_size = int(runtime.world_size)
    ddp = runtime.is_ddp
    if ddp:
        device_map = {"": int(runtime.local_rank)}

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    _prepare_tokenizer_and_model(tokenizer, model)
    use_chat_template = _should_use_chat_template(base_model, chat_template_mode)
    print(f"[prompting] use_chat_template={use_chat_template} (mode={chat_template_mode})")

    generate_and_tokenize_prompt = _tokenize_example_builder(
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        train_on_inputs=train_on_inputs,
        add_eos_token=add_eos_token,
        use_chat_template=use_chat_template,
    )

    current_data = train_dataset_raw.shuffle(seed=seed + task_id).map(generate_and_tokenize_prompt)

    memory_data = None
    if task_id > 0:
        memory_data_raw = _build_memory_buffer(
            history_train_sets=history_train_sets,
            memory_data_ratio=memory_data_ratio,
            seed=seed + task_id * 17,
        )
        if memory_data_raw is not None and len(memory_data_raw) > 0:
            memory_data = memory_data_raw.shuffle(seed=seed + task_id).map(
                generate_and_tokenize_prompt
            )
            print(f"Memory Data loaded: {len(memory_data)} samples.")

    safety_memory_data = None
    if task_id > 0 and safety_memory_raw is not None and len(safety_memory_raw) > 0:
        safety_memory_data = safety_memory_raw.shuffle(seed=seed + 4200 + task_id).map(
            generate_and_tokenize_prompt
        )
        print(f"Safety Memory Data loaded: {len(safety_memory_data)} samples.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=list(lora_target_modules),
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if task_id == 0:
        model = get_peft_model(model, config)
        print("Fine tune lora from scratch!")
    else:
        model = PeftModel.from_pretrained(model, lora_weights, is_trainable=True)
        print("Continual fine tune lora!")

    if resume_this_task:
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)

    if runtime.is_main_process:
        model.print_trainable_parameters()

    theta_star = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    importance = {name: 1.0 for name, _ in model.named_parameters() if _.requires_grad}

    t1_reference_path = _get_t1_reference_path(output_path, model_name)
    t1_theta_star = None
    t1_importance = None
    if task_id > 0 and use_safety_reference_model and (
        enable_safety_layer_reg_boost or enable_safety_token_kl
    ):
        t1_theta_star, t1_importance = _load_t1_reference(t1_reference_path)
        if t1_theta_star is None or t1_importance is None:
            print(
                f"[safety_ref] Missing/invalid T1 reference at {t1_reference_path}; "
                f"falling back to per-task reference."
            )
            t1_theta_star, t1_importance = None, None
        else:
            print(f"[safety_ref] Loaded T1 safety reference from {t1_reference_path}")

    per_device_train_batch_size = micro_batch_size
    gradient_accumulation_steps = batch_size // micro_batch_size
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    gradient_accumulation_steps = max(1, gradient_accumulation_steps)

    num_samples = len(current_data)
    if num_samples == 0:
        print("No samples in current_data, skip task.")
        _save_pretrained_main_process(model, output_dir, runtime)
        return model, tokenizer, output_dir

    total_steps_current = math.ceil(num_samples / batch_size * effective_epochs)
    print(f"Total estimated steps for Current Task: {total_steps_current}")
    current_warmup_budget = max(1, int(round(0.05 * float(max(1, total_steps_current)))))
    current_steps_done = 0

    def _phase_warmup_steps(phase_steps: int) -> int:
        remaining = max(0, current_warmup_budget - current_steps_done)
        if remaining <= 0:
            return 0
        return int(min(max(1, int(phase_steps)), remaining))

    drift_recorder = DriftRecorderCallback(
        output_dir,
        resume=resume_this_task,
        write_logs=runtime.is_main_process,
    )

    replay_strength_log_path = os.path.join(output_dir, "replay_strength_log.csv")
    replay_loss_log_path = os.path.join(output_dir, "replay_loss_log.csv")
    replay_strength_header_written = os.path.isfile(replay_strength_log_path) and os.path.getsize(replay_strength_log_path) > 0
    replay_loss_header_written = os.path.isfile(replay_loss_log_path) and os.path.getsize(replay_loss_log_path) > 0

    steps_per_day = max(1, int(steps_per_day))
    trigger_days_human = [1, 2, 4, 7, 15, 30, 60, 90, 120]

    calibrated = False
    model_day = None
    standard_trigger_thresholds: List[float] = []
    safety_trigger_thresholds: List[float] = []
    next_standard_trigger_idx = 0
    next_safety_trigger_idx = 0
    phase_idx = 0
    steps_done = drift_recorder.step

    ema_alpha = 0.05
    beta_base = 1.0
    beta_scale = 1.0
    beta_min, beta_max = 0.5, 3.0

    print(
        f"[Scheduler] steps_per_day={steps_per_day}, "
        f"effective_epochs={effective_epochs}, total_steps_current={total_steps_current}"
    )

    while steps_done < total_steps_current:
        remaining_steps = total_steps_current - steps_done

        if not calibrated:
            calib_steps = min(steps_per_day, remaining_steps)
            phase_steps_before = int(drift_recorder.step)
            phase_idx += 1
            print(
                f"\n=== Phase {phase_idx}: CALIBRATION CURRENT, "
                f"steps={calib_steps}, steps_done(before)={steps_done}/{total_steps_current} ==="
            )
            phase_warmup = _phase_warmup_steps(calib_steps)

            calib_args = transformers.TrainingArguments(
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=int(phase_warmup),
                lr_scheduler_type="linear",
                max_steps=calib_steps,
                learning_rate=learning_rate,
                weight_decay=0.0,
                bf16=True,
                logging_steps=10,
                optim="adamw_torch",
                save_strategy="steps",
                save_steps=40,
                save_total_limit=2,
                output_dir=output_dir,
                report_to=report_to_target,
                run_name=(
                    f"{wandb_run_name}_phase_{phase_idx}" if (len(wandb_run_name) > 0) else None
                ),
            )

            trainer_calib = MyTrainer(
                model=model,
                train_dataset=current_data,
                eval_dataset=None,
                args=calib_args,
                callbacks=[drift_recorder],
                data_collator=transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                    padding=True,
                ),
            )
            trainer_calib.train()
            current_steps_done += max(0, int(drift_recorder.step) - phase_steps_before)

            steps_done = drift_recorder.step
            current_tau = drift_recorder.tau
            print(
                f"[Scheduler] After CALIBRATION, tau={current_tau:.6f}, "
                f"steps_done={steps_done}/{total_steps_current}"
            )

            model_day = max(current_tau, 1e-8)
            standard_trigger_thresholds = [d * model_day for d in trigger_days_human]
            if safety_memory_data is not None:
                safety_trigger_thresholds = [x / 2.0 for x in standard_trigger_thresholds]
            calibrated = True
            print(
                f"[Scheduler] Calibrated model_day={model_day:.6f}. "
                f"standard_trigger_thresholds={standard_trigger_thresholds}"
            )
            if safety_trigger_thresholds:
                print(f"[Scheduler] safety_trigger_thresholds={safety_trigger_thresholds}")

            if steps_done > 0:
                drift_recorder.mu0 = current_tau / float(steps_done)
                drift_recorder.mu = drift_recorder.mu0
                drift_recorder.ema_alpha = ema_alpha
                print(f"[Scheduler] Initialized mu0={drift_recorder.mu0:.6f}")

            _save_pretrained_main_process(model, output_dir, runtime)
            del trainer_calib
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        def _compute_beta_t() -> Tuple[float, float, float, float, float]:
            mu = drift_recorder.mu if drift_recorder.mu is not None else drift_recorder.mu0
            mu0 = drift_recorder.mu0 if drift_recorder.mu0 is not None else 1.0
            if mu is None:
                mu = mu0
            instability_ratio = mu / (mu0 + 1e-12)
            scale_unclipped = 1.0 + beta_scale * (instability_ratio - 1.0)
            scale = min(max(scale_unclipped, beta_min), beta_max)
            beta_t = beta_base * scale
            return beta_t, scale, float(mu), float(mu0), float(instability_ratio)

        def _run_replay_phase(
            *,
            replay_type: str,
            replay_dataset,
            replay_threshold: Optional[float],
            replay_epochs: int,
            use_safety_heuristics: bool,
        ) -> None:
            nonlocal replay_strength_header_written, replay_loss_header_written

            beta_t, scale, mu, mu0, instability_ratio = _compute_beta_t()
            print(
                f"[Scheduler] Replay({replay_type}) beta_t={beta_t:.6f} "
                f"(mu={mu:.6e}, mu0={mu0:.6e}, r={instability_ratio:.4f})"
            )

            if runtime.is_main_process:
                os.makedirs(output_dir, exist_ok=True)
                with open(replay_strength_log_path, "a", encoding="utf-8") as f:
                    if not replay_strength_header_written:
                        f.write("step,replay_type,threshold,scale,beta_min,beta_max,mu,mu0\n")
                        replay_strength_header_written = True
                    threshold_val = (
                        "" if replay_threshold is None else f"{replay_threshold:.6f}"
                    )
                    f.write(
                        f"{drift_recorder.step},{replay_type},{threshold_val},{scale:.6f},"
                        f"{beta_min:.6f},{beta_max:.6f},{mu:.6e},{mu0:.6e}\n"
                    )

            use_t1_layer_anchor = (
                use_safety_reference_model
                and use_safety_heuristics
                and enable_safety_layer_reg_boost
                and t1_theta_star is not None
                and t1_importance is not None
            )

            if regularize_at_standard:
                use_t1_token_kl = (
                    use_safety_reference_model
                    and enable_safety_token_kl
                    and t1_theta_star is not None
                )
                use_token_kl_loss = bool(use_t1_token_kl)
                early_layer_boost_factor = (
                    safety_layer_reg_boost_factor if enable_safety_layer_reg_boost else 1.0
                )
                early_layer_count = safety_layer_count if enable_safety_layer_reg_boost else 0
            else:
                use_token_kl_loss = bool(use_safety_heuristics and enable_safety_token_kl)
                use_t1_token_kl = (
                    use_safety_reference_model
                    and use_safety_heuristics
                    and enable_safety_token_kl
                    and t1_theta_star is not None
                )
                early_layer_boost_factor = (
                    safety_layer_reg_boost_factor
                    if (use_safety_heuristics and enable_safety_layer_reg_boost)
                    else 1.0
                )
                early_layer_count = safety_layer_count if use_safety_heuristics else 0

            replay_args = transformers.TrainingArguments(
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=5,
                num_train_epochs=replay_epochs,
                learning_rate=learning_rate,
                bf16=True,
                logging_steps=10,
                optim="adamw_torch",
                save_strategy="no",
                output_dir=output_dir,
                report_to=report_to_target,
            )

            trainer_memory = ReplayTrainer(
                beta_t=beta_t,
                theta_star=t1_theta_star if use_t1_layer_anchor else theta_star,
                importance=t1_importance if use_t1_layer_anchor else importance,
                kl_theta_star=t1_theta_star if use_t1_token_kl else None,
                early_layer_boost_factor=early_layer_boost_factor,
                early_layer_count=early_layer_count,
                use_token_kl_loss=use_token_kl_loss,
                token_kl_weight=safety_token_weight,
                token_kl_first_n=safety_token_count,
                token_kl_warmup_frac=safety_token_warmup_frac,
                model=model,
                train_dataset=replay_dataset,
                eval_dataset=None,
                args=replay_args,
                callbacks=[],
                data_collator=transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                    padding=True,
                ),
            )
            trainer_memory.train()
            drift_recorder.update_reference_point(model)

            if runtime.is_main_process:
                with open(replay_loss_log_path, "a", encoding="utf-8") as f:
                    if not replay_loss_header_written:
                        f.write(
                            "step,replay_type,base_loss,loss_reg,beta_t,"
                            "beta_loss_reg,token_kl,token_kl_weight\n"
                        )
                        replay_loss_header_written = True
                    base_loss = float(trainer_memory.last_base_loss)
                    loss_reg = float(trainer_memory.last_loss_reg)
                    beta_loss_reg = beta_t * loss_reg
                    token_kl = float(getattr(trainer_memory, "last_token_kl", 0.0))
                    token_kl_weight = float(getattr(trainer_memory, "last_token_kl_weight", 0.0))
                    f.write(
                        f"{drift_recorder.step},{replay_type},"
                        f"{base_loss:.6f},{loss_reg:.6e},{beta_t:.6f},{beta_loss_reg:.6e},"
                        f"{token_kl:.6e},{token_kl_weight:.6f}\n"
                    )

            del trainer_memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        steps_done = drift_recorder.step
        current_tau = drift_recorder.tau
        print(
            f"\n[Scheduler] Loop start: tau={current_tau:.6f}, "
            f"steps_done={steps_done}/{total_steps_current}, "
            f"next_standard_idx={next_standard_trigger_idx}, "
            f"next_safety_idx={next_safety_trigger_idx}"
        )

        replay_triggered = False
        if (
            memory_data is not None
            and next_standard_trigger_idx < len(standard_trigger_thresholds)
            and current_tau >= standard_trigger_thresholds[next_standard_trigger_idx]
        ):
            phase_idx += 1
            replay_threshold = standard_trigger_thresholds[next_standard_trigger_idx]
            print(
                f"=== Phase {phase_idx}: STANDARD MEMORY REPLAY, "
                f"threshold={replay_threshold:.6f}, tau={current_tau:.6f} ==="
            )
            _run_replay_phase(
                replay_type="standard",
                replay_dataset=memory_data,
                replay_threshold=replay_threshold,
                replay_epochs=memory_epochs,
                use_safety_heuristics=False,
            )
            next_standard_trigger_idx += 1
            replay_triggered = True

        if (
            safety_memory_data is not None
            and next_safety_trigger_idx < len(safety_trigger_thresholds)
            and current_tau >= safety_trigger_thresholds[next_safety_trigger_idx]
        ):
            phase_idx += 1
            replay_threshold = safety_trigger_thresholds[next_safety_trigger_idx]
            print(
                f"=== Phase {phase_idx}: SAFETY MEMORY REPLAY, "
                f"threshold={replay_threshold:.6f}, tau={current_tau:.6f} ==="
            )
            _run_replay_phase(
                replay_type="safety",
                replay_dataset=safety_memory_data,
                replay_threshold=replay_threshold,
                replay_epochs=safety_memory_epochs,
                use_safety_heuristics=True,
            )
            next_safety_trigger_idx += 1
            replay_triggered = True

        if replay_triggered:
            print(
                f"[Scheduler] Replays finished. next_standard_idx={next_standard_trigger_idx}, "
                f"next_safety_idx={next_safety_trigger_idx}, tau={drift_recorder.tau:.6f}"
            )
            continue

        remaining_steps = total_steps_current - steps_done
        if remaining_steps <= 0:
            break

        phase_idx += 1
        print(
            f"=== Phase {phase_idx}: CONTINUE CURRENT, "
            f"remaining_steps={remaining_steps}, "
            f"next_standard_idx={next_standard_trigger_idx}, "
            f"next_safety_idx={next_safety_trigger_idx} ==="
        )

        target_tau = None
        tau_callback = None
        target_candidates: List[float] = []
        if memory_data is not None and next_standard_trigger_idx < len(standard_trigger_thresholds):
            target_candidates.append(standard_trigger_thresholds[next_standard_trigger_idx])
        if safety_memory_data is not None and next_safety_trigger_idx < len(
            safety_trigger_thresholds
        ):
            target_candidates.append(safety_trigger_thresholds[next_safety_trigger_idx])
        if target_candidates:
            target_tau = min(target_candidates)
            tau_callback = TauTriggerStopCallback(drift_recorder, target_tau)
            print(f"[Scheduler] Will watch tau to reach target_tau={target_tau:.6f}")

        current_args = transformers.TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=int(_phase_warmup_steps(remaining_steps)),
            lr_scheduler_type="linear",
            max_steps=remaining_steps,
            learning_rate=learning_rate,
            weight_decay=0.0,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            save_strategy="steps",
            save_steps=40,
            save_total_limit=2,
            output_dir=output_dir,
            report_to=report_to_target,
            run_name=(
                f"{wandb_run_name}_phase_{phase_idx}" if (len(wandb_run_name) > 0) else None
            ),
        )

        callbacks = [drift_recorder]
        if tau_callback is not None:
            callbacks.append(tau_callback)

        trainer_current = MyTrainer(
            model=model,
            train_dataset=current_data,
            eval_dataset=None,
            args=current_args,
            callbacks=callbacks,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True,
            ),
        )
        phase_steps_before = int(drift_recorder.step)
        trainer_current.train()
        current_steps_done += max(0, int(drift_recorder.step) - phase_steps_before)

        steps_done = drift_recorder.step
        current_tau = drift_recorder.tau
        print(
            f"[Scheduler] After CURRENT phase, tau={current_tau:.6f}, "
            f"steps_done={steps_done}/{total_steps_current}"
        )

        _save_pretrained_main_process(model, output_dir, runtime)
        del trainer_current
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll CURRENT phases completed for this task.")

    if memory_data is None and safety_memory_data is None:
        print("[Scheduler] No replay buffer available for final replay.")
    else:
        phase_idx += 1
        if memory_data is not None:
            print(
                f"=== Phase {phase_idx}: FINAL STANDARD MEMORY REPLAY, "
                f"memory_epochs={memory_epochs} ==="
            )
            _run_replay_phase(
                replay_type="standard_final",
                replay_dataset=memory_data,
                replay_threshold=None,
                replay_epochs=memory_epochs,
                use_safety_heuristics=False,
            )

        if safety_memory_data is not None:
            phase_idx += 1
            print(
                f"=== Phase {phase_idx}: FINAL SAFETY MEMORY REPLAY, "
                f"safety_memory_epochs={safety_memory_epochs} ==="
            )
            _run_replay_phase(
                replay_type="safety_final",
                replay_dataset=safety_memory_data,
                replay_threshold=None,
                replay_epochs=safety_memory_epochs,
                use_safety_heuristics=True,
            )

        _save_pretrained_main_process(model, output_dir, runtime)
        print("[Scheduler] Final replay completed and model saved.")

    if (
        task_id == 0
        and use_safety_reference_model
        and (enable_safety_layer_reg_boost or enable_safety_token_kl)
    ):
        if runtime.is_main_process:
            t1_theta = _snapshot_trainable_params(model)
            t1_imp = {name: 1.0 for name in t1_theta}
            _save_t1_reference(t1_reference_path, t1_theta, t1_imp)
            print(f"[safety_ref] Saved T1 safety reference to {t1_reference_path}")
        ddp_barrier(runtime)

    return model, tokenizer, output_dir


def train(
    base_model: str = "Qwen/Qwen3-0.6B",
    output_path: str = "./checkpoints_safety_forever",
    cache_dir: str = "./dataset_cache",
    batch_size: int = 8,
    micro_batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
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
    memory_data_ratio: int = 2,
    safety_memory_ratio: int = 2,
    first_task_epochs: Optional[int] = None,
    memory_epochs: int = 2,
    safety_memory_epochs: int = 2,
    steps_per_day: int = 24,
    enable_safety_layer_reg_boost: bool = False,
    safety_layer_reg_boost_factor: float = 1.5,
    safety_layer_count: int = 4,
    enable_safety_token_kl: bool = False,
    use_safety_reference_model: bool = False,
    regularize_at_standard: bool = False,
    enable_safety_token_reweight: Optional[bool] = None,
    safety_token_weight: float = 1.5,
    safety_token_count: int = 5,
    safety_token_warmup_frac: float = 0.2,
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
):
    runtime = get_ddp_runtime()
    setup_ddp_device(runtime)
    set_seed(int(seed))
    chat_template_mode = _normalize_chat_template_mode(chat_template_mode)
    use_chat_template = _should_use_chat_template(base_model, chat_template_mode)

    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(output_path, exist_ok=True)

    if enable_safety_token_reweight is not None:
        enable_safety_token_kl = bool(enable_safety_token_reweight)
        print("[flags] --enable_safety_token_reweight is deprecated; use --enable_safety_token_kl.")
    if regularize_at_standard and not use_safety_reference_model:
        print("[flags] --regularize_at_standard implies --use_safety_reference_model.")
    use_safety_reference_model = bool(use_safety_reference_model or regularize_at_standard)

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

    safety_memory_raw = _sample_ratio(
        safety_ds,
        ratio_pct=safety_memory_ratio,
        seed=seed + 4242,
        min_samples=1,
    )
    if safety_memory_raw is not None:
        print(
            f"[safety_replay] sampled {len(safety_memory_raw)} safety examples "
            f"({safety_memory_ratio}% of safety task) for replay."
        )
    else:
        print("[safety_replay] safety replay buffer is empty; only standard replay will run.")

    eval_sets: Dict[str, Dataset] = {
        task: _load_capability_dataset(
            task, split=test_split_map[task], n_samples=test_n_map[task]
        )
        for task in perf_order
    }

    if advbench_n is not None and advbench_n <= 0:
        advbench_n = None
    harmful_prompts = load_advbench_harmful(n_samples=advbench_n)

    task_order = ["safety"] + perf_order
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{runtime.local_rank}" if runtime.is_ddp else "cuda")
    else:
        device = torch.device("cpu")

    print(
        f"[sequence] task_order={task_order} | alignment_source={safety_source} "
        f"| advbench_n={len(harmful_prompts)}"
    )

    rows: List[Tuple[str, Dict[str, float], str]] = []

    for task_id, task_name in enumerate(task_order):
        history = [train_sets[t] for t in task_order[:task_id]]

        model, tokenizer, out_dir = _run_task_training(
            runtime=runtime,
            base_model=base_model,
            task_id=task_id,
            task_name=task_name,
            task_order=task_order,
            train_dataset_raw=train_sets[task_name],
            history_train_sets=history,
            safety_memory_raw=safety_memory_raw,
            output_path=output_path,
            memory_data_ratio=memory_data_ratio,
            safety_memory_epochs=safety_memory_epochs,
            memory_epochs=memory_epochs,
            steps_per_day=steps_per_day,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            num_epochs=num_epochs,
            first_task_epochs=first_task_epochs,
            learning_rate=learning_rate,
            max_input_length=max_input_length,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            train_on_inputs=train_on_inputs,
            add_eos_token=add_eos_token,
            seed=seed,
            chat_template_mode=chat_template_mode,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            resume_from_checkpoint=resume_from_checkpoint,
            prompt_template_name=prompt_template_name,
            enable_safety_layer_reg_boost=enable_safety_layer_reg_boost,
            safety_layer_reg_boost_factor=safety_layer_reg_boost_factor,
            safety_layer_count=safety_layer_count,
            enable_safety_token_kl=enable_safety_token_kl,
            use_safety_reference_model=use_safety_reference_model,
            regularize_at_standard=regularize_at_standard,
            safety_token_weight=safety_token_weight,
            safety_token_count=safety_token_count,
            safety_token_warmup_frac=safety_token_warmup_frac,
        )

        row = _eval_suite(
            model=model,
            tokenizer=tokenizer,
            harmful_prompts=harmful_prompts,
            eval_datasets=eval_sets,
            device=device,
            eval_batch_size=eval_batch_size,
            use_chat_template=use_chat_template,
            save_dir=out_dir,
            runtime=runtime,
        )

        if runtime.is_main_process:
            stage_label = f"After T{task_id + 1}_{task_name}"
            rows.append((stage_label, row, out_dir))

        del model, tokenizer
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
            "safety_memory_ratio": safety_memory_ratio,
            "enable_safety_layer_reg_boost": enable_safety_layer_reg_boost,
            "safety_layer_reg_boost_factor": safety_layer_reg_boost_factor,
            "safety_layer_count": safety_layer_count,
            "enable_safety_token_kl": enable_safety_token_kl,
            "use_safety_reference_model": use_safety_reference_model,
            "regularize_at_standard": regularize_at_standard,
            "safety_token_weight": safety_token_weight,
            "safety_token_count": safety_token_count,
            "safety_token_warmup_frac": safety_token_warmup_frac,
            "stages": [
                {"label": label, "checkpoint": ckpt, **metrics}
                for (label, metrics, ckpt) in rows
            ],
        }
        out_path = Path(results_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[sequence] results written to {out_path}")
    ddp_barrier(runtime)


if __name__ == "__main__":
    fire.Fire(train)
