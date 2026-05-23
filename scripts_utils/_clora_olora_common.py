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


def _parse_performance_tasks_arg(performance_tasks: object) -> List[str]:
    if isinstance(performance_tasks, str):
        parts = [performance_tasks]
    elif isinstance(performance_tasks, (list, tuple)):
        parts = [str(x) for x in performance_tasks]
    else:
        parts = [str(performance_tasks)]
    tasks: List[str] = []
    for part in parts:
        for token in str(part).split(","):
            t = token.strip().lower()
            if t:
                tasks.append(t)
    return tasks


def _method_key(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode == "clora_random":
        return "clora_random"
    if mode == "clora_safety":
        return "clora_safety"
    if mode == "olora_standard":
        return "olora_standard"
    if mode == "olora_safety":
        return "olora_safety"
    raise ValueError(f"Unsupported mode: {mode}")


def _maybe_init_process_group(runtime) -> None:
    if runtime.is_ddp and dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)


def train(
    mode: str = "clora_random",
    base_model: str = "Qwen/Qwen3-0.6B",
    output_path: str = "./checkpoints",
    results_json: str = "",
    seed: int = 0,
    chat_template_mode: str = "always",
    performance_tasks: str = "gsm8k,sst2,mbpp,xsum,sciq,samsum",
    align_n: int = 10000,
    alignment_source: str = "wildjailbreak_chat",
    batch_size: int = 8,
    micro_batch_size: Optional[int] = None,
    eval_batch_size: int = 64,
    num_epochs: int = 3,
    learning_rate: float = 1e-3,
    max_seq_len: int = 512,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ("q_proj", "v_proj"),
    weight_decay: float = 0.0,
    clora_lambda: float = 1.0,
    clora_k: int = 256,
    olora_lambda_1: float = 0.5,
    olora_safety_lambda_1: float = 2.5,
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

    chat_template_mode = _normalize_chat_template_mode(chat_template_mode)
    if abs(float(weight_decay)) > 0.0:
        print(f"[warn] ignoring non-zero weight_decay={weight_decay}; O-LoRA/CLoRA trainers use weight_decay=0")

    mode_key = _method_key(mode)
    if micro_batch_size is not None and int(micro_batch_size) != int(batch_size):
        print(
            f"[warn] micro_batch_size={micro_batch_size} ignored for CLoRA/O-LoRA; "
            f"using global batch_size={batch_size}."
        )
    _ = wandb_run_name
    if _is_base_model_name(base_model):
        chat_template_mode = "never"
    elif chat_template_mode != "always":
        chat_template_mode = "always"
    use_chat_template = _should_use_chat_template(base_model, chat_template_mode)

    perf_order = _parse_performance_tasks_arg(performance_tasks)
    if len(perf_order) != len(set(perf_order)):
        raise ValueError("performance_tasks contains duplicate tasks")
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
    method_root = out_root / f"{base_model.split('/')[-1]}_{mode_key}"
    method_root.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{runtime.local_rank}" if runtime.is_ddp else "cuda")
    else:
        device = torch.device("cpu")
    rows: List[Tuple[str, Dict[str, float], str]] = []

    stage1_ckpt: Optional[Path] = None
    cap_adapters: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = []

    for task_id, task_name in enumerate(task_order):
        stage_dir = method_root / f"{task_id}-{task_name}"
        final_ckpt = stage_dir / f"epoch_{num_epochs}"
        trainer_cfg = {
            "model_name": base_model if (task_id == 0 or mode_key in {"olora_standard", "olora_safety"}) else str((method_root / f"{task_id - 1}-{task_order[task_id - 1]}" / f"epoch_{num_epochs}")),
            "mode": "lora" if task_id == 0 else mode_key,
            "rank": int(lora_r),
            "alpha": int(lora_alpha),
            "lora_dropout": float(lora_dropout),
            "lora_target_modules": list(lora_target_modules),
            "k": int(clora_k),
            "lam": float(clora_lambda),
            "lam_orth": float(olora_lambda_1),
            "lam_safety": float(olora_safety_lambda_1),
            "lr": float(learning_rate),
            "epochs": int(num_epochs),
            "batch_size": int(batch_size),
            "max_seq_len": int(max_seq_len),
            "seed": int(seed + task_id),
            "use_chat_template": bool(use_chat_template),
        }
        trainer = Trainer(trainer_cfg)

        extra_prev = list(cap_adapters) if (task_id > 0 and mode_key in {"olora_standard", "olora_safety"}) else None
        save_path = trainer.train(
            train_dataset=train_sets[task_name],
            aligned_model_name=(str(stage1_ckpt / f"epoch_{num_epochs}") if (stage1_ckpt is not None and task_id > 0) else None),
            base_model_name_for_s=base_model,
            extra_prev_adapters=extra_prev,
            save_dir=str(stage_dir),
        )
        final_ckpt = Path(save_path) / f"epoch_{num_epochs}"
        if task_id == 0:
            stage1_ckpt = stage_dir
        ddp_barrier(runtime)

        if mode_key in {"olora_standard", "olora_safety"} and task_id > 0:
            ad_path = final_ckpt / "olora_adapters.pt"
            if ad_path.exists():
                data = torch.load(str(ad_path), map_location="cpu", weights_only=False)
                cap_adapters.append({name: (d["A"], d["B"]) for name, d in data.items()})
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
            "mode": mode_key,
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
