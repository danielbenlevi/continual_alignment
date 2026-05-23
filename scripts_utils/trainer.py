from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os
import random
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from peft import LoraConfig, get_peft_model, PeftModel

from scripts_utils.clora import apply_clora_to_model, merge_clora_to_base_linear
from scripts_utils.olora import (
    apply_olora_to_model,
    extract_peft_lora_adapters,
    merge_olora_to_base_linear,
    olora_orth_loss_for_model,
)
from scripts_utils.losses import clora_regularization_loss
from scripts_utils.prompt_tokenization import build_tokenize_example_fn
from scripts_utils.ddp_runtime import ddp_barrier, get_ddp_runtime, setup_ddp_device


def _normalize_target_modules(value) -> list[str]:
    if value is None:
        return ["q_proj", "v_proj"]
    if isinstance(value, str):
        mods = [x.strip() for x in value.split(",") if x.strip()]
        return mods or ["q_proj", "v_proj"]
    if isinstance(value, (list, tuple)):
        mods = [str(x).strip() for x in value if str(x).strip()]
        return mods or ["q_proj", "v_proj"]
    return ["q_proj", "v_proj"]


def _resolve_effective_target_module_names(
    model: "torch.nn.Module",
    requested_targets,
) -> list[str]:
    """
    Resolve concrete module names for LoRA-style wrapping.

    For multimodal backbones (e.g., Gemma-3) this avoids attaching adapters to
    vision q/v projections when training text-only data, which would otherwise
    create trainable-but-never-used parameters and break DDP.
    """
    suffixes = _normalize_target_modules(requested_targets)
    matched: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(name.endswith(s) for s in suffixes):
            matched.append(name)

    if not matched:
        return suffixes

    language_only = [n for n in matched if ".language_model." in n]
    if language_only:
        return language_only

    non_vision = [n for n in matched if "vision_tower" not in n and ".vision_model." not in n]
    if non_vision:
        return non_vision

    return matched


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    from transformers import set_seed as _hf_set_seed
    _hf_set_seed(seed)


def _set_hf_scratch_env():
    # Use per-job scratch if provided.
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        hf_home = os.environ.get("HF_HOME", os.path.join(tmpdir, "hf"))
        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
        os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device, *, trainable: bool = False):
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        tok = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, dtype=torch.bfloat16).to(device)
        return model, tok

    tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    marker = ckpt / "CHECKPOINT_TYPE"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == "peft_lora_adapter":
        base_id_file = ckpt / "BASE_MODEL_ID"
        if not base_id_file.exists():
            raise FileNotFoundError(f"Missing BASE_MODEL_ID in {ckpt} for PEFT adapter checkpoint.")
        base_id = base_id_file.read_text(encoding="utf-8").strip()
        base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16).to(device)
        model = PeftModel.from_pretrained(base, ckpt / "adapter", is_trainable=trainable).to(device)
        return model, tok

    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16).to(device)
    return model, tok


def _make_lm_dataset(tokenizer, ds, max_seq_len: int, *, use_chat_template: bool = False):
    tok_fn = build_tokenize_example_fn(
        tokenizer=tokenizer,
        max_input_length=max_seq_len,
        train_on_inputs=False,
        add_eos_token=True,
        use_chat_template=use_chat_template,
    )
    return ds.map(tok_fn, remove_columns=ds.column_names, keep_in_memory=True)


def _total_grad_norm_from_grads(grads: tuple) -> float:
    """L2 norm of concatenated gradient tensors (autograd.grad output)."""
    t = 0.0
    for g in grads:
        if g is not None:
            t += float(g.detach().float().pow(2).sum().item())
    return t ** 0.5


def _count_non_none_grads(params: list[torch.nn.Parameter]) -> int:
    return sum(1 for p in params if p.grad is not None)


def _make_collate_fn(tokenizer):
    """
    Pads variable-length `input_ids`/`attention_mask` and pads `labels` with -100.
    """

    def _collate(features):
        labels = [f.pop("labels") for f in features]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].size(1)
        padded_labels = []
        for lab in labels:
            if len(lab) < max_len:
                lab = lab + ([-100] * (max_len - len(lab)))
            padded_labels.append(lab[:max_len])
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

    return _collate


def _apply_peft_lora(
    model,
    rank: int,
    alpha: int,
    *,
    dropout: float = 0.05,
    target_modules=None,
):
    if isinstance(model, PeftModel):
        return model
    modules = _normalize_target_modules(target_modules)
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=float(dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules,
    )
    model = get_peft_model(model, lora_cfg)
    return model


def _maybe_merge_peft(model):
    # For CLoRA/Safety-CLoRA we want a plain model with real Linear layers.
    if isinstance(model, PeftModel):
        return model.merge_and_unload()
    return model


def _freeze_all_but_clora(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if hasattr(m, "A") and hasattr(m, "B"):
            m.A.requires_grad_(True)
            m.B.requires_grad_(True)


def _freeze_all_but_olora(model):
    """Freeze everything; unfreeze only the current-task A, B in OLoRALinear modules."""
    from scripts_utils.olora import OLoRALinear
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, OLoRALinear):
            m.A.requires_grad_(True)
            m.B.requires_grad_(True)


def _extract_safety_adapters_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    target_module_names: Optional[list[str]] = None,
) -> Dict[str, tuple]:
    """
    Load a Stage-1 PEFT checkpoint and extract per-layer (A, B) adapter weights
    WITHOUT merging them into the base weights.

    Returns {base_layer_name: (A, B)} keyed by the base model's layer name
    (i.e., with 'base_model.model.' prefix stripped).
    """
    model, _ = load_model_and_tokenizer(checkpoint_path, device=device, trainable=False)
    if not isinstance(model, PeftModel):
        raise RuntimeError(
            f"Expected a PeftModel at {checkpoint_path} but got {type(model).__name__}. "
            "O-LoRA requires a PEFT checkpoint so adapter weights can be extracted before merging."
        )
    adapters = extract_peft_lora_adapters(model, target_module_names=target_module_names)
    if not adapters:
        raise RuntimeError(
            f"No LoRA adapter weights found in PEFT checkpoint at {checkpoint_path}. "
            "Check that the checkpoint was saved with q_proj/v_proj as target_modules."
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return adapters


def _save_checkpoint(model, tokenizer, ckpt_dir: Path) -> None:
    """
    Saves a checkpoint directory that downstream code can reload.

    - If `model` is a PEFT LoRA model, we save adapters under `adapter/`
      and write a small marker file so loaders can detect it.
    - If `model` is a plain Transformers model, we save it directly.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(ckpt_dir)

    if isinstance(model, PeftModel):
        adapter_dir = ckpt_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        (ckpt_dir / "CHECKPOINT_TYPE").write_text("peft_lora_adapter\n", encoding="utf-8")
        base_id = getattr(model.base_model.model.config, "_name_or_path", None) or getattr(
            model.base_model.model.config, "name_or_path", "unknown"
        )
        (ckpt_dir / "BASE_MODEL_ID").write_text(f"{base_id}\n", encoding="utf-8")
    else:
        model.save_pretrained(ckpt_dir)
        (ckpt_dir / "CHECKPOINT_TYPE").write_text("full_model\n", encoding="utf-8")


def _save_olora_adapters(model: "torch.nn.Module", ckpt_dir: "Path") -> None:
    """Save raw A_curr, B_curr from each OLoRALinear so sequential stages can load them."""
    from scripts_utils.olora import OLoRALinear
    adapters = {}
    for full_name, module in model.named_modules():
        if isinstance(module, OLoRALinear):
            adapters[full_name] = {
                "A": module.A.detach().cpu(),
                "B": module.B.detach().cpu(),
            }
    torch.save(adapters, ckpt_dir / "olora_adapters.pt")


class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        _set_hf_scratch_env()
        self.runtime = get_ddp_runtime()
        setup_ddp_device(self.runtime)
        if self.runtime.is_ddp and dist.is_available() and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        if torch.cuda.is_available() and self.runtime.is_ddp:
            self.device = torch.device(f"cuda:{int(self.runtime.local_rank)}")
        else:
            self.device = _resolve_device()
        self.is_main_process = bool(self.runtime.is_main_process)
        if "seed" in self.cfg and self.cfg["seed"] is not None:
            _set_seed(int(self.cfg["seed"]))

    def train(
        self,
        train_dataset,
        aligned_model_name: Optional[str] = None,
        base_model_name_for_s: Optional[str] = None,
        extra_prev_adapters: Optional[list] = None,
        save_dir: str = "checkpoints/run",
    ) -> str:
        """
        mode:
          - 'lora'           : task loss only (standard PEFT LoRA)
          - 'clora_random'   : task + lambda*reg  (random S matrix)
          - 'clora_safety'   : task + lambda*reg  (S from alignment direction)
          - 'olora_standard' : task + lam_orth * orth_loss  (uniform lambda for all prev tasks)
          - 'olora_safety'   : task + lam_safety * orth_loss_safety  (asymmetric lambda for safety adapter)
        """
        mode = self.cfg["mode"]
        model_name = self.cfg["model_name"]
        rank = int(self.cfg.get("rank", 8))
        alpha = int(self.cfg.get("alpha", 16))
        lora_dropout = float(self.cfg.get("lora_dropout", 0.05))
        lora_target_modules = _normalize_target_modules(
            self.cfg.get("lora_target_modules", ["q_proj", "v_proj"])
        )
        k = int(self.cfg.get("k", 256))
        lam = float(self.cfg.get("lam", 0.5))
        lr = float(self.cfg.get("lr", 2e-4))
        epochs = int(self.cfg.get("epochs", 1))
        batch_size = int(self.cfg.get("batch_size", 4))
        max_seq_len = int(self.cfg.get("max_seq_len", 512))

        model, tokenizer = load_model_and_tokenizer(model_name, device=self.device, trainable=True)
        # Prefer non-reentrant checkpointing in partially-frozen adapter tuning;
        # it is more robust with DDP + checkpointing.
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        # Required when using gradient checkpointing with mostly-frozen backbones
        # (e.g., adapter tuning). Without this, checkpointed blocks can produce
        # no trainable gradients, which then triggers DDP reduction failures.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        aligned_model = None
        base_model = None
        olora_lam_list_by_name: Dict[str, list] = {}

        if mode == "lora":
            effective_target_module_names = _resolve_effective_target_module_names(model, lora_target_modules)
            if self.is_main_process:
                print(
                    f"[targets] mode=lora requested={lora_target_modules} resolved_count={len(effective_target_module_names)}",
                    flush=True,
                )
            model = _apply_peft_lora(
                model,
                rank=rank,
                alpha=alpha,
                dropout=lora_dropout,
                target_modules=effective_target_module_names,
            )
        elif mode in {"clora_random", "clora_safety"}:
            model = _maybe_merge_peft(model)
            effective_target_module_names = _resolve_effective_target_module_names(model, lora_target_modules)
            if self.is_main_process:
                print(
                    f"[targets] mode={mode} requested={lora_target_modules} resolved_count={len(effective_target_module_names)}",
                    flush=True,
                )
            if mode == "clora_safety":
                if aligned_model_name is None or base_model_name_for_s is None:
                    raise ValueError("clora_safety requires aligned_model_name and base_model_name_for_s")
                # Load reference models on CPU: build_safety_s_matrices moves
                # individual layer tensors to GPU one at a time for SVD, so we never
                # need all three 7B models resident on GPU simultaneously.
                _cpu = torch.device("cpu")
                aligned_model, _tok2 = load_model_and_tokenizer(aligned_model_name, device=_cpu, trainable=False)
                aligned_model = _maybe_merge_peft(aligned_model)
                aligned_model.eval()
                for p in aligned_model.parameters():
                    p.requires_grad_(False)
                base_model = AutoModelForCausalLM.from_pretrained(base_model_name_for_s, dtype=torch.bfloat16)
                base_model.eval()
                for p in base_model.parameters():
                    p.requires_grad_(False)

            model, _clora_mods = apply_clora_to_model(
                model=model,
                rank=rank,
                alpha=alpha,
                k=k,
                lam=lam,
                mode="safety" if mode == "clora_safety" else "random",
                base_model=base_model,
                aligned_model=aligned_model,
                target_module_names=effective_target_module_names,
            )
            _freeze_all_but_clora(model)
        elif mode in {"olora_standard", "olora_safety"}:
            # model_name should be the base model (e.g. Qwen/Qwen3-0.6B).
            # aligned_model_name should be the Stage-1 PEFT checkpoint for adapter extraction.
            if aligned_model_name is None:
                raise ValueError("olora modes require aligned_model_name (Stage-1 PEFT checkpoint path)")

            # model is already loaded above as the Qwen3-0.6B base.
            model = _maybe_merge_peft(model)  # no-op if not PeftModel; keeps base weights clean
            effective_target_module_names = _resolve_effective_target_module_names(model, lora_target_modules)
            if self.is_main_process:
                print(
                    f"[targets] mode={mode} requested={lora_target_modules} resolved_count={len(effective_target_module_names)}",
                    flush=True,
                )

            # Extract safety adapter from Stage-1 PEFT checkpoint (do NOT merge).
            print(f"[olora] extracting safety adapter from {aligned_model_name}", flush=True)
            safety_adapters = _extract_safety_adapters_from_checkpoint(
                aligned_model_name,
                device=self.device,
                target_module_names=effective_target_module_names,
            )
            print(f"[olora] extracted adapters for {len(safety_adapters)} layers", flush=True)

            # Build prev_adapters_by_name: each layer has one prev adapter (the safety one).
            prev_adapters_by_name = {name: [(A, B)] for name, (A, B) in safety_adapters.items()}

            # Lambda list per layer — one value since there is one prev adapter (safety).
            lam_orth = float(self.cfg.get("lam_orth", 0.1))
            if mode == "olora_standard":
                olora_lam_list_by_name = {name: [lam_orth] for name in safety_adapters}
            else:  # olora_safety
                lam_safety = float(self.cfg.get("lam_safety", 1.0))
                olora_lam_list_by_name = {name: [lam_safety] for name in safety_adapters}

            # Append capability adapters from prior sequential stages (each is a
            # {layer_name: (A, B)} dict corresponding to one completed task).
            if extra_prev_adapters:
                for cap_dict in extra_prev_adapters:
                    for name, (A, B) in cap_dict.items():
                        if name in prev_adapters_by_name:
                            prev_adapters_by_name[name].append((A, B))
                            olora_lam_list_by_name[name].append(lam_orth)

            model, _olora_mods = apply_olora_to_model(
                model=model,
                rank=rank,
                alpha=alpha,
                prev_adapters_by_name=prev_adapters_by_name,
                target_module_names=effective_target_module_names,
            )
            _freeze_all_but_olora(model)
        else:
            raise ValueError("mode must be one of: lora, clora_random, clora_safety, olora_standard, olora_safety")

        tokenized = _make_lm_dataset(
            tokenizer,
            train_dataset,
            max_seq_len=max_seq_len,
            use_chat_template=bool(self.cfg.get("use_chat_template", False)),
        )
        world_size = int(self.runtime.world_size)
        if self.runtime.is_ddp:
            if int(batch_size) % world_size != 0:
                raise ValueError(
                    f"batch_size ({batch_size}) must be divisible by world_size ({world_size}) in DDP mode."
                )
            per_rank_batch_size = int(batch_size) // world_size
            sampler = DistributedSampler(
                tokenized,
                num_replicas=world_size,
                rank=int(self.runtime.rank),
                shuffle=True,
                seed=int(self.cfg.get("seed", 0)),
                drop_last=False,
            )
            dl = DataLoader(
                tokenized,
                batch_size=max(1, int(per_rank_batch_size)),
                shuffle=False,
                sampler=sampler,
                collate_fn=_make_collate_fn(tokenizer),
            )
        else:
            sampler = None
            dl = DataLoader(tokenized, batch_size=batch_size, shuffle=True, collate_fn=_make_collate_fn(tokenizer))

        model = model.to(self.device)
        raw_model = model
        train_model = model
        trainable_params = [p for p in raw_model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError(
                f"No trainable parameters found for mode='{mode}'. "
                "Check target module matching and adapter wrapping."
            )
        if self.is_main_process:
            print(f"[train] mode={mode} trainable_params={len(trainable_params)}", flush=True)

        ddp_find_unused_cfg = self.cfg.get("ddp_find_unused_parameters", None)
        if ddp_find_unused_cfg is None:
            ddp_find_unused = mode in {"clora_random", "clora_safety", "olora_standard", "olora_safety"}
        else:
            ddp_find_unused = bool(ddp_find_unused_cfg)

        if self.runtime.is_ddp:
            train_model = DDP(
                model,
                device_ids=[int(self.runtime.local_rank)] if torch.cuda.is_available() else None,
                output_device=int(self.runtime.local_rank) if torch.cuda.is_available() else None,
                find_unused_parameters=ddp_find_unused,
            )
            if self.is_main_process:
                print(f"[ddp] find_unused_parameters={ddp_find_unused}", flush=True)
        opt = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=0.0,
        )
        total_steps = max(1, int(epochs) * max(1, len(dl)))
        warmup_steps = max(1, int(round(0.05 * float(total_steps))))
        scheduler = get_scheduler(
            name="linear",
            optimizer=opt,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        loss_diag_every = int(self.cfg.get("loss_diag_every", 0) or 0)
        global_step = 0

        for epoch in range(1, epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            train_model.train()
            pbar = tqdm(dl, desc=f"epoch {epoch}/{epochs}", leave=True, disable=not self.is_main_process)
            for batch in pbar:
                global_step += 1
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = train_model(**batch)
                task_loss = out.loss

                reg_loss = torch.tensor(0.0, device=self.device)
                orth_loss = torch.tensor(0.0, device=self.device)

                loss = task_loss
                if mode in {"clora_random", "clora_safety"}:
                    reg_loss = clora_regularization_loss(raw_model)
                    loss = loss + lam * reg_loss

                if mode in {"olora_standard", "olora_safety"}:
                    orth_loss = olora_orth_loss_for_model(raw_model, olora_lam_list_by_name)
                    loss = loss + orth_loss

                do_diag = (
                    loss_diag_every > 0
                    and global_step % loss_diag_every == 0
                    and mode in {"clora_random", "clora_safety", "olora_standard", "olora_safety"}
                )
                if do_diag:
                    params = [p for p in raw_model.parameters() if p.requires_grad]
                    opt.zero_grad(set_to_none=True)
                    g_task = torch.autograd.grad(
                        task_loss, params, retain_graph=True, allow_unused=True
                    )
                    gn_task = _total_grad_norm_from_grads(g_task)
                    g_reg = torch.autograd.grad(
                        lam * reg_loss, params, retain_graph=True, allow_unused=True
                    )
                    gn_reg = _total_grad_norm_from_grads(g_reg)
                    if self.is_main_process:
                        print(
                            f"[loss_diag] step={global_step} "
                            f"L_task={float(task_loss):.6f} L_reg={float(reg_loss):.6f} "
                            f"|g_task|={gn_task:.6f} |g_lam_reg|={gn_reg:.6f}",
                            flush=True,
                        )
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if global_step == 1:
                        n_local = _count_non_none_grads(trainable_params)
                        n_min = n_local
                        if self.runtime.is_ddp and dist.is_available() and dist.is_initialized():
                            t = torch.tensor([n_local], device=self.device, dtype=torch.long)
                            dist.all_reduce(t, op=dist.ReduceOp.MIN)
                            n_min = int(t.item())
                        if n_min == 0:
                            raise RuntimeError(
                                "No trainable parameters received gradients on step 1. "
                                "This usually indicates a broken autograd path."
                            )
                    opt.step()
                    scheduler.step()
                else:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if global_step == 1:
                        n_local = _count_non_none_grads(trainable_params)
                        n_min = n_local
                        if self.runtime.is_ddp and dist.is_available() and dist.is_initialized():
                            t = torch.tensor([n_local], device=self.device, dtype=torch.long)
                            dist.all_reduce(t, op=dist.ReduceOp.MIN)
                            n_min = int(t.item())
                        if n_min == 0:
                            raise RuntimeError(
                                "No trainable parameters received gradients on step 1. "
                                "This usually indicates a broken autograd path."
                            )
                    opt.step()
                    scheduler.step()

                pbar.set_postfix(
                    task=float(task_loss.detach().cpu()),
                    reg=float(reg_loss.detach().cpu()),
                    orth=float(orth_loss.detach().cpu()),
                )

            if self.is_main_process and epoch == epochs:
                ckpt_dir = save_path / f"epoch_{epoch}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                # Save only the post-task checkpoint used by downstream stages/eval.
                if mode in {"clora_random", "clora_safety"}:
                    raw_model = merge_clora_to_base_linear(raw_model)
                if mode in {"olora_standard", "olora_safety"}:
                    _save_olora_adapters(raw_model, ckpt_dir)
                    raw_model = merge_olora_to_base_linear(raw_model)
                _save_checkpoint(raw_model, tokenizer, ckpt_dir)
            ddp_barrier(self.runtime)

        return str(save_path)
