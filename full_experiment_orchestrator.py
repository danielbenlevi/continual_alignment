#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_BASE_MODEL_ALIGN_N = 10000
DEFAULT_WILDJAILBREAK_ALIGN_N = 10000
GLOBAL_TRAIN_EPOCHS = 3
DEFAULT_RUNS_ROOT = Path(__file__).resolve().parent / "orchestrator_runs"


@dataclass
class ExperimentSpec:
    key: str
    script_path: Path
    fire_args: Dict[str, Any]


@dataclass
class JobSpec:
    job_id: str
    experiment_key: str
    script_path: Path
    base_model: str
    model_alias: str
    seed: int
    ordering_id: str
    task_order: List[str]
    output_path: Path
    results_json: Path
    fire_args: Dict[str, Any]


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _sanitize_slug(text: str) -> str:
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("_")
    return out or "value"


def _model_alias(model_name: str) -> str:
    return _sanitize_slug(str(model_name).split("/")[-1])


def _is_base_model_name(model_name: str) -> bool:
    name = str(model_name).strip().lower()
    if not name:
        return False
    if ("instruct" in name) or ("chat" in name):
        return False
    leaf = name.split("/")[-1]
    return (
        leaf.endswith("-base")
        or leaf.endswith("_base")
        or ("-base-" in leaf)
        or ("_base_" in leaf)
        or leaf in {"llama-2-7b", "llama-2-7b-hf"}
    )


def _parse_csv_ints(raw: str) -> List[int]:
    vals = []
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    if not vals:
        raise ValueError("Expected at least one integer value.")
    return vals


def _parse_csv_strs(raw: str) -> List[str]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one value.")
    return vals


def _parse_gpus(raw: str) -> List[str]:
    gpus = _parse_csv_strs(raw)
    if len(gpus) > 8:
        raise ValueError("At most 8 GPUs are supported by this orchestrator.")
    return gpus


def _parse_task_orderings(raw: str) -> List[List[str]]:
    orderings: List[List[str]] = []
    for part in str(raw).split(";"):
        p = part.strip()
        if not p:
            continue
        tasks = [x.strip().lower() for x in p.split(",") if x.strip()]
        if not tasks:
            continue
        if len(tasks) != len(set(tasks)):
            raise ValueError(f"Task ordering has duplicates: {tasks}")
        if tasks[0] != "safety":
            raise ValueError("Each task ordering must start with 'safety'.")
        orderings.append(tasks)
    if not orderings:
        raise ValueError("Expected at least one task ordering.")
    return orderings


def _fire_arg(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"--{key}={'True' if value else 'False'}"
    return f"--{key}={value}"


def _build_experiments(project_root: Path) -> List[ExperimentSpec]:
    training_dir = project_root / "scripts_training"
    forever = training_dir / "finetune_forever.py"
    safety_forever = training_dir / "finetune_safety_forever.py"
    ewcdr = training_dir / "finetune_ewcdr.py"
    safety_ewcdr = training_dir / "finetune_safety_ewcdr.py"
    clora_random = training_dir / "finetune_clora.py"
    clora_safety = training_dir / "finetune_safety_clora.py"
    olora_standard = training_dir / "finetune_olora.py"
    olora_safety = training_dir / "finetune_safety_olora.py"

    for p in (
        forever,
        safety_forever,
        ewcdr,
        safety_ewcdr,
        clora_random,
        clora_safety,
        olora_standard,
        olora_safety,
    ):
        if not p.exists():
            raise FileNotFoundError(f"Missing required script: {p}")

    common_data_args: Dict[str, Any] = {
        "num_epochs": int(GLOBAL_TRAIN_EPOCHS),
        "align_n": int(DEFAULT_WILDJAILBREAK_ALIGN_N),
        "alignment_source": "wildjailbreak_chat",
        "gsm8k_train_n": 2000,
        "sst2_train_n": 2000,
        "mbpp_train_n": -1,
        "xsum_train_n": 2000,
        "sciq_train_n": 2000,
        "samsum_train_n": 2000,
        "gsm8k_test_n": 500,
        "sst2_test_n": 500,
        "mbpp_test_n": 500,
        "xsum_test_n": 500,
        "sciq_test_n": 500,
        "samsum_test_n": 500,
        "advbench_n": -1,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    }

    return [
        ExperimentSpec(
            "forever_base",
            forever,
            dict(common_data_args, learning_rate=3e-4),
        ),
        ExperimentSpec(
            "safety_forever_base",
            safety_forever,
            dict(common_data_args, learning_rate=3e-4),
        ),
        ExperimentSpec(
            "safety_forever_v2_kl",
            safety_forever,
            dict(
                common_data_args,
                learning_rate=3e-4,
                enable_safety_token_kl=True,
                use_safety_reference_model=True,
            ),
        ),
        ExperimentSpec(
            "safety_forever_v2_layer_reg",
            safety_forever,
            dict(
                common_data_args,
                learning_rate=3e-4,
                enable_safety_layer_reg_boost=True,
                use_safety_reference_model=True,
            ),
        ),
        ExperimentSpec(
            "ewcdr_base",
            ewcdr,
            dict(common_data_args, learning_rate=1e-4, lamda=10000.0, omegamax=1e-4),
        ),
        ExperimentSpec(
            "ewcdr_safety",
            safety_ewcdr,
            dict(common_data_args, learning_rate=1e-4, lamda=10000.0, omegamax=1e-4),
        ),
        ExperimentSpec(
            "clora_random",
            clora_random,
            dict(common_data_args, learning_rate=1e-3, clora_lambda=1.0, clora_k=256),
        ),
        ExperimentSpec(
            "clora_safety",
            clora_safety,
            dict(common_data_args, learning_rate=1e-3, clora_lambda=1.0, clora_k=256),
        ),
        ExperimentSpec(
            "olora_standard",
            olora_standard,
            dict(common_data_args, learning_rate=1e-3, olora_lambda_1=0.5),
        ),
        ExperimentSpec(
            "olora_safety",
            olora_safety,
            dict(common_data_args, learning_rate=1e-3, olora_lambda_1=0.5, olora_safety_lambda_1=2.5),
        ),
    ]


def _build_jobs(
    experiments: Sequence[ExperimentSpec],
    models: Sequence[str],
    seeds: Sequence[int],
    task_orderings: Sequence[Sequence[str]],
    batch_size: int,
    eval_batch_size: int,
    base_model_align_n: int,
    run_root: Path,
) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    for model in models:
        alias = _model_alias(model)
        world_size = _world_size_for_model(model)
        if world_size <= 0:
            raise ValueError(f"Invalid world_size={world_size} for model={model}")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be > 0")
        if int(eval_batch_size) <= 0:
            raise ValueError("eval_batch_size must be > 0")
        if (int(batch_size) % int(world_size) != 0):
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by world_size ({world_size}) for model={model}."
            )
        effective_eval_batch_size = int(eval_batch_size)
        if effective_eval_batch_size % int(world_size) != 0:
            raise ValueError(
                "effective eval_batch_size "
                f"({effective_eval_batch_size}) must be divisible by world_size ({world_size}) for model={model}."
            )
        effective_batch_size = int(batch_size)
        if effective_batch_size % int(world_size) != 0:
            raise ValueError(
                f"effective batch_size ({effective_batch_size}) must be divisible by world_size ({world_size}) for model={model}."
            )
        micro_batch_size = int(effective_batch_size) // int(world_size)
        for seed in seeds:
            for ordering_idx, task_order in enumerate(task_orderings):
                ordering_tag = f"order_{ordering_idx}"
                perf_order = list(task_order[1:])
                perf_csv = ",".join(perf_order)
                for exp in experiments:
                    seed_tag = f"seed_{seed}"
                    job_id = f"{exp.key}__{alias}__{seed_tag}__{ordering_tag}"
                    output_path = run_root / "artifacts" / exp.key / alias / seed_tag / ordering_tag
                    results_json = (
                        run_root / "results" / exp.key / alias / seed_tag / f"{ordering_tag}.json"
                    )
                    fire_args: Dict[str, Any] = {
                        "base_model": model,
                        "seed": int(seed),
                        "chat_template_mode": "never" if _is_base_model_name(model) else "always",
                        "performance_tasks": perf_csv,
                        "batch_size": int(effective_batch_size),
                        "micro_batch_size": int(micro_batch_size),
                        "eval_batch_size": int(effective_eval_batch_size),
                        "output_path": str(output_path),
                        "results_json": str(results_json),
                        "wandb_run_name": job_id,
                    }
                    if _is_base_model_name(model):
                        fire_args["align_n"] = int(base_model_align_n)
                    fire_args.update(exp.fire_args)

                    jobs.append(
                        JobSpec(
                            job_id=job_id,
                            experiment_key=exp.key,
                            script_path=exp.script_path,
                            base_model=model,
                            model_alias=alias,
                            seed=int(seed),
                            ordering_id=ordering_tag,
                            task_order=list(task_order),
                            output_path=output_path,
                            results_json=results_json,
                            fire_args=fire_args,
                        )
                    )
    return jobs


def _pick_master_port() -> int:
    # Reserve a currently free local TCP port for this job's rendezvous.
    # This avoids collisions when multiple single-GPU jobs run concurrently.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _prepare_env(gpu_id: str, master_port: int) -> Dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["WORLD_SIZE"] = "1"
    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(int(master_port))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _world_size_for_model(base_model: str) -> int:
    model_name = str(base_model).lower()
    if ("llama-2-7b" in model_name) and _is_base_model_name(base_model):
        return 4
    if "llama" in model_name:
        return 2
    if "qwen3-4b-base" in model_name:
        return 2
    return 1


def _launch_job(
    *,
    job: JobSpec,
    gpu_ids: Sequence[str],
    world_size: int,
    run_root: Path,
    repo_root: Path,
    python_exe: str,
) -> Dict[str, Any]:
    job_dir = run_root / "jobs" / job.job_id
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if len(gpu_ids) != int(world_size):
        raise ValueError(
            f"world_size={world_size} must match allocated GPUs={list(gpu_ids)} for job={job.job_id}"
        )

    base_cmd = [str(job.script_path)] + [_fire_arg(k, v) for k, v in job.fire_args.items()]
    master_port = _pick_master_port()
    master_addr = "127.0.0.1"
    if int(world_size) > 1:
        cmd = [
            python_exe,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={int(world_size)}",
            f"--master_addr={master_addr}",
            f"--master_port={int(master_port)}",
        ] + base_cmd
    else:
        cmd = [python_exe] + base_cmd
    env = _prepare_env(",".join(str(g) for g in gpu_ids), master_port)
    env["WORLD_SIZE"] = str(int(world_size))
    env["MASTER_ADDR"] = master_addr
    env.pop("RANK", None)
    env.pop("LOCAL_RANK", None)

    stdout_path = logs_dir / "stdout.log"
    stderr_path = logs_dir / "stderr.log"

    (logs_dir / "command.txt").write_text(shlex.join(cmd) + "\n", encoding="utf-8")
    _write_json(
        logs_dir / "launch_env.json",
        {
            "gpu_ids": [str(g) for g in gpu_ids],
            "world_size": int(world_size),
            "cwd": str(repo_root),
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "WORLD_SIZE": env.get("WORLD_SIZE"),
            "RANK": env.get("RANK"),
            "LOCAL_RANK": env.get("LOCAL_RANK"),
            "MASTER_ADDR": env.get("MASTER_ADDR"),
            "MASTER_PORT": env.get("MASTER_PORT"),
        },
    )

    out_f = stdout_path.open("w", encoding="utf-8")
    err_f = stderr_path.open("w", encoding="utf-8")

    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=out_f,
        stderr=err_f,
        text=True,
    )

    _write_json(
        job_dir / "job_spec.json",
        {
            "job_id": job.job_id,
            "experiment_key": job.experiment_key,
            "script_path": str(job.script_path),
            "base_model": job.base_model,
            "model_alias": job.model_alias,
            "seed": int(job.seed),
            "ordering_id": job.ordering_id,
            "task_order": job.task_order,
            "output_path": str(job.output_path),
            "results_json": str(job.results_json),
            "gpu_ids": [str(g) for g in gpu_ids],
            "world_size": int(world_size),
            "master_addr": env.get("MASTER_ADDR"),
            "master_port": env.get("MASTER_PORT"),
            "pid": int(proc.pid),
            "command": cmd,
        },
    )

    return {
        "job": job,
        "gpu_ids": [str(g) for g in gpu_ids],
        "world_size": int(world_size),
        "proc": proc,
        "pid": int(proc.pid),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "_stdout_handle": out_f,
        "_stderr_handle": err_f,
        "start_time": time.time(),
    }


def _resolve_python(python_value: str) -> str:
    if os.path.isabs(python_value):
        if not Path(python_value).exists():
            raise FileNotFoundError(f"Python executable not found: {python_value}")
        return python_value
    found = shutil.which(python_value)
    if not found:
        raise FileNotFoundError(f"Could not resolve python executable: {python_value}")
    return found


def _job_from_manifest_entry(entry: Dict[str, Any]) -> JobSpec:
    base_model = str(entry["base_model"])
    seed = int(entry["seed"])
    fire_args = dict(entry.get("fire_args", {}))
    ordering_id = str(entry.get("ordering_id", "order_0"))
    task_order = entry.get("task_order")
    if not isinstance(task_order, list) or not task_order:
        perf_csv = str(fire_args.get("performance_tasks", "gsm8k,sst2,mbpp,xsum,sciq,samsum"))
        perf_order = [x.strip().lower() for x in perf_csv.split(",") if x.strip()]
        task_order = ["safety"] + perf_order
    script_path = Path(str(entry["script_path"]))
    if not script_path.exists():
        # Backward compatibility for manifests created before training script reorganization.
        legacy_map = {
            "finetune_forever.py": Path(__file__).resolve().parent / "scripts_training" / "finetune_forever.py",
            "finetune_safety_forever.py": Path(__file__).resolve().parent / "scripts_training" / "finetune_safety_forever.py",
            "finetune_ewcdr.py": Path(__file__).resolve().parent / "scripts_training" / "finetune_ewcdr.py",
            "finetune_safety_ewcdr.py": Path(__file__).resolve().parent / "scripts_training" / "finetune_safety_ewcdr.py",
        }
        remapped = legacy_map.get(script_path.name)
        if remapped is not None and remapped.exists():
            script_path = remapped

    return JobSpec(
        job_id=str(entry["job_id"]),
        experiment_key=str(entry["experiment_key"]),
        script_path=script_path,
        base_model=base_model,
        model_alias=_model_alias(base_model),
        seed=seed,
        ordering_id=ordering_id,
        task_order=list(task_order),
        output_path=Path(str(entry["output_path"])),
        results_json=Path(str(entry["results_json"])),
        fire_args=fire_args,
    )


def _load_jobs_from_manifest(manifest_path: Path) -> Tuple[Dict[str, Any], List[JobSpec]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    jobs_raw = manifest.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise ValueError(f"Manifest has no jobs: {manifest_path}")
    jobs = [_job_from_manifest_entry(x) for x in jobs_raw]
    return manifest, jobs


def _is_job_completed(results_json: Path) -> bool:
    if not results_json.exists():
        return False
    try:
        with results_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return False
    stages = payload.get("stages")
    return isinstance(stages, list) and len(stages) > 0


def _load_failed_job_ids(run_root: Path) -> set[str]:
    status_path = run_root / "final_status.json"
    if not status_path.exists():
        return set()
    try:
        with status_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return set()
    failed = payload.get("failures")
    if not isinstance(failed, list):
        return set()
    out: set[str] = set()
    for row in failed:
        if isinstance(row, dict):
            jid = row.get("job_id")
            if jid is not None:
                out.add(str(jid))
    return out


def _safe_remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _cleanup_incomplete_job(run_root: Path, job: JobSpec) -> None:
    _safe_remove_path(job.output_path)
    _safe_remove_path(job.results_json)
    _safe_remove_path(run_root / "jobs" / job.job_id)


def _find_latest_run_root(runs_root: Path, run_name_prefix: str) -> Path:
    prefix = _sanitize_slug(run_name_prefix)
    candidates = [p for p in runs_root.glob(f"{prefix}_*") if p.is_dir() and (p / "manifest.json").exists()]
    if not candidates:
        raise FileNotFoundError(
            f"No prior runs found under {runs_root} matching '{prefix}_*' with a manifest.json."
        )
    # Directory names include sortable timestamps: {prefix}_YYYYmmdd_HHMMSS
    return sorted(candidates, key=lambda p: p.name)[-1]


def _is_port_bind_collision(stderr_path: str) -> bool:
    path = Path(stderr_path)
    if not path.exists():
        return False
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    has_addr_collision = ("EADDRINUSE" in txt) or ("address already in use" in txt)
    has_dist_context = ("DistNetworkError" in txt) or ("TCPStore" in txt)
    return has_addr_collision and has_dist_context


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run full continual-learning experiment suite across available GPUs with queue backfilling."
        )
    )
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU ids, e.g. 0,1,2")
    parser.add_argument(
        "--models",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Comma-separated base model names.",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2,3", help="Comma-separated integer seeds.")
    parser.add_argument(
        "--task-orderings",
        type=str,
        default=(
            "safety,gsm8k,sst2,mbpp,xsum,sciq,samsum;"
            "safety,mbpp,xsum,sst2,sciq,samsum,gsm8k;"
            "safety,sciq,gsm8k,samsum,xsum,mbpp,sst2"
        ),
        help=(
            "Semicolon-separated task orderings. "
            "Each ordering is comma-separated and must start with safety."
        ),
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
        help="Global train batch size passed to all trainers.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=128,
        help="Global eval generation batch size passed to all trainers.",
    )
    parser.add_argument(
        "--base-model-align-n",
        type=int,
        default=DEFAULT_BASE_MODEL_ALIGN_N,
        help="Alignment examples for base checkpoints (e.g., *-Base).",
    )
    parser.add_argument("--python", type=str, default="python", help="Python executable for child jobs.")
    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(DEFAULT_RUNS_ROOT),
        help="Output root folder.",
    )
    parser.add_argument("--run-name-prefix", type=str, default="full_suite", help="Run folder name prefix.")
    parser.add_argument("--poll-seconds", type=float, default=20.0, help="Polling interval in seconds.")
    parser.add_argument(
        "--max-port-retries",
        type=int,
        default=2,
        help="Retries for transient torch TCPStore port-collision failures (EADDRINUSE).",
    )
    parser.add_argument(
        "--complete-last",
        nargs="?",
        const="__LATEST__",
        default=None,
        help=(
            "Resume an in-progress run from an existing run directory. "
            "Provide a path to a run folder containing manifest.json, or pass flag with no value "
            "to use the most recent prior run for --run-name-prefix under --runs-root."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Only write the manifest and print commands.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    repo_root = project_root
    pending: List[JobSpec] = []
    completed: List[Dict[str, Any]] = []

    if args.complete_last is not None:
        if str(args.complete_last) == "__LATEST__":
            runs_root = Path(args.runs_root).resolve()
            run_root = _find_latest_run_root(runs_root, args.run_name_prefix)
        else:
            run_root = Path(str(args.complete_last)).expanduser().resolve()
            if not run_root.exists() or not run_root.is_dir():
                raise FileNotFoundError(f"Run directory not found: {run_root}")
        manifest_path = run_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest.json in run directory: {run_root}")
        manifest, jobs = _load_jobs_from_manifest(manifest_path)

        if args.gpus:
            gpus = _parse_gpus(args.gpus)
        else:
            manifest_gpus = manifest.get("gpus")
            if not isinstance(manifest_gpus, list) or not manifest_gpus:
                raise ValueError(
                    f"Manifest is missing a valid non-empty 'gpus' list: {manifest_path}"
                )
            gpus = _parse_gpus(",".join(str(x) for x in manifest_gpus))

        failed_job_ids = _load_failed_job_ids(run_root)

        for job in jobs:
            if job.job_id in failed_job_ids:
                pending.append(job)
            elif _is_job_completed(job.results_json):
                completed.append(
                    {
                        "job_id": job.job_id,
                        "experiment_key": job.experiment_key,
                        "base_model": job.base_model,
                        "seed": int(job.seed),
                        "ordering_id": job.ordering_id,
                        "task_order": job.task_order,
                        "gpu_id": None,
                        "pid": None,
                        "returncode": 0,
                        "elapsed_sec": None,
                        "stdout": None,
                        "stderr": None,
                        "output_path": str(job.output_path),
                        "results_json": str(job.results_json),
                        "preexisting_completed": True,
                    }
                )
            else:
                pending.append(job)

        if not args.dry_run:
            for job in pending:
                _cleanup_incomplete_job(run_root, job)

        print(
            f"[orchestrator] resuming run_root={run_root} | total={len(jobs)} "
            f"already_completed={len(completed)} pending={len(pending)} "
            f"known_failed_prior={len(failed_job_ids)}"
        )
    else:
        gpus = _parse_gpus(args.gpus or "0")
        models = _parse_csv_strs(args.models)
        seeds = _parse_csv_ints(args.seeds)
        task_orderings = _parse_task_orderings(args.task_orderings)
        experiments = _build_experiments(project_root)

        ts = _now_tag()
        run_root = Path(args.runs_root).resolve() / f"{_sanitize_slug(args.run_name_prefix)}_{ts}"
        run_root.mkdir(parents=True, exist_ok=True)

        jobs = _build_jobs(
            experiments=experiments,
            models=models,
            seeds=seeds,
            task_orderings=task_orderings,
            batch_size=int(args.train_batch_size),
            eval_batch_size=int(args.eval_batch_size),
            base_model_align_n=int(args.base_model_align_n),
            run_root=run_root,
        )
        manifest = {
            "timestamp": ts,
            "run_root": str(run_root),
            "repo_root": str(repo_root),
            "gpus": gpus,
            "models": models,
            "seeds": seeds,
            "task_orderings": task_orderings,
            "train_batch_size": int(args.train_batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "base_model_align_n": int(args.base_model_align_n),
            "poll_seconds": float(args.poll_seconds),
            "num_experiments": len(experiments),
            "num_jobs": len(jobs),
            "experiments": [
                {
                    "key": exp.key,
                    "script_path": str(exp.script_path),
                    "fire_args": exp.fire_args,
                }
                for exp in experiments
            ],
            "jobs": [
                {
                    "job_id": job.job_id,
                    "experiment_key": job.experiment_key,
                    "script_path": str(job.script_path),
                    "base_model": job.base_model,
                    "seed": int(job.seed),
                    "ordering_id": job.ordering_id,
                    "task_order": job.task_order,
                    "output_path": str(job.output_path),
                    "results_json": str(job.results_json),
                    "fire_args": job.fire_args,
                }
                for job in jobs
            ],
        }
        _write_json(run_root / "manifest.json", manifest)
        pending = list(jobs)

    python_exe = _resolve_python(args.python)

    print(f"[orchestrator] run_root={run_root}")
    print(f"[orchestrator] python={python_exe}")
    print(f"[orchestrator] jobs={len(jobs)} | gpus={gpus}")

    if args.dry_run:
        for j in pending:
            cmd = [python_exe, str(j.script_path)] + [_fire_arg(k, v) for k, v in j.fire_args.items()]
            print(shlex.join(cmd))
        print("[orchestrator] dry-run complete")
        return 0

    running: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []
    next_gpu_idx = 0
    max_port_retries = max(0, int(args.max_port_retries))
    retry_counts: Dict[str, int] = {job.job_id: 0 for job in jobs}

    def pick_next_free_gpus(required_world_size: int) -> Tuple[Optional[List[str]], int]:
        nonlocal next_gpu_idx
        if required_world_size <= 0 or required_world_size > len(gpus):
            return None, next_gpu_idx

        used_gpus = set()
        for info in running.values():
            for gpu in info.get("gpu_ids", []):
                used_gpus.add(str(gpu))

        for offset in range(len(gpus)):
            idx = (next_gpu_idx + offset) % len(gpus)
            ordered = gpus[idx:] + gpus[:idx]
            free = [gpu for gpu in ordered if gpu not in used_gpus]
            if len(free) >= required_world_size:
                selected = free[:required_world_size]
                selected_last_pos = max(ordered.index(x) for x in selected)
                return selected, (idx + selected_last_pos + 1) % len(gpus)
        return None, next_gpu_idx

    try:
        while pending or running:
            while pending:
                job = pending[0]
                world_size = _world_size_for_model(job.base_model)
                gpu_ids, proposed_idx = pick_next_free_gpus(world_size)
                if gpu_ids is None:
                    break
                next_gpu_idx = proposed_idx
                pending.pop(0)
                info = _launch_job(
                    job=job,
                    gpu_ids=gpu_ids,
                    world_size=world_size,
                    run_root=run_root,
                    repo_root=repo_root,
                    python_exe=python_exe,
                )
                running[job.job_id] = info
                print(
                    f"[orchestrator] launched job={job.job_id} gpus={','.join(gpu_ids)} "
                    f"world_size={world_size} pid={info['pid']} "
                    f"remaining={len(pending)}"
                )

            if not running:
                break

            time.sleep(max(0.5, float(args.poll_seconds)))
            finished_jobs: List[str] = []

            for running_job_id, info in list(running.items()):
                rc = info["proc"].poll()
                if rc is None:
                    continue

                elapsed = time.time() - float(info["start_time"])
                job = info["job"]
                info["_stdout_handle"].close()
                info["_stderr_handle"].close()

                if int(rc) != 0:
                    retry_count = int(retry_counts.get(job.job_id, 0))
                    if retry_count < max_port_retries and _is_port_bind_collision(info["stderr"]):
                        retry_counts[job.job_id] = retry_count + 1
                        _cleanup_incomplete_job(run_root, job)
                        pending.insert(0, job)
                        print(
                            f"[orchestrator] retrying job={job.job_id} after EADDRINUSE "
                            f"(attempt {retry_counts[job.job_id]}/{max_port_retries})"
                        )
                        finished_jobs.append(running_job_id)
                        continue

                result = {
                    "job_id": job.job_id,
                    "experiment_key": job.experiment_key,
                    "base_model": job.base_model,
                    "seed": int(job.seed),
                    "ordering_id": job.ordering_id,
                    "task_order": job.task_order,
                    "gpu_ids": [str(g) for g in info.get("gpu_ids", [])],
                    "world_size": int(info.get("world_size", 1)),
                    "pid": int(info["pid"]),
                    "returncode": int(rc),
                    "elapsed_sec": float(elapsed),
                    "stdout": info["stdout"],
                    "stderr": info["stderr"],
                    "output_path": str(job.output_path),
                    "results_json": str(job.results_json),
                }
                completed.append(result)
                if int(rc) != 0:
                    failures.append(result)
                    print(
                        f"[orchestrator] FAILED job={job.job_id} "
                        f"gpus={','.join(result['gpu_ids'])} rc={rc}"
                    )
                else:
                    print(
                        f"[orchestrator] finished job={job.job_id} "
                        f"gpus={','.join(result['gpu_ids'])} rc=0"
                    )

                finished_jobs.append(running_job_id)

            for running_job_id in finished_jobs:
                running.pop(running_job_id, None)

    except KeyboardInterrupt:
        print("[orchestrator] interrupt received; terminating running jobs")
        for info in running.values():
            info["proc"].terminate()
        for info in running.values():
            try:
                info["proc"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                info["proc"].kill()
            info["_stdout_handle"].close()
            info["_stderr_handle"].close()
        _write_json(
            run_root / "final_status.json",
            {
                "status": "interrupted",
                "completed": completed,
                "failures": failures,
                "remaining_jobs": [job.job_id for job in pending],
            },
        )
        return 130

    final_payload = {
        "status": "completed_with_failures" if failures else "success",
        "total_jobs": len(jobs),
        "completed_jobs": len(completed),
        "failed_jobs": len(failures),
        "completed": completed,
        "failures": failures,
    }
    _write_json(run_root / "final_status.json", final_payload)

    if failures:
        print(f"[orchestrator] done with failures: {len(failures)}/{len(jobs)}")
        return 1

    print(f"[orchestrator] all jobs completed successfully: {len(jobs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
