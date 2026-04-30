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


def _fire_arg(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"--{key}={'True' if value else 'False'}"
    return f"--{key}={value}"


def _build_experiments(script_dir: Path) -> List[ExperimentSpec]:
    forever = script_dir / "finetune_forever.py"
    safety_forever = script_dir / "finetune_safety_forever.py"
    ewcdr = script_dir / "finetune_ewcdr.py"
    safety_ewcdr = script_dir / "finetune_safety_ewcdr.py"

    for p in (forever, safety_forever, ewcdr, safety_ewcdr):
        if not p.exists():
            raise FileNotFoundError(f"Missing required script: {p}")

    return [
        ExperimentSpec("forever_base", forever, {}),
        ExperimentSpec("safety_forever_base", safety_forever, {}),
        ExperimentSpec(
            "safety_forever_v2_kl",
            safety_forever,
            {"enable_safety_token_kl": True, "use_safety_reference_model": True},
        ),
        ExperimentSpec(
            "safety_forever_v2_layer_reg",
            safety_forever,
            {"enable_safety_layer_reg_boost": True, "use_safety_reference_model": True},
        ),
        ExperimentSpec("ewcdr_base", ewcdr, {}),
        ExperimentSpec("ewcdr_safety", safety_ewcdr, {}),
    ]


def _build_jobs(
    experiments: Sequence[ExperimentSpec],
    models: Sequence[str],
    seeds: Sequence[int],
    run_root: Path,
) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    for model in models:
        alias = _model_alias(model)
        for seed in seeds:
            for exp in experiments:
                seed_tag = f"seed_{seed}"
                job_id = f"{exp.key}__{alias}__{seed_tag}"
                output_path = run_root / "artifacts" / exp.key / alias / seed_tag
                results_json = run_root / "results" / exp.key / alias / f"{seed_tag}.json"

                fire_args: Dict[str, Any] = {
                    "base_model": model,
                    "seed": int(seed),
                    "chat_template_mode": "auto",
                    "output_path": str(output_path),
                    "results_json": str(results_json),
                    "wandb_run_name": job_id,
                }
                fire_args.update(exp.fire_args)

                jobs.append(
                    JobSpec(
                        job_id=job_id,
                        experiment_key=exp.key,
                        script_path=exp.script_path,
                        base_model=model,
                        model_alias=alias,
                        seed=int(seed),
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


def _launch_job(
    *,
    job: JobSpec,
    gpu_id: str,
    run_root: Path,
    repo_root: Path,
    python_exe: str,
) -> Dict[str, Any]:
    job_dir = run_root / "jobs" / job.job_id
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [python_exe, str(job.script_path)] + [_fire_arg(k, v) for k, v in job.fire_args.items()]
    master_port = _pick_master_port()
    env = _prepare_env(gpu_id, master_port)

    stdout_path = logs_dir / "stdout.log"
    stderr_path = logs_dir / "stderr.log"

    (logs_dir / "command.txt").write_text(shlex.join(cmd) + "\n", encoding="utf-8")
    _write_json(
        logs_dir / "launch_env.json",
        {
            "gpu_id": str(gpu_id),
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
            "output_path": str(job.output_path),
            "results_json": str(job.results_json),
            "gpu_id": str(gpu_id),
            "master_addr": env.get("MASTER_ADDR"),
            "master_port": env.get("MASTER_PORT"),
            "pid": int(proc.pid),
            "command": cmd,
        },
    )

    return {
        "job": job,
        "gpu_id": str(gpu_id),
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run full continual-learning experiment suite across available GPUs with queue backfilling."
        )
    )
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids, e.g. 0,1,2")
    parser.add_argument(
        "--models",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Comma-separated base model names.",
    )
    parser.add_argument("--seeds", type=str, default="42,0,1,2,3,4", help="Comma-separated integer seeds.")
    parser.add_argument("--python", type=str, default="python", help="Python executable for child jobs.")
    parser.add_argument("--runs-root", type=str, default="./orchestrator_runs", help="Output root folder.")
    parser.add_argument("--run-name-prefix", type=str, default="full_suite", help="Run folder name prefix.")
    parser.add_argument("--poll-seconds", type=float, default=20.0, help="Polling interval in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Only write the manifest and print commands.")
    args = parser.parse_args()

    gpus = _parse_gpus(args.gpus)
    models = _parse_csv_strs(args.models)
    seeds = _parse_csv_ints(args.seeds)

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    experiments = _build_experiments(script_dir)

    ts = _now_tag()
    run_root = Path(args.runs_root).resolve() / f"{_sanitize_slug(args.run_name_prefix)}_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(experiments=experiments, models=models, seeds=seeds, run_root=run_root)
    manifest = {
        "timestamp": ts,
        "run_root": str(run_root),
        "repo_root": str(repo_root),
        "gpus": gpus,
        "models": models,
        "seeds": seeds,
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
                "output_path": str(job.output_path),
                "results_json": str(job.results_json),
                "fire_args": job.fire_args,
            }
            for job in jobs
        ],
    }
    _write_json(run_root / "manifest.json", manifest)

    python_exe = _resolve_python(args.python)

    print(f"[orchestrator] run_root={run_root}")
    print(f"[orchestrator] python={python_exe}")
    print(f"[orchestrator] jobs={len(jobs)} | gpus={gpus}")

    if args.dry_run:
        for j in jobs:
            cmd = [python_exe, str(j.script_path)] + [_fire_arg(k, v) for k, v in j.fire_args.items()]
            print(shlex.join(cmd))
        print("[orchestrator] dry-run complete")
        return 0

    pending = list(jobs)
    running: Dict[str, Dict[str, Any]] = {}
    completed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    next_gpu_idx = 0

    def pick_next_free_gpu() -> Tuple[Optional[str], int]:
        nonlocal next_gpu_idx
        for offset in range(len(gpus)):
            idx = (next_gpu_idx + offset) % len(gpus)
            gpu = gpus[idx]
            if gpu not in running:
                return gpu, (idx + 1) % len(gpus)
        return None, next_gpu_idx

    try:
        while pending or running:
            while pending:
                gpu, proposed_idx = pick_next_free_gpu()
                if gpu is None:
                    break
                next_gpu_idx = proposed_idx
                job = pending.pop(0)
                info = _launch_job(
                    job=job,
                    gpu_id=gpu,
                    run_root=run_root,
                    repo_root=repo_root,
                    python_exe=python_exe,
                )
                running[gpu] = info
                print(
                    f"[orchestrator] launched job={job.job_id} gpu={gpu} pid={info['pid']} "
                    f"remaining={len(pending)}"
                )

            if not running:
                break

            time.sleep(max(0.5, float(args.poll_seconds)))
            finished_gpus: List[str] = []

            for gpu, info in list(running.items()):
                rc = info["proc"].poll()
                if rc is None:
                    continue

                elapsed = time.time() - float(info["start_time"])
                job = info["job"]
                result = {
                    "job_id": job.job_id,
                    "experiment_key": job.experiment_key,
                    "base_model": job.base_model,
                    "seed": int(job.seed),
                    "gpu_id": str(gpu),
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
                    print(f"[orchestrator] FAILED job={job.job_id} gpu={gpu} rc={rc}")
                else:
                    print(f"[orchestrator] finished job={job.job_id} gpu={gpu} rc=0")

                info["_stdout_handle"].close()
                info["_stderr_handle"].close()
                finished_gpus.append(gpu)

            for gpu in finished_gpus:
                running.pop(gpu, None)

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
