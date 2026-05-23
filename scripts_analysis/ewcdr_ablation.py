#!/usr/bin/env python3
from __future__ import annotations

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
from typing import Any, Dict, List


DEFAULT_LAMBDAS = [15000, 17500, 20000, 22500, 25000, 30000, 35000, 40000]
DEFAULT_GPUS = ["0", "1", "2", "3", "4", "5", "6", "7"]


@dataclass
class JobSpec:
    job_id: str
    lamda: int
    gpu: str
    script_path: Path
    output_path: Path
    results_json: Path
    fire_args: Dict[str, Any]


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_csv_ints(raw: str) -> List[int]:
    vals = []
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    if not vals:
        raise ValueError("Expected at least one integer lambda value.")
    return vals


def _parse_csv_strs(raw: str) -> List[str]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one GPU id.")
    return vals


def _pick_master_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _resolve_python(python_value: str) -> str:
    if os.path.isabs(python_value):
        if not Path(python_value).exists():
            raise FileNotFoundError(f"Python executable not found: {python_value}")
        return python_value
    found = shutil.which(python_value)
    if not found:
        raise FileNotFoundError(f"Could not resolve python executable: {python_value}")
    return found


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fire_arg(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"--{key}={'True' if value else 'False'}"
    return f"--{key}={value}"


def _prepare_env(gpu: str, master_port: int) -> Dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["WORLD_SIZE"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(int(master_port))
    env.pop("RANK", None)
    env.pop("LOCAL_RANK", None)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _is_port_bind_collision(stderr_path: Path) -> bool:
    if not stderr_path.exists():
        return False
    try:
        txt = stderr_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    has_addr_collision = ("EADDRINUSE" in txt) or ("address already in use" in txt)
    has_dist_context = ("DistNetworkError" in txt) or ("TCPStore" in txt)
    return has_addr_collision and has_dist_context


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


def _build_jobs(
    *,
    project_root: Path,
    base_model: str,
    seed: int,
    lambdas: List[int],
    gpus: List[str],
    run_root: Path,
) -> List[JobSpec]:
    if len(lambdas) != len(gpus):
        raise ValueError(
            f"Need same number of lambdas and GPUs, got lambdas={len(lambdas)} gpus={len(gpus)}"
        )

    train_script = project_root / "scripts_training" / "finetune_ewcdr.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Missing training script: {train_script}")

    jobs: List[JobSpec] = []
    for lamda, gpu in zip(lambdas, gpus):
        tag = f"lamda_{int(lamda)}"
        output_path = run_root / "artifacts" / tag
        results_json = run_root / "results" / f"{tag}.json"
        fire_args: Dict[str, Any] = {
            "base_model": base_model,
            "seed": int(seed),
            "lamda": int(lamda),
            "output_path": str(output_path),
            "results_json": str(results_json),
            "wandb_run_name": f"ewcdr_base_qwen3_0.6b_base_{tag}_seed_{seed}",
        }
        jobs.append(
            JobSpec(
                job_id=tag,
                lamda=int(lamda),
                gpu=str(gpu),
                script_path=train_script,
                output_path=output_path,
                results_json=results_json,
                fire_args=fire_args,
            )
        )
    return jobs


def _launch_job(*, job: JobSpec, run_root: Path, repo_root: Path, python_exe: str) -> Dict[str, Any]:
    job_dir = run_root / "jobs" / job.job_id
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [str(job.script_path)] + [_fire_arg(k, v) for k, v in job.fire_args.items()]
    cmd = [python_exe] + base_cmd

    master_port = _pick_master_port()
    env = _prepare_env(job.gpu, master_port)

    stdout_path = logs_dir / "stdout.log"
    stderr_path = logs_dir / "stderr.log"

    (logs_dir / "command.txt").write_text(shlex.join(cmd) + "\n", encoding="utf-8")
    _write_json(
        logs_dir / "launch_env.json",
        {
            "gpu": job.gpu,
            "cwd": str(repo_root),
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "WORLD_SIZE": env.get("WORLD_SIZE"),
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
            "lamda": int(job.lamda),
            "gpu": job.gpu,
            "script_path": str(job.script_path),
            "output_path": str(job.output_path),
            "results_json": str(job.results_json),
            "master_addr": env.get("MASTER_ADDR"),
            "master_port": env.get("MASTER_PORT"),
            "pid": int(proc.pid),
            "command": cmd,
            "fire_args": job.fire_args,
        },
    )

    return {
        "job": job,
        "proc": proc,
        "pid": int(proc.pid),
        "stdout": stdout_path,
        "stderr": stderr_path,
        "_stdout_handle": out_f,
        "_stderr_handle": err_f,
        "start_time": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run EWCDR-base lambda ablation with full-orchestrator-style lifecycle/retry handling."
    )
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lambdas",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LAMBDAS),
        help="Comma-separated lamda values.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=",".join(DEFAULT_GPUS),
        help="Comma-separated GPU ids; must match --lambdas length.",
    )
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "orchestrator_runs"),
    )
    parser.add_argument(
        "--run-name-prefix",
        type=str,
        default="ewcdr_ablation_qwen3_0.6b_base",
    )
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument(
        "--max-port-retries",
        type=int,
        default=2,
        help="Retries for transient TCPStore EADDRINUSE failures.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root
    python_exe = _resolve_python(args.python)
    lambdas = _parse_csv_ints(args.lambdas)
    gpus = _parse_csv_strs(args.gpus)

    run_root = Path(args.runs_root).resolve() / f"{args.run_name_prefix}_{_now_tag()}"
    run_root.mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(
        project_root=project_root,
        base_model=args.base_model,
        seed=int(args.seed),
        lambdas=lambdas,
        gpus=gpus,
        run_root=run_root,
    )

    manifest = {
        "run_root": str(run_root),
        "base_model": args.base_model,
        "seed": int(args.seed),
        "python": python_exe,
        "gpus": gpus,
        "poll_seconds": float(args.poll_seconds),
        "max_port_retries": int(args.max_port_retries),
        "jobs": [
            {
                "job_id": j.job_id,
                "lamda": int(j.lamda),
                "gpu": j.gpu,
                "script_path": str(j.script_path),
                "output_path": str(j.output_path),
                "results_json": str(j.results_json),
                "fire_args": j.fire_args,
            }
            for j in jobs
        ],
    }
    _write_json(run_root / "manifest.json", manifest)

    print(f"[ewcdr-ablation] run_root={run_root}")
    print(f"[ewcdr-ablation] python={python_exe}")
    print(f"[ewcdr-ablation] jobs={len(jobs)} | gpus={gpus}")

    if args.dry_run:
        for j in jobs:
            cmd = [python_exe, str(j.script_path)] + [_fire_arg(k, v) for k, v in j.fire_args.items()]
            print(f"[dry-run] job={j.job_id} gpu={j.gpu} :: {shlex.join(cmd)}")
        print("[ewcdr-ablation] dry-run complete")
        return 0

    pending = list(jobs)
    running: Dict[str, Dict[str, Any]] = {}
    completed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    max_port_retries = max(0, int(args.max_port_retries))
    retry_counts: Dict[str, int] = {j.job_id: 0 for j in jobs}

    try:
        while pending or running:
            while pending:
                job = pending.pop(0)
                info = _launch_job(job=job, run_root=run_root, repo_root=repo_root, python_exe=python_exe)
                running[job.job_id] = info
                print(
                    f"[ewcdr-ablation] launched job={job.job_id} gpu={job.gpu} pid={info['pid']} "
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
                            f"[ewcdr-ablation] retrying job={job.job_id} after EADDRINUSE "
                            f"(attempt {retry_counts[job.job_id]}/{max_port_retries})"
                        )
                        finished_jobs.append(running_job_id)
                        continue

                result = {
                    "job_id": job.job_id,
                    "lamda": int(job.lamda),
                    "gpu": job.gpu,
                    "pid": int(info["pid"]),
                    "returncode": int(rc),
                    "elapsed_sec": float(elapsed),
                    "stdout": str(info["stdout"]),
                    "stderr": str(info["stderr"]),
                    "output_path": str(job.output_path),
                    "results_json": str(job.results_json),
                }
                completed.append(result)
                if int(rc) != 0:
                    failures.append(result)
                    print(f"[ewcdr-ablation] FAILED job={job.job_id} gpu={job.gpu} rc={rc}")
                else:
                    print(f"[ewcdr-ablation] finished job={job.job_id} gpu={job.gpu} rc=0")
                finished_jobs.append(running_job_id)

            for running_job_id in finished_jobs:
                running.pop(running_job_id, None)

    except KeyboardInterrupt:
        print("[ewcdr-ablation] interrupt received; terminating running jobs")
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
                "remaining_jobs": [j.job_id for j in pending],
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
        print(f"[ewcdr-ablation] done with failures: {len(failures)}/{len(jobs)}")
        return 1

    print(f"[ewcdr-ablation] all jobs completed successfully: {len(jobs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
