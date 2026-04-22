#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RunSpec:
    name: str
    script_path: Path
    gpu_id: str


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _prepare_env(gpu_id: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["WORLD_SIZE"] = "1"
    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29500"
    return env


def _launch(spec: RunSpec, run_root: Path, python_exe: str) -> Dict:
    workdir = run_root / spec.name
    logs_dir = workdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = logs_dir / "stdout.log"
    stderr_path = logs_dir / "stderr.log"

    cmd = [python_exe, str(spec.script_path)]
    env = _prepare_env(spec.gpu_id)

    with (logs_dir / "command.txt").open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")
    _write_json(
        logs_dir / "launch_env.json",
        {
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "WORLD_SIZE": env.get("WORLD_SIZE"),
            "RANK": env.get("RANK"),
            "LOCAL_RANK": env.get("LOCAL_RANK"),
            "cwd": str(workdir),
        },
    )

    out_f = stdout_path.open("w", encoding="utf-8")
    err_f = stderr_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(workdir),
        env=env,
        stdout=out_f,
        stderr=err_f,
        text=True,
    )

    return {
        "name": spec.name,
        "gpu_id": spec.gpu_id,
        "pid": proc.pid,
        "proc": proc,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "workdir": str(workdir),
        "command": cmd,
        "_stdout_handle": out_f,
        "_stderr_handle": err_f,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run standard and safety Qwen continual finetuning in parallel on two single GPUs, "
            "with unique run folders/logs."
        )
    )
    parser.add_argument("--gpus", type=str, default="6,7", help="Two GPU ids as comma-separated values, e.g. 6,7")
    parser.add_argument(
        "--runs-root",
        type=str,
        default="./orchestrator_runs",
        help="Root folder for timestamped run outputs.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=20.0,
        help="Polling interval for status updates.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default="python",
        help="Python executable used for child training runs.",
    )
    args = parser.parse_args()

    gpu_ids: List[str] = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if len(gpu_ids) != 2:
        raise ValueError("--gpus must contain exactly two GPU ids, e.g. --gpus 6,7")

    script_dir = Path(__file__).resolve().parent
    std_script = script_dir / "finetune_qwen_forever.py"
    safety_script = script_dir / "finetune_qwen_safety_forever.py"
    if not std_script.exists() or not safety_script.exists():
        raise FileNotFoundError("Expected finetune scripts next to orchestrator.")

    ts = _now_tag()
    run_root = Path(args.runs_root).resolve() / f"dual_qwen_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    specs = [
        RunSpec(name="standard", script_path=std_script, gpu_id=gpu_ids[0]),
        RunSpec(name="safety", script_path=safety_script, gpu_id=gpu_ids[1]),
    ]

    manifest = {
        "timestamp": ts,
        "run_root": str(run_root),
        "runs": [
            {
                "name": s.name,
                "script": str(s.script_path),
                "gpu_id": s.gpu_id,
            }
            for s in specs
        ],
    }
    _write_json(run_root / "manifest.json", manifest)

    python_exe = args.python
    resolved_python = shutil.which(python_exe) if not os.path.isabs(python_exe) else python_exe
    if not resolved_python or not Path(resolved_python).exists():
        raise FileNotFoundError(f"Could not resolve python executable: {python_exe}")

    print(f"[orchestrator] run_root={run_root}")
    print(f"[orchestrator] python={resolved_python}")

    launched = []
    for spec in specs:
        info = _launch(spec, run_root, resolved_python)
        launched.append(info)
        print(
            f"[orchestrator] launched {spec.name} on GPU {spec.gpu_id} "
            f"pid={info['pid']} stdout={info['stdout']}"
        )

    _write_json(
        run_root / "pids.json",
        {
            "standard_pid": launched[0]["pid"],
            "safety_pid": launched[1]["pid"],
        },
    )

    pending = {x["name"]: x for x in launched}
    completed: Dict[str, int] = {}

    try:
        while pending:
            finished = []
            for name, info in pending.items():
                rc = info["proc"].poll()
                if rc is not None:
                    completed[name] = int(rc)
                    finished.append(name)
                    print(f"[orchestrator] {name} finished rc={rc}")
            for name in finished:
                info = pending.pop(name)
                info["_stdout_handle"].close()
                info["_stderr_handle"].close()
            if pending:
                print("[orchestrator] still running: " + ", ".join(sorted(pending.keys())))
                time.sleep(max(1.0, float(args.poll_seconds)))
    except KeyboardInterrupt:
        print("[orchestrator] interrupt received, terminating child processes...")
        for info in pending.values():
            info["proc"].terminate()
        for info in pending.values():
            try:
                info["proc"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                info["proc"].kill()
        for info in pending.values():
            info["_stdout_handle"].close()
            info["_stderr_handle"].close()
        raise

    _write_json(run_root / "final_status.json", completed)
    failures = {k: v for k, v in completed.items() if v != 0}
    if failures:
        print(f"[orchestrator] completed with failures: {failures}")
        return 1

    print("[orchestrator] all runs finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
