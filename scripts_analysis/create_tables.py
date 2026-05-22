#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_CONFIG = {
    "qwen": {
        "short_model": "Qwen3-0.6B",
        "default_run_root": str(PROJECT_ROOT / "orchestrator_runs" / "updated_full_results"),
        "mode_label": "qwen",
        "default_out": str(PROJECT_ROOT / "results_summary" / "results_qwen_updated.md"),
        "qwen_base_default_run_root": str(PROJECT_ROOT / "orchestrator_runs" / "qwen_base_updated_full_results"),
        "qwen_base_default_out": str(PROJECT_ROOT / "results_summary" / "results_qwen_base_updated.md"),
        "qwen_base_sem_default_run_root": str(PROJECT_ROOT / "orchestrator_runs" / "qwen_base_sem_results"),
        "qwen_base_sem_default_out": str(PROJECT_ROOT / "results_summary" / "results_qwen_base_sem.md"),
    },
    "llama": {
        "short_model": "Llama-3.2-3B-Instruct",
        "default_run_root": str(PROJECT_ROOT / "orchestrator_runs" / "llama_updated_full_results"),
        "mode_label": "llama",
        "default_out": str(PROJECT_ROOT / "results_summary" / "results_llama_updated.md"),
        "sem_default_run_root": str(PROJECT_ROOT / "orchestrator_runs" / "llama_7b_sem_results"),
        "sem_default_out": str(PROJECT_ROOT / "results_summary" / "results_llama_sem.md"),
    },
}

TASK_COLUMNS = [
    ("asr", "ASR"),
    ("gsm8k_acc", "gsm8k"),
    ("sst2_acc", "sst2"),
    ("mbpp_acc", "mbpp"),
    ("xsum_acc", "xsum"),
    ("sciq_acc", "sciq"),
    ("samsum_acc", "samsum"),
]


def pct(x: float) -> float:
    return 100.0 * x


def fmt_mean_std(values, include_std: bool = True):
    if not values:
        return "n/a"
    m = pct(mean(values))
    if not include_std:
        return f"{m:.2f}%"
    s = pct(stdev(values)) if len(values) > 1 else 0.0
    return f"{m:.2f}% ± {s:.2f}%"


def infer_results_paths(run_root: Path, method: str, base_model: str, seed: int, llama_guard: bool = False) -> list[Path]:
    short_model = base_model.split("/")[-1]
    results_root = run_root / ("lg_artifacts/results" if llama_guard else "results")
    flat = results_root / method / short_model / f"seed_{seed}.json"
    nested_seed_dir = results_root / method / short_model / f"seed_{seed}"
    candidates = [flat, nested_seed_dir / "order_0.json"]
    if nested_seed_dir.exists():
        candidates.extend(sorted(nested_seed_dir.glob("*.json")))
    return candidates


def load_by_method(run_root: Path, final_status_path: Path, llama_guard: bool = False):
    status = json.loads(final_status_path.read_text())
    grouped = {}
    for item in status.get("completed", []):
        method = item["experiment_key"]
        base_model = item["base_model"]
        seed = int(item["seed"])
        p = None
        for candidate in infer_results_paths(run_root, method, base_model, seed, llama_guard=llama_guard):
            if candidate.exists():
                p = candidate
                break
        if p is None and not llama_guard:
            fallback = Path(item.get("results_json", ""))
            if fallback.exists():
                p = fallback
        if p is None:
            continue
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        grouped.setdefault(method, []).append(data)
    return grouped


def build_markdown(
    run_name: str,
    run_root: Path,
    grouped: dict,
    mode_label: str,
    grouped_string_asr: dict | None = None,
    llama_guard: bool = False,
) -> str:
    lines = []
    lines.append("# Continual Learning Summary")
    lines.append("")
    lines.append(f"- Run: `{run_name}`")
    lines.append(f"- Run path: `{run_root}`")
    lines.append(f"- Modes found in run: `{mode_label}`")
    lines.append("")
    lines.append(f"## Mode: {mode_label}")
    lines.append("")

    for method in sorted(grouped.keys()):
        seeds = grouped[method]
        if not seeds:
            continue
        stages = seeds[0]["stages"]

        lines.append(f"### Method: {method}")
        lines.append("")
        lines.append(f"_Seeds aggregated: {len(seeds)}_")
        lines.append("")

        header = ["After Training"] + [display for _, display in TASK_COLUMNS]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        for stage_idx, stage in enumerate(stages):
            row = [stage["label"]]
            per_metric_values = {
                key: [seed["stages"][stage_idx][key] for seed in seeds if key in seed["stages"][stage_idx]]
                for key, _ in TASK_COLUMNS
            }
            diag_key = TASK_COLUMNS[stage_idx][0] if stage_idx < len(TASK_COLUMNS) else None

            for key, _ in TASK_COLUMNS:
                include_std = True
                cell = fmt_mean_std(per_metric_values[key], include_std=include_std)
                if llama_guard and key == "asr":
                    lg_cell = fmt_mean_std(per_metric_values[key], include_std=False)
                    string_cell = "n/a"
                    if grouped_string_asr and method in grouped_string_asr:
                        string_seeds = grouped_string_asr[method]
                        string_values = [
                            seed["stages"][stage_idx][key]
                            for seed in string_seeds
                            if key in seed["stages"][stage_idx]
                        ]
                        string_cell = fmt_mean_std(string_values, include_std=False)
                    cell = f"{lg_cell} ({string_cell})"
                if diag_key is not None and key == diag_key:
                    cell = f"**{cell}**"
                row.append(cell)
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def resolve_output_path(requested_out: str | None, model_name: str, default_out: str | None = None) -> Path:
    cfg = MODEL_CONFIG[model_name]
    effective_default_out = default_out or cfg["default_out"]
    out = Path(requested_out).resolve() if requested_out else Path(effective_default_out).resolve()
    stem = out.stem.lower()
    # Guard against cross-model overwrite when an explicit conflicting filename is passed.
    if model_name == "qwen" and "llama" in stem:
        raise ValueError("Refusing to write qwen output to a llama-named file. Use a qwen output path.")
    if model_name == "llama" and "qwen" in stem:
        raise ValueError("Refusing to write llama output to a qwen-named file. Use a llama output path.")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=None, help="Path to orchestrator run root (contains final_status.json and results/)")
    parser.add_argument("--out", default=None, help="Output markdown file (defaults to model-specific path in results_summary/)")
    parser.add_argument("--run-name", default=None, help="Display name for the run")
    parser.add_argument("--model-name", choices=sorted(MODEL_CONFIG.keys()), default="qwen")
    parser.add_argument(
        "--qwen-base",
        action="store_true",
        help="Use qwen-base defaults (qwen_base_updated_full_results + results_qwen_base_updated.md).",
    )
    parser.add_argument(
        "--sem",
        action="store_true",
        help="Use SEM run roots and SEM output defaults (llama_7b_sem_results or qwen_base_sem_results).",
    )
    parser.add_argument("--llama-guard", action="store_true", help="Use lg_artifacts/results/* ASR annotations")
    args = parser.parse_args()

    model_cfg = MODEL_CONFIG[args.model_name]
    default_run_root = model_cfg["default_run_root"]
    default_out = model_cfg["default_out"]
    if args.qwen_base and args.model_name == "qwen":
        default_run_root = model_cfg["qwen_base_default_run_root"]
        default_out = model_cfg["qwen_base_default_out"]
    if args.sem:
        if args.model_name == "llama":
            default_run_root = model_cfg["sem_default_run_root"]
            default_out = model_cfg["sem_default_out"]
        elif args.model_name == "qwen":
            if not args.qwen_base:
                raise ValueError("For qwen SEM results, pass both --sem and --qwen-base.")
            default_run_root = model_cfg["qwen_base_sem_default_run_root"]
            default_out = model_cfg["qwen_base_sem_default_out"]
    run_root = Path(args.run_root or default_run_root).resolve()
    final_status = run_root / "final_status.json"
    if not final_status.exists():
        raise FileNotFoundError(f"Missing {final_status}")

    grouped = load_by_method(run_root, final_status, llama_guard=args.llama_guard)
    grouped_string_asr = None
    if args.llama_guard:
        grouped_string_asr = load_by_method(run_root, final_status, llama_guard=False)
    md = build_markdown(
        args.run_name or run_root.name,
        run_root,
        grouped,
        model_cfg["mode_label"],
        grouped_string_asr=grouped_string_asr,
        llama_guard=args.llama_guard,
    )
    out_path = resolve_output_path(args.out, args.model_name, default_out=default_out)
    out_path.write_text(md)
    print(out_path)


if __name__ == "__main__":
    main()
