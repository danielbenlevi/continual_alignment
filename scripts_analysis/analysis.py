#!/usr/bin/env python3
"""Summarize orchestrator full-suite runs into markdown tables (+ optional plots)."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Sequence, Tuple

KNOWN_MODES = ("qwen", "llama")
DEFAULT_METHOD_ORDER = (
    "forever_base",
    "safety_forever_base",
    "safety_forever_v2_kl",
    "safety_forever_v2_layer_reg",
    "ewcdr_base",
    "ewcdr_safety",
)


@dataclass
class RunContext:
    run_dir: Path
    manifest: Dict
    methods: List[str]
    models: List[str]
    used_modes: List[str]


def _detect_mode(model_alias: str) -> str:
    text = str(model_alias).lower()
    if "llama" in text:
        return "llama"
    if "qwen" in text:
        return "qwen"
    return text


def _pick_latest_full_suite_run(runs_root: Path) -> Path:
    candidates = sorted(p for p in runs_root.glob("full_suite_*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No full_suite_* run found under: {runs_root}")
    return candidates[-1]


def _load_manifest(run_dir: Path) -> Dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _discover_methods(run_dir: Path, manifest: Dict) -> List[str]:
    methods_from_manifest = [x.get("key") for x in manifest.get("experiments", []) if x.get("key")]
    results_root = run_dir / "results"
    methods_from_fs = sorted(p.name for p in results_root.iterdir() if p.is_dir()) if results_root.exists() else []
    methods = list(dict.fromkeys([*methods_from_manifest, *methods_from_fs]))
    if not methods:
        raise FileNotFoundError(f"No methods found in manifest/results for run: {run_dir}")
    ordered = [m for m in DEFAULT_METHOD_ORDER if m in methods]
    ordered.extend(m for m in methods if m not in ordered)
    return ordered


def _discover_models(run_dir: Path, methods: Sequence[str], manifest: Dict) -> List[str]:
    models_manifest = [str(m).split("/")[-1] for m in manifest.get("models", [])]
    models_fs: List[str] = []
    results_root = run_dir / "results"
    for method in methods:
        method_dir = results_root / method
        if not method_dir.exists():
            continue
        for model_dir in method_dir.iterdir():
            if model_dir.is_dir():
                models_fs.append(model_dir.name)
    models = list(dict.fromkeys([*models_manifest, *sorted(set(models_fs))]))
    if not models:
        raise FileNotFoundError(f"No model directories found in run results: {results_root}")
    return models


def _used_modes_from_results(run_dir: Path, methods: Sequence[str], models: Sequence[str]) -> List[str]:
    results_root = run_dir / "results"
    used: set[str] = set()
    for method in methods:
        for model in models:
            model_dir = results_root / method / model
            if any(model_dir.glob("seed_*.json")):
                used.add(_detect_mode(model))
    known = [m for m in KNOWN_MODES if m in used]
    unknown = sorted(m for m in used if m not in KNOWN_MODES)
    return known + unknown


def _build_context(run_dir: Path) -> RunContext:
    manifest = _load_manifest(run_dir)
    methods = _discover_methods(run_dir, manifest)
    models = _discover_models(run_dir, methods, manifest)
    used_modes = _used_modes_from_results(run_dir, methods, models)
    if not used_modes:
        raise RuntimeError(f"No mode data found in run: {run_dir}")
    return RunContext(run_dir=run_dir, manifest=manifest, methods=methods, models=models, used_modes=used_modes)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _seed_files_for_method_mode(run_dir: Path, method: str, models: Iterable[str], mode: str) -> List[Path]:
    files: List[Path] = []
    for model in models:
        if _detect_mode(model) != mode:
            continue
        files.extend(sorted((run_dir / "results" / method / model).glob("seed_*.json")))
    return files


def _format_cell(values: Sequence[float], digits: int = 2) -> str:
    if not values:
        return "NA"
    pct_values = [100.0 * v for v in values]
    if len(values) == 1:
        return f"{pct_values[0]:.{digits}f}% ± 0.00%"
    return f"{mean(pct_values):.{digits}f}% ± {stdev(pct_values):.{digits}f}%"


def _human_metric(metric: str) -> str:
    if metric == "asr":
        return "ASR"
    if metric.endswith("_acc"):
        return metric[:-4]
    return metric


def _aggregate_method_mode(
    run_dir: Path,
    method: str,
    mode: str,
    models: Sequence[str],
) -> Tuple[List[str], List[str], List[Dict[str, str]], int]:
    seed_files = _seed_files_for_method_mode(run_dir, method, models, mode)
    if not seed_files:
        return [], [], [], 0

    loaded = []
    for file_path in seed_files:
        with file_path.open("r", encoding="utf-8") as f:
            loaded.append(json.load(f))

    task_order = loaded[0].get("task_order", [])
    max_stages = min(len(obj.get("stages", [])) for obj in loaded)
    if max_stages == 0:
        return [], [], [], len(seed_files)

    metric_keys: List[str] = []
    seen = set()
    for obj in loaded:
        for stage in obj.get("stages", []):
            for key, value in stage.items():
                if _is_number(value) and key not in seen:
                    seen.add(key)
                    metric_keys.append(key)

    rows: List[Dict[str, str]] = []
    stage_labels: List[str] = []
    for stage_idx in range(max_stages):
        labels = [obj["stages"][stage_idx].get("label", f"Stage {stage_idx + 1}") for obj in loaded]
        stage_label = labels[0]
        if len(set(labels)) > 1:
            stage_label = f"Stage {stage_idx + 1}"
        stage_labels.append(stage_label)

        row = {"after_training": stage_label}
        for metric in metric_keys:
            vals: List[float] = []
            for obj in loaded:
                v = obj["stages"][stage_idx].get(metric)
                if _is_number(v) and not math.isnan(float(v)):
                    vals.append(float(v))
            cell = _format_cell(vals)
            diagonal_metric = None
            if stage_idx < len(task_order):
                stage_task = str(task_order[stage_idx]).lower()
                diagonal_metric = "asr" if stage_task == "safety" else f"{stage_task}_acc"
            if diagonal_metric is not None and metric == diagonal_metric and cell != "NA":
                cell = f"**{cell}**"
            row[metric] = cell
        rows.append(row)

    return stage_labels, metric_keys, rows, len(seed_files)


def _make_markdown_table(metric_keys: Sequence[str], rows: Sequence[Dict[str, str]]) -> str:
    header = ["After Training", *[_human_metric(m) for m in metric_keys]]
    sep = ["---"] * len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        values = [row.get("after_training", "")]
        values.extend(row.get(m, "NA") for m in metric_keys)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_results_markdown(ctx: RunContext, out_md: Path) -> None:
    lines: List[str] = []
    run_name = ctx.run_dir.name
    lines.append("# Continual Learning Summary")
    lines.append("")
    lines.append(f"- Run: `{run_name}`")
    lines.append(f"- Run path: `{ctx.run_dir}`")
    lines.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Modes found in run: `{', '.join(ctx.used_modes)}`")
    lines.append("")

    for mode in ctx.used_modes:
        lines.append(f"## Mode: {mode}")
        lines.append("")
        for method in ctx.methods:
            _, metric_keys, rows, n_seeds = _aggregate_method_mode(
                run_dir=ctx.run_dir,
                method=method,
                mode=mode,
                models=ctx.models,
            )
            lines.append(f"### Method: {method}")
            if not rows:
                lines.append("")
                lines.append("_No results found for this method/mode in this run._")
                lines.append("")
                continue
            lines.append("")
            lines.append(f"_Seeds aggregated: {n_seeds}_")
            lines.append("")
            lines.append(_make_markdown_table(metric_keys, rows))
            lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _try_generate_plots(ctx: RunContext, plot_root: Path) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    plot_root.mkdir(parents=True, exist_ok=True)
    made_any_plot = False
    for mode in ctx.used_modes:
        for method in ctx.methods:
            _, metric_keys, _, _ = _aggregate_method_mode(
                run_dir=ctx.run_dir,
                method=method,
                mode=mode,
                models=ctx.models,
            )
            seed_files = _seed_files_for_method_mode(ctx.run_dir, method, ctx.models, mode)
            if not seed_files:
                continue

            loaded = []
            for file_path in seed_files:
                with file_path.open("r", encoding="utf-8") as f:
                    loaded.append(json.load(f))
            max_stages = min(len(obj.get("stages", [])) for obj in loaded)
            if max_stages == 0:
                continue

            x = list(range(1, max_stages + 1))
            for metric in metric_keys:
                means, stds = [], []
                for i in range(max_stages):
                    vals = []
                    for obj in loaded:
                        v = obj["stages"][i].get(metric)
                        if _is_number(v) and not math.isnan(float(v)):
                            vals.append(float(v))
                    if vals:
                        means.append(mean(vals))
                        stds.append(stdev(vals) if len(vals) > 1 else 0.0)
                    else:
                        means.append(float("nan"))
                        stds.append(0.0)

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(x, means, marker="o")
                lower = [m - s if not math.isnan(m) else float("nan") for m, s in zip(means, stds)]
                upper = [m + s if not math.isnan(m) else float("nan") for m, s in zip(means, stds)]
                ax.fill_between(x, lower, upper, alpha=0.2)
                ax.set_title(f"{method} | {mode} | {_human_metric(metric)}")
                ax.set_xlabel("Task index (after training)")
                ax.set_ylabel(_human_metric(metric))
                ax.grid(alpha=0.3)

                out_dir = plot_root / mode / method
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{metric}.png"
                fig.tight_layout()
                fig.savefig(out_path, dpi=160)
                plt.close(fig)
                made_any_plot = True
    return made_any_plot


def main() -> int:
    our_scripts_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Summarize latest/full-suite orchestrator results.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=our_scripts_dir / "orchestrator_runs",
        help="Directory containing full_suite_* runs.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional explicit run directory (e.g., .../full_suite_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--results-md",
        type=Path,
        default=our_scripts_dir / "results.md",
        help="Output markdown file path.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Optional output directory for plots (default: <run_dir>/analysis_plots).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve() if args.run_dir else _pick_latest_full_suite_run(args.runs_root.resolve())
    ctx = _build_context(run_dir)

    _write_results_markdown(ctx, args.results_md.resolve())

    plots_attempted = False
    plots_made = False
    if not args.skip_plots:
        plots_dir = args.plots_dir.resolve() if args.plots_dir else (ctx.run_dir / "analysis_plots")
        plots_attempted = True
        plots_made = _try_generate_plots(ctx, plots_dir)

    print(f"[analysis] run_dir={ctx.run_dir}")
    print(f"[analysis] modes={ctx.used_modes}")
    print(f"[analysis] wrote markdown: {args.results_md.resolve()}")
    if plots_attempted and plots_made:
        print(f"[analysis] plots dir: {plots_dir}")
    elif plots_attempted:
        print("[analysis] plots skipped (matplotlib unavailable or no plottable data).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
