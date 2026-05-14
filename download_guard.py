#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Llama Guard model locally for vLLM serving")
    parser.add_argument(
        "--repo-id",
        default="meta-llama/Llama-Guard-4-12B",
        help="Hugging Face repo id to download",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Directory to store downloaded models (default: <this_script_dir>/models)",
    )
    parser.add_argument(
        "--local-name",
        default=None,
        help="Optional override for local folder name (default: repo name part)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HF revision (branch/tag/commit)",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Optional allow pattern(s), repeatable (e.g. --allow-pattern '*.safetensors')",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install with `pip install huggingface_hub`."
        ) from exc

    script_dir = Path(__file__).resolve().parent
    models_dir = Path(args.models_dir).expanduser().resolve() if args.models_dir else (script_dir / "models").resolve()
    local_name = args.local_name or args.repo_id.split("/")[-1]
    local_dir = (models_dir / local_name).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repo: {args.repo_id}")
    print(f"Destination: {local_dir}")
    if args.revision:
        print(f"Revision: {args.revision}")

    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        revision=args.revision,
        allow_patterns=args.allow_pattern,
        resume_download=True,
    )

    print("Download complete.")
    print(path)


if __name__ == "__main__":
    main()
