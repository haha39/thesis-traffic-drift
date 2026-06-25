#!/usr/bin/env python3
"""EXP-001 deterministic preprocessing entry point.

This script currently performs public-safe configuration validation and
metadata-only planning. It does not materialize full Milan data, generate
training-ready tensors, train models, evaluate metrics, or implement bridge
methods.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thesis_traffic_drift.exp001 import construct_split_intervals, drift_point, load_exp001_config


def is_relative_to(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_output_dir(output_dir):
    local_root = (REPO_ROOT / "artifacts" / "local").resolve()
    resolved = output_dir.resolve()
    if output_dir.is_absolute():
        if is_relative_to(resolved, REPO_ROOT.resolve()) and not is_relative_to(resolved, local_root):
            raise SystemExit("ERROR: output-dir inside this repository must be under artifacts/local/.")
        return
    if not is_relative_to((REPO_ROOT / output_dir).resolve(), local_root):
        raise SystemExit("ERROR: relative output-dir must be under artifacts/local/.")


def resolve_raw_dir(config, cli_raw_dir):
    if cli_raw_dir is not None:
        return Path(cli_raw_dir), "--raw-dir"
    configured = config["dataset"].get("raw_dir")
    if configured:
        return Path(configured), "dataset.raw_dir"
    env_name = config["dataset"]["raw_dir_env"]
    env_value = os.environ.get(env_name)
    if env_value:
        return Path(env_value), env_name
    raise SystemExit(
        "ERROR: raw dir is required but unavailable. Provide --raw-dir, "
        "dataset.raw_dir, or the environment variable named by dataset.raw_dir_env."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate config without writing metadata.")
    args = parser.parse_args()

    config = load_exp001_config(args.config)
    raw_dir, raw_source = resolve_raw_dir(config, args.raw_dir)
    output_dir = args.output_dir or Path(config["outputs"]["root"])

    validate_output_dir(output_dir)
    if not raw_dir.exists():
        raise SystemExit(f"ERROR: raw dir from {raw_source} does not exist: {raw_dir}")

    intervals = construct_split_intervals(config)
    metadata = {
        "experiment_id": config["experiment_id"],
        "status": "metadata_only",
        "raw_dir_source": raw_source,
        "output_dir": str(output_dir),
        "splits": [
            {"name": interval.name, "start": interval.start.isoformat(), "end": interval.end.isoformat() if interval.end else "dataset_end"}
            for interval in intervals
        ],
        "drift_point": drift_point(config).isoformat(),
        "model_training": "not_implemented",
        "training_ready_tensors": "not_generated_missing_null_policy_requires_human_approval",
    }

    print("EXP-001 deterministic preprocessing metadata validated.")
    print("EXP-001 model training is not implemented.")
    print("Training-ready tensors are not generated because modeling-time missing/null policy still requires human approval.")
    if args.dry_run:
        print("Dry run requested; no files were written.")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata_only_preprocessing_plan.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Metadata-only output: {output_dir / 'metadata_only_preprocessing_plan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
