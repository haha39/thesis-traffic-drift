#!/usr/bin/env python3
"""Run focused EXP-001 local-only baseline-result diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thesis_traffic_drift.exp001_training import (
    load_baseline_config,
    run_baseline_diagnostics,
    write_diagnostic_results,
)


DEFAULT_CONFIG = Path("configs/EXP-001-baselines.template.yaml")
DEFAULT_INPUT_DIR = Path("artifacts/local/exp-001")
DEFAULT_OUTPUT_DIR = Path("artifacts/local/exp-001/baselines/diagnostics")


def is_relative_to(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_repo_path(path):
    path = Path(path).expanduser()
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def validate_output_dir(output_dir):
    local_root = (REPO_ROOT / "artifacts" / "local").resolve()
    resolved = resolve_repo_path(output_dir)
    if not is_relative_to(resolved, local_root):
        raise SystemExit("ERROR: output-dir must be under artifacts/local/.")


def ensure_writable_output(output_dir, overwrite):
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise SystemExit("ERROR: output-dir is non-empty; pass --overwrite to replace local diagnostic outputs.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def display_path(path):
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return "local-only-input-artifacts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Validate planned local diagnosis without writing outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty local diagnostic output directory.")
    args = parser.parse_args()

    config = load_baseline_config(resolve_repo_path(args.config))
    input_dir = resolve_repo_path(args.input_dir)
    output_dir = resolve_repo_path(args.output_dir)
    validate_output_dir(output_dir)
    input_length = int(config["window"]["input_length"])
    horizon = int(config["window"]["horizon"])

    planned = [
        "diagnosis_summary.json",
        "diagnosis_summary.csv",
        "group_metrics.csv",
        "per_cell_metrics.csv",
        "comparison_ratios.csv",
        "counts_summary.csv",
    ]
    print("EXP-001 local baseline diagnosis configured.")
    print(f"Input directory: {display_path(input_dir)}")
    print(f"Output status: local-only; planned file count: {len(planned)}")
    print("Last Value treatment: adaptive sanity baseline, not stale-model evidence.")
    if args.dry_run:
        print("Dry run requested; no files were written.")
        for relative in planned:
            print(f"planned: {relative}")
        return 0

    ensure_writable_output(output_dir, args.overwrite)
    results = run_baseline_diagnostics(input_dir, input_length=input_length, horizon=horizon, input_label=display_path(input_dir))
    write_diagnostic_results(output_dir, results)
    print("EXP-001 local baseline diagnosis complete.")
    print(f"Output directory: {display_path(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
