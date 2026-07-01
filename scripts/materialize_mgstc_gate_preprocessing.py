#!/usr/bin/env python3
"""MGSTC feasibility-gate preprocessing materialization entry point."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thesis_traffic_drift.mgstc_gate import (
    discover_milan_daily_files,
    load_mgstc_gate_config,
    materialize_mgstc_gate,
    resolve_env_path,
)


DEFAULT_CONFIG = Path("configs/MGSTC-feasibility-gate.template.yaml")


def is_relative_to(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_repo_path(path: Path) -> Path:
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def validate_output_dir(output_dir: Path) -> None:
    local_root = (REPO_ROOT / "artifacts" / "local").resolve()
    resolved = resolve_repo_path(output_dir)
    if not is_relative_to(resolved, local_root):
        raise SystemExit("ERROR: output-dir must be under artifacts/local/.")


def ensure_writable_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise SystemExit("ERROR: output-dir is non-empty; pass --overwrite to replace local artifacts.")
        shutil.rmtree(output_dir)


def display_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def resolve_raw_dir(config, cli_raw_dir: Path | None):
    if cli_raw_dir is not None:
        return cli_raw_dir, "--raw-dir"
    configured = str(config["raw_milan_daily_dir"])
    try:
        return resolve_env_path(configured)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned outputs without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty local output directory.")
    args = parser.parse_args()

    config = load_mgstc_gate_config(resolve_repo_path(args.config))
    raw_dir, raw_source = resolve_raw_dir(config, args.raw_dir)
    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise SystemExit(f"ERROR: raw dir from {raw_source} is missing or invalid.")
    try:
        files = discover_milan_daily_files(raw_dir)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    output_dir = args.output_dir or Path(config["output_dir"])
    validate_output_dir(output_dir)
    resolved_output = resolve_repo_path(output_dir)

    planned = [
        "manifest.json",
        "metadata.json",
        "indices/cell_index.csv",
        "indices/timestamp_index.csv",
        "indices/split_index.csv",
        "arrays/clean_train.npz",
        "arrays/clean_validation.npz",
        "arrays/clean_test.npz",
        "summaries/validation_checks.json",
        "summaries/observation_counts_by_split.csv",
    ]
    print("MGSTC feasibility-gate preprocessing inputs validated.")
    print(f"Raw dir source: {raw_source}; daily files discovered: {len(files)}")
    print(f"Output status: local-only; planned file count: {len(planned)}")
    if args.dry_run:
        print("Dry run requested; no files were written.")
        for relative in planned:
            print(f"planned: {relative}")
        return 0

    ensure_writable_output(resolved_output, args.overwrite)
    materialize_mgstc_gate(config, raw_dir, resolved_output, overwrite=False, raw_source=raw_source)
    print("MGSTC feasibility-gate local materialization complete.")
    print(f"Output directory: {display_path(resolved_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
