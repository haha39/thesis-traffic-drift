#!/usr/bin/env python3
"""EXP-001 local preprocessing materialization entry point.

This writes clean/drifted dense Top-100 arrays with explicit observation masks.
It does not create model training windows, train models, run baselines, evaluate
metrics, implement bridge methods, or claim EXP-001 results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thesis_traffic_drift.exp001 import (
    STATE_CODE_BY_NAME,
    build_dense_arrays,
    construct_split_intervals,
    datetime_to_ms,
    dense_shape,
    discover_milan_daily_files,
    drift_point,
    load_exp001_config,
    ms_to_datetime,
    observation_counts,
    observed_values,
    rank_topk_from_means,
    read_day_groups,
    scan_training_means,
    select_affected_cells,
    timestamp_range_ms,
    write_csv_rows,
    write_json,
    write_npz,
)


def is_relative_to(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_output_dir(output_dir):
    local_root = (REPO_ROOT / "artifacts" / "local").resolve()
    resolved = (REPO_ROOT / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    if not is_relative_to(resolved, local_root):
        raise SystemExit("ERROR: output-dir must be under artifacts/local/.")


def resolve_output_dir(output_dir):
    return (REPO_ROOT / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()


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


def ensure_writable_output(output_dir, overwrite):
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise SystemExit("ERROR: output-dir is non-empty; pass --overwrite to replace local artifacts.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def merge_group(target, incoming):
    target.raw_rows += incoming.raw_rows
    target.numeric_rows += incoming.numeric_rows
    target.null_rows += incoming.null_rows
    target.invalid_rows += incoming.invalid_rows
    target.finite_sum += incoming.finite_sum


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_config_snapshot(config):
    snapshot = json.loads(json.dumps(config))
    if "dataset" in snapshot:
        snapshot["dataset"]["raw_dir"] = None
    return snapshot


def mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance ** 0.5


def split_timestamp_plan(intervals, drift_ms, dataset_end_ms):
    by_name = {interval.name: interval for interval in intervals}
    train = timestamp_range_ms(by_name["train"].start, by_name["train"].end)
    validation = timestamp_range_ms(by_name["validation"].start, by_name["validation"].end)
    test_start = by_name["test"].start
    test_end = by_name["test"].end
    test_predrift = list(range(datetime_to_ms(test_start), drift_ms, 10 * 60 * 1000))
    test_postdrift = list(range(drift_ms, datetime_to_ms(test_end), 10 * 60 * 1000))
    holdout = list(range(datetime_to_ms(by_name["holdout"].start), dataset_end_ms, 10 * 60 * 1000))
    return {
        "train": train,
        "validation": validation,
        "test_predrift": test_predrift,
        "test_postdrift": test_postdrift,
        "holdout": holdout,
    }


def write_checksums(output_dir, checksums):
    lines = [f"{digest}  {relative}\n" for relative, digest in sorted(checksums.items())]
    (output_dir / "checksums.sha256").write_text("".join(lines), encoding="utf-8")


def display_output_dir(path):
    return str(path.relative_to(REPO_ROOT))


def materialize(config, raw_dir, output_dir, overwrite=False, raw_source="unknown"):
    intervals = construct_split_intervals(config)
    point = drift_point(config)
    drift_ms = datetime_to_ms(point)
    files = discover_milan_daily_files(raw_dir)
    expected_cells = int(config["dataset"]["expected_cells"])
    k = int(config["selection"]["k"])

    training_means, all_timestamps, min_ts, max_ts, file_stats = scan_training_means(files, intervals, expected_cells)
    dataset_end_ms = max_ts + 10 * 60 * 1000
    topk_cells = rank_topk_from_means(training_means, k)
    affected_cells = select_affected_cells(topk_cells, training_means, float(config["drift"]["affected_fraction_of_topk"]))

    keep_cells = set(topk_cells)
    groups_by_ts = defaultdict(dict)
    for path in files:
        groups, _stats = read_day_groups(path, keep_cells=keep_cells)
        for (ts_ms, cell_id), group in groups.items():
            existing = groups_by_ts[ts_ms].get(cell_id)
            if existing is None:
                groups_by_ts[ts_ms][cell_id] = group
            else:
                merge_group(existing, group)

    ensure_writable_output(output_dir, overwrite)
    for child in ("selection", "indices", "arrays", "summaries"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    split_timestamps = split_timestamp_plan(intervals, drift_ms, dataset_end_ms)
    clean_train = build_dense_arrays(groups_by_ts, split_timestamps["train"], topk_cells)
    clean_validation = build_dense_arrays(groups_by_ts, split_timestamps["validation"], topk_cells)
    clean_predrift = build_dense_arrays(groups_by_ts, split_timestamps["test_predrift"], topk_cells)
    clean_postdrift = build_dense_arrays(groups_by_ts, split_timestamps["test_postdrift"], topk_cells)
    drifted_postdrift = build_dense_arrays(
        groups_by_ts,
        split_timestamps["test_postdrift"],
        topk_cells,
        drift_timestamp_ms=drift_ms,
        affected_cells=affected_cells,
        factor=float(config["drift"]["factor"]),
    )
    holdout_clean = build_dense_arrays(groups_by_ts, split_timestamps["holdout"], topk_cells)

    arrays = {
        "arrays/clean_train.npz": ("train", "clean", clean_train),
        "arrays/clean_validation.npz": ("validation", "clean", clean_validation),
        "arrays/clean_test_predrift.npz": ("test_predrift", "clean", clean_predrift),
        "arrays/clean_test_postdrift.npz": ("test_postdrift", "clean", clean_postdrift),
        "arrays/drifted_test_postdrift.npz": ("test_postdrift", "drifted", drifted_postdrift),
        "arrays/holdout_clean.npz": ("holdout", "clean_reserved_not_used_exp001_v0", holdout_clean),
    }
    for relative, (_split, _label, dense) in arrays.items():
        write_npz(output_dir / relative, dense)

    top_rows = [
        {"rank": rank, "cell_id": cell_id, "training_mean_observed_numeric": training_means[cell_id]}
        for rank, cell_id in enumerate(topk_cells, start=1)
    ]
    affected_set = set(affected_cells)
    write_csv_rows(output_dir / "selection" / "top100_cells.csv", top_rows, ["rank", "cell_id", "training_mean_observed_numeric"])
    write_csv_rows(
        output_dir / "selection" / "affected20_cells.csv",
        [row for row in top_rows if row["cell_id"] in affected_set],
        ["rank", "cell_id", "training_mean_observed_numeric"],
    )
    write_csv_rows(
        output_dir / "selection" / "training_cell_means.csv",
        [{"cell_id": cell_id, "training_mean_observed_numeric": training_means[cell_id]} for cell_id in range(1, expected_cells + 1)],
        ["cell_id", "training_mean_observed_numeric"],
    )
    write_csv_rows(
        output_dir / "indices" / "cell_index.csv",
        [{"cell_position": idx, "cell_id": cell_id} for idx, cell_id in enumerate(topk_cells)],
        ["cell_position", "cell_id"],
    )

    timestamp_rows = []
    split_rows = []
    for split_name, timestamps in split_timestamps.items():
        split_rows.append(
            {
                "split": split_name,
                "start": ms_to_datetime(timestamps[0], config["dataset"]["timezone"]).isoformat() if timestamps else None,
                "end_exclusive": ms_to_datetime(timestamps[-1] + 10 * 60 * 1000, config["dataset"]["timezone"]).isoformat() if timestamps else None,
                "samples": len(timestamps),
            }
        )
        for pos, ts_ms in enumerate(timestamps):
            timestamp_rows.append(
                {
                    "split": split_name,
                    "split_position": pos,
                    "timestamp_ms": ts_ms,
                    "timestamp_local": ms_to_datetime(ts_ms, config["dataset"]["timezone"]).isoformat(),
                }
            )
    write_csv_rows(output_dir / "indices" / "timestamp_index.csv", timestamp_rows, ["split", "split_position", "timestamp_ms", "timestamp_local"])
    write_csv_rows(output_dir / "indices" / "split_index.csv", split_rows, ["split", "start", "end_exclusive", "samples"])

    observation_split_rows = []
    for relative, (split_name, label, dense) in arrays.items():
        counts = observation_counts(dense)
        expected = dense_shape(dense)[0] * dense_shape(dense)[1]
        observation_split_rows.append({"array": relative, "split": split_name, "label": label, "expected_pairs": expected, **counts})
    write_csv_rows(
        output_dir / "summaries" / "observation_counts_by_split.csv",
        observation_split_rows,
        ["array", "split", "label", "expected_pairs", "NUMERIC_OBSERVED", "ALL_NULL", "ABSENT", "INVALID_ONLY"],
    )

    cell_count_rows = []
    for col, cell_id in enumerate(topk_cells):
        counts = {name: 0 for name in STATE_CODE_BY_NAME}
        for dense in (clean_train, clean_validation, clean_predrift, clean_postdrift, holdout_clean):
            for row in dense.state_code:
                code = row[col]
                for name, value in STATE_CODE_BY_NAME.items():
                    if value == code:
                        counts[name] += 1
                        break
        cell_count_rows.append({"cell_id": cell_id, **counts})
    write_csv_rows(
        output_dir / "summaries" / "observation_counts_by_cell.csv",
        cell_count_rows,
        ["cell_id", "NUMERIC_OBSERVED", "ALL_NULL", "ABSENT", "INVALID_ONLY"],
    )

    drift_changed_observed = 0
    for clean_row, drifted_row, mask_row in zip(clean_postdrift.values, drifted_postdrift.values, clean_postdrift.observed_mask):
        for cell_id, clean_value, drifted_value, mask in zip(topk_cells, clean_row, drifted_row, mask_row):
            if mask and cell_id in affected_set and clean_value != drifted_value:
                drift_changed_observed += 1
    drift_summary = [
        {
            "drift_point": point.isoformat(),
            "factor": config["drift"]["factor"],
            "affected_cell_count": len(affected_cells),
            "postdrift_samples": len(split_timestamps["test_postdrift"]),
            "changed_observed_values": drift_changed_observed,
            "mask_and_state_codes_preserved": clean_postdrift.observed_mask == drifted_postdrift.observed_mask
            and clean_postdrift.state_code == drifted_postdrift.state_code,
        }
    ]
    write_csv_rows(
        output_dir / "summaries" / "drift_application_summary.csv",
        drift_summary,
        ["drift_point", "factor", "affected_cell_count", "postdrift_samples", "changed_observed_values", "mask_and_state_codes_preserved"],
    )

    train_observed = observed_values(clean_train)
    norm_mean, norm_std = mean_std(train_observed)
    validation_checks = {
        "placeholder_zero_mask_check": all(
            value == 0.0
            for dense in (clean_train, clean_validation, clean_predrift, clean_postdrift, holdout_clean)
            for value_row, mask_row in zip(dense.values, dense.observed_mask)
            for value, mask in zip(value_row, mask_row)
            if not mask
        ),
        "drift_preserves_mask_and_state_code": clean_postdrift.observed_mask == drifted_postdrift.observed_mask
        and clean_postdrift.state_code == drifted_postdrift.state_code,
        "topk_count": len(topk_cells),
        "affected_count": len(affected_cells),
        "no_boundary_sample_dropped": bool(split_timestamps["test_postdrift"] and split_timestamps["test_postdrift"][0] == drift_ms),
    }
    write_json(output_dir / "summaries" / "validation_checks.json", validation_checks)

    metadata = {
        "experiment_id": config["experiment_id"],
        "local_artifact_status": "local-only",
        "config_snapshot": safe_config_snapshot(config),
        "timezone": config["dataset"]["timezone"],
        "resolution_minutes": config["dataset"]["resolution_minutes"],
        "source_summary": {
            "raw_dir_source": raw_source,
            "daily_file_count": len(files),
            "file_names": [path.name for path in files],
            "timestamp_min": ms_to_datetime(min_ts, config["dataset"]["timezone"]).isoformat(),
            "timestamp_max": ms_to_datetime(max_ts, config["dataset"]["timezone"]).isoformat(),
            "timestamp_count_present_in_raw": len(all_timestamps),
            "country_code_handling": "collapsed_by_summing_internet_values_over_countryCode_for_each_timestamp_cell",
        },
        "split_boundaries": split_rows,
        "drift": {
            "point": point.isoformat(),
            "factor": config["drift"]["factor"],
            "affected_fraction_of_topk": config["drift"]["affected_fraction_of_topk"],
            "boundary_inclusive": True,
            "applies_when": "t >= drift_point",
        },
        "observation_policy": config["missing_policy"],
        "state_codes": STATE_CODE_BY_NAME,
        "selection": {
            "k": k,
            "topk_source_segment": "train",
            "rank_metric": "training_period_observed_numeric_mean_traffic",
            "tie_break": "ascending_cell_id",
            "affected_cell_count": len(affected_cells),
        },
        "normalization_statistics": {
            "scope": "training_period_observed_numeric_values_only",
            "status": "statistics_only_not_applied",
            "observed_count": len(train_observed),
            "mean": norm_mean,
            "std": norm_std,
        },
        "holdout_status": "reserved_not_used_exp001_v0",
        "file_parse_summary": file_stats,
    }
    write_json(output_dir / "metadata.json", metadata)

    manifest_files = {}
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file() and p.name != "checksums.sha256"):
        relative = path.relative_to(output_dir).as_posix()
        manifest_files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    array_summaries = {
        relative: {
            "split": split,
            "label": label,
            "shape": dense_shape(dense),
            "dtypes": {"values": "float64", "observed_mask": "uint8", "state_code": "uint8", "timestamp_ms": "int64", "cell_ids": "int64"},
        }
        for relative, (split, label, dense) in arrays.items()
    }
    manifest = {
        "experiment_id": config["experiment_id"],
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "local_artifact_status": "local-only",
        "row_ordering": "timestamp ascending within each split",
        "column_ordering": "Top-100 cells ranked by train-period observed numeric mean, descending, ties ascending cell_id",
        "raw_aggregation": "countryCode rows are collapsed by summing Internet values for each timestamp/cell before dense materialization",
        "split_names": list(split_timestamps),
        "arrays": array_summaries,
        "files": manifest_files,
    }
    write_json(output_dir / "manifest.json", manifest)
    manifest_files["manifest.json"] = {"sha256": sha256_file(output_dir / "manifest.json"), "bytes": (output_dir / "manifest.json").stat().st_size}
    write_checksums(output_dir, {relative: item["sha256"] for relative, item in manifest_files.items()})
    return metadata, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned outputs without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty local output directory.")
    args = parser.parse_args()

    config = load_exp001_config(args.config)
    raw_dir, raw_source = resolve_raw_dir(config, args.raw_dir)
    output_dir = args.output_dir or Path(config["outputs"]["root"])
    validate_output_dir(output_dir)
    resolved_output = resolve_output_dir(output_dir)

    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise SystemExit(f"ERROR: raw dir from {raw_source} is missing or invalid.")
    try:
        files = discover_milan_daily_files(raw_dir)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    planned = [
        "manifest.json",
        "metadata.json",
        "checksums.sha256",
        "selection/top100_cells.csv",
        "selection/affected20_cells.csv",
        "selection/training_cell_means.csv",
        "indices/cell_index.csv",
        "indices/timestamp_index.csv",
        "indices/split_index.csv",
        "arrays/clean_train.npz",
        "arrays/clean_validation.npz",
        "arrays/clean_test_predrift.npz",
        "arrays/clean_test_postdrift.npz",
        "arrays/drifted_test_postdrift.npz",
        "arrays/holdout_clean.npz",
        "summaries/observation_counts_by_split.csv",
        "summaries/observation_counts_by_cell.csv",
        "summaries/drift_application_summary.csv",
        "summaries/validation_checks.json",
    ]
    print("EXP-001 local preprocessing inputs validated.")
    print(f"Raw dir source: {raw_source}; daily files discovered: {len(files)}")
    print(f"Output status: local-only; planned file count: {len(planned)}")
    if args.dry_run:
        print("Dry run requested; no files were written.")
        for relative in planned:
            print(f"planned: {relative}")
        return 0

    materialize(config, raw_dir, resolved_output, overwrite=args.overwrite, raw_source=raw_source)
    print("EXP-001 local materialization complete.")
    print(f"Output directory: {display_output_dir(resolved_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
