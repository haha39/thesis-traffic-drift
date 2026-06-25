"""EXP-001 non-neural sanity baselines.

This module consumes local-only EXP-001 materialized arrays, constructs masked
sliding windows in chronological order, and evaluates Last Value and Historical
Average baselines on original traffic scale. It intentionally does not train
neural models, read raw Milan data, use holdout, or write tensor artifacts.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from thesis_traffic_drift.exp001 import DenseArrays, parse_simple_yaml, write_json


REQUIRED_ARRAY_MEMBERS = ("values", "observed_mask", "state_code", "timestamp_ms", "cell_ids")
TRAIN_SPLIT = "clean_train"
EVALUATION_SPLITS = ("clean_test_predrift", "drifted_test_postdrift")
BASELINES = ("last_value", "historical_average")
METRICS = ("masked_mae", "masked_rmse", "masked_smape")


@dataclass(frozen=True)
class WindowTarget:
    input_start: int
    target_index: int
    target_timestamp_ms: int


@dataclass
class MetricAccumulator:
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    smape_sum: float = 0.0
    valid_count: int = 0
    unavailable_prediction_count: int = 0
    unobserved_target_count: int = 0

    def add(self, target: float, prediction: Optional[float], target_observed: int) -> None:
        if not target_observed:
            self.unobserved_target_count += 1
            return
        if prediction is None:
            self.unavailable_prediction_count += 1
            return
        error = abs(float(prediction) - float(target))
        self.absolute_error_sum += error
        self.squared_error_sum += error * error
        denominator = abs(float(target)) + abs(float(prediction))
        self.smape_sum += 0.0 if denominator == 0.0 else (2.0 * error / denominator)
        self.valid_count += 1

    def metrics(self) -> Dict[str, Optional[float]]:
        if self.valid_count == 0:
            return {"masked_mae": None, "masked_rmse": None, "masked_smape": None}
        return {
            "masked_mae": self.absolute_error_sum / self.valid_count,
            "masked_rmse": math.sqrt(self.squared_error_sum / self.valid_count),
            "masked_smape": self.smape_sum / self.valid_count,
        }


def load_baseline_config(path: Path) -> Dict[str, Any]:
    config = parse_simple_yaml(Path(path).read_text(encoding="utf-8"))
    validate_baseline_config(config)
    return config


def _get(config: Mapping[str, Any], path: Sequence[str]) -> Any:
    cursor: Any = config
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise ValueError(f"Missing required config field {'.'.join(path)}")
        cursor = cursor[key]
    return cursor


def validate_baseline_config(config: Mapping[str, Any]) -> None:
    expected = {
        ("experiment_id",): "EXP-001-stale-degradation-v0",
        ("input_artifacts", "root"): "artifacts/local/exp-001",
        ("output", "root"): "artifacts/local/exp-001/baselines",
        ("window", "input_length"): 144,
        ("window", "horizon"): 1,
        ("baselines", "enabled", "last_value"): True,
        ("baselines", "enabled", "historical_average"): True,
        ("evaluation", "splits", "clean_test_predrift"): True,
        ("evaluation", "splits", "drifted_test_postdrift"): True,
        ("metrics", "masked_mae"): True,
        ("metrics", "masked_rmse"): True,
        ("metrics", "masked_smape"): True,
        ("metrics", "degradation_ratio"): True,
        ("unreliable_gap", "status"): "deferred_human_threshold_required",
        ("holdout", "use"): False,
        ("dependencies", "neural"): "deferred",
        ("gpu", "use"): False,
    }
    for path, expected_value in expected.items():
        actual = _get(config, path)
        if actual != expected_value:
            raise ValueError(f"{'.'.join(path)} must be {expected_value!r}, got {actual!r}")


def _npy_payload(blob: bytes) -> Tuple[Mapping[str, Any], bytes]:
    if not blob.startswith(b"\x93NUMPY"):
        raise ValueError("Invalid .npy payload")
    major = blob[6]
    if major == 1:
        header_len = struct.unpack("<H", blob[8:10])[0]
        offset = 10
    elif major == 2:
        header_len = struct.unpack("<I", blob[8:12])[0]
        offset = 12
    else:
        raise ValueError(f"Unsupported .npy version {major}")
    header = ast.literal_eval(blob[offset : offset + header_len].decode("latin1").strip())
    return header, blob[offset + header_len :]


def _load_npy_array(blob: bytes) -> Any:
    header, payload = _npy_payload(blob)
    if header.get("fortran_order"):
        raise ValueError("Fortran-order arrays are not supported")
    dtype = header["descr"]
    shape = tuple(header["shape"])
    if dtype == "<f8":
        fmt, size, cast = "d", 8, float
    elif dtype == "|u1":
        fmt, size, cast = "B", 1, int
    elif dtype == "<i8":
        fmt, size, cast = "q", 8, int
    else:
        raise ValueError(f"Unsupported dtype {dtype!r}")
    count = 1
    for dim in shape:
        count *= int(dim)
    expected = count * size
    if len(payload) != expected:
        raise ValueError(f"Unexpected payload length for dtype {dtype!r}: expected {expected}, got {len(payload)}")
    values = [cast(v) for v in struct.unpack("<" + fmt * count, payload)] if count else []
    if len(shape) == 1:
        return values
    if len(shape) == 2:
        rows, cols = shape
        return [values[index * cols : (index + 1) * cols] for index in range(rows)]
    raise ValueError(f"Unsupported array shape {shape!r}")


def load_materialized_npz(path: Path) -> DenseArrays:
    if not path.exists():
        raise FileNotFoundError(f"Missing required EXP-001 array artifact: {path}")
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = [f"{member}.npy" for member in REQUIRED_ARRAY_MEMBERS if f"{member}.npy" not in names]
        if missing:
            raise ValueError(f"{path} is missing required member(s): {', '.join(missing)}")
        for member in REQUIRED_ARRAY_MEMBERS:
            arrays[member] = _load_npy_array(archive.read(f"{member}.npy"))
    dense = DenseArrays(
        values=arrays["values"],
        observed_mask=arrays["observed_mask"],
        state_code=arrays["state_code"],
        timestamp_ms=arrays["timestamp_ms"],
        cell_ids=arrays["cell_ids"],
    )
    validate_dense_arrays(path, dense)
    return dense


def validate_dense_arrays(path: Path, dense: DenseArrays) -> None:
    rows = len(dense.timestamp_ms)
    cols = len(dense.cell_ids)
    for name, matrix in (("values", dense.values), ("observed_mask", dense.observed_mask), ("state_code", dense.state_code)):
        if len(matrix) != rows or any(len(row) != cols for row in matrix):
            raise ValueError(f"{path} member {name} shape does not match timestamp_ms/cell_ids")
    if dense.timestamp_ms != sorted(dense.timestamp_ms):
        raise ValueError(f"{path} timestamps must be chronological")


def artifact_path(input_dir: Path, split_name: str) -> Path:
    return input_dir / "arrays" / f"{split_name}.npz"


def load_required_artifacts(input_dir: Path) -> Dict[str, DenseArrays]:
    required = (TRAIN_SPLIT,) + EVALUATION_SPLITS
    return {split: load_materialized_npz(artifact_path(input_dir, split)) for split in required}


def iter_window_targets(dense: DenseArrays, input_length: int, horizon: int) -> Iterator[WindowTarget]:
    if input_length < 1:
        raise ValueError("input_length must be positive")
    if horizon != 1:
        raise ValueError("EXP-001 baseline sanity pipeline supports horizon 1 only")
    stop = len(dense.timestamp_ms) - input_length - horizon + 1
    for start in range(max(0, stop)):
        target_index = start + input_length + horizon - 1
        yield WindowTarget(start, target_index, dense.timestamp_ms[target_index])


def training_observed_statistics(train: DenseArrays) -> Dict[str, Optional[float]]:
    values = [
        float(value)
        for row_values, row_mask in zip(train.values, train.observed_mask)
        for value, mask in zip(row_values, row_mask)
        if mask
    ]
    if not values:
        return {"scope": "clean_train_observed_numeric_values_only", "observed_count": 0, "mean": None, "std": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "scope": "clean_train_observed_numeric_values_only",
        "observed_count": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
    }


def fit_historical_average(train: DenseArrays) -> List[Optional[float]]:
    means: List[Optional[float]] = []
    for col in range(len(train.cell_ids)):
        values = [
            float(row[col])
            for row, mask_row in zip(train.values, train.observed_mask)
            if mask_row[col]
        ]
        means.append((sum(values) / len(values)) if values else None)
    return means


def last_value_prediction(dense: DenseArrays, target: WindowTarget, cell_index: int, input_length: int) -> Optional[float]:
    for row_index in range(target.input_start + input_length - 1, target.input_start - 1, -1):
        if dense.observed_mask[row_index][cell_index]:
            return float(dense.values[row_index][cell_index])
    return None


def evaluate_last_value(dense: DenseArrays, input_length: int, horizon: int) -> Dict[str, Any]:
    accumulator = MetricAccumulator()
    window_count = 0
    for target in iter_window_targets(dense, input_length, horizon):
        window_count += 1
        target_values = dense.values[target.target_index]
        target_mask = dense.observed_mask[target.target_index]
        for col in range(len(dense.cell_ids)):
            accumulator.add(target_values[col], last_value_prediction(dense, target, col, input_length), target_mask[col])
    return metric_payload(accumulator, window_count)


def evaluate_historical_average(dense: DenseArrays, means: Sequence[Optional[float]], input_length: int, horizon: int) -> Dict[str, Any]:
    accumulator = MetricAccumulator()
    window_count = 0
    for target in iter_window_targets(dense, input_length, horizon):
        window_count += 1
        target_values = dense.values[target.target_index]
        target_mask = dense.observed_mask[target.target_index]
        for col, prediction in enumerate(means):
            accumulator.add(target_values[col], prediction, target_mask[col])
    return metric_payload(accumulator, window_count)


def metric_payload(accumulator: MetricAccumulator, window_count: int) -> Dict[str, Any]:
    payload = accumulator.metrics()
    payload.update(
        {
            "window_count": window_count,
            "valid_position_count": accumulator.valid_count,
            "unobserved_target_count": accumulator.unobserved_target_count,
            "unavailable_prediction_count": accumulator.unavailable_prediction_count,
            "smape_denominator": "2*abs(error)/(abs(target)+abs(prediction)); zero denominator contributes 0.0",
        }
    )
    if accumulator.valid_count == 0:
        payload["status"] = "unavailable_no_valid_target_prediction_pairs"
    else:
        payload["status"] = "ok"
    return payload


def degradation_ratio(pre_value: Optional[float], post_value: Optional[float]) -> Optional[float]:
    if pre_value is None or post_value is None or pre_value == 0.0:
        return None
    return post_value / pre_value


def public_input_label(input_dir: Path, label: Optional[str] = None) -> str:
    if label:
        return label
    path = Path(input_dir)
    if not path.is_absolute():
        return path.as_posix()
    return "local-only-exp001-artifacts"


def run_baselines(input_dir: Path, input_length: int = 144, horizon: int = 1, input_label: Optional[str] = None) -> Dict[str, Any]:
    artifacts = load_required_artifacts(input_dir)
    train = artifacts[TRAIN_SPLIT]
    historical_means = fit_historical_average(train)
    results: Dict[str, Any] = {
        "experiment_id": "EXP-001-stale-degradation-v0",
        "local_artifact_status": "local-only",
        "input_artifacts_root": public_input_label(input_dir, input_label),
        "holdout_status": "not_loaded_not_used_exp001_v0",
        "window": {"input_length": input_length, "horizon": horizon},
        "normalization_statistics": training_observed_statistics(train),
        "baselines": {},
    }
    for baseline in BASELINES:
        results["baselines"][baseline] = {}
        for split in EVALUATION_SPLITS:
            dense = artifacts[split]
            if baseline == "last_value":
                results["baselines"][baseline][split] = evaluate_last_value(dense, input_length, horizon)
            else:
                results["baselines"][baseline][split] = evaluate_historical_average(dense, historical_means, input_length, horizon)
        ratios = {}
        for metric in METRICS:
            pre = results["baselines"][baseline]["clean_test_predrift"][metric]
            post = results["baselines"][baseline]["drifted_test_postdrift"][metric]
            ratios[metric] = degradation_ratio(pre, post)
        results["baselines"][baseline]["degradation_ratio"] = ratios
    return results


def summary_rows(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for baseline in BASELINES:
        baseline_result = results["baselines"][baseline]
        for split in EVALUATION_SPLITS:
            metrics = baseline_result[split]
            row = {"baseline": baseline, "split": split, "kind": "metric"}
            row.update({key: metrics[key] for key in METRICS})
            row.update(
                {
                    "window_count": metrics["window_count"],
                    "valid_position_count": metrics["valid_position_count"],
                    "unobserved_target_count": metrics["unobserved_target_count"],
                    "unavailable_prediction_count": metrics["unavailable_prediction_count"],
                    "status": metrics["status"],
                }
            )
            rows.append(row)
        ratio_row = {"baseline": baseline, "split": "drifted_over_predrift", "kind": "degradation_ratio"}
        ratio_row.update(baseline_result["degradation_ratio"])
        ratio_row.update({"window_count": "", "valid_position_count": "", "unobserved_target_count": "", "unavailable_prediction_count": "", "status": ""})
        rows.append(ratio_row)
    return rows


def write_results(output_dir: Path, results: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics_summary.json", results)
    rows = summary_rows(results)
    fieldnames = [
        "baseline",
        "split",
        "kind",
        "masked_mae",
        "masked_rmse",
        "masked_smape",
        "window_count",
        "valid_position_count",
        "unobserved_target_count",
        "unavailable_prediction_count",
        "status",
    ]
    with (output_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
