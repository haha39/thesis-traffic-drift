"""Deterministic EXP-001 preprocessing and drift materialization helpers.

This module materializes clean/drifted local arrays only. It intentionally does
not create sliding-window training tensors, train models, evaluate metrics,
implement baselines, implement bridge methods, or make EXP-001 result claims.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import struct
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone as datetime_timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9 fallback
    ZoneInfo = None


SEGMENT_ORDER = ("train", "validation", "test", "holdout")
REQUIRED_MISSING_POLICY_STATUS = "approved_exp001_v0"
TEN_MINUTES_MS = 10 * 60 * 1000

STATE_NUMERIC_OBSERVED = 1
STATE_ALL_NULL = 2
STATE_ABSENT = 3
STATE_INVALID_ONLY = 4

STATE_CODE_BY_NAME = {
    "NUMERIC_OBSERVED": STATE_NUMERIC_OBSERVED,
    "ALL_NULL": STATE_ALL_NULL,
    "ABSENT": STATE_ABSENT,
    "INVALID_ONLY": STATE_INVALID_ONLY,
}
STATE_NAME_BY_CODE = {value: key for key, value in STATE_CODE_BY_NAME.items()}


@dataclass(frozen=True)
class SplitInterval:
    """End-exclusive time interval for an EXP-001 segment."""

    name: str
    start: datetime
    end: Optional[datetime]

    def contains(self, timestamp: datetime) -> bool:
        local = timestamp.astimezone(self.start.tzinfo) if timestamp.tzinfo else timestamp.replace(tzinfo=self.start.tzinfo)
        return local >= self.start and (self.end is None or local < self.end)


@dataclass(frozen=True)
class TrafficObservation:
    """Minimal synthetic/testable traffic observation."""

    timestamp: datetime
    cell_id: int
    value: Optional[float]


@dataclass
class GroupState:
    """Aggregated raw rows for one timestamp/cell pair."""

    raw_rows: int = 0
    numeric_rows: int = 0
    null_rows: int = 0
    invalid_rows: int = 0
    finite_sum: float = 0.0

    def add_null(self) -> None:
        self.raw_rows += 1
        self.null_rows += 1

    def add_invalid(self) -> None:
        self.raw_rows += 1
        self.invalid_rows += 1

    def add_numeric(self, value: float) -> None:
        self.raw_rows += 1
        self.numeric_rows += 1
        self.finite_sum += value

    @property
    def state_name(self) -> str:
        if self.numeric_rows > 0:
            return "NUMERIC_OBSERVED"
        if self.raw_rows > 0 and self.null_rows == self.raw_rows:
            return "ALL_NULL"
        return "INVALID_ONLY"

    @property
    def state_code(self) -> int:
        return STATE_CODE_BY_NAME[self.state_name]


@dataclass(frozen=True)
class DenseArrays:
    values: List[List[float]]
    observed_mask: List[List[int]]
    state_code: List[List[int]]
    timestamp_ms: List[int]
    cell_ids: List[int]


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "null":
        return None
    if value in ("true", "false"):
        return value == "true"
    if value.startswith('"') and value.endswith('"'):
        return ast.literal_eval(value)
    if value == "dataset_end":
        return value
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Parse the small YAML subset used by the public EXP-001 template."""

    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent % 2:
            raise ValueError(f"Unsupported indentation on line {line_number}")
        line = raw_line.strip()
        if ":" not in line:
            raise ValueError(f"Expected key/value YAML entry on line {line_number}")
        key, raw_value = line.split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid YAML nesting on line {line_number}")
        parent = stack[-1][1]
        key = key.strip()
        if raw_value.strip() == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(raw_value)
    return root


def load_exp001_config(path: Path) -> Dict[str, Any]:
    """Load and validate an EXP-001 config template."""

    config = parse_simple_yaml(Path(path).read_text(encoding="utf-8"))
    validate_exp001_config(config)
    return config


def validate_exp001_config(config: Mapping[str, Any]) -> None:
    """Validate decisions that this implementation is required to preserve."""

    expected = {
        ("experiment_id",): "EXP-001-stale-degradation-v0",
        ("dataset", "name"): "milan_internet",
        ("dataset", "raw_dir_env"): "MILAN_RAW_DIR",
        ("dataset", "timezone"): "Europe/Rome",
        ("dataset", "resolution_minutes"): 10,
        ("dataset", "expected_cells"): 10000,
        ("selection", "k"): 100,
        ("selection", "topk_source_segment"): "train",
        ("selection", "rank_metric"): "training_period_mean_traffic",
        ("selection", "tie_break"): "ascending_cell_id",
        ("drift", "rule"): "multiplicative_level_shift",
        ("drift", "factor"): 1.5,
        ("drift", "affected_fraction_of_topk"): 0.20,
        ("drift", "boundary_inclusive"): True,
        ("missing_policy", "modeling_policy"): "observed_numeric_with_explicit_mask",
        ("missing_policy", "status"): REQUIRED_MISSING_POLICY_STATUS,
        ("missing_policy", "placeholder_zero_requires_observed_mask_zero"): True,
        ("missing_policy", "normalization_fit_scope"): "training_period_observed_numeric_values_only",
    }
    for path, expected_value in expected.items():
        actual = _get(config, path)
        if actual != expected_value:
            joined = ".".join(path)
            raise ValueError(f"{joined} must be {expected_value!r}, got {actual!r}")


def _get(config: Mapping[str, Any], path: Sequence[str]) -> Any:
    cursor: Any = config
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise ValueError(f"Missing required config field {'.'.join(path)}")
        cursor = cursor[key]
    return cursor


def timezone(name: str):
    if ZoneInfo is None:
        if name == "Europe/Rome":
            return datetime_timezone(timedelta(hours=1), name="CET")
        raise RuntimeError("zoneinfo is required for non-Europe/Rome timezones")
    return ZoneInfo(name)


def parse_local_time(text: str, timezone_name: str = "Europe/Rome") -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=timezone(timezone_name))


def datetime_to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def ms_to_datetime(ms: int, timezone_name: str = "Europe/Rome") -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone(timezone_name))


def timestamp_range_ms(start: datetime, end: datetime) -> List[int]:
    return list(range(datetime_to_ms(start), datetime_to_ms(end), TEN_MINUTES_MS))


def construct_split_intervals(config: Mapping[str, Any]) -> List[SplitInterval]:
    tz_name = _get(config, ("dataset", "timezone"))
    splits = _get(config, ("splits",))
    intervals = []
    for name in SEGMENT_ORDER:
        split = _get(splits, (name,))
        end_value = _get(split, ("end",))
        end = None if end_value == "dataset_end" else parse_local_time(end_value, tz_name)
        intervals.append(SplitInterval(name=name, start=parse_local_time(_get(split, ("start",)), tz_name), end=end))
    return intervals


def segment_for_timestamp(timestamp: datetime, intervals: Sequence[SplitInterval]) -> Optional[str]:
    for interval in intervals:
        if interval.contains(timestamp):
            return interval.name
    return None


def drift_point(config: Mapping[str, Any]) -> datetime:
    return parse_local_time(_get(config, ("drift", "point")), _get(config, ("dataset", "timezone")))


def discover_milan_daily_files(raw_dir: Path) -> List[Path]:
    files = sorted(path for path in raw_dir.glob("sms-call-internet-mi-*.txt") if path.is_file())
    if not files:
        raise ValueError("No sms-call-internet-mi-*.txt files found in raw dir")
    return files


def parse_internet_value(text: str) -> Tuple[str, Optional[float]]:
    if text == "":
        return "null", None
    try:
        value = float(text)
    except ValueError:
        return "invalid", None
    if not math.isfinite(value):
        return "invalid", None
    return "numeric", value


def read_day_groups(path: Path, keep_cells: Optional[Set[int]] = None) -> Tuple[Dict[Tuple[int, int], GroupState], Dict[str, Any]]:
    """Parse one Milan daily file, summing Internet over countryCode rows."""

    groups: Dict[Tuple[int, int], GroupState] = defaultdict(GroupState)
    stats = {
        "file": path.name,
        "raw_rows": 0,
        "bad_rows": 0,
        "aggregated_timestamp_cell_records": 0,
        "min_timestamp_ms": None,
        "max_timestamp_ms": None,
    }
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stats["raw_rows"] += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 8:
                stats["bad_rows"] += 1
                continue
            try:
                cell_id = int(parts[0])
                timestamp_ms = int(parts[1])
            except ValueError:
                stats["bad_rows"] += 1
                continue
            stats["min_timestamp_ms"] = timestamp_ms if stats["min_timestamp_ms"] is None else min(stats["min_timestamp_ms"], timestamp_ms)
            stats["max_timestamp_ms"] = timestamp_ms if stats["max_timestamp_ms"] is None else max(stats["max_timestamp_ms"], timestamp_ms)
            if keep_cells is not None and cell_id not in keep_cells:
                continue
            group = groups[(timestamp_ms, cell_id)]
            kind, value = parse_internet_value(parts[7])
            if kind == "null":
                group.add_null()
            elif kind == "invalid":
                group.add_invalid()
            else:
                assert value is not None
                group.add_numeric(value)
    stats["aggregated_timestamp_cell_records"] = len(groups)
    return dict(groups), stats


def scan_training_means(
    files: Sequence[Path],
    intervals: Sequence[SplitInterval],
    expected_cells: int,
) -> Tuple[Dict[int, float], List[int], int, int, List[Dict[str, Any]]]:
    """First pass over raw files: train-only observed means and timestamp span."""

    train = next(interval for interval in intervals if interval.name == "train")
    train_start_ms = datetime_to_ms(train.start)
    train_end_ms = datetime_to_ms(train.end) if train.end else None
    sums = defaultdict(float)
    counts = defaultdict(int)
    all_timestamps: Set[int] = set()
    min_ts = None
    max_ts = None
    file_stats = []
    for path in files:
        train_groups = defaultdict(float)
        train_numeric_keys = set()
        stats = {
            "file": path.name,
            "raw_rows": 0,
            "bad_rows": 0,
            "aggregated_timestamp_cell_records": 0,
            "min_timestamp_ms": None,
            "max_timestamp_ms": None,
        }
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stats["raw_rows"] += 1
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 8:
                    stats["bad_rows"] += 1
                    continue
                try:
                    cell_id = int(parts[0])
                    ts_ms = int(parts[1])
                except ValueError:
                    stats["bad_rows"] += 1
                    continue
                all_timestamps.add(ts_ms)
                min_ts = ts_ms if min_ts is None else min(min_ts, ts_ms)
                max_ts = ts_ms if max_ts is None else max(max_ts, ts_ms)
                stats["min_timestamp_ms"] = ts_ms if stats["min_timestamp_ms"] is None else min(stats["min_timestamp_ms"], ts_ms)
                stats["max_timestamp_ms"] = ts_ms if stats["max_timestamp_ms"] is None else max(stats["max_timestamp_ms"], ts_ms)
                if not (1 <= cell_id <= expected_cells and train_start_ms <= ts_ms and (train_end_ms is None or ts_ms < train_end_ms)):
                    continue
                kind, value = parse_internet_value(parts[7])
                if kind != "numeric":
                    continue
                assert value is not None
                key = (ts_ms, cell_id)
                train_groups[key] += value
                train_numeric_keys.add(key)
        stats["aggregated_timestamp_cell_records"] = len(train_numeric_keys)
        file_stats.append(stats)
        for (_ts_ms, cell_id), value in train_groups.items():
            sums[cell_id] += value
            counts[cell_id] += 1
    if min_ts is None or max_ts is None:
        raise ValueError("No valid timestamped rows were found")
    means = {
        cell_id: (sums[cell_id] / counts[cell_id] if counts[cell_id] else float("-inf"))
        for cell_id in range(1, expected_cells + 1)
    }
    return means, sorted(all_timestamps), min_ts, max_ts, file_stats


def _as_observation(row: Any) -> TrafficObservation:
    if isinstance(row, TrafficObservation):
        return row
    if isinstance(row, Mapping):
        return TrafficObservation(timestamp=row["timestamp"], cell_id=int(row["cell_id"]), value=row.get("value"))
    timestamp, cell_id, value = row
    return TrafficObservation(timestamp=timestamp, cell_id=int(cell_id), value=value)


def rank_topk_cells(
    observations: Iterable[Any],
    intervals: Sequence[SplitInterval],
    k: int,
    candidate_cell_ids: Optional[Iterable[int]] = None,
) -> List[int]:
    """Rank Top-K cells by training-period mean traffic with ascending id ties."""

    train = next(interval for interval in intervals if interval.name == "train")
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    cells: Set[int] = set(candidate_cell_ids or [])
    for row in observations:
        obs = _as_observation(row)
        cells.add(obs.cell_id)
        if not train.contains(obs.timestamp) or obs.value is None:
            continue
        value = float(obs.value)
        if not math.isfinite(value):
            continue
        sums[obs.cell_id] = sums.get(obs.cell_id, 0.0) + value
        counts[obs.cell_id] = counts.get(obs.cell_id, 0) + 1
    if k < 1:
        raise ValueError("k must be positive")
    if not cells:
        return []

    def key(cell_id: int) -> Tuple[float, int]:
        count = counts.get(cell_id, 0)
        mean = sums[cell_id] / count if count else float("-inf")
        return (-mean, cell_id)

    return sorted(cells, key=key)[:k]


def rank_topk_from_means(training_means: Mapping[int, float], k: int) -> List[int]:
    if k < 1:
        raise ValueError("k must be positive")
    return sorted(training_means, key=lambda cell_id: (-training_means[cell_id], cell_id))[:k]


def select_affected_cells(
    topk_cells: Sequence[int],
    training_means: Mapping[int, float],
    fraction: float,
) -> List[int]:
    """Select top fraction of Top-K by training mean traffic."""

    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    count = int(len(topk_cells) * fraction)
    ranked = sorted(topk_cells, key=lambda cell_id: (-training_means.get(cell_id, float("-inf")), cell_id))
    return ranked[:count]


def training_means_by_cell(
    observations: Iterable[Any],
    intervals: Sequence[SplitInterval],
    cell_ids: Optional[Iterable[int]] = None,
) -> Dict[int, float]:
    train = next(interval for interval in intervals if interval.name == "train")
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    selected = set(cell_ids) if cell_ids is not None else None
    for row in observations:
        obs = _as_observation(row)
        if selected is not None and obs.cell_id not in selected:
            continue
        if not train.contains(obs.timestamp) or obs.value is None:
            continue
        value = float(obs.value)
        if not math.isfinite(value):
            continue
        sums[obs.cell_id] = sums.get(obs.cell_id, 0.0) + value
        counts[obs.cell_id] = counts.get(obs.cell_id, 0) + 1
    result = {cell_id: sums[cell_id] / counts[cell_id] for cell_id in sums}
    if selected is not None:
        for cell_id in selected:
            result.setdefault(cell_id, float("-inf"))
    return result


def is_post_drift_test(timestamp: datetime, intervals: Sequence[SplitInterval], point: datetime) -> bool:
    return segment_for_timestamp(timestamp, intervals) == "test" and timestamp.astimezone(point.tzinfo) >= point


def apply_multiplicative_level_shift(
    value: Optional[float],
    timestamp: datetime,
    cell_id: int,
    intervals: Sequence[SplitInterval],
    point: datetime,
    affected_cells: Iterable[int],
    factor: float,
) -> Optional[float]:
    """Apply EXP-001 drift only to affected cells in post-drift test samples."""

    if value is None:
        return None
    if is_post_drift_test(timestamp, intervals, point) and cell_id in set(affected_cells):
        return float(value) * factor
    return value


def transform_observation(
    observation: Any,
    intervals: Sequence[SplitInterval],
    point: datetime,
    affected_cells: Iterable[int],
    factor: float,
) -> TrafficObservation:
    obs = _as_observation(observation)
    return TrafficObservation(
        timestamp=obs.timestamp,
        cell_id=obs.cell_id,
        value=apply_multiplicative_level_shift(obs.value, obs.timestamp, obs.cell_id, intervals, point, affected_cells, factor),
    )


def build_dense_arrays(
    groups_by_ts: Mapping[int, Mapping[int, GroupState]],
    timestamp_ms: Sequence[int],
    cell_ids: Sequence[int],
    drift_timestamp_ms: Optional[int] = None,
    affected_cells: Optional[Iterable[int]] = None,
    factor: float = 1.0,
) -> DenseArrays:
    """Build dense values plus observation mask/state code for selected cells."""

    affected = set(affected_cells or [])
    values: List[List[float]] = []
    observed_mask: List[List[int]] = []
    state_code: List[List[int]] = []
    for ts_ms in timestamp_ms:
        row_values = []
        row_mask = []
        row_states = []
        present = groups_by_ts.get(ts_ms, {})
        drift_row = drift_timestamp_ms is not None and ts_ms >= drift_timestamp_ms
        for cell_id in cell_ids:
            group = present.get(cell_id)
            if group is None:
                row_values.append(0.0)
                row_mask.append(0)
                row_states.append(STATE_ABSENT)
            elif group.numeric_rows > 0:
                value = group.finite_sum
                if drift_row and cell_id in affected:
                    value *= factor
                row_values.append(float(value))
                row_mask.append(1)
                row_states.append(STATE_NUMERIC_OBSERVED)
            else:
                row_values.append(0.0)
                row_mask.append(0)
                row_states.append(group.state_code)
        values.append(row_values)
        observed_mask.append(row_mask)
        state_code.append(row_states)
    return DenseArrays(values, observed_mask, state_code, list(timestamp_ms), list(cell_ids))


def _npy_header(dtype: str, shape: Tuple[int, ...]) -> bytes:
    header = f"{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape}, }}"
    padding = 16 - ((10 + len(header) + 1) % 16)
    header_bytes = (header + (" " * padding) + "\n").encode("latin1")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes


def _flatten_2d(rows: Sequence[Sequence[Any]]) -> Iterable[Any]:
    for row in rows:
        for value in row:
            yield value


def npy_bytes(name: str, array: Any) -> bytes:
    """Serialize the small numeric array subset needed for EXP-001 NPZ files."""

    if name in ("values",):
        shape = (len(array), len(array[0]) if array else 0)
        payload = struct.pack("<" + "d" * (shape[0] * shape[1]), *[float(v) for v in _flatten_2d(array)]) if shape[0] and shape[1] else b""
        return _npy_header("<f8", shape) + payload
    if name in ("observed_mask", "state_code"):
        shape = (len(array), len(array[0]) if array else 0)
        payload = bytes(int(v) & 0xFF for v in _flatten_2d(array))
        return _npy_header("|u1", shape) + payload
    if name in ("timestamp_ms", "cell_ids"):
        shape = (len(array),)
        payload = struct.pack("<" + "q" * len(array), *[int(v) for v in array]) if array else b""
        return _npy_header("<i8", shape) + payload
    raise ValueError(f"Unsupported NPZ array {name!r}")


def write_npz(path: Path, arrays: DenseArrays) -> None:
    """Write compressed NumPy-compatible NPZ without requiring numpy at runtime."""

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("values.npy", npy_bytes("values", arrays.values))
        archive.writestr("observed_mask.npy", npy_bytes("observed_mask", arrays.observed_mask))
        archive.writestr("state_code.npy", npy_bytes("state_code", arrays.state_code))
        archive.writestr("timestamp_ms.npy", npy_bytes("timestamp_ms", arrays.timestamp_ms))
        archive.writestr("cell_ids.npy", npy_bytes("cell_ids", arrays.cell_ids))


def dense_shape(arrays: DenseArrays) -> List[int]:
    return [len(arrays.timestamp_ms), len(arrays.cell_ids)]


def observation_counts(arrays: DenseArrays) -> Dict[str, int]:
    counts = {name: 0 for name in STATE_CODE_BY_NAME}
    for row in arrays.state_code:
        for code in row:
            counts[STATE_NAME_BY_CODE[int(code)]] += 1
    return counts


def observed_values(arrays: DenseArrays) -> List[float]:
    values = []
    for value_row, mask_row in zip(arrays.values, arrays.observed_mask):
        for value, mask in zip(value_row, mask_row):
            if mask:
                values.append(float(value))
    return values


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
