"""MGSTC feasibility-gate preprocessing helpers.

This module implements a public-safe scaffold for MGSTC-aligned Milan data
materialization only. It does not implement the MGSTC model, online updates,
feedback-delay diagnostics, or any training/evaluation pipeline.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import os
import re
import struct
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone as datetime_timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9 fallback
    ZoneInfo = None


ACTIVITY_FIELDS = ("sms_in", "sms_out", "call_in", "call_out", "internet")
TEN_MINUTES_MS = 10 * 60 * 1000

STATE_NUMERIC_OBSERVED = 1
STATE_ALL_NULL = 2
STATE_ABSENT = 3
STATE_INVALID_ONLY = 4
STATE_PARTIAL_MISSING = 5

STATE_CODE_BY_NAME = {
    "NUMERIC_OBSERVED": STATE_NUMERIC_OBSERVED,
    "ALL_NULL": STATE_ALL_NULL,
    "ABSENT": STATE_ABSENT,
    "INVALID_ONLY": STATE_INVALID_ONLY,
    "PARTIAL_MISSING": STATE_PARTIAL_MISSING,
}
STATE_NAME_BY_CODE = {value: key for key, value in STATE_CODE_BY_NAME.items()}

ENV_VAR_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


@dataclass
class FieldAggregate:
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


@dataclass
class MilanCellGroup:
    fields: Dict[str, FieldAggregate] = field(default_factory=lambda: {name: FieldAggregate() for name in ACTIVITY_FIELDS})


@dataclass(frozen=True)
class DenseArrays:
    values: List[List[float]]
    observed_mask: List[List[int]]
    fully_observed_mask: List[List[int]]
    state_code: List[List[int]]
    component_observed_count: List[List[int]]
    component_missing_count: List[List[int]]
    all_components_observed_mask: List[List[int]]
    timestamp_ms: List[int]
    cell_ids: List[int]


@dataclass(frozen=True)
class NormalizationStats:
    minimum: float
    maximum: float
    observed_count: int


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "null":
        return None
    if value in ("true", "false"):
        return value == "true"
    if value.startswith('"') and value.endswith('"'):
        return ast.literal_eval(value)
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Parse the small YAML subset used by tracked config templates."""

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


def load_mgstc_gate_config(path: Path) -> Dict[str, Any]:
    config = parse_simple_yaml(Path(path).read_text(encoding="utf-8"))
    validate_mgstc_gate_config(config)
    return config


def validate_mgstc_gate_config(config: Mapping[str, Any]) -> None:
    required = {
        ("experiment_id",): "MGSTC-feasibility-gate-v0",
        ("grid_rows",): 100,
        ("grid_cols",): 100,
        ("center_rows",): 30,
        ("center_cols",): 30,
        ("center_row_start",): 40,
        ("center_col_start",): 40,
        ("resolution_minutes",): 10,
        ("train_days",): 5,
        ("validation_days",): 2,
        ("test_days",): 55,
        ("input_length",): 128,
        ("prediction_horizon",): 60,
        ("normalization",): "train_only_minmax",
    }
    for path, expected in required.items():
        actual = _get(config, path)
        if actual != expected:
            raise ValueError(f"{'.'.join(path)} must be {expected!r}, got {actual!r}")
    traffic_field = _get(config, ("traffic_field",))
    if traffic_field not in ("total", "internet"):
        raise ValueError("traffic_field must be 'total' or 'internet'")
    if traffic_field == "internet" and not bool(_get(config, ("allow_internet_fallback",))):
        raise ValueError("traffic_field='internet' requires allow_internet_fallback=true")
    training_mask = str(_get(config, ("total_training_mask",)))
    if training_mask != "fully_observed":
        raise ValueError("Builder-1 requires total_training_mask == 'fully_observed'")
    expected_timestamps = int(_get(config, ("expected_total_timestamps",)))
    total_days = int(_get(config, ("train_days",))) + int(_get(config, ("validation_days",))) + int(_get(config, ("test_days",)))
    expected_total_days = int(_get(config, ("expected_total_days",)))
    if expected_total_days != total_days:
        raise ValueError("expected_total_days must match train_days + validation_days + test_days")
    slots_per_day = slots_per_day_for_resolution(int(_get(config, ("resolution_minutes",))))
    if expected_timestamps != total_days * slots_per_day:
        raise ValueError("expected_total_timestamps must match train/validation/test days at the configured resolution")


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


def ms_to_datetime(ms: int, timezone_name: str = "Europe/Rome") -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone(timezone_name))


def datetime_to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def parse_activity_value(text: str) -> Tuple[str, Optional[float]]:
    if text == "":
        return "null", None
    try:
        value = float(text)
    except ValueError:
        return "invalid", None
    if not math.isfinite(value):
        return "invalid", None
    return "numeric", value


def discover_milan_daily_files(raw_dir: Path) -> List[Path]:
    files = sorted(path for path in raw_dir.glob("sms-call-internet-mi-*.txt") if path.is_file())
    if not files:
        raise ValueError("No sms-call-internet-mi-*.txt files found in raw dir")
    return files


def read_day_field_groups(
    path: Path,
    keep_cells: Optional[Set[int]] = None,
) -> Tuple[Dict[Tuple[int, int], MilanCellGroup], Set[int], Dict[str, Any]]:
    """Parse one Milan daily file and aggregate activity fields by timestamp/cell."""

    groups: Dict[Tuple[int, int], MilanCellGroup] = defaultdict(MilanCellGroup)
    timestamps: Set[int] = set()
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
            timestamps.add(timestamp_ms)
            stats["min_timestamp_ms"] = timestamp_ms if stats["min_timestamp_ms"] is None else min(stats["min_timestamp_ms"], timestamp_ms)
            stats["max_timestamp_ms"] = timestamp_ms if stats["max_timestamp_ms"] is None else max(stats["max_timestamp_ms"], timestamp_ms)
            if keep_cells is not None and cell_id not in keep_cells:
                continue
            group = groups[(timestamp_ms, cell_id)]
            for field_name, raw_value in zip(ACTIVITY_FIELDS, parts[3:8]):
                kind, value = parse_activity_value(raw_value)
                aggregate = group.fields[field_name]
                if kind == "null":
                    aggregate.add_null()
                elif kind == "invalid":
                    aggregate.add_invalid()
                else:
                    assert value is not None
                    aggregate.add_numeric(value)
    stats["aggregated_timestamp_cell_records"] = len(groups)
    return dict(groups), timestamps, stats


def merge_cell_group(target: MilanCellGroup, incoming: MilanCellGroup) -> None:
    for field_name in ACTIVITY_FIELDS:
        target_field = target.fields[field_name]
        source_field = incoming.fields[field_name]
        target_field.raw_rows += source_field.raw_rows
        target_field.numeric_rows += source_field.numeric_rows
        target_field.null_rows += source_field.null_rows
        target_field.invalid_rows += source_field.invalid_rows
        target_field.finite_sum += source_field.finite_sum


def load_selected_milan_groups(
    files: Sequence[Path],
    keep_cells: Set[int],
) -> Tuple[Dict[int, Dict[int, MilanCellGroup]], List[int], int, int, List[Dict[str, Any]]]:
    groups_by_ts: Dict[int, Dict[int, MilanCellGroup]] = defaultdict(dict)
    all_timestamps: Set[int] = set()
    min_ts = None
    max_ts = None
    file_stats = []
    for path in files:
        groups, timestamps, stats = read_day_field_groups(path, keep_cells=keep_cells)
        file_stats.append(stats)
        all_timestamps.update(timestamps)
        if stats["min_timestamp_ms"] is not None:
            min_ts = stats["min_timestamp_ms"] if min_ts is None else min(min_ts, stats["min_timestamp_ms"])
            max_ts = stats["max_timestamp_ms"] if max_ts is None else max(max_ts, stats["max_timestamp_ms"])
        for (timestamp_ms, cell_id), group in groups.items():
            existing = groups_by_ts[timestamp_ms].get(cell_id)
            if existing is None:
                groups_by_ts[timestamp_ms][cell_id] = group
            else:
                merge_cell_group(existing, group)
    if min_ts is None or max_ts is None:
        raise ValueError("No valid timestamped rows were found")
    return groups_by_ts, sorted(all_timestamps), min_ts, max_ts, file_stats


def select_center_cell_ids(
    grid_rows: int,
    grid_cols: int,
    center_rows: int,
    center_cols: int,
    row_start: Optional[int] = None,
    col_start: Optional[int] = None,
) -> List[int]:
    if center_rows < 1 or center_cols < 1:
        raise ValueError("center_rows and center_cols must be positive")
    if center_rows > grid_rows or center_cols > grid_cols:
        raise ValueError("center selection cannot exceed the grid size")
    if row_start is None:
        row_start = (grid_rows - center_rows) // 2
    if col_start is None:
        col_start = (grid_cols - center_cols) // 2
    if row_start < 0 or col_start < 0:
        raise ValueError("center selection start cannot be negative")
    row_end = row_start + center_rows
    col_end = col_start + center_cols
    if row_end > grid_rows or col_end > grid_cols:
        raise ValueError("configured center selection exceeds the grid bounds")
    cell_ids = []
    for row_index in range(row_start, row_end):
        for col_index in range(col_start, col_end):
            cell_ids.append(row_index * grid_cols + col_index + 1)
    return cell_ids


def cell_position(cell_id: int, grid_cols: int) -> Tuple[int, int]:
    zero_based = cell_id - 1
    return zero_based // grid_cols, zero_based % grid_cols


def slots_per_day_for_resolution(resolution_minutes: int) -> int:
    if resolution_minutes < 1 or 1440 % resolution_minutes:
        raise ValueError("resolution_minutes must evenly divide 1440")
    return 1440 // resolution_minutes


def build_chronological_splits(
    timestamps: Sequence[int],
    train_days: int,
    validation_days: int,
    test_days: int,
    resolution_minutes: int,
) -> Dict[str, List[int]]:
    ordered = sorted(set(int(ts) for ts in timestamps))
    slots_per_day = slots_per_day_for_resolution(resolution_minutes)
    expected_total = (train_days + validation_days + test_days) * slots_per_day
    if len(ordered) != expected_total:
        raise ValueError(f"Expected {expected_total} timestamps, found {len(ordered)}")
    step_ms = resolution_minutes * 60 * 1000
    for earlier, later in zip(ordered, ordered[1:]):
        if later - earlier != step_ms:
            raise ValueError("Timestamps must form a continuous regular series at the configured resolution")
    train_count = train_days * slots_per_day
    validation_count = validation_days * slots_per_day
    return {
        "train": ordered[:train_count],
        "validation": ordered[train_count:train_count + validation_count],
        "test": ordered[train_count + validation_count:],
    }


def _field_state_name(aggregate: FieldAggregate) -> str:
    if aggregate.numeric_rows > 0:
        return "NUMERIC_OBSERVED"
    if aggregate.raw_rows > 0 and aggregate.null_rows == aggregate.raw_rows:
        return "ALL_NULL"
    return "INVALID_ONLY"


def group_value_and_state(group: Optional[MilanCellGroup], traffic_field: str) -> Tuple[float, int, int, int, int]:
    if group is None:
        return 0.0, STATE_ABSENT, 0, 1 if traffic_field == "internet" else len(ACTIVITY_FIELDS), 0
    if traffic_field == "internet":
        aggregate = group.fields["internet"]
        state_name = _field_state_name(aggregate)
        if state_name == "NUMERIC_OBSERVED":
            return float(aggregate.finite_sum), STATE_NUMERIC_OBSERVED, 1, 0, 1
        if state_name == "ALL_NULL":
            return 0.0, STATE_ALL_NULL, 0, 1, 0
        return 0.0, STATE_INVALID_ONLY, 0, 1, 0
    assert traffic_field == "total"
    aggregates = [group.fields[field_name] for field_name in ACTIVITY_FIELDS]
    observed_count = sum(1 for aggregate in aggregates if aggregate.numeric_rows > 0)
    missing_count = len(ACTIVITY_FIELDS) - observed_count
    all_observed = 1 if observed_count == len(ACTIVITY_FIELDS) else 0
    if observed_count > 0:
        total_value = sum(aggregate.finite_sum for aggregate in aggregates if aggregate.numeric_rows > 0)
        state = STATE_NUMERIC_OBSERVED if all_observed else STATE_PARTIAL_MISSING
        return float(total_value), state, observed_count, missing_count, all_observed
    if all(aggregate.raw_rows > 0 and aggregate.null_rows == aggregate.raw_rows for aggregate in aggregates):
        return 0.0, STATE_ALL_NULL, 0, len(ACTIVITY_FIELDS), 0
    if all(_field_state_name(aggregate) == "INVALID_ONLY" for aggregate in aggregates):
        return 0.0, STATE_INVALID_ONLY, 0, len(ACTIVITY_FIELDS), 0
    return 0.0, STATE_PARTIAL_MISSING, 0, len(ACTIVITY_FIELDS), 0


def build_dense_arrays(
    groups_by_ts: Mapping[int, Mapping[int, MilanCellGroup]],
    timestamp_ms: Sequence[int],
    cell_ids: Sequence[int],
    traffic_field: str,
) -> DenseArrays:
    values: List[List[float]] = []
    observed_mask: List[List[int]] = []
    fully_observed_mask: List[List[int]] = []
    state_code: List[List[int]] = []
    component_observed_count: List[List[int]] = []
    component_missing_count: List[List[int]] = []
    all_components_observed_mask: List[List[int]] = []
    for ts_ms in timestamp_ms:
        row_values = []
        row_mask = []
        row_fully_observed_mask = []
        row_states = []
        row_component_observed = []
        row_component_missing = []
        row_all_components_observed = []
        present = groups_by_ts.get(ts_ms, {})
        for cell_id in cell_ids:
            value, state, observed_count, missing_count, all_observed = group_value_and_state(present.get(cell_id), traffic_field)
            is_observed = 1 if observed_count > 0 else 0
            row_values.append(value if is_observed else 0.0)
            row_mask.append(is_observed)
            row_fully_observed_mask.append(all_observed)
            row_states.append(state)
            row_component_observed.append(observed_count)
            row_component_missing.append(missing_count)
            row_all_components_observed.append(all_observed)
        values.append(row_values)
        observed_mask.append(row_mask)
        fully_observed_mask.append(row_fully_observed_mask)
        state_code.append(row_states)
        component_observed_count.append(row_component_observed)
        component_missing_count.append(row_component_missing)
        all_components_observed_mask.append(row_all_components_observed)
    return DenseArrays(
        values,
        observed_mask,
        fully_observed_mask,
        state_code,
        component_observed_count,
        component_missing_count,
        all_components_observed_mask,
        list(timestamp_ms),
        list(cell_ids),
    )


def compute_train_only_minmax(arrays: DenseArrays, mask_name: str = "observed_mask") -> NormalizationStats:
    observed = observed_values(arrays, mask_name=mask_name)
    if not observed:
        raise ValueError(f"Training split has no values selected by {mask_name} for normalization")
    return NormalizationStats(minimum=min(observed), maximum=max(observed), observed_count=len(observed))


def apply_train_only_minmax(arrays: DenseArrays, stats: NormalizationStats) -> DenseArrays:
    scale = stats.maximum - stats.minimum
    values: List[List[float]] = []
    for value_row, mask_row in zip(arrays.values, arrays.observed_mask):
        normalized_row = []
        for value, mask in zip(value_row, mask_row):
            if not mask:
                normalized_row.append(0.0)
            elif scale == 0:
                normalized_row.append(0.0)
            else:
                normalized_row.append((float(value) - stats.minimum) / scale)
        values.append(normalized_row)
    return DenseArrays(
        values,
        [list(row) for row in arrays.observed_mask],
        [list(row) for row in arrays.fully_observed_mask],
        [list(row) for row in arrays.state_code],
        [list(row) for row in arrays.component_observed_count],
        [list(row) for row in arrays.component_missing_count],
        [list(row) for row in arrays.all_components_observed_mask],
        list(arrays.timestamp_ms),
        list(arrays.cell_ids),
    )


def observed_values(arrays: DenseArrays, mask_name: str = "observed_mask") -> List[float]:
    if mask_name == "observed_mask":
        mask_rows = arrays.observed_mask
    elif mask_name == "fully_observed_mask":
        mask_rows = arrays.fully_observed_mask
    elif mask_name == "all_components_observed_mask":
        mask_rows = arrays.all_components_observed_mask
    else:
        raise ValueError(f"Unsupported mask_name {mask_name!r}")
    values = []
    for value_row, mask_row in zip(arrays.values, mask_rows):
        for value, mask in zip(value_row, mask_row):
            if mask:
                values.append(float(value))
    return values


def dense_shape(arrays: DenseArrays) -> List[int]:
    return [len(arrays.timestamp_ms), len(arrays.cell_ids)]


def observation_counts(arrays: DenseArrays) -> Dict[str, int]:
    counts = {name: 0 for name in STATE_CODE_BY_NAME}
    for row in arrays.state_code:
        for code in row:
            counts[STATE_NAME_BY_CODE[int(code)]] += 1
    return counts


def safe_config_snapshot(config: Mapping[str, Any]) -> Dict[str, Any]:
    snapshot = json.loads(json.dumps(config))
    if "raw_milan_daily_dir" in snapshot:
        snapshot["raw_milan_daily_dir"] = "${MILAN_RAW_DIR}"
    return snapshot


def resolve_env_path(value: str) -> Tuple[Path, str]:
    match = ENV_VAR_PATTERN.fullmatch(value)
    if not match:
        return Path(value), "config.raw_milan_daily_dir"
    env_name = match.group(1)
    env_value = os.environ.get(env_name)
    if not env_value:
        raise ValueError(f"Environment variable {env_name} is required by raw_milan_daily_dir")
    return Path(env_value), env_name


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
    if name == "values":
        shape = (len(array), len(array[0]) if array else 0)
        payload = struct.pack("<" + "d" * (shape[0] * shape[1]), *[float(v) for v in _flatten_2d(array)]) if shape[0] and shape[1] else b""
        return _npy_header("<f8", shape) + payload
    if name in ("observed_mask", "fully_observed_mask", "state_code", "component_observed_count", "component_missing_count", "all_components_observed_mask"):
        shape = (len(array), len(array[0]) if array else 0)
        payload = bytes(int(v) & 0xFF for v in _flatten_2d(array))
        return _npy_header("|u1", shape) + payload
    if name in ("timestamp_ms", "cell_ids"):
        shape = (len(array),)
        payload = struct.pack("<" + "q" * len(array), *[int(v) for v in array]) if array else b""
        return _npy_header("<i8", shape) + payload
    raise ValueError(f"Unsupported NPZ array {name!r}")


def write_npz(path: Path, arrays: DenseArrays) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("values.npy", npy_bytes("values", arrays.values))
        archive.writestr("observed_mask.npy", npy_bytes("observed_mask", arrays.observed_mask))
        archive.writestr("fully_observed_mask.npy", npy_bytes("fully_observed_mask", arrays.fully_observed_mask))
        archive.writestr("state_code.npy", npy_bytes("state_code", arrays.state_code))
        archive.writestr("component_observed_count.npy", npy_bytes("component_observed_count", arrays.component_observed_count))
        archive.writestr("component_missing_count.npy", npy_bytes("component_missing_count", arrays.component_missing_count))
        archive.writestr("all_components_observed_mask.npy", npy_bytes("all_components_observed_mask", arrays.all_components_observed_mask))
        archive.writestr("timestamp_ms.npy", npy_bytes("timestamp_ms", arrays.timestamp_ms))
        archive.writestr("cell_ids.npy", npy_bytes("cell_ids", arrays.cell_ids))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def output_file_inventory(output_dir: Path, extra_files: Optional[Sequence[str]] = None) -> List[str]:
    files = {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    if extra_files:
        files.update(str(path) for path in extra_files)
    return sorted(files)


def materialize_mgstc_gate(
    config: Mapping[str, Any],
    raw_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    raw_source: str = "unknown",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    files = discover_milan_daily_files(raw_dir)
    center_cells = select_center_cell_ids(
        int(config["grid_rows"]),
        int(config["grid_cols"]),
        int(config["center_rows"]),
        int(config["center_cols"]),
        row_start=int(config["center_row_start"]),
        col_start=int(config["center_col_start"]),
    )
    groups_by_ts, all_timestamps, min_ts, max_ts, file_stats = load_selected_milan_groups(files, keep_cells=set(center_cells))
    splits = build_chronological_splits(
        all_timestamps,
        int(config["train_days"]),
        int(config["validation_days"]),
        int(config["test_days"]),
        int(config["resolution_minutes"]),
    )

    traffic_field = str(config["traffic_field"])
    raw_arrays = {name: build_dense_arrays(groups_by_ts, split_timestamps, center_cells, traffic_field) for name, split_timestamps in splits.items()}
    normalization_mask = "fully_observed_mask" if traffic_field == "total" else "observed_mask"
    stats = compute_train_only_minmax(raw_arrays["train"], mask_name=normalization_mask)
    normalized_arrays = {name: apply_train_only_minmax(arrays, stats) for name, arrays in raw_arrays.items()}

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise ValueError("output_dir is non-empty; pass overwrite=True to replace local artifacts")
        for child in sorted(output_dir.iterdir(), reverse=True):
            if child.is_dir():
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file():
                        nested.unlink()
                    elif nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in ("indices", "arrays", "summaries"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    array_targets = {
        "train": "arrays/clean_train.npz",
        "validation": "arrays/clean_validation.npz",
        "test": "arrays/clean_test.npz",
    }
    for split_name, relative in array_targets.items():
        write_npz(output_dir / relative, normalized_arrays[split_name])

    cell_rows = []
    for position, cell_id in enumerate(center_cells):
        row_index, col_index = cell_position(cell_id, int(config["grid_cols"]))
        cell_rows.append(
            {
                "cell_position": position,
                "cell_id": cell_id,
                "row_index": row_index,
                "col_index": col_index,
            }
        )
    write_csv_rows(output_dir / "indices" / "cell_index.csv", cell_rows, ["cell_position", "cell_id", "row_index", "col_index"])

    timestamp_rows = []
    split_rows = []
    timezone_name = str(config["timezone"])
    resolution_ms = int(config["resolution_minutes"]) * 60 * 1000
    for split_name, split_timestamps in splits.items():
        split_rows.append(
            {
                "split": split_name,
                "start": ms_to_datetime(split_timestamps[0], timezone_name).isoformat(),
                "end_exclusive": ms_to_datetime(split_timestamps[-1] + resolution_ms, timezone_name).isoformat(),
                "samples": len(split_timestamps),
            }
        )
        for position, timestamp_ms in enumerate(split_timestamps):
            timestamp_rows.append(
                {
                    "split": split_name,
                    "split_position": position,
                    "timestamp_ms": timestamp_ms,
                    "timestamp_local": ms_to_datetime(timestamp_ms, timezone_name).isoformat(),
                }
            )
    write_csv_rows(output_dir / "indices" / "timestamp_index.csv", timestamp_rows, ["split", "split_position", "timestamp_ms", "timestamp_local"])
    write_csv_rows(output_dir / "indices" / "split_index.csv", split_rows, ["split", "start", "end_exclusive", "samples"])

    validation_checks = {
        "center_cell_count": len(center_cells),
        "train_samples": len(splits["train"]),
        "validation_samples": len(splits["validation"]),
        "test_samples": len(splits["test"]),
        "placeholder_zero_mask_check": all(
            value == 0.0
            for dense in normalized_arrays.values()
            for value_row, mask_row in zip(dense.values, dense.observed_mask)
            for value, mask in zip(value_row, mask_row)
            if not mask
        ),
        "fully_observed_mask_subset_check": all(
            fully <= observed
            for dense in normalized_arrays.values()
            for fully_row, observed_row in zip(dense.fully_observed_mask, dense.observed_mask)
            for fully, observed in zip(fully_row, observed_row)
        ),
        "component_count_consistency_check": all(
            observed + missing == (1 if traffic_field == "internet" else len(ACTIVITY_FIELDS))
            for dense in normalized_arrays.values()
            for observed_row, missing_row in zip(dense.component_observed_count, dense.component_missing_count)
            for observed, missing in zip(observed_row, missing_row)
        ),
        "normalization_source": "train_only_minmax",
        "normalization_mask": normalization_mask,
        "traffic_field": traffic_field,
    }
    write_json(output_dir / "summaries" / "validation_checks.json", validation_checks)

    observation_summary_rows = []
    for split_name, dense in normalized_arrays.items():
        counts = observation_counts(dense)
        observation_summary_rows.append(
            {
                "split": split_name,
                "expected_pairs": len(dense.timestamp_ms) * len(dense.cell_ids),
                **counts,
            }
        )
    write_csv_rows(
        output_dir / "summaries" / "observation_counts_by_split.csv",
        observation_summary_rows,
        ["split", "expected_pairs", "NUMERIC_OBSERVED", "ALL_NULL", "ABSENT", "INVALID_ONLY", "PARTIAL_MISSING"],
    )

    metadata = {
        "experiment_id": config["experiment_id"],
        "scaffold_only": True,
        "not_full_mgstc_reproduction": True,
        "paper_alignment_status": "feasibility_scaffold_with_assumptions",
        "normalization_policy_not_paper_verified": True,
        "fig6_inferred_900_region": True,
        "square_id_mapping_assumption": "row_major_unverified",
        "local_artifact_status": "local-only",
        "config_snapshot": safe_config_snapshot(config),
        "source_summary": {
            "raw_dir_source": raw_source,
            "daily_file_count": len(files),
            "file_names": [path.name for path in files],
            "timestamp_min": ms_to_datetime(min_ts, timezone_name).isoformat(),
            "timestamp_max": ms_to_datetime(max_ts, timezone_name).isoformat(),
            "timestamp_count_present_in_raw": len(all_timestamps),
        },
        "selection": {
            "grid_rows": config["grid_rows"],
            "grid_cols": config["grid_cols"],
            "center_rows": config["center_rows"],
            "center_cols": config["center_cols"],
            "center_row_start": config["center_row_start"],
            "center_col_start": config["center_col_start"],
            "cell_count": len(center_cells),
            "selection_rule": "deterministic_fig6_inferred_30x30_region_under_row_major_mapping",
            "square_id_mapping": {
                "assumption": "row_major_unverified",
                "row_formula": "(square_id - 1) // 100",
                "col_formula": "(square_id - 1) % 100",
                "selected_region_definition": "rows_40_70_cols_40_70_fig6_inferred_30x30_region_under_row_major_mapping",
                "range_source": "fig6_visual_inference",
                "revisit_condition": "revisit_if_future_source_evidence_contradicts_row_major_mapping_or_fig6_inferred_range",
            },
        },
        "traffic_field": traffic_field,
        "traffic_field_note": (
            "MGSTC-target total traffic scaffold across five Milan activity fields; not a faithful MGSTC reproduction"
            if traffic_field == "total"
            else "Internet-only fallback or ablation; not full MGSTC reproduction"
        ),
        "split_boundaries": split_rows,
        "protocol": {
            "input_length": config["input_length"],
            "prediction_horizon": config["prediction_horizon"],
        },
        "normalization": {
            "policy": config["normalization"],
            "fit_split": "train",
            "fit_mask": normalization_mask,
            "min_value": stats.minimum,
            "max_value": stats.maximum,
            "observed_count": stats.observed_count,
            "total_training_mask_contract": "fully_observed_only_for_builder1",
        },
        "missing_policy": config["missing_policy"],
        "partial_total_policy": {
            "value_semantics": "lower_bound_sum_of_available_numeric_components",
            "observed_mask_semantics": "at_least_one_numeric_component",
            "fully_observed_mask_semantics": "all_five_components_numeric",
            "default_safe_downstream_mask_for_total": "fully_observed_mask",
            "partial_totals_not_clean_training_by_default": True,
        },
        "component_observation_semantics": {
            "total": {
                "observed_definition": "at_least_one_numeric_component",
                "value_definition": "sum_of_available_numeric_components_only",
                "component_observed_count_definition": "number_of_numeric_components_among_five_fields",
                "component_missing_count_definition": "number_of_unavailable_components_among_five_fields",
                "all_components_observed_mask_definition": "1_only_when_all_five_components_are_numeric",
                "fully_observed_mask_definition": "default_safe_mask_for_normalization_and_clean_total_training",
            },
            "internet": {
                "observed_definition": "internet_component_numeric",
                "component_count_scope": "internet_component_only",
            },
        },
        "state_codes": STATE_CODE_BY_NAME,
        "file_parse_summary": file_stats,
    }
    write_json(output_dir / "metadata.json", metadata)

    arrays_manifest = {
        relative: {
            "split": split_name,
            "shape": dense_shape(normalized_arrays[split_name]),
            "value_transform": config["normalization"],
            "traffic_field": traffic_field,
            "dtypes": {
                "values": "float64",
                "observed_mask": "uint8",
                "fully_observed_mask": "uint8",
                "state_code": "uint8",
                "component_observed_count": "uint8",
                "component_missing_count": "uint8",
                "all_components_observed_mask": "uint8",
                "timestamp_ms": "int64",
                "cell_ids": "int64",
            },
        }
        for split_name, relative in array_targets.items()
    }
    manifest = {
        "experiment_id": config["experiment_id"],
        "scaffold_only": True,
        "not_full_mgstc_reproduction": True,
        "paper_alignment_status": "feasibility_scaffold_with_assumptions",
        "normalization_policy_not_paper_verified": True,
        "fig6_inferred_900_region": True,
        "square_id_mapping_assumption": "row_major_unverified",
        "local_artifact_status": "local-only",
        "row_ordering": "timestamp ascending within each split",
        "column_ordering": "fig6-inferred 900-region cells in deterministic row-major rows_40_70_cols_40_70 inferred range order",
        "partial_total_policy": {
            "default_safe_downstream_mask_for_total": "fully_observed_mask",
            "partial_totals_not_clean_training_by_default": True,
        },
        "arrays": arrays_manifest,
        "files": output_file_inventory(output_dir, extra_files=("manifest.json",)),
    }
    write_json(output_dir / "manifest.json", manifest)
    return metadata, manifest
