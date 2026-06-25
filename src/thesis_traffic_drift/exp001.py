"""Deterministic EXP-001 preprocessing and drift transformation helpers.

This module intentionally stops before modeling-time missing/null handling,
tensor materialization, model training, metrics, baselines, bridge methods, or
retraining. Functions are pure enough to test with synthetic observations.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone as datetime_timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9 fallback
    ZoneInfo = None


SEGMENT_ORDER = ("train", "validation", "test", "holdout")
REQUIRED_MISSING_POLICY_STATUS = "human_decision_required"


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
        ("missing_policy", "modeling_policy"): None,
        ("missing_policy", "status"): REQUIRED_MISSING_POLICY_STATUS,
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
