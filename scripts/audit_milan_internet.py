#!/usr/bin/env python3
"""Read-only Milan Internet traffic data audit.

The corrected v2 audit preserves raw and aggregated observation states. It does
not train models, inject drift, choose K, approve a split, or select a final
missing-data policy.
"""

import argparse
import csv
import json
import math
import os
import struct
import sys
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9 fallback
    ZoneInfo = None


TEN_MINUTES_MS = 10 * 60 * 1000
EXPECTED_CELLS = 10000
TZ_NAME = "Europe/Rome"
TRAIN_START = "2013-11-01 00:00"
TRAIN_END = "2013-11-29 00:00"
VAL_START = "2013-11-29 00:00"
VAL_END = "2013-12-06 00:00"
TEST_START = "2013-12-06 00:00"
TEST_END = "2013-12-20 00:00"
HOLDOUT_START = "2013-12-20 00:00"
DRIFT_POINT = "2013-12-13 00:00"
TOPK_VALUES = (60, 80, 100)
POLICIES = ("P1_OBSERVED_ONLY", "P2_NULL_ZERO_ABSENT_MASKED", "P3_LEGACY_ZERO_FILL")


@dataclass(frozen=True)
class Segment:
    name: str
    start_ms: int
    end_ms: Optional[int]


@dataclass
class GroupState:
    raw_rows: int = 0
    numeric_rows: int = 0
    positive_rows: int = 0
    explicit_zero_rows: int = 0
    null_rows: int = 0
    negative_rows: int = 0
    nonfinite_rows: int = 0
    malformed_rows: int = 0
    finite_sum: float = 0.0

    def add_null(self) -> None:
        self.raw_rows += 1
        self.null_rows += 1

    def add_malformed(self) -> None:
        self.raw_rows += 1
        self.malformed_rows += 1

    def add_nonfinite(self) -> None:
        self.raw_rows += 1
        self.nonfinite_rows += 1

    def add_numeric(self, value: float) -> None:
        self.raw_rows += 1
        self.numeric_rows += 1
        self.finite_sum += value
        if value > 0.0:
            self.positive_rows += 1
        elif value == 0.0:
            self.explicit_zero_rows += 1
        else:
            self.negative_rows += 1

    @property
    def state(self) -> str:
        if self.numeric_rows > 0:
            return "NUMERIC_OBSERVED"
        if self.raw_rows > 0 and self.null_rows == self.raw_rows:
            return "ALL_NULL"
        return "INVALID_ONLY"

    @property
    def mixed_numeric_plus_null(self) -> bool:
        return self.numeric_rows > 0 and self.null_rows > 0


@dataclass
class CellPolicyStats:
    observed_sum: float = 0.0
    present_count: int = 0
    numeric_count: int = 0
    all_null_count: int = 0
    absent_count: int = 0
    explicit_zero_count: int = 0


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def timezone_obj():
    if ZoneInfo is None:
        return timezone(timedelta(hours=1), name="CET")
    return ZoneInfo(TZ_NAME)


def timezone_assumption() -> str:
    if ZoneInfo is None:
        return "fixed CET UTC+01 fallback for 2013-11 to 2014-01 Milan data"
    return TZ_NAME


def local_ms(text: str) -> int:
    dt = datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=timezone_obj())
    return int(dt.timestamp() * 1000)


def ms_to_local(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone_obj())


def ms_to_iso(ms: Optional[int]) -> Optional[str]:
    if ms is None:
        return None
    return ms_to_local(ms).isoformat()


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    mu = sum(values) / len(values)
    var = max(0.0, sum((x - mu) * (x - mu) for x in values) / len(values))
    return mu, math.sqrt(var)


def ks_distance(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if not a or not b:
        return None
    xs = sorted(a)
    ys = sorted(b)
    i = j = 0
    best = 0.0
    while i < len(xs) or j < len(ys):
        if j >= len(ys) or (i < len(xs) and xs[i] <= ys[j]):
            x = xs[i]
            while i < len(xs) and xs[i] == x:
                i += 1
        else:
            x = ys[j]
            while j < len(ys) and ys[j] == x:
                j += 1
        while i < len(xs) and xs[i] <= x:
            i += 1
        while j < len(ys) and ys[j] <= x:
            j += 1
        best = max(best, abs(i / len(xs) - j / len(ys)))
    return best


def safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den in (None, 0, 0.0):
        return None
    return num / den


def spearman_rank_correlation(rank_a: Dict[int, int], rank_b: Dict[int, int]) -> Optional[float]:
    common = sorted(set(rank_a) & set(rank_b))
    n = len(common)
    if n < 2:
        return None
    diffsq = sum((rank_a[c] - rank_b[c]) ** 2 for c in common)
    return 1.0 - (6.0 * diffsq) / (n * (n * n - 1))


def discover_daily_files(raw_dir: Path) -> List[Path]:
    files = sorted(p for p in raw_dir.glob("sms-call-internet-mi-*.txt") if p.is_file())
    if not files:
        fail("No sms-call-internet-mi-*.txt files found in raw directory.")
    return files


def load_grid_summary(raw_dir: Path) -> Dict[str, object]:
    grid = raw_dir / "milano-grid.geojson"
    if not grid.exists():
        return {"available": False, "warning": "milano-grid.geojson not found"}
    payload = json.loads(grid.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    cell_ids = []
    centroids = {}
    for feature in features:
        props = feature.get("properties", {})
        if "cellId" not in props:
            continue
        cell_id = int(props["cellId"])
        cell_ids.append(cell_id)
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [])
        if geom.get("type") == "Polygon" and coords:
            ring = coords[0]
            pts = ring[:-1] if ring and ring[0] == ring[-1] else ring
            if pts:
                centroids[cell_id] = (
                    sum(float(x) for x, _ in pts) / len(pts),
                    sum(float(y) for _, y in pts) / len(pts),
                )
    return {
        "available": True,
        "feature_count": len(features),
        "cell_id_min": min(cell_ids) if cell_ids else None,
        "cell_id_max": max(cell_ids) if cell_ids else None,
        "cell_ids_contiguous_1_to_10000": sorted(cell_ids) == list(range(1, EXPECTED_CELLS + 1)),
        "centroids": centroids,
    }


def parse_internet_value(text: str) -> Tuple[str, Optional[float]]:
    if text == "":
        return "null", None
    try:
        value = float(text)
    except ValueError:
        return "malformed", None
    if not math.isfinite(value):
        return "nonfinite", None
    return "numeric", value


def read_day_aggregates(path: Path) -> Tuple[Dict[Tuple[int, int], GroupState], Dict[str, object]]:
    groups: Dict[Tuple[int, int], GroupState] = defaultdict(GroupState)
    timestamps = set()
    cells = set()
    country_codes = set()
    stats = {
        "file": path.name,
        "raw_rows": 0,
        "bad_rows": 0,
        "null_internet_rows": 0,
        "positive_internet_rows": 0,
        "explicit_zero_internet_rows": 0,
        "negative_internet_rows": 0,
        "nonfinite_internet_rows": 0,
        "malformed_internet_rows": 0,
        "min_timestamp_ms": None,
        "max_timestamp_ms": None,
        "min_cell_id": None,
        "max_cell_id": None,
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
                country_code = parts[2]
            except ValueError:
                stats["bad_rows"] += 1
                continue
            key = (timestamp_ms, cell_id)
            kind, value = parse_internet_value(parts[7])
            group = groups[key]
            if kind == "null":
                group.add_null()
                stats["null_internet_rows"] += 1
            elif kind == "malformed":
                group.add_malformed()
                stats["malformed_internet_rows"] += 1
            elif kind == "nonfinite":
                group.add_nonfinite()
                stats["nonfinite_internet_rows"] += 1
            else:
                assert value is not None
                group.add_numeric(value)
                if value > 0.0:
                    stats["positive_internet_rows"] += 1
                elif value == 0.0:
                    stats["explicit_zero_internet_rows"] += 1
                else:
                    stats["negative_internet_rows"] += 1
            timestamps.add(timestamp_ms)
            cells.add(cell_id)
            country_codes.add(country_code)
            stats["min_timestamp_ms"] = timestamp_ms if stats["min_timestamp_ms"] is None else min(stats["min_timestamp_ms"], timestamp_ms)
            stats["max_timestamp_ms"] = timestamp_ms if stats["max_timestamp_ms"] is None else max(stats["max_timestamp_ms"], timestamp_ms)
            stats["min_cell_id"] = cell_id if stats["min_cell_id"] is None else min(stats["min_cell_id"], cell_id)
            stats["max_cell_id"] = cell_id if stats["max_cell_id"] is None else max(stats["max_cell_id"], cell_id)
    duplicate_groups = sum(1 for g in groups.values() if g.raw_rows > 1)
    duplicate_extra = sum(g.raw_rows - 1 for g in groups.values() if g.raw_rows > 1)
    stats.update(
        {
            "unique_timestamps": len(timestamps),
            "unique_cells": len(cells),
            "unique_country_codes": len(country_codes),
            "duplicate_timestamp_cell_groups": duplicate_groups,
            "duplicate_extra_timestamp_cell_records": duplicate_extra,
            "aggregated_timestamp_cell_records": len(groups),
            "numeric_observed_timestamp_cell_groups": sum(1 for g in groups.values() if g.state == "NUMERIC_OBSERVED"),
            "all_null_timestamp_cell_groups": sum(1 for g in groups.values() if g.state == "ALL_NULL"),
            "invalid_only_timestamp_cell_groups": sum(1 for g in groups.values() if g.state == "INVALID_ONLY"),
            "mixed_numeric_plus_null_timestamp_cell_groups": sum(1 for g in groups.values() if g.mixed_numeric_plus_null),
        }
    )
    return dict(groups), stats


def segment_for_timestamp(ms: int, segments: Sequence[Segment]) -> Optional[str]:
    for segment in segments:
        if ms >= segment.start_ms and (segment.end_ms is None or ms < segment.end_ms):
            return segment.name
    return None


def timestamps_in_range(start_ms: int, end_ms: int) -> List[int]:
    return list(range(start_ms, end_ms, TEN_MINUTES_MS))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def save_png(path: Path, width: int, height: int, pixels: List[List[Tuple[int, int, int]]]) -> None:
    raw = b"".join(b"\x00" + b"".join(bytes(rgb) for rgb in row) for row in pixels)
    data = b"\x89PNG\r\n\x1a\n"
    data += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += png_chunk(b"IDAT", zlib.compress(raw, 9))
    data += png_chunk(b"IEND", b"")
    path.write_bytes(data)


def draw_line_plot(path: Path, series: Sequence[float], color=(40, 90, 150)) -> None:
    width, height = 900, 300
    pixels = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]
    left, right, top, bottom = 48, 16, 18, 36
    for x in range(left, width - right):
        pixels[height - bottom][x] = (210, 210, 210)
    for y in range(top, height - bottom + 1):
        pixels[y][left] = (210, 210, 210)
    if series:
        max_points = width - left - right
        if len(series) > max_points:
            bucket = len(series) / max_points
            values = []
            for i in range(max_points):
                start = int(i * bucket)
                end = max(start + 1, int((i + 1) * bucket))
                values.append(sum(series[start:end]) / (end - start))
        else:
            values = list(series)
        lo, hi = min(values), max(values)
        if hi == lo:
            hi = lo + 1.0
        points = []
        for i, value in enumerate(values):
            x = left + int(i * (width - left - right - 1) / max(1, len(values) - 1))
            y = top + int((hi - value) * (height - top - bottom - 1) / (hi - lo))
            points.append((x, y))
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for step in range(steps + 1):
                x = round(x0 + (x1 - x0) * step / steps)
                y = round(y0 + (y1 - y0) * step / steps)
                if 0 <= x < width and 0 <= y < height:
                    pixels[y][x] = color
    save_png(path, width, height, pixels)


def draw_scatter(path: Path, points: Sequence[Tuple[float, float]]) -> None:
    width, height = 500, 500
    pixels = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmin == xmax:
            xmax = xmin + 1
        if ymin == ymax:
            ymax = ymin + 1
        for x, y in points:
            px = 24 + int((x - xmin) * (width - 48) / (xmax - xmin))
            py = height - 24 - int((y - ymin) * (height - 48) / (ymax - ymin))
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    qx, qy = px + dx, py + dy
                    if 0 <= qx < width and 0 <= qy < height:
                        pixels[qy][qx] = (170, 60, 55)
    save_png(path, width, height, pixels)


def policy_value(group: Optional[GroupState], policy: str) -> Optional[float]:
    if group is None:
        return 0.0 if policy == "P3_LEGACY_ZERO_FILL" else None
    if group.numeric_rows > 0:
        return group.finite_sum
    if group.state == "ALL_NULL":
        return 0.0 if policy in ("P2_NULL_ZERO_ABSENT_MASKED", "P3_LEGACY_ZERO_FILL") else None
    return None


def policy_denominator(expected: int, stats: CellPolicyStats, policy: str) -> int:
    if policy == "P1_OBSERVED_ONLY":
        return stats.numeric_count
    if policy == "P2_NULL_ZERO_ABSENT_MASKED":
        return stats.numeric_count + stats.all_null_count
    return expected


def policy_mean(expected: int, stats: CellPolicyStats, policy: str) -> Optional[float]:
    den = policy_denominator(expected, stats, policy)
    return safe_ratio(stats.observed_sum, den)


def rank_cells(
    cell_stats: Dict[int, CellPolicyStats],
    expected_per_cell: int,
    policy: str,
    n_cells: int,
) -> List[int]:
    def key(cell_id: int) -> Tuple[float, int]:
        mean = policy_mean(expected_per_cell, cell_stats[cell_id], policy)
        return (-(mean if mean is not None else float("-inf")), cell_id)

    return sorted(range(1, n_cells + 1), key=key)


def jaccard(a: Iterable[int], b: Iterable[int]) -> Optional[float]:
    sa = set(a)
    sb = set(b)
    union = sa | sb
    if not union:
        return None
    return len(sa & sb) / len(union)


def summarize_rank_comparison(rank_a: Dict[int, int], rank_b: Dict[int, int], k: int) -> Dict[str, object]:
    set_a = set(rank_a)
    set_b = set(rank_b)
    common = sorted(set_a & set_b)
    displacements = [abs(rank_a[c] - rank_b[c]) for c in common]
    return {
        "candidate_k": k,
        "jaccard": jaccard(set_a, set_b),
        "rank_correlation_common_cells": spearman_rank_correlation(rank_a, rank_b),
        "entered": sorted(set_b - set_a),
        "left": sorted(set_a - set_b),
        "max_rank_displacement": max(displacements) if displacements else None,
        "median_rank_displacement": median(displacements) if displacements else None,
    }


def observation_row(scope: str, expected: int, counts: Dict[str, float]) -> Dict[str, object]:
    row = {"scope": scope, "expected_timestamp_cell_pairs": expected}
    for name in (
        "numeric_observed",
        "explicit_zero",
        "all_null",
        "absent",
        "invalid_only",
        "mixed_numeric_plus_null",
        "negative",
        "nonfinite",
        "malformed",
    ):
        count = counts.get(name, 0)
        row[f"{name}_count"] = count
        row[f"{name}_ratio"] = safe_ratio(count, expected)
    return row


def add_group_to_counts(counts: Dict[str, float], group: GroupState) -> None:
    counts["present"] = counts.get("present", 0) + 1
    if group.numeric_rows > 0:
        counts["numeric_observed"] = counts.get("numeric_observed", 0) + 1
    elif group.state == "ALL_NULL":
        counts["all_null"] = counts.get("all_null", 0) + 1
    else:
        counts["invalid_only"] = counts.get("invalid_only", 0) + 1
    if group.explicit_zero_rows > 0:
        counts["explicit_zero"] = counts.get("explicit_zero", 0) + 1
    if group.mixed_numeric_plus_null:
        counts["mixed_numeric_plus_null"] = counts.get("mixed_numeric_plus_null", 0) + 1
    if group.negative_rows > 0:
        counts["negative"] = counts.get("negative", 0) + group.negative_rows
    counts["nonfinite"] = counts.get("nonfinite", 0) + group.nonfinite_rows
    counts["malformed"] = counts.get("malformed", 0) + group.malformed_rows


def sum_count_dicts(dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    total = defaultdict(float)
    for counts in dicts:
        for key, value in counts.items():
            total[key] += value
    return dict(total)


def build_observation_diagnostics(
    groups_by_ts: Dict[int, Dict[int, GroupState]],
    timestamps: Sequence[int],
    segments: Sequence[Segment],
    n_cells: int,
) -> Tuple[List[Dict[str, object]], Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    scope_counts: Dict[str, Dict[str, float]] = {"global": defaultdict(float)}
    scope_expected = {"global": len(timestamps) * n_cells}
    segment_by_ts = {}
    for segment in segments:
        ts_list = set(timestamps_in_range(segment.start_ms, segment.end_ms or segment.start_ms))
        scope = f"split:{segment.name}"
        scope_counts[scope] = defaultdict(float)
        scope_expected[scope] = len(ts_list) * n_cells
        for ts in ts_list:
            segment_by_ts[ts] = scope
    per_cell_global = {cell_id: defaultdict(float) for cell_id in range(1, n_cells + 1)}
    per_cell_train = {cell_id: defaultdict(float) for cell_id in range(1, n_cells + 1)}
    timestamp_set = set(timestamps)
    for ts, cell_groups in groups_by_ts.items():
        if ts not in timestamp_set:
            continue
        split_scope = segment_by_ts.get(ts)
        for cell_id, group in cell_groups.items():
            if not 1 <= cell_id <= n_cells:
                continue
            add_group_to_counts(scope_counts["global"], group)
            if split_scope is not None:
                add_group_to_counts(scope_counts[split_scope], group)
            add_group_to_counts(per_cell_global[cell_id], group)
            if split_scope == "split:train":
                add_group_to_counts(per_cell_train[cell_id], group)
    rows = []
    for scope, expected in scope_expected.items():
        counts = scope_counts[scope]
        counts["absent"] = expected - counts.get("present", 0)
        rows.append(observation_row(scope, expected, counts))
    for cell_id in range(1, n_cells + 1):
        counts = per_cell_global[cell_id]
        counts["absent"] = len(timestamps) - counts.get("present", 0)
        rows.append(observation_row(f"cell:{cell_id}", len(timestamps), counts))
        train_expected = (segments[0].end_ms - segments[0].start_ms) // TEN_MINUTES_MS
        train_counts = per_cell_train[cell_id]
        train_counts["absent"] = train_expected - train_counts.get("present", 0)
    return rows, per_cell_global, per_cell_train


def aggregate_observation_counts(
    groups_by_ts: Dict[int, Dict[int, GroupState]],
    timestamps: Sequence[int],
    cells: Sequence[int],
) -> Dict[str, float]:
    counts = defaultdict(float)
    cell_set = set(cells)
    for ts in timestamps:
        present = groups_by_ts.get(ts, {})
        scoped_present = 0
        for cell_id, group in present.items():
            if cell_id not in cell_set:
                continue
            scoped_present += 1
            if group.numeric_rows > 0:
                counts["numeric_observed"] += 1
            elif group.state == "ALL_NULL":
                counts["all_null"] += 1
            else:
                counts["invalid_only"] += 1
            if group.explicit_zero_rows > 0:
                counts["explicit_zero"] += 1
            if group.mixed_numeric_plus_null:
                counts["mixed_numeric_plus_null"] += 1
            if group.negative_rows > 0:
                counts["negative"] += group.negative_rows
            counts["nonfinite"] += group.nonfinite_rows
            counts["malformed"] += group.malformed_rows
        counts["absent"] += len(cells) - scoped_present
    return dict(counts)


def build_cell_policy_stats(
    groups_by_ts: Dict[int, Dict[int, GroupState]],
    timestamps: Sequence[int],
    n_cells: int,
) -> Dict[int, CellPolicyStats]:
    stats = {cell_id: CellPolicyStats() for cell_id in range(1, n_cells + 1)}
    for ts in timestamps:
        for cell_id, group in groups_by_ts.get(ts, {}).items():
            if not 1 <= cell_id <= n_cells:
                continue
            cell = stats[cell_id]
            cell.present_count += 1
            if group.numeric_rows > 0:
                cell.numeric_count += 1
                cell.observed_sum += group.finite_sum
                if group.explicit_zero_rows > 0:
                    cell.explicit_zero_count += 1
            elif group.state == "ALL_NULL":
                cell.all_null_count += 1
    expected = len(timestamps)
    for cell in stats.values():
        cell.absent_count = expected - cell.present_count
    return stats


def per_cell_observation_rows(
    groups_by_ts: Dict[int, Dict[int, GroupState]],
    timestamps: Sequence[int],
    n_cells: int,
) -> List[Dict[str, object]]:
    expected = len(timestamps)
    counts = {
        cell_id: defaultdict(float)
        for cell_id in range(1, n_cells + 1)
    }
    for ts in timestamps:
        for cell_id, group in groups_by_ts.get(ts, {}).items():
            if not 1 <= cell_id <= n_cells:
                continue
            cell = counts[cell_id]
            cell["present"] += 1
            if group.numeric_rows > 0:
                cell["numeric_observed"] += 1
            elif group.state == "ALL_NULL":
                cell["all_null"] += 1
            else:
                cell["invalid_only"] += 1
            if group.explicit_zero_rows > 0:
                cell["explicit_zero"] += 1
            if group.mixed_numeric_plus_null:
                cell["mixed_numeric_plus_null"] += 1
            if group.negative_rows > 0:
                cell["negative"] += group.negative_rows
            cell["nonfinite"] += group.nonfinite_rows
            cell["malformed"] += group.malformed_rows
    rows = []
    for cell_id in range(1, n_cells + 1):
        cell = counts[cell_id]
        cell["absent"] = expected - cell.get("present", 0)
        rows.append(observation_row(f"cell:{cell_id}", expected, cell))
    return rows


def aggregate_series(
    groups_by_ts: Dict[int, Dict[int, GroupState]],
    timestamps: Sequence[int],
    cells: Sequence[int],
    policy: str,
    timestamp_numeric_sum: Optional[Dict[int, float]] = None,
    timestamp_numeric_count: Optional[Dict[int, int]] = None,
    timestamp_all_null_count: Optional[Dict[int, int]] = None,
) -> Tuple[List[float], int]:
    if len(cells) > 1000 and timestamp_numeric_sum is not None and timestamp_numeric_count is not None and timestamp_all_null_count is not None:
        values = []
        denominator = 0
        for ts in timestamps:
            total = timestamp_numeric_sum.get(ts, 0.0)
            numeric_count = timestamp_numeric_count.get(ts, 0)
            all_null_count = timestamp_all_null_count.get(ts, 0)
            if policy == "P1_OBSERVED_ONLY":
                denominator += numeric_count
                if numeric_count > 0:
                    values.append(total)
            elif policy == "P2_NULL_ZERO_ABSENT_MASKED":
                ts_den = numeric_count + all_null_count
                denominator += ts_den
                if ts_den > 0:
                    values.append(total)
            else:
                denominator += len(cells)
                values.append(total)
        return values, denominator

    if len(cells) > 1000:
        values = []
        denominator = 0
        for ts in timestamps:
            total = 0.0
            ts_den = 0
            for group in groups_by_ts.get(ts, {}).values():
                if group.numeric_rows > 0:
                    total += group.finite_sum
                    ts_den += 1
                elif group.state == "ALL_NULL" and policy in ("P2_NULL_ZERO_ABSENT_MASKED", "P3_LEGACY_ZERO_FILL"):
                    ts_den += 1
            if policy == "P3_LEGACY_ZERO_FILL":
                denominator += len(cells)
                values.append(total)
            else:
                denominator += ts_den
                if ts_den > 0:
                    values.append(total)
        return values, denominator

    values = []
    denominator = 0
    cell_set = set(cells)
    for ts in timestamps:
        total = 0.0
        has_value = False
        present = groups_by_ts.get(ts, {})
        for cell_id, group in present.items():
            if cell_id not in cell_set:
                continue
            value = policy_value(group, policy)
            if value is not None:
                total += value
                has_value = True
        if policy == "P3_LEGACY_ZERO_FILL":
            denominator += len(cells)
        elif has_value:
            denominator += sum(1 for cell_id, group in present.items() if cell_id in cell_set and policy_value(group, policy) is not None)
        if has_value or policy == "P3_LEGACY_ZERO_FILL":
            values.append(total)
    return values, denominator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=os.environ.get("MILAN_RAW_DIR"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/local/data_audit_v2"))
    parser.add_argument("--topk", type=int, nargs="*", default=list(TOPK_VALUES))
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    args = parser.parse_args()

    if args.raw_dir is None:
        fail("Provide --raw-dir or set MILAN_RAW_DIR.")
    raw_dir = args.raw_dir.expanduser().resolve()
    if not raw_dir.exists():
        fail("Raw directory does not exist.")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_daily_files(raw_dir)
    grid = load_grid_summary(raw_dir)
    centroids = grid.get("centroids", {}) if grid.get("available") else {}
    n_cells = EXPECTED_CELLS
    segments = [
        Segment("train", local_ms(TRAIN_START), local_ms(TRAIN_END)),
        Segment("validation", local_ms(VAL_START), local_ms(VAL_END)),
        Segment("test", local_ms(TEST_START), local_ms(TEST_END)),
        Segment("reserved_holdout", local_ms(HOLDOUT_START), None),
    ]
    drift_ms = local_ms(DRIFT_POINT)
    train_week_starts = [local_ms("2013-11-01 00:00"), local_ms("2013-11-08 00:00"), local_ms("2013-11-15 00:00"), local_ms("2013-11-22 00:00")]
    train_week_ends = [local_ms("2013-11-08 00:00"), local_ms("2013-11-15 00:00"), local_ms("2013-11-22 00:00"), local_ms("2013-11-29 00:00")]

    groups_by_ts: Dict[int, Dict[int, GroupState]] = defaultdict(dict)
    timestamp_total_numeric = defaultdict(float)
    timestamp_numeric_count = defaultdict(int)
    timestamp_all_null_count = defaultdict(int)
    all_timestamps = set()
    all_cells = set()
    daily_stats = []
    raw_rows = bad_rows = 0
    null_count = positive_count = explicit_zero_count = 0
    nonfinite_count = malformed_count = negative_count = 0
    duplicate_groups = duplicate_extra = 0
    min_ts = max_ts = None
    min_cell = max_cell = None
    last_progress = time.time()

    for index, path in enumerate(files, start=1):
        groups, stats = read_day_aggregates(path)
        daily_stats.append({**{k: v for k, v in stats.items() if k != "file"}, "file": stats["file"]})
        raw_rows += int(stats["raw_rows"])
        bad_rows += int(stats["bad_rows"])
        null_count += int(stats["null_internet_rows"])
        positive_count += int(stats["positive_internet_rows"])
        explicit_zero_count += int(stats["explicit_zero_internet_rows"])
        nonfinite_count += int(stats["nonfinite_internet_rows"])
        malformed_count += int(stats["malformed_internet_rows"])
        negative_count += int(stats["negative_internet_rows"])
        duplicate_groups += int(stats["duplicate_timestamp_cell_groups"])
        duplicate_extra += int(stats["duplicate_extra_timestamp_cell_records"])
        if stats["min_timestamp_ms"] is not None:
            min_ts = stats["min_timestamp_ms"] if min_ts is None else min(min_ts, stats["min_timestamp_ms"])
            max_ts = stats["max_timestamp_ms"] if max_ts is None else max(max_ts, stats["max_timestamp_ms"])
        if stats["min_cell_id"] is not None:
            min_cell = stats["min_cell_id"] if min_cell is None else min(min_cell, stats["min_cell_id"])
            max_cell = stats["max_cell_id"] if max_cell is None else max(max_cell, stats["max_cell_id"])
        for (ts_ms, cell_id), group in groups.items():
            groups_by_ts[ts_ms][cell_id] = group
            all_timestamps.add(ts_ms)
            all_cells.add(cell_id)
            if group.numeric_rows > 0:
                timestamp_total_numeric[ts_ms] += group.finite_sum
                timestamp_numeric_count[ts_ms] += 1
            elif group.state == "ALL_NULL":
                timestamp_all_null_count[ts_ms] += 1
        if time.time() - last_progress >= args.progress_seconds:
            print(f"processed {index}/{len(files)} daily files", file=sys.stderr, flush=True)
            last_progress = time.time()

    if min_ts is None or max_ts is None:
        fail("No valid timestamped rows were read.")

    all_expected_timestamps = timestamps_in_range(min_ts, max_ts + TEN_MINUTES_MS)
    missing_timestamps = [ts for ts in all_expected_timestamps if ts not in all_timestamps]
    complete_timestamps = [ts for ts in sorted(all_timestamps) if len(groups_by_ts[ts]) == n_cells]
    incomplete_timestamps = [ts for ts in sorted(all_timestamps) if len(groups_by_ts[ts]) < n_cells]
    end_for_holdout = max_ts + TEN_MINUTES_MS
    segments[-1] = Segment("reserved_holdout", local_ms(HOLDOUT_START), end_for_holdout)

    train_timestamps = timestamps_in_range(segments[0].start_ms, segments[0].end_ms or segments[0].start_ms)
    train_cell_stats = build_cell_policy_stats(groups_by_ts, train_timestamps, n_cells)
    rankings = {
        policy: rank_cells(train_cell_stats, len(train_timestamps), policy, n_cells)
        for policy in POLICIES
    }
    max_topk = max(args.topk)

    observation_rows, _per_cell_global_counts, per_cell_train_counts = build_observation_diagnostics(
        groups_by_ts,
        all_expected_timestamps,
        segments,
        n_cells,
    )
    for policy in POLICIES:
        for k in args.topk:
            selected = rankings[policy][:k]
            counts = sum_count_dicts(per_cell_train_counts[cell_id] for cell_id in selected)
            observation_rows.append(observation_row(f"topk:{policy}:K={k}:train", len(train_timestamps) * k, counts))
    observation_by_scope = {row["scope"]: row for row in observation_rows}

    topk_rows: List[Dict[str, object]] = []
    topk_rank_maps: Dict[Tuple[str, int], Dict[int, int]] = {}
    train_total_observed_sum = sum(s.observed_sum for s in train_cell_stats.values())
    weekly_stats = [
        (timestamps_in_range(ws, we), build_cell_policy_stats(groups_by_ts, timestamps_in_range(ws, we), n_cells))
        for ws, we in zip(train_week_starts, train_week_ends)
    ]
    for policy in POLICIES:
        weekly_sets_by_k: Dict[int, List[set]] = {k: [] for k in args.topk}
        weekly_ranks_by_k: Dict[int, List[Dict[int, int]]] = {k: [] for k in args.topk}
        for week_ts, week_stats in weekly_stats:
            week_ranking = rank_cells(week_stats, len(week_ts), policy, n_cells)
            for k in args.topk:
                week_top = week_ranking[:k]
                weekly_sets_by_k[k].append(set(week_top))
                weekly_ranks_by_k[k].append({cell: rank + 1 for rank, cell in enumerate(week_top)})
        for k in args.topk:
            selected = rankings[policy][:k]
            selected_total = sum(train_cell_stats[c].observed_sum for c in selected)
            selected_centroids = [centroids[c] for c in selected if c in centroids]
            spatial_summary = {}
            if selected_centroids:
                xs = [p[0] for p in selected_centroids]
                ys = [p[1] for p in selected_centroids]
                spatial_summary = {
                    "spatial_centroid_x_min": min(xs),
                    "spatial_centroid_x_max": max(xs),
                    "spatial_centroid_x_mean": sum(xs) / len(xs),
                    "spatial_centroid_y_min": min(ys),
                    "spatial_centroid_y_max": max(ys),
                    "spatial_centroid_y_mean": sum(ys) / len(ys),
                }
            jaccards = []
            spearmans = []
            for i in range(len(weekly_sets_by_k[k])):
                for j in range(i + 1, len(weekly_sets_by_k[k])):
                    jaccards.append(jaccard(weekly_sets_by_k[k][i], weekly_sets_by_k[k][j]))
                    spearmans.append(spearman_rank_correlation(weekly_ranks_by_k[k][i], weekly_ranks_by_k[k][j]))
            topk_rank_maps[(policy, k)] = {cell: rank + 1 for rank, cell in enumerate(selected)}
            for rank, cell_id in enumerate(selected, start=1):
                stats = train_cell_stats[cell_id]
                expected = len(train_timestamps)
                row = {
                    "policy": policy,
                    "candidate_k": k,
                    "rank": rank,
                    "cell_id": cell_id,
                    "observed_traffic_sum": stats.observed_sum,
                    "mean_under_policy": policy_mean(expected, stats, policy),
                    "observation_denominator": policy_denominator(expected, stats, policy),
                    "expected_training_pairs_for_cell": expected,
                    "observation_coverage": safe_ratio(policy_denominator(expected, stats, policy), expected),
                    "numeric_observed_count": stats.numeric_count,
                    "all_null_count": stats.all_null_count,
                    "absent_count": stats.absent_count,
                    "explicit_zero_group_count": stats.explicit_zero_count,
                    "all_null_ratio": safe_ratio(stats.all_null_count, expected),
                    "absent_ratio": safe_ratio(stats.absent_count, expected),
                    "observed_traffic_fraction": safe_ratio(stats.observed_sum, train_total_observed_sum),
                    "cumulative_traffic_volume_coverage_at_k": safe_ratio(selected_total, train_total_observed_sum),
                    "weekly_membership_count": sum(1 for s in weekly_sets_by_k[k] if cell_id in s),
                    "weekly_jaccard_median": quantile([x for x in jaccards if x is not None], 0.50),
                    "weekly_jaccard_min": min([x for x in jaccards if x is not None], default=None),
                    "weekly_spearman_median": quantile([x for x in spearmans if x is not None], 0.50),
                    "weekly_spearman_min": min([x for x in spearmans if x is not None], default=None),
                }
                if cell_id in centroids:
                    row["centroid_x"] = centroids[cell_id][0]
                    row["centroid_y"] = centroids[cell_id][1]
                row.update(spatial_summary)
                topk_rows.append(row)

    policy_sensitivity_rows = []
    for k in args.topk:
        base = topk_rank_maps[("P1_OBSERVED_ONLY", k)]
        for policy in POLICIES:
            comp = summarize_rank_comparison(base, topk_rank_maps[(policy, k)], k)
            comp.update({"baseline_policy": "P1_OBSERVED_ONLY", "comparison_policy": policy})
            policy_sensitivity_rows.append(comp)

    comparison_json = {"status": "draft", "comparison_note": "Evidence only; no policy or K is recommended.", "topk_policy_comparisons": []}
    for k in args.topk:
        policies = list(POLICIES)
        for i, a in enumerate(policies):
            for b in policies[i + 1 :]:
                comp = summarize_rank_comparison(topk_rank_maps[(a, k)], topk_rank_maps[(b, k)], k)
                comp.update({"policy_a": a, "policy_b": b})
                high_missing = []
                for cell_id in sorted(set(topk_rank_maps[(a, k)]) | set(topk_rank_maps[(b, k)])):
                    s = train_cell_stats[cell_id]
                    expected = len(train_timestamps)
                    if safe_ratio(s.all_null_count, expected) and safe_ratio(s.all_null_count, expected) > 0.05:
                        high_missing.append({"cell_id": cell_id, "all_null_ratio": safe_ratio(s.all_null_count, expected), "absent_ratio": safe_ratio(s.absent_count, expected)})
                    elif safe_ratio(s.absent_count, expected) and safe_ratio(s.absent_count, expected) > 0.05:
                        high_missing.append({"cell_id": cell_id, "all_null_ratio": safe_ratio(s.all_null_count, expected), "absent_ratio": safe_ratio(s.absent_count, expected)})
                comp["selected_cells_with_unusually_high_all_null_or_absent_ratios"] = high_missing
                comparison_json["topk_policy_comparisons"].append(comp)

    split_rows: List[Dict[str, object]] = []
    train_policy_series = {}
    all_cell_ids = list(range(1, n_cells + 1))
    for policy in POLICIES:
        train_policy_series[policy] = aggregate_series(
            groups_by_ts,
            train_timestamps,
            all_cell_ids,
            policy,
            timestamp_total_numeric,
            timestamp_numeric_count,
            timestamp_all_null_count,
        )[0]
    for segment in segments:
        ts_list = timestamps_in_range(segment.start_ms, segment.end_ms or end_for_holdout)
        structural = {
            "segment": segment.name,
            "row_type": "structural",
            "policy": "",
            "start_local": ms_to_iso(segment.start_ms),
            "end_local": ms_to_iso(segment.end_ms),
            "expected_timestamp_count": len(ts_list),
            "actual_timestamp_count": sum(1 for ts in ts_list if ts in all_timestamps),
            "weekday_timestamp_count": sum(1 for ts in ts_list if ms_to_local(ts).weekday() < 5),
            "weekend_timestamp_count": sum(1 for ts in ts_list if ms_to_local(ts).weekday() >= 5),
            "complete_timestamp_count": sum(1 for ts in ts_list if len(groups_by_ts.get(ts, {})) == n_cells),
            "incomplete_timestamp_count": sum(1 for ts in ts_list if ts in all_timestamps and len(groups_by_ts.get(ts, {})) < n_cells),
        }
        split_rows.append(structural)
        coverage = dict(observation_by_scope[f"split:{segment.name}"])
        coverage.update({"segment": segment.name, "row_type": "observation_state", "policy": ""})
        split_rows.append(coverage)
        for policy in POLICIES:
            series, denom = aggregate_series(
                groups_by_ts,
                ts_list,
                all_cell_ids,
                policy,
                timestamp_total_numeric,
                timestamp_numeric_count,
                timestamp_all_null_count,
            )
            mean, std = mean_std(series)
            split_rows.append(
                {
                    "segment": segment.name,
                    "row_type": "numeric_distribution",
                    "policy": policy,
                    "expected_timestamp_cell_pairs": len(ts_list) * n_cells,
                    "observation_denominator": denom,
                    "coverage_ratio": safe_ratio(denom, len(ts_list) * n_cells),
                    "aggregate_timestamp_count_used": len(series),
                    "aggregate_mean": mean,
                    "aggregate_std": std,
                    "aggregate_q10": quantile(series, 0.10),
                    "aggregate_q50": quantile(series, 0.50),
                    "aggregate_q90": quantile(series, 0.90),
                    "train_vs_segment_ks_distance": None if segment.name == "train" else ks_distance(train_policy_series[policy], series),
                    "train_vs_segment_mean_ratio": None if segment.name == "train" else safe_ratio(mean, mean_std(train_policy_series[policy])[0]),
                }
            )

    drift_rows: List[Dict[str, object]] = []
    drift_group_counts = aggregate_observation_counts(groups_by_ts, [drift_ms], all_cell_ids)
    drift_rows.append(
        {
            "row_type": "boundary_observation_state",
            "boundary_local": ms_to_iso(drift_ms),
            "timestamp_exists": drift_ms in all_timestamps,
            "expected_cells": n_cells,
            "numeric_observed_cells": drift_group_counts.get("numeric_observed", 0),
            "all_null_cells": drift_group_counts.get("all_null", 0),
            "absent_cells": drift_group_counts.get("absent", 0),
            "invalid_only_cells": drift_group_counts.get("invalid_only", 0),
        }
    )
    for policy in POLICIES:
        for k in args.topk:
            selected = rankings[policy][:k]
            counts = aggregate_observation_counts(groups_by_ts, [drift_ms], selected)
            drift_rows.append(
                {
                    "row_type": "boundary_topk_completeness",
                    "boundary_local": ms_to_iso(drift_ms),
                    "policy": policy,
                    "candidate_k": k,
                    "expected_cells": k,
                    "numeric_observed_cells": counts.get("numeric_observed", 0),
                    "all_null_cells": counts.get("all_null", 0),
                    "absent_cells": counts.get("absent", 0),
                    "coverage_ratio": safe_ratio(counts.get("numeric_observed", 0), k),
                }
            )
    for hours in (1, 6, 24, 168):
        span = hours * 60 * 60 * 1000
        pre_ts = timestamps_in_range(drift_ms - span, drift_ms)
        post_ts = timestamps_in_range(drift_ms, drift_ms + span)
        drift_rows.append(
            {
                "row_type": "window_structural_coverage",
                "boundary_local": ms_to_iso(drift_ms),
                "window_hours_each_side": hours,
                "pre_expected_timestamp_count": len(pre_ts),
                "post_expected_timestamp_count": len(post_ts),
                "pre_existing_timestamp_count": sum(1 for ts in pre_ts if ts in all_timestamps),
                "post_existing_timestamp_count": sum(1 for ts in post_ts if ts in all_timestamps),
                "pre_complete_timestamp_count": sum(1 for ts in pre_ts if len(groups_by_ts.get(ts, {})) == n_cells),
                "post_complete_timestamp_count": sum(1 for ts in post_ts if len(groups_by_ts.get(ts, {})) == n_cells),
            }
        )
        for policy in POLICIES:
            pre, pre_den = aggregate_series(
                groups_by_ts,
                pre_ts,
                all_cell_ids,
                policy,
                timestamp_total_numeric,
                timestamp_numeric_count,
                timestamp_all_null_count,
            )
            post, post_den = aggregate_series(
                groups_by_ts,
                post_ts,
                all_cell_ids,
                policy,
                timestamp_total_numeric,
                timestamp_numeric_count,
                timestamp_all_null_count,
            )
            pre_mean, pre_std = mean_std(pre)
            post_mean, post_std = mean_std(post)
            drift_rows.append(
                {
                    "row_type": "window_policy_aggregate",
                    "boundary_local": ms_to_iso(drift_ms),
                    "window_hours_each_side": hours,
                    "policy": policy,
                    "pre_observation_denominator": pre_den,
                    "post_observation_denominator": post_den,
                    "pre_coverage_ratio": safe_ratio(pre_den, len(pre_ts) * n_cells),
                    "post_coverage_ratio": safe_ratio(post_den, len(post_ts) * n_cells),
                    "pre_aggregate_mean": pre_mean,
                    "post_aggregate_mean": post_mean,
                    "pre_aggregate_std": pre_std,
                    "post_aggregate_std": post_std,
                    "post_over_pre_mean_ratio": safe_ratio(post_mean, pre_mean),
                    "ks_distance": ks_distance(pre, post),
                }
            )
    boundary = local_ms(TEST_START) + 24 * 60 * 60 * 1000
    while boundary < local_ms(TEST_END):
        pre_start = boundary - 7 * 24 * 60 * 60 * 1000
        post_end = boundary + 7 * 24 * 60 * 60 * 1000
        if pre_start >= min_ts and post_end <= max_ts + TEN_MINUTES_MS:
            pre_slots = timestamps_in_range(pre_start, boundary)
            post_slots = timestamps_in_range(boundary, post_end)
            if all(ts in all_timestamps for ts in pre_slots + post_slots):
                drift_rows.append(
                    {
                        "row_type": "structurally_eligible_midnight_alternative",
                        "boundary_local": ms_to_iso(boundary),
                        "pre_expected_timestamp_count": len(pre_slots),
                        "post_expected_timestamp_count": len(post_slots),
                        "pre_weekend_timestamp_count": sum(1 for ts in pre_slots if ms_to_local(ts).weekday() >= 5),
                        "post_weekend_timestamp_count": sum(1 for ts in post_slots if ms_to_local(ts).weekday() >= 5),
                    }
                )
        boundary += 24 * 60 * 60 * 1000

    warnings = []
    if missing_timestamps:
        warnings.append(f"{len(missing_timestamps)} expected 10-minute timestamps are missing from actual min/max coverage.")
    if incomplete_timestamps:
        warnings.append(f"{len(incomplete_timestamps)} timestamps have fewer than {n_cells} explicit cell records; this is reported as sparse representation evidence, not automatically a data-quality defect.")
    if negative_count:
        warnings.append(f"{negative_count} raw Internet values are negative.")
    if nonfinite_count:
        warnings.append(f"{nonfinite_count} raw Internet values are non-finite.")
    if malformed_count:
        warnings.append(f"{malformed_count} raw Internet values are malformed.")
    if len(files) != 62:
        warnings.append("Actual daily file count differs from 62; local data is treated as authoritative.")

    integrity = {
        "status": "draft",
        "dataset": "Milan Internet traffic",
        "script_policy": "read-only v2 audit; K, split, drift point, and missing-data policy remain UNDECIDED",
        "detected_format": {
            "daily_files": "tab-separated text, no header",
            "columns": ["gridID", "timestamp_ms", "countryCode", "sms_in", "sms_out", "call_in", "call_out", "internet"],
            "grid_geojson_available": bool(grid.get("available")),
            "grid_feature_count": grid.get("feature_count"),
        },
        "timestamp_interpretation": {
            "raw_unit": "milliseconds since Unix epoch",
            "local_timezone_assumption": timezone_assumption(),
            "evidence": "first daily file date aligns to local midnight under Europe/Rome; raw 10-minute step is 600000 ms",
        },
        "coverage": {
            "min_timestamp_local": ms_to_iso(min_ts),
            "max_timestamp_local": ms_to_iso(max_ts),
            "min_timestamp_ms_mod_10min": min_ts % TEN_MINUTES_MS,
            "actual_10min_timestamp_count": len(all_timestamps),
            "expected_10min_timestamp_count_from_actual_minmax": len(all_expected_timestamps),
            "missing_10min_timestamp_count": len(missing_timestamps),
            "cell_id_min": min_cell,
            "cell_id_max": max_cell,
            "unique_cell_count": len(all_cells),
            "complete_timestamp_count": len(complete_timestamps),
            "incomplete_timestamp_count": len(incomplete_timestamps),
        },
        "record_integrity": {
            "raw_row_count": raw_rows,
            "bad_row_count": bad_rows,
            "positive_internet_row_count": positive_count,
            "explicit_zero_internet_row_count": explicit_zero_count,
            "null_internet_row_count": null_count,
            "negative_internet_row_count": negative_count,
            "nonfinite_internet_row_count": nonfinite_count,
            "malformed_internet_row_count": malformed_count,
            "duplicate_timestamp_cell_groups_before_country_aggregation": duplicate_groups,
            "duplicate_extra_timestamp_cell_records_before_country_aggregation": duplicate_extra,
        },
        "observation_state_summary": observation_rows[0],
        "policy_views": {
            "P1_OBSERVED_ONLY": "ALL_NULL and ABSENT are masked; means use only NUMERIC_OBSERVED pairs.",
            "P2_NULL_ZERO_ABSENT_MASKED": "ALL_NULL is treated as zero for sensitivity analysis; ABSENT remains masked.",
            "P3_LEGACY_ZERO_FILL": "ALL_NULL and ABSENT are treated as zero as an explicit assumption, not verified semantics.",
        },
        "daily_file_stats": daily_stats,
        "warnings": warnings,
        "unresolved_schema_assumptions": [
            "Same timestamp/cell rows are summed across countryCode for the Internet field.",
            "The semantic meaning of null Internet fields is unresolved.",
            "The semantic meaning of absent timestamp/cell pairs is unresolved.",
        ],
        "human_decisions_pending": ["final K", "final chronological split", "final drift point", "final missing-data policy"],
    }

    write_json(output_dir / "dataset_integrity.json", integrity)
    write_json(output_dir / "policy_comparison.json", comparison_json)
    write_csv(output_dir / "observation_state_diagnostics.csv", observation_rows, list(observation_rows[0].keys()))
    topk_fields = [
        "policy",
        "candidate_k",
        "rank",
        "cell_id",
        "observed_traffic_sum",
        "mean_under_policy",
        "observation_denominator",
        "expected_training_pairs_for_cell",
        "observation_coverage",
        "numeric_observed_count",
        "all_null_count",
        "absent_count",
        "explicit_zero_group_count",
        "all_null_ratio",
        "absent_ratio",
        "observed_traffic_fraction",
        "cumulative_traffic_volume_coverage_at_k",
        "weekly_membership_count",
        "weekly_jaccard_median",
        "weekly_jaccard_min",
        "weekly_spearman_median",
        "weekly_spearman_min",
        "centroid_x",
        "centroid_y",
        "spatial_centroid_x_min",
        "spatial_centroid_x_max",
        "spatial_centroid_x_mean",
        "spatial_centroid_y_min",
        "spatial_centroid_y_max",
        "spatial_centroid_y_mean",
    ]
    write_csv(output_dir / "topk_diagnostics.csv", topk_rows, topk_fields)
    sensitivity_fields = [
        "baseline_policy",
        "comparison_policy",
        "candidate_k",
        "jaccard",
        "rank_correlation_common_cells",
        "entered",
        "left",
        "max_rank_displacement",
        "median_rank_displacement",
    ]
    write_csv(output_dir / "topk_policy_sensitivity.csv", policy_sensitivity_rows, sensitivity_fields)
    split_fields = sorted({k for row in split_rows for k in row})
    drift_fields = sorted({k for row in drift_rows for k in row})
    write_csv(output_dir / "split_diagnostics.csv", split_rows, split_fields)
    write_csv(output_dir / "drift_point_diagnostics.csv", drift_rows, drift_fields)

    draw_line_plot(output_dir / "numeric_observed_aggregate_timeseries.png", [timestamp_total_numeric[ts] for ts in sorted(all_timestamps)])
    p3_series, _ = aggregate_series(
        groups_by_ts,
        train_timestamps,
        all_cell_ids,
        "P3_LEGACY_ZERO_FILL",
        timestamp_total_numeric,
        timestamp_numeric_count,
        timestamp_all_null_count,
    )
    draw_line_plot(output_dir / "train_p3_legacy_aggregate_profile.png", p3_series, (70, 130, 70))
    scatter_points = [centroids[c] for c in rankings["P1_OBSERVED_ONLY"][:max_topk] if c in centroids]
    draw_scatter(output_dir / "top100_p1_spatial_distribution.png", scatter_points)

    global_obs = observation_rows[0]
    summary = [
        "# Milan Internet Data Audit v2",
        "",
        "Status: draft",
        "",
        "Reproduce with:",
        "",
        "```bash",
        "MILAN_RAW_DIR=/path/to/local/milan/daily python3 scripts/audit_milan_internet.py --output-dir artifacts/local/data_audit_v2",
        "```",
        "",
        "## Verified Structural Facts",
        "",
        f"- Detected {len(files)} raw daily text files and {len(all_cells)} Cell IDs.",
        f"- Timestamp unit is interpreted as Unix epoch milliseconds; local reporting assumes {timezone_assumption()}.",
        f"- Actual coverage is {ms_to_iso(min_ts)} through {ms_to_iso(max_ts)}.",
        f"- Actual unique 10-minute timestamps: {len(all_timestamps)}.",
        f"- Complete explicit timestamp coverage: {len(complete_timestamps)} timestamps; incomplete explicit timestamp coverage: {len(incomplete_timestamps)} timestamps.",
        "- Provisional split boundaries and drift timestamp are reported only as diagnostics.",
        "",
        "## Observation-State Facts",
        "",
        f"- Expected timestamp/cell pairs: {global_obs['expected_timestamp_cell_pairs']}.",
        f"- NUMERIC_OBSERVED pairs: {global_obs['numeric_observed_count']} ({global_obs['numeric_observed_ratio']:.6f}).",
        f"- Explicit-zero groups: {global_obs['explicit_zero_count']} ({global_obs['explicit_zero_ratio']:.6f}).",
        f"- ALL_NULL pairs: {global_obs['all_null_count']} ({global_obs['all_null_ratio']:.6f}).",
        f"- ABSENT pairs: {global_obs['absent_count']} ({global_obs['absent_ratio']:.6f}).",
        f"- Mixed numeric-plus-null groups: {global_obs['mixed_numeric_plus_null_count']} ({global_obs['mixed_numeric_plus_null_ratio']:.6f}).",
        "",
        "## Policy-Sensitive Results",
        "",
        "- P1_OBSERVED_ONLY masks ALL_NULL and ABSENT.",
        "- P2_NULL_ZERO_ABSENT_MASKED treats ALL_NULL as zero and masks ABSENT.",
        "- P3_LEGACY_ZERO_FILL treats ALL_NULL and ABSENT as zero as an explicit unverified assumption.",
        "- Top-K, split, and drift-point numeric diagnostics are emitted under all three views without selecting a policy.",
        "",
        "## Unresolved Semantic Assumptions",
        "",
        "- Same timestamp/cell rows are summed across countryCode for Internet.",
        "- The meaning of null Internet fields is unresolved.",
        "- The meaning of absent timestamp/cell pairs is unresolved.",
        "",
        "## Human Decisions Still Pending",
        "",
        "- Final K: UNDECIDED.",
        "- Final train/validation/test split: UNDECIDED.",
        "- Final drift point: UNDECIDED.",
        "- Final missing-data policy: UNDECIDED.",
        "",
        "## Warnings",
        "",
    ]
    summary.extend([f"- {w}" for w in warnings] or ["- No warnings generated by the audit script."])
    (output_dir / "audit_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"wrote audit outputs under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
