import io
import ast
import json
import shutil
import struct
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.materialize_exp001_preprocessing import main as cli_main
from scripts.materialize_exp001_preprocessing import resolve_raw_dir, validate_output_dir
from thesis_traffic_drift.exp001 import (
    REQUIRED_MISSING_POLICY_STATUS,
    STATE_ABSENT,
    STATE_ALL_NULL,
    STATE_INVALID_ONLY,
    STATE_NUMERIC_OBSERVED,
    DenseArrays,
    GroupState,
    TrafficObservation,
    apply_multiplicative_level_shift,
    build_dense_arrays,
    construct_split_intervals,
    drift_point,
    is_post_drift_test,
    load_exp001_config,
    parse_local_time,
    rank_topk_cells,
    segment_for_timestamp,
    select_affected_cells,
    training_means_by_cell,
    write_npz,
)


CONFIG_PATH = Path("configs/EXP-001-stale-degradation.template.yaml")


def npy_header(payload):
    if not payload.startswith(b"\x93NUMPY\x01\x00"):
        raise AssertionError("expected NumPy v1.0 payload")
    header_len = struct.unpack("<H", payload[8:10])[0]
    self_aligned_len = 10 + header_len
    if self_aligned_len % 16:
        raise AssertionError("npy header is not 16-byte aligned")
    header = ast.literal_eval(payload[10:self_aligned_len].decode("latin1").strip())
    return header, payload[self_aligned_len:]


class Exp001PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.config = load_exp001_config(CONFIG_PATH)
        self.intervals = construct_split_intervals(self.config)
        self.point = drift_point(self.config)

    def dt(self, text):
        return parse_local_time(text)

    def test_topk_ranking_uses_train_only_validation_and_test_spikes_do_not_affect_ranking(self):
        observations = [
            TrafficObservation(self.dt("2013-11-01 00:00"), 1, 10.0),
            TrafficObservation(self.dt("2013-11-01 00:00"), 2, 5.0),
            TrafficObservation(self.dt("2013-11-29 00:00"), 2, 9999.0),
            TrafficObservation(self.dt("2013-12-06 00:00"), 3, 9999.0),
        ]

        self.assertEqual(rank_topk_cells(observations, self.intervals, k=3), [1, 2, 3])

    def test_deterministic_tie_break_by_ascending_cell_id(self):
        observations = [
            TrafficObservation(self.dt("2013-11-01 00:00"), 3, 7.0),
            TrafficObservation(self.dt("2013-11-01 00:00"), 1, 7.0),
            TrafficObservation(self.dt("2013-11-01 00:00"), 2, 7.0),
        ]

        self.assertEqual(rank_topk_cells(observations, self.intervals, k=3), [1, 2, 3])

    def test_affected_cell_selection_selects_twenty_for_k100_fraction_point20(self):
        topk = list(range(1, 101))
        means = {cell_id: float(101 - cell_id) for cell_id in topk}

        affected = select_affected_cells(topk, means, 0.20)

        self.assertEqual(len(affected), 20)
        self.assertEqual(affected, list(range(1, 21)))

    def test_split_boundaries_are_end_exclusive_and_no_boundary_sample_is_dropped(self):
        self.assertEqual(segment_for_timestamp(self.dt("2013-11-28 23:50"), self.intervals), "train")
        self.assertEqual(segment_for_timestamp(self.dt("2013-11-29 00:00"), self.intervals), "validation")
        self.assertEqual(segment_for_timestamp(self.dt("2013-12-06 00:00"), self.intervals), "test")
        self.assertEqual(segment_for_timestamp(self.dt("2013-12-20 00:00"), self.intervals), "holdout")

    def test_drift_point_timestamp_belongs_to_post_drift_test(self):
        self.assertEqual(segment_for_timestamp(self.dt("2013-12-13 00:00"), self.intervals), "test")
        self.assertTrue(is_post_drift_test(self.dt("2013-12-13 00:00"), self.intervals, self.point))

    def test_drift_not_applied_before_drift_point(self):
        value = apply_multiplicative_level_shift(10.0, self.dt("2013-12-12 23:50"), 1, self.intervals, self.point, {1}, 1.5)

        self.assertEqual(value, 10.0)

    def test_drift_applied_for_affected_cell_at_drift_point(self):
        value = apply_multiplicative_level_shift(10.0, self.dt("2013-12-13 00:00"), 1, self.intervals, self.point, {1}, 1.5)

        self.assertEqual(value, 15.0)

    def test_drift_applied_for_affected_cell_after_drift_point(self):
        value = apply_multiplicative_level_shift(10.0, self.dt("2013-12-13 00:10"), 1, self.intervals, self.point, {1}, 1.5)

        self.assertEqual(value, 15.0)

    def test_unaffected_cells_remain_unchanged(self):
        value = apply_multiplicative_level_shift(10.0, self.dt("2013-12-13 00:00"), 2, self.intervals, self.point, {1}, 1.5)

        self.assertEqual(value, 10.0)

    def test_training_means_support_affected_selection_from_train_only(self):
        observations = [
            TrafficObservation(self.dt("2013-11-01 00:00"), 1, 4.0),
            TrafficObservation(self.dt("2013-11-01 00:10"), 1, 6.0),
            TrafficObservation(self.dt("2013-11-01 00:00"), 2, 3.0),
            TrafficObservation(self.dt("2013-12-13 00:00"), 2, 1000.0),
        ]

        means = training_means_by_cell(observations, self.intervals, [1, 2])

        self.assertEqual(means[1], 5.0)
        self.assertEqual(means[2], 3.0)

    def test_config_template_contains_no_private_path_patterns(self):
        text = CONFIG_PATH.read_text(encoding="utf-8")
        forbidden = [
            "/" + name + "/"
            for name in ("home", "mnt", "data")
        ] + [
            "." + suffix
            for suffix in ("npy", "h5", "pt", "pth", "ckpt", "pkl", "pickle")
        ] + [
            "".join(parts)
            for parts in (("PRI", "VATE"), ("SEC", "RET"), ("TO", "KEN"))
        ]

        self.assertFalse([pattern for pattern in forbidden if pattern in text])

    def test_missing_null_absent_modeling_policy_is_approved_exp001_v0(self):
        self.assertEqual(self.config["missing_policy"]["modeling_policy"], "observed_numeric_with_explicit_mask")
        self.assertEqual(self.config["missing_policy"]["status"], REQUIRED_MISSING_POLICY_STATUS)
        self.assertTrue(self.config["missing_policy"]["placeholder_zero_requires_observed_mask_zero"])

    def test_config_loader_rejects_changed_missing_policy_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            text = CONFIG_PATH.read_text(encoding="utf-8").replace("status: approved_exp001_v0", "status: human_decision_required")
            path.write_text(text, encoding="utf-8")

            with self.assertRaises(ValueError):
                load_exp001_config(path)

    def test_dense_arrays_preserve_numeric_all_null_and_absent_states(self):
        numeric = GroupState()
        numeric.add_numeric(12.5)
        all_null = GroupState()
        all_null.add_null()
        ts0 = int(self.dt("2013-11-01 00:00").timestamp() * 1000)
        ts1 = int(self.dt("2013-11-01 00:10").timestamp() * 1000)
        groups = {
            ts0: {1: numeric, 2: all_null},
        }

        dense = build_dense_arrays(groups, [ts0, ts1], [1, 2])

        self.assertEqual(dense.values, [[12.5, 0.0], [0.0, 0.0]])
        self.assertEqual(dense.observed_mask, [[1, 0], [0, 0]])
        self.assertEqual(dense.state_code, [[STATE_NUMERIC_OBSERVED, STATE_ALL_NULL], [STATE_ABSENT, STATE_ABSENT]])

    def test_dense_arrays_preserve_invalid_only_state(self):
        invalid = GroupState()
        invalid.add_invalid()
        ts = int(self.dt("2013-11-01 00:00").timestamp() * 1000)

        dense = build_dense_arrays({ts: {1: invalid}}, [ts], [1])

        self.assertEqual(dense.values, [[0.0]])
        self.assertEqual(dense.observed_mask, [[0]])
        self.assertEqual(dense.state_code, [[STATE_INVALID_ONLY]])

    def test_dense_drift_preserves_mask_and_state_codes(self):
        numeric = GroupState()
        numeric.add_numeric(10.0)
        all_null = GroupState()
        all_null.add_null()
        ts = int(self.dt("2013-12-13 00:00").timestamp() * 1000)
        groups = {ts: {1: numeric, 2: all_null}}

        clean = build_dense_arrays(groups, [ts], [1, 2])
        drifted = build_dense_arrays(groups, [ts], [1, 2], drift_timestamp_ms=ts, affected_cells={1, 2}, factor=1.5)

        self.assertEqual(drifted.values, [[15.0, 0.0]])
        self.assertEqual(drifted.observed_mask, clean.observed_mask)
        self.assertEqual(drifted.state_code, clean.state_code)

    def test_cli_raw_dir_resolution_order(self):
        config = {
            "dataset": {
                "raw_dir": "configured_raw",
                "raw_dir_env": "MILAN_RAW_DIR",
            }
        }

        with patch.dict("os.environ", {"MILAN_RAW_DIR": "env_raw"}):
            self.assertEqual(resolve_raw_dir(config, Path("cli_raw")), (Path("cli_raw"), "--raw-dir"))
            self.assertEqual(resolve_raw_dir(config, None), (Path("configured_raw"), "dataset.raw_dir"))
            config["dataset"]["raw_dir"] = None
            self.assertEqual(resolve_raw_dir(config, None), (Path("env_raw"), "MILAN_RAW_DIR"))

    def test_cli_raw_dir_resolution_fails_when_unavailable(self):
        config = {"dataset": {"raw_dir": None, "raw_dir_env": "MILAN_RAW_DIR"}}

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit):
                resolve_raw_dir(config, None)

    def test_cli_output_dir_guard_rejects_tracked_repo_paths(self):
        validate_output_dir(Path("artifacts/local/exp-001"))

        with self.assertRaises(SystemExit):
            validate_output_dir(Path("runs/exp-001"))
        with self.assertRaises(SystemExit):
            validate_output_dir(REPO_ROOT / "configs" / "generated")
        with self.assertRaises(SystemExit):
            validate_output_dir(Path(tempfile.gettempdir()) / "exp001-external-output")

    def test_cli_dry_run_does_not_write_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            output_dir = REPO_ROOT / "artifacts" / "local" / "exp001-dry-run-test"
            raw_dir.mkdir()
            (raw_dir / "sms-call-internet-mi-2013-11-01.txt").write_text(
                "1\t1383260400000\t39\t0\t0\t0\t0\t1.0\n",
                encoding="utf-8",
            )
            argv = [
                "materialize_exp001_preprocessing.py",
                "--config",
                str(CONFIG_PATH),
                "--raw-dir",
                str(raw_dir),
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ]

            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(), 0)

            self.assertFalse(output_dir.exists())

    def test_npz_writer_structural_fallback_and_numpy_roundtrip_when_available(self):
        output_dir = REPO_ROOT / "artifacts" / "local" / "exp001-npz-writer-test"
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True)
        path = output_dir / "sample.npz"
        try:
            write_npz(
                path,
                DenseArrays(
                    values=[[1.5, 0.0]],
                    observed_mask=[[1, 0]],
                    state_code=[[STATE_NUMERIC_OBSERVED, STATE_ABSENT]],
                    timestamp_ms=[1383260400000],
                    cell_ids=[7, 8],
                ),
            )

            with zipfile.ZipFile(path) as archive:
                self.assertEqual(
                    sorted(archive.namelist()),
                    ["cell_ids.npy", "observed_mask.npy", "state_code.npy", "timestamp_ms.npy", "values.npy"],
                )
                expected = {
                    "values.npy": ("<f8", (1, 2), 16),
                    "observed_mask.npy": ("|u1", (1, 2), 2),
                    "state_code.npy": ("|u1", (1, 2), 2),
                    "timestamp_ms.npy": ("<i8", (1,), 8),
                    "cell_ids.npy": ("<i8", (2,), 16),
                }
                for name, (dtype, shape, payload_len) in expected.items():
                    header, payload = npy_header(archive.read(name))
                    self.assertEqual(header["descr"], dtype)
                    self.assertEqual(header["shape"], shape)
                    self.assertEqual(len(payload), payload_len)

            try:
                import numpy as np
            except ModuleNotFoundError:
                np = None
            if np is not None:
                loaded = np.load(path)
                self.assertEqual(loaded["values"].shape, (1, 2))
                self.assertEqual(float(loaded["values"][0, 0]), 1.5)
                self.assertEqual(int(loaded["observed_mask"][0, 1]), 0)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_cli_synthetic_materialization_writes_local_outputs_without_raw_path_leak(self):
        output_dir = REPO_ROOT / "artifacts" / "local" / "exp001-synthetic-materialization-test"
        shutil.rmtree(output_dir, ignore_errors=True)
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "synthetic_raw"
            raw_dir.mkdir()
            raw_text = (
                "1\t1383260400000\t39\t0\t0\t0\t0\t10.0\n"
                "1\t1383260400000\t40\t0\t0\t0\t0\t2.0\n"
                "2\t1383260400000\t39\t0\t0\t0\t0\t\n"
                "3\t1383260400000\t39\t0\t0\t0\t0\tbad\n"
                "4\t1386892800000\t39\t0\t0\t0\t0\t30.0\n"
            )
            (raw_dir / "sms-call-internet-mi-2013-11-01.txt").write_text(raw_text, encoding="utf-8")
            argv = [
                "materialize_exp001_preprocessing.py",
                "--config",
                str(CONFIG_PATH),
                "--raw-dir",
                str(raw_dir),
                "--output-dir",
                str(output_dir),
            ]

            try:
                stdout = io.StringIO()
                with patch.object(sys, "argv", argv), redirect_stdout(stdout):
                    self.assertEqual(cli_main(), 0)

                self.assertIn("Output directory: artifacts/local/exp001-synthetic-materialization-test", stdout.getvalue())
                self.assertNotIn(str(raw_dir), stdout.getvalue())
                required = [
                    "manifest.json",
                    "metadata.json",
                    "checksums.sha256",
                    "arrays/clean_train.npz",
                    "arrays/drifted_test_postdrift.npz",
                    "summaries/validation_checks.json",
                ]
                for relative in required:
                    self.assertTrue((output_dir / relative).exists(), relative)

                metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
                manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
                checks = json.loads((output_dir / "summaries" / "validation_checks.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["source_summary"]["raw_dir_source"], "--raw-dir")
                self.assertEqual(
                    metadata["source_summary"]["country_code_handling"],
                    "collapsed_by_summing_internet_values_over_countryCode_for_each_timestamp_cell",
                )
                self.assertIn("countryCode rows are collapsed", manifest["raw_aggregation"])
                split_counts = {row["split"]: row["samples"] for row in metadata["split_boundaries"]}
                self.assertEqual(split_counts["train"], 4032)
                self.assertEqual(split_counts["validation"], 1008)
                self.assertEqual(split_counts["test_predrift"], 1008)
                self.assertEqual(split_counts["test_postdrift"], 1008)
                self.assertEqual(split_counts["holdout"], 0)
                self.assertTrue(checks["placeholder_zero_mask_check"])
                self.assertTrue(checks["drift_preserves_mask_and_state_code"])

                serialized_text = "\n".join(
                    path.read_text(encoding="utf-8", errors="ignore")
                    for path in output_dir.rglob("*")
                    if path.is_file() and path.suffix != ".npz"
                )
                self.assertNotIn(str(raw_dir), serialized_text)
            finally:
                shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
