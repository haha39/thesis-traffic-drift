import io
import tempfile
import unittest
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
    TrafficObservation,
    apply_multiplicative_level_shift,
    construct_split_intervals,
    drift_point,
    is_post_drift_test,
    load_exp001_config,
    parse_local_time,
    rank_topk_cells,
    segment_for_timestamp,
    select_affected_cells,
    training_means_by_cell,
)


CONFIG_PATH = Path("configs/EXP-001-stale-degradation.template.yaml")


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

    def test_missing_null_absent_modeling_policy_remains_human_decision_required(self):
        self.assertIsNone(self.config["missing_policy"]["modeling_policy"])
        self.assertEqual(self.config["missing_policy"]["status"], REQUIRED_MISSING_POLICY_STATUS)

    def test_config_loader_rejects_changed_missing_policy_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            text = CONFIG_PATH.read_text(encoding="utf-8").replace("status: human_decision_required", "status: approved")
            path.write_text(text, encoding="utf-8")

            with self.assertRaises(ValueError):
                load_exp001_config(path)

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

    def test_cli_dry_run_does_not_write_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            output_dir = Path(tmp) / "out"
            raw_dir.mkdir()
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


if __name__ == "__main__":
    unittest.main()
