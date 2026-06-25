import io
import json
import shutil
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

from scripts.run_exp001_baselines import main as cli_main
from scripts.run_exp001_baselines import ensure_writable_output
from scripts.run_exp001_baselines import validate_output_dir
from thesis_traffic_drift.exp001 import DenseArrays, STATE_ABSENT, STATE_NUMERIC_OBSERVED, write_npz
from thesis_traffic_drift.exp001_training import (
    fit_historical_average,
    iter_window_targets,
    load_baseline_config,
    load_materialized_npz,
    run_baselines,
)


CONFIG_PATH = Path("configs/EXP-001-baselines.template.yaml")


def dense(values, masks=None):
    if masks is None:
        masks = [[1 for _ in row] for row in values]
    states = [[STATE_NUMERIC_OBSERVED if mask else STATE_ABSENT for mask in row] for row in masks]
    return DenseArrays(
        values=values,
        observed_mask=masks,
        state_code=states,
        timestamp_ms=[index * 600000 for index in range(len(values))],
        cell_ids=list(range(1, len(values[0]) + 1)),
    )


class Exp001TrainingTests(unittest.TestCase):
    def test_baseline_config_is_public_safe_and_preserves_approved_scope(self):
        config = load_baseline_config(CONFIG_PATH)

        self.assertEqual(config["experiment_id"], "EXP-001-stale-degradation-v0")
        self.assertFalse(config["holdout"]["use"])
        self.assertFalse(config["gpu"]["use"])
        self.assertEqual(config["dependencies"]["neural"], "deferred")
        text = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertNotIn("/home/", text)
        self.assertNotIn("PyTorch", text)
        self.assertNotIn("CUDA", text)

    def test_window_targets_preserve_chronology_and_do_not_cross_split_boundaries(self):
        arrays = dense([[1.0], [2.0], [3.0], [4.0], [5.0]])

        targets = list(iter_window_targets(arrays, input_length=3, horizon=1))

        self.assertEqual([(t.input_start, t.target_index) for t in targets], [(0, 3), (1, 4)])
        self.assertEqual([t.target_timestamp_ms for t in targets], [1800000, 2400000])

    def test_historical_average_uses_train_observed_values_only_and_ignores_placeholder_zero(self):
        train = dense([[2.0, 0.0], [4.0, 0.0], [0.0, 0.0]], masks=[[1, 0], [1, 0], [0, 0]])

        means = fit_historical_average(train)

        self.assertEqual(means, [3.0, None])

    def test_run_baselines_masks_unobserved_targets_and_unavailable_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "exp-001"
            arrays_dir = root / "arrays"
            arrays_dir.mkdir(parents=True)
            train = dense([[2.0, 0.0], [4.0, 0.0]], masks=[[1, 0], [1, 0]])
            pre = dense(
                [[0.0, 0.0], [10.0, 0.0], [20.0, 5.0], [30.0, 7.0]],
                masks=[[0, 0], [1, 0], [1, 1], [1, 1]],
            )
            post = dense(
                [[0.0, 0.0], [10.0, 0.0], [20.0, 5.0], [50.0, 7.0]],
                masks=[[0, 0], [1, 0], [1, 1], [1, 1]],
            )
            write_npz(arrays_dir / "clean_train.npz", train)
            write_npz(arrays_dir / "clean_test_predrift.npz", pre)
            write_npz(arrays_dir / "drifted_test_postdrift.npz", post)

            results = run_baselines(root, input_length=2, horizon=1)

        pre_last = results["baselines"]["last_value"]["clean_test_predrift"]
        post_last = results["baselines"]["last_value"]["drifted_test_postdrift"]
        hist_pre = results["baselines"]["historical_average"]["clean_test_predrift"]

        self.assertEqual(pre_last["valid_position_count"], 3)
        self.assertEqual(pre_last["unavailable_prediction_count"], 1)
        self.assertAlmostEqual(pre_last["masked_mae"], 22.0 / 3.0)
        self.assertAlmostEqual(pre_last["masked_rmse"], (204.0 / 3.0) ** 0.5)
        self.assertAlmostEqual(pre_last["masked_smape"], ((20.0 / 30.0) + (20.0 / 50.0) + (4.0 / 12.0)) / 3.0)
        self.assertAlmostEqual(post_last["masked_mae"], 42.0 / 3.0)
        self.assertAlmostEqual(post_last["masked_rmse"], (1004.0 / 3.0) ** 0.5)
        self.assertAlmostEqual(post_last["masked_smape"], ((20.0 / 30.0) + (60.0 / 70.0) + (4.0 / 12.0)) / 3.0)
        self.assertAlmostEqual(results["baselines"]["last_value"]["degradation_ratio"]["masked_mae"], (42.0 / 3.0) / (22.0 / 3.0))
        self.assertAlmostEqual(
            results["baselines"]["last_value"]["degradation_ratio"]["masked_rmse"],
            ((1004.0 / 3.0) ** 0.5) / ((204.0 / 3.0) ** 0.5),
        )
        self.assertEqual(hist_pre["valid_position_count"], 2)
        self.assertEqual(hist_pre["unavailable_prediction_count"], 2)
        self.assertAlmostEqual(hist_pre["masked_mae"], 22.0)

    def test_smape_zero_denominator_contributes_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "exp-001"
            arrays_dir = root / "arrays"
            arrays_dir.mkdir(parents=True)
            train = dense([[0.0], [0.0]])
            pre = dense([[0.0], [0.0], [0.0]])
            post = dense([[0.0], [0.0], [0.0]])
            write_npz(arrays_dir / "clean_train.npz", train)
            write_npz(arrays_dir / "clean_test_predrift.npz", pre)
            write_npz(arrays_dir / "drifted_test_postdrift.npz", post)

            results = run_baselines(root, input_length=1, horizon=1)

        metrics = results["baselines"]["last_value"]["clean_test_predrift"]
        self.assertEqual(metrics["valid_position_count"], 2)
        self.assertEqual(metrics["masked_mae"], 0.0)
        self.assertEqual(metrics["masked_rmse"], 0.0)
        self.assertEqual(metrics["masked_smape"], 0.0)

    def test_npz_loader_fails_clearly_for_missing_required_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("values.npy", b"not enough")

            with self.assertRaisesRegex(ValueError, "missing required member"):
                load_materialized_npz(path)

    def test_cli_output_guard_and_overwrite_behavior(self):
        validate_output_dir(Path("artifacts/local/exp-001/baselines"))
        with self.assertRaises(SystemExit):
            validate_output_dir(Path("runs/exp-001-baselines"))
        with self.assertRaises(SystemExit):
            validate_output_dir(REPO_ROOT / "configs" / "generated")

    def test_cli_rejects_non_empty_output_dir_without_overwrite(self):
        output_dir = REPO_ROOT / "artifacts" / "local" / "exp001-baseline-non-empty-test"
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True)
        (output_dir / "existing.txt").write_text("keep\n", encoding="utf-8")

        try:
            with self.assertRaises(SystemExit):
                ensure_writable_output(output_dir, overwrite=False)
            self.assertTrue((output_dir / "existing.txt").exists())
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_cli_dry_run_does_not_create_output(self):
        output_dir = REPO_ROOT / "artifacts" / "local" / "exp001-baseline-dry-run-test"
        shutil.rmtree(output_dir, ignore_errors=True)
        argv = [
            "run_exp001_baselines.py",
            "--config",
            str(CONFIG_PATH),
            "--input-dir",
            "artifacts/local/exp-001",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]

        try:
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(), 0)
            self.assertFalse(output_dir.exists())
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_cli_writes_public_safe_local_metric_outputs_from_synthetic_artifacts(self):
        output_dir = REPO_ROOT / "artifacts" / "local" / "exp001-baseline-cli-test"
        shutil.rmtree(output_dir, ignore_errors=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "exp-001"
            arrays_dir = root / "arrays"
            arrays_dir.mkdir(parents=True)
            train = dense([[1.0], [3.0]])
            pre = dense([[10.0], [20.0], [30.0]])
            post = dense([[10.0], [20.0], [40.0]])
            write_npz(arrays_dir / "clean_train.npz", train)
            write_npz(arrays_dir / "clean_test_predrift.npz", pre)
            write_npz(arrays_dir / "drifted_test_postdrift.npz", post)
            argv = [
                "run_exp001_baselines.py",
                "--input-dir",
                str(root),
                "--output-dir",
                str(output_dir),
                "--overwrite",
            ]

            try:
                stdout = io.StringIO()
                with patch.object(sys, "argv", argv), redirect_stdout(stdout):
                    self.assertEqual(cli_main(), 0)
                self.assertTrue((output_dir / "metrics_summary.json").exists())
                self.assertTrue((output_dir / "metrics_summary.csv").exists())
                payload = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
                serialized = json.dumps(payload)
                self.assertNotIn(str(root), stdout.getvalue())
                self.assertNotIn(str(root), serialized)
                self.assertEqual(payload["input_artifacts_root"], "local-only-input-artifacts")
                self.assertEqual(payload["holdout_status"], "not_loaded_not_used_exp001_v0")
                self.assertEqual(payload["baselines"]["last_value"]["clean_test_predrift"]["window_count"], 0)
            finally:
                shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
