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
from scripts.diagnose_exp001_baselines import main as diagnose_cli_main
from scripts.diagnose_exp001_baselines import validate_output_dir as validate_diagnostic_output_dir
from thesis_traffic_drift.exp001 import DenseArrays, STATE_ABSENT, STATE_NUMERIC_OBSERVED, write_npz
from thesis_traffic_drift.exp001_training import (
    fit_historical_average,
    iter_window_targets,
    load_baseline_config,
    load_materialized_npz,
    run_baseline_diagnostics,
    run_baselines,
    validate_diagnostic_grouping,
)


CONFIG_PATH = Path("configs/EXP-001-baselines.template.yaml")


def dense(values, masks=None, cell_ids=None):
    if masks is None:
        masks = [[1 for _ in row] for row in values]
    if cell_ids is None:
        cell_ids = list(range(1, len(values[0]) + 1))
    states = [[STATE_NUMERIC_OBSERVED if mask else STATE_ABSENT for mask in row] for row in masks]
    return DenseArrays(
        values=values,
        observed_mask=masks,
        state_code=states,
        timestamp_ms=[index * 600000 for index in range(len(values))],
        cell_ids=cell_ids,
    )


def write_selection(path, cell_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("rank,cell_id,training_mean_observed_numeric\n")
        for rank, cell_id in enumerate(cell_ids, start=1):
            handle.write(f"{rank},{cell_id},{1000.0 - rank}\n")


def synthetic_diagnostic_root(tmp_path):
    root = Path(tmp_path) / "exp-001"
    arrays_dir = root / "arrays"
    arrays_dir.mkdir(parents=True)
    cell_ids = list(range(1001, 1101))
    affected = cell_ids[:20]
    train = dense([[10.0] * 100, [10.0] * 100], cell_ids=cell_ids)
    pre = dense([[10.0] * 100, [10.0] * 100, [10.0] * 100], cell_ids=cell_ids)
    clean_post = dense([[20.0] * 100, [20.0] * 100, [20.0] * 100], cell_ids=cell_ids)
    drifted_rows = []
    for _ in range(3):
        drifted_rows.append([30.0 if index < 20 else 20.0 for index in range(100)])
    drifted_post = dense(drifted_rows, cell_ids=cell_ids)
    write_npz(arrays_dir / "clean_train.npz", train)
    write_npz(arrays_dir / "clean_test_predrift.npz", pre)
    write_npz(arrays_dir / "clean_test_postdrift.npz", clean_post)
    write_npz(arrays_dir / "drifted_test_postdrift.npz", drifted_post)
    write_selection(root / "selection" / "affected20_cells.csv", affected)
    return root, cell_ids, affected


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

    def test_diagnostic_grouping_requires_top100_affected20_and_same_window_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cell_ids, affected = synthetic_diagnostic_root(tmp)
            artifacts = {
                split: load_materialized_npz(root / "arrays" / f"{split}.npz")
                for split in ("clean_test_predrift", "clean_test_postdrift", "drifted_test_postdrift")
            }

            grouping = validate_diagnostic_grouping(root, artifacts)

        self.assertEqual(grouping["cell_ids"], cell_ids)
        self.assertEqual(len(grouping["groups"]["all_top100"]), 100)
        self.assertEqual(len(grouping["groups"]["affected20"]), 20)
        self.assertEqual(len(grouping["groups"]["unaffected80"]), 80)
        self.assertEqual(grouping["affected_cell_ids"], affected)

    def test_diagnostic_historical_average_same_window_ratios_and_per_cell_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cell_ids, _ = synthetic_diagnostic_root(tmp)

            results = run_baseline_diagnostics(root, input_length=1, horizon=1)

        ratios = {
            (row["baseline"], row["group"], row["comparison"]): row
            for row in results["comparison_ratios"]
        }
        affected_ratio = ratios[("historical_average", "affected20", "clean_postdrift_vs_drifted_postdrift")]
        unaffected_ratio = ratios[("historical_average", "unaffected80", "clean_postdrift_vs_drifted_postdrift")]
        all_ratio = ratios[("historical_average", "all_top100", "clean_postdrift_vs_drifted_postdrift")]
        self.assertEqual(affected_ratio["denominator_split"], "clean_test_postdrift")
        self.assertEqual(affected_ratio["numerator_split"], "drifted_test_postdrift")
        self.assertEqual(affected_ratio["denominator_label"], "clean_test_postdrift_same_window_metric")
        self.assertAlmostEqual(affected_ratio["masked_mae_ratio"], 2.0)
        self.assertAlmostEqual(unaffected_ratio["masked_mae_ratio"], 1.0)
        self.assertAlmostEqual(all_ratio["masked_mae_ratio"], 1.2)

        first_cell = [
            row for row in results["per_cell_metrics"]
            if row["baseline"] == "historical_average"
            and row["split"] == "drifted_test_postdrift"
            and row["cell_id"] == cell_ids[0]
        ][0]
        self.assertEqual(first_cell["group"], "affected20")
        self.assertEqual(first_cell["window_count"], 2)
        self.assertEqual(first_cell["valid_position_count"], 2)
        self.assertAlmostEqual(first_cell["masked_mae"], 20.0)

    def test_diagnostic_last_value_is_labeled_adaptive_not_stale_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = synthetic_diagnostic_root(tmp)

            results = run_baseline_diagnostics(root, input_length=1, horizon=1)

        self.assertEqual(results["last_value_treatment"], "adaptive_sanity_baseline_not_stale_model_evidence")
        last_rows = [row for row in results["group_metrics"] if row["baseline"] == "last_value"]
        self.assertTrue(last_rows)
        self.assertTrue(all(row["baseline_treatment"] == "adaptive_sanity_baseline_not_stale_model_evidence" for row in last_rows))

    def test_diagnostic_missing_prediction_and_unobserved_target_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = synthetic_diagnostic_root(tmp)
            cell_ids = list(range(1001, 1101))
            train_masks = [[1] * 99 + [0], [1] * 99 + [0]]
            target_masks = [[1] * 100, [1] * 99 + [0], [1] * 100]
            train = dense([[10.0] * 100, [10.0] * 100], masks=train_masks, cell_ids=cell_ids)
            pre = dense([[10.0] * 100, [10.0] * 100, [10.0] * 100], masks=target_masks, cell_ids=cell_ids)
            write_npz(root / "arrays" / "clean_train.npz", train)
            write_npz(root / "arrays" / "clean_test_predrift.npz", pre)

            results = run_baseline_diagnostics(root, input_length=1, horizon=1)

        pre_all = [
            row for row in results["group_metrics"]
            if row["baseline"] == "historical_average" and row["split"] == "clean_test_predrift" and row["group"] == "all_top100"
        ][0]
        self.assertEqual(pre_all["window_count"], 2)
        self.assertEqual(pre_all["evaluated_position_count"], 200)
        self.assertEqual(pre_all["unobserved_target_count"], 1)
        self.assertEqual(pre_all["unavailable_prediction_count"], 1)
        self.assertEqual(pre_all["valid_position_count"], 198)
        self.assertAlmostEqual(pre_all["mask_coverage"], 199 / 200)

    def test_diagnostic_cli_output_guard_dry_run_and_public_safe_payload(self):
        validate_diagnostic_output_dir(Path("artifacts/local/exp-001/baselines/diagnostics"))
        with self.assertRaises(SystemExit):
            validate_diagnostic_output_dir(Path("runs/exp-001-diagnostics"))

        output_dir = REPO_ROOT / "artifacts" / "local" / "exp001-diagnostic-cli-test"
        shutil.rmtree(output_dir, ignore_errors=True)
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = synthetic_diagnostic_root(tmp)
            argv = [
                "diagnose_exp001_baselines.py",
                "--input-dir",
                str(root),
                "--output-dir",
                str(output_dir),
                "--overwrite",
            ]

            try:
                stdout = io.StringIO()
                with patch.object(sys, "argv", argv), redirect_stdout(stdout):
                    self.assertEqual(diagnose_cli_main(), 0)
                self.assertTrue((output_dir / "diagnosis_summary.json").exists())
                self.assertTrue((output_dir / "group_metrics.csv").exists())
                self.assertTrue((output_dir / "per_cell_metrics.csv").exists())
                self.assertTrue((output_dir / "comparison_ratios.csv").exists())
                self.assertTrue((output_dir / "counts_summary.csv").exists())
                payload = json.loads((output_dir / "diagnosis_summary.json").read_text(encoding="utf-8"))
                serialized = json.dumps(payload)
                self.assertNotIn(str(root), stdout.getvalue())
                self.assertNotIn(str(root), serialized)
                self.assertNotIn("/home/", serialized)
                self.assertEqual(payload["input_artifacts_root"], "local-only-input-artifacts")
                self.assertNotIn("holdout", serialized.lower())
            finally:
                shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
