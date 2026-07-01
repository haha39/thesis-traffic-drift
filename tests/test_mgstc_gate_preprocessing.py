import ast
import io
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

from scripts.materialize_mgstc_gate_preprocessing import main as cli_main
from scripts.materialize_mgstc_gate_preprocessing import resolve_raw_dir, validate_output_dir
from thesis_traffic_drift.mgstc_gate import (
    STATE_ABSENT,
    STATE_ALL_NULL,
    STATE_INVALID_ONLY,
    STATE_NUMERIC_OBSERVED,
    STATE_PARTIAL_MISSING,
    TEN_MINUTES_MS,
    DenseArrays,
    apply_train_only_minmax,
    build_chronological_splits,
    build_dense_arrays,
    compute_train_only_minmax,
    load_mgstc_gate_config,
    materialize_mgstc_gate,
    observed_values,
    parse_activity_value,
    read_day_field_groups,
    select_center_cell_ids,
    validate_mgstc_gate_config,
    write_npz,
)


CONFIG_PATH = REPO_ROOT / "configs" / "MGSTC-feasibility-gate.template.yaml"


def npy_header(payload):
    if not payload.startswith(b"\x93NUMPY\x01\x00"):
        raise AssertionError("expected NumPy v1.0 payload")
    header_len = struct.unpack("<H", payload[8:10])[0]
    self_aligned_len = 10 + header_len
    if self_aligned_len % 16:
        raise AssertionError("npy header is not 16-byte aligned")
    header = ast.literal_eval(payload[10:self_aligned_len].decode("latin1").strip())
    return header, payload[self_aligned_len:]


def dense(
    values,
    observed_mask,
    state_code,
    component_observed_count,
    component_missing_count,
    all_components_observed_mask,
    fully_observed_mask=None,
    timestamp_ms=None,
    cell_ids=None,
):
    if timestamp_ms is None:
        timestamp_ms = list(range(len(values)))
    if cell_ids is None:
        cell_ids = list(range(1, len(values[0]) + 1))
    if fully_observed_mask is None:
        fully_observed_mask = all_components_observed_mask
    return DenseArrays(
        values=values,
        observed_mask=observed_mask,
        fully_observed_mask=fully_observed_mask,
        state_code=state_code,
        component_observed_count=component_observed_count,
        component_missing_count=component_missing_count,
        all_components_observed_mask=all_components_observed_mask,
        timestamp_ms=timestamp_ms,
        cell_ids=cell_ids,
    )


def write_synthetic_milan_day(path, start_ms, day_index):
    rows = []
    for slot in range(144):
        timestamp_ms = start_ms + slot * TEN_MINUTES_MS
        internet_value = 5 + slot + (day_index * 1000)
        rows.append(f"4041\t{timestamp_ms}\t39\t1\t2\t3\t4\t{internet_value}\n")
        if slot % 2 == 0:
            rows.append(f"4042\t{timestamp_ms}\t39\t1\t\t3\t4\t5\n")
        else:
            rows.append(f"4042\t{timestamp_ms}\t39\t\t\t\t\t\n")
    path.write_text("".join(rows), encoding="utf-8")


class MgstcGatePreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.config = load_mgstc_gate_config(CONFIG_PATH)

    def test_fig6_inferred_900_region_selection_is_deterministic_for_100x100_grid(self):
        cell_ids = select_center_cell_ids(100, 100, 30, 30, row_start=40, col_start=40)

        self.assertEqual(len(cell_ids), 900)
        self.assertEqual(cell_ids[0], 4041)
        self.assertEqual(cell_ids[-1], 6970)
        self.assertEqual(len(set(cell_ids)), 900)

    def test_total_traffic_summation_is_correct_after_country_code_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sms-call-internet-mi-2013-11-01.txt"
            path.write_text(
                "3536\t1383260400000\t39\t1\t2\t3\t4\t5\n"
                "3536\t1383260400000\t40\t10\t20\t30\t40\t50\n",
                encoding="utf-8",
            )

            groups, _timestamps, _stats = read_day_field_groups(path, keep_cells={3536})
            dense = build_dense_arrays({1383260400000: {3536: groups[(1383260400000, 3536)]}}, [1383260400000], [3536], "total")

        self.assertEqual(dense.values, [[165.0]])
        self.assertEqual(dense.observed_mask, [[1]])
        self.assertEqual(dense.fully_observed_mask, [[1]])
        self.assertEqual(dense.state_code, [[STATE_NUMERIC_OBSERVED]])
        self.assertEqual(dense.component_observed_count, [[5]])
        self.assertEqual(dense.component_missing_count, [[0]])
        self.assertEqual(dense.all_components_observed_mask, [[1]])

    def test_total_traffic_partial_component_availability_remains_observed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sms-call-internet-mi-2013-11-01.txt"
            path.write_text(
                "3536\t1383260400000\t39\t1\t2\t3\t4\t5\n"
                "3537\t1383260400000\t39\t1\t\t3\t4\t5\n"
                "3538\t1383260400000\t39\t\t\t\t\t\n",
                encoding="utf-8",
            )

            groups, _timestamps, _stats = read_day_field_groups(path, keep_cells={3536, 3537, 3538})
            dense = build_dense_arrays(
                {
                    1383260400000: {
                        3536: groups[(1383260400000, 3536)],
                        3537: groups[(1383260400000, 3537)],
                        3538: groups[(1383260400000, 3538)],
                    }
                },
                [1383260400000],
                [3536, 3537, 3538, 3539],
                "total",
            )

        self.assertEqual(dense.values, [[15.0, 13.0, 0.0, 0.0]])
        self.assertEqual(dense.observed_mask, [[1, 1, 0, 0]])
        self.assertEqual(dense.fully_observed_mask, [[1, 0, 0, 0]])
        self.assertEqual(
            dense.state_code,
            [[STATE_NUMERIC_OBSERVED, STATE_PARTIAL_MISSING, STATE_ALL_NULL, STATE_ABSENT]],
        )
        self.assertEqual(dense.component_observed_count, [[5, 4, 0, 0]])
        self.assertEqual(dense.component_missing_count, [[0, 1, 5, 5]])
        self.assertEqual(dense.all_components_observed_mask, [[1, 0, 0, 0]])

    def test_total_traffic_all_missing_or_invalid_is_unobserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sms-call-internet-mi-2013-11-01.txt"
            path.write_text(
                "3536\t1383260400000\t39\t\t\t\t\t\n"
                "3537\t1383260400000\t39\tbad\tbad\tbad\tbad\tbad\n"
                "3538\t1383260400000\t39\t\tbad\t\tbad\t\n",
                encoding="utf-8",
            )

            groups, _timestamps, _stats = read_day_field_groups(path, keep_cells={3536, 3537, 3538})
            dense_total = build_dense_arrays(
                {
                    1383260400000: {
                        3536: groups[(1383260400000, 3536)],
                        3537: groups[(1383260400000, 3537)],
                        3538: groups[(1383260400000, 3538)],
                    }
                },
                [1383260400000],
                [3536, 3537, 3538],
                "total",
            )

        self.assertEqual(dense_total.values, [[0.0, 0.0, 0.0]])
        self.assertEqual(dense_total.observed_mask, [[0, 0, 0]])
        self.assertEqual(dense_total.fully_observed_mask, [[0, 0, 0]])
        self.assertEqual(dense_total.state_code, [[STATE_ALL_NULL, STATE_INVALID_ONLY, STATE_PARTIAL_MISSING]])
        self.assertEqual(dense_total.component_observed_count, [[0, 0, 0]])
        self.assertEqual(dense_total.component_missing_count, [[5, 5, 5]])
        self.assertEqual(dense_total.all_components_observed_mask, [[0, 0, 0]])

    def test_internet_fallback_component_counts_remain_single_component_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sms-call-internet-mi-2013-11-01.txt"
            path.write_text(
                "3536\t1383260400000\t39\t1\t\t3\t4\t5\n"
                "3537\t1383260400000\t39\t1\t2\t3\t4\t\n",
                encoding="utf-8",
            )

            groups, _timestamps, _stats = read_day_field_groups(path, keep_cells={3536, 3537})
            dense_internet = build_dense_arrays(
                {
                    1383260400000: {
                        3536: groups[(1383260400000, 3536)],
                        3537: groups[(1383260400000, 3537)],
                    }
                },
                [1383260400000],
                [3536, 3537],
                "internet",
            )

        self.assertEqual(dense_internet.values, [[5.0, 0.0]])
        self.assertEqual(dense_internet.observed_mask, [[1, 0]])
        self.assertEqual(dense_internet.fully_observed_mask, [[1, 0]])
        self.assertEqual(dense_internet.component_observed_count, [[1, 0]])
        self.assertEqual(dense_internet.component_missing_count, [[0, 1]])
        self.assertEqual(dense_internet.all_components_observed_mask, [[1, 0]])

    def test_parse_activity_value_preserves_numeric_null_and_invalid(self):
        self.assertEqual(parse_activity_value("12.5"), ("numeric", 12.5))
        self.assertEqual(parse_activity_value(""), ("null", None))
        self.assertEqual(parse_activity_value("bad"), ("invalid", None))

    def test_split_lengths_match_mgstc_5_2_55_setting(self):
        start = 1383260400000
        timestamps = [start + index * 600000 for index in range(8928)]

        splits = build_chronological_splits(timestamps, 5, 2, 55, 10)

        self.assertEqual(len(splits["train"]), 720)
        self.assertEqual(len(splits["validation"]), 288)
        self.assertEqual(len(splits["test"]), 7920)
        self.assertEqual(splits["validation"][0], start + 720 * 600000)
        self.assertEqual(splits["test"][0], start + (720 + 288) * 600000)

    def test_train_only_normalization_uses_train_statistics_only(self):
        train = dense(
            values=[[10.0, 20.0], [0.0, 30.0]],
            observed_mask=[[1, 1], [0, 1]],
            state_code=[[STATE_NUMERIC_OBSERVED, STATE_NUMERIC_OBSERVED], [STATE_ABSENT, STATE_NUMERIC_OBSERVED]],
            component_observed_count=[[5, 5], [0, 4]],
            component_missing_count=[[0, 0], [5, 1]],
            all_components_observed_mask=[[1, 1], [0, 0]],
            fully_observed_mask=[[1, 1], [0, 0]],
            timestamp_ms=[0, 1],
            cell_ids=[1, 2],
        )
        validation = dense(
            values=[[40.0, 0.0]],
            observed_mask=[[1, 0]],
            state_code=[[STATE_NUMERIC_OBSERVED, STATE_ABSENT]],
            component_observed_count=[[5, 0]],
            component_missing_count=[[0, 5]],
            all_components_observed_mask=[[1, 0]],
            fully_observed_mask=[[1, 0]],
            timestamp_ms=[2],
            cell_ids=[1, 2],
        )

        stats = compute_train_only_minmax(train, mask_name="fully_observed_mask")
        normalized_validation = apply_train_only_minmax(validation, stats)

        self.assertEqual((stats.minimum, stats.maximum, stats.observed_count), (10.0, 20.0, 2))
        self.assertEqual(observed_values(train, mask_name="fully_observed_mask"), [10.0, 20.0])
        self.assertEqual(normalized_validation.values, [[3.0, 0.0]])
        self.assertEqual(normalized_validation.observed_mask, validation.observed_mask)
        self.assertEqual(normalized_validation.fully_observed_mask, validation.fully_observed_mask)
        self.assertEqual(normalized_validation.component_observed_count, validation.component_observed_count)

    def test_builder1_config_rejects_observed_total_training_mask(self):
        config = dict(self.config)
        config["total_training_mask"] = "observed"

        with self.assertRaisesRegex(ValueError, "Builder-1 requires total_training_mask == 'fully_observed'"):
            validate_mgstc_gate_config(config)

    def test_cli_raw_dir_resolution_uses_env_placeholder_and_cli_override(self):
        config = load_mgstc_gate_config(CONFIG_PATH)
        with patch.dict("os.environ", {"MILAN_RAW_DIR": "/tmp/from-env"}):
            self.assertEqual(resolve_raw_dir(config, Path("/tmp/from-cli")), (Path("/tmp/from-cli"), "--raw-dir"))
            self.assertEqual(resolve_raw_dir(config, None), (Path("/tmp/from-env"), "MILAN_RAW_DIR"))

    def test_cli_raw_dir_resolution_fails_when_env_var_is_missing(self):
        config = load_mgstc_gate_config(CONFIG_PATH)
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit):
                resolve_raw_dir(config, None)

    def test_cli_output_dir_guard_rejects_tracked_or_external_paths(self):
        validate_output_dir(Path("artifacts/local/mgstc_gate"))
        with self.assertRaises(SystemExit):
            validate_output_dir(Path("runs/mgstc-gate"))
        with self.assertRaises(SystemExit):
            validate_output_dir(REPO_ROOT / "configs" / "generated")
        with self.assertRaises(SystemExit):
            validate_output_dir(Path(tempfile.gettempdir()) / "mgstc-gate-output")

    def test_cli_dry_run_validates_inputs_without_writing_outputs(self):
        output_dir = REPO_ROOT / "artifacts" / "local" / "mgstc-gate-dry-run-test"
        shutil.rmtree(output_dir, ignore_errors=True)
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            raw_dir.mkdir()
            (raw_dir / "sms-call-internet-mi-2013-11-01.txt").write_text(
                "3536\t1383260400000\t39\t1\t2\t3\t4\t5\n",
                encoding="utf-8",
            )
            argv = [
                "materialize_mgstc_gate_preprocessing.py",
                "--config",
                str(CONFIG_PATH),
                "--raw-dir",
                str(raw_dir),
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ]

            stdout = io.StringIO()
            with patch.object(sys, "argv", argv), redirect_stdout(stdout):
                self.assertEqual(cli_main(), 0)

            self.assertIn("MGSTC feasibility-gate preprocessing inputs validated.", stdout.getvalue())
            self.assertNotIn(str(raw_dir), stdout.getvalue())
            self.assertFalse(output_dir.exists())

    def test_npz_writer_emits_required_public_members(self):
        output_dir = REPO_ROOT / "artifacts" / "local" / "mgstc-gate-npz-test"
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True)
        path = output_dir / "sample.npz"
        try:
            write_npz(
                path,
                dense(
                    values=[[0.5, 0.0]],
                    observed_mask=[[1, 0]],
                    fully_observed_mask=[[0, 0]],
                    state_code=[[STATE_NUMERIC_OBSERVED, STATE_ABSENT]],
                    component_observed_count=[[3, 0]],
                    component_missing_count=[[2, 5]],
                    all_components_observed_mask=[[0, 0]],
                    timestamp_ms=[1383260400000],
                    cell_ids=[3536, 3537],
                ),
            )

            with zipfile.ZipFile(path) as archive:
                self.assertEqual(
                    sorted(archive.namelist()),
                    [
                        "all_components_observed_mask.npy",
                        "cell_ids.npy",
                        "component_missing_count.npy",
                        "component_observed_count.npy",
                        "fully_observed_mask.npy",
                        "observed_mask.npy",
                        "state_code.npy",
                        "timestamp_ms.npy",
                        "values.npy",
                    ],
                )
                expected = {
                    "values.npy": ("<f8", (1, 2), 16),
                    "observed_mask.npy": ("|u1", (1, 2), 2),
                    "fully_observed_mask.npy": ("|u1", (1, 2), 2),
                    "state_code.npy": ("|u1", (1, 2), 2),
                    "component_observed_count.npy": ("|u1", (1, 2), 2),
                    "component_missing_count.npy": ("|u1", (1, 2), 2),
                    "all_components_observed_mask.npy": ("|u1", (1, 2), 2),
                    "timestamp_ms.npy": ("<i8", (1,), 8),
                    "cell_ids.npy": ("<i8", (2,), 16),
                }
                for name, (dtype, shape, payload_len) in expected.items():
                    header, payload = npy_header(archive.read(name))
                    self.assertEqual(header["descr"], dtype)
                    self.assertEqual(header["shape"], shape)
                    self.assertEqual(len(payload), payload_len)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_synthetic_end_to_end_materialization_writes_metadata_manifest_and_expected_npz_members(self):
        output_dir = REPO_ROOT / "artifacts" / "local" / "mgstc-gate-synthetic-e2e-test"
        shutil.rmtree(output_dir, ignore_errors=True)
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            raw_dir.mkdir()
            start_ms = 1383260400000
            for day_index, day_name in enumerate(("2013-11-01", "2013-11-02", "2013-11-03")):
                write_synthetic_milan_day(
                    raw_dir / f"sms-call-internet-mi-{day_name}.txt",
                    start_ms + day_index * 144 * TEN_MINUTES_MS,
                    day_index,
                )

            config = {
                "experiment_id": "MGSTC-feasibility-gate-v0",
                "raw_milan_daily_dir": "${MILAN_RAW_DIR}",
                "output_dir": "artifacts/local/mgstc_gate",
                "traffic_field": "total",
                "grid_rows": 100,
                "grid_cols": 100,
                "center_rows": 30,
                "center_cols": 30,
                "center_row_start": 40,
                "center_col_start": 40,
                "timezone": "Europe/Rome",
                "resolution_minutes": 10,
                "expected_total_days": 3,
                "expected_total_timestamps": 432,
                "train_days": 1,
                "validation_days": 1,
                "test_days": 1,
                "input_length": 128,
                "prediction_horizon": 60,
                "normalization": "train_only_minmax",
                "allow_internet_fallback": True,
                "total_training_mask": "fully_observed",
                "missing_policy": self.config["missing_policy"],
            }

            try:
                metadata, manifest = materialize_mgstc_gate(config, raw_dir, output_dir, overwrite=False, raw_source="--raw-dir")

                metadata_path = output_dir / "metadata.json"
                manifest_path = output_dir / "manifest.json"
                self.assertTrue(metadata_path.exists())
                self.assertTrue(manifest_path.exists())

                metadata_disk = json.loads(metadata_path.read_text(encoding="utf-8"))
                manifest_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
                split_counts = {row["split"]: row["samples"] for row in metadata_disk["split_boundaries"]}
                self.assertEqual(split_counts, {"train": 144, "validation": 144, "test": 144})
                self.assertEqual(metadata_disk["normalization"]["policy"], "train_only_minmax")
                self.assertEqual(metadata_disk["normalization"]["fit_split"], "train")
                self.assertEqual(metadata_disk["normalization"]["fit_mask"], "fully_observed_mask")
                self.assertEqual(metadata_disk["normalization"]["total_training_mask_contract"], "fully_observed_only_for_builder1")
                self.assertEqual(metadata_disk["normalization"]["min_value"], 15.0)
                self.assertEqual(metadata_disk["normalization"]["max_value"], 158.0)
                self.assertEqual(
                    metadata_disk["traffic_field_note"],
                    "MGSTC-target total traffic scaffold across five Milan activity fields; not a faithful MGSTC reproduction",
                )
                self.assertTrue(metadata_disk["scaffold_only"])
                self.assertTrue(metadata_disk["not_full_mgstc_reproduction"])
                self.assertEqual(metadata_disk["paper_alignment_status"], "feasibility_scaffold_with_assumptions")
                self.assertTrue(metadata_disk["normalization_policy_not_paper_verified"])
                self.assertTrue(metadata_disk["fig6_inferred_900_region"])
                self.assertEqual(metadata_disk["square_id_mapping_assumption"], "row_major_unverified")
                self.assertEqual(metadata_disk["selection"]["center_row_start"], 40)
                self.assertEqual(metadata_disk["selection"]["center_col_start"], 40)
                self.assertEqual(metadata_disk["selection"]["square_id_mapping"]["assumption"], "row_major_unverified")
                self.assertEqual(metadata_disk["selection"]["square_id_mapping"]["row_formula"], "(square_id - 1) // 100")
                self.assertEqual(metadata_disk["selection"]["square_id_mapping"]["col_formula"], "(square_id - 1) % 100")
                self.assertEqual(metadata_disk["selection"]["square_id_mapping"]["range_source"], "fig6_visual_inference")
                self.assertEqual(
                    metadata_disk["selection"]["square_id_mapping"]["selected_region_definition"],
                    "rows_40_70_cols_40_70_fig6_inferred_30x30_region_under_row_major_mapping",
                )
                self.assertEqual(metadata_disk["partial_total_policy"]["default_safe_downstream_mask_for_total"], "fully_observed_mask")
                self.assertEqual(metadata["selection"]["square_id_mapping"], metadata_disk["selection"]["square_id_mapping"])
                self.assertEqual(manifest["arrays"], manifest_disk["arrays"])
                self.assertTrue(manifest_disk["scaffold_only"])
                self.assertTrue(manifest_disk["not_full_mgstc_reproduction"])
                self.assertEqual(manifest_disk["paper_alignment_status"], "feasibility_scaffold_with_assumptions")
                self.assertTrue(manifest_disk["fig6_inferred_900_region"])

                expected_npz_members = {
                    "values.npy",
                    "observed_mask.npy",
                    "fully_observed_mask.npy",
                    "state_code.npy",
                    "timestamp_ms.npy",
                    "cell_ids.npy",
                    "component_observed_count.npy",
                    "component_missing_count.npy",
                    "all_components_observed_mask.npy",
                }
                for relative in ("arrays/clean_train.npz", "arrays/clean_validation.npz", "arrays/clean_test.npz"):
                    with zipfile.ZipFile(output_dir / relative) as archive:
                        self.assertEqual(set(archive.namelist()), expected_npz_members)
            finally:
                shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
