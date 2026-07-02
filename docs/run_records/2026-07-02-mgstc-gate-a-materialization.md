# MGSTC Gate A Materialization Run Record

Date: 2026-07-02

Status: run-record

Scope: local-only MGSTC feasibility-gate materialization

Reviewer artifact check verdict: PASS

## Purpose

This record documents an already executed local-only MGSTC Gate A
materialization run for the Builder-1 preprocessing scaffold. This
documentation task did not rerun materialization.

This run is a preprocessing/materialization scaffold result only. It is not
full MGSTC reproduction, not EXP-002, not a Proceed-style feedback-delay
diagnostic, not model training, and not thesis evidence.

## Command

Public-safe command form:

```bash
python3 scripts/materialize_mgstc_gate_preprocessing.py \
  --config configs/MGSTC-feasibility-gate.template.yaml \
  --output-dir artifacts/local/mgstc_gate \
  --overwrite
```

The input raw-data mechanism is provided through
`configs/MGSTC-feasibility-gate.template.yaml`, which resolves
`raw_milan_daily_dir` from `${MILAN_RAW_DIR}`.

## Execution Summary

- Exit status: `0`
- Runtime: about `7m51s`
- Artifact root: `artifacts/local/mgstc_gate`
- Artifact size: `61M`
- Artifact status: local-only and ignored by git

## Outputs

Generated local-only artifacts:

- `artifacts/local/mgstc_gate/metadata.json`
- `artifacts/local/mgstc_gate/manifest.json`
- `artifacts/local/mgstc_gate/arrays/clean_train.npz`
- `artifacts/local/mgstc_gate/arrays/clean_validation.npz`
- `artifacts/local/mgstc_gate/arrays/clean_test.npz`
- `artifacts/local/mgstc_gate/indices/cell_index.csv`
- `artifacts/local/mgstc_gate/indices/split_index.csv`
- `artifacts/local/mgstc_gate/indices/timestamp_index.csv`
- `artifacts/local/mgstc_gate/summaries/observation_counts_by_split.csv`
- `artifacts/local/mgstc_gate/summaries/validation_checks.json`

These artifacts remained local-only ignored outputs and were not committed.

## Metadata Summary

- `daily_file_count = 62`
- `timestamp_count_present_in_raw = 8928`
- `selection.cell_count = 900`
- Selected region: Fig. 6 inferred `rows 40:70`, `cols 40:70`
- `square_id_mapping_assumption = row_major_unverified`
- `fig6_inferred_900_region = true`

## Split Counts

- Train = `720`
- Validation = `288`
- Test = `7920`

## Normalization

- Policy = `train_only_minmax`
- Fit split = `train`
- Fit mask = `fully_observed_mask`
- Observed count = `546428`
- Min = `3.7073461442022992`
- Max = `8816.218846058686`

## Observation Summary

Train:

- Expected pairs = `648000`
- `NUMERIC_OBSERVED = 546428`
- `PARTIAL_MISSING = 101572`
- Partial missing ratio is about `0.156747`

Validation:

- Expected pairs = `259200`
- `NUMERIC_OBSERVED = 216826`
- `PARTIAL_MISSING = 42374`
- Partial missing ratio is about `0.163480`

Test:

- Expected pairs = `7128000`
- `NUMERIC_OBSERVED = 5908219`
- `PARTIAL_MISSING = 1219416`
- Partial missing ratio is about `0.171074`
- `ABSENT = 365`
- Absent ratio is about `0.000051`

## Validation Checks

All reported checks passed:

- `center_cell_count = 900`
- `train_samples = 720`
- `validation_samples = 288`
- `test_samples = 7920`
- `placeholder_zero_mask_check = true`
- `fully_observed_mask_subset_check = true`
- `component_count_consistency_check = true`
- `normalization_source = train_only_minmax`
- `normalization_mask = fully_observed_mask`
- `traffic_field = total`

## Caveats

- This is not full MGSTC reproduction.
- This is not thesis evidence.
- This does not validate the MGSTC model architecture.
- This does not implement CGTA / FGSA.
- This does not implement an MGSTC drift monitor.
- This does not run a feedback-delay diagnostic.
- This does not train any model.
- Center-900 selection uses the Fig. 6 inferred row/col range and the
  row-major `square_id` mapping assumption.
- The normalization policy is documented but not paper-verified.
- Partial totals are preserved; clean normalization and downstream clean
  training defaults use `fully_observed_mask`.
- The partial missing ratio is non-trivial, about `15%` to `17%`, and must be
  considered before downstream modeling.

## Privacy And Repository Hygiene

- Privacy scan across generated JSON/CSV summaries and indices found no private
  raw-data path leakage.
- The raw-data input mechanism remained `${MILAN_RAW_DIR}` in recorded public
  metadata.
- Generated artifacts remained ignored and local-only under
  `artifacts/local/mgstc_gate/`.
- Nothing from this run was staged or committed.

## What This Run Does Not Claim

- No MGSTC reproduction result.
- No model accuracy result.
- No MSE / MAE result.
- No online forecasting result.
- No feedback-delay result.
- No EXP-002 result.
- No thesis-conclusion claim suitable for evidence use.

## Suggested Next Step

Create a small diagnostic/reporting task to summarize the partial-missing
distribution and decide whether downstream Builder-2 should use only fully
observed values, allow partial totals with explicit mask handling, or require
further paper/code evidence before modeling.

Do not proceed to model training yet.
