# EXP-001 Baseline Diagnosis Run Record

Date: 2026-06-25

Status: run-record

Scope: local-only baseline-result sanity diagnosis

Reviewer verdict: PASS

## Purpose

This record documents a focused EXP-001 local-only diagnosis of existing
non-neural baseline outputs and materialized arrays. The diagnosis checks
whether affected cells behave differently from unaffected cells, whether the
same-window synthetic drift comparison shows measurable degradation, and whether
aggregate metrics can hide cell-level degradation.

This record does not claim thesis results. The generated diagnostic files remain
local-only ignored artifacts and must not be committed.

## Command

```bash
python3 scripts/diagnose_exp001_baselines.py \
  --config configs/EXP-001-baselines.template.yaml \
  --input-dir artifacts/local/exp-001 \
  --output-dir artifacts/local/exp-001/baselines/diagnostics \
  --overwrite
```

The command completed successfully.

## Inputs

Public-safe inputs:

- Baseline config:
  `configs/EXP-001-baselines.template.yaml`
- Existing local materialized EXP-001 artifacts under:
  `artifacts/local/exp-001/`
- Existing local non-neural baseline output area under:
  `artifacts/local/exp-001/baselines/`

The local materialized artifacts and generated diagnostics are ignored by git
and are not committed.

## Outputs

Generated local-only diagnostics:

- `artifacts/local/exp-001/baselines/diagnostics/diagnosis_summary.json`
- `artifacts/local/exp-001/baselines/diagnostics/diagnosis_summary.csv`
- `artifacts/local/exp-001/baselines/diagnostics/group_metrics.csv`
- `artifacts/local/exp-001/baselines/diagnostics/per_cell_metrics.csv`
- `artifacts/local/exp-001/baselines/diagnostics/comparison_ratios.csv`
- `artifacts/local/exp-001/baselines/diagnostics/counts_summary.csv`

These files are local-only ignored artifacts.

## Baselines And Treatment

- Historical Average is treated as the primary stale-baseline diagnostic.
- Last Value / Naive is labeled as an adaptive sanity baseline and is not used
  as stale-model degradation evidence.

No GRU, LSTM, TCN, PyTorch, CUDA, GPU execution, bridge method, retraining, or
unreliable-gap duration logic was used.

## Diagnostic Scope

Evaluated splits:

- `clean_test_predrift`
- `clean_test_postdrift`
- `drifted_test_postdrift`

Cell groups:

- All Top-100 cells.
- Affected 20 cells.
- Unaffected 80 cells.
- Per-cell metrics for all Top-100 cells.

Same-window synthetic drift effect is evaluated by comparing:

```text
clean_test_postdrift -> drifted_test_postdrift
```

The diagnosis required matching timestamps and cell IDs for the clean and
drifted post-drift arrays.

## Metrics And Counts

Metrics computed:

- Masked MAE.
- Masked RMSE.
- Masked SMAPE.
- Valid target/prediction counts.
- Unobserved target counts.
- Unavailable prediction counts.
- Window counts.
- Mask coverage.
- Degradation ratios with explicit denominator labels.

Counts summary:

| Baseline | Split | Group | Windows | Cells | Valid Count | Unobserved Targets | Unavailable Predictions | Mask Coverage |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Last Value / Naive | all evaluated splits | All Top-100 | 864 | 100 | 86400 | 0 | 0 | 1.0 |
| Last Value / Naive | all evaluated splits | Affected 20 | 864 | 20 | 17280 | 0 | 0 | 1.0 |
| Last Value / Naive | all evaluated splits | Unaffected 80 | 864 | 80 | 69120 | 0 | 0 | 1.0 |
| Historical Average | all evaluated splits | All Top-100 | 864 | 100 | 86400 | 0 | 0 | 1.0 |
| Historical Average | all evaluated splits | Affected 20 | 864 | 20 | 17280 | 0 | 0 | 1.0 |
| Historical Average | all evaluated splits | Unaffected 80 | 864 | 80 | 69120 | 0 | 0 | 1.0 |

## Historical Average Same-Window Summary

Same-window ratio definition:

```text
drifted_test_postdrift metric / clean_test_postdrift metric
```

Explicit denominator label:

```text
clean_test_postdrift_same_window_metric
```

| Group | MAE Ratio | RMSE Ratio | SMAPE Ratio |
| --- | ---: | ---: | ---: |
| All Top-100 | 1.1214897504703127 | 1.3144368543888254 | 0.9897587211871991 |
| Affected 20 | 1.3829849232160076 | 1.5917419902671204 | 0.9520033973629918 |
| Unaffected 80 | 1.0 | 1.0 | 1.0 |

This table is a local diagnostic summary only. It is not a thesis-ready result
or a final validity claim.

## Verification Evidence

The following checks passed:

```bash
python3 -m py_compile src/thesis_traffic_drift/exp001_training.py scripts/run_exp001_baselines.py scripts/diagnose_exp001_baselines.py tests/test_exp001_training.py
python3 -m unittest tests.test_exp001_training
python3 -m unittest discover
```

Observed test results:

- `tests.test_exp001_training`: 15 tests passed.
- Full unittest discovery: 48 tests passed.
- Reviewer verdict: PASS.
- Sensitive-pattern scan found no private absolute paths or forbidden result
  framing in generated diagnostic files.

## What This Run Does Not Claim

- No thesis result is claimed.
- No result is claimed as thesis-ready evidence.
- No stale-model degradation conclusion is claimed from Last Value behavior.
- No bridge or correction method was evaluated.
- No neural baseline was run.
- No retraining was run.
- No unreliable-gap duration was computed.
- No holdout data was loaded or used.

## Suggested Next Step

Use this run record as public-safe traceability for the local diagnosis. The
diagnostics support moving to separately approved neural baseline planning as a
planning recommendation only, not as a thesis claim.
