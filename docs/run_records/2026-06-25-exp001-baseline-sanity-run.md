# EXP-001 Baseline Sanity Run Record

Date: 2026-06-25

Status: run-record

Scope: local-only non-neural sanity baseline evaluation

## Purpose

This record documents the completed EXP-001 local-only sanity evaluation for
the approved non-neural baselines. It records command evidence, local artifact
locations, masking policy, and metric outputs for review traceability.

This record does not claim thesis results. The generated metric files remain
local-only ignored artifacts and must not be committed.

## Command

```bash
python3 scripts/run_exp001_baselines.py \
  --config configs/EXP-001-baselines.template.yaml \
  --input-dir artifacts/local/exp-001 \
  --output-dir artifacts/local/exp-001/baselines \
  --overwrite
```

The command completed successfully.

## Inputs

Public-safe inputs:

- Baseline config:
  `configs/EXP-001-baselines.template.yaml`
- Local materialized EXP-001 artifacts under:
  `artifacts/local/exp-001/`

The local materialized artifacts are ignored by git and are not committed.

## Outputs

Generated local-only outputs:

- `artifacts/local/exp-001/baselines/metrics_summary.json`
- `artifacts/local/exp-001/baselines/metrics_summary.csv`

These outputs are ignored local artifacts and must not be committed.

## Baselines

- Last Value / Naive.
- Historical Average.

No GRU, LSTM, TCN, PyTorch, CUDA, GPU execution, bridge method, retraining, or
unreliable-gap threshold logic was used in this run.

## Window And Evaluation

- Input length: 144 samples.
- Prediction horizon: 1 sample.
- Evaluation scale: original traffic scale.

Evaluated splits:

- `clean_test_predrift`
- `drifted_test_postdrift`

Holdout status:

- Holdout was not loaded.
- Holdout was not used.

## Mask Policy

- `NUMERIC_OBSERVED` uses actual numeric traffic values.
- `ALL_NULL` and `ABSENT` are missing / unobserved.
- Placeholder zeros are excluded by `observed_mask`.
- Metrics include only observed targets with available predictions.

## Metrics

Metrics computed:

- Masked MAE.
- Masked RMSE.
- Masked SMAPE.
- Degradation ratio = post-drift metric / pre-drift metric.

## Result Summary

| Baseline | Split | Masked MAE | Masked RMSE | Masked SMAPE |
| --- | --- | ---: | ---: | ---: |
| Last Value / Naive | `clean_test_predrift` | 56.645525139820855 | 89.51043085968864 | 0.08622294665348468 |
| Last Value / Naive | `drifted_test_postdrift` | 46.9390409588411 | 74.62113867358478 | 0.07071215929478207 |
| Historical Average | `clean_test_predrift` | 403.29214456436415 | 512.6806003741872 | 0.5825123512805551 |
| Historical Average | `drifted_test_postdrift` | 423.19296073690487 | 624.5489773703408 | 0.5700412387181879 |

Degradation ratios:

| Baseline | MAE Ratio | RMSE Ratio | SMAPE Ratio |
| --- | ---: | ---: | ---: |
| Last Value / Naive | 0.8286451726412496 | 0.833658579864916 | 0.8201083590771049 |
| Historical Average | 1.0493459057925305 | 1.2182028672715624 | 0.9785908186582627 |

Counts:

| Split | Window Count | Valid Position Count | Unobserved Target Count | Unavailable Prediction Count |
| --- | ---: | ---: | ---: | ---: |
| `clean_test_predrift` | 864 | 86400 | 0 | 0 |
| `drifted_test_postdrift` | 864 | 86400 | 0 | 0 |

## Reviewer Caveat

Last Value post-drift error is lower than pre-drift. This is not a thesis-result
claim and is not a blocker. Because Last Value predicts from recent values
within the same drifted post-drift split, it can adapt immediately to
level-shifted recent values. Do not frame this as stale-model degradation
evidence.

## Verification Evidence

The following checks passed:

```bash
python3 -m py_compile src/thesis_traffic_drift/exp001_training.py scripts/run_exp001_baselines.py tests/test_exp001_training.py
python3 -m unittest tests.test_exp001_training
python3 -m unittest discover
```

Observed test results:

- `tests.test_exp001_training`: 10 tests passed.
- Full unittest discovery: 43 tests passed.
- Reviewer verdict: PASS.
- Sensitive-pattern scan found no private paths or credentials in generated
  JSON/CSV; only a deliberate test assertion string matched `/home/`.

## What This Run Does Not Claim

- No thesis result is claimed.
- No stale-model degradation conclusion is claimed from Last Value behavior.
- No bridge or correction method was evaluated.
- No neural baseline was run.
- No retraining was run.
- No unreliable-gap duration was computed.
- Generated JSON/CSV outputs are not committed evidence by themselves.

## Suggested Next Step

Use this run record as public-safe traceability for the local sanity run. The
next Builder or Planner task should decide whether to perform a focused
baseline-result sanity diagnosis or proceed to separately approved neural
baseline planning.
