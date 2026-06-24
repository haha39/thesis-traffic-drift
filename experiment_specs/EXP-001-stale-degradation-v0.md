# EXP-001 Stale Model Degradation v0

Status: proposed draft

Owner: human decision required before implementation

Scope: offline experiment only

## Research Question

EXP-001 asks whether a model trained only on pre-drift Milan Internet traffic
becomes stale and degrades after a controlled post-drift traffic regime change.

EXP-001-v0 does not propose a bridge method yet. It does not evaluate
retraining yet. It only establishes stale degradation and the unreliable-gap
measurement basis for later approved experiments.

## Dataset Setting

- Dataset: Telecom Italia Milan Internet traffic.
- Temporal resolution: 10-minute intervals.
- Cell subset: Top-K active cells.
- K: 100.
- Top-K ranking: computed using only the training period.
- Supporting audit: observation-state audit v2, documented in
  `docs/run_records/2026-06-25-milan-internet-audit-v2.md`.
- Null / absent final semantic policy: remains documented as an assumption to
  be handled by implementation. Audit v2 policy views are sensitivity
  diagnostics only and are not canonical.

## Chronological Split

All boundaries are end-exclusive and expressed in `Europe/Rome`.

| Segment | Start | End | Samples |
| --- | --- | --- | ---: |
| Train | 2013-11-01 00:00 | 2013-11-29 00:00 | 4032 |
| Validation | 2013-11-29 00:00 | 2013-12-06 00:00 | 1008 |
| Test | 2013-12-06 00:00 | 2013-12-20 00:00 | 2016 |
| Reserved holdout | 2013-12-20 00:00 | dataset end | 1872 |

Ranking and normalization must use training data only.

The reserved holdout is not used in EXP-001-v0.

## Drift Injection Point

Drift point: 2013-12-13 00:00 `Europe/Rome`.

This point is selected as the structural midpoint of the 14-day primary test
period. The clean pre-drift test window is 7 days and the post-drift test window
is 7 days. The point is not selected based on model outcome.

## Controlled Drift Scenario

- Scenario: controlled synthetic drift.
- Drift type: upward multiplicative level shift.
- Affected cells: top 20% of Top-100 active cells, selected by
  training-period mean traffic ranking.
- Number of affected cells: 20.
- Unaffected cells: remaining 80.
- Main factor: 1.5.

Interval semantics:

- Clean pre-drift test window: [2013-12-06 00:00 `Europe/Rome`,
  2013-12-13 00:00 `Europe/Rome`).
- Post-drift test window: [2013-12-13 00:00 `Europe/Rome`,
  2013-12-20 00:00 `Europe/Rome`).
- The exact timestamp 2013-12-13 00:00 `Europe/Rome` is included in the
  post-drift window.
- Drift is applied when `t >= t_drift`.
- No 10-minute sample is dropped at the boundary.

Formal definition:

- Before the drift point, when `t < t_drift`, `x_prime[c,t] = x[c,t]`.
- At or after the drift point, when `t >= t_drift`, if `c` is affected,
  `x_prime[c,t] = 1.5 * x[c,t]`.
- At or after the drift point, when `t >= t_drift`, if `c` is unaffected,
  `x_prime[c,t] = x[c,t]`.

Factor sensitivity candidates are 1.2, 1.5, and 2.0. EXP-001-v0 main setting
uses 1.5 only unless the human later approves sensitivity runs.

## Baseline Candidates

The following baselines are proposed candidates, not final implementation
requirements:

- Last Value / Naive.
- Historical Average.
- GRU or LSTM.
- TCN.

The exact baseline set, hyperparameters, input length, prediction horizon, and
normalization implementation still require human approval before coding.

## Evaluation Protocol

- Train models on the train segment only.
- Select hyperparameters using the validation segment only.
- Evaluate the clean pre-drift test window and post-drift test window
  separately.
- Treat stale model deployment as no retraining and no parameter update after
  the drift point.
- Compare prediction error before and after drift.
- Do not include any bridge method in EXP-001-v0.

## Metrics

Proposed metrics:

- MAE.
- RMSE.
- SMAPE.
- Pre-drift error.
- Post-drift error.
- Degradation ratio.
- Peak degradation.
- Cumulative post-drift error.
- Unreliable gap duration as a planned metric. The threshold remains a
  human-approved decision for a later task.

## Leakage And Validity Checks

Implementation must enforce:

- Top-K ranking uses training data only.
- Normalization uses training data only.
- Validation is not used for final test selection.
- No post-drift data is used for training or hyperparameter selection.
- Drift factor is not chosen based on model results.
- Affected cells are selected before seeing model outcomes.
- Holdout remains untouched.

## Expected Artifacts

Future artifacts, if implementation is later approved:

- Config file.
- Deterministic drift transformation script/config.
- Generated drifted data, if materialized.
- Baseline training logs.
- Prediction outputs.
- Metric summary CSV.
- Degradation curve plot.
- Model checkpoints, if any are ever produced.
- Run record.
- Reviewer report.

These are expected future artifacts only. This specification does not claim that
any EXP-001 results or implementation artifacts exist.

Future generated artifacts should go under an ignored local directory by
default, such as `artifacts/local/`. This includes drifted data, prediction
outputs, logs, plots, metric CSVs, and model checkpoints, if any are ever
produced. Processed tensors, large artifacts, checkpoints, and private paths
must not be committed. Only small public-safe summaries or run records may be
committed after human approval.

## Human-Only Pending Decisions

- Final baseline model set.
- Input window and prediction horizon.
- Normalization policy implementation.
- Null / absent handling policy for modeling.
- Unreliable-gap threshold.
- Whether to run factor sensitivity.
- Whether results are reportable.
- Commit / push / release decisions.

## Public Boundary

EXP-001-v0 artifacts must not include:

- Raw Milan data.
- Processed tensors.
- Checkpoints.
- Private server paths.
- Large artifacts.
- Unapproved thesis claims.

Current scope-control status:

- No EXP-001 training code has been implemented yet.
- No EXP-001 drifted data has been generated yet.
- No EXP-001 model training or evaluation has been run yet.
- No EXP-001 results exist yet.

## Next Step After Spec Approval

After this spec is approved and committed, the next step is implementation
planning for deterministic preprocessing and drift-transformation code.
Implementation still requires a separate approved Builder task and independent
review.
