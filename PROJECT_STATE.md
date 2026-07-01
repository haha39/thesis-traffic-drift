# Project State

Status: draft

## Current Thesis Direction

NWDAF closed-loop operation remains the system context for this repository.
Within that context, the working research direction is drifting toward
drift-aware online network traffic forecasting rather than only offline model
comparison.

The current evaluation anchors under consideration are:

- Proceed-style feedback delay / temporal-gap framing.
- MGSTC-style online cellular traffic forecasting.
- Drift-aware handling of stale-model degradation and the analytics unreliable
  gap.

These anchors are still under evaluation. They do not yet define a final thesis
method or contribution claim.

## Reproduction-Stage Conclusion

The reproduction stage is paused. STDenseNet is the strongest full-pipeline
reproduction record. HGCRN and FML are diagnostic or partial reproduction
records. FedDA is feasibility-only.

These records do not define the final thesis method and should not be treated as
final experiment results for this repository.

## Current Working Assumptions

- The main problem framing remains inside NWDAF closed-loop network analytics.
- The active direction is no longer only "offline stale degradation" but the
  broader question of drift-aware online traffic forecasting under deployment
  mismatch.
- Proceed-style feedback delay and MGSTC-style cellular forecasting are
  candidate anchors, not approved final framing.
- Recovery should be evaluated as a model-agnostic layer where possible, unless
  a later approved direction requires a different framing.
- Repository artifacts should remain public-safe and avoid private data,
  infrastructure details, and unpublished claims.
- Any thesis-evidence claim requires explicit approved run records.

## Completed Work

- EXP-001 controlled local-shift harness is completed as local public-safe
  sanity evidence.
- EXP-001 non-neural baseline sanity / diagnosis is completed and documented in
  `docs/run_records/2026-06-25-exp001-baseline-diagnosis.md`.
- EXP-001 is not approved thesis evidence and should not be cited as a final
  experiment result.
- MGSTC Feasibility Gate Builder-1 scaffold is implemented, reviewed,
  red-teamed, fixed, and committed as preprocessing/materialization scaffold
  work.
- The MGSTC gate is not full MGSTC reproduction, not EXP-002, and not a
  Proceed-style feedback-delay diagnostic.
- Full real-data MGSTC materialization has not been run.

## Candidate Dataset Settings

- EXP-001: Milan Internet Top-100 controlled local-shift harness at 10-minute
  resolution. This setting exists as sanity evidence only and is not a
  thesis-evidence result.
- MGSTC gate: Milan total-traffic 900-cell feasibility scaffold using the
  Fig. 6 inferred rows `40:70` / cols `40:70` region under the explicit
  row-major `square_id` mapping assumption.
- MGSTC full reproduction has not been achieved.

## Candidate Baseline Model Set

The candidate baseline set is Last Value / Historical Average, LSTM or GRU, and
TCN. ARIMA / exponential smoothing, DLinear / MLP, and STDenseNet may be added
later if approved.

## Offline-vs-Testbed Strategy

Main quantitative experiments should run offline on the server first. The
testbed should be used later for a small-scale NWDAF closed-loop demonstration,
not as the first place to establish the core quantitative evidence.

## Next Planned Task

MGSTC Gate A real-data dry-run / materialization planning is the next planned
task.

That next step is planning-only. Any full real-data materialization execution
requires separate human approval before it is run.

## Human-Only Pending Decisions

- Final thesis method, framing, and contribution claims.
- Final dataset setting, split, and disclosure boundaries.
- Final baseline model set.
- Drift scenario definition.
- Metrics, loss functions, and unreliable-gap threshold.
- Whether Proceed-style feedback-delay framing or MGSTC-style online forecasting
  becomes the primary problem anchor.
- Whether MGSTC real-data dry-run or materialization is approved to execute.
- Whether any result is approved as thesis evidence, reportable, or
  publishable.

## Public Repository Boundary Reminder

This public repository may contain code, configs, experiment specs, run records,
and review notes. It must not contain raw datasets, processed tensors,
checkpoints, credentials, private server paths, private testbed configs, or
generated artifacts unless explicitly approved as public-safe.
