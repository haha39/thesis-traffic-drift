# Project State

Status: draft

## Current Thesis Direction

This thesis is not primarily about proposing a stronger traffic prediction
model. The current working focus is traffic drift, stale model degradation, and
reducing the analytics unreliable gap.

The preferred direction is a model-agnostic bridge or correction mechanism that
can improve reliability when deployed predictors become stale under drift.

## Reproduction-Stage Conclusion

The reproduction stage is paused. STDenseNet is the strongest full-pipeline
reproduction record. HGCRN and FML are diagnostic or partial reproduction
records. FedDA is feasibility-only.

These records do not define the final thesis method and should not be treated as
final experiment results for this repository.

## Current Working Assumptions

- The main problem framing should emphasize stale model degradation under traffic
  drift.
- Recovery should be evaluated as a model-agnostic layer where possible.
- Repository artifacts should remain public-safe and avoid private data,
  infrastructure details, and unpublished claims.
- Any thesis-ready claim requires explicit approved run records.

## Candidate Main Dataset Setting

The candidate dataset setting is Milan Internet traffic with 10-minute
aggregation over topK active cells. This setting is not yet approved as final.

## Candidate Baseline Model Set

The candidate baseline set is Last Value / Historical Average, LSTM or GRU, and
TCN. ARIMA / exponential smoothing, DLinear / MLP, and STDenseNet may be added
later if approved.

## Offline-vs-Testbed Strategy

Main quantitative experiments should run offline on the server first. The
testbed should be used later for a small-scale NWDAF closed-loop demonstration,
not as the first place to establish the core quantitative evidence.

## Next Planned Experiment

EXP-001 stale model degradation v0 is the next planned experiment. It is not
implemented yet.

## Human-Only Pending Decisions

- Final thesis method, framing, and contribution claims.
- Final dataset setting, split, and disclosure boundaries.
- Final baseline model set.
- Drift scenario definition.
- Metrics, loss functions, and unreliable-gap threshold.
- Whether EXP-001 is approved to implement, run, or publish.

## Public Repository Boundary Reminder

This public repository may contain code, configs, experiment specs, run records,
and review notes. It must not contain raw datasets, processed tensors,
checkpoints, credentials, private server paths, private testbed configs, or large
generated artifacts unless explicitly approved as public-safe.
