# MGSTC Feasibility Gate v0

Status: proposed draft

Scope: preprocessing and materialization scaffold only

## Purpose

This specification defines a narrow feasibility gate for MGSTC-aligned Milan
data preparation. It is not a full MGSTC reproduction, not EXP-001, and not a
feedback-delay diagnostic.

Builder-1 only aligns public-safe preprocessing definitions, split metadata,
normalization policy, and local-only materialization scaffolding with the MGSTC
paper setting described in local paper notes.

## Dataset Setting

- Dataset: Telecom Italia Milan traffic.
- Original grid: `100 x 100`.
- Evaluated region: deterministic `30 x 30` region = `900` cells targeted to
  the MGSTC Fig. 6 visual range.
- Total timeline: `8928` timestamps at `10`-minute resolution.
- Default traffic field: `total`.
- Total traffic definition:
  `SMS-in + SMS-out + Call-in + Call-out + Internet`.
- Optional fallback or ablation field: `internet`.
- Internet-only must be labeled as an optional fallback and not as full MGSTC
  reproduction.

## Cell Selection

- The selected 900-cell region is deterministic within this scaffold.
- The feasibility gate assumes Milan `square_id` values map to a `100 x 100`
  row-major grid.
- Mapping assumption:
  - `row = (square_id - 1) // 100`
  - `col = (square_id - 1) % 100`
- The default selected `30 x 30` region is rows `40:70` and cols `40:70`
  under this mapping.
- This range is inferred from the provided MGSTC Fig. 6 visual evidence. It is
  not proven by paper text or official code in the current repository context.
- Builder-1 must not replace this with Top-K selection or any ranking-based
  subset.
- This is an implementation assumption for the feasibility gate. If future
  source evidence contradicts the row-major mapping or the Fig. 6 inferred
  range, the materialization logic must be revisited.

## Chronological Split

- Train: first `5` days.
- Validation: next `2` days.
- Test: remaining `55` days.
- The split is the MGSTC-aligned `5:2:55` chronological protocol.

At `10`-minute resolution this corresponds to:

- Train: `720` timestamps.
- Validation: `288` timestamps.
- Test: `7920` timestamps.

## Recorded Protocol Parameters

- Historical input length: `T = 128`.
- Prediction horizon: `tau = 60`.

Builder-1 records these values for future protocol compatibility only. It does
not build model windows, train models, or run online updates.

## Missingness And Normalization Policy

- Builder-1 preserves explicit observation masks.
- Placeholder zeros for unobserved entries are allowed only when accompanied by
  `observed_mask = 0`.
- Default normalization policy: `train_only_minmax`.
- Train-only normalization means min/max statistics are fit on the training
  split only, then applied unchanged to validation and test.
- If `traffic_field = total`, an entry is treated as observed when at least one
  of the five Milan activity fields is numeric after timestamp/cell
  aggregation.
- Total traffic is the sum of available numeric components only.
- Missing components are tracked explicitly through component-level metadata and
  are not interpreted as confirmed zero traffic.
- Component metadata includes the numeric-component count, missing-component
  count, and an explicit mask for whether all five components were observed.
- `observed_mask` means at least one numeric component exists and therefore
  lower-bound partial totals may be present.
- `fully_observed_mask` means all five components are numeric and is the
  default safe mask for total-traffic normalization and downstream clean
  training use in this feasibility scaffold.
- Builder-1 keeps `total_training_mask: "fully_observed"` as a fixed contract.
  `"observed"` is not a supported alternative in this scaffold, even though
  partial totals remain preserved in the exported arrays and metadata.
- Any future Builder-2 use of partial totals for training or MGSTC-style model
  inputs must explicitly opt in and document that deviation.

This policy is a documented Builder-1 normalization choice. It is not a claim
that the paper’s exact original normalization has been fully resolved.

## Caveat Status

- This scaffold is not a faithful MGSTC reproduction.
- It is a feasibility scaffold toward the MGSTC paper setting with explicit
  assumptions.
- The total-traffic normalization policy is not paper-verified from current
  notes alone.

## Explicit Non-Goals

Builder-1 does not:

- implement MGSTC architecture;
- implement CGTA or FGSA;
- implement drift monitoring, fine-tuning, or aggressive update;
- train models;
- compute MSE / MAE result tables;
- compute cumulative online error curves;
- compute feedback-delay diagnostics;
- modify EXP-001 behavior or artifacts.

## Outputs

Generated materialization outputs are local-only and belong under:

```text
artifacts/local/mgstc_gate/
```

Expected local-only categories:

- `manifest.json`
- `metadata.json`
- `indices/`
- `arrays/clean_train.npz`
- `arrays/clean_validation.npz`
- `arrays/clean_test.npz`
- `summaries/`

Generated arrays and local artifacts must not be committed.
