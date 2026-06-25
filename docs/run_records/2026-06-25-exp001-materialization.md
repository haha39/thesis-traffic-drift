# EXP-001 Local Materialization Run Record

Date: 2026-06-25

Status: run-record

Scope: local-only materialization, not model training

## Purpose

This run materialized EXP-001 Top-100 clean and drifted Milan Internet time
series artifacts with observation masks for future baseline training planning.
It produced local-only preprocessing artifacts and did not train, evaluate, or
claim any EXP-001 result.

## Inputs

Public-safe inputs:

- EXP-001 specification:
  `experiment_specs/EXP-001-stale-degradation-v0.md`.
- Public config template:
  `configs/EXP-001-stale-degradation.template.yaml`.
- Local Milan Internet daily raw files. The private raw directory path is not
  recorded in this run record.
- Observation-state audit context:
  `docs/run_records/2026-06-25-milan-internet-audit-v2.md`.
- EXP-001-v0 observation policy:
  `NUMERIC_OBSERVED` uses actual numeric values; `ALL_NULL` and `ABSENT` are
  modeled as missing / unobserved, not semantic zero; placeholder zeros require
  explicit masks.

## Outputs

Generated outputs are local-only artifacts under:

```text
artifacts/local/exp-001/
```

Output categories:

- `manifest.json`
- `metadata.json`
- `checksums.sha256`
- selection CSVs
- index CSVs
- value/mask/state `.npz` arrays
- summaries and validation checks

These outputs are ignored local artifacts. Generated arrays are not committed.
Raw data and private paths are not committed.

## Materialized Splits

Materialized split artifacts:

| Artifact | Split | Shape | Status |
| --- | --- | ---: | --- |
| `clean_train` | train | 4032 x 100 | clean |
| `clean_validation` | validation | 1008 x 100 | clean |
| `clean_test_predrift` | test pre-drift | 1008 x 100 | clean |
| `clean_test_postdrift` | test post-drift | 1008 x 100 | clean |
| `drifted_test_postdrift` | test post-drift | 1008 x 100 | drifted |
| `holdout_clean` | holdout | 1872 x 100 | reserved, not used in EXP-001-v0 |

## Observation Policy

- `NUMERIC_OBSERVED` uses the actual numeric traffic value.
- `ALL_NULL` is missing / unobserved, not semantic zero.
- `ABSENT` is missing / unobserved, not semantic zero.
- Missing positions may use `0` only as a placeholder.
- `observed_mask` distinguishes observed numeric values from placeholder zeros.
- `state_code` preserves `NUMERIC_OBSERVED`, `ALL_NULL`, `ABSENT`, and
  `INVALID_ONLY`.
- Drifted arrays preserve the same masks and state codes as the corresponding
  clean arrays.

## Drift Transformation

- Drift point: 2013-12-13 00:00 `Europe/Rome`.
- Drift applies when `t >= drift_point`.
- Affected cells: 20 cells from the Top-100 training-period ranking.
- Factor: 1.5.
- Drift was applied only to affected observed post-drift entries.
- No 10-minute boundary sample was dropped.

## Verification Summary

Reviewer verdict: PASS.

Reviewer-confirmed facts:

- Checksums were validated for all 19 generated local files.
- Split and array shapes match EXP-001.
- Drift boundary was verified.
- Drifted arrays preserve mask and state codes.
- Holdout is reserved and not used in EXP-001-v0.
- No private paths or sensitive patterns were found.
- Drift spot-check confirmed 20,160 changed entries.
- Changed entries were affected cells only.
- Drift factor was 1.5.

Local artifact validation checks recorded:

- `topk_count`: 100.
- `affected_count`: 20.
- `placeholder_zero_mask_check`: true.
- `drift_preserves_mask_and_state_code`: true.
- `no_boundary_sample_dropped`: true.

Additional materialization facts:

- Local daily files discovered: 62.
- Raw timestamps present: 8928.
- Timestamp range: 2013-11-01T00:00:00+01:00 through
  2014-01-01T23:50:00+01:00.
- `countryCode` rows were collapsed by summing Internet values for each
  timestamp/cell before dense materialization.

## Reviewer Caveat

- `metadata.json.file_parse_summary[].aggregated_timestamp_cell_records`
  reflects the optimized train-ranking scan.
- Non-train files may show `0` for that field.
- Do not interpret it as full-file aggregation coverage in this run record.

## What This Run Does Not Claim

- No model training was run.
- No baseline evaluation was run.
- No metrics were computed.
- No bridge or correction method was evaluated.
- No EXP-001 thesis result is claimed.
- Generated artifacts are not thesis-ready evidence by themselves.

## Next Step

Review this run record. After explicit human approval, commit public-safe code,
config, tests, and this run record, excluding local generated artifacts. After
that, proceed to model-training planning.
