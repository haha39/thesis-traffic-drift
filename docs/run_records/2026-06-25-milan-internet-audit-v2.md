# Milan Internet Audit v2 Run Record

Date: 2026-06-25

Status: run-record

## Purpose

This record documents the pre-EXP-001 Milan Internet data audit v2 review
result. The audit verified dataset structure, observation-state semantics,
Top-K evidence, split diagnostics, and drift-point diagnostics before any
EXP-001 implementation or run.

## Scope

- Dataset scope: Milan Internet traffic.
- Temporal resolution: 10-minute intervals.
- Audit type: descriptive audit only.
- Exclusions: no training, no drift injection, and no K, split, or drift-point
  decision.

## Source Changes Summarized

- `.gitignore` adds `artifacts/local/`.
- `scripts/audit_milan_internet.py` contains the read-only audit v2 workflow.
- `tests/test_audit_milan_internet.py` covers parsing, observation states,
  policy denominators, timestamp conversion, split boundaries, deterministic
  Top-K ranking, and train-only Top-K selection.
- `tests/__init__.py` is present.

## Local Artifact Location

Local audit artifacts are under:

```text
artifacts/local/data_audit_v2/
```

The `artifacts/local/` directory is ignored by git and is not committed.

## Verification Commands

```bash
python3 -m py_compile scripts/audit_milan_internet.py tests/test_audit_milan_internet.py
python3 -m unittest discover
MILAN_RAW_DIR=/path/to/local/milan/daily python3 scripts/audit_milan_internet.py --output-dir artifacts/local/data_audit_v2
```

## Reviewer Verdict

Reviewer verdict: PASS.

The review reported no blocker, major, or minor findings.

## Key Verified Facts

- Daily files: 62.
- Cells: 10000.
- Timestamps: 8928.
- Expected dense timestamp/cell pairs: 89,280,000.
- `NUMERIC_OBSERVED`: 89,127,473.
- `ALL_NULL`: 117,845.
- `ABSENT`: 34,682.
- Explicit zero: 0.
- Negative: 0.
- Non-finite: 0.
- Malformed: 0.
- Counts sum to the expected dense pairs.

## Policy Sensitivity

The audit reports three policy views:

- P1 `OBSERVED_ONLY`: masks `ALL_NULL` and `ABSENT`.
- P2 `NULL_ZERO_ABSENT_MASKED`: treats `ALL_NULL` as zero and masks `ABSENT`.
- P3 `LEGACY_ZERO_FILL`: treats `ALL_NULL` and `ABSENT` as zero.

No policy is canonical or approved. These views are sensitivity diagnostics
only.

## Top-K Evidence

- K=60, K=80, and K=100 were evaluated descriptively.
- Top-K membership is identical across P1, P2, and P3.
- Policy-pair Jaccard is 1.0.
- Rank correlation is 1.0.
- No final K decision was made.

## Split Diagnostics

The audit reports provisional chronological boundaries only:

- Train: 2013-11-01T00:00:00+01:00 to 2013-11-29T00:00:00+01:00,
  4032 timestamps.
- Validation: 2013-11-29T00:00:00+01:00 to 2013-12-06T00:00:00+01:00,
  1008 timestamps.
- Test: 2013-12-06T00:00:00+01:00 to 2013-12-20T00:00:00+01:00,
  2016 timestamps.
- Reserved holdout: 2013-12-20T00:00:00+01:00 to
  2014-01-02T00:00:00+01:00, 1872 timestamps.

These boundaries are structurally verified but not approved as final.

## Drift-Point Diagnostics

- Candidate boundary `2013-12-13 00:00 Europe/Rome` exists.
- 7-day pre-window structural coverage: 1008 timestamps.
- 7-day post-window structural coverage: 1008 timestamps.
- Top-60, Top-80, and Top-100 boundary completeness: 1.0.

The drift point is structurally verified but not approved as final.

## Caveats

- `countryCode` summation remains an explicit audit assumption, not externally
  proven semantics.
- The final null/absent semantic policy remains undecided.
- K, chronological split, and drift point remain human-only decisions.

## Performance And Process Note

Audit v2 passed performance/process review. Future full-data audit changes
should first use a small-file smoke/profile workflow before a full 62-file scan.

## Next Step

Resume human discussion of K, then chronological split, then drift injection
point.
