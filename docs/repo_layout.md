# Repository Layout

Status: draft

## 1. Purpose

This document is a lightweight guide to what belongs in this public-safe
repository and where contributors should put small, reviewable artifacts.
It describes directory intent only. It does not approve experiments, define
roles, set stop conditions, or make thesis claims.
Use it when deciding whether a file is a source file, a public record, a draft
specification, a review note, or a local-only generated artifact.

## 2. Source-of-truth note

`THESIS_HARNESS.md` is the authority for operating rules, authority order,
roles, evidence standards, stop conditions, and the public repository boundary.
When this layout guide and `THESIS_HARNESS.md` appear to differ, follow
`THESIS_HARNESS.md` and ask the human to resolve the conflict.
`PROJECT_STATE.md` records the current draft project state and human-only
pending decisions. It should not be treated as final thesis approval.

## 3. Public repository boundary

This repository may contain public-safe code, configs, experiment specs, run
records, review notes, and small documentation files.
This repository must not contain raw Milan data, processed tensors,
checkpoints, credentials, private server paths, internal repository links,
private testbed configs, large generated artifacts, or unapproved thesis
claims.
Local artifacts may be described by category or ignored directory name, but not
by private/raw paths.

## 4. Top-level directory guide

`README.md` gives the short project description and public boundary summary.
`THESIS_HARNESS.md` defines the operating rules and public-safe boundary for
agent and human work in this repository.
`PROJECT_STATE.md` tracks current draft direction, paused or planned work, and
human-only pending decisions.
`configs/` is for small public-safe configuration files that can be reviewed
without private data or generated outputs.
`docs/` is for public-safe documentation, run records, layout notes, and other
small written records.
`docs/run_records/` is where committed public-safe run records belong after the
run record is approved for repository inclusion.
`experiment_specs/` is for proposed or approved experiment specifications,
including scope, assumptions, planned inputs, and planned outputs.
`rdr/` is for Research Decision Records that document human-approved research
decisions and their rationale.
`red_team/` is for public-safe challenge notes about assumptions, leakage
risks, validity risks, or claim boundaries.
`reviews/` is for public-safe review reports, implementation reviews, and
scope-control checks.
`runs/` is an ignored working area for local run outputs. Only `runs/.gitkeep`
is intended to be committed.
`scripts/` is for small public-safe utility scripts, audits, and automation
that do not embed private paths, data, credentials, or large generated outputs.
`src/` is for reusable public-safe source code for approved implementation
work.
`tests/` is for public-safe tests that verify code behavior without requiring
private datasets or committed generated artifacts.
`artifacts/local/` is an ignored local-only area for generated files that must
not be committed.

## 5. Local-only artifacts and ignored outputs

The `.gitignore` file treats `runs/*` as ignored while allowing
`runs/.gitkeep`.
The `.gitignore` file also ignores `artifacts/local/`.
Generated run outputs, temporary audit products, plots, metric files, logs,
drifted data, processed tensors, and checkpoints belong outside committed
history unless a later human-approved public-safe exception is made.
Committed run records should summarize evidence under `docs/run_records/`, not
store generated outputs under `runs/`.
Ignored local directories may be referenced in documentation by directory
category, but documentation should not reveal private machine locations.

## 6. Examples of what belongs where

A short public-safe config belongs under `configs/`.
A proposed experiment plan belongs under `experiment_specs/`.
A committed audit or experiment run record belongs under `docs/run_records/`.
A review of a script, spec, or run record belongs under `reviews/`.
A challenge note about leakage, assumptions, or claim risk belongs under
`red_team/`.
A human-approved research decision record belongs under `rdr/`.
A reusable implementation module belongs under `src/`.
A lightweight helper or audit entry point belongs under `scripts/`.
A test for a public-safe script or module belongs under `tests/`.
A generated local output belongs under an ignored local artifact area, not in
committed history.

## 7. Examples of what must not be committed

Do not commit raw Milan data.
Do not commit processed tensors or serialized training arrays.
Do not commit model checkpoints or large generated artifacts.
Do not commit credentials, tokens, environment secrets, or private configs.
Do not commit private server paths, internal repository links, or private
testbed details.
Do not commit local run outputs from `runs/`.
Do not commit local generated artifacts from `artifacts/local/`.
Do not commit thesis conclusions or result claims unless they are explicitly
approved and supported by the required evidence.

## 8. Maintenance rule

Keep this document descriptive, lightweight, and aligned with
`THESIS_HARNESS.md`, `PROJECT_STATE.md`, and `.gitignore`.
When a new top-level directory is added, update this guide with one concise
public-boundary line and avoid adding process rules already owned by
`THESIS_HARNESS.md`.
