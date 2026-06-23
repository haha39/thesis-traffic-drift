# Thesis Harness

## Purpose

This file defines the operating rules for thesis work in this public repository.
It keeps agent sessions scoped, reviewable, and separated from private data,
testbed details, and unpublished thesis claims.

## Authority Order

1. Human instructions in the current session.
2. Human-approved Research Decision Records (RDRs).
3. Human-approved experiment specifications.
4. This `THESIS_HARNESS.md`.
5. Repository documentation and committed project files.
6. Agent assumptions, only when they do not conflict with higher authority.

When instructions conflict, stop and ask the human to resolve the conflict.

## Human-Only Decisions

- Final thesis method, framing, and contribution claims.
- Whether an experiment is approved to run or publish.
- Whether a result is valid, reportable, or thesis-ready.
- Dataset inclusion, licensing interpretation, and disclosure boundaries.
- Train / validation / test split.
- Baseline model set.
- Drift scenario definition.
- Loss function, metric definition, and unreliable-gap threshold.
- Any commit, push, release, or public sharing action.

## Active Roles

- Planner: turns goals into scoped tasks, acceptance checks, and handoff notes.
- Builder: implements approved code, configs, specs, or documentation changes.
- Reviewer: checks correctness, scope control, evidence, and repository hygiene.
- Red-Team: optional role for challenging assumptions, leakage risks, and claims.

A session may switch modes within the same role after declaring the change.
Switching roles requires explicit human approval. Builder and Reviewer should
not review the same artifact in the same session unless the human explicitly
accepts the loss of independence.

## Session Start Template

```text
Role:
Task:
Approved scope:
Allowed actions:
Forbidden actions:
Expected output:
Evidence needed:
Stop conditions:
```

## Session End Handoff Template

```text
Role:
Completed:
Files changed:
Evidence checked:
Artifact labels:
Open questions:
Risks:
Suggested next step:
Proposed commit message:
```

## Evidence Standard

- Claims about repository state must cite files, diffs, commands, or review notes.
- Claims about experiment results require explicit run records and approved
  artifacts; otherwise label them as planned, proposed, or unverified.
- Do not infer thesis conclusions from code structure, draft specs, or partial
  outputs.
- Prefer concise evidence summaries over copying large logs into the repository.

## Artifact Status Labels

- `draft`: incomplete and not approved for use as evidence.
- `proposed`: ready for human review, not yet accepted.
- `approved`: accepted by the human for the stated purpose.
- `run-record`: documents an executed run without implying thesis validity.
- `rejected`: retained only for traceability or review context.

Every generated or edited artifact should have an appropriate status in its file,
handoff, or surrounding review notes.

## Stop Conditions

Stop and ask the human before continuing if:

- The task would add raw data, processed tensors, checkpoints, logs, or run
  artifacts.
- The task requires private paths, credentials, internal links, or testbed secrets.
- The task would define the final thesis method or claim a result exists.
- The task needs changes outside the approved scope.
- Instructions conflict or evidence is missing for a requested claim.
- A commit, push, release, or external publication step is requested implicitly.

## Public Repository Boundary

This repository may contain code, configs, experiment specs, run records, and
review notes suitable for public review. It must not contain raw datasets,
processed tensors, checkpoints, credentials, private server paths, internal repo
links, private testbed configs, or large generated artifacts unless the human
explicitly approves a public-safe exception.
