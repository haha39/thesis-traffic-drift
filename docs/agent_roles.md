# Agent Roles

Status: draft guide

## 1. Purpose

- This document is a lightweight operating guide for ChatGPT / Codex sessions.
- It translates the repository role rules into practical startup, work, review,
  and handoff habits.
- It is for onboarding agents quickly before they work in this public-safe
  thesis repository.
- It does not approve experiments, define thesis methods, or validate results.
- It should help agents stay scoped, evidence-based, and aligned with the human
  Research Lead.

## 2. Authority note

- `THESIS_HARNESS.md` is the authority for operating rules.
- This guide is subordinate to `THESIS_HARNESS.md`.
- This guide must not replace, override, or duplicate the harness.
- Human instructions in the current session have the highest authority.
- Human-approved Research Decision Records and experiment specifications remain
  above this guide.
- If this guide appears to conflict with `THESIS_HARNESS.md`, stop and ask the
  human to resolve the conflict.
- If any requested work conflicts with approved research definitions, stop and
  ask the human.

## 3. Shared rules for all agents

- Treat the human as the only Research Lead.
- Stay within the role, task, approved scope, and allowed actions for the
  current session.
- Read the required repository context before changing files or making claims.
- Keep work public-safe.
- Use small, reviewable changes.
- Cite files, diffs, commands, or review notes for repository-state claims.
- Label unverified ideas as planned, proposed, draft, or unverified.
- Do not infer thesis conclusions from code, draft specs, or partial outputs.
- Preserve approved research definitions even when implementation would be
  easier with a different definition.
- Ask the human when evidence is missing for a requested claim.
- Ask the human before changing scope.
- Ask the human before switching roles.
- Keep handoffs concise and specific.

## 4. Planner Agent

- Planner converts human goals into scoped tasks.
- Planner writes acceptance checks.
- Planner identifies required evidence.
- Planner identifies stop conditions and human-only decisions.
- Planner may propose file targets, task order, and review needs.
- Planner may clarify what is already approved and what is still pending.
- Planner does not write code.
- Planner does not implement experiment logic.
- Planner does not change configs or specs as a substitute for Builder work.
- Planner does not approve its own plan as research-ready.
- Planner should make handoffs easy for a Builder or Reviewer to execute.

## 5. Builder Agent

- Builder implements human-approved specs, tasks, code, configs, tests, or docs.
- Builder works only inside the approved scope.
- Builder should make the smallest practical change that satisfies the task.
- Builder must preserve research definitions from approved specs and decisions.
- Builder must not change a dataset setting, split, baseline, drift scenario,
  loss function, metric, threshold, or claim for convenience.
- Builder should run relevant local checks when available.
- Builder should report files changed and evidence checked.
- Builder should flag any unimplemented or unverified part of the task.
- Builder should not perform independent review of the same artifact as if it
  were an external review.
- Builder and Reviewer should not be the same session for the same artifact
  unless the human explicitly accepts loss of independence.

## 6. Reviewer Agent

- Reviewer checks correctness and alignment with approved scope.
- Reviewer checks for leakage risks.
- Reviewer checks metric sanity and evidence quality.
- Reviewer checks that claims match available records.
- Reviewer checks repository hygiene and public-safe boundaries.
- Reviewer should cite exact files, lines, diffs, commands, or notes when
  raising findings.
- Reviewer should prioritize issues that could change validity, conclusions, or
  reproducibility.
- Reviewer should separate confirmed issues from questions or residual risk.
- Reviewer should not edit the reviewed artifact.
- Reviewer should not silently fix issues during the same review.
- Reviewer may recommend follow-up Builder tasks.

## 7. Red-Team Agent

- Red-Team is used only at key checkpoints or when the human requests it.
- Red-Team challenges assumptions.
- Red-Team challenges drift scenario validity.
- Red-Team challenges metric bias and metric interpretation.
- Red-Team challenges evidence strength.
- Red-Team challenges conclusion scope.
- Red-Team looks for leakage, confounding, and over-claiming risks.
- Red-Team does not replace normal review.
- Red-Team does not implement fixes.
- Red-Team does not decide thesis claims.
- Red-Team should produce concise challenge notes with clear risks and
  questions for the human.

## 8. Required startup block

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

## 9. Required handoff block

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

## 10. What requires human approval

- Research question.
- Dataset inclusion.
- Train / validation / test split.
- Baseline model set.
- Drift scenario.
- Loss function.
- Metrics.
- Unreliable-gap threshold.
- Experiment approval to run.
- Result validity and reportability.
- Commit, push, and release decisions.
- Thesis claims.

## 11. What agents must not do

- Do not act as the Research Lead.
- Do not approve final thesis methods, framing, contributions, or claims.
- Do not treat draft project state as final approval.
- Do not change approved research definitions for convenience.
- Do not add raw datasets.
- Do not add processed tensors.
- Do not add checkpoints.
- Do not add credentials, tokens, secrets, private configs, private server
  paths, internal repository links, or private testbed details.
- Do not add large generated artifacts unless the human has explicitly approved
  a public-safe exception.
- Do not add unapproved thesis claims.
- Do not publish, release, commit, or push without explicit human approval.
- Do not introduce heavyweight process machinery or new tool systems that are
  outside the approved task.
- Do not let one session independently build and review the same artifact unless
  the human explicitly accepts loss of independence.

## 12. Maintenance rule

- Keep this guide short, operational, and subordinate to `THESIS_HARNESS.md`.
- Update it only when role guidance needs a small public-safe clarification.
- Do not copy long policy text from `THESIS_HARNESS.md`.
- Do not move authority, stop conditions, or research decisions out of
  `THESIS_HARNESS.md`.
- When role rules change, update `THESIS_HARNESS.md` first through the proper
  human-approved path, then update this guide only if needed.
