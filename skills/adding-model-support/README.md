# Adding Model Support

`adding-model-support` is an agent skill for contributing support for one
unsupported or partially supported model to
[`microsoft/winml-cli`](https://github.com/microsoft/winml-cli).

The skill covers the full contribution lifecycle: it diagnoses the current
failure, selects the narrowest reusable fix, implements and validates the
change, opens a draft pull request, runs an independent review, and records
reusable findings for future model families.

> This is a contribution workflow, not a command for building an already
> supported model. For normal model builds, optimization, or benchmarking, use
> the [`use-winml-cli`](../use-winml-cli/SKILL.md) skill instead.

## When to use this skill

Use it when a Hugging Face model or architecture is not fully supported by the
WinML CLI, including cases where:

- `winml inspect`, `config`, `build`, `run`, `perf`, or `eval` fails for the
  model.
- The model needs a recipe, exporter, resolver, task mapping, dataset adapter,
  or evaluator.
- Existing support works only for part of the model or only at an insufficient
  validation level.
- A model-support contribution needs reproducible evidence and an independently
  reviewed draft pull request.

Do not use it to:

- Run, benchmark, or tune an already supported model.
- Optimize an existing working recipe.
- Add a new execution provider backend.
- Process multiple models as one contribution.

## Requirements

Run the skill from a current
[`microsoft/winml-cli`](https://github.com/microsoft/winml-cli) checkout with:

- Python 3.11 and [`uv`](https://docs.astral.sh/uv/).
- [GitHub CLI](https://cli.github.com/) installed and authenticated.
- Permission to push a branch to the target remote.
- An agent runtime that supports fresh subagent delegation.

The workflow deliberately assigns planning, implementation, testing, reporting,
and review to separate agents. If the runtime cannot create fresh subagents,
the skill stops as `BLOCKED` rather than presenting self-review as independent
validation.

## Quick start

Make this directory available to your agent runtime as a skill, then describe
one model-support goal in natural language. Include the model ID and any known
failure or target hardware when available.

For example:

```text
Add winml support for <organization>/<model>. The current build fails during
export. Target CPU first, then validate the execution providers available on
this machine.
```

The skill entry point is [`SKILL.md`](./SKILL.md). The agent begins by loading
the [orchestrator contract](./agents/orchestrator.md), checking the repository
and GitHub prerequisites, and reproducing the problem on the current `main`
branch before proposing a change.

## What the workflow does

Each run handles exactly one model and delegates seven roles:

| Stage | Responsibility |
|---|---|
| Orchestrator | Verifies prerequisites, dispatches fresh agents, drives repair loops, and cleans up run-owned state |
| Planner | Reproduces the current-main baseline and freezes the contribution scope and success criteria |
| Producer | Implements the narrowest reusable code, configuration, or recipe change |
| Tester | Runs the repair loop and records exact validation evidence for each required target |
| Learner | Captures model-family findings and workflow lessons with their evidence and limits |
| Explainer | Creates or updates the draft pull request and presents the tested evidence |
| Reviewer | Independently verifies the final pushed commit and reports a verdict |

```text
planner -> producer -> tester -> learner -> explainer -> reviewer
   ^           ^                                      |
   |           +--------- artifact fixes -------------+
   +---------------- scope corrections ---------------+
```

The orchestrator continues these loops until the run reaches one terminal
state:

- `APPROVE` — independent review accepts the exact final commit; the pull
  request remains a draft.
- `REJECT` — the contribution has a structural or evidence-integrity failure.
- `BLOCKED` — an external dependency, environment limitation, or user decision
  prevents completion.

`REQUEST_CHANGES` is not terminal. It routes the contribution back to the role
that owns the problem.

## Expected output

A successful run produces:

- A focused model-support change based on current-main evidence.
- Pytest coverage for source changes and regressions.
- A checked-in recipe when the validated contribution requires one.
- Per-target build, analysis, runtime, and performance evidence.
- One bounded FP32 CPU functional smoke evaluation.
- A pushed branch and draft pull request labeled `model-scale-by-skill`.
- An independent verdict for the exact final commit.
- Reusable, scoped findings in the
  [model knowledge base](./model_knowledge/README.md) or
  [skill meta-findings](./skill_meta/README.md).

The skill does not claim broad model quality from a smoke test, infer support
for one execution-provider tuple from another, or convert blocked evidence into
a successful result.

## Optional integrations

- **`model-breakdown`** can provide a structured model profile when a compatible
  version is installed. The planner falls back to pinned source, configuration,
  and model-card evidence when it is unavailable.
- **[`auto-optimize`](../auto-optimize/SKILL.md)** is required only for a
  `--promotion-handoff <absolute path>` run. Normal model-support contributions
  do not depend on it.

See [`SKILL.md`](./SKILL.md) for the integration contracts and non-negotiable
workflow boundaries.

## Skill development

The root files are organized by responsibility:

```text
adding-model-support/
├── SKILL.md                 # trigger and workflow dispatcher
├── agents/                  # role contracts
├── model_knowledge/         # evidence-scoped model-family findings
├── skill_meta/              # findings about the workflow itself
├── evals/                   # trigger and response evaluations
├── scripts/                 # bounded workflow utilities
└── tests/                   # skill contract tests
```

When changing the skill:

1. Keep methodology changes separate from model-support changes.
2. Update the owning role contract instead of duplicating operational rules in
   this README.
3. Run the relevant Pytest contract tests.
4. Run the trigger or response evaluations when changing the skill description
   or behavior. See the [evaluation guide](./evals/README.md).
