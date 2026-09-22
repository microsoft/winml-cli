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
> the [`use-winml-cli`](use-winml-cli.md) skill instead.

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

## Quick start

Make the
[`skills/adding-model-support`](https://github.com/microsoft/winml-cli/tree/main/skills/adding-model-support)
directory available to your agent runtime as a skill, then ask it to add
support for one model. The model ID is enough to start; a known failure, target
device, or execution provider is useful but optional.

For example:

```text
Add winml support for <organization>/<model>.
```

You can include more context when you have it:

```text
Add winml support for <organization>/<model>. Export currently fails, and I
need it validated on the execution providers available on this machine.
```

The skill handles one model per run.

## What the skill handles automatically

The skill drives the contribution from diagnosis through review:

- Checks the checkout, Python tooling, GitHub authentication, push access, and
  agent delegation support before starting expensive work.
- Reproduces the problem against the current `main` branch and determines
  whether the gap is in a recipe, exporter, resolver, task, dataset adapter,
  evaluator, or shared infrastructure.
- Chooses the narrowest reusable fix and avoids model-name-specific logic.
- Implements the change and adds relevant Pytest coverage.
- Validates the required build, analysis, runtime, performance, and functional
  smoke evidence without inferring one target's result from another.
- Uses separate agents for planning, implementation, testing, reporting, and
  final review, then handles review feedback until the run reaches a final
  result.
- Pushes the contribution and creates a draft pull request. If the GitHub API
  cannot create the pull request, it provides the branch link and prepared body
  for the user to submit in the browser.
- Records reusable findings with their evidence and scope.

If a required capability is unavailable, the skill stops early as `BLOCKED`
and explains the action needed. Users do not need to run the preflight checks
themselves.

## What you get

A successful run produces:

- A focused model-support change based on current-main evidence.
- Pytest coverage for source changes and regressions.
- A checked-in recipe when the validated contribution requires one.
- Per-target build, analysis, runtime, and performance evidence.
- One bounded FP32 CPU functional smoke evaluation.
- A pushed branch and draft pull request labeled `model-scale-by-skill`.
- An independent verdict for the exact final commit.
- Reusable, scoped findings in the
  [model knowledge base](https://github.com/microsoft/winml-cli/tree/main/skills/adding-model-support/model_knowledge)
  or
  [skill meta-findings](https://github.com/microsoft/winml-cli/tree/main/skills/adding-model-support/skill_meta).

The final result is `APPROVE`, `REJECT`, or `BLOCKED`, with the supporting
evidence and next action. A smoke evaluation proves bounded end-to-end
operability; it is not presented as representative model accuracy.

## Maintainer resources

The entry point is
[`SKILL.md`](https://github.com/microsoft/winml-cli/blob/main/skills/adding-model-support/SKILL.md).
It dispatches the internal role contracts and defines the workflow boundaries.
Users do not need to invoke the roles directly.

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

The skill can use `model-breakdown` when it is available and falls back to
pinned source and model metadata when it is not. The
[`auto-optimize`](https://github.com/microsoft/winml-cli/blob/main/skills/auto-optimize/SKILL.md)
integration applies only to explicit promotion handoffs; normal model-support
runs do not depend on it.

When changing the skill:

1. Keep methodology changes separate from model-support changes.
2. Update the owning role contract instead of duplicating operational rules in
   this README.
3. Run the relevant Pytest contract tests.
4. Run the trigger or response evaluations when changing the skill description
   or behavior. See the
   [evaluation guide](https://github.com/microsoft/winml-cli/blob/main/skills/adding-model-support/evals/README.md).
