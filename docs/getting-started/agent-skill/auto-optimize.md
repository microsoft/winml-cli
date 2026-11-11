# Auto Optimize

`auto-optimize` is an agent skill for reducing ONNX inference latency with
[`microsoft/winml-cli`](https://github.com/microsoft/winml-cli) on a specified
execution provider (EP) and device. It profiles the baseline, tests graph
optimizations, checks correctness before performance, and publishes a
reproducible output bundle.

When a validated optimization needs a generic CLI capability, the skill can
implement that capability and prepare an independently reviewed draft PR.
It does not guarantee a speedup or automatically merge changes.

## When to use this skill

Use it when you want to:

- Reduce latency for an ONNX model on a specific EP/device, including QNN NPU.
- Investigate operator hotspots, graph interactions, layout, transfers,
  partitioning, or fallback.
- Validate an existing optimization candidate or resume an interrupted search.
- Turn a proven optimization into reusable CLI behavior.

For routine builds, optimization commands, or benchmarks without an experimental
search, use [`use-winml-cli`](use-winml-cli.md). For an unsupported model,
missing exporter, or model-support recipe contribution, use
[`adding-model-support`](adding-model-support.md).

## Quick start

Make the [skills/auto-optimize directory](https://github.com/microsoft/winml-cli/tree/main/skills/auto-optimize) available to your agent runtime as a skill. Provide the
model, target EP/device, optimization goal, working directory, and WinML CLI
checkout (`WINML_CLI_REPO`). The agent asks for missing values.

```text
Optimize ./model.onnx for QNN on the NPU. Aim to reduce inference latency
without changing the model's I/O or correctness. Use ./optimization-run for
artifacts and the checkout at <path-to-winml-cli> as WINML_CLI_REPO.
```

## Sample prompt template

The following prompt keeps the request concise while making the result
reproducible and useful to another developer. Replace the angle-bracketed
values; omit only fields that genuinely do not apply.

Report structure, delivery artifacts, and final composed-config validation are
standard behavior owned by the skill. Users do not need to repeat those
requirements in each prompt; state only the optimization task and its inputs.

```text
Use the auto-optimize skill to optimize <MODEL_OR_ONNX> for <EP> on <DEVICE>.

Goal: minimize median batch-1 latency while preserving <QUALITY_METRIC> >=
<QUALITY_THRESHOLD> against <REFERENCE_MODEL> on <REFERENCE_DEVICE_AND_EP>.
Use the fixed real inputs in <INPUT_DATA>; use <SHAPE_OR_SEQUENCE_LENGTH> and
the same inputs, provider options, and timing protocol for every comparison.
Write all run evidence to <RUN_DIRECTORY>. Use <WINML_CLI_CHECKOUT> as
WINML_CLI_REPO. Do not modify generic source or create a PR unless a confirmed
leader requires a reusable CLI capability.
```

For example, a BGE embedding-model request can be:

```text
Use the auto-optimize skill to optimize ./models/fp32/model.onnx for QNN on the
NPU. Minimize median batch-1 latency while keeping mean output cosine similarity
>= 0.99 against the same FP32 model on CPU. Use sequence length 128 and fixed,
meaningful text inputs saved under ./optimization-run/inputs. Write all evidence
to ./optimization-run and use <path-to-winml-cli> as WINML_CLI_REPO.
```

For a candidate you already have:

```text
Validate ./candidate.onnx against ./baseline.onnx using the inputs and
provider options in my existing run directory. Check correctness first;
do not start a new optimization search.
```

The host needs a working WinML CLI environment and the requested provider/device
for target measurements. Replay requires PowerShell 7.3 or newer. Draft PR
creation also needs GitHub access and the existing `model-opt-by-skill` label
in the target repository; the skill does not create that label automatically.

## What the skill handles automatically

- Inspects CLI capabilities and collects baseline analysis, performance, and
  operator traces, recording gaps in provider attribution.
- Freezes model/input hashes, provider options, toolchain versions, and cache
  identity so comparisons use compatible conditions.
- Selects bounded hotspot probes or a normal loop with at most three active
  hypotheses, using relevant reusable findings.
- Validates ONNX structure, shapes, I/O, and numerical correctness before
  measuring a candidate's performance.
- Screens candidates with alternating A/B and B/A runs and confirms leaders
  with paired evidence. Statistical ties are labeled as ties, not speedups.
- Reviews remaining optimization opportunities and retains failed experiments
  instead of discarding inconvenient results.
- When necessary, implements generic CLI behavior in an isolated current-main
  worktree, adds tests, and rebuilds through the public CLI.
- Replays the final result in a fresh directory, validates the published
  bundle, then independently rebuilds and executes the final composed config
  before recording a separate promotion handoff.

On resume, the skill starts at the first unverified gate. Changes to the model,
inputs, options, or toolchain invalidate affected evidence. A validation-only
request does not start a new optimization loop.

## What you get

A successfully replayed and validated result includes:

- `champion.onnx` and any companion files.
- `winml_config.json` for the built champion and a separate
  `rebuild_config.json` for reproducing the build.
- `report.json` and `report.html` with the experiment evidence.
- A concise report opening with matched benchmark results, accepted
  optimizations, and the delivered outcome before the detailed evidence.
- A hash-bound `manifest.json`.
- `repro.ps1` (including `-ValidateOnly`), `repro-run.ps1`, and `repro.lock.json`.
- `config_validation.json`, proving the exact final rebuild config completed a
  clean build, target runtime load, and fixed-input correctness comparison.
- `perf_input.npz`, `eval_inputs.npz`, and `inputs_manifest.json`.
- A standalone `promotion_handoff.json` created after bundle validation.

Optimizer contributions remain draft PRs labeled `model-opt-by-skill`. The
independent check-in reviewer returns `READY_FOR_CHECK_IN`, `CHANGES_REQUESTED`,
`PERF_NOT_PROVEN`, or `BLOCKED`; a readiness verdict does not merge the PR or
convert it to ready for review. Recipe contributions are routed separately to
`adding-model-support`.

The search stops when the target is confirmed, hypotheses are exhausted, the
budget is reached, or you ask it to stop. If a task evaluator is unavailable,
the result may be provisional-quality after tensor validation, with that gap
disclosed. Failed replay blocks publication. Artifacts that depend on unmerged
code are marked `requires-unmerged-pr`.

## Maintainer resources

[`SKILL.md`](https://github.com/microsoft/winml-cli/blob/main/skills/auto-optimize/SKILL.md) is the entry point. Keep operational rules in their
owning contracts rather than duplicating them in this README.

```text
auto-optimize/
├── SKILL.md       # workflow and gates
├── roles/         # scouting, engineering, performance, and review contracts
├── references/    # resume, capability closure, complexity, and PR routing
├── knowledge/     # model-agnostic, evidence-backed optimization findings
├── scripts/       # planning, reports, publication, promotion, and case storage
├── tests/         # helper and skill contract tests
└── evals/         # live-agent behavioral evaluations
```

See [resume behavior](https://github.com/microsoft/winml-cli/blob/main/skills/auto-optimize/references/resume.md),
[PR routing](https://github.com/microsoft/winml-cli/blob/main/skills/auto-optimize/references/pr-routing.md), and the
[check-in review gates](https://github.com/microsoft/winml-cli/blob/main/skills/auto-optimize/roles/checkin-reviewer.md) for the detailed contracts.
Model identities, source paths, tensors, and generated artifacts stay run-local.
Only reviewed, reusable, model-agnostic findings enter bundled knowledge.

When changing the skill, run the relevant Pytest tests. For workflow behavior
changes, also use the [behavioral evaluation guide](https://github.com/microsoft/winml-cli/blob/main/skills/auto-optimize/evals/README.md). Those
evaluations simulate hardware and GitHub actions; they do not measure actual
model performance or certify reproduction on a target device.
## Loading and verification

The repository `skills/` directory is a source distribution, not proof that a
host has registered a skill. Make the complete auto-optimize directory available
through the host's supported skill installation mechanism, or explicitly ask
the agent to read its absolute `SKILL.md` path. Keep references and scripts
together. Start a fresh session after changing installed skills and verify
that the agent read the intended file. Record its resolved path and SHA-256
in run-local evidence to distinguish stale copies from workflow failures.

## Entry and completion boundaries

New searches collect a baseline before planning. Supplied candidates and
validation-only requests resume at the first unverified gate without restarting
planning. Hash or option changes invalidate downstream evidence. Roles must be
read before use; unavailable independent review remains unverified.

Reaching the target or budget ends additional search, not automatically a
validated delivery. Retain partial evidence when replay or closure is incomplete.
A user stop request ends additional work immediately. Generic implementation
returns public-CLI evidence first; only the main workflow creates an eligible
optimizer Draft PR after bundle validation and handoff creation. No generic
source change means no optimizer PR requirement.
## Evidence-bound workflow entry

Use scripts/workflow.py as documented in references/workflow.md. `prepare`
imports raw timing metrics and analyzer tables; `status` identifies the first
unrecorded or invalidated gate; `record` binds evidence hashes and invalidates
downstream records; `deliver` invokes finalizer and promotion and writes a
receipt only after successful validation. These records are attestations, not
authentication of reviewers or proof that hardware commands ran. Receipt claims
keep artifact validation, recorded model/review/replay evidence, and visual
inspection separate. Use a new directory for each delivery attempt.
