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
search, use [`use-winml-cli`](../use-winml-cli/SKILL.md). For an unsupported model,
missing exporter, or model-support recipe contribution, use
[`adding-model-support`](../adding-model-support/SKILL.md).

## Quick start

Make this directory available to your agent runtime as a skill. Provide the
model, target EP/device, optimization goal, working directory, and WinML CLI
checkout (`WINML_CLI_REPO`). The agent asks for missing values.

```text
Optimize ./model.onnx for QNN on the NPU. Aim to reduce inference latency
without changing the model's I/O or correctness. Use ./optimization-run for
artifacts and the checkout at <path-to-winml-cli> as WINML_CLI_REPO.
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
  bundle, and records a separate promotion handoff.

On resume, the skill starts at the first unverified gate. Changes to the model,
inputs, options, or toolchain invalidate affected evidence. A validation-only
request does not start a new optimization loop.

## What you get

A successfully replayed and validated result includes:

- `champion.onnx` and any companion files.
- `winml_config.json` for the built champion and a separate
  `rebuild_config.json` for reproducing the build.
- `report.json` and `report.html` with the experiment evidence.
- A hash-bound `manifest.json`.
- `repro.ps1` (including `-ValidateOnly`), `repro-run.ps1`, and `repro.lock.json`.
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

[`SKILL.md`](./SKILL.md) is the entry point. Keep operational rules in their
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

See [resume behavior](./references/resume.md),
[PR routing](./references/pr-routing.md), and the
[check-in review gates](./roles/checkin-reviewer.md) for the detailed contracts.
Model identities, source paths, tensors, and generated artifacts stay run-local.
Only reviewed, reusable, model-agnostic findings enter bundled knowledge.

When changing the skill, run the relevant Pytest tests. For workflow behavior
changes, also use the [behavioral evaluation guide](./evals/README.md). Those
evaluations simulate hardware and GitHub actions; they do not measure actual
model performance or certify reproduction on a target device.
