# Auto Optimize with an AI Agent

The `auto-optimize` skill helps an AI agent investigate and reduce ONNX inference
latency on a specified execution provider (EP) and device. It profiles the
baseline, tests optimization candidates, checks correctness before performance,
and publishes a reproducible output bundle. It does not guarantee a speedup.

For routine CLI builds and benchmarks, see [Use with AI Agent](agent-skill.md).
Use `auto-optimize` when you need an experimental optimization search, hotspot
investigation, or validation of an existing candidate.

## Get started

Make the repository's
[`skills/auto-optimize/` directory](https://github.com/microsoft/winml-cli/tree/main/skills/auto-optimize)
available to your agent runtime as a skill, including its roles, references,
knowledge, and scripts. Provide the model, target EP/device, optimization goal,
working directory, and WinML CLI checkout (`WINML_CLI_REPO`).

```text
Optimize ./model.onnx for QNN on the NPU. Reduce inference latency without
changing I/O or correctness. Use ./optimization-run for artifacts and
<path-to-winml-cli> as WINML_CLI_REPO.
```

The host needs a working WinML CLI environment and the target provider/device
for measurements. Replay requires PowerShell 7.3 or newer. Optimizer draft PRs
also require GitHub access and the existing `model-opt-by-skill` repository label.

## Workflow and results

The agent collects baseline analysis and traces, fixes input and provider
settings for comparable experiments, and validates each candidate before
paired performance measurements. Statistical ties are reported as ties.
Independent roles review coverage and disputed performance evidence.

A validated result includes the champion ONNX model, build/rebuild configs,
JSON and HTML reports, a hash-bound manifest, input data, and PowerShell replay
assets. Failed replay blocks publication. Missing task-level quality evaluation
is disclosed; tensor validation alone is not a claim of representative accuracy.

When a generic CLI capability is needed, the workflow can create a tested,
independently reviewed optimizer draft PR. It does not automatically merge or
mark the PR ready. Recipe contributions are routed to `adding-model-support`.

## Resume or validate a candidate

```text
Validate ./candidate.onnx against ./baseline.onnx using the inputs and
provider options in my existing run directory. Check correctness first;
do not start a new optimization search.
```

The skill resumes at the first unverified gate. Changed model, inputs, options,
or toolchain invalidate affected evidence. A validation-only request does not
authorize a new optimization search.

See the [Auto Optimize README](https://github.com/microsoft/winml-cli/blob/main/skills/auto-optimize/README.md)
for the full output list, stopping conditions, and maintainer resources.
