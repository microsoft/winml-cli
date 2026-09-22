# Use WinML CLI with AI Agents

This section documents two complementary agent workflows: using the CLI to
build and run models, and contributing support for models the CLI does not yet
handle.

## Choose a skill

| Goal | Skill | Guide |
|---|---|---|
| Build, optimize, compile, benchmark, or troubleshoot a supported model | `use-winml-cli` | [Use WinML CLI with an AI agent](use-winml-cli.md) |
| Add or repair model support and prepare a validated contribution | `adding-model-support` | [Add model support with an AI agent](adding-model-support.md) |

Use `use-winml-cli` for normal model workflows, including targeting available
CPU, GPU, and NPU execution providers.

Use `adding-model-support` when a model needs a recipe, exporter, resolver,
task mapping, dataset adapter, evaluator, or shared CLI fix. It handles one
model per run and drives the contribution through independent validation and
review.

## How to use a skill

Make the selected directory under
[`skills/`](https://github.com/microsoft/winml-cli/tree/main/skills) available
to an agent runtime that supports skills or custom instructions, then describe
your goal in natural language. Each guide provides example prompts and explains
what the skill handles automatically.
