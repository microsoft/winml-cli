# Use with AI Agent

WinML CLI provides skills for running supported models, contributing missing
model support, and investigating ONNX latency. Choose the workflow that matches
your goal.

## Choose a skill

| Your goal | Skill | Result |
|---|---|---|
| Build, optimize, quantize, compile, or benchmark a supported model | [`use-winml-cli`](https://github.com/microsoft/winml-cli/tree/main/skills/use-winml-cli) | Model artifacts and measurements through existing CLI features |
| Add or repair a recipe, exporter, resolver, task, dataset adapter, or evaluator | [`adding-model-support`](https://github.com/microsoft/winml-cli/tree/main/skills/adding-model-support) | A validated model-support contribution and independently reviewed draft PR |
| Investigate latency or validate optimization candidates on an EP/device | [`auto-optimize`](auto-optimize.md) | Correctness and paired performance evidence, a replayable bundle, and an optimizer draft PR when needed |

Routine optimization with existing CLI features belongs to `use-winml-cli`.
Choose `auto-optimize` for an experimental search or candidate validation.

## Add model support

Use `adding-model-support` for one unsupported or partially supported model.
A model ID is enough to start; include a known failure or target when available.

```text
Add winml support for <organization>/<model>. Export currently fails;
validate the fix on the execution providers available on this machine.
```

The workflow diagnoses current-main behavior, implements a reusable fix,
validates it, and prepares a draft PR with independent review. It requires a
WinML CLI checkout, Python 3.11 with `uv`, authenticated GitHub CLI with push
access, and an agent runtime that can delegate fresh subagents. Missing required
capabilities stop the workflow as `BLOCKED`.

A successful contribution includes relevant Pytest coverage, per-target
validation evidence, and a bounded FP32 CPU functional smoke evaluation. That
smoke evaluation does not certify representative accuracy. The PR remains draft
even when the workflow's review verdict is `APPROVE`.

## Optimize ONNX latency

Use `auto-optimize` to profile bottlenecks and test candidates on a specified
EP/device. Provide the model, target, goal, working directory, and WinML CLI
checkout (`WINML_CLI_REPO`); the agent asks for missing values.

```text
Optimize ./model.onnx for QNN on the NPU without changing I/O or correctness.
Use ./optimization-run for artifacts and <path-to-winml-cli> as WINML_CLI_REPO.
```

Correctness is checked before performance. Paired measurements distinguish
confirmed gains from noise, and the final bundle is replayed before publication.
Existing candidates can be validated without starting another search. See
[Auto Optimize with AI](auto-optimize.md) for prerequisites, outputs, and resume
behavior.

## Make a skill available

Use your agent runtime's skill-loading mechanism with the selected directory
under [`skills/`](https://github.com/microsoft/winml-cli/tree/main/skills).
The entry point is `SKILL.md`. Keep the complete folder available: contribution
and optimization workflows also load roles, references, knowledge, and scripts.
Text context alone does not provide command execution, hardware access, or
independent agent delegation. Target measurements require the actual requested
provider/device; a cloud agent cannot infer them from a different host.

For CLI setup, see [Installation](installation.md). The following sections
describe the general-purpose `use-winml-cli` skill.

## Run a supported model

winml-cli ships a **Copilot Skill** (`use-winml-cli`) that lets AI coding agents
drive the entire model-building pipeline on your behalf. When a coding agent has
this skill attached, it can inspect models, generate configs, run builds, and
interpret results — without you having to remember exact flags or stage ordering.

---

### What the skill provides

The skill teaches the agent:

| Capability | What the agent learns |
|---|---|
| **Pipeline shape** | The stage order (`inspect → export → analyze → optimize → quantize → compile → perf`) and when to enter mid-pipeline |
| **Flag discovery** | Always run `winml <command> --help` before quoting a command — never fabricate flags |
| **Output mapping** | Which command's `-o` produces the artifact the user actually needs |
| **Scope awareness** | Which model architectures are supported (classic DL) vs. out-of-scope (LLMs, diffusion) |
| **Hardware detection** | Use `winml sys --list-ep` to confirm what's available before targeting an EP |
| **Two paths** | When to use primitives (debugging, exploring) vs. config + build (production, CI) |

---

### How to use it

#### With GitHub Copilot Coding Agent

To make the [Copilot Coding Agent](https://docs.github.com/en/copilot/how-tos/copilot-on-github/use-copilot-agents/overview)
(the cloud agent that creates PRs) follow the skill's guidance, reference it in
`.github/copilot-instructions.md`. The Coding Agent reads that file automatically
when working on this repository.

#### With other AI agents

For agents that support custom instructions (e.g., Copilot Extensions, Claude,
ChatGPT with file uploads, or custom MCP tool servers), attach the skill file
as context:

```
skills/use-winml-cli/SKILL.md
```

You can copy the file contents into your agent's system prompt, upload it as a
reference document, or include it in a `.github/copilot-instructions.md` for
VS Code Copilot Chat. The skill uses standard markdown with YAML front-matter —
any agent that accepts text context can benefit from it.

---

### Skill location

```
winml-cli/
└── skills/
    └── use-winml-cli/
        └── SKILL.md          ← the skill definition
```

---

### Example agent interaction

```
User: Can I run ConvNeXt on my Snapdragon X Elite NPU?

Agent (with skill):
1. Runs `winml sys --list-ep` → confirms QNNExecutionProvider is registered
2. Runs `winml inspect -m microsoft/convnext-tiny-224` → confirms supported
3. Runs `winml config --onnx ... -d npu -o config.json`
4. Runs `winml build -c config.json -m microsoft/convnext-tiny-224 -o output/`
5. Runs `winml perf -m output/model.onnx -d npu --monitor`
6. Reports latency + NPU utilization to user
```
