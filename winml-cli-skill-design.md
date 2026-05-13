# WinML CLI Skill — Design One-Pager

**Date:** 2026-05-13
**Status:** Draft

---

## Overview

A skill is a markdown document that teaches a coding-assistant agent how you want things done. The WinML CLI skill teaches the agent to use the `winml` CLI to build optimized models for Windows — following a consistent, repeatable pattern without requiring the user to know the CLI flags.

The skill is agent-agnostic: it ships through FoundryTK to GitHub Copilot and is also usable in Claude Code, Cursor, or any agent that consumes the standard skill format.

---

## How It Works

FoundryTK installs the skill and exposes its name + description to the agent. The agent sees this metadata at all times; when a user task looks relevant, it loads the full skill body and follows its guidance to drive the `winml` CLI on the user's behalf.

The skill teaches the **shape** of the work (what stages exist, which hardware needs which execution provider, what's in and out of scope) and points the agent at the CLI's own `--help` for current flag spelling. This loose coupling is deliberate: the CLI evolves, the skill shouldn't break with every release.

**Example.** A user on a Snapdragon X Elite laptop asks the agent, "how fast can `microsoft/resnet-50` run on my NPU?" The agent loads the skill, identifies the right hardware path for Snapdragon, drives the `winml` CLI through the build and benchmark steps for the user, and reports the resulting latency. The user never types a command or looks up a flag.

---

## Out of Scope

- Fixing errors when WinML doesn't support a model.
- Editing or customizing the config file for winml build.

---

## Skill Evaluation

A skill is only useful if the agent applies it correctly. We need a measurable, repeatable loop — not monthly eyeballing.

### TL;DR

We evaluate the skill along three dimensions:

- **Trigger correctness** — does the agent load the skill at the right time, and stay away when it shouldn't apply?
- **Response correctness** — once loaded, does the agent give sound advice and use the right commands?
- **Prescription correctness** — when the agent actually runs those commands on the target hardware, do they deliver?

### What we evaluate

Dimensions 1 and 2 are fast (seconds per response) and run every iteration. Dimension 3 is slower (minutes per case) and runs before release and after structural changes.

For each dimension below: what failure it catches, how we measure it, and what we change in the skill when it fails.

#### 1. Trigger correctness

**What.** The agent decides on its own — from the user's prompt — whether to load the skill. Two failures: it doesn't load when it should (under-trigger), or it loads for an adjacent question it shouldn't apply to like "how do I call Windows ML from C#?" (over-trigger).

**How we measure.** A curated set of ~20 prompts, half should-trigger and half should-not-trigger. Score whether the agent decided to load the skill on each one. The negative cases are where the work is — sharp near-misses are what stress-test the skill's boundary.

**What we change to fix.** The short **description** of the skill — the one-line summary the agent sees when deciding whether to load. The body of the skill has no effect on this dimension; the agent doesn't see it yet at this stage.

#### 2. Response correctness

**What.** Once loaded, does the agent give sound advice? Two failures: wrong reasoning (recommends the wrong hardware path, skips a prerequisite step, tries to push an out-of-scope model through) or wrong commands (hallucinated flags, malformed CLI invocations).

**How we measure.** Two checks on every response:
- A per-case checklist of objectively verifiable statements (e.g., "identifies the right execution provider for Snapdragon", "refuses out-of-scope models").
- A syntax check: every CLI command the agent quoted must use real flags listed in the CLI's own `--help`. No invented flags, no wrong command shapes.

**What we change to fix.** The **body** of the skill — the workflow guidance, scope rules, error-recovery heuristics.

#### 3. Prescription correctness

**What.** A response can pass the first two checks and still be wrong in execution: commands in the wrong order, one step's output filename doesn't match the next step's input, the agent can't recover when something fails. Only catchable by actually running the task end-to-end as a real agent would.

**How we measure.** Spawn the agent in **agent mode** — with shell-tool access on a machine that has the target hardware — and give it the user's prompt. The agent runs the `winml` commands itself, reads the outputs, recovers from errors as needed. After it finishes, grade outcomes: did the expected artifacts get produced? Did the agent report a real result to the user? Did it stay efficient (no thrashing or infinite loops)? We don't grade stdout content — that's CLI behavior, not skill behavior. Steps requiring hardware the host doesn't have are skipped with a clear reason rather than silently substituted.

**What we change to fix.** Usually the skill body — but sometimes the failure points at a real CLI bug or an agent-runtime quirk, in which case we push back to the project rather than work around it in the skill.

### Out of scope for skill eval

To set expectations clearly: we are deliberately **not** signing up to test the following.

- **The `winml` CLI's own behavior.** The project's existing CI tests whether `winml` commands behave correctly. Our eval uses the CLI but doesn't validate it. If a command produces the wrong output, that's a CLI bug, not a skill bug.
- **What CLI commands print to the console.** We check whether each step completes (via exit code or the agent's reaction). We don't grade the human-readable output text — that's the CLI's job to format.
- **Model accuracy after the pipeline.** Whether a quantized model gives correct predictions is the model's own quality problem, separate from whether the skill drove the build pipeline correctly.

### Known limitation

Our eval drives the agent through a single runtime (Claude Code in the current setup). Behavior on the actual product target — Copilot via FoundryTK — and other runtimes (Cursor, Codex, etc.) is **not** validated by the current loop. The skill content is identical across runtimes, but runtime-specific differences in how agents interpret skills, sequence tool calls, or recover from errors would only surface in post-ship telemetry, not in pre-ship eval. Cross-runtime validation is a known gap to address before broad release.

---

## Telemetry & Feedback

**Telemetry to collect:** trigger rate, command success/failure rate, user correction rate (proxy for skill quality).

**Feedback loop:** Review failures monthly. Update skill guidance when new WinML CLI versions change command structure.