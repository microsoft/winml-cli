# E2E Eval — Runbook

How to run the Pillar 3 E2E eval on any machine.

## What this measures

Pillar 3 from the design doc — **prescription correctness**. We spawn a real agent against each test case, let it execute `winml` commands on the actual hardware, then grade the outcome artifacts.

This is the only pillar that requires the target hardware. Pillars 1 and 2 run anywhere.

## Prerequisites

The only truly manual step is **cloning the repo**. Everything else is handled by the slash command:

- `uv` itself — slash command detects if missing and runs the official installer for your OS.
- `winml` Python package — slash command runs `uv run winml --version` which triggers uv to sync `pyproject.toml` and build winml from source on first invocation.
- Hardware detection — slash command runs `uv run winml sys --list-ep` and filters `cases.json` accordingly.

Setup on a fresh machine reduces to: clone the repo, open it in Claude Code, type `/run-e2e`.

**Hardware decides what cases are eligible.** See `cases.json` — each case has a `required_ep` field:
- `null` → runs on any machine
- `"QNNExecutionProvider"` → requires Snapdragon X Elite (Qualcomm NPU)
- `"VitisAIExecutionProvider"` → requires AMD Ryzen AI

Cases for unavailable EPs are skipped with a reason — not silently substituted.

**Claude Code (or equivalent agent runtime)** must be running in this workspace so the orchestration step can spawn subagents.

## How to run

### Option A — slash command (recommended)

In Claude Code, type:

```
/run-e2e
```

This loads `.claude/commands/run-e2e.md` and the agent executes the full pipeline:
1. Verifies `winml --version` works, then detects registered EPs via `winml sys --list-ep`
2. Filters cases.json down to eligible ones (skips the rest with reason)
3. Spawns one subagent per case in parallel
4. Captures each agent's final message + telemetry
5. Runs `grade_case.py` per case
6. Writes a per-run report to `e2e_eval/runs/<datetime>_<hardware>/report.md` — one directory per machine + skill version, not numbered iterations.

### Option B — manual (if no slash command or for debugging)

1. Check available EPs:
   ```bash
   uv run winml sys --list-ep
   ```

2. For each case in `cases.json` you want to run, ask Claude in the workspace:

   > Spawn a subagent that completes this task: [paste case.prompt]. The agent has shell tool access. Save its final message to `e2e_eval/scratch/<case_id>/agent_summary.md`. Save `tool_uses` and `duration_ms` from the completion notification.

3. Grade each case:
   ```bash
   python e2e_eval/grade_case.py <case_id> e2e_eval/scratch/<case_id>/agent_summary.md <tool_uses> <duration_ms>
   ```

4. Aggregate results manually or by reading each `e2e_eval/scratch/<case_id>/grading.json`.

## Interpreting results

- **PASS** — all outcome assertions on that case were met (artifact produced, agent's final message matches expected pattern, tool calls within budget).
- **FAIL** — at least one assertion failed. Read `grading.json` for details.
- **SKIP** — case requires hardware not on this machine. **Don't treat SKIP as PASS** — it just means "untested here, run on the right hardware before shipping".

Pass@K is built in. `cases.json`'s `default_runs` field (typically 3) controls how many independent trials each eligible case gets per `/run-e2e` invocation. The report shows Pass@K per case: how many trials out of K had every assertion pass. Per-case override via `runs` field.

A case with 2/3 trials passing is a real signal — agent-mode behavior is non-deterministic, and a flaky case is worth investigating before shipping. Don't average across cases into one composite score; the per-case Pass@K is what matters.

## Naming convention: why no "iteration-N"

Pillar 3 is a **gate**, not a trajectory. Each run is evidence — "skill version X, on hardware Y, on date Z, passed/failed". We don't number them in sequence because they're not comparable across machines (a run on CPU-only is fundamentally a different gate than a run on Snapdragon). Per-run directories live under `runs/<datetime>_<hardware>/` and stay side-by-side rather than being rolled into one trajectory.

If you want the trajectory view (Pillar 2 pass rate over time), look at the workspace-root `reports/<UTC-datetime>.md` cross-pillar snapshots instead. Those pull in the latest E2E gate evidence at the time of writing.

## Adding new cases

Edit `cases.json`. Each case needs:

| Field | What it does |
|---|---|
| `id` | Unique identifier, used for scratch dir name |
| `hardware` | Free-form label (`any`, `snapdragon`, `ryzen-ai`) for human readers |
| `required_ep` | Canonical EP name (e.g. `QNNExecutionProvider`). `null` = no hardware constraint |
| `prompt` | The user message the agent will be given verbatim |
| `expected_outcomes` | Outcome assertions — see `grade_case.py` for what each field means |

After adding cases, the slash command picks them up automatically on the next run.

## What this does NOT test

- The `winml` CLI's own behavior — that's the project's CI.
- Model accuracy after the pipeline.
- Stdout/stderr content of CLI commands beyond the agent's interpretation.

See the design doc's "Out of scope for skill eval" section for the full list.
