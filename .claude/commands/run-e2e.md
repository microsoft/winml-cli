---
description: Run the winml-cli skill's E2E eval suite on this machine
---

You are running the Pillar 3 E2E eval for the `winml-cli` skill. Follow these steps exactly.

## Workspace

- Cases:        `C:/repo/WinML-ModelKit/.claude/skills/winml-cli-workspace/e2e_eval/cases.json`
- Grader:       `C:/repo/WinML-ModelKit/.claude/skills/winml-cli-workspace/e2e_eval/grade_case.py`
- Scratch root: `C:/repo/WinML-ModelKit/.claude/skills/winml-cli-workspace/e2e_eval/scratch/`
- Skill being tested: `C:/repo/WinML-ModelKit/.claude/skills/winml-cli/SKILL.md`

## Procedure

1. **Ensure uv is installed.** Run `uv --version`. If it errors with "command not found":
   - On Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - On macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Then re-source the shell or use the install path uv reports. Verify `uv --version` works before continuing. If the install itself fails (corporate-locked machine, network blocked), surface the error and stop.

2. **Sync the environment, then discover hardware.** From `C:/repo/WinML-ModelKit`, run `uv run winml --version`. On a fresh clone this triggers uv to create a venv, resolve `pyproject.toml`, and build winml from source — first invocation can take a minute or two; subsequent ones are instant. If `--version` errors out (not the uv-not-found case, which step 1 covered), surface it; the project's environment is broken in a way beyond this skill to fix.

   On success, run `uv run winml sys --list-ep` and parse out the registered ExecutionProvider names (e.g. `QNNExecutionProvider`, `OpenVINOExecutionProvider`).

3. **Read cases.** Load `cases.json`. For each case, check the `required_ep` field:
   - If `required_ep` is `null` → eligible to run
   - If `required_ep` is in the registered EP set → eligible
   - Otherwise → mark as SKIP with reason "EP not registered on this machine"

4. **Determine K (trials per case).** Read `cases.json`'s `default_runs` field (typically 3). For each case, K = `case.runs` if specified, otherwise `default_runs`. Pass@K is the goal: how many of K trials fully pass.

5. **Prepare scratch dirs.** For each eligible case, ensure `scratch/<case_id>/` exists, then create K empty `scratch/<case_id>/trial-<N>/` subdirs (N = 1..K). Clean any prior trial-* contents first.

6. **Spawn agents in parallel.** For each eligible case, spawn K subagents (one per trial). Each subagent uses its own `trial-<N>/` directory as scratch. Prompt structure per trial:

   ```
   You are Claude in an interactive coding-assistant session. A user just said:

   > "<case.prompt>"

   Treat this as a real session — actually run commands, read output, complete the task.

   **Environment**
   - `winml` is at `C:/repo/WinML-ModelKit`, invoke via `uv run winml ...`. Run commands with `cwd=C:/repo/WinML-ModelKit`.
   - Scratch dir for this trial: `<scratch>/<case_id>/trial-<N>/`. Save artifacts there using absolute paths.
   - Registered EPs on this machine: <list from step 1>

   **Skill**
   Read first: `C:/repo/WinML-ModelKit/.claude/skills/winml-cli/SKILL.md`. Follow its guidance.

   **Done when**
   <case-specific done criteria — extract from expected_outcomes>

   Be efficient: aim for ≤ <case.expected_outcomes.max_tool_uses> tool calls.
   ```

   Total agents in flight = sum(K_per_case) across eligible cases. You can run them all in parallel; trial-N subdirs ensure no file collisions. If you hit runtime limits, fall back to serial-per-case (still parallel-across-cases) — but for ≤ 20 total agents, full parallel is fine.

7. **Capture agent output per trial.** As each subagent completes, the task notification contains:
   - Final text reply → save to `scratch/<case_id>/trial-<N>/agent_summary.md`
   - `total_tokens`, `tool_uses`, `duration_ms` → save to `scratch/<case_id>/trial-<N>/telemetry.json` as `{"total_tokens": ..., "tool_uses": ..., "duration_ms": ...}`

8. **Grade each trial.** Per completed trial:
   ```bash
   python <workspace>/e2e_eval/grade_case.py <case_id> <scratch>/<case_id>/trial-<N> <tool_uses> <duration_ms>
   ```
   Writes `scratch/<case_id>/trial-<N>/grading.json`.

8. **Generate the run id.** Build the run-id string:
   - `<UTC-datetime>` = `YYYYMMDD-HHMMSS` from `datetime.now(timezone.utc)`
   - `<hardware-tag>` = derived from registered EPs (step 1):
     - `cpu-only` if only `CPUExecutionProvider`, `DmlExecutionProvider`, and/or `OpenVINOExecutionProvider` are registered (no NPU)
     - `snapdragon` if `QNNExecutionProvider` is registered
     - `ryzen-ai` if `VitisAIExecutionProvider` is registered
     - Combine with `+` if multiple NPU EPs present (e.g. `snapdragon+ryzen-ai`)
   - Full run-id: `<UTC-datetime>_<hardware-tag>` (e.g. `20260513-160500_snapdragon`)

9. **Build the meta file** at a temp path (e.g. `/tmp/run_meta.json`) with the following fields:
   ```json
   {
     "host": "<from `hostname` command>",
     "skill_commit": "<from `git rev-parse HEAD:.claude/skills/winml-cli/SKILL.md` — null if not in git>",
     "winml_version": "<from `uv run winml --version`>",
     "registered_eps": [<from step 1>],
     "started_at_utc": "<recorded at start of step 1>"
   }
   ```

10. **Archive the run with Pass@K aggregation.** Call:
    ```bash
    python <workspace>/e2e_eval/archive_run.py <run-id> \
        --meta /tmp/run_meta.json \
        --cases <space-separated eligible case ids> \
        --skipped '<JSON array of {case_id, reason}>'
    ```
    This script:
    - For each case, walks all `scratch/<case>/trial-*` subdirs and copies each trial's lightweight files into `runs/<run-id>/cases/<case>/trial-<N>/`. Heavy binaries (>1 MB) are registered in `artifacts_manifest.json` but not duplicated.
    - Computes Pass@K per case (K = number of trials, pass = how many trials had every assertion pass) and writes `runs/<run-id>/cases/<case>/aggregate.json`.
    - Writes `runs/<run-id>/meta.json` and `runs/<run-id>/report.md` (the report shows the Pass@K column per case).

11. **Tell the user** where the archived report lives (`e2e_eval/runs/<run-id>/report.md`) and print the Pass@K summary table from the script's stdout to the chat.

## Notes

- Don't grade the CLI's stdout content — only check exit codes, artifact existence, and the agent's final message regex match. See `grade_case.py` for the canonical assertions.
- If an EP-required case ran on the wrong hardware (slipped through somehow), the agent's commands may silent-fallback. Flag this in the report.
- Pass@K cost: K=3 default means roughly 3× the wall time of a single-run setup. With ≤6 eligible cases and parallel spawn, expect 5–15 minutes total on a fast box; longer if HF model downloads serialize.
- If 2 of 3 trials pass for a case, that's a real signal — the case is flaky, worth investigating before shipping. Don't average to look like 67% in some composite score; report Pass@K honestly per case.
