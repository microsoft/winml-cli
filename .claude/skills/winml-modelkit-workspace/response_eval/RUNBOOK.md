# Response eval — Runbook

Pillar 2: given the skill loads, does the agent give sound advice? Iterate the skill body against case-by-case concept assertions + static command-shape checks.

This RUNBOOK is **agent-facing** — when a user says "iterate response", "add a response case", or "let's push iter-N", the agent driving the conversation follows these steps.

## What this measures

For each case, with a fresh subagent:
- **Concept assertions** — does the response identify the right hardware path, refuse out-of-scope work, suggest the right command order? Judged by reading the response text.
- **Static command check** — does every `winml ...` command in the response use real flags from `winml <cmd> --help`? Runs `verify_commands.py` automatically.

Each case is run twice per iteration: **with_skill** (skill loaded) and **without_skill** (baseline). The delta tells us the skill is actually doing work.

## How to iterate

### Step 1 — Read the current state

```
response_eval/iterations/iter-N/benchmark.md           — latest iteration's pass rate
response_eval/iterations/iter-N/eval-*/grading.json    — per-case results
response_eval/cases.json                                — canonical case list
```

Look at which assertions failed. Each failure is one of:
- **Skill content gap** — the body doesn't teach what the assertion checks. Fix body.
- **Static failure** — the response quoted a bad flag / used a positional arg where flag-only. Fix body (usually a "consult --help" reminder) OR document the flag explicitly.
- **Coverage gap** — the case exposes a scenario the skill doesn't address. Either add to body OR mark as out-of-scope explicitly.

### Step 2 — Decide what to change

| Action | When |
|---|---|
| **Edit SKILL.md body** | A specific assertion fails because the skill doesn't cover that case well. Most common move. |
| **Add a new case** | Suspected coverage gap; want to lock in a scenario the current set doesn't test. See "Adding a case" section. |
| **Both** | Discover gap via new case, then fix body. |

### Step 3 — Edit SKILL.md body

`.claude/skills/winml-modelkit/SKILL.md` — everything below the YAML frontmatter is the body. The `description` field belongs to Pillar 1 (Trigger); don't change it as part of a response iteration unless you explicitly want to.

### Step 4 — Set up the next iteration directory

Pick `N+1` where N is the highest existing `iter-*` number.

```bash
cd response_eval/iterations
cp -r iter-N iter-(N+1)

# Clear with_skill outputs and gradings — we'll regenerate
rm iter-(N+1)/eval-*/with_skill/run-1/outputs/response.md
rm iter-(N+1)/eval-*/with_skill/run-1/grading.json
rm iter-(N+1)/eval-*/with_skill/run-1/timing.json

# baselines are reused if the SKILL.md body is the only change
# (the baseline doesn't load the skill, so its responses are skill-version-agnostic)
```

### Step 5 — Spawn with_skill agents (one per case)

For each case in `cases.json`, spawn a subagent (parallel). The prompt should:
- Include the case prompt verbatim
- Give the path to the new SKILL.md and tell the agent to read it first
- Explicitly tell the agent **this is chat mode** — it should write a response with commands as text, NOT execute any commands
- Specify where to save the response: `iter-(N+1)/eval-<case_id>/with_skill/run-1/outputs/response.md`

After each subagent completes, save its `total_tokens`, `tool_uses`, `duration_ms` as `timing.json` in the same directory.

### Step 6 — Grade

This is the **manual judgment step**. For each case, read the new response and compare against the assertions in `cases.json`. Write your judgments into `iter-(N+1)/grade.py`:

```python
# iterations/iter-(N+1)/grade.py
# Copy structure from iter-N/grade.py; update or replace per-case entries.
gradings = {
    "eval-<case_id>": [
        ("<assertion text>", True|False, "<evidence quote from response>"),
        ...
    ],
    ...
}
```

`grade.py` walks each case's response, applies these judgments, and writes per-case `grading.json` files. It also calls `verify_commands.py` for the static check assertion.

Run it:

```bash
python iterations/iter-(N+1)/grade.py
```

### Step 7 — Static CLI command verification

```bash
python response_eval/run_full_verify.py iter-(N+1)
```

Writes `iter-(N+1)/cli_verification.md` — flags any case where the agent quoted an invalid flag.

### Step 8 — Aggregate benchmark

If skill-creator tooling is available:

```bash
cd <path-to-skill-creator>
python -m scripts.aggregate_benchmark <workspace>/response_eval/iterations/iter-(N+1) --skill-name winml-modelkit
```

Otherwise build benchmark.md by hand:
- Tally per-case with_skill pass rate vs baseline pass rate
- Compute overall pass rate and delta

### Step 9 — Compare and decide

Read `iter-(N+1)/benchmark.md` against `iter-N/benchmark.md`:
- Did with_skill pass rate go up?
- Did the assertion that was failing now pass?
- Any regressions in other cases?

If yes → ship iter-(N+1) as the new baseline. If no → revert SKILL.md edits or try a different angle.

## Adding a new case

1. Decide the scenario and write the prompt in real user voice. Avoid sterile "format X for Y" framing.

2. Write concept assertions — each one a single objectively-checkable statement. Aim for 4–7 per case.

3. Add the case to `response_eval/cases.json`:
   ```json
   {
     "id": <next-id>,
     "name": "<descriptive-slug>",
     "prompt": "...",
     "assertions": [
       {"id": "<short-id>", "text": "..."},
       ...
     ]
   }
   ```

4. Create the per-iteration directory for the next iter:
   ```
   iter-(N+1)/eval-<name>/{with_skill,without_skill}/run-1/outputs/
   iter-(N+1)/eval-<name>/eval_metadata.json     ← copy from cases.json with the same prompt + assertions
   ```

5. Spawn BOTH with_skill AND baseline subagents for the new case (baseline can't be reused from previous iter since this case didn't exist).

6. Grade both. The case has discrimination if baseline fails some assertions that with_skill passes — that's the signal the skill is doing real work for this scenario.

7. If baseline passes everything → the case is too easy; the skill isn't needed for it. Make the case harder or drop it.

## Cost per iteration

- 6 cases × 1 with_skill subagent + 0 baselines (reused) = 6 subagents
- Wall time: ~3–5 min (parallel)
- Grading: human judgment, ~5 min reading responses + writing evidence quotes

Total: 10–15 min per iteration.

## K=1 caveat

This pillar runs each case once per iteration (no Pass@K). LLM variance in response wording is small enough that single-trial signal is reliable for concept assertions. If a specific case ever shows flaky behavior (passing sometimes, failing other times when nothing changed), upgrade to K=3 for that case — see e2e_eval's pattern for how to extend.

## When to stop iterating

- Pass rate plateaued at ≥ 95% for two consecutive iterations
- Remaining failures are structural (skill internal tension, not fixable by wording)
- Delta vs baseline ≥ 30pp — confirms the skill is doing work

## Common pitfalls

- **Forgetting to reuse baseline**: if you only changed SKILL.md body, baselines from iter-N are still valid for iter-(N+1). Don't waste subagent calls.
- **Writing grade.py before reading responses**: judgments must be backed by evidence quotes. Read first, judge second.
- **Adding too many cases at once**: each new case should expose a specific gap. Don't bulk-add untargeted cases.
- **Not running `run_full_verify.py`**: the static command check is cheap and catches fabricated flags. Always run it as part of the iteration.
