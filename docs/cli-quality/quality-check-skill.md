---
name: cli-quality-check-workflow
description: Operational workflow an AI agent follows when asked to audit the winml CLI against the quality checklist. Provides the 5-phase audit procedure (Setup, Static triangulation, Runtime probing, Rule decomposition, Cross-command sweep, Wrap-up) plus failure-mode catalog and regression-test template. The actual rules live in [quality-checklist.md](../../../docs/cli-quality/quality-checklist.md). Use when reviewing a command before declaring it "Ready", scoping a CLI cleanup PR, investigating a CLI quality regression, or running a tool-wide consistency sweep.
---

# winml CLI Quality-Check Skill (Agent Workflow)

This skill is the **operational workflow** an agent follows when asked to audit the winml CLI. The rules being audited live in [quality-checklist.md](../../../docs/cli-quality/quality-checklist.md) (single source of truth, written for human review). This file tells an agent how to apply those rules systematically without missing the failure modes that have tripped past audits.

## Companion documents

| File | Role |
|---|---|
| [`quality-checklist.md`](quality-checklist.md) | The 6 rule sections (R1.x … R6.x). Single source of truth for what "Ready" means. |
| [`CLI_commands.md`](CLI_commands.md) | **Per-command invocation matrix (success + failure scenarios). This is the seed input for every Phase 2 probe — do not invent invocations, take them from this file.** |
| [`CLI_quality_check_report.md`](CLI_quality_check_report.md) | Latest audit findings; the artifact this workflow produces. |
| [`CLI_UX_Capture.md`](CLI_UX_Capture.md) | Captured `--help` + happy-path output per command (cited as evidence from the report). |

This workflow is structured as **5 phases with hard gates between them**. Do not advance to the next phase until the current phase's exit criteria are satisfied. Each phase ends with a self-check that names the most common ways agents fail this audit, so the agent can catch its own mistakes before the user has to.

### Phase 0 — Setup (must complete before any other phase)

- **0.1 Resolve scope.** Default = every file under `src/winml/modelkit/commands/*.py` + `cli.py` + shared modules (`utils/cli.py`, option decorators, error classes). If the user names a specific command, narrow to that file + its shared imports.
- **0.2 Enumerate commands.** Run `winml --help` and list every subcommand. **Use this list verbatim throughout the audit** — never invent or omit commands.
- **0.3 Configure the shell for capture.** Windows-specific: set `[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)`, `$env:PYTHONIOENCODING='utf-8'`, `$env:PYTHONUTF8='1'` in the PowerShell session. Without this, captured stdout will be cp1252-mangled and you will mis-diagnose Unicode bugs.
- **0.4 Build the coverage matrix file** (`temp/audit/coverage.md`). Rows = every rule (R1.1 … R6.4 + every sub-point from §4 below). Columns = every command from 0.2 + a `cross-cutting` column. Every cell starts as `TODO`. Every cell must end as `PASS` / `FAIL <issue-id>` / `N/A <reason>` / `DEFERRED <reason>` before Phase 5 can complete.
- **0.5 Diff against any prior audit.** If `CLI_AUDIT*.md` / `cli-quality-*.md` exists, list every finding ID. Each one must be re-verified in this run (PASS with evidence, re-FAIL, or marked obsolete with reason).

**Phase 0 exit criteria**: scope decided · command list captured · UTF-8 shell configured · empty coverage matrix on disk · prior-audit IDs listed.

---

### Phase 1 — Static triangulation (no runtime yet)

For every command from 0.2, read **three sources** and reconcile them. Any disagreement is a finding before a single command runs.

1. **`--help` output** — list every option, default, accepted values, advertised behavior.
2. **README / docs references** — note documented usage and examples.
3. **Source file** — enumerate every `@click.option` / `@click.argument`, every `if/elif` branch, every `# TODO` / `hack_` / `raise NotImplementedError` / `sys.exit`, every top-level import (R4.3), every silent `pass` in an `except`.

**Mandatory grep sweeps** (run all of these once, log results in the matrix; absences are findings):

| Anti-pattern | Grep | Rule it violates |
|---|---|---|
| Heavy import at module top | `grep -n "^\s*import \(torch\\|onnxruntime\\|transformers\\|optimum\)" src/winml/modelkit/commands/*.py` | R4.3 |
| EP hard-coded by string | `grep -rn "if ep ==\\|== \"qnn\\|== \"QNN" src/winml/modelkit/` | R5.2 |
| Bare `sys.exit` outside `cli.py` | `grep -rn "sys\\.exit(" src/winml/modelkit/commands/` | R4.5 |
| Forbidden jargon in help strings | `grep -rn "WinMLSession\\|TasksManager\\|_run_onnx_benchmark\\|auto_class" src/winml/modelkit/commands/` | R1.1 |
| `print(` for logs/progress | `grep -rn "^\\s*print(" src/winml/modelkit/commands/` | R3.3 |
| Missing `--force` machinery | `grep -rn "force\\|exist_ok=False\\|FileExistsError" src/winml/modelkit/commands/` | R3.4 / R5.5 |
| Missing SIGINT / atomic rename | `grep -rn "SIGINT\\|signal\\.\\|os\\.replace\\|tempfile\\." src/winml/modelkit/commands/` | R5.6 |
| Missing `NO_COLOR` handling | `grep -rn "NO_COLOR\\|no_color\\|isatty" src/winml/modelkit/` | R3.3 |
| Missing `winml cache` group | `grep -rn "def cache\\|@.*group.*cache" src/winml/modelkit/` | R5.4 |
| Swallowed `ImportError` | `grep -rn "except ImportError" src/winml/modelkit/commands/` (each must include an install hint) | R6.2 |
| In-progress flags | `grep -rn "TODO\\|hack_\\|NotImplementedError\\|Falling back to default" src/winml/modelkit/commands/` | R1.2 |

**Phase 1 self-check (answer YES to all before advancing)**:
- [ ] Did I read source for *every* command from 0.2, not just the ones the user mentioned?
- [ ] Did I cross-reference `--help` against source for every option (`--help` may advertise dead flags or hide live ones)?
- [ ] Did I run *every* anti-pattern grep above and record results in the matrix?
- [ ] Did I list every `@click.option` per command (this is the option-coverage seed for Phase 2)?

---

### Phase 2 — Runtime probing (the part agents skip; do not skip)

Static evidence is necessary but **never sufficient**. The following bugs are invisible to source reading and must be discovered by execution: silent flag overrides, late validation, raw third-party tracebacks, slow imports, sidecar files, CWD pollution, Unicode crashes, missing `--force`, missing dataset/config validation. The bias is **fewer well-targeted runs over many ad-hoc runs** — but the runs below are mandatory, not optional.

**Source of probes**: open [`CLI_commands.md`](CLI_commands.md) before starting Phase 2. It enumerates the success and failure scenarios for every command in this CLI, plus the 10 cross-cutting probes. The tables below are the *categories* of probes the matrix file fills in concretely. **Do not invent your own invocations** — if a probe is missing from `CLI_commands.md`, add it there first (so the next audit picks it up), then run it.

#### 2A. Per-command negative-input probe set (run for *every* output-producing command)

For every command run **at least these 6 invocations**, capture exit code + last 15 lines of stdout/stderr, and record evidence in the matrix:

| # | Probe | Rule(s) | What it catches |
|---|---|---|---|
| 1 | **Happy path, defaults** — `winml <cmd> -m <known-good> -o <out>` | baseline | Reference for echo-back checks. |
| 2 | **Happy path, explicit non-default flags** — add `--device`, `--precision`, `--ep`, etc. Compare output **line-by-line** against (1). | R2.7, R2.8 | Inconsistent value spelling, missing echo-back. |
| 3 | **Same `-o` re-run** | R3.4, R5.5 | Silent overwrite. |
| 4 | **No `-o`** (where applicable) | R3.4 | Default file dropped in CWD. |
| 5 | **Bad `Choice` value** — `--ep bogus`, `--precision banana`, `--device gpyu`, `--task xyz` | R1.5, R3.1 | Missing did-you-mean, missing valid-set listing. |
| 6 | **Conflicting flag combo** — `--device cpu --ep qnn`, `--device gpu --ep qnn`, `--iterations 0`, `-1`, `--samples 0` | R4.2 | Silent override, late-stage crash, garbage output. |

**For each invocation also verify** (these are easy to forget):
- (i) **Echo-back**: every flag the user passed appears in the output (e.g. `--precision int8` †’ `Precision: int8`). Missing echo = FAIL R2.7 or R3.3.
- (ii) **No undeclared sidecars**: `dir <output-dir>` shows only files the command announced in stdout. Hidden `*_qnn.bin`, `*.data`, `report.html` = FAIL.
- (iii) **No CWD pollution**: `dir <CWD>` after the run shows no new files unless `-o` was a CWD-relative path.
- (iv) **Wall-clock time** captured for every probe (needed for R4.3 / R4.4).

#### 2B. Input-validation probe set (the part most often missed — covers R3.1, R4.1, R4.2 fully)

For every command that accepts `-c/--config`, `--dataset`, `--task`, `--shape-config`, or any other structured input:

| # | Probe | Expected | If fails |
|---|---|---|---|
| 1 | `-c <missing-path>.json` | Fast Click error: `Path '...' does not exist.` | R4.1 FAIL |
| 2 | `-c <truncated-json>` (e.g. `{ "key": `) | Fast `Invalid JSON in config: ...` | R4.1 FAIL |
| 3 | `-c <wrong-schema>.json` (e.g. `{ "this": "is wrong" }`) | Fast schema error **before** any banner / Setup line | R4.1 FAIL — late validation is a real bug; mark as FAIL even if the error message itself is good |
| 4 | `-c <empty-object>.json` (`{}`) | Same as 3 | R4.1 FAIL |
| 5 | `--dataset does-not-exist` | One-line `Dataset 'does-not-exist' not found on the Hub. Check spelling.` — **no traceback frames** at default verbosity | R3.1 FAIL |
| 6 | `--dataset <valid-but-incompatible>` (e.g. `glue/mrpc` on an image model) | One-line `Dataset 'glue/mrpc' is incompatible with task 'image-classification'. Expected ...; found ...` — **no traceback** | R3.1 FAIL |
| 7 | `--task <bogus>` | Click `Choice` error listing valid tasks (or one-line app error with valid set + `did you mean`) — **no third-party class name** like `TasksManager` | R1.5 / R3.1 FAIL |
| 8 | Required flag missing — `winml <cmd>` (no args) | `Error: Missing option '--model' / '-m'.` **plus a runnable example** | R3.1 sub-point (e) FAIL |

**This block exists because input validation is the single biggest source of missed findings.** A command that runs `--help` cleanly and produces a valid output on the happy path can still leak a 4-frame `datasets/load.py` traceback on a typo'd `--dataset`. That bug is invisible to Phase 1 entirely.

#### 2C. Cross-cutting environment probes (run once across the tool, not per command)

| # | Probe | Rule | FAIL condition |
|---|---|---|---|
| 1 | `Measure-Command { uv run winml --help }` × 3 warm trials | R4.3a | Median > 500 ms |
| 2 | `Measure-Command { uv run winml --version }` × 3 warm trials; capture full output | R4.3b, R1.1e | > 500 ms; or output lacks Python/ORT/EP provenance |
| 3 | `Measure-Command { uv run winml sys }` warm | R4.3 | > 500 ms (local-state only) |
| 4 | First-output latency on each long-running command (`inspect`, `perf`, `export`, `build`) | R4.4a | Cursor blank > 500 ms |
| 5 | `$env:NO_COLOR='1'; winml <cmd> > out.txt 2>&1` then `Select-String -Pattern '\x1b\['` | R3.3d | Any ANSI escape in `out.txt` |
| 6 | `$env:CI='true'; winml <cmd> > out.txt 2>&1` then check ANSI | R3.3e | Any ANSI escape |
| 7 | `winml --help \| Select-String 'no-color'` | R3.3f | No `--no-color` flag |
| 8 | `winml <cmd> > out.txt 2> err.txt`; verify data †’ `out.txt`, logs/banners/progress †’ `err.txt` | R3.3c | Data in `err.txt` or banners in `out.txt` |
| 9 | `cmd /c "uv run winml inspect -m microsoft/resnet-50"` (legacy cp1252 codepage) | R3.3h | `UnicodeEncodeError`, mojibake, or literal `\u2550` |
| 10 | `$env:HF_HUB_OFFLINE='1'; winml inspect -m <cached-model>` | R6.3 | Network call attempted; or fails despite cache |
| 11 | SIGINT each pipeline command at ~50% progress; check output dir + CWD | R5.6 | Partial files left behind |
| 12 | EP-requiring command in venv missing the EP package | R6.2 | Raw `ImportError` traceback |
| 13 | Run all 13 subcommand `--help`s with `Select-String 'NO_COLOR\|HF_HUB_OFFLINE\|HF_HOME\|HF_TOKEN\|WINML_'` | R1.6 | Zero hits = env vars not documented |
| 14 | For every `-c/--config` command: `Select-String 'schema\|fields\|keys\|JSON Schema'` in `--help` | R1.3 | No schema link / no field list documented |

**Why both PowerShell 7 *and* `cmd.exe`**: PS7 with UTF-8 stdout silently masks cp1252 bugs. The `cmd /c "..."` path is the only reliable way to trigger the legacy-codepage crash on a modern dev box. Skip this and you will mark R3.3h as PASS when it is FAIL.

#### 2D. Static-data vs live-measurement labeling

When a command displays performance numbers, latency, accuracy, or "verdict" data, verify the source: live measurement vs canned data from a JSON catalog shipped in the wheel? If canned, output **must label it** (`Source: catalog (last updated YYYY-MM-DD)`). A table that mixes EPs the host machine cannot run is canned by definition; presenting it identically to live measurements is a P1 finding.

**Phase 2 self-check (answer YES to all before advancing)**:
- [ ] Did I run all 6 negative-input probes (2A) for *every* output-producing command, not just `export`?
- [ ] Did I run the 8 input-validation probes (2B) for *every* command that accepts `-c`, `--dataset`, `--task`, `--shape-config`?
- [ ] Did I run all 14 cross-cutting probes (2C) including the `cmd.exe` Unicode probe and the `HF_HUB_OFFLINE` probe?
- [ ] Did I capture wall-clock time and last 15 lines of stdout/stderr for every invocation?
- [ ] Did I check for **echo-back** of every flag the user passed in every probe?
- [ ] Did I check for **sidecar files** in every output directory and **CWD pollution** after every run?

---

### Phase 3 — Decompose multi-clause rules

Several rules pack multiple requirements into one sentence. Record one PASS/FAIL per sub-point in the matrix; collapsing them into a single cell hides real findings.

| Rule | Sub-points |
|---|---|
| **R1.1** | (a) one-line purpose · (b) runnable example in `--help` · (c) no internal class names / private symbols · (d) top-level grouping when many subcommands · (e) `LEARN MORE` footer + multi-line `--version` provenance |
| **R3.1** | (a) correct diagnosis · (b) actionable next step · (c) no traceback at default verbosity · (d) did-you-mean for typo'd subcommand/flag/Choice · (e) missing-required-option error includes a runnable example · (f) Choice errors list the valid set |
| **R3.3** | (a) jargon-free table cells · (b) `…` truncation marker · (c) data †’ stdout / logs †’ stderr · (d) `NO_COLOR` honored · (e) `CI=true` honored · (f) `--no-color` flag exists · (g) `not isatty()` strips color · (h) Windows non-UTF-8 codepage falls back to ASCII without crash · (i) no literal `\u2550` / `\U0001f4cb` when piped |
| **R4.3** | (a) `winml --help` ‰¤ 500 ms warm · (b) `winml --version` ‰¤ 500 ms · (c) heavy deps lazy-imported (greps in 1 above return empty) |
| **R4.4** | (a) first-output ‰¤ 500 ms · (b) every multi-second stage labeled · (c) single progress mechanism · (d) no double-printed `100%` |

---

### Phase 4 — Cross-command consistency sweep

After per-command audits are complete, build a **shared-concept matrix**: rows = every shared concept (`--model`, `--device`, `--ep`, `--task`, `-o`, `-v`, `-q`, `--format`, `--precision`, `--samples`, `-c`, `--trust-remote-code`, …); columns = every command. Each cell records: **type · default · short-flag · accepted-values · help wording**.

Any cell that disagrees with its column is an R2.x finding. Without this step, an agent will produce a strong per-command audit and still miss "the same flag means different things in 10 commands."

**Probe to seed the matrix**: for each shared concept, run

```powershell
foreach ($cmd in @('analyze','build','compile','config','eval','export','hub','inspect','optimize','perf','quantize','sys')) {
  "--- $cmd ---"
  uv run winml $cmd --help 2>&1 | Select-String '<concept>' -Context 0,2
}
```

…and visually diff. Any difference in `Choice`, default, or wording is a finding.

---

### Phase 5 — Wrap-up validation (the step that catches "I forgot to test X")

Before declaring the audit done, run the following **mandatory closing checklist**. Skipping any of these is the most common cause of incomplete audits.

1. **Coverage matrix completeness**: open `temp/audit/coverage.md`. **Zero cells may be `TODO`.** Every cell is `PASS` (with evidence link), `FAIL <issue-id>`, `N/A <reason>`, or `DEFERRED <reason>`. If any cell is `DEFERRED`, the report's Summary must list the deferred items at the top.

2. **Re-read the checklist end-to-end against the report.** For every rule R1.1 – R6.4, ask: *"What artifact would prove this rule was tested?"* Then `Select-String` the report for that artifact. Examples:

   | Rule | Artifact to grep for in report |
   |---|---|
   | R1.3 | "config schema" / "JSON Schema" / "field list" |
   | R1.5 | "did you mean" / "Possible options" / "valid set" |
   | R1.6 | "NO_COLOR" / "HF_HUB_OFFLINE" / "WINML_" / "Environment Variables" |
   | R3.1 (e) | "Missing option" + "Example:" |
   | R3.3 (h) | "cmd.exe" / "cp1252" / "UnicodeEncodeError" |
   | R3.4 | "--force" / "overwrite" / "clobber" |
   | R4.1 | "malformed config" / "invalid dataset" / "fail-fast" / "before banner" |
   | R4.3 | "warm" / "‰¤ 500 ms" / "lazy import" |
   | R5.6 | "SIGINT" / "Ctrl+C" / "atomic rename" |
   | R6.2 | "ImportError" / "uv pip install" |
   | R6.3 | "HF_HUB_OFFLINE" / "offline" |

   **A rule with zero artifact hits in the report is a missed audit, not a clean pass.** Either run the probes and add the finding, or explicitly mark `N/A` in the matrix with a one-line reason.

3. **Diff against prior audit**. Every finding ID from the prior audit must appear in the new report as one of: re-FAIL, PASS-with-evidence, or obsolete-with-reason. Silent omission is forbidden.

4. **Severity sanity check**. Re-grade every finding against the severity legend in the report header. Common mistakes: marking a raw traceback as P2 (it is at least P1), marking a security flag with no warning as P2 (it is P1 minimum), marking a documented-example crash as P1 (it is P0).

5. **Findings-outside-the-six-sections audit**. Real bugs that no rule covers go in a `Notes` appendix — not silently dropped. Then file an issue against this checklist to add the missing rule.

**Phase 5 self-check (answer YES to all before delivering)**:
- [ ] Coverage matrix has zero `TODO` cells.
- [ ] Every rule R1.1 – R6.4 has at least one artifact reference in the report **or** an explicit `N/A` justification.
- [ ] Every prior-audit finding is accounted for.
- [ ] No raw third-party traceback is filed at less than P1.
- [ ] Report header severity counts match the body (P0/P1/P2/P3 totals are accurate).

---

### Failure-mode catalog (lessons from past audits)

These are the specific ways past agents failed this audit. If your draft matches any of these patterns, redo the affected phase before delivering.

| Failure mode | Symptom | Phase that catches it |
|---|---|---|
| **One-happy-path-per-command** | Report has a section per command; each has 1–2 findings; cross-cutting section is short. | Phase 2A (6 mandatory probes per command). |
| **Trusted `--help`** | Reported a flag as working because help shows it; never invoked it. | Phase 1 (cross-reference source) + Phase 2 (every option exercised). |
| **Skipped input validation** | No findings about bad config, missing dataset, malformed JSON, or typo'd `--task`. | Phase 2B (mandatory). |
| **Skipped env vars** | No mention of `NO_COLOR`, `HF_HUB_OFFLINE`, `WINML_*`, `--no-color`. | Phase 2C probes 5–7 + 13. |
| **PowerShell-only** | Marked R3.3h PASS without ever running `cmd /c`. | Phase 2C probe 9. |
| **No `--version` provenance check** | Accepted `winml, version 0.0.2` as good enough. | Phase 3 R1.1 sub-point (e). |
| **Late-validation accepted as PASS** | Marked malformed-config as "errors cleanly" without noting the Setup banner printed first. | Phase 2B probes 3–4 (the *order* matters). |
| **Coverage matrix not maintained** | Report exists; no matrix file; some rules untested. | Phase 0.4 + Phase 5.1. |
| **Static-only on runtime rules** | R4.3 / R4.4 / R5.6 marked PASS without execution evidence. | Phase 2C + Phase 5.2. |
| **Wrap-up skipped** | Delivered the report; user has to point out missing rules. | Phase 5 (do not skip — this is the step that prevents the user having to remind you). |

---

### Regression-prevention test suite (for CI)

To prevent the failure modes above from re-occurring after each fix, codify the high-value Phase 2 probes as pytest fixtures under `tests/cli_quality/`. Each test runs the CLI as a subprocess and asserts on captured output. Suggested structure:

```text
tests/cli_quality/
”œ”€”€ conftest.py              # UTF-8 shell setup, common fixtures, CLI runner helpers
”œ”€”€ test_discoverability.py  # R1.x: --help shape, --version provenance, env-var docs
”œ”€”€ test_consistency.py      # R2.x: shared-concept matrix asserts (--model, --device, --ep)
”œ”€”€ test_ux.py               # R3.x: NO_COLOR, --no-color, cmd.exe encoding, missing-option example
”œ”€”€ test_reliability.py      # R4.x: warm timing budgets, fail-fast on bad config/dataset
”œ”€”€ test_correctness.py      # R5.x: idempotency, --force, SIGINT, cache visibility
”””€”€ test_environment.py      # R6.x: HF_HUB_OFFLINE, optional-EP install hints
```

Examples of high-value regression tests (each one corresponds to a real past finding):

```python
def test_help_warm_under_500ms():
    # R4.3a — runs `winml --help` 3 times, asserts median < 500 ms
    ...

def test_version_includes_provenance():
    # R1.1e — asserts python/onnxruntime/transformers strings appear in `--version` output
    ...

@pytest.mark.parametrize("cmd", OUTPUT_PRODUCING_COMMANDS)
def test_overwrite_requires_force(tmp_path, cmd):
    # CC-P1-2 — first run succeeds, second run errors without --force
    ...

@pytest.mark.parametrize("cmd", CONFIG_ACCEPTING_COMMANDS)
def test_malformed_config_fails_before_banner(tmp_path, cmd):
    # BLD-P1-2 — schema validation runs before any "Setup" / "Stages" banner
    ...

def test_eval_bogus_dataset_no_traceback():
    # EVL-P0-1 — `--dataset does-not-exist` produces a one-line error, no `datasets/load.py` frames
    out = run_cli("eval", "-m", "microsoft/resnet-50", "--dataset", "does-not-exist", "--samples", "3")
    assert "Traceback" not in out.stderr
    assert "datasets/load.py" not in out.stderr

def test_no_color_strips_ansi():
    # CC-P2-7 — NO_COLOR=1 produces zero ANSI escapes when piped
    out = run_cli("sys", "--format", "compact", env={"NO_COLOR": "1"})
    assert "\x1b[" not in out.stdout

def test_cmd_exe_no_unicode_crash():
    # INS-P1-3 — `cmd /c winml inspect ...` does not crash on cp1252
    result = subprocess.run(["cmd", "/c", "uv run winml inspect -m microsoft/resnet-50"], capture_output=True)
    assert "UnicodeEncodeError" not in result.stderr.decode("cp1252", errors="replace")
```

A new finding from a future audit should arrive as **(a) a report entry + (b) a corresponding pytest test**. The pytest test then prevents regression and serves as the executable specification of the fix.

---

### Feature-owner self-check (human-only — agents cannot verify these reliably)

The phases above catch UX, consistency, performance, and validation issues that an automated agent can detect by reading help text, source, and runtime output. They **cannot** catch the following three classes of issue, because each one requires (a) the design spec the command was built against, (b) hardware diversity the audit machine does not have, or (c) a clean install state the audit cannot create. These are explicitly the **feature owner's responsibility** before declaring a command "Ready" and must be re-checked on every release.

The audit report shall include a **Feature-Owner Self-Check** appendix listing each of the three checks below with a one-line owner-signed status: `PASS <date> <owner>` / `FAIL <issue-id>` / `DEFERRED <reason>`. An audit that does not include this appendix is incomplete.

#### SC-1 — Functional correctness against the design spec (covers R1.2 + R5.1)

**Owner action**: open the design doc / PRD / issue that authorized this command. For every behavior the spec promises, run the corresponding invocation and confirm the observable output matches the spec.

**Why agents cannot do this**: agents do not have access to the design intent. An agent can verify that `--precision int8` *runs without crashing* but cannot tell you whether the resulting model is actually quantized to INT8 — because the spec for "what int8 means in this command" lives outside the codebase. Same for `--preset qnn-compatible` (does the output actually pass QNN compilation?), `--task image-classification` (does the auto-discovered class match the documented task?), `--shape-config` (does the model accept the declared shape at inference?).

**Minimum check per command**:
- List every flag whose name implies a behavior (`--quantize`, `--preset X`, `--precision int8`, `--use-cache`, `--trust-remote-code`, `--shape-config`).
- For each, run an invocation that should produce a *visible* difference vs. omitting the flag.
- Confirm the difference is real (output file size, opset, EP-specific node count, presence of QDQ ops, cache-hit log line, observed download of remote code, model input shape).
- A flag whose presence vs. absence produces *byte-identical* output is a feature gap or a silent fallback (R1.2 FAIL).

#### SC-2 — Cross-EP coverage (covers R5.2)

**Owner action**: for every EP the command's `--ep` choice list advertises, run a happy-path invocation on host hardware that supports that EP. Confirm correct results, not just "exit 0".

**Why agents cannot do this**: the audit machine has only the EPs it has. The Snapdragon X Elite audit box has QNN + CPU; it has no DML GPU, no CUDA, no OpenVINO, no VitisAI, no NV TensorRT-RTX. An agent running on that box will mark `--ep dml` and `--ep openvino` as `DEFERRED <no hardware>` no matter how thorough Phase 2 is. Only a feature owner with access to the full EP matrix (or an internal CI lab) can close those cells.

**Minimum check per command that exposes `--ep`**:
- Build an EP × command matrix (rows = EPs, columns = commands that take `--ep`).
- For each cell, run on hardware that supports that EP and capture: exit code, EP actually selected (R2.7 echo-back), one piece of EP-specific evidence (e.g. `QNNExecutionProvider` appears in `winml sys` providers list during the run, or the output `.onnx` contains EP-context nodes).
- Confirm `--ep <X>` on a host that lacks `<X>` produces the correct error from R6.2 (one-line install hint, exit 3) — not a raw `ImportError` and not a silent fallback to CPU.
- Hard-coded `if ep == "qnn"` branches are R5.2 violations even when the test passes — re-grep Phase 1 anti-pattern table.

#### SC-3 — First-run success on a fresh install (covers R6.1 + R6.2)

**Owner action**: in a clean Python venv (no `winml`, no model cache, no `~/.cache/winml`, no `~/.cache/huggingface`), `pip install` the wheel and run each command's documented happy-path example exactly as printed in `--help`. Every example must succeed without manual setup, without script downloads, and without `ImportError`.

**Why agents cannot do this**: the audit machine has the dev venv with every optional dep installed and every model already cached. Of course `winml export -m microsoft/resnet-50` works — the model has been on disk since the project started. A first-time user without the cache, without `optimum`, without `onnxruntime-qnn`, on a corporate proxy, hits errors the audit never sees.

**Minimum check before each release**:
1. `python -m venv .venv-fresh && .venv-fresh\Scripts\activate`.
2. `pip install <wheel>` (not `uv run` from the source tree — that bypasses the wheel).
3. Clear caches: `Remove-Item -Recurse -Force ~\.cache\huggingface, ~\.cache\winml -ErrorAction SilentlyContinue`.
4. For each command, run the *first* example printed in `winml <cmd> --help` verbatim. Time-to-first-success and exit code recorded.
5. Repeat with `HF_HUB_OFFLINE=1` to verify the offline path.
6. Repeat behind a proxy (`HTTP_PROXY=http://...`) if your shipping target includes corporate users.
7. Any failure = R6.1 / R6.2 violation. Common findings: `Run scripts/download_rules.py` referenced but file is not in the wheel; raw `ModuleNotFoundError: No module named 'optimum'` instead of an install hint; first-run download has no progress bar.

---

#### Why these three are called out separately

They are mentioned *implicitly* in the rule tables (R1.2, R5.1, R5.2, R6.1, R6.2), but the rule wording assumes the reviewer has the **domain knowledge** (what the design promises), the **hardware** (every EP), and the **install isolation** (fresh venv) needed to test them. An automated agent has none of these. Without an explicit "feature owner signs here" step, these issues hide behind passing CI and only surface in user bug reports.

The agent shall include this appendix template in every audit report:

```markdown
## Feature-Owner Self-Check Appendix

Owner: <github handle>
Sign-off date: <YYYY-MM-DD>
Wheel under test: <version + commit>

| Check | Scope | Status | Evidence / Issue |
|---|---|---|---|
| SC-1 design-spec correctness | per-command flag-behavior matrix | PASS / FAIL <id> / DEFERRED | <link to spec doc + per-flag evidence> |
| SC-2 cross-EP coverage      | EP × command matrix              | PASS / FAIL <id> / DEFERRED | <link to lab run, list any EP not exercised> |
| SC-3 first-run success      | clean venv + cleared cache       | PASS / FAIL <id> / DEFERRED | <link to clean-install transcript> |
```

If any cell is `DEFERRED`, the report's Summary section lists the deferred check and the owner's plan to close it before release.

---
