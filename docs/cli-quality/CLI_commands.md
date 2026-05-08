# `winml` CLI — Per-Command Invocation Matrix

A catalog of representative invocations for every `winml` subcommand, covering both **success scenarios** (what users typically do) and **failure scenarios** (what auditors must probe). This document is the **input** to Phase 2 of the [quality-check skill](quality-check-skill.md): every row in the failure tables below should be executed during an audit, and the result captured in [`CLI_quality_check_report.md`](CLI_quality_check_report.md).

This file is intentionally redundant with the [quality checklist](quality-checklist.md) and [audit report](CLI_quality_check_report.md) — its purpose is to give a feature owner or auditor a single dense reference of *"how do I invoke this command in N realistic ways"* without reading the entire report.

## Conventions

- All invocations assume a PowerShell 7 session with `[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)` and `$env:PYTHONIOENCODING='utf-8'` (per Phase 0.3 of the skill).
- `<m>` = a known-good HuggingFace id (`microsoft/resnet-50` for image, `prajjwal1/bert-tiny` for text).
- `<onnx>` = a local `.onnx` file that exists (e.g. `temp\cli-audit\resnet.onnx`).
- `<out>` = an output path under `temp\` that the user wants populated.
- `<bad>` = a deliberately invalid value (path, id, choice value, etc.).
- **Status legend** in the success column: ✅ = expected to succeed; ⚠ = expected to succeed but with a known issue (links to finding ID); ❌ = expected to fail (this is the test).

## Cross-cutting probes (run once per audit, not per command)

| # | Probe | Expected | Catches |
|---|---|---|---|
| 1 | `Measure-Command { uv run winml --help }` × 3 warm | median ≤ 500 ms | R4.3a |
| 2 | `Measure-Command { uv run winml --version }` × 3 warm; capture output | ≤ 500 ms; multi-line provenance (Python, ORT, EPs) | R4.3b · R1.1e · TOP-P2-5 |
| 3 | `winml exprt` (typo) | "did you mean export" suggestion | R1.5 · TOP-P1-1 |
| 4 | `winml --help \| Select-String 'no-color\|NO_COLOR\|HF_HUB_OFFLINE'` | non-empty | R1.6 · CC-P2-7 · CC-P2-8 |
| 5 | `$env:NO_COLOR='1'; winml sys --format compact > out.txt 2>&1` then grep `\x1b\[` | zero ANSI escapes | R3.3d |
| 6 | `cmd /c "uv run winml inspect -m microsoft/resnet-50"` | no `UnicodeEncodeError`, no mojibake | R3.3h · INS-P1-3 |
| 7 | `$env:HF_HUB_OFFLINE='1'; winml inspect -m microsoft/resnet-50` (cached) | succeeds against cache | R6.3 |
| 8 | `winml <any-output-cmd>` (no required args) | error includes runnable example, not just `Missing option '-m'` | R3.1e · CC-P1-3 |
| 9 | Any output-producing command twice with same `-o` | second run errors without `--force` | R3.4 · R5.5 · CC-P1-2 |
| 10 | SIGINT (`Ctrl+C`) any pipeline command at ~50% | output dir + CWD have zero partial files | R5.6 |

---

## `winml` (top-level)

### Success scenarios

| # | Invocation | Status | Notes |
|---|---|---|---|
| 1 | `winml --help` | ✅ | Should warm-return ≤ 500 ms. |
| 2 | `winml --version` | ⚠ TOP-P2-5 | Returns `winml, version 0.0.2` only — no Python/ORT/EP provenance. |
| 3 | `winml <subcmd> --help` for every subcommand | ✅ | Each must include at least one runnable example (R1.1b). |

### Failure scenarios

| # | Invocation | Expected | Rule |
|---|---|---|---|
| 1 | `winml exprt` | "did you mean: export" + exit 2 | R1.5 |
| 2 | `winml --bogus-flag` | clean error + valid-flag suggestion | R1.5 |

---

## `winml analyze`

### Success scenarios

| # | Invocation | Status | Notes |
|---|---|---|---|
| 1 | `winml analyze -m <onnx>` | ✅ | Default analysis on local ONNX. |
| 2 | `winml analyze -m <onnx> --output <out>.json` | ⚠ ANA-P1-2 | `-o` short alias missing. |

### Failure scenarios

| # | Invocation | Expected | Rule |
|---|---|---|---|
| 1 | `winml analyze` (no args) | error with example | R3.1e |
| 2 | `winml analyze -m <bad>.onnx` | file-not-found, not "Network error" | R3.1a |
| 3 | `cmd /c "winml analyze -m <onnx>"` | no UnicodeEncodeError, no `\u2550` literals | R3.3h · ANA-P1-3 |

---

## `winml build`

### Success scenarios

| # | Invocation | Status | Notes |
|---|---|---|---|
| 1 | `winml build -m <m> -o <out-dir>` | ✅ | Default HuggingFace build path. |
| 2 | `winml build -c <valid-cfg>.json -m <m> -o <out-dir>` | ✅ | Config-driven build. |
| 3 | `winml build -m <onnx> -o <out-dir>` | ✅ | Build from local ONNX. |
| 4 | `winml build --trust-remote-code -m <m> -c <cfg>.json -o <out-dir>` | ⚠ BLD-P1-1 | Must emit security warning to stderr — currently silent. |

### Failure scenarios

| # | Invocation | Expected | Rule |
|---|---|---|---|
| 1 | `winml build` (no args) | error + example | R3.1e · CC-P1-3 |
| 2 | `winml build -c <missing>.json -m <m> -o <out-dir>` | fast `Path '...' does not exist` | R4.1 |
| 3 | `winml build -c <truncated-json>.json -m <m> -o <out-dir>` | fast `Invalid JSON in config` | R4.1 |
| 4 | `winml build -c <wrong-schema>.json -m <m> -o <out-dir>` | schema error **before** Setup banner | R4.1 · BLD-P1-2 |
| 5 | `winml build -c <empty>.json -m <m> -o <out-dir>` | same as 4 | R4.1 · BLD-P1-2 |
| 6 | `winml build -m <m> --device foo -o <out-dir>` | rejected at Click parse time | R1.5 · BLD-P2-3 |
| 7 | `winml build --help \| Select-String 'schema\|fields\|keys'` | non-empty | R1.3 · BLD-P2-4 |
| 8 | `winml build -m <m> -o <existing-dir>` (re-run) | error without `--force` | R3.4 · CC-P1-2 |

---

## `winml compile`

### Success scenarios

| # | Invocation | Status | Notes |
|---|---|---|---|
| 1 | `winml compile -m <onnx> --device npu --ep qnn -o <out>.onnx` | ✅ | Happy path on QNN NPU. |
| 2 | `winml compile -m <onnx> -d cpu --ep cpu -o <out>.onnx` | ✅ | CPU EP path. |
| 3 | First example printed in `winml compile --help` | ✅ | If ❌, that's a P0 (CMP-P0-x). |

### Failure scenarios

| # | Invocation | Expected | Rule |
|---|---|---|---|
| 1 | `winml compile -m <onnx> --device gpyu --ep qnn -o <out>.onnx` | did-you-mean for `gpu`; valid set listed | R1.5 |
| 2 | `winml compile -m <onnx> --device npu --ep qnnn -o <out>.onnx` | valid EP set listed | R1.5 |
| 3 | `winml compile -m <onnx> --device cpu --ep qnn -o <out>.onnx` | reject conflict (exit 2), name both flags | R4.2 |
| 4 | `winml compile -m <onnx> --device gpu --ep qnn -o <out>.onnx` | reject conflict, no silent override | R4.2 |
| 5 | `winml compile -m <bad>.onnx --device npu --ep qnn -o <out>.onnx` | file-not-found at entry, not deep crash | R4.1 |
| 6 | Re-run with same `-o` | error without `--force` | R3.4 · CC-P1-2 |

---

## `winml config`

### Success scenarios

| # | Invocation | Status | Notes |
|---|---|---|---|
| 1 | `winml config -m <m>` | ✅ | Print build config to stdout. |
| 2 | `winml config -m <m> -o <out>.json` | ✅ | Write to file. |
| 3 | `winml config -m <onnx> -o <out>.json` | ✅ | Build config for local ONNX. |
| 4 | `winml config -m <m> --precision int8 -o <out>.json` | ✅ | Precision-specific config. |

### Failure scenarios

| # | Invocation | Expected | Rule |
|---|---|---|---|
| 1 | `winml config` (no args) | error + example | R3.1e |
| 2 | `winml config -m <bogus-id>` | model-not-found, not "Network error" | R3.1a |
| 3 | `winml config -m <m> --precision banana` | valid set listed | R1.5 |
| 4 | `cmd /c "winml config -m <m>"` | no Unicode crash | R3.3h |

---

## `winml eval`

### Success scenarios

| # | Invocation | Status | Notes |
|---|---|---|---|
| 1 | `winml eval -m <m>` | ⚠ EVL-P1-1 | Auto-discovered dataset; `--samples` may be ignored. |
| 2 | `winml eval -m <m> --dataset imagenet-1k --samples 10` | ✅ | Explicit dataset. |
| 3 | `winml eval -m <onnx> --model-id <m> --dataset <ds>` | ✅ | ONNX model with HF metadata. |
| 4 | `winml eval -m <m> --schema --task image-classification` | ✅ | Print expected dataset schema. |

### Failure scenarios

| # | Invocation | Expected | Rule |
|---|---|---|---|
| 1 | `winml eval -m <m> --dataset does-not-exist --samples 3` | one-line `Dataset not found on Hub`; **no traceback frames** | R3.1c · EVL-P0-1 |
| 2 | `winml eval -m <m> --dataset glue --dataset-name mrpc --samples 3` | one-line incompatibility error; **no `KeyError` traceback** | R3.1c · EVL-P0-1 |
| 3 | `winml eval -m <m> --task bogus-task` | valid task set listed | R1.5 · INS-P1-4 |
| 4 | `winml eval --help \| Select-String 'samples'` | default value documented; mismatch with `quantize` is filed | EVL-P2-1 |
| 5 | `winml eval -m <m> --samples 0` | rejected at entry | R4.1 |
| 6 | `winml eval -m a -m b --model-id c` | mutually-exclusive guard | EVL-P2-2 |

---

## `winml export`

### Success scenarios

| # | Invocation | Status | Notes |
|---|---|---|---|
| 1 | `winml export -m microsoft/resnet-50 -o resnet50.onnx` | ✅ | Standard image-model export. |
| 2 | `winml export -m <m> -o <out>.onnx --task image-classification` | ✅ | Explicit task. |
| 3 | `winml export -m <m> -o <out>.onnx --shape-config <shape>.json` | ✅ | Custom input shapes. |
| 4 | First example printed in `winml export --help` | ⚠ EXP-P0-1 | Documented `prajjwal1/bert-tiny` example currently crashes. |

### Failure scenarios

| # | Invocation | Expected | Rule |
|---|---|---|---|
| 1 | `winml export` (no args) | error + example | R3.1e · CC-P1-3 |
| 2 | `winml export -m <m>` (no `-o`) | error or default to safe location, never CWD | R3.4 |
| 3 | `winml export -m <bad-id>` | model-not-found, not "Network error" | R3.1a |
| 4 | `winml export -m <m> -o <readonly>\out.onnx` | writability probe at entry | R4.1 |
| 5 | `winml export -m <m> -o <existing>.onnx` (re-run) | error without `--force` | R3.4 · CC-P1-2 |
| 6 | `winml export -m <m> -o <out>.onnx --shape '{...}'` (PowerShell unquoted) | clean error or working example | R1.3 |

---

## `winml hub`

### Success scenarios

| # | Invocation | Status | Notes |
|---|---|---|---|
| 1 | `winml hub` | ✅ | List curated catalog. |
| 2 | `winml hub -k image-classification` | ✅ | Filter by task. |
| 3 | `winml hub -o catalog.json` | ✅ | Write to file. |
| 4 | `winml hub` piped to `jq` (no `-o`) | ✅ | stdout = data. |

### Failure scenarios

| # | Invocation | Expected | Rule |
|---|---|---|---|
| 1 | `winml hub -k bogus-task` | valid task set listed; non-zero exit | R1.5 |
| 2 | `winml hub -o <existing>.json` (re-run) | error without `--force` | R3.4 · CC-P1-2 |
| 3 | `winml hub -t bogus-type` | grandfathered exception (R2.5 INFO) — no FAIL | R2.5 |

---

## `winml inspect`

### Success scenarios

| # | Invocation | Status | Notes |
|---|---|---|---|
| 1 | `winml inspect -m <m>` | ⚠ INS-P1-1 | Takes ~24 s end-to-end; first-output silent for ~14 s. |
| 2 | `winml inspect -m <onnx>` | ⚠ | "ONNX inspection not yet supported" (see R5.1). |
| 3 | `winml inspect --list-tasks` | ⚠ INS-P1-2 | Takes ~12.6 s for a static dict lookup. |
| 4 | `winml inspect -m <m> --task image-classification` | ✅ | Explicit task. |

### Failure scenarios

| # | Invocation | Expected | Rule |
|---|---|---|---|
| 1 | `winml inspect -m <bogus-id>` | model-not-found, not "Network error" | R3.1a · INS-P0-1 |
| 2 | `winml inspect -m ./does-not-exist.onnx` | file-not-found, not "Network error" | R3.1a · INS-P0-1 |
| 3 | `cmd /c "winml inspect -m <m>"` | no UnicodeEncodeError on cp1252 | R3.3h · INS-P1-3 |
| 4 | `winml inspect -m <m> --task bogus-task` | valid task set listed; no `TasksManager` jargon | R1.5 · INS-P1-4 |

---

## `winml optimize`

### Success scenarios

| # | Invocation | Status | Notes |
|---|---|---|---|
| 1 | `winml optimize -m <onnx> -o <out>.onnx` | ✅ | Default optimization passes. |
| 2 | `winml optimize -m <onnx> --preset basic -o <out>.onnx` | ✅ | Basic preset. |
| 3 | `winml optimize -m <onnx> --preset qnn-compatible -o <out>.onnx` | ❌ OPT-P0-1 | Advertised preset ships broken. |

### Failure scenarios

| # | Invocation | Expected | Rule |
|---|---|---|---|
| 1 | `winml optimize -m <onnx> --preset bogus -o <out>.onnx` | valid set listed | R1.5 |
| 2 | `winml optimize -m <bad>.onnx -o <out>.onnx` | file-not-found at entry | R4.1 |
| 3 | Re-run with same `-o` | error without `--force` | R3.4 · CC-P1-2 |

---

## `winml perf`

### Success scenarios

| # | Invocation | Status | Notes |
|---|---|---|---|
| 1 | `winml perf -m <onnx> --ep qnn -o <out>.json` | ✅ | QNN benchmark, explicit output. |
| 2 | `winml perf -m <m>` | ⚠ PRF-P1-1 | Without `-o`, drops `<slug>_perf.json` in CWD (R3.4 violation). |
| 3 | `winml perf -m <onnx> --ep cpu --iterations 100 -o <out>.json` | ✅ | CPU benchmark. |
| 4 | `winml perf -m <onnx> --ep dml -o <out>.json` | DEFERRED SC-2 | Requires DML hardware. |

### Failure scenarios

| # | Invocation | Expected | Rule |
|---|---|---|---|
| 1 | `winml perf` (no args) | error + example | R3.1e |
| 2 | `winml perf -m <onnx> --device gpu --ep qnn` | reject conflict | R4.2 |
| 3 | `winml perf -m <onnx> --iterations 0` | rejected at entry | R4.1 |
| 4 | `winml perf -m <onnx> --iterations -1` | rejected at entry | R4.1 |
| 5 | `winml perf -m <onnx> --ep <not-installed>` | one-line install hint, exit 3 | R6.2 |
| 6 | Re-run with same `-o` | error without `--force` | R3.4 · CC-P1-2 |

---

## `winml quantize`

### Success scenarios

| # | Invocation | Status | Notes |
|---|---|---|---|
| 1 | `winml quantize -m <onnx> -o <out>.onnx` | ✅ | Default INT8 calibration. |
| 2 | `winml quantize -m <onnx> --weight-type uint8 -o <out>.onnx` | ✅ | Explicit weight type. |
| 3 | `winml quantize -m <onnx> --samples 50 -o <out>.onnx` | ✅ | Custom calibration size (default 10 — see EVL-P2-1). |
| 4 | First example in `winml quantize --help` | ⚠ QNT-P0-x | Document the example actually works. |

### Failure scenarios

| # | Invocation | Expected | Rule |
|---|---|---|---|
| 1 | `winml quantize -m <onnx> --weight-type bogus` | valid set listed | R1.5 |
| 2 | `winml quantize -m <onnx> --samples 0` | rejected at entry | R4.1 |
| 3 | `winml quantize -m <bad>.onnx -o <out>.onnx` | file-not-found at entry | R4.1 |
| 4 | Re-run with same `-o` | error without `--force` | R3.4 · CC-P1-2 |

---

## `winml sys`

### Success scenarios

| # | Invocation | Status | Notes |
|---|---|---|---|
| 1 | `winml sys` | ⚠ SYS-P1-1 | Takes >7 s — should be < 500 ms (R4.3). |
| 2 | `winml sys --format compact` | ✅ | Compact one-screen output. |
| 3 | `winml sys --format json` | ✅ | Pipeable JSON. |
| 4 | `winml sys -v` | ⚠ SYS-P2-1 | DEBUG lines mixed with Rich tables. |

### Failure scenarios

| # | Invocation | Expected | Rule |
|---|---|---|---|
| 1 | `winml sys --format bogus` | valid set listed | R1.5 |
| 2 | `winml sys > out.txt 2> err.txt` | data → out.txt; logs/banners → err.txt | R3.3c |
| 3 | `$env:NO_COLOR='1'; winml sys --format compact > out.txt` | zero ANSI escapes | R3.3d |

---

## Reading order for an audit

1. Read [`quality-checklist.md`](quality-checklist.md) end-to-end (the rules).
2. Read [`quality-check-skill.md`](quality-check-skill.md) (the workflow).
3. Use **this file** as the Phase-2 invocation seed: every "Failure scenarios" row corresponds to a probe the agent must run.
4. Capture verbatim console output in [`CLI_UX_Capture.md`](CLI_UX_Capture.md) (replace the existing one) and reference those captures from new findings in [`CLI_quality_check_report.md`](CLI_quality_check_report.md).
5. Sign off via the Feature-Owner Self-Check appendix at the bottom of the report (SC-1 / SC-2 / SC-3).
