# `winml` CLI Quality Audit

**Audit date**: 2026-05-08
**Test machine**: Snapdragon X Elite ARM64 · Windows 11 · Python 3.10.19 · onnxruntime 1.23.4 (windowsml) · QNN NPU only · PowerShell 7 (UTF-8 stdout) + cmd.exe (cp1252).

Severity legend:

- **P0** — release-blocking. Crash, silent corruption, advertised feature broken, or example in `--help` does not run.
- **P1** — must fix before next minor. Slow (>500 ms warm) local-state command, unsafe defaults, missing safety rails, confusing error.
- **P2** — polish / cross-command consistency.
- **P3** — advisory.

Every issue has a stable ID (`<CMD>-<SEV>-<N>`) so reviewers can refer to issues by ID across iterations. **IDs do not change in update rounds**; they are renumbered only at the end. Each issue carries a *Category* (the kind of problem) and the four standard sub-sections **Repro / Actual / Expected / Why it matters**.

Command prefixes: `TOP` top-level, `ANA` analyze, `BLD` build, `CMP` compile, `CFG` config, `EVL` eval, `EXP` export, `HUB` hub, `INS` inspect, `OPT` optimize, `PRF` perf, `QNT` quantize, `SYS` sys, `CC` cross-cutting.

---

## Summary

| Command | P0 | P1 | P2 | P3 | Total |
|---|---:|---:|---:|---:|---:|
| Top-level (`winml`) | 0 | 2 | 3 | 0 | 5 |
| `winml analyze` | 0 | 3 | 0 | 0 | 3 |
| `winml build` | 0 | 2 | 2 | 0 | 4 |
| `winml compile` | 2 | 1 | 1 | 0 | 4 |
| `winml config` | 1 | 2 | 0 | 0 | 3 |
| `winml eval` | 1 | 1 | 2 | 0 | 4 |
| `winml export` | 2 | 2 | 0 | 0 | 4 |
| `winml hub` | 0 | 2 | 2 | 0 | 4 |
| `winml inspect` | 1 | 3 | 0 | 0 | 4 |
| `winml optimize` | 1 | 1 | 1 | 0 | 3 |
| `winml perf` | 1 | 4 | 0 | 0 | 5 |
| `winml quantize` | 2 | 0 | 1 | 0 | 3 |
| `winml sys` | 0 | 2 | 1 | 0 | 3 |
| Cross-cutting | 0 | 2 | 6 | 1 | 9 |
| **Total** | **11** | **27** | **19** | **1** | **58** |

Out of 6 high-traffic commands (`export`, `inspect`, `compile`, `quantize`, `perf`, `sys`), every one ships at least one P0 or P1 issue. Two `--help` examples are non-runnable. Three commands silently override or ignore user input and return success. Three local-state commands take >7 s when they should take <500 ms. The wheel is 144 MB.

---

## Navigation

- [Top-level (`winml`)](#top-level-winml)
- [`winml analyze`](#winml-analyze)
- [`winml build`](#winml-build)
- [`winml compile`](#winml-compile)
- [`winml config`](#winml-config)
- [`winml eval`](#winml-eval)
- [`winml export`](#winml-export)
- [`winml hub`](#winml-hub)
- [`winml inspect`](#winml-inspect)
- [`winml optimize`](#winml-optimize)
- [`winml perf`](#winml-perf)
- [`winml quantize`](#winml-quantize)
- [`winml sys`](#winml-sys)
- [Cross-cutting](#cross-cutting)
- [Feature-Owner Self-Check](#feature-owner-self-check)
---

## Top-level (`winml`)

### TOP-P1-1 — `winml exprt` (typo) returns "No such command" with no suggestion

**Category**: Did-you-mean — typos on subcommand names should suggest the closest valid name.

**Repro**:
```
uv run winml exprt
```

**Actual**:
```
Usage: winml [OPTIONS] COMMAND [ARGS]...
Try 'winml --help' for help.

Error: No such command 'exprt'.
```
Exit 2.

**Expected**: Suggest the closest match, the way `git`, `gh`, `cargo`, and `kubectl` do.
```
Error: No such command 'exprt'. Did you mean 'export'?
```

**Why it matters**: First-time users will mistype subcommand names. A bare "no such command" forces them back to `winml --help`; a one-line suggestion keeps them moving. Implementation is a one-method override of `LazyGroup.resolve_command` using `difflib.get_close_matches`.

---

### TOP-P1-2 — Bare `sys.exit(N)` calls in command modules; no documented exit-code contract

**Category**: Exit-Code Contract — exit codes must be documented and consistent so CI/scripts can branch on them.

**Repro**: Run several failure modes and observe the exit codes.
```
uv run winml export                                # missing required option
uv run winml inspect -m bogus/x                    # bogus model id
uv run winml exprt                                 # typo
uv run winml --version                             # success
uv run winml perf -m model.onnx --module NoSuchClass --iterations 3 --warmup 1
uv run winml quantize -m model.onnx --precision banana -o bad.onnx
```

**Actual**: 4 distinct exit codes observed across the CLI with no documented meaning:
- `winml export` (no args) → **2** (Click usage error).
- `winml inspect -m bogus/x` → **1** (generic).
- `winml exprt` → **2**.
- `winml --version` → **0**.
- `winml perf --module NoSuchClass` → **3** (undocumented; only path that emits 3).
- `winml quantize --precision banana` → **1** but stdout reads `Success! Model quantized` (contradiction).

**Expected**: A documented exit-code table in `CONTRIBUTING.md` and `winml --help`, e.g.:
```
0  success
1  negative result (e.g. eval below threshold)
2  usage error (bad flag, conflicting flags, missing input)
3  hardware/EP unavailable
4  network / Hub failure
5  I/O failure
70+ internal error
```
Replace every `sys.exit(N)` in command modules with `raise click.ClickException(...)` or a typed `ModelKitError`. Never combine `Success!` stdout with non-zero exit.

**Why it matters**: CI pipelines and shell scripts use exit codes to decide whether to publish, retry, alert, or block a release. With four distinct codes and no contract, every consumer has to read the source.

---

### TOP-P2-3 — Multiple brand spellings in use (`WML`, `WinML`, `winml`, `Windows ML`, `ModelKit`)

**Category**: Naming Consistency — pick one canonical product name and use it everywhere.

**Repro**:
```
uv run winml --help
uv run winml --version
uv run winml export --help
```

**Actual**: Five different forms appear in user-visible output:
- `winml --help` first line: `WML ModelKit - Accelerate Model Deployment on WinML.` (uses **WML** and **WinML** in the same sentence).
- `winml --version` prints: `winml, version 0.0.2` (lowercase **winml**).
- `winml export --help` first line: `Export HuggingFace model to ONNX format with HTP.` (no brand).
- `winml inspect` panel header reads `WinML Inference Class`.
- `winml sys` table headers read `Windows ML` for the OS service and `winml` for the CLI.

**Expected**: Pick one canonical capitalization (e.g. **WinML ModelKit** for the product, `winml` only as the literal command name) and use it in every place a user can read it: top-level docstring, every subcommand docstring, `--version` output, every panel/table title, `README.md`, and `pyproject.toml`. Concretely:
- Replace `WML ModelKit` → `WinML ModelKit` in the top-level docstring.
- Make `winml --version` print `WinML ModelKit, version 0.0.2 (winml CLI)`.

**Why it matters**: Users searching docs/issues for the product name need one term. Marketing/legal usually require one canonical capitalization too.

---

### TOP-P2-4 — Subcommand descriptions in `winml --help` are truncated mid-word

**Category**: Unicode/Output Rendering — table column widths must respect terminal width without hiding meaning.

**Repro**:
```
uv run winml --help
```

**Actual** (from the captured 80-column run):
```
Commands:
  analyze       Analyze ONNX model for runtime support with live progress.
  build         Build a WinML-optimized ONNX model from a HuggingFace model
  compile       Compile ONNX model to EP-specific format.
  config        Generate WinMLBuildConfig for a HuggingFace model or .onnx f
  eval          Evaluate model accuracy on a dataset.
  expand_rules  Expand runtime rules zip files in-place when directories and
  export        Export HuggingFace model to ONNX format with HTP.
  hub           Browse ModelKit's curated built-in model catalog.
```
`config` is cut off at `.onnx f` (intended: `…or .onnx file`). `expand_rules` is cut off at `directories and` (intended: `…directories and zips coexist`). `build` ends at `from a HuggingFace model` with no period.

**Expected**: Either trim each first-line description so it fits in 80 cols without `…`, or wrap to the next line with continued indent (Click supports both). No mid-word truncation.

**Why it matters**: The first-line subcommand summary is the single most-read line of help. Truncation here gives users a wrong or incomplete idea of what each subcommand does and undermines the rest of the documentation.

---

### TOP-P2-5 — `winml --version` prints only `winml, version 0.0.2`; no provenance

**Category**: Version & Provenance — `--version` should let a bug report identify the build.

**Repro**:
```
uv run winml --version
```

**Actual**:
```
winml, version 0.0.2
```
No Python version, no `onnxruntime` build (`windowsml` vs upstream), no `transformers` / `optimum` / `torch` versions, no commit hash, no platform.

**Expected**: Print a multi-line provenance block, e.g.:
```
winml ModelKit 0.0.2 (commit a1b2c3d, built 2026-04-30)
  python   3.10.19  (CPython, win-arm64)
  torch    2.11.0
  onnxruntime 1.23.4 (windowsml)
  transformers 4.57.6 | optimum 1.x | onnx 1.18.0
  EPs available: cpu, qnn
```
The same block should be printed at the bottom of `winml sys --format compact` and at the top of every `--verbose` run, so a bug report inevitably contains it.

**Why it matters**: `winml, version 0.0.2` is useless for triage. Reporters omit Python/ORT/EP info because they don't know which build they're on; maintainers can't reproduce.

---

## `winml analyze`

### ANA-P1-1 — `--ep cpu` silently changes target to NPU then exits 1

**Category**: Conflicting Flag Combinations — when flags conflict, reject at entry; never silently override.

**Repro**:
```
uv run winml analyze -m temp\cli-audit\resnet.onnx --ep cpu
```

**Actual**:
```
Target: CPUExecutionProvider on NPU
WARNING: No runtime check data for CPUExecutionProvider on NPU — skipping op analysis.
No runtime check results for CPUExecutionProvider on NPU — no rule data available.
```
Exit 1. The user passed `--ep cpu` and did **not** pass `--device`, but the target line says `on NPU`. Then the rule lookup fails because the (CPU, NPU) pair is impossible, and the whole command exits 1.

**Expected**: When `--ep cpu` is given without `--device`, default `--device cpu`. When `--ep cpu --device npu` is given explicitly, reject at entry:
```
Error: --ep cpu cannot run on --device npu. CPUExecutionProvider runs on CPU.
```
Exit 2.

**Why it matters**: Silent override means the command appears to "do something" but produces no result. The user has no way to know their flag was ignored.

---

### ANA-P1-2 — Lacks `-o` short alias; only `--output` works (drift)

**Category**: Cross-Command Flag Consistency — same flag should have same form (long + short) across commands.

**Repro**:
```
uv run winml analyze -m temp\cli-audit\resnet.onnx --ep qnn -o temp\ux\out_analyze.json
```

**Actual**:
```
Usage: winml analyze [OPTIONS]
Try 'winml analyze --help' for help.

Error: No such option: -o
```
Exit 2.

**Expected**: Both `-o` and `--output` should work, the same way they do for `winml export`, `winml optimize`, `winml quantize`, `winml compile`, `winml hub`, `winml perf`, `winml config`, `winml eval`. Add the `-o` alias.

**Why it matters**: One subcommand using a different flag name forces users to memorize an exception. Most users will type `-o` from muscle memory and get an "unknown option" error.

---

### ANA-P1-3 — Box-drawing renders as literal `\u2550…\U0001f4ca` under cp1252

**Category**: Unicode/Output Rendering — output must survive non-UTF-8 stdout (cmd.exe, redirected pipes, `tee` to files).

**Repro**:
```
cmd /c "uv run winml analyze -m temp\cli-audit\resnet.onnx --ep qnn"
```
(Default cmd.exe stdout codepage on Windows is cp1252.)

**Actual**: stdout literally contains the escape sequences:
```
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550...
\U0001f4ca ANALYSIS SUMMARY
\u2550\u2550...
```
The Python `errors='backslashreplace'` codec fallback prints the escape codes instead of the box-drawing and chart emoji.

**Expected**: Detect non-UTF-8 stdout (`sys.stdout.encoding != "utf-8"`) and either (a) substitute ASCII art (`---` for `•`, `[chart]` for `📊`), or (b) reopen stdout in UTF-8 at startup with `sys.stdout.reconfigure(encoding="utf-8")`. Never let `backslashreplace` reach the user.

**Why it matters**: Many enterprise users still use cmd.exe (or redirect output to a log file). For them every emoji becomes a `\Uxxxxxxxx` literal, making the output unreadable. PowerShell 7's UTF-8 default hides this.

---

## `winml build`

### BLD-P1-1 — `--trust-remote-code` runs custom code with no warning

**Category**: Security Flag Warning — flags that loosen security must emit a warning to stderr before doing the loosened operation.

**Repro**:
```
uv run winml build --trust-remote-code -m microsoft/resnet-50 -c temp\ux\build_cfg.json -o temp\ux\trc_out
```

**Actual**: stderr contains only the normal Setup banner and download progress; **no** warning about `--trust-remote-code` is emitted before the model is downloaded and its custom Python is executed. Filtering stderr for `WARNING|trust` returns zero matches.

**Expected**: Emit a warning to stderr **before** any download starts, not suppressed by `-q/--quiet`:
```
WARNING: --trust-remote-code allows running custom Python from microsoft/resnet-50.
         Proceed only if you trust the publisher.
```
Same warning should fire for `winml config --trust-remote-code` and `winml inspect --trust-remote-code`.

**Why it matters**: `--trust-remote-code` causes `transformers` to import and execute arbitrary Python from the model repo. Without a visible warning at the call site, this becomes an attack vector: a user copy-pasting a `winml` command from a blog post may not realize they just authorized arbitrary code execution.

---

### BLD-P1-2 — Config-schema validation fires *after* printing Setup banner and Stages header

**Category**: Fail Fast — validation of all inputs (flags, files, config schema) must complete before any user-visible work or banner.

**Repro**:
```
'{ "this": "is not a valid build config" }' | Out-File temp\ux\malformed_cfg.json -Encoding ascii
uv run winml build -c temp\ux\malformed_cfg.json -m microsoft/resnet-50 -o temp\ux\bo2
```

**Actual**:
```
════════════════════════════════════════════════════════════
🔧 Setup — HuggingFace
════════════════════════════════════════════════════════════
   📦 Model:     microsoft/resnet-50  (pretrained)
   📁 Config:    malformed_cfg.json
   📂 Output:    temp\ux\bo2

════════════════════════════════════════════════════════════
🎯 Stages
════════════════════════════════════════════════════════════
Usage: winml build [OPTIONS]
Try 'winml build --help' for help.

Error: Config validation failed: Invalid WinMLBuildConfig:
  - loader.task is required for full model builds
```
The Setup banner and Stages header are printed *before* the config is validated. The same happens with an empty `{}` config. (Truncated/invalid JSON does fail fast — that path runs through Click's `File(... json)` decoder.)

**Expected**: Validate the config object against the `WinMLBuildConfig` schema as the first action inside the command callback (right after Click parses arguments), before printing any banner. On failure, exit with code 2 and the same error.

**Why it matters**: The banner implies work is starting. Users who passed the wrong file path or hand-edited a config will assume the build started and may wait. Worse, when validation runs late, partial scratch directories (`temp\ux\bo2`) may already have been created and need manual cleanup.

---

### BLD-P2-3 — `--device` flag is shaped differently in every command that uses it

**Category**: Default Value Consistency — same flag should have same `Choice`, same default, and same casing across commands.

**Repro**:
```
uv run winml build --help    | Select-String '\-\-device' -Context 0,2
uv run winml compile --help  | Select-String '\-\-device' -Context 0,2
uv run winml eval --help     | Select-String '\-\-device' -Context 0,2
uv run winml perf --help     | Select-String '\-\-device' -Context 0,2
```

**Actual**: Four different shapes:

| Command | Click `Choice` | Default | Help-text example |
|---|---|---|---|
| `build` | none (free string) | `None` | `NPU` (uppercase) |
| `compile` | `[auto, npu, gpu, cpu]` | `auto` | `auto` (lowercase) |
| `eval` | `[auto, cpu, gpu, npu]` | `auto` | `auto` (different ordering) |
| `perf` | `[auto, cpu, gpu, npu]` | `auto` | `auto` |

So `winml build --device npu` and `winml build --device NPU` both work; `winml compile --device NPU` errors with a `Choice` mismatch. `winml build --device foo` accepts `foo` and fails later.

**Expected**: Define one shared `device_option` decorator in a common module. All commands that take `--device` import and apply it. One canonical `Choice([auto, cpu, gpu, npu])`, default `auto`, lowercase only.

**Why it matters**: Inconsistency forces users to reverse-engineer each command. The `build` command's free-string acceptance is the worst case: it accepts a typo silently and fails deep inside the build, possibly after expensive work.

---

### BLD-P2-4 — `--config` accepts a JSON file but `--help` documents no schema

**Category**: Help & Examples — every flag that takes a structured input must point the user at the schema.

**Repro**:
```
uv run winml build --help | Select-String 'config|schema|json|fields|keys'
```

**Actual**: The only mention of the file format is one line — `WinMLBuildConfig JSON file (from winml config)`. No list of required keys, no link to a schema file, no example shown inline, no `--print-schema` flag. Users only learn the required keys (e.g. `loader.task`) by submitting an invalid config and reading the validation error. The same pattern applies to every other command that takes `-c/--config` (`export`, `optimize`, `compile`, `quantize`, `perf`, `eval`) — none documents the schema in `--help`.

**Expected**:
1. Add a one-paragraph schema summary to each command's `--help` epilog: `"Config keys: loader.task (required), exporter.opset, optimizer.preset, quantizer.calibration_samples. Full schema: docs/config-schema.md."`
2. Add a `winml config --schema` (or `winml config --print-schema`) subcommand that prints the JSON Schema to stdout, so it can be used with editor validation (`"$schema": ...`).
3. Reference both from each affected command's `--help`.

**Why it matters**: Cross-cutting documentation gap. Today the only path to discover required keys is trial-and-error — which is especially painful given BLD-P1-2 (validation runs late). New users cannot author a valid config from `--help` alone.

---

## `winml compile`

### CMP-P0-1 — `--device cpu --ep qnn` silently overrides device to NPU and exits 0

**Category**: Conflicting Flag Combinations — when flags conflict, reject at entry; never silently override.

**Repro**:
```
uv run winml compile -m temp\cli-audit\resnet.onnx --device cpu --ep qnn -o temp\ux\out_cpu_qnn.onnx
```

**Actual** (also reproduces with `--device gpu --ep qnn`):
```
Input:    temp\cli-audit\resnet.onnx
Device:   npu          ← user passed cpu
EP:       qnn
Success! Model compiled
```
Exit 0. A 51 MB `out_cpu_qnn.onnx` is written and the user has no idea it was compiled for NPU.

**Expected**: At entry:
```python
if device == "cpu" and ep == "qnn":
    raise click.ClickException(
        "--device cpu and --ep qnn are incompatible. QNN runs on NPU only. "
        "Either pass --device npu, or use --ep CPUExecutionProvider for CPU."
    )
```
Exit 2.

**Why it matters**: Silent device override produces an artifact that does not match what the user asked for. If the user then ships that artifact thinking it runs on CPU, deployment fails in production.

---

### CMP-P0-2 — `--device cpu --ep cpu` raises raw `AttributeError` traceback

**Category**: Error Messages — no third-party class names or tracebacks at default verbosity.

**Repro**:
```
uv run winml compile -m temp\cli-audit\resnet.onnx --device cpu --ep cpu -o temp\ux\out_cpu_cpu.onnx
```

**Actual**:
```
Traceback (most recent call last):
  ...
  File "...\src\winml\modelkit\commands\compile.py", line 204, in compile
    config.validate = validate
AttributeError: 'NoneType' object has no attribute 'validate'
```
Exit 1.

**Expected**: Either reject `cpu+cpu` at entry with a clear message ("compile-to-CPU is not supported; use `winml export` to produce a CPU ONNX model"), or implement it. Either way: no traceback at default verbosity. Tracebacks only with `-vv`.

**Why it matters**: Tracebacks expose internal class names (`NoneType`, line numbers in compile.py) and force users to file an issue rather than self-correct.

---

### CMP-P1-1 — Leaves 51 MB `*_qnn.bin` sidecar files per run, undocumented

**Category**: Output File Safety — every file the command creates must be documented in `--help` and the success message.

**Repro**:
```
uv run winml compile -m temp\cli-audit\resnet.onnx -o temp\ux\c1.onnx
uv run winml compile -m temp\cli-audit\resnet.onnx -o temp\ux\c2.onnx
uv run winml compile -m temp\cli-audit\resnet.onnx -o temp\ux\c3.onnx
Get-ChildItem temp\ux\c*.* | Format-Table Name, Length
```

**Actual**:
```
Name        Length
----        ------
c1.onnx     384
c1_qnn.bin  51,761,152
c2.onnx     384
c2_qnn.bin  51,761,152
c3.onnx     384
c3_qnn.bin  51,761,152
```
Total ~150 MB written, none of which is mentioned in `--help`, the success message, or the printed `Output:` line. The `.onnx` is just a stub; the real compiled blob is `_qnn.bin`.

**Expected**: One of:
1. Document the sidecar in `--help` and print it on success: `Output: c1.onnx + c1_qnn.bin (51.7 MB)`.
2. Write the sidecar inside a directory next to the ONNX: `<output>.bin/`.
3. Fold the binary into the ONNX as `external_data`.

In all cases, the success message must list every file the command wrote.

**Why it matters**: Users assume `-o c1.onnx` means "one file at c1.onnx". They later wonder why their disk is full, or — worse — they ship `c1.onnx` without `c1_qnn.bin` and the model fails to load on the target machine.

---

### CMP-P2-1 — `Input:` line appears with default flags but disappears with explicit ones

**Category**: Unicode/Output Rendering — same field set should appear regardless of which flags were defaulted.

**Repro**:
```
uv run winml compile -m temp\cli-audit\resnet.onnx -o temp\ux\c1.onnx
uv run winml compile -m temp\cli-audit\resnet.onnx --device npu --ep qnn -o temp\ux\c2.onnx
```

**Actual**:

Default flags (first invocation) prints:
```
Input:    temp\cli-audit\resnet.onnx
Device:   npu
EP:       qnn
Success! Model compiled
```

Explicit flags (second invocation) prints:
```
Device:   npu
EP:       qnn
Success! Model compiled
```
The `Input:` line is **omitted** when `--device` and `--ep` are passed explicitly.

**Expected**: Same field block in both cases. `Input:` should always appear (it's the most important context line — it tells the user what was compiled).

**Why it matters**: Users grep build logs for `Input:` to confirm which model was compiled. When the line vanishes, automation breaks.

---

## `winml config`

### CFG-P0-1 — `--precision int8` silently produces `uint8/uint8`

**Category**: Conflicting Flag Combinations — reject unsupported values at entry; never substitute silently.

**Repro**:
```
uv run winml config -m microsoft/resnet-50 --device npu --precision int8 -o temp\ux\cfg_int8.json
```

**Actual**:
```
Quant: uint8/uint8  (weight/activation)
Config saved to: temp\ux\cfg_int8.json
```
Exit 0. `cfg_int8.json` contains `"weight_type": "uint8", "activation_type": "uint8"`. The user asked for `int8`; the config silently records `uint8`.

**Expected**: Declare `--precision` as `click.Choice([auto, fp32, fp16, int8, int16, w4a16, w8a8, w8a16])`. Either honor `int8`, or reject:
```
Error: --precision int8 not supported on --device npu. Supported: auto, fp16, w8a8, w8a16.
```
Exit 2.

**Why it matters**: A config silently set to a different precision than asked will produce a different model than the user expects, and they'll spend hours debugging a "regression" that's actually a flag override.

---

### CFG-P1-1 — Bogus HF model id reported as "Network error"

**Category**: Error Messages — three distinct conditions need three distinct messages.

**Repro**:
```
uv run winml config -m bogus/x -o temp\ux\cfg_bad.json
```

**Actual**:
```
Error: Unexpected error: bogus/x is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo …
```
Exit 1.

**Expected**: Three distinct messages for three distinct conditions:
- Local file/dir missing → `Error: Local path 'bogus/x' does not exist.`
- Valid HF id format but 404 → `Error: Model 'bogus/x' not found on Hugging Face Hub. Check spelling, or pass a token if private.`
- Actual network outage → `Error: Network error fetching 'bogus/x' (DNS / TLS / timeout).`

The "private repository" hint should fire only when the response is HTTP 401/403, not unconditionally.

**Why it matters**: When the user sees "private repository" on a public-id typo, they assume their token is broken and waste time troubleshooting auth instead of fixing the typo.

---

### CFG-P1-2 — Backslash-replace fallback prints `\U0001f9e9` literals when stdout codec can't encode emoji

**Category**: Unicode/Output Rendering — output must survive non-UTF-8 stdout.

**Repro**:
```
cmd /c "uv run winml config -m microsoft/resnet-50 --device npu --precision int8 -o temp\ux\cfg_int8.json"
```

**Actual**: stdout literally contains:
```
\U0001f9e9 Model class:  AutoModelForImageClassification
\U0001f3f7\ufe0f  Task:        image-classification
\u2699\ufe0f  Resolution:   224x224
\u2705 Config saved to: temp\ux\cfg_int8.json
```

Same pattern reproduces in `analyze`, `quantize`, `build`, `compile`, `optimize`.

**Expected**: Detect `sys.stdout.encoding != "utf-8"` at startup and either substitute ASCII (e.g. `[+]` for `✅`, `[i]` for `🧩`) or reopen stdout in UTF-8. Never let `backslashreplace` reach the user.

**Why it matters**: Same as ANA-P1-3 — cmd.exe and redirected output are common in enterprise/CI; the fallback turns the entire UI into unreadable escape sequences.

---

## `winml eval`

### EVL-P0-1 — Bogus or mismatched `--dataset` leaks raw multi-frame `datasets`/internal traceback

**Category**: Error Messages — never leak third-party traceback frames at default verbosity.

**Repro 1 — dataset that doesn't exist on the Hub**:
```
uv run winml eval -m microsoft/resnet-50 --dataset does-not-exist --samples 3
```

**Actual** (default verbosity, last 10 lines):
```
  File "...\datasets\load.py", line 1393, in dataset_module_factory
    raise DatasetNotFoundError(...) from e
datasets.exceptions.DatasetNotFoundError: Dataset 'does-not-exist' doesn't exist on the Hub or cannot be accessed.
Error: Evaluation failed: Dataset 'does-not-exist' doesn't exist on the Hub or cannot be accessed.
```
Four frames of `datasets/load.py` internals are printed before the friendly error.

**Repro 2 — valid but task-incompatible dataset**:
```
uv run winml eval -m microsoft/resnet-50 --dataset glue --dataset-name mrpc --samples 3
```

**Actual**: ~18 frames of traceback through `datasets/arrow_dataset.py`, `winml/modelkit/eval/base_evaluator.py`, ending in:
```
KeyError: 'equivalent'
...
RuntimeError: Label alignment failed for dataset 'glue': 'equivalent'
Error: Evaluation failed: Label alignment failed for dataset 'glue': 'equivalent'
```
No guidance to the user about *why* `glue/mrpc` is incompatible with an image-classification model, or how to find a compatible dataset.

**Expected**:
1. Wrap dataset loading in a guard that converts `DatasetNotFoundError` into a clean Click error with no traceback at default verbosity:
   ```
   Error: Dataset 'does-not-exist' was not found on the Hugging Face Hub.
          Check the spelling, or run 'winml eval --help' for the auto-discovered defaults.
          Use -v to see the underlying loader exception.
   ```
2. Validate dataset/task compatibility *before* iterating samples. If the model task is `image-classification` and the dataset has no image column or has incompatible labels, fail with:
   ```
   Error: Dataset 'glue/mrpc' is not compatible with task 'image-classification'.
          Expected an image column and integer labels; found columns: sentence1, sentence2, label (string).
          Run 'winml eval --schema --task image-classification' for compatible dataset shape.
   ```
3. Only show the raw traceback when `-v` is passed.

**Why it matters**: Eval is a high-traffic command and `--dataset` is the flag users are most likely to typo or misuse. Today the user sees pages of traceback through unfamiliar libraries (`datasets`, `arrow`) and has no clue whether the bug is in `winml`, `datasets`, or their own input. This is the single biggest UX cliff in eval.

---

### EVL-P1-1 — `--samples N` silently ignored; output reports the wrong sample count

**Category**: Conflicting Flag Combinations — never silently ignore a user-supplied flag.

**Repro**:
```
uv run winml eval -m microsoft/resnet-50 --samples 5
```

**Actual** (~31 s warm): Output reports the default sample count, not 5. Stats are computed from the default count. The `--samples` flag is parsed by Click but never plumbed to the eval loop for the auto-discovered dataset path.

**Expected**: Either honor `--samples` (preferred), or reject the combination at entry with a clear message: `Error: --samples is only supported with --dataset; the auto-discovered eval set uses a fixed split.`

**Why it matters**: `--samples` is the main knob for shortening eval during development. Silently ignoring it means dev iterations take 10× longer than the user expects, and the reported metrics don't match what the user thought they ran.

---

### EVL-P2-1 — `--samples` default differs from `quantize` (eval=100, quantize=10)

**Category**: Default Value Consistency — same flag should have same default across commands.

**Repro**:
```
uv run winml eval --help     | Select-String '\-\-samples'
uv run winml quantize --help | Select-String '\-\-samples'
```

**Actual**:
- `eval`: `--samples INTEGER  Number of samples to evaluate. [default: 100]`
- `quantize`: `--samples INTEGER  Number of calibration samples [default: 10]`

**Expected**: Both flags do conceptually the same thing (pull N rows from a dataset). Pick one default and apply everywhere, or rename one of them so the difference is explicit (e.g. `--calibration-samples` for quantize).

**Why it matters**: Users learn one flag and assume it behaves the same. Different defaults silently change runtime by 10×.

---

### EVL-P2-2 — Confusing dual `-m/--model` (`multiple=True`) + `--model-id`; no mutual-exclusion check; no role-key list in `--help`

**Category**: Canonical Flag Names — one concept, one flag.

**Repro**:
```
uv run winml eval --help | Select-String '\-\-model|\-\-model-id'
```

**Actual**: `--help` shows two model-input flags:
- `-m, --model TEXT  HuggingFace model id or path (can be passed multiple times with role keys e.g. -m generator=foo -m discriminator=bar)`
- `--model-id TEXT   Hugging Face model id` (single).

The role-key syntax (`generator=foo`) is mentioned but the list of valid role keys is **not** in `--help`. Passing both `-m` and `--model-id` is not flagged as conflicting.

**Expected**: One canonical input flag (`-m/--model`, `multiple=True`). Document the role keys in `--help` (`Valid roles: generator, discriminator, embedder, …`). Drop `--model-id` (or alias it to `-m`).

**Why it matters**: Three ways to spell "the model to evaluate" and zero documentation of the role keys forces users to read source.

---

## `winml export`

### EXP-P0-1 — Documented basic example (`prajjwal1/bert-tiny`) crashes with raw 6-frame traceback

**Category**: Help & Examples — every example in `--help` must run.

**Repro**: This is the *first* example in `winml export --help`:
```
uv run winml export -m prajjwal1/bert-tiny -o temp\ux\bt.onnx
```

**Actual** (~14 s wait, then):
```
Traceback (most recent call last):
  File "...\task.py", line 128, in resolve_task
  File "...\task.py", line 245, in _detect_task
  File "...\task.py", line 393, in _detect_task_from_config
  File "...\hf.py", line 223, in _load_config
  File "...\hf.py", line 229, in _load_config
  File "...\export.py", line 365, in export
ValueError: Cannot resolve task/model for prajjwal1/bert-tiny.
Original error: Cannot detect task: config has no 'architectures' field.
Please specify task explicitly.

Error: Export failed: Cannot resolve task/model for prajjwal1/bert-tiny. ...
```
Note also the double `Error:` prefix (the ValueError message starts with `Cannot resolve…` and the Click wrapper prepends `Error: Export failed: …` containing the same text). Exit 1. Compare: `winml inspect -m prajjwal1/bert-tiny` succeeds — the same model loads fine for `inspect`.

**Expected**: Either (a) the example works out of the box (auto-fall-back to `--task feature-extraction` when `architectures` is missing from config), or (b) `--help` cites a model that does work (e.g. `microsoft/resnet-50`). Hide tracebacks behind `-vv`. Add a regression test that runs every example in every `--help`.

**Why it matters**: The first command a user copy-pastes from `--help` should succeed. When the canonical example crashes, every "follow the README" tutorial breaks and users assume the entire tool is broken.

---

### EXP-P0-2 — `--dynamo` and `--torch-module` advertised; both fail on the documented example model

**Category**: Advertised Flags — every flag in `--help` must work.

**Repro**:
```
uv run winml export -m prajjwal1/bert-tiny -o temp\ux\bt_dyn.onnx --dynamo
uv run winml export -m prajjwal1/bert-tiny -o temp\ux\bt_lm.onnx --torch-module LayerNorm
```

**Actual** (each ~13–14 s):
```
ValueError: Cannot resolve task/model for prajjwal1/bert-tiny ...
```
Same failure path as EXP-P0-1.

**Expected**: Examples must use a model that exports successfully. Add a regression test that runs every advertised flag against the example model in every `--help`. Mark `hidden=True` on flags that don't work yet.

**Why it matters**: Same as EXP-P0-1 — these flags are the headline features in the help; they must work on the example.

---

### EXP-P1-1 — `UnicodeEncodeError` on rocket emoji `\U0001f680` under cp1252

**Category**: Unicode/Output Rendering — output must survive non-UTF-8 stdout.

**Repro**:
```
cmd /c "uv run winml export -m microsoft/resnet-50 -o temp\ux\rn.onnx"
```

**Actual**: stderr contains:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f680' in position 0: character maps to <undefined>
```
Crash before the export starts. The triggering line is `console.print("🚀 Starting export …")`.

**Expected**: ASCII fallback (`>> Starting export …`) when `sys.stdout.encoding != "utf-8"`, or reopen stdout in UTF-8 at startup with `sys.stdout.reconfigure(encoding="utf-8")`.

**Why it matters**: For cmd.exe users on Windows (common in enterprise), the command crashes before doing any work.

---

### EXP-P1-2 — Library `TracerWarning` from torch leaks at default verbosity

**Category**: Quiet at Default — library WARNING noise should be suppressed unless `-v`.

**Repro**:
```
uv run winml export -m microsoft/resnet-50 -o temp\ux\rn.onnx
```

**Actual**: stderr contains lines like:
```
C:\...\modeling_resnet.py:72: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect ...
```
Multiple TracerWarning lines from torch leak to default-verbosity output.

**Expected**: At default verbosity, suppress `torch.jit.TracerWarning` and other library warnings. Show them only when `-v`.

**Why it matters**: Users see scary `WARNING` lines for benign torch internals and assume their export is broken. A clean default output is a quality signal.

---

## `winml hub`

### HUB-P1-1 — Default invocation takes 7 s for a static curated table

**Category**: Local Command Latency — informational queries should return in <500 ms warm.

**Repro**:
```
Measure-Command { uv run winml hub --task image-classification }
```

**Actual**: ~7.1 s warm, ~8.1 s on a freshly-warmed machine. Output is a static curated table.

**Expected**: <500 ms. The catalog should be a JSON file loaded lazily, not built by importing every model loader / EP probe at import time.

**Why it matters**: `winml hub` is the discovery entry point — users will run it dozens of times to browse models. Seven seconds per invocation feels broken.

---

### HUB-P1-2 — `--model <id>` displays static OV / VitisAI latency on a machine with neither EP installed

**Category**: Single Source of Truth — one capability check; never display canned data alongside live data without a label.

**Repro** (on QNN-only Snapdragon X with no OpenVINO and no Vitis AI installed):
```
uv run winml hub --model microsoft/resnet-50
```

**Actual**:
```
Latency (ms)
EP        Avg   P50   P90   P95   P99   Min   Max  QPS
QNN      1.22  1.14  1.94  1.94  1.94  1.09  1.94  823
OV       3.96  4.06  4.13  4.13  4.13  3.60  4.13  252      ← OV not installed
VitisAI  2.32  2.33  2.41  2.41  2.41  2.28  2.41  431      ← VitisAI not installed
```

These OV / VitisAI numbers are static catalog entries from the curated catalog — but the table renders them identically to the live QNN measurement, with no provenance label. `winml sys --format compact` on the same machine reports `OpenVINO: N/A`.

**Expected**: Either (a) hide rows whose EP isn't installed on the host, or (b) label the table source explicitly: `Source: ModelKit catalog (last updated 2026-04-15)` and add a column distinguishing `live` from `catalog`.

**Why it matters**: Users assume those latency numbers came from their hardware. They'll cite them in design reviews ("OV is 3× slower than QNN") not knowing both are static.

---

### HUB-P2-1 — Table truncates model and task names mid-word

**Category**: Unicode/Output Rendering — terminal-width-aware sizing required; no silent mid-word truncation of identifiers.

**Repro**:
```
uv run winml hub
```

**Actual**: Cells truncated mid-identifier:
```
microsoft/swin-large-patch4-windo
dbmdz/bert-large-cased-finetuned-
image-classifica
feature-extracti
```

**Expected**: Use Rich `overflow="ellipsis"` so truncation is visible (`microsoft/swin-large-patch4-window…`), offer `--full` for non-truncated output, and never silently truncate model identifiers (a truncated id is unusable — you can't copy-paste it).

**Why it matters**: Users will copy-paste truncated model ids and get "model not found" errors.

---

### HUB-P2-2 — `-t` short flag means `--model-type` here, but means `--task` in `inspect` / `export` / `config`

**Category**: Reserved Short Flags — short flags must mean the same thing across commands.

**Repro**:
```
uv run winml hub --help     | Select-String '^\s+-[tk],'
uv run winml inspect --help | Select-String '^\s+-t,'
uv run winml export --help  | Select-String '^\s+-t,'
uv run winml config --help  | Select-String '^\s+-t,'
```

**Actual**:
- `winml hub`: `-t/--model-type` and `-k/--task`.
- `winml inspect`, `winml export`, `winml config`: `-t/--task`.

**Expected**: `-t` should mean `--task` in `winml hub` too. Use a different short letter (or no short) for `--model-type`. The fix lives in `commands/hub.py`.

**Why it matters**: A user who has memorized `-t` to mean `--task` in 3 commands will type `-t image-classification` against `winml hub` and silently get `--model-type=image-classification` (no such model type) or a confusing error.

---

## `winml inspect`

### INS-P0-1 — Bogus HF id and missing local file both reported as "Network error"

**Category**: Error Messages — three distinct conditions need three distinct messages.

**Repro**:
```
uv run winml inspect -m totally-bogus/does-not-exist
uv run winml inspect -m .\does-not-exist.onnx
```

**Actual**:
```
Error: Network error: totally-bogus/does-not-exist is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token …

Error: Network error: Can't load the configuration of '.\does-not-exist.onnx' …
If you were trying to load it from 'https://huggingface.co/models' …
```

A 404 reported as "Network error". A missing local file reported as "Network error". The "private repository" hint fires unconditionally.

**Expected**: At entry, classify the input:
- Looks like a path or ends in `.onnx` → check existence first; report `Error: Local file '.\does-not-exist.onnx' does not exist.`
- Matches HF id pattern (`org/name`) → catch `RepositoryNotFoundError` and report `Error: Model 'totally-bogus/does-not-exist' not found on Hugging Face Hub.`
- Real network failure → `Error: Network error fetching '...': <DNS / TLS / timeout>.`

**Why it matters**: "Network error" sends users down the wrong debugging path (proxy, VPN, firewall) when the real issue is a typo. P0 because it's the most common failure mode for new users.

---

### INS-P1-1 — `winml inspect -m <hf_id>` takes 24 s end-to-end; first user-visible output silent for ~14 s

**Category**: Time-to-First-Output — first byte to user must arrive within 500 ms of entry.

**Repro**:
```
Measure-Command { uv run winml inspect -m microsoft/resnet-50 }
```

**Actual**: ~24.3 s wall time on warm cache. The first ~14 s after pressing Enter shows nothing on screen, then the panels render almost instantly.

**Expected**: Print a banner immediately on entry (`Inspecting microsoft/resnet-50 …`), show a `rich.status` spinner during HF metadata fetch. `inspect` should never download model weights — only `config.json` is needed. With cached `config.json`, drop to <2 s.

**Why it matters**: A 14-second silence makes the user assume the command hung and Ctrl-C; then they re-run, get the same silence, and the issue compounds. A spinner inside the first 500 ms removes the ambiguity.

---

### INS-P1-2 — `--list-tasks` takes 12.6 s for what should be a static dict lookup

**Category**: Local Command Latency — informational queries should return in <500 ms.

**Repro**:
```
Measure-Command { uv run winml inspect --list-tasks }
```

**Actual**: ~12.6 s warm. Output is a static list of ~22 task names.

**Expected**: <200 ms. Move the task list to a hand-coded constant in `loader/task.py`; do not import `optimum.exporters` to enumerate it at runtime.

**Why it matters**: This is a pure documentation-display call. A 12-second wait to read 22 strings is indefensible.

---

### INS-P1-3 — `UnicodeEncodeError` on `→` (U+2192) under cp1252

**Category**: Unicode/Output Rendering — output must survive non-UTF-8 stdout.

**Repro**:
```
cmd /c "uv run winml inspect -m microsoft/resnet-50"
```

**Actual** (cmd.exe with default cp1252 stdout):
```
File "…\rich\_win32_console.py", line 402, in write_text
    self.write(text)
File "…\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 4
```
Exit 1. PowerShell 7 with UTF-8 stdout does not crash.

**Expected**: Detect `sys.stdout.encoding != "utf-8"` and substitute ASCII fallbacks (`->` for `→`); or re-open stdout in UTF-8 mode at startup.

**Why it matters**: For cmd.exe users, `winml inspect` is unusable.

---

### INS-P1-4 — `--task bogus-task` leaks `TasksManager` jargon and points to Optimum docs

**Category**: Error Messages — no third-party class names; suggest valid values.

**Repro**:
```
uv run winml inspect -m microsoft/resnet-50 --task bogus-task
```

**Actual**:
```
Error: Inspection error: Task 'bogus-task' not supported by TasksManager. Check optimum documentation for supported tasks.
```

**Expected**: Validate `--task` at Click time as `Choice(KNOWN_TASKS)`. Error becomes:
```
Error: Invalid task 'bogus-task'. Valid: image-classification, text-classification, ... (22 total).
       See 'winml inspect --list-tasks' for the full list.
```
Exit 2.

**Why it matters**: "TasksManager" is an internal `optimum` class name. Users have no idea what it is. The message also points them to Optimum's docs — irrelevant.

---

## `winml optimize`

### OPT-P0-1 — `--preset qnn-compatible` (advertised in `--help`) ships broken

**Category**: Advertised Flags — every value documented in `--help` must work.

**Repro**:
```
uv run winml optimize -m temp\cli-audit\resnet.onnx --preset qnn-compatible -o temp\ux\opt_qnn.onnx
```

**Actual**:
```
Preset: qnn-compatible
Applied preset: qnn-compatible
Configuration validation errors:
  * Unknown capability 'graph-optimization-level'
```
Exit 1.

**Expected**: Either remove the preset (mark `hidden=True`), or update its capability list to the current schema. Add a regression test that runs every preset against the current capability registry.

**Why it matters**: A QNN-targeted preset that fails is misleading on a tool whose primary EP is QNN.

---

### OPT-P1-1 — Several `sys.exit()` paths in `optimize.py` instead of `click.ClickException`

**Category**: Exit-Code Contract — see TOP-P1-2.

**Repro**: Triggering an internal error (e.g. malformed `--preset-config` JSON) prints the error and `sys.exit(1)` immediately.

**Actual**: Exit 1 with no documented meaning vs. Click's exit 2 for usage errors. Inconsistent with the rest of the CLI (most paths use `click.ClickException` → exit 1).

**Expected**: Replace `sys.exit(N)` with `raise click.ClickException(...)` (exit 1) or `raise click.UsageError(...)` (exit 2), matching the contract proposed in TOP-P1-2.

**Why it matters**: Same as TOP-P1-2 — CI scripts can't branch on exit codes.

---

### OPT-P2-1 — `-p` short flag means `--preset` here, but means `--precision` in `quantize` / `config` / `build`

**Category**: Reserved Short Flags — short flags must mean the same thing across commands.

**Repro**:
```
uv run winml optimize --help | Select-String '^\s+-p,'
uv run winml quantize --help | Select-String '^\s+-p,'
uv run winml config --help   | Select-String '^\s+-p,'
uv run winml build --help    | Select-String '^\s+-p,'
```

**Actual**:
- `winml optimize`: `-p, --preset TEXT`.
- `winml quantize`, `winml config`, `winml build`: `-p, --precision TEXT`.

**Expected**: `-p` should mean `--precision` (3 commands already use it that way). Use a different short letter for `--preset` (e.g. `-P` or no short). The fix lives in `commands/optimize.py:207`.

**Why it matters**: A user who has used `-p int8` in three commands will type `-p int8` against `winml optimize` and get "no such preset 'int8'" — or worse, a silent fall-through to a wrong preset.

---

## `winml perf`

### PRF-P0-1 — `--compare-devices` advertised but unimplemented

**Category**: Advertised Flags — every flag in `--help` must work.

**Repro**:
```
uv run winml perf -m temp\cli-audit\resnet.onnx --compare-devices "cpu,npu" --iterations 5
```

**Actual**:
```
--compare-devices is not yet implemented. Run benchmarks separately and compare JSON outputs.
```
Exit 1.

**Expected**: Hide the flag (`hidden=True`) until implemented, or remove it. `--help` should not advertise non-functional features.

**Why it matters**: Users plan workflows around advertised flags. An "advertised but not implemented" flag is worse than no flag at all.

---

### PRF-P1-1 — Writes results to CWD by default (50+ stranded `*_perf.json` files in repo root)

**Category**: Output File Safety — never write to CWD by default.

**Repro**:
```
cd c:\Users\zhenni\repos\wmk
uv run winml perf -m temp\cli-audit\resnet.onnx --iterations 5
Get-ChildItem *_perf.json | Measure-Object
```

**Actual**: After previous runs, `Get-ChildItem c:\Users\zhenni\repos\wmk -Filter *_perf.json` returns **50+** stranded `<modelname>_perf.json` files in the repo root.

**Expected**: Either no file written unless `-o` given, or default to `~/.cache/winml/perf/<timestamp>/<modelname>.json` and tell the user where it went on success.

**Why it matters**: Pollutes the user's working directory (or their git repo). Easy to commit a perf file by accident. Easy for the user to lose track of which file is from which run.

---

### PRF-P1-2 — `--iterations 0` succeeds with garbage stats and exit 0

**Category**: Input Validation — validate ranges at parse time.

**Repro**:
```
uv run winml perf -m temp\cli-audit\resnet.onnx --iterations 0 --warmup 1 -o temp\ux\zero.json
```

**Actual**:
```
Latency (ms)
Avg=0.00 P50=0.00 P90=0.00 ... Std=0.00
Throughput: 0.00 samples/sec
Results saved to: temp\ux\zero.json
```
Exit 0. `zero.json` contains all-zero stats and is indistinguishable in shape from a real result.

**Expected**: Click `IntRange(min=1)` on `--iterations` (and `--warmup`); reject at parse time:
```
Error: Invalid value for --iterations: 0 is not in the range x>=1.
```
Exit 2.

**Why it matters**: A `*_perf.json` file with all zeros that has exit code 0 will be silently ingested by downstream tooling (dashboards, regression bots) and cause false alarms.

---

### PRF-P1-3 — `--module NoSuchClass` exits 3 with wrong error (blames the model file)

**Category**: Error Messages — the error message must point at the actual problem.

**Repro**:
```
uv run winml perf -m temp\cli-audit\resnet.onnx --iterations 3 --warmup 1 --module NoSuchClass
```

**Actual**:
```
Generating module configs for NoSuchClass...
Error generating module configs: It looks like the config file at 'temp\cli-audit\resnet.onnx' is not a valid JSON file.
```
Exit 3 (a code emitted nowhere else in the CLI).

**Expected**: Validate `--module` against the registered class list at entry:
```
Error: Module 'NoSuchClass' not found. Valid: WinMLModelForImageClassification, WinMLModelForCausalLM, ...
```
Exit 2.

**Why it matters**: The error blames `resnet.onnx` (which is a perfectly valid ONNX) when the actual problem is the unknown class name. Also introduces a 4th distinct exit code in the CLI (0/1/2/3) with no documentation.

---

### PRF-P1-4 — Several `sys.exit()` paths in `perf.py`, including `sys.exit(0)` on the no-modules-matched user-error path

**Category**: Exit-Code Contract — see TOP-P1-2. The `sys.exit(0)` is particularly bad — it masks a user error as success.

**Repro**: Trigger the `--module pattern` matches nothing path:
```
uv run winml perf -m temp\cli-audit\resnet.onnx --module DoesNotExist*
```

**Actual**: Prints `No modules matched pattern 'DoesNotExist*'`, exits 0. CI sees success.

**Expected**: Treat "no modules matched" as a usage error → exit 2. Replace every `sys.exit(N)` in `perf.py` with `click.ClickException` / `click.UsageError`.

**Why it matters**: A user error returning success in CI is the worst possible outcome — silent passing tests.

---

## `winml quantize`

### QNT-P0-1 — `--precision banana` silently falls back to defaults, prints "Success!", exits 1

**Category**: Input Validation — reject unknown values at parse time; do not contradict the exit code.

**Repro**:
```
uv run winml quantize -m temp\cli-audit\resnet.onnx --precision banana -o temp\ux\bad.onnx
```

**Actual**:
```
Precision: banana          ← lying — actually used uint8/uint8
Weight type: uint8
Activation type: uint8
Success! Model quantized
QDQ nodes inserted: 256
```
Followed by two near-duplicate ORT WARNING lines (~7 s apart). Exit code **1** despite the `Success!` line in stdout.

**Expected**: Declare `--precision` as `click.Choice([auto, fp32, fp16, int8, int16, w4a16, w8a8, w8a16])`. Fail at parse time:
```
Error: Invalid value for --precision: 'banana' is not one of [auto, fp32, fp16, int8, int16, w4a16, w8a8, w8a16].
```
Exit 2.

**Why it matters**: Three things wrong at once: (a) the flag value is silently ignored; (b) stdout says "Success!"; (c) exit code is 1. Any one of those is a P1; together they form a P0.

---

### QNT-P0-2 — `--precision int8` does not echo the precision in output (user cannot verify the flag took effect)

**Category**: Output Rendering — always echo what the command did, before doing it.

**Repro**:
```
uv run winml quantize -m temp\cli-audit\resnet.onnx --precision int8 --samples 3 -o temp\ux\q_int8.onnx
```

**Actual**: Output omits the `Precision:` line entirely. Other invocations (e.g. `--precision banana`) print `Precision: banana`. The output is non-deterministic in what fields it shows.

**Expected**: Always print, before quantization starts:
```
Precision:        int8
Weight type:      int8
Activation type:  int8
Calibration:      3 samples (minmax)
```

**Why it matters**: Users cannot verify their flag was honored without diffing the resulting ONNX.

---

### QNT-P2-1 — Two near-duplicate ORT WARNING lines per run

**Category**: Quiet at Default — library WARNING noise should be suppressed or de-duplicated.

**Repro**:
```
uv run winml quantize -m temp\cli-audit\resnet.onnx --samples 3 -o temp\ux\q.onnx
```

**Actual**:
```
WARNING: Please consider to run pre-processing before quantization. Refer to example: …
WARNING: Please consider pre-processing before quantization. See …
```
Two ORT layers emit the same advice with slightly different wording, ~7 s apart.

**Expected**: Silence both at default verbosity, or de-duplicate so only one fires per run.

**Why it matters**: Duplicate warnings make users assume something is wrong twice.

---

## `winml sys`

### SYS-P1-1 — Takes 11.5 s warm

**Category**: Local Command Latency — informational queries should return in <500 ms.

**Repro**:
```
Measure-Command { uv run winml sys }
Measure-Command { uv run winml sys --format compact }
Measure-Command { uv run winml sys --list-device }
Measure-Command { uv run winml sys --list-ep }
```

**Actual**:
- `winml sys` → **11.5 s** warm.
- `winml sys --format compact` → **3.8 s** warm (proves a fast path exists, but is opt-in).
- `winml sys --list-device` → 6.3 s.
- `winml sys --list-ep` → 1.3 s.

**Expected**: <500 ms warm. Replace `import torch; torch.__version__` with `importlib.metadata.version("torch")`. Cache EP probes within process. Make `compact` the default text rendering.

**Why it matters**: Users will run `winml sys` first to debug *anything*. A 12-second response makes the whole tool feel slow.

---

### SYS-P1-2 — Internally contradictory about the same EP

**Category**: Single Source of Truth — one capability check per EP; tables must agree.

**Repro** (on a machine without OpenVINO installed):
```
uv run winml sys
```

**Actual**: Three tables disagree about OpenVINO:
- `Backend SDKs:` row → `OpenVINO  Not found`.
- `Export Readiness:` row → `OpenVINO Conversion  Not installed`.
- `Available Execution Providers:` row → `OpenVINOExecutionProvider` (i.e. *available*).

**Expected**: One capability check (e.g. `is_ep_runnable("OpenVINO")`), consumed by all three tables. One verdict, displayed three times.

**Why it matters**: A diagnostic command that contradicts itself defeats its purpose. Users will pick whichever row supports their hypothesis and waste time.

---

### SYS-P2-1 — `winml sys -v` leaks DEBUG log lines inside Rich tables

**Category**: Verbosity Convention — log output goes to stderr; UI output goes to stdout.

**Repro**:
```
uv run winml sys -v
```

**Actual**:
```
[05/08/26 16:06:12] DEBUG    OpenVINO not available
┌─────────────────────────────┐
│ ModelKit System Information │
└─────────────────────────────┘

... (tables for Environment / ML Libraries / Backend SDKs / Available Devices) ...

[05/08/26 16:06:19] DEBUG    Found EP: QNNExecutionProvider at C:\Program
                             Files\WindowsApps\...\onnxruntime_providers_qnn.dll
                    DEBUG    WinML EP discovery successful:
                             ['QNNExecutionProvider']

Available Execution Providers
  QNNExecutionProvider           -> NPU/GPU
  ...
```
DEBUG lines appear *between* (and in some cases interleaved with) Rich panels on stdout, breaking the visual structure and making the output unparseable when piped.

**Expected**: Route logger output to stderr; tables to stdout. Run with `winml sys -v 2>&1` and the user sees a clean table; run with `winml sys -v 2>nul` and the tables are pristine.

**Why it matters**: Mixed stderr/stdout makes both verbose debugging and clean piping impossible. Also see CC-P2-6 (verbosity convention).

---

## Cross-cutting

These issues affect ≥3 commands. Fix once, fix everywhere.

### CC-P1-2 — No safe-overwrite control on any output-producing command

**Category**: Output File Safety — destructive overwrite must be opt-in.

**Repro**: Run any output-producing command twice with the same `-o`:
```
uv run winml export -m microsoft/resnet-50 -o temp\ux\out.onnx
"hello world" | Out-File temp\ux\out.onnx -Force
uv run winml export -m microsoft/resnet-50 -o temp\ux\out.onnx
Get-Item temp\ux\out.onnx | Select-Object Length
```

**Actual**: The second invocation silently overwrites the file the user just wrote. No warning, no prompt, no `--force` requirement. Reproduced for: `export`, `optimize`, `quantize`, `compile`, `perf -o`, `build -o`, `config -o`, `eval -o`, `analyze --output`, `hub -o`. None of these commands have a `--force` flag.

**Expected**: Add a shared `--force / -f` flag (default `False`) to every output-producing command. At entry:
```python
if output.exists() and not force:
    raise click.ClickException(
        f"Output '{output}' already exists. Re-run with --force to overwrite."
    )
```

**Why it matters**: Users keep multiple variant outputs in one folder (`q1.onnx`, `q1_int8.onnx`, etc.) and easily clobber yesterday's good model with today's broken one. There is no recovery — the old file is gone.

---

### CC-P1-3 — Missing-required-option errors lack a runnable example

**Category**: Error Messages — when a required flag is missing, show a copy-paste example that would succeed.

**Repro**:
```
uv run winml export
uv run winml compile
uv run winml quantize
uv run winml optimize
uv run winml inspect
```

**Actual** (all five):
```
Usage: winml export [OPTIONS]
Try 'winml export --help' for help.
Error: Missing option '--model' / '-m'.
```
No example. No suggested HF id. No hint that `microsoft/resnet-50` would work as a smoke test.

**Expected**: When Click reports `Missing option`, append one example line lifted from the command's own `--help` epilog. For example:
```
Error: Missing option '--model' / '-m'.
Example: winml export -m microsoft/resnet-50 -o resnet50.onnx
```
Implement once via a shared `MissingOptionWithExample` exception class on the Click group; each command supplies its own example string.

**Why it matters**: Most first-time users discover a command by typing its name with no flags. Today they get a one-line error and a `--help` reference. Showing one runnable example halves the discovery loop and increases the chance the user's *second* command succeeds.

---

### CC-P2-1 — `--model` semantic drift across 10 commands

**Category**: Cross-Command Flag Consistency — same flag should mean the same thing everywhere.

**Repro**:
```
uv run winml analyze --help | Select-String '\-\-model'
uv run winml build --help   | Select-String '\-\-model'
uv run winml compile --help | Select-String '\-\-model'
uv run winml config --help  | Select-String '\-\-model'
uv run winml eval --help    | Select-String '\-\-model|\-\-model-id'
uv run winml export --help  | Select-String '\-\-model'
uv run winml hub --help     | Select-String '\-\-model'
uv run winml inspect --help | Select-String '\-\-model'
uv run winml optimize --help| Select-String '\-\-model'
uv run winml perf --help    | Select-String '\-\-model'
uv run winml quantize --help| Select-String '\-\-model'
```

**Actual**:
- 4 different Python types (`str`, `Path`, `Tuple[str, ...]`, `List[str]`).
- 3 different parameter names: `model`, `model_id`, `hf_model`.
- Inconsistent acceptance: some accept HF ids only, some accept `.onnx` only, some accept either, some accept directories of HF snapshots.
- `eval` uniquely uses `multiple=True` plus a separate `--model-id` flag.

**Expected** — propose this canonical pattern:

```
-m, --model VALUE         # required for commands that need a model; lowercase, singular
```

- Accepts three concrete kinds, auto-detected at entry:
  1. **HF id** — matches `^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$`. Fetch from Hub.
  2. **Local file** — exists on disk and ends in `.onnx` / `.pt` / `.safetensors`.
  3. **Local directory** — exists on disk and contains `config.json` (HF snapshot layout).
- One internal Python type `ModelRef` with `kind: Literal["hf", "file", "dir"]` discriminator.
- One shared decorator `@model_option(required=True, multiple=False)` applied to every command.
- Three concrete error messages — see INS-P0-1.
- `eval`'s multi-role syntax (`-m generator=foo -m discriminator=bar`) becomes a separate decorator `@multi_role_model_option`, used only by commands that need it.

**Why it matters**: Users learn one flag and expect it to behave the same way everywhere. Type drift means automation has to special-case every command.

---

### CC-P2-2 — `-o/--output` overloaded: file vs directory vs stdout vs CWD-clutter

**Category**: Cross-Command Flag Consistency — same flag should mean the same thing everywhere.

**Repro**:
```
uv run winml perf --help     | Select-String '\-\-output|\-o,'   # writes <model>_perf.json to CWD if omitted
uv run winml export --help   | Select-String '\-\-output|\-o,'   # file path required
uv run winml hub --help      | Select-String '\-\-output|\-o,'   # writes JSON to stdout if omitted
uv run winml build --help    | Select-String '\-\-output|\-o,'   # directory
uv run winml analyze --help  | Select-String '\-\-output'        # no -o short alias (see ANA-P1-2)
```

**Actual**: Five different behaviors in five commands.
- `perf`: optional; if omitted, writes `<modelname>_perf.json` to **CWD** (see PRF-P1-1).
- `export`: required; file path.
- `hub`: optional; if omitted, prints to stdout; if given, writes file.
- `build`: required; **directory** path (output is a folder of artifacts).
- `analyze`: only the long `--output` works; no `-o` short (see ANA-P1-2).

**Expected** — propose this canonical pattern:

```
-o, --output PATH         # always, lowercase, singular, type = pathlib.Path
```

Two flavors based on what the command produces:

| Command produces   | `--output` semantics                                                    |
|--------------------|-------------------------------------------------------------------------|
| One file           | `PATH` is a file path. If omitted, command writes nothing to disk (still prints UI to stdout). If given and path exists, refuse unless `--force` (CC-P1-2). |
| Directory of files | `PATH` is a directory. Created if missing. If non-empty, refuse unless `--force`. |

- Never write to CWD by default (kills PRF-P1-1).
- Never default to stdout for binary output. JSON output → stdout only when `-o -` is passed explicitly.
- Always print on success: `Saved to: <abs path>` (one line, last line of normal stdout).
- All commands import the same shared decorator: `@output_option(kind=Literal["file","dir"], required=False)`.

**Why it matters**: Users don't memorize per-command behaviors. They expect `-o foo.json` to write `foo.json` in every command, and to fail loudly on overwrite.

---

### CC-P2-5 — `-d` short used only by `compile --device`; everyone else uses `--device` with no short

**Category**: Reserved Short Flags — each short flag means one thing, used the same way across commands.

**Repro**:
```
uv run winml compile --help | Select-String '^\s+-d,'
uv run winml build --help   | Select-String '^\s+-d,'
uv run winml eval --help    | Select-String '^\s+-d,'
uv run winml perf --help    | Select-String '^\s+-d,'
```

**Actual**: Only `compile` declares `-d, --device`. Other commands take `--device` long-form only.

**Expected**: Either drop `-d` from `compile` (so `--device` is consistently long-form everywhere), or add `-d, --device` to all four commands. The shared `device_option` decorator from BLD-P2-3 should make this trivial.

**Why it matters**: A user who runs `winml compile -d npu` then tries `winml build -d npu` gets "no such option" with no hint that `--device` is the long form.

---

### CC-P2-6 — Verbosity is declared inconsistently; logger output mixes with UI

**Category**: Verbosity Convention — one declaration, one log format, one stream.

**Repro**:
```
uv run winml export -m microsoft/resnet-50 --verbose -o temp\ux\rn.onnx
uv run winml export -m microsoft/resnet-50 -vv -o temp\ux\rn.onnx     # error: extra argument
uv run winml --quiet export -m microsoft/resnet-50 -o temp\ux\rn.onnx # works
uv run winml export --quiet -m microsoft/resnet-50 -o temp\ux\rn.onnx # error: no --quiet on export
```

**Actual**: Mix:
- Top-level group declares `-v/--verbose` (count) and `-q/--quiet`.
- 12 of 13 subcommands **redeclare** `--verbose` as `is_flag=True` (so `-vv` after subcommand fails).
- No subcommand exposes `-q/--quiet` (parent-level only).
- Only `analyze` uses the shared `verbosity_options` decorator.
- DEBUG / INFO log lines are interleaved with Rich tables on stdout (see SYS-P2-1).

**Expected** — propose this canonical pattern:

1. **Declaration**: top-level group declares `-v/--verbose` (count, `-v` = INFO, `-vv` = DEBUG) and `-q/--quiet` (errors only). **No subcommand redeclares either flag.** Apply via shared `verbosity_options` decorator.

2. **Default level**: `WARNING` and above. `-q` raises to `ERROR`. `-v` lowers to `INFO`. `-vv` lowers to `DEBUG`.

3. **Log format** (stderr only):
   ```python
   logging.Formatter(
       "[%(asctime)s %(levelname)-7s %(name)s] %(message)s",
       datefmt="%H:%M:%S",
   )
   ```
   Sample line: `[14:32:11 INFO    winml.export] Loaded config.json (cached)`

4. **Streams**:
   - **stdout** = user-facing UI (Rich panels, success messages, JSON results, `--help`).
   - **stderr** = log lines (DEBUG/INFO/WARNING/ERROR), library warnings, progress bars, banners.
   - This way `winml export ... > out.onnx 2> log.txt` does the right thing.

5. **Compliance** — current state vs. the proposal:

   | Command | Inherits top-level `-v/-q` | No subcommand redeclare | Logger → stderr only |
   |---|---|---|---|
   | `analyze` | ✅ | ✅ | ⚠ (some `print` to stdout) |
   | `build` | ✅ | ❌ redeclares `--verbose` | ❌ |
   | `compile` | ✅ | ❌ | ❌ |
   | `config` | ✅ | ❌ | ❌ |
   | `eval` | ✅ | ❌ | ❌ |
   | `export` | ✅ | ❌ | ❌ |
   | `hub` | ✅ | ❌ | ❌ |
   | `inspect` | ✅ | ❌ | ❌ |
   | `optimize` | ✅ | ❌ | ❌ |
   | `perf` | ✅ | ❌ | ❌ |
   | `quantize` | ✅ | ❌ | ❌ |
   | `sys` | ✅ | ❌ | ❌ (DEBUG inside tables — SYS-P2-1) |
   | `expand_rules` | ✅ | ✅ | ✅ |

   So today only `analyze` and `expand_rules` partially comply. The rest must remove their local `--verbose` declarations and route loggers through `logging.basicConfig(stream=sys.stderr)`.

**Why it matters**: Without a single convention, users learn the flags 13 times. Mixed streams make piping (`> file 2> log`) impossible and break automation.

---

### CC-P2-7 — No `--no-color` flag (NO_COLOR works only via Rich's tty auto-detection)

**Category**: Color & Theming — color must be controllable both per-invocation (flag) and per-environment (env var).

**Repro**:
```
uv run winml --help        | Select-String 'no-color'
uv run winml export --help | Select-String 'no-color'
$env:NO_COLOR='1'; uv run winml sys --format compact > out.txt 2>&1; Remove-Item Env:NO_COLOR
Get-Content out.txt -Raw | Select-String -Pattern '\x1b\['
```

**Actual**:
- No `--no-color` flag exists in `winml --help` or any subcommand.
- `NO_COLOR=1` does suppress ANSI in piped output (verified — 0 escape sequences in capture). This works because Rich auto-detects no-tty / NO_COLOR.
- But there is no per-invocation flag, so a user running interactively in a terminal cannot disable color without setting an env var.

**Expected**: Add a top-level `--no-color` flag that sets the Rich `Console(no_color=True, force_terminal=False)` for the rest of the run. Document `NO_COLOR=1` and `CI=true` in the help epilog. Do not regress the current Rich auto-detection.

**Why it matters**: Some terminals (older Windows consoles, log viewers, screen-reader sessions) render the color codes as garbage. A per-invocation flag is the standard escape hatch.

---

### CC-P2-8 — No environment variables documented in any `--help`

**Category**: Discoverability — env vars that affect behavior must be listed in `--help`.

**Repro**: Greps across all 13 subcommand `--help` outputs find zero references to `NO_COLOR`, `CI`, `WINML_*`, `HF_HUB_OFFLINE`, `HF_HOME`, `HF_TOKEN`, `TRANSFORMERS_CACHE`. (Word matches like "HuggingFace" or "environment" are unrelated.)

**Actual**: Users must read source to discover that:
- `NO_COLOR=1` and `CI=true` suppress color (works via Rich auto-detection — see CC-P2-7).
- `HF_HUB_OFFLINE=1` is honored (verified: `winml inspect -m microsoft/resnet-50` succeeds against cache).
- `HF_HOME` / `TRANSFORMERS_CACHE` redirect the model cache (used transitively by `transformers`).
- No `WINML_*` env vars are documented anywhere.

**Expected**: Add a single `Environment Variables` epilog to `winml --help` listing each variable, what it controls, and one example. Same epilog should appear in any subcommand whose behavior depends on the var (e.g. `eval` and `inspect` should mention `HF_HUB_OFFLINE`).

**Why it matters**: Env vars are invisible by definition. Without documentation, CI authors guess and copy from blog posts; offline users don't know they can run without the network; debug users don't know how to silence color.

---

### CC-P3-1 — No `winml cache`; no shell-completion subcommand

**Category**: Cache Visibility / Shell Completion.

**Repro**:
```
uv run winml cache --help        # no such command
uv run winml completion --help   # no such command
```

**Actual**: `winml build` has caching infrastructure, but no way to inspect or prune it. The wheel ships no completion files for bash/zsh/PowerShell.

**Expected**:
- `winml cache list` — show cache directory, size, items.
- `winml cache prune [--older-than 30d]` — delete old entries.
- `winml completion bash | zsh | powershell` — print a completion script suitable for `eval $(...)`.

**Why it matters**: Users with multi-GB caches need a way to clean up without resorting to `rm -rf`. Tab completion is a quality-of-life expectation for any modern CLI.

---

## Feature-Owner Self-Check

Three categories of issue an automated audit cannot detect — they require the design spec, hardware diversity, or a clean install state. The feature owner shall confirm each item below before declaring a command "Ready".

### SC-1 — Functional correctness against the design spec

**Category**: Functional Correctness (covers R1.2 + R5.1).

**What to check** (per command):
- Every flag whose name implies a behavior (`--quantize`, `--preset X`, `--precision int8`, `--use-cache`, `--trust-remote-code`, `--shape-config`) produces a *visible* difference vs. omitting the flag (output size, opset, EP-specific node count, presence of QDQ ops, cache-hit log, observed download, model input shape).
- A flag whose presence vs. absence yields byte-identical output is a feature gap or a silent fallback.
- The observable behavior matches what the design doc / PRD / authorizing issue promised.

**Why it matters**: An agent can verify a command runs without crashing but cannot tell whether the resulting artifact matches the spec. "Quantized to int8" and "compatible with QNN" are claims only the feature owner can validate.

---

### SC-2 — Cross-EP coverage

**Category**: Functional Correctness (covers R5.2).

**What to check** (per command that exposes `--ep`):
- Every EP listed in the command's `--ep` choice set produces a correct result on hardware that supports it — not just exit 0.
- The selected EP appears in the run's echo-back (R2.7) and in EP-specific evidence (e.g. `QNNExecutionProvider` in `winml sys`, EP-context nodes in the output `.onnx`).
- `--ep <X>` on a host that lacks `<X>` produces the R6.2 install hint and exits 3 — never `ImportError`, never silent fallback to CPU.
- No `if ep == "<name>"` branches in command code (R5.2 violation even when the test passes).

**Why it matters**: The audit machine has only the EPs it has. Cells the agent marks `DEFERRED <no hardware>` can only be closed by an owner with access to the full EP matrix or an internal CI lab.

---

### SC-3 — First-run success on a fresh install

**Category**: Install & Environment (covers R6.1 + R6.2).

**What to check** (before each release):
- In a clean venv with cleared model caches, the *first* example printed in every `winml <cmd> --help` runs verbatim and succeeds.
- No `Run scripts/download_rules.py` references to files not in the wheel.
- No raw `ModuleNotFoundError` for an optional dep — must be the R6.2 install hint.
- First-run model download shows progress, not silent multi-minute hang.
- Same checks pass with `HF_HUB_OFFLINE=1` against a pre-warmed cache.
- Same checks pass behind a corporate proxy if the shipping target includes corporate users.

**Why it matters**: The audit machine has every optional dep installed and every model already cached. First-time users without those caches and packages hit errors the audit never sees.

---
