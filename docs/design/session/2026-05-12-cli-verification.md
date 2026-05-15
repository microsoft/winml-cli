# CLI Verification — 2026-05-12

Branch: `feat/op-tracing-refactor` at `db39b80d`.

Run started: 2026-05-12 15:32 (wall clock), completed: 2026-05-12 16:09.

## Command results

| # | Command (short) | Exit | Notes |
|---|---|---|---|
| 1 | `winml export facebook/convnext-base-224` | **0** | Success. Wrote `convnext/model.onnx` (300 KB header + 354 MB `.data` external tensors), `model_htp_metadata.json`. |
| 2 | `winml analyze ... --optim-config optim.json` | **TIMEOUT (killed at 30+ min)** | Per-node ORT fallback because rule zip missing; processed ~270/667 nodes in 18 min then stdout plateaued for 10+ min while CPU kept climbing. Killed at 16:03. `optim.json` was never written. |
| 3 | `winml optimize -m convnext/model.onnx` | **0** | Bypassed missing `optim.json` by using capability defaults. Wrote `convnext/model_opt.onnx`. Nodes 667 → 557 (16.5% reduction). |
| 4 | `winml quantize -m convnext/model_opt.onnx` | **0** | 27.5s. Wrote `convnext/model_opt_qdq.onnx`. QDQ nodes inserted: 1463. Weight/activation uint8, minmax, 10 random samples. |
| 5 | `winml compile -m convnext/model_opt_qdq.onnx` | **1** | All QNN HTP graph-prep / sequencing / VTCM / parallelization stages completed (4.6s of QNN work) but `EPContext model not found in work directory` → no output file written. Real exit 1; CLI banner printed `Error: No output file produced. Check EP context support for provider 'qnn'.` (Earlier tee-piped run reported `EXIT=0` only because `tee` masked the real `sys.exit(1)`.) |
| 6 | `winml perf -m microsoft/resnet-50 --ep qnn --device npu` | **127** | Reaches the analyze pre-check phase (`PatternMatcher`, then 0/122 op probes), then a child subprocess dies ~60s in, propagating exit 127. Benchmark never starts. Tested twice (`--iterations 1 --warmup 0` and default) — identical failure. (Earlier tee-piped run reported `EXIT=0` for same reason as cmd5.) |

## Per-command detail

### Command 1: `winml export -m facebook/convnext-base-224 -o convnext/model.onnx`

**Exit code:** 0
**Output:**
```
Model: facebook/convnext-base-224
Output: convnext\model.onnx
Auto-resolved input specs: ['pixel_values']
Auto-resolved output specs: ['logits']

Starting HTP export...
Detected task: image-classification

Success! Model exported to: convnext\model.onnx
```
**Verdict:** PASS
**Log file:** `D:/BYOM/ModelKit_PRs/op_tracing/temp/verify/cmd1_export.log`

---

### Command 2: `winml analyze -m convnext/model.onnx --optim-config optim.json`

**Exit code:** TIMEOUT — killed at 30+ min (user-specified timeout was 600s; per-task brief, "kill if hangs > 900s").
**Output (decoded — full stderr is ANSI-spaced ORT cpuid spam):**
```
═══════════════════════════════════════════════════════════════════════════════
📊 OP CHECK
═══════════════════════════════════════════════════════════════════════════════
   📦 Model: model.onnx
   🔧 Opset: 17  Producer: pytorch v2.11.0
   📋 Operators: 667 total, 11 unique types
───────────────────────────────────────────────────────────────────────────────
💻 EP 1: QNNExecutionProvider
───────────────────────────────────────────────────────────────────────────────
[15:34:15] WARNING  Rule zip not found:
            D:\...\analyze\rules\runtime_check_rules\QNNExecutionProvider_NPU_ai.onnx_opset17.zip.
            Run 'uv run python scripts/download_rules.py' to download rule files,
            or set MODELKIT_RULES_DIR to a directory containing the zip.
            WARNING  Rule zip file not found: <same>
<then 270+ identical ORT "Unknown CPU vendor" cpuid warnings, one per per-node
ORT subprocess spawn, at ~4-second intervals between 15:34:18 and 15:51:34>
<then 10-min stdout plateau at 298 lines while PID 1014936 kept burning CPU
 (54410s → 55155s in last 5 min observed → ~150% CPU sustained — alive but
 hung on stdout flush via the multiprocessing _OutputCapturingWrapper)>
```
**Verdict:** FAIL (timeout). Root cause **not** in the refactor — this is the missing rule-zip fallback path: when no `QNNExecutionProvider_NPU_ai.onnx_opset17.zip` is available, `RuntimeCheckerQuery._is_ep_available_locally()` activates per-node ORT execution (via `multiprocessing` with stdout/stderr capture), and `runtime_checker_query._build_single_node_model` + ORT session creation is invoked once per ONNX node. 667 nodes × ~4s ≈ 44 min, plus apparent stdout-flush stalls in the `_OutputCapturingWrapper` wrapper.

The `--optim-config` flag is **not** the trigger; analyze always runs this loop when the rule zip is missing (the `optim_config` argument only controls whether to save the file at the end — see `commands/analyze.py:673-684`).

**Log file:** `D:/BYOM/ModelKit_PRs/op_tracing/temp/verify/cmd2_analyze.log` (299 lines, 90 KB)

---

### Command 3: `winml optimize -m convnext/model.onnx`

(Run **without** `-c optim.json` because cmd2 never produced that file. Falls back to capability defaults.)

**Exit code:** 0
**Output:**
```
Input: convnext\model.onnx
Output: convnext\model_opt.onnx

Loading model...
Running optimizer...
Saving optimized model...

Success! Model optimized: convnext\model_opt.onnx
Nodes: 667 -> 557 (16.5% reduction)
```
**Verdict:** PASS (with caveat — ran against capability defaults rather than the analyze-derived config the user prescribed)
**Log file:** `D:/BYOM/ModelKit_PRs/op_tracing/temp/verify/cmd3_optimize.log`

---

### Command 4: `winml quantize -m convnext/model_opt.onnx`

**Exit code:** 0
**Output:**
```
Input: convnext\model_opt.onnx
Output: convnext\model_opt_qdq.onnx
Weight type: uint8
Activation type: uint8
Samples: 10
Method: minmax
Dataset: Random data (synthetic from ONNX I/O specs)

Running quantization...
[2026-05-12T16:04:26] WARNING: Please consider to run pre-processing before quantization. ...
[2026-05-12T16:04:45] WARNING: Please consider pre-processing before quantization. ...

Success! Model quantized
Output: convnext\model_opt_qdq.onnx
QDQ nodes inserted: 1463
Total time: 27.48s
```
**Verdict:** PASS
**Log file:** `D:/BYOM/ModelKit_PRs/op_tracing/temp/verify/cmd4_quantize.log`

---

### Command 5: `winml compile -m convnext/model_opt_qdq.onnx`

**Exit code:** 1 (real — the earlier tee-wrapped run reported `EXIT=0` because `tee` masks the real exit code of the upstream `sys.exit(1)`)
**Output (decoded):**
```
Input: convnext\model_opt_qdq.onnx
Device: npu
Provider: qnn
Compiler: ort

Compiling model...
Starting stage: Graph Preparation Initializing
Completed stage: Graph Preparation Initializing (1861 us)
Starting stage: Graph Optimizations
Completed stage: Graph Optimizations (1182873 us)
Starting stage: Post Graph Optimization
Completed stage: Post Graph Optimization (37398 us)
Starting stage: Graph Sequencing for Target
Completed stage: Graph Sequencing for Target (339136 us)
Starting stage: VTCM Allocation
Completed stage: VTCM Allocation (37545 us)
Starting stage: Parallelization Optimization
Completed stage: Parallelization Optimization (15172 us)
Starting stage: Finalizing Graph Sequence

====== DDR bandwidth summary ======
spill_bytes=0
fill_bytes=0
write_total_bytes=65536
read_total_bytes=113817600

Completed stage: Finalizing Graph Sequence (16163 us)
Starting stage: Completion
Completed stage: Completion (1006 us)
[ORT WARNING: Some nodes were not assigned to the preferred execution providers...]
[2026-05-12T16:04:58] WARNING: EPContext model not found in work directory

Warning: Compilation finished but no output file was written to the output directory.
Error: No output file produced. Check EP context support for provider 'qnn'.
```
**Verdict:** FAIL — exits 1 with the message `Error: No output file produced. Check EP context support for provider 'qnn'.`

The QNN HTP backend ran the full compile pipeline (all stages completed; DDR bandwidth summary printed), but the EPContext file is never materialized. The CLI then explicitly reports "no output file was written to the output directory" and exits 1. No `_ctx.onnx` exists in `convnext/`.

Likely root cause: `WinMLSession._build_session_options` (the **legacy instance method** at `session.py:462-485`) is what `compile()` calls (per audit Gap §1.1 in `2026-05-12-impl-status.md` line 45), and that method does **not** set `ep.context_enable=1` / `ep.context_file_path=...` the way the new free-function `_build_session_options` does. The compile graph happens in-memory in ORT-QNN, but the on-disk context dump is never wired in.

A second contributing factor: the audit also flags a `premature ort.InferenceSession in __init__` (compile no-op bug) — see `2026-05-12-impl-status.md` discussion of `WinMLSession.__init__`.

**Log file:** `D:/BYOM/ModelKit_PRs/op_tracing/temp/verify/cmd5_compile.log`

---

### Command 6: `winml perf -m microsoft/resnet-50 --ep qnn --device npu`

**Exit code:** 127 (real — earlier tee-wrapped run reported `EXIT=0`, again because of tee masking)
**Output (decoded — total runtime ~60s before subprocess crash):**
```
Loading model: microsoft/resnet-50
[ORT cpuid warning, ANSI-spaced]
[2026-05-12T16:07:07] WARNING: No pattern matches found by PatternMatcher
  0%|          | 0/122 [00:00<?, ?it/s]
[2026-05-12T16:07:07] WARNING: Rule zip not found:
    D:\...\analyze\rules\runtime_check_rules\QNNExecutionProvider_npu_ai.onnx_opset17.zip.
[2026-05-12T16:07:07] WARNING: Rule zip file not found: <same>
<process exits 127 after that, before the benchmark loop ever runs>
```

Two consecutive runs (default and `--iterations 1 --warmup 0`) reproduce: identical 1076-byte stderr, exit 127 ~60s after start, no python processes left running. The HF model loads, the perf pre-check enters the analyze runtime-checker loop (same code path that hung cmd2 — note the rule-zip warning casing differs: `QNNExecutionProvider_npu` lowercase here vs `_NPU_` in cmd2), but the per-node ORT subprocess fails ~immediately and the parent reports exit 127.

Exit 127 from a subprocess is "command not found" in POSIX shells, but the parent is Python; this is most likely a child process that died via SIGKILL/OOM/initialization failure inside the `_OutputCapturingWrapper` multiprocessing pool, propagated up by `concurrent.futures`. Without a stack trace it's impossible to be certain, but **the failure clearly happens in the analyze-pre-check phase before perf even runs the benchmark.**

**Verdict:** FAIL — but the failure mode is different from what the audit Gap §3 predicts. The audit said `models/auto.py:355-362` would raise a `TypeError` for the HF path because `WinMLPreTrainedModel.__init__` now needs `ep_device: EPDevice` rather than `device: str` (audit Gap #1). We do **not** see a TypeError in the output — instead we see exit 127 from a child process. Two possible reasons:
1. The TypeError happened in a child process and got swallowed by `_OutputCapturingWrapper`, returning a non-zero subprocess exit that bash translated to 127.
2. The migration of `models/auto.py` was completed at some point not reflected in the audit doc (need to verify by source inspection — pending).

**Log file:** `D:/BYOM/ModelKit_PRs/op_tracing/temp/verify/cmd6_perf.log`

---

## Summary

- Total commands: 6
- Passed: 3 (cmd1 export, cmd3 optimize, cmd4 quantize)
- Failed: 3 (cmd2 analyze TIMEOUT; cmd5 compile no-output, exit 1; cmd6 perf-HF subprocess crash, exit 127)
- Pattern of failures:
  - **All three failing commands hit the same code path family**: the analyze runtime-checker fallback (cmd2 hang, cmd6 crash) and the QNN compile output-not-written bug (cmd5). The post-export pipeline (cmd3/4) — which doesn't touch the analyzer or QNN-EPContext machinery — works cleanly.

## Likely root causes (cross-referenced with `2026-05-12-impl-status.md`)

### Cmd2 analyze TIMEOUT
**Not a refactor-introduced bug.** Pre-existing per-node ORT fallback in `RuntimeCheckerQuery` activates when the rule zip is missing. With 667 nodes × ~4s/node ≈ 44 minutes, plus a stdout-buffering stall in `_OutputCapturingWrapper`, the run exceeds any reasonable CLI timeout. Same fallback also runs when **no** `--optim-config` is given (see `commands/analyze.py:673-684` — the flag only gates the final save). Suggested mitigations:
- Add a `--skip-runtime-check` / `--fast` flag that classifies based on schema lookup alone.
- Run `uv run python scripts/download_rules.py` once at setup.
- Cache the per-node ORT results across runs.
- Cap the loop at N nodes per op_type (already noted as `n_cases` in `runtime_checker/check_ops.py:54`).

### Cmd5 compile failure (no output, exit 1)
**Direct match to Audit Gap #1.1 / Audit Gap #3 — legacy `WinMLSession._build_session_options` instance method.** The new free function `_build_session_options` (`session.py:162-205`) wires QNN EPContext options correctly, but `compile()` (`session.py:317, 338`) still calls the **legacy instance method** at `session.py:462-485`, which uses `set_provider_selection_policy(PREFER_NPU)` and does not set `ep.context_enable=1` / `ep.context_file_path=...`. Net effect: the ORT-QNN session compiles fine in memory, but the EPContext-file dump never happens. The status doc (line 45) explicitly marks this as `TODO Task 8 [bridge]`.

Recommendation: complete Task 8 [bridge] removal — migrate `compile()`, `is_compatible()`, and `WinMLQairtSession._create_inference_session()` to call the new free-function `_build_session_options(ep_device, ep_config, ep_monitor, base)` with appropriate ep_config that includes `context_enable=1` / `context_file_path=<output_dir>`.

There may also be the "premature `ort.InferenceSession` in `__init__`" issue (per the user's prompt header) — the constructor builds a bare session before `compile()` gets a chance to set up EPContext options, so by the time `compile()` runs there's no fresh session to attach EPContext options to. Need a source-level inspection of `WinMLSession.__init__` flow + `compile()` flow to confirm, but the symptom — "QNN graph stages all complete, but no output file" — is exactly what a no-op `compile()` would produce.

### Cmd6 perf HF-path crash (exit 127)
**Matches Audit Gap #1 (HF auto-build path — `models/auto.py:163-168, 196-203, 355-362`).** The audit predicts a `TypeError` from `winml_class(..., device=device)` because `WinMLPreTrainedModel.__init__` now requires `ep_device: EPDevice`. We see exit 127 not a TypeError, but the failure mode (~60s in, before any benchmark, while still in analyzer pre-check) is consistent with a child subprocess exception propagating up through `concurrent.futures` / `_OutputCapturingWrapper`. Either:
- The TypeError is raised inside a multiprocessing child, swallowed by the output capture, and the parent gets a generic non-zero subprocess exit.
- Or there's a downstream consequence of the still-broken HF path (e.g. monitor mismatch, EP registration failure) that crashes the same subprocess pool.

Either way the audit Gap #1 fix is on the critical path: migrate `models/auto.py` to call `resolve_device(ep_arg, device_arg)` at line 163/196/355 and pass `ep_device=` instead of `device=`. This is also explicitly called out in the status doc as **the blocker for Task 14 / §6 E2E gate**.

### Audit Gap #2 (legacy `device=` in e2e tests)
Not exercised by this CLI verification (test sweep is a separate pass).

## Recommendations

In priority order:

1. **Fix Audit Gap #1** (`models/auto.py:163-168, 196-203, 355-362`): replace the three `winml_class(..., device=device)` calls with `winml_class(..., ep_device=resolve_device(ep_arg, device_arg))`. This unblocks cmd6 and Task 14 E2E.
2. **Fix Audit Gap #3 / Task 8 bridge removal** (`WinMLSession._build_session_options` instance method): migrate `compile()`, `is_compatible()`, and `qairt_session._create_inference_session()` to use the new free-function `_build_session_options(...)` and ensure QNN EPContext is enabled when `compile()` is called. This unblocks cmd5.
3. **Investigate analyze runtime-check perf** (`RuntimeCheckerQuery._is_ep_available_locally` + the per-node ORT fallback): either ship the rule zip out of band, or add `--fast`/`--skip-runtime-check` to skip the local ORT loop, or aggressively cap to one probe per op_type (the model has 667 nodes but only 11 unique op types — running 11 probes instead of 667 would take ~45s instead of 45min).
4. **Cease piping through `tee`** in verification scripts — it masks real exit codes. Capture stdout/stderr to file and read `$?` from the upstream command directly. This bit cmd5 and cmd6 in the first attempt — both reported `EXIT=0` from `tee` while the actual underlying command exited 1 and 127 respectively.
5. **Once Gaps #1 and #3 are fixed**, re-run this full 6-command sequence. Expected: cmd2 will still take 45 min unless mitigation #3 is also applied; cmd5 and cmd6 should pass.
