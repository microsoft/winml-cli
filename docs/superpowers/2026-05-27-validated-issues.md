# v3 docs validated issues

Validated against microsoft/winml-cli @ 5e25579 on docs/draft.

## Critical (factually wrong; user would hit error)

### docs/getting-started/installation.md
- Python version wrong: doc states `3.10` / `requires-python = ">=3.10,<3.11"` but `pyproject.toml:13` reads `requires-python = ">=3.11,<3.12"`. Install step (`uv python install 3.10`) and verify output (`Python Version 3.10.x`) are both wrong.

### docs/getting-started/end-to-end.md
- DML and CPU artifact filenames are wrong: doc claims GPU produces `convnext_tiny_dml_ctx.onnx` and CPU produces `convnext_tiny.onnx`. `compiler/configs.py:175` (`for_dml`) and `:165` (`for_cpu`) both set `enable_ep_context=False`; `CompileStage.process` only calls `_finalize_output` when `ep_config.enable_ep_context` is True (`compile.py:102`). Neither DML nor CPU produces a `_ctx.onnx` file — the compile step is a no-op for both.

### docs/commands/overview.md
- `winml hub` command does not exist. Source registers the function as `catalog` (`catalog.py:387`). Every `winml hub` invocation in the doc will fail at the CLI.

### docs/commands/build.md
- `--random-init` flag does not exist. Source `build.py` has no such option (grep returns no hits). Passing it will produce "No such option".
- `--config / -c` documented as *(required)* but `build.py:237` sets `required=False`. When omitted, config is auto-generated from `-m`.
- `--qnn-sdk-root` listed in the flag table but does not exist on `winml build` (zero hits in `build.py`). It is a `winml compile`-only flag. Users will get "No such option".

### docs/commands/compile.md
- `--device` default documented as `npu` but `compile.py:62` sets `default="auto"`. Users expecting NPU-only targeting without `--device npu` will get auto-detection instead.
- `--no-quant` flag does not exist in `compile.py` (zero occurrences). Users who pass it get "No such option".

### docs/commands/config.md
- `--no-compile` default documented as `off` (compile included by default). `config.py:163` shows `default=True` for `no_compile`, meaning compile is *excluded* by default. The framing is entirely backwards — users need `--compile` to include compilation, not `--no-compile` to exclude it.

### docs/commands/eval.md
- `--device` type column shows `cpu|gpu|npu`, default `cpu`. Source `eval.py:69` defines `click.Choice(["auto", "cpu", "gpu", "npu"])` with `default="auto"`. `auto` is missing and the default is wrong.
- `-n` listed as a short alias for `--samples`. Source defines `--samples` with no short flag. The `-n` alias does not exist.

### docs/commands/hub.md
- All `winml hub` invocations fail: source registers the command as `winml catalog` (`catalog.py:387`).
- `--model / -m` flag does not exist in `catalog.py` (confirmed full read). Users who run `winml hub --model <id>` get "No such option".

### docs/commands/analyze.md
- `--device` default documented as `NPU`; `analyze.py:644` sets `default="auto"`. Users will not get NPU-specific analysis by default.
- `--ep` default documented as "all supported EPs analyzed"; `analyze.py:633` sets `default="auto"` (infers from local availability, not "all").
- `--run-unknown-op` default documented as "enabled"; `analyze.py:668` sets `default=False`. The pitfall note that says "disable when libraries are missing" compounds this by implying it is on.

### docs/commands/optimize.md
- `--preset / -p` flag does not exist. `optimize.py` command definition (lines 151–187) has no `--preset` option. The entire "Built-in presets" table and all preset-based examples are invalid; users get "Error: no such option: --preset".

### docs/commands/inspect.md
- `--list-tasks`, `--model-type`, and `--model-class` flags are not documented. All three are defined in source (`inspect.py:98–116`) and functional.

### docs/concepts/quantization.md
- `int8` row annotated "default for NPU via QNN EP". Actual NPU auto-precision default is `w8a16` (`precision.py:33`: `_AUTO_PRECISION = {"npu": "w8a16", ...}`). `int8` is valid for QNN but is not the default.
- `auto` row: "Resolves to `int8` (NPU)…" is wrong for NPU; resolves to `w8a16` per `_AUTO_PRECISION["npu"]` (`precision.py:33`).
- `w4a16` row: "Recognized as a precision string but raises error at quantization time" is wrong. `is_quantized_precision("w4a16")` returns `False` (4 not in `_BITS_TO_WEIGHT_TYPE`, `precision.py:57`) so it is rejected before quantization, not recognized at all.

### docs/concepts/compile-and-epcontext.md
- `--no-quant` on `winml compile` (line 29): flag does not exist in `compile.py`. Users get "Error: No such option".

### docs/concepts/config-and-build.md
- JSON `compile` section uses nested `ep_config.provider` structure. `WinMLCompileConfig.to_dict()` (`configs.py:230–245`) serializes flat with `execution_provider`, not nested under `ep_config`. Copy-pasting the example silently uses defaults instead of the specified values.

### docs/samples/bert-config-build.md
- Final artifact documented as `bert_out/bert-base-uncased_ctx.onnx`. `build.py:714` writes `final_path = resolved_dir / "model.onnx"` for non-cached builds. The file is `bert_out/model.onnx`; the ctx-name variant does not exist. Step 3 `winml perf` reference to `bert-base-uncased_ctx.onnx` will also fail.

### docs/samples/convnext-primitives.md
- CPU (`--device cpu`) and GPU (`--device gpu`) compile steps documented as producing `_cpu_ctx.onnx` and `_dml_ctx.onnx`. Both are wrong: `for_cpu()` and `for_dml()` set `enable_ep_context=False` (`configs.py:165,175`); no `_ctx.onnx` is written. The GPU perf tab referencing `convnext_int8_dml_ctx.onnx` will fail.
- Note claims "`--device` does not accept `auto` on `winml eval`". `eval.py:69` lists `auto` as a valid choice with `default="auto"`.

### docs/tutorials/npu-convnext.md
- CPU artifact named `convnext_int8_cpu_ctx.onnx` (steps 7–8): `for_cpu()` sets `enable_ep_context=False` (`configs.py:165`); no such file is produced. The Step 8 perf command referencing it will fail.
- Python 3.10 listed in prerequisites; `pyproject.toml:13` requires `>=3.11,<3.12`.

### docs/concepts/perf-and-monitoring.md
- `--device` described as accepting only `cpu`, `gpu`, `npu`. `perf.py:1113` uses `device_option(include_auto=True, default="auto")`; `auto` is valid and is the default.
- Default output path stated as `{model_slug}_perf.json` (current directory). Source `perf.py:871` writes to `~/.cache/winml/perf/<slug>/<timestamp>.json`.

---

## Important (misleading or stale)

### docs/getting-started/installation.md
- "No NPU?" callout claims `winml eval` accepts only `cpu|gpu|npu` (no `auto`). `eval.py:69` defines `click.Choice(["auto", "cpu", "gpu", "npu"])` — `auto` is valid.

### docs/getting-started/end-to-end.md
- Sample `sys` output shows `QNNExecutionProvider -> NPU`. `get_ep_device_map()` returns `"npu/gpu"` for QNN (`device.py:49`, `constants.py:183`); actual rendered output would be `QNNExecutionProvider -> NPU/GPU`.

### docs/commands/build.md
- `--no-compile/--compile` documented as a simple `--no-compile` flag; source `build.py:275–282` is a boolean toggle pair; `--compile` (force enable) is undocumented.
- `--trust-remote-code` absent from flag table; `build.py:312–314` defines it.
- `--max-optim-iterations` table default shown as `3`; `build.py:309` sets `default=None` (3 is enforced inside pipeline helpers, not at Click layer).

### docs/commands/config.md
- `--no-compile` framing is backwards (see Critical). The entire usage example `winml config ... --no-compile` implies the flag does work when it is a no-op (already the default).

### docs/commands/hub.md
- "How it works" describes per-EP latency stats and accuracy verdicts (PASS/AT_RISK/REGRESSION) that do not appear anywhere in `catalog.py`. The rendered catalog shows only Model, Task, Size, Model Type columns.
- `--ep` and `--device` filter flags (`catalog.py:377–385`) absent from the flag table entirely.

### docs/commands/analyze.md
- `--ep` valid special values `"all"` and `"auto"` not mentioned; `analyze.py:634` includes both in `Choice`. Related: "Omitting `--ep` analyzes every EP" (pitfall line 82) repeats the incorrect default claim.
- `--model` short form `-m` shown with empty Short column; `cli.py:68` defines `"--model", "-m"`.
- `--verbose/-v`, `--quiet/-q`, and `--config/-c` absent from flag table; all defined via decorators (`analyze.py:651–652`).

### docs/commands/optimize.md
- `--verbose/-v` absent from flag table; `optimize.py:180–185` defines it.
- `--model` Short column is empty; `optimize.py:167` defines `-m`.
- "Configuration precedence" describes 4 levels (including "preset"); source has 3 levels (`optimize.py:363–383`). The preset level does not exist.

### docs/commands/inspect.md
- `-v`/`--verbose` absent from flag table; `inspect.py:78–83` defines it.

### docs/commands/perf.md
- `--compare-devices` listed as "Not yet implemented" but the flag is not registered at all in `perf.py`. Passing it will error, not silently be ignored.
- `--op-tracing` documented as a user-facing feature; `perf.py:1183` decorates it `hidden=True`.
- Default output path documented as `{model_slug}_perf.json`; actual path is `~/.cache/winml/perf/<slug>/<timestamp>.json` (`perf.py:871`).

### docs/commands/sys.md
- `--verbose/-v` absent from flag table; `sys.py:653–659` defines it. Verbose mode surfaces Backend SDKs and Export Readiness sections (`sys.py:392`).

### docs/concepts/config-and-build.md
- `WinMLBuildConfig` described as having five sub-configs; `config/build.py:132–138` also has `eval: WinMLEvaluationConfig | None` and `auto: bool`.

### docs/samples/npu-convnext.md
- Step 7 OpenVINO artifact named `convnext_int8_openvino_ctx.onnx`; `compile.py:230` uses `{stem}_{device}_ctx.onnx` where device is the resolved device string (`"npu"`), not the EP name. Actual filename would be `convnext_int8_npu_ctx.onnx`.

### docs/concepts/perf-and-monitoring.md
- `--monitor` described as streaming "NPU utilisation". Source resolves from model device at runtime (`perf.py:409`); it monitors whichever device is being benchmarked, not NPU specifically.
- `--op-tracing` documented as a supported feature; it is `hidden=True` (`perf.py:1183`).

### docs/commands/overview.md
- `src/winml/modelkit/commands/_options.py` cited as "canonical contract" for global flags. This file does not exist (`_options.py` absent from `commands/` directory). Global flags are in `cli.py`.

---

## Rejected (claimed by an agent but not a real defect)

### docs/concepts/quantization.md
- ["`_KNOWN_PRECISIONS` from `_options.py`" is fabricated] — REJECTED: The claim itself is being kept as Critical because both `_KNOWN_PRECISIONS` and `_options.py` are absent from the codebase (confirmed `_options.py` not in commands/, and `grep` for `_KNOWN_PRECISIONS` returns nothing). The actual symbol is `_NAMED_PRECISIONS` at `precision.py:71`. The finding is genuine, not a false positive.

### docs/concepts/compile-and-epcontext.md
- [External EPContext described as "default"] — REJECTED as false positive: `EPConfig.embed_context: bool = False` at `configs.py:46` confirms external is the default. Doc is correct.
- [`--no-validate` flag] — REJECTED as false positive: `compile.py:72–74` defines `--validate/--no-validate`; the doc's use of `--no-validate` correctly names the negative form of the toggle.

### docs/getting-started/end-to-end.md
- [`QNN_SDK_ROOT` from environment] — REJECTED: `build.py` has no `--qnn-sdk-root` flag (confirmed zero hits). Reading from environment is the correct description.
- [`--device auto` priority order "NPU first, then GPU, then CPU"] — REJECTED: `device.py:62` confirms `_DEVICE_PRIORITY = ("npu", "gpu", "cpu")`. Claim is correct.

### docs/commands/compile-and-epcontext.md
- [`for_vitisai` and `for_qnn` described as interchangeable "QNN-family EPs"] — REJECTED as below threshold: both produce EPContext, the distinction noted by the agent is a simplification, not a user-facing error.

### docs/concepts/quantization.md
- [`int16` weight dtype listed as `int16`] — REJECTED: `_WEIGHT_TYPE["int16"] = "int16"` at `precision.py:43`. Doc row is correct.

### docs/commands/eval.md
- [`winml eval` loads via `WinMLAutoModel`] — this claim is kept as Important (class name misrepresents implementation) but the agent's flag about missing flags is correct and retained above.
