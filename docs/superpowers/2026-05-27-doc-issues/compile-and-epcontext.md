# Issues: docs/concepts/compile-and-epcontext.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical (factually wrong; user would hit error)

- **`--no-quant` on `winml compile`** (line 29): The doc says "`winml compile` also accepts `--no-quant` to skip the quantization pass for already-quantized (QDQ) models." There is no `--no-quant` flag on `winml compile`. The `commands/compile.py` file was fully read and contains no `--no-quant` option. This is a flag that exists on `winml build`, not `winml compile`. A user passing `--no-quant` to `winml compile` will get `Error: No such option: --no-quant`.

## Important (misleading or stale claim)

- **`--ep qnn` and `--ep vitisai` described as "QNN-family EPs"** (line 11): The doc lumps these together as both producing "EP context blobs". Source shows `WinMLCompileConfig.for_provider()` treats them distinctly — `vitisai` uses `VitisAIExecutionProvider` and `qnn` uses `QNNExecutionProvider` (`commands/compile.py` lines 214-221, `compiler/configs.py` lines 209-221). Both do produce EPContext, but the doc's grouping as interchangeable is a simplification that may mislead users.

- **External EPContext described as "default"** (lines 17-21): Doc says "By default the blob is written as a sidecar `.bin` file alongside the `.onnx`." Source confirms `embed_context: bool = False` as default in `EPConfig` (`compiler/configs.py` line 46), so external is indeed the default. Correct.

- **`--embed` flag** (line 17): Doc says "Passing `--embed` instead inlines the blob". Source confirms `--embed` is a flag on `winml compile` (`commands/compile.py` lines 96-99), which sets `embed_context=True`. Correct.

- **`--compiler qairt` and `--qnn-sdk-root`** (line 13): Doc says "select `--compiler qairt` and point `--qnn-sdk-root`". Source confirms both flags on `winml compile` (`commands/compile.py` lines 83-93). Correct.

- **`--no-validate` flag** (line 34): The actual flag on `winml compile` is `--validate/--no-validate` (source: `commands/compile.py` lines 72-74). The doc says "The `--no-validate` flag skips that pass." This is accurate — `--no-validate` is the negative form of the `--validate/--no-validate` pair.

## Minor (style, polish, low-impact)

- **Validation described as "default: enabled"** (line 33): Confirmed — `WinMLCompileConfig.validate: bool = True` (`compiler/configs.py` line 86) and `--validate/--no-validate` defaults to `True` (`commands/compile.py` line 74). Correct.

- **Cross-links** `[eps-and-devices.md]`, `[analyze-and-optimize.md]`, `[../commands/compile.md]`, `[../commands/build.md]` (lines 39-43): All target files exist.

## Verified correct (anchored claims you checked)

- `winml compile` `--ep` flag exists → `commands/compile.py` lines 66-69
- `winml compile` `--device` flag with auto/npu/gpu/cpu choices → `commands/compile.py` lines 58-65
- `winml compile` `--compiler` flag with choices `["ort", "qairt"]` → `commands/compile.py` lines 83-87
- `winml compile` `--qnn-sdk-root` flag exists → `commands/compile.py` lines 88-93
- `winml compile` `--embed` flag exists → `commands/compile.py` lines 96-99
- `winml compile` `--validate/--no-validate` flag exists, default enabled → `commands/compile.py` lines 72-74
- `EPConfig.embed_context` defaults to `False` (external sidecar) → `compiler/configs.py` line 46
- `EPConfig.enable_ep_context` defaults to `True` → `compiler/configs.py` line 45
- Compiler backend `ort` is the default → `commands/compile.py` line 87 (`default="ort"`)
- No `wmk` or `ModelKit` strings in prose → verified by grep
