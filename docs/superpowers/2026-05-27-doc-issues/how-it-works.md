# Issues: docs/concepts/how-it-works.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical (factually wrong; user would hit error)
- (none)

## Important (misleading or stale claim)
- Line 80: Doc says `winml build` auto-detects ONNX vs HF and calls "`build_hf_model` or `build_onnx_model`". This is inaccurate at the CLI layer. The build command (`src/winml/modelkit/commands/build.py`) orchestrates stages directly via `_build_hf_pipeline()` / `_build_onnx_pipeline()` inline functions. The named public API functions `build_hf_model` / `build_onnx_model` (from `src/winml/modelkit/build/hf.py` and `build/onnx.py`) are only called in module-mode (`_build_modules()`), not in the single-model code path. Telling readers "calls `build_hf_model` or `build_onnx_model`" misrepresents the actual dispatch.

- Line 88: Example flag `--no-optimize` is valid (`src/winml/modelkit/commands/build.py:300`), but the comment "Skip optimization (for pre-quantized input)" is misleading. The source docstring says "Skip optimization (for pre-quantized ONNX models)" (`build.py:303`), and the flag is general-purpose (not limited to pre-quantized inputs). The doc's narrower framing could confuse users with other reasons to skip optimization.

## Minor (style, polish, low-impact)
- Line 12: Claims the pipeline API "powers `WinMLAutoModel.from_pretrained()`". `WinMLAutoModel` exists (`src/winml/modelkit/models/auto.py`) but the connection to the pipeline described here is not verifiable from the source at the cited commit; may be aspirational or referring to an internal API not exposed in this path.

- Lines 116–122: Cross-links — `../commands/build.md`, `../commands/export.md`, `eps-and-devices.md`, and `config-and-build.md` all resolve to files that exist on disk. No broken links.

## Verified correct (anchored claims you checked)
- Lines 88-91: `--no-quant` and `--no-compile` flags exist in `src/winml/modelkit/commands/build.py:274` and `279-282` respectively. `--no-optimize` exists at line 300.
- Lines 99-105: `WinMLBuildConfig` structure (loader/export/optim/quant/compile) matches `src/winml/modelkit/config/build.py:97-138`.
- Lines 109-110: Setting `quant` or `compile` to null skips that stage; confirmed by `src/winml/modelkit/commands/build.py:948-949` (quant) and `src/winml/modelkit/commands/build.py:1038-1039` (compile).
- Line 113: Config file written after optimize stage; confirmed by `src/winml/modelkit/commands/build.py:1192` (`config_path.write_text(...)`).
