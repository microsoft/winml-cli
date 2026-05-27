# Issues: docs/tutorials/npu-convnext.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical
- Step 7 CPU compile artifact named `convnext_int8_cpu_ctx.onnx` (npu-convnext.md:164): `compiler/configs.py` `for_cpu()` sets `enable_ep_context=False`, so `CompileStage._finalize_output` is never invoked and no `_cpu_ctx.onnx` file is written. The CPU compile step is silently skipped by `for_provider()` returning `None` when `enable_ep_context=False`. The CPU tab in Step 7 describes a compile command that produces no artifact, and the named output file does not exist.
- Step 8 CPU perf command references `convnext_int8_cpu_ctx.onnx` (npu-convnext.md:190): this file is never produced (same root cause as above). The CPU benchmark tab would fail to find the input model.
- Step 9 eval uses `--device npu` (npu-convnext.md:224): `eval.py` declares `--device` as `click.Choice(["auto", "cpu", "gpu", "npu"])` — `npu` is a valid value. However, the tutorial is evaluating `convnext_int8.onnx` (the quantized float ONNX before compilation) on the NPU. This will attempt to run the uncompiled model through QNN EP, which requires JIT compilation at load time and may fail or be extremely slow. This is a usage problem but `npu` is a legal value, so it is not a flag-existence error.

## Important
- Step 7 OpenVINO compile (npu-convnext.md:155): `winml compile -m convnext_int8.onnx --device npu --ep openvino`. In `compile.py`, `--device` accepts `["auto", "npu", "gpu", "cpu"]` and `--ep` accepts EP aliases. `OpenVINOExecutionProvider` maps to `("npu", "gpu", "cpu")` in `EP_SUPPORTED_DEVICES`, so `--device npu --ep openvino` is a valid combination. No error here.
- Step 7 claims OpenVINO produces `convnext_int8_openvino_ctx.onnx` (npu-convnext.md:164): `for_openvino()` sets `enable_ep_context=True`, so an EPContext file is produced. The filename pattern `{stem}_{device}_ctx.onnx` is used in `CompileStage._finalize_output` where `device` comes from the resolved device string. With `--device npu`, `device="npu"`, so the file would be `convnext_int8_npu_ctx.onnx`, not `convnext_int8_openvino_ctx.onnx`. The EP name is not used in the filename; the device name is.
- Section B `winml build` command (npu-convnext.md:239): `uv run winml build -c convnext_config.json -m facebook/convnext-tiny-224 -o convnext_out/`. Source `build.py` uses `-c` (config), `-m` (model), `-o` (output-dir). The flag signatures match. No error.
- Section B states "The QNN SDK path is read from the `QNN_SDK_ROOT` environment variable, not from the config or CLI flags." (npu-convnext.md:257): correct for `winml build` — `build.py` has no `--qnn-sdk-root` option. But note: `winml compile` *does* expose `--qnn-sdk-root` (compile.py:89–93). The tutorial does not use `winml compile --qnn-sdk-root` so this nuance is not wrong in context, but it may confuse users who read both pages.
- Prerequisites list Python 3.10 (npu-convnext.md:22): `pyproject.toml` requires `>=3.11,<3.12`. This propagates the same Python version error found in installation.md.

## Minor
- Section B perf command at the end uses `convnext_out/model.onnx` (npu-convnext.md:262): `winml build` does not write a file named `model.onnx`; it writes the compiled artifact under its EP-derived name (e.g., `convnext_int8_npu_ctx.onnx`). The placeholder path is misleading — users must look up the actual output filename from the build log.
