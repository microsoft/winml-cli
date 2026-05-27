# Issues: docs/getting-started/end-to-end.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical
- Artifact filename pattern is wrong for DML and CPU (end-to-end.md:123–125). The doc claims the GPU artifact is named `convnext_tiny_dml_ctx.onnx` and the CPU artifact is `convnext_tiny.onnx`. Source: `compiler/configs.py` `for_dml()` sets `enable_ep_context=False` and `for_cpu()` also sets `enable_ep_context=False`. When `enable_ep_context=False`, `compile.py` `_finalize_output` is never called (the `if ep_config.enable_ep_context:` guard in `CompileStage.process`), meaning no `_ctx.onnx` is produced and `winml build --no-compile` leaves only the quantized ONNX. Neither `convnext_tiny_dml_ctx.onnx` nor a special CPU variant filename is produced; the DML and CPU "compile" steps are no-ops that return `None` from `for_provider`. The correct behavior is that only QNN (and OpenVINO, VitisAI, NvTensorRTRTX) produce `_ctx.onnx` artifacts; DML/CPU compile is skipped entirely.
- `winml build` `--no-quant` / `--no-compile` flags exist in source (build.py:270, 276), but the doc also mentions `--no-optimize` (end-to-end.md:106) — this flag exists (`build.py:300`), so that claim is correct. However, the doc omits any mention that `--no-compile/--compile` is actually a toggle pair and `--compile` can be used to force enable compilation (build.py:277–280). Minor gap but not a factual error.

## Important
- `winml build` warning box (end-to-end.md:111–113): states the build reads `QNN_SDK_ROOT` from the environment. This is correct for the `winml build` wrapper, which does NOT expose `--qnn-sdk-root` (build.py has no such option). The doc is consistent with the source. No error here.
- `--device auto` priority order claimed as "NPU first, then GPU, then CPU" (end-to-end.md:7–8): confirmed correct by `sysinfo/device.py` `_DEVICE_PRIORITY: tuple[str, ...] = ("npu", "gpu", "cpu")`.
- Tabbed `sys` output EP names (end-to-end.md:54–57): `QNNExecutionProvider -> NPU`, `DmlExecutionProvider -> GPU`, `CPUExecutionProvider -> CPU`. Cross-referencing `EP_SUPPORTED_DEVICES` in `constants.py`: `QNNExecutionProvider` maps to `("npu", "gpu")` not just `"npu"`. The display in `_output_ep_text` shows the first device from `get_ep_device_map()` which joins with `/`, so it would render `QNNExecutionProvider -> NPU/GPU`, not just `NPU`. The sample output in the doc shows only `-> NPU`, which is inaccurate.

## Minor
- Step 3 perf command uses placeholder `<artifact>.onnx` (end-to-end.md:119). Given the critical artifact naming issue above, the example filenames shown in the tabbed blocks (`convnext_tiny_qnn_ctx.onnx`, `convnext_tiny_dml_ctx.onnx`, `convnext_tiny.onnx`) are not the actual file stems that `winml build` produces for a model named `convnext-tiny-224`. The actual stem would depend on the slug generated from the model ID (not verified here), but the `_dml_ctx` and plain `.onnx` names are definitely wrong per the critical issue above.
