# Issues: docs/concepts/eps-and-devices.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical (factually wrong; user would hit error)
- (none)

## Important (misleading or stale claim)
- Line 13 (table row for `QNNExecutionProvider`): The table lists QNN's device as `npu` only. However, `src/winml/modelkit/utils/constants.py:184` declares `"QNNExecutionProvider": ("npu", "gpu")` — QNN also supports `gpu` as a secondary device. The table is therefore incomplete and will mislead users who want to run QNN on a GPU target.

- Lines 35-38: The `--device` description says the default is `auto` and it picks "NPU > GPU > CPU". The source at `src/winml/modelkit/commands/build.py:289-290` sets `default="auto"` for `--device` in the build command, and `src/winml/modelkit/commands/analyze.py:645` also defaults to `"auto"`. Priority logic `NPU > GPU > CPU` is consistent with `EP_SUPPORTED_DEVICES` key order in `src/winml/modelkit/utils/constants.py:178-187`. So far accurate. However, `--device` on `winml analyze` accepts `CPU/GPU/NPU/all/auto` (uppercase; `src/winml/modelkit/commands/analyze.py:644-648`), not lowercase as shown in the doc examples on lines 37-40. The CLI itself normalizes case, so commands work, but showing `--device npu` (lowercase) in examples while the `type=click.Choice([*SUPPORTED_DEVICES, ...])` enumerates uppercase `"CPU"`, `"GPU"`, `"NPU"` (`src/winml/modelkit/utils/constants.py:163-167`) could be confusing. Since Click's `case_sensitive=False` is set on the analyze command, the examples aren't wrong, but readers inspecting help output will see uppercase choices.

- Lines 48-53: Example shows `winml analyze --model model.onnx --ep QNNExecutionProvider --device npu`. The `analyze` command uses `--model` (confirmed at `src/winml/modelkit/utils/cli.py:69`), not `--model-path` or another variant. The example is correct in flag name.

## Minor (style, polish, low-impact)
- Lines 57-63: All cross-links (`graphs-and-ir.md`, `weight-and-activation.md`, `../commands/sys.md`, `../commands/analyze.md`) resolve to files on disk.
- Line 22: `winml sys --list-ep` — flag `--list-ep` confirmed at `src/winml/modelkit/commands/sys.py:668-671`.

## Verified correct (anchored claims you checked)
- Lines 11-19 (EP table): `CPUExecutionProvider`, `DmlExecutionProvider`, `MIGraphXExecutionProvider`, `NvTensorRTRTXExecutionProvider`, `OpenVINOExecutionProvider`, `QNNExecutionProvider`, `VitisAIExecutionProvider` — all seven are in `EPName` Literal at `src/winml/modelkit/utils/constants.py:24-33`.
- Table: `OpenVINOExecutionProvider` listed as supporting `npu / gpu / cpu` — confirmed by `"OpenVINOExecutionProvider": ("npu", "gpu", "cpu")` at `src/winml/modelkit/utils/constants.py:185`.
- Table: `VitisAIExecutionProvider` listed as `npu` only — confirmed by `"VitisAIExecutionProvider": ("npu",)` at `src/winml/modelkit/utils/constants.py:183`.
- Table: `DmlExecutionProvider` listed as `gpu` only — confirmed by `"DmlExecutionProvider": ("gpu",)` at `src/winml/modelkit/utils/constants.py:186`.
- Table: `MIGraphXExecutionProvider` listed as `gpu` only — confirmed by `"MIGraphXExecutionProvider": ("gpu",)` at `src/winml/modelkit/utils/constants.py:182`.
- Table: `NvTensorRTRTXExecutionProvider` listed as `gpu` only — confirmed by `"NvTensorRTRTXExecutionProvider": ("gpu",)` at `src/winml/modelkit/utils/constants.py:179`.
- Lines 44-45: `--ep` accepts aliases `qnn`, `vitisai`, `dml`, `openvino` — confirmed in `EP_ALIASES` at `src/winml/modelkit/utils/constants.py:59-69`.
