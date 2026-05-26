# winml analyze

> Verify an ONNX model is compatible with a target execution provider before deployment.

## When to use this

Use `winml analyze` before running the full build pipeline to confirm that your ONNX model's operators are supported by the intended execution provider and device. It surfaces operator gaps and actionable recommendations early, saving time that would otherwise be spent on a failed compile or quantize run.

## Synopsis

```bash
$ winml analyze [options]
```

## Flags

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--model` | | `PATH` | *(required)* | Path to the ONNX model file to analyze. |
| `--ep` | | choice | *(none)* | Target execution provider. Accepts full names (`QNNExecutionProvider`, `OpenVINOExecutionProvider`, `VitisAIExecutionProvider`) or short aliases (`qnn`, `ov`/`openvino`, `vitis`/`vitisai`). When omitted, all supported EPs are analyzed. |
| `--device` | | `CPU\|GPU\|NPU` | `NPU` | Target device type. Filters the analysis for the named device class when `--ep` is also supplied. When omitted, defaults to NPU. |
| `--output` | | `PATH` | *(none)* | Save the full JSON result to a file in addition to printing the console summary. |
| `--information` / `--no-information` | | flag | enabled | Include detailed per-operator recommendations and remediation hints in the output. Pass `--no-information` for a compact pass/fail summary. |
| `--htp-metadata` | | `PATH` | *(none)* | Path to an HTP metadata JSON file. Enables enhanced Qualcomm-specific pattern extraction when targeting QNN. |
| `--run-unknown-op` / `--no-run-unknown-op` | | flag | enabled | Attempt to run operators unknown to the EP locally to infer shape and type information. Disable when the local machine lacks the required libraries. |
| `--save-node` | | `partial\|unsupported` | *(none)* | Save partial or unsupported node subgraphs to disk for further investigation. Can be specified multiple times: `--save-node partial --save-node unsupported`. |

## How it works

`winml analyze` loads the ONNX model and runs a static analysis pass via `ONNXStaticAnalyzer`. It checks each operator in the graph against the EP's capability list, classifies nodes as fully supported, partially supported, or unsupported, and optionally runs unknown operators locally to infer missing shape information. The command exits with code `0` when all operators are supported, `1` when at least one operator is unsupported or only partially supported, and `2` on any input or runtime error — making it safe to use in CI pipelines with exit-code checks.

## Examples

Analyze against all supported EPs using the default NPU device:

```bash
$ winml analyze --model microsoft/resnet-50.onnx
```

```text
Analyzing microsoft/resnet-50.onnx against all supported EPs...

QNNExecutionProvider (NPU): FULLY SUPPORTED
  Operators checked : 142
  Unsupported       : 0
  Partial           : 0

OpenVINOExecutionProvider (NPU): FULLY SUPPORTED
  Operators checked : 142
  Unsupported       : 0
  Partial           : 0
```

Check QNN NPU support using the short alias:

```bash
$ winml analyze --model bert-base-uncased.onnx --ep qnn --device NPU
```

Check Intel OpenVINO GPU support and print operator-level recommendations:

```bash
$ winml analyze --model bert-base-uncased.onnx --ep ov --device GPU --information
```

Save the full JSON result for offline inspection while still printing the console summary:

```bash
$ winml analyze --model facebook/convnext-tiny-224.onnx --output results.json
```

Use QNN with HTP metadata for enhanced Qualcomm pattern extraction:

```bash
$ winml analyze --model bert-base-uncased.onnx \
    --ep QNNExecutionProvider --device NPU \
    --htp-metadata htp_metadata.json
```

## Common pitfalls

- **Omitting `--ep` analyzes every EP** — this is slower and may produce confusing output when one EP shows unsupported operators that another handles fine. Specify `--ep` when you know your target hardware.
- **Exit code 1 is not a hard failure** — it means at least one operator is unsupported, not that the model cannot run at all. Many EPs fall back unsupported nodes to the CPU EP automatically; review the recommendations before deciding to restructure the model.
- **`--htp-metadata` is QNN-specific** — passing a QNN HTP metadata file while targeting a different EP has no effect. Ensure the EP and metadata file correspond to the same hardware.
- **`--no-run-unknown-op` may widen the unsupported list** — if local execution is disabled, operators whose support cannot be verified statically are conservatively marked as unsupported.
- **The model path must point to an existing `.onnx` file** — symbolic HuggingFace model IDs are not accepted; export the model first with `winml export`.

## See also

- [eps-and-devices.md](../concepts/eps-and-devices.md) — background on ONNX operators and execution providers
- [export.md](export.md) — convert a HuggingFace model to ONNX before analyzing
- [compile.md](compile.md) — compile the model for the target EP after analysis passes
- [sys.md](sys.md) — list EPs available on the current machine
