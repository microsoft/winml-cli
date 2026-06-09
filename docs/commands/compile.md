# winml compile

> Compile an ONNX model to an EP-specific format for fast runtime loading.

## When to use this

Use `winml compile` as the final pipeline stage after `winml quantize` to
produce an execution-provider-native artifact (for example, a QNN EPContext
model) that loads faster and avoids online graph compilation at inference time.

## Synopsis

```bash
$ winml compile [options]
```

## Flags

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--model` | `-m` | path | *(required unless `--list`)* | Input ONNX model file. |
| `--output-dir` | | path | same dir as input | Directory to write compiled output artifacts. |
| `--device` | `-d` | choice | `auto` | Target device: `auto`, `npu`, `gpu`, or `cpu`. |
| `--ep` | | choice | `None` | Force a specific execution provider, overriding device-to-provider mapping. Choices: `cpu`, `cuda`, `dml`, `migraphx`, `openvino`, `qnn`, `tensorrt`, `vitisai`. |
| `--no-validate` | | flag | `false` | Skip validation of the compiled model after compilation. |
| `--compiler` | | choice | `ort` | Compiler backend: `ort` (ONNX Runtime) or `qairt` (Qualcomm AI Runtime Tools). |
| `--qnn-sdk-root` | | path | `None` | Path to the QNN SDK root directory. |
| `--embed` | | flag | `false` | Embed the EP context blob inside the ONNX file instead of writing a separate `.bin` file. |
| `--list` | | flag | `false` | List available compiler backends for the selected device and exit without compiling. |
| `--help` | `-h` | flag | | Show this message and exit. |

## How it works

`winml compile` resolves the target execution provider from `--device` and
`--ep`, then calls the winml-cli compiler API to hand the ONNX graph to the
EP's offline compilation toolchain. When `--device auto` (the default), the
target EP is determined by auto-detecting available hardware. For NPU targets,
ONNX Runtime's QNN EP generates a binary `.bin` context file (or embeds it
inline with `--embed`) that encodes the hardware-optimized execution plan,
eliminating graph partitioning at load time. An optional post-compilation
validation pass runs a forward pass through the
target EP; skip it with `--no-validate` when the target hardware is absent.

## Examples

```bash
# Compile with auto device detection (default compiler)
winml compile -m resnet50_qdq.onnx
```

```text
Input: resnet50_qdq.onnx
Device: npu
Provider: qnn
Compiler: ort

Compiling model...

Success! Model compiled
Output: resnet50_qdq_ctx.onnx
Compile time: 12.40s
Total time: 13.05s
```

```bash
# List available compiler backends for NPU before committing to a run
winml compile --list --device npu
```

```bash
# Compile a pre-quantized BERT model for NPU with context embedded inline
winml compile -m bert-base-uncased_qdq.onnx --embed
```

```bash
# Compile for GPU using the MIGraphX execution provider
winml compile -m microsoft_resnet50.onnx --device gpu --ep migraphx
```

## Common pitfalls

- **`--embed` inflates the `.onnx` file significantly.** Embedding the EP
  context produces a single portable file but can make it impractical to open or
  inspect the ONNX graph with standard tooling.
- **Validation requires the target hardware.** The post-compilation validation
  step runs an actual inference pass; on a machine without the NPU driver or the
  relevant EP installed, always pass `--no-validate`.
- **`--device auto` auto-detects the best available hardware.** Pass `--device npu`,
  `--device gpu`, or `--device cpu` explicitly when targeting specific hardware
  regardless of what is auto-detected.

## See also

- [winml quantize](quantize.md)
- [winml build](build.md)
- [ONNX and execution providers](../concepts/eps-and-devices.md)
