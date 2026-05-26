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
| `--device` | `-d` | choice | `npu` | Target device: `auto`, `npu`, `gpu`, or `cpu`. |
| `--ep` | | choice | `None` | Force a specific execution provider, overriding device-to-provider mapping. Choices: `cpu`, `cuda`, `dml`, `migraphx`, `openvino`, `qnn`, `tensorrt`, `vitisai`. |
| `--no-quant` | | flag | `false` | Flag retained for compatibility; quantization is no longer performed during compile. Use `winml quantize` beforehand. |
| `--no-validate` | | flag | `false` | Skip validation of the compiled model after compilation. |
| `--compiler` | | choice | `ort` | Compiler backend: `ort` (ONNX Runtime) or `qairt` (Qualcomm AI Runtime Tools). |
| `--qnn-sdk-root` | | path | `None` | Path to the QAIRT/QNN SDK root directory. Required when `--compiler qairt` is set. |
| `--embed` | | flag | `false` | Embed the EP context blob inside the ONNX file instead of writing a separate `.bin` file. |
| `--list` | | flag | `false` | List available compiler backends for the selected device and exit without compiling. |
| `--help` | `-h` | flag | | Show this message and exit. |

## How it works

`winml compile` resolves the target execution provider from `--device` and
`--ep`, then calls the winml-cli compiler API to hand the ONNX graph to the
EP's offline compilation toolchain. For the default NPU target, ONNX Runtime's
QNN EP generates a binary `.bin` context file (or embeds it inline with
`--embed`) that encodes the hardware-optimized execution plan, eliminating
graph partitioning at load time. When `--compiler qairt` is used, the
Qualcomm AI Runtime Tools SDK is invoked directly (requires `--qnn-sdk-root`).
An optional post-compilation validation pass runs a forward pass through the
target EP; skip it with `--no-validate` when the target hardware is absent.

## Examples

```bash
# Compile for NPU (default device and compiler)
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

```bash
# Compile using the QAIRT SDK and skip post-compilation validation
winml compile -m facebook_convnext_qdq.onnx \
  --compiler qairt \
  --qnn-sdk-root /opt/qnn-sdk \
  --no-validate
```

## Common pitfalls

- **`--no-quant` is a no-op in the current release.** Quantization is no longer
  performed during compile; run `winml quantize` on your model first, then pass
  the QDQ model to this command.
- **`--compiler qairt` requires `--qnn-sdk-root`.** Without a valid SDK path,
  compilation will fail immediately with a missing-executable error.
- **`--embed` inflates the `.onnx` file significantly.** Embedding the EP
  context produces a single portable file but can make it impractical to open or
  inspect the ONNX graph with standard tooling.
- **Validation requires the target hardware.** The post-compilation validation
  step runs an actual inference pass; on a machine without the NPU driver or the
  relevant EP installed, always pass `--no-validate`.
- **`--device` default is `npu`, not `auto`.** Unlike other commands, compile
  defaults to NPU targeting. Pass `--device cpu` or `--device gpu` explicitly
  when targeting other hardware.

## See also

- [winml quantize](quantize.md)
- [winml build](build.md)
- [ONNX and execution providers](../concepts/eps-and-devices.md)
