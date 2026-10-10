# winml compile

> Compile ONNX to an EP-specific format, or explicitly compile CGC Input IR to device-targeted Output IR.

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
| `--model` | `-m` | path | *(required unless `--list`)* | Input ONNX model, or complete CGC Input IR in explicit IR mode. |
| `--input-format` | | choice | `onnx` | Input representation: `onnx` or `cgc-input-ir`. CGC input content is parsed and validated, not inferred from its extension. |
| `--target` | | choice | `epcontext` | Output representation: `epcontext` or `cgc-output-ir`. |
| `--output` | `-o` | path | — | Output file path (e.g., `model_compiled.onnx`). Takes precedence over `--output-dir`. |
| `--output-dir` | | path | same dir as input | Directory to write compiled output artifacts. |
| `--device` | `-d` | choice | `auto` | Target device: `auto`, `npu`, `gpu`, or `cpu`. |
| `--device-luid` | | string | — | Explicit CGC IR mode only: GPU adapter LUID from `winml sys`, in `0xHHHHHHHH_0xLLLLLLLL` form. |
| `--ep` | | `TEXT` | — | Force a specific execution provider, overriding device-to-provider mapping. Accepts full names (e.g., `QNNExecutionProvider`) or aliases (`qnn`, `dml`, `openvino`, `vitisai`, `migraphx`, `cpu`, `nvtensorrtrtx`). |
| `--ep-options` | | `KEY=VALUE` | — | Execution-provider option for the compilation session. Repeat for multiple options; later duplicate keys win. Explicit CLI values override matching `compile.provider_options` keys from `--config`. |
| `--validate` / `--no-validate` | | flag | `--validate` | EPContext: post-compilation inference validation. CGC Output IR: reload, pipeline build and I/O schema validation, without inference. |
| `--compiler` | | choice | `ort` | EPContext backends: `ort`, `ort_session`, `qairt`. `winml-runtime` is only for the explicit CGC Input IR / Output IR mode. |
| `--qnn-sdk-root` | | path | `None` | Path to the QNN SDK root directory. |
| `--embed/--no-embed` | | flag | `false` | Embed the EP context blob inside the ONNX file instead of writing a separate `.bin` file. |
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

If a provider option names a compiler input file, list its key in
`compile.provider_option_file_keys` in the JSON config. The CLI canonicalizes
that option path and fingerprints its contents for EPContext cache identity;
other provider-option strings are always passed through unchanged.

## CGC Input IR to Output IR

This explicit mode uses the Windows ML Runtime API directly, without an ONNX
Runtime EP. All three selectors are required so the input representation,
output representation and compiler implementation are unambiguous:

```powershell
winml compile `
  -m input.mlir `
  --input-format cgc-input-ir `
  --target cgc-output-ir `
  --compiler winml-runtime `
  --device gpu `
  -o output.mlir
```

The command parses the input with the preview wheel's FoundryToolbox, rejecting
ONNX, malformed IR and already compiled Output IR. Both textual and bytecode
CGC Input IR are accepted. Supply a complete executable model with its weights,
not a graph-only report projection.

Compilation calls the selected GPU target's Runtime
`ModelCompiler.compile_to_file()` with `CompiledModelForm.DEVICE_TARGETED`.
The saved artifact is **MLIR bytecode**, even when named `.mlir`; text dumping
is not provided by this command. `.mlirbc` is also accepted as an output suffix.
No optimization or quantization pipeline runs.

This first version supports GPU only (the default in explicit IR mode). A single
GPU is selected automatically; multiple GPUs require `--device-luid`. GPU
discovery uses DXCore, not EP discovery or provider installation.

Default `--validate` reloads the compiled artifact, builds a Runtime pipeline and
checks that its I/O schema matches the input. Unlike the EPContext validation
path, this does not execute inference or certify accuracy. `--no-validate` skips
that reload/build check; genuine Input IR parsing and compilation remain required.
Run `winml perf` separately for performance.

The command publishes the artifact and any Runtime-produced resource companions
only after successful compilation and requested validation. It preserves an
existing input's `<stem>_metadata.json` I/O bindings as output metadata and writes
`<output-stem>_compile.json` with input/artifact hashes, output encoding, GPU LUID
and validation status. Metadata and resource collisions also require
`--overwrite`; the input and its metadata are never replaced. Without `-o`, the
default is `<input-stem>_output.mlir` in `--output-dir` or beside the input.

`--ep`, `--ep-options`, `--embed`/`--no-embed`, `--qnn-sdk-root`, `--list`,
`--config` and multiple `-m` inputs are rejected in this mode. Use the existing
ONNX-to-EPContext mode for EP compilation. Omitting the new selectors preserves
the original command behavior.

The preview `windowsml` payload must include both Runtime and FoundryToolbox
APIs compatible with the active CLI. If preview wheels override stable project
pins, invoke the environment directly or use `uv run --no-sync` to avoid restoring
stable packages.

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
# Pass provider-specific options to the compilation session
winml compile -m model.onnx --ep qnn \
  --ep-options htp_performance_mode=burst \
  --ep-options soc_model=57
```

```bash
# Compile for GPU using the OpenVINO execution provider
winml compile -m microsoft_resnet50.onnx --device gpu --ep openvino
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
