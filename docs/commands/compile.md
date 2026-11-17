# winml compile

> Compile ONNX to an EP-specific format, or explicitly compile CGIR to device-targeted CG Output IR.

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
| `--model` | `-m` | path | *(required unless `--list`)* | Input model. |
| `--output` | `-o` | path | — | Output file path (e.g., `model_compiled.onnx`). Takes precedence over `--output-dir`. |
| `--output-dir` | | path | same dir as input | Directory to write compiled output artifacts. |
| `--device` | `-d` | choice | `auto` | Target device: `auto`, `npu`, `gpu`, or `cpu`. |
| `--device-luid` | | string | — | Explicit CGC IR mode only: GPU adapter LUID from `winml sys`, in `0xHHHHHHHH_0xLLLLLLLL` form. |
| `--ep` | | `TEXT` | — | Force a specific execution provider, overriding device-to-provider mapping. Accepts full names (e.g., `QNNExecutionProvider`) or aliases (`qnn`, `dml`, `openvino`, `vitisai`, `migraphx`, `cpu`, `nvtensorrtrtx`). |
| `--ep-options` | | `KEY=VALUE` | — | Execution-provider option for the compilation session. Repeat for multiple options; later duplicate keys win. Explicit CLI values override matching `compile.provider_options` keys from `--config`. |
| `--validate` / `--no-validate` | | flag | `--validate` | Post-compilation validation. |
| `--compiler` | | choice | `ort` | Backends: `ort`, `ort_session`, `qairt`, `cgc`. `cgc` is inferred for a single `.mlir` input. |
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

## Compilation without target hardware

Use the explicit `--off-device` route; `--no-validate` alone does not bypass
physical-device discovery. This route uses ORT's `ModelCompiler`, disables CPU EP
fallback, embeds the context, and never constructs an inference session.

```powershell
winml compile --off-device --list-eps
winml compile --off-device --ep openvino --list-targets
winml compile --off-device --ep openvino --target 4000 -m model.onnx -o compiled.onnx
winml compile --off-device --ep qnn --target 60 -m model.onnx -o compiled-qnn.onnx
winml compile --off-device --ep vitisai --target X2 -m quantized.onnx -o compiled-vitisai.onnx
```

OpenVINO targets are the virtual platform IDs returned by `--list-targets`.
QNN's target is a positive numeric QAIRT `soc_model` ID; `60` is an example,
not an automatically selected target. Use the ID documented for your deployment
device. QNN does not enumerate SoC IDs, so `--list-targets --ep qnn` reports that
limitation instead of inventing a list.

QNN requires a compatible plugin payload exposing a CPU compilation host and
containing `QnnHtp.dll`; set `WINMLCLI_EP_PATH` to the payload directory if needed.
The host CPU runs the HTP compiler, not the QNN CPU reference backend or the ORT
CPU execution provider. No full QAIRT SDK is required for this route.

### QNN EP devices and filtering

The CPU compilation host is an actual `ort.OrtEpDevice` advertised by the QNN
plugin. It represents the host CPU's availability for offline compilation, not a
physical Qualcomm NPU or an ORT `CPUExecutionProvider` device.

The CLI registers `onnxruntime_providers_qnn.dll` with
`ort.register_execution_provider_library()`, then calls `ort.get_ep_devices()`.
That enumeration can return devices from multiple registered EPs.
`_registered_compilation_devices()` in `src/winml/modelkit/compiler/offdevice.py`
filters the enumeration by all three of the following:

- The EP name is `QNNExecutionProvider`.
- The device's `library_path` matches the exact selected QNN plugin DLL.
- The underlying hardware-device type is `OrtHardwareDeviceType.CPU`.

The helper yields only those filtered QNN CPU compilation hosts. The subsequent
`matches = list(devices)` copies that filtered list; it does not enumerate again
or select physical QNN NPUs. Compilation requires exactly one matching host.
For OpenVINO, the corresponding filter selects only NPU devices marked
`is_virtual=1`, and compilation then selects the requested `npu_platform`.

The DLLs have distinct roles: `onnxruntime_providers_qnn.dll` exposes the QNN EP
and its devices to ORT; `QnnHtp.dll` is explicitly selected through
`backend_path` to compile for the destination NPU specified by `soc_model`.
The CLI binds the filtered EP device with `SessionOptions.add_provider_for_devices()`
and calls `ort.ModelCompiler(...).compile_to_file(...)` without running inference.

QNN uses the canonical provider registration name because it is stored in the
compiled context. A conflicting in-process registration is an error, never
reused or unregistered; run compilation in a fresh CLI process.

### VitisAI virtual NPU compilation

VitisAI requires an AMD plugin exposing an NPU EP device marked `is_virtual=1`.
If discovery cannot find it, set `WINMLCLI_EP_PATH` to the full plugin DLL path,
for example `...\ExecutionProvider\onnxruntime_vitisai_ep.dll`, and inspect it
with `--off-device --ep vitisai --list-eps`. Use the plugin, not the similarly
named legacy `onnxruntime_providers_vitisai.dll` wrapper. Unknown DLL filenames
require an explicit `--ep`; unqualified inventory does not guess their provider.
The CLI never selects the physical AMD NPU, even when one is present. It filters
the canonical `VitisAIExecutionProvider` devices by the exact selected DLL,
NPU type, and virtual-device metadata, and requires exactly one match.

ORT enables `allow_virtual_devices` while a library whose registration name ends
in `.virtual` is registered. The CLI temporarily registers the selected VitisAI
DLL with such a name, then registers it as `VitisAIExecutionProvider`. This lets
the canonical factory enumerate a virtual NPU and keeps the compiled EPContext's
provider identity stable. Both registrations remain owned by this invocation and
are released in reverse order before the output is published. A conflicting
canonical registration fails; it is never reused or unregistered.

This two-registration workaround is tracked by
[AMD bug 65345287](https://dev.azure.com/microsoft/OS/_workitems/edit/65345287).
The tested AMD plugin reports the registration alias as its EP identity.
Single-alias compilation failed for a model with a boundary `QuantizeLinear`,
but succeeded for an integer-input/output variant. A mismatch between the aliased
EP identity and canonical helper-kernel identities is suspected, not confirmed.
The workaround binds only the canonical registration's virtual device; it does
not enable CPU fallback or rewrite model I/O. Remove it once a fixed plugin passes
the unchanged boundary-operation model with a single `.virtual` registration.

`--target X2` selects [AMD's documented compiler backend](https://ryzenai.docs.amd.com/en/latest/modelrun.html).
It is not a SoC ID or an
advertised destination architecture. The installed virtual device does not
enumerate architecture profiles, so `--list-targets --ep vitisai` reports that
limitation. This route does not claim arbitrary AMD NPU generation targeting,
artifact portability, or independence from installed driver/runtime software.
X1 and additional compiler options are not supported by this route.

Use an X2-compatible model. Local validation compiled a generated INT8 QDQ Conv
with integer output into an embedded VitisAI EPContext, binding only the virtual
NPU. Model preparation was test-only; the CLI does not quantize or rewrite model
I/O. CPU fallback remains disabled, so unsupported operations, including any
residual CPU-only boundary operations, fail rather than silently compiling on CPU.
For INT8/X2 compilation, the CLI sets `enable_cache_file_io_in_mem=1` to keep
the provider's intermediate cache in memory. It does not configure `cache_dir`
or `cache_key`, or create a separate provider cache directory. The compiled data
is embedded in the final ONNX output, which is still staged and published on disk.
This policy does not claim support for in-memory BF16 compilation.
No inference or physical-NPU deployment validation is performed.

All three providers require a compatible model and target. Compilation success does
not establish deployment compatibility or inference correctness on that target.
Inputs and existing outputs are preserved on failure. Replacing an existing
output requires `--overwrite`.

## CGIR to CG Output IR

This explicit mode uses the Windows ML Runtime API directly, without an ONNX
Runtime EP. The input and output selectors make the representation unambiguous:

```powershell
winml compile `
  -m input.mlir `
  --compiler cgc `
  --device gpu `
  -o output.mlir
```

`--compiler cgc` is optional for `.mlir` input.

The command loads CGIR through Windows ML Runtime. Both textual and bytecode
CGIR are accepted; invalid or unsupported input is reported by the Runtime.
Supply a complete executable model with its weights, not a graph-only report
projection.

Compilation calls the selected GPU target's Runtime
`ModelCompiler.compile_to_file()` with `CompiledModelForm.DEVICE_TARGETED`.
The output uses the `.mlir` suffix.
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

The preview `windowsml` payload must include Runtime APIs compatible with the
active CLI. If preview wheels override stable project pins, invoke the
environment directly or use `uv run --no-sync` to avoid restoring stable
packages.

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
