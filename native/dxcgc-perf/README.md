# Native DXCGC perf host (experimental)

`winml perf --runtime dxcgc` loads local MLIR in a native process that calls
`DXCGCCreateDevice`, `IDXCGCCompiler::Compile`, and `IDXCGCExecutable::Record`.
D3D12 owns buffers, command lists, submission, and fence synchronization.
Neither Windows ML Runtime API nor ORT participates in that process. The CLI
launch and Python overhead are outside the measured loop.

Build with Visual Studio 2022 C++ tools, Windows SDK 10.0.26100 or newer, CMake,
and the matching DXCGC Drop11.3 `include/dxcgc.h`:

```powershell
cmake -S native/dxcgc-perf -B temp/dxcgc-build -A x64 -DDXCGC_INCLUDE_DIR=C:/DXCGC/include
cmake --build temp/dxcgc-build --config Release
winml perf -m model.mlir --runtime dxcgc --dxcgc-host temp/dxcgc-build/Release/winml-dxcgc-perf.exe --dxcgc-dir C:/DXCGC --dxcgc-sdk-path C:/DXCGC/D3D12 --dxcgc-sdk-version 721 --dxcgc-adapter 0 --input-data inputs.npz --iterations 100 --warmup 10 -o perf.json
```

Supply the matching DLL set, compatible DirectML dependency, Agility SDK, and
GPU driver separately. No private SDK headers or binaries are redistributed.
This helper is built separately from the Python wheel.

The input NPZ must contain arrays in the model's logical input order. Shape,
dtype and byte count are checked against native tensor reflection. Input names
are not used for binding. Output views and initializer exports are resolved
through the DXCGC reflection API, including offsets into packed arguments.
The adjacent `perf.outputs.npz` contains the final inference outputs.

JSON reports separate GPU timestamps from host recording/submission/fence
time. Compilation, executable creation, weight preprocessing, input uploads
and output readback occur outside both inference measurements. The
`compile_ms` field includes loading and compilation, but excludes executable
creation. GPU timing cannot be subtracted from a Runtime API host latency to
claim Runtime API overhead. A controlled same-boundary comparison is needed.

Initial scope: one static program, buffer bindings, embedded constants,
explicit NPZ inputs, fixed iteration count. External safetensors weights,
texture bindings, dynamic shapes, resource monitoring and other perf build/EP
options are unsupported. Unsupported CLI options fail explicitly. The SDK
may use DirectML for compiled partitions. This is not a guarantee of exclusive
driver-native MLIR execution.

Validation uses pytest. Set WINML_DXCGC_MODEL, WINML_DXCGC_HOST,
WINML_DXCGC_DIR, WINML_DXCGC_SDK, WINML_DXCGC_INPUT, and
WINML_DXCGC_REFERENCE (a CPU-reference NPZ for the same inputs), then run
`pytest tests/unit/commands/test_perf_dxcgc.py`. The hardware test executes
100 measured iterations after 10 warmups and checks relative L2 error <1%.

### CPU round-trip comparison

Add `--dxcgc-io-mode cpu-roundtrip` to include CPU input copying, upload,
GPU execution, readback and a caller-owned CPU output copy in every host
sample. Staging GPU resources are reused; output CPU buffers are allocated
per iteration. Disk I/O and compilation remain excluded. The default
`resident` mode retains GPU-resident inputs and excludes tensor transfers.
Compare `host_ms` in round-trip mode with the Runtime API feed/infer/fetch
latency, not with `gpu_ms`. Runtime API Python projection, resource allocation
and scheduling policies still differ; matching compiler DLLs and compiled
work is required before attributing the entire gap to API overhead.
