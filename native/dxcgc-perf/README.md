# Native DXCGC perf host (experimental)

### Fixed-shape replay and changing inputs

ABI 5 supplies CPU input pointers and byte sizes on every call. The worker
validates all sequence shapes and dtypes before timing. GPU resources and
bindings remain fixed for the session. The first invocation records the upload,
inference and readback commands; subsequent invocations resubmit the closed
list only after the preceding fence completed. Each call refreshes upload bytes
and returns independently owned outputs. Failure aborts the session; this is
not a concurrent or dynamic-binding execution API.

Paired input scheduling, repeat-order control, comparison and report generation
belong to the benchmark agent. WinML CLI provides per-path measurements and
explicit timing_scope metadata; it does not orchestrate cross-path experiments.

`winml perf --runtime dxcgc` compiles local MLIR with dxcgc.dll and executes
through D3D12 in a native host, without the Windows ML Runtime API.

By default the host loads dxcgc.dll, its dependencies and D3D12Core.dll from
the installed windowsml package's lib directory. Explicit --dxcgc-dir and
--dxcgc-sdk-path remain available for diagnostics. SDK version defaults to 721.
The native host must be built against compatible DXCGC headers. No SDK
headers or DLLs are redistributed here.

```powershell
cmake -S native/dxcgc-perf -B temp/dxcgc-build -A x64 -DDXCGC_INCLUDE_DIR=C:/DXCGC/include
cmake --build temp/dxcgc-build --config Release
winml perf -m model.mlir --runtime dxcgc --dxcgc-host temp/dxcgc-build/Release/winml-dxcgc-perf.exe --input-data inputs.npz --iterations 500 --warmup 50 -o perf.json
```

Host timing wraps the Python-to-native Run call: input upload, command recording/replay, GPU
submission/completion, readback and native CPU output materialization.
Python buffer preparation and separate output fetch are excluded. Run call overhead is included.
Runtime API timing remains Pipeline.run() only. Compilation and disk I/O are excluded. GPU timestamps are disabled
by default to avoid instrumenting only one side of the comparison. Use
--dxcgc-gpu-timing for a separate diagnostic run; uncollected GPU times are null.

Inputs copy directly into reusable upload buffers on every call. Executable
arguments remain bound to the same resources; redundant intermediate CPU
copies and rebinding are removed. Each call still uploads inputs, executes,
waits for completion, and returns independently owned CPU outputs. Results
record io_binding_policy=bound_buffers_replay_v3; rebuild both native
binaries. The Runtime Offline implementation is unchanged. Internal allocation
and scheduling can differ, so this measures complete path latency, not isolated
API-call overhead. Historical results retain their original protocol.

Inputs are NPZ arrays in logical input order; shapes, dtypes and byte counts
are validated using native reflection. Final CPU outputs are saved next to
the JSON report as .outputs.npz. Initializer exports and tensor-view offsets
are resolved using DXCGC reflection.

Initial scope: Windows, one static program, buffer bindings, embedded weights,
explicit NPZ input and fixed iteration count. External weights, textures and
dynamic shapes are unsupported. The helper is built separately from the wheel.

For the GPU pytest set WINML_DXCGC_MODEL, WINML_DXCGC_HOST, WINML_DXCGC_INPUT
and WINML_DXCGC_REFERENCE (CPU-reference NPZ for the same inputs). Run
pytest tests/unit/commands/test_perf_dxcgc.py. It uses the installed Runtime
DLLs by default and checks output relative L2 error below 1%.

### Disable D3D12 MLIR programs

Add `--disable-d3d-mlir-programs` to `winml perf --runtime dxcgc`.
The native host sets the DXCGC internal compiler option of the same name to
true before creating the compiler. Default compilation is unchanged when the
flag is absent. Unsupported DLLs fail explicitly; precompiled Output IR is
rejected with this flag because its targets are already selected.

The report records `disable_d3d_mlir_programs`. CPU input/output round-trip
timing and the default Runtime-package DLL selection remain unchanged. The
flag disables a compiler target; it does not disable D3D12 command submission
or independently certify that every compiled operation uses DirectML.

### Python-driven per-call measurement

The CLI starts one Python worker for setup and execution. The worker loads
winml-dxcgc-python.dll beside --dxcgc-host and drives one in-process ctypes
call per iteration; no per-inference process launch or IPC is timed. Compilation
and setup precede the Python measurement loop.

Each sample uses a Python stopwatch around Run only. ABI 5 separates Run from Fetch. Fetch copies native outputs into independent NumPy arrays outside timing.
Buffers and bindings are reused. Writing
outputs and timing CSV happens outside the measured loop. This aligns the
Python call boundary without claiming identical ctypes and COM projections.

Rebuild both the EXE and winml-dxcgc-python.dll. Results include
call_driver=python_ctypes_per_call plus the worker/DLL hashes; old results must
not be relabeled as this protocol.

### Build and compatibility

Requires Visual Studio 2022 C++ tools, Windows SDK 10.0.26100 or newer
(including DirectXPackedVector.h), CMake 3.24+, and compatible DXCGC headers.
The Python worker requires native host ABI version 5; older DLLs are rejected
before callbacks. Native calls validate output byte sizes before copying into
Python-owned arrays.

When tensor reflection returns E_NOTIMPL for a constant export, a fail-closed
text-IR fallback supports direct embedded resources, explicit static int64
lists, and exactly representable scalar float16 literals. It resolves exact
initializer/export/SSA mappings and verifies byte counts; unsupported layouts
or syntax fail rather than guessing. Printing compiled IR can increase setup
memory; this happens outside inference timing.
