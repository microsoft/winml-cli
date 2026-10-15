# Native DXCGC perf host (experimental)

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

Host timing always includes CPU input copy, upload, command recording, GPU
submission/completion, readback and caller-owned CPU output copy on every
iteration. Compilation and disk I/O are excluded. GPU timestamps separately
measure only execution. There is no resident-only timing option.

Each measured iteration creates owned CPU input copies and rebinds executable
arguments, then uploads inputs and returns owned CPU output copies. GPU and
staging buffers remain pooled/reused; the Runtime's internal allocation policy
is unknown, so this does not force fresh committed GPU allocations. Results
record `io_binding_policy=per_call_cpu_ownership_and_binding`, acknowledged by
the native executable (older hosts must be rebuilt). The Offline CLI path is
restored to its original per-call behavior. Python projection and scheduling
policies
can still differ; the gap describes the complete execution paths rather than
an isolated native API-call cost.

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

Each sample includes Python-owned NumPy output allocation, ctypes argument
marshalling, native CPU input ownership/copy, argument rebinding, GPU upload,
execution, fence completion and direct readback into the owned NumPy outputs.
Native input temporaries are released before the call returns. Output arrays
survive the call; writing outputs and timing CSV is outside the measured loop.
GPU/staging resources remain reusable; the original Offline path is unchanged.
This aligns the Python call boundary but does not claim ctypes and Runtime COM
projections, internal allocators or scheduling are identical.

Rebuild both the EXE and winml-dxcgc-python.dll. Results include
call_driver=python_ctypes_per_call plus the worker/DLL hashes; old results must
not be relabeled as this protocol.

### Build and compatibility

Requires Visual Studio 2022 C++ tools, Windows SDK 10.0.26100 or newer
(including DirectXPackedVector.h), CMake 3.24+, and compatible DXCGC headers.
The Python worker requires native host ABI version 2; older DLLs are rejected
before callbacks. Native calls validate output byte sizes before copying into
Python-owned arrays.

When tensor reflection returns E_NOTIMPL for a constant export, a fail-closed
text-IR fallback supports direct embedded resources, explicit static int64
lists, and exactly representable scalar float16 literals. It resolves exact
initializer/export/SSA mappings and verifies byte counts; unsupported layouts
or syntax fail rather than guessing. Printing compiled IR can increase setup
memory; this happens outside inference timing.
