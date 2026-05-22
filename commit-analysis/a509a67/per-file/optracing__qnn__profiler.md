# src/winml/modelkit/optracing/qnn/profiler.py (DELETED)

## TL;DR
This file is removed. The `QNNProfiler` class — which previously owned the entire QNN profiling workflow (build ORT session, drive warmup/iterations, tear down, parse CSV/QHAS) — is replaced by `QNNMonitor` at `session/monitor/qnn_monitor.py`. The new design is an `EPMonitor` context manager that contributes session/provider options to a `WinMLSession` and parses artifacts in `__exit__`; the inference loop is owned by `WinMLSession.perf()`, not the monitor.

## Diff metrics
- Lines deleted: 351
- Status: DELETED

## What this file did (pre-state)
Implemented the only concrete `OpTracer` subclass. The class orchestrated the end-to-end profiling workflow:
1. Build an `ort.SessionOptions` with `session.disable_cpu_ep_fallback=1`, `ep.context_enable=1`, `ep.context_embed_mode=0`.
2. Build provider options dict with `backend_path=QnnHtp.dll`, HTP perf knobs (`htp_performance_mode=high_performance`, `htp_graph_finalization_optimization_mode=3`, `enable_htp_fp16_precision=1`), and the profiling controls (`profiling_level` and `profiling_file_path`).
3. `chdir` into `output_dir` so `*_schematic.bin` lands there.
4. Create `InferenceSession`, generate random inputs matching model I/O, run warmup + measured iterations, `del session` to flush profiling data.
5. In detail mode, locate the schematic, shell out to `run_qhas_viewer`, parse the resulting QHAS JSON via `qhas_parser.parse_qhas`, and return an `OpTraceResult` populated from the QHAS dict.
6. Otherwise (basic mode or QHAS fallback), parse the CSV via `csv_parser.parse_qnn_profiling_csv` and build an `OpTraceResult` from cycle counts.

## Public symbols (pre-deletion)
- `QNNProfiler(OpTracer)` — the concrete class. Constructor: `(onnx_path: Path, *, output_dir: Path, level: str = "basic")`.
  - `is_available() -> bool` — checks `QNNExecutionProvider in ort.get_available_providers()`.
  - `run(iterations=5, warmup=2) -> OpTraceResult` — full end-to-end execution.
  - Private helpers: `_build_session_options`, `_build_provider_options`, `_generate_inputs`, `_collect_results`, `_find_schematic`, `_try_qhas`, `_from_csv`.
- `_ort_type_to_numpy`, `_resolve_shape` — module-level helpers for input generation.
- `_working_directory` — module-level `@contextlib.contextmanager` for safe CWD switching.

## Where the functionality moved
| Pre-state symbol | Where it lives now |
|---|---|
| `QNNProfiler` class | Replaced by `QNNMonitor` at `src/winml/modelkit/session/monitor/qnn_monitor.py`. Different ABC (`EPMonitor` not `OpTracer`), different lifecycle (context manager not `run()`-driven). |
| `QNNProfiler.__init__(onnx_path, ..., output_dir, level)` | `QNNMonitor.__init__(level, output_dir=None, extra_provider_options=None)`. **`onnx_path` is dropped** — the session owns the model now. `output_dir` is optional (`None` mints a `qnn_profile_*` tempdir under the OS tempdir which is *not* auto-cleaned). New `extra_provider_options` lets callers pass HTP/backend knobs. |
| `QNNProfiler.is_available()` (instance) | `QNNMonitor.is_available()` (`@classmethod`). Extended: also probes the `onnxruntime-windowsml` path via `session.ep_registry.ensure_initialized()` and `ort.get_ep_devices()` scan for QNN devices. |
| `QNNProfiler.run(iterations, warmup)` | **Dropped entirely.** No `run` method on `QNNMonitor`. The inference loop lives in `WinMLSession.perf()`; the monitor only contributes options via `get_session_options()` / `get_provider_options()` and parses on `__exit__`. |
| `_build_session_options` | `QNNMonitor.get_session_options()`. **Behavior changed:** only returns `{"ep.context_enable": "1", "ep.context_embed_mode": "0"}`. The `session.disable_cpu_ep_fallback=1` entry is intentionally dropped — the docstring explains that under `onnxruntime-windowsml`, the WinML-registered QNN partitions QDQ-wrapped EPContext models into Q/DQ-on-CPU + EPContext-on-QNN, and disabling CPU fallback would wrongly reject that valid partition. |
| `_build_provider_options` | `QNNMonitor.get_provider_options()`. **Significantly changed:** only the two profiling-control keys (`profiling_level` mapped from `_LEVEL_TO_PROFILING`, and `profiling_file_path`) are owner-set. The HTP/backend defaults (`backend_path=QnnHtp.dll`, `htp_performance_mode`, etc.) are **no longer hardcoded** — callers pass them via `extra_provider_options`. The docstring documents that under `onnxruntime-windowsml` the device source already supplies an absolute `backend_path` and tuned defaults; baking our own would overwrite WinML's and break DLL loading. |
| `run()` warmup/iterations driver | Moved to `WinMLSession.perf()` + `PerfStats`. |
| `_generate_inputs` static method | Moved to the session layer (input generation is now an `EPDevice` / session concern, no longer a profiler concern). |
| `_working_directory` context manager (chdir) | **Dropped entirely.** New `QNNMonitor._find_schematic` and `_try_qhas` use `Path.glob` rather than mutating CWD (C-5 / FR-12 in the design spec); the CWD-glob fallback is **mtime-gated against the profiling CSV** to reject stale schematics. |
| `_ort_type_to_numpy`, `_resolve_shape` | Moved to the session/input-generation layer (out of scope for this file's analysis). |
| `_collect_results` orchestration | Replaced by `QNNMonitor.__exit__` → `_parse_artifacts_safe` → `_parse_artifacts`. Wraps the parser in a try/except that converts exceptions into `OpTraceResult(status="parse_failed", error=str(exc))` (the parse-failure contract). |
| `_try_qhas` | Replaced by `QNNMonitor._try_qhas`. Adds a `qhas_override: Path | None` parameter so offline analysis (`parse_existing_artifacts`) can skip the viewer shell-out. Returns `(summary, operators, qhas_path)` tuple instead of an `OpTraceResult | None`. Adds the v2.4 op-type resolver call: `name = self._resolve_op_type(op["op_path"], ep_authoritative=op["name"])`. |
| `_from_csv` | Replaced by the CSV path inside `QNNMonitor._parse_artifacts`. Adds: (a) Windows file-handle-lag retry — sleeps 50ms once if CSV is absent; (b) `int("0" or 0)` → `round(float(...))` for cycle metadata so float-string SDK output parses correctly; (c) v2.4 resolver call `_resolve_op_type(op["op_path"], ep_authoritative=None)`; (d) sets `status="basic_fallback"` when detail mode falls back to CSV. |
| `_find_schematic` | Replaced by `QNNMonitor._find_schematic`. **Significantly hardened:** glob `_output_dir` first, then glob CWD as read-only fallback gated by mtime ≥ csv_mtime − 5s tolerance to reject stale schematics from prior CI runs. |
| (new) `parse_existing_artifacts(level, artifacts, onnx_op_types) -> OpTraceResult` | New classmethod on `QNNMonitor` for offline analysis of pre-existing CSV/QHAS files without running a benchmark. No pre-state equivalent. |

## Net behavior change
- The control flow inversion is the headline change: `QNNProfiler.run(iters, warmup)` is replaced by `with session.perf(monitor=QNNMonitor(...)) as ctx: for _ in range(N): session.run(...)`. The caller drives the loop, the monitor only contributes options + parses artifacts.
- `session.disable_cpu_ep_fallback=1` is no longer hardcoded — required for `onnxruntime-windowsml` correctness (WinML-QNN partitions Q/DQ to CPU legitimately).
- HTP/backend provider options are no longer hardcoded — required for `onnxruntime-windowsml` correctness (overwriting WinML's `backend_path` breaks DLL loading).
- `os.chdir` is gone — replaced by `Path.glob` with an mtime-gated CWD fallback so a stale schematic in CWD can't silently corrupt a new run's QHAS metrics.
- CSV metadata parsing tolerates float-string values via `round(float(...))` instead of `int("0" or 0)`.
- Parse failures now produce `OpTraceResult(status="parse_failed", error=...)` rather than raising or producing an empty result.

## Risks
- Out-of-tree code that bakes `backend_path=QnnHtp.dll` / `htp_performance_mode=high_performance` assumptions because the profiler used to set them will lose those defaults. Callers running on bundled `onnxruntime-qnn` need to supply them via `extra_provider_options`.
- `session.disable_cpu_ep_fallback=1` is no longer set; users who relied on the loud failure when QNN couldn't claim the graph will now see partial-CPU partitions silently. The docstring notes the new guarantee comes from `add_provider_for_devices` failing loudly when the device is absent, but this is a behavioral shift.
- Tracers that drove a synchronous `tracer.run()` workflow must restructure to use `WinMLSession.perf()`.
