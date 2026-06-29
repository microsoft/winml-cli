# src/winml/modelkit/optracing/qnn/profiler.py (DELETED)

## TL;DR
This file is removed. It defined `QNNProfiler` — the only production `OpTracer` subclass, which **owned the entire inference-loop**: built the ORT session, generated inputs, ran warmup/measured iterations, tore down the session to flush profiling data, and parsed CSV/QHAS artifacts. The biggest architectural shift of the commit lives here: the new `session/monitor/qnn_monitor.QNNMonitor` is **invoked-by-session** — it observes via `__enter__`/`__exit__` and contributes ORT options, but `WinMLSession.perf()` owns the loop.

## Diff metrics
- Lines deleted: 351
- Status: DELETED

## What this file did (pre-state)
A self-contained QNN EP profiling driver. Step-by-step:

1. Constructor took `onnx_path`, `output_dir`, `level` (`"basic"` | `"detail"`).
2. `run(iterations, warmup)`:
   - Built `ort.SessionOptions` with `disable_cpu_ep_fallback`, `ep.context_enable`, `ep.context_embed_mode=0`.
   - Built QNN provider options dict (`backend_path=QnnHtp.dll`, `htp_performance_mode=high_performance`, `htp_graph_finalization_optimization_mode=3`, `enable_htp_fp16_precision=1`, `profiling_level={detailed|optrace}`, `profiling_file_path=<csv>`).
   - Changed CWD to `output_dir` (so `*_schematic.bin` would land there).
   - Created the `InferenceSession` with `providers=["QNNExecutionProvider"]`.
   - Generated random NumPy inputs matching the model's input spec (`_ort_type_to_numpy`, `_resolve_shape` to fill symbolic dims with 1).
   - Ran `warmup` un-measured iterations + `iterations` measured iterations.
   - `del session` to flush the profiling CSV.
3. `_collect_results`:
   - In `detail` mode, looked for a `*_schematic.bin` alongside the `_qnn.log`, called `run_qhas_viewer(...)` to produce QHAS JSON, parsed it via `parse_qhas`.
   - Otherwise (basic mode or QHAS failure), parsed the CSV via `parse_qnn_profiling_csv`.
   - Built and returned an `OpTraceResult`.

Also contained module-level helpers `_ort_type_to_numpy`, `_resolve_shape`, and `_working_directory` (CWD context manager).

## Public symbols (pre-deletion)
- `QNNProfiler(OpTracer)` — concrete profiler with `__init__(onnx_path, *, output_dir, level)`, `is_available()`, `run(iterations, warmup)`.
- Module-level helpers: `_ort_type_to_numpy`, `_resolve_shape`, `_working_directory`.

## Where the functionality moved
| Pre-state symbol / behaviour | Where it lives now |
|---|---|
| `QNNProfiler` class | **Replaced by `session.monitor.qnn_monitor.QNNMonitor`** — but with inverted control: no `run()` method, no inference loop, no session construction. |
| `QNNProfiler.run()` inference loop (warmup + iterations + `del session`) | **Moved into `WinMLSession.perf()`.** The session drives the loop; `QNNMonitor.requires_session_teardown = True` declares the C-2 invariant that ensures `del session` runs before `__exit__` flushes artifacts. |
| `_build_session_options(ort)` | `QNNMonitor.get_session_options() -> dict[str, str]` — returns the same `disable_cpu_ep_fallback` / `ep.context_enable` / `ep.context_embed_mode` keys, but as a plain dict that `WinMLSession` applies via `add_session_config_entry`. |
| `_build_provider_options(csv_path)` | `QNNMonitor.get_provider_options() -> dict[str, str]` — returns the same QNN provider options. Routed via `add_provider_for_devices` when `ep_name = "qnn"` pins the EP. |
| `_generate_inputs(session)` (random NumPy input synthesis) | **Folded into the session's input-synthesis path** (the perf loop now owns input generation). |
| `_working_directory(path)` CWD context manager | **Replaced by `cwd` logic inside `QNNMonitor.__enter__`/`__exit__`** that establishes CWD before session creation and restores after teardown. The post-state monitor also logs when `*_schematic.bin` lands in CWD instead of `output_dir`. |
| `_collect_results` / `_try_qhas` / `_from_csv` (post-hoc artifact parsing) | Moved into `QNNMonitor` private methods (`_parse_csv_artifacts`, `_try_qhas`, etc.), invoked during `__exit__`. The result is stored on `self._result` and exposed via the `result` property. |
| `_ort_type_to_numpy`, `_resolve_shape` | Folded into the session's input generator — no longer monitor-owned. |
| CSV parsing — `parse_qnn_profiling_csv` import | Re-imported from the new home `session.monitor.qnn._internal`. |
| QHAS parsing — `parse_qhas` import | Re-imported from the new home `session.monitor.qnn._internal`. |
| Viewer invocation — `find_qnn_sdk`, `run_qhas_viewer` | Relocated to `session.monitor.qnn.viewer` (along with a new `run_basic_viewer` helper). |
| `is_available()` (instance method, ORT provider-name probe) | Promoted to `QNNMonitor.is_available()` classmethod with a richer WinML EP probe (catches and logs probe failures, falls back gracefully). |

## Net behavior change
- **Architectural inversion.** Pre-state: `QNNProfiler.run()` was the entry point, owning session lifecycle. Post-state: `WinMLSession.perf(monitor=QNNMonitor(...))` drives the loop; the monitor contributes config and observes.
- **Teardown ordering invariant is now declared.** `requires_session_teardown: ClassVar[bool] = True` on `QNNMonitor` tells the session to destroy the `InferenceSession` *before* calling `monitor.__exit__`. This is what guarantees the CSV is flushed by the time parsing runs.
- **Single CWD switch** is preserved but moves into the monitor's `__enter__` rather than wrapping a complete run.
- **Failure reporting** is now structured via the new `TraceStatus` (`"ok"`, `"no_data"`, `"parse_failed"`, `"basic_fallback"`, `"not_run"`) on `OpTraceResult`. Pre-state, a parse failure silently returned an empty `OpTraceResult` with `num_samples=0`; post-state, the status is reported and the user-facing log/report can distinguish causes.
- **Detail-mode fallback** logic is unchanged in spirit (no schematic / QHAS unavailable → fall back to basic CSV) but the post-state surfaces this as `status="basic_fallback"` rather than silently degrading.

## Risks
- The architectural inversion means out-of-tree callers that constructed a `QNNProfiler` and called `.run()` have **no migration path** other than rewriting against `WinMLSession.perf(monitor=...)`. The monitor cannot be driven standalone.
- The `_working_directory` CWD switch was a global mutation; the new monitor preserves the same global-CWD-mutation behavior inside `__enter__`/`__exit__`. Concurrent sessions would still race on CWD — this is a pre-existing risk, not new.
- The new `QNNMonitor` enforces `ep_name = "qnn"` so that provider options actually flow through `add_provider_for_devices`. If a caller previously relied on QNN being selected via ORT's policy-based selection without naming it, that path may now refuse the monitor.
- Input generation is now session-owned. If the session's default input synthesis differs from the pre-state `np.random.rand(*shape).astype(dtype)` (e.g. quantization-aware ranges), measurements may shift — verify against the post-state `WinMLSession` input generator.
