# src/winml/modelkit/session/monitor/qnn_monitor.py

## TL;DR
New concrete `EPMonitor` for the Qualcomm QNN EP — owns QNN-EP profiling provider options (`profiling_level`, `profiling_file_path`), drives the QHAS-viewer shell-out (detail mode) or CSV-only path (basic mode), and turns the resulting artifacts into an `OpTraceResult` of `OperatorMetrics`. Critical hardening: `_to_int` helper replaces a fragile `int(meta.get("...", 0) or 0)` pattern with `round(float(...))` so float-string SDK output (e.g. `"12345.6"`) no longer raises `ValueError` and silently zeros out `cycle_to_us` / corrupts every `duration_us` downstream.

## Diff metrics
- Lines added / removed: **+633 / 0** (effectively new — diff metrics report `633 ++++++++++++++++++++++` with no `-` rows; the file in this form replaces the old `optracing/qnn/*` tree wholesale)
- New / modified: **new file at this path** (the wider commit deletes the old `src/winml/modelkit/optracing/` tree it supersedes)

## Role before vs after
- **Before**: No file at this path. The old per-op QNN tracing logic lived under `src/winml/modelkit/optracing/qnn/` (deleted by this commit).
- **After**: `QNNMonitor` is the QNN concrete implementation of the new `EPMonitor` ABC. Plugged into `WinMLSession.perf()` via two hooks (`get_session_options`, `get_provider_options`) plus a class-var contract (`ep_name="qnn"`, `requires_session_teardown=True`). Drives both live profiling and offline parsing via `parse_existing_artifacts`.

## Symbol-level changes
All additions:
- Module constant `_LEVEL_TO_PROFILING: dict[str, str] = {"basic": "detailed", "detail": "optrace"}` — maps the user-facing level word to QNN EP's `profiling_level` provider option.
- Class `QNNMonitor(EPMonitor)` with ClassVars `requires_session_teardown=True` and `ep_name="qnn"`.
- `__init__(level, output_dir=None, extra_provider_options=None)` — mints a per-monitor `qnn_profile_*` tempdir when `output_dir is None`; deliberately **does not** register a cleanup finalizer (artifacts persist for inspection).
- `output_dir` property — read-only accessor for the artifact directory.
- `is_available()` classmethod — two-path probe: `onnxruntime-qnn` bundled wheel via `get_available_providers()`, then `onnxruntime-windowsml` via `ep_registry.ensure_initialized()` + `get_ep_devices()`. Logs a WARNING on probe exception (NFR-2).
- `get_session_options()` — emits `ep.context_enable=1`, `ep.context_embed_mode=0`. Intentionally does **not** set `session.disable_cpu_ep_fallback` (Q/DQ-on-CPU + EPContext-on-QNN is a valid partition under WinML).
- `get_provider_options()` — pass-through of `extra_provider_options` with the two owner-enforced keys (`profiling_level`, `profiling_file_path`) applied last per PRD C-3.
- `__enter__` / `__exit__` — `__exit__` routes through `_parse_artifacts_safe`, never suppresses caller exceptions.
- `_parse_artifacts_safe(qhas_override=None)` — single source of truth for the parse-failure contract; both live and offline paths route through it.
- `result` property — accessor for the parsed `OpTraceResult`.
- `set_onnx_op_types(onnx_op_types)` — defensively copies the ONNX `node.name → node.op_type` map (drives L1 of the fallback chain).
- `_resolve_op_type(op_path, ep_authoritative=None)` — implements the v2.4 FR-14 fallback chain L1 → L2 → L3 → L4 (ONNX → EP-authoritative → heuristic → raw).
- `_heuristic_op_type(op_path)` — token-suffix strip + leaf-split with trailing-slash fallback.
- `parse_existing_artifacts(level, artifacts, onnx_op_types=None)` classmethod — offline / post-hoc analysis entry point; raises `ValueError` if `artifacts` lacks `"csv"`.
- `_parse_artifacts(qhas_override=None)` — windows file-handle-lag retry (R-2, single 50ms sleep), CSV-only basic path, optional QHAS detail path; status `ok` → `basic_fallback` when QHAS unavailable; `no_data` when CSV never appeared.
- `_to_int(val, field)` — **the key hardening**: `round(float(val))` parses both `"12345"` and `"12345.6"`; logs a WARNING on `TypeError`/`ValueError` and returns 0.
- `_try_qhas(artifacts, qhas_override=None)` — locates `*_qnn.log` + schematic, shells out to viewer, parses JSON; degrades to `(None, None, None)` silently on any failure (caller flips status to `basic_fallback`). Forbids `os.chdir` per C-5/FR-12.
- `_find_schematic()` — output_dir glob first; **mtime-gated** CWD fallback (CSV mtime − 5s tolerance) so stale schematics from prior CI runs cannot poison QHAS results with `status="ok"`.
- `_make_failure_result(status, error)` — synthesises a minimal `OpTraceResult` for parse-time failures.

## Behavior / contract changes
- **`int("0" or 0) → round(float(...))` fix (the called-out hardening)**: the metadata extraction in `_parse_artifacts` previously did `int(meta.get("accel_execute_cycles", 0) or 0)` (per the commit body). A QNN SDK that emits `"12345.6"` instead of `"12345"` for `accel_execute_us` would raise `ValueError: invalid literal for int() with base 10: '12345.6'`, which the surrounding `except Exception` would catch and convert into a `parse_failed` result with all op rows lost. The new `_to_int` does `round(float(val))`, so float-string inputs are now lossy-rounded rather than silently dropping the entire trace. Worse latent bug avoided: with `total_cycles=0` and `accel_us=0`, `cycle_to_us=0.0` and every `OperatorMetrics.duration_us=0.0` — the trace would render as "the entire model ran in 0 µs."
- **Owner-enforced provider keys (C-3)**: `profiling_level` and `profiling_file_path` are applied AFTER the `extra` `dict.update`, so callers cannot override them even by passing them in `extra_provider_options`. This is a deliberate behavioral change from the legacy `optracing/` API where these keys were caller-mutable.
- **EP pinning**: `ep_name="qnn"` ClassVar forces `WinMLSession` onto the QNN EP path via `add_provider_for_devices`, bypassing ORT's policy-based selection (which would silently drop provider options).
- **`requires_session_teardown=True`**: QNN EP flushes the profiling CSV only on `InferenceSession.__del__`, so `WinMLSession.perf().__exit__` must drop the session **before** calling `monitor.__exit__` — encoded as a class-level contract, not an implementation detail.
- **No tempdir cleanup**: when `output_dir=None`, a `qnn_profile_*` directory is minted under the OS tempdir and **never auto-removed**. Disk hygiene is explicitly the caller's problem; documented loudly in both class and property docstrings.
- **mtime-gated CWD schematic fallback**: prior implementations that fell back to CWD without an mtime check could silently consume a stale `*_schematic.bin` from a previous CI run and produce QHAS metrics for the wrong graph with `status="ok"` — a true silent-corruption path now closed.
- **Empty-string `op_type` is no longer truthy**: `_resolve_op_type` uses `if mapped:` (truthy) so a defensive `""` op_type in the ONNX map falls through to L2/L3/L4 instead of short-circuiting with `""`.

## Cross-file impact
- **Direct sibling imports**: `from .ep_monitor import EPMonitor`, `from .op_metrics import OperatorMetrics, OpTraceResult, TraceStatus`, `from .qnn._internal import _TOKEN_SUFFIX, parse_qhas, parse_qnn_profiling_csv`, `from .qnn.viewer import find_qnn_sdk, run_qhas_viewer`.
- **Note on `_TOKEN_SUFFIX` import**: this is the **CLAUDE.md `_`-prefixed-private exception** in use — `qnn_monitor.py` is the only module allowed to import non-`_`-prefixed names from `qnn._internal`, and it imports the private regex anyway so the heuristic strip semantics stay aligned with the CSV path.
- **EP registry coupling**: `is_available()` imports `from ..ep_registry import ensure_initialized` inside the try-block — defers the WinML EP probe until `is_available()` is actually called.
- **Caller contract**: `WinMLSession.perf()` is expected to call `set_onnx_op_types(map)` once **before** `__enter__` so the L1 lookup is primed.

## Risks / subtleties
- The R-2 file-handle-lag retry is a single 50ms sleep — if Windows takes longer than 50ms to flush the CSV under load, the monitor returns `no_data`. Tunable point if real-world misses appear.
- `_find_schematic()`'s 5s mtime tolerance is arbitrary; on a heavily clock-skewed filesystem this could either reject a genuine artifact or accept a stale one. Documented but not parameterised.
- `_LEVEL_TO_PROFILING["basic"] = "detailed"` is intentionally counterintuitive — the user-facing level word does not equal the QNN-EP option string. Anyone debugging at the ORT level needs the map in hand to make sense of `profiling_level="detailed"` traces being labeled "basic" in the report.
- `_resolve_op_type` mixes vocabularies across paths: CSV (`ep_authoritative=None`) falls through to leaf-split heuristic that yields ONNX op symbols (`"Conv"`, `"Add"`), whereas QHAS path passes `qnn_op_type` ("Conv2d", "ElementWiseAdd"). The ONNX-primary L1 lookup is what reconciles them — when the map is unavailable, basic and detail mode produce different `name` columns for the same model.
- `_try_qhas` swallows all exceptions to `(None, None, None)`; the only signal that QHAS failed is the `basic_fallback` status — caller must check it.

## Open questions / TODOs surfaced
- No tempdir cleanup on `output_dir=None`: long-running services that exercise this path will leak `qnn_profile_*` directories under `%TEMP%`. Should there be a max-N-keep policy or an explicit `cleanup()` method?
- `_to_int` defaults silently to 0 on parse failure but only logs WARNING — a corrupted SDK output produces an `ok`-status result with zeroed metadata. Consider escalating to `status="parse_failed"` when metadata fields fail to parse.
- `parse_existing_artifacts` quietly overwrites `instance._csv_path` after the constructor pinned it to `<output_dir>/profiling_output.csv` — a private-attribute reach-around. A constructor argument for `csv_filename` would be cleaner.
