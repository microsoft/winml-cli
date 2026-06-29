# src/winml/modelkit/session/session.py

## TL;DR

`WinMLSession` is the user-facing session wrapper bound to one `WinMLEPDevice` pair. Three module-level free functions (`_ep_defaults`, `_build_provider_options`, `_build_session_options`) extracted from the prior class body do the three-layer provider-options merge (catalog → user → monitor). The `perf()` context manager retains its complex auto-reset / monitor lifecycle logic. `SessionState` enum, `PerfContext` dataclass, `_suppress_native_output` ctx manager, and 5 exception classes complete the file.

## Diff metrics

- 928 lines (parent: 579 → +349 net inferred from commit stat).
- Free functions extracted: `_ep_defaults`, `_build_provider_options`, `_build_session_options`.
- Class methods: `WinMLSession.__init__`, `compile`, `run`, `reset`, `__del__`, `_is_verbose`, `_build_op_type_map` (staticmethod), `_validate_inputs`, `_prepare_inputs`, `_detect_best_device`, `_get_compile_suggestion`, `_get_install_suggestion`, `perf` (contextmanager), `io_config` (property), `_load_input_value_ranges`, `is_compatible`, plus `state` / `device` / `is_compiled` / `perf_stats` properties.

## Role before vs after

**Before.** `WinMLSession.__init__` consumed loose `(ep: str, device: str)` strings, internally called `WinMLEPRegistry.register_ep` to derive the `OrtEpDevice` handle. `_build_session_options` was a method that re-registered the EP, filtered handles, and built the SessionOptions.

**After.** `WinMLSession.__init__(onnx_path, ep_device: WinMLEPDevice, ...)` takes the pre-resolved pair. The handle is reached via `ep_device.device._ort`. The session-options helpers are module-level free functions taking the `WinMLEPDevice` directly. No `register_ep` call inside the session. Matches `2_coreloop.md` §5.7 + §5.9.

## Symbol-level changes

### Top-level utilities

- `_suppress_native_output(log_path)` ctxmanager — redirects fd 1 (stdout) to log file or `/dev/null`. Used by `compile()` for QNN's native stdout chatter.
- `SessionState` enum: `INITIALIZED`, `COMPILED`, `INFERRING`, `ERROR`.
- `PerfContext` frozen dataclass: `stats: PerfStats`, `monitor: WinMLEPMonitor`. Yielded by `perf()`. Not re-exported from `__init__.py`.

### `_ep_defaults(ep_device)` (lines 85-101)

Returns a fresh dict copy of `EP_DEVICE_SPECS[(ep, device.lower())].default_provider_options` if found, else `{}`. The `device_type.lower()` call is the fix mentioned in the commit body ("auto.py:411 passes device_type.lower() to match the other 3 call sites"). Most EPs return `{}` — the only non-empty default is QNN/NPU's two HTP options.

### `_build_provider_options(ep_device, ep_config, ep_monitor)` (lines 104-125)

Three-layer merge:
1. `options = _ep_defaults(ep_device)` (catalog).
2. If `ep_config.provider_options` is non-empty, `options.update(...)` (user).
3. If `ep_monitor is not None`, `options.update(ep_monitor.get_provider_options())` (monitor).

Monitor wins last. Documented as: "Callers who want to disable tracing should drop the monitor, not override its keys."

### Exception classes (lines 128-166)

`WinMLSessionError` base class with `message`, `context: dict`, `suggestion: str | None` fields. `_format_message()` produces a pipe-separated string. Four subclasses: `CompilationError`, `DeviceNotAvailableError`, `InferenceError`, `NotCompiledError`. Only `InferenceError` is re-exported from `session/__init__.py`.

### `_build_session_options(ep_device, ep_config, ep_monitor, base_session_options)` (lines 168-190)

```python
so = base_session_options if base_session_options is not None else ort.SessionOptions()
if ep_monitor is not None:
    for key, value in ep_monitor.get_session_options().items():
        so.add_session_config_entry(key, value)
handle = ep_device.device._ort
options = _build_provider_options(ep_device, ep_config, ep_monitor)
so.add_provider_for_devices([handle], options)
return so
```

Documented as free function (not method). The `ep_device.device._ort` reach into the private attribute conflicts with `WinMLDevice.ort_handle` public accessor (see `session__ep_device.md` issue).

### `WinMLSession.__init__` (lines 196-269)

Eager session construction unless `ep_config.enable_ep_context` is True (the compile workflow). Stores:
- `_onnx_path`, `_ep_device`, `_ep_config`, `_ep_monitor`, `_base_session_options`.
- `_provider_options` (initial snapshot from `_build_provider_options`).
- `_active_session_option_entries: dict[str, str] = {}`.
- `_ep: str` (canonical name from `ep_device.device.ep_name`).
- `_device: str` (lowercased `device_type`).
- `_persist_jit: bool`, `_embed_context: bool` (from `ep_config`).
- `_session: ort.InferenceSession | None = None`.
- `_state = SessionState.INITIALIZED`.
- `_last_error`, `_io_config`, `_perf_stats` (None).

Lifecycle invariant: `_session` must exist before any call that could raise (`__del__` reads it). Comment lines 239-240.

### `compile()` (lines 271-366)

Path for `_persist_jit=True`. Three cache cases:
1. Existing `.ctx.onnx` is fresher than source → use it.
2. Source is already an EPContext model → use it as-is.
3. Run `ort.ModelCompiler` to produce ctx, fall back to source on failure.

Then build a runtime InferenceSession against the (possibly compiled) model. Wraps in `try/except` that converts any exception to `CompilationError` with structured context + suggestion via `_get_compile_suggestion`.

### `run(inputs)` (lines 368-431)

Standard inference path: validate inputs not empty, auto-compile on first call, check ERROR state, call `_validate_inputs`, prep numpy arrays, run session, optionally record latency via `_perf_stats.record(lambda: ...)`. Build output dict via `dict(zip(output_names, outputs, strict=True))`. Exception wraps to `InferenceError`. Finally clause resets `INFERRING → COMPILED`.

### `reset()` (lines 433-441)

Clears `_session`, `_state = INITIALIZED`, `_last_error = None`. Logs INFO.

### `__del__` (lines 443-448)

Best-effort cleanup with broad exception swallow.

### `_is_verbose()` (lines 450-452)

Reads `WINMLSESSION_VERBOSE` env var.

### `_build_op_type_map(onnx_path)` (staticmethod, lines 454-478)

Loads ONNX model (no external data), returns `{n.name: n.op_type for n in model.graph.node if n.name and n.op_type}`. Empty dict on any failure. Used by `perf()` to inject the map into op-tracing monitors.

### `_validate_inputs(inputs)` (lines 480-501)

Compares input names to `io_config["input_names"]`. Raises `ValueError` for missing required. WARN-logs for unexpected (extra) inputs.

### `_prepare_inputs(inputs, session)` (lines 503-536)

Converts torch tensors via `.cpu().numpy()`, ndarrays via passthrough, anything else via `np.array(...)`. Coerces dtype via `io_config["input_types"]`.

### `_detect_best_device()` (lines 538-549)

Returns `"auto"` and logs INFO with the comment "With PREFER_NPU policy, ORT will automatically select..." — **dead code**. Never called. The `device` is set in `__init__` from `ep_device.device.device_type.lower()`. See Risks #1.

### `_get_compile_suggestion(device, error)` (lines 551-563) and `_get_install_suggestion(device)` (lines 565-571)

String-based device→hint maps. Suggestions like "Ensure NPU backend DLLs are in PATH (e.g., Qualcomm AI Stack)." `_get_install_suggestion` is **dead** — only `_get_compile_suggestion` is called (from `compile()` exception handler).

### Properties (lines 573-595)

`state`, `device`, `is_compiled`, `perf_stats`.

### `perf(warmup, monitor)` (contextmanager, lines 597-766)

This is the heaviest method in the file. Lifecycle steps:

1. Lazy import `NullEPMonitor`.
2. Re-entry guard: raise `RuntimeError` if `_perf_stats is not None`.
3. Build `effective_monitor` (the user's monitor or `NullEPMonitor`).
4. Validate `monitor.ep_name` (if non-None) against `self._ep` via `expand_ep_name`. Raise `WinMLEPMonitorMismatch` on mismatch.
5. Compute `new_prov = _build_provider_options(self._ep_device, self._ep_config, monitor)`.
6. **Auto-reset**: if compiled AND `new_prov != self._provider_options`, log WARN and `self.reset()`.
7. Snapshot `saved_sess_entries`, `saved_prov`, `saved_ep` for restore-on-exit.
8. Inject op-type map: `effective_monitor.set_onnx_op_types(self._build_op_type_map(self._onnx_path))`.
9. Activate `PerfStats(warmup)` and set `self._perf_stats`.
10. **Rebuild session conditionally**: `_session_rebuilt = new_prov != self._provider_options or self._session is None`. If True, update `_provider_options`, build new session with the monitor's options included.
11. Manually enter the monitor (`effective_monitor.__enter__()`). On `__enter__` failure: restore state, rebuild bare session (no monitor), do NOT call `__exit__`. Re-raise.
12. Yield `PerfContext(stats=stats, monitor=effective_monitor)`. Capture `exc_info` from the body via `BaseException` catch.
13. **Finally clause**: C-2 ordering: if `requires_session_teardown` (e.g., QNNMonitor needs `reset()` to flush CSV before `__exit__`), reset first. Call `monitor.__exit__(*exc_info)` — swallow any exception from `__exit__`. Restore snapshots. If `_session_rebuilt`, build a bare baseline session (no monitor) so the next `run()` call sees no monitor options. Re-raise body exception with traceback.

### `io_config` (property, lines 768-792)

Lazy loads ONNX I/O metadata via `load_onnx(..., load_weights=False, validate=False)` → `get_io_config(model)`. Enriches with `input_value_ranges` from `winml_build_config.json` if present.

### `_load_input_value_ranges()` (lines 794-833)

Scans `self._onnx_path.parent` for `winml_build_config.json` or `*_winml_build_config.json` glob; reads `export.input_tensors[].value_range`. Returns `{name: [low, high]}` dict.

### `is_compatible(node, graph)` (lines 835-927)

Wraps a single ONNX node in a minimal graph; builds InferenceSession with the session's EP binding. Returns True/False. Documented as "standalone utility, not wired into the build pipeline." WARN-logs when called without `graph` context.

## Behavior / contract changes

1. **`WinMLSession.__init__` no longer registers EPs.** It takes a pre-resolved `WinMLEPDevice`. Matches `2_coreloop.md` §5.7.
2. **The free functions `_build_session_options`, `_build_provider_options`, `_ep_defaults` are exported (private name) for re-use by `WinMLQairtSession._create_inference_session`.** See cross-file impact.
3. **`compile()` no longer takes a device argument.** Uses `self._device` set at `__init__`. Documented as "Device is immutable - set at __init__ time."
4. **`perf()`'s session rebuilding** is conditional on `new_prov != self._provider_options`. When the monitor contributes the same options as the session already has (e.g., default settings), the session is reused — preserves object identity for tests asserting `assert old_session is new_session`. Documented.
5. **`perf()` accepts BaseException** in its body try/except (line 725) — KeyboardInterrupt and SystemExit are captured into `exc_info` and re-raised. Per usual Python convention this is acceptable inside a context manager (the cleanup is important enough to run before the exception propagates).
6. **The `_session is None` guard in `run()` triggers `self.compile()` automatically.** Matches `1_req.md` flow — "Auto-compiles if not compiled."
7. **`reset()` clears `_last_error` to None**, so the next `run()` after a reset doesn't see the error state.
8. **`is_compatible(node, graph=None)`** builds a probe InferenceSession. WARN-logs without context. Expensive (one session per check) — must not be called inside hot loops. Documented.

## Cross-file impact

- `commands/perf.py`, `commands/compile.py`, `commands/build.py`, `models/auto.py`, `eval/evaluate.py` all construct `WinMLSession` (or its qairt subclass) with `ep_device=<WinMLEPDevice>` pre-resolved.
- `compiler/configs.py` and `compiler/stages/compile.py` consume `EPDeviceTarget` and `resolve_device` (not WinMLSession directly).
- `qairt/qairt_session.py` imports `_build_session_options` from `..session` (the private name escapes the module boundary).
- `monitor/qnn_monitor.py` reads `requires_session_teardown` — the session reaches that attribute via `getattr(effective_monitor, "requires_session_teardown", False)` defensively.

## Risks / subtleties

1. **`_detect_best_device()` is dead code** (line 538). It returns the literal string `"auto"` and is never called. Either delete it or wire it up; right now it's misleading documentation.
2. **`_get_install_suggestion()` is dead code** (line 565). Only `_get_compile_suggestion` is called from the `compile()` exception handler. Delete or wire.
3. **`compile()` catches a broad `Exception`** at line 330 from `ModelCompiler.compile_to_file` and falls back to the original model. A real EP compile failure (corrupt ONNX, missing driver) silently logs WARN and proceeds — the user may not notice until inference fails. Acceptable for the cache-miss-fall-back UX but obscures real failures.
4. **The qairt subclass imports `_build_session_options`** by underscore-prefixed name from `..session` (not from `..session.session`). This requires either `_build_session_options` to be in `session/__init__.py` `__all__` or for `from ..session import _build_session_options` to work via implicit module attribute access. Looking at `session/__init__.py`: it's not in `__all__`. The import works only because `from ..session import X` falls through to the package, which then accesses `session.session._build_session_options` via attribute. Actually, looking at qairt line 238: `from ..session import _build_session_options` — this imports from the **package** (`session/__init__.py`), not the **module** (`session/session.py`). And since `_build_session_options` is NOT in the package's namespace, this would normally fail. The only way this works is if Python's attribute lookup falls through to the submodule, which happens only if the submodule has been imported. Since `session/__init__.py` does `from .session import InferenceError, SessionState, WinMLSession`, the `.session` submodule IS loaded. But `_build_session_options` is not re-exported — so the qairt import is **technically broken**. It works only because Python `from package import name` looks up `name` as an attribute of the package, which falls through to `package.session._build_session_options` ONLY when no `name` shadow exists. This is fragile. See Simplification #1.
5. **`perf()`'s finally clause rebuilds the baseline session unconditionally** when `_session_rebuilt and self._session is not None` — but `self.reset()` (called when `requires_session_teardown=True`) sets `self._session = None`, so the conditional re-checks after reset. Looking at the sequence: line 732 calls `self.reset()`; line 733 calls `monitor.__exit__`; line 753 checks `_session_rebuilt and self._session is not None`. After `reset()`, `self._session is None` — so the rebuild is skipped. For non-reset monitors, the rebuild runs. This is the correct semantic, but the dependency between the C-2 ordering and the rebuild conditional is subtle.
6. **Inference error path doesn't reset `_session_rebuilt`**. If `monitor.__enter__` fails AND `_session_rebuilt=True`, the session is rebuilt to bare-baseline. But there's a subtle case: if `_session_rebuilt=False` (monitor was a no-op) and `monitor.__enter__` fails anyway (defensive only — `NullEPMonitor.__enter__` should never raise), nothing rebuilds. Probably fine.
7. **`__del__` reads `self._session`** which must exist as an attribute. The comment in `__init__` (line 239-240) acknowledges this. If a future `__init__` reorders so `_session = None` happens after a raise-able call, `__del__` would `AttributeError`. The current code is correct but the contract is implicit.
8. **`_load_input_value_ranges()` uses `model_dir.glob("*_winml_build_config.json")`** — could match unexpected files. Acceptable best-effort.
9. **`is_compatible()`** builds a fresh InferenceSession per call. The doc-comment says "not wired into the build pipeline" — but `analyze/core/runtime_checker_query.py` likely calls it (per the commit body's "is_compatible" references). Per-node session construction is expensive; if there's a hot consumer, it should batch.
10. **The `_provider_options` snapshot at `__init__` time** (line 226-228) is based on the initial `ep_monitor` arg. The `perf()` method later computes `new_prov` based on the perf-window monitor (which may differ). The auto-reset compares `new_prov` to the *current* `self._provider_options` — which would have been updated by a prior `perf()` exit. So a sequence of `perf(monitor=A)` then `perf(monitor=B)` correctly compares B's options against the post-A baseline (which was restored to the pre-A snapshot in A's finally clause). Subtle but correct.

## Simplification opportunities

1. **`_build_session_options` should be added to `session/__init__.py` `__all__`** OR qairt should import from `..session.session`. The current import is technically valid Python but fragile. Cleanest fix: move the three free functions to a private submodule like `session/_session_options.py` and have both `session.py` and `qairt/qairt_session.py` import from there.
2. **Delete `_detect_best_device()` and `_get_install_suggestion()`.** Both are dead. The `_detect_best_device` mentions a "PREFER_NPU policy" that doesn't exist in this codebase any longer.
3. **Replace the `WinMLSessionError._format_message` pipe-joining** with structured logging. Currently every exception message is mashed into one string with `|` separators — debuggers and tests both have to parse it. Use `__notes__` (PEP 678) for context, or just include context in `__str__` via a multi-line format.
4. **The `perf()` method is 170 lines.** Could split into 3-4 helpers: `_prepare_perf_window`, `_enter_monitor`, `_teardown_monitor_and_session`. Currently the lifecycle is documented via inline comments. A refactor that extracts the C-2-ordering teardown into a named method would make the contract explicit.
5. **The `_get_compile_suggestion` table is hand-written**. Could be a `_COMPILE_SUGGESTIONS: dict[str, list[tuple[str, str]]]` keyed on device, with (error-substring, suggestion-text) tuples. Marginal.
6. **`_build_op_type_map` is a staticmethod on `WinMLSession`** but used only by `perf()`. Could be a module-level free function. As a staticmethod it's awkward to test (tests have to reach `WinMLSession._build_op_type_map`).
7. **`io_config`'s lazy-load + cache** could use `@functools.cached_property` for clarity.
8. **`__del__` could be removed entirely.** Python's garbage collector handles cleanup; setting `self._session = None` adds nothing. The defensive `try/except` swallowing is for interpreter shutdown — but if the InferenceSession's destructor needs to run, dropping the reference is enough.
9. **The `_active_session_option_entries` field is initialized to `{}` and snapshotted in `perf()`** but never populated outside of perf snapshots. Looking at the code, `_active_session_option_entries` is only ever read in the `perf()` save/restore lines — never written. The field is dead state. Either populate it from `_build_session_options`' contribution to track which keys came from the session-options layer, or delete it.
10. **The `_persist_jit` and `_embed_context` reads from `ep_config`** could be inlined where used (in `compile()`). Currently they're cached as instance attrs but only `_persist_jit` is read in `__init__` (line 254) and `_embed_context` in `compile()` (line 321) — they're not configurable post-init.

## Open questions / TODOs surfaced

- The `_detect_best_device`'s reference to "PREFER_NPU policy" is stale documentation — the auto-detect is now in `auto_detect_device()` in `ep_device.py`. Either delete or update.
- The qairt subclass's import path for `_build_session_options` (Risk #4 + Simplification #1) is fragile — should be made explicit one way.
- `is_compatible`'s caller (likely `analyze/`) needs verification that per-call session construction is acceptable for the workload.
- `_active_session_option_entries` field is dead state (Simplification #9). The save/restore snapshot suggests it WAS planned to be populated but the population path was never added.
- Is `__del__` necessary at all (Simplification #8)? If kept, the `try/except` could narrow to `AttributeError` only (interpreter shutdown case).
