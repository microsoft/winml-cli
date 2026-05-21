# src/winml/modelkit/session/session.py

## TL;DR
Hard-break refactor of `WinMLSession`: the policy/device-string surface (`DEVICE_POLICY_MAP`, `_EP_NAME_MAP`, `_find_ep_device`, the `device=`/`ep=`/`session_options=` kwargs) is removed and replaced with a required `EPDevice` descriptor plus a free-function `_build_session_options`. `perf()` now yields a new `PerfContext` (stats + monitor) and gains a save/restore lifecycle that auto-tears-down and rebuilds the `InferenceSession` only when the monitor contributes differing provider options. `compile()` is rewritten to actually call `ort.ModelCompiler.compile_to_file` and to defer eager `InferenceSession` creation when `enable_ep_context=True` (Bugs A & B).

## Diff metrics
- Lines added: ~287
- Lines removed: ~134
  (`git show --stat` not shown in diff; counts approximated from the unified diff hunks — large net add owing to the new `perf()` lifecycle and `_build_session_options` free function.)

## Role before vs after
- **Before:** Session manager that took a free-form `device` string ("auto"/"npu"/"gpu"/"cpu") and optional `ep` short name, mapped device -> ORT `OrtExecutionProviderDevicePolicy` and ep -> full provider name via two internal dicts, picked an OrtEpDevice by first-match (`_find_ep_device`), and lazily compiled inside `_build_session_options` as an instance method. `perf()` yielded a `PerfStats`.
- **After:** Session manager bound to one explicit `EPDevice` (vendor_id + device_id + ep + device). `_build_session_options` is a module-level free function that builds a `SessionOptions`, resolves the exact `OrtEpDevice` via `WinMLEPRegistry.register_ep`, validates uniqueness (raising `DeviceNotFound` / `AmbiguousMatch`), and applies a three-layer `provider_options` merge (ep_defaults -> ep_config.provider_options -> monitor). `perf()` yields a `PerfContext(stats, monitor)` and manages save/restore around an optional `EPMonitor`.

## Symbol-level changes

- **`PerfContext`** — added
  - New frozen dataclass with `stats: PerfStats` and `monitor: EPMonitor`; yielded by `perf()`.

- **`_ep_defaults`** — added (module-level helper)
  - Returns a fresh dict copy of `EPDeviceSpec.default_provider_options` for the resolved spec, or `{}` if the (ep, device) pair has no catalog entry. Embeds the comment that QNN backend_type must not be passed explicitly (crashes ORT 1.23.5 exit 127).

- **`_build_provider_options`** — added (module-level helper)
  - Three-layer merge: catalog defaults -> `ep_config.provider_options` -> `ep_monitor.get_provider_options()`. Monitor wins last (tracing correctness invariant).

- **`_build_session_options`** — added as free function (replaces the deleted instance method of the same name)
  - Calls `WinMLEPRegistry.get_instance().register_ep(ep_device.ep)`, filters by `(device.type.name, vendor_id, device_id)`, raises `DeviceNotFound` with the available device list and `AmbiguousMatch` if dedup leaves >1. Adds monitor's `get_session_options()` entries via `add_session_config_entry`, then `add_provider_for_devices([matching[0]], options)`.

- **`DEVICE_POLICY_MAP`** — removed
  - Old policy-string -> `OrtExecutionProviderDevicePolicy` mapping deleted; no policy-based selection path remains.

- **`WinMLSession._EP_NAME_MAP`** — removed
  - Class-level short-name -> full-provider-name dict deleted; `expand_ep_name` from `.ep_device` is the single source of truth.

- **`WinMLSession._eps_initialized` / `_init_winml_eps_once`** — removed
  - Lazy class-level WinML EP-init hook deleted; registry resolution now happens on-demand inside `_build_session_options` via `WinMLEPRegistry.get_instance().register_ep`.

- **`WinMLSession._find_ep_device`** — removed
  - First-match `for ep_dev in ort.get_ep_devices(): if ep_dev.ep_name == ep_name` selection deleted; deterministic vendor_id/device_id filter inside `_build_session_options` replaces it.

- **`WinMLSession._build_session_options` (instance method)** — removed
  - Replaced by the module-level free function of the same name.

- **`WinMLSession.__init__`** — signature-changed (hard break)
  - New: `__init__(onnx_path, ep_device: EPDevice, *, ep_config=None, base_session_options=None)`.
  - Removed kwargs: `device=`, `ep=`, `session_options=`.
  - `EPDevice` is positional and required. Stores `_ep_device`, `_ep_config`, `_base_session_options`. Derives `_device`/`_ep` as legacy aliases. Pre-creates `_session = None` before any call that could raise (so `__del__` is safe). Eagerly builds the `InferenceSession` when `not self._persist_jit`; defers to `compile()` otherwise (Bug A fix). Drops the `FileNotFoundError` upfront check; drops `_init_winml_eps_once()` call; drops `logger.info("WinMLSession initialized: %s", ...)` trailing log.

- **`WinMLSession.compile`** — refactored (Bug A + Bug B)
  - Bug A: now defers ORT `InferenceSession` creation when `enable_ep_context=True`, so the compile workflow runs.
  - Bug B: calls the new free `_build_session_options(...)` and `ort.ModelCompiler.compile_to_file(...)`. Cache-freshness check (`ctx_path.exists() and mtime>=`) and `is_compiled_onnx` short-circuit are restructured into a single if/elif/else. On `ModelCompiler` exception, logs a warning and falls back to the original model rather than failing hard. After compile, creates a runtime `InferenceSession` against the resulting model and stamps `SessionState.COMPILED`.
  - Removed: auto-device resolution branch (`if target_device == "auto"`) and `get_ep_device_map()` post-resolution; device is now fixed at `__init__`.

- **`WinMLSession.perf`** — signature-changed + heavy refactor
  - New signature: `perf(warmup=0, monitor: EPMonitor | None = None)`; yields `PerfContext` (was `PerfStats`).
  - Re-entry guard: raises `RuntimeError` if `_perf_stats` is non-None.
  - EP-monitor cross-check: raises `EPMonitorMismatch` if `expand_ep_name(monitor.ep_name) != self._ep_device.ep`.
  - Three-layer merged `new_prov` computed; if `_session is not None and new_prov != self._provider_options` the compiled session is auto-reset (with a WARNING) so new options take effect.
  - Snapshots `_active_session_option_entries`, `_provider_options`, `_ep` for restore-on-exit.
  - Injects ONNX op-type map into the monitor via `set_onnx_op_types(self._build_op_type_map(self._onnx_path))` before `__enter__`.
  - Manual `__enter__`/`__exit__` of the monitor so it can sequence around `self.reset()`.
  - C-2 invariant: monitors with `requires_session_teardown=True` get `self.reset()` called *before* `monitor.__exit__` (so QNNMonitor's CSV flush data is visible inside `__exit__`).
  - On body exception, captures `sys.exc_info()`, runs teardown, restores snapshots, rebuilds baseline session, then re-raises.
  - Baseline rebuild on exit only when `_session_rebuilt` was True (preserves pre-perf `InferenceSession` object identity when monitor contributed nothing — tests assert on this).

- **`WinMLSession._build_op_type_map`** — added (staticmethod)
  - Builds `{node.name: node.op_type}` from an ONNX file; returns `{}` on any failure (None path, missing file, corrupt protobuf, missing `onnx` package). Loaded without external data. Used by `perf()` to feed op-tracing monitors.

- **`WinMLSession.is_compatible`** — signature-changed
  - Removed the `device: str | None = None` keyword; the session's `_ep_device` is now the sole binding. Internal call switched from `self._build_session_options(target_device)` to the free `_build_session_options(self._ep_device, self._ep_config, None, self._base_session_options)`.

- **`WinMLSession` class docstring** — refactored
  - Shrunk from a multi-paragraph "policy-based device selection" description to "ONNX Runtime session bound to one explicit (EP, device) target."

## Behavior / contract changes
- Constructing `WinMLSession` now **requires** an `EPDevice`. Callers that previously passed `device="npu"` or `ep="qnn"` must call `resolve_device(ep, device)` first.
- No more `FileNotFoundError` raised eagerly in `__init__`; missing ONNX files surface later when ORT tries to load.
- No more class-level `_init_winml_eps_once` side effect at first construction; registry registration happens on demand inside `_build_session_options`.
- For runtime workflows (`persist_jit=False`) the `InferenceSession` is now created in `__init__`, not lazily on first `run()`. For compile workflows (`enable_ep_context=True`) creation is deferred to `compile()`.
- `perf()` yields `PerfContext` not `PerfStats`. Callers must use `ctx.stats` / `ctx.monitor` (caller-visible breaking change).
- `perf()` may auto-reset a compiled session with a WARNING log when the monitor's provider_options differ from current.
- `perf()` may raise `RuntimeError` on nested entry and `EPMonitorMismatch` on monitor/session EP disagreement.
- `perf()` preserves the pre-perf `InferenceSession` object identity when no rebuild is needed (tests rely on this).
- `_build_session_options` failures now raise `DeviceNotFound` (with the available-device list) or `AmbiguousMatch`, not a silent fallback. Previously the instance method warned and fell back to policy selection.
- `compile()` no longer auto-detects "auto" -> best device; device is immutable from `__init__`.
- `compile()` no longer post-resolves device label from selected providers via `get_ep_device_map()`.
- `is_compatible()` no longer accepts a `device=` override.

## Cross-file impact
- **Imports added:**
  - `from dataclasses import dataclass`
  - `from .ep_device import AmbiguousMatch, DeviceNotFound, EPDevice, EPMonitorMismatch, expand_ep_name, lookup_device_spec`
  - `from .monitor.ep_monitor import EPMonitor` (and lazy `NullEPMonitor` inside `perf`)
- **Imports removed:**
  - `from typing import ClassVar`
  - `from collections.abc import Generator` (was TYPE_CHECKING-only)
  - Inline `from ..sysinfo.device import get_ep_device_map` inside `compile()` removed.
- **Depends on:** `.ep_device` (EPDevice, exceptions, name helpers, catalog lookup), `.ep_registry.WinMLEPRegistry`, `.monitor.ep_monitor.EPMonitor` and `NullEPMonitor`, `.stats.PerfStats`, `..onnx.is_compiled_onnx`, `..core.onnx_utils.get_io_config`.
- **Depended on by:** `qairt/qairt_session.py` (subclass; also imports `_build_session_options`), `commands/perf.py`, `eval/evaluate.py`, `models/auto.py`, `models/winml/base.py`, `compiler/stages/compile.py` — all migrated to pass `EPDevice` per the commit body.

## Risks / subtleties
- **Auto-reset of compiled session inside `perf()`**: silently destroys the existing `InferenceSession` (logger.warning only). If a caller holds a reference to provider state on the previous session it will be invalidated. Mitigated by the object-identity-preserving fast path when monitor contributes no differing options.
- **Save/restore lifecycle in `perf()`** snapshots `_active_session_option_entries`, `_provider_options`, `_ep`. Note `_active_session_option_entries` is initialized to `{}` and never written elsewhere in this file — the save/restore appears to be future-proofing or maintained for symmetry with monitor `get_session_options()`; verify monitor pipeline isn't expected to mutate it.
- **Monitor `__exit__` errors are swallowed** (`except Exception: pass`) so they cannot override body exceptions. This is a deliberate exception-transparency choice but could mask real teardown bugs in monitors.
- **C-2 ordering invariant** is encoded by a `getattr(effective_monitor, "requires_session_teardown", False)` check. Monitors that forget to declare the attribute fall to the default-False path; the CSV-flush correctness of QNNMonitor depends on its class-level flag being set correctly.
- **Eager session construction in `__init__`** changes failure timing — registration / device-mismatch errors that used to surface on first `run()` now surface at construction. Catch-and-handle code at call sites needs review.
- **Bug A fix lifecycle:** when `enable_ep_context=True`, `_session` stays `None` after `__init__`. `__del__` reads `self._session = None` defensively but anything that calls `run()` between `__init__` and `compile()` triggers `auto-compile`. Compile errors raise `CompilationError` from `compile()`, not from `__init__`.
- **Bug B fix:** `ort.ModelCompiler` is wrapped in `_suppress_native_output(compile_log)` to redirect QNN SDK native stdout. The compile-log path is `<onnx_path>.parent/compile.log` — implicit filesystem write.
- **`_build_op_type_map` swallows all exceptions** and logs at DEBUG. Op-tracing monitors must tolerate `{}` and fall back to EP-authoritative source. Failure is silent in production logs.
- **`expand_ep_name(monitor.ep_name)`** comparison in `perf()` will raise from `.ep_device` if `monitor.ep_name` is an unknown short name. Confirm `expand_ep_name` is safe on `None` (guarded by the prior `monitor.ep_name is not None` check) and on unknown names.
- The `_ep` instance attribute is still maintained as a "legacy alias; TODO Task 10: replace consumers and remove" — downstream consumers still read it.
- **`_detect_best_device`, `_get_compile_suggestion`, `_get_install_suggestion`** still exist and reference the old "auto"/"npu"/"gpu" string device taxonomy, but `_detect_best_device` is no longer called by `compile()`. Dead code or transitional?

## Open questions / TODOs surfaced
- `_ep` legacy alias is explicitly flagged "TODO Task 10: replace consumers and remove" — outstanding cleanup.
- `_active_session_option_entries` snapshot/restore in `perf()` appears unused inside this file — is anything outside the class writing to it (e.g. via monitor `__enter__`)? If not, the snapshot dance is dead weight.
- `_detect_best_device` and the auto/npu/gpu-string helper methods (`_get_compile_suggestion`, `_get_install_suggestion`) are no longer reachable in the auto path — candidates for deletion.
- Should `_build_op_type_map` be a free function (parity with `_build_session_options`)? Currently a `@staticmethod` on `WinMLSession` despite not using cls/self.
- The class-docstring shrunk dramatically; module-level API docs / sphinx-style examples are now missing.
