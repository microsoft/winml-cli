# EP + Device Selection Refactor — Design Spec

**Initial Date:** 2026-05-11
**Version:** 1.2
**Status:** Spec — implementation pending
**Branch:** `feat/op-tracing-refactor`
**Scope:** This PR (in-flight). The op-tracing refactor is already merged into the branch; this is the focused EP+device selection slice being added on top.
**Companion branch (dependency):** `feat/update-pkg-deps` — provides `ep_path.py`, revised `ep_registry.py`, `EP_DLL_NAMES`, `canonicalize_ep_name()`, `discover_eps()`, `MODELKIT_EP_PATH`. This PR rebases ON it after merge.
**Backward compatibility:** None. Hard break (Option A). Every callsite is updated in this PR.

## 1. Background

`WinMLSession` today resolves `(EP, device)` in two paths and neither is deterministic. The explicit path goes through `_find_ep_device(ep_name)` in `src/winml/modelkit/session/session.py:460-469`, which iterates registered `OrtEpDevice`s and returns the **first** match by EP name. When a single EP exposes multiple `OrtEpDevice` entries — one per claimed hardware device — this is silently arbitrary. Verified on Snapdragon X-Elite: `QNNExecutionProvider` exposes four entries (NPU, two GPU entries, CPU). `_find_ep_device("QNNExecutionProvider")` could return the CPU entry instead of the NPU. It happens to return NPU today on the test box; that is luck, not design.

The second path is the policy-based autoep route: `WinMLSession.__init__` accepts `device="auto"` and high-level `prefer_npu` / `prefer_gpu` policies, which feed `ort.SessionOptions.set_provider_selection_policy(...)`. Two selection mechanisms in one constructor doubles the surface area and forces every call site to reason about whether the policy will pick the same device the user assumes. Neither path produces a portable, serializable record of *what was actually selected*, which the perf monitor and CLI both need.

The op-tracing refactor (already shipped in this branch) made the monitor pipeline EP-aware and added structured op-trace results that flow through CLI commands. That work surfaced the need for a plain-data `(EP, device)` descriptor that the same modules can carry across boundaries without depending on ORT runtime types. This refactor introduces that descriptor and replaces the two ambiguous paths with one explicit, deterministic resolution step.

## 2. Goals & non-goals

### Goals

- Single, explicit, deterministic `(EP, device)` selection performed once at session construction.
- Plain-data descriptor `EPDevice` that serializes to JSON and flows through CLI args, perf-monitor configs, and module boundaries with no ORT runtime dependency.
- Native multi-device support: QNN-NPU, QNN-GPU, and QNN-CPU are independently addressable and distinguishable.
- Hard break (Option A): no backward-compat shims, no deprecation period. Every call site is updated in this PR.

### Non-goals

- Modifying `feat/update-pkg-deps` work (`ep_path.py`, `ep_registry.py`). This PR rebases on it after merge and **only adds** one method (`register_ep`) to `ep_registry.py`.
- Inventing new EP discovery, plugin loading, or DLL search mechanisms. `feat/update-pkg-deps` already provides `discover_eps()`, `MODELKIT_EP_PATH`, `EP_DLL_NAMES`, and `canonicalize_ep_name()`.
- Adding a hardware-compatibility error path beyond what is available today. The other PR may optionally expose `_ep_is_compatible` publicly; if so, we consume it. If not, this PR ships without it.
- Refactoring `WinMLQairtSession` (the parallel QAIRT direct-binary session class at `session/qairt/qairt_session.py`). It is fixed in a follow-up PR. This PR focuses solely on `WinMLSession` and the register-based ORT path.

## 3. Design

### 3.1 The `EPDevice` descriptor

```python
# src/winml/modelkit/session/ep_device.py  (new file)

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EPDevice:
    """Pure-data identifier of one (EP, hardware-device) binding target.

    Frozen, JSON-serializable, no ORT runtime dependency. Constructed by
    resolve_device() or rehydrated from JSON via from_dict(). Flows through
    CLI args, perf-monitor configs, and WinMLSession constructors.
    """

    ep: str           # canonical EP name (e.g. "QNNExecutionProvider")
    device: str       # "cpu" | "gpu" | "npu" (lowercase invariant — see __post_init__)
    vendor_id: int    # PCI/ACPI vendor ID
    device_id: int    # PCI/ACPI device ID
    vendor: str = ""  # display-only vendor string (e.g. "Qualcomm")

    def __post_init__(self) -> None:
        # Enforce lowercase device invariant — lets us compare against
        # OrtHardwareDeviceType.name.lower() downstream without a helper.
        if self.device != self.device.lower():
            object.__setattr__(self, "device", self.device.lower())

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EPDevice": ...
```

Naming convention: `EPDevice` (all-caps `EP`) matches existing `EPMonitor` / `EPConfig` in this codebase.

**Invariants:**

- `EPDevice.ep` always stores the canonical EP name. `canonicalize_ep_name()` is applied at the boundary (inside `resolve_device`); user-typed aliases (`"qnn"`, `"QNN"`) never reach an `EPDevice` instance.
- `EPDevice.device` is always lowercase (`"cpu"` / `"gpu"` / `"npu"`), enforced by `__post_init__`. This lets the downstream filter compare directly against `OrtHardwareDeviceType.name.lower()` — no string-to-enum helper needed.

`EPDevice` is the only shape that crosses module boundaries. ORT `OrtEpDevice` is a runtime handle and stays inside `session.py`; it is never stored on `WinMLSession` and never serialized.

### 3.2 `resolve_device(ep, device)`

```python
# new public function in src/winml/modelkit/session/ep_device.py
def resolve_device(ep: str, device: str) -> EPDevice:
    """Resolve a (user-friendly EP name, device kind) pair to a frozen
    EPDevice descriptor.

    Args:
        ep: User-supplied EP name. Short forms ("qnn") and aliases
            expanded via expand_ep_name() (see helper below).
        device: "cpu" | "gpu" | "npu" (case-insensitive).

    Raises:
        EPNotDiscovered:      EP plugin not in catalog or MODELKIT_EP_PATH.
        EPRegistrationFailed: ort.register_execution_provider_library raised.
        DeviceNotFound:       EP registered, but no matching OrtEpDevice.
        AmbiguousMatch:       multiple OrtEpDevice match the descriptor
                              after dedup (should not occur; raised loudly).
    """
```

Implementation pseudo-code:

1. `ep_canonical = expand_ep_name(ep)` — short → canonical, with passthrough for already-canonical names.
2. `devices = WinMLEPRegistry.get_instance().register_ep(ep_canonical)` — new method, see §3.5.
3. Filter `devices` by `d.device.type.name.lower() == device.lower()` (both sides reduce to lowercase strings — no helper).
4. Dedup by `(vendor_id, device_id)` — handles QNN's duplicate-GPU entries on X-Elite.
5. Expect exactly one match. Build and return `EPDevice(ep=ep_canonical, device=device.lower(), vendor_id=..., device_id=..., vendor=...)`.
6. Zero matches → `DeviceNotFound`. More than one after dedup → `AmbiguousMatch`.

#### `expand_ep_name(name)` — short-form → canonical

```python
# in session/ep_device.py
_SHORT_TO_CANONICAL: Final[dict[str, str]] = {
    "qnn":             "QNNExecutionProvider",
    "openvino":        "OpenVINOExecutionProvider",
    "vitisai":         "VitisAIExecutionProvider",
    "migraphx":        "MIGraphXExecutionProvider",
    "nv_tensorrt_rtx": "NvTensorRtRtxExecutionProvider",
    "dml":             "DmlExecutionProvider",
    "cpu":             "CPUExecutionProvider",
}


def expand_ep_name(name: str) -> str:
    """Expand a short EP name to its canonical form; passthrough if already canonical.

    Universal rule in this codebase: "xxx" is the short name of
    "xxxExecutionProvider" (case-folded for lookup). Examples:
      "qnn"                                -> "QNNExecutionProvider"
      "QNNExecutionProvider"               -> "QNNExecutionProvider" (passthrough)
      "NvTensorRTRTXExecutionProvider"     -> "NvTensorRtRtxExecutionProvider"
        (via canonicalize_ep_name() — alias casing)

    Used at every public boundary that takes an EP name: resolve_device(),
    perf-time monitor validation, CLI parsing.
    """
    canonical = _SHORT_TO_CANONICAL.get(name.lower())
    if canonical is not None:
        return canonical
    # Already-canonical names (or unknown ones) flow through ep_path's
    # alias table for casing fixes (e.g. NvTensorRTRTX -> NvTensorRtRtx).
    return canonicalize_ep_name(name)
```

This is **purely additive** to `feat/update-pkg-deps`'s `canonicalize_ep_name()`: we delegate to it for canonical-form aliasing and only add the short-form lookup layer here. The other PR is not modified.

**Naming collision (must fix in this PR):** `src/winml/modelkit/sysinfo/device.py:146` already defines `resolve_device(device="auto") -> tuple[str, list[str]]`. That function returns a *category* (the kind of device available on the host), not an `EPDevice`. **Rename it to `resolve_device_category`** in this PR so the names are unambiguous in a single namespace.

### 3.3 `WinMLSession.__init__` (hard break)

```python
# revised src/winml/modelkit/session/session.py
class WinMLSession:
    def __init__(
        self,
        onnx_path: str | Path,
        ep_device: EPDevice,                           # required, no default
        *,
        ep_config: EPConfig | None = None,
        base_session_options: ort.SessionOptions | None = None,
    ): ...
```

`EPMonitor` is **not** a constructor parameter. It is passed to `WinMLSession.perf(monitor=...)` per benchmark window — preserving today's pattern of running plain inference on the same session and only attaching a monitor for the tracing-enabled perf run. See §3.4 for how `perf()` rebuilds session options with the monitor.

Removed (deleted, not deprecated):

- `device="auto"` parameter.
- `ep="qnn"` short-name keyword on `WinMLSession.__init__`. (Short forms still work as user input through `resolve_device("qnn", "npu")` via `expand_ep_name`.)
- Policy-based autoep (`PREFER_NPU` / `PREFER_GPU` paths through `set_provider_selection_policy`).
- `_EP_NAME_MAP`, `DEVICE_POLICY_MAP`.
- The `_find_ep_device(ep_name)` helper.

**Not removed:** the save/restore lifecycle inside `perf()` (`session.py:670-707` — `saved_sess_entries`, `saved_prov`, `saved_ep`). Those guard against re-entry while the session is rebuilt with monitor options; they remain. What changes is *how* the merge happens: instead of mutating `self._provider_options` mid-flight, `perf()` calls the pure `_build_provider_options(ep_device, ep_config, monitor)` once at perf-window start and rebuilds the `InferenceSession` with the result. The save/restore protects the same `self` fields as today.

Every call site (CLI commands, perf monitor, tests, internal helpers) constructs an `EPDevice` first via `resolve_device("qnn", "npu")` (or `EPDevice.from_dict(...)` for rehydrated configs) and passes it explicitly. Op-tracing perf runs additionally pass the matching `EPMonitor` to `session.perf(monitor=...)`; that path validates `expand_ep_name(monitor.ep_name) == ep_device.ep` and raises `EPMonitorMismatch` on conflict.

### 3.4 `_build_session_options` (new flow)

`_build_session_options` and `_build_provider_options` are private **free functions** in `session.py` (not methods, not a new module). Pure inputs → pure outputs — unit-testable without instantiating `WinMLSession`. The descriptor-to-handle bridge is inlined; no separate `_select_one` / `to_ort_ep_device` helper.

```python
def _build_session_options(
    ep_device: EPDevice,
    ep_config: EPConfig | None = None,
    ep_monitor: EPMonitor | None = None,
    base_session_options: ort.SessionOptions | None = None,
) -> ort.SessionOptions:
    so = base_session_options or ort.SessionOptions()

    # Monitor's session-level config (e.g. session.disable_cpu_ep_fallback=1).
    if ep_monitor is not None:
        for key, value in ep_monitor.get_session_options().items():
            so.add_session_config_entry(key, value)

    # Bridge descriptor → ORT handle: register, filter, validate.
    devices = WinMLEPRegistry.get_instance().register_ep(ep_device.ep)
    matching = [
        d for d in devices
        if d.device.type.name.lower() == ep_device.device
        and d.device.vendor_id == ep_device.vendor_id
        and d.device.device_id == ep_device.device_id
    ]
    if not matching:
        raise DeviceNotFound(
            f"No OrtEpDevice for {ep_device.ep} matches device="
            f"{ep_device.device}, vendor_id=0x{ep_device.vendor_id:x}, "
            f"device_id=0x{ep_device.device_id:x}. Available: "
            f"{[(d.device.type.name, hex(d.device.vendor_id), hex(d.device.device_id)) for d in devices]}"
        )
    if len(matching) > 1:
        raise AmbiguousMatch(
            f"Multiple OrtEpDevices match {ep_device!r} after dedup — "
            f"registry bug. Matched: {matching}"
        )

    options = _build_provider_options(ep_device, ep_config, ep_monitor)
    so.add_provider_for_devices([matching[0]], options)
    return so
```

`_build_session_options` has **two call sites**, distinguished only by whether `ep_monitor` is set:

```python
# (a) WinMLSession.__init__ (or lazy compile) — plain inference, no monitor
so = _build_session_options(
    self._ep_device,
    self._ep_config,
    None,                                           # no monitor at session creation
    self._base_session_options,
)
self._session = ort.InferenceSession(self._onnx_path, sess_options=so)

# (b) WinMLSession.perf(monitor=...) — tracing-enabled rebuild for the perf window
def perf(self, monitor: EPMonitor | None = None, ...):
    if monitor is not None and monitor.ep_name is not None:
        if expand_ep_name(monitor.ep_name) != self._ep_device.ep:
            raise EPMonitorMismatch(...)

    saved_sess_entries = dict(self._active_session_option_entries)
    saved_prov = dict(self._provider_options)
    saved_ep = self._ep                              # preserved for legacy state
    try:
        so = _build_session_options(
            self._ep_device, self._ep_config, monitor, self._base_session_options
        )
        # Update self._provider_options snapshot so other code paths see the
        # current merge; rebuild the InferenceSession with the monitor-aware so.
        self._provider_options = _build_provider_options(self._ep_device, self._ep_config, monitor)
        self._session = ort.InferenceSession(self._onnx_path, sess_options=so)
        with monitor:
            ...                                     # run benchmark
    finally:
        self._active_session_option_entries = saved_sess_entries
        self._provider_options = saved_prov
        self._ep = saved_ep
        # Rebuild the base session without the monitor on the way out so
        # post-perf state is clean (today's restore semantics).
        self._session = ort.InferenceSession(
            self._onnx_path,
            sess_options=_build_session_options(
                self._ep_device, self._ep_config, None, self._base_session_options
            ),
        )
```

#### Provider-options layering — `_build_provider_options`

```python
def _build_provider_options(
    ep_device: EPDevice,
    ep_config: EPConfig | None,
    ep_monitor: EPMonitor | None,
) -> dict[str, str]:
    """Flat provider_options for add_provider_for_devices().

    Three layers, each overrides the previous:
      1. EP-specific defaults from ep_device (e.g. QNN backend_type — must
         be present for ORT to dispatch to the right backend).
      2. User overrides from ep_config.provider_options.
      3. EPMonitor-required options (e.g. QNN profiling_level,
         profiling_file_path).

    Monitor wins last because tracing correctness depends on its options
    actually reaching the EP. Callers who want to disable tracing should
    drop the monitor, not override its keys.
    """
    options = _ep_defaults(ep_device)
    if ep_config and ep_config.provider_options:
        options.update(ep_config.provider_options)
    if ep_monitor is not None:
        options.update(ep_monitor.get_provider_options())
    return options


def _ep_defaults(ep_device: EPDevice) -> dict[str, str]:
    """EP-specific defaults driven by ep_device.device.

    Most EPs return {} — they pick up settings via
    ep_config.provider_options and ep_monitor.get_provider_options().
    Only EPs that must signal a backend/device kind at registration
    appear here.
    """
    match ep_device.ep:
        case "QNNExecutionProvider":
            return {"backend_type": _QNN_BACKEND[ep_device.device]}
        case _:
            return {}


_QNN_BACKEND: Final[dict[str, str]] = {"npu": "htp", "gpu": "gpu", "cpu": "cpu"}
```

**Matching strictness:** strict 4-tuple `(ep, device.type, vendor_id, device_id)` everywhere, **including CPU**. Rationale: reproducibility. QNN-CPU's `vendor_id` reflects the host CPU (AMD on X-Elite, Intel elsewhere), so a serialized `EPDevice` for `qnn+cpu` will not roundtrip across host-CPU vendors. That is the correct failure mode — the user re-resolves on the target machine. A predictable error beats a host-dependent silent success.

**Relation to today's `perf()` mid-flight rewrite.** Today, `session.py:679` mutates `self._provider_options = {**saved, **extra}` inside `perf()`, then restores in `finally`. The new design preserves the **save/restore lifecycle** (still needed to protect re-entry) but replaces the inline `{**saved, **extra}` expression with one call to the pure `_build_provider_options(ep_device, ep_config, monitor)`. The merge order is identical (monitor wins last); the merge is now pure, upfront, and inspectable from the call site.

### 3.5 `WinMLEPRegistry.register_ep(name)` — new method

```python
# ADDITION to src/winml/modelkit/session/ep_registry.py (post-rebase)
def register_ep(self, ep_name: str) -> list[ort.OrtEpDevice]:
    """Register a single discovered EP and return its claimed devices.

    Idempotent: if already registered, returns the current device list
    without re-loading the DLL. Callers must pass canonicalize_ep_name(...)
    on user-supplied names first.

    Raises:
        EPNotDiscovered:      ep_name absent from self._ep_paths.
        EPRegistrationFailed: ort.register_execution_provider_library
                              raised (wrapped with the original exception).
    """
```

This is **additive**. It does not modify `register_to_ort()` (the bulk registration used by `winml sys --list-ep`) or any other existing public surface on the registry. `register_to_ort` continues to work unchanged for catalog-style listings.

### 3.6 Layering vs `feat/update-pkg-deps`

```
   feat/update-pkg-deps (merge first)
   ┌──────────────────────────────────────────────┐
   │ session/ep_path.py                           │
   │   EpSource ABCs, discover_eps(),             │
   │   canonicalize_ep_name(), MODELKIT_EP_PATH,  │
   │   EP_DLL_NAMES                               │
   │                                              │
   │ session/ep_registry.py                       │
   │   WinMLEPRegistry, register_to_ort()         │
   └──────────────────────────────────────────────┘
                       ▲
                       │ consumes
                       │
   this PR (feat/op-tracing-refactor, rebased)
   ┌──────────────────────────────────────────────┐
   │ ADD to session/ep_registry.py:               │
   │   WinMLEPRegistry.register_ep(name)          │
   │     -> list[OrtEpDevice]                     │
   │                                              │
   │ NEW session/ep_device.py:                    │
   │   EPDevice (frozen dataclass)                │
   │   resolve_device(ep, device) -> EPDevice     │
   │   exceptions (EPNotDiscovered, ...)          │
   │                                              │
   │ MODIFY session/session.py:                   │
   │   WinMLSession.__init__ hard-break           │
   │     (ep_monitor stays on perf(), not ctor)   │
   │   _build_session_options new flow            │
   │     (inlines descriptor → handle bridge,     │
   │      accepts ep_monitor=None|monitor)        │
   │   _build_provider_options (3-layer merge)    │
   │   _ep_defaults (per-EP, QNN backend_type)    │
   │   perf() validates monitor.ep_name match     │
   │     via expand_ep_name; save/restore kept    │
   │                                              │
   │ MODIFY sysinfo/device.py:                    │
   │   resolve_device  ->  resolve_device_category│
   │                                              │
   │ MODIFY every call site (CLI, perf, tests)    │
   └──────────────────────────────────────────────┘
```

## 4. Error taxonomy

New exception types defined in `session/ep_device.py`:

| Exception | Condition | Actionable message includes |
|---|---|---|
| `EPNotDiscovered` | EP plugin not installed, not in catalog, not in `MODELKIT_EP_PATH`. | Discovered EP names, hint to set `MODELKIT_EP_PATH` or install the plugin package. |
| `EPRegistrationFailed` | `ort.register_execution_provider_library(...)` raised. | EP name, DLL path attempted, original exception chained. |
| `DeviceNotFound` | EP registered, no `OrtEpDevice` matches `(device.type, vendor_id, device_id)`. | List of `OrtEpDevice`s the EP did expose so the user can pick a different `device` or vendor. |
| `AmbiguousMatch` | More than one `OrtEpDevice` matches after dedup by `(vendor_id, device_id)`. Should not occur. | Full list of conflicting devices; this is a bug signal, not a user error. |
| `EPMonitorMismatch` | `WinMLSession.__init__` received an `ep_monitor` whose `ep_name` does not match `ep_device.ep`. | Both names, hint that monitor and session must agree. |

Each error message is actionable: it states what was requested, what was found, and what the user can do next.

## 5. Migration plan (this PR)

1. **Wait for `feat/update-pkg-deps` to merge into `main`.** Rebase `feat/op-tracing-refactor` onto the new tip.
2. Add `WinMLEPRegistry.register_ep(...)` to `ep_registry.py` (additive only).
3. Add new file `session/ep_device.py` with `EPDevice` (incl. `__post_init__` lowercase invariant), `resolve_device`, `expand_ep_name` helper + `_SHORT_TO_CANONICAL` table, and the five exception types (including `EPMonitorMismatch`).
4. Rename `sysinfo/device.py::resolve_device` → `resolve_device_category` and update every existing caller.
5. Rewrite `WinMLSession.__init__` per §3.3 (hard break — old kwargs deleted; ep_monitor is *not* a ctor param).
6. Rewrite `_build_session_options` per §3.4 as a private free function — inlined descriptor → handle bridge, accepts `ep_monitor` (`None` from `__init__` call site, the actual monitor from `perf()` call site).
7. Add `_build_provider_options` and `_ep_defaults` as private free functions in `session.py` (three-layer merge: ep defaults → user → monitor).
8. Refactor `WinMLSession.perf()` to call `_build_session_options(monitor=monitor)` once at window start (and again with `monitor=None` in the `finally` to restore the bare session). **Preserve the save/restore lifecycle around `session.py:670-707`** — `saved_sess_entries`, `saved_prov`, `saved_ep` remain. Add the `expand_ep_name(monitor.ep_name) == self._ep_device.ep` validation that raises `EPMonitorMismatch` at the top of `perf()`.
9. Update every call site: CLI commands (`wmk perf`, `wmk run`, others), perf monitor wiring, internal helpers, all tests.
10. Update test fixtures: replace `device="auto"` and `ep="qnn"` ctor kwargs with `ep_device=resolve_device("qnn", "npu")` (or `EPDevice(...)` literals where pure-data is preferred). `session.perf(monitor=...)` call sites unchanged.

## 6. Verification plan (this PR)

- **E2E (must work):** `uv run wmk perf <convnext-onnx> --ep qnn --device npu` produces the same per-op timings and HW chart as today's `--ep qnn`. Today's path is non-deterministic by first-match-by-name but happens to land on NPU on X-Elite; the new path produces the same result *deterministically*.
- **Unit tests:** `uv run pytest tests/` — all pass. Skips only for hardware-required tests (CUDA, DirectML, AVX), per `CLAUDE.md`.
- **Lint:** `uv run ruff check --fix` clean.
- **Architecture regression test:** add a test that asserts `WinMLSession.__init__` rejects string `ep=` and `device=` kwargs (catches accidental revival of autoep / policy paths in future edits).
- **Roundtrip test:** `EPDevice.from_dict(ep_device.to_dict()) == ep_device` for a representative QNN-NPU descriptor.
- **Provider-options layering test:** `_build_provider_options(ep_device=qnn_npu, ep_config=EPConfig(provider_options={"profiling_level":"off"}), ep_monitor=QNNMonitor(...))` returns `{"backend_type":"htp","profiling_level":"detailed","profiling_file_path":"..."}` — confirms monitor wins over user override (tracing-correctness invariant).
- **Monitor/ep_device mismatch test:** Construct `session = WinMLSession(..., ep_device=resolve_device("openvino","npu"))` then call `session.perf(monitor=QNNMonitor(...))` — must raise `EPMonitorMismatch`.
- **`expand_ep_name` test:** asserts `expand_ep_name("qnn") == expand_ep_name("QNNExecutionProvider") == "QNNExecutionProvider"`, plus alias casing via `NvTensorRTRTX -> NvTensorRtRtx`.
- **`perf()` save/restore test:** assert that after `session.perf(monitor=QNNMonitor(...))` raises mid-run, `session._provider_options`, `session._active_session_option_entries`, and `session._ep` are restored to their pre-perf values.

## 7. Open decisions resolved during brainstorming

- Class name is **`EPDevice`** (all-caps `EP`), matching `EPMonitor` / `EPConfig`.
- `EPDevice` is a **pure-data descriptor**: frozen dataclass, no ORT runtime in `__init__`, no methods that touch ORT. `device` field is lowercase-invariant (`__post_init__` enforces).
- **Strict 4-tuple matching** `(ep, device.type, vendor_id, device_id)` everywhere, including CPU. Host-dependent serialization is an accepted, predictable failure mode.
- **Hard break (Option A)**: no autoep, no `device="auto"`, no policy paths, no compatibility shims. Every call site updated in this PR.
- `_build_session_options` and `_build_provider_options` are private **free functions** inside `session.py`, not methods and not a new module. The descriptor → `OrtEpDevice` bridge is **inlined** in `_build_session_options` (no `_select_one` / `to_ort_ep_device` helper). The string → `OrtHardwareDeviceType` mapping is **inlined** via `.name.lower()` (no `_to_ort_device_type` helper).
- `_build_provider_options` is **three-layer**: ep defaults → `ep_config.provider_options` → `ep_monitor.get_provider_options()`. Monitor wins last because tracing correctness depends on its options actually reaching the EP.
- `_build_session_options` plumbs `ep_monitor.get_session_options()` via `SessionOptions.add_session_config_entry()`. The pure-function merge replaces the inline `{**saved, **extra}` expression inside today's `perf()`; the save/restore lifecycle around `session.py:670-707` is **preserved** (still needed to protect re-entry).
- `EPMonitor` integrates via `WinMLSession.perf(monitor=...)`, **not** the constructor. Plain inference sessions are constructed without a monitor; tracing-enabled perf runs attach one for the perf window only. Validation (`expand_ep_name(monitor.ep_name) == ep_device.ep`) happens at the top of `perf()` and raises `EPMonitorMismatch` on conflict.
- **Short-form EP names** (`"qnn"`, `"openvino"`, `"dml"`, ...) remain valid user input. `expand_ep_name(name)` in `session/ep_device.py` maps short → canonical via a `_SHORT_TO_CANONICAL` lookup, falling through to `canonicalize_ep_name()` for already-canonical names. Monitors keep short-form `ep_name` ClassVars (`QNNMonitor.ep_name = "qnn"` etc.); validation expands at the comparison site.
- `register_ep` lives in **`ep_registry.py`** as an additive method on `WinMLEPRegistry`; bulk `register_to_ort()` is unchanged.
- This PR is a **layered consumer** of `feat/update-pkg-deps`: rebase first, then add. No edits to that PR's surface beyond the one additive method.
- `sysinfo/device.py::resolve_device` is **renamed to `resolve_device_category`** to keep one unambiguous `resolve_device` in the namespace.

## 8. References

- `feat/update-pkg-deps` branch (remote `gh`, repo `microsoft/ModelKit`) — source of `ep_path.py`, the revised `ep_registry.py`, `EP_DLL_NAMES`, and `canonicalize_ep_name()`.
- ORT 1.23+ register-based API: `SessionOptions.add_provider_for_devices(list[OrtEpDevice], dict[str, str])`, `register_execution_provider_library(name, dll_path)`.
- `src/winml/modelkit/session/session.py:460-469` — the `_find_ep_device(ep_name)` helper being replaced.
- `src/winml/modelkit/sysinfo/device.py:146` — the `resolve_device` being renamed to `resolve_device_category`.
