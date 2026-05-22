# Review: `src/winml/modelkit/session/ep_device.py`

**Status:** new file
**Lines added/removed:** 213+ / 0-
**Diff command:** `git diff 1bea4cf..HEAD -- src/winml/modelkit/session/ep_device.py`

---

## 1. Purpose of this file

`ep_device.py` introduces the `EPDevice` frozen dataclass — the single plain-data descriptor that represents one `(EP, hardware-device)` binding target throughout the refactored architecture. It also houses the five exception types that constitute the new EP resolution error taxonomy, the `resolve_device(ep, device)` public entry-point, and a set of EP name canonicalization helpers (`expand_ep_name`, `short_ep_name`, `canonicalize_ep_name`). The module is intentionally ORT-runtime-free at import time; the ORT handle is derived inside `session.py` at session-build time and never stored here.

---

## 2. Changes summary

- New file; no prior version.
- Defines 5 public exception types: `EPNotDiscovered`, `EPRegistrationFailed`, `DeviceNotFound`, `AmbiguousMatch`, `EPMonitorMismatch`.
- Defines `EPDevice` frozen dataclass with `__post_init__` lowercase invariant, `to_dict`, `from_dict`.
- Defines `canonicalize_ep_name` stub (with `MIGRATION:` marker awaiting `feat/update-pkg-deps`) backed by `_EP_NAME_ALIASES`.
- Defines `_SHORT_TO_CANONICAL` table and `expand_ep_name` public function.
- Defines `_CANONICAL_TO_SHORT` inverse table and `short_ep_name` public function.
- Module-level `WinMLEPRegistry: Any = None` sentinel + `_get_ep_registry()` lazy importer (circular-import avoidance).
- Defines `resolve_device(ep, device) -> EPDevice` public function.

---

## 3. Per-symbol review

### Exception types (`EPNotDiscovered`, `EPRegistrationFailed`, `DeviceNotFound`, `AmbiguousMatch`, `EPMonitorMismatch`)

- **Role:** Typed error taxonomy for EP resolution. Each maps to one failure mode in the resolution pipeline.
- **Signature:** `class X(Exception):`
- **Behavior:** Plain exceptions with `# noqa: N818` (exception names don't end in `Error` — intentional, matches `EPConfig`, `EPMonitor` naming convention in this codebase). All five match spec §4 exactly.
- **Invariants:** None beyond standard exception semantics.
- **Risks / concerns:** `EPMonitorMismatch` is raised inside `session.py:perf()`, not here. The placement in this file is correct because the exception semantically belongs to the EP-device contract, not to session lifecycle.
- **Tests:** `tests/unit/session/test_ep_device.py` (indirect via `resolve_device` tests), `tests/unit/session/test_winml_session.py:670-721` (mismatch test).

---

### `EPDevice`

- **Role:** Frozen, JSON-serializable pure-data identifier of one `(EP, hardware-device)` binding target.
- **Signature:** `@dataclass(frozen=True) class EPDevice: ep: str, device: str, vendor_id: int, device_id: int, vendor: str = ""`
- **Behavior:** Constructed by `resolve_device` or rehydrated via `from_dict`. `__post_init__` enforces lowercase `device` invariant using `object.__setattr__` (required for frozen dataclasses). `to_dict` / `from_dict` provide symmetric JSON round-trip via `dataclasses.asdict`.
- **Invariants:** `EPDevice.device` is always lowercase (`"cpu"` / `"gpu"` / `"npu"`). `EPDevice.ep` is always the canonical EP name (enforced by `resolve_device` before construction; callers using the literal constructor bypass this — acceptable for test fixtures).
- **Risks / concerns:** `ep` is not validated at construction time — a caller using the literal constructor can pass a non-canonical string. This is acceptable because `resolve_device` is the authoritative path and tests use literal construction for fixture values. `from_dict` uses `d.get("vendor", "")` for backward-compatible deserialization of persisted configs that predate the `vendor` field. No issues.
- **Tests:** `tests/unit/session/test_ep_device.py:22-48`.

---

### `canonicalize_ep_name`

- **Role:** Stub for EP name casing normalization, covering the `NvTensorRtRtx` alias until `feat/update-pkg-deps` merges.
- **Signature:** `def canonicalize_ep_name(name: str) -> str`
- **Behavior:** Looks up `name.lower()` in `_EP_NAME_ALIASES`; if absent, returns `name` unchanged. This is a deliberate no-op stub for all EP names other than the NvTensorRt variant.
- **Invariants:** Never raises. Passthrough for unknown names.
- **Risks / concerns:** The `MIGRATION:` comment at line 82-85 is explicit. One-line replacement when the dependency merges. The stub is safe but only covers the single alias known today; any new alias added to `feat/update-pkg-deps` will not be visible here until the rebase. This is the accepted tradeoff.
- **Tests:** Covered indirectly via `expand_ep_name` alias-casing tests.

---

### `expand_ep_name`

- **Role:** Canonical entry point for all user-supplied EP name strings — maps short forms to canonical, with casing-fix passthrough for already-canonical names.
- **Signature:** `def expand_ep_name(name: str) -> str`
- **Behavior:** Looks up `name.lower()` in `_SHORT_TO_CANONICAL`; if found, returns the canonical string directly. Otherwise delegates to `canonicalize_ep_name(name)` for casing fixes. The passthrough chain means an already-canonical name like `"QNNExecutionProvider"` returns unchanged.
- **Invariants:** Never raises; unknown short names flow through `canonicalize_ep_name` which returns them unchanged.
- **Risks / concerns:** An unrecognized short form (e.g. `"cuda"`) will not expand. `"cuda"` is not in `_SHORT_TO_CANONICAL`, so it returns `"cuda"` (not `"CUDAExecutionProvider"`). If a caller passes it to `resolve_device`, the downstream `register_ep` will raise `EPNotDiscovered` with the unrecognized string, which produces a usable error. Not a bug but worth noting — adding `"cuda"` to `_SHORT_TO_CANONICAL` if needed is a one-line change.
- **Tests:** `tests/unit/session/test_ep_device.py:51-69`.

---

### `short_ep_name`

- **Role:** Inverse of `expand_ep_name` — canonical → display-friendly short form for CLI and log output.
- **Signature:** `def short_ep_name(canonical: str) -> str`
- **Behavior:** Looks up in `_CANONICAL_TO_SHORT`; falls back to `canonical.removesuffix("ExecutionProvider").lower()` for unknown names. Never raises.
- **Invariants:** None — fallback may produce an unexpected string for non-`ExecutionProvider`-suffixed names, but all ORT EPs follow this convention.
- **Risks / concerns:** `_CANONICAL_TO_SHORT` is built at module import time from `_SHORT_TO_CANONICAL`. Any future addition to `_SHORT_TO_CANONICAL` is automatically picked up. No issues.
- **Tests:** Not directly tested; used in CLI display paths.

---

### `WinMLEPRegistry` module-level sentinel + `_get_ep_registry`

- **Role:** Lazy circular-import avoidance shim. `ep_registry.py` imports from `ep_device.py`; `ep_device.py` cannot import from `ep_registry.py` at module load time without a circular import. The sentinel is `None` until first call to `_get_ep_registry`, which uses `importlib.import_module` to defer the import.
- **Signature:** `WinMLEPRegistry: Any = None` (module-level) + `def _get_ep_registry() -> Any`
- **Behavior:** `_get_ep_registry()` uses `global` to rebind `WinMLEPRegistry` on first call. Test patches replace `winml.modelkit.session.ep_device.WinMLEPRegistry` directly, bypassing the lazy-load branch entirely.
- **Invariants:** After the first real call to `resolve_device` in production, `WinMLEPRegistry` is permanently bound to the registry class.
- **Risks / concerns:** The module-level public name `WinMLEPRegistry` (capitalized, `Any`-typed) is unconventional — it looks like a class import but it is a mutable binding. The inline comment at lines 140-145 documents the pattern for future readers. The test-patch documentation is clear. Acceptable, but this is the most unusual design element in the file. Documented in impl-status §4 as "Subtle smell."
- **Tests:** Covered by `test_ep_device.py` resolve tests which mock this binding.

---

### `resolve_device`

- **Role:** Primary public API — maps a `(user-ep-string, device-string)` pair to a frozen `EPDevice` descriptor with exactly one ORT match.
- **Signature:** `def resolve_device(ep: str, device: str) -> EPDevice`
- **Behavior:** (1) Expands `ep` via `expand_ep_name`. (2) Calls `WinMLEPRegistry.get_instance().register_ep(ep_canonical)`. (3) Filters by `d.device.type.name.lower() == device_lower`. (4) Deduplicates by `(vendor_id, device_id)` — handling QNN's duplicate-GPU rows on X-Elite hardware. (5) Raises `DeviceNotFound` / `AmbiguousMatch` on 0 or >1 matches. (6) Returns `EPDevice(ep=ep_canonical, device=device_lower, vendor_id=..., device_id=..., vendor=...)`.
- **Invariants:** Returned `EPDevice.ep` is always canonical. Returned `EPDevice.device` is always lowercase (enforced by `__post_init__` as backup even if `device_lower` is already lowercase). Dedup guarantees at most one unique `(vendor_id, device_id)` row.
- **Risks / concerns:** The `vendor` field uses `getattr(chosen.device, "vendor", "") or ""` (line 212) to handle ORT versions where `vendor` may not exist. The double-fallback (`getattr` with default then `or ""`) is defensive but slightly redundant — `getattr(..., "")` already returns `""` on missing attribute. Harmless but could be simplified. The dedup by `(vendor_id, device_id)` happens *before* raising `AmbiguousMatch`, so duplicates at the ORT level are silently collapsed. This matches spec §3.2 intent (dedup for QNN-GPU rows) but means `AmbiguousMatch` can only be raised if two distinct physical devices claim the same `(vendor_id, device_id)` — an extreme edge case.
- **Tests:** `tests/unit/session/test_ep_device.py:84-132`.

---

## 4. Cross-cutting concerns

**Spec drift:** None. Every symbol matches spec §3.1, §3.2, §4 exactly. The lazy-import shim (`_get_ep_registry`) is not in the spec pseudocode but is documented in `impl-status.md §4` as an intentional deviation with clear justification. The stub `canonicalize_ep_name` carries the `MIGRATION:` marker per spec §3.6.

**Deferred work markers:**
- Line 82: `MIGRATION:` — replace stub with `from .ep_path import canonicalize_ep_name` after `feat/update-pkg-deps` merges. Low-risk, one-line change.

**Dependencies on other files in this group:**
- `ep_registry.py` — imported lazily via `_get_ep_registry()`. Direct dependency on `WinMLEPRegistry.register_ep` (spec §3.5).
- `session.py` — imports `EPDevice`, `resolve_device`, `AmbiguousMatch`, `DeviceNotFound`, `EPMonitorMismatch`, `expand_ep_name` at top level.
- `qairt/qairt_session.py` — imports `EPDevice`, `resolve_device`.

---

## 5. Confidence level

**High.**

The new-file nature means no legacy behavior to regress. Logic is concise and well-commented. Dedup behavior is correct for the stated X-Elite QNN-GPU duplicate use case. The lazy-import shim is unusual but works correctly and is well-documented. The only subtlety is that `ep` is not validated at `EPDevice` construction time (only at `resolve_device` time), which is acceptable given that `resolve_device` is the authoritative path.

What to verify before declaring production-ready:
- Confirm the `vendor` attribute name on `OrtEpDevice` in ORT 1.23+ (line 212 uses `getattr` defensively, which implies uncertainty).
- Confirm `AmbiguousMatch` is actually reachable in practice (dedup by `vendor_id, device_id` may always collapse duplicates, making it a dead raise).

---

## 6. Verbatim risk inventory

| Severity | Location | Description |
|---|---|---|
| MINOR | `ep_device.py:212` | `getattr(chosen.device, "vendor", "") or ""` — double-fallback is redundant; `getattr(..., "")` already covers missing attribute. Simplify to `getattr(chosen.device, "vendor", "")`. |
| MINOR | `ep_device.py:146` | Module-level public name `WinMLEPRegistry: Any = None` is easy to confuse with a class import. Consider renaming to `_WinMLEPRegistry` (private) and updating test patch targets, or add a module-level comment marking it as an internal binding, not a re-export. |
