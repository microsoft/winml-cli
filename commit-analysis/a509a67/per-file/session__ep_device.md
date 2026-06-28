# src/winml/modelkit/session/ep_device.py

## TL;DR
Brand-new module (471 lines) establishing the EP/device taxonomy as a single source of truth: the `EPDeviceSpec` catalog (13 entries) plus the `EPDevice` frozen runtime dataclass, name canonicalization (`expand_ep_name` / `short_ep_name` / `canonicalize_ep_name`), the `resolve_device(ep, device)` deduction-and-resolution entry point used at CLI boundaries, and a 5-exception taxonomy. Replaces the pre-state `_EP_TO_DEVICE`, `_DEVICE_TO_PROVIDER`, `get_provider_for_device`, and the `_ep_defaults` if/else ladder that previously lived elsewhere.

## Diff metrics
- Lines added: 471
- Lines removed: 0
- **New file**

## Role before vs after
- **Before:** Did not exist. EP→device and device→EP relationships were spread across `_EP_TO_DEVICE` / `_DEVICE_TO_PROVIDER` constants and a `_ep_defaults` if/else ladder (locations not visible in this file's diff but described in the commit body), with `_find_ep_device(ep_name)` doing non-deterministic first-match selection.
- **After:** Authoritative location for EP/device identity, naming, catalog, deduction, resolution, and the exception taxonomy. The only file allowed to define `_EP_NAME_ALIASES`, `_SHORT_TO_FULL`, `_FULL_TO_SHORT`, `_BY_KEY` (per the commit-body directive). All EP/device queries elsewhere in the repo route through helpers exported from here.

## Symbol-level changes
- **`EPNotDiscovered`** — added (exception). EP plugin missing from catalog/`MODELKIT_EP_PATH` and not bundled with ORT.
- **`EPRegistrationFailed`** — added (exception). `ort.register_execution_provider_library` raised; original chained.
- **`DeviceNotFound`** — added (exception). EP registered, but no `OrtEpDevice` matches the descriptor (raised by `resolve_device` line 452).
- **`AmbiguousMatch`** — added (exception). Multiple `OrtEpDevice` match after dedup — labeled in the docstring as a registry bug, not a user error (`resolve_device` line 459).
- **`EPMonitorMismatch`** — added (exception). `Monitor.ep_name` disagrees with `EPDevice.ep`. Not raised in this file — defined for use by consumers (monitor code).
- **`EPDevice`** — added (`@dataclass(frozen=True)`, lines 54–82). Fields: `ep: str`, `device: str`, `vendor_id: int`, `device_id: int`, `vendor: str = ""`. `__post_init__` lowercases `device` via `object.__setattr__` (frozen workaround). Has `to_dict` and `from_dict` for JSON round-trip. `OrtEpDevice` handle is **not** stored; commit-body comment says it's re-derived at session-build time inside `session.py`.
- **`canonicalize_ep_name`** — added (line 96). Reads from `_EP_NAME_ALIASES` (currently one entry: `nvtensorrtrtxexecutionprovider → NvTensorRtRtxExecutionProvider`). Marked as a migration stub to be replaced by `from .ep_path import canonicalize_ep_name` once `feat/update-pkg-deps` merges.
- **`_EP_NAME_ALIASES`** — added private constant; the migration stub.
- **`_SHORT_TO_FULL`** — added private constant (9 entries): `qnn`, `openvino`, `vitisai`, `migraphx`, `nv_tensorrt_rtx`, `cuda`, `tensorrt`, `dml`, `cpu`. Per commit body, `cuda` and `tensorrt` were newly added here — previously `VALID_EPS` listed them but `expand_ep_name` passed them through unchanged, causing `EPNotDiscovered` at register time.
- **`expand_ep_name`** — added (line 114). Short→full lookup; falls through to `canonicalize_ep_name` for already-full names.
- **`_FULL_TO_SHORT`** — added; computed at module load as inverse of `_SHORT_TO_FULL`.
- **`short_ep_name`** — added (line 132). Full→short via `_FULL_TO_SHORT`, falls back to `full.removesuffix("ExecutionProvider").lower()` — never raises.
- **`EPDeviceSpec`** — added (`@dataclass(frozen=True, kw_only=True, slots=True)`, lines 150–163). Fields: `ep`, `device`, `default_provider_options: Mapping[str, str]`.
- **`EP_DEVICE_SPECS`** — added catalog (lines 166–199), 13 entries. Order encodes preference: QNN-NPU (primary), DML-GPU (primary), CPU-CPU (primary), then QNN secondary (GPU, CPU), OpenVINO (NPU/GPU/CPU), VitisAI-NPU, MIGraphX-GPU, Tensorrt-GPU, CUDA-GPU, NvTensorRtRtx-GPU. QNN-NPU embeds `htp_performance_mode="burst"` + `htp_graph_finalization_optimization_mode="3"` defaults (commit body claims +3× ResNet-50 throughput vs empty defaults).
- **`_BY_KEY`** — added; O(1) `(ep, device) → EPDeviceSpec` lookup dict built from the catalog.
- **`VALID_EPS`** — added; frozenset of **short** names derived from the catalog.
- **`VALID_DEVICES`** — added; frozenset of device categories from the catalog.
- **`lookup_device_spec(ep, device)`** — added (line 211). O(1) exact-match catalog query using full EP names.
- **`default_device_for_ep(ep)`** — added (line 224). Replaces `_EP_TO_DEVICE`; returns first device for matching EP via catalog scan.
- **`default_ep_for_device(device)`** — added (line 240). Replaces `_DEVICE_TO_PROVIDER`; returns full canonical EP name (docstring explicitly notes this differs from the old function, which returned short).
- **`eps_for_device(device)`** — added (line 261). Replaces inline `candidate_eps` lists; returns frozenset of canonical (full) EP names, empty for unknown devices (no raise).
- **`ep_to_device(ep)`** — added (line 280). Short EP name → device. Raises `ValueError` for unknown EP.
- **`auto_detect_device() -> str`** — added. Returns the auto-picked device category (`"npu" | "gpu" | "cpu"`). Lazily imports `get_available_devices` from `..sysinfo.hardware` and `available_eps` from `.ep_registry`, walks the hardware priority list, and returns the first category for which at least one compatible EP is currently available. Falls back to `"cpu"` when no compatible EP exists; warns but does not raise. This replaces the device-string-only contract that lived under `sysinfo.resolve_device(device="auto")` in the pre-branch tree.
- **`WinMLEPRegistry`** — added as **module-level sentinel `Any = None`**, patched lazily by `_get_ep_registry()` to break the `ep_registry → ep_device` circular import. Tests can patch `winml.modelkit.session.ep_device.WinMLEPRegistry` directly.
- **`_get_ep_registry()`** — added. Lazy `importlib.import_module(".ep_registry", ...)` to populate the module-level `WinMLEPRegistry` sentinel on first real call.
- **`resolve_device(ep=None, device=None)`** — added. The primary public resolver and the typed replacement for `sysinfo.resolve_device(device="auto")` from the pre-branch tree. Deduction matrix: both given → validate; ep only → look up device from catalog; device only → look up default EP from catalog; neither (or `device="auto"`) → call `auto_detect_device()` then fall through. Resolution phase calls `registry.register_ep(ep_full)`, filters by device-type lowercase match, dedups by `(vendor_id, device_id)`, raises `DeviceNotFound` if none / `AmbiguousMatch` if >1 after dedup. Returns an `EPDevice` carrying the chosen device's IDs and vendor — a fully typed object, not the old `(category, available_devices_list)` tuple.

## Behavior / contract changes
- **`EPDevice` is the new abstraction at the EP-resolution boundary.** Where the pre-branch code returned an unstructured `(category, available_devices_list)` tuple from `sysinfo.resolve_device`, the post-branch code returns a frozen `EPDevice` dataclass carrying the full `(ep, device, vendor_id, device_id, vendor)` quintuple. Downstream consumers (`WinMLSession.__init__`, `commands/perf.py`, `eval/evaluate.py`, `compiler/stages/compile.py`) consume the typed object rather than parsing the tuple shape.
- **5 new exception types** form a structured failure taxonomy (`EPNotDiscovered`, `EPRegistrationFailed`, `DeviceNotFound`, `AmbiguousMatch`, `EPMonitorMismatch`). All `# noqa: N818` for ruff's "exception names should end in `Error`" rule. The pre-branch path could only raise `ValueError`.
- **Non-deterministic `_find_ep_device(ep_name)` first-match selection is replaced** by `resolve_device(ep, device)` with explicit (EP, device) descriptors. Per commit body: hard break, no compat shims.
- **`EPDevice.device` is normalized to lowercase** at construction (line 67) — callers cannot rely on case-preserving behavior.
- **`EPDevice` does not carry an `OrtEpDevice` handle** — only `vendor_id`/`device_id`/`vendor` as plain data. The handle is re-derived at session-build time per the module docstring (lines 9–13).
- **Catalog order is load-bearing** (per docstring at line 166): `default_device_for_ep("QNNExecutionProvider") == "npu"` (not `"gpu"` or `"cpu"`); `default_ep_for_device("gpu") == "DmlExecutionProvider"` (not OpenVINO/CUDA/etc.). Reordering the tuple silently changes deduction behavior.
- **`default_ep_for_device` now returns full canonical names** (e.g. `"DmlExecutionProvider"`), not short names — explicit `NOTE` in docstring at line 248. Callers that need short must wrap with `short_ep_name(...)`.
- **`eps_for_device("nonexistent")` returns empty frozenset, does not raise** (line 277). Callers must check membership.
- **`ep_to_device("nonexistent")` raises `ValueError`** (line 295), not one of the new EP exceptions.
- **`auto_detect_device()` warns but does not raise** when no compatible EP is available for the prioritised hardware; it falls back to `"cpu"`. This warn-but-don't-raise behaviour was inherited from the pre-branch `sysinfo.resolve_device("auto")` path.
- **`resolve_device` performs side-effectful registration** via `registry.register_ep(ep_full)`. Any caller is implicitly triggering DLL loads. The pre-branch `sysinfo.resolve_device` was pure introspection and did not load DLLs.
- **QNN-NPU gets burst-mode defaults baked into the catalog** (lines 175–180). All session builds targeting QNN-NPU through the catalog inherit `htp_performance_mode='burst'` and `htp_graph_finalization_optimization_mode='3'` unless overridden by the three-layer merge (ep_defaults → ep_config → monitor) described in the commit body.

## Cross-file impact
- **Imports added (this file):** `importlib`, `logging`, `collections.abc.Mapping`, `dataclasses.asdict/dataclass/field`, `typing.Any/Final`.
- **Imports removed:** N/A (new file).
- **Public API exported via `__init__.py`:** all top-level non-`_` names — see the `__init__.py` analysis.
- **Modules that now depend on this file:** per commit body, `models/auto.py`, `models/winml/base.py`, `commands/perf.py`, `eval/evaluate.py`, `compiler/stages/compile.py`, plus `session/session.py` (consumes `EPDevice` as `WinMLSession.__init__` positional arg), `session/ep_registry.py` (imports `EPNotDiscovered` + `EPRegistrationFailed`), and `config/precision.py` (commit-body comment at line 147 mentions this).
- **Modules this file now depends on:** lazy/internal — `..sysinfo.hardware.get_available_devices` (lazy import inside `auto_detect_device`), `.ep_registry.available_eps` (lazy inside `auto_detect_device`) and `.ep_registry.WinMLEPRegistry` (lazy via `_get_ep_registry`, inside `resolve_device`).

## Risks / subtleties
- **Circular-import side-step is fragile.** `WinMLEPRegistry: Any = None` at line 362 is intentionally a module-level rebindable name. The comment (lines 356–361) warns that tests patch this binding directly. If a future refactor changes `ep_registry` to no longer require `ep_device`, the lazy load becomes dead code; if `ep_device` ever gains an eager `from .ep_registry import WinMLEPRegistry`, the import cycle reactivates.
- **`_EP_NAME_ALIASES` is a migration stub** (comment at lines 87–90). It must be removed and replaced with `from .ep_path import canonicalize_ep_name` when `feat/update-pkg-deps` merges. If the merge is delayed, the alias-table approach must continue to be maintained — every new casing-mismatch must be added by hand.
- **Catalog order is undocumented contract.** Two preference axes (per-device-first-EP, per-EP-first-device) are both encoded by the same tuple ordering. The comment at lines 167–172 explains the intent but not the constraint; future editors may not understand that swapping QNN-NPU and DML-GPU breaks the "primary NPU" / "primary GPU" semantics.
- **`EPDeviceSpec.default_provider_options` is a `Mapping[str, str]`** — values are `str`, not `int`/`bool`. The QNN entry uses `"3"` (string), not `3` (int). Anyone wiring new defaults must stringify.
- **`expand_ep_name` is case-insensitive only on the short side** (`name.lower()` at line 121). A full-form name with wrong casing falls through to `canonicalize_ep_name`, which only fixes the one entry currently in `_EP_NAME_ALIASES`.
- **`resolve_device` dedups by `(vendor_id, device_id)` only** (line 442). If two `OrtEpDevice` rows have identical IDs but differ on other fields, only the first survives — the docstring at line 462 calls this a "registry bug" but the dedup itself happens here silently.
- **`EPDevice.from_dict` requires `ep`, `device`, `vendor_id`, `device_id` keys** (lines 77–82); only `vendor` is optional. A round-trip from an older serialized form missing `vendor_id`/`device_id` raises `KeyError`, not a typed exception.
- **`getattr(chosen.device, "vendor", "") or ""`** at line 470 — defends against both missing attribute and falsy value (e.g. `None`). The double `or ""` is intentional but easy to mis-read as redundant.

## Open questions / TODOs surfaced
- **Migration TODO** (lines 87–90): replace `canonicalize_ep_name` stub with `from .ep_path import canonicalize_ep_name` once `feat/update-pkg-deps` merges; delete `_EP_NAME_ALIASES`.
- **OpenVINO TODO** (line 188): "verify whether `device_type` is needed under `add_provider_for_devices`, or auto-derived from `OrtEpDevice` handle (like QNN's `backend_type`)." Affects OpenVINO NPU/GPU/CPU rows in the catalog.
- **QNN-GPU TODO** (line 185): comment "TODO: measure" — the QNN secondary-GPU row exists in the catalog but its performance characteristics are unverified.
- **`EPMonitorMismatch` is defined but not raised in this file.** Its production-call site lives elsewhere (presumably `session.py` or `monitor/*`); the relationship between `Monitor.ep_name` and `EPDevice.ep` is not documented here.
- **`get_provider_for_device` removal** (commit body) — this function name does not appear in the post-state, so it has been deleted somewhere else; any external caller that imported it now breaks at import time.
