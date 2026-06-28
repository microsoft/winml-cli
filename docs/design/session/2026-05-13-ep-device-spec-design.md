# `EPDeviceSpec` — Single Source of Truth for EP/Device Taxonomy

**Date:** 2026-05-13
**Status:** Design finalized; implementation pending
**Branch:** `feat/op-tracing-refactor`
**Base:** `d271dfb3`
**Driver:** Phase 1 experiment showed +3× throughput from `htp_performance_mode='burst'` on QNN-NPU. The current `_ep_defaults()` if-ladder doesn't scale to 13 (EP, device) variants. Two duplicate dicts (`_EP_TO_DEVICE`, `_DEVICE_TO_PROVIDER`) need consolidation.

## 1. Decision

Replace **three** existing structures with **one** typed dataclass catalog:

| Removed | Reason |
|---|---|
| `_EP_TO_DEVICE: dict[str, str]` | EP→device mapping; one direction of the (EP, device) relation |
| `_DEVICE_TO_PROVIDER: dict[str, str \| None]` | device→EP mapping; the other direction of the same relation |
| `_ep_defaults(ep_device)` if/elif ladder | Would grow linearly with each new (EP, device) variant |

| Added | Purpose |
|---|---|
| `EPDeviceSpec` (frozen dataclass) | One catalog entry per supported (EP, device) target |
| `EP_DEVICE_SPECS: tuple[EPDeviceSpec, ...]` | The single ordered catalog (order encodes deduction preference) |
| `lookup_device_spec(ep, device)` | O(N=13) lookup helper |
| `default_device_for_ep(ep)` / `default_ep_for_device(device)` | Derived from `EP_DEVICE_SPECS` (replaces the two dicts above) |

## 2. Type relationship: `EPDeviceSpec` (template) → `EPDevice` (instance)

```
EPDeviceSpec
  ep: str
  device: str
  default_provider_options: Mapping[str, str]
       │
       │  + machine info from OrtEpDevice handle
       │  + vendor_id, device_id, vendor (read at resolve_device time)
       ▼
EPDevice
  ep: str
  device: str
  vendor_id: int
  device_id: int
  vendor: str
```

One `EPDeviceSpec` materializes to many `EPDevice` instances (one per machine). This mirrors the well-known Kubernetes `PodSpec`→`Pod` pattern, the OpenAPI spec→instance pattern, and the Pydantic config-vs-runtime distinction.

## 3. The catalog — 13 variants

Order matters: first variant matching a given `ep` wins for "ep-only" deduction; first variant matching a given `device` wins for "device-only" deduction.

```python
@dataclass(frozen=True, kw_only=True, slots=True)
class EPDeviceSpec:
    """One supported (EP, device) target in the catalog.

    Distinct from EPDevice:
      - EPDeviceSpec is the *kind-of-target* (machine-independent).
      - EPDevice is the *runtime instance* (machine-specific, carries
        vendor_id / device_id from the OrtEpDevice handle).
    Many EPDevices map to one EPDeviceSpec.
    """
    ep: str
    device: str
    default_provider_options: Mapping[str, str] = field(default_factory=dict)


EP_DEVICE_SPECS: tuple[EPDeviceSpec, ...] = (
    # ---- QNN family ----
    EPDeviceSpec(
        ep="QNNExecutionProvider",
        device="npu",
        default_provider_options={
            # Verified 2026-05-13: +3× throughput on ResNet-50 vs default mode
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
        },
    ),
    EPDeviceSpec(ep="QNNExecutionProvider", device="gpu"),   # TODO: measure
    EPDeviceSpec(ep="QNNExecutionProvider", device="cpu"),

    # ---- OpenVINO family ----
    # TODO: verify whether `device_type` is needed under add_provider_for_devices,
    # or auto-derived from OrtEpDevice handle (like QNN's backend_type).
    EPDeviceSpec(ep="OpenVINOExecutionProvider", device="npu"),
    EPDeviceSpec(ep="OpenVINOExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="OpenVINOExecutionProvider", device="cpu"),

    # ---- Single-device EPs ----
    EPDeviceSpec(ep="VitisAIExecutionProvider", device="npu"),
    EPDeviceSpec(ep="DmlExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="MIGraphXExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="TensorrtExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="CUDAExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="NvTensorRtRtxExecutionProvider", device="gpu"),

    # ---- Bundled CPU ----
    EPDeviceSpec(ep="CPUExecutionProvider", device="cpu"),
)
```

**Why most entries have empty `default_provider_options`:** we only override an EP's own default when we have **measured evidence** ours is better. Today that's only QNN-NPU. The 12 TODOs are documented so future contributors with the right hardware can fill them in with citations, not speculation.

## 4. Derived helpers (no more if/else ladder)

```python
_BY_KEY: Final = {(s.ep, s.device): s for s in EP_DEVICE_SPECS}


def lookup_device_spec(ep: str, device: str) -> EPDeviceSpec | None:
    """O(1) lookup by exact (ep, device) match."""
    return _BY_KEY.get((ep, device))


def default_device_for_ep(ep: str) -> str | None:
    """First variant in catalog whose ep matches. Replaces _EP_TO_DEVICE.

    Order in EP_DEVICE_SPECS encodes preference:
      QNN's first variant is npu  → default_device_for_ep("QNN...") == "npu"
      DML only has gpu             → default_device_for_ep("Dml...") == "gpu"
    """
    return next((s.device for s in EP_DEVICE_SPECS if s.ep == ep), None)


def default_ep_for_device(device: str) -> str | None:
    """First variant in catalog whose device matches. Replaces _DEVICE_TO_PROVIDER.

    Order in EP_DEVICE_SPECS encodes preference:
      Among npu variants, QNN comes first → default_ep_for_device("npu") == "QNNExecutionProvider"
      Among gpu variants, OpenVINO        → default_ep_for_device("gpu") == "OpenVINOExecutionProvider"

    NOTE: this changes behavior slightly from the old _DEVICE_TO_PROVIDER:
      Old: _DEVICE_TO_PROVIDER["gpu"] = "dml" (short name)
      New: returns the canonical full name; callers that need short form
           must call short_ep_name() on the result.
    """
    return next((s.ep for s in EP_DEVICE_SPECS if s.device == device), None)


def _ep_defaults(ep_device: EPDevice) -> dict[str, str]:
    """Per-variant default provider options. Replaces the if/elif ladder.

    Returns a fresh dict (callers may mutate; we don't share the catalog entry's storage).
    """
    spec = lookup_device_spec(ep_device.ep, ep_device.device)
    return dict(spec.default_provider_options) if spec else {}
```

## 5. Why Pattern A (dataclass tuple) over alternatives

| Pattern | Add a new *property* (e.g. `default_datatype`) | Add a new *variant* | Pros | Cons |
|---|---|---|---|---|
| **A — dataclass tuple** ✅ | 1 line on dataclass + zero or more overrides | 1 line in tuple | Pure data; single file; type-safe via dataclass; iteration is `for s in EP_DEVICE_SPECS` | Methods on EPDeviceSpec read the data fields (no per-variant overrides) — fine for our use case |
| B — Enum with rich values | Touch every member tuple + `__init__` signature (positional, fragile) | 1 line | Named identity (`EPVariant.QNN_NPU`) | Position-based init; refactoring footgun |
| C — class hierarchy | 1 line on base + override where differs | A whole class per variant (13 classes) | Per-variant methods natural | Heavy; `__subclasses__()` registration fragile to import order |
| D — module-level constants + dispatch dict | Add a new module-level constant | Touch the dispatch dict | Simple | Spreads config across many names; doesn't unify with EP↔device lookups |

**Pattern A** chosen because the new properties we anticipate (`default_datatype`, `supports_op_tracing`, `typical_use_case`, vendor compatibility hints, etc.) are all **data**. Adding a field to `EPDeviceSpec` is one line; existing entries don't need to be touched (defaults take over). If we later need **behavior per variant**, we can add methods on `EPDeviceSpec` that read its own fields — no migration.

## 6. Future scalability — adding more properties

Example: add `default_datatype` and `supports_op_tracing`.

```python
@dataclass(frozen=True, kw_only=True, slots=True)
class EPDeviceSpec:
    ep: str
    device: str
    default_provider_options: Mapping[str, str] = field(default_factory=dict)
    default_datatype: Literal["fp32", "fp16", "int8"] = "fp32"          # NEW
    supports_op_tracing: bool = False                                    # NEW
```

Catalog entries pick up defaults automatically; only entries that differ need overrides:

```python
EPDeviceSpec(
    ep="QNNExecutionProvider", device="npu",
    default_provider_options={...},
    default_datatype="fp16",        # overrides default
    supports_op_tracing=True,       # overrides default
),
EPDeviceSpec(ep="OpenVINOExecutionProvider", device="npu"),   # all defaults
# … 11 more untouched
```

`kw_only=True` makes adding/reordering fields safe — no positional-arg footguns.
`slots=True` saves memory and prevents accidental attribute creation.

## 7. Naming decision — why `EPDeviceSpec`

Considered:
- `EPVariantSpec` — explicit but verbose
- `EPVariant` — clean but slightly ambiguous (variant of an EP? or of an EP+device combination?)
- `EPSpec` — too short; loses the "device" half
- `EPDeviceSpec` — chosen ← mirrors `EPDevice`, makes the spec→instance relationship explicit
- `EPProfile` — collides with profiling vocabulary in this codebase
- `EPTarget` — "target" is overloaded in build tooling
- `EPCatalogEntry` — verbose; describes catalog-membership rather than the thing

`EPDeviceSpec` wins because it pairs cleanly with `EPDevice` and follows the well-understood K8s `PodSpec`→`Pod` convention.

## 8. Migration plan (the implementation work)

1. Add `EPDeviceSpec` dataclass to `src/winml/modelkit/session/ep_device.py`.
2. Define `EP_DEVICE_SPECS` tuple — 13 entries.
3. Add `lookup_device_spec(ep, device)`, `default_device_for_ep(ep)`, `default_ep_for_device(device)`.
4. Rewrite `_ep_defaults(ep_device)` to consult the catalog (single lookup, no if/elif).
5. **Delete `_EP_TO_DEVICE`** — replaced by `default_device_for_ep`.
6. **Delete `_DEVICE_TO_PROVIDER`** and `get_provider_for_device()` — replaced by `default_ep_for_device`. (Note: `get_provider_for_device` returned the short name; callers must update or use `short_ep_name(default_ep_for_device(device))`.)
7. Update `resolve_device()` deduction matrix to use the new helpers.
8. Update `session/__init__.py` re-exports: add `EPDeviceSpec`, `EP_DEVICE_SPECS`, `lookup_device_spec`, `default_device_for_ep`, `default_ep_for_device`. Remove `get_provider_for_device` from the exports if any consumer used it.
9. Audit all callers via grep:
   - `_EP_TO_DEVICE` → must be 0 hits
   - `_DEVICE_TO_PROVIDER` → must be 0 hits
   - `get_provider_for_device` → migrate or delete callers
10. Update tests:
    - `tests/unit/session/test_ep_device.py` — new tests for `lookup_device_spec`, `default_device_for_ep`, `default_ep_for_device`
    - Tests that previously poked `_EP_TO_DEVICE` keys directly → switch to `EP_DEVICE_SPECS` introspection or the helpers
11. **DO commit the Phase 1 burst-mode change in the same commit** — the catalog entry for QNN-NPU includes the `htp_performance_mode='burst'` defaults. This delivers both the +3× perf win and the refactor in one atomic landing.

## 9. Verification

```bash
cd D:/BYOM/ModelKit_PRs/op_tracing
uv run ruff check --fix src/ tests/

# Architecture regression
uv run pytest tests/unit/architecture/ -v --tb=short

# Session + commands + models suites
uv run pytest tests/unit/session/ tests/unit/commands/ tests/unit/models/ tests/unit/eval/ -v --tb=short

# Full unit suite
uv run pytest tests/unit/ --tb=no -q

# Sanity smoke — must produce the same numbers as the Phase 1 experiment
uv run winml perf -m C:/Users/zhengte/.cache/winml/artifacts/microsoft_resnet-50/imgcls_69f0345d0dbeb3b1_export.onnx --ep qnn --device npu --iterations 100 --warmup 10 2>&1 | tail -20
# Expected: Avg ~1.90 ms, Throughput ~525 samp/s

# Regression — direct ONNX + ep/device deduction
uv run winml perf -m <fp32>.onnx --ep qnn               # device deduced
uv run winml perf -m <fp32>.onnx --device npu           # ep deduced

# Catalog introspection
uv run python -c "
from winml.modelkit.session import EP_DEVICE_SPECS, lookup_device_spec, default_device_for_ep, default_ep_for_device
print('Total variants:', len(EP_DEVICE_SPECS))
print('default_device_for_ep(qnn):    ', default_device_for_ep('QNNExecutionProvider'))
print('default_ep_for_device(npu):    ', default_ep_for_device('npu'))
print('default_ep_for_device(gpu):    ', default_ep_for_device('gpu'))
print('lookup(qnn,npu).default_options:', lookup_device_spec('QNNExecutionProvider', 'npu').default_provider_options)
"
# Expected output:
#   Total variants: 13
#   default_device_for_ep(qnn):     npu
#   default_ep_for_device(npu):     QNNExecutionProvider
#   default_ep_for_device(gpu):     OpenVINOExecutionProvider
#   lookup(qnn,npu).default_options: {'htp_performance_mode': 'burst', 'htp_graph_finalization_optimization_mode': '3'}
```

## 10. What this does NOT change

- `EPDevice` runtime instance — same shape, same fields.
- `resolve_device(ep, device)` public API — same signature; only internal deduction uses new helpers.
- `WinMLSession.__init__(ep_device=...)` — unchanged.
- CLI flag shapes — `--ep` and `--device` still optional, still deduced via `resolve_device`.
- Existing tests (mostly) — they import via the session facade, so as long as the facade re-exports the new helpers, no consumer breaks.

## 11. Out of scope

- Adding speculative defaults for unverified variants (OpenVINO, TensorRT, CUDA, etc.). Each new default needs hardware measurement first.
- Per-variant session-config defaults (e.g., custom `ep.context_*` keys). Session options stay separate from provider options; they'll be addressed when concrete callers need them.
- Removing the lazy `_get_ep_registry` shim. Out of scope.

## 12. Rollback

The change is a single atomic commit. If anything goes wrong, `git revert <sha>` cleanly restores the previous state (which already works for the 6 CLI commands per `2026-05-13-remaining-issues.md`).
