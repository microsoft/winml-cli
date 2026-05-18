# Review: `src/winml/modelkit/config/precision.py`

**Status:** modified
**Lines added/removed:** 1+ / 1-

## 1. Purpose

`config/precision.py` is a pure-function module: given device, precision,
EP hint, and available-device list, it emits a `PrecisionPolicy` dataclass
with fully resolved fields. It also defines the `VALID_EPS` set and the
`_EP_TO_DEVICE` mapping, both of which are consumed by other modules in
this same PR. This diff is a one-line docstring update — but the module as
a whole is a key dependency of the `compile.py` change.

## 2. Changes summary

Single change: in the `resolve_precision` docstring (line 227), the
`available_devices` argument description is updated from
`"…from resolve_device()…"` to `"…from resolve_device_category()…"` to
match the renamed function in `sysinfo`. No logic changes, no new symbols.

## 3. Per-symbol review

### `_EP_TO_DEVICE` (dict, module-level, line 85-94)

- **Role:** Maps short EP alias → device category. Used by `resolve_precision`
  when inferring device from an explicit `ep=` hint, and imported directly
  by `compiler/stages/compile.py` to reconstruct device from context.
- **Current entries:**
  ```
  "qnn":      "npu"
  "vitisai":  "npu"
  "dml":      "gpu"
  "migraphx": "gpu"
  "tensorrt": "gpu"
  "cuda":     "gpu"
  "openvino": "gpu"
  "cpu":      "cpu"
  ```
- **Invariants:** Keys must be short names (lower-case, no
  `ExecutionProvider` suffix). Values must be one of `{"npu", "gpu",
  "cpu"}`. The set of keys equals `VALID_EPS`.
- **Risks / concerns:**
  - `"cuda"` maps to `"gpu"`. This is consistent with the device category
    model, but `"cuda"` is not present in `session/ep_device.py`'s
    `_SHORT_TO_CANONICAL`. If a caller constructs an `EPDevice` via
    `resolve_device("cuda", "gpu")`, `expand_ep_name("cuda")` returns
    `"cuda"` (unknown short form → passthrough via `canonicalize_ep_name`),
    not `"CUDAExecutionProvider"`. This is a latent mismatch between
    `_EP_TO_DEVICE` and `_SHORT_TO_CANONICAL`.
  - `"tensorrt"` similarly maps GPU → but `_SHORT_TO_CANONICAL` has no
    `"tensorrt"` entry (it has `"nv_tensorrt_rtx"` for the new provider
    and the alias `"nvtensorrtrtxexecutionprovider"`). The old TensorRT EP
    canonical name `"TensorrtExecutionProvider"` would not be reachable via
    `expand_ep_name("tensorrt")` — it returns `"tensorrt"` unchanged.
  - `"openvino"` maps to `"gpu"` but OpenVINO can target CPU as well.
    The category mapping is a simplification; it becomes incorrect if a
    caller passes `ep="openvino"` expecting a CPU build.
  - **`compile.py` dependency**: `_EP_TO_DEVICE` is imported directly by
    `compile.py` (not via a public API). This creates a tight coupling
    between the compiler and the precision config. If `_EP_TO_DEVICE` is
    renamed or moved, `compile.py` breaks. Adding a public accessor
    function would be safer.

---

### `VALID_EPS` (frozenset, module-level, line 97)

- **Role:** Set of recognized short EP names. Used by `resolve_precision`
  to validate the `ep=` argument before proceeding.
- **Definition:** `frozenset(_EP_TO_DEVICE.keys())` — derived from the
  mapping, so it stays in sync automatically.
- **Invariants:** Always equals the key set of `_EP_TO_DEVICE`.
- **Risks / concerns:** `cuda` and `tensorrt` are valid per `VALID_EPS` but
  may not produce a correct `EPDevice` (see `_EP_TO_DEVICE` note above).
  Validation passes but downstream behavior is undefined for these two EPs
  when used with the new `resolve_device()` path.

---

### `resolve_precision` (docstring only change)

- **Role:** Pure function: `(device, precision, ep, available_devices, task)` →
  `PrecisionPolicy`.
- **Signature:** Unchanged.
- **Behavior:** Unchanged. Docstring line 227 updated to reference
  `resolve_device_category()`.
- **Invariants:** Unchanged.
- **Risks / concerns:** None introduced by this diff.
- **Tests:** `tests/unit/config/test_precision.py` — comprehensive coverage
  of the resolution logic.

## 4. Cross-cutting

- `_EP_TO_DEVICE` is a private symbol (underscore prefix) but is now
  imported directly by `compile.py`. This makes it a de-facto semi-public
  API. Consider promoting it to `EP_TO_DEVICE` (public) with an explicit
  export, or adding a `get_device_for_ep(ep: str) -> str` accessor in this
  module to formalize the contract.
- The gap between `_EP_TO_DEVICE` keys (`cuda`, `tensorrt`) and
  `ep_device._SHORT_TO_CANONICAL` entries is a pre-existing inconsistency
  not introduced by this diff, but it becomes more visible now that
  `compile.py` uses this map to construct `EPDevice` objects.

## 5. Confidence level

High for the diff itself (one-line docstring). Medium for the broader
`_EP_TO_DEVICE` usage given the `cuda`/`tensorrt` coverage gaps.

## 6. Verbatim risk inventory

| # | Location | Risk |
|---|----------|------|
| R1 | `precision.py:91-92` | `"cuda"` and `"tensorrt"` in `_EP_TO_DEVICE` / `VALID_EPS` are valid strings but have no matching entries in `ep_device._SHORT_TO_CANONICAL`, so `expand_ep_name("cuda")` returns `"cuda"` unchanged — `resolve_device("cuda", "gpu")` would attempt to register `"cuda"` as a canonical EP name and likely fail. |
| R2 | `precision.py:85` (`_EP_TO_DEVICE`) | Private symbol imported externally by `compile.py` — any rename or restructuring of `precision.py` breaks the compiler without a static dependency signal. |
| R3 | `precision.py:92` | `"openvino": "gpu"` oversimplifies; OpenVINO targets CPU too. A caller passing `ep="openvino"` without explicit `device` always gets a GPU `EPDevice`. |
