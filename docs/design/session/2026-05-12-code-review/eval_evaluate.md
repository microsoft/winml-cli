# Review: `src/winml/modelkit/eval/evaluate.py`

**Status:** modified
**Lines added/removed:** 14+ / 3-

## 1. Purpose

`evaluate.py` contains `_load_model`, which is the internal model loader for the eval
pipeline. This PR moves EPDevice construction from inside `WinMLAutoModel` to the
`_load_model` boundary — the same "CLI boundary resolve" pattern applied in `perf.py`.
The eval command itself (`eval.py`) remains responsible only for device-category
resolution; `_load_model` now resolves the full EPDevice before calling into
`WinMLAutoModel`.

## 2. Changes summary

- Import `resolve_device` from `..session.ep_device`.
- New inline `_default_ep_for_device` dict to derive `ep_str` from `config.device`.
- `resolve_device(ep=ep_str, device=device)` constructs `ep_device`.
- `WinMLAutoModel.from_onnx`: `device=config.device` → `ep_device=ep_device`.
- `WinMLAutoModel.from_pretrained`: `device=config.device` dropped; `ep_device`
  passed as positional argument (not keyword).

## 3. Per-symbol review

### `_load_model`

- **Role:** Load a `WinMLPreTrainedModel` from either a cached ONNX path or a HF model
  ID, using the eval config's device/task settings.
- **Signature:** `def _load_model(config: WinMLEvaluationConfig) -> WinMLPreTrainedModel`
- **Behavior:**
  1. Derives `ep_str` via `_default_ep_for_device.get(device, "cpu")`.
  2. Calls `resolve_device(ep=ep_str, device=device)` to get `ep_device`.
  3. On the ONNX path: `WinMLAutoModel.from_onnx(onnx_path=..., ep_device=ep_device, task=..., skip_build=True)`.
  4. On the HF path: `WinMLAutoModel.from_pretrained(config.model_id, ep_device, task=config.task)`.
- **Invariants:** `config.model_id` is required (ValueError guard on line 133).
  `config.device` is always a lowercase string from `eval.py:resolved_device`.
- **Risks / concerns:**
  - **Positional argument on `from_pretrained`:** `ep_device` is passed as the second
    positional argument at line 156–158:
    ```python
    return WinMLAutoModel.from_pretrained(
        config.model_id,
        ep_device,       # positional — no keyword name
        task=config.task,
    )
    ```
    The `from_onnx` call uses `ep_device=ep_device` (keyword). The asymmetry is a
    readability hazard: if `WinMLAutoModel.from_pretrained` reorders its parameters in
    a future refactor, this silently passes `ep_device` to the wrong parameter. Should
    use `ep_device=ep_device` for consistency.
  - **HF auto-build crash path (same as perf.py):** `WinMLAutoModel.from_pretrained`
    triggers model build/compile via `compile.py`. The audit-flagged premature
    `ort.InferenceSession` inside `WinMLSession.__init__` still applies here. This PR
    does not fix it.
  - `_default_ep_for_device` is duplicated from `perf.py` (third copy). See Cross-cutting.
  - `device = config.device.lower()` is called on line 139. `config.device` is already
    lowercased by `eval.py` (`resolved_device` from `resolve_device_category`), so the
    double lower is harmless but redundant.
  - `resolve_device` can raise `EPNotDiscovered`, `DeviceNotFound`, or `AmbiguousMatch`
    (from `ep_device.py`). None of these are caught in `_load_model`. They will
    propagate up through `evaluate()` and surface as unformatted tracebacks to the
    user. The eval command's `try/except` block in `eval.py` only catches explicit
    exception types; a `DeviceNotFound` exception would leak as a raw traceback.
- **Tests:** `tests/unit/eval/test_eval.py` (config roundtrip, `_resolve_task` — the
  `_load_model` function itself is not directly unit-tested in the reviewed files;
  it is exercised via mocked integration tests).

## 4. Cross-cutting concerns

- **Audit gap — HF auto-build crash:** Same as perf.py: `from_pretrained` still
  triggers the broken compile path for QNN+NPU. Not introduced by this PR.
- **`_default_ep_for_device` duplication:** Third copy of the same dict across
  `eval.py:_load_model`, `perf.py:_load_model`, and `perf.py:perf`. Should be
  centralized.
- **Legacy `device=` callers:** Both `WinMLAutoModel` call sites in this file have been
  updated. No legacy `device=` kwarg remains.
- **Missing `--ep` on eval CLI:** `WinMLEvaluationConfig` has no `ep` field. The EP is
  fully determined by the device string via the hardcoded map. Users on a machine with
  both QNN and VitisAI NPU support cannot specify which EP to use for evaluation.

## 5. Confidence level

**Medium.** The EPDevice construction logic is correct. The positional-argument
asymmetry on `from_pretrained` is a low-risk but real consistency gap. The lack of
exception handling for `resolve_device` errors is a UX regression risk for error
messages.

## 6. Verbatim risk inventory

| Severity | Location | Description |
|----------|----------|-------------|
| **Medium** | `evaluate.py:156–158` | `ep_device` passed as positional arg to `from_pretrained`; `from_onnx` uses keyword. Positional binding is fragile to future parameter reordering. |
| **Medium** | `evaluate.py:141` + whole function | `resolve_device` exceptions (`EPNotDiscovered`, `DeviceNotFound`, `AmbiguousMatch`) are uncaught; propagate as raw tracebacks through `evaluate()` to the user. No `try/except` at the `_load_model` boundary. |
| **Medium** | `evaluate.py:138` + `perf.py:472` + `perf.py:1552` | `_default_ep_for_device` dict duplicated at 3 inline sites across two files. |
| **Low** | `evaluate.py:139` | `config.device.lower()` redundant — `eval.py` already lowercases via `resolve_device_category`. Harmless but noisy. |
| **Low** | `evaluate.py` (whole file) | No `--ep` option in the eval CLI; EP choice is always hardcoded via device map. Pre-existing gap. |
