# Review: `src/winml/modelkit/models/winml/base.py`

**Status:** modified
**Lines added/removed:** 12+ / 5-

## 1. Purpose

`WinMLPreTrainedModel` is the abstract base class for all task-specific
WinML inference wrappers (e.g., `WinMLModelForImageClassification`). It
provides HF pipeline compatibility via duck typing, delegates all ORT
operations to `WinMLSession`, and exposes `forward()`, `to()`, `perf()`,
and standard HF properties. This diff replaces the loose `device: str`
constructor parameter with a required `ep_device: EPDevice` value object
and threads it through to `WinMLSession`.

## 2. Changes summary

- `__init__` signature: `device: str = "auto"` → `ep_device: EPDevice`
  (required, no default), position moved to slot 2 (before `config`).
- `self._device = device` → `self._ep_device = ep_device`.
- `WinMLSession(onnx_path=..., device=device)` → `WinMLSession(onnx_path=..., ep_device=ep_device)`.
- `TYPE_CHECKING` block: adds `from ...session.ep_device import EPDevice`.
- `perf()` docstring: `PerfStats` → `ctx.stats` — cosmetic correctness fix.
- `WinMLModelForGenericTask` (concrete subclass in same file) is unaffected.

## 3. Per-symbol review

### `WinMLPreTrainedModel.__init__`

- **Role:** Construct the base model, store references, and create the
  underlying `WinMLSession`.
- **Signature:** `def __init__(self, onnx_path: str | Path, ep_device: EPDevice, config: PretrainedConfig | None = None) -> None`
- **Behavior:** Stores `onnx_path` as `Path`, stores `ep_device`, sets
  `config`, clears `_build_config`, and immediately creates a
  `WinMLSession`. The session creation is eager — it happens at
  construction, not on first `forward()` call.
- **Invariants:**
  - `ep_device` is always required; there is no "auto" fallback at this
    layer. The caller (i.e., `WinMLAutoModel` or direct instantiation) must
    supply a fully resolved `EPDevice`.
  - `config` remains optional (keyword, defaults to `None`) — backwards-
    compatible for bare-ONNX builds.
  - Parameter order changed: old order was `(onnx_path, config=None,
    device="auto")`; new order is `(onnx_path, ep_device, config=None)`.
    Any caller passing `config` positionally (position 2) will now bind
    an `EPDevice`-typed config to `ep_device`, causing a hard runtime error
    from `WinMLSession`. Keyword-call style is safe.
- **Risks / concerns:**
  - **Positional order change** is the primary risk. The test helper
    `_make_mock_model()` in `test_automodel.py` bypasses `__init__`
    entirely (uses `__new__` + manual attribute assignment at line 32-48),
    so it is shielded but does not verify the new constructor contract.
  - `self._device` is gone. Any code outside this file that reads
    `model._device` directly (bypassing the `device` property which
    delegates to `self._session.device`) will get `AttributeError`. Grep
    confirms no such external access exists in the reviewed diff, but it
    remains a migration hazard for calling code not in this repo slice.
  - There is no `device` property override on `WinMLPreTrainedModel` that
    exposes `self._ep_device.device`. The `device` property at line 208
    delegates to `self._session.device` — which is correct, but means the
    session must be successfully created before `device` is readable. If
    session construction fails, `_ep_device` is set but `device` is
    inaccessible.
- **Tests:** `tests/unit/models/auto/test_automodel.py` (via `_make_mock_model`);
  `tests/unit/models/auto/test_auto_onnx.py` (via factory call).
  No test directly instantiates `WinMLPreTrainedModel.__init__` with the
  new signature and asserts `_ep_device` is stored; the mock-bypass pattern
  means the constructor path is not exercised in unit tests.

---

### `WinMLPreTrainedModel.perf`

- **Role:** Context manager for scoped performance tracking, delegating to `WinMLSession.perf()`.
- **Signature:** `def perf(self, warmup: int = 0) -> contextlib.AbstractContextManager`
- **Behavior:** Unchanged functionally. Docstring updated: `PerfStats` →
  `ctx.stats`, `with model.perf(warmup=5) as stats:` → `as ctx:`.
- **Invariants:** Unchanged.
- **Risks / concerns:** Docstring-only change; no behavioral impact.
  The type annotation `contextlib.AbstractContextManager` is a forward
  reference under `TYPE_CHECKING`, consistent with existing style.
- **Tests:** Covered transitively by perf CLI tests.

---

### `WinMLModelForGenericTask`

- **Role:** Concrete fallback subclass for unknown/unsupported tasks.
- **Changes:** None. The class inherits the new `__init__` signature
  automatically. Its `forward()` calls `_format_inputs` and
  `_run_inference`, both unchanged.
- **Risks / concerns:** None introduced by this diff.
- **Tests:** `tests/unit/models/auto/` suite exercises via `get_winml_class(None, None)`.

## 4. Cross-cutting

- The removal of `_device` is intentional but is a semver-breaking change
  for any consumer that accessed the private attribute directly. No public
  `ep` or `ep_device` property is added to expose the stored `_ep_device`
  to subclasses or external callers. Consider adding a read-only
  `ep_device` property for diagnostic access.
- The `device` property (line 208) still returns a `str` (via session). If
  HF pipeline compatibility ever requires comparing against `ep_device.device`,
  callers must go through `model._ep_device.device` (private) rather than
  `model.device` (which could differ if the session remaps the device).
- Task-specific subclasses (e.g., `WinMLModelForImageClassification` in
  separate files) call `super().__init__` — those were not in scope for
  this diff; verify they propagate `ep_device` and not `device=`.

## 5. Confidence level

High for the mechanical correctness of the change. Medium for test
coverage: the constructor is not directly tested with the new signature.

## 6. Verbatim risk inventory

| # | Location | Risk |
|---|----------|------|
| R1 | `base.py:64-68` | Parameter order change from `(onnx_path, config, device)` to `(onnx_path, ep_device, config)` — any positional caller of a subclass `__init__` that passed `config` in slot 2 will silently bind to `ep_device`. |
| R2 | `base.py:79` | `self._device` removed — external direct attribute access breaks with `AttributeError`; no public replacement property added. |
| R3 | `base.py:32-48` (test) | `_make_mock_model()` bypasses `__init__` entirely; the new constructor contract is not covered by any test that actually calls `__init__`. |
