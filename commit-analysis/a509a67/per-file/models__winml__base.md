# src/winml/modelkit/models/winml/base.py

## TL;DR

`WinMLPreTrainedModel.__init__` switched from `device: str = "auto"` to a required `ep_device: EPDevice` argument and forwards it through to the underlying `WinMLSession`. The model now stores `self._ep_device` instead of `self._device`. The `perf()` context-manager docstring was also updated to reflect that the yielded value is a `PerfContext` (referenced as `ctx`), not a bare `PerfStats` (used to be `stats`).

## Diff metrics

- Lines changed: +9 / -6 (~17 line shift)
- Functions touched: `__init__`, `perf` (docstring only)
- New imports: `EPDevice` (TYPE_CHECKING only)
- New instance attribute: `self._ep_device`
- Removed instance attribute: `self._device`

## Role before vs after

Role is unchanged: `WinMLPreTrainedModel` is the abstract base that adapts a `WinMLSession` to HF's duck-typed `pipeline(...)` contract. It still owns:

- The ONNX path
- An optional HF `PretrainedConfig`
- A delegated `WinMLSession` for ORT operations
- `forward`/`__call__`/`to`/`device`/`dtype`/`perf` HF surface

Before, the constructor accepted a free-form `device: str` (e.g. `"auto"`/`"npu"`/`"cpu"`) and forwarded it to `WinMLSession(device=device)`, which then ran `_find_ep_device(ep_name)` internally. After, the constructor accepts a fully-resolved `EPDevice` and forwards it as `WinMLSession(ep_device=ep_device)`. Resolution is now done **upstream** at the CLI boundary, not inside the model.

## Symbol-level changes

### `__init__`

- Signature parameter order changed:
  - **Before:** `(self, onnx_path, config=None, device="auto")`
  - **After:**  `(self, onnx_path, ep_device, config=None)`
- `ep_device` is **positional** (no `*` separator). Callers must pass it positionally or as a keyword.
- Removed `device` parameter entirely (no compat shim).
- Removed default value for the relocated parameter — `ep_device` has no default and is required.
- Body changes:
  - `self._device = device` removed.
  - `self._ep_device = ep_device` added.
  - `WinMLSession(onnx_path=self._onnx_path, device=device)` → `WinMLSession(onnx_path=self._onnx_path, ep_device=ep_device)`.
- Docstring updated: removed `device` arg description, added `ep_device` arg description pointing at `resolve_device(ep, device)`.

### TYPE_CHECKING imports

- Added `from ...session import EPDevice` to the second TYPE_CHECKING block (the one that also imports `PretrainedConfig`).

### `perf(self, warmup=0)` — docstring only

- Reworded "records timing in PerfStats" → "records timing in ``ctx.stats``".
- Example renamed from `with model.perf(warmup=5) as stats:` to `with model.perf(warmup=5) as ctx:` and `stats.p99_ms` → `ctx.stats.p99_ms`. Reflects the op-tracing refactor where `WinMLSession.perf()` now yields a `PerfContext` (with `.stats` + `.monitor`) instead of a bare `PerfStats`.
- No code change to `perf` body — still `return self._session.perf(warmup=warmup)`.

### Unchanged but worth noting

- `device` property still returns `self._session.device` (NOT `self._ep_device.device`). This means after construction, the `device` property reflects whatever the session decided post-resolve, not the descriptor passed in. In practice they should agree, but it's a subtle re-derivation path.
- `to(...)` is still a no-op with the same FIXME comment.
- `_ep_device` is currently **write-only** — nothing in this file or its subclasses reads it back. The session is the source of truth at run-time.

## Behavior / contract changes

1. **Construction contract is now strict.** Every subclass instantiation (e.g. `WinMLModelForImageClassification(onnx_path, ep_device)`) must supply `ep_device`. The factory (`WinMLAutoModel`) was updated in lockstep — see `models/auto.py` analysis.
2. **No "auto" semantics in the model.** Previously, the device string "auto" propagated to `WinMLSession` and then to `_find_ep_device`. Now `auto`-style deduction must happen **before** `WinMLPreTrainedModel.__init__` is called, in `resolve_device(...)`. If a user constructs the model directly with an `EPDevice`, it has already been resolved.
3. **`config` is now after `ep_device` in positional order.** Anyone constructing subclasses with positional args (`WinMLModelForXxx("model.onnx", my_hf_config)`) breaks — `my_hf_config` would be misinterpreted as `ep_device`. The factory always uses keyword args, so internal callers are fine; external user-code that bypasses the factory would break with a non-obvious TypeError.
4. **`perf` context-manager semantics changed.** The yielded value is no longer the stats object directly — clients must now do `ctx.stats` instead of using the yielded value as stats. This is the op-tracing refactor (PerfContext = stats + monitor). Although `perf()`'s code in this file is unchanged, the **yielded type** changed because `WinMLSession.perf()` changed — and the docstring is the only signal here.

## Cross-file impact

- **Subclasses** under `src/winml/modelkit/models/winml/` (e.g. classification, detection, segmentation, QA, feature-extraction wrappers) inherit `__init__` and get the new contract for free. They are unchanged unless they override `__init__`.
- **`WinMLAutoModel`** (in `auto.py`) is the primary caller — confirmed updated to pass `ep_device=ep_device` at every `winml_class(...)` call site.
- **`WinMLSession.__init__`** (in `session/session.py`) must accept `ep_device=` kwarg — this is the new mandatory positional/keyword per commit body. Failure to pass it would be a TypeError at session construction.
- **Tests** that directly instantiate a `WinMLPreTrainedModel` subclass need to supply `ep_device` — likely there are several test-side callers (the commit reports ~720 passing, so they were updated).

## Risks / subtleties

1. **`self._ep_device` is dead state.** Stored but never read in this file. Subclasses may consume it, or it may exist for introspection/debugging — but a year from now it risks bit-rot.
2. **`device` property re-derives from session.** If `WinMLSession.device` ever diverges from `EPDevice.device` (e.g. session falls back from NPU to CPU at compile time), `model.device` reports the **effective** device, while `model._ep_device.device` is the **requested** device. This divergence is unflagged.
3. **Positional ordering of `ep_device` before `config`.** Existing test code or notebooks that did `WinMLModelForX("path", hf_cfg)` silently break to a TypeError in the best case, or a confusing type-mismatch deeper down in the worst case.
4. **`perf()` body unchanged but contract changed.** A consumer reading just the diff would see the docstring update and notice nothing else; they might miss that downstream `WinMLSession.perf()` now yields a different object type. The docstring example is the only signal — anyone with old code using `as stats` plus `stats.p99_ms` is now broken because the yielded object has no `.p99_ms` (only `.stats.p99_ms`).
5. **No validation of `ep_device`** at construction. The model trusts the caller. If `None` is passed (e.g. `ep_device=None`), the TypeError surfaces only inside `WinMLSession`, not here.

## Open questions / TODOs surfaced

- Should `self._ep_device` be exposed via a public `ep_device` property for symmetry with `device`/`task`/`precision`?
- Should the `device` property prefer `self._ep_device.device` (requested) over `self._session.device` (effective)? Or should there be two properties (`requested_device`, `effective_device`)?
- Is the `perf()` docstring change documented anywhere else as a breaking change to users? The commit body mentions the PerfContext yield but doesn't enumerate this as a public-API break.
- The `to(...)` FIXME is still present and unchanged — should it also be aware of `_ep_device` (e.g. reject `.to("cpu")` if `ep_device.device != "cpu"`)?
