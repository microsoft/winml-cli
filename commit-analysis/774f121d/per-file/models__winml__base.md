# src/winml/modelkit/models/winml/base.py

## TL;DR

`WinMLPreTrainedModel.__init__` swapped `device: str = "auto"` and `session_options: Any | None = None` for a single required `ep_device: WinMLEPDevice`. The model now stores `self._ep_device` instead of `self._device`. The `perf()` context-manager docstring was also updated to reflect that the yielded value is a `PerfContext` (referenced as `ctx`) with a `.stats` attribute, not a bare `PerfStats` (used to be `stats`).

## Diff metrics

- Lines changed: +10 / -10 (~20 total)
- Methods touched: `__init__` (signature + body + docstring), `perf` (docstring only)
- New imports: `WinMLEPDevice` under TYPE_CHECKING (lazy)
- New instance attribute: `self._ep_device`
- Removed instance attribute: `self._device`
- Removed parameter: `session_options: Any | None = None` — alongside `device`

## Role before vs after

Role is unchanged: `WinMLPreTrainedModel` is the abstract base that adapts a `WinMLSession` to HF's duck-typed `pipeline(...)` contract. It still owns:

- The ONNX path
- An optional HF `PretrainedConfig`
- A delegated `WinMLSession` for ORT operations
- `forward`/`__call__`/`to`/`device`/`dtype`/`perf` HF surface

Before, the constructor accepted a free-form `device: str` (e.g. `"auto"`/`"npu"`/`"cpu"`) and an optional `session_options` blob, forwarding both to `WinMLSession`. After, the constructor accepts a fully-resolved `WinMLEPDevice` and forwards it as `WinMLSession(ep_device=ep_device)`. Resolution and session-options construction now happen **upstream**.

## Symbol-level changes

### `__init__`

- Signature parameter order changed:
  - **Before:** `(self, onnx_path, config=None, device="auto", session_options=None)`
  - **After:**  `(self, onnx_path, ep_device, config=None)`
- `ep_device` is **positional** (before the `*`-less divider — there is none here). Callers must pass it positionally or as a keyword.
- Removed `device` and `session_options` parameters entirely (no compat shims).
- `ep_device` has no default; required.
- Body changes:
  - `self._device = device` removed.
  - `self._ep_device = ep_device` added.
  - `WinMLSession(onnx_path=self._onnx_path, device=device, session_options=session_options)` → `WinMLSession(onnx_path=self._onnx_path, ep_device=ep_device)`.
- Docstring updated: removed `device` and `session_options` arg descriptions, added `ep_device` arg description pointing at `resolve_device(EPDeviceTarget(...))` from `session.ep_device`.

### TYPE_CHECKING imports

- Added `from ...session import WinMLEPDevice` to the second TYPE_CHECKING block (the one that also imports `PretrainedConfig`). Kept lazy — purely a type hint.

### `perf(self, warmup=0)` — docstring only

- Reworded "records timing in PerfStats" → "records timing in ``ctx.stats``".
- Example renamed from `with model.perf(warmup=5) as stats:` to `with model.perf(warmup=5) as ctx:` and `stats.p99_ms` → `ctx.stats.p99_ms`. Reflects the op-tracing refactor where `WinMLSession.perf()` now yields a `PerfContext` (with `.stats` + `.monitor`) instead of a bare `PerfStats`.
- No code change to `perf` body — still `return self._session.perf(warmup=warmup)`.

### Unchanged but worth noting

- `device` property still returns `self._session.device` (NOT `self._ep_device.device`). After construction, `model.device` reflects whatever the session decided post-resolve, not the descriptor passed in. In practice they should agree.
- `to(...)` is still a no-op with the same FIXME comment.
- `_ep_device` is currently **write-only** — nothing in this file reads it back. The session is the source of truth at runtime.

## Behavior / contract changes

1. **Construction contract is now strict.** Every subclass instantiation (e.g. `WinMLModelForImageClassification(onnx_path, ep_device)`) must supply `ep_device`. The factory (`WinMLAutoModel`) was updated in lockstep.
2. **No "auto" semantics in the model.** Previously, the device string "auto" propagated to `WinMLSession` and then to `_find_ep_device`. Now `auto`-style deduction must happen **before** `WinMLPreTrainedModel.__init__` is called, in `resolve_device(...)`.
3. **`session_options=` escape hatch is gone.** Callers who used to thread a custom `ort.SessionOptions` through the model constructor (e.g. for `graph_optimization_level`, `intra_op_num_threads`, custom config entries) have no equivalent here. The commit body mentions `_build_session_options` is now a module-level free function inside `session/session.py` — but it's a private helper, not the public API. Public escape hatch for custom session options is unclear.
4. **`config` is now after `ep_device` in positional order.** Anyone constructing subclasses with positional args (`WinMLModelForXxx("model.onnx", my_hf_config)`) breaks — `my_hf_config` would be misinterpreted as `ep_device`. The factory always uses keyword args, so internal callers are fine; external user-code that bypasses the factory would break with a non-obvious TypeError later when `WinMLSession` tried to consume it.
5. **`perf` context-manager semantics changed.** The yielded value is no longer the stats object directly — clients must now do `ctx.stats` instead of using the yielded value as stats. This is the op-tracing refactor (`PerfContext` = stats + monitor). Although `perf()`'s code in this file is unchanged, the **yielded type** changed because `WinMLSession.perf()` changed — and the docstring is the only signal here.

## Cross-file impact

- **Subclasses** under `src/winml/modelkit/models/winml/` (e.g. classification, detection, segmentation, QA, feature-extraction wrappers) inherit `__init__` and get the new contract for free. They are unchanged unless they override `__init__`.
- **`WinMLAutoModel`** (in `auto.py`) is the primary caller — confirmed updated to pass `ep_device=ep_device` at every `winml_class(...)` call site.
- **`WinMLSession.__init__`** (in `session/session.py`) must accept `ep_device=` kwarg — this is the new mandatory keyword per commit body. Failure to pass it would be a TypeError at session construction.
- **Tests** that directly instantiate a `WinMLPreTrainedModel` subclass need to supply `ep_device`.

## Risks / subtleties

1. **`self._ep_device` is dead state.** Stored but never read in this file. Subclasses may consume it, or it may exist for introspection/debugging — but a year from now it risks bit-rot. A property exposing it would at least make the intent explicit.
2. **`device` property re-derives from session.** If `WinMLSession.device` ever diverges from `WinMLEPDevice`'s device (e.g. session falls back from NPU to CPU at compile time), `model.device` reports the **effective** device, while `model._ep_device` carries the **requested** device. Divergence is unflagged.
3. **Positional ordering of `ep_device` before `config`.** Existing test code or notebooks that did `WinMLModelForX("path", hf_cfg)` silently break to a TypeError in the best case, a confusing deeper type-mismatch in the worst.
4. **`perf()` body unchanged but contract changed.** A consumer reading just the diff would see the docstring update and miss that downstream `WinMLSession.perf()` now yields a different object type. Anyone with old code using `as stats` plus `stats.p99_ms` is now broken because the yielded object has no `.p99_ms` (only `.stats.p99_ms`).
5. **`session_options=` removed without a replacement.** Power users who customized session options now need an internal path that isn't documented. Worth a follow-up to expose `WinMLSession`'s `_build_session_options` overrides via the model constructor or a factory hook.
6. **No validation of `ep_device`** at construction. The model trusts the caller. If `None` is passed, the TypeError surfaces only inside `WinMLSession`.

## Open questions / TODOs surfaced

- Should `self._ep_device` be exposed via a public `ep_device` property for symmetry with `device`/`task`/`precision`?
- Should the `device` property prefer `self._ep_device.device` (requested) over `self._session.device` (effective)? Or should there be two properties (`requested_device`, `effective_device`)?
- Was the `session_options=` removal intentional? If yes, what's the replacement path for users who need custom `graph_optimization_level` etc.?
- The `to(...)` FIXME is still present and unchanged — should it also be aware of `_ep_device` (e.g. reject `.to("cpu")` if `ep_device.device != "cpu"`)?

## Simplification opportunities

- **Expose `_ep_device` publicly.** Either via property or by renaming to `self.ep_device`. The current write-only state is dead code-smell.
- **The `device` property** is now a one-line forward to the session — fine as-is, but worth noting: the model has both `_ep_device` (requested) and access to `self._session.device` (effective), without a property naming the distinction.
- **`perf()` body is a one-liner forward to `self._session.perf(...)`.** If subclasses don't override it (they likely don't), this is candidate boilerplate. A class-level mixin or direct delegation pattern would remove the wrapper.
- **The two `if TYPE_CHECKING:` blocks at the top of the file** import 1+ symbols each. Consolidating them into a single `if TYPE_CHECKING:` block reduces noise — currently the second block is essentially needless duplication of the import-guard.
