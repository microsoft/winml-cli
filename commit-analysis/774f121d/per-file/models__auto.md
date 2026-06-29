# src/winml/modelkit/models/auto.py

## TL;DR

`WinMLAutoModel.from_onnx` and `from_pretrained` migrated from the loose `device: str = "auto"` / `ep: str | None = None` pair to a single resolved `ep_device: WinMLEPDevice` argument. Inside both classmethods, the descriptor is unpacked into the legacy build-layer dialect with `ep_device.device.device_type.lower()` for the device string and `short_ep_name(ep_device.device.ep_name)` for the EP short name. The hand-off to per-task `WinMLPreTrainedModel` subclasses passes the full `ep_device` through. `session_options=` was dropped from both call sites — it was previously a smuggled session-options pass-through that the session layer no longer accepts at this level.

**This file contains the auto.py:411 `.lower()` bug fix called out in the commit body.** Confirmed landed (see below).

## Diff metrics

- Lines changed: +20 / -21 (~41 total)
- Public methods touched: `from_onnx`, `from_pretrained` (signatures + bodies)
- New imports: `from ..session import short_ep_name`; `WinMLEPDevice` under `TYPE_CHECKING`
- Removed parameters: `device: str = "auto"`, `ep: str | None = None` (both methods); `session_options: Any | None` smuggled via `**kwargs`-popped at the wrapper-construction call sites
- Added parameters: `ep_device: WinMLEPDevice` (both methods)
- Removed inline comment: `# pass user's original device string; WinMLSession handles "auto"` (deleted because the assertion is no longer true)

## Role before vs after

Unchanged at the conceptual level: `WinMLAutoModel` is still the factory orchestrating CONFIG → LOAD → BUILD → RUNTIME for both HF model IDs and bare ONNX files, returning a task-specific `WinMLPreTrainedModel` subclass.

What changed: the **input contract** is now a resolved `WinMLEPDevice`. The factory internally adapts back to the legacy `(device: str, ep: str | None)` form for the build/config layer (`generate_onnx_build_config`, `generate_hf_build_config`, `build_onnx_model`, `build_hf_model`) — those entry points still accept strings. `auto.py` therefore acts as the **adapter** between the new typed-descriptor boundary and the legacy string-based build pipeline.

The per-task model constructors (e.g. `WinMLModelForImageClassification`) get the full `ep_device` object — that's the only path where the descriptor travels intact, because `WinMLPreTrainedModel.__init__` (see `models__winml__base.md`) now requires it.

## Symbol-level changes

### `from_onnx(cls, onnx_path, *, ep_device, task=None, config=None, precision="auto", cache_dir=None, use_cache=True, force_rebuild=False, skip_build=False, **kwargs)`

- Signature: removed `device: str = "auto"`, removed `ep: str | None = None`; inserted `ep_device: WinMLEPDevice` as the first keyword-only argument (immediately after `*`).
- Docstring: removed `device` and `ep` arg descriptions; added `ep_device` arg description pointing at `resolve_device(EPDeviceTarget(...))` from `session.ep_device`.
- Body call sites:
  - `generate_onnx_build_config(onnx_path, task=task, device=ep_device.device.device_type.lower(), precision=precision, ep=short_ep_name(ep_device.device.ep_name), override=config)` — was `device=device, ep=ep`.
  - Skip-build branch: `winml_class(onnx_path=onnx_path, config=None, ep_device=ep_device)` — was `device=device, session_options=session_options`. The `session_options=...` argument is **dropped entirely**.
  - `build_onnx_model(..., ep=short_ep_name(ep_device.device.ep_name), device=ep_device.device.device_type.lower(), **kwargs)`.
  - Final wrap: `winml_class(onnx_path=result.final_onnx_path, config=None, ep_device=ep_device)` — `session_options=...` also dropped here.

### `from_pretrained(cls, model_id_or_path, ep_device, *, task=None, config=None, precision="auto", cache_dir=None, use_cache=True, force_rebuild=False, trust_remote_code=False, shape_config=None, **kwargs)`

- Signature: removed `device: str = "auto"`; inserted `ep_device: WinMLEPDevice` as a **positional** parameter immediately after `model_id_or_path`, **before** the `*`. Hard break — clients must now supply `ep_device` positionally or by keyword.
- Body call sites:
  - ONNX delegate path: `cls.from_onnx(onnx_path=onnx_file, ep_device=ep_device, task=task, ...)` — `device=device` and `ep=kwargs.pop("ep", None)` removed.
  - `generate_hf_build_config(model_id, ..., device=ep_device.device.device_type.lower(), precision=precision, ep=short_ep_name(ep_device.device.ep_name))` — was `device=device, ep=kwargs.get("ep")`.
  - `build_hf_model(..., ep=resolved_ep, device=ep_device.device.device_type.lower())` — **this is line 411 in the post-fix file.**
  - Final wrap: `winml_class(onnx_path=onnx_path, config=hf_config, ep_device=ep_device)` — `device=device` and the trailing `# pass user's original device string; WinMLSession handles "auto"` comment are both removed.

### `.lower()` bug fix verification (line 411)

The commit body calls out: *"auto.py:411 passes device_type.lower() to match the other 3 call sites"*.

Confirmed at HEAD (verified by `Read` on the post-commit file):

```python
410        ep=resolved_ep,
411        device=ep_device.device.device_type.lower(),
412    )
```

This call site (`build_hf_model(...)`) now matches the other three call sites in the file:

1. `generate_onnx_build_config(..., device=ep_device.device.device_type.lower(), ...)` (line 173)
2. `build_onnx_model(..., device=ep_device.device.device_type.lower(), ...)` (line 218)
3. `generate_hf_build_config(..., device=ep_device.device.device_type.lower(), ...)` (line ~356)
4. `build_hf_model(..., device=ep_device.device.device_type.lower(), ...)` (line 411 — **the fix**)

Before the fix, this fourth call site likely passed `ep_device.device.device_type` (un-lowered) or `device` (the old string), producing a case mismatch with the other three sites. The fix is a one-token addition (`.lower()`) but it eliminates a real footgun where the build pipeline got `"GPU"` from one entry point and `"gpu"` from another.

## Behavior / contract changes

1. **Hard-break API.** No `device=`/`ep=` compat shims. The commit body lists this as deliberate.
2. **`from_pretrained`'s second positional arg is now `ep_device`.** Callers using `from_pretrained(model_id)` get `TypeError: missing 1 required positional argument: 'ep_device'`. Callers using kwargs are mostly safe — but kwargs callers passing `device=` or `ep=` now hit `**kwargs` and silently drop on the floor inside the function body. No error.
3. **`session_options=` parameter dropped at `winml_class(...)` construction.** Previously the factory smuggled `session_options=` via `kwargs.pop("session_options", None)` (implicit) and passed it to the model constructor, which forwarded to `WinMLSession`. The new `WinMLSession(ep_device=...)` does not take `session_options=` at this level — internal session-option construction is now via `_build_session_options` (module-level free function per the commit body).
4. **Two-direction string conversion** at boundaries: `WinMLEPDevice` carries the canonical EP (e.g. `"QNNExecutionProvider"`), but downstream `generate_onnx_build_config` / `build_onnx_model` / `generate_hf_build_config` / `build_hf_model` still expect the short form (`"qnn"`). `short_ep_name(ep_device.device.ep_name)` is called four times. This is technical debt — the build subsystem hasn't been migrated to `WinMLEPDevice`.
5. **Stale comment removed**, semantics confirmed: `WinMLSession` no longer accepts `"auto"`. The boundary is now the caller's responsibility (CLI/`resolve_device(EPDeviceTarget(...))`).
6. **`resolved_ep` (HF path) is still computed from `config.compile.ep_config.provider`,** not directly from `ep_device.device.ep_name`. So if user-provided `config` has a different compile EP than the run-time `ep_device`, the build step uses the compile EP and the runtime uses `ep_device`. Latent inconsistency surface that pre-dates the refactor.

## Cross-file impact

- **Direct consumers.** Anywhere that called `WinMLAutoModel.from_onnx(..., device=..., ep=...)` or `from_pretrained(..., device=...)` — per commit body: `commands/perf.py`, `compile.py`, `eval/evaluate.py`, and tests. The commit reports ~720 passing, so these are all migrated.
- **Downstream contracts unchanged.** `generate_onnx_build_config`, `generate_hf_build_config`, `build_onnx_model`, `build_hf_model` all keep `device: str` and `ep: str | None` parameters — `auto.py` does the adaptation.
- **`WinMLPreTrainedModel` constructor** changed in lockstep (see `models__winml__base.md`).
- **`session.short_ep_name`** must remain stable; four call sites in this file depend on it for round-tripping. Confirmed exported in `session/__init__.py`.
- **Note:** `ep_device.device.ep_name` is the access path (not `ep_device.ep` as in the earlier a509a67 commit). This suggests a structural change in `WinMLEPDevice`: the EP name now lives on `device`, not at the top level. Worth verifying `WinMLEPDevice` shape.

## Risks / subtleties

1. **`short_ep_name` round-trip is silent on miss.** If an EP from the catalog isn't reverse-mapped, behavior depends on the helper's fallback — could yield `None` or the canonical name. Failure mode not documented here.
2. **Stale class docstring examples.** The `>>> WinMLAutoModel.from_onnx("model.onnx", device="npu")` and `>>> model.to("npu")` examples are likely still present in the class docstring (the diff didn't touch the class docstring) and now broken. Worth a docstring sweep.
3. **`**kwargs` smuggling lost.** Previously `from_pretrained(..., ep="qnn")` worked via `kwargs.pop("ep", None)`. Now `ep=` falls into `**kwargs` and is silently dropped. No warning. Anyone with an old script still passing `ep=` sees their EP choice silently ignored.
4. **Positional-vs-keyword asymmetry across methods.** `from_pretrained` has `ep_device` positional; `from_onnx` has it keyword-only. The asymmetry is a footgun — IDE autocomplete + linter may flag one and miss the other. Could be intentional (force attention on the new arg) but is undocumented.
5. **`ep_device.device.ep_name` access path is fragile.** If `WinMLEPDevice`'s internal shape changes (e.g. EP name moves back to top level), all four `short_ep_name(ep_device.device.ep_name)` call sites silently break. A property like `ep_device.short_ep` would isolate the access.

## Open questions / TODOs surfaced

- Should the build layer (`generate_onnx_build_config`, `build_onnx_model`, etc.) be migrated to take `WinMLEPDevice` directly so the four `short_ep_name(ep_device.device.ep_name)` adapters here go away?
- Why is `ep_device` positional on `from_pretrained` but keyword-only on `from_onnx`? Intentional or accident?
- Should `resolved_ep` (HF path) be replaced by `ep_device.device.ep_name` (or vice versa)? They can disagree if user-provided `config.compile.ep_config.provider` differs from the runtime `ep_device`.
- Stale class docstring examples — needs a follow-up sweep.
- `session_options=` is gone from this file — is any caller still trying to pass it (e.g. a notebook with custom graph_optimization_level)? If so, they need a new escape hatch.

## Simplification opportunities

- **The four `ep_device.device.device_type.lower()` and four `short_ep_name(ep_device.device.ep_name)` calls are pure boilerplate.** Two short properties on `WinMLEPDevice` — `device_str` (lowercase device_type) and `short_ep` — would collapse 8 call sites to 8 attribute reads and prevent the kind of casing-drift bug the commit just fixed at line 411.
- **The latent `resolved_ep` vs `ep_device.device.ep_name` disagreement** could be made an assertion at the top of the HF path body: `assert resolved_ep is None or resolved_ep == ep_device.device.ep_name`. This would surface the latent bug instead of silently picking one.
- **Stale class docstring examples** should be updated in the same sweep.
- **The asymmetry between `from_onnx` (kw-only `ep_device`) and `from_pretrained` (positional `ep_device`)** should be reconciled. Pick one ergonomic and apply it to both.
- **`from_pretrained` and `from_onnx` share substantial structure** (cache-dir resolution, output-dir derivation, winml_class lookup, final wrapper construction). Extracting a private `_construct_wrapper(winml_class, onnx_path, hf_config, ep_device, build_config)` helper would reduce duplication.
