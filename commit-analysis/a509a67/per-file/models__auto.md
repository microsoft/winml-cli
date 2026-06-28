# src/winml/modelkit/models/auto.py

## TL;DR

`WinMLAutoModel.from_onnx` and `from_pretrained` now require a resolved `EPDevice` instead of the loose `device: str = "auto"` / `ep: str | None = None` pair. Internally the factory unpacks `ep_device.device` and `short_ep_name(ep_device.ep)` to feed the lower-level config-generation and build functions that still speak the legacy (device-str, ep-short-name) wire format. The hand-off to the per-task `WinMLPreTrainedModel` constructor switched from `device=...` to `ep_device=...`. `from_pretrained`'s second positional argument signature was changed (hard break).

## Diff metrics

- Lines changed: +19 / -19 (38 total)
- Public methods touched: `from_onnx`, `from_pretrained` (both signatures and bodies)
- New imports: `from ..session import short_ep_name`; `EPDevice` under `TYPE_CHECKING`
- Removed parameters: `device: str = "auto"`, `ep: str | None = None` (both methods)
- Added parameters: `ep_device: EPDevice` (both methods)
- Removed example in module docstring? No — only signatures, the broken examples like `WinMLAutoModel.from_onnx("model.onnx", device="npu")` in the docstring are now **stale** (see Risks).

## Role before vs after

Unchanged at the conceptual level: `WinMLAutoModel` is still the factory orchestrating CONFIG → LOAD → BUILD → RUNTIME for both HF model IDs and bare ONNX files, returning a task-specific `WinMLPreTrainedModel` subclass.

What changed is the **input contract**: callers now pre-resolve (EP, device) into a single `EPDevice` descriptor and pass it in. The factory's internal use of `device`/`ep` strings to feed `generate_onnx_build_config`, `generate_hf_build_config`, `build_onnx_model`, and `build_hf_model` is preserved — those lower-level call sites still take `device: str` and `ep: str | None`. So `auto.py` is the **adapter layer** between the new `EPDevice` boundary protocol and the legacy string-based build/config layer.

The per-task model constructors (e.g. `WinMLModelForImageClassification`) get the full `ep_device` object — this is the only path where the descriptor travels intact, because that's what `WinMLSession.__init__` now requires.

## Symbol-level changes

### `from_onnx(cls, onnx_path, *, ep_device, task=None, config=None, precision="auto", cache_dir=None, use_cache=True, force_rebuild=False, skip_build=False, **kwargs)`

- Signature: removed `device: str = "auto"` and `ep: str | None = None`; inserted `ep_device: EPDevice` as the first keyword-only argument (after `*`).
- Docstring: removed `device` and `ep` arg descriptions; added `ep_device` arg description with construction hint pointing at `resolve_device(ep, device)`.
- Body call sites:
  - `generate_onnx_build_config(onnx_path, task=task, device=ep_device.device, precision=precision, ep=short_ep_name(ep_device.ep), override=config)` — was `device=device, ep=ep`.
  - Skip-build branch: `winml_class(onnx_path=onnx_path, config=None, ep_device=ep_device)` — was `device=device`.
  - `build_onnx_model(... ep=short_ep_name(ep_device.ep), device=ep_device.device, **kwargs)` — was `ep=ep, device=device`.
  - Final wrap: `winml_class(onnx_path=result.final_onnx_path, config=None, ep_device=ep_device)` — was `device=device`.

### `from_pretrained(cls, model_id_or_path, ep_device, *, task=None, config=None, precision="auto", cache_dir=None, use_cache=True, force_rebuild=False, trust_remote_code=False, shape_config=None, **kwargs)`

- Signature: removed `device: str = "auto"`; inserted `ep_device: EPDevice` as a **positional** parameter immediately after `model_id_or_path`, **before** the `*`. This is a deliberate API choice — clients are forced to write `from_pretrained(model_id, ep_device)`, not pass it by keyword.
- Note: the old signature had no `ep` kwarg at this level; `ep` used to leak through `**kwargs` (the diff shows `ep=kwargs.pop("ep", None)` being deleted from the ONNX delegate path).
- Body call sites:
  - ONNX delegate path: `cls.from_onnx(onnx_path=onnx_file, ep_device=ep_device, task=task, config=config, precision=precision, ...)` — `device=device` and `ep=kwargs.pop("ep", None)` removed.
  - `generate_hf_build_config(model_id, ..., device=ep_device.device, precision=precision, ep=short_ep_name(ep_device.ep))` — was `device=device, ep=kwargs.get("ep")`.
  - `build_hf_model(..., ep=resolved_ep, device=ep_device.device)` — was `device=device`. `resolved_ep` is still derived locally from `config.compile.ep_config.provider`, unchanged.
  - Final wrap: `winml_class(onnx_path=onnx_path, config=hf_config, ep_device=ep_device)` — was `device=device` with a comment about "pass user's original device string; WinMLSession handles 'auto'". That comment is gone — the new model is, `WinMLSession` does **not** handle "auto"; the caller already resolved it.

## Behavior / contract changes

1. **Hard-break API.** No `device=`/`ep=` compat shims on either classmethod. The commit body lists this as deliberate ("hard break Option A").
2. **`from_pretrained`'s second positional arg is now `ep_device`.** Any user code passing positional args (`WinMLAutoModel.from_pretrained("model-id", "npu")`) used to interpret the second arg as `task` (no — task was kw-only) — actually the old signature was `from_pretrained(cls, model_id_or_path, *, task=None, ..., device="auto", ...)`. So previously you could not pass *anything* positionally except `model_id_or_path`. The new API allows (and requires) `ep_device` positionally. Callers using kwargs are mostly unaffected; callers using only positional `model_id_or_path` now error with "missing 1 required positional argument: 'ep_device'".
3. **Two-direction string conversion** at boundaries: `EPDevice` carries the canonical EP (e.g. `"QNNExecutionProvider"`), but downstream `generate_onnx_build_config` / `build_onnx_model` / `generate_hf_build_config` still expect the short form (e.g. `"qnn"`). `short_ep_name(ep_device.ep)` is called four times to perform this conversion. This is technical debt — the build subsystem hasn't been migrated to `EPDevice`. The commit body doesn't mention this asymmetry.
4. **Lost stale comment about WinMLSession handling "auto".** The deleted comment confirms a real semantic shift: `WinMLSession` no longer accepts `"auto"` — the boundary is responsible for resolution.
5. **`resolved_ep` (HF path) is still computed from `config.compile.ep_config.provider`,** not directly from `ep_device.ep`. So if user-provided `config` has a different compile EP than the run-time `ep_device`, the build step uses the compile EP and the runtime uses `ep_device`. This is a latent bug surface that pre-dates the refactor.
6. **Stale docstring examples.** The class docstring still shows:
   ```python
   >>> model = WinMLAutoModel.from_onnx("model.onnx", device="npu")
   >>> model = WinMLAutoModel.from_pretrained("model.onnx", config=my_config)
   >>> model.to("npu")
   ```
   These no longer work — `device=` is gone, `ep_device=` is required, and `.to("npu")` is a no-op per `base.py`. The commit did not refresh these.

## Cross-file impact

- **Direct consumers.** Eval CLI (`eval/evaluate.py`) and any other `winml.modelkit` entrypoint that constructed models. The commit body lists `commands/perf.py`, `compiler/stages/compile.py` as also migrated.
- **Downstream contracts unchanged.** `generate_onnx_build_config`, `generate_hf_build_config`, `build_onnx_model`, `build_hf_model` all keep `device: str` and `ep: str | None` parameters — `auto.py` does the adaptation.
- **`WinMLPreTrainedModel` constructor** changed in lockstep (see `models__winml__base.md`).
- **`session.short_ep_name`** must remain stable; four call sites in this file depend on it for round-tripping canonical EP → short EP at boundaries. If it's renamed or its mapping for an EP changes (e.g. for new EPs not in `_SHORT_TO_FULL`), `auto.py` silently passes wrong values to the build layer.

## Risks / subtleties

1. **`short_ep_name` round-trip is silent on miss.** If an EP from the catalog isn't reverse-mapped in `_SHORT_TO_FULL`, behavior depends on its fallback — could yield `None` or the canonical name. The commit body's directive ("do not import private symbols...use the session/ facade and public helpers") suggests `short_ep_name` is the public helper, but its failure mode isn't documented here.
2. **Stale class docstring** (see above) will mislead users running `help(WinMLAutoModel)`. Probably worth a docstring fix in a follow-up.
3. **`**kwargs` shape changed in `from_pretrained`.** Previously `ep` could be smuggled through `**kwargs` and was popped in the ONNX delegate path; now any caller that did `from_pretrained(..., ep="qnn")` will silently get `ep` swallowed into `**kwargs` and dropped on the floor. There's no error.
4. **`ep_device.device` is a string.** The migration to a typed descriptor only flips the outermost API; underneath, everything is still string-keyed. If `EPDevice.device` is itself a string field (most likely), the type-safety win is modest — clients still pass a string-encoded device, just one that's been validated against the catalog.
5. **Positional-vs-keyword asymmetry across methods.** `from_pretrained` has `ep_device` positional; `from_onnx` has it keyword-only. The asymmetry is a footgun — IDE autocomplete + linter may flag one and miss the other.

## Open questions / TODOs surfaced

- Should the build layer (`generate_onnx_build_config`, `build_onnx_model`, etc.) be migrated to take `EPDevice` directly so the `short_ep_name(ep_device.ep)` plumbing here goes away?
- Why is `ep_device` positional on `from_pretrained` but keyword-only on `from_onnx`? Intentional or accident? (Looks intentional — kwargs are cleaner for `from_onnx` because of the long parameter list, while `from_pretrained` is the user-facing entry point and forcing positional draws attention to the new required argument.)
- Should `resolved_ep` (HF path) be replaced by `ep_device.ep` (or vice versa)? They can disagree.
- Stale docstring example in the class — needs cleanup but is not behavior-affecting.
