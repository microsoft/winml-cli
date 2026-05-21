# src/winml/modelkit/compiler/stages/compile.py

## TL;DR
`CompileStage` is migrated to the new `EPDevice` plumbing. (1) The `process()` method now resolves an `EPDevice` from a 3-way priority chain (context dict → config field → `resolve_device(ep=...)`) and constructs `WinMLSession(... ep_device=ep_device ...)` instead of the legacy `device=context.execution_provider` kwarg. (2) `_finalize_output` is taught the new naming protocol — `WinMLSession.compile()` now writes `{stem}_{ep_device.device}_ctx.onnx` (e.g. `_npu_ctx.onnx`), not `{stem}_{provider}_ctx.onnx` (e.g. `_qnn_ctx.onnx`), so the file-search list is widened with an additional pattern derived from `ep_to_device(execution_provider)`. (3) The import of the private `_EP_TO_DEVICE` map (cross-package, previously from `config/precision.py`) is gone — `_finalize_output` now calls the public `ep_to_device` from `..session` instead.

## Diff metrics
- ~22 LOC net added across two methods (`process`, `_finalize_output`).
- Module-level import switched: `from ...session import WinMLQairtSession, WinMLSession` → `from ...session import EPDevice, WinMLQairtSession, WinMLSession, resolve_device`. (Two new public symbols pulled in.)
- One new in-function lazy import added: `from ...session import ep_to_device` inside `_finalize_output`.

## Role before vs after
- **Before:** Built a `WinMLCompileConfig` from `context.config` solely to extract `ep_config`, then passed `device=context.execution_provider` (a short provider string) to `WinMLSession`. `_finalize_output` searched for compiled-context artefacts using only the provider short name in the filename.
- **After:** Same overall structure, but the session is bound to a concrete `(EP, device)` pair via `ep_device`, and the artefact lookup understands that `WinMLSession.compile()` names its outputs by the *device* (e.g. `npu`), not the EP short name (e.g. `qnn`). Avoidance of `_EP_TO_DEVICE` cross-package import was the architectural prize.

## Symbol-level changes
- **`process()`:**
  - Now binds `compile_cfg = WinMLCompileConfig.from_dict(context.config)` (previously was a throwaway temporary) so `compile_cfg.ep_device` is reachable.
  - New 3-way resolver:
    ```
    ep_device_dict = context.config.get("ep_device")
    if ep_device_dict:
        ep_device = EPDevice.from_dict(ep_device_dict)
    elif compile_cfg.ep_device is not None:
        ep_device = compile_cfg.ep_device
    else:
        ep_device = resolve_device(ep=context.execution_provider)
    ```
    The dict branch is for callers that hand a raw dict (e.g. round-tripped JSON or `WinMLBuildConfig.to_dict`); the field branch is for in-memory configs built via `WinMLCompileConfig.for_ep_device`; the fallback covers older API callers that only know an EP string.
  - Log line restructured: `f"Creating {session_cls.__name__} for {ep_device.ep}/{ep_device.device}"` (was `f"... for device: {context.execution_provider}"`).
  - Session construction: `session_cls(onnx_path=model_path, ep_device=ep_device, ep_config=ep_config)`. The previous `device=context.execution_provider` kwarg is gone — this is one of the "hard break" sites the commit body calls out (`WinMLSession.__init__` requires `ep_device` positional; `device=` kwarg removed).
- **`_finalize_output()`:**
  - Same `device = context.execution_provider.lower()` retained for the *output* filename (so the final emitted artefact keeps the EP-short-name convention, e.g. `..._qnn_ctx.onnx`).
  - Added in-function `from ...session import ep_to_device`.
  - Computes `ep_device_str = ep_to_device(context.execution_provider)` (guarded by try/except `ValueError`; on failure `ep_device_str = None`).
  - When `ep_device_str` is set (e.g. `"npu"`), it is *prepended* (`insert(0, ...)`) onto `ctx_patterns` so the new naming wins the search-order tie. The two legacy patterns (`_{device}_ctx.onnx` and `_ctx.onnx`) are preserved as fallbacks.
  - The rest of `_finalize_output` (copy to output dir, rename .bin, rewrite `ep_cache_context`) is unchanged.

## Behavior / contract changes
- **The `_finalize_output` naming protocol is now asymmetric:**
  - **Input-search list (work_dir):** prefers `{stem}_{device_category}_ctx.onnx` (e.g. `..._npu_ctx.onnx`), then falls back to `{stem}_{provider_short}_ctx.onnx` (e.g. `..._qnn_ctx.onnx`), then `{stem}_ctx.onnx`.
  - **Output filename (output_dir):** still `{original_stem}_{provider_short}_ctx.onnx` (e.g. `resnet50_qnn_ctx.onnx`). Same for the renamed `.bin` sidecar.
  - Implication: a model compiled today on NPU will land as `*_qnn_ctx.onnx` in the final output, matching the verification line `winml compile --ep qnn --device npu (fp32) → *_qnn_ctx.onnx + .bin` in the commit body.
- **`WinMLSession` is now called with the new positional `ep_device` kwarg.** Old code that monkey-patched or subclassed `WinMLSession.__init__` will break (intentional — Option A hard break).
- **`ep_to_device` accepting a full provider name (e.g. `"QNNExecutionProvider"`) is assumed.** The `try/except ValueError` is the defensive branch for the case where `context.execution_provider` is a short string (e.g. `"qnn"`) that `ep_to_device` may not accept, or some other unknown EP slips through. On failure, the lookup degrades silently to only the legacy patterns.
- **No more cross-package access to `_EP_TO_DEVICE`.** This file used to (or its sibling did) import the private map directly; that violation is removed. The commit-body directive ("do not import private symbols ... outside session/ep_device.py") is honoured.

## Cross-file impact
- Reads from `WinMLCompileConfig.ep_device` (added in the sibling `configs.py` change).
- Reads from `context.config["ep_device"]` (a dict shape produced by `WinMLCompileConfig.to_dict` / `EPDevice.to_dict`).
- Calls into the session package's public helpers: `EPDevice.from_dict`, `resolve_device(ep=...)`, `ep_to_device(full_provider_name)`. All four are now part of `winml.modelkit.session.__init__.__all__`.
- The downstream `WinMLSession.compile()` (modified elsewhere in this squash) must honour the new ctx-filename protocol (`{stem}_{ep_device.device}_ctx.onnx`) for the `ctx_patterns.insert(0, ...)` branch to ever match.

## Risks / subtleties
- **Three-way precedence has a subtle priority inversion.** `context.config.get("ep_device")` (a dict) wins over `compile_cfg.ep_device` (the rehydrated object), even though the latter is derived from the former through `WinMLCompileConfig.from_dict`. They should be equivalent in practice; if they ever disagree (e.g. someone mutates the dict after rehydrating), the dict wins. Probably fine but worth documenting.
- **`context.execution_provider` is still consulted in two places** — once as the fallback EP for `resolve_device(ep=...)`, and once as the `device` variable used to compose output filenames. If a user threads an `ep_device` whose EP disagrees with `context.config["execution_provider"]`, the *input* search uses `ep_device`-derived patterns but the *output* still uses `context.execution_provider`-derived names. Hand-built configs could drift.
- **`ep_to_device` is called with `context.execution_provider` (a short name like `"qnn"` per `CompileContext.execution_provider`'s default).** Whether `ep_to_device` accepts short names is a contract question for `session/ep_device.py`. If it accepts only full canonical names, the try/except will swallow a `ValueError` and silently disable the new pattern — meaning the new naming protocol only kicks in when callers pass full names. (Worth verifying against `ep_device.py`.)
- **The `_finalize_output` fallback chain has overlapping cases.** If both `_{device_short}_ctx.onnx` (where `device_short = "qnn"`) and `_{device_category}_ctx.onnx` (`"npu"`) happen to exist (e.g. left over from a previous CLI), the `insert(0, ...)` ordering means `npu` wins — which is correct for the current `WinMLSession.compile()` but assumes the session never falls back to writing the older name.
- **Session class still resolved via `COMPILER_SESSION_MAPPING[compiler]`.** `WinMLQairtSession` accepts the same `ep_device=` kwarg signature — assumed but not verified here.

## Open questions / TODOs surfaced
- Should `_finalize_output` also reuse the resolved `ep_device.device` directly (from `process`) instead of re-deriving via `ep_to_device(context.execution_provider)`? Today `_finalize_output` re-runs the inference. Threading the `EPDevice` onto `context` would eliminate the second lookup and the try/except.
- Should the *output* filename adopt the device-category convention (`*_npu_ctx.onnx`) for consistency, or stay on the provider-short convention (`*_qnn_ctx.onnx`)? Today the file is named after the provider, which mismatches the work_dir convention.
- Validate that `ep_to_device` actually accepts the value of `context.execution_provider` — otherwise the try/except is masking the new code path entirely.
