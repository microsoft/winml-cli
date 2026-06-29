# src/winml/modelkit/compiler/stages/compile.py

## TL;DR
`CompileStage` is migrated to the new `EPDeviceTarget` plumbing. (1) The `process()` method now resolves a `WinMLEPDevice` from a 3-way priority chain (context dict → config field → fresh `resolve_device()`), then calls `WinMLEPRegistry.instance().auto_device(target)` to materialize the registered pair, and constructs `WinMLSession(... ep_device=ep_device, ...)` instead of the legacy `device=context.execution_provider` kwarg. (2) `_finalize_output` is taught the new naming protocol — `WinMLSession.compile()` writes `{stem}_{ep_device.device}_ctx.onnx` (e.g. `_npu_ctx.onnx`), not `{stem}_{provider}_ctx.onnx` (e.g. `_qnn_ctx.onnx`), so the file-search list is widened with an additional pattern derived from `ep_to_device(execution_provider)`. (3) The module-level import broadens from `WinMLQairtSession, WinMLSession` to also pull in `EPDeviceTarget, WinMLEPRegistry, resolve_device`; no more cross-package access to the private `_EP_TO_DEVICE` map.

## Diff metrics
- ~22 LOC net added (`process` grows ~13 LOC; `_finalize_output` grows ~10 LOC).
- Module-level import switched: `from ...session import WinMLQairtSession, WinMLSession` → `from ...session import (EPDeviceTarget, WinMLEPRegistry, WinMLQairtSession, WinMLSession, resolve_device,)`.
- One new in-function lazy import added: `from ...session import ep_to_device` inside `_finalize_output`.

## Role before vs after
- **Before:** Built a `WinMLCompileConfig` from `context.config` solely to extract `ep_config`, then passed `device=context.execution_provider` (a short provider string) to `WinMLSession`. `_finalize_output` searched for compiled-context artefacts using only the provider short name in the filename.
- **After:** Same overall structure, but the session is bound to a concrete `WinMLEPDevice` pair via `ep_device`, and the artefact lookup understands that `WinMLSession.compile()` names its outputs by the *device* (e.g. `npu`), not the EP short name (e.g. `qnn`). Avoidance of `_EP_TO_DEVICE` cross-package import was the architectural prize.

## Symbol-level changes
- **`process()`:**
  - Now binds `compile_cfg = WinMLCompileConfig.from_dict(context.config)` (previously was a throwaway temporary) so `compile_cfg.ep_device` is reachable.
  - New 3-way resolver:
    ```python
    ep_device_dict = context.config.get("ep_device")
    if ep_device_dict:
        target: EPDeviceTarget = EPDeviceTarget.from_dict(ep_device_dict)
    elif compile_cfg.ep_device is not None:
        target = compile_cfg.ep_device
    else:
        ep_str = context.execution_provider
        target = resolve_device(
            EPDeviceTarget(ep=ep_str or "auto", device="auto")
        )
    ep_device = WinMLEPRegistry.instance().auto_device(target)
    ```
    The dict branch is for callers that hand a raw dict (round-tripped JSON or `WinMLBuildConfig.to_dict`); the field branch is for in-memory configs built via `WinMLCompileConfig.for_ep_device`; the fallback covers older API callers / the broken sub-CLI that only know an EP string. Note the variable shadowing: `target` is the *intent* (`EPDeviceTarget`); `ep_device` is the *registered pair* (`WinMLEPDevice`). The naming inside the local scope reads cleanly.
  - **`auto_device(target)` may raise `WinMLEPNotDiscovered`, `UnknownListingPick`, `WinMLEPRegistrationFailed`, or `DeviceNotFound`.** None of these are caught here; they bubble up through the outer `except Exception as e:` and become `Compilation failed: {e}`. No remediation hints — by design, that's the CLI layer's job (`commands/compile.py`).
  - Log line restructured: `f"Creating {session_cls.__name__} for {target.ep}/{target.device}"` (was `f"... for device: {context.execution_provider}"`).
  - Session construction: `session_cls(onnx_path=model_path, ep_device=ep_device, ep_config=ep_config)`. The previous `device=context.execution_provider` kwarg is gone — this is one of the "hard break" sites the commit body calls out (`WinMLSession.__init__` now takes a positional `ep_device: WinMLEPDevice`; `device=` kwarg removed).
- **`_finalize_output()`:**
  - Same `device = context.execution_provider.lower()` retained for the *output* filename (so the final emitted artefact keeps the EP-short-name convention, e.g. `..._qnn_ctx.onnx`).
  - Added in-function `from ...session import ep_to_device`.
  - Computes `ep_device_str = ep_to_device(context.execution_provider)` (guarded by `try/except ValueError`; on failure `ep_device_str = None`).
  - When `ep_device_str` is set (e.g. `"npu"`), it is *prepended* (`insert(0, ...)`) onto `ctx_patterns` so the new naming wins the search-order tie. The two legacy patterns (`_{device}_ctx.onnx` and `_ctx.onnx`) are preserved as fallbacks.
  - The rest of `_finalize_output` (copy to output dir, rename `.bin`, rewrite `ep_cache_context`) is unchanged.

## Behavior / contract changes
- **The `_finalize_output` naming protocol is now asymmetric:**
  - **Input-search list (work_dir):** prefers `{stem}_{device_category}_ctx.onnx` (e.g. `..._npu_ctx.onnx`), then falls back to `{stem}_{provider_short}_ctx.onnx` (e.g. `..._qnn_ctx.onnx`), then `{stem}_ctx.onnx`.
  - **Output filename (output_dir):** still `{original_stem}_{provider_short}_ctx.onnx` (e.g. `resnet50_qnn_ctx.onnx`). Same for the renamed `.bin` sidecar.
  - Implication: a model compiled today on NPU will land as `*_qnn_ctx.onnx` in the final output, even though `WinMLSession.compile()` wrote `*_npu_ctx.onnx` to the work dir.
- **`WinMLSession` is called with `ep_device: WinMLEPDevice`** (positional, second arg per `session.py:196-199`). Old code that monkey-patched or subclassed `WinMLSession.__init__` with the `device=` kwarg will break (intentional — Option A hard break).
- **`ep_to_device` accepts a short or full EP name** (it calls `expand_ep_name` internally per `session/ep_device.py:419`). `context.execution_provider` defaults to `"qnn"` (a short name) per `compiler/context.py:88`, so the new naming branch *does* fire on normal CLI invocations. The `try/except ValueError` is the defensive branch for unknown EPs (e.g. a custom EP not in `EP_DEVICE_SPECS`).
- **No more cross-package access to `_EP_TO_DEVICE`.** This file used to (via `WinMLCompileConfig.from_dict(...).ep_config`'s provider-string fall-through) import the private map directly; that violation is removed. The commit-body directive ("do not import private symbols ... outside session/ep_device.py") is honoured.

## Cross-file impact
- Reads from `WinMLCompileConfig.ep_device` (added in the sibling `configs.py` change).
- Reads from `context.config["ep_device"]` (a dict shape produced by `WinMLCompileConfig.to_dict` / `EPDeviceTarget.to_dict`).
- Calls into the session package's public helpers: `EPDeviceTarget.from_dict`, `resolve_device(target)`, `WinMLEPRegistry.instance().auto_device(target)`, `ep_to_device(full_or_short_ep_name)`. All five are part of `winml.modelkit.session.__init__.__all__`.
- The downstream `WinMLSession.compile()` (modified elsewhere in this squash) must honour the new ctx-filename protocol (`{stem}_{ep_device.device}_ctx.onnx`) for the `ctx_patterns.insert(0, ...)` branch to ever match.
- **Affected by the sibling `configs.py` bug:** if `compile_cfg.ep_device` is reached via a config that was rehydrated from a dict missing the `ep_device` key, the field is `None` and the fallback `resolve_device(...)` branch fires — which is fine. But the `WinMLCompileConfig.from_dict` call on line 70 triggers the deferred `from ..session import EPDeviceTarget` import, which loads onnxruntime as a side effect (slow path; first invocation only).

## Risks / subtleties
- **Three-way precedence has a subtle priority inversion.** `context.config.get("ep_device")` (a dict) wins over `compile_cfg.ep_device` (the rehydrated `EPDeviceTarget`), even though the latter is *derived* from the former through `WinMLCompileConfig.from_dict`. They should be equivalent in practice; if they ever disagree (e.g. someone mutates the dict after rehydrating), the dict wins. Probably fine but worth documenting.
- **`context.execution_provider` is still consulted in two places** — once as the fallback EP for `resolve_device(EPDeviceTarget(ep=ep_str or "auto", device="auto"))`, and once as the `device` variable used to compose output filenames. If a user threads an `ep_device` whose EP disagrees with `context.config["execution_provider"]`, the *input* search uses `ep_device`-derived patterns but the *output* still uses `context.execution_provider`-derived names. Hand-built configs could drift.
- **`auto_device` raises an unhandled `ValueError` if the threaded `target` still has `"auto"` on either axis.** The three branches above all produce concrete `EPDeviceTarget`s — but only because `resolve_device` is run in the fallback path. The `if ep_device_dict:` and `elif compile_cfg.ep_device is not None:` branches trust upstream to have already resolved. If a caller passes an unresolved `EPDeviceTarget(ep="auto", device="auto")` via `for_ep_device`, the `auto_device` call will raise `ValueError("auto_device requires a resolved EPDeviceTarget; call resolve_device(target) first")` — surfaced verbatim as `Compilation failed: auto_device requires...`. No defensive `resolve_device(target)` call here means upstream gets to be the gate.
- **The `_finalize_output` fallback chain has overlapping cases.** If both `_{device_short}_ctx.onnx` (where `device_short = "qnn"`) and `_{device_category}_ctx.onnx` (`"npu"`) happen to exist (e.g. left over from a previous CLI), the `insert(0, ...)` ordering means `npu` wins — which is correct for the current `WinMLSession.compile()` output but assumes the session never falls back to writing the older name.
- **Session class still resolved via `COMPILER_SESSION_MAPPING[compiler]`.** `WinMLQairtSession` accepts the same `ep_device=` kwarg signature (per `session/qairt/qairt_session.py`); assumed but not verified here.
- **`compile_cfg.ep_config` is the EP-config seam.** After the 3-way ep_device resolution, the `ep_config` still comes from the round-tripped `WinMLCompileConfig`, meaning `provider_options`, `enable_ep_context`, `embed_context`, `compiler`, `qnn_sdk_root` all flow through `from_dict`. If `ep_config.provider` disagrees with `target.ep`, the session uses the `ep_device` pair (correctly) but `ep_config.provider_options` may have been computed against the wrong EP.

## Open questions / TODOs surfaced
- Should `_finalize_output` also reuse the resolved `target.device` directly (from `process`) instead of re-deriving via `ep_to_device(context.execution_provider)`? Today `_finalize_output` re-runs the inference; threading the resolved `target` (or `ep_device.device.device_type.lower()`) onto `context` would eliminate the second lookup and the try/except. The current code couples `process` and `_finalize_output` only through `context.execution_provider`, which is fragile.
- Should the *output* filename adopt the device-category convention (`*_npu_ctx.onnx`) for consistency, or stay on the provider-short convention (`*_qnn_ctx.onnx`)? Today the file is named after the provider, which mismatches the work_dir convention.
- Should the three-way precedence be flattened — pick one canonical source (`compile_cfg.ep_device`) and require callers to put it there, instead of also supporting the raw dict? Today the dict path is the load-bearing branch (CLI passes `to_dict` output through `context.config`), and the field path is dead weight.

## Simplification opportunities
- **Collapse the 3-way resolver to 2 cases.** The `context.config.get("ep_device")` branch and the `compile_cfg.ep_device` branch are aliases — `compile_cfg` is just `WinMLCompileConfig.from_dict(context.config)`, so `compile_cfg.ep_device is None` iff `context.config.get("ep_device")` is missing/None. Drop the dict-path; rely entirely on `compile_cfg.ep_device or resolve_device(...)`. Saves ~6 LOC, kills the priority-inversion question above.
- **Thread the resolved `target.device` onto `context`** (e.g. `context.config["ep_device_category"] = target.device`) so `_finalize_output` doesn't need to re-call `ep_to_device`. Eliminates the in-function lazy import, the try/except, and the `ctx_patterns.insert(0, ...)` dance.
- **Unify the output naming convention.** Either:
  - Always emit `*_{device_category}_ctx.onnx` (e.g. `*_npu_ctx.onnx`), aligning with `WinMLSession.compile()`'s work-dir convention. Single search pattern, single output pattern; the entire fallback chain collapses.
  - Or always emit `*_{provider_short}_ctx.onnx` (e.g. `*_qnn_ctx.onnx`), and update `WinMLSession.compile()` to match. Either is fine — the current asymmetry is what causes the workaround in `_finalize_output`.
- **`_build_provider_options` (line 165)** is defined on `CompileStage` but never called by `process()` — `ep_config` from `compile_cfg.from_dict(context.config)` already carries `provider_options`. Dead method. (Verified: only `process()` uses `ep_config`, and `_build_provider_options` reads `context.config["provider_options"]` directly which is the same data.)
- **`_finalize_output`'s `ep_to_device` try/except is overcautious.** `context.execution_provider` is sourced from `EPConfig.provider` upstream, which is validated by click's `Choice(VALID_EPS)` in `commands/compile.py`. So `ep_to_device` only fails on a programmatic call with a custom provider name. Either let the `ValueError` propagate (it's a real misconfig) or remove the path entirely once the unified naming convention above is adopted.
