# Review: `src/winml/modelkit/compiler/stages/compile.py`

**Status:** modified
**Lines added/removed:** 8+ / 3-

## 1. Purpose

`CompileStage` is the final stage in the compiler pipeline. Its `process()`
method takes a `CompileContext` (which carries only an EP short name via
`context.execution_provider`), creates a `WinMLSession` or
`WinMLQairtSession`, calls `compile()`, validates outputs, and finalizes the
EPContext artifact. This diff migrates the session-creation call inside
`process()` from the old `device=` string API to the new `ep_device=EPDevice`
API, using `_EP_TO_DEVICE` from `precision.py` to reconstruct the device
category from the EP short name stored in the context.

## 2. Changes summary

- New imports: `from ...config.precision import _EP_TO_DEVICE` and
  `from ...session.ep_device import resolve_device`.
- In `process()` (lines 67-76): the single-line
  `session_cls(onnx_path=model_path, device=context.execution_provider, ep_config=ep_config)`
  call is replaced with a three-step sequence:
  1. `ep_str = context.execution_provider`
  2. `device_str = _EP_TO_DEVICE.get(ep_str, "cpu")` — map EP → device with
     `"cpu"` fallback for unknown EPs.
  3. `ep_device = resolve_device(ep_str, device_str)` — full EP registration
     via `WinMLEPRegistry`.
  Then `session_cls(onnx_path=model_path, ep_device=ep_device, ep_config=ep_config)`.
- Log message updated: `f"for device: {context.execution_provider}"` →
  `f"for {ep_device.ep}/{ep_device.device}"`.

## 3. Per-symbol review

### `CompileStage.process`

- **Role:** Execute model compilation: create session → compile → validate → finalize.
- **Signature:** `def process(self, context: CompileContext) -> CompileContext` (unchanged).
- **Behavior:** The core compilation loop is unchanged. The only behavioral
  delta is that session construction now goes through full EP registration
  (`WinMLEPRegistry.register_ep`) instead of passing a raw device string.
  This means `process()` can now raise `EPNotDiscovered`, `DeviceNotFound`,
  or `AmbiguousMatch` where previously it would have passed a string and
  deferred the error to `WinMLSession`'s internal handling.
- **Invariants:**
  - `context.execution_provider` must be a recognized short EP name (i.e.,
    a key in `_EP_TO_DEVICE`). If it is an unrecognized string, `_EP_TO_DEVICE.get(ep_str, "cpu")`
    silently defaults to `"cpu"`, then `resolve_device(ep_str, "cpu")` will
    attempt to register and likely raise `EPNotDiscovered` — which propagates
    out of `process()` wrapped in the `except Exception` at line 99 and re-raised.
  - `_EP_TO_DEVICE` fallback to `"cpu"` for unknown EP names is a soft
    default. An unknown EP that is genuinely GPU-bound would produce a
    `DeviceNotFound` error from `resolve_device`, which is the right outcome.
  - **Audit Gap #3**: `process()` uses the free function `_build_session_options`
    via `WinMLCompileConfig.from_dict(context.config).ep_config` at line 66 —
    this is correct. The legacy `_build_provider_options` method also exists
    in this class (line 149) but is NOT called in `process()`. This is
    correct and consistent with the design; the `_build_provider_options`
    method appears to be dead code (it is not referenced from `process()` or
    any other stage method in this file). It should be removed or explicitly
    documented as unused.
- **Risks / concerns:**
  - **`_EP_TO_DEVICE` completeness**: `process()` calls
    `_EP_TO_DEVICE.get(ep_str, "cpu")`. If the EP is `"dml"` → `"gpu"`,
    `"qnn"` → `"npu"`, `"vitisai"` → `"npu"`, `"migraphx"` → `"gpu"`,
    `"tensorrt"` → `"gpu"`, `"cpu"` → `"cpu"`. All EPs reachable via the
    build pipeline flow through `WinMLCompileConfig.for_provider(policy.compile_provider)`,
    which is itself keyed from `_EP_TO_DEVICE`-adjacent logic. The mapping
    appears complete for all EPs currently supported by the build pipeline.
    However, `"tensorrt"` and `"cuda"` (if ever passed) have no matching
    `_SHORT_TO_CANONICAL` entry in `ep_device.py` — `resolve_device` would
    call `expand_ep_name("tensorrt")` which returns `"tensorrt"` (unknown →
    passthrough), causing `WinMLEPRegistry.register_ep("tensorrt")` to fail
    with `EPNotDiscovered`. The `except Exception` wrapper at line 99 catches
    this and re-raises as a compile error.
  - **`resolve_device` is a heavyweight call**: it triggers
    `WinMLEPRegistry.get_instance().register_ep(ep_canonical)`, which
    loads/validates the EP plugin. This call was NOT present before this
    diff. For compilation pipelines that are already running in a session
    context (e.g., the EP was already registered), this is likely a no-op.
    For test environments without a real EP, it requires mocking
    `WinMLEPRegistry` or `resolve_device`.
  - **`_EP_TO_DEVICE` is a private import**: cross-package import of a
    private symbol. See `config_precision.md` risk R2.
  - **`_finalize_output` uses `context.execution_provider.lower()`** at
    line 213 for naming the EPContext file — it does NOT use `ep_device.device`
    or `ep_device.ep`. This is inconsistent with the new approach: the
    EPContext filename is derived from the EP short name (e.g., `qnn_ctx.onnx`),
    not from the device kind. This is pre-existing behavior, not introduced
    by this diff, but it's worth noting that `ep_device` is not propagated
    to the finalization step.
- **Tests:** `tests/unit/compiler/test_compiler_stages.py` — `TestCompileStageFinalizeOutput`
  tests `_finalize_output` directly; `TestCompilerPipeline` verifies stage
  ordering. Neither test exercises `CompileStage.process()` with the new
  `ep_device` path. The `process()` method requires a real ORT session and
  EP plugin, so it is not covered by unit tests. This was true before the
  diff as well.

---

### `_build_provider_options` (method, line 149)

- **Role:** Build provider options dict from context config.
- **Signature:** `def _build_provider_options(self, context: CompileContext) -> dict[str, str]`
- **Behavior:** Returns `dict(context.config.get("provider_options", {}))`.
- **Invariants:** Pure function; no side effects.
- **Risks / concerns:** This method is not called anywhere in `process()` or
  other stage methods. It is dead code. The actual provider options are
  delegated to `WinMLCompileConfig.from_dict(context.config).ep_config`
  at line 66.
- **Tests:** Not tested.

## 4. Cross-cutting

- `CompileContext.execution_provider` carries only the short EP name
  (e.g., `"qnn"`), not a full `EPDevice`. The choice to reconstruct
  `EPDevice` at compile time rather than propagating it through the context
  is a design decision — it keeps `CompileContext` serializable (plain
  dict). The trade-off is that the `_EP_TO_DEVICE` lookup becomes a
  correctness gate: if the context carries an EP not in the map, the
  fallback to `"cpu"` is silently wrong before the `resolve_device` call
  surfaces the error.
- Consider whether `CompileContext` should eventually carry an `EPDevice`
  directly, or whether the reconstruction from EP string is permanent. The
  current approach is a reasonable bridge but creates a point of failure
  for non-standard EPs.

## 5. Confidence level

Medium-high. The mapping from EP → device is correct for all EPs currently
used in production (QNN, DML, VitisAI). The `tensorrt`/`cuda` gap in
`_SHORT_TO_CANONICAL` would surface as an `EPNotDiscovered` error rather
than a silent wrong-device scenario, which is an acceptable failure mode.

## 6. Verbatim risk inventory

| # | Location | Risk |
|---|----------|------|
| R1 | `compile.py:69` | `_EP_TO_DEVICE.get(ep_str, "cpu")` — unknown EP defaults silently to `"cpu"` before `resolve_device` raises; if `resolve_device` ever becomes more lenient, the wrong-device scenario would be silent. |
| R2 | `compile.py:70` | `resolve_device(ep_str, device_str)` is a heavyweight EP-registration call; in test contexts this requires mocking `WinMLEPRegistry`. No test currently exercises `process()` end-to-end with a mock registry. |
| R3 | `compile.py:17` | `from ...config.precision import _EP_TO_DEVICE` — private symbol imported across package boundary; rename in `precision.py` breaks this file at import time. |
| R4 | `compile.py:149` (`_build_provider_options`) | Dead method — not called from `process()` or elsewhere. Should be removed to avoid confusion about where provider options are actually applied. |
| R5 | `compile.py:213` (`_finalize_output`) | EPContext naming uses `context.execution_provider.lower()`, not `ep_device.ep` or `ep_device.device`; `ep_device` is not propagated to finalization. Pre-existing issue, not introduced here. |
