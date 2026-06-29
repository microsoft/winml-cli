# src/winml/modelkit/compiler/configs.py

## TL;DR
`WinMLCompileConfig` gains an optional `ep_device: EPDeviceTarget | None` slot plus a new factory `for_ep_device(ep_device)`, and the `to_dict`/`from_dict` serializers learn to round-trip it. The intent is to thread a fully-resolved `EPDeviceTarget` (intent type, not the registered `WinMLEPDevice` pair) from the CLI boundary down to `CompileStage` so `resolve_device()` is called exactly once at the `winml compile` entry point. `EPDeviceTarget` is imported under `TYPE_CHECKING` plus deferred local imports in `for_ep_device` / `from_dict` to break the `compiler ↔ session` import cycle. **Two latent bugs slipped through the squash:** (1) `import warnings` was removed from the imports but eight `warnings.warn(...)` calls in the `for_qnn` / `for_cpu` / ... factories still reference it — calling any factory with `quantize=True` raises `NameError: name 'warnings' is not defined`; (2) `_EP_CONTEXT_DEFAULTS: Final[frozenset[str]] = frozenset({"qnn", "openvino"})` was added at module level with a comment ("replaces the per-EP factory boilerplate") but **is never read anywhere** in the codebase — the per-EP `enable_ep_context=...` literals in the eight `for_*` factories remain intact. A half-finished consolidation.

## Diff metrics
- ~32 LOC added net; 4 LOC structurally restructured.
- **Imports churn:** removed `import warnings` and `from typing import Any`; added `from typing import TYPE_CHECKING, Final` and the `TYPE_CHECKING` block for `EPDeviceTarget`.
- New module-level constant `_EP_CONTEXT_DEFAULTS` (3 lines with comment + assignment). Dead.
- New `WinMLCompileConfig.ep_device` field (1 line) plus the explanatory comment.
- New classmethod `WinMLCompileConfig.for_ep_device` (~18 LOC including docstring).
- `to_dict` body switched from single-expression `return { ... }` to `d: dict[str, Any] = { ... }; if self.ep_device is not None: d["ep_device"] = self.ep_device.to_dict(); return d`. (`Any` is referenced in the annotation despite the import removal — survives only because `from __future__ import annotations` stringifies all type hints.)
- `from_dict` gains the deferred-local `from ..session import EPDeviceTarget as _EPDeviceTarget`, an `if "ep_device" in data and data["ep_device"] is not None: ep_device = _EPDeviceTarget.from_dict(...)` branch, and `ep_device=ep_device` in the `cls(...)` call.
- Docstring `Examples:` block rewritten to demonstrate `for_ep_device(resolve_device(EPDeviceTarget(ep="qnn", device="npu")))` instead of `for_qnn()` / `for_cpu()`.
- No deletions of pre-existing public symbols. All previous `for_*` factories (`for_qnn`, `for_cpu`, `for_cuda`, `for_dml`, `for_nv_tensorrt_rtx`, `for_openvino`, `for_vitisai`, `for_migraphx`, `for_provider`) and the `device` property are intact.

## Role before vs after
- **Before:** Pure dataclass over `EPConfig` (provider as a free-form short string) + a battery of `for_*` factories. The CLI built an `EPConfig`, dropped it into `WinMLCompileConfig`, and the downstream stage rediscovered the device by `resolve_device(ep=...)` or `device=context.execution_provider`.
- **After:** Same surface plus an explicit "resolved intent" slot. Callers that have already gone through `resolve_device(EPDeviceTarget(...))` (i.e. the top-level CLI in `commands/compile.py`) attach the resolved `EPDeviceTarget` via `for_ep_device`; callers that don't (the broken sub-CLI in `compiler/cli.py`, plus any direct API consumers that hand-construct `WinMLCompileConfig`) leave it `None` and the stage falls back to `resolve_device(EPDeviceTarget(ep=ep_str or "auto", device="auto"))`.

## Symbol-level changes
- **New imports:**
  - `from typing import TYPE_CHECKING, Final` (was `from typing import Any` — both `Any` and `warnings` deleted).
  - Under `if TYPE_CHECKING:` block: `from ..session import EPDeviceTarget  # noqa: TC004`. (`TC004` suppression because the symbol is also reached at runtime via local imports inside `for_ep_device` / `from_dict`.)
- **New module-level constant** `_EP_CONTEXT_DEFAULTS: Final[frozenset[str]] = frozenset({"qnn", "openvino"})` — annotated as "Single source of truth — replaces the per-EP factory boilerplate that used to encode this same bit across 8 methods" but **never referenced**. Dead.
- **New field on `WinMLCompileConfig`:** `ep_device: EPDeviceTarget | None = None`. Placed between `ep_config` and `validate`, with a leading comment explaining the "None means infer in CompileStage" contract.
- **New classmethod `for_ep_device(cls, ep_device: EPDeviceTarget) -> WinMLCompileConfig`:**
  - Local-imports `short_ep_name` from `..session` (runtime-deferred to break the cycle).
  - Resolves the short provider via `short_ep_name(ep_device.ep)`.
  - `base = cls.for_provider(provider) or cls(ep_config=EPConfig(provider=provider))` — reuses an existing `for_*` factory if the short name matches a known EP, else falls back to a generic `EPConfig(provider=provider)` (note: in the generic branch, **`enable_ep_context` defaults to `EPConfig.__init__`'s default `True`**, not False — unlike `for_provider`'s generic fallback which forces False).
  - Stamps `base.ep_device = ep_device` and returns it.
- **`to_dict` modified:**
  - Body switched from single-expression `return { ... }` to a typed local `d: dict[str, Any] = { ... }`.
  - Appends `d["ep_device"] = self.ep_device.to_dict()` when `self.ep_device is not None`. Otherwise the key is omitted (consumers must use `data.get("ep_device")`).
- **`from_dict` modified:**
  - Adds local-import `from ..session import EPDeviceTarget as _EPDeviceTarget` (same cycle-avoidance pattern as `for_ep_device`).
  - Reads `data["ep_device"]` only when both present and non-None, rehydrates via `_EPDeviceTarget.from_dict(...)`, and threads it into the `cls(... ep_device=ep_device ...)` call.

## Behavior / contract changes
- **New optional config field.** Existing call sites that construct `WinMLCompileConfig(ep_config=...)` or use `for_*` factories continue to work — `ep_device` defaults to `None`. The CLI (`commands/compile.py:220`) calls `WinMLCompileConfig.for_ep_device(ep_device_resolved)` after running `resolve_device`.
- **`to_dict` payload now has a conditional `ep_device` key.** Downstream consumers (notably `CompileStage.process`, which reads `context.config.get("ep_device")`) must use `.get()` to remain safe. The dict has the shape `{"ep": str, "device": str, "source": str | None}` per `EPDeviceTarget.to_dict()` (`asdict` on the frozen dataclass). **Note:** this is the *intent type*, not the *handle pair* — it carries the user-craftable triple, not vendor/device IDs.
- **`from_dict` is forward-compatible.** Missing or `None` `ep_device` is treated as "not set"; presence triggers rehydration. `EPDeviceTarget.from_dict` ignores legacy `vendor_id`/`device_id`/`vendor` keys (Batch C strip), so previously persisted v1 JSONs round-trip cleanly.
- **No mutation of `EPConfig`'s `provider` field via `for_ep_device`.** Well — there is, transitively, via `cls.for_provider(provider)` setting `EPConfig(provider=provider)`. The two (provider short name and `ep_device.ep`) must stay consistent; nothing enforces that invariant.
- **Backward-compatible `device` property.** Still returns `ep_config.provider`, not `ep_device.device`. So `cfg.device` is a provider short name (e.g. `"qnn"`), not a hardware category (e.g. `"npu"`) — a long-standing naming wart preserved.
- **NEW RUNTIME BUG: `warnings.warn` calls raise `NameError`.** Eight `for_*` factories (`for_qnn`, `for_cpu`, `for_cuda`, `for_dml`, `for_nv_tensorrt_rtx`, `for_openvino`, `for_vitisai`, `for_migraphx`) emit a `DeprecationWarning` when `quantize is not None`. The `import warnings` line was removed in this squash. Direct verification:
  ```
  >>> from winml.modelkit.compiler import configs
  >>> configs.WinMLCompileConfig.for_qnn(quantize=True)
  NameError: name 'warnings' is not defined. Did you mean: 'Warning'?
  ```
  This is only triggered on the legacy `quantize=` path; current callers leave it as `None`, so the bug is dormant in CI but lethal for anyone passing the kwarg.

## Cross-file impact
- **Producer:** `commands/compile.py:220` calls `WinMLCompileConfig.for_ep_device(ep_device_resolved)` after running `resolve_device(EPDeviceTarget(ep=..., device=...))` at the CLI boundary.
- **Consumer:** `compiler/stages/compile.py` (line 70-84) reads via `WinMLCompileConfig.from_dict(context.config)` then prefers `context.config.get("ep_device")` (dict from `to_dict`) over `compile_cfg.ep_device` (the rehydrated `EPDeviceTarget`) over `resolve_device(EPDeviceTarget(ep=ep_str or "auto", device="auto"))` (fallback). It calls `WinMLEPRegistry.instance().auto_device(target)` to materialize the `WinMLEPDevice` pair, then passes that pair (not the `EPDeviceTarget`) to `WinMLSession(... ep_device=...)`. This is exactly the path that eliminates the cross-package `_EP_TO_DEVICE` import.
- `WinMLBuildConfig.to_dict / from_dict` (in `config/build.py`) round-trip through `WinMLCompileConfig.to_dict / from_dict`, so the new `ep_device` key flows transparently through the full build config.
- **`compiler/cli.py` still imports `CalibrationConfig` and `QDQConfig` from this module** — those symbols don't exist here (whether they ever did is moot; the import fails today). See `compiler__cli.md`. Combined with the `warnings` deletion, this commit shipped at least two import-time / call-time `NameError`s in this corner of the package.

## Risks / subtleties
- **`warnings.warn` in eight factories — `NameError` at runtime** when the deprecated `quantize=` kwarg is passed. Has to be fixed before the legacy API surface can be exercised.
- **`_EP_CONTEXT_DEFAULTS` is dead code.** Comment claims it replaces "the per-EP factory boilerplate that used to encode this same bit across 8 methods" but the eight `for_*` methods still hand-encode `enable_ep_context=False` (or `True`) inline. The consolidation was started, not finished. Either delete the constant or actually use it (e.g. `cls(ep_config=EPConfig(provider=ep, enable_ep_context=ep in _EP_CONTEXT_DEFAULTS))`).
- **The `EPConfig.provider` vs `ep_device.ep` consistency contract is implicit.** `for_ep_device` derives `provider` from `ep_device.ep`, but if a caller hand-builds a config and sets `ep_device` and a different `EPConfig.provider`, nothing validates the mismatch. `CompileStage` will trust `ep_device` (preferred branch).
- **`for_ep_device` generic fallback has a different `enable_ep_context` default than `for_provider`'s generic fallback.** `for_provider`'s fallback at line 147 is `cls(ep_config=EPConfig(provider=provider, enable_ep_context=False))` (forced False); `for_ep_device`'s fallback at line 117 is `cls(ep_config=EPConfig(provider=provider))` (defaults to True via `EPConfig.__init__`). For known EPs the for_provider table wins, but for an unknown EP threaded via `EPDeviceTarget(ep="customep", ...)`, the two factories disagree.
- **`EPDeviceTarget.__post_init__` validates `ep` against `known_ep_short_names() | _FULL_TO_SHORT`.** So `for_ep_device` can never receive an unknown EP unless the caller hand-bypasses the dataclass validation — meaning the generic fallback at line 117 is effectively dead. Worth simplifying away.
- **`TYPE_CHECKING` import requires `from __future__ import annotations`** (already present at the top of the file). Annotations are stringified, so the runtime side never resolves `EPDeviceTarget` for the field annotation itself. The `Any` typing annotation in `dict[str, Any]` survives the same way — annotations only — which is why the missing `from typing import Any` doesn't bite at import.
- **Deferred local imports inside `for_ep_device` and `from_dict`** are necessary because module-level `from ..session import ...` would create a cycle once `session` (or its transitive imports) needs anything from `compiler`. But `WinMLCompileConfig.from_dict({"ep_device": {...}})` will trigger a `session` import on first call, which loads onnxruntime as a side effect — slow path that callers should know about.

## Open questions / TODOs surfaced
- **Fix `warnings.warn` NameError** (1-line `import warnings` add).
- **Use or delete `_EP_CONTEXT_DEFAULTS`.** If used, the eight `for_*` factories collapse into a 1-line `for_provider`-driven dispatcher.
- Should `for_ep_device` also harden the `EPConfig.enable_ep_context` setting based on the EP family (i.e. drive it from `_EP_CONTEXT_DEFAULTS`)?
- Should `to_dict`/`from_dict` enforce that `ep_device.ep` (in target) and `ep_config.provider` (short) refer to the same EP? Cheap invariant check that would catch hand-built mismatches.
- Should `WinMLCompileConfig.device` (the legacy property) be deprecated now that an `EPDeviceTarget` is available — its return shape (provider short name) is confusing relative to `EPDeviceTarget.device` (hardware category).

## Simplification opportunities
- **Collapse eight `for_*` factories into one.** With `_EP_CONTEXT_DEFAULTS` defined and the `quantize=` kwarg now a deprecated no-op, the entire `for_qnn` / `for_cpu` / `for_cuda` / `for_dml` / `for_nv_tensorrt_rtx` / `for_openvino` / `for_vitisai` / `for_migraphx` block (~120 LOC, lines 149-265) reduces to:
  ```python
  @classmethod
  def for_provider(cls, provider: str | None) -> WinMLCompileConfig | None:
      if provider is None:
          return None
      return cls(ep_config=EPConfig(
          provider=provider,
          enable_ep_context=provider in _EP_CONTEXT_DEFAULTS,
      ))
  ```
  This eliminates the `factories: dict[str, Any]` dispatch table too. Per the project's CLAUDE.md and feedback_no_back_compat note, the user prefers hard-break consolidation.
- **Drop `for_ep_device`'s generic fallback** (`or cls(ep_config=EPConfig(provider=provider))` at line 117). `EPDeviceTarget.__post_init__` validates `ep`, so `for_provider(provider)` always returns a config — the `None` branch is dead.
- **Consider deleting `WinMLCompileConfig.device` property** — its return value (provider short string) is semantically conflated with `EPDeviceTarget.device` (hardware class). One-line property; rename or remove.
- **`_EP_CONTEXT_DEFAULTS` is dead today**; either wire it into the factory simplification above or remove it.
- **`Any` and `warnings` imports**: with the factory simplification, `warnings` goes away entirely (no more deprecation shims); `Any` should be re-added to satisfy strict-runtime tools even though `from __future__ import annotations` saves us at import.
