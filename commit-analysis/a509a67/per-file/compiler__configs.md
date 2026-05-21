# src/winml/modelkit/compiler/configs.py

## TL;DR
`WinMLCompileConfig` gains an optional `ep_device: EPDevice | None` field plus a new factory `for_ep_device(ep_device)`, and the `to_dict`/`from_dict` serializers learn to round-trip it. The intent is to thread a fully-resolved `(EP, device)` binding from the CLI boundary down to `CompileStage`, so `resolve_device()` is called exactly once and `CompileStage` no longer has to import the private `_EP_TO_DEVICE` map. `EPDevice` is imported under `TYPE_CHECKING` (with deferred `from ..session import ...` inside method bodies) to avoid a config↔session import cycle at module load time.

## Diff metrics
- ~30 LOC added; 2 LOC restructured (the `to_dict` `return { ... }` became `d = { ... }; ...; return d`).
- No deletions of pre-existing public symbols. All previous factories (`for_qnn`, `for_cpu`, `for_cuda`, `for_dml`, `for_tensorrt`, `for_openvino`, `for_vitisai`, `for_migraphx`, `for_provider`) and the `device` property are intact.

## Role before vs after
- **Before:** Pure dataclass over `EPConfig` (provider as a free-form short string) + a battery of `for_*` factories. The CLI built an `EPConfig`, dropped it into `WinMLCompileConfig`, and the downstream stage rediscovered the device by string-matching the provider against a private map.
- **After:** Same surface plus an explicit "resolved binding" slot. Callers that have an `EPDevice` (i.e. the top-level CLI after `resolve_device(ep, device)`) attach it via `for_ep_device`; callers that don't (older API consumers, calibrate-only flows) leave it `None` and the stage falls back to `resolve_device(ep=...)`.

## Symbol-level changes
- **New imports:**
  - `from typing import TYPE_CHECKING, Any` (added `TYPE_CHECKING`).
  - Under `if TYPE_CHECKING:` block: `from ..session import EPDevice  # noqa: TC004`. (`TC004` suppression because the symbol is also used at runtime via local imports inside method bodies.)
- **New field on `WinMLCompileConfig`:** `ep_device: EPDevice | None = None`. Placed between `ep_config` and `validate`, with a docstring comment explaining the "None means infer in CompileStage" contract.
- **New classmethod `for_ep_device(cls, ep_device: EPDevice) -> WinMLCompileConfig`:**
  - Local-imports `short_ep_name` from `..session` (runtime-deferred to break the cycle).
  - Resolves the short provider via `short_ep_name(ep_device.ep)`.
  - Reuses `for_provider(provider)` if the short name matches a known EP factory; otherwise falls back to a generic `cls(ep_config=EPConfig(provider=provider))`.
  - Stamps `base.ep_device = ep_device` and returns it.
- **`to_dict` modified:**
  - Body switched from single-expression `return { ... }` to a typed local `d: dict[str, Any] = { ... }`.
  - Appends `d["ep_device"] = self.ep_device.to_dict()` when `self.ep_device is not None`. Otherwise the key is omitted (consumers must use `data.get("ep_device")`).
- **`from_dict` modified:**
  - Adds local-import `from ..session import EPDevice as _EPDevice` (same cycle-avoidance pattern).
  - Reads `data["ep_device"]` only when both present and non-None, rehydrates via `_EPDevice.from_dict(...)`, and threads it into the `cls(... ep_device=ep_device ...)` call.

## Behavior / contract changes
- **New optional config field.** Existing call sites that construct `WinMLCompileConfig(ep_config=...)` or use `for_*` factories continue to work — `ep_device` defaults to `None`. The CLI is expected (per commit body) to set it via the new factory.
- **`to_dict` payload now has a conditional `ep_device` key.** Downstream consumers (notably `CompileStage.process`, which reads `context.config.get("ep_device")`) must use `.get()` to remain safe. The dict has the shape `{"ep": str, "device": str, "vendor_id": int, "device_id": int, "vendor": str}` per `EPDevice.to_dict()` (`asdict` on the frozen dataclass).
- **`from_dict` is forward-compatible.** Missing or `None` `ep_device` is treated as "not set"; presence triggers rehydration. Field-name spelling matches `EPDevice.from_dict` exactly (`ep`, `device`, `vendor_id`, `device_id`, `vendor`).
- **No mutation of `EPConfig`'s `provider` field.** When `for_ep_device` is used, the short name is written to `EPConfig.provider` AND `ep_device` is attached. The two must stay consistent; nothing enforces that invariant.
- **Backward-compatible `device` property.** Still returns `ep_config.provider`, not `ep_device.device`. So `cfg.device` is a provider short name (e.g. `"qnn"`), not a hardware category (e.g. `"npu"`) — a long-standing naming wart preserved.

## Cross-file impact
- Producers of `ep_device`-bearing configs (commit body lists `commands/compile.py`, `commands/perf.py`, `models/auto.py`, `models/winml/base.py`, `eval/evaluate.py`) call `for_ep_device` after running `resolve_device(ep, device)` at the CLI boundary.
- Consumer: `compiler/stages/compile.py` reads `context.config.get("ep_device")` (the dict produced by `to_dict`) or `compile_cfg.ep_device` (after `from_dict`) and instantiates `WinMLSession(..., ep_device=...)` accordingly. This is exactly the path that eliminates the cross-package `_EP_TO_DEVICE` import.
- `WinMLBuildConfig.to_dict / from_dict` (in `config/build.py`) round-trip through `WinMLCompileConfig.to_dict / from_dict`, so the new `ep_device` key flows transparently through the full build config.

## Risks / subtleties
- **The `EPConfig.provider` vs `ep_device.ep` consistency contract is implicit.** `for_ep_device` derives `provider` from `ep_device.ep`, but if a caller hand-builds a config and sets `ep_device` and a different `EPConfig.provider`, nothing validates the mismatch. `CompileStage` will trust `ep_device` (preferred branch).
- **`EPDevice` is a frozen dataclass whose `ep` is stored as the canonical full name** (e.g. `"QNNExecutionProvider"`), not the short name. `for_ep_device` calls `short_ep_name(ep_device.ep)` to recover the short form for `EPConfig.provider`. If `short_ep_name` ever falls back to its safe-default branch (`full.removesuffix("ExecutionProvider").lower()`), the resulting provider may not match any `for_*` factory; the code then takes the generic `cls(ep_config=EPConfig(provider=provider))` branch, which sets `enable_ep_context` per the EPConfig default (True), not per the EP-specific policy (most non-QNN factories disable it). This is a subtle quirk.
- **`TYPE_CHECKING` import for `EPDevice` requires `from __future__ import annotations`** (already present at the top of the file). Annotations are stringified, so the runtime side never resolves `EPDevice` for the field annotation itself.
- **The deferred local imports inside `for_ep_device` and `from_dict`** are idiomatic and necessary: any module-level `from ..session import ...` would create a cycle once `session` (or its transitive imports) needs anything from `config`. But it does mean that `WinMLCompileConfig.from_dict({"ep_device": {...}})` will trigger a `session` import on first call, which loads onnxruntime as a side effect.

## Open questions / TODOs surfaced
- Should `for_ep_device` also harden the `EPConfig.enable_ep_context` setting based on the EP family? Today the result depends on whether the short name happens to match `for_provider`'s factory table.
- Should `to_dict`/`from_dict` enforce that `ep_device.ep` (full) and `ep_config.provider` (short) refer to the same EP? Cheap invariant check that would catch hand-built mismatches.
- Whether `WinMLCompileConfig.device` (the legacy property) should be deprecated now that an `EPDevice` is available — its return shape (provider short name) is confusing relative to `EPDevice.device` (hardware category).
