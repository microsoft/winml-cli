# src/winml/modelkit/eval/evaluate.py

## TL;DR

Migrated the `winml eval` CLI boundary off the legacy `device=` string passthrough onto the new `WinMLEPDevice` runtime descriptor. `_load_model` now constructs an `EPDeviceTarget(ep="auto", device=device)`, resolves it to a concrete target via `resolve_device(...)`, then binds the registry's `WinMLEPDevice` via `WinMLEPRegistry.instance().auto_device(target)`. The resolved `ep_device` is forwarded keyword-wise to `WinMLAutoModel.from_onnx` and positionally to `from_pretrained`.

No CLI / user-facing change — `WinMLEvaluationConfig.device: str` is still a string and there is no `ep` field in eval config. But the internal handoff is now an immutable (EP, device) binding pair instead of a raw string.

## Diff metrics

- Lines changed: +8 / -2 (11 total per `git show --stat`).
- Functions touched: `_load_model` only.
- New inline imports: `from ..session import EPDeviceTarget, WinMLEPRegistry, resolve_device`.
- Removed call args: `device=config.device` (×2, one on `from_onnx`, one on `from_pretrained`).
- Added call args: `ep_device=ep_device` (keyword, on `from_onnx`); `ep_device` (positional, on `from_pretrained`).

## Role before vs after

Role unchanged: `_load_model(config)` is still the eval pipeline's entry point that translates a `WinMLEvaluationConfig` into a constructed `WinMLPreTrainedModel`, delegating to `WinMLAutoModel`. What changed is the wire format crossing the eval→models boundary:

- **Before (parent `7a66c024`):** raw `device: str` (e.g. `"npu"`, `"cpu"`, `"auto"`) forwarded straight through to `WinMLAutoModel`, which internally invoked the now-deleted `_find_ep_device(ep_name)` non-deterministic first-match resolver.
- **After:** the eval boundary is now responsible for resolving `(ep, device) → EPDeviceTarget → WinMLEPDevice` exactly twice: first by `resolve_device` (pure deduction, picks the concrete `EPDeviceTarget`), then by `WinMLEPRegistry.instance().auto_device(target)` (binds an `OrtEpDevice` handle from the registry). Downstream code receives the immutable bound descriptor.

This file is the canonical example of the commit-message bullet: eval/evaluate.py migrated to ep_device at CLI boundaries.

## Symbol-level changes

`_load_model(config: WinMLEvaluationConfig) -> WinMLPreTrainedModel`:

- **Added import inside function:** `from ..session import EPDeviceTarget, WinMLEPRegistry, resolve_device` (deferred to keep import graph lazy, matching the existing pattern with `WinMLAutoModel`).

- **Added 4-line pre-amble** after the `model_id is None` guard:
  ```python
  # Resolve EPDeviceTarget then bind a WinMLEPDevice at the boundary. Eval
  # config has no explicit ep field; resolve_device deduces from device.
  device = config.device.lower()
  target = resolve_device(EPDeviceTarget(ep="auto", device=device))
  ep_device = WinMLEPRegistry.instance().auto_device(target)
  ```
  Two-step resolution:
  1. `EPDeviceTarget(ep="auto", device=device)` — validates the `device` string against `VALID_DEVICES ∪ {"auto"}` at construction time (the dataclass's `__post_init__`).
  2. `resolve_device(target)` — pure-deduction, picks the concrete `(ep, device)` pair (uses `default_ep_for_device` when ep="auto", filtered by `available_eps()`).
  3. `WinMLEPRegistry.instance().auto_device(target)` — actually binds an `OrtEpDevice` handle, after registering the EP if needed (Path A registration).

- **`from_onnx(...)` call:** `device=config.device` removed, `ep_device=ep_device` inserted as keyword (`from_onnx` makes `ep_device` keyword-only after `*` per `models/auto.py:104-105`).

- **`from_pretrained(...)` call:** `device=config.device` removed, `ep_device` passed **positionally** as the second positional arg (after `config.model_id`). Matches the new `from_pretrained` signature `(cls, model_id_or_path, ep_device, *, ...)` — `ep_device` is positional-or-keyword in the new API, not keyword-only (`models/auto.py:233-237`).

## Behavior / contract changes

- **Resolution timing shifts upstream.** Device→EP deduction now happens once at the eval boundary instead of being re-derived inside `WinMLSession.__init__` per construction.
- **Case-folding is now eval's responsibility.** `_load_model` lowercases `config.device` before calling `resolve_device`. Previously the raw string was passed and lower-casing (if any) happened deep in `WinMLSession` / `_find_ep_device`.
- **Determinism.** With the catalog-ordered preference list backing `resolve_device` + `auto_device`, the same `device` input now always resolves to the same `WinMLEPDevice`. Previously `_find_ep_device(ep_name)` used "first-match" iteration over a non-canonical map (per commit body — "auto_detect_device walks a single unified list").
- **Public eval CLI/API surface unchanged.** `WinMLEvaluationConfig.device: str` is still a string; no `ep`/`ep_device` field added to user-facing config. Users of `winml eval` see the same flags.
- **Implicit "auto" handling.** `config.device` defaults to `"cpu"` in `WinMLEvaluationConfig`, not `"auto"`. The lowercase + `resolve_device(EPDeviceTarget(ep="auto", device=device))` path will accept whatever the catalog accepts; "auto" semantics live inside `resolve_device`, not in `WinMLSession` anymore.
- **`auto_device` may raise** — `WinMLEPRegistrationFailed` (registration failed at ORT level), `WinMLEPNotDiscovered` (no plugin found), or `DeviceNotFound` (no `OrtEpDevice` matched after registration). None of these are caught here — they bubble up to the click command layer, which the commit body promises catches them at "CLI boundaries with remediation hints". This file is a CLI boundary itself for `winml eval`, but the catch is elsewhere (likely in `commands/eval.py`).

## Cross-file impact

This file is purely a **consumer** of the new session API. It depends on:

- `src/winml/modelkit/session/__init__.py` exporting `EPDeviceTarget`, `WinMLEPRegistry`, `resolve_device` (confirmed at l.46, l.63, l.73).
- `WinMLEPRegistry.auto_device(target)` — the method that binds an `OrtEpDevice` handle for a resolved target (`session/ep_registry.py:357`).
- `src/winml/modelkit/models/auto.py`'s new `from_onnx(*, ep_device: WinMLEPDevice, ...)` keyword-only signature and `from_pretrained(model_id, ep_device: WinMLEPDevice, *, ...)` positional-then-keyword signature.

If the upstream signatures change again (e.g. `ep_device` becomes keyword-only on `from_pretrained`), this file will break — the positional call `WinMLAutoModel.from_pretrained(config.model_id, ep_device, task=...)` is fragile to that.

No other files in the eval/ subtree appear to construct `WinMLSession`/`WinMLAutoModel` — this is the single migration point for the eval CLI.

## Risks / subtleties

1. **Positional vs keyword inconsistency.** `from_onnx` requires `ep_device` keyword-only (it's after `*` in the signature); `from_pretrained` takes it positionally. The eval code reflects both — easy to introduce a bug if anyone refactors. A future caller might mis-match. Verified at `models/auto.py:101-115` (kw-only) and `models/auto.py:233-247` (positional).
2. **Two-step resolution couples eval to registry I/O.** `WinMLEPRegistry.instance().auto_device(target)` may register a plugin EP DLL — a side-effect that previously happened deeper inside `WinMLSession`. The eval pipeline now incurs registration cost at `_load_model` time, even before any data is loaded. For lightweight tasks (small models, small datasets), this slightly skews startup-time perception.
3. **No `ep` plumbing.** Eval cannot currently target a non-default EP for a given device (e.g. force the CPU EP on an NPU device, or pick between two NPU EPs). The commit message admits this is intentional ("eval config has no explicit ep field") but it's a regression compared to fine-grained `--ep` exposure in `winml perf`.
4. **Lower-casing only.** `config.device.lower()` handles case but not whitespace/aliases. If `EPDeviceTarget(...)` raises `ValueError` for invalid inputs (it validates in `__post_init__`), or `resolve_device` raises `DeviceNotFound`, those exceptions bubble up unwrapped to the eval CLI — the commit body claims `DeviceNotFound / EPNotDiscovered` are caught at CLI boundaries with remediation hints, but I see no `try/except` here. The catch lives in the click command that invokes `evaluate()`, not in `_load_model`.
5. **`config.model_path` branch ignores `ep` entirely.** Same as before, but worth noting: the skip-build path also takes `ep_device` now, so a pre-built ONNX will still be wrapped with the resolved (EP, device) for inference. That is the desired behavior.
6. **`WinMLEPRegistry.instance()` is now the sole singleton entrypoint** (per commit body — `__new__` + `_initialized` guard deleted). This file uses `.instance()` correctly. If a test wants to inject a `fresh_registry` fixture (also mentioned in the commit body), the singleton swap must happen before `_load_model` is called — fragile coupling for tests.
7. **`EPDeviceTarget(ep="auto", device=device)` construction can raise** if `device` is not in `VALID_DEVICES ∪ {"auto"}` (it validates in `__post_init__` per `session/ep_device.py:187-217`). The `.lower()` normalisation is needed to avoid spurious `ValueError` on `"CPU"` / `"Cpu"` / `"AUTO"` inputs.

## Open questions / TODOs surfaced

- Should `WinMLEvaluationConfig` grow an `ep: str | None = None` field for symmetry with `perf` and `compile`? The commit message lists eval among CLI boundaries migrated to ep_device but stops short of giving eval the `--ep` flag.
- Where exactly is `DeviceNotFound` / `WinMLEPRegistrationFailed` / `WinMLEPNotDiscovered` caught for `winml eval`? Not in this file — needs a grep at the click command level. If no command-layer catch exists, headless servers will surface raw tracebacks.
- Is there a test covering `_load_model` with the new EP-device resolution? The commit reports ~720 passing but doesn't enumerate eval-specific coverage of the EPDevice migration.
- The `target` local variable is bound and used only once — could be inlined as `WinMLEPRegistry.instance().auto_device(resolve_device(EPDeviceTarget(ep="auto", device=device)))`. Verbose vs readable tradeoff.

## Simplification opportunities

- **Inline the `target` local.** `target` is named once, used once. Inline as: `ep_device = WinMLEPRegistry.instance().auto_device(resolve_device(EPDeviceTarget(ep="auto", device=device)))`. Three lines collapse to one. The current form is more debuggable (you can set a breakpoint after `target = ...`), but for production code the one-liner is fine.
- **Push the two-step (resolve + bind) into a single facade helper.** Almost every CLI boundary will do this same two-step. A `WinMLEPRegistry.instance().resolve_and_bind(EPDeviceTarget)` method, or a free function `bind_ep_device(target: EPDeviceTarget) -> WinMLEPDevice`, would collapse every CLI boundary's 3-line incantation to 1 line. This pattern is also in `commands/perf.py`, `commands/build.py`, `commands/compile.py` — DRY win across the codebase.
- **`device = config.device.lower()`** is a one-line normalisation. If `WinMLEvaluationConfig` ever validates `device` at construction (via a `__post_init__` that lower-cases), this line vanishes. Currently the config dataclass does not — but symmetric normalisation in the config layer would simplify all CLI boundaries.
- **The positional `from_pretrained(config.model_id, ep_device, task=...)` call vs the keyword `from_onnx(..., ep_device=ep_device, ...)`** is asymmetric. A consistent calling convention (force keyword for both, or force positional for both at the signature level in `models/auto.py`) would simplify here. Currently the inconsistency leaks from `models/auto.py` into every caller.
- **No need for `from ..models import WinMLAutoModel`** to be inside the function — both inline imports could be hoisted to module scope if the lazy-import concern is gone post-refactor. The session and models packages are presumably not circular against eval. (Check: `eval` is imported by `commands/eval.py`; `session` and `models` are not imported by `eval/*.py` at module scope today. If they were hoisted, no cycle would form.)
