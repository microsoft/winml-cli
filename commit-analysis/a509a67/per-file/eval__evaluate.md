# src/winml/modelkit/eval/evaluate.py

## TL;DR

Migrated the `winml eval` CLI boundary off the legacy `device=` string passthrough onto the new `EPDevice` descriptor. `_load_model` now resolves an `EPDevice` from `config.device` via `resolve_device(device=...)` and forwards it positionally/keyword-wise to `WinMLAutoModel.from_pretrained` and `from_onnx`. No behavior change for users — the public eval config still exposes a `device` string only — but the internal handoff is now an immutable resolved (EP, device) pair.

## Diff metrics

- Lines changed: +8 / -2 (10 total)
- Functions touched: `_load_model` only
- New imports: `from ..session import resolve_device`
- Removed call args: `device=config.device` (×2)
- Added call args: `ep_device=ep_device` for `from_onnx`; bare positional `ep_device` for `from_pretrained`

## Role before vs after

Role unchanged: `_load_model(config)` is still the eval pipeline's entry point that translates a `WinMLEvaluationConfig` into a constructed `WinMLPreTrainedModel`, delegating to `WinMLAutoModel`. What changed is the wire format crossing the eval→models boundary:

- **Before:** raw `device: str` (e.g. `"npu"`, `"cpu"`, `"auto"`) forwarded straight through to `WinMLAutoModel`, which internally invoked the now-deleted `_find_ep_device(ep_name)` non-deterministic first-match resolver.
- **After:** the eval boundary is now responsible for resolving `(EP, device) → EPDevice` exactly once via the session-package helper, and downstream code receives the immutable descriptor.

This file is the canonical example of the commit-message bullet: "eval/evaluate.py migrated to ep_device at CLI boundaries".

## Symbol-level changes

`_load_model(config: WinMLEvaluationConfig) -> WinMLPreTrainedModel`:

- Added import inside function: `from ..session import resolve_device` (deferred to keep import graph lazy, matching the existing pattern with `WinMLAutoModel`).
- Added 3-line pre-amble after the `model_id is None` guard:
  ```python
  device = config.device.lower()
  ep_device = resolve_device(device=device)
  ```
  Note `resolve_device` is called **with only `device=`** — no `ep=`. The inline comment ("Eval config has no explicit ep field; resolve_device deduces the ep from the device automatically") is correct: `WinMLEvaluationConfig` exposes `device: str = "cpu"` but no `ep` field (verified in eval/config.py at this commit).
- `from_onnx(...)` call: `device=config.device` removed, `ep_device=ep_device` inserted as a keyword arg (`from_onnx` makes `ep_device` keyword-only after `*`).
- `from_pretrained(...)` call: `device=config.device` removed, `ep_device` passed **positionally** as the second positional arg (after `config.model_id`). Matches the new `from_pretrained` signature `(cls, model_id_or_path, ep_device, *, ...)` — `ep_device` is positional in the new API, not keyword-only.

## Behavior / contract changes

- **Resolution timing shifts upstream.** Device→EP deduction now happens once at the eval boundary instead of being re-derived inside `WinMLSession.__init__` per construction.
- **Case-folding is now eval's responsibility.** `_load_model` lowercases `config.device` before calling `resolve_device`. Previously the raw string was passed and lower-casing (if any) happened deep in `WinMLSession` / `_find_ep_device`.
- **Determinism.** With the catalog-ordered preference list backing `resolve_device`, the same `device` input now always resolves to the same `EPDevice`. Previously `_find_ep_device(ep_name)` used "first-match" iteration over a non-canonical map (per commit body).
- **Public eval CLI/API surface unchanged.** `WinMLEvaluationConfig.device: str` is still a string; no `ep`/`ep_device` field added to user-facing config. Users of `winml eval` see the same flags.
- **Implicit "auto" handling.** `config.device` defaults to `"cpu"` in `WinMLEvaluationConfig`, not `"auto"`. The lowercase + `resolve_device(device=...)` path will accept whatever the catalog accepts; "auto" semantics (if any) now live inside `resolve_device`, not in `WinMLSession`.

## Cross-file impact

This file is purely a **consumer** of the new session API. It depends on:

- `src/winml/modelkit/session/__init__.py` exporting `resolve_device` (confirmed: line 71).
- `src/winml/modelkit/models/auto.py`'s new `from_onnx(*, ep_device, ...)` keyword-only signature and `from_pretrained(model_id, ep_device, *, ...)` positional-ep_device signature.

If the upstream signatures change again (e.g. ep_device becomes keyword-only on `from_pretrained`), this file will break — the positional call `WinMLAutoModel.from_pretrained(config.model_id, ep_device, task=...)` is fragile to that.

No other files in the eval/ subtree appear to construct `WinMLSession`/`WinMLAutoModel` — this is the single migration point for the eval CLI.

## Risks / subtleties

1. **Positional vs keyword inconsistency.** `from_onnx` requires `ep_device` keyword-only (it's after `*` in the signature); `from_pretrained` takes it positionally. The eval code reflects both — easy to introduce a bug if anyone refactors. A future caller might mis-match.
2. **No `ep` plumbing.** Eval cannot currently target a non-default EP for a given device (e.g. force `cpu` EP on an NPU device, or pick between two NPU EPs). The commit message admits this is intentional ("eval config has no explicit ep field") but it's a regression compared to fine-grained `--ep` exposure in `winml perf`.
3. **Lower-casing only.** `config.device.lower()` handles case but not whitespace/aliases. If `resolve_device` raises `DeviceNotFound` for invalid inputs, that exception bubbles up unwrapped to the eval CLI — the commit body claims `DeviceNotFound / EPNotDiscovered` are caught at CLI boundaries with remediation hints, but I see no `try/except` here. The catch probably lives in the click/typer command that invokes `evaluate()`, not in `_load_model`.
4. **`config.model_path` branch ignores `ep` entirely.** Same as before, but worth noting: the skip-build path also takes `ep_device` now, so a pre-built ONNX will still be wrapped with the resolved (EP, device) for inference. That is the desired behavior.

## Open questions / TODOs surfaced

- Should `WinMLEvaluationConfig` grow an `ep: str | None = None` field for symmetry with `perf` and `compile`? The commit message lists eval among CLI boundaries migrated to ep_device but stops short of giving eval the `--ep` flag.
- Where exactly is `DeviceNotFound` caught for `winml eval`? Not in this file — needs grep at the click command level.
- Is there a test covering `_load_model` with the new EP-device resolution? The commit reports "~720 passing" but doesn't enumerate eval-specific coverage of the EPDevice migration.
