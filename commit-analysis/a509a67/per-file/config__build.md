# src/winml/modelkit/config/build.py

## TL;DR
Three taxonomy import sites are migrated from the deprecated `..sysinfo`/`..config.precision` private functions to the new `..session` and `..sysinfo.hardware` facades. (a) `resolve_quant_compile_config` and `generate_hf_build_config` drop `from ..sysinfo import resolve_device` (which returned `(category, available_devices)`) in favour of two split helpers: `available_devices = get_available_devices()` (from `..sysinfo.hardware`) and `resolved_device = auto_detect_device() if device == "auto" else device.lower()` (from `..session`). The `available_devices` list still flows into `resolve_precision`. (b) The HF auto-mode fallback that previously called `get_provider_for_device(resolved_device)` (a private helper in `config/precision.py`) now derives the same value structurally via `default_ep_for_device(resolved_device)` + `short_ep_name(...)` from the session catalog, with an explicit "cpu → None" mapping. No new fields, no API changes, no signature changes.

## Diff metrics
- 6 LOC changed in two function bodies; all changes are inside local-imported lines inside `resolve_quant_compile_config` and `generate_hf_build_config`. No top-level additions or deletions.
- Net behavior: same return values for non-CPU paths; CPU branch behaviour is preserved by an explicit short-name guard.

## Role before vs after
- **Before:** `config/build.py` reached into two private taxonomy layers — `sysinfo.device.resolve_device(device=...)` for hardware detection, and `config.precision.get_provider_for_device(resolved_device)` for the device→EP mapping. Both functions are gone in this commit (one renamed, one deleted).
- **After:** All taxonomy lookups go through `winml.modelkit.session.*` public helpers. Build-config generation no longer knows about the EPDeviceSpec catalog directly; it just calls the documented session facade.

## Symbol-level changes
- **`resolve_quant_compile_config` (l.252-307):**
  - The previous `from ..sysinfo import resolve_device` (which returned a `(resolved_device, available_devices)` tuple) is replaced with two inline imports: `from ..session import auto_detect_device` and `from ..sysinfo.hardware import get_available_devices`.
  - Call site reconstructs the same pair separately:
    ```python
    available_devices = get_available_devices()
    resolved_device = auto_detect_device() if device == "auto" else device.lower()
    ```
    See `src/winml/modelkit/config/build.py:276-281`. The `available_devices` list still flows into `resolve_precision` unchanged.
  - The unused `get_provider_for_device` import inside this function is also gone in tandem (the function still calls `WinMLCompileConfig.for_provider(policy.compile_provider)` where `policy.compile_provider` is now a short EP name produced by `resolve_precision` — see `config/precision.py`).
- **`generate_hf_build_config` (l.437-652):**
  - First migration site (l.570-577): same fold as above — drops the old `sysinfo.resolve_device` tuple call in favour of `auto_detect_device()` + `get_available_devices()` split. Adjacent log line ("Device resolved: %s ...") is unchanged.
  - Second migration site (l.604-616, the `else` branch covering "auto/auto"): the inline import `from .precision import get_provider_for_device` is replaced with `from ..session import default_ep_for_device, short_ep_name`.
  - Calculation replaced:
    ```
    # was
    hw_provider = get_provider_for_device(resolved_device)

    # now
    _canonical = default_ep_for_device(resolved_device)
    _short = short_ep_name(_canonical) if _canonical is not None else None
    hw_provider = _short if _short != "cpu" else None
    ```
    Reason for the post-filter: `default_ep_for_device("cpu")` returns the canonical `"CPUExecutionProvider"` (i.e. it is *not* None for CPU), so a naïve translation would change behaviour. The explicit `if _short != "cpu" else None` re-establishes the old semantic where "CPU resolves to no compile stage".

## Behavior / contract changes
- **Function signatures unchanged.** `resolve_quant_compile_config`, `generate_hf_build_config`, `generate_onnx_build_config`, `generate_build_config` keep the same parameters and return types.
- **Public return types unchanged:** `WinMLCompileConfig.for_provider(short_ep_name)` is still expected to accept the same short EP strings (`"qnn"`, `"dml"`, …); `policy.compile_provider` was already a short name pre-commit but its derivation moved (see `precision.py`).
- **CPU auto-mode parity preserved.** Without the `if _short != "cpu" else None` guard, `WinMLCompileConfig.for_provider("cpu")` would be called and would produce a non-None compile stage (the `for_cpu` factory returns a config with `enable_ep_context=False`); the pre-commit code path skipped the assignment entirely (`hw_provider is None`), leaving the *default* `parent_config.compile`. The new code preserves the "leave the default in place" behaviour for CPU-only machines.
- **Symbol surfaces:** the original `sysinfo.resolve_device` (returning `(category, available_devices)`) is gone. In its place this file now uses two helpers with disjoint responsibilities: `session.auto_detect_device()` (returns the lowercase str category) and `sysinfo.hardware.get_available_devices()` (returns the available-devices list). The session module also has a typed `resolve_device(ep, device)` that returns an `EPDevice` — but this file does not use it, since the build pipeline only needs the str category and the available-devices list, not an `EPDevice` descriptor.

## Cross-file impact
- Consumers of `resolve_quant_compile_config` / `generate_hf_build_config` / `generate_build_config` see no change (same signatures, same returns).
- Inside the session package, this file is now a *consumer* of `auto_detect_device`, `default_ep_for_device`, `short_ep_name` (all confirmed in `winml.modelkit.session.__init__.__all__`), plus `get_available_devices` from `winml.modelkit.sysinfo.hardware`.
- The deleted symbol `config.precision.get_provider_for_device` is no longer referenced from this file; the sibling deletion in `precision.py` is consistent.

## Risks / subtleties
- **No tuple-unpack footgun.** Earlier drafts of this refactor had two same-named `resolve_device` functions (one tuple-returning, one `EPDevice`-returning) and relied on the build module to use the right one. The final design splits responsibilities by name — `auto_detect_device()` returns a single `str` and `get_available_devices()` returns the list — so there is no shape-overload risk anymore.
- **The "cpu" string sentinel is fragile.** `short_ep_name("CPUExecutionProvider") == "cpu"` is assumed here; if the short-name convention ever changes (e.g. uppercase, or returns `None` for CPU), this branch will silently start to invoke the `for_cpu` factory and emit a compile stage where none is wanted. Worth a regression test.
- **`default_ep_for_device(category)` returning `None` is treated as "no EP for this category"** — i.e. the user is on a machine where no compile is meaningful. The chain `_canonical → _short → hw_provider` survives that case via three sequential None-guards.
- **Local imports are kept inline** inside both functions (matching the pre-existing style in this file), so the session module is only loaded when these code paths actually fire. This avoids forcing an onnxruntime import for read-only consumers of `WinMLBuildConfig`.
- **`policy.compile_provider` shape contract.** This file passes `policy.compile_provider` directly to `WinMLCompileConfig.for_provider(...)`. The commit changes `policy.compile_provider`'s derivation (see `config/precision.py`) — it's still a short string per the dataclass docstring update, but the lineage now goes via `short_ep_name(default_ep_for_device(...))`, so a future refactor of `short_ep_name` could change the shape unexpectedly.

## Open questions / TODOs surfaced
- The taxonomy imports (`from ..session import auto_detect_device` and `from ..sysinfo.hardware import get_available_devices`) are kept inline inside the two functions rather than hoisted to module-level — preserving the existing lightweight-import discipline (note the existing comment about circular import with `models.hf`).
- The "cpu" guard duplicates logic that arguably belongs in `default_ep_for_device` itself — should `default_ep_for_device("cpu")` return `None`? It does not today, by deliberate choice. The duplication of `if _short != "cpu" else None` between this file and `config/precision.py` is a code smell.
- No new test surface added in this file by the diff alone; the contract changes (function rename) rely on consumer tests at the CLI / build pipeline level.
