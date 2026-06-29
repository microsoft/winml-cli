# src/winml/modelkit/config/build.py

## TL;DR

Three taxonomy import sites are migrated from the deprecated `..sysinfo`/`..config.precision` private functions to the new `..session` and `..sysinfo.hardware` facades.

(a) `resolve_quant_compile_config` and `generate_hf_build_config` drop `from ..sysinfo import resolve_device` (which returned `(category, available_devices)`) in favour of two split helpers: `available_devices = get_available_devices()` (from `..sysinfo.hardware`) and `resolved_device = auto_detect_device() if device.lower() == "auto" else device.lower()` (from `..session`). The `available_devices` list still flows into `resolve_precision`.

(b) The HF auto-mode fallback that previously called `get_provider_for_device(resolved_device)` (a private helper in `config/precision.py`) now derives the same value structurally via `default_ep_for_device(resolved_device)` + `short_ep_name(...)` from the session catalog, with an explicit "cpu → None" mapping.

No new fields, no API changes, no signature changes. The CLI surface is preserved.

## Diff metrics

- Lines changed: 18 total (per `git show --stat`); 6 LOC of substantive edits across two functions plus context.
- Two inline-import sites refactored (one in `resolve_quant_compile_config` at l.276-278, one in `generate_hf_build_config` at l.570-572).
- One inline-import site replaced (the `else` branch fallback in `generate_hf_build_config` at l.610).
- No top-level imports added or removed.
- No symbols added or removed at module scope.

## Role before vs after

- **Before (parent `7a66c024`):** `config/build.py` reached into two private taxonomy layers — `sysinfo.device.resolve_device(device=...)` for hardware detection (returning a `(resolved_device, available_devices)` tuple), and `config.precision.get_provider_for_device(resolved_device)` for the device→EP mapping. Both functions are gone in this commit (one renamed/split, one deleted).
- **After:** All taxonomy lookups go through `winml.modelkit.session.*` public helpers. Build-config generation no longer knows about the EPDeviceSpec catalog directly; it just calls the documented session facade. The split responsibilities (`auto_detect_device()` returns a single str category, `get_available_devices()` returns the list) replace the tuple-returning function.

## Symbol-level changes

- **`resolve_quant_compile_config` (l.252-309):**
  - The previous `from ..sysinfo import resolve_device` (which returned a `(resolved_device, available_devices)` tuple) is replaced with two inline imports: `from ..session import auto_detect_device` and `from ..sysinfo.hardware import get_available_devices`.
  - Call site reconstructs the same pair separately:
    ```python
    available_devices = get_available_devices()
    resolved_device = auto_detect_device() if device.lower() == "auto" else device.lower()
    ```
    See `src/winml/modelkit/config/build.py:276-281`. The `available_devices` list still flows into `resolve_precision` unchanged.
  - Note the `.lower()` on the LHS of the ternary: the `device` parameter could legitimately arrive as `"AUTO"` (from non-click callers); the prior `== "auto"` check would miss the case. Both branches of the ternary now normalise.

- **`generate_hf_build_config` (l.437-652):**
  - **First migration site (l.570-577):** same fold as above — drops the old `sysinfo.resolve_device` tuple call in favour of `auto_detect_device()` + `get_available_devices()` split. Adjacent log line ("Device resolved: %s ...") unchanged.
  - **Second migration site (l.604-616, the `else` branch covering "auto/auto"):** the inline import `from .precision import get_provider_for_device` is replaced with `from ..session import default_ep_for_device, short_ep_name`.
  - Calculation replaced:
    ```python
    # before (parent 7a66c024)
    hw_provider = get_provider_for_device(resolved_device)

    # after (774f121d)
    _canonical = default_ep_for_device(resolved_device)
    _short = short_ep_name(_canonical) if _canonical is not None else None
    hw_provider = _short if _short != "cpu" else None
    ```
    Reason for the post-filter: `default_ep_for_device("cpu")` returns the canonical `"CPUExecutionProvider"` (i.e. it is *not* None for CPU), so a naïve translation would change behaviour. The explicit `if _short != "cpu" else None` re-establishes the old semantic where "CPU resolves to no compile stage".

- **No public-surface changes.** `resolve_quant_compile_config`, `generate_hf_build_config`, `generate_onnx_build_config`, `generate_build_config` keep identical signatures.

## Behavior / contract changes

- **Function signatures unchanged.** Same parameters, same return types.
- **Public return types unchanged:** `WinMLCompileConfig.for_provider(short_ep_name)` is still expected to accept the same short EP strings (`"qnn"`, `"dml"`, …); `policy.compile_provider` was already a short name pre-commit but its derivation moved (see `precision.py`).
- **CPU auto-mode parity preserved.** Without the `if _short != "cpu" else None` guard, `WinMLCompileConfig.for_provider("cpu")` would be called and would produce a non-None compile stage; the pre-commit code path skipped the assignment entirely (`hw_provider is None`), leaving the *default* `parent_config.compile`. The new code preserves the "leave the default in place" behaviour for CPU-only machines.
- **`"AUTO"` / `"Auto"` now resolve correctly.** Previously `device="AUTO"` would have failed `== "auto"` and fed `resolved_device = "auto"` into downstream code — which `resolve_precision` would treat as a literal device class and reject. Post-commit, both branches normalize through `.lower()`.
- **Headless-server safety (transitive).** `default_ep_for_device` now catches `RuntimeError` from `EP_CATALOG.is_compatible` (per commit body), so on a server without GPU/NPU vendor detection the `hw_provider` chain returns `None` cleanly instead of bubbling a WMI traceback through `_canonical = default_ep_for_device(...)`. This file is a transitive beneficiary, not the locus of the change.
- **Symbol surfaces:** the original `sysinfo.resolve_device` (returning `(category, available_devices)`) is gone. In its place this file now uses two helpers with disjoint responsibilities: `session.auto_detect_device()` and `sysinfo.hardware.get_available_devices()`. The session module also has a typed `resolve_device(EPDeviceTarget) -> EPDeviceTarget` — but this file does not use it, since the build pipeline only needs the str category and the available-devices list, not an `EPDeviceTarget`.

## Cross-file impact

- Consumers of `resolve_quant_compile_config` / `generate_hf_build_config` / `generate_build_config` see no change (same signatures, same returns).
- Inside the session package, this file is now a *consumer* of `auto_detect_device`, `default_ep_for_device`, `short_ep_name` (all confirmed in `winml.modelkit.session.__init__.__all__` at l.66, 68, 74), plus `get_available_devices` from `winml.modelkit.sysinfo.hardware`.
- The deleted symbol `config.precision.get_provider_for_device` is no longer referenced from this file; the sibling deletion in `precision.py` is consistent.
- A grep for `from ..sysinfo import resolve_device` across the codebase should now be empty — this file was the canonical consumer. (Other callers of the tuple-returning form would break, but the commit removes the producer too.)

## Risks / subtleties

- **No tuple-unpack footgun.** The new design splits responsibilities by name — `auto_detect_device()` returns a single `str` and `get_available_devices()` returns the list — so there is no shape-overload risk between the two `resolve_device` signatures (the session-module's `EPDeviceTarget`-returning `resolve_device` is not used here).
- **The "cpu" string sentinel is fragile.** `short_ep_name("CPUExecutionProvider") == "cpu"` is assumed here; if the short-name convention ever changes (e.g. uppercase, or returns `None` for CPU), this branch will silently start to invoke the `for_provider("cpu")` factory and emit a compile stage where none is wanted. Worth a regression test.
- **`default_ep_for_device(category)` returning `None` is treated as "no EP for this category"** — i.e. the user is on a machine where no compile is meaningful. The chain `_canonical → _short → hw_provider` survives that case via three sequential None-guards.
- **Local imports are kept inline** inside both functions (matching the pre-existing style in this file), so the session module is only loaded when these code paths actually fire. This avoids forcing an onnxruntime import for read-only consumers of `WinMLBuildConfig`. There's a pre-existing module-level comment about a circular import with `models.hf` that justifies this discipline.
- **`policy.compile_provider` shape contract.** This file passes `policy.compile_provider` directly to `WinMLCompileConfig.for_provider(...)`. The commit changes `policy.compile_provider`'s derivation (see `config/precision.py`) — it's still a short string per the dataclass docstring update, but the lineage now goes via `short_ep_name(default_ep_for_device(...))`, so a future refactor of `short_ep_name` could change the shape unexpectedly.
- **Duplicated `.lower()` calls.** Both branches of the ternary call `.lower()`; functionally harmless but a hair redundant. A leading `_dev = device.lower()` would remove the duplication.

## Open questions / TODOs surfaced

- The taxonomy imports (`from ..session import auto_detect_device` and `from ..sysinfo.hardware import get_available_devices`) are kept inline inside the two functions rather than hoisted to module-level — preserving the existing lightweight-import discipline.
- The "cpu" guard duplicates logic that arguably belongs in `default_ep_for_device` itself — should `default_ep_for_device("cpu")` return `None`? It does not today, by deliberate choice. The duplication of `if _short != "cpu" else None` between this file and `config/precision.py` is a code smell.
- No new test surface added in this file by the diff alone; the contract changes (function rename) rely on consumer tests at the CLI / build pipeline level.

## Simplification opportunities

- **Hoist `.lower()` once.** Both call sites read `device.lower() == "auto"` then immediately call `device.lower()` again on the else branch. Local `_dev = device.lower()` removes the duplication and shortens the ternary.
- **Inline-import duplication.** Both `resolve_quant_compile_config` and the first STEP 4.5 block in `generate_hf_build_config` inline-import `auto_detect_device` and `get_available_devices` with identical lines. A module-private helper `_resolve_device_and_inventory(device: str) -> tuple[str, list[str]]` would consolidate ~8 LOC. Tradeoff: the original inline-import pattern was deliberate to keep the import graph lazy. A module-private helper at the bottom of the file would keep that property.
- **Single-call-site `_short != "cpu"` block.** This pattern occurs in both `config/build.py` and `config/precision.py`. Best simplification is structural in `session/ep_device.py` — give CPUExecutionProvider a marker (`no_compile: bool = True` on `EPDeviceSpec`) — but that's out of scope for this file.
- **The "auto/auto" `else` branch in `generate_hf_build_config` (l.607-616) is a 6-line dance** for what could be a one-call `compile_provider_for_device(device)` helper in the session facade. Three local underscored names (`_canonical`, `_short`, `hw_provider`) hint that the chain is doing too much arithmetic at the call site.
- **Two ternaries with the same RHS.** `device.lower() if device.lower() == "auto" else device.lower()` could be replaced with the call form: `auto_detect_device() if device.lower() == "auto" else device.lower()` — which is what's there — but the redundant `.lower()` is the noisy part.
