# src/winml/modelkit/utils/constants.py

## TL;DR
Deletes the legacy second source of truth (`SUPPORTED_EPS`, `EP_ALIASES`, `ALL_EP_NAMES`, `SUPPORTED_DEVICES`, `SUPPORTED_DEVICES_WITH_AUTO`) and the local `normalize_ep_name` lookup. The file now delegates to `session.expand_ep_name` for the canonical EP-name resolution. `normalize_ep_name` keeps its public signature but its body shrinks to a small "non-canonical short → canonical short" alias table (`ov` → `openvino`, `vitis` → `vitisai`, `nv_tensorrt_rtx` → `nvtensorrtrtx`) plus a single call into the session facade. `extract_ep_options` keeps its name but sources its prefix list from a module-private `_EP_CLI_PREFIXES` tuple instead of `EP_ALIASES.keys()`. The ORT-enum bridge maps (`DEVICE_TO_DEVICE_TYPE`, `DEVICE_TYPE_TO_DEVICE`) are untouched and remain uppercase-keyed — a known casing-mismatch footgun deferred to a follow-up.

## Diff metrics
- Lines: `+25 / -43` (net `-18`)
- Hunks: 3 (imports + EP-knowledge block rewritten; `extract_ep_options` rebound to `_EP_CLI_PREFIXES`; `SUPPORTED_DEVICES` block deleted)
- Symbols removed: 4 module-level (`SUPPORTED_EPS`, `EP_ALIASES`, `ALL_EP_NAMES`, `SUPPORTED_DEVICES`, `SUPPORTED_DEVICES_WITH_AUTO`), 1 private helper (`_get_supported_eps`).
- Symbols added: 1 module-private constant (`_EP_CLI_PREFIXES`).

## Role before vs after
- Before: a co-equal second source of truth for EP/device knowledge. Held its own `SUPPORTED_EPS` (3 ExecutionProvider full names), `EP_ALIASES` (9 shorthand mappings), and `SUPPORTED_DEVICES` (uppercase list) with a local `normalize_ep_name` doing a case-insensitive lookup. The `_get_supported_eps` helper called into `sysinfo.device.get_ep_device_map` — now-deleted — to derive the EP list. Three risks: drift versus the session-side mapping; inconsistent uppercase / lowercase device conventions; and a circular dep with the soon-to-be-deleted `sysinfo.device`.
- After: a thin translation layer that defers to `session.expand_ep_name` for the canonical mapping. The only remaining locally-owned table is `_EP_CLI_PREFIXES` — used by `extract_ep_options` to recognize EP-prefixed CLI kwargs like `qnn_qairt`, `ov_*`, `vitis_*`. The file's responsibility shrinks to "CLI-shaped helpers" (normalization + kwarg extraction) plus the ORT-enum bridge.

## Symbol-level changes
- Removed: `def _get_supported_eps()` helper (called the now-deleted `from ..sysinfo.device import get_ep_device_map`). Importing this private function out-of-tree would now break.
- Removed: `SUPPORTED_EPS: list[str]` — the three full ExecutionProvider names. Catalog now owns the canonical list (`EP_DEVICE_SPECS` → `VALID_EPS`).
- Removed: `EP_ALIASES: dict[str, str]` — shorthand → full mapping. Replaced by `session.ep_device._SHORT_TO_FULL` (which is broader: also covers `migraphx`, `nv_tensorrt_rtx`, `cuda`, `tensorrt`, `dml`, `cpu`).
- Removed: `ALL_EP_NAMES: list[str]` — union of full names + aliases. Removed; `utils/cli.py` now uses `session.VALID_EPS` directly.
- Removed: `SUPPORTED_DEVICES: list[str]` (uppercase `["CPU","GPU","NPU"]`). `utils/cli.py` now uses `session.VALID_DEVICES` (lowercase frozenset).
- Removed: `SUPPORTED_DEVICES_WITH_AUTO = ["auto", "cpu", "gpu", "npu"]`. Inline-reconstructed by `utils/cli.py` as `[*_DEVICE_CHOICES, "auto"]`.
- Added: import `from ..session import expand_ep_name`.
- Added: `_EP_CLI_PREFIXES: tuple[str, ...] = ("qnn", "openvino", "ov", "vitisai", "vitis")`. Module-private (single underscore). Drives `extract_ep_options`'s prefix-match. Comment marks it as not duplicating the session taxonomy.
- `normalize_ep_name(ep)` rewritten:
  - Old: in-house lookup against `SUPPORTED_EPS` (case-sensitive full-name pass-through) → `EP_ALIASES` (case-insensitive) → pass-through fallback.
  - New: builds an inline `_short_aliases` dict (`ov` → `openvino`, `vitis` → `vitisai`, `nv_tensorrt_rtx` → `nvtensorrtrtx`) covering the non-canonical short-form spellings, applies the rewrite, then delegates to `expand_ep_name(ep)` for canonical expansion.
  - Signature, return type, and `None`-passthrough unchanged.
- `extract_ep_options(kwargs)`:
  - Body: `ep_aliases = list(EP_ALIASES.keys())` → reads `_EP_CLI_PREFIXES` directly.
  - Otherwise unchanged.
- Untouched: `DEVICE_TO_DEVICE_TYPE` and `DEVICE_TYPE_TO_DEVICE` (the ORT-enum ↔ uppercase-device-string bridges). Both still use `"CPU"`/`"GPU"`/`"NPU"` keys; the rest of the codebase moved to lowercase, but this map's call sites still expect uppercase. Known footgun.

## Behavior / contract changes
- `normalize_ep_name` is now broader. Old code only recognized the 9 entries in `EP_ALIASES`; new code recognizes all entries in `session._SHORT_TO_FULL` (which adds `migraphx`, `cuda`, `tensorrt`, `dml`, `cpu`, etc.). Calls that previously fell into the pass-through branch (e.g. `normalize_ep_name("migraphx")`) now correctly return the canonical full name.
- The `nv_tensorrt_rtx` short form is supported (mapped to `nvtensorrtrtx` then expanded by `expand_ep_name`). The casing bug (`NvTensorRTRTX` → `NvTensorRtRtx`) is fixed at the catalog layer.
- `extract_ep_options` is structurally identical but no longer depends on a public `EP_ALIASES` table. Direct or test-time monkeypatching of `EP_ALIASES` is dead — tests that did so will silently still pass for unrelated reasons but no longer affect behavior.
- `SUPPORTED_DEVICES` deletion is a hard public-API break for any external code that imported it. Importers in this codebase: only `utils/cli.py`, which migrated. Anything outside the package that did `from winml.modelkit.utils.constants import SUPPORTED_DEVICES` will now `ImportError`. Same for `SUPPORTED_EPS`, `EP_ALIASES`, `ALL_EP_NAMES`.
- `normalize_ep_name(unknown_string)` behavior changed: old code returned the input unchanged ("let validation catch invalid names"). New code delegates to `expand_ep_name`, which **may raise** (`KeyError` or `ValueError` depending on its implementation) for unknown input. Callers downstream of `normalize_ep_name` may now see exceptions where they previously got pass-through strings. Verify against `session/ep_device.py::expand_ep_name`.

## Cross-file impact
- Direct consumers in `src/` that previously imported removed names — `utils/cli.py` was the primary in-tree consumer; migrated in the same commit. Out-of-tree consumers will break on import.
- `_get_supported_eps`'s call into `sysinfo.device.get_ep_device_map` was the only remaining importer of `sysinfo.device`. Removing it un-blocks the deletion of `sysinfo/device.py`.
- `DEVICE_TO_DEVICE_TYPE` / `DEVICE_TYPE_TO_DEVICE` consumers: `analyze/runtime_checker/check_ops.py`, `analyze/pattern/check_patterns.py`, `analyze/core/runtime_checker_query.py`, `utils/cli.py` (transitively). All still expect uppercase keys; any new lowercase-pipeline caller must `.upper()` before indexing.

## Risks / subtleties
- **`DEVICE_TO_DEVICE_TYPE` still uses uppercase keys.** The rest of the codebase migrated to lowercase device strings; this map is one of the few places that still expects `"CPU"`/`"GPU"`/`"NPU"`. Any caller that grabbed `device` out of the new lowercase pipeline needs to `.upper()` before indexing. Not changed in this commit; remaining footgun.
- **`_short_aliases` is built inside the function on every call.** Cheap but wasteful. A module-level constant would be marginally faster (the function may run thousands of times in test runs).
- **`expand_ep_name` failure modes are now in this function's contract.** Without a docstring update or wrapping `try/except`, callers that previously got a pass-through string for an unknown EP now see whatever exception `expand_ep_name` raises. If a downstream caller had implicit "look up first, fall back to default" logic, it may now crash.
- **`_EP_CLI_PREFIXES` is hardcoded.** If a new EP introduces a CLI prefix (e.g. `migraphx_options`), `extract_ep_options` won't recognize it without an edit here. The list was previously derived from `EP_ALIASES`; now it's a separate manual list with the same drift risk the refactor was meant to eliminate.
- **The comment "Kept as a local tuple — not exported; does not duplicate the session taxonomy"** is mildly misleading. The set of prefixes *does* overlap with session aliases (`qnn`, `openvino`, `ov`, `vitisai`, `vitis`) — it's just narrower (only those with `_*` CLI options). A more accurate comment would say "subset of EP shorts that own dedicated CLI namespaces".

## Simplification opportunities
- **Move `DEVICE_TO_DEVICE_TYPE` / `DEVICE_TYPE_TO_DEVICE` into `session/` and lowercase the keys.** Today they're the only remaining unique content in this file other than the prefix tuple — both arguably belong elsewhere. Doing so would let `utils/constants.py` be deleted entirely.
- **Hoist `_short_aliases` to a module-level constant** with a short doc comment.
- **Inline `extract_ep_options` into `commands/_cli_helpers.py`** (or wherever the EP option plumbing lives). It's one-shot logic that doesn't really fit a "constants" file.
- **Document the `normalize_ep_name` → `expand_ep_name` failure surface**. The migration brought richer behavior but the docstring still reads as if it pass-throughs unknown input.
- **Test the `DEVICE_TO_DEVICE_TYPE` casing footgun** with a unit test that asserts the map is uppercase-keyed and consumers of it know — or migrate it lowercase. Either move beats the silent footgun.

## Open questions / TODOs surfaced
- Should this file be deleted entirely after migrating `DEVICE_TO_DEVICE_TYPE` to the session package? The remaining content is two small helpers — could fit into `commands/_cli_helpers.py` or `session/ep_device.py`.
- Is there an in-tree call site that depends on `normalize_ep_name`'s old pass-through-on-unknown behavior? Worth a quick grep for `normalize_ep_name(` to check call sites pass only known input.
- The pre-existing TODO comment (`# TODO: unify casing with SUPPORTED_DEVICES (uppercase) and DEVICE_TO_DEVICE_TYPE keys`) is *removed* because `SUPPORTED_DEVICES` itself is removed — but the underlying problem (uppercase keys in `DEVICE_TO_DEVICE_TYPE`) is still there. A new TODO should call out the remaining inconsistency.
