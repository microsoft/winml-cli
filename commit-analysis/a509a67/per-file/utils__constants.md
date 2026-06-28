# src/winml/modelkit/utils/constants.py

## TL;DR

Stripped the EP/device taxonomy duplication out of this file: `SUPPORTED_EPS`, `EP_ALIASES`, `ALL_EP_NAMES`, and `SUPPORTED_DEVICES` are deleted, and `normalize_ep_name` now delegates to the session facade's `expand_ep_name` (which reads from the EPDeviceSpec catalog). The file's role shrinks from "secondary EP/device allowlist" to "thin alias-normaliser + ORT device-type enum bridges". A small private tuple `_EP_CLI_PREFIXES` is kept locally to drive `extract_ep_options`'s `prefix_param` parsing.

## Diff metrics

`+18 / -38` (net `-20`). One large hunk removes the four list/dict constants and rewrites `normalize_ep_name`; one small hunk rewrites `extract_ep_options`'s prefix source; one final hunk deletes `SUPPORTED_DEVICES` while leaving the ORT device-type enum maps intact.

## Role before vs after

Before — co-equal second source of truth: held its own `SUPPORTED_EPS` / `EP_ALIASES` / `SUPPORTED_DEVICES` tables and its own `normalize_ep_name` lookup logic. Risked drift versus session-side mappings.

After — derived/delegate layer: `normalize_ep_name` is a thin shim over `session.expand_ep_name`; the only taxonomy-shaped constant retained is `_EP_CLI_PREFIXES`, kept deliberately local because it is CLI-keyword-parsing surface (not part of the runtime EP catalog) — comment makes this carve-out explicit. The two ORT-enum bridge dicts (`DEVICE_TO_DEVICE_TYPE`, `DEVICE_TYPE_TO_DEVICE`) stayed because they bind `ort.OrtHardwareDeviceType` to the canonical uppercase device strings and are not duplicated anywhere in the session package.

## Symbol-level changes

Deleted (top-level):

- `SUPPORTED_EPS: list[str]` — the three full ExecutionProvider names. Catalog now owns the canonical list (`EP_DEVICE_SPECS` → `VALID_EPS`).
- `EP_ALIASES: dict[str, str]` — shorthand → full mapping. Replaced by `session.ep_device._SHORT_TO_FULL` (which is broader: also covers `migraphx`, `nv_tensorrt_rtx`, `cuda`, `tensorrt`, `dml`, `cpu` — the latter three were already added in this commit for the deduction paths).
- `ALL_EP_NAMES: list[str]` — union of full names + aliases. Removed; `utils/cli.py` now uses `session.VALID_EPS` directly.
- `SUPPORTED_DEVICES: list[str]` (uppercase `["CPU","GPU","NPU"]`) — removed. `utils/cli.py` now uses `session.VALID_DEVICES` (lowercase frozenset).

Added (top-level):

- Import `from ..session import expand_ep_name` — establishes the new dependency direction (utils → session, never the reverse).
- `_EP_CLI_PREFIXES: tuple[str, ...] = ("qnn", "openvino", "ov", "vitisai", "vitis")` — private, not exported, used only by `extract_ep_options`. Comment explicitly states "not exported; does not duplicate the session taxonomy" — this is the part of the old `EP_ALIASES` that earns its keep because it has to match CLI option keyword prefixes like `qnn_qairt`, `ov_device_type`, etc. Notably includes the legacy two-letter `ov` and `vitis` (still parseable from `--qnn_qairt`-style kwargs), even though `utils/cli.py`'s `_EP_CHOICES` no longer accepts them as `--ep` values.

Rewritten:

- `normalize_ep_name(ep)`:
  - Old: in-house lookup against `SUPPORTED_EPS` (case-sensitive full-name pass-through) → `EP_ALIASES` (case-insensitive) → pass-through fallback.
  - New: special-cases the two legacy two-letter aliases (`ov` → `openvino`, `vitis` → `vitisai`) and then delegates everything to `expand_ep_name`. The function preserves the `None → None` contract and keeps the same docstring examples (`"qnn"`, `"ov"`, full name). One semantic shift: under the old impl, an unknown EP string would be returned as-is; under the new impl it goes through `expand_ep_name` → `canonicalize_ep_name`, which only normalises *casing for known full names*, otherwise passes through unchanged. So the visible behaviour for unknown EP strings is the same (passthrough), but with the side-benefit that mis-cased canonical names like `nvtensorrtrtxexecutionprovider` will now be auto-fixed to `NvTensorRtRtxExecutionProvider`.

- `extract_ep_options(kwargs)`:
  - Body change: `ep_aliases = list(EP_ALIASES.keys())` → reads `_EP_CLI_PREFIXES` directly.
  - Quoting style normalised to double-quoted (cosmetic; matches ruff's preferred style).
  - Behaviour preserved: same `(prefix, suffix) = name.split("_", 1)` filter, same `str(value)` coercion, same `parts[0] in <prefix-set>` gate.

Untouched:

- `DEVICE_TO_DEVICE_TYPE: dict[str, ort.OrtHardwareDeviceType]` (keys still uppercase `"CPU"`/`"GPU"`/`"NPU"`).
- `DEVICE_TYPE_TO_DEVICE: dict[ort.OrtHardwareDeviceType, str]` (values still uppercase).
- The `import onnxruntime as ort` at the top of the file (kept for the two enum bridge dicts).

## Behavior / contract changes

- `normalize_ep_name` now recognises more short names than before. Anything in `session.ep_device._SHORT_TO_FULL` is normalised — that includes `cuda`, `tensorrt`, `dml`, `cpu`, `migraphx`, `nv_tensorrt_rtx` in addition to the original three. Callers that built on the assumption "only `qnn/openvino/ov/vitisai/vitis` get expanded, everything else is left raw" will now see expanded forms for the additional shorts.
- `extract_ep_options` is structurally identical but no longer depends on a public `EP_ALIASES` table — direct or test-time monkeypatching of `EP_ALIASES` is dead.
- `SUPPORTED_DEVICES` deletion is a hard public-API break for any external code that imported it. Importers in this codebase: only `utils/cli.py`, which migrated. Anything outside the package that did `from winml.modelkit.utils.constants import SUPPORTED_DEVICES` will now `ImportError`. Same for `SUPPORTED_EPS`, `EP_ALIASES`, `ALL_EP_NAMES`.
- The case-sensitivity of `normalize_ep_name` changes subtly: the old implementation accepted full-name exact-case (`"QNNExecutionProvider"`) and then case-insensitively looked up aliases. The new implementation lowercases `ep` before checking `_legacy`, then forwards to `expand_ep_name`, which lowercases for `_SHORT_TO_FULL` and falls through to `canonicalize_ep_name` (which only fixes the one casing-bug entry, `NvTensorRtRtxExecutionProvider`). Net: full-name input flows through unchanged (with the casing fix), and short/alias inputs are case-insensitive — same observable behaviour, but the implementation now goes through `canonicalize_ep_name` for free.

## Cross-file impact

- This is the consumer-side end of the new `from ..session import expand_ep_name` dependency edge. `utils → session` is now allowed; the inverse direction would create a cycle.
- The commit's Directive in the body explicitly says "do not import private symbols `_EP_TO_DEVICE`, `_DEVICE_TO_PROVIDER`, `_SHORT_TO_FULL` outside `session/ep_device.py` — use the session/ facade and public helpers". This file follows that rule (imports `expand_ep_name`, not `_SHORT_TO_FULL`).
- `_EP_CLI_PREFIXES` lives here on purpose: it is keyword-argument prefix parsing for Click, which is upstream of the catalog. Putting it on the catalog would conflate runtime EP identity with CLI keyword conventions.

## Risks / subtleties

- **Legacy two-letter aliases (`ov`, `vitis`) are now partly orphaned.** They appear in three places now: `_EP_CLI_PREFIXES` (for `--ov_device_type`-style kwargs), `normalize_ep_name`'s `_legacy` dict (for the `normalize_ep_name("ov")` path), but *not* in `utils/cli.py`'s `_EP_CHOICES` — so `--ep ov` is rejected by Click. Inconsistent surface. Either consolidate or document.
- **`DEVICE_TO_DEVICE_TYPE` still uses uppercase keys.** The rest of the codebase migrated to lowercase device strings; this map is one of the few places that still expects `"CPU"`/`"GPU"`/`"NPU"`. Any caller that grabbed `device` out of the new lowercase pipeline needs to `.upper()` before indexing. Not changed in this commit; remaining footgun.
- **`expand_ep_name` is imported at module top-level.** Means importing `utils/constants.py` now eagerly imports `session/ep_device.py`. Light, but worth noting because `utils/constants` was previously a self-contained leaf module. Touches the import-time-budget work covered in `docs/design/importtime/`.
- **No coverage path for the `_legacy` shim.** If `ov`/`vitis` are removed from the codebase entirely in a follow-up, the `_legacy = {"ov": "openvino", "vitis": "vitisai"}` branch becomes dead code without anything failing.

## Open questions / TODOs surfaced

- Should `_EP_CLI_PREFIXES` be folded into the catalog as a per-spec `cli_kwarg_prefixes` attribute, so adding a new EP also wires up its CLI keyword namespace?
- Should the `DEVICE_TO_DEVICE_TYPE` map be moved into the session package and made lowercase-keyed to match the catalog, with this file deleted entirely? Today, this file's only remaining unique content is the ORT-enum bridge and the prefix tuple — both arguably belong elsewhere.
- The `_legacy` two-letter alias map is private and inline. If retained, it should arguably live alongside `_SHORT_TO_FULL` in `session/ep_device.py` so all alias logic is in one place.
