# src/winml/modelkit/utils/cli.py

## TL;DR
Switches the `--ep` / `--device` Click choices from the local `utils/constants.py` allowlists (`ALL_EP_NAMES`, `SUPPORTED_DEVICES`, `SUPPORTED_DEVICES_WITH_AUTO`) to the session facade's catalog-derived `VALID_EPS` / `VALID_DEVICES`. Fixes the casing bug — `--device` now accepts lowercase `cpu/gpu/npu` (previously case-sensitive uppercase only) — and gives both options a single source of truth (the `EPDeviceSpec` catalog). The `--device` default is normalised to `"npu"` (lowercase), help text updated to mirror, and `show_default=True` dropped from `--device` because the default is now informative in the help string itself. No new options, no decorator-signature breakage for callers.

## Diff metrics
- Lines: +14 / -11 (net +3 logical, mostly comment + helper extraction)
- Hunks: 4 (imports, two new module-level helpers, `ep_option` choice change, `device_option` body change)
- New module-level symbols: `_DEVICE_CHOICES`, `_EP_CHOICES`. Both private.

## Role before vs after
- Before: Click decorator factory pulling allowlists from `utils/constants.py`. Two issues: (1) `SUPPORTED_DEVICES = ["CPU", "GPU", "NPU"]` was uppercase, so `--device cpu` was rejected (only `--device CPU` worked) — a long-standing latent bug. (2) `ALL_EP_NAMES` was a union of `SUPPORTED_EPS` (full names) + `EP_ALIASES.keys()` (short names) — both representations accepted as input.
- After: Same Click decorator factory, but sourcing choices from `session.VALID_EPS` / `session.VALID_DEVICES`. The casing bug is fixed (case-insensitive + the underlying set is lowercase). Full ExecutionProvider names (e.g. `QNNExecutionProvider`) are *no longer* in the choice set — only short names. Calls passing the full name now fail at click parse-time. This is the "Option A hard break" the commit body advertises.

## Symbol-level changes
- Removed import: `from .constants import ALL_EP_NAMES, SUPPORTED_DEVICES, SUPPORTED_DEVICES_WITH_AUTO`.
- Added import: `from ..session import VALID_DEVICES, VALID_EPS`.
- Added module-level constants (with explanatory comments):
  - `_DEVICE_CHOICES = sorted(VALID_DEVICES)` — sorted lowercase device strings.
  - `_EP_CHOICES = sorted(VALID_EPS)` — sorted short EP names from the catalog.
  Comment marks the previous `SUPPORTED_DEVICES = ["CPU","GPU","NPU"]` shape as a bug.
- `ep_option(...)`:
  - Body: `click.Choice(ALL_EP_NAMES, case_sensitive=False)` → `click.Choice(_EP_CHOICES, case_sensitive=False)`.
- `device_option(...)`:
  - Default value: `default="NPU"` → `default="npu"` (lowercase, matches new VALID_DEVICES).
  - Help-text `optional_message` updated to mirror lowercase (`"uses NPU as default"` → `"uses npu as default"`).
  - Choice computation: `choices = SUPPORTED_DEVICES_WITH_AUTO if include_auto else SUPPORTED_DEVICES` → `choices = [*_DEVICE_CHOICES, "auto"] if include_auto else _DEVICE_CHOICES`. The "auto" sentinel is appended on the fly.
  - `case_sensitive=False` — implicit through the `click.Choice` defaulting + the lowercase choice set.
  - `show_default=True` removed (was previously on the Click option). Default is now reflected in `help` text instead.
  - New docstring line: `"auto" defers device selection to runtime via auto_detect_device()`.

## Behavior / contract changes
- **`--ep` choice set changes shape.** Old `ALL_EP_NAMES` was `["QNNExecutionProvider", "OpenVINOExecutionProvider", "VitisAIExecutionProvider", "qnn", "openvino", "ov", "vitisai", "vitis"]` (with one or two more depending on which EPs were registered). New `_EP_CHOICES = sorted(VALID_EPS)` is short names only — the canonical short forms from the catalog (`"qnn"`, `"openvino"`, `"vitisai"`, `"migraphx"`, `"nvtensorrtrtx"`, `"cuda"`, `"tensorrt"`, `"dml"`, `"cpu"`, etc.). Consequences:
  1. Full-name forms are no longer accepted by Click — calls that previously passed `QNNExecutionProvider` now fail with `"Invalid value for '--ep': ..."`.
  2. Aliases `ov` and `vitis` (which were *only* in the constants.py `EP_ALIASES`, not in the session catalog) are no longer in the choice set. Calls passing `--ep ov` now fail. The commit's `normalize_ep_name` migration in `utils/constants.py` still handles these post-parse, but click rejects them before parse — so the migration is incomplete here. Verify against `EpAtSourceParamType` in `commands/_cli_helpers.py` (or wherever the `@source` parsing now lives); it may have its own normalization layer that doesn't go through `ep_option`.
- **`--device` is now case-insensitive.** A long-standing bug where `--device cpu` was rejected is fixed. `--device CPU`, `--device cpu`, `--device Cpu` all work.
- **`--device` default normalized to lowercase.** Same effective value, but the type is now consistent with the rest of the codebase.
- **`SUPPORTED_DEVICES_WITH_AUTO` no longer used here.** The constants.py companion file deleted that symbol; this file now reconstructs the `[*_DEVICE_CHOICES, "auto"]` set inline. The decorator's `include_auto` flag still drives whether "auto" is in the choices.

## Cross-file impact
- New dependency on `session.VALID_EPS` and `session.VALID_DEVICES`. These are exported from `session/__init__.py` (verified in the prior batch's `session____init__.md`).
- Consumers of `ep_option` and `device_option` are the CLI command modules (`commands/perf.py`, `commands/build.py`, `commands/compile.py`, `commands/config.py`, `commands/eval.py`). The `--ep` short-name-only constraint means any documentation snippet showing `--ep QNNExecutionProvider` is now stale.
- Tests that exercise the CLI parse step (`tests/unit/commands/test_cli.py`) should have been updated to use short names; verify.
- Removed dependency on `utils/constants.py::ALL_EP_NAMES`, `SUPPORTED_DEVICES`, `SUPPORTED_DEVICES_WITH_AUTO`. The first three are deleted in this commit (see `utils__constants.md`).

## Risks / subtleties
- **Hard break on full EP names at click parse time.** Documented above; non-obvious to a user who learned the old syntax. The error message Click produces is generic (`Invalid value for '--ep'`) and doesn't direct the user to the new short form. A `EpAtSourceParamType` (mentioned in the commit body) handles the `--ep <name>[@<source-tag>]` syntax — verify it's used here, not raw `click.Choice`, or the `@source-tag` form is rejected too.
- **`show_default=True` removal.** Click previously appended `[default: NPU]` to the help string. Now the user has to read the help text manually. Functionally fine but a UX regression for the `--device` option.
- **`_DEVICE_CHOICES` and `_EP_CHOICES` are evaluated at import time.** If `session.VALID_EPS` or `session.VALID_DEVICES` change shape (e.g. the catalog grows a new EP), they need to be re-importable for the change to land. Today they're frozensets, so the snapshot is durable for the process lifetime — that's fine.
- **`case_sensitive=False` retained on `ep_option`, but the underlying choice set is already lowercase.** The flag is redundant. Same for `device_option`. Not harmful, but a future cleanup item.
- **The "auto" sentinel is hardcoded inline.** If a future caller wanted a third sentinel (e.g. `"none"`), they'd have to add another `include_X` kwarg. A more general `extra_choices: list[str] | None` would be cleaner.

## Simplification opportunities
- **Inline `_DEVICE_CHOICES` and `_EP_CHOICES` into their call sites**, or hoist them onto the session facade itself as `session.DEVICE_CLI_CHOICES` / `session.EP_CLI_CHOICES`. The current shape (two module-private sorted views) is fine but adds a layer of indirection.
- **Drop the redundant `case_sensitive=False`** since the choice sets are already lowercase. Documentation noise that may mislead.
- **Re-add a `--device` default to the help string** explicitly (e.g. `"Target device type (cpu, gpu, npu) [default: npu]"`) to compensate for the dropped `show_default=True`.
- **`device_option(default="npu")` is duplicated as a string literal at the def signature *and* as the help message.** Wire the default to a single source via `default_str = default if not include_auto else f"{default}"` — small but symmetric with the choice computation.
- **The two module-level constants could live as `@functools.cache`-decorated helpers** if there's any concern about session import cost at click decorator definition time. Today this is import-evaluated; cached lazy evaluation would defer to first use.

## Open questions / TODOs surfaced
- Does the `@source-tag` syntax (`--ep qnn@msix`) still work? `click.Choice` rejects values it doesn't recognize verbatim. The commit body mentions `EpAtSourceParamType` is added — but if `ep_option` here still uses `click.Choice`, the `@source` syntax is rejected at parse. Either this file should switch to the param type, or there's a parser-precedence story I'm missing.
- Should there be a deprecation translation layer for `--ep QNNExecutionProvider`? Today users get a generic Click error; a one-line "use --ep qnn instead" hint would be friendlier.
- Should `show_default=True` be restored for `--device` to match `--ep`'s behavior?
