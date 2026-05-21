# src/winml/modelkit/utils/cli.py

## TL;DR

Switched the `--ep` / `--device` Click choices from the local `utils/constants.py` allowlists (`ALL_EP_NAMES`, `SUPPORTED_DEVICES`) to the session facade's catalog-derived `VALID_EPS` / `VALID_DEVICES`. Fixes a casing bug — `--device` now accepts lowercase `cpu/gpu/npu` (previously case-sensitive uppercase only) — and gives both options a single source of truth (the EPDeviceSpec catalog). No new options, no decorator-signature breakage for callers.

## Diff metrics

`+15 / -7` (net `+8`). Hunks: 2 (top of file imports/constants; `device_option` body and signature).

## Role before vs after

| | Before | After |
|---|---|---|
| EP choice source | `ALL_EP_NAMES` (a list in `utils/constants.py` built from the local `SUPPORTED_EPS` + `EP_ALIASES`) | `sorted(VALID_EPS)` (frozenset derived from `EP_DEVICE_SPECS` catalog in `session/ep_device.py`) |
| Device choice source | `SUPPORTED_DEVICES = ["CPU","GPU","NPU"]` (uppercase list) | `sorted(VALID_DEVICES)` (frozenset of lowercase device strings — `{"cpu","gpu","npu"}`) |
| `device_option` case-sensitivity | `case_sensitive=True` (uppercase only) | `case_sensitive=False` |
| `device_option` default | `"NPU"` | `"npu"` |
| `device_option` help text | `"Target device type (CPU, GPU, NPU)"` | `"Target device type (cpu, gpu, npu)"` |

Role of the file is unchanged — it is still the Click-decorator factory for the shared CLI options (`--model`, `--ep`, `--device`, `--verbose`/`--quiet`).

## Symbol-level changes

Top-level module:

- Removed import: `from .constants import ALL_EP_NAMES, SUPPORTED_DEVICES`.
- Added import: `from ..session import VALID_DEVICES, VALID_EPS` — first time `utils/cli.py` reaches across into the session package.
- Added module-level constants:
  - `_DEVICE_CHOICES = sorted(VALID_DEVICES)` — private; comment flags the previous uppercase as a bug.
  - `_EP_CHOICES = sorted(VALID_EPS)` — private; "single source of truth" comment.

Function `ep_option(required=True, optional_message=None)`:

- Body — `click.Choice(ALL_EP_NAMES, …)` → `click.Choice(_EP_CHOICES, …)`. `case_sensitive=False` retained.
- Help text unchanged (still lists the three full-name EPs and aliases `qnn`, `ov/openvino`, `vitis/vitisai`). NOTE: the help text now lies by omission — `_EP_CHOICES` is derived from the catalog and includes `cuda`, `tensorrt`, `migraphx`, `nv_tensorrt_rtx`, `dml`, `cpu` short names too, but the help string only advertises the three NPU/Vitis aliases. (Documented as an open question below.)

Function `device_option(required=True, optional_message=None, default="npu")`:

- Default kwarg `"NPU"` → `"npu"` (lowercase).
- `click.Choice(SUPPORTED_DEVICES, case_sensitive=True)` → `click.Choice(_DEVICE_CHOICES, case_sensitive=False)`.
- Help text `"Target device type (CPU, GPU, NPU)"` → lowercase.
- Docstring updated correspondingly (mention of `NPU as default` → `npu as default`).

Functions left untouched: `model_option`, `verbosity_options`.

## Behavior / contract changes

- `winml … --device CPU` (uppercase) now succeeds — previously rejected by Click because `case_sensitive=True`. Conversely `winml … --device npu` also works (it already did, but only via Click's case-insensitive default; the explicit `case_sensitive=True` had made it strict).
- The default device value flowing into command bodies is now the string `"npu"` not `"NPU"`. Any downstream consumer that does an exact-uppercase comparison (e.g. `device == "NPU"`) would break. The commit migrated the session/CLI boundary to lowercase consistently and dropped uppercase comparisons, so this should be safe inside the package, but is worth checking when revising any external script that wraps these commands.
- `--ep` choice set expanded substantially. Old `ALL_EP_NAMES` was `["QNNExecutionProvider", "OpenVINOExecutionProvider", "VitisAIExecutionProvider", "qnn", "openvino", "ov", "vitisai", "vitis"]`. New `_EP_CHOICES = sorted(VALID_EPS)` is the short names only — currently `{"qnn", "openvino", "vitisai", "migraphx", "nv_tensorrt_rtx", "cuda", "tensorrt", "dml", "cpu"}`. Two consequences:
  1. The full-name forms (`QNNExecutionProvider`, etc.) are **no longer accepted by Click** (they were before via `ALL_EP_NAMES`). Any user CLI invocation that previously passed full ExecutionProvider names will now be rejected at the click-parse step. The commit message frames this as part of the "Option A hard-break" stance.
  2. The two-letter legacy aliases `ov` and `vitis` are also no longer in the Click `Choice` allowlist (they live only inside `constants.normalize_ep_name`'s legacy-alias map). Passing `--ep ov` to a CLI command will now fail at click-parse time, before `normalize_ep_name` is ever called.

## Cross-file impact

- This is one of two main consumers of the new `VALID_DEVICES`/`VALID_EPS` exports added to `session/__init__.py`. The other intra-package consumers are `utils/constants.py` (`expand_ep_name`) and the per-command modules that hold `--ep` / `--device` defaults.
- The file no longer imports anything from `utils/constants.py`, but `utils/constants.py` is still imported by other modules in the package for `normalize_ep_name`, `extract_ep_options`, and the `DEVICE_TO_DEVICE_TYPE` / `DEVICE_TYPE_TO_DEVICE` maps.
- Click resolves choices at decoration time (module import), so the catalog is materialised at import of any CLI command module. That keeps the import-time cost on the session package, which is the same thing the eager `from ..session import VALID_DEVICES, VALID_EPS` already pays.

## Risks / subtleties

- **Help-text drift.** The `--ep` help text was not updated to reflect the broader `_EP_CHOICES` set. Users running `winml … --help` will see only three aliases advertised, but Click will accept any of nine. Minor UX inconsistency, not a correctness bug.
- **Hidden breakage of full-name CLI invocations.** Anyone scripting `winml perf --ep QNNExecutionProvider …` will see a sudden `Invalid value for '--ep'` after this commit. The commit body's "Option A hard-break — no compat shims" justification covers it explicitly, but it is a real wire-level change.
- **Two-letter alias loss at Click layer.** As above for `ov` / `vitis`. The `normalize_ep_name` function in `constants.py` still handles them, but only if something else parses them first; Click will short-circuit.
- **Default-value semantics.** When `required=False` the option default is `default if not required else None`, which is now `"npu"` (lowercase) instead of `"NPU"`. Any downstream `if device == "NPU"` check or dict lookup keyed by uppercase will silently miss.
- **Module-level eager sort.** `sorted(VALID_EPS)` runs at import; `VALID_EPS` is a frozenset, so `sorted` returns an alphabetically-ordered list. The Click help renders choices in that alphabetical order — a cosmetic change from the old hand-ordered list.

## Open questions / TODOs surfaced

- Should `ep_option`'s help text be regenerated from `_EP_CHOICES` so it stays in sync with the catalog? Right now it is a hand-maintained string that will silently desynchronise the next time the catalog grows.
- The legacy two-letter aliases `ov` and `vitis` are dead at the Click boundary. If they are still considered supported, they need to be added to the catalog's `_SHORT_TO_FULL` (in `session/ep_device.py`). Otherwise they should be removed from `_EP_CLI_PREFIXES` and the `_legacy` dict in `utils/constants.py` (they currently route through `normalize_ep_name` but nothing reaches it anymore).
- The file no longer carries a "case-insensitive when the EP catalog is case-mixed" comment. As more full-cased EP names enter the catalog (e.g. `NvTensorRtRtxExecutionProvider`), the relationship between Click input casing and the catalog's full-name canonical casing depends entirely on `canonicalize_ep_name`. Worth a brief comment if anyone touches this again.
