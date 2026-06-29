# src/winml/modelkit/commands/config.py

## TL;DR
Similar shape to `build.py`: imports the new shared `EpAtSourceParamType`,
swaps the `--device` Choice's hardcoded `["auto","npu","gpu","cpu"]` for
`["auto", *sorted(VALID_DEVICES)]`, rejects the `@<source-tag>` syntax at
the CLI boundary (config's pipeline takes a bare EP short-name), and
replaces the legacy `sysinfo.resolve_device(device)` device-resolution
call in the "Resolution" display section with the session-package's pure
`auto_detect_device()` (no EP registration side effects). Everything else
in the 623-line file is unchanged.

## Diff metrics
- 31 lines changed (23 insertions / 8 deletions per `--stat`).
- Three small hunks: imports, `--device` Choice, `--ep` type swap +
  rejection block, and the device-resolution display call.

## Role before vs after
Before: `--ep` was an untyped `str`; the device-resolution display used
`from ..sysinfo import resolve_device as _rd` (the legacy
`(category, info)`-tuple returner).

After: `--ep` uses `EpAtSourceParamType()`, source-tag rejected at the
boundary; the display resolution uses `from ..session import
auto_detect_device` — a pure detection that doesn't trigger EP
registration as a side effect.

The config-generation pipeline downstream (`generate_hf_build_config`,
`generate_onnx_build_config`, the composite-model branch, the module-mode
loop) still receives `ep: str | None` / `device: str` as before. So
config-generation is still string-keyed end-to-end; only the CLI parser
and the display layer changed.

## Symbol-level changes

### Imports
- Added top-level: `from ..session import VALID_DEVICES`,
  `from ._ep_arg import EpAtSourceParamType`.
- Removed from inline call site: `from ..sysinfo import resolve_device as _rd`.
- Added at the inline call site: `from ..session import auto_detect_device`.

The sysinfo→session swap shifts the "device deduction" responsibility
from `sysinfo` (which historically also touched WMI / hardware queries)
to the catalog-driven `session` module. The commit body confirms the
`session.auto_detect_device` is the public surface for "what device
class should I pick?" without doing any EP registration.

### `--device` Click decorator
```python
type=click.Choice(["auto", *sorted(VALID_DEVICES)], case_sensitive=False),
```
Same shape change as `eval.py`, `perf.py`, `compile.py`. Single
source-of-truth from the `..session` package.

### `--ep` Click decorator
```python
type=EpAtSourceParamType(),
help="... When used without --device, device is inferred from EP. "
"(Source-pinning ``@<source-tag>`` is rejected: config's pipeline "
"takes a bare EP short-name.)",
```
Type swap with documented rejection in the help text — same pattern as
`build.py`.

### Source-tag rejection block (after pre-flight validation)
```python
if ep:
    ep_part, ep_source = ep
    if ep_source is not None:
        raise click.UsageError(
            f"`winml config` does not yet support source pinning "
            f"(got --ep {ep_part}@{ep_source!r}); "
            f"use --ep {ep_part!r} without '@'."
        )
    ep = ep_part
```

**Verbatim duplicate** of the block in `build.py`. The only difference
is the command name in the message. This is the strongest case in the
batch for hoisting to a shared `_reject_ep_source(ep, *, command_name)`
helper in `commands/_ep_arg.py`.

### Display-resolution call site (line ~478)
Before:
```python
from ..sysinfo import resolve_device as _rd
_resolved_dev, _ = _rd(device)
```

After:
```python
from ..session import auto_detect_device
_resolved_dev = auto_detect_device() if device.lower() == "auto" else device.lower()
```

The new shape:
- Calls `auto_detect_device()` only when `device == "auto"`. Otherwise
  takes the user-typed value verbatim (lowercased). Skips the catalog
  walk when there's nothing to deduce.
- Returns just the resolved category (str), discarding the
  `(category, info)` tuple shape from `sysinfo.resolve_device`. The
  display only ever read the first element, so the simplification is
  free.
- Inline comment notes the design intent: *"resolves "auto" to a
  concrete category without registering EPs"*. The previous
  `sysinfo.resolve_device` did do some EP-registry probing as part of
  its decision; the new path is pure deduction.

## Behavior / contract changes

### (a) Source-tag rejection
Same hard reject as `build.py`. `winml config --ep openvino@pypi`
emits a UsageError. Source pin support is deferred until the
config pipeline can persist the source tag into a generated
WinMLBuildConfig (which would require a `WinMLBuildConfig.ep_source`
field — not in this commit).

### (b) Device deduction no longer touches the EP registry
The display's `Device: ...` line is now computed without
side-effects. The pre-state's `sysinfo.resolve_device` had nontrivial
behavior (would consult hardware queries, sometimes load DLLs to
probe EP availability). New behavior is faster and cheaper, but
**might also be less informative** — if a user has an NPU but no QNN
EP installed, the new `auto_detect_device()` will likely say "NPU"
based on hardware, even though no EP is available. The pre-state may
have returned "CPU" because the EP probe failed. Whether this is a
regression depends on `auto_detect_device`'s implementation.

### (c) `--device` Choice list
Same source-of-truth swap as the other commands. CUDA / TensorRT / etc.
device categories that the catalog supports are now reachable; pre-state
hardcoded the four-element list.

## Cross-file impact
- Depends on `commands/_ep_arg.py` (NEW) for `EpAtSourceParamType`.
- Depends on `..session` exporting `VALID_DEVICES`, `auto_detect_device`.
  Both are confirmed in the prior commit's session-public-surface
  inventory.
- `generate_hf_build_config(..., ep=ep)` and
  `generate_onnx_build_config(..., ep=ep)` still take a string —
  the config-generation pipeline is unchanged.
- `_generate_pipeline_configs` (the composite-model branch) — same
  `ep=ep` string signature, unchanged.

## Risks / subtleties
- **Source-tag rejection is now duplicated across `build.py` /
  `config.py`**. Verbatim except for the command name. A shared
  helper (e.g., `_reject_ep_source(ep, *, command_name)`) would
  collapse both to one line.
- **`auto_detect_device()` vs `sysinfo.resolve_device` semantic drift**:
  the prior call returned a `(category, info)` tuple where `info`
  carried hardware details (model name, etc.). The new call just
  returns a category. The display only used the category, so this is
  fine — but any consumer outside this file that imported
  `sysinfo.resolve_device` for the `info` half is affected by the
  broader migration (cf. `eval.py` and `perf.py` which both did the
  same swap).
- **`device.lower()` after `device == "auto"`**: the user could have
  typed `--device GPU` (case-insensitive Choice match). The
  `auto_detect_device()` branch is only taken on `"auto"`; otherwise
  the value goes through `.lower()` to normalize. Fine.
- **`device` is `str` not `str | None` here** (per the Click default
  of `"auto"`); `compile.py` chose `None` as the default but
  `config.py` keeps `"auto"`. Cross-file inconsistency in
  `--device` defaults.

## Simplification opportunities
- **Shared source-tag rejection helper**: as noted, `_reject_ep_source(
  ep, *, command_name="config")` in `commands/_ep_arg.py` would
  collapse the duplicated block.
- **`_apply_stage_overrides` is fine as-is** — called 4× across the
  ONNX / single / module / pipeline branches. Tight, focused.
- **The 100-line "Resolution display" block** could be factored into
  a `_print_resolution(console, *, device, ep, quant_cfg)` helper.
  Tangential to this commit's focus.

## Open questions / TODOs surfaced
- Should `config` accept source-tag and persist it into the generated
  `WinMLBuildConfig`? Today it rejects; the persisted config can
  therefore not reproduce a Scenario A.5/A.6 invocation.
- `auto_detect_device()` semantics vs old `sysinfo.resolve_device`:
  is the new path strictly equivalent for the display, or does it
  diverge when EP probes used to influence the answer? Worth a
  behavioral spot-check on a machine with NPU hardware but no QNN
  package installed.
- `--device` default: `compile.py` uses `None`, `config.py` uses
  `"auto"`, `perf.py` uses `"auto"`, `eval.py` uses `"auto"`. The
  inconsistency is minor (all four end up with the same auto-detect
  behavior), but a shared decorator factory would tighten this.
