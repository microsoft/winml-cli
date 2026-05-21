# src/winml/modelkit/commands/analyze.py

## TL;DR
Cosmetic-only update: lowercases the default device label and the example
device/EP names in the `--device` option description, default value, and
docstring example. Pure case normalization to match the new `VALID_DEVICES`
catalog (lowercase short names) and the resolver's canonicalize-on-input
contract. No code-path, control-flow, or contract changes.

## Diff metrics
- Lines added: 2
- Lines removed: 2
- Net: 0
- New / modified: modified (existing file, ~700 lines untouched)

## Role before vs after
Before: `analyze` CLI command; device flag's `optional_message` advertised
"uses NPU as default" and `default="NPU"`; docstring example showed
`--ep ov --device GPU`.

After: same command, same wiring; only string casing changed:
- `optional_message="If not specified, uses npu as default"`
- `default="npu"`
- example: `--ep openvino --device gpu`

Aligns with the new policy that EP short-names and device names are
**lowercase** throughout the CLI surface — the catalog (`VALID_DEVICES`,
`expand_ep_name`, `canonicalize_ep_name`) is the single source of truth, and
this command was the last spot still emitting uppercase tokens.

## Symbol-level changes
- `@cli_utils.device_option(...)`: `default` kwarg `"NPU"` → `"npu"`,
  message string updated correspondingly.
- Docstring example line in the `analyze` function: `--ep ov --device GPU`
  → `--ep openvino --device gpu` (also drops the `ov` short alias in favor
  of full `openvino` — consistent with the commit-body note that the
  short-form `ov` is no longer the favored CLI surface; `openvino` is the
  canonical short name and `OpenVINOExecutionProvider` is the full form).

No imports, no signatures, no body code changed. The downstream call
`analyzer.analyze(..., ep=ep_normalized, device=device, ...)` still goes
through `normalize_ep_name()` (line 467) and passes raw `device` through —
so functional behavior matches whatever case-insensitive handling
ONNXStaticAnalyzer already had.

## Behavior / contract changes
- Default device when `--device` is omitted changes from string `"NPU"` to
  string `"npu"`. This is **the only runtime-visible change**. Downstream
  consumers must treat it case-insensitively (the analyzer evidently does,
  since the example with `gpu` and `npu` is expected to work).
- Help text in `--help` output is changed (npu lowercase).
- Examples in the docstring reflect the new canonical short-form
  (`openvino`, not `ov`).

## Cross-file impact
- Depends on `cli_utils.device_option` accepting an arbitrary lowercase
  default string — confirm that helper does not validate against
  uppercase-only `click.Choice`. (Not verified here; this file alone was
  the diff.)
- `analyzer.analyze()` receives `device="npu"` (default) instead of
  `device="NPU"`. The analyzer must be case-tolerant. The commit body does
  not flag any analyzer change, suggesting it already was.
- The analyze CLI was **not migrated** to the typed `EPDevice` contract.
  It still passes `ep` and `device` to the analyzer as raw strings,
  whereas sibling commands (`perf`, `compile`, `build`) now resolve a
  typed `EPDevice` at the CLI boundary via `session.resolve_device(ep,
  device)` (which handles `"auto"` internally). This is a **gap**
  relative to the new EPDevice-at-CLI-boundary pattern.

## Risks / subtleties
- Inconsistency with sibling commands: `compile.py` now uses
  `resolve_device(ep, device)` at the CLI boundary and catches
  `DeviceNotFound`/`EPNotDiscovered` with remediation hints. `build.py`
  uses `auto_detect_device()` + `get_available_devices()` for auto-select.
  `analyze.py` does neither — it keeps the legacy "string passed
  straight through" approach. This may be intentional (analyzer treats
  EP+device as hints/filters, not as a session creation), but it leaves
  `analyze` outside the new error-UX umbrella.
- The string `"npu"` is hard-coded; if `VALID_DEVICES` changes its
  canonical spelling, this drifts.
- Case-sensitivity contract for `cli_utils.device_option` is implicit —
  no type hint or validator visible in this file.

## Open questions / TODOs surfaced
- Should `analyze` adopt `resolve_device(ep, device)` at the CLI boundary
  like `perf` and `compile` did, for unified error UX?
- Is the `cli_utils.device_option` Click decorator backed by a
  case-insensitive `click.Choice` (so users can still type `NPU`)?
- The commit-body's "device deduced" flows (`winml perf --ep qnn` →
  device auto-selected) are not mirrored for `analyze`; intentional, or
  TODO?
