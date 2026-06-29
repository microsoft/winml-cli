# src/winml/modelkit/commands/analyze.py

## TL;DR
Two-line change. The `@cli_utils.device_option(...)` decorator's default
and help text are case-normalized from `"NPU"` → `"npu"` (to match the
lowercase convention adopted across the new `--device` Choice lists in
`build.py`, `compile.py`, `config.py`, `eval.py`, `perf.py`). The
docstring example is corrected from `--ep ov --device GPU` to
`--ep openvino --device gpu` — the short alias `ov` is no longer a valid
short-name (the catalog uses `openvino`), and lowercase `gpu` matches the
new Choice convention. Nothing else in this 775-line file is touched.

## Diff metrics
- 4 lines changed (2 insertions / 2 deletions per `--stat`).
- Single hunk in the `@cli_utils.device_option` decorator + a doc
  example.

## Role before vs after
Role unchanged: `winml analyze` still runs the static analyzer with
Rich Live stacked-bar progress display. The change is purely cosmetic
— case-normalization in user-facing strings.

## Symbol-level changes

### `@cli_utils.device_option`
```python
@cli_utils.device_option(
    required=False, optional_message="If not specified, uses npu as default", default="npu"
)
```

- `default="NPU"` → `default="npu"` (lowercased).
- `optional_message="... uses NPU as default"` → `"... uses npu as default"`.

The `cli_utils.device_option` helper in `utils/cli.py` (per CLAUDE.md it
lives under `utils.cli`) wraps a `click.option` decorator. Whether
`cli_utils.device_option` itself normalizes the default to a Choice
match is not changed by this diff; if `cli_utils` does
`click.Choice(["auto","npu","gpu","cpu"], case_sensitive=False)`, the
default must match a Choice element. Lowercased `"npu"` does.

### Docstring example
```
- winml analyze --model model.onnx --ep ov --device GPU
+ winml analyze --model model.onnx --ep openvino --device gpu
```

`ov` was a short alias for OpenVINO that the catalog presumably no
longer recognizes (the catalog uses `openvino`). `GPU` → `gpu` matches
the lowercase convention.

## Behavior / contract changes
- **None for the actual analyzer pipeline**. The Choice
  case-insensitivity (presumed `case_sensitive=False` in
  `cli_utils.device_option`) means both `"NPU"` and `"npu"` parse to the
  same value; the default change is invisible to users who already
  passed `--device`. Users relying on the implicit default get `"npu"`
  (lowercase) routed to the analyzer instead of `"NPU"` — and if any
  downstream layer is case-sensitive about the device string, that
  layer's behavior might shift.
- The `--ep ov` shorthand example is now wrong if the catalog only
  knows `openvino`. The diff just updates the docstring; the
  underlying validity of `ov` as an alias depends on
  `normalize_ep_name` (used inside `analyze`) and the EP catalog
  alias map.

## Cross-file impact
- Depends on `cli_utils.device_option`'s default-passing behavior.
- `normalize_ep_name(ep)` is still called inside the body — if
  `ov` worked pre-diff because of an alias, the example update is
  a documentation correction; if it never worked, the example was
  a pre-existing bug now fixed.

## Risks / subtleties
- **Downstream case-sensitivity**: any consumer reading the device
  string for an exact match (e.g., `if device == "NPU": ...`) is
  affected by the lowercased default. The diff doesn't touch any
  such consumer in this file, so the risk is mostly in
  `analyze/utils/ep_utils.py` (e.g., `has_rule_data_for_ep`,
  `get_devices_with_rule_data`). The body's call sites pass `device`
  through without normalization, so any case-sensitive check
  downstream that previously matched `"NPU"` now sees `"npu"`. Worth
  verifying.
- **Example reflects current alias state**: if at some point
  `winml analyze --ep ov` is meant to work via an alias map, the
  docstring update either accidentally drops the alias or correctly
  reflects its removal.

## Cross-file impact (verbose)
- `cli_utils.device_option(...)` — not touched in this commit; its
  behavior is the load-bearing piece. If it does
  `click.Choice([...], case_sensitive=False)`, the new default lands
  cleanly; if case-sensitive, the change is required for compatibility
  with the Choice element list.

## Simplification opportunities
- **`cli_utils.device_option` should derive defaults from
  `VALID_DEVICES`**: same observation as for the other commands. If
  the helper's signature accepted `default=NPU_DEFAULT` (where
  `NPU_DEFAULT` is a module-level constant), the case-normalization
  would propagate automatically.
- **Centralize the device Choice list**: all four touched commands
  (`build`, `config`, `eval`, `perf`) use
  `["auto", *sorted(VALID_DEVICES)]`. `analyze` uses
  `cli_utils.device_option(...)` which presumably wraps the same.
  If yes, the only-touched-here case fixes are nominally
  redundant — they should follow from a single source-of-truth update.

## Open questions / TODOs surfaced
- Is `--ep ov` still a valid short-form? If yes, the example update
  is a regression in documentation discoverability. If no, the prior
  example was broken (`ov` not in the catalog) and this fixes it.
- Should the `--ep` option here also adopt `EpAtSourceParamType`? The
  other four commands all did. The analyzer pipeline takes a bare
  EP short-name (cf. `build`, `config` rejection rationale), so
  source-tag rejection would be the consistent thing — but this
  commit doesn't make that change. Carried over as a smaller follow-up.
- `cli_utils.device_option(default="npu")` — does the helper use
  `VALID_DEVICES` for the Choice, or does it still hardcode the list?
  Worth verifying. If hardcoded, this single-file fix doesn't
  propagate.
