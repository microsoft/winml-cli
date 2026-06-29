# src/winml/modelkit/commands/build.py

## TL;DR
Small, surgical change: imports the new shared `EpAtSourceParamType` for
`--ep`, switches the auto-EP-selection path from `resolve_device(ep=None,
device=None)` (positional kwargs in the prior commit) to the typed
`resolve_device(EPDeviceTarget(ep="auto", device="auto"))` API, and rejects
the `@<source-tag>` form at the CLI boundary because the build pipeline
downstream takes a bare EP short-name. Everything else in the 1228-line
file is unchanged. Same partial-migration shape as the prior commit (the
build pipeline is still string-keyed end-to-end; `EPDevice` resolution
stops at the CLI).

## Diff metrics
- 58 lines changed (33 insertions / 25 deletions per `--stat`).
- One hunk for the `--ep` Click decorator; one hunk for the auto-select
  block at the top of the callback.

## Role before vs after
Before (prior commit): `--ep` was an untyped `str`; an auto-select block
called the resolver positionally as `resolve_device(ep=None, device=None)`.

After: `--ep` uses `EpAtSourceParamType()` (the shared click ParamType
in `commands/_ep_arg.py`), arriving as `(ep, source)` or `None`. The
source-tag is **rejected** at the CLI boundary with a custom UsageError
because the build pipeline takes a bare EP short-name. The auto-select
block now constructs an `EPDeviceTarget(ep="auto", device="auto")` and
passes it to `resolve_device(target)`.

The downstream call signature (`build_hf_model(..., ep=ep, device=device,
...)`) is **still string-keyed**. So the build pipeline is unchanged
end-to-end; only the CLI's resolver-input shape changed.

## Symbol-level changes

### Imports
- Added: `from ._ep_arg import EpAtSourceParamType` (top-level).
- Added inside auto-select: `from ..session import EPDeviceTarget,
  resolve_device, short_ep_name`. The pre-state imported `resolve_device,
  short_ep_name` (no `EPDeviceTarget`); commit-deltas are the
  `EPDeviceTarget` import and its construction.

### `--ep` Click decorator
```python
@click.option(
    "--ep",
    type=EpAtSourceParamType(),
    default=None,
    help="Target execution provider for analyzer (e.g., 'qnn'). "
    "Falls back to compile config EP if not set. (Source-pinning "
    "``@<source-tag>`` is rejected: build's analyzer pipeline takes a "
    "bare EP short-name.)",
)
```

The help text now documents the source-tag rejection. The type swap is
load-bearing: without it, click would pass the raw `qnn@pypi` string
through to the callback, and the explicit `if ep_source is not None`
check would never fire (because `ep` would be `"qnn@pypi"` not `(qnn,
pypi)`).

### Callback body — source-tag rejection
```python
if ep:
    ep_part, ep_source = ep
    if ep_source is not None:
        raise click.UsageError(
            f"`winml build` does not yet support source pinning "
            f"(got --ep {ep_part}@{ep_source!r}); "
            f"use --ep {ep_part!r} without '@'."
        )
    ep = ep_part
```

After this block, `ep` is either a bare short-name (`"qnn"`) or `None`.
Downstream code reads `ep` as a string, identical to the pre-state.

### Auto-select block
```python
if ep is None:
    from ..session import EPDeviceTarget, resolve_device, short_ep_name

    try:
        _auto_ep_device = resolve_device(EPDeviceTarget(ep="auto", device="auto"))
    except Exception as exc:
        logger.warning("EP unspecified for build, and auto-selection failed: %s. "
                       "Proceeding without EP hints.", exc)
    else:
        ep = short_ep_name(_auto_ep_device.ep)
        logger.info("EP unspecified for build, auto-selecting: %s (device: %s)",
                    ep, _auto_ep_device.device)
```

The `EPDeviceTarget` construction is the new shape — the pre-state called
`resolve_device(ep=None, device=None)` with positional defaults. Both
paths reach the same internal deduction (`"auto"` axis triggers
catalog-driven first-compatible-EP pick); the typed shape just makes the
input intent explicit.

## Behavior / contract changes

### (a) `--ep <name>@<source>` is now a hard reject
Previously the CLI took an untyped string. If a user typed
`winml build --ep openvino@pypi -c config.json -o out/`, the click parser
silently passed `"openvino@pypi"` through to `build_hf_model`, which
then either auto-matched a substring or threw `unknown EP`. Now the
ParamType catches the syntax at parse time and the rejection block
at line ~380 emits a precise UsageError with a remediation hint.

### (b) Auto-select call signature
The typed `EPDeviceTarget` construction is the new shape. Internal
behaviour is identical (the resolver consumes both forms). No
observable user-facing difference here — same catalog ordering,
same `EPDevice` output, same `short_ep_name` extraction.

### (c) Failure mode is unchanged
Broad `except Exception` swallows everything from `resolve_device` —
including `DeviceNotFound`, `WinMLEPNotDiscovered`,
`WinMLEPRegistrationFailed`, and bare `ValueError`. Unlike `compile.py`
which surfaces these with remediation hints, build silently proceeds
without EP hints. Same as the prior commit.

## Cross-file impact
- Depends on `commands/_ep_arg.py` (NEW in this commit) exporting
  `EpAtSourceParamType`.
- Depends on `..session` re-exporting `EPDeviceTarget`, `resolve_device`,
  `short_ep_name` (already re-exported in the prior commit; the commit
  body confirms `EPDeviceTarget` is added to the session public surface
  in the same change).
- Downstream `build_hf_model`, `_build_modules`, `_run_single_build`,
  `_build_hf_pipeline`, `_build_onnx_pipeline`, `_run_optimize_stage`,
  `run_optimize_analyze_loop` still receive `ep: str | None, device:
  str | None`. The build pipeline is **still string-keyed end-to-end**;
  only the CLI auto-select source changed. Same partial-migration
  story as the prior commit.

## Risks / subtleties
- **`except Exception` swallows specific resolver exceptions** —
  including the new `WinMLEPNotDiscovered` / `WinMLEPRegistrationFailed`
  classes. Build silently proceeds with `ep=None`; user gets no
  signal that auto-select failed beyond a single `logger.warning`.
  Compile.py handles the same exceptions with `click.ClickException`
  remediation hints. Cross-file inconsistency unchanged.
- **`device` CLI param is still not auto-set** when `resolve_device`
  succeeds — only `ep` is filled in. So `winml build -m model.id`
  ends up with `ep="qnn"` and `device=None` (Click default), passed
  through to `build_hf_model`. Downstream code must tolerate that.
  Same observation as the prior commit.
- **The `--device` help text** still says `"Default: NPU."` (line ~282).
  Stale since the prior commit's catalog-driven resolver — the catalog
  picks NPU by ordering, not by hardcode, but the help text doesn't
  document it. Minor doc drift.
- **Source-tag rejection is a build-specific limitation, not a design
  principle**: `compile.py` accepts the `@<tag>` syntax and threads it
  through. The "build's analyzer pipeline takes a bare EP short-name"
  rationale is true today, but the rejection is a maintenance
  obligation (every CLI command must re-check whether it now supports
  sources). A single decorator that takes a `support_source=False`
  flag would make this declarative.

## Simplification opportunities
- **Hoist `EPDeviceTarget` import to top-level**: it's already loaded
  for `VALID_DEVICES` consumers elsewhere in the codebase; the local
  scope-import inside `if ep is None:` doesn't save anything
  meaningful for a CLI command that just spent 50ms parsing click
  options.
- **Source-tag rejection helper**: this same block is duplicated
  verbatim in `compile.py` (no, wait — `compile.py` accepts it),
  `config.py` (yes — same block), and `analyze.py` (untouched here
  but a candidate). A `_reject_ep_source(ep, *, command_name)` helper
  in `commands/_ep_arg.py` would collapse to one line per call site:
  `ep = _reject_ep_source(ep, command_name="build")`.
- **`build_hf_model` should take an `EPDevice`**: same observation
  as the prior commit. Migration deferred — would touch
  `_build_modules`, `_run_single_build`, `_build_hf_pipeline`,
  `_build_onnx_pipeline`, `_run_optimize_stage`,
  `run_optimize_analyze_loop`.

## Open questions / TODOs surfaced
- Should `device` symmetrically get auto-filled when `--device` is
  omitted but `resolve_device` succeeds? Currently asymmetric: `ep`
  gets the auto-fill, `device` stays at Click's None default.
- Should `except Exception` be narrowed to the documented resolver
  exceptions (`DeviceNotFound`, `WinMLEPNotDiscovered`,
  `WinMLEPRegistrationFailed`) to match `compile.py`'s remediation-hint
  UX?
- `--device` Click option help text (`"Default: NPU."`) is stale.
- Should the source-tag rejection be a shared helper? Cf. duplicated
  in `config.py`.
