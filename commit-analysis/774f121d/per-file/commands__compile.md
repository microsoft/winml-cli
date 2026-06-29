# src/winml/modelkit/commands/compile.py

## TL;DR
The compile command becomes the **canonical reference implementation**
for the new EP-resolution boundary. `--ep` swaps from a `click.Choice(VALID_EPS)`
to the shared `EpAtSourceParamType` (so `--ep openvino@pypi` is a first-class
syntax — Scenarios A.5/A.6 in `2_coreloop.md` §6.2). `--device` swaps from a
hardcoded `["auto","npu","gpu","cpu"]` Choice to
`["auto", *sorted(VALID_DEVICES)]` and defaults to `None` (not `"npu"`). The
legacy `_resolve_compile_provider(device, ep)` helper that relied on the
`_DEVICE_TO_PROVIDER` / `_EP_TO_DEVICE` hardcodes in `config/precision.py`
is **deleted**; in its place is a single `resolve_device(EPDeviceTarget(...))`
call at the CLI boundary that catches `DeviceNotFound`,
`WinMLEPNotDiscovered`, `WinMLEPRegistrationFailed`, and `ValueError` with
explicit remediation hints. The resolved `WinMLEPDevice` is then passed
unmodified to `WinMLCompileConfig.for_ep_device(ep_device_resolved)`
(replacing the legacy `for_provider(provider_str)`).

## Diff metrics
- 76 lines changed (47 insertions / 29 deletions per `--stat`).
- Three concentrated hunks: imports, decorator, callback body.
- Net deletion: `_resolve_compile_provider` helper at the file tail.

## Role before vs after
Before: callback collapsed `--device` + `--ep` into a single `provider`
string via `_resolve_compile_provider(device, ep)` — a manual lookup
into the prebuilt `_DEVICE_TO_PROVIDER` / `_EP_TO_DEVICE` dicts in
`config/precision.py`. The result was a string (`"qnn"`, `"vitisai"`, etc.)
that `WinMLCompileConfig.for_provider(provider)` consumed. No source-tag
support; no `WinMLEPDevice` typing; no remediation errors for unknown
devices.

After: callback parses `--ep` as `(ep, source) | None` via
`EpAtSourceParamType`, constructs an `EPDeviceTarget`, runs the unified
resolver, and consumes the typed `WinMLEPDevice` for both config building
and console display. `WinMLCompileConfig.for_ep_device(ep_device)`
replaces the string-based `for_provider`. Source-tag pinning works
end-to-end.

## Symbol-level changes

### Imports
- Removed: `from ..config import VALID_EPS`
- Removed: `from ..config.precision import _DEVICE_TO_PROVIDER, _EP_TO_DEVICE`
- Added: `from ..session import VALID_DEVICES, EPDeviceTarget, resolve_device`
- Added: `from ..session.ep_device import DeviceNotFound,
  WinMLEPNotDiscovered, WinMLEPRegistrationFailed`
- Added: `from ._ep_arg import EpAtSourceParamType`

The removed imports came from the legacy `config.precision` private
underscore-name reach. Importing private symbols across packages was the
exact CLAUDE.md import rule violation in the prior commit ("Never reach
into internal submodules for symbols exported by `__init__.py`"). The
swap to the `session` public surface is a hygiene win.

### `--device` Click decorator
```python
@click.option(
    "--device", "-d",
    type=click.Choice(["auto", *sorted(VALID_DEVICES)], case_sensitive=False),
    default=None,
    help="Target device (default: deduced from --ep, or 'npu' if neither given)",
)
```

- Choice list source-of-truth swap: `["auto", "npu", "gpu", "cpu"]`
  → `["auto", *sorted(VALID_DEVICES)]`. Now matches `eval.py`, `perf.py`,
  `config.py`.
- Default: `"npu"` → `None`. Previously a hardcoded NPU bias. Now `None`
  triggers the resolver's catalog-ordered auto-pick. Help text updated
  to describe the new behavior accurately.
- Type annotation in callback: `device: str` → `device: str | None`.

### `--ep` Click decorator
```python
@click.option(
    "--ep",
    type=EpAtSourceParamType(),
    default=None,
    help="Force specific EP, optionally pinned to a source (e.g. 'openvino@pypi'). "
    "Overrides device-to-provider mapping.",
)
```

Type swap from `click.Choice(sorted(VALID_EPS))` to
`EpAtSourceParamType()`. The Choice-list constraint is dropped (the
ParamType doesn't whitelist; it just parses the `@<tag>` syntax). EP
name validation now happens later inside `EPDeviceTarget.__post_init__`
or `resolve_device`. Help text now documents the source-tag syntax.

### Callback body — resolver block
```python
_device_arg = "auto" if (device is None or device.lower() == "auto") else device.lower()
ep_part, source_part = ep if ep else (None, None)
try:
    ep_device_resolved = resolve_device(
        EPDeviceTarget(
            ep=ep_part or "auto",
            device=_device_arg,
            source=source_part,
        )
    )
except DeviceNotFound as e:
    raise click.ClickException(str(e)) from e
except WinMLEPNotDiscovered as e:
    raise click.ClickException(
        f"EP plugin not found: {e}. Install the required EP package (e.g. onnxruntime-qnn)."
    ) from e
except WinMLEPRegistrationFailed as e:
    raise click.ClickException(f"EP registration failed: {e}") from e
except ValueError as e:
    raise click.UsageError(str(e)) from e
logger.info("Resolved to: %s", ep_device_resolved)
```

Key properties:
- Single resolution point. Subsequent code paths
  (`list_compilers_flag`, the `compile_onnx` call) consume
  `ep_device_resolved.device`, `ep_device_resolved.ep` — never the
  raw `--device` / `--ep` strings.
- **Explicit per-exception handlers with remediation hints**. The
  three named resolver exceptions each get a tailored message. Compare
  with `build.py`'s broad `except Exception` that swallows everything
  silently — `compile.py` is the model for the right error UX, and
  `build.py` should adopt the same pattern.
- `ValueError → click.UsageError` is the right mapping: arg-shape
  problems should look like usage errors, not internal failures.

### `_resolve_compile_provider` — deleted
End-of-file helper removed. The 12-line function did:
```python
def _resolve_compile_provider(device: str, ep: str | None) -> str:
    if ep: return ep.lower()
    provider = _DEVICE_TO_PROVIDER.get(device.lower())
    if provider is None:
        return "cpu" if device.lower() == "cpu" else "qnn"  # NPU bias
    return provider
```

The "if no mapping found, default to qnn" branch was the legacy
NPU-bias hardcode. The catalog-driven resolver is the replacement,
and the hardcode is correctly removed.

### `WinMLCompileConfig` consumption
- Before: `config = WinMLCompileConfig.for_provider(provider)` —
  takes a string.
- After: `config = WinMLCompileConfig.for_ep_device(ep_device_resolved)` —
  takes a `WinMLEPDevice`. The config class must have grown the new
  factory method; this commit's `config/*` changes (not in this
  batch) presumably add it.

### Console output
Pre-state: `console.print(f"... {_EP_TO_DEVICE.get(provider, device)}")`
— used the dict reach into precision.py.

Post-state: `console.print(f"[bold blue]Device:[/bold blue] {ep_device_resolved.device}")`
— reads off the resolved device directly. Same for `Provider`.

### `--list` handler
Pre-state: `provider = _resolve_compile_provider(device, ep)` ;
`list_compilers(provider)`.

Post-state: `click.echo(list_compilers(ep_device_resolved.device))`.

A subtle semantic shift: pre-state passed the EP/provider short-name
("qnn", "vitisai") to `list_compilers`; post-state passes the device
class ("NPU", "GPU", "CPU"). If `list_compilers` actually wants a
provider name, this is a bug. Worth verifying in `compiler/__init__.py`.
Quick check: `list_compilers` in `compiler/cli.py` per `git status`
modifications — might be intentional if the prior commit already
adjusted that API.

## Behavior / contract changes

### (a) `--device` default is no longer hardcoded NPU
A user typing `winml compile -m model.onnx` (no `--device`, no `--ep`)
now triggers the resolver's auto-detection (catalog ordering picks the
first compatible device, which on an NPU-equipped machine is QNN/NPU;
on a CPU-only machine is CPU). Behavioral parity with prior on
NPU systems; new graceful CPU fallback on non-NPU systems instead of
"defaults to NPU which then errors".

### (b) `--ep` accepts source-tag
Same syntactic story as `perf.py`. Source-tag rejection is
**not** in this file — it's accepted and threaded through to
`EPDeviceTarget(source=source_part)`. The `WinMLCompileConfig` must
likewise tolerate the source pin.

### (c) Error UX is now structured
`DeviceNotFound` / `WinMLEPNotDiscovered` / `WinMLEPRegistrationFailed`
each surface as a `ClickException` with a tailored message and
remediation hint. The compiler-callsite errors look professional now
("EP plugin not found: ... . Install the required EP package
(e.g. onnxruntime-qnn).") versus the legacy "could not resolve device"
opacity.

### (d) `ValueError` from EPDeviceTarget becomes UsageError
Construction-time validation in `EPDeviceTarget.__post_init__` (per
`2_coreloop.md` §4 class taxonomy) raises `ValueError` on bad input.
The handler maps it to `click.UsageError` so it renders with click's
usage formatting. This is the right mapping for user-input problems.

### (e) `list_compilers(ep_device.device)` semantic
Confirm against `compiler/__init__.py`: if `list_compilers` expects a
provider string ("qnn") and now gets a device class ("NPU"), it
silently misbehaves. The variable name `provider` was renamed away;
the API contract may have shifted in lockstep, but this would be
worth a spot-check in the corresponding compiler-side review.

## Cross-file impact
- **`WinMLCompileConfig.for_ep_device(ep_device)`** must exist. Replaces
  `for_provider(provider_str)`. `compile.py` is the sole caller in the
  visible diff.
- **`commands/_ep_arg.py` (NEW)** consumer.
- **`session.ep_device`** must export `DeviceNotFound`,
  `WinMLEPNotDiscovered`, `WinMLEPRegistrationFailed` directly (not via
  `session/__init__.py`). The compile.py import is
  `from ..session.ep_device import ...` — a deliberate exception to
  the CLAUDE.md "use the package facade" rule that the commit body
  flags as acceptable for exception types (they're rarely-changing
  symbols and module-level import keeps the callsite syntactically
  light).
- **`config.precision._DEVICE_TO_PROVIDER` / `_EP_TO_DEVICE`** are no
  longer imported here. If the prior commit didn't remove them from
  `config/precision.py`, they may be dead code — worth a separate
  follow-up grep.

## Risks / subtleties
- **`list_compilers(ep_device_resolved.device)` semantic shift**: if
  `list_compilers` expects a provider string (e.g., "qnn") and now
  gets a device class enum/str (e.g., "NPU"), the output silently
  changes shape. The diff doesn't include `compiler/__init__.py`
  changes, so verification is needed. (Most likely the commit author
  updated `list_compilers` in lockstep.)
- **`EPDeviceTarget(ep=ep_part or "auto", device=_device_arg,
  source=source_part)`** — `ep_part` is `None` when `--ep` was
  omitted, so the `or "auto"` fallback kicks in. `_device_arg` is
  pre-coerced to `"auto"` when device is `None`. The two axes use
  the same sentinel pattern, which is the right hygiene.
- **`from ..session.ep_device import ...`** reaches past the
  `session/__init__.py` facade. The CLAUDE.md import rule says "Never
  use absolute `from winml.modelkit.*` paths in source code", but this
  is a relative-from-submodule import — which the rule allows. Still,
  the import target is a submodule rather than the package: the
  cleaner path would be `from ..session import DeviceNotFound, ...`.
  Worth confirming `session/__init__.py` re-exports these (the prior
  commit's session refactor likely added them — the public surface
  for catchable exceptions is necessary if any external caller is
  going to handle them).
- **No `WinMLCompileConfig` validation of the source-tag**: if the user
  passes `--ep openvino@msix` and the build pipeline downstream
  produces an MSIX-style provider config, the source-tag is implicitly
  honored. If downstream uses just the EP short-name, the source-tag
  is silently dropped at compile config time. The diff doesn't show
  which way it goes — depends on `for_ep_device`'s implementation.
- **`device = None` parameter type**: the type hint `device: str | None`
  is correct, but the callback body's first action is to coerce
  `None | "auto" → "auto"` before passing to the resolver. If anyone
  reads the value before that coercion they get a `None`. Minor.

## Simplification opportunities
- **Hoist the resolver block to a helper**: identical resolver-pattern
  appears in `perf.py` (two call sites) and could be shared. A
  `_resolve_ep_device(ep, device, *, allow_source_tag=True) ->
  WinMLEPDevice` helper in `commands/_ep_arg.py` (or a new
  `commands/_resolver.py`) would collapse the 20-line try/except block
  to one call per CLI command. The exception-to-ClickException mapping
  would also be centralized.
- **Drop the `_device_arg` local**: it's only used once, and the
  ternary inline `(device or "auto").lower()` reads as well.
- **Multi-line tuple-unpack**: `ep_part, source_part = ep if ep else
  (None, None)` is concise but cryptic. A small helper or named
  default would help readability. Minor.

## Open questions / TODOs surfaced
- **Does `list_compilers` accept a device class?** If not, this is a
  regression. Cross-reference `compiler/__init__.py` modifications
  (not in this batch).
- **Does `WinMLCompileConfig.for_ep_device` honor the source pin?**
  The implementation likely lives in `config/*`; worth verifying
  that `ep_device.source` (or the equivalent) ends up in
  `ep_config.provider` or as a separate pin.
- **`from ..session.ep_device import ...`** — should be promoted to
  `from ..session import ...` once `session/__init__.py` re-exports
  the exception types.
- **Source-tag rejection is opt-out**: this command accepts it. The
  reverse decision (build.py / config.py reject it) deserves a
  shared helper to make the decision declarative.
