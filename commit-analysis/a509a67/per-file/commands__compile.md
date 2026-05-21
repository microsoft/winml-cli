# src/winml/modelkit/commands/compile.py

## TL;DR
The flagship CLI-boundary change of the commit. `winml compile` migrates from
"device defaults to npu / EP derived via private `_DEVICE_TO_PROVIDER` map"
to "both `--ep` and `--device` optional / `resolve_device(ep, device)` called
at the CLI boundary / downstream gets an `EPDevice` object via
`WinMLCompileConfig.for_ep_device()`". The legacy `_resolve_compile_provider`
helper is deleted. Four resolver-domain exceptions are caught and re-raised
as Click errors with remediation hints (previously: opaque tracebacks).

## Diff metrics
- Lines added: ~30
- Lines removed: ~20 (incl. 16-line `_resolve_compile_provider` helper)
- Net: +10
- New / modified: modified (existing file, ~250 lines)

## Role before vs after
Before:
- `--device` was `click.Choice(["auto", "npu", "gpu", "cpu"], …)` with
  `default="npu", show_default=True`.
- Required signature param `device: str` (always non-None at call time).
- Imported private symbols `_DEVICE_TO_PROVIDER` and `_EP_TO_DEVICE` from
  `..config.precision`.
- Resolved EP via local helper `_resolve_compile_provider(device, ep)`:
  - If `ep` given → returns `ep.lower()`.
  - Else looks up `_DEVICE_TO_PROVIDER[device.lower()]`, defaulting CPU →
    `"cpu"` else `"qnn"` (hardcoded NPU bias).
- Downstream `WinMLCompileConfig.for_provider(provider_str)` was called
  with the resolved string.
- Display panel used `_EP_TO_DEVICE.get(provider, device)` for the Device
  row.
- No structured error handling for unavailable EP/device; whatever ORT or
  registry raised propagated as an opaque traceback.

After:
- `--device` is `click.Choice(["auto", *sorted(VALID_DEVICES)],
  case_sensitive=False)` with `default=None`. `show_default` dropped.
  Help text: `"Target device (default: deduced from --ep, or 'npu' if
  neither given)"`.
- Signature param is now `device: str | None`.
- Private `_DEVICE_TO_PROVIDER` / `_EP_TO_DEVICE` imports **deleted**; new
  imports: `from ..session import VALID_DEVICES, resolve_device` and
  `from ..session.ep_device import DeviceNotFound, EPNotDiscovered,
  EPRegistrationFailed`.
- New CLI-boundary resolution block (lines 152–167):
  - Treats `"auto"` and `None` as "let resolver decide" → passes
    `device=None`.
  - Calls `ep_device_resolved = resolve_device(ep=ep, device=_device_arg)`.
  - Wraps in try/except for 4 distinct error classes:
    - `DeviceNotFound` → `click.ClickException(str(e))`
    - `EPNotDiscovered` → `click.ClickException("EP plugin not found:
      {e}. Install the required EP package (e.g. onnxruntime-qnn).")`
    - `EPRegistrationFailed` → `click.ClickException("EP registration
      failed: {e}")`
    - `ValueError` → `click.UsageError(str(e))` (CLI-input validation
      error, exit code 2)
  - `logger.info("Resolved to: %s", ep_device_resolved)` after success.
- `--list` flag path: `list_compilers(provider)` → `list_compilers(
  ep_device_resolved.device)` (still string, but pulled from the typed
  EPDevice).
- Config construction: `WinMLCompileConfig.for_provider(provider)` →
  `WinMLCompileConfig.for_ep_device(ep_device_resolved)`. **New
  factory** required on `WinMLCompileConfig` (per commit body:
  "WinMLCompileConfig.ep_device; CompileContext carries EPDevice").
- Display panel: `Device:` row now uses `ep_device_resolved.device`;
  `Provider:` row uses `ep_device_resolved.ep` (full EP name); the
  optional `EP:` row uses the raw user-supplied short alias if `ep` was
  set.
- Helper `_resolve_compile_provider(device, ep)` deleted (16 lines).

## Symbol-level changes
- Removed imports: `from ..config.precision import _DEVICE_TO_PROVIDER,
  _EP_TO_DEVICE` (per Directive in commit body — these private symbols
  must not be imported outside `session/ep_device.py`).
- Added imports: `from ..session import VALID_DEVICES, resolve_device`;
  `from ..session.ep_device import DeviceNotFound, EPNotDiscovered,
  EPRegistrationFailed`.
- `--device` Click option: type narrowed via `VALID_DEVICES` enumeration
  (sorted, lowercase), default changed `npu → None`, `show_default`
  dropped, help text rewritten.
- `compile()` parameter: `device: str` → `device: str | None`.
- New local `_device_arg` (None | lowercased str).
- New local `ep_device_resolved: EPDevice` carried through the rest of
  the function.
- Deleted module-level function `_resolve_compile_provider`.

## Behavior / contract changes
- **Both `--ep` and `--device` optional** — first invocation with neither
  flag now works (resolver picks the catalog's top entry that's actually
  available on the host, with NPU-bias encoded in catalog order, not in
  CLI code).
- `--device auto` and `--device <unset>` are now equivalent; both feed
  `device=None` to the resolver. Previously `auto` was a special string
  passed to `_DEVICE_TO_PROVIDER` which silently fell back to `qnn`.
- **Choice expansion**: `--device` no longer hard-codes
  `["auto","npu","gpu","cpu"]` — it accepts whatever is in
  `VALID_DEVICES` (derived from the EPDeviceSpec catalog). This admits
  any future device the catalog adds (e.g., NPU variants) without
  editing the CLI.
- **Error UX**: 4 specific exceptions caught and re-raised as
  user-readable Click errors instead of opaque ORT/native tracebacks.
  Notably `EPNotDiscovered` includes the install hint
  (`onnxruntime-qnn`).
- **Down-stream contract**: compile config is built from `EPDevice`
  (typed), not provider string. The whole `CompileContext` chain
  (commit body) now carries `EPDevice`, eliminating the cross-package
  private-symbol import in `CompileStage`.
- Display strings: the Device/Provider rows now reflect the **resolved**
  values (e.g., display always shows full provider name
  `QNNExecutionProvider`, not short `qnn`), whereas previously Provider
  was the user-supplied `ep` lowercased (`"qnn"`). Slight cosmetic
  change for users used to seeing the short name in the panel.
- `if ep:` branch for the `EP:` row still uses the user-supplied raw
  string, so when user explicitly passes `--ep qnn` they still see
  `EP: qnn` and `Provider: QNNExecutionProvider`. When `--ep` is
  omitted, no `EP:` row appears (matches previous behavior).
- `--no-quantize` deprecation notice unchanged.

## Cross-file impact
- Requires `..session` package to publicly export `VALID_DEVICES` and
  `resolve_device` (per commit body, `session/ep_device.py` is the
  source).
- Requires `..session.ep_device` to expose `DeviceNotFound`,
  `EPNotDiscovered`, `EPRegistrationFailed` (3 of the "5 exceptions"
  named in commit body).
- Requires `WinMLCompileConfig.for_ep_device(ep_device: EPDevice) ->
  WinMLCompileConfig` factory (new — commit body confirms
  `WinMLCompileConfig.ep_device` field added).
- Downstream `compile_onnx(model, output_path, config)` is unchanged
  signature-wise — it just consumes the config differently inside.
- Removes the last consumer of `_DEVICE_TO_PROVIDER` and `_EP_TO_DEVICE`
  outside `session/ep_device.py`, fulfilling the Directive in the
  commit body.

## Risks / subtleties
- The bare `except ValueError` will swallow **any** ValueError from
  `resolve_device`, including ones the user did not cause (e.g., a
  bug in the catalog). UsageError → exit code 2; if the resolver
  raises ValueError on a programming error, the user sees a confusing
  "usage error" instead of a traceback.
- `_device_arg = None if (device is None or device.lower() == "auto")
  else device.lower()`: the `.lower()` is now redundant because
  `case_sensitive=False` already normalizes Click input, but it's
  defensive.
- The `--list` path now requires `resolve_device` to succeed even
  when the user just wants to list compilers. If no EP is registered
  on the host, `winml compile --list` fails with `EPNotDiscovered` —
  arguably a regression in pure-CLI listing UX. (Previously,
  `_resolve_compile_provider` would default to `"qnn"` string and
  list against that without checking availability.)
- The `EP:` display row uses the user-supplied string, not
  `short_ep_name(ep_device_resolved.ep)` — so if the user passed
  `--ep openvino` the panel shows `EP: openvino` but
  `Provider: OpenVINOExecutionProvider`. Cosmetic split between user
  intent and resolved name.
- Per commit body, `WinMLSession.compile()` now actually runs
  `ort.ModelCompiler` (Bug A: defer InferenceSession when
  `enable_ep_context=True`; Bug B: free `_build_session_options` +
  `ort.ModelCompiler.compile_to_file`). The legacy instance method is
  deleted. This CLI compile path is the entry point for that fix.
- Hardcoded install hint `"onnxruntime-qnn"` only fits QNN — if the
  missing EP is OpenVINO/VitisAI/CUDA, the hint is misleading.

## Open questions / TODOs surfaced
- Should `winml compile --list` skip resolution entirely (since it's a
  pure listing) and just enumerate catalog entries?
- The `EPNotDiscovered` install hint is QNN-specific
  (`onnxruntime-qnn`); should it be parameterized by the resolved EP?
- `VALID_EPS` (imported from `..config`) and `VALID_DEVICES` (imported
  from `..session`) come from different packages — confirm they stay
  in sync. Commit body says the catalog in `session/ep_device.py` is
  the single source of truth, so `..config.VALID_EPS` may need to be
  derived from there.
- `--ep` Click choice is `sorted(VALID_EPS)` — these are short names;
  user cannot pass `QNNExecutionProvider` directly. Confirmed by the
  examples in the docstring.
- `--no-quantize` is still accepted but emits a deprecation notice —
  unrelated to this commit's scope but worth flagging for follow-up
  removal.
