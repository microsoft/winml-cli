# src/winml/modelkit/commands/build.py

## TL;DR
Replaces the hand-rolled NPU-biased EP auto-selection ladder (linear scan of
hardcoded `["QNNExecutionProvider", "OpenVINOExecutionProvider",
"VitisAIExecutionProvider"]` against `WinMLEPRegistry`) with a single call to
the unified `resolve_device(ep=None, device=None)` resolver. Localized 20-line
swap inside the `build` command; everything else in the 1200-line file is
unchanged.

## Diff metrics
- Lines added: 22
- Lines removed: 20
- Net: +2 (one extra import on the new path, blank-line tweaks)
- New / modified: modified (existing file)

## Role before vs after
Before: `build` command's pre-flight had a private `_select_default_ep()`-style
inline block. When `--ep` was unset, it imported `WinMLEPRegistry`, walked a
hardcoded candidate list, picked the first available registry entry, and
logged the short alias (the registry was queried directly via
`is_ep_available()`).

After: same code path, same trigger (`if ep is None:`), but the body now
delegates to the new `resolve_device(ep=None, device=None)` (the session
package's unified resolver) and unwraps its `EPDevice` result into the short
EP name via `short_ep_name(_auto_ep_device.ep)`. Logged message now includes
the resolved device too (`auto-selecting: qnn (device: NPU)` style).

The downstream call signature (`build_hf_model(..., ep=ep, device=device, ...)`
and the module-mode `_build_modules(..., ep=ep, device=device)` etc.) is
**unchanged** — `ep` and `device` are still passed as separate string
arguments. So this refactor stops at the CLI boundary and does **not**
propagate `EPDevice` deeper through the build pipeline.

## Symbol-level changes
- The auto-select block at lines 364–384 swapped:
  - Removed imports/symbols: `from ..session import WinMLEPRegistry`,
    `WinMLEPRegistry.get_instance()`, `is_ep_available()`, the hardcoded
    `candidate_eps` list, the iterating `for candidate_ep in candidate_eps`,
    and the separate warning-only branch at the end.
  - Added imports/symbols: `from ..session import resolve_device,
    short_ep_name` (local function-scoped import), `_auto_ep_device =
    resolve_device(ep=None, device=None)`, `short_ep_name(...)`.
  - Error handling pattern changed from "registry says none available →
    warn-and-continue" to `try/except Exception → warn-and-continue`; the
    new resolver raises domain-specific exceptions (`DeviceNotFound`,
    `EPNotDiscovered`, etc., per `compile.py`), all swallowed by the
    broad `except Exception`.
  - Log line on success now reports both EP and device.

No function signatures, no Click decorators, no docstring changes. The 17
Click options on the `build` command (`-c`, `-m`, `-o`, `--use-cache`,
`--rebuild`, `--no-quant`, `--no-compile`, `--ep`, `--device`,
`--no-analyze`, `--no-optimize`, `--max-optim-iterations`, `-v`) are
identical to before.

## Behavior / contract changes
- **Auto-select policy now lives in one place** (the EPDeviceSpec catalog's
  ordered tuple — see commit body: "Order encodes deduction preference").
  Build no longer encodes its own NPU-bias hardcode; deduction priority is
  whatever the catalog says.
- The `--ep` flag still accepts an arbitrary string (no `click.Choice`
  validation in `build` — unlike `compile.py` which gates on
  `VALID_EPS`). Downstream `build_hf_model` is the only validator.
- Device is now reported in the log when auto-selected (new info), but
  `--device` remains an independent CLI option that is **not** auto-set
  here — `device` stays at its CLI default (`None`) even after auto-EP
  resolution. So the deduction is one-way: `ep=None → auto ep + auto device`
  in the log only; `device` parameter passed downstream is still whatever
  the user typed.
- Failure mode equivalence: previously, exhausting all 3 candidates emitted
  `"EP unspecified for build, and auto-selection failed. Proceeding without
  EP hints."` Now, if `resolve_device` raises (for any reason — no NPU/GPU,
  ORT plugin DLL not registered, schema bug), the same warning is logged
  with `%s` formatted exception detail, and the build proceeds with `ep
  = None`. Behavior is **less aggressive about logging** (no warning when
  registry is empty and exception is not raised — but new resolver always
  resolves to *something* or raises, so this case shifts to the exception
  path).
- Hardcoded 3-EP candidate list is gone — CUDA/TensorRT/MIGraphX EPs are now
  reachable via auto-select if present in the catalog and on the host.

## Cross-file impact
- Depends on `..session` re-exporting `resolve_device` and `short_ep_name`
  (both new per commit body; `ep_device.py` defines them).
- No longer touches `WinMLEPRegistry` directly from `build` — registry is
  used **inside** `resolve_device` (per the commit body, `register_ep` does
  the fallback to `ort.get_ep_devices()` for built-in EPs).
- Downstream `build_hf_model`, `_build_modules`, `_run_single_build`,
  `_build_hf_pipeline`, `_build_onnx_pipeline`, `_run_optimize_stage`,
  `run_optimize_analyze_loop` still receive `ep: str | None, device: str |
  None` — none migrated to `EPDevice`. So the build pipeline is **still
  string-keyed end-to-end**, only the CLI auto-select source changed.

## Risks / subtleties
- Broad `except Exception` swallows everything from `resolve_device` —
  including `DeviceNotFound`, `EPNotDiscovered`, `EPRegistrationFailed`,
  and bare `ValueError`. Unlike `compile.py` which surfaces these with
  remediation hints, build silently proceeds without EP hints. If the
  user has no NPU and asked for `winml build` without `--ep`, they get
  a CPU-targeted build with only a `logger.warning` (no Rich panel) —
  matches prior behavior but loses the new error UX.
- The `device` CLI parameter is **not** auto-set even when `resolve_device`
  succeeds — only `ep` is filled in. So a user running `winml build -m
  ... --use-cache` ends up with `ep="qnn"` (auto) and `device=None`
  passed to `build_hf_model`. Downstream code must tolerate that.
- Inconsistency with `compile.py`: compile resolves at the CLI boundary
  and hands `EPDevice` downstream; build resolves at the CLI boundary and
  hands strings downstream. The commit body explicitly migrated
  "compiler/stages/compile.py" but not the analyzer/build pipeline.
- `short_ep_name` is a new helper from `session.ep_device` — coupling
  point: if its canonicalization rules change, build's `--ep` value
  changes shape too.

## Open questions / TODOs surfaced
- Should the `device` CLI value be filled in symmetrically when `--device`
  is omitted but `resolve_device` succeeds? Currently a discrepancy:
  ep gets auto-filled, device does not.
- Should build adopt `EPDevice` end-to-end (commit body says it migrated
  `compiler/stages/compile.py` but not `build_hf_model`)? The commit
  leaves this as a partial migration.
- Should `except Exception` be narrowed to the documented resolver
  exceptions (`DeviceNotFound`, `EPNotDiscovered`, `EPRegistrationFailed`)
  to match `compile.py`'s remediation-hint UX?
- `--device` Click option says `"Default: NPU."` (line 277) — stale help
  text now that the catalog-driven resolver decides (no longer
  NPU-biased).
