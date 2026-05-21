# src/winml/modelkit/compiler/cli.py

## TL;DR
Tiny surface-area change: the `--ep` option's hard-coded `click.Choice` list (`["qnn", "cpu", "cuda", "dml"]`) is replaced by `sorted(VALID_EPS)` imported from the new `..session` facade. Everything else (calibrate, info, list-providers commands) is unchanged. No `--device` flag is added in this file — the commit body's "winml compile gets --device" claim was implemented in `commands/compile.py` (the top-level `winml` CLI), not in this compiler sub-CLI.

## Diff metrics
- 2 lines changed (1 added, 1 modified):
  - `+from ..session import VALID_EPS` (new import).
  - `type=click.Choice(sorted(VALID_EPS))` replaces `type=click.Choice(["qnn", "cpu", "cuda", "dml"])` on the `--ep` option (line 54).
- No other changes; the file is otherwise byte-identical to its pre-state.

## Role before vs after
- **Before:** Self-contained click group with a tiny, hand-maintained EP allowlist. Knew only 4 EPs (qnn/cpu/cuda/dml), so newer EPs (openvino, vitisai, tensorrt, migraphx) were silently unreachable through this CLI.
- **After:** Delegates EP allowlist to the session catalog. Now exposes every short name in `VALID_EPS` (≥8 EPs: qnn, openvino, vitisai, migraphx, nv_tensorrt_rtx, cuda, tensorrt, dml, cpu — driven by `_SHORT_TO_FULL`/`EP_DEVICE_SPECS`), with no manual maintenance.

## Symbol-level changes
- **Added import:** `from ..session import VALID_EPS`.
- **No new functions/classes**; the `compile`, `calibrate`, `info`, `list_providers` commands keep their signatures and bodies.
- **Mutated literal:** the `type=` argument of `--ep` on `compile`.

## Behavior / contract changes
- **Accepted EP values widen.** Users can now pass `--ep openvino|vitisai|tensorrt|migraphx|nv_tensorrt_rtx` to `python -m winml.modelkit.compiler compile`; previously click rejected them at parse time.
- **Help-text ordering changes** to alphabetical (`sorted(VALID_EPS)`).
- **Default remains `"qnn"`.** Still passes `--ep` through to `EPConfig.provider` as a free-form string; nothing here calls `resolve_device` (that wiring lives in `commands/compile.py` for the top-level CLI).
- Failure-mode contract unchanged: invalid `--ep` still raises `click.BadParameter` at parse time, just based on a larger set.

## Cross-file impact
- Pulls `VALID_EPS` from `..session` (new public re-export from `session/ep_device.py`); previously this constant lived in `config/precision.py` as `VALID_EPS = frozenset(_EP_TO_DEVICE.keys())`, which was deleted in this same commit. The session package is the new single source of truth, consistent with the commit's "Directive: do not import private symbols ... outside session/ep_device.py".
- No callers of `compiler.cli` are affected (no symbols added/removed from the public surface).

## Risks / subtleties
- **EPs in `VALID_EPS` may not be runnable on the host.** click only validates spelling; if the user picks an EP that isn't installed / discoverable, the failure will surface later inside `compile_onnx` → `WinMLSession`. Most CLI-boundary remediation hints described in the commit body (`DeviceNotFound`/`EPNotDiscovered`) are implemented in the top-level `commands/compile.py`, not here — this sub-CLI will still emit a generic traceback (or the styled `Error:` line on line 226) for those cases.
- **Compile sub-CLI is now out of sync with the top-level CLI.** It still has no `--device` flag and does not call `resolve_device` at the boundary, so it cannot exercise the EPDevice threading path. Only the `_EP_TO_DEVICE` fallback in `CompileStage.process` will run for invocations through this CLI.

## Open questions / TODOs surfaced
- Should `python -m winml.modelkit.compiler compile` mirror the `--device` flag from `commands/compile.py`? Today the two entry points have diverged behaviorally re. EPDevice threading.
- Should `--ep` choices for `calibrate` and other subcommands also gain symmetric session-driven validation? They currently don't accept `--ep` at all, but if any do in the future, the same `VALID_EPS` import is now available.
