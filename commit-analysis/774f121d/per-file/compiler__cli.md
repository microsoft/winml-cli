# src/winml/modelkit/compiler/cli.py

## TL;DR
Two-line widening of the `--ep` allowlist: imports `VALID_EPS` from the new `..session` facade and feeds `sorted(VALID_EPS)` into the click `Choice` instead of the hand-maintained four-element list `["qnn", "cpu", "cuda", "dml"]`. Everything else is byte-identical to its pre-state. **This file is also collateral damage of the same squash: the `from .configs import (CalibrationConfig, EPConfig, QDQConfig, WinMLCompileConfig)` import at the top is now broken at module-load time because `CalibrationConfig` and `QDQConfig` were deleted from `configs.py` in this commit (see commit body: "Quantization concerns ... moved to WinMLQuantizationConfig in modelkit.quant.config"). The sub-CLI fails with `ImportError` before any command runs.**

## Diff metrics
- 2 lines changed (1 added, 1 modified):
  - `+from ..session import VALID_EPS` (new import, line 13).
  - `type=click.Choice(sorted(VALID_EPS))` replaces `type=click.Choice(["qnn", "cpu", "cuda", "dml"])` on the `--ep` option (line 54).
- No other lines moved. The file is otherwise byte-identical to its pre-state — including the leftover `CalibrationConfig` / `QDQConfig` import and the `WinMLCompileConfig(qdq_config=..., calibration_config=...)` constructor call which both reference symbols that no longer exist.

## Role before vs after
- **Before:** Self-contained click group with a tiny, hand-maintained EP allowlist. Knew only 4 EPs (qnn/cpu/cuda/dml), so newer EPs (openvino, vitisai, tensorrt, migraphx, nv_tensorrt_rtx) were silently unreachable through this CLI.
- **After:** Delegates EP allowlist to the session catalog. Now exposes every short name in `VALID_EPS` (≥8 EPs: cpu, cuda, dml, migraphx, nv_tensorrt_rtx, openvino, qnn, vitisai — driven by `_SHORT_TO_FULL`/`EP_DEVICE_SPECS`), with no manual maintenance.

## Symbol-level changes
- **Added import:** `from ..session import VALID_EPS`.
- **No new functions/classes**; the `compile`, `calibrate`, `info`, `list_providers` commands keep their signatures and bodies.
- **Mutated literal:** the `type=` argument of `--ep` on `compile`.
- **Stale imports / call sites (NOT updated by this commit, but should have been):**
  - `from .configs import (CalibrationConfig, EPConfig, QDQConfig, WinMLCompileConfig)` (line 14) — `CalibrationConfig` and `QDQConfig` no longer exist in `configs.py`. `ImportError` at module load.
  - `qdq_config = QDQConfig(...)`, `calibration_config = CalibrationConfig(...)`, and `WinMLCompileConfig(qdq_config=..., calibration_config=...)` (lines 179-196) — all reference removed symbols / fields.

## Behavior / contract changes
- **Accepted EP values widen** (assuming the import-error is fixed). Users can now pass `--ep openvino|vitisai|migraphx|nv_tensorrt_rtx|cuda|cpu|dml|qnn` to `python -m winml.modelkit.compiler compile`; previously click rejected anything outside the original 4.
- **Help-text ordering changes** to alphabetical (`sorted(VALID_EPS)`).
- **Default remains `"qnn"`.** Still passes `--ep` through to `EPConfig.provider` as a free-form string; nothing here calls `resolve_device` (that wiring lives in `commands/compile.py` for the top-level CLI).
- **Failure-mode contract:** invalid `--ep` still raises `click.BadParameter` at parse time, just based on a larger set.
- **Sub-CLI is dead.** As shipped on this commit, `python -m winml.modelkit.compiler compile ...` fails at import (`ImportError: cannot import name 'CalibrationConfig' from 'winml.modelkit.compiler.configs'`). Verified by direct import in the venv.

## Cross-file impact
- Pulls `VALID_EPS` from `..session` (new public re-export from `session/ep_device.py`); previously this constant lived in `config/precision.py` as `VALID_EPS = frozenset(_EP_TO_DEVICE.keys())`, which was deleted in earlier sweep work. The session package is the new single source of truth, consistent with the commit's "no private symbol imports outside session/".
- No callers of `compiler.cli` are affected (no symbols added/removed from the public surface — the sub-CLI is only invoked via `python -m winml.modelkit.compiler`).

## Risks / subtleties
- **The sub-CLI is non-functional after this squash** (see TL;DR). The top-level `winml compile` entry point (`commands/compile.py`) is the supported path; this sub-CLI appears to have been left behind during the quant-config split (#241 reference in `configs.py`'s docstring). If anyone tries to use it for a smoke test, they'll hit the `ImportError` immediately.
- **EPs in `VALID_EPS` may not be runnable on the host.** click only validates spelling; if the user picks an EP that isn't installed / discoverable, the failure surfaces later inside `compile_onnx` → `WinMLSession`. Remediation hints (`WinMLEPNotDiscovered` / `DeviceNotFound`) are implemented in the top-level `commands/compile.py`, not here.
- **Sub-CLI diverged from top-level CLI on EPDevice wiring.** It still has no `--device` flag and does not call `resolve_device` at the boundary, so even after the `ImportError` is fixed it cannot exercise the EPDevice threading path. Only the `WinMLCompileConfig.from_dict({...})` → `resolve_device(EPDeviceTarget(ep=ep_str, device="auto"))` fallback in `CompileStage.process` (line 81-83 of `stages/compile.py`) will run.

## Open questions / TODOs surfaced
- **Bug**: fix the `CalibrationConfig` / `QDQConfig` import and the `qdq_config=...` / `calibration_config=...` constructor call. Likely the sub-CLI should either (a) drop the quantization options entirely (since quantize is now its own pipeline), or (b) import from `winml.modelkit.quant.config` (the new home per the docstring on `configs.py`).
- **Decision**: should `python -m winml.modelkit.compiler compile` mirror the `--device` flag from `commands/compile.py`? Today the two entry points have diverged behaviorally re. EPDevice threading.
- **Decision**: should this sub-CLI be deleted altogether? It duplicates `commands/compile.py` and is the only consumer of `WinMLCompileConfig`'s now-removed quantization fields.

## Simplification opportunities
- **Delete the entire file.** It is unreachable from the top-level `winml` CLI, ships broken, and duplicates `commands/compile.py` semantically. The `info` and `list-providers` subcommands could move to `commands/sys.py` if anyone uses them.
- **If kept:** drop `quantize` / `calibration_*` / `weight_type` / `activation_type` / `per_channel` flags (the `quant.config` package owns those now) and replace lines 175-196 with `config = WinMLCompileConfig(ep_config=ep_config, verbose=verbose)`. That's a 22-LOC delete plus the import fix.
- The `sorted(VALID_EPS)` call is evaluated at module-import time inside the `@click.option(...)` decorator argument; if `VALID_EPS` ever becomes runtime-mutable, this would silently freeze the snapshot. Today it's a `Final[frozenset[str]]` so this is fine.
