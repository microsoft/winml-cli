# src/winml/modelkit/session/monitor/qnn/viewer.py

## TL;DR
Thin shell-out wrapper around `qnn-profile-viewer.exe` from the Qualcomm QNN SDK. Locates the SDK strictly via the `QNN_SDK_ROOT` env var (no auto-discovery), finds the viewer EXE under `<sdk>/bin/<arch>/`, and runs it in two modes: basic CSV and detail QHAS (with schematic + optrace-reader config). All failure paths return `None` so `QNNMonitor._try_qhas` degrades cleanly to `basic_fallback`.

## Diff metrics
- Lines added / removed: **+206 / 0** at this path (new file in the squash)
- Net change vs the a509a67 baseline (the pre-squash version of the same logical file at `optracing/qnn/viewer.py`): **−24 / +21** ≈ the 49-line edit count, concentrated entirely in `find_qnn_sdk()`'s resolution policy.
- New / modified: **new file** at this path; the predecessor lived at `optracing/qnn/viewer.py` until the relocate commit `14b7c3e5`, then was deleted by `5f86d9e8`. The squash records the file as net-new at the destination.

## Role before vs after
- **Before (a509a67 baseline)**: `find_qnn_sdk()` checked `QNN_SDK_ROOT`, then walked a hardcoded `_COMMON_SDK_PATHS = [r"D:\QC", r"C:\Qualcomm\AIStack\qairt"]` list and returned the highest-versioned child containing `bin/`. The two `run_*_viewer` warning messages also mentioned "set QNN_SDK_ROOT to enable detail mode (falling back to basic CSV)".
- **After (774f121d)**: `find_qnn_sdk()` is env-var-only — no `_COMMON_SDK_PATHS` constant, no version-sorted directory walk. Warning strings are shortened to "qnn-profile-viewer not found; (falling back to basic CSV)" with the env-var hint kept. The result: the function is a pure `QNN_SDK_ROOT` reader — no Windows-path heuristics in the source tree.

## Symbol-level changes
- Module constant `_DEFAULT_CONFIG: dict[str, Any]` — unchanged from baseline (`qhas_json`, `qhas_schema`, `htp_json`, `runtrace`, `memory_info`, `traceback`, `enable_input_output_flow_events`, `enable_sequencer_flow_events`, all `True`).
- **Removed** module constant `_COMMON_SDK_PATHS: list[str]` (was `[r"D:\QC", r"C:\Qualcomm\AIStack\qairt"]` in baseline).
- `find_qnn_sdk() -> Path | None` (public) — **simplified**: reads only `QNN_SDK_ROOT`, returns `None` when unset or not a directory. The fallback walk over `_COMMON_SDK_PATHS` was removed.
- `_find_viewer_exe(sdk_root=None) -> Path | None` (private) — unchanged: iterates `<sdk>/bin/<arch>/qnn-profile-viewer.exe` then falls back to `<sdk>/bin/qnn-profile-viewer.exe`.
- `run_basic_viewer(qnn_log, output, *, sdk_root=None) -> Path | None` (public) — unchanged behaviour; warning text restored to the env-var hint shape.
- `run_qhas_viewer(qnn_log, schematic, output, config=None, *, sdk_root=None) -> Path | None` (public) — unchanged behaviour; warning text restored to the env-var hint shape.

## Behavior / contract changes
- **Auto-discovery removed**: `find_qnn_sdk()` no longer walks `D:\QC` or `C:\Qualcomm\AIStack\qairt`. Detail-mode QHAS now requires an explicit `QNN_SDK_ROOT` env var. Any deployment that was implicitly relying on the hardcoded paths will now fall back to basic CSV.
- **Always silent on failure**: every failure mode (no viewer, missing schematic, viewer returncode non-zero, viewer missing post-discovery) is logged at `warning`/`error` and returns `None`. Caller (`qnn_monitor.py:558-561`) treats `None` from `find_qnn_sdk` as "skip QHAS"; `None` from `run_qhas_viewer` similarly.
- **Config file write side-effect** in `run_qhas_viewer`: writes `<output.parent>/optrace_config.json` (no cleanup, matches the broader "artifacts persist for inspection" stance).
- **Subprocess flags**: `check=True, capture_output=True, text=True` with `# noqa: S603` (trusted input from `_find_viewer_exe`).

## Cross-file impact
- Only direct importer: `src/winml/modelkit/session/monitor/qnn_monitor.py:31` — `from .qnn.viewer import find_qnn_sdk, run_qhas_viewer`. Used at `qnn_monitor.py:558` and `:564`.
- `run_basic_viewer` is **not imported anywhere** in `src/` or `tests/` (verified by grep) — `QNNMonitor`'s basic path parses the EP's profiling CSV directly via `parse_qnn_profiling_csv`, not the viewer-converted CSV.
- Env-var coupling: `QNN_SDK_ROOT`. The squash makes this the *only* discovery channel, so the env var is now load-bearing for detail-mode QHAS.
- Not re-exported by `qnn/__init__.py` — accessible only via the fully-qualified `winml.modelkit.session.monitor.qnn.viewer.*` import path.

## Risks / subtleties
- **Auto-discovery removal is a behaviour change for developer machines**: anyone with QNN SDK installed at `D:\QC\<version>` or `C:\Qualcomm\AIStack\qairt\<version>` but no `QNN_SDK_ROOT` set will silently drop from detail to basic mode after this squash. The warning message still says "set QNN_SDK_ROOT to enable detail mode", which is now the only path — that's consistent — but earlier docs / muscle memory may suggest the SDK is auto-found.
- `_find_viewer_exe` iterates `bin/`'s children and returns the **first** match — non-deterministic if multiple arch directories contain the viewer (QNN SDK has shipped multi-arch bins historically).
- `run_basic_viewer` has no caller. Either dead code or future external surface; the relocate didn't prune it.
- `run_qhas_viewer` writes `optrace_config.json` next to `output` with no cleanup; two concurrent monitors sharing an output dir would race. `QNNMonitor` uses per-monitor dirs, so theoretical only.
- `subprocess.run(..., capture_output=True)` buffers entire stdout/stderr in memory; no timeout — hung viewer hangs parent.
- The viewer EXE search is Windows-only (`.exe` suffix); the rest of the package is platform-agnostic. Acceptable today (QNN EP is Windows-on-Snapdragon) but a hard assumption.

## Open questions / TODOs
- Removing `_COMMON_SDK_PATHS` is a quiet UX regression for developers. Was this intentional (env-var-only is the supported contract) or accidental? If intentional, the design doc should call out the breaking change explicitly.
- `run_basic_viewer` — keep, delete, or wire in as an alternative basic-mode path? The squash retained it despite no caller.
- Should `viewer.py` be renamed `_viewer.py` to match `_internal.py`'s privacy stance, or should `find_qnn_sdk` / `run_qhas_viewer` be re-exported from `qnn/__init__.py` to make the public surface explicit?
- `_DEFAULT_CONFIG` is a module-scope mutable dict shared across calls (callers passing `config=None` share the same reference). Not a current bug but a foot-gun.
- No timeout on `subprocess.run` — a hung viewer hangs the parent process indefinitely.

## Simplification opportunities
- **Delete `run_basic_viewer`** (~50 lines). No source-tree caller exists; the basic path in `QNNMonitor` parses the EP CSV directly. If a future caller needs viewer-converted CSV they can resurrect from git.
- **Inline `_find_viewer_exe` into `run_qhas_viewer`**: it's called twice (once per public runner). If `run_basic_viewer` is deleted, the helper has a single caller and the 18-line search loop can move into `run_qhas_viewer`'s prelude — closer to where the failure-warning is emitted, fewer indirection hops.
- **Wrap `_DEFAULT_CONFIG` in a factory** (`def _default_config() -> dict[...]`) so each call gets a fresh dict — closes the shared-mutable-state foot-gun without behaviour change.
- **Add a `timeout=` kwarg to both `subprocess.run` calls** (default ~120s) to prevent a hung viewer from blocking the parent monitor indefinitely.
- **Hoist the duplicated `viewer is None` warning** into a small helper — both runners emit verbatim-identical 3-line warnings; collapsing them saves ~6 lines and ensures the message stays consistent across modes.
