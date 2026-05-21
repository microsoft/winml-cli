# src/winml/modelkit/session/monitor/qnn/viewer.py

## TL;DR
New thin shell-out wrapper around the `qnn-profile-viewer.exe` post-processing binary from the Qualcomm QNN SDK. Locates the SDK via `QNN_SDK_ROOT` env var, finds the viewer EXE under `<sdk>/bin/<arch>/`, and runs it in two modes: basic CSV and detail QHAS (with schematic + optrace-reader config). Both run paths catch `CalledProcessError` and `FileNotFoundError`, log, and return `None` — never raise — so `QNNMonitor._try_qhas` can degrade cleanly to `basic_fallback`.

## Diff metrics
- Lines added / removed: **+206 / 0**
- New / modified: **new file** (succeeds an analogous wrapper in the deleted `optracing/qnn/` tree)

## Role before vs after
- **Before**: No file at this path. Equivalent shell-out lived in the now-deleted `optracing/qnn/` tree.
- **After**: Public sibling of `_internal.py` inside the `qnn` subpackage — `viewer.py` is **not** `_`-prefixed, so it counts as part of the `qnn/` public surface (though `qnn/__init__.py` does not re-export it). `qnn_monitor.py` imports it directly: `from .qnn.viewer import find_qnn_sdk, run_qhas_viewer`.

## Symbol-level changes
- Module constant `_DEFAULT_CONFIG: dict[str, Any]` — fixed feature set for QHAS post-processing (`qhas_json`, `qhas_schema`, `htp_json`, `runtrace`, `memory_info`, `traceback`, `enable_input_output_flow_events`, `enable_sequencer_flow_events`, all `True`).
- `find_qnn_sdk() -> Path | None` (public) — reads `QNN_SDK_ROOT`; returns `None` when unset or pointing at a non-directory. Documented as "the" trigger for `basic_fallback`.
- `_find_viewer_exe(sdk_root=None) -> Path | None` (private) — searches `<sdk>/bin/<arch>/qnn-profile-viewer.exe` (every arch subdirectory), then falls back to `<sdk>/bin/qnn-profile-viewer.exe`. Auto-calls `find_qnn_sdk()` when `sdk_root` is None.
- `run_basic_viewer(qnn_log, output, *, sdk_root=None) -> Path | None` (public) — runs `qnn-profile-viewer --input_log <log> --output <csv>`. Returns the CSV path on success, `None` otherwise.
- `run_qhas_viewer(qnn_log, schematic, output, config=None, *, sdk_root=None) -> Path | None` (public) — writes an `optrace_config.json` next to `output`, then runs the viewer with `--reader optrace --schematic ... --config ...`. Returns the QHAS JSON path on success, `None` otherwise.

## Behavior / contract changes
- **Always silent on failure**: every failure mode (no viewer, missing schematic, viewer returncode non-zero, viewer missing post-discovery) is logged at `warning`/`error` and returns `None`. Caller (`QNNMonitor._try_qhas`) treats `None` as "fall back to basic CSV with status='basic_fallback'".
- **Config file write side-effect** in `run_qhas_viewer`: the function unconditionally writes `<output.parent>/optrace_config.json`. The file is left on disk (no cleanup), which matches the broader "artifacts persist for inspection" stance from `QNNMonitor`.
- **No CWD mutation**: matches PRD C-5 / FR-12 — `subprocess.run` inherits the caller's CWD; viewer outputs land at the explicit `--output` path.
- **Subprocess flags**: `check=True, capture_output=True, text=True`. Hostile binary stdout/stderr can grow unbounded in memory in pathological cases; not parameterised.
- **`# noqa: S603`** marks both `subprocess.run` calls as intentionally trusted input (the command paths come from `_find_viewer_exe`, not user input).

## Cross-file impact
- Only direct importer in source: `qnn_monitor.py` — `from .qnn.viewer import find_qnn_sdk, run_qhas_viewer`. (Not `run_basic_viewer` — the basic path in `QNNMonitor` parses the EP's profiling CSV directly, not the viewer-converted one.)
- Not re-exported by `qnn/__init__.py` despite being a non-`_` module — it's accessible only via the fully-qualified `winml.modelkit.session.monitor.qnn.viewer.*` import path.
- Env-var coupling: `QNN_SDK_ROOT`. No other module reads it (verified by the design docs and import locality).

## Risks / subtleties
- `_find_viewer_exe` iterates `bin/`'s children and returns the **first** match — non-deterministic if multiple arch directories contain the viewer (unlikely in practice, but the QNN SDK has been known to ship multi-arch bins).
- `run_basic_viewer` is unused in the current monitor pipeline (no source-tree caller), making it dead code at HEAD. Likely kept for future parity / external consumers, but worth flagging.
- `run_qhas_viewer` writes `optrace_config.json` next to `output` and never cleans it up; if two concurrent monitors share the same output dir they race on this file. The current `QNNMonitor` design uses per-monitor tempdirs, so the race is theoretical, but the API itself permits it.
- The function-doc says "schematic" must exist (`if not schematic.is_file()` early return), but `qnn_monitor._try_qhas` already validated this via `_find_schematic()` — defensive double-check, not a contract change.
- `subprocess.run(..., capture_output=True)` buffers entire stdout/stderr in memory; on a very long log this could be noticeable. Not parameterised.
- The viewer EXE search is Windows-only (`.exe` suffix); the rest of the package is platform-agnostic. Acceptable today (QNN EP is Windows-on-Snapdragon) but a hard assumption.

## Open questions / TODOs surfaced
- `run_basic_viewer` has no caller — keep, delete, or wire it in as an alternative basic-mode path?
- Should `viewer.py` be renamed `_viewer.py` to match `_internal.py`'s privacy stance? Either rename and re-export selectively, or leave public and explicitly document the public surface in `qnn/__init__.py`.
- `_DEFAULT_CONFIG` is module-scope and mutable in principle — callers passing `config=None` get a reference to the same dict each time. Not a current bug (no one mutates it) but a foot-gun for future callers.
- No timeout on `subprocess.run` — a hung viewer hangs the parent process. Add an explicit timeout?
