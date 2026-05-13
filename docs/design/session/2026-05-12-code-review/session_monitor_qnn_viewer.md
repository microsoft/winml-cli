# Review: `src/winml/modelkit/session/monitor/qnn/viewer.py`

**Status:** new file (relocated from `optracing/qnn/viewer.py`)
**Lines added/removed:** 206+ / 0-
**Diff command:** `git diff 1bea4cf..HEAD -- src/winml/modelkit/session/monitor/qnn/viewer.py`

## 1. Purpose of this file

Thin wrapper around the `qnn-profile-viewer.exe` post-processing tool shipped in the QNN SDK. Provides two functions: `run_basic_viewer` (CSV output) and `run_qhas_viewer` (QHAS JSON output with optrace reader), plus `find_qnn_sdk` for SDK root detection and `_find_viewer_exe` for binary location. Called by `QNNMonitor._try_qhas()` for the live profiling path; not called on the offline `parse_existing_artifacts` path.

## 2. Changes summary

- Relocated from `optracing/qnn/viewer.py` to `session/monitor/qnn/viewer.py`.
- No functional changes.
- `# noqa: S603` on both `subprocess.run` calls (trusts arguments built internally, not from user input).

## 3. Per-symbol review

### `find_qnn_sdk()`

- **Role:** Resolve QNN SDK root from the `QNN_SDK_ROOT` environment variable.
- **Signature:** `def find_qnn_sdk() -> Path | None:`
- **Behavior:** Returns `None` when `QNN_SDK_ROOT` is unset or points to a non-existent directory. This degrades detail mode to `status="basic_fallback"` (FR-5).
- **Risks / concerns:** Only checks `QNN_SDK_ROOT`. If the QNN SDK is installed in a standard system location (e.g. `C:\Qualcomm\QNN\<version>\`) without `QNN_SDK_ROOT` being set, it would not be found. This is by design — the SDK location is not standard enough to probe heuristically. The behavior is documented in the function docstring.
- **Tests:** `tests/unit/session/monitor/qnn/test_viewer.py`.

---

### `_find_viewer_exe()`

- **Role:** Locate `qnn-profile-viewer.exe` within the SDK directory tree.
- **Signature:** `def _find_viewer_exe(sdk_root: Path | None = None) -> Path | None:`
- **Behavior:** Searches `<sdk_root>/bin/<arch>/qnn-profile-viewer.exe` by iterating arch subdirectories, with a fallback to `<sdk_root>/bin/qnn-profile-viewer.exe`. Returns `None` if not found.
- **Risks / concerns:** `bin_dir.iterdir()` iterates ALL entries in `bin/` (not just directories). If the `bin/` directory contains files at the top level, the code checks `file_path / "qnn-profile-viewer.exe"` where `file_path` is not a directory — this silently skips (the path would not be a file). No performance concern (the SDK `bin/` is typically small). On a malformed SDK installation (e.g. a symlink in `bin/` pointing to a non-directory), `iterdir()` would follow the symlink if it points to a directory, potentially searching unexpected locations. Acceptable risk for a development tool.
- **Tests:** `test_viewer.py`.

---

### `run_basic_viewer()`

- **Role:** Shell out to `qnn-profile-viewer` to produce a basic CSV.
- **Signature:** `def run_basic_viewer(qnn_log, output, *, sdk_root=None) -> Path | None:`
- **Behavior:** Builds a command list, runs via `subprocess.run(check=True, capture_output=True, text=True)`. Returns the output path if it exists post-run, `None` otherwise. Logs errors at WARNING/ERROR.
- **Risks / concerns:**
  1. `# noqa: S603` suppresses `subprocess` call with a list (not a shell string). This is the correct form — argument injection is not possible since all arguments are derived from internal `Path` objects, not user input. The suppression is justified.
  2. `CalledProcessError` is caught and the `exc.stderr` is logged. If stderr contains non-UTF-8 characters (rare but possible on some Windows locales for tool error messages), `text=True` may raise a `UnicodeDecodeError` from `subprocess.run`. This would not be caught by the `CalledProcessError` handler and would propagate to `_try_qhas`'s `except Exception`, resulting in `(None, None, None)` with a WARNING log. Acceptable as an edge case.
  3. No timeout on `subprocess.run`. A hung `qnn-profile-viewer` would hang `monitor.__exit__` indefinitely. The PRD does not specify a timeout for the viewer shell-out, and adding one was apparently considered out of scope. This is a potential reliability risk in adversarial environments.
- **Tests:** `test_viewer.py`.

---

### `run_qhas_viewer()`

- **Role:** Shell out to `qnn-profile-viewer` with the optrace reader to produce QHAS JSON.
- **Signature:** `def run_qhas_viewer(qnn_log, schematic, output, config=None, *, sdk_root=None) -> Path | None:`
- **Behavior:** Writes a `optrace_config.json` to `output.parent` before running the viewer. Passes `--reader optrace --schematic <path> --config <config_path>` arguments. Returns the output QHAS JSON path on success.
- **Risks / concerns:**
  1. `config_path = output.parent / "optrace_config.json"` — this file is written unconditionally each run. If multiple concurrent `QNNMonitor` instances share the same `output.parent` (impossible with unique tempdirs, but possible if user passes the same `output_dir`), the config file would be clobbered. Not a practical risk with unique temp dirs.
  2. Same `subprocess.run` timeout and Unicode-stderr concerns as `run_basic_viewer`.
  3. `config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")` correctly specifies UTF-8 encoding. This is more robust than `report.py`'s `write_text` which omits encoding.
  4. If `output.is_file()` returns `False` after the viewer runs (e.g. the viewer completed with exit code 0 but wrote to a different path), the function returns `None` silently. The caller logs this at DEBUG level.
- **Tests:** `test_viewer.py`.

---

### `_DEFAULT_CONFIG`

- **Role:** Default QHAS post-processing feature flags for the optrace reader.
- **Value:** All features enabled: `qhas_json`, `qhas_schema`, `htp_json`, `runtrace`, `memory_info`, `traceback`, `enable_input_output_flow_events`, `enable_sequencer_flow_events`.
- **Risks / concerns:** Enabling all features maximizes artifact richness but may slow the viewer shell-out significantly for large models. The PRD does not specify a performance budget for the viewer call. Acceptable for the current use case (offline analysis, not latency-critical).
- **Tests:** Not directly tested; used by `run_qhas_viewer`.

## 4. Cross-cutting concerns

**Spec drift:** FR-12 (no `os.chdir`) — this file does not call `os.chdir`. The schematic file location is handled in `QNNMonitor._find_schematic()` (not in this file), so the `os.chdir` constraint is upheld. The docstring in this file does not mention the schematic-location strategy, which is appropriate since that logic belongs to the monitor.

**Information-hiding contract:** `viewer.py` is part of the `session/monitor/qnn/` package and is imported by `qnn_monitor.py` (`from .qnn.viewer import find_qnn_sdk, run_qhas_viewer`). This is the only `src/` importer — verified by grep. The file itself does not import from `_internal`.

**Deferred work:** No TODO markers. The `run_basic_viewer` function is defined but not currently called by `QNNMonitor` (which parses the CSV directly from the path set in `get_provider_options()`). The viewer is only called for the QHAS post-processing step. `run_basic_viewer` appears to be a preserved API for potential future use or standalone scripting.

**EPDevice / ep_name:** Not referenced.

## 5. Confidence level

**High.** The module is a thin subprocess wrapper with appropriate error handling. The main risks are the absent subprocess timeout (potential hang on viewer failure) and the Unicode-stderr edge case. Neither is a current practical concern for QNN EP development environments.

## 6. Verbatim risk inventory

| Severity | Location | Description |
|----------|----------|-------------|
| Medium | `viewer.py:123`, `viewer.py:196` | `subprocess.run(check=True, capture_output=True, text=True)` — no `timeout=` parameter. A hung or non-terminating `qnn-profile-viewer.exe` would block `monitor.__exit__` indefinitely. Consider adding `timeout=60` for robustness. |
| Low | `viewer.py:123`, `viewer.py:196` | `text=True` decodes stderr with the system's default encoding. On Windows locales where the QNN SDK's error messages are not UTF-8, `CalledProcessError.stderr` may raise `UnicodeDecodeError` in the `logger.error` call. This escapes the `CalledProcessError` handler and propagates to `_try_qhas`'s `except Exception`. Safe (returns `(None, None, None)`) but obscures the actual tool error. |
| Low | `viewer.py:79-96` | `_find_viewer_exe` iterates `bin_dir.iterdir()` including non-directory entries. Checking `file_path / "qnn-profile-viewer.exe"` on a non-directory silently produces a non-existent path (no crash), but the iteration over all `bin/` entries is slightly inefficient for large `bin/` directories. |
| Info | `viewer.py:94-124` | `run_basic_viewer` is defined but not called by `QNNMonitor`. Its existence could confuse future maintainers into thinking there are two viewer call paths. A docstring note that this function is for standalone use would clarify. |
