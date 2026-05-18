# Review: `src/winml/modelkit/session/monitor/qnn_monitor.py`

**Status:** modified (placeholder → full implementation)
**Lines added/removed:** 379+ / 47-
**Diff command:** `git diff 1bea4cf..HEAD -- src/winml/modelkit/session/monitor/qnn_monitor.py`

## 1. Purpose of this file

The concrete `QNNMonitor(EPMonitor)` implementation. Replaces the previous stub (`is_available()` returned `False`, `to_dict()` returned `{"status": "not_implemented"}`). This module owns all QNN-specific knowledge: contributing ORT session/provider options, parsing the profiling CSV and QHAS JSON artifacts produced by QNN EP, resolving op-type labels via the four-layer fallback chain, and exposing the result as a typed `OpTraceResult`. It is the single concrete class that satisfies PRD §4.4 (FR-4).

## 2. Changes summary

- Rewrote `QNNMonitor` from placeholder to full implementation.
- Added `requires_session_teardown = True` and `ep_name = "qnn"` class vars.
- `__init__` now accepts `level`, `output_dir`, and `extra_provider_options`.
- Added `is_available()` with dual-path check (bundled ORT + WinML-registered ORT).
- Added `get_session_options()` returning EPContext caching options.
- Added `get_provider_options()` with owner-enforced `profiling_level` / `profiling_file_path` (C-3).
- `__enter__` adds a double-entry guard.
- `__exit__` calls `_parse_artifacts_safe()` and stores the result.
- Added `result` property override.
- Added `set_onnx_op_types()` override that stores a defensive copy.
- Added `_resolve_op_type()` (four-layer fallback chain).
- Added `_heuristic_op_type()` (leaf-split + `_token_N` strip).
- Added `parse_existing_artifacts()` classmethod for offline analysis.
- Added `_parse_artifacts()`, `_try_qhas()`, `_find_schematic()`, `_parse_artifacts_safe()`, `_make_failure_result()`.
- Removed `to_dict()` and the placeholder `is_available()`.

## 3. Per-symbol review

### `QNNMonitor`

- **Role:** Qualcomm NPU per-op profiler via ORT's QNN EP.
- **Signature:** `class QNNMonitor(EPMonitor):`
- **Behavior:** Registers two ORT hook contributions at construction time, manages a unique per-monitor output directory, parses artifacts on context-manager exit.
- **Invariants:** `requires_session_teardown = True` — `WinMLSession.perf().__exit__` must destroy the ORT session before calling `monitor.__exit__`; this is asserted by the integration test at `tests/unit/session/test_perf_monitor_integration.py`. `ep_name = "qnn"` — `WinMLSession.perf()` validates this against the session's `EPDevice`.

---

### `QNNMonitor.__init__`

- **Role:** Initialize the monitor with profiling level, output directory, and extra provider options.
- **Signature:** `def __init__(self, level="basic", output_dir=None, extra_provider_options=None)`
- **Behavior:** Mints a unique temp directory if `output_dir=None` (via `tempfile.mkdtemp`) and pins `_csv_path` to `<output_dir>/profiling_output.csv`. The temp directory is NOT auto-cleaned — documented intentionally for post-run artifact inspection.
- **Invariants:** NFR-4 (idempotency) — paths are produced at `__init__`, not per-call. NFR-6 (no module-level caches) — no global state is mutated.
- **Risks / concerns:** The temp directory leak is intentional but will accumulate on machines running many profiles without cleanup. The docstring warns the caller explicitly. Callers on CI pipelines that need disk hygiene must pass an explicit `output_dir` they manage.
- **Tests:** `test_qnn_monitor.py::test_ctor_defaults`, `test_ctor_accepts_custom_output_dir`, `test_ctor_rejects_invalid_level`.

---

### `QNNMonitor.is_available()`

- **Role:** Returns `True` iff QNN EP is usable on this system (bundled or WinML-registered).
- **Signature:** `@classmethod def is_available(cls) -> bool:`
- **Behavior:** Two-path check: (1) `"QNNExecutionProvider" in ort.get_available_providers()` for bundled wheels; (2) `ensure_initialized()` + `ort.get_ep_devices()` for WinML-registered ORT. Any failure in the WinML path is caught and logged at WARNING (NFR-2 — no silent failures).
- **Invariants:** FR-8 — must return `True` on any machine where `wmk perf --device npu` currently works.
- **Risks / concerns:** The `getattr(d, "ep_name", None)` call on ORT device objects (`ep_device_py:159`) is a duck-typing probe. If ORT changes the attribute name of `OrtEpDevice`, this silently returns `False` instead of raising. The `logger.warning` in the except handler ensures the failure is observable per NFR-2.
- **Tests:** `test_qnn_monitor.py::test_is_available_via_bundled`, `test_is_available_via_winml`, `test_is_available_neither`, `test_is_available_winml_path_failure_logs_warning`.

---

### `QNNMonitor.get_provider_options()`

- **Role:** Build provider options dict with owner-enforced profiling keys (C-3).
- **Signature:** `def get_provider_options(self) -> dict[str, str]:`
- **Behavior:** Merges `self._extra` first, then assigns `profiling_level` and `profiling_file_path` last. The last-write-wins ordering enforces C-3 (these two keys are never user-overridable). The docstring explains the reason for explicit assignment after `update()` (vs dict literal with duplicate keys, which would trigger ruff F601).
- **Invariants:** C-3 — `profiling_level` and `profiling_file_path` must always reflect the monitor's internal state.
- **Risks / concerns:** If `self._extra` contains a key that shadows a WinML-registered QNN device's pre-configured `backend_path`, it will overwrite the WinML default. The docstring warns about this and says callers who need `backend_path` for the bundled ORT path should pass it via `extra_provider_options`. This is correct behavior (three-layer merge: defaults → user → monitor wins last, per impl-status §2.3).
- **Tests:** `test_qnn_monitor.py::test_get_provider_options_owner_keys_only`, `test_profiling_keys_not_user_overridable`, `test_get_provider_options_idempotent`, `test_extra_provider_options_pass_through`.

---

### `QNNMonitor.__exit__`

- **Role:** Parse artifacts and store result. Never suppresses caller exceptions.
- **Signature:** `def __exit__(self, exc_type, exc_val, exc_tb) -> None:`
- **Behavior:** Calls `_parse_artifacts_safe()` and stores to `self._result`. Implicit `None` return means caller exceptions propagate normally (NFR-5).
- **Invariants:** `self._result` is always set after `__exit__`, even on parse failure (status becomes `"no_data"` or `"parse_failed"`).
- **Risks / concerns:** If `_parse_artifacts_safe()` itself raises unexpectedly (which it shouldn't given the try/except wrapper), `self._result` would remain `None` and the exception would propagate. The `except Exception` in `_parse_artifacts_safe` is intentionally broad to prevent this.
- **Tests:** `test_qnn_monitor.py::test_exit_with_no_csv_reports_no_data`, `test_exit_parse_failure_caught`, `test_exit_does_not_suppress_caller_exception`.

---

### `QNNMonitor.set_onnx_op_types()`

- **Role:** Override of the EPMonitor no-op. Stores a defensive copy of the injected ONNX map.
- **Signature:** `def set_onnx_op_types(self, onnx_op_types: dict[str, str]) -> None:`
- **Behavior:** `self._onnx_op_types = dict(onnx_op_types)`. Defensive copy prevents later caller mutation from corrupting the L1 lookup table. Called by `WinMLSession.perf().__enter__` before `__enter__`.
- **Tests:** `test_qnn_monitor_resolve.py::test_set_onnx_op_types_copies_input`, `test_set_onnx_op_types_overwrites_previous_call`.

---

### `QNNMonitor._resolve_op_type()`

- **Role:** Walk the four-layer fallback chain: L1 ONNX → L2 EP-authoritative → L3 heuristic → L4 raw.
- **Signature:** `def _resolve_op_type(self, op_path: str, ep_authoritative: str | None = None) -> str:`
- **Behavior:** L1 uses a truthy check (`if mapped:`) — an empty-string value in the ONNX map falls through to L2/L3/L4 rather than short-circuiting with `""`. This is the correct defensive behavior documented in FR-14.
- **Invariants:** Never returns an empty string — L4 returns `op_path` verbatim which may be empty only if the caller passes an empty `op_path`. In practice `op_path` is always non-empty from the CSV/QHAS parsers.
- **Risks / concerns:** If `op_path` is empty and the ONNX map doesn't contain `""`, L4 returns `""` verbatim. Not a practical risk since parsers always produce non-empty paths, but worth noting.
- **Tests:** `test_qnn_monitor_resolve.py` (all 15 tests covering all four layers and edge cases).

---

### `QNNMonitor._heuristic_op_type()`

- **Role:** L3 fallback: strip `_token_N` suffix, then leaf-split on `/`.
- **Signature:** `def _heuristic_op_type(self, op_path: str) -> str:`
- **Behavior:** Strips `_TOKEN_SUFFIX` regex, strips whitespace, splits on the trailing `/`, strips whitespace around the leaf. Falls back to the cleaned input for trailing-slash paths to avoid returning an empty string.
- **Invariants:** The strip semantics match `_split_op_event_id` in `_internal.py` — both use the same `_TOKEN_SUFFIX` regex imported from `_internal`. Shared regex eliminates divergence risk.
- **Tests:** `test_qnn_monitor_resolve.py::test_heuristic_*` (7 tests).

---

### `QNNMonitor.parse_existing_artifacts()`

- **Role:** Classmethod for offline analysis of pre-existing CSV/QHAS artifacts.
- **Signature:** `@classmethod def parse_existing_artifacts(cls, level, artifacts, onnx_op_types=None) -> OpTraceResult:`
- **Behavior:** Constructs a monitor instance with `output_dir = csv_path.parent`, then overrides `_csv_path` to honor the caller's explicit path (which may differ from the `profiling_output.csv` default). Routes through `_parse_artifacts_safe` for consistent parse-failure handling.
- **Invariants:** Must raise `ValueError` if `artifacts` lacks `"csv"`. The `instance._result = result` assignment at the end keeps the instance internally consistent per the `M-2` note in the comment.
- **Risks / concerns:** The `instance._csv_path = csv_path.resolve()` assignment mutates a private attribute of the freshly-constructed instance, bypassing the constructor's path-pinning logic. This is intentional (the classmethod IS the constructor for the offline case) but is a subtle internal consistency risk if the classmethod is extended.
- **Tests:** `test_qnn_monitor_parse_existing.py` (9 tests).

---

### `QNNMonitor._parse_artifacts()`

- **Role:** Parse CSV (always) and optionally QHAS (detail mode).
- **Signature:** `def _parse_artifacts(self, qhas_override=None) -> OpTraceResult:`
- **Behavior:** Checks CSV existence with a 50ms retry (R-2 mitigation for Windows file-handle flush lag). Builds operators from CSV data using `cycle_to_us` ratio derived from `accel_execute_cycles` and `accel_execute_us`. For detail mode, calls `_try_qhas()` and replaces operators with QHAS-enriched data if available; otherwise sets `status="basic_fallback"`.
- **Risks / concerns:**
  1. **`cycle_to_us` precision loss:** `total_cycles` and `accel_us` are both truncated to `int` before computing the ratio (`int(meta.get(...) or 0)`). If the CSV metadata contains float strings (e.g. `"12345.6"`), `int()` truncates silently. The CSV format appears to always emit integers for these fields, but this is an assumption.
  2. **Single 50ms retry:** The R-2 mitigation is one retry only. On heavily loaded machines or network-mounted paths, the CSV flush lag might exceed 50ms. This is a known acceptable risk per design (PRD R-2/M-2).
  3. **`model=None`:** The `OpTraceResult` is constructed with `model=None` here. The caller (`WinMLSession.perf()`) is expected to set the model name post-hoc or accept `None`. This is correct per the relaxed `model: str | None` type.
- **Tests:** `test_qnn_monitor.py::test_exit_with_no_csv_reports_no_data`, `test_detail_mode_falls_back_to_basic_when_qhas_unavailable`; `test_qnn_monitor_parse_existing.py`.

---

### `QNNMonitor._find_schematic()`

- **Role:** Locate `*_schematic.bin` without calling `os.chdir()` (FR-12).
- **Behavior:** Searches `_output_dir` first, then falls back to a glob of the process CWD. The CWD fallback rejects schematics older than the CSV (mtime gate with 5s tolerance for clock skew) to prevent stale CI artifacts from silently corrupting results.
- **Risks / concerns:** The 5s mtime tolerance is hardcoded. On machines where the filesystem clock resolution is coarser than 5s (e.g. FAT32 at 2s), this could accept a slightly stale schematic. For the standard NTFS dev environment this is not a concern. The `logger.warning` on CWD fallback ensures the non-ideal path is observable.
- **Tests:** `test_qnn_monitor.py::test_find_schematic_rejects_stale_cwd_candidate`, `test_find_schematic_accepts_fresh_cwd_candidate`, `test_find_schematic_prefers_output_dir_over_cwd`.

---

### `QNNMonitor._try_qhas()`

- **Role:** Attempt QHAS post-processing; return `(summary, operators, path)` or `(None, None, None)` on failure.
- **Behavior:** On the live path, locates the QNN log, schematic, and SDK root, shells out to `run_qhas_viewer`, then calls `parse_qhas`. On the offline path (`qhas_override` provided), skips the shell-out. Never raises — all failures are logged at DEBUG/WARNING and return `(None, None, None)`.
- **Risks / concerns:** `qnn_logs[0]` picks the first log file found if multiple `*_qnn.log` files exist (e.g. from a failed prior run). No deterministic tie-breaking. This could pick an older log from a previous run if output dirs overlap. Since the output dir is unique per `__init__` (tempdir), this should not be an issue in practice.
- **Tests:** `test_qnn_monitor.py::test_detail_mode_falls_back_to_basic_when_qhas_unavailable`; QHAS success path covered by `test_qnn_monitor_parse_existing.py::test_parse_existing_artifacts_detail_qhas_override`.

## 4. Cross-cutting concerns

**Spec drift:**
- `get_session_options()` intentionally does NOT set `"session.disable_cpu_ep_fallback": "1"` — the docstring explains why (WinML-registered QNN uses Q/DQ-on-CPU for the EPContext boundary nodes; disabling CPU fallback would incorrectly reject this valid partition). This is a deliberate deviation from the "no silent CPU fallback" intent in PRD §4.1, justified inline.
- `model=None` in `_make_failure_result` and `_parse_artifacts` is correct per the relaxed `model: str | None` type.

**Information-hiding contract:** `qnn_monitor.py` imports `_TOKEN_SUFFIX`, `parse_qhas`, `parse_qnn_profiling_csv` from `.qnn._internal`. This is the ONLY source file in `src/` that does so — verified by the architecture regression test at `tests/unit/architecture/test_qnn_imports.py`. The test confirms no other `src/` module breaches the boundary.

**Deferred work:** No TODO or MIGRATION markers in this file.

**EPDevice / ep_name:** `ep_name = "qnn"` is the short form; `WinMLSession.perf()` applies `expand_ep_name("qnn")` → `"QNNExecutionProvider"` and compares against `ep_device.ep`. This is the correct short-form convention.

## 5. Confidence level

**High** overall. The four-layer resolver, the parse-failure contract, and the double-entry guard are all well-tested. The main residual risks are:

- `int()` truncation of `accel_execute_us` / `accel_execute_cycles` metadata (low practical risk for known CSV formats).
- Single 50ms retry for Windows file-handle flush lag (acceptable per design).
- `qnn_logs[0]` non-deterministic log selection when multiple logs exist (practically avoided by per-init tempdir).

## 6. Verbatim risk inventory

| Severity | Location | Description |
|----------|----------|-------------|
| Medium | `qnn_monitor.py:439-441` | `int(meta.get("accel_execute_cycles", 0) or 0)` — truncates float strings silently. If QNN EP ever emits `"12345.6"` for the cycle count, the `int()` call truncates and the `cycle_to_us` ratio is wrong, producing systematically incorrect `duration_us` values. Should use `round(float(...))` for robustness. |
| Low | `qnn_monitor.py:425-427` | Single 50ms retry for CSV flush lag (R-2). May be insufficient on heavily loaded machines or network-mounted paths. |
| Low | `qnn_monitor.py:501-502` | `qnn_logs[0]` picks the lexicographically first log when multiple `*_qnn.log` files exist. In practice avoided by unique tempdir per `__init__`, but if `output_dir` is user-supplied and shared across runs, an older log could be selected. |
| Low | `qnn_monitor.py:390-394` | `instance._csv_path = csv_path.resolve()` in `parse_existing_artifacts` mutates a private attribute post-construction. No other caller should do this, but the pattern is fragile if the classmethod is extended. |
| Info | `qnn_monitor.py:115` | `self._result` initialized to `None` in `__init__`, but the base-class `result` property uses `getattr(self, "_result", None)`. Since `QNNMonitor` overrides `result` directly, the base-class `getattr` is never called. The direct override is correct but redundant with the base-class default. |
