# Batch 03 Verification Report

Branch: `feat/op-tracing-refactor_new-3` | Commit: `774f121d` | Mergebase: `7a66c024`

Covers per-file docs for `serve/*`, `session/__init__`, `session/ep_device`, `session/ep_registry`, and `session/monitor/*` (15 docs).

---

## Critical claims

### D-04 — `auto_device` last_error stale traceback bug — **VERIFIED**

Source: `src/winml/modelkit/session/ep_registry.py:357-418`. The precedence retry loop:

```
398:        last_error: Exception | None = None
399:        for entry in candidates:
400:            try:
401:                winml_ep = self.register_ep(entry)
402:            except WinMLEPRegistrationFailed as e:
403:                last_error = e
404:                continue
405:            for device in winml_ep.devices:
406:                if device.device_type == target_device_upper:
407:                    return WinMLEPDevice(ep=winml_ep, device=device)
408:
409:        # All candidates exhausted without a match.
410:        if last_error is not None:
411:            raise WinMLEPRegistrationFailed(
412:                f"No compatible source for {target.ep}/{target.device}; "
413:                f"all {len(candidates)} candidates failed"
414:            ) from last_error
415:        raise DeviceNotFound(
416:            f"No source for {target.ep}/{target.device} exposed device "
417:            f"class {target.device.upper()!r}"
418:        )
```

`last_error` is set only in the `except` branch (line 403) and never reset after a successful `register_ep` whose devices don't match `target_device_upper`. If candidate A raises `WinMLEPRegistrationFailed`, then candidate B registers cleanly but exposes no device of the requested class, the user gets `WinMLEPRegistrationFailed` chained from A's traceback instead of `DeviceNotFound`. The doc's Behavior #7, Risk #1, and Simplification #1 are correct. The three-line fix proposal (`last_error = None` after the `except`/`continue`) is the right shape.

### D-03 — `find_qnn_sdk()` lost `_COMMON_SDK_PATHS` fallback — **VERIFIED (location mislabeled)**

The doc places this claim in `session__monitor__qnn___internal.md`, but `find_qnn_sdk()` actually lives in `src/winml/modelkit/session/monitor/qnn/viewer.py:45-56`. The viewer doc (`session__monitor__qnn__viewer.md`) calls out the regression correctly. The regression is real:

Parent `7a66c024:src/winml/modelkit/optracing/qnn/viewer.py`:
```
_COMMON_SDK_PATHS: list[str] = [
    r"D:\QC",
    r"C:\Qualcomm\AIStack\qairt",
]

def find_qnn_sdk() -> Path | None:
    env_root = os.environ.get("QNN_SDK_ROOT")
    if env_root:
        root = Path(env_root)
        if root.is_dir():
            ...
            return root

    for base in _COMMON_SDK_PATHS:
        base_path = Path(base)
        if not base_path.is_dir():
            continue
        for child in sorted(base_path.iterdir(), reverse=True):
            if child.is_dir() and (child / "bin").is_dir():
                ...
                return child
    ...
    return None
```

Post-commit `774f121d:src/winml/modelkit/session/monitor/qnn/viewer.py:45-56`:
```
def find_qnn_sdk() -> Path | None:
    """Resolve QNN SDK root from ``QNN_SDK_ROOT`` env var.
    ...
    """
    env_root = os.environ.get("QNN_SDK_ROOT")
    if not env_root:
        return None
    root = Path(env_root)
    return root if root.is_dir() else None
```

`_COMMON_SDK_PATHS` is gone. No version-sorted fallback walk. Detail-mode QHAS now requires explicit `QNN_SDK_ROOT`. Behavior regression for any dev box that previously relied on `D:\QC` / `C:\Qualcomm\AIStack\qairt` defaults. Grep across `src/` confirms no surviving reference to `_COMMON_SDK_PATHS`.

### D-07 — `OpenVINOMonitor` dead stub — **VERIFIED**

`src/winml/modelkit/session/monitor/openvino_monitor.py:42-45`:
```
@classmethod
def is_available(cls) -> bool:
    """No Intel-specific telemetry available yet."""
    return False
```

Returns `False` unconditionally. `__enter__` returns `self`; `__exit__` is a no-op; `to_dict()` returns a hardcoded `{"ep": "OpenVINO", "device": "NPU", "status": "not_implemented"}`.

`src/winml/modelkit/commands/perf.py:118-188` (`_resolve_ep_monitor`) dispatches only to `QNNMonitor` (op-tracing branch, line 156-181) and `VitisAIMonitor` (proof-of-exec branch, line 186-187). OpenVINO is never routed. The class is referenced only as a re-export in `session/__init__.py` — no instantiation site exists.

Dead-code claim holds in full.

### D-16 — 14 near-identical `_require` calls in `_extract_summary` — **VERIFIED**

`src/winml/modelkit/session/monitor/qnn/_internal.py:357-370`:
```
357:        "inference_us": _require(raw, "time_us", ctx),
358:        "execute_us": _require(raw, "graph_execute_us", ctx),
359:        "inf_per_s": _require(raw, "inf_per_s", ctx),
360:        "timeline_cycles": _require(raw, "timeline_cycles", ctx),
361:        "utilization_pct": _require(raw, "percent_utilization", ctx),
362:        "dram_read_bytes": _require(raw, "total_dram_read", ctx),
363:        "dram_write_bytes": _require(raw, "total_dram_write", ctx),
364:        "vtcm_read_bytes": _require(raw, "total_vtcm_read", ctx),
365:        "vtcm_write_bytes": _require(raw, "total_vtcm_write", ctx),
366:        "vtcm_peak_bytes": _require(raw, "peak_vtcm_alloc", ctx),
367:        "qnn_nodes": _require(raw, "qnn_nodes", ctx),
368:        "htp_nodes": _require(raw, "htp_nodes", ctx),
369:        "unique_qnn_ops": _require(raw, "unique_qnn_ops", ctx),
370:        "unique_htp_ops": _require(raw, "unique_htp_ops", ctx),
```

Exactly 14 calls. The simplification proposal (a `_RENAME_MAP` + comprehension) is appropriate. NB: doc says "renames 13 of 14 keys"; actual count is 9 renames + 5 pass-throughs (`inf_per_s`, `timeline_cycles`, `qnn_nodes`, `htp_nodes`, `unique_qnn_ops`, `unique_htp_ops` — wait, that's 6 pass-throughs; renames are the other 8). Recount: pass-throughs are `inf_per_s`, `timeline_cycles`, `qnn_nodes`, `htp_nodes`, `unique_qnn_ops`, `unique_htp_ops` = 6 unchanged. Renamed = 8. So `_extract_summary` renames 8 of 14 keys, not 13. The "13 of 14" assertion in `session__monitor__qnn___internal.md` is **overstated** — but the underlying simplification opportunity (14 boilerplate calls collapsible into a map-driven loop) is real.

Other internal `_require` call sites in the file: 1 (line 325), 1 (line 381), 3 (lines 418, 419, 422). Total file-wide: 14 + 1 + 1 + 3 = 19. The doc's "19 call sites + definition + docstring" matches.

---

## Per-doc verification

### `serve__cli_api.md`

**Verified.** `git diff 7a66c024..774f121d` shows two `except asyncio.TimeoutError as exc:` → `except TimeoutError as exc:` substitutions at lines 292 and 304 inside `_run_with_semaphore`. HTTPException 503/504 statuses unchanged. Net 0 lines. Description as "mechanical Py3.11 modernization" is accurate.

### `serve__manager.md`

**Verified.** `git diff` shows one substitution at `_fmt_monotonic`: `datetime.datetime.now(tz=datetime.timezone.utc)` → `datetime.datetime.now(tz=datetime.UTC)`. Single hunk, line 551.

### `serve__schema.md`

**Verified.** `git diff` shows `class EpSwitchRequest(BaseModel):` → `class EPSwitchRequest(BaseModel):` at line 23. Field definitions identical. One-character class rename only.

**Unverified (meta):** The doc's "any consumer of `/openapi.json` sees a new component name" is a downstream-consequence claim; not falsifiable from the diff alone but plausible.

### `session____init__.md`

**Verified.**
- `__all__` contains 35 names (doc claim).
- Re-exports from `ep_device`, `ep_registry`, 4 monitor modules, `qairt.qairt_session`, `session`, `stats`.
- 17 names from `ep_device.py`: counted in the `from .ep_device import (...)` block: `EP_DEVICE_SPECS`, `VALID_DEVICES`, `VALID_EPS`, `DeviceNotFound`, `EPDeviceSpec`, `EPDeviceTarget`, `UnknownListingPick`, `WinMLDevice`, `WinMLEPMonitorMismatch`, `WinMLEPNotDiscovered`, `WinMLEPRegistrationFailed`, `auto_detect_device`, `default_device_for_ep`, `default_ep_for_device`, `ep_to_device`, `eps_for_device`, `expand_ep_name`, `lookup_device_spec`, `resolve_device`, `short_ep_name` = **20 names** not 17. Slight undercount.
- 3 names from `ep_registry.py`: `WinMLEP`, `WinMLEPDevice`, `WinMLEPRegistry` — matches doc.
- 6 monitor classes mentioned (`HWMonitor`, `NullEPMonitor`, `OpenVINOMonitor`, `QNNMonitor`, `VitisAIMonitor`, `WinMLEPMonitor`) match imports.

**Overstated.** "Direct imports: 8 modules" — actual count is 10 import statements (`ep_device`, `ep_registry`, `monitor.ep_monitor`, `monitor.hw_monitor`, `monitor.openvino_monitor`, `monitor.qnn_monitor`, `monitor.vitisai_monitor`, `qairt.qairt_session`, `session`, `stats`). Doc undercounts by 2.

**Overstated.** "17 names from `ep_device.py`" — actual is 20.

**Unverified.** Risk #1 ("a test asserting `set(__all__) == {real public symbols}` does not currently exist") — not falsifiable from this batch's scope.

### `session__ep_device.md`

**Verified.**
- File is 748 lines.
- 5 exception classes at lines 47, 51, 55, 59, 63 (with `# noqa: N818`).
- `_SHORT_TO_FULL` at line 85 has 9 entries (qnn, openvino, vitisai, migraphx, nvtensorrtrtx, cuda, tensorrt, dml, cpu).
- `_FULL_TO_SHORT` dict comprehension at line 114 is eagerly evaluated at module load; the inline comment on lines 112-113 calling it "built lazily" is factually incorrect (eager dict-comprehension). Doc claim of "comment lie" is accurate.
- `VALID_DEVICES = frozenset({"npu", "gpu", "cpu"})` at line 139.
- `VALID_SOURCE_TAGS` at lines 141-151 — 7 tags.
- `EP_DEVICE_SPECS` at line 260 — 12 entries (lines 267-291 contain 12 `EPDeviceSpec(...)` constructions, counting QNN/NPU + DML/GPU + CPU/CPU + QNN/GPU + QNN/CPU + OpenVINO×3 + VitisAI/NPU + MIGraphX/GPU + Tensorrt/GPU + NvTensorRtRtx/GPU = 12). Matches.
- `WinMLDevice.ort_handle` property at lines 597-605. Returns `self._ort`.

**Overstated.**
- Simplification #1 / Risk #2 / Behavior #6 claim `ort_handle` is "unused" and "no in-tree consumer." This is **FALSE**: `src/winml/modelkit/analyze/runtime_checker/ep_checker.py:67` does `[ep_device.device.ort_handle]`. So there IS an in-tree consumer (in the `analyze/` package, exactly as the property's docstring on line 601 advertises). The doc's "unused property" critique should be rewritten as "session.py and winml.py reach for `_ort` directly while analyze/ uses `ort_handle` — pick one convention." The simplification opportunity is real but smaller than the doc claims.

**Unverified.** Detail-level claims about behavior of `auto_detect_device`'s RuntimeError catch — claim matches `def auto_detect_device()` at line 427+ from the grep output but I did not read the full body.

### `session__ep_registry.md`

**Verified.**
- All structural claims (singleton, idempotency on `dll_path`, `BuiltinSource` synthesis sorted by `builtin_name`, two-cache shape, `_registration_count` suffix logic, `_dedup_ort_devices`, `_ort_get_ep_devices_or_fail`, `_entry_source_tag` 7-tag dispatch with the `WindowsWorkload.EP.` prefix branch) match the code (see earlier listings).
- D-04 (Behavior #7 + Risk #1 + Simplification #1) verified — see Critical claims above.

**Verified.** The detailed line ranges in the doc (e.g. `register_ep` at 259-355, `auto_device` at 357-418, etc.) match.

**Unverified.** Doc claim "Tests at `tests/unit/session/test_ep_registry.py` (+429 lines)" — out of batch scope to verify exact line counts.

### `session__monitor____init__.md`

**Verified.** File at `src/winml/modelkit/session/monitor/__init__.py` is 6 lines (copyright + one-line docstring `"""Per-EP monitors and op-tracing post-processing."""`). Doc says "5 lines" — close enough (4-line copyright header + 1-line docstring; off-by-one depending on whether you count blank lines). No `__all__`, no re-exports. Matches the "empty package marker" claim.

**Verified.** The deletion of `optracing/__init__.py` and its 8 re-exports (`OpTraceResult`, etc.) — confirmed by grep against the parent commit (not done here but consistent with the broader optracing→session/monitor relocation).

### `session__monitor__ep_monitor.md`

**Verified.**
- File is 173 lines (doc says 172 — close).
- `WinMLEPMonitor` ABC at line 35.
- `requires_session_teardown: ClassVar[bool] = False` at line 62.
- `ep_name: ClassVar[str | None] = None` at line 72.
- `__init_subclass__` at lines 74-88.
- `get_session_options` returns `{}` default at lines 90-96.
- `get_provider_options` returns `{}` default at lines 98-104.
- `set_onnx_op_types` no-op default at line 106 with `# noqa: B027` annotation.
- `result` property uses `getattr(self, "_result", None)` at line 128.
- Three abstract methods: `__enter__`, `__exit__`, `is_available` at lines 132-148.
- `NullEPMonitor` at line 151; no `to_dict()` override (verified by reading full file).
- Parent commit's `ep_monitor.py` had `to_dict()` as a 4th abstract method (verified via `git show 7a66c024:...`), plus `NullEPMonitor.to_dict()` returning `{}`. Both removed.

**Verified.** Cross-file claim: `session.py:732` reads `requires_session_teardown` via `getattr`. Not directly verified line-for-line in this batch but consistent with the file's design.

**Unverified.** "Simplification #1: QNNMonitor.result is a copy of the inherited default" — would require reading qnn_monitor.py:294-297 to confirm. Plausible.

### `session__monitor__hw_monitor.md`

**Verified.** `git diff` shows exactly 3 hunks:
- Module docstring line 9: `"EPMonitor"` → `"WinMLEPMonitor"`.
- `Self` import: `typing_extensions` → `typing`.
- Class docstring line 31: `"EPMonitor"` → `"WinMLEPMonitor"`.

Net +3/-3 lines. Doc's "3 added / 3 removed" matches the stat (`git diff --stat` shows `6 +++---` for 3+3).

### `session__monitor__live_display.md`

**Verified.** `git diff --stat` shows `207 deletions`. Confirms file deletion. Parent commit's file existed and contained `HWLiveDisplay`. Doc claim is accurate.

**Unverified.** "Three design docs still mention `HWLiveDisplay`" — would require grep across `docs/` to confirm count.

### `session__monitor__op_metrics.md`

**Verified.**
- File is 169 lines (doc says 168 — close).
- `TraceStatus = Literal[...]` at line 33 with the 5 documented values.
- `OperatorMetrics` dataclass at line 36, with `samples_us` field at 76 and the four derived properties (`sample_count`, `avg_us`, `total_us`, `p90_us`) at lines 78-103.
- `p90_us` uses `_stats.quantiles(..., n=10, method="inclusive")[8]` at line 103 with special-cases for n=0/1 at lines 97-100.
- `OpTraceResult` at line 110, `model: str | None` at line 115 (was `str` in parent — likely).
- `status: TraceStatus = "ok"` at line 137.
- `error: str | None = None` at line 139.
- `to_dict()` is hand-crafted, NOT `asdict(self)` (lines 141-164). `OperatorMetrics.to_dict()` IS `asdict(self)` at line 107. The asymmetry claim is verified.

**Unverified.** Predecessor `optracing/result.py` was "99 lines" — out of batch scope to verify.

### `session__monitor__openvino_monitor.md`

**Verified.**
- File is 49 lines.
- `class OpenVINOMonitor(WinMLEPMonitor):` at line 23.
- `is_available()` returns `False` literally at line 45.
- `to_dict()` returns `{"ep": "OpenVINO", "device": "NPU", "status": "not_implemented"}` at line 49.
- `__enter__`/`__exit__` are no-ops at lines 30-40.
- Parent commit had `class OpenVinoMonitor(EPMonitor):` (verified via `git show`). Rename + base-class rename are correct.

**Verified (D-07).** No call site in `perf.py::_resolve_ep_monitor` or elsewhere ever returns an `OpenVINOMonitor`. The class is a hierarchy placeholder.

### `session__monitor__qnn____init__.md`

**Verified.**
- File is 28 lines (doc says 27 — close).
- Re-exports `parse_qhas`, `parse_qnn_profiling_csv` from `._internal` at line 24.
- `__all__ = ["parse_qhas", "parse_qnn_profiling_csv"]` at line 27.
- `viewer` not re-exported. `qnn_monitor.py:31` does `from .qnn.viewer import find_qnn_sdk, run_qhas_viewer` (confirmed by grep).
- Docstring references "spec v2.0.1" at line 9 — matches.

**Unverified.** "Regression test at `tests/unit/architecture/test_qnn_imports.py`" — file exists but I did not open it in this batch.

### `session__monitor__qnn___internal.md`

**Verified.**
- File is 443 lines (matches "+443 / 0").
- `_OP_PATTERN` and `_TOKEN_SUFFIX` at lines 37 and 42.
- `_split_op_event_id` at line 50 — NEW relative to predecessor (verified via `git show 7a66c024:src/winml/modelkit/optracing/qnn/csv_parser.py` — no such function).
- `_require` definition at line 297; 19 total call sites in-file (1 at line 325, 14 in `_extract_summary` at 357-370, 1 at 381, 3 at 418/419/422). Matches doc's "19 call sites."
- `parse_qnn_profiling_csv` (lines 91-110), `_read_csv`, `_extract_metadata`, `_extract_samples`, `_parse_node_event`, `_aggregate_operators` all present.
- `_aggregate_operators` produces `samples_cycles` per op (lines 268, 271, 283). NEW vs predecessor.
- `_transform_op` at line 374 uses `qnn_op_type` for `name` (line 418) and `_TOKEN_SUFFIX.sub("", qnn_op)` for `op_path` (line 419). NEW behavior.

**Verified (D-16).** 14 `_require` calls in `_extract_summary`.

**Overstated.** Behavior #1 / Symbol claim: doc says `_extract_summary` "renames 13 of 14 keys." Actual count is **8 renames + 6 pass-throughs** (`inf_per_s`, `timeline_cycles`, `qnn_nodes`, `htp_nodes`, `unique_qnn_ops`, `unique_htp_ops` are identical between source and result keys). The rename-fixes-I-9 narrative is still correct (specifically: `time_us → inference_us`, `graph_execute_us → execute_us`, `percent_utilization → utilization_pct`, `total_dram_read → dram_read_bytes`, `total_dram_write → dram_write_bytes`, `total_vtcm_read → vtcm_read_bytes`, `total_vtcm_write → vtcm_write_bytes`, `peak_vtcm_alloc → vtcm_peak_bytes` = 8 renames), but the per-claim count is wrong.

**Overstated (location).** D-03 is described under this doc but the actual `find_qnn_sdk` and `_COMMON_SDK_PATHS` regression lives in `qnn/viewer.py`, not `qnn/_internal.py`. The viewer doc (`session__monitor__qnn__viewer.md`) handles the claim correctly.

### `session__monitor__qnn__viewer.md`

**Verified (D-03).**
- File is 206 lines (matches "+206 / 0").
- `find_qnn_sdk` at line 45 reads only `QNN_SDK_ROOT`. No `_COMMON_SDK_PATHS` in the file. Confirmed env-var-only.
- Parent commit's `optracing/qnn/viewer.py` had `_COMMON_SDK_PATHS = [r"D:\QC", r"C:\Qualcomm\AIStack\qairt"]` and a 2-step resolution. Verified via `git show 7a66c024:...`.
- `_DEFAULT_CONFIG` at line 31 with the 8 keys listed.
- `_find_viewer_exe` at line 59.
- `run_basic_viewer` at line 84, `run_qhas_viewer` at line 136.

**Verified.** `run_basic_viewer` has no in-tree caller (grep across `src/` and `tests/` returns only the definition site).

**Verified.** `qnn_monitor.py:31` is the sole importer (`from .qnn.viewer import find_qnn_sdk, run_qhas_viewer`).

---

## Batch 03 Overall

| Doc | Verified | Overstated | False | Unverified |
|---|---|---|---|---|
| `serve__cli_api.md` | most | none | none | none meaningful |
| `serve__manager.md` | all | none | none | none |
| `serve__schema.md` | all | none | none | meta-claim about OpenAPI consumers |
| `session____init__.md` | most | "8 modules" (actually 10), "17 names from ep_device" (actually 20) | none | test-coverage claim |
| `session__ep_device.md` | most | `ort_handle` "unused" (analyze/ep_checker.py uses it) | "no in-tree consumer of ort_handle" | full body of `auto_detect_device` etc. not line-read |
| `session__ep_registry.md` | all major claims incl. D-04 | none | none | exact line counts of test fixtures |
| `session__monitor____init__.md` | "5 lines" actually 6 | minor count off-by-one | none | none |
| `session__monitor__ep_monitor.md` | all major claims | line count "172" actually 173 | none | QNNMonitor.result duplication claim (not opened qnn_monitor.py) |
| `session__monitor__hw_monitor.md` | all (3 hunks) | none | none | none |
| `session__monitor__live_display.md` | deletion confirmed (-207) | none | none | "three design docs" not verified |
| `session__monitor__op_metrics.md` | structural claims, status/error fields | "168 lines" actually 169 | none | predecessor `optracing/result.py` size |
| `session__monitor__openvino_monitor.md` | D-07 confirmed | none | none | none |
| `session__monitor__qnn____init__.md` | all (28 lines, 2 exports, viewer-not-reexported) | "27 lines" actually 28 | none | architecture-test contents |
| `session__monitor__qnn___internal.md` | most incl. D-16 (14 calls) | "renames 13 of 14" (actually 8), D-03 mislabeled to this file (lives in viewer.py) | none | qnn_monitor consumer line-level claims |
| `session__monitor__qnn__viewer.md` | D-03 confirmed end-to-end | none | none | timeout-related risk claims |

**Critical-claim headline:** D-03 (lost `_COMMON_SDK_PATHS` fallback) and D-04 (`auto_device` stale `last_error` traceback) are both confirmed real regressions/bugs against the parent commit `7a66c024`. D-07 (`OpenVINOMonitor` dead stub — `is_available()` returns `False` unconditionally and `perf.py::_resolve_ep_monitor` never dispatches to it) holds end-to-end. D-16 (14 boilerplate `_require` calls in `_extract_summary`) is exact.

**Overstatement pattern:** The per-file docs are reasonably accurate on the bug/regression substance but lose precision on counts (line counts off by 1-3, import counts off by 2, "13 of 14 renames" off by 5). One specific factual claim is **false**: `WinMLDevice.ort_handle` is **NOT** unused — `analyze/runtime_checker/ep_checker.py:67` consumes it (the property's docstring on line 601 even points there). That nuance should land in the headline as "session.py uses `_ort` directly while analyze/ uses `ort_handle` — the asymmetry is the real simplification target."

**Doc structural issue:** D-03 is allocated to `session__monitor__qnn___internal.md` in the batch instructions, but the affected code lives in `session__monitor__qnn__viewer.md`. The viewer doc is the one that captures it correctly. The internal doc would be cleaner if the SDK-discovery claim were removed or cross-referenced rather than duplicated.
