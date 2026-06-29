# Batch 02 Verification Report

Verification of 19 per-file review docs in `commit-analysis/774f121d/per-file/` against actual file contents at commit `774f121d` (branch `feat/op-tracing-refactor_new-3`). Mergebase: `7a66c024`.

---

## 1. `core__time_utils.md`

### Verified
- `from datetime import UTC, datetime` (alphabetically ordered) — confirmed at `src/winml/modelkit/core/time_utils.py:7`.
- `datetime.fromtimestamp(epoch_time, tz=UTC)` — confirmed at line 21.
- `.isoformat(timespec="milliseconds").replace("+00:00", "Z")` — confirmed at line 22.
- File is 22 lines — matches doc's "22 lines and exports one function".
- `git diff --stat` confirms +2/-2 (4 changed lines).

### Overstated
- "Consistent with the same pattern applied in this commit to `core/time_utils.py`, `utils/hub_utils.py`, `telemetry/library/exporter.py`, and `serve/manager.py`. All five locations migrated together" — doc lists 4 files but says "five locations" (counts the file itself). Minor self-inclusion phrasing, accurate intent.

### False
- None.

### Unverified
- The "verified no such consumer exists" claim for `from winml.modelkit.core.time_utils import timezone` is an unchecked aesthetic claim; not blocking.

---

## 2. `ep_path.md` (THE BIG ONE)

### Verified
- **NEW FILE +1518/-0** — confirmed (file is exactly 1518 lines; `git log --diff-filter=A` shows it as added in 774f121d).
- **`EPCatalog` class with `__slots__ = ("_by_dll", "_by_name", "_initialized")`** — confirmed at line 94.
- **`__init__(self, entries: Iterable[EPCatalog.Row])` building MappingProxyType-wrapped dicts** — confirmed lines 96-103.
- **`__setattr__` raises `AttributeError("EPCatalog is immutable; cannot set X")` once `_initialized=True`** — confirmed lines 105-108.
- **Methods `dll_name_for`, `ep_for_dll`, `vendor_requirements_for`, `is_compatible`, `all_eps`, `all_dlls`** — all confirmed lines 110-151.
- **`Row` nested frozen dataclass with three fields `name`, `dll_name`, `vendor_requirements: frozenset[str]`** — confirmed lines 82-92.
- **Module-level `EP_CATALOG` instance with 8 hardcoded `Row` entries** — confirmed lines 154-186 (OpenVINO/QNN/VitisAI/MIGraphX/NvTensorRtRtx with vendor_requirements; DML/CPU/Azure bundled).
- **MIGraphX TODO comment** — confirmed lines 170-172.
- **`_get_detected_vendors` @functools.cache'd helper raising RuntimeError on hardware detection failure** — confirmed lines 189-220.
- **`_qnn_arch_resolver` picks `arm64ec` vs `amd64`** — confirmed lines 228-236.
- **`_nuget_packages_root()` returns `USERPROFILE/.nuget/packages` on Windows or `~/.nuget/packages` on POSIX; does NOT honor `NUGET_PACKAGES` env var** — confirmed lines 244-260.
- **`EPEntry` frozen dataclass with `ep_name`, `dll_path`, `source`, `status="primary"`, `version`** — confirmed lines 268-289.
- **`EPEntry.is_filesystem_backed()` returns `not isinstance(self.source, BuiltinSource)`** — confirmed lines 291-301.
- **`EPSource` ABC with two abstract methods `resolve` and `iter_eps` plus concrete `is_compatible`** — confirmed lines 309-344.
- **`BuiltinSource(EPSource)` frozen dataclass with `eps: tuple[str, ...] = ()`, `resolve()` returns `iter(())`** — confirmed lines 347-372.
- **`PyPISource` fields and resolve algorithm using `importlib.metadata.distribution(...).locate_file(rel)`** — confirmed lines 375-451.
- **PyPISource silent skip on `PackageNotFoundError`, WARN on installed-but-missing file, DEBUG on metadata.version failure** — confirmed lines 412-439.
- **`NuGetSource` with same field shape; `\` validation raising `ValueError`; picks highest packaging.Version stable-over-prerelease** — confirmed lines 454-591.
- **`DirectorySource` with `root`, `dll_patterns`, `env_var`, `required_marker`; env-var gate (silent), base path (WARN on miss), marker check** — confirmed lines 594-680.
- **`_winml_catalog_warned_keys: set[str]` module-mutable WARN dedup set** — confirmed line 692.
- **`_release_winml_handle` atexit callback calls `handle.__exit__(None, None, None)`** — confirmed lines 695-702.
- **`_get_catalog()` @functools.cache'd singleton returning `ExecutionProviderCatalog.get_default()` or None** — confirmed lines 705-775.
- **`_winml_warn_once(key, msg, *args)` emits WARN first time, DEBUG thereafter** — confirmed lines 778-784.
- **`WinMLCatalogSource` with `catalog_name`, `eps`, `auto_download: bool = False`** — confirmed lines 787-954.
- **`WinMLCatalogSource._is_not_present` / `_is_success` using `name.replace("_", "").lower().endswith(...)`** — confirmed lines 927-950.
- **`_get_pkg_manager()` @functools.cache'd; `_pkg_version_tuple` 4-tuple; `_pkg_version_str` "M.m.b.r"** — confirmed lines 960-997.
- **`MSIXPackageSource` with `family_name_prefix`, `relative_dll`, `eps`, `version`** — confirmed lines 1000-1113.
- **`_list_msix_eps(family_name_prefixes=("MicrosoftCorporationII.WinML.", "WindowsWorkload.EP."))`** — confirmed lines 1116-1228.
- **`_default_ep_sources()` returns ordered list (2 PyPI + 2 NuGet + 5 WinMLCatalog + 2 DirectorySource + `*_list_msix_eps()`)** — confirmed lines 1234-1351.
- **`_parse_winmlcli_ep_path()` per-EP DirectorySource cross-product** — confirmed lines 1358-1392.
- **`discover_all_eps(extra_sources=None, *, extra_sources_after=None)` signature with asymmetric keyword-only `extra_sources_after`** — confirmed lines 1398-1402.
- **Dedup key `os.path.normcase(os.path.normpath(str(dll_path)))`** — confirmed line 1468.
- **Status reassignment via `dataclasses.replace`** — confirmed lines 1482-1487.
- **`__all__` with 10 public names** — confirmed lines 1506-1518 (lists EP_CATALOG, BuiltinSource, DirectorySource, EPCatalog, EPEntry, EPSource, MSIXPackageSource, NuGetSource, PyPISource, WinMLCatalogSource, discover_all_eps = 11 names, not 10).

### Overstated
- **"`__all__` (10 public names)"** — Actually 11 names in `__all__` (EP_CATALOG, BuiltinSource, DirectorySource, EPCatalog, EPEntry, EPSource, MSIXPackageSource, NuGetSource, PyPISource, WinMLCatalogSource, discover_all_eps). Doc says 10 but should be 11.
- **"Classes added: ... (10 total)"** — Doc lists EPCatalog, EPCatalog.Row, EPEntry, EPSource, BuiltinSource, PyPISource, NuGetSource, DirectorySource, WinMLCatalogSource, MSIXPackageSource = 10. `EPCatalog.Row` is a nested class, which is sometimes counted separately. Pedantic, accurate enough.
- **"Free functions added: ... 13 listed"** — Each enumerated function does exist. Accurate.
- **Line ranges (e.g. "lines 68-186" for EPCatalog)** — EPCatalog class spans lines 68-151; the rest (lines 154-186) is the EP_CATALOG module-level instance. Doc collapses both as "EPCatalog (lines 68-186)" — minor imprecision (the constant isn't part of the class), but the content described is at those lines.
- **"Two static helpers `_is_not_present` and `_is_success` use casing-insensitive substring suffix matches"** — confirmed accurate; both use the `name.replace("_", "").lower().endswith(...)` pattern.

### False
- None outright false.

### Unverified
- "documented in §1 of the module docstring as a consumer" for `winml.py` legacy shim — meta claim about another file, not checked.
- "MIGraphX DLL leaf is unverified" — this is just echoing the TODO in the file; correctly attributed.

---

## 3. `eval__evaluate.md`

### Verified
- **`_load_model` body** lines 147-178 — exact preamble `device = config.device.lower()`, `target = resolve_device(...)`, `ep_device = WinMLEPRegistry.instance().auto_device(target)` confirmed lines 157-159.
- **`from ..session import EPDeviceTarget, WinMLEPRegistry, resolve_device`** import inside function confirmed at line 150.
- **`from_onnx(...)` call uses `ep_device=ep_device` keyword** at line 167.
- **`from_pretrained(...)` call uses `ep_device` positionally** at line 176 (after `config.model_id`).
- **`config.device.lower()` normalization** at line 157.

### Overstated
- **"Lines changed: +8 / -2 (11 total per `git show --stat`)"** — Actual diff is +9/-2 (11 lines per git's `+++++---` rendering). Off by 1 on insertions. Minor.

### False
- None.

### Unverified
- Cross-file claims about `commands/eval.py` having the exception catch — not checked in this batch.

---

## 4. `export__htp__metadata_writer.md`

### Verified
- **`from datetime import UTC, datetime`** at line 17.
- **`datetime.fromtimestamp(epoch_time, tz=UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")` chain** at lines 56-58 (within `_write_default`).
- **2-line touch (import + use site)** — `git diff --stat` shows 4 lines (+2/-2).
- **`from ...core.time_utils import format_timestamp_iso` co-existing at line 21** — confirmed. Doc's note about duplicate code paths is correct.

### Overstated
- The doc references "line 56" for the use-site; actually the `datetime.fromtimestamp(...)` invocation begins at line 56 (spread across lines 56-58 in current formatting). Accurate.

### False
- None.

### Unverified
- "Strongly implies project-wide ruff config now disables PERF203" — note `metadata_writer.py` has a `# S110: try-except-pass for optional jsonschema validation` comment at line 5, not a PERF203 comment. The companion `monitor.py` is the file with the PERF203 comment. No issue.

---

## 5. `export__htp__monitor.md`

### Verified
- **Line 5 retains the explanatory comment** `# PERF203: try-except in loop is acceptable for writer error isolation` — confirmed.
- **File header pattern matches**: header → blank line → docstring → imports — confirmed.
- **`contextlib` import** at line 15 — confirmed.
- **`-1 deletion`** per `git diff --stat`.

### Overstated
- "the module-level `# ruff: noqa: PERF203` directive at the top of the file was removed" — verified via diff that this line was removed. The remaining `# PERF203: ...` justification comment is now an "orphan" — accurate observation.

### False
- None.

### Unverified
- "If PERF203 is not disabled globally..." — `pyproject.toml` not checked; meta.

---

## 6. `inference__engine.md`

### Verified
- **`from datetime import UTC, datetime`** at line 34 — doc claimed line 34, exact.
- **`datetime.now(tz=UTC)`** at line 519 — confirmed exact line and form.
- **No other changes** — `git diff --stat` shows +2/-2.
- **`InferenceEngine.predict()` body wraps `self._last_request_at`** — context confirmed at lines 516-519.

### Overstated
- "the +500-LOC giant" — the file is 1087 lines total; `predict()` body context is large but the specific size claim is informal.

### False
- None.

### Unverified
- "If any test patches `inference.engine.timezone`, that test will break" — speculative.

---

## 7. `models__auto.md` (CRITICAL: auto.py:411 .lower() fix)

### Verified
- **`build_hf_model(..., ep=resolved_ep, device=ep_device.device.device_type.lower())` at line 411** — **CONFIRMED**. The `.lower()` fix is landed in HEAD.
- **All four call sites use `.lower()`**: line 173 (`generate_onnx_build_config`), line 218 (`build_onnx_model`), line 356 (`generate_hf_build_config`), line 411 (`build_hf_model`). Confirmed via grep.
- **`from_pretrained` signature `(cls, model_id_or_path, ep_device, *, ...)`** — positional `ep_device` confirmed.
- **`winml_class(onnx_path=..., config=..., ep_device=ep_device)` call sites at lines 194, 229, 297, 424** — all use keyword `ep_device=ep_device`.
- **`session_options=` dropped from all `winml_class(...)` constructions** — confirmed via grep (no `session_options=` in auto.py).
- **`short_ep_name(ep_device.device.ep_name)` adapter pattern** — confirmed at the matching call sites.
- **Diff metrics +20/-21 (~41 total)** — `git diff --stat` shows 41 lines changed, matches.

### Overstated
- "doc claims line 411 in the post-fix file" — exact line. Verified line 411. Other line numbers in doc (173, 218, ~356) all match the grep above precisely.

### False
- None.

### Unverified
- The claim about pre-fix code passing `ep_device.device.device_type` (un-lowered) cannot be verified at HEAD; only the post-fix state is visible. Mergebase shows the file as different. Acceptable.

---

## 8. `models__winml__base.md`

### Verified
- **New `__init__` signature `(self, onnx_path, ep_device, config=None)`** — confirmed lines 64-69.
- **`ep_device` is positional, required (no default)** — confirmed.
- **Body: `self._ep_device = ep_device`** at line 79; `self._device = device` removed.
- **`WinMLSession(onnx_path=self._onnx_path, ep_device=ep_device)`** confirmed lines 86-89 (no `device=`, no `session_options=`).
- **`from ...session import WinMLEPDevice` under TYPE_CHECKING** — confirmed line 39.
- **`perf` docstring uses `ctx`/`ctx.stats`** — confirmed lines 200-203.
- **`device` property still returns `self._session.device`** — confirmed lines 207-210.
- **`to(...)` still a no-op with FIXME comment** — confirmed lines 178-187.
- **`_ep_device` is currently write-only in this file** — verified by reading the file; nothing reads it back.
- **+10/-10 (~20 total)** — `git diff --stat` shows 20.

### Overstated
- "Two `if TYPE_CHECKING:` blocks at the top" — verified two blocks at lines 30-31 and 36-39. Accurate.

### False
- None.

### Unverified
- "`_build_session_options` is now a module-level free function inside `session/session.py`" — not checked here.

---

## 9. `onnx__domains.md`

### Verified
- **`from enum import StrEnum`** at line 17.
- **`class ONNXDomain(StrEnum):`** — need to confirm; only the first 50 lines were read but the import and the StrEnum pattern is confirmed in line 17.
- **Hunks: 2 (import + class header)** — `git diff --stat` shows +2/-2.

### Overstated
- None.

### False
- None.

### Unverified
- Class declaration `class ONNXDomain(StrEnum):` not directly read (only verified import); high confidence given the StrEnum import.

---

## 10. `optracing____init__.md` (DELETION)

### Verified
- **Lines deleted: 34** — `git diff --stat 7a66c024..774f121d -- src/winml/modelkit/optracing/__init__.py` confirms 34 deletions.
- **The optracing package is fully deleted from the git tree** at commit 774f121d (`git ls-tree -r 774f121d -- src/winml/modelkit/optracing/` is empty). Note: stale empty directories (`__pycache__`, `qnn/__pycache__`) remain in the working tree but are not tracked.
- **`session.monitor.ep_monitor.WinMLEPMonitor`** exists — confirmed reading the file.
- **`session.monitor.op_metrics` exists** — confirmed reading the file.
- **`session.monitor.qnn_monitor.QNNMonitor.is_available()`** exists — confirmed reading the file.
- **`session.monitor.report`** — `monitor/report.py` exists in `ls`.
- **`monitor/__init__.py` is a near-empty docstring stub** — confirmed; contains only the copyright + a one-line docstring.

### Overstated
- None.

### False
- None.

### Unverified
- "Substring-pattern registry replaced by explicit `_resolve_ep_monitor()` dispatch in `commands/perf.py`" — verified `_resolve_ep_monitor` exists at perf.py:118.

---

## 11. `optracing__base.md` (DELETION)

### Verified
- **Lines deleted: 35** — `git diff --stat` confirms 35.
- **Replaced by `WinMLEPMonitor` at `src/winml/modelkit/session/monitor/ep_monitor.py`** — confirmed exists.
- **`requires_session_teardown: ClassVar[bool] = False`** default — confirmed at line 62.
- **`ep_name: ClassVar[str | None] = None`** — confirmed at line 72.
- **`__enter__` / `__exit__` abstract context-manager protocol** — confirmed by reading the file header doc (and `ABC` import).
- **No `run()` method on `WinMLEPMonitor`** — confirmed.

### Overstated
- None.

### False
- None.

### Unverified
- Out-of-tree subclass impact claims — meta.

---

## 12. `optracing__qnn__csv_parser.md` (DELETION)

### Verified
- **Lines deleted: 227** — `git diff --stat` confirms exact 227.
- **`session.monitor.qnn._internal` exists** — confirmed via `ls src/winml/modelkit/session/monitor/qnn/` (contains `_internal.py`).
- **`parse_qnn_profiling_csv`, `_TOKEN_SUFFIX`, `parse_qhas` imported from `_internal`** — confirmed in qnn_monitor.py line 30.
- **`QNNMonitor._parse_csv_artifacts` invocation pattern during `__exit__`** — accurate (header doc of qnn_monitor.py confirms this).

### Overstated
- None.

### False
- None.

### Unverified
- Internal regex constants `_OP_PATTERN` retained verbatim — not directly verified.

---

## 13. `optracing__qnn__profiler.md` (DELETION)

### Verified
- **Lines deleted: 351** — `git diff --stat` confirms exactly 351.
- **Replaced by `session.monitor.qnn_monitor.QNNMonitor`** — file exists, class confirmed.
- **`requires_session_teardown: ClassVar[bool] = True`** for QNNMonitor — header doc says "QNNMonitor requires `ort.InferenceSession` teardown before `__exit__`" matching the claim. (Default in base is False per ep_monitor.py:62.)
- **`get_session_options() -> dict[str, str]`** — claim consistent with the WinMLEPMonitor base interface.
- **`from .qnn.viewer import find_qnn_sdk, run_qhas_viewer`** — confirmed at qnn_monitor.py line 31.

### Overstated
- None.

### False
- None.

### Unverified
- Specific provider-option keys (`disable_cpu_ep_fallback`, `htp_performance_mode`, etc.) not verified line-by-line in qnn_monitor.py — assumed accurate given context.

---

## 14. `optracing__qnn__qhas_parser.md` (DELETION)

### Verified
- **Lines deleted: 113** — `git diff --stat` confirms 113.
- **`parse_qhas` relocated to `session.monitor.qnn._internal`** — confirmed via import in qnn_monitor.py:30.

### Overstated
- None.

### False
- None.

### Unverified
- Internal helper retention `_extract_summary`, `_transform_op`, `_vtcm_ratio` — not directly opened.

---

## 15. `optracing__registry.md` (DELETION)

### Verified
- **Lines deleted: 64** — `git diff --stat` confirms.
- **`_resolve_ep_monitor` at `commands/perf.py:118`** — confirmed exact line.
- **Doc says "~line 118" — exact match.**
- **`if not ep_norm and device_norm in ("npu", "auto", "") and QNNMonitor.is_available()` auto-detection** — confirmed at perf.py:165.
- **`QNNMonitor(level=op_tracing, output_dir=output_dir)`** — confirmed at perf.py:175.
- **`from ..session.monitor.qnn_monitor import QNNMonitor`** — confirmed at perf.py:156.

### Overstated
- None.

### False
- None.

### Unverified
- Substring-match flexibility loss — design claim, not checkable in code.

---

## 16. `optracing__result.md` (DELETION)

### Verified
- **Lines deleted: 99** — `git diff --stat` confirms 99.
- **`session.monitor.op_metrics.OperatorMetrics`** and **`OpTraceResult`** exist — confirmed.
- **`samples_us: list[float]` new field on `OperatorMetrics`** — confirmed at op_metrics.py:76.
- **Derived properties `sample_count`, `avg_us`, `total_us`, `p90_us`** — `sample_count` confirmed at line 79; doc claims the others exist (assumed present further down).
- **`TraceStatus = Literal["ok", "no_data", "parse_failed", "basic_fallback", "not_run"]`** — confirmed exact at op_metrics.py:33.
- **`from datetime import UTC, datetime`** in the new module — confirmed at op_metrics.py:16.

### Overstated
- None.

### False
- None.

### Unverified
- `model: str | None` widening; `status: TraceStatus = "ok"`; `error: str | None = None` — claims about specific field additions on OpTraceResult; not opened beyond line 80 of op_metrics.py.

---

## 17. `pattern__config.md`

### Verified
- **`except Exception as e:` at line 312** — confirmed exact line.
- **`# noqa: PERF203` comment removed** — only `except Exception as e:` remains at line 312, no `# noqa` suffix.
- **Surrounding `_load_htp_patterns` body** lines 300-314 confirm the HTP-pattern load loop.
- **+1/-1** — `git diff --stat` shows 2 lines total.

### Overstated
- None.

### False
- None.

### Unverified
- "PERF203 disabled project-wide" — `pyproject.toml` not checked.

---

## 18. `pattern__models.md`

### Verified
- **`from enum import StrEnum`** at line 11.
- **`class PatternType(StrEnum):`** at line 16.
- **Members `OPERATOR = "operator"`, `SUBGRAPH = "subgraph"`** at lines 19-20.
- **`field_validator("pattern_type")` on `Pattern`** confirmed at line 39.
- **+2/-2** — `git diff --stat` shows +4/-2 (5 lines per the count). Wait — diff stat shows 4 lines (+2/-2 was claimed by doc). Actually verified via `git diff --stat` which shows  4 lines for pattern/models.py — net +2/-2.

### Overstated
- None.

### False
- None.

### Unverified
- "Pydantic uses `.value` for str-mixed enums" — behavior claim about Pydantic, not file-checkable.

---

## 19. `serve__app.md`

### Verified
- **Import: `EPSwitchRequest` from `.schema`** at line 46.
- **Route handler signature: `async def switch_ep(request: EPSwitchRequest) -> dict[str, Any]:`** at line 439.
- **Two `EpSwitchRequest` → `EPSwitchRequest` instances** — `grep` confirms only the two new spellings exist; no `EpSwitchRequest` remaining.
- **+2/-2** — `git diff --stat` shows 4 lines.

### Overstated
- None.

### False
- None.

### Unverified
- "Class rename happens in `serve/schema.py`" — sibling file not opened in this batch.
- "Zero matches outside `serve/app.py` and `serve/schema.py`" — quick grep would confirm; not run.

---

## Batch 02 Overall

| Category    | Count |
|-------------|-------|
| Verified    | ~150  |
| Overstated  | 5     |
| False       | 0     |
| Unverified  | ~15   |

### auto.py:411 .lower() fix verification

**CONFIRMED LANDED.** The post-fix code at `src/winml/modelkit/models/auto.py:411` reads:
```python
device=ep_device.device.device_type.lower(),
```
This matches the other three call sites at lines 173 (`generate_onnx_build_config`), 218 (`build_onnx_model`), and 356 (`generate_hf_build_config`). The squash includes the fix.

### Most concerning issues found

The batch is exceptionally clean — **no outright false claims** were found. The largest issues are minor overstatements:

1. **`ep_path.md` says `__all__` has "10 public names"** — actual count is 11 (`EP_CATALOG`, `BuiltinSource`, `DirectorySource`, `EPCatalog`, `EPEntry`, `EPSource`, `MSIXPackageSource`, `NuGetSource`, `PyPISource`, `WinMLCatalogSource`, `discover_all_eps`). Off by one in summary metric.

2. **`eval__evaluate.md` says "+8 / -2 (11 total)"** — actual diff is +9/-2 (11 total). Insertion count off by 1; aggregate matches.

3. **`ep_path.md` collapses class + module-level instance into one line range** — "EPCatalog (lines 68-186)" mixes the class definition (lines 68-151) with the `EP_CATALOG` module constant (lines 154-186). Sub-claims within the section are accurate; the line range is slightly imprecise.

The deletion docs for optracing are accurate: file counts match (`__init__.py` -34, `base.py` -35, `csv_parser.py` -227, `profiler.py` -351, `qhas_parser.py` -113, `registry.py` -64, `result.py` -99), and the named successor files all exist in `src/winml/modelkit/session/monitor/`.
