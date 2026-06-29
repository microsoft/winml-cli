# Commit 774f121d — Summary

> **Fact-check pass applied (2026-06-28).** This SUMMARY was independently verified against actual source by 5 verification agents (reports at `verification/batch-{00..04}.md`); the consolidated verdict is at [`FINAL-VERDICTS.md`](FINAL-VERDICTS.md). All four 🔴 regression claims hold up under scrutiny. Two claims from per-file reviews were retracted as FALSE and are NOT present in this SUMMARY's body: (a) `qairt_session.py from ..session import _build_session_options` is NOT fragile (standard PEP-328 sibling-submodule resolution); (b) `WinMLDevice.ort_handle` IS used by `analyze/runtime_checker/ep_checker.py:67` — it is not unused.

## What this commit is

Commit `774f121d`, titled **"feat(session): v2.9 unified-source EP refactor + WinMLSession redesign"**, is the squash of 45 development commits from branch `feat/op-tracing-refactor_new-2` against mergebase `7a66c024`, totalling 362 files at +70,114 / −4,491 lines (Python under `src/winml/modelkit/`: 81 files reviewed individually under `commit-analysis/774f121d/per-file/`). The squash continues the architectural arc started by the prior `a509a67` op-tracing + EPDevice refactor and pushes it through to a **unified-source EP model**: built-in EPs (CPU, Dml, Azure) are no longer special-cased — they flow through the same `EPSource → register_ep → WinMLEP` pipeline as plugin EPs via a new `BuiltinSource` marker class. The commit collapses the previously parallel "registry merge" / "ORT-providers" / "built-ins block" code paths in `commands/sys.py` into a single broad-walk pass over `WinMLEPRegistry.all_discovered()`, with L1 (registration outcome) and L2 (vendor compatibility) status taxonomy derived in exactly one place. It also ships an end-to-end **`--ep <name>[@<source-tag>]` CLI syntax** for Scenarios A.5/A.6 via a new `EpAtSourceParamType`; tears out the old `optracing/` package wholesale in favour of a `session/monitor/` tree (a near-duplicate of the a509a67 move that landed partially); deletes the `wrap_ort_device` factory shim in favour of direct `WinMLDevice(handle)` construction; and consolidates the `WinMLEPRegistry` singleton into a single `instance()` classmethod (the `__new__` + `_initialized` guard is gone).

The change set also folds in several pieces of **squash collateral** that the title does not advertise: a new `_transformers_compat.py` (+304 LOC) that monkey-patches transformers 5.x symbols so optimum-onnx 0.1.0 survives, a coordinated Python-3.11 modernization sweep (`StrEnum`, `datetime.UTC`, `TimeoutError`, `PERF203` ruff suppression strip) across five files in `onnx/` `pattern/` `export/htp/`, and a `winml.py` legacy-deprecation pass that issues `DeprecationWarning` on every entry point of the old top-level `WinML` singleton.

## File-count by theme

| Theme | Files touched | New | Deleted | Modified |
|---|---|---|---|---|
| Session core (`ep_path`, `ep_device`, `ep_registry`) | 4 | 1 (`ep_path.py` +1518) | 0 | 3 |
| Session lifecycle (`session/session.py`, qairt) | 2 | 0 | 0 | 2 |
| Op-tracing monitor pipeline (new tree) | 12 | 4 | 0 | 8 |
| Legacy `optracing/` tree (deleted wholesale) | 7 | 0 | 7 | 0 |
| CLI commands (`commands/`) | 9 | 2 (`_ep_arg.py`, `_pre_bench.py`) | 0 | 7 |
| Compiler + config | 5 | 0 | 0 | 5 |
| Sysinfo redistribution | 4 | 0 | 1 (`device.py`) | 3 |
| Analyze subsystem (downstream) | 12 | 0 | 0 | 12 |
| Models + build (downstream) | 4 | 0 | 0 | 4 |
| Utils + telemetry + onnx/pattern/export (Py3.11 + facade) | 11 | 0 | 0 | 11 |
| Top-level (`__init__`, `winml.py`, `_transformers_compat.py`, `core/`) | 5 | 1 (`_transformers_compat.py`) | 0 | 4 |
| Serve | 4 | 0 | 0 | 4 (mostly EPSwitchRequest casing) |
| **Total (Python under `src/winml/modelkit/`)** | **81** | **8** | **8** | **65** |

Three of the eight "new" files are renames from `optracing/*` (the `__name-status` shows `R` at git plumbing level), but they are net new at their destination paths.

## The eight architectural moves

### 1. `BuiltinSource` unified-source synthesis — CPU/Dml/Azure flow through the same `EPSource → register_ep → WinMLEP` pipeline as plugin EPs

`ep_path.py` (new, +1518 LOC) introduces an `EPSource` ABC with six concrete subclasses: `BuiltinSource(eps: tuple[str, ...])` for ORT-bundled providers; `PyPISource`, `NuGetSource`, `DirectorySource`, `WinMLCatalogSource`, `MSIXPackageSource` for plugin discovery channels. Each subclass implements `resolve() → Iterable[EPEntry]` (one entry per `(ep_name, dll_path, source)` triple). At `WinMLEPRegistry.__init__`, the discovery cache `_discovered` is built as `list(discover_all_eps()) + [EPEntry(ep_name=n, dll_path=Path(), source=BuiltinSource(eps=(n,))) for n in builtin_names - discovered_names]` where `builtin_names = ort.get_available_providers() ∩ {d.ep_name for d in ort.get_ep_devices()}`. The intersection guard is load-bearing: it suppresses ORT misconfig states (provider listed but no `OrtEpDevice` exposed) that would otherwise drive `default_ep_for_device` to pick a bundled EP that crashes session-build.

The unified-source thesis pays dividends downstream. `commands/sys.py:_gather_ep_info` no longer has a separate built-ins block — every entry flows through one main loop, and the BuiltinSource branch of `register_ep` (no `register_execution_provider_library` call; direct `ort.get_ep_devices()` filter by `ep_name`) is dispatched on `isinstance(entry.source, BuiltinSource)`. The CLI's `[bundled]` source-kind tag is dispatched the same way via `_entry_source_tag` (which also handles `pypi`, `nuget`, `msix-microsoft`, `msix-workload`, `winml-catalog`, `directory`).

Per the per-file analyses: `ep_path.py` carries the EP catalog (`EP_CATALOG`) with vendor-requirement strings (`Intel`, `Qualcomm`, `AMD`, `NVIDIA`) used by L2's `is_compatible(ep)` check, plus `EPCatalog.Row` is locked via `__setattr__` after `__init__` so the catalog is immutable post-construction. The catalog ordering encodes deduction preference (QNN-NPU first, DML-GPU second, …) — same convention as a509a67's `EP_DEVICE_SPECS`.

Key files: `ep_path.py` (NEW, +1518), `session/ep_registry.py` (synthesis loop, `_entry_source_tag`), `session/__init__.py` (re-exports `BuiltinSource`, `EP_CATALOG`, `EPEntry`, `EPSource` and the five plugin sources), `commands/sys.py:_describe_source` (dispatches the `[bundled]` source-kind label).

### 2. `register_ep` idempotency — `dll_path` cache hit returns the cached `WinMLEP` instead of raising

The pre-refactor `register_ep` raised `WinMLEPRegistrationFailed("library already registered")` on a second call with the same `dll_path`. This broke two paths: (a) `auto_device`'s precedence retry loop (a `WinMLEPRegistrationFailed` from an earlier candidate left the registry primed; a later candidate using the same DLL never got a chance), and (b) `commands/sys.py:_gather_ep_info`'s broad walk over `all_discovered()` (every `(dll_path, ep)` row in the inventory triggered the second-call raise after the first row succeeded). The new shape is:

```python
if entry.dll_path in self._registered:
    return self._registered[entry.dll_path]
```

`auto_device` walks candidates in catalog precedence order, calls `register_ep(entry)`, and on success looks for a `WinMLDevice` with the requested `device_type`. On failure it records `last_error` and continues; if every candidate fails registration, it raises `WinMLEPRegistrationFailed` with the chained last-error traceback. The BuiltinSource branch goes through a separate `_builtin_registered: dict[str, WinMLEP]` cache keyed by `ep_name` (because all BuiltinSource entries share `Path("")` as `dll_path` and would collide in `_registered`); same idempotency contract.

⚠️ **Confirmed bug (Batch A finding):** the `last_error` variable is **never reset** after a successful registration. If candidate #1 raises `WinMLEPRegistrationFailed` and candidate #2 registers cleanly but exposes no matching device class, the precedence loop exhausts and `auto_device` re-raises `WinMLEPRegistrationFailed` with **candidate #1's** stale traceback — when the correct exception is `DeviceNotFound`. Single-line fix: `last_error = None` after the successful `register_ep` return inside the loop body.

Key files: `session/ep_registry.py` (`register_ep`, `auto_device`, `_builtin_registered`, `_available_eps_cache`), `commands/sys.py:_gather_ep_info` (now safely walks `all_discovered()`).

### 3. `EpAtSourceParamType` CLI parser — `--ep <name>[@<source-tag>]` wired at click parse time

`commands/_ep_arg.py` (NEW, +98 LOC) defines `split_ep_at_source(value: str) → tuple[str, str | None]` and `EpAtSourceParamType(click.ParamType)`. The pre-refactor pattern at every command (`perf`, `compile`, `build`, `config`) wrapped the split inside a try/except → `click.UsageError` block; the new ParamType collapses that to a one-line `type=EpAtSourceParamType()` on the option decorator. Source-tag validation runs at parse time against `VALID_SOURCE_TAGS = {"pypi", "nuget", "msix-microsoft", "msix-workload", "winml-catalog", "directory", "bundled"}`. EP-name case is preserved (`OpenVINOExecutionProvider` survives intact for the case-sensitive `_FULL_TO_SHORT` match inside `EPDeviceTarget.__post_init__`); only the source tag is lowercased.

The CLI migration is **deliberately partial**: `perf.py` and `compile.py` accept the source-tag and flow it through to `EPDeviceTarget.source`. `build.py` and `config.py` reject it at the CLI boundary with verbatim-identical try/except blocks (a simplification target — see DEEP-DIVE D-8). The `analyze` and `eval` commands continue to use the legacy short-string path.

Key files: `commands/_ep_arg.py` (NEW), `commands/perf.py:1269` (`type=EpAtSourceParamType()`), `commands/compile.py`, `commands/build.py` (rejects @-form), `commands/config.py` (rejects @-form).

### 4. `WinMLDevice(handle)` direct construction — `wrap_ort_device` factory deleted

The previous `wrap_ort_device(d: ort.OrtEpDevice) → WinMLDevice` was a one-line forward to `WinMLDevice(d)`. Its only historical justification was a per-EP `_DEVICE_CLASSES` dispatch map (selecting an ABC subclass at construction); that map was already collapsed into property-access-time dispatch in v1.4 of `docs/design/session/4_winml_device.md`. v2.10 of the doc records the shim's deletion in this commit. Production callers (`ep_registry.py:310, 352`) now construct via `WinMLDevice(h)` directly. Test sites (~70) were migrated mechanically.

⚠️ **Adjacent dead code (Batch A finding):** `WinMLDevice.ort_handle` is a public accessor documented as the "external API escape hatch", but **nobody uses it** — `session.py:_build_session_options` reaches for `ep_device.device._ort` directly. Either inline `_ort` or delete `ort_handle`.

Key files: `session/ep_device.py:WinMLDevice` (constructor + property-access dispatch on `self._ort.ep_name`), `session/__init__.py` (drops `wrap_ort_device` from `__all__`), `session/ep_registry.py:310,352` (direct construction).

### 5. `sys --list-ep` cleanup — static catalog claim deleted, L1/L2 status taxonomy in one place

Pre-refactor each EP header carried a static "→ Qualcomm NPU/GPU/CPU"-style line built by `_format_device_types(ep_name)` from `EP_DEVICE_SPECS`. The line was misleading for `[incompatible]` rows: the actual contributed devices were rendered per-source below it anyway, and the static claim made unsupported hardware look supported. **Removed wholesale** — `_format_device_types` is deleted, `EP_DEVICE_SPECS` is no longer imported into `sys.py`, the `record_by_ep[ep_name]` dict drops the `"device_types"` field, and the `_output_ep_text` header line collapses to just `f"  [bold]{ep_name}[/bold]{compat_tag}"`. The compact renderer (`_render_compact`) drops the `(device_types)` parenthetical.

The L1/L2 status taxonomy from `2_coreloop.md` §7.1 is implemented in exactly one place — `_gather_ep_info`'s status derivation loop (commands/sys.py:662):
```python
if err is not None:     desc["status"] = "failed"          # L1
elif not compatible:    desc["status"] = "incompatible"    # L2
else:                   desc["status"] = "primary" if not primary_seen else "shadowed"
```
The per-EP `[incompatible]` header tag is derived at render time (`_output_ep_text:779`) by checking `any(e["status"] in ("primary","shadowed") for e in entries)` — no duplicate field on the record.

A separate concern handled in the same pass: BuiltinSource entries carry `dll_path = Path()`, whose `str()` is `"."`. The renderer's `if entry.get("dll_path"):` guard previously rendered a stray `Path: .` line under every bundled row; this commit gates the assignment via `entry.is_filesystem_backed()` (new method on `EPEntry`, returns False for `BuiltinSource`) so `desc["dll_path"]` is `None` for bundled rows.

Indent constants `_INDENT_L2 = "    "`, `_INDENT_L3 = " " * 14`, `_INDENT_L4 = " " * 16` replace literal-space-count copies that were drifting across helpers.

Key files: `commands/sys.py` (~618 lines changed), `ep_path.py:EPEntry.is_filesystem_backed`.

### 6. Op-tracing → `session/monitor/` refactor (continuation of a509a67)

The `optracing/` package is deleted wholesale a second time (a509a67 had partially landed it under tests; 774f121d completes the move into `src/`). The new `session/monitor/` tree centres on `WinMLEPMonitor` ABC (formerly `EPMonitor`), with the **control-inversion** thesis from a509a67 made firmer: the old `OpTracer.run(iters, warmup)` owned the inference loop; the new `WinMLEPMonitor` is a passive context manager driven by `WinMLSession.perf(monitor=...)`. Three subclasses: `NullEPMonitor` (default), `QNNMonitor` (the concrete op-tracing implementation; ~649 lines of changes), and stubs `OpenVINOMonitor` / `VitisAIMonitor`.

QNN's resolution chain (FR-14) `L1 ONNX node.op_type` → `L2 EP-authoritative QHAS` → `L3 leaf-token heuristic` → `L4 raw op_path` is correctly wired in `QNNMonitor._resolve_op_type`; the QHAS parser + CSV parser are merged into a private `qnn/_internal.py` (+443 LOC) behind a `_`-prefixed boundary that a regression test enforces.

⚠️ **Real regression (Batch C finding):** `find_qnn_sdk()` lost its `_COMMON_SDK_PATHS` fallback (`D:\QC`, `C:\Qualcomm\AIStack\qairt`). Dev machines without `QNN_SDK_ROOT` set silently degrade to `basic_fallback` (no QHAS detail).
⚠️ **Lost defaults:** the old `QNNProfiler` hardcoded `backend_path=QnnHtp.dll`, `htp_performance_mode=high_performance`, `enable_htp_fp16_precision=1` are gone with no migration note. Bundled-ORT users who relied on those defaults need to set them explicitly.
🟡 **Dead code:** `OpenVINOMonitor` is a stub — `is_available()` returns `False`, never selected by `_resolve_ep_monitor`'s explicit dispatch in `commands/perf.py`.
🟡 **Lost extension point:** the pluggable-tracer `register_tracer` registry is dropped with no replacement. Adding a new EP monitor now requires editing `commands/perf.py`.

Key files: `session/monitor/{__init__,ep_monitor,hw_monitor,live_display,op_metrics,openvino_monitor,vitisai_monitor,qnn_monitor,report}.py`, `session/monitor/qnn/{__init__,_internal,viewer}.py`. Old `optracing/{__init__,base,registry,result,qnn/csv_parser,qnn/profiler,qnn/qhas_parser}.py` deleted.

### 7. Sysinfo refactor — `device.py` deleted, EP-vs-device routing moves to `session/ep_device.py`

The `sysinfo/device.py` (-191 LOC) is deleted. The EP-vs-device routing it owned (`_DEVICE_EP_MAP`, `_EP_DEVICE_MAP`, `get_ep_for_device`, etc.) was already migrated to `session/ep_device.py` in a509a67; this commit completes the deletion. Hardware-only inspection (`Cpu`, `Gpu`, `Npu`, `MemorySize`) lives in `sysinfo/hardware.py` (+29 LOC). `sysinfo/sysinfo.py` (-38) drops the old aggregation layer. `sysinfo/__init__.py` is updated to re-export only the hardware classes.

This refactor pairs with the `utils/constants.py` retirement: `SUPPORTED_EPS`, `EP_ALIASES`, `ALL_EP_NAMES`, `SUPPORTED_DEVICES` are gone. The remaining `utils/constants.py` carries only two ORT-enum bridge maps + a CLI-prefix tuple; the per-file review notes it is near-empty enough to delete entirely (DEEP-DIVE D-9).

Key files: `sysinfo/device.py` (DELETED), `sysinfo/hardware.py`, `sysinfo/__init__.py`, `sysinfo/sysinfo.py`, `utils/constants.py`.

### 8. `WinMLEPRegistry.instance()` singleton — `__new__` + `_initialized` guard deleted

The pre-refactor singleton used `__new__` to return the cached instance and a `_initialized` guard inside `__init__` to skip re-initialization. Per the per-file review (`session__ep_registry.md`), this was over-engineered: all 19 callers across `src/` and tests went through `.instance()` exclusively; zero direct `WinMLEPRegistry()` calls. The new shape:

```python
@classmethod
def instance(cls) -> WinMLEPRegistry:
    if cls._instance is None:
        cls._instance = cls()
    return cls._instance
```

with `__init__` unguarded. Tests reset via `WinMLEPRegistry._instance = None` then re-invoke `instance()`.

Key files: `session/ep_registry.py`.

## Internal inconsistencies

- **`VALID_EPS` vs `known_ep_short_names()` disagree.** `VALID_EPS` is 8 short names derived from `EP_DEVICE_SPECS` (catalog rows); `known_ep_short_names()` is 9 short names derived from `_SHORT_TO_FULL` (the short→full alias map), including `cuda`. `EPDeviceTarget(ep="cuda", ...)` passes validation but has no catalog row downstream — a silent latent crash when `default_device_for_ep("CUDAExecutionProvider")` returns `None`. (Batch A.)
- **`auto.py` `device_type` casing inconsistency** — `auto.py:411` originally passed `ep_device.device.device_type` (uppercase `"GPU"`) to `build_hf_model`, while three sibling call sites at `:173 :218 :356` used `.lower()`. The fix landed in this squash (verified in `models__auto.md`).
- **`commands/sys.py:_describe_source` vs `session/ep_registry.py:_entry_source_tag`** — same isinstance-on-`EPSource` dispatch implemented twice, with the risk that adding a new `EPSource` subclass requires updating both. The audit doc recommends `_describe_source` delegate to `_entry_source_tag`. Not done in this squash.
- **`session/monitor/__init__.py` empty `__all__`** — same pattern as a509a67; consumers reach for monitor classes via their concrete-class modules, which is fine, but the empty `__all__` makes the public surface implicit.

## Breaking changes for callers

This commit follows the **Option A — no compat shims** discipline (per `feedback_no_back_compat` memory). Externally visible breaks:

- **`wrap_ort_device(handle)` is gone.** Construct `WinMLDevice(handle)` directly.
- **`SUPPORTED_EPS` / `EP_ALIASES` / `ALL_EP_NAMES` / `SUPPORTED_DEVICES`** are deleted from `utils/constants.py`. Use `session.VALID_EPS` / `session.VALID_DEVICES` / `session.expand_ep_name` / `session.short_ep_name`.
- **`get_ep_device_map()`** and the rest of `sysinfo/device.py` are deleted. Use `session.default_ep_for_device` / `session.eps_for_device` / `session.lookup_device_spec`.
- **`WinML` top-level class is deprecated.** Every public method emits `DeprecationWarning`; new code should construct `WinMLSession(...)` directly.
- **`OpTracer` ABC and `optracing.*` are deleted.** Use `WinMLEPMonitor` subclasses with `WinMLSession.perf(monitor=...)`.
- **`commands.config.py` no longer accepts `--ep <name>@<source>`** — same for `build.py`. Use `--ep <name>` only.

## Verification

The change set was exercised end-to-end via live commands on a Lunar Lake box (Intel Core Ultra 7 258V, Arc 140V GPU, AI Boost NPU):

- `winml sys --list-ep` → renders four EPs (OpenVINO PyPI primary + 3 MSIX shadowed; QNN PyPI `[incompatible]`; CPU bundled; DML bundled) without the `-> Qualcomm NPU/GPU/CPU` static-claim header; no spurious `Path: .` lines under bundled rows.
- `winml sys --list-ep --format json` → top-level fields per EP: `["entries"]`; first entry fields: `["source_kind", "version", "distribution", "status", "dll_path", "devices"]`. No `"device_types"` field.
- `winml sys --list-ep --format compact` → `EPs: OpenVINOExecutionProvider, QNNExecutionProvider, CPUExecutionProvider, DmlExecutionProvider`. No `(device_types)` parenthetical.
- `winml perf -m microsoft/resnet-50 --device gpu --iterations 30` → 169.74 samples/sec on Intel Arc 140V; resolve_device log: `auto/gpu -> DmlExecutionProvider/gpu`.
- `winml perf -m microsoft/resnet-50 --monitor --iterations 30` → 311.35 samples/sec on Intel AI Boost NPU.

Test suite: **990/990 pass** across `tests/unit/commands/`, `tests/unit/session/`, `tests/unit/ep_path/test_discover_eps_dedup.py` (8 hardware-gated skips). Ruff: `src/winml/modelkit/commands/sys.py`, `session/ep_registry.py` clean; `models/auto.py` has 3 pre-existing `F821` errors (unrelated `Undefined name 'device'/'ep'` in the composite-model dispatch branch at `auto.py:148-150`, on baseline).

## Reading order recommended for newcomers

1. **`SUMMARY.md`** (this doc) — orient.
2. **`per-file/ep_path.md`** — understand the unified `EPSource` taxonomy + `EPCatalog`.
3. **`per-file/session__ep_device.md`** — `EPDeviceTarget`, `WinMLDevice`, `resolve_device`, deduction helpers.
4. **`per-file/session__ep_registry.md`** — `WinMLEPRegistry.{register_ep, auto_device}` + the BuiltinSource synthesis + the `auto_device` `last_error` bug.
5. **`per-file/commands__sys.md`** — Path B / Tier-2 broad-walk inventory + L1/L2 status taxonomy.
6. **`per-file/commands__perf.md`** — Path A typed flow (`resolve_device → auto_device → WinMLEPDevice` → `WinMLSession.perf(monitor=...)`).
7. **`per-file/session__monitor__qnn_monitor.md`** + **`per-file/session__monitor__qnn___internal.md`** — op-tracing pipeline and the FR-14 fallback chain.
8. **`DEEP-DIVE.md`** — impl vs design divergences, regressions, simplification opportunities.
9. **`DESIGN-DOCS-INDEX.md`** — catalog of every design / plan / review doc that the squash references or supersedes.
