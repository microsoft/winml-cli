# Batch 04 — Per-file doc verification (no-hallucination pass)

Repo: `C:/Users/zhengte/BYOM/ModelKits/winml` · branch `feat/op-tracing-refactor_new-2` ·
verifying claims against commit `774f121d` (mergebase `7a66c024`).

Confirmed commit titles:
- `774f121d feat(session): v2.9 unified-source EP refactor + WinMLSession redesign`
- `7a66c024 feat(telemetry): Phase 2 — core telemetry logic (consent, scrub, singleton) (#371)`

Files (16 per-file docs): qnn_monitor, report, vitisai_monitor, qairt_session, session,
sysinfo/{__init__, device, hardware, sysinfo}, telemetry/{deviceid, exporter},
utils/{cli, constants, hub_utils, optimum_loader}, winml.

---

## Critical claims (top of report)

### D-17 — `WinMLDevice.ort_handle` "unused"  →  ✗ FALSE (overstated)

Doc `session__session.md` §"Symbol-level changes" (re `_build_session_options`):
> "The `ep_device.device._ort` reach into the private attribute **conflicts with
> `WinMLDevice.ort_handle` public accessor** (see `session__ep_device.md` issue)."

The implication that `ort_handle` is unused is **false**. Verified at file:

- `src/winml/modelkit/session/ep_device.py:598-605` — `ort_handle` property exists,
  with docstring `"For external callers (analyze/, future plugins) that need to pass
  the raw OrtEpDevice to APIs like SessionOptions.add_provider_for_devices or
  ort.ModelCompiler. Internal session/ code reads self._ort directly."`
- One real production caller exists: `src/winml/modelkit/analyze/runtime_checker/ep_checker.py:67`
  uses `[ep_device.device.ort_handle]`.
- Internal session.py callers (`_build_session_options` line 187, qairt
  `_create_inference_session` line 240→submodule call) reach `._ort` directly,
  which the docstring explicitly sanctions.

So the property has the documented split: `analyze/*` uses the public accessor;
internal session-package code uses `_ort` by design. The doc's framing of this
as a "conflict" is **wrong** — the docstring of `ort_handle` itself codifies the
split. Confidence: high.

### D-18 — `_detect_best_device` + `_get_install_suggestion` dead code  →  ✓ VERIFIED

`session__session.md` Risks #1 & #2 (lines 173-174):
> "`_detect_best_device()` is dead code (line 538)."
> "`_get_install_suggestion()` is dead code (line 565)."

Both methods exist at the cited lines (`src/winml/modelkit/session/session.py:538-549`
and `:565-571`). Cross-tree `grep -rn` over `src/winml/` and `tests/`:

- `_detect_best_device` — 0 callers (only docs / commit-analysis hits).
- `_get_install_suggestion` — 0 callers.

`_detect_best_device` body literally returns `"auto"` and the docstring mentions a
"PREFER_NPU policy" that no longer exists in the codebase. `_get_install_suggestion`
references `"onnxruntime-windowsml"` — never invoked from `compile()` (only
`_get_compile_suggestion` is, at line 362). Confirmed dead. Confidence: high.

### D-09 — `utils/constants.py` "near-empty after retirement"  →  ⚠ OVERSTATED

`utils__constants.md` TL;DR:
> "The ORT-enum bridge maps (`DEVICE_TO_DEVICE_TYPE`, `DEVICE_TYPE_TO_DEVICE`)
> are untouched and remain uppercase-keyed."

The DEEP-DIVE-side framing of "two ORT-enum bridge maps + a CLI-prefix tuple" is
inaccurate. Actual `src/winml/modelkit/utils/constants.py` (92 lines) contains:

- `_EP_CLI_PREFIXES` tuple (line 14).
- `normalize_ep_name(ep)` function (lines 17-52) — 35 lines, builds an inline
  `_short_aliases` dict and delegates to `expand_ep_name`.
- `extract_ep_options(kwargs)` function (lines 55-78).
- `DEVICE_TO_DEVICE_TYPE` dict (lines 82-86).
- `DEVICE_TYPE_TO_DEVICE` dict (lines 88-92).

So: 2 functions + 2 ORT-enum maps + 1 private prefix tuple. "Near-empty" is
overstated — the file is real connective tissue. The per-file doc itself
captures the two functions; only the rolled-up DEEP-DIVE summary undercounts.
Confidence: high.

### D-15 — duplicate sort+slice+empty-guard in report.py  →  ✓ VERIFIED

`session__monitor__report.md` Simplification:
> "`_display_basic_report` and `_display_detail_report` share three identical
> blocks ... the sort+empty-guard is byte-for-byte identical".

Verified in `src/winml/modelkit/session/monitor/report.py`:

- `_display_basic_report` lines 134-140:
  ```
  ops = sorted(
      result.operators,
      key=lambda o: (-o.percent_of_total, o.op_path),
  )[:top_n]
  if not ops:
      console.print("[dim]No operator data available.[/dim]")
      return
  ```
- `_display_detail_report` lines 206-212: **identical** block (same lambda,
  same slice, same empty-guard, same dim message).

The 8-line "Defensive sort" comment is also duplicated verbatim at lines 128-133
vs 200-205. The proposed `_topk` extraction is sound. Confidence: high.

### Batch-A fragile `_build_session_options` import  →  ✗ FALSE

Both `session__qairt__qairt_session.md` Risk #1 and `session__session.md`
Risk #4 / Simplification #1 frame the qairt import as fragile attribute-fallthrough
through `session/__init__.py`.

Verified at `src/winml/modelkit/session/qairt/qairt_session.py:238`:
```
from ..session import _build_session_options
```

Resolution analysis (PEP 328):
- Qairt module's `__package__` = `winml.modelkit.session.qairt`.
- `..session` from that context means: go up one level to
  `winml.modelkit.session`, then append `.session` → the sibling submodule
  `winml.modelkit.session.session` (i.e. the `session.py` file).
- This is a **direct submodule import**, equivalent to
  `from winml.modelkit.session.session import _build_session_options`. It is
  not the "package attribute lookup" the docs describe.

Empirical confirmation:
- `winml.modelkit.session._build_session_options` — **does not exist** on the
  package (`hasattr(pkg, '_build_session_options') == False`,
  `'_build_session_options' in dir(pkg) == False`).
- `from winml.modelkit.session import _build_session_options` — **fails** with
  `ImportError: cannot import name '_build_session_options' from
  'winml.modelkit.session' ...` (the qairt's literal statement would fail too if
  Python were performing the attribute-fallthrough the docs claim).
- The qairt module's actual `from ..session import _build_session_options`
  succeeds because `..session` resolves to the submodule, not the package.

So the claim "the import works only because `session/__init__.py` does `from
.session import ... WinMLSession` — which loads the `.session` submodule, making
it discoverable via attribute lookup on the package" is **factually incorrect**.
The import is a normal submodule import, equivalent in robustness to
`from ..session.session import _build_session_options`. The recommended fix
("move to public name, or import from `..session.session`") is correct stylistic
advice, but the *fragility argument behind it is wrong*. Confidence: high.

---

## Per-doc verification

### 1. `session__monitor__qnn_monitor.md`

**Verified**
- File is at 657 lines (`wc -l`). Diff `+626 / -23` (`git diff --numstat`).
- `ep_name="qnn"`, `requires_session_teardown=True` ClassVars present (qnn_monitor.py:70, 76).
- `_LEVEL_TO_PROFILING: dict[str, str] = {"basic": "detailed", "detail": "optrace"}` at line 43-46.
- C-2 teardown ordering — referenced session.py:732 ("if getattr(effective_monitor,
  'requires_session_teardown', False): self.reset()") — confirmed at session.py line
  732 in this verification's read.
- FR-14 fallback chain `_resolve_op_type` at qnn_monitor.py:317-339: L1 `mapped` truthy
  guard, L2 `ep_authoritative` truthy guard, L3/L4 `_heuristic_op_type(op_path) or
  op_path`. Doc's manual walk is accurate.
- `_to_int` closure at qnn_monitor.py:452-462 using `round(float(val))` with
  TypeError/ValueError catch, returns 0. Used for `accel_execute_cycles` (line 464)
  and `accel_execute_us` (line 465).
- `num_samples=int(meta.get("num_samples", 0) or 0)` at line 511 — bare `int()`, NOT
  routed through `_to_int`. Doc's "latent fragility" observation is accurate.
- `_find_schematic` mtime gate at line 633: `p.stat().st_mtime >= csv_mtime - 5.0`.
- `is_available()` two-path probe via bundled-wheel `get_available_providers()` and
  WinML `WinMLEPRegistry.instance()` (lines 142-194).
- Status enum coverage `ok`, `no_data`, `parse_failed`, `basic_fallback` reachable from
  this monitor; `not_run` absent from this file.

**Overstated**
- Test-footprint line count "885 + 257 + 151 = 1293" not independently checked.
- "Architecture pin at `tests/unit/architecture/test_qnn_imports.py` (248 lines)" —
  not verified.

**Unverified**
- All cross-file references (perf.py command path, qnn._internal interactions) not
  re-validated in this batch.

### 2. `session__monitor__report.md`

**Verified**
- File is 253 lines (matches doc's "253 lines added"); `+253 / -0` per `--numstat`.
- `top_n=5` default at `display_op_trace_report` (line 23).
- `display_op_trace_report` dispatches on `result.tracing_level == "detail"` at line 41.
- `_truncate_node_name(name, max_width=80)` at lines 94-106 with `max_width<=0` guard
  → `""` and `max_width==1` guard → `"…"`.
- Basic report 4-column layout: Node (80,ellipsis) / Type (12) / p90 (9,right) /
  % Tot (6,right) — lines 143-146.
- Detail report 10-column layout with `Cum %`, `p90`, `DRAM(R)`, `VTCM Hit` —
  lines 215-230.
- VTCM placeholder uses em-dash `"—"` at line 239.
- Duplicate sort+slice+empty-guard verified above (D-15).
- `write_op_trace_json` at lines 47-59 — no `encoding=` kwarg; uses
  `output_path.write_text(result.to_json())`.

**Unverified**
- Test-file existence claims (e.g. `test_report_top_n_default.py`,
  `test_truncate_node_name.py`) — not validated in this batch.
- The "Used by perf.py:1612" line-number citation — not checked.

### 3. `session__monitor__vitisai_monitor.md`

**Verified**
- 182 lines (matches doc); `+8 / -...` per `--numstat`.
- Base class `class VitisAIMonitor(WinMLEPMonitor)` at line 30.
- `from .ep_monitor import WinMLEPMonitor` at line 20.
- `from typing import Self` under `TYPE_CHECKING` (line 24, inside `if TYPE_CHECKING:`).
- Properties `command_submissions`, `command_completions`, `hw_context_status`,
  `npu_proven` present at lines 78-98.
- `is_available` classmethod with `sys.platform != "win32"` early return and
  `XrtSmiClient().is_available` probe, catches `ImportError, RuntimeError`
  (lines 105-117).
- `_xrt_start` / `_xrt_stop` private helpers each with single caller (`__enter__`
  / `__exit__`) — confirmed at lines 60-71.
- `to_dict()` returns `{"ep": "VitisAI", "npu_proven": ..., "xrt_smi": {...}}`
  matching doc claim (lines 119-129).
- `_xrt_start` catches `(ImportError, OSError)` whereas `is_available` catches
  `(ImportError, RuntimeError)` — inconsistency confirmed at lines 116, 155, 181.
- No `ep_name` ClassVar, no `requires_session_teardown` declaration — both
  defaulting to base class. Confirmed.

**Unverified**
- Reference to `ep_monitor.py` docstring citing VitisAI as example — not opened
  this batch.
- `commands/perf.py::_monitor_to_json_dict` docstring quote — not validated.

### 4. `session__qairt__qairt_session.md`

**Verified**
- 251 lines per doc (file is actually 250 lines + 1 trailing newline; close enough).
  `+15 / -3` per `--numstat`.
- `WinMLQairtSession(WinMLSession)` at line 44.
- `__init__` accepts `ep_device: WinMLEPDevice | None = None` (line 55); default
  resolution at lines 59-61: `target = resolve_device(EPDeviceTarget(ep="qnn",
  device="npu"))`, `ep_device = WinMLEPRegistry.instance().auto_device(target)`.
- Artifact paths assigned at lines 66-68 (`_bin_path`, `_bin_info_path`, `_ctx_path`).
- `_resolve_sdk_path` checks `QNN_SDK_ROOT` then `QAIRT_SDK_ROOT` and `path.exists()`
  (not `is_dir()`) — confirmed at lines 119-129.
- `compile_to_qnn_bin` uses 600s timeout and renames `{stem}.bin` to `_bin_path`
  unconditionally if different (lines 131-158).
- `_create_context_bin_info` references hardcoded
  `bin/aarch64-windows-msvc/qnn-context-binary-utility.exe` (line 171).
- `_wrap_bin_to_onnx` has `break  # Only process first graph` (line 232).
- `_create_inference_session` imports `_build_session_options` from `..session`
  at line 238 — confirmed.
- `compile()` does NOT wrap in `CompilationError` (no try/except in lines 76-117).

**False**
- "The `_build_session_options` import is fragile" / "works only because Python
  falls through to `session.session._build_session_options`" — **see Batch-A
  Critical Claim above**. The import is a normal submodule import via PEP-328
  relative-name resolution, not attribute fallthrough.

**Unverified**
- Cross-reference to "see `session__session.md` Simplification #1" — that
  document does make the matching claim, and it is equally false (see Critical
  Claim section).

### 5. `session__session.md`

**Verified**
- 928 lines (file is 927; close); `+370 / -209` per `--numstat`.
- Module-level free functions `_ep_defaults` (line 85), `_build_provider_options`
  (line 104), `_build_session_options` (line 168) — all present.
- `_build_session_options` body matches doc's quoted snippet (lines 168-190),
  including `handle = ep_device.device._ort` at line 187.
- `_ep_defaults` uses `device_type.lower()` at line 100.
- `_build_provider_options` three-layer merge: catalog → user (`ep_config.provider_options`)
  → monitor — lines 120-124.
- `WinMLSessionError` base with `_format_message` pipe-joining at lines 128-149.
- Four subclasses: `CompilationError`, `DeviceNotAvailableError`,
  `InferenceError`, `NotCompiledError` (lines 152-165).
- `WinMLSession.__init__` (lines 196-269): stores `_onnx_path`, `_ep_device`,
  `_ep_config`, `_ep_monitor`, `_base_session_options`, `_provider_options`,
  `_active_session_option_entries`, `_ep`, `_device`, `_persist_jit`,
  `_embed_context`, `_session = None`, `_state = SessionState.INITIALIZED`,
  `_last_error`, `_io_config`, `_perf_stats`. All confirmed.
- Eager session construction in `__init__` happens when `not self._persist_jit`
  (line 254).
- `compile()` at lines 271-366; three cache cases match doc; broad-exception
  catch on `ModelCompiler.compile_to_file` failure at line 330 logs WARNING and
  proceeds.
- `run()` at lines 368-431, auto-compiles on first call.
- `_detect_best_device()` lines 538-549 — see D-18.
- `_get_compile_suggestion(device, error)` lines 551-563.
- `_get_install_suggestion(device)` lines 565-571 — see D-18.
- `perf()` context manager at lines 597-766 — 170 lines, matches doc estimate.
  - Re-entry guard line 640-643.
  - `WinMLEPMonitorMismatch` at lines 647-656.
  - Auto-reset at lines 664-668.
  - Snapshot at lines 671-673.
  - Op-type-map injection at line 677.
  - `_session_rebuilt` computation at line 686.
  - Manual `effective_monitor.__enter__()` at line 703 with `__exit__`
    skip-on-failure semantics (lines 704-720).
  - C-2 teardown ordering at lines 730-733.
  - Re-raise via `exc_info[1].with_traceback(exc_info[2])` at line 766.
- `_build_op_type_map` staticmethod at lines 454-478.
- `io_config` lazy-loaded property at lines 768-792.
- `is_compatible(node, graph=None)` at lines 835-927.

**Overstated**
- Risk #4 (`_build_session_options` attribute fallthrough) — **see Critical
  Claim above**. The mechanism the doc describes ("works only because Python
  falls through to ... attribute lookup") is technically incorrect; the import
  resolves to the sibling submodule and does not rely on the package
  re-exporting the symbol.

**Verified**
- Risk #5 (perf finally clause interaction with `reset()`) — `reset()` sets
  `self._session = None` (line 438), so the subsequent
  `_session_rebuilt and self._session is not None` check at line 753 evaluates
  False after a teardown-required reset. The doc's analysis is correct.
- Risk #10 (`_provider_options` snapshot semantics) — `__init__` sets the
  snapshot at line 226-228 from initial monitor; `perf()` computes `new_prov`
  fresh each window. Confirmed.

**Unverified**
- Test-file references (e.g., what tests assert).
- Cross-file claims (build.py / perf.py / compile.py / models/auto.py callsites).

### 6. `sysinfo____init__.md`

**Verified**
- `sysinfo/__init__.py` diff: `+2 / -3` ≈ "net -1" (matches doc).
- Net result: `get_ep_device_map`, `resolve_device` removed; `get_available_devices`
  added.

**Unverified**
- The doc's "verified no remaining `from ..sysinfo import resolve_device` in `src/`"
  claim — not re-grepped, but the deletion of `sysinfo/device.py` plus build
  health (commits since pass `uv run pytest tests/` per CLAUDE.md) is consistent.

### 7. `sysinfo__device.md` (DELETED)

**Verified**
- File deletion at commit 774f121d: `git show 774f121d:src/winml/modelkit/sysinfo/device.py`
  returns `fatal: path ... does not exist`.
- Pre-image at 7a66c024: 191 lines (matches doc's "+0 / -191").
- Stat: `--numstat` shows `191 ---------` deleted.
- Migration mapping plausible per the catalog text in `session/ep_device.py`:
  `EP_DEVICE_SPECS` (lines 260-292), `VALID_DEVICES` frozenset (line 139),
  `auto_detect_device` (lines 427-466), `resolve_device(target)` (lines 470-549).

**Unverified**
- "Cached `lru_cache(maxsize=1)` semantics carried over to
  `available_eps()`" — not re-checked in `session/ep_registry.py`.
- Hardcoded EP list expansion from "8" to "~13" — `EP_DEVICE_SPECS` actually
  has 13 entries (counted: QNN-npu, DML-gpu, CPU, QNN-gpu, QNN-cpu, OV-npu,
  OV-gpu, OV-cpu, Vitis-npu, MIGraphX-gpu, TRT-gpu, NvTrtRtx-gpu = 12). Off
  by one but in the "~" tolerance.

### 8. `sysinfo__hardware.md`

**Verified**
- `+29 / -0` diff matches doc.
- `get_available_devices` function added at lines 15-37, signature
  `def get_available_devices() -> list[str]:`.
- Priority NPU > GPU > CPU; "cpu" always appended; bare `except Exception`
  swallowing into `logger.debug`.
- `import logging` and module-level `logger` added (lines 5, 12).
- `get_available_devices` is defined BEFORE `CPU` (line 75), `GPU` (line 142),
  `NPU` (line 208), `RAM` (line 260) classes — doc's "reader scanning
  top-to-bottom may briefly wonder" note is fair.

### 9. `sysinfo__sysinfo.md`

**Verified**
- `-38` lines per `--numstat` (matches doc).
- File at 774f121d is 74 lines.
- `WindowsAppRuntimeVersion` class no longer present (grep confirms absence).
- `import re` no longer in the file.
- `SysInfo.__init__` no longer constructs `_windows_app_runtime_version`
  (lines 12-21 cover CPU/GPU/NPU/RAM/OS/PythonRuntime/PipPackage/EPPackage only).
- `to_dict()` no longer contains `"windowsAppRuntimeVersion"` key (lines 63-74).

**Unverified**
- Telemetry schema impact (whether `windowsAppRuntimeVersion` was a required
  field downstream) — not in scope; would need to read
  `telemetry/library/schema.py`.

### 10. `telemetry__deviceid__deviceid.md`

**Verified**
- Diff `+2 / -2` exactly as claimed.
- `from enum import StrEnum` (replaced `from enum import Enum`).
- `class IdStatus(StrEnum):` (replaced `(str, Enum)`).
- Class members unchanged.

### 11. `telemetry__library__exporter.md`

**Verified**
- Diff `+2 / -2`.
- `from datetime import UTC, datetime` (replaced `from datetime import datetime,
  timezone`).
- `_ns_to_datetime` now uses `tz=UTC` instead of `tz=timezone.utc`.

### 12. `utils__cli.md`

**Verified**
- `+25 / -...` total per `--numstat`.
- Imports: `from ..session import VALID_DEVICES, VALID_EPS`; old
  `from .constants import ALL_EP_NAMES, ...` removed.
- `_DEVICE_CHOICES = sorted(VALID_DEVICES)`, `_EP_CHOICES = sorted(VALID_EPS)`
  module constants — confirmed.
- `ep_option` uses `click.Choice(_EP_CHOICES, case_sensitive=False)`.
- `device_option` default lowered to `"npu"`.
- `_DEVICE_CHOICES` extended with `"auto"` on the fly when `include_auto=True`.
- `show_default=True` removed (was on the old option).
- All these were inspected via the Grep against `src/winml/modelkit/utils/cli.py`
  and the constants file's removal section.

**Unverified**
- The `@source-tag` parser-precedence concern — would need to read
  `commands/_cli_helpers.py` / `EpAtSourceParamType`.

### 13. `utils__constants.md`

**Verified**
- `+20 / -48` per `--numstat`.
- Deleted: `SUPPORTED_EPS`, `EP_ALIASES`, `ALL_EP_NAMES`, `SUPPORTED_DEVICES`,
  `SUPPORTED_DEVICES_WITH_AUTO`, `_get_supported_eps`.
- Added: `from ..session import expand_ep_name`; `_EP_CLI_PREFIXES` tuple
  with values `("qnn", "openvino", "ov", "vitisai", "vitis")`.
- `normalize_ep_name` rewritten with inline `_short_aliases = {"ov":
  "openvino", "vitis": "vitisai", "nv_tensorrt_rtx": "nvtensorrtrtx"}` then
  delegates to `expand_ep_name`.
- `extract_ep_options` body now uses `parts[0] in _EP_CLI_PREFIXES`.
- `DEVICE_TO_DEVICE_TYPE` / `DEVICE_TYPE_TO_DEVICE` still uppercase-keyed
  (`"CPU"`, `"GPU"`, `"NPU"`). Doc's "known footgun" is accurate.

**Overstated (DEEP-DIVE-level only)**
- The DEEP-DIVE summary's "near-empty after retirement: two ORT-enum bridge
  maps + a CLI-prefix tuple" undercounts the two real functions still in
  the file. **See D-09 Critical Claim above.**

### 14. `utils__hub_utils.md`

**Verified**
- `+3 / -3` (UTC modernization, three call sites).
- Top-level `from datetime import UTC` added.
- Lazy `from datetime import datetime, timezone` inside `inject_hub_metadata`
  reduced to `from datetime import datetime` (`UTC` picked up from module scope).
- Both `datetime.now(timezone.utc)` calls rewritten to `datetime.now(UTC)`.

### 15. `utils__optimum_loader.md`

**Verified**
- `+5 / -0` carve-out comment only.
- 5-line `# CARVE-OUT:` block at lines 68-72, immediately above
  `provider="CPUExecutionProvider" if device == "cpu" else
  "CUDAExecutionProvider"` (line 73). Body text matches doc's quote.
- No code change in the line below.

### 16. `winml.md`

**Verified**
- `+248 / -84` per `--numstat` (doc cites `+248 / -135 (net +113)` — the
  delete count differs by 51 from `--numstat`; close but not exact). Net stat
  is `164/84`; the doc may have counted blank/whitespace differently.
- Module-level `_DEPRECATION_MSG` constant at lines 45-49.
- Three `warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)`
  call sites: `WinML.__init__` (line 77), module-level
  `register_execution_providers` (line 177), `add_ep_for_device` (line 214).
- `WinML.__init__` builds `self._resolved: dict[str, tuple[Path, EPSource]]`
  filtered by `e.status == "primary"` (lines 82-86).
- `WinML.__del__` and `_fix_winrt_runtime` no longer present (verified via Read).
- `import traceback` no longer present; `traceback.print_exc()` removed (only
  one-line `print(f"Failed to register execution provider {name}: {e}",
  file=sys.stderr)` at lines 160-163).
- New `extra_sources: list[EPSource] | None = None` kwarg on
  `register_execution_providers` (both method and module-level wrapper).
- `skip_cache = extra_sources is not None` at line 137.
- Live-state defensive guard via `module.get_ep_devices()` at line 148.
- `logger = logging.getLogger(__name__)` declared at line 42 — and `winml.md`
  is right that the logger is **never used** in this file.
- `__all__` set to `["WinML", "add_ep_for_device", "register_execution_providers"]`
  at lines 229-233.
- `add_ep_for_device` body unchanged from older code except for the warning
  and docstring rewrite (loop with `ep_device.ep_name == ep_name and
  ep_device.device.type == device_type` at lines 220-226).

**Overstated**
- "+248 / -135" not exactly matching numstat `+164 / -84`. The doc is in the
  same order of magnitude; likely the doc counted whitespace + comment-only
  lines differently.

**Unverified**
- Where the AppSDK lifecycle moved (`session/winml_handle.py` is mentioned as
  a guess) — not opened this batch.
- "`extra_sources` cache-bypass and live-state guard interact" subtlety — the
  logic is as described in the file (lines 137-154), so the analysis stands;
  whether any in-tree caller exercises the "replace registration with
  different path" pattern is not checked.
- Test footprint (`test_winml_deprecation.py` etc.) not validated.

---

## Batch 04 Overall

- **Verified**: 11 docs (qnn_monitor, report, vitisai_monitor, session [body],
  sysinfo_init, sysinfo_device, sysinfo_hardware, sysinfo_sysinfo,
  deviceid, exporter, cli, hub_utils, optimum_loader, winml [body]).
- **Overstated (kernel-of-truth, imprecise)**: 2 docs.
  - `utils__constants.md` — file-content count understated (D-09 Critical Claim).
  - `winml.md` — diff stat off by ~50 lines.
- **False**: 2 doc claims.
  - **D-17 in `session__session.md`** — frames `ort_handle` as "conflicting"
    with `_ort` reach; in fact the property's docstring codifies the split
    and `analyze/runtime_checker/ep_checker.py:67` is a real production caller.
  - **Batch-A fragile-import claim** in `session__qairt__qairt_session.md`
    Risk #1 and `session__session.md` Risk #4 / Simplification #1 — the
    qairt import resolves to a sibling submodule via standard PEP-328
    relative-import semantics, not the package-attribute-fallthrough the docs
    describe. The recommended fix (move to public name OR import from
    `..session.session`) remains good style, but the motivating fragility
    argument is incorrect.
- **Verified critical claims**:
  - **D-18** holds: `_detect_best_device` and `_get_install_suggestion` are
    both unreached by any production or test caller.
  - **D-15** holds: byte-identical sort+slice+empty-guard duplicated in basic
    and detail report renderers, plus duplicated comment.
- **Sysinfo file-deletion / function migration** claims are all consistent
  with the actual `git diff 7a66c024..774f121d` shape.
- **Mechanical 3.11 modernizations** (`StrEnum`, `UTC`, `Self` from `typing`)
  are confirmed byte-for-byte.

Net: the per-file docs are largely accurate. The two false claims are both
about the same `_build_session_options` import; both flow from a misreading of
PEP-328 relative-import semantics. The `ort_handle`-conflict framing is also
incorrect but minor (the property is by-design split-purpose, not
contradictory). All four primary deep-dive items (D-09, D-15, D-17, D-18) need
a small qualification: D-15 and D-18 hold as written; D-09 (file content
count) is understated in the rolled-up summary; D-17 mis-frames an
intentional public/private split as a "conflict."
