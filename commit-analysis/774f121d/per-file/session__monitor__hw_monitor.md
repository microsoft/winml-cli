# src/winml/modelkit/session/monitor/hw_monitor.py

## TL;DR
Cosmetic-only diff: two doc references to the old `EPMonitor` class name renamed to `WinMLEPMonitor` (the post-refactor name in `ep_monitor.py`), and the `Self` type alias is sourced from stdlib `typing` instead of `typing_extensions` (Python 3.11+ baseline). No behavior, no API surface, no metric-collection logic changed.

## Diff metrics
- Lines added: 3
- Lines removed: 3
- Modified — file existed at 170 lines pre-commit, 171 post-commit.

## Role before vs after
- **Before:** PDH-backed system-wide hardware monitor — `HWMonitor` context manager polling CPU, RAM, and NPU/GPU utilization via Windows PDH counters. Doc strings referenced the sibling ABC as `EPMonitor`.
- **After:** Identical responsibility. Doc strings updated for the renamed ABC (`WinMLEPMonitor`). `Self` import migrated from `typing_extensions` to `typing` (PEP 673 is stdlib-native on 3.11+).

## Symbol-level changes
- **Module docstring** — modified
  - Line 9: `"Works independently of the EPMonitor hierarchy."` → `"Works independently of the WinMLEPMonitor hierarchy."`.
- **`HWMonitor` class docstring** — modified
  - Line 31: `"Independent of the EPMonitor hierarchy — provides system-wide …"` → `"Independent of the WinMLEPMonitor hierarchy …"`.
- **`TYPE_CHECKING` block** — modified
  - `from typing_extensions import Self` → `from typing import Self`.
- All other symbols unchanged: `__init__(poll_interval_ms=200)`, `__enter__`/`__exit__`, the metric properties (`mean_utilization_pct`, `peak_utilization_pct`, `peak_memory_mb`, `peak_memory_local_mb`, `peak_memory_shared_mb`, `mean_cpu_pct`, `peak_cpu_pct`, `ram_used_mb`, `peak_ram_used_mb`), `is_available()`, `to_dict()`, and the chart-compatible properties (`utilization_samples`, `cpu_samples`, `memory_samples_mb`).

## Behavior / contract changes
- None. Pure rename-pass-through.

## Cross-file impact
- **Used by which modules:** Imported into `session/__init__.py` (`HWMonitor` is in `__all__`); used as the background hardware sampler by `commands/perf.py`, `commands/eval.py`, and `commands/_live_chart.py::LiveMonitorDisplay`.
- **Depends on which modules:** `._pdh.PdhPoller` (sibling module).

## Risks / subtleties
- `HWMonitor` is deliberately **not** a `WinMLEPMonitor` subclass — the class docstring is explicit ("Independent of the WinMLEPMonitor hierarchy"). It still defines `to_dict()` (returning a dict-of-dicts of CPU/RAM/NPU metrics) and `is_available()` (returning `sys.platform == "win32"`); those happen to match the deleted ABC contract but are not enforced by the ABC. This is intentional — a single shared concrete `HWMonitor` runs alongside whichever per-EP `WinMLEPMonitor` (or `NullEPMonitor`) is active.
- The `typing_extensions` → `typing` migration assumes the repo's minimum Python is ≥ 3.11 (where `Self` landed in stdlib). `pyproject.toml` should confirm; an older minimum would re-break this.

## Open questions / TODOs surfaced
- None surfaced by this diff.

## Simplification opportunities
- The `to_dict()` here pre-dates the `WinMLEPMonitor.result: OpTraceResult | None` typed-accessor pattern. Op-tracing monitors now expose data via `monitor.result`, leaving `HWMonitor` as the only first-party monitor still emitting a dict-shaped serialization. The PRD/perf docs reference an `HWStats` typed accessor as the long-term home; until that lands, `HWMonitor.to_dict()` is the de-facto schema and consumers (`commands/perf.py`, `commands/eval.py`) read it positionally.
- The `is_available()` classmethod returns `sys.platform == "win32"` — that's the same predicate `WinMLEPMonitor.is_available()` subclasses use, but `HWMonitor` is not a subclass. If `HWMonitor` were folded into the `WinMLEPMonitor` hierarchy (it would not need `ep_name` / op-tracing hooks, just the lifecycle), the docstring carve-out ("Independent of the WinMLEPMonitor hierarchy") could go away. The current asymmetry is documented but not justified by any unique requirement.
- Two near-identical "Independent of …" sentences in module + class docstrings — one is enough.
