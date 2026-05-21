# src/winml/modelkit/session/monitor/live_display.py

## TL;DR
File was **deleted** in a509a67 (-207 lines, no replacement at this path). `HWLiveDisplay` was an orphaned duplicate of `commands/_live_chart.py::LiveMonitorDisplay` with zero call-sites. The commit body's "HW chart widened to 15s/120c" actually lives in the **surviving** `LiveMonitorDisplay` (in `src/winml/modelkit/commands/_live_chart.py`), not here.

## Diff metrics
- Lines added / removed: **0 / -207**
- New / modified: **deleted** (status `D`, mode 100644 → /dev/null)

## Role before vs after
- **Before**: Self-contained live HW utilization chart context manager — wrapped `HWMonitor` + plotext + Rich `Live` in `__enter__`/`__exit__`, drove an off-thread `_update_loop` at 5 FPS, rendered NPU (green) + CPU (cyan) traces with a hardcoded 10s window and 72-col chart.
- **After**: Does not exist. `session/monitor/` no longer exposes a live-display widget; live charting is centralized in `commands/_live_chart.py::LiveMonitorDisplay` (15.0s window, 120c width, 15-row height — matches the commit-body claim).

## Symbol-level changes
All removed:
- Class `HWLiveDisplay(title, poll_interval_ms=200, chart_width=72, chart_height=12)` — context-manager API
- Methods: `__enter__`, `__exit__`, `_update_loop`, `_render_once`, `_render_chart`, `_render_status`
- Property `hw` (exposed wrapped `HWMonitor`)
- Module constants `_CHART_WINDOW_SECONDS = 10.0`, `_REFRESH_FPS = 5`, `_DEFAULT_CHART_WIDTH = 72`, `_DEFAULT_CHART_HEIGHT = 12`

## Behavior / contract changes
- Anything importing `from winml.modelkit.session.monitor.live_display import HWLiveDisplay` (or any other symbol) will now fail with `ModuleNotFoundError`. Per `docs/design/perf/2026-05-01-op-tracing-production-lift-summary.md` there were no production or test call sites at delete-time — this is a clean removal, not a breaking change in practice.
- The chart-window/refresh constants this module owned (10s, 72×12 plot) no longer exist; the surviving `LiveMonitorDisplay` uses 15s / 120×15 (commit body: "HW chart widened to 15s/120c").

## Cross-file impact
- None expected within the repo at HEAD; grep confirms only doc references remain (`docs/design/perf/2026-04-28-console-mockup-design.md`, `2026-04-29-session-handoff.md`, `2026-05-01-op-tracing-production-lift-summary.md`).
- External consumers (if any) that referenced the symbol must migrate to `winml.modelkit.commands._live_chart.LiveMonitorDisplay` (note the `_`-prefixed private module — there is no public live-chart API exported from `session/monitor/__init__.py`).

## Risks / subtleties
- The deletion notes file lives entirely under `session/monitor/` but the survivor lives under `commands/` — caller migration crosses a package boundary.
- `LiveMonitorDisplay` is a *private* (`_live_chart`) module by naming convention; if anyone needs the live-chart capability from outside `commands/`, that surface needs to be promoted (or duplicated, which is what this deleted file used to do).

## Open questions / TODOs surfaced
- Should `LiveMonitorDisplay` be promoted out of the `_`-prefixed `commands/_live_chart.py` to make it the supported public live-chart entry point? Currently no module exports it; only `commands/perf.py` consumes it.
- Doc cleanup: three design docs still reference `HWLiveDisplay` and the deleted path; harmless but could be pruned.
