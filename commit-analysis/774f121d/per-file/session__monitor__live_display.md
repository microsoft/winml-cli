# src/winml/modelkit/session/monitor/live_display.py

## TL;DR
File **deleted** in this commit (-207 lines, no replacement at this path). `HWLiveDisplay` was a self-contained `HWMonitor` + Rich `Live` + plotext widget with **zero callers** in the pre-state tree — its functionality was already duplicated (and superseded) by `commands/_live_chart.py::LiveMonitorDisplay`, which is the one `commands/perf.py` actually uses. No behavior was lost; this is dead-code removal.

## Diff metrics
- Lines added / removed: **0 / -207**
- New / modified: **deleted** (mode 100644 → /dev/null)

## Role before vs after
- **Before:** Self-contained live HW utilization chart context manager. Wrapped `HWMonitor` + plotext + Rich `Live` in `__enter__`/`__exit__`, drove an off-thread `_update_loop` at 5 FPS, rendered NPU (green) + CPU (cyan) traces with a hardcoded 10-second window and 72-col / 12-row chart. Module constants: `_CHART_WINDOW_SECONDS = 10.0`, `_REFRESH_FPS = 5`, `_DEFAULT_CHART_WIDTH = 72`, `_DEFAULT_CHART_HEIGHT = 12`.
- **After:** Does not exist. The live-chart capability is centralized in `src/winml/modelkit/commands/_live_chart.py::LiveMonitorDisplay` (which `commands/perf.py` imports). The deleted file's identical responsibility shows it was already a stale parallel implementation when the monitor package was conceived.

## Symbol-level changes (all removed)
- Class `HWLiveDisplay(title, poll_interval_ms=200, chart_width=72, chart_height=12)` — context manager
- Methods: `__enter__`, `__exit__`, `_update_loop`, `_render_once`, `_render_chart`, `_render_status`
- Property `hw` (exposed wrapped `HWMonitor`)
- Module constants `_CHART_WINDOW_SECONDS`, `_REFRESH_FPS`, `_DEFAULT_CHART_WIDTH`, `_DEFAULT_CHART_HEIGHT`

## Behavior / contract changes
- `from winml.modelkit.session.monitor.live_display import HWLiveDisplay` now raises `ModuleNotFoundError`. Codebase grep at HEAD confirms **no production or test call sites** existed — `HWLiveDisplay` was never wired into any command. Three design docs (`docs/design/perf/2026-04-28-console-mockup-design.md`, `2026-04-29-session-handoff.md`, `2026-05-01-op-tracing-production-lift-summary.md`) still mention it but that's pure prose. This is a no-op deletion in terms of runtime behavior.
- The chart constants this module owned (10 s / 72×12) no longer exist anywhere; the surviving `LiveMonitorDisplay` uses different geometry (15 s / 120×15 per prior commit-analysis notes).

## Cross-file impact
- **Used by which modules at HEAD:** none. The codebase grep returns only this `commit-analysis` directory and three design docs.
- **Depends on which modules (pre-deletion):** `.hw_monitor.HWMonitor`, `rich.console`, `rich.live`, `rich.panel`, `rich.text`, `rich.console.Group`, `plotext` (lazy/optional), stdlib `threading`.

## Risks / subtleties
- The deletion crosses a package boundary in spirit: the surviving widget lives in `commands/_live_chart.py` (private module by underscore convention), so any future caller outside `commands/` would need to either import from a private module or promote `LiveMonitorDisplay`. This is an architectural smell but not a regression from the deleted state.
- The deleted class had a subtle threading invariant — its `_update_loop` swallowed all exceptions (`except Exception: pass`) to "not let display errors kill the thread". The surviving `LiveMonitorDisplay` should be audited for the same defensive pattern; if it doesn't have one, a single bad chart render could nuke the worker thread. (Out of scope here, but worth noting.)

## Open questions / TODOs surfaced
- Should `LiveMonitorDisplay` be promoted out of the `_`-prefixed `commands/_live_chart.py` to make it the supported public live-chart entry point? Currently no module exports it; only `commands/perf.py` consumes it. If the answer is "yes," the new home should arguably be `session/monitor/live_display.py` — i.e., putting back the file just deleted, but with the live consumer.
- Doc cleanup: three design docs still mention `HWLiveDisplay` and the deleted path.

## Simplification opportunities
- The deletion itself is the simplification — this file was an orphaned duplicate. The fact that it survived as long as it did suggests "I'll wire it up later" intent that never materialized.
- Lookup the fate of plotext: deleting this was the last in-tree consumer if `commands/_live_chart.py` doesn't also use plotext. (Verifying that here is out of scope, but if plotext is only referenced from the surviving file, both can be checked together for whether the dependency is still pulling its weight.)
- The `commit-analysis/a509a67/per-file/session__monitor__live_display.md` doc records the same deletion at the earlier commit — confirming this file was already dead in the predecessor commit chain. The "delete twice" pattern (this commit re-deletes the same file) is a side-effect of the squash; nothing actually lived in this path between a509a67 and 774f121d.
