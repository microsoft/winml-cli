# src/winml/modelkit/commands/_live_chart.py

## TL;DR
Two-line tweak that widens the HW utilization moving-window chart from a 10 s /
80-column view to a 15 s / 120-column view to fit the new pre-bench layout and
to keep the chart legible while the wider perf console is in play. Pure display
tuning — no behavioral or contract change.

## Diff metrics
- Lines added: 2
- Lines removed: 2
- Net: 0
- New / modified: modified (existing file)

## Role before vs after
Before: private helper rendering a live NPU/CPU plot via `plotext` inside a
Rich `Live` panel during `winml perf`. X-axis window = 10 s; default chart
width = 80 columns.

After: same private helper, same responsibilities; only the visual dimensions
changed. X-axis window = 15 s; default chart width = 120 columns. This aligns
with the commit-body note "HW chart widened to 15s/120c" and complements the
new wider Perf console (4-col basic / 10-col detail tables + pre-bench identity
block).

## Symbol-level changes
- `_CHART_WINDOW_SECONDS` module constant: `10.0` → `15.0`. Drives both the
  sliding-window slicing inside `_render_chart` (samples kept =
  `_CHART_WINDOW_SECONDS / poll_interval_s`) and the displayed `x_min/x_max`
  bounds.
- `LiveMonitorDisplay.__init__(..., chart_width: int = 80, ...)` →
  `chart_width: int = 120`. Default panel width grew by 50 %; callers that
  pass `chart_width` explicitly are unaffected.

No other functions, methods, attributes, or imports changed.

## Behavior / contract changes
- Wider default `plt.plotsize(120, 15)` means the rendered ANSI block produced
  by `plt.build()` now emits ~120-column lines instead of ~80. Terminals
  narrower than 120 cols may wrap; the surrounding Rich `Panel` does not
  truncate.
- 15 s window keeps 1.5× more samples on screen at any given poll cadence
  (e.g., at the default 100 ms poll, that's 150 samples vs 100 previously).
  The x-axis lower bound calculation `max(0.0, elapsed - 15.0)` extends the
  visible history correspondingly.
- The inline comment on line 138 ("e.g., at 15s elapsed with 10s window:
  x-axis shows 5.0 -> 15.0") was not refreshed and now mismatches the new
  constant — minor doc drift, not a bug.

## Cross-file impact
None at the symbol level. `LiveMonitorDisplay` is consumed by `commands/perf.py`
(see commit body — `commands/perf.py` migrated to ep_device); any caller that
relied on the 80-column default now gets 120 implicitly. Callers passing
`chart_width` explicitly are untouched. No public-API signatures changed.

## Risks / subtleties
- Hard-coded width=120 may overflow on narrow terminals (Windows Terminal
  default is 120, but ssh/pty contexts can be 80). No dynamic sizing.
- The "10s window" example comment is now stale (says 10 s, code says 15 s) —
  cosmetic only.
- Behavior under `plotext` ImportError fallback path is unchanged (still
  emits a 50-char ASCII bar).
- No test changes accompany this — there is no pytest coverage for live
  rendering, so this is verified visually only (consistent with the file
  being a private CLI helper).

## Open questions / TODOs surfaced
- Should `chart_width` be derived from `Console.size.width` instead of a fixed
  constant, so narrow terminals do not wrap?
- The stale comment on line 138 should be updated to "at 15 s window".
- No automated regression test for the live chart; only manual verification
  via the 6-command CLI matrix in the commit body.
