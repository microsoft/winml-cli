# src/winml/modelkit/commands/_live_chart.py

## TL;DR
Trivial: two constants tuned. `_CHART_WINDOW_SECONDS` from 10.0 → 15.0,
and `LiveMonitorDisplay.__init__` default `chart_width` from 80 → 120.
Pure UX tweak — wider charts, longer history window. Nothing else
touches the file.

## Diff metrics
- 4 lines changed (2 insertions / 2 deletions per `--stat`).
- One hunk near the top of the file.

## Role before vs after
Role unchanged: renders the live NPU/CPU utilization chart during
`winml perf --monitor`. The chart is fed by `HWMonitor.utilization_samples`
and re-rendered via Rich Live + plotext.

## Symbol-level changes

### Constants
- `_CHART_WINDOW_SECONDS = 10.0` → `15.0` (moving window for the x-axis).
- `LiveMonitorDisplay.__init__(chart_width=80)` → `chart_width=120`.

### Behavioral consequences
- 15 s window means the in-chart history covers 50% more samples at
  the default `poll_interval_ms=100`. At `_HW_POLL_INTERVAL_MS=200`
  (perf.py's default), that's 75 samples instead of 50.
- 120-wide chart fits wider terminals comfortably (the 80-wide default
  was clamped for narrow terminals; the new 120 still degrades
  gracefully if the terminal is narrower — plotext handles it).

## Behavior / contract changes
- The persisted final-frame display (`transient=False`) now shows a
  wider, longer-history snapshot. Acceptable for the modern wide-terminal
  baseline.

## Cross-file impact
- None. `LiveMonitorDisplay` constructor is called from
  `_run_monitored_loop` in `perf.py`; the call doesn't pass
  `chart_width`, so it picks up the new default.

## Risks / subtleties
- Narrow terminals (< 120 cols) will see the chart get clipped or
  word-wrapped by Rich. plotext typically degrades gracefully but
  the UX might look worse on small terminals. Worth a manual check
  at 80 cols.
- The `_poll_interval_s` math (`int(_CHART_WINDOW_SECONDS /
  self._poll_interval_s)`) yields more samples in the sliding
  window — both the NPU and CPU traces will have 50% more data
  points, which makes the chart busier. Visually it's still readable.

## Simplification opportunities
- **`chart_width=120` as a constructor default** is a magic number. A
  module-level `_DEFAULT_CHART_WIDTH = 120` would surface the choice
  for future tuning. Minor.
- The `chart_width` kwarg is never explicitly passed by the caller
  (`perf.py` builds `LiveMonitorDisplay(total_iterations=..., warmup=...,
  model_id=..., device=...)`); removing the kwarg and inlining the
  constant would simplify the API. Worth keeping for hypothetical
  external callers that want to override.

## Open questions / TODOs surfaced
- Should `chart_width` be derived from `Console.size.width`? That would
  auto-fit to terminal width. Trivial follow-up.
- Are there any tests that mock `LiveMonitorDisplay` constructor args?
  None visible in the diff. Likely none — the live display is hard to
  unit-test.
