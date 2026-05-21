# Review: `src/winml/modelkit/commands/_live_chart.py`

**Status:** modified
**Lines added/removed:** 2+ / 2-

## 1. Purpose

`_live_chart.py` provides the `LiveMonitorDisplay` class used by `_run_monitored_loop`
in `perf.py` to render a real-time plotext/Rich Live chart of NPU and CPU utilization
during a benchmark run. It is a display-only module with no inference logic. This PR
makes two cosmetic constant adjustments to better fit wider terminals.

## 2. Changes summary

- `_CHART_WINDOW_SECONDS` bumped from `10.0` to `15.0` — broader time window on the
  x-axis so the chart shows more history before the oldest samples scroll off.
- `chart_width` default parameter in `LiveMonitorDisplay.__init__` bumped from `80`
  to `120` — matches a standard wide terminal and the mockup-approved geometry.

## 3. Per-symbol review

### `_CHART_WINDOW_SECONDS`

- **Role:** Module constant controlling how many seconds of utilization history are
  shown on the scrolling x-axis.
- **Signature:** `_CHART_WINDOW_SECONDS = 15.0`
- **Behavior:** Consumed in `_render_chart` via `window_samples = int(_CHART_WINDOW_SECONDS / self._poll_interval_s)`. At `poll_interval_ms=100` (default) the window holds 150 samples. Larger window means the x-axis tick at the far left of the chart is 15 s before the most recent sample.
- **Invariants:** Must be > 0. Should be greater than a single benchmark run's total elapsed time to keep all samples visible for short runs.
- **Risks / concerns:** None. The test in `test_live_chart_constants.py` pins this to `15.0`, enforcing the mockup spec. No coupling to inference logic.
- **Tests:** `tests/unit/commands/test_live_chart_constants.py::test_chart_window_seconds_is_fifteen`

### `LiveMonitorDisplay.__init__` — `chart_width` default

- **Role:** Default terminal width for the plotext canvas.
- **Signature:** `def __init__(self, ..., chart_width: int = 120, ...)`
- **Behavior:** Passed to `plt.plotsize(self._chart_width, self._chart_height)` in `_render_chart`. Increasing from 80 to 120 avoids spurious wrapping on 120-column terminals and matches the mockup spec.
- **Invariants:** Overridable by caller; callers in `perf.py` use the default.
- **Risks / concerns:** None. On narrow terminals (< 120 columns) Rich wraps gracefully; no crash path.
- **Tests:** `tests/unit/commands/test_live_chart_constants.py::test_default_chart_width_is_one_hundred_twenty`

## 4. Cross-cutting concerns

- No audit gap: changes are purely cosmetic constants, no behavior path affected.
- No legacy `device=` callers in this file.
- CLI option changes: none. Constants are internal to the display layer.

## 5. Confidence level

**High.** Two constant changes, both guarded by pinning tests. No logic altered.

## 6. Verbatim risk inventory

| Severity | Location | Description |
|----------|----------|-------------|
| Info | `_live_chart.py:20` | `_CHART_WINDOW_SECONDS` is a bare module-level constant; if a future caller caches the old value at import time the pin test still protects the class default. |
