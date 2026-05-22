# Review: `src/winml/modelkit/session/monitor/report.py`

**Status:** new file (relocated from `optracing/report.py`)
**Lines added/removed:** 253+ / 0-
**Diff command:** `git diff 1bea4cf..HEAD -- src/winml/modelkit/session/monitor/report.py`

## 1. Purpose of this file

Provides two public functions — `display_op_trace_report` and `write_op_trace_json` — and the internal helpers that format `OpTraceResult` for human display (via Rich) and JSON serialization. This file was relocated from `optracing/report.py` as part of the refactor; its public interface is unchanged per PRD OOS-4.

## 2. Changes summary

- Relocated from `optracing/report.py` to `session/monitor/report.py`.
- No functional changes to the rendering logic.
- `OpTraceResult` import moved from `optracing.result` to `.op_metrics`.
- Added `# noqa: TC001` on the `OpTraceResult` import (used at runtime by type annotations, not TYPE_CHECKING).

## 3. Per-symbol review

### `display_op_trace_report()`

- **Role:** Render an `OpTraceResult` as a Rich console table.
- **Signature:** `def display_op_trace_report(result: OpTraceResult, console: Console | None = None, top_n: int = 5) -> None:`
- **Behavior:** Dispatches on `result.tracing_level` — `"detail"` calls `_display_detail_report`, everything else calls `_display_basic_report`. Creates a default `Console` if `None` is passed.
- **Invariants:** OOS-4 — function signature and behavior are unchanged. `top_n=5` matches `OP_TRACING_TOP_K_DEFAULT` in `docs/design/perf/console_mockup.py`.
- **Risks / concerns:** The dispatch is string-comparison (`== "detail"`). If `tracing_level` is misspelled (e.g. `"Detail"`) by a monitor, the basic report renders silently. No validation of the `tracing_level` value. Acceptable given the closed set from `QNNMonitor.__init__`.
- **Tests:** `test_report.py`, `test_report_basic.py`, `test_report_detail.py`, `test_report_top_n_default.py`.

---

### `write_op_trace_json()`

- **Role:** Write `OpTraceResult` to a JSON file.
- **Signature:** `def write_op_trace_json(result: OpTraceResult, output_path: Path | str) -> None:`
- **Behavior:** Creates parent directories if needed; writes `result.to_json()`. The encoding is whatever Python's default write mode gives (platform-dependent). On Windows this may be `cp1252` rather than UTF-8 if the operator paths contain non-ASCII characters.
- **Risks / concerns:** `output_path.write_text(result.to_json())` uses Python's default encoding (platform locale on Windows). Should use `encoding="utf-8"` explicitly for portability. Operator node names can contain Unicode characters (e.g. Chinese model layers). Not a current practical risk (QNN op paths are ASCII), but a latent portability issue.
- **Tests:** `test_report.py::test_creates_file`.

---

### `_format_bytes()`

- **Role:** Format a byte count to a human-readable string with unit suffix.
- **Behavior:** Handles `None` and `0` → `"0"`. Handles integer values below 1024 without decimal (e.g. `"42 B"`). Iterates through `("B", "KB", "MB", "GB")` dividing by 1024 each time; falls through to TB.
- **Risks / concerns:** The `if unit == "B" and value == int(value)` branch checks for integer-valued floats in the loop — this is correct but subtle. If `n` is a large negative float (which should not occur for byte counts), the loop terminates at TB with negative values. Acceptable — byte counts should always be non-negative.
- **Tests:** Not directly tested; exercised via `test_report_detail.py`.

---

### `_truncate_node_name()`

- **Role:** Left-truncate a node path with a leading `…` to preserve the leaf identifier.
- **Behavior:** Preserves right side since the leaf operator name lives at the tail. `max_width <= 0` returns `""`. `max_width == 1` returns `"…"`.
- **Tests:** `tests/unit/session/monitor/test_truncate_node_name.py`.

---

### `_display_basic_report()`

- **Role:** Render a 4-column basic op-trace table (Node, Type, p90, % Tot).
- **Behavior:** Sorts operators descending by `percent_of_total` with `op_path` tie-break before slicing to `top_n`. The comment explains the motivation: upstream parsers have varying sort order; the render layer normalizes. `p90_us` is displayed as `"—"` when `samples_us` is empty (QHAS path fallback case).
- **Invariants:** Defensive sort in the render layer means the `% Tot` column always reads naturally (largest first) and `Cum %` in detail mode is monotonically increasing.
- **Risks / concerns:** The `Node` column is `max_width=80` hardcoded. If QNN node paths exceed 80 characters (which can happen for deeply nested models), the `overflow="ellipsis"` setting in Rich truncates, but the `_truncate_node_name()` pre-truncation is also called with `max_width=80`. The two-stage truncation is redundant but harmless.
- **Tests:** `test_report_basic.py`.

---

### `_display_detail_report()`

- **Role:** Render a 10-column detail op-trace table with cumulative % column.
- **Behavior:** Same defensive sort as basic mode. `avg_str` uses `op.avg_us` when `samples_us` is populated; falls back to `op.duration_us` for QHAS path. `total_str` and `p90_str` are `"—"` for QHAS path.
- **Risks / concerns:**
  1. The `Cum %` column is computed by accumulating `percent_of_total` over the sorted `top_n` slice. If the upstream parser's `percent_of_total` values don't sum to 100 (which can happen if the CSV has rounding errors or the QHAS summary is from a different inference run than the CSV), the cumulative percentage will be misleading. No validation is performed.
  2. `backend_suffix` in the rule header is taken from `result.tracing_backend`, which is `"qnn"` in normal operation. No special formatting is applied (e.g. `"qnn"` appears in lowercase). Acceptable.
- **Tests:** `test_report_detail.py`.

## 4. Cross-cutting concerns

**Spec drift:** None. This file is OOS-4 (unchanged from optracing/report.py). The `top_n=5` default matches `OP_TRACING_TOP_K_DEFAULT` in `docs/design/perf/console_mockup.py`. The 10-column detail table matches `console_mockup.py:448-465`.

**Information-hiding contract:** Only imports `OpTraceResult` from `.op_metrics`. No imports from `qnn/_internal` or any monitor. Correct.

**Deferred work:** No TODO markers.

**EPDevice / ep_name:** Not referenced.

## 5. Confidence level

**High.** The render logic is straightforward, the defensive sort is well-motivated, and the `# noqa: TC001` comment is correct (the import IS used at runtime as a type annotation in the function signature, not just for type checking). The main latent risk is the `write_op_trace_json` encoding issue.

## 6. Verbatim risk inventory

| Severity | Location | Description |
|----------|----------|-------------|
| Low | `report.py:52` | `output_path.write_text(result.to_json())` uses platform default encoding. Should use `encoding="utf-8"` to guarantee portability across Windows locales. |
| Low | `report.py:226` | `Cum %` column accumulates `percent_of_total` values from QHAS or CSV. If upstream values don't sum to 100 (rounding, multi-sample averaging), the final `Cum %` may be misleading (e.g. `97.3%` for the full `top_n`). No validation or note is shown to the user. |
| Info | `report.py:110` | `if result.tracing_level == "detail":` is a string-equality dispatch. A misspelled `tracing_level` would silently render as basic. Not a current risk given the closed set in `QNNMonitor.__init__`. |
