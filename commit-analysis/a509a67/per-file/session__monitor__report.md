# src/winml/modelkit/session/monitor/report.py

## TL;DR
New file providing the public render layer for `OpTraceResult`: a 4-column "basic" Rich table, a 10-column "detail" Rich table (with running cumulative percentage), and a JSON writer. Relocated from the deleted `optracing/report.py` with stricter column widths, defensive sort by `percent_of_total`, and a left-truncating node-path formatter.

## Diff metrics
- Lines added: 253
- Lines removed: 0
- New file (relocated content from deleted `optracing/report.py`)

## Role before vs after
- **Before:** Logically equivalent renderer lived in `src/winml/modelkit/optracing/report.py` (deleted). Pre-refactor column layout and sort assumptions were embedded in CSV/QHAS parsers respectively.
- **After:** Single canonical renderer that defensively re-sorts operators in the display layer (so upstream parsers can preserve whatever order they like) and width-locks both tables for stable 120-column console output. Pairs with `op_metrics.py` schema.

## Symbol-level changes
### Public
- **`display_op_trace_report(result, console=None, top_n=5)`** — added (new)
  - Dispatches on `result.tracing_level` to either `_display_basic_report` or `_display_detail_report`. Creates a default `rich.console.Console()` when not provided. Default `top_n=5` matches the mockup `OP_TRACING_TOP_K_DEFAULT` constant.
- **`write_op_trace_json(result, output_path)`** — added (new)
  - Coerces `output_path` to `Path`, creates parent dirs (`mkdir(parents=True, exist_ok=True)`), writes `result.to_json()` via `Path.write_text`. No explicit encoding argument (system default).

### Internal helpers
- **`_format_bytes(n)`** — added (private)
  - Human-readable byte formatter, B/KB/MB/GB/TB, integer-rendered when value is whole at the B unit (e.g. `"42 B"`). Returns `"0"` for `None` or zero (no unit).
- **`_format_number(n)`** — added (private)
  - Comma-thousands formatter. Float → `:,.1f`, int → `:,`, None → `"-"`.
- **`_truncate_node_name(name, max_width=80)`** — added (private)
  - **Left-truncates** with a leading ellipsis (`"…"`), preserving the right side because the leaf operator name (the differentiator) lives at the tail. Guards `max_width <= 0` (returns `""`) and `max_width == 1` (returns `"…"`).
- **`_display_basic_report(result, console, top_n)`** — added (private)
  - 4-column table: Node (min/max 80), Type (12), p90 (9, right-justified), % Tot (6). Header rule "Op-Tracing (basic)". Summary line shows HVX threads, accel-execute µs, sample count when present. Renders `"—"` for p90 when `samples_us` is empty. Empty-ops case prints `"[dim]No operator data available.[/dim]"`.
- **`_display_detail_report(result, console, top_n)`** — added (private)
  - 10-column table: `#`, Node, Type, Avg, Total, % Tot, Cum %, p90, DRAM(R), VTCM Hit. Header rule "Op-Tracing (detail)" with `tracing_backend` suffix when present (`" -- {backend}"`). Two-line summary: inference/execute µs/utilization%, then DRAM read/write + VTCM peak. Maintains running `cum` of `percent_of_total` across rows. Avg falls back to `duration_us` when `samples_us` is empty; Total and p90 render `"—"` in that case. VTCM-hit-ratio rendered as `* 100` percentage.

## Behavior / contract changes
- **Defensive sort:** Both render paths sort `result.operators` by `(-percent_of_total, op_path)` before slicing `[:top_n]`. This means the rendered top-K is the top-K by % of total, regardless of upstream order. Comment explicitly calls out that CSV parsers sort by cycles and QHAS preserves JSON order — the renderer normalizes both.
- **Tie-break stability:** Ascending `op_path` tie-break gives deterministic output across runs for ops with equal `percent_of_total`.
- **Cumulative %:** detail-mode `Cum %` is the running sum over the *displayed* top-K rows (not over all operators). Caller should not interpret `Cum %` of the last row as "100%" of the model.
- **Width contract:** column widths are hard-coded to fit a 120-column console (the basic-mode comment explicitly says "width-locked at 120"); detail-mode is also width-locked.
- **Empty-state contract:** prints a dim "No operator data available." line and returns without rendering the table header.
- **Output dir for JSON:** silently creates missing parent dirs.

## Cross-file impact
- **Used by which modules:** `commands/perf.py` (after benchmark, calls `display_op_trace_report` for console rendering and `write_op_trace_json` for artifact persistence); the diff message describes `commands/perf` now writing benchmark JSON only after the op-trace status check.
- **Depends on which modules:** stdlib `pathlib`, third-party `rich.console.Console` / `rich.table.Table`, sibling `.op_metrics.OpTraceResult`.

## Risks / subtleties
- `# noqa: TC001 (used at runtime)` annotation on the `OpTraceResult` import is correct — it's used in the runtime function signatures.
- `_truncate_node_name` uses Unicode ellipsis `"…"` (U+2026), single character — width is 1 column. Good for console alignment but may not survive non-UTF terminals.
- `_format_bytes` returns `"0"` (no unit) for `None`/zero — slight inconsistency with the unit-suffixed format used otherwise, but the byte columns are too narrow (8 chars) for `"0 B"` to matter.
- Default `top_n=5` is hard-coded; the docstring references `OP_TRACING_TOP_K_DEFAULT` in `docs/design/perf/console_mockup.py` as the canonical source but no actual constant import — drift risk between mockup and code.
- `write_op_trace_json` uses `Path.write_text` with no `encoding=` — falls back to platform default (Windows: cp1252 historically, now UTF-8 since Python 3.15 on opted-in PEP 686 paths). For ASCII JSON this is fine, but a non-ASCII model/path/op name could corrupt on legacy Windows.
- `_display_basic_report` shows samples count and HVX/accel from `result.summary` keys — if a future non-QNN backend doesn't populate `hvx_threads` / `accel_execute_us`, the basic header just degrades gracefully (the `parts` list filter handles `None`), but that gives a less useful summary line.
- The detail table mixes `Avg` (computed property) with `duration_us` (raw field) as fallback — readers cannot tell from the column whether they're seeing a sample mean or a single aggregated number.

## Open questions / TODOs surfaced
- `OP_TRACING_TOP_K_DEFAULT` is mentioned in docstring as living in `docs/design/perf/console_mockup.py` — should this be a proper imported constant rather than a docstring reference?
- No tests visible for the renderer in the diff for this file specifically — the new architecture-regression test mentioned in the commit message targets `qnn._internal` information hiding, not rendering output.
- `write_op_trace_json` lacks an `encoding="utf-8"` argument — minor but worth adding for cross-platform robustness.
- Should "basic" / "detail" string check (`if result.tracing_level == "detail"`) be backed by a Literal type alias paralleling `TraceStatus`?
