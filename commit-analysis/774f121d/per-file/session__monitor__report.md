# src/winml/modelkit/session/monitor/report.py

## TL;DR
Relocated renderer for `OpTraceResult`: `display_op_trace_report` dispatches on `tracing_level` to a 4-column basic table or a 10-column detail table; `write_op_trace_json` persists results. Move from `optracing/report.py` rewrote both tables to width-locked columns matching the v2.9 console mockup, lowered the default `top_n` from 15 to 5, switched to left-truncating node paths (new `_truncate_node_name`), and added a defensive sort by `percent_of_total` plus a running `Cum %` column in detail mode.

## Diff metrics
- File status: renamed `optracing/report.py` → `session/monitor/report.py` (recorded as delete + add).
- Old: 206 lines deleted. New: 253 lines added. Net +47.
- Commit's stat collapses the pair to "145 lines changed" because the renderer bodies were largely rewritten.

## Role before vs after
- **Before (`optracing/report.py`):** Renderer for the standalone `optracing` package. Imported `OpTraceResult` from sibling `.result`. Tables sized with `min_width` only (no upper bound), titled `"Top Operators by Duration"`, used raw `op.duration_us` as the primary timing column, displayed `op_path` as-is (no truncation), and showed 15 rows by default.
- **After (`session/monitor/report.py`):** Same public API, but anchored to the consolidated `session.monitor` namespace. Imports `OpTraceResult` from sibling `.op_metrics` (which now carries `samples_us` + derived `avg_us`/`total_us`/`p90_us` properties and `TraceStatus`). Render layer normalizes upstream order, width-locks every column to fit a 120-column console, and aligns column inventory with `docs/design/perf/console_mockup.py:448-465`.

## Symbol-level changes
### Public
- **`display_op_trace_report(result, console=None, top_n=5)`**
  - Default `top_n` changed `15 → 5` to match `OP_TRACING_TOP_K_DEFAULT` in the mockup (pinned by `tests/unit/session/monitor/test_report_top_n_default.py`).
  - Docstring now cites the mockup as the canonical source of the default.
  - Dispatch logic (`if tracing_level == "detail"`) unchanged.
- **`write_op_trace_json(result, output_path)`** — byte-for-byte identical (sole behavior: `mkdir(parents=True, exist_ok=True)` + `Path.write_text(result.to_json())`; no `encoding=` kwarg).

### Internal helpers
- **`_format_bytes`**, **`_format_number`** — unchanged from `optracing/report.py`.
- **`_truncate_node_name(name, max_width=80)`** — **new.** Left-truncates with leading `"…"` (U+2026); guards `max_width <= 0` (`""`) and `max_width == 1` (`"…"`). Preserves the right side because the differentiating leaf op name lives at the tail of the path.
- **`_display_basic_report`** — completely re-skinned:
  - Header rule renamed `"Op-Level Profiling (basic)"` → `"Op-Tracing (basic)"`.
  - Table now 4 cols (was `# / Operator / Avg Cyc / % Tot`, 4 cols): `Node` (min/max 80, ellipsis), `Type` (width 12), `p90` (width 9, right), `% Tot` (width 6, right). The `#` index column was dropped and `Type` was added.
  - Primary timing column switched from `op.duration_us` to `op.p90_us`, with `"—"` fallback when `samples_us` is empty.
  - Node path now passes through `_truncate_node_name` before render.
  - Defensive sort on `(-percent_of_total, op_path)` before slicing top-N (upstream parsers may sort by cycles or preserve JSON order — comment in source spells this out).
- **`_display_detail_report`** — re-skinned:
  - Header rule renamed `"Op-Level Profiling (detail)"` → `"Op-Tracing (detail)"`; backend suffix appended unchanged.
  - Table expanded `7 → 10` columns: adds `Avg`, `Total`, `Cum %`, `p90` (previously only `Dur(us)` + memory cols).
  - Primary timing splits: `Avg` uses `op.avg_us` when `samples_us` populated else falls back to `op.duration_us`; `Total` and `p90` render `"—"` when `samples_us` is empty.
  - Running `cum` accumulator added — `Cum %` is the partial sum of `percent_of_total` over the *displayed* rows, not over all operators.
  - Same defensive sort by `(-percent_of_total, op_path)` as basic.
  - Node column gains `max_width=80` + ellipsis overflow (previously `min_width=25` only).
  - VTCM "no data" sentinel changed `"-"` → `"—"` (em-dash, matches the new `"—"` placeholders).

## Behavior / contract changes
- **Defaults:** `top_n` default lowered from 15 to 5 (visible to every `wmk perf --op-tracing` caller that omits `--top-k`).
- **Display sort:** Render layer now *guarantees* descending `percent_of_total` order; upstream parsers no longer need to pre-sort and `Cum %` is monotonic by construction. Tie-break on `op_path` makes output deterministic.
- **Column inventory drift:** Anyone scraping the old basic table for an `Avg Cyc` column will break — that column is gone. Anyone scraping detail for `Dur(us)` will break — replaced by `Avg`/`Total`/`p90`.
- **Width contract:** All columns now have hard `width=` or `max_width=` caps so the table fits 120 cols on Windows terminals. Long node paths get left-ellipsised, not wrapped.
- **Status branches NOT handled here:** The renderer never inspects `result.status`. The full status fan-out (`no_data`, `parse_failed`, `basic_fallback`, `ok`) is enforced in `commands/perf.py` *before* it calls `display_op_trace_report`. `basic_fallback`'s yellow notice is printed by the perf command; the renderer just sees a `tracing_level == "basic"` result and proceeds. Only `not_run` is not explicitly named anywhere on the render path.
- **Empty-operators contract preserved:** Both modes still print `"[dim]No operator data available.[/dim]"` and return without drawing a table header.
- **JSON output:** Unchanged byte-for-byte.

## Cross-file impact
- **Used by:** `src/winml/modelkit/commands/perf.py:1612` (lazy import inside the `op_tracing` branch). Two call shapes: `display_op_trace_report(trace_result, console, top_n=top_k)` when `--top-k` was passed and `display_op_trace_report(trace_result, console)` otherwise (relying on the new `top_n=5` default). `write_op_trace_json(trace_result, trace_output)` follows immediately, after the status guards have already filtered out failure states.
- **Depends on:** `pathlib.Path`, `rich.console.Console`, `rich.table.Table`, sibling `.op_metrics.OpTraceResult` (and transitively its `OperatorMetrics` properties `avg_us`, `total_us`, `p90_us`, `samples_us`).
- **Tests:** `tests/unit/session/monitor/test_report.py` (general), `test_report_basic.py`, `test_report_detail.py`, `test_report_top_n_default.py` (pins default to 5), `test_truncate_node_name.py` (left-ellipsis contract).
- **Old `optracing/report.py` deleted** along with the rest of the standalone `optracing/` package; no shim left behind (matches the user's "no back-compat" preference).

## Risks / subtleties
- **`# noqa: TC001` retained** for the `OpTraceResult` import — correct, the symbol is referenced at runtime in the signatures.
- **Hard-coded `top_n=5` literal** in the signature with the only "source of truth" reference being a docstring pointer to `docs/design/perf/console_mockup.py`. Drift risk: the mockup constant is *not* imported. A future change to the mockup will not propagate.
- **`samples_us`-empty branch** silently changes the meaning of the `Avg` column (`op.avg_us` returns 0.0 from a property, so the fallback to `op.duration_us` is necessary — but the reader sees no indicator that this is a single aggregated value rather than a sample mean).
- **`Cum %` semantics:** the displayed running total runs only over the rendered top-K rows. The last row's `Cum %` is therefore not "100% of the model" — a user comparing two runs with different `--top-k` will see different terminal values for the same row. No on-screen disclaimer.
- **Unicode em-dash and ellipsis (`"—"`, `"…"`):** rely on UTF-8 terminal. Legacy Windows code pages will mojibake. Tests assert on the literal characters.
- **`write_op_trace_json` still lacks `encoding="utf-8"`** — inherited from the old file; non-ASCII model names / op paths could corrupt on legacy Windows defaults.
- **Defensive sort runs twice** (once per render path) on the same `result.operators` list. Cheap, but it is the same lambda + slice copied verbatim.
- **No backend-tag branching:** `result.tracing_backend` is rendered only as a string suffix on the detail header; the renderer does not adjust columns or fall back when summary keys (e.g. `hvx_threads`, `dram_read_bytes`) are absent from a non-QNN backend — it just filters them out of the summary line.

## Open questions / TODOs surfaced
- Should `OP_TRACING_TOP_K_DEFAULT` be imported as a module-level constant rather than referenced only by docstring?
- Should `tracing_level` be a `Literal["basic", "detail"]` paired with `TraceStatus` so the renderer's `if result.tracing_level == "detail"` branch is type-checked? Today a typo (`"details"`) silently falls into the basic branch.
- Is the renderer the right place to enforce display sort, or should the parsers normalize? Today it is duplicated across both render paths.
- Should the JSON writer accept a `console`/logger so the success path can announce its own path, instead of `perf.py` doing the printing?
- No status-aware rendering: should the renderer at least refuse to draw on `status == "not_run"` rather than relying on the caller to check?

## Simplification opportunities
- **`_display_basic_report` and `_display_detail_report` share three identical blocks** (head/blank lines, the 12-line sort+slice+empty-guard, the trailing `console.print(table)`). The sort+empty-guard is byte-for-byte identical and could collapse to a private `_topk(result, top_n) -> list[OperatorMetrics] | None` that returns `None` (caller prints the dim message + returns) or a list. That removes ~14 duplicated lines and centralizes the "comment block about CSV vs QHAS ordering" that currently appears twice.
- **Summary-line construction** in `_display_basic_report` (`hvx_threads`, `accel_execute_us`, samples) and `_display_detail_report` (`inference_us`/`execute_us`/`utilization_pct`, then DRAM/VTCM) follow the same `parts: list[str]` + `" | ".join(parts)` pattern. A `_format_summary_line(summary, keys: tuple[tuple[str, str, Callable], ...])` helper, or a tiny declarative `[(label, key, formatter), ...]` table per mode, would dedupe the boilerplate.
- **Per-op cell construction repeats `if op.samples_us else "—"` three times** in detail mode (Avg, Total, p90). A helper like `_us_or_dash(value: float, samples: list[float]) -> str` would dedupe.
- **Two callers, single dispatcher:** `display_op_trace_report` is just a 4-line `if/else`. Given the call sites in `commands/perf.py`, the dispatcher itself is fine — but the two `_display_*` functions could plausibly be one parametrized renderer that takes a `(rule_text, summary_lines, columns, row_builder)` tuple. Higher risk of obscuring intent; only worth it if a third mode is added.
- **`_truncate_node_name(max_width=80)` is called with the same literal `80` at every site** — making `_MAX_NODE_WIDTH = 80` a module-level constant would tie the helper default, the basic column `min_width/max_width`, and the detail column `max_width` together so they cannot drift.
- **`# noqa: TC001` comment** could be dropped by simply moving the import out of any `TYPE_CHECKING` guard convention (it already is) — the noqa is defensive but no longer load-bearing now that the import is unconditional. Minor.
- **Defensive sort comment is duplicated verbatim** (8 lines, identical in both functions). Folding it into the proposed `_topk` helper eliminates the duplication.
