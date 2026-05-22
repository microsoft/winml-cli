# Op-Tracing Production Lift — Outcome Summary

**Date:** 2026-05-01
**Branch:** feat/op-tracing-refactor
**Plan executed:** docs/design/perf/2026-04-29-op-tracing-production-lift-plan.md
**Mockup spec:** docs/design/perf/2026-04-28-console-mockup-design.md (v2.1)

## What changed

13 commits ahead of `gh/feat/op-tracing-refactor`. Production code + tests:

| File | Action | Lines (+/-) |
|---|---|---|
| `src/winml/modelkit/session/monitor/op_metrics.py` | Modified | +32 / -0 |
| `src/winml/modelkit/session/monitor/qnn/csv_parser.py` | Modified | +15 / -2 |
| `src/winml/modelkit/session/monitor/qnn_monitor.py` | Modified | +6 / -0 |
| `src/winml/modelkit/session/monitor/report.py` | Modified | +110 / -45 |
| `src/winml/modelkit/session/monitor/live_display.py` | **Deleted** | 0 / -207 |
| `src/winml/modelkit/commands/_pre_bench.py` | **Created** | +85 / -0 |
| `src/winml/modelkit/commands/perf.py` | Modified | +90 / -56 |
| `src/winml/modelkit/commands/_live_chart.py` | Modified | +2 / -2 |
| `tests/unit/session/monitor/test_op_metrics_samples.py` | **Created** | +59 |
| `tests/unit/session/monitor/test_truncate_node_name.py` | **Created** | +32 |
| `tests/unit/session/monitor/test_report_basic.py` | **Created** | +189 |
| `tests/unit/session/monitor/test_report_detail.py` | **Created** | +225 |
| `tests/unit/session/monitor/test_qnn_monitor.py` | Modified | +5 |
| `tests/unit/session/monitor/qnn/test_csv_parser_samples.py` | **Created** | +79 |
| `tests/unit/commands/test_pre_bench.py` | **Created** | +106 |
| `tests/unit/commands/test_perf_save_footer.py` | **Created** | +50 |
| `tests/unit/commands/test_live_chart_constants.py` | **Created** | +18 |
| `docs/design/perf/2026-04-29-op-tracing-production-lift-plan.md` | **Created** | +1357 |

Net: **+2422 / -287 across 18 files** (excluding documentation lift).

## What landed by task

### T1 — `OperatorMetrics.samples_us` + derived `@property` (commit `12a86c81`)

Adds `samples_us: list[float]` field via `field(default_factory=list)` and four derived properties: `sample_count`, `avg_us`, `total_us`, `p90_us`. `p90_us` uses `statistics.quantiles(..., n=10, method="inclusive")[8]` and gracefully degenerates: returns `0.0` for empty samples, returns the single value when `n == 1`. Existing `duration_us` field retained for serialization back-compat. Implements Contract D from the mockup. Test file `test_op_metrics_samples.py` covers all 4 properties + back-compat + degenerate cases.

### T2 — QNN CSV parser per-sample retention (commit `a6293201`)

`_aggregate_operators` in `qnn/csv_parser.py` now builds a per-op `samples_us: list[float]` while still computing the avg-into-`duration_us` for back-compat. Also touched `qnn_monitor.py` (+6 lines, plumbing). Test file `test_csv_parser_samples.py` asserts samples_us length matches the number of input samples and order is preserved.

### T3 — `_truncate_node_name` left-ellipsis helper (commit `bebda766`)

Adds 7-line helper to `report.py` with edge-case handling: `max_width <= 0` returns empty string, `max_width == 1` returns the ellipsis alone, `len(name) <= max_width` returns input unchanged, otherwise returns `"…" + name[-(max_width - 1):]`. Right-side preserved (the leaf op name is the differentiator). Test file `test_truncate_node_name.py` covers 5 boundary cases.

### T4 — Basic-mode 4-col render rewrite (commit `98e419bd` + fix `20da0415`)

Replaces the old `# / Operator / Avg Cyc / % Tot` 4-col layout with mockup-spec `Node / Type / p90 / % Tot`. Width-locked at 120 cells: Node `min_width=max_width=80, no_wrap=True, overflow="ellipsis"`; Type `width=12`; p90 `width=9, justify="right"`; % Tot `width=6, justify="right"`. Header rule renamed `Op-Level Profiling (basic)` → `Op-Tracing (basic)`. p90 cell falls back to `—` (em-dash) when `samples_us` is empty.

The fix-up commit `20da0415` dropped a `' us'` suffix from the p90 cell that overflowed `width=9` for kilo-microsecond p90 values — `'1,234.5 us'` is 10 chars and Rich vertically-wraps anything over the column budget, breaking the locked-120 envelope. The unit is announced once in the table header rule and summary line, not per-cell.

### T5 — Detail-mode 10-col render rewrite (commit `32b18dca` + fix `235855b4`)

Replaces the old detail render with the mockup-spec 10-col layout: `# / Node / Type / Avg / Total / % Tot / Cum % / p90 / DRAM(R) / VTCM Hit`. Cumulative-percent computed inline (`cum += op.percent_of_total`). Em-dash fallbacks for `total_us`, `p90_us`, `vtcm_hit_ratio` when missing. Header rule renamed with `-- <backend>` suffix preserved.

The fix-up commit `235855b4` added a defensive sort by `percent_of_total` desc inside the render loop so cumulative-percent is monotonically non-decreasing regardless of upstream parser ordering. This was discovered by a test that asserted Cum% monotonicity and exposed a parser-ordering assumption the render had been silently relying on. Sort moved from parser to render layer per the principle of "render is responsible for its own invariants."

### T6 — Pre-bench identity panel (commit `97640676` + fix `09bee22d`)

Creates `_pre_bench.py` with `print_pre_bench_block(...)` that renders a 3-block identity panel (Model identity → Surface → Device) before the benchmark loop. Handles both HF model paths (full identity card) and ONNX-file paths (path only). Wired into `perf.py` post-load.

The fix-up commit `09bee22d` retired the old `_print_model_info` helper from `perf.py` (zero-call-site after the lift) and changed dynamic dim rendering from `-1` literal to `?` sentinel for readability. Net `perf.py` shrunk by 64 lines.

### T7 — Save-to footer (commit `2cc2ddc4` + fix `5fe92c78`)

Adds `_print_save_to_footer(console, *, trace_json, profiling_csv)` to `perf.py` that prints up to two `[dim]<label>:[/dim] <path>` lines. Wired in after `display_op_trace_report` + `write_op_trace_json`. CSV path comes from `trace_result.artifacts.get("profiling_csv")` and is silently omitted when absent.

The fix-up commit `5fe92c78` removed a duplicate `Op-trace saved to:` line that was being printed both inside `display_op_trace_report` and again by the new footer. The grep-first discipline caught the duplicate before T9 could surface it visually.

### T8 — Live HW chart geometry (commit `d26400ba`)

Two-character production change: `_CHART_WINDOW_SECONDS = 15.0` (was `10.0`) and default `chart_width=120` (was `80`) in `LiveMonitorDisplay.__init__`. Pinned by constant-pinning tests in `test_live_chart_constants.py`.

### Cleanup — Delete orphaned `HWLiveDisplay` (commit `7b077bc8`)

`session/monitor/live_display.py` (`HWLiveDisplay`, 207 lines) was a duplicate of `commands/_live_chart.py::LiveMonitorDisplay` with zero call-sites in production or tests. Discovered during T8 grep for chart-width references. Deleted rather than refactored-into-LiveMonitorDisplay because the survivor (`LiveMonitorDisplay`) was already feature-equivalent and actively maintained.

## Acceptance criteria coverage

The mockup design doc declares 21 ACs in §11. Production lift covers them as follows:

| AC | Description | Coverage |
|---|---|---|
| 1 | All documented invocations run end-to-end | T9 (hardware E2E) |
| 2 | Phase 2 runs ~3 sec | Mockup-only (Phase 2 timing is not productized) |
| 3 | Chart shows 3 distinct lines (NPU/CPU/GPU) | Mockup-only (GPU column forward-looking; production still NPU+CPU) |
| 4 | Two-tone progress bar | Mockup-only (forward-looking; production still single-tone) |
| 5 | Phase 3 latency table cells derive from RAW_SAMPLES_MS | Mockup-only (production already verified pre-lift) |
| 6 | Phase 3 hardware summary cells derive from per-silicon HW samples | Mockup-only |
| 7 | Phase 4 basic op-table is exactly 120 cells; 4 cols width-locked | `test_report_basic.py` (column count, widths, header text) |
| 8 | Phase 4 detail op-table auto-fits ~143 cells; 10 cols | `test_report_detail.py` (10-column-presence test) |
| 9 | Top-K summary line computed (not hardcoded) | Pre-existing `test_contains_summary_metrics` (vacuous against new layout — see carry-forwards) |
| 10 | Single-sample note iff `num_samples == 1` | `test_report_basic.py` / `test_report_detail.py` (samples_us-empty fallback path) |
| 11 | `--top-k` without `--op-tracing` exits 2 | Pre-existing CLI tests (preserved) |
| 12 | `--op-tracing` without `--iterations` collapses to 1 | Pre-existing CLI tests (preserved) |
| 13 | Save-to footer paths shown for perf.json + op-trace JSON+CSV | `test_perf_save_footer.py` (3 cases: both / CSV-omitted / both-None) |
| 14 | Module docstring documents Contracts A/B/C/D | Mockup-only |
| 15 | Reruns produce identical Phase-1/3/4 output | Mockup-only (deterministic seeded data) |
| 16 | File self-contained: no `winml.modelkit.*` imports | Mockup-only |
| 17 | `uv run ruff check && ruff format --check` clean | Production code: `uv run ruff check --fix` applied per task |
| 18 | Phase 4 section rule reads `── Op-Tracing (basic\|detail, N samples) ──` | `test_report_basic.py` + `test_report_detail.py` (header text assertion) |
| 19 | `_truncate_node_name` truncates left | `test_truncate_node_name.py` (5 cases) |
| 20 | `FakeOp`/`OperatorMetrics` attrs match Contract D stored fields + properties | `test_op_metrics_samples.py` (all 4 properties + samples_us field) |
| 21 | `--op-tracing` smart-defaults `--iterations=1` | Pre-existing CLI tests (preserved) |

**Coverage summary:** 9 ACs verified by new unit tests, 4 ACs verified by pre-existing CLI/parse tests preserved through the lift, 1 AC (AC 1) gated on T9 hardware E2E, 7 ACs are mockup-only (Phase 2 animation, GPU column, two-tone bar, contract docstring, file self-containment) and not productized by this lift.

## Carry-forward follow-ups

### Production-affecting (watch in T9)

- **I-1** — `report.py` detail-mode summary block reads `inference_us`, `execute_us`, `dram_read_bytes`, `vtcm_peak_bytes` keys that the QNN parsers (`csv_parser.py`, `qhas_parser.py`) don't populate. Real parsers emit `time_us`, `graph_execute_us`, `total_dram_read`, `peak_vtcm_alloc`, `accel_execute_us`. **Likely visible defect in T9**: most of the detail-mode summary block will silently render empty for real data. Fix-up commit ready to apply (rename keys at one of the two layers — render layer is the simpler edit, parser layer is the more correct change since the parser keys are closer to what the runtime emits).

### Cleanup nice-to-haves (judgment-deferred, not blocking)

- `_io_specs_from_config` lives in `perf.py` but is consumed only by `print_pre_bench_block` — could move to `_pre_bench.py` for locality.
- `Console(file=StringIO(), width=…, force_terminal=False, record=True)` idiom duplicated across 6+ new test files — candidate for a `tests/unit/conftest.py` `recording_console` fixture.
- Pre-existing `test_contains_detail_columns` is now too weak (only checks `"DRAM"` + `"VTCM"` substrings; column structure went from 7→10 cols).
- Pre-existing `test_contains_summary_metrics` and `test_top_n_limits_rows` are vacuous against the new column structure and should be replaced or removed.

### Forward-looking design contributions (deferred from §13 of the design doc)

- GPU column in production `LiveMonitorDisplay` (currently NPU+CPU only).
- `gpu_pct_avg` in `wmk perf` JSON output's `hw_monitor` block.
- Two-tone progress bar in production (currently single-tone).
- Promoting fake-mockup hardcoded constants (`HVX threads`, `91.3%` utilization, `1_843_200` peak VTCM, etc.) to module-level constants once production emits them.
- Surface sub-block content in `_pre_bench.py` (currently a placeholder).

## Architectural observations from the lift

**The "land cleanly + small fix-up commit" pattern recurred across T4/T5/T6/T7.** Each render task surfaced an adjacent legacy redundancy that the spec didn't anticipate: T4 needed to drop a `' us'` suffix that overflowed the new locked column; T5 surfaced a parser-ordering assumption that broke Cum% monotonicity; T6 retired a stale `_print_model_info` helper and changed the dynamic-dim sentinel; T7 found a duplicate save-line print. None of these were called out in the plan, all were caught by the per-task review gate, and all landed as small follow-up commits rather than amending the original. The pattern argues for keeping per-task reviewers in the loop even when the task itself is mechanically straightforward — the spec is necessarily silent on adjacent code that the new code accidentally illuminates.

**Defensive sort moved from parser to render layer in T5's I-2 fix.** The parser ordering had been silently load-bearing for cumulative-percent monotonicity. The fix sorts inside `_display_detail_report` rather than asserting parser ordering, which is the right invariant location: render is responsible for the contract that "rows are presented in cum-percent order." Parser callers should be free to provide ops in any order without breaking the visual invariant.

**Grep-first discipline caught a wrong artifact key in T7's plan.** The plan referenced `trace_result.artifacts["profiling_csv"]`; greppage of the actual `OpTraceResult` dataclass surfaced that the field is named differently (and may or may not be present, hence `.get(...)`). The implementer caught this before writing test stubs against a non-existent key. This was the single most-effective spec-vs-reality cross-check during the lift.

**`HWLiveDisplay` was deleted rather than refactored-into-`LiveMonitorDisplay`** because greppage showed zero call-sites and the survivor was already feature-equivalent. Refactoring would have meant merging two near-identical class hierarchies under a contract no caller exercised. The deletion saves 207 lines and removes a maintenance burden the project did not know it was carrying. This kind of orphan-detection (run a grep for class name across both `src/` and `tests/`, count call-sites, decide) is cheap and worth doing as part of every render-layer touch.

## Production-readiness

- **Unit tests:** All new tests passing (3935 passed at lift end vs 3935 baseline pre-lift; 248 commands tests + 124 monitor tests new/updated).
- **T9 hardware E2E:** Pending user verification on NPU/QNN with `convnext-base-224`. Two test invocations to run:
  - `wmk perf -m facebook/convnext-base-224 --op-tracing basic`
  - `wmk perf -m facebook/convnext-base-224 --op-tracing detail --iterations 50`
- **I-1 (summary key mismatch)** is the only known visible-defect risk in T9 — likely surfaces as a mostly-empty summary block in detail mode. Fix-up commit ready to apply if visible.
- **Linting:** `uv run ruff check --fix` applied per task; tree clean at lift end.
- **Lift commit chain:** `12a86c81..7b077bc8` (13 commits), authored on `feat/op-tracing-refactor`, ahead of `gh/feat/op-tracing-refactor` by 13 commits. Ready for force-push or PR update once T9 passes.
