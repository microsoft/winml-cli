# Perf Console Mockup — Design

**Initial Date**: 2026-04-28
**Last Revised**: 2026-05-01
**Version**: 2.1
**Status**: Production-lifted; mockup retained as reference (T9 hardware E2E pending)
**Target file**: `docs/design/perf/console_mockup.py`
**Pattern reference**: `docs/design/static_analyzer/console_mockup.py`
**Companion**: `docs/design/perf/op_tracing_mockup_plan.md` (executed)
**Lift plan**: `docs/design/perf/2026-04-29-op-tracing-production-lift-plan.md` (executed T1-T8 + cleanup)
**Lift outcome**: `docs/design/perf/2026-05-01-op-tracing-production-lift-summary.md`

## Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-28 | Initial spec — 3-phase mockup (header, live monitor, post-bench summary). Chart 80 wide, NPU+CPU lines. |
| 2.0 | 2026-04-29 | Adds Phase 4 (op-tracing) per `op_tracing_mockup_plan.md`. UX revisions: 3-silicon chart (+GPU), chart and basic op-table widened to 120, header reordered to 3-block layout, smart `--iterations=1` default with `--op-tracing`, save-to footer for op-trace artifacts, "Op-Tracing" rename. New Contract D for op-tracing per-instance schema. New `WINDOW_SECONDS=15` (was 10). |
| 2.1 | 2026-05-01 | Production-lifted across 13 commits on feat/op-tracing-refactor. Mockup retained as reference. |

## 1. Purpose

Port the production `wmk perf -m <model> --monitor [--iterations N] [--op-tracing basic|detail]` console output to an executable mockup that simultaneously:

1. Demonstrates the visual UX with hand-tuned fake data (so design reviewers can iterate without running real benchmarks)
2. Formalizes the four dict shapes that cross production code boundaries (in-memory + on-disk)
3. Provides pure rendering helpers that can be lifted into production
4. Acts as a forward-looking spec for changes the production code should make (GPU column, basic-mode op-table layout, smart `--iterations` default, etc.)

Self-contained — no imports from `winml.modelkit.*`. The static_analyzer mockup pattern.

## 2. Scope

### In scope

- Faithful port of current `wmk perf` rendering for `--monitor`, `--monitor --iterations N`, `--op-tracing basic|detail`, `--op-tracing basic --top-k K`, `--op-tracing basic --iterations 1`, and the `--top-k` without `--op-tracing` hard-error
- Three-silicon utilization chart (NPU + CPU + GPU) — current `LiveMonitorDisplay` plots NPU+CPU only; **this mockup adds GPU as a forward-looking design contribution**
- Two-tone progress bar (warmup dim + measured green + pending `░`) — current bar is single-tone; **this mockup proposes the two-tone improvement**
- Four data contracts: on-disk perf.json + 2 in-memory dicts + per-instance op-tracing schema
- Smart-default behavior: `--op-tracing` without explicit `--iterations` collapses to 1 (mirroring the production-side fix landed earlier in this PR)
- Op-tracing artifact save-to footer (JSON + CSV paths) — production currently shows JSON only; **this mockup proposes also surfacing the CSV path**

### Out of scope

- Importing `commands/_live_chart.py::LiveMonitorDisplay` (mockup is self-contained)
- Real plotext-vs-Rich coupling (chart still hard-coded width via `CHART_WIDTH`; Panel auto-fills terminal — known asymmetry, see §10)
- Writing actual files to disk (Approach A — print-only)
- Failure scenarios (`no_data`, `parse_failed`, `basic_fallback`) — single happy-path scenario only
- Refactoring `LiveMonitorDisplay` itself (the mockup is a target spec, not a refactor)
- Pytest-based tests for the mockup (visual inspection)

## 3. Output sections

Four phases in execution order. All in one file, all driven by `demo()` with optional kwargs.

### Phase 1 — Pre-bench header (static, three sub-blocks)

```
Model:    facebook/convnext-base-224  (HF)
ONNX:     C:\Users\zhengte\.cache\winml\artifacts\facebook_convnext-base-224\imgcls_db24bf8910f169d6_compiled.onnx

Task:     image-classification
Opset:    17
Inputs:   pixel_values   [1, 3, 224, 224]   float32
Outputs:  logits         [1, 1000]

Device:   auto → npu  (QNN)
```

Three blocks separated by blank lines:
- **Identity**: `Model:` and `ONNX:` — what the user is benchmarking
- **Surface**: `Task:` / `Opset:` / `Inputs:` / `Outputs:` — model contract
- **Device**: `Device:` — requested → resolved (with EP suffix when known)

The `(HF)` annotation flips to `(local)` when the user passes a direct `.onnx` file.

### Phase 2 — Live monitor (animated, ~3 sec, single `rich.Live` region)

`rich.Panel` titled `HW Monitor - <model>` with blue border. Inside:
- **Chart**: plotext braille-marker line plot, 120 cells wide × 15 cells tall
  - NPU (`SILICON_COLORS["npu"]` = green)
  - CPU (`SILICON_COLORS["cpu"]` = cyan)
  - GPU (`SILICON_COLORS["gpu"]` = magenta) — **forward-looking; not in current `LiveMonitorDisplay`**
  - Y-axis fixed `0..100` with ticks `[0, 20, 40, 60, 80, 100]`
  - X-axis sliding window: last `WINDOW_SECONDS = 15.0` seconds
  - Title with Rich-coloured legend swatches
- **Progress bar** (3 rows below chart):
  - Row 1 — two-tone progress (**forward-looking — see §10**): `[░░░░████████████░░░░░░░░] 47%   |  Iter: 470/1000   |  Device: auto`
    - Warmup chunk: `dim` style
    - Measured chunk: `green` style
    - Pending: `░` light shade in dim
    - Production today renders a single-tone bar; this is a UX improvement the mockup proposes
  - Row 2 — hardware: `NPU: 73.5% avg (88.0% now) | CPU: 31.2% | GPU: 8.4% | Sys Mem: 52490 MB | Device Mem: 95/119 MB (local/shared)`
  - Row 3 — inference: `Latency: 3.18 ms | Throughput: ~315 smp/s`

`transient=False` — last frame stays in scrollback after Live exits. Phase 3+4 print below it.

### Phase 3 — Post-bench summary (static)

```
Latency (ms)
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│  Avg │  P50 │  P90 │  P95 │  P99 │  Min │  Max │  Std │
├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ 3.16 │ 3.13 │ 3.31 │ 3.35 │ 3.45 │ 3.04 │ 3.45 │ 0.09 │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
  Warmup: 6.69 ms avg (first 10 iterations)

Throughput: 316.21 samples/sec

Hardware (during benchmark)
  NPU: 73.5% avg, 100.0% peak  |  CPU: 31.3% avg  |  GPU: 8.4% avg
  Sys Mem: 52,496 MB  |  Device Mem: 95/119 MB (local/shared)

  📁 Results saved to: ./facebook_convnext-base-224_perf.json
```

### Phase 4 — Op-tracing (only when `--op-tracing` is set)

Activates only when `op_tracing` parameter is `"basic"` or `"detail"`. Collapses `--iterations` to **1** when not explicitly passed (the smart default).

Layout:
1. **Section rule**: `── Op-Tracing (basic|detail, N samples) ──` (note: "Op-Tracing" not "Operator Tracing")
2. **Rich Table** titled `Top K Operator Instances by Avg Duration  (timings in μs)`:
   - **Basic mode (4 cols)**: `Node | Type | p90 | % Tot` — slim scan view
     - Total table width: **120 cells** (matches chart width)
     - Column widths: Node 80 (fixed), Type 12 (fixed), p90 9 (fixed), % Tot 6 (fixed)
     - Node uses **left-ellipsis truncation** (preserves trailing op name; `…0/Conv_token_2` not `/resnet/encoder/sta…`)
   - **Detail mode (10 cols)**: `# | Node | Type | Avg | Total | % Tot | Cum % | p90 | DRAM(R) | VTCM Hit` — power-user view
     - Total width: auto-fit to content (~143 chars natural)
3. **Summary line**: `Top K instances account for X% of execute time (sum_top / sum_all μs accumulated)`
   - Switches to `All N instances account for 100.0% ...` when `K >= len(ops)`
4. **Mode-specific footer**:
   - Basic: `HVX threads: 4   Accel execute: ... μs   Samples: N`
   - Detail (2 dim lines): inference/execute/utilization on line 1; DRAM/VTCM totals on line 2
5. **Single-sample note** (only when `num_samples == 1`): `Note: p90 reflects single-sample data; ...`
6. **Save-to footer** (forward-looking — production today shows JSON only):
   ```
     📁 Op-trace JSON: ./<model_slug>_op_trace.json
     📁 Profiling CSV: ./profiling_output.csv
   ```

## 4. Data contracts (the docstring's job)

The mockup's docstring declares **four** dict shapes that cross production code boundaries.

### 4.1 Contract A — On-disk `<model>_perf.json` schema

```python
{
    "benchmark_info": {
        "model_id": str, "task": str, "device": str, "precision": str,
        "iterations": int, "warmup": int, "batch_size": int,
        "timestamp": str,                              # ISO-8601 UTC
    },
    "model_info": {
        "input_names":   list[str],
        "input_shapes":  list[list[int]],
        "input_types":   list[str],
        "output_names":  list[str],
        "output_shapes": list[list[int]],
    },
    "latency_ms": {
        "mean": float, "min": float, "max": float,
        "p50": float, "p90": float, "p95": float, "p99": float,
        "std": float, "warmup_mean": float,
    },
    "throughput": {"samples_per_sec": float, "batches_per_sec": float},
    "raw_samples_ms": list[float],                    # length == iterations (post-warmup)
    "hw_monitor": {
        "monitor": str,
        "npu_pct_avg": float, "npu_pct_peak": float,
        "cpu_pct_avg": float,
        "gpu_pct_avg": float,                         # NEW (forward-looking; not in current schema)
        "ram_mb_avg": float,
        "device_mem_local_mb_peak": float,
        "device_mem_shared_mb_peak": float,
    },
}
```

### 4.2 Contract B — In-memory HW sample (chart consumer)

```python
{
    "t": float,             # elapsed seconds since benchmark start
    "npu_pct": float,       # 0..100
    "cpu_pct": float,       # 0..100
    "gpu_pct": float,       # 0..100  (NEW — forward-looking)
    "ram_mb": float,
    "mem_local_mb": float,  # NPU device memory (local), current value
    "mem_shared_mb": float, # NPU device memory (shared), current value
}
```

### 4.3 Contract C — In-memory progress state

```python
{
    "iteration": int,    # 1..total_iterations (includes warmup)
    "total": int,        # total_iterations (warmup + measured)
    "warmup": int,       # warmup iteration count
    "latency_ms": float, # most recent measurement
}
```

### 4.4 Contract D — Op-tracing per-instance schema (NEW in v2)

Implemented as the `FakeOp` dataclass (production: a corresponding type in `session/monitor/op_metrics.py` or its successor). Stored fields are dataclass attributes; derived values are `@property` methods computed from `sample_durations_us`.

**Stored fields (dataclass attributes):**

| Field | Type | Notes |
|---|---|---|
| `node_name` | `str` | full ONNX node path |
| `op_type` | `str` | e.g. `"Conv2d"`, `"MatMul"`, `"Add"` |
| `sample_durations_us` | `list[float]` | per-iteration durations — source of truth |
| `dram_read_bytes` | `int \| None` | optional; detail-mode column |
| `vtcm_hit_ratio` | `float \| None` | 0..1; optional; detail-mode column |

**Derived `@property` methods (computed from `sample_durations_us`):**

| Property | Body |
|---|---|
| `sample_count` | `len(sample_durations_us)` |
| `avg_us` | `statistics.fmean(sample_durations_us)` |
| `total_us` | `float(sum(sample_durations_us))` |
| `p90_us` | inclusive 90th percentile via `statistics.quantiles(..., n=10, method="inclusive")[8]`; degenerates to `sample_durations_us[0]` when `sample_count == 1` |

### Contract-to-derivation invariant

The displayed `latency_ms` dict is **derived from** `RAW_SAMPLES_MS` via `compute_latency_stats()`. The displayed `hw_monitor` dict is **derived from** the HW sample lists via `compute_hw_aggregates()`. Op-tracing's per-instance `avg_us`/`total_us`/`p90_us` are **derived from** `sample_durations_us` via `FakeOp` properties. Reruns are deterministic (`np.random.seed(42)` for HW/latency, `OP_TRACING_SEED=42` for op timings via `random.Random(seed)`).

The table on screen is provably the right summary of the displayed raw data, not just plausibly.

## 5. Module constants

```python
# Live-monitor / chart
SILICON_COLORS    = {"npu": "green", "cpu": "cyan", "gpu": "magenta"}
CHART_HEIGHT      = 15
CHART_WIDTH       = 120                    # was 80 in v1
WINDOW_SECONDS    = 15.0                   # was 10.0; bumped proportionally with chart width
REFRESH_FPS       = 5
PROGRESS_WIDTH    = 20
POLL_INTERVAL_S   = 0.1

# Op-tracing
OP_TRACING_TOP_K_DEFAULT       = 5
OP_TRACING_NUM_SAMPLES         = 100        # default for --iterations; collapses to 1 with --op-tracing
OP_TRACING_NODE_NAME_MAX_WIDTH = 80
OP_TRACING_SEED                = 42
```

## 6. Helper inventory

All pure (`state → Rich primitive`). Each is a candidate for lifting into production at refactor time.

| Category | Helper | Signature |
|---|---|---|
| Stat derivation | `compute_latency_stats` | `(raw_samples_ms, warmup_samples_ms) → dict` |
|  | `compute_hw_aggregates` | `(npu, cpu, gpu, ram, mem_local, mem_shared) → dict` |
| Phase 1 | `build_pre_bench_block` | `(model_id, opset, task, device, device_resolved, io_config, *, is_hf, onnx_path, ep_resolved="") → Group` |
| Phase 2 | `build_chart` | `(npu, cpu, gpu, *, t_now, window_s, poll_interval_s) → Group` |
|  | `build_progress_bar` | `(iteration, total, warmup, *, width=20) → str` (Rich markup, two-tone) |
|  | `build_status_lines` | `(progress, hw_now, latency_ms, *, device_label) → Text` (3-row joined) |
|  | `build_live_panel` | `(chart, status_lines, model_id) → Panel` |
| Phase 3 | `build_latency_table` | `(latency_ms_dict) → Table` |
|  | `build_throughput_line` | `(samples_per_sec) → Text` |
|  | `build_hw_summary_block` | `(hw_aggregates) → Group` |
|  | `build_save_footer` | `(json_path) → Text` |
| Phase 4 | `render_op_tracing` | `(console, ops, *, level, top_k, num_samples, json_path="", csv_path="") → None` (~157 LoC; see §13.4 for split trigger) |
|  | `_build_op_tracing_summary_line` | `(top_ops, all_ops, k) → str` |
|  | `_format_number` | `(n: float\|int\|None) → str` |
|  | `_format_bytes` | `(n: int\|float\|None) → str` |
|  | `_truncate_node_name` | `(name, max_width) → str` (left-ellipsis) |
| CLI | `_parse_args` | `(argv) → argparse.Namespace` |
|  | `_user_passed_top_k` | `(argv) → bool` — sys.argv inspection because argparse cannot distinguish "default" from "explicit value matching default" |
|  | `_user_passed_iterations` | `(argv) → bool` — same pattern; enables the `--iterations=1` smart default with `--op-tracing` |
|  | `main` | `(argv=None) → int` (exit code) |

## 7. Animation plan (Phase 2)

| Knob | Value | Rationale |
|---|---|---|
| Wall-clock | ~3.0 sec | Matches real QNN convnext at ~3 ms/iter × 1010 iters |
| Live refresh | 5 fps (`REFRESH_FPS=5`) | Matches `LiveMonitorDisplay` |
| Total live ticks | 15 (= 5 fps × 3 sec) | Set in `demo()` |
| Iterations per tick | ~67 (= 1010/15) | Counter races by chunks (matches real cadence) |
| HW poll interval | 100 ms (`POLL_INTERVAL_S=0.1`) | Matches `LiveMonitorDisplay` |
| HW samples per silicon | 30 (= 0.1s × 30 = 3.0s) | Pre-generated at module load |
| Phase transition (warmup→measure) | At tick 1 | 10 warmup iters fit in first ~30ms; mockup's tick granularity collapses cleanly |

`transient=False` mirrors production. Phase 3+4 print below the persisted last live frame.

## 8. Fake data plan

All canned data lives in one section near module top, generated deterministically at module load.

```python
np.random.seed(42)

# Identity
MODEL_ID = "facebook/convnext-base-224"
IS_HF_MODEL = True              # toggle for HF model_id (True) vs direct .onnx (False)
CACHED_ONNX_PATH = r"C:\Users\zhengte\.cache\winml\artifacts\..."

# Op-tracing artifact paths (mockup defaults; production resolves from --output-dir)
OP_TRACE_JSON_PATH = "./facebook_convnext-base-224_op_trace.json"
OP_TRACE_CSV_PATH  = "./profiling_output.csv"

# Run config
TASK = "image-classification"
DEVICE = "auto";  DEVICE_RESOLVED = "npu";  EP_RESOLVED = "QNN"
PRECISION = "auto";  ITERATIONS = 1000;  WARMUP = 10;  BATCH_SIZE = 1
OPSET = 17;  PRODUCER = "pytorch v2.1.0"  # PRODUCER kept as constant though no longer rendered

# Latency data (cold-cache shape)
WARMUP_SAMPLES_MS = [25.1, 8.2, 5.4, 4.5, 4.1, 3.95, 3.92, 3.93, 3.91, 3.91]
RAW_SAMPLES_MS = (3.16 + 0.09 * np.random.randn(ITERATIONS)).round(3).tolist()

# HW data per silicon (30 samples, cold-start ramp + jitter around steady)
NPU_SAMPLES_FULL  = _hw_curve(steady=73, ramp_start=8,  jitter=8.0)
CPU_SAMPLES_FULL  = _hw_curve(steady=31, ramp_start=12, jitter=4.0)
GPU_SAMPLES_FULL  = _hw_curve(steady=8,  ramp_start=2,  jitter=3.0)

# Op-tracing data (20 ResNet-50-ish op templates; lognormal jitter via random.Random(42))
_OP_TEMPLATES = [...]  # 20 rows of (node_name, op_type, base_avg_us, dram_read_bytes, vtcm_hit_ratio)
generate_fake_ops(num_samples) -> list[FakeOp]
```

## 9. CLI surface

```
uv run python docs/design/perf/console_mockup.py [--op-tracing basic|detail] [--top-k N] [--iterations N]
```

Validation rules (in `main()` order):
1. `--top-k` without `--op-tracing` → exit 2 with `Error: --top-k requires --op-tracing to be set.`
2. `--top-k < 1` → exit 2 with `Error: --top-k must be >= 1.`
3. `--op-tracing` set AND user did not pass explicit `--iterations` → silently override `args.iterations = 1`
4. `--iterations < 1` → exit 2 with `Error: --iterations must be >= 1.`
5. Otherwise dispatch to `demo(...)` and return 0

User-typed flag detection uses `sys.argv` inspection helpers (`_user_passed_top_k`, `_user_passed_iterations`) because argparse cannot distinguish "default" from "explicit value matching default."

## 10. Known asymmetries & limitations

1. **Panel auto-fills terminal; chart is hard-coded width.** `Panel(...)` defaults to `expand=True` (terminal width); chart uses `plt.plotsize(CHART_WIDTH=120, CHART_HEIGHT=15)`. When terminal > ~124 chars, the chart sits left-aligned inside the Panel with whitespace to its right. Three resolution options if alignment matters:
   - `Panel(expand=False)` to shrink Panel to chart
   - Compute chart width from `console.width` at render time
   - Cap both at a known number (current state)
2. **Op-table fixed-width columns require `min_width=max_width=N` to stop Rich from redistributing spare cells**, NOT just `width=N`. The `Table(width=120)` model treats per-column `width=` as a target hint. Today's basic mode achieves the 120-cell envelope by forcing `Node` to `min_width=max_width=80` and metric columns to `width=12/9/6` — total falls out arithmetically. Documented for the next implementer.
3. **GPU column / 15s window / save-CSV footer / two-tone progress bar are forward-looking** — they don't exist in production today. The mockup is the spec for these proposed changes.
4. **Mockup runs in ~3 seconds** because the live phase fakes a 1000-iter benchmark via `time.sleep(0.2) × 15`. Real production timing depends on actual model + hardware.
5. **No file is written to disk** — the `📁 Results saved to:` and `📁 Op-trace JSON:` lines are print-only. Approach A locked this in v1.
6. **Phase 2 runs even when `--op-tracing` is set with `--iterations=1`.** Op-tracing's `--iterations` controls the *fake-sample count per operator* in Phase 4, NOT the live-phase iteration count. Phase 2 always animates ~3 sec of HW polling. This is intentional (HW polling is independent of op-tracing data) and matches what production should do, but a reader might wonder.
7. **`_OP_TEMPLATES` is a fixed 20-row backbone.** `--top-k > 20` clamps to 20 and the summary line switches to `All 20 instances account for 100.0% ...`. Real production data could have hundreds of instances; the mockup is bounded for demo readability.
8. **Chart width and basic op-table width are coupled by hand at 120.** Bumping one without the other will cause visual drift. There's no enforcement in code that they share a value.

## 11. Acceptance criteria

The implementation is acceptable when:

1. `uv run python docs/design/perf/console_mockup.py [args...]` runs to completion for every documented invocation (no flags, `--op-tracing basic`, `--op-tracing detail`, `--top-k N`, `--iterations N`, hard-error case)
2. Phase 2 runs in ~3 seconds (±0.5 sec)
3. The chart shows three distinct overlaid lines (NPU/CPU/GPU) with a sliding x-axis covering 0..15 sec
4. The progress bar visually distinguishes warmup (dim) from measured (green) sections
5. Phase 3 latency table cells are derivable from `RAW_SAMPLES_MS` (numerically verified: `np.mean / np.percentile / np.std` match)
6. Phase 3 hardware summary cells are derivable from per-silicon HW sample lists
7. Phase 4 basic op-table is exactly 120 cells wide; columns Node 80 / Type 12 / p90 9 / % Tot 6 (fixed)
8. Phase 4 detail op-table auto-fits content (~143 cells); 10 columns
9. Phase 4 Top-K summary line is computed from `top_ops`/`all_ops`/`k` (not hardcoded)
10. Phase 4 single-sample note appears iff `num_samples == 1`
11. `--top-k` without `--op-tracing` exits 2 with the spec'd error
12. `--op-tracing` without `--iterations` collapses to 1
13. Save-to footer paths shown for both perf.json (Phase 3) and op-trace JSON+CSV (Phase 4)
14. Module docstring documents Contracts A/B/C/D
15. Reruns produce identical Phase-1/3/4 output (Phase-2 live frames may interleave differently when piped, but underlying seeded data is identical)
16. File is self-contained: no `winml.modelkit.*` imports
17. `uv run ruff check` and `uv run ruff format --check` both clean
18. Phase 4 section rule reads exactly `── Op-Tracing (basic|detail, N samples) ──` (the rename — NOT "Operator Tracing")
19. `_truncate_node_name` truncates **left** (preserves trailing op suffix). For input `"/some/path/with/lots/of/nesting/to/Conv_final"` (45 chars) at `max_width=32`, output starts with `…` and ends with `Conv_final`
20. `FakeOp` dataclass attributes match Contract D's stored fields exactly (`node_name`, `op_type`, `sample_durations_us`, `dram_read_bytes`, `vtcm_hit_ratio`); `@property` methods match Contract D's derived list (`sample_count`, `avg_us`, `total_us`, `p90_us`)
21. `--op-tracing` smart-defaults `--iterations=1` (rule shows `(basic|detail, 1 samples)` and single-sample note appears) UNLESS user explicitly passes `--iterations N`

## 12. References

- v1 design doc (this file's history)
- Pattern reference: `docs/design/static_analyzer/console_mockup.py`
- Op-tracing implementation plan: `docs/design/perf/op_tracing_mockup_plan.md` (executed via subagent-driven-development on 2026-04-29)
- Production target the helpers could be lifted into: `src/winml/modelkit/commands/_live_chart.py::LiveMonitorDisplay`, `src/winml/modelkit/session/monitor/report.py::display_op_trace_report`
- Sibling mockups: `docs/design/build/console_mockup.py`, `docs/design/config/console_mockup.py`, `docs/design/static_analyzer/console_mockup.py`

## 12.1 Production lift outcome (v2.1)

The mockup was lifted into `src/winml/modelkit/` across 13 commits on `feat/op-tracing-refactor`. T9 (hardware E2E with `convnext-base-224`) is the only outstanding step and is gated on user-bound NPU/QNN access. Per-task SHAs:

| Task | Description | Commit(s) |
|---|---|---|
| T1 | `OperatorMetrics.samples_us` + 4 derived `@property` (avg/p90/total/count) | `12a86c81` |
| T2 | QNN CSV parser retains per-sample timings | `a6293201` |
| T3 | `_truncate_node_name` left-ellipsis helper | `bebda766` |
| T4 | Basic-mode 4-col render (Node/Type/p90/% Tot, width-locked 120) | `98e419bd` + fix `20da0415` (drop `" us"` suffix) |
| T5 | Detail-mode 10-col render (with Cum%, em-dash fallbacks) | `32b18dca` + fix `235855b4` (defensive sort) |
| T6 | Pre-bench identity panel (`_pre_bench.py`, 3-block layout) | `97640676` + fix `09bee22d` (retire `_print_model_info`, dynamic dim `?` sentinel) |
| T7 | Save-to footer (trace JSON + profiling CSV paths) | `2cc2ddc4` + fix `5fe92c78` (drop duplicate `Op-trace saved to:` line) |
| T8 | Live HW chart geometry: window 10s→15s, default width 80→120 | `d26400ba` |
| Cleanup | Delete orphaned `HWLiveDisplay` (zero call-sites; was duplicate of `LiveMonitorDisplay`) | `7b077bc8` |
| T9 | Hardware E2E with convnext-base-224 | **pending** |
| T10 | Documentation landing (this revision + handoff update + outcome summary) | this commit |

See the outcome summary (`2026-05-01-op-tracing-production-lift-summary.md`) for AC mapping, carry-forward follow-ups, and architectural observations from the lift.

## 13. Open follow-ups (not part of this design; flag for production lift)

- Extending production `LiveMonitorDisplay` to plot GPU samples (today: NPU+CPU only). Mockup is the spec.
- Adding `gpu_pct_avg` to the real `wmk perf` JSON output's `hw_monitor` block.
- Promoting fake-mockup hardcoded constants (`HVX threads: 4`, `91.3%` utilization, `1_843_200` peak VTCM, `* 1.009` inference factor, `* 8` VTCM:DRAM ratio) to module-level constants once the production path emits them.
- Splitting `render_op_tracing` (~120 lines) into `_build_op_tracing_table` / `_build_op_tracing_basic_footer` / `_build_op_tracing_detail_footer` if a third mode is added.
- Making `WINDOW_SECONDS` adapt to chart width dynamically rather than as a hand-tuned proportional bump.
- Optional Approach C upgrade: have the mockup write a fake JSON file and verify it round-trips through Contract A.
- Optional failure-scenario expansion: `--scenario {ok|cold_compile_only|no_data|basic_fallback}` switches.
- Resolving the Panel auto-width vs chart hard-coded-width asymmetry (§10.1).
