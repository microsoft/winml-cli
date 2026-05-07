# Perf UX Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve `wmk perf --monitor` console output UX — dual CPU/NPU chart, 3-row status, full percentiles, HW metrics in JSON, ExitStack fix, EP registry, and more.

**Architecture:** All changes are in `modelkit/commands/perf.py` (display layer) with minor additions to `BenchmarkResult` (data model). The monitoring infrastructure (`HWMonitor`, `PdhPoller`, `_pdh.py`) is unchanged — only the presentation layer is modified.

**Tech Stack:** Python, Rich (Panel, Table, Live, Progress), plotext, click

**Issue:** https://github.com/gim-home/ModelKit/issues/257
**Branch:** `feat/perf-ux-257`

---

### Task 1: Fix ExitStack + EP Registry (Issue 1.5, 1.6)

Fix the two code quality issues from PR #256 review before adding new features.

**Files:**
- Modify: `modelkit/commands/perf.py` (`_run_benchmark_monitored`, lines 329-394)
- Test: `tests/session/test_ep_monitor.py`

**Step 1: Fix manual `__enter__`/`__exit__` with `contextlib.ExitStack`**

In `_run_benchmark_monitored()`, replace the manual ep_monitor context management:

```python
# BEFORE (lines 355-392):
ep_ctx = ep_monitor.__enter__() if ep_monitor else None
try:
    ...
finally:
    if ep_monitor:
        ep_monitor.__exit__(None, None, None)

# AFTER:
import contextlib

with contextlib.ExitStack() as stack:
    if ep_monitor:
        stack.enter_context(ep_monitor)
    ...
```

**Step 2: Replace hardcoded EP string match with registry**

Add a module-level registry dict and use it in `_run_benchmark_monitored`:

```python
# Module level:
from ..session.perf import VitisAIMonitor, QNNMonitor, OpenVinoMonitor

_EP_MONITOR_REGISTRY: dict[str, type] = {
    "vitisai": VitisAIMonitor,
    "qnn": QNNMonitor,
    "openvino": OpenVinoMonitor,
}

# In _run_benchmark_monitored:
monitor_cls = _EP_MONITOR_REGISTRY.get(ep)
ep_monitor = monitor_cls() if monitor_cls and monitor_cls.is_available() else None
```

Note: The registry import must be lazy (inside function) since the monitors import `_pdh` which is Windows-only. Move the dict inside the method or use lazy imports.

**Step 3: Run tests + lint**

Run: `uv run pytest tests/session/test_ep_monitor.py -v`
Run: `uv run ruff check modelkit/commands/perf.py`

**Step 4: Commit**

```bash
git add modelkit/commands/perf.py
git commit -m "fix(perf): use ExitStack for EP monitor + EP registry pattern (#257)"
```

---

### Task 2: Add HW Metrics to BenchmarkResult + JSON (Issue 1.3)

Wire `_hw_metrics` into `BenchmarkResult` so JSON reports include hardware data.

**Files:**
- Modify: `modelkit/commands/perf.py` (`BenchmarkResult`, `_collect_results`, `to_dict`)
- Test: `tests/session/test_ep_monitor.py` or `tests/commands/test_perf_cli.py`

**Step 1: Add `hw_monitor` field to BenchmarkResult dataclass**

```python
@dataclass
class BenchmarkResult:
    ...
    # Hardware monitor metrics (from HWMonitor.to_dict())
    hw_monitor: dict[str, Any] | None = None
```

**Step 2: Update `to_dict()` to include hw_monitor**

```python
def to_dict(self) -> dict[str, Any]:
    result = {
        "benchmark_info": { ... },
        ...
    }
    if self.hw_monitor:
        result["hw_monitor"] = self.hw_monitor
    return result
```

**Step 3: Update `_collect_results()` to pass `_hw_metrics`**

In `PerfBenchmark._collect_results()`, add:
```python
hw_monitor=getattr(self, '_hw_metrics', None),
```

**Step 4: Write test verifying hw_monitor appears in JSON**

**Step 5: Run tests + commit**

```bash
git commit -m "feat(perf): include HW monitor metrics in JSON report (#257)"
```

---

### Task 3: Expand Console Report (Issue 1.4)

Add full percentile table + conditional HW section to `display_console_report`.

**Files:**
- Modify: `modelkit/commands/perf.py` (`display_console_report`, lines 600-657)

**Step 1: Expand latency table to 8 columns**

```python
table = Table(show_header=True, header_style="bold cyan")
for col in ["Avg", "P50", "P90", "P95", "P99", "Min", "Max", "Std"]:
    table.add_column(col, justify="right")

table.add_row(
    f"{result.mean_ms:.2f}",
    f"{result.p50_ms:.2f}",
    f"{result.p90_ms:.2f}",
    f"{result.p95_ms:.2f}",
    f"{result.p99_ms:.2f}",
    f"{result.min_ms:.2f}",
    f"{result.max_ms:.2f}",
    f"{result.std_ms:.2f}",
)
```

**Step 2: Add conditional HW section**

After the throughput line, if `result.hw_monitor` is populated:
```python
if result.hw_monitor:
    console.print("[bold]Hardware (during benchmark)[/bold]")
    npu = result.hw_monitor.get("npu", {})
    cpu = result.hw_monitor.get("cpu", {})
    ram = result.hw_monitor.get("ram", {})
    dev_mem = result.hw_monitor.get("device_memory", {})
    console.print(
        f"  NPU: {npu.get('mean_pct', 0):.1f}% avg, {npu.get('peak_pct', 0):.1f}% peak  |  "
        f"CPU: {cpu.get('mean_pct', 0):.1f}% avg"
    )
    console.print(
        f"  RAM: {ram.get('used_mb', 0):.0f} MB  |  "
        f"Device Memory: {dev_mem.get('peak_mb', 0):.0f} MB peak"
    )
```

**Step 3: Run tests + commit**

```bash
git commit -m "feat(perf): expand console report with full percentiles + HW section (#257)"
```

---

### Task 4: Dual CPU/NPU Chart + 3-Row Status (Issue 1.1, 1.2)

Redesign the live monitor display: dual-series chart and structured status rows.

**Files:**
- Modify: `modelkit/commands/perf.py` (`_LiveMonitorDisplay._render_chart`, `_render_status`, `update`)

**Step 1: Update `update()` to also pass `cpu_samples`**

Add `cpu_samples: list[float] = None` parameter. Pass from `_run_benchmark_monitored`:
```python
display.update(
    ...
    cpu_samples=hw.cpu_samples,
)
```

**Step 2: Update `_render_chart` for dual-series plotext**

```python
def _render_chart(self, util_samples, cpu_samples=None):
    ...
    if util_samples:
        plt.plot(util_samples, label="NPU %", marker="braille")
    if cpu_samples:
        plt.plot(cpu_samples, label="CPU %", marker="braille")
    plt.title("Utilization %")
    plt.ylabel("Usage %")
    plt.ylim(0, 100)
    ...
```

**Step 3: Restructure `_render_status` into 3 rows**

```python
def _render_status(self, iteration, latency_ms, util_samples, memory_mb,
                   cpu_pct=0.0, ram_mb=0.0) -> str:
    ...
    # Row 1: Progress
    row1 = f"  {progress}  |  Device: {self._device}  |  Batch: 1"

    # Row 2: Hardware
    row2 = (f"  NPU: [bold]{mean_util:.1f}%[/bold] avg ({current_util:.1f}% now)  |  "
            f"CPU: {cpu_pct:.1f}%  |  RAM: {ram_mb:.0f} MB  |  Dev Mem: {memory_mb:.0f} MB")

    # Row 3: Inference
    throughput = 1000.0 / latency_ms if latency_ms > 0 else 0
    row3 = f"  Lat: {latency_ms:.2f} ms  |  ~{throughput:.0f} smp/s"

    return f"{row1}\n{row2}\n{row3}"
```

**Step 4: Update `print_final_snapshot` similarly with CPU chart + 3-row summary**

**Step 5: Run live test + commit**

Run: `uv run wmk perf -m microsoft/resnet-50 --iterations 200 --monitor`
Verify: Dual chart visible, 3-row status, CPU + NPU both plotted.

```bash
git commit -m "feat(perf): dual CPU/NPU chart + 3-row status in live monitor (#257)"
```

---

### Task 5: Surface Monitor Failure + Live Throughput (Issue 2.1, 2.2)

**Files:**
- Modify: `modelkit/commands/perf.py` (`_run_benchmark_monitored`)

**Step 1: Replace silent logger.info with console warning**

When `HWMonitor.is_available()` returns False:
```python
from rich.console import Console
console = Console(stderr=True)
console.print("[yellow]Warning: HWMonitor unavailable on this system. "
              "Running without hardware monitoring.[/yellow]")
```

**Step 2: Throughput is already in Row 3 from Task 4**

The 3-row status from Task 4 already includes `~{throughput:.0f} smp/s`. Verify it works.

**Step 3: Commit**

```bash
git commit -m "feat(perf): visible monitor failure warning + live throughput (#257)"
```

---

### Task 6: Unified Report — Merge Snapshot with Console Report (Issue 2.3)

**Files:**
- Modify: `modelkit/commands/perf.py` (`_run_benchmark_monitored`, `display_console_report`)

**Step 1: Remove `print_final_snapshot` call from `_run_benchmark_monitored`**

The `print_final_snapshot` currently prints a separate panel. Instead, pass `hw_dict` through to `display_console_report` which now handles everything (from Task 3).

**Step 2: Update the CLI `perf` function flow**

After `benchmark.run()` returns `result`, call `display_console_report(result, console)` which now shows latency + throughput + HW section (all in one). The plotext chart can be printed as a final standalone panel above the unified report.

**Step 3: Remove or simplify `print_final_snapshot` to only render the chart panel**

```python
def print_final_chart(self, util_samples, cpu_samples=None):
    """Print only the final chart (no status lines — those go in console report)."""
    console = Console()
    chart = self._render_chart(util_samples, cpu_samples)
    panel = Panel(chart, title="[bold]HW Monitor[/bold]", border_style="green")
    console.print(panel)
```

**Step 4: Test + commit**

```bash
git commit -m "feat(perf): unified report — merge chart + latency + HW into one output (#257)"
```

---

### Task 7: ONNX + Module Mode Monitor Support (Issue 2.4, 2.5)

**Files:**
- Modify: `modelkit/commands/perf.py` (`_run_onnx_benchmark`, `_perf_modules`)

**Step 1: Add monitoring to `_run_onnx_benchmark`**

Wrap the inference loop with HWMonitor when `config.monitor` is True:

```python
def _run_onnx_benchmark(..., config: BenchmarkConfig) -> BenchmarkResult:
    ...
    if config.monitor:
        from ..session.perf import HWMonitor
        if HWMonitor.is_available():
            hw_monitor = HWMonitor(poll_interval_ms=100)
            with session.perf(warmup=warmup) as stats, hw_monitor as hw:
                for i in range(total_iterations):
                    session.run(inputs)
            # Store hw_metrics on the result
            ...
    else:
        # existing simple loop
        ...
```

**Step 2: Add monitoring to `_perf_modules`**

Similar pattern — wrap each module's benchmark loop with HWMonitor when `--monitor` is passed. Since module mode benchmarks each module separately, HWMonitor runs per-module.

**Step 3: Pass `--monitor` and `--ep` through to module mode**

The `_perf_modules` function currently ignores `monitor` and `ep` from the click context. Add them as parameters.

**Step 4: Test + commit**

```bash
git commit -m "feat(perf): support --monitor for ONNX direct + module mode benchmarks (#257)"
```

---

### Task 8: P3 Polish — Progress Bar, Warmup, Module JSON, Precision, Comparison (Issue 3.1-3.5)

**Files:**
- Modify: `modelkit/commands/perf.py`

**Step 1: Add Rich Progress bar to live display (3.1)**

In `_render_status`, add a compact progress indicator with elapsed time:
```python
from rich.progress import Progress, BarColumn, TimeElapsedColumn
```
Or simpler: compute a text-based bar inline:
```python
pct = iteration / self._total
bar_len = int(pct * 30)
bar = f"[{'=' * bar_len}{' ' * (30 - bar_len)}] {pct:.0%}"
```

**Step 2: Add warmup latency section (3.2)**

In `BenchmarkResult`, add `warmup_mean_ms` field. In `_collect_results`, compute mean of warmup samples from `stats.all_samples_ms[:warmup]`. Show in console report:
```
Warmup: 2.15 ms avg (first 10 iterations)
```

**Step 3: Module JSON parity (3.3)**

Expand `_perf_modules` JSON per-instance to include p50, p95, p99, std, throughput.

**Step 4: Fix --precision flag (3.4)**

Remove non-auto choices from `click.Choice` or change to read-only display of detected precision.

**Step 5: Comparison mode placeholder (3.5)**

Add `--compare-devices` option stub that prints "Not yet implemented" for now. Or implement the basic sequential-run comparison if time permits.

**Step 6: Final lint + test + commit**

```bash
git commit -m "feat(perf): progress bar, warmup latency, module JSON parity, precision fix (#257)"
```

---

### Task 9: Final Integration Test + Lint + PR

**Step 1: Run full test suite**

```bash
uv run pytest tests/session/test_ep_monitor.py -v
uv run pytest tests/commands/ -v
uv run ruff check modelkit/commands/perf.py tests/session/test_ep_monitor.py
```

**Step 2: Live verification**

```bash
uv run wmk perf -m microsoft/resnet-50 --iterations 500 --monitor
```

Verify:
- Dual CPU/NPU chart rendered
- 3-row status line (progress, hardware, inference)
- Unified report at end (chart + latency table + HW section)
- JSON includes hw_monitor section

**Step 3: Squash commits and create PR**

```bash
git rebase -i main  # squash to single commit
git push -u origin feat/perf-ux-257
gh pr create --title "feat(perf): improve wmk perf --monitor console output UX (#257)" ...
```

---

## Dependency Graph

```
Task 1 (ExitStack + EP registry) → no deps
Task 2 (BenchmarkResult JSON)    → no deps
Task 3 (Console report expand)   → depends on Task 2
Task 4 (Dual chart + 3-row)      → no deps
Task 5 (Warnings + throughput)   → depends on Task 4
Task 6 (Unified report)          → depends on Task 3, 4
Task 7 (ONNX + module monitor)  → depends on Task 2
Task 8 (P3 polish)               → depends on Task 4, 6
Task 9 (Integration)             → depends on all
```

Parallelizable: Tasks 1, 2, 4 can run in parallel.
Sequential chain: 4 → 5 → 6 → 8 → 9
