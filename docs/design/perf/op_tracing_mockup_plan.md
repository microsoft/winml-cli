# Op-Tracing Mockup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `docs/design/perf/console_mockup.py` to render an op-tracing section (Top-K operator instances with avg/total/%/cum %/p90) when invoked with `--op-tracing [basic|detail]`. Default `--top-k` is 5. `--top-k` without `--op-tracing` must hard-error.

**Architecture:** Keep the existing 3-phase rendering (pre-bench → live monitor → post-bench summary) unchanged. Add:
1. CLI argument parsing via `argparse` (script currently runs `demo()` unconditionally).
2. Fake per-instance operator data with per-sample timings (lognormal jitter so p90 ≠ avg).
3. A `render_op_tracing()` function that emits a Rich `Table` with 8 columns (basic) or 10 columns (detail), plus a "Top K instances account for X%" summary line and a mode-specific footer.
4. A new Phase 4 in `demo()` that runs only when `--op-tracing` is set.
5. Pre-flight validation: `--top-k` without `--op-tracing` exits 2 with a clear message before any rendering.

**Tech Stack:** Python stdlib (`argparse`, `statistics`, `dataclasses`, `random`), Rich (`Console`, `Table`), NumPy (already imported by the mockup).

**Why these decisions (locked from prior conversation, not subject to relitigation):**
- Top-K **instances**, not op-types. Both `Node Name` and `Op Type` columns are shown.
- Percentiles are **across samples for a given instance** — i.e. given an op like `Conv_0` measured 100 times, what's the spread? Requires preserving raw per-sample durations.
- Only `avg` + `p90`. No p50/p95/p99 in this report.
- Sort by **avg μs descending**, tiebreak on `node_name` for determinism.
- "Accumulated time consumption" is realised as **two columns**: per-instance `Total μs` (sum across samples) and the running `Cum %` down the table.
- Detail mode keeps its existing DRAM(R) / VTCM Hit columns, sourced from the same fake fixture.

**Reference implementation that proved the approach:** A working version of this exact UX was prototyped in the parallel `optimize` worktree. Numbers below match that prototype's seed=42 output so visual diffs against it are trivially reproducible.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `docs/design/perf/console_mockup.py` | Modify | Existing 3-phase mockup; add CLI parsing + op-tracing section |

Single-file change. No new modules, no test files (this is a design mockup, validated by visual inspection — not production code).

---

## Task 1: Add the new imports and module constants

**Files:**
- Modify: `docs/design/perf/console_mockup.py:85-95` (the `from __future__ import annotations` block and existing imports)

- [ ] **Step 1: Add stdlib imports for argparse, statistics, dataclasses, random**

After the existing `from __future__ import annotations` (line 85) and before `import time` (line 87), add:

```python
import argparse
import random
import statistics
import sys
from dataclasses import dataclass, field
```

The final import block should look like:

```python
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import plotext as plt
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
```

- [ ] **Step 2: Add op-tracing module constants below the existing constants block (after line 107 `POLL_INTERVAL_S = 0.1`)**

```python
# ── Op-tracing constants ────────────────────────────────────────────────────

OP_TRACING_TOP_K_DEFAULT = 5
OP_TRACING_NUM_SAMPLES = 100  # fake per-instance sample count
OP_TRACING_NODE_NAME_MAX_WIDTH = 32
OP_TRACING_SEED = 42  # deterministic jitter for reproducible visuals
```

- [ ] **Step 3: Verify the script still runs (no behaviour change yet)**

Run: `uv run python docs/design/perf/console_mockup.py`
Expected: Existing 3-phase animation runs to completion, no tracebacks. (Same output as before this task.)

- [ ] **Step 4: Commit**

```bash
git add docs/design/perf/console_mockup.py
git commit -m "chore(perf-mockup): add imports and constants for op-tracing section"
```

---

## Task 2: Add the `FakeOp` dataclass

**Files:**
- Modify: `docs/design/perf/console_mockup.py` — append after the `RAM_MB_FULL` line (~line 159, end of the existing fake-data block, before `# ── Stat derivation helpers ──`)

- [ ] **Step 1: Add the dataclass and its derived properties**

```python
# ── Op-tracing fake data ────────────────────────────────────────────────────


@dataclass
class FakeOp:
    """A single op instance with per-sample timing and hardware metrics.

    ``sample_durations_us`` is the source of truth.  ``avg_us``, ``total_us``,
    and ``p90_us`` are computed from it on demand.
    """

    node_name: str
    op_type: str
    sample_durations_us: list[float] = field(default_factory=list)
    dram_read_bytes: int | None = None
    vtcm_hit_ratio: float | None = None

    @property
    def sample_count(self) -> int:
        return len(self.sample_durations_us)

    @property
    def avg_us(self) -> float:
        return statistics.fmean(self.sample_durations_us)

    @property
    def total_us(self) -> float:
        return float(sum(self.sample_durations_us))

    @property
    def p90_us(self) -> float:
        # Inclusive percentile; degenerate to the single value when n == 1.
        if self.sample_count == 1:
            return self.sample_durations_us[0]
        deciles = statistics.quantiles(
            self.sample_durations_us, n=10, method="inclusive"
        )
        return deciles[8]  # 9 cut points -> [8] is the 90th percentile
```

- [ ] **Step 2: Verify the script still runs**

Run: `uv run python docs/design/perf/console_mockup.py`
Expected: Existing animation runs unchanged. (`FakeOp` is defined but unused so far.)

- [ ] **Step 3: Commit**

```bash
git add docs/design/perf/console_mockup.py
git commit -m "feat(perf-mockup): add FakeOp dataclass for per-instance op-tracing data"
```

---

## Task 3: Add the operator-templates fixture and generator

**Files:**
- Modify: `docs/design/perf/console_mockup.py` — append after the `FakeOp` dataclass

- [ ] **Step 1: Add the realistic ResNet-50-ish op templates**

```python
# Backbone: realistic ResNet-50-ish op mix (matches optrace_resnet50.csv shape).
# Each tuple: (node_name, op_type, base_avg_us, dram_read_bytes, vtcm_hit_ratio)
_OP_TEMPLATES: list[tuple[str, str, float, int, float]] = [
    ("/resnet/encoder/stage.3/block.0/Conv_token_2",     "Conv2d",    1247.0, 18_432, 0.942),
    ("/resnet/encoder/stage.2/block.0/Conv_token_2",     "Conv2d",     982.0, 12_800, 0.917),
    ("/resnet/encoder/stage.1/attention/MatMul_0",       "MatMul",     873.0,  8_192, 0.971),
    ("/resnet/encoder/stage.0/block.0/Conv_token_2",     "Conv2d",     641.0,  6_344, 0.894),
    ("/resnet/encoder/stage.3/block.2/Add_2",            "Add",        420.0,  4_096, 0.990),
    ("/resnet/encoder/stage.2/block.1/Add_1",            "Add",        351.0,  3_584, 0.985),
    ("/resnet/encoder/stage.3/block.0/LayerNorm_0",      "LayerNorm",  298.0,  2_304, 0.998),
    ("/resnet/encoder/stage.0/block.0/Add_0",            "Add",        284.0,  2_048, 0.991),
    ("/resnet/encoder/stage.1/block.0/Conv_token_2",     "Conv2d",     256.0,  3_072, 0.880),
    ("/resnet/encoder/stage.2/attention/MatMul_0",       "MatMul",     231.0,  4_352, 0.964),
    ("/resnet/encoder/stage.0/block.1/Conv_token_1",     "Conv2d",     188.0,  2_560, 0.873),
    ("/resnet/encoder/stage.1/block.1/Conv_token_1",     "Conv2d",     162.0,  1_792, 0.901),
    ("/resnet/encoder/stage.3/block.1/Mul_0",            "Mul",        134.0,  1_024, 0.996),
    ("/resnet/embedder/embedder/Conv_token_0",           "Conv2d",     128.0,  4_608, 0.812),
    ("/resnet/encoder/stage.2/block.0/Softmax_0",        "Softmax",    119.0,    768, 0.983),
    ("/resnet/encoder/stage.1/block.0/Reshape_0",        "Reshape",     94.0,    512, 1.000),
    ("/resnet/encoder/stage.0/block.0/Relu_0",           "Relu",        82.0,      0, 1.000),
    ("/resnet/encoder/stage.3/Transpose_0",              "Transpose",   71.0,    256, 1.000),
    ("/resnet/encoder/stage.2/block.1/Gemm_0",           "Gemm",        63.0,  1_024, 0.945),
    ("/resnet/encoder/stage.1/block.1/Sigmoid_0",        "Sigmoid",     54.0,    128, 1.000),
]


def generate_fake_ops(num_samples: int, seed: int = OP_TRACING_SEED) -> list[FakeOp]:
    """Build per-sample durations around each template's base avg.

    Uses lognormal multiplicative jitter so p90 sits noticeably above avg,
    matching NPU jitter in practice (occasional spikes from cache misses
    or thermal throttling).
    """
    rng = random.Random(seed)
    ops: list[FakeOp] = []

    for node_name, op_type, base_us, dram_r, vtcm in _OP_TEMPLATES:
        samples: list[float] = []
        for _ in range(num_samples):
            jitter = rng.lognormvariate(mu=0.0, sigma=0.08)
            # 5% chance of a moderate spike (1.15x to 1.45x).
            if rng.random() < 0.05:
                jitter *= rng.uniform(1.15, 1.45)
            samples.append(base_us * jitter)

        ops.append(
            FakeOp(
                node_name=node_name,
                op_type=op_type,
                sample_durations_us=samples,
                dram_read_bytes=dram_r,
                vtcm_hit_ratio=vtcm,
            )
        )

    return ops
```

- [ ] **Step 2: Smoke-test the generator from a Python REPL**

```bash
uv run python -c "
from docs.design.perf.console_mockup import generate_fake_ops
ops = generate_fake_ops(100)
print(f'{len(ops)} ops')
top = sorted(ops, key=lambda o: -o.avg_us)[0]
print(f'top: {top.node_name}  avg={top.avg_us:.1f}  p90={top.p90_us:.1f}  total={top.total_us:.1f}')
"
```

Expected output (seed=42, n=100):
```
20 ops
top: /resnet/encoder/stage.3/block.0/Conv_token_2  avg=1270.4  p90=1386.4  total=127037.8
```

If numbers differ, the seed or jitter math is wrong — fix before continuing.

- [ ] **Step 3: Commit**

```bash
git add docs/design/perf/console_mockup.py
git commit -m "feat(perf-mockup): add op templates fixture and lognormal-jitter generator"
```

---

## Task 4: Add formatting helpers

**Files:**
- Modify: `docs/design/perf/console_mockup.py` — append after `generate_fake_ops()`

- [ ] **Step 1: Add the formatters**

```python
# ── Op-tracing formatters (mirror modelkit/optracing/report.py) ─────────────


def _format_number(n: float | int | None) -> str:
    """Format a number with comma separators; one decimal for floats."""
    if n is None:
        return "-"
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{n:,}"


def _format_bytes(n: int | float | None) -> str:
    """Format a byte count to a human-readable string."""
    if n is None or n == 0:
        return "0"
    value: float = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024:
            if unit == "B" and value == int(value):
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _truncate_node_name(
    name: str, max_width: int = OP_TRACING_NODE_NAME_MAX_WIDTH
) -> str:
    """Trailing-ellipsis truncation; preserves the leading path context."""
    if len(name) <= max_width:
        return name
    return name[: max_width - 1] + "…"
```

- [ ] **Step 2: Verify with a quick eval**

```bash
uv run python -c "
from docs.design.perf.console_mockup import _format_number, _format_bytes, _truncate_node_name
print(_format_number(1247.34))         # 1,247.3
print(_format_bytes(18432))            # 18.0 KB
print(_truncate_node_name('/very/long/node/name/that/exceeds/the/width', 20))  # /very/long/node/nam…
"
```

Expected:
```
1,247.3
18.0 KB
/very/long/node/nam…
```

- [ ] **Step 3: Commit**

```bash
git add docs/design/perf/console_mockup.py
git commit -m "feat(perf-mockup): add op-tracing formatters (number, bytes, truncate)"
```

---

## Task 5: Add the summary-line builder

**Files:**
- Modify: `docs/design/perf/console_mockup.py` — append after the formatters

- [ ] **Step 1: Add the helper that builds the `Top K instances account for X%` line**

```python
def _build_op_tracing_summary_line(
    top_ops: list[FakeOp], all_ops: list[FakeOp], k: int
) -> str:
    """Render the 'Top K instances account for X%' summary string.

    Switches phrasing to 'All N instances account for 100.0%' when ``k``
    covers the full operator set.
    """
    sum_top = sum(op.total_us for op in top_ops)
    sum_all = sum(op.total_us for op in all_ops)
    pct = (sum_top / sum_all * 100.0) if sum_all > 0 else 0.0

    if len(top_ops) >= len(all_ops):
        prefix = f"All {len(all_ops)} instances"
    else:
        prefix = f"Top {k} instances"

    return (
        f"{prefix} account for [bold]{pct:.1f}%[/bold] of execute time "
        f"([bold]{_format_number(sum_top)}[/bold] / "
        f"{_format_number(sum_all)} μs accumulated)"
    )
```

- [ ] **Step 2: Quick sanity check**

```bash
uv run python -c "
from docs.design.perf.console_mockup import generate_fake_ops, _build_op_tracing_summary_line
ops = generate_fake_ops(100)
ops_sorted = sorted(ops, key=lambda o: -o.avg_us)
top5 = ops_sorted[:5]
print(_build_op_tracing_summary_line(top5, ops_sorted, 5))
"
```

Expected output (with markup intact):
```
Top 5 instances account for [bold]62.4%[/bold] of execute time ([bold]424,919.6[/bold] / 681,386.1 μs accumulated)
```

- [ ] **Step 3: Commit**

```bash
git add docs/design/perf/console_mockup.py
git commit -m "feat(perf-mockup): add op-tracing summary line builder"
```

---

## Task 6: Add the `render_op_tracing()` function

**Files:**
- Modify: `docs/design/perf/console_mockup.py` — append after the summary-line helper

- [ ] **Step 1: Add the renderer**

```python
def render_op_tracing(
    console: Console,
    ops: list[FakeOp],
    *,
    level: str,
    top_k: int,
    num_samples: int,
) -> None:
    """Render the operator tracing section (basic or detail mode).

    Layout:
      1. Section rule: '── Operator Tracing (basic|detail, N samples) ──'
      2. Rich Table titled 'Top K Operator Instances by Avg Duration  (timings in μs)'
         - Basic columns: # | Node Name | Op Type | Avg | Total | % Tot | Cum % | p90
         - Detail adds:   DRAM(R) | VTCM Hit
      3. Summary line: 'Top K instances account for X% of execute time (...)'
      4. Mode-specific footer:
         - Basic:  HVX threads, Accel execute, Samples
         - Detail: HVX threads, Inference, Execute, Utilization (line 1)
                   DRAM read total, VTCM read total, Peak VTCM alloc (line 2)
      5. If num_samples == 1, an italic note explaining degenerate p90.
    """
    console.print()
    console.rule(
        f"[bold]Operator Tracing ({level}, {num_samples} samples)[/bold]"
    )

    if not ops:
        console.print(
            "[yellow]Warning:[/yellow] No operator data available; "
            "profiling artifacts may be missing."
        )
        return

    # Sort by avg desc; tiebreak on node_name for determinism.
    ops_sorted = sorted(ops, key=lambda o: (-o.avg_us, o.node_name))
    k = min(top_k, len(ops_sorted))
    top_ops = ops_sorted[:k]
    sum_all_total = sum(o.total_us for o in ops_sorted)

    table = Table(
        title=f"Top {k} Operator Instances by Avg Duration  (timings in μs)",
        show_lines=False,
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column(
        "Node Name",
        min_width=24,
        max_width=OP_TRACING_NODE_NAME_MAX_WIDTH,
        no_wrap=True,
        overflow="ellipsis",
    )
    table.add_column("Op Type", min_width=9, no_wrap=True)
    table.add_column("Avg", justify="right", width=9)
    table.add_column("Total", justify="right", width=10)
    table.add_column("% Tot", justify="right", width=6)
    table.add_column("Cum %", justify="right", width=6)
    table.add_column("p90", justify="right", width=9)

    if level == "detail":
        table.add_column("DRAM(R)", justify="right", width=8)
        table.add_column("VTCM Hit", justify="right", width=8)

    cum_total_us = 0.0
    for i, op in enumerate(top_ops, 1):
        cum_total_us += op.total_us
        pct_total = (op.total_us / sum_all_total * 100.0) if sum_all_total > 0 else 0.0
        cum_pct = (cum_total_us / sum_all_total * 100.0) if sum_all_total > 0 else 0.0

        row = [
            str(i),
            _truncate_node_name(op.node_name),
            op.op_type,
            _format_number(op.avg_us),
            _format_number(op.total_us),
            f"{pct_total:.1f}%",
            f"{cum_pct:.1f}%",
            _format_number(op.p90_us),
        ]

        if level == "detail":
            vtcm_str = (
                f"{op.vtcm_hit_ratio * 100:.1f}%"
                if op.vtcm_hit_ratio is not None
                else "-"
            )
            row.extend([_format_bytes(op.dram_read_bytes), vtcm_str])

        table.add_row(*row)

    console.print(table)
    console.print(_build_op_tracing_summary_line(top_ops, ops_sorted, k))

    # Mode-specific footer
    if level == "basic":
        accel_us = int(sum_all_total)
        console.print(
            f"[dim]HVX threads: 4   Accel execute: {_format_number(accel_us)} μs"
            f"   Samples: {num_samples}[/dim]"
        )
    else:  # detail
        total_dram_r = sum((op.dram_read_bytes or 0) for op in ops_sorted)
        # VTCM read ~= 8x DRAM read approximates typical NPU residency win.
        total_vtcm_r = total_dram_r * 8
        peak_vtcm = 1_843_200  # 1.8 MB
        execute_us = int(sum_all_total)
        inference_us = int(execute_us * 1.009)  # ~1% session overhead
        utilization = 91.3
        console.print(
            f"[dim]HVX threads: 4   Inference: {_format_number(inference_us)} μs"
            f"   Execute: {_format_number(execute_us)} μs"
            f"   Utilization: {utilization}%[/dim]"
        )
        console.print(
            f"[dim]DRAM read total: {_format_bytes(total_dram_r)}"
            f"   VTCM read total: {_format_bytes(total_vtcm_r)}"
            f"   Peak VTCM alloc: {_format_bytes(peak_vtcm)}[/dim]"
        )

    if num_samples == 1:
        console.print()
        console.print(
            "[dim]Note: p90 reflects single-sample data; "
            "increase --iterations for meaningful percentiles.[/dim]"
        )
```

- [ ] **Step 2: Quick smoke-test from REPL (without integrating into demo() yet)**

```bash
uv run python -c "
from rich.console import Console
from docs.design.perf.console_mockup import generate_fake_ops, render_op_tracing
console = Console()
ops = generate_fake_ops(100)
render_op_tracing(console, ops, level='basic', top_k=5, num_samples=100)
"
```

Expected: A correctly formatted table appears with 5 rows, top row is `/resnet/encoder/stage.3/block.0/Conv_token_2  Conv2d  1,270.4  127,037.8  18.6%  18.6%  1,386.4`. Summary line reads `Top 5 instances account for 62.4% of execute time (424,919.6 / 681,386.1 μs accumulated)`. No tracebacks.

- [ ] **Step 3: Verify detail mode renders the two extra columns**

```bash
uv run python -c "
from rich.console import Console
from docs.design.perf.console_mockup import generate_fake_ops, render_op_tracing
console = Console()
ops = generate_fake_ops(100)
render_op_tracing(console, ops, level='detail', top_k=5, num_samples=100)
"
```

Expected: Same table plus `DRAM(R)` (e.g. `18.0 KB`) and `VTCM Hit` (e.g. `94.2%`) columns. Footer shows `DRAM read total: 76.1 KB   VTCM read total: 608.6 KB   Peak VTCM alloc: 1.8 MB`.

- [ ] **Step 4: Commit**

```bash
git add docs/design/perf/console_mockup.py
git commit -m "feat(perf-mockup): add render_op_tracing for top-K instance report"
```

---

## Task 7: Refactor `demo()` to accept arguments

**Files:**
- Modify: `docs/design/perf/console_mockup.py:493-594` (the existing `demo()` function and `if __name__ == "__main__":` block)

- [ ] **Step 1: Change `demo()` signature to accept op-tracing parameters**

Replace the `def demo() -> None:` line (currently at line 493) with:

```python
def demo(
    *,
    op_tracing: str | None = None,
    top_k: int = OP_TRACING_TOP_K_DEFAULT,
    op_tracing_iterations: int = OP_TRACING_NUM_SAMPLES,
) -> None:
    """Render the full --monitor --iterations 1000 output sequence.

    When ``op_tracing`` is ``"basic"`` or ``"detail"``, append a Phase 4
    operator-tracing section after the post-bench summary.
    """
```

- [ ] **Step 2: Append Phase 4 to the body of `demo()` — after the existing `console.print(build_save_footer(...))` line and its trailing `console.print()` (currently lines 589-590)**

Add this new block at the very end of `demo()`, just before the function ends:

```python
    # ── Phase 4 — Op-tracing (only when requested) ───────────────────────────
    if op_tracing is not None:
        ops = generate_fake_ops(num_samples=op_tracing_iterations)
        render_op_tracing(
            console,
            ops,
            level=op_tracing,
            top_k=top_k,
            num_samples=op_tracing_iterations,
        )
        console.print()
```

- [ ] **Step 3: Verify the script still runs without args (no behaviour change for legacy invocation)**

Run: `uv run python docs/design/perf/console_mockup.py`
Expected: Identical 3-phase output to before this task. No Phase 4 appears (since `op_tracing` defaults to None).

- [ ] **Step 4: Commit**

```bash
git add docs/design/perf/console_mockup.py
git commit -m "refactor(perf-mockup): make demo() take op_tracing/top_k/iterations args"
```

---

## Task 8: Add `main()` with argparse and the hard-error precondition

**Files:**
- Modify: `docs/design/perf/console_mockup.py:593-594` (replace the `if __name__ == "__main__":` block)

- [ ] **Step 1: Add `main()` between `demo()` and `if __name__ == "__main__":`**

Insert this function above the existing `if __name__ == "__main__":` line:

```python
# ── CLI entry point ─────────────────────────────────────────────────────────


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="console_mockup.py",
        description=(
            "Render a mockup of `wmk perf` output. "
            "Pass --op-tracing to append the operator tracing section."
        ),
    )
    parser.add_argument(
        "--op-tracing",
        choices=["basic", "detail"],
        default=None,
        help="Enable operator tracing section (basic or detail mode).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=OP_TRACING_TOP_K_DEFAULT,
        help=f"Number of top operator instances to show (default: {OP_TRACING_TOP_K_DEFAULT}).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=OP_TRACING_NUM_SAMPLES,
        help=(
            "Number of fake inference samples per operator instance "
            f"(default: {OP_TRACING_NUM_SAMPLES})."
        ),
    )
    return parser.parse_args(argv)


def _user_passed_top_k(argv: list[str]) -> bool:
    """Detect whether the user explicitly typed --top-k on the CLI.

    Argparse's default value is indistinguishable from an explicit
    ``--top-k 5``, so we inspect ``sys.argv`` directly.
    """
    return any(a == "--top-k" or a.startswith("--top-k=") for a in argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = _parse_args(raw_argv)

    console = Console()

    # Hard-error: --top-k without --op-tracing is meaningless.
    if args.op_tracing is None and _user_passed_top_k(raw_argv):
        console.print(
            "[red]Error:[/red] --top-k requires --op-tracing to be set.",
            highlight=False,
        )
        return 2

    if args.top_k < 1:
        console.print(
            "[red]Error:[/red] --top-k must be >= 1.", highlight=False
        )
        return 2

    if args.iterations < 1:
        console.print(
            "[red]Error:[/red] --iterations must be >= 1.", highlight=False
        )
        return 2

    demo(
        op_tracing=args.op_tracing,
        top_k=args.top_k,
        op_tracing_iterations=args.iterations,
    )
    return 0
```

- [ ] **Step 2: Replace the entry-point block at the bottom of the file**

Replace:

```python
if __name__ == "__main__":
    demo()
```

with:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Verify the no-flag invocation still produces the original 3-phase output**

Run: `uv run python docs/design/perf/console_mockup.py`
Expected: Original 3-phase output (no Phase 4).

- [ ] **Step 4: Verify the hard-error case**

Run: `uv run python docs/design/perf/console_mockup.py --top-k 7; echo "EXIT=$?"`
Expected:
```
Error: --top-k requires --op-tracing to be set.
EXIT=2
```

- [ ] **Step 5: Verify basic-mode op-tracing renders end-to-end**

Run: `uv run python docs/design/perf/console_mockup.py --op-tracing basic`
Expected: Original 3 phases run, then Phase 4 appears with:
- Rule: `──── Operator Tracing (basic, 100 samples) ────`
- Table with 8 columns and 5 rows
- Summary line: `Top 5 instances account for 62.4% of execute time (...)`
- Footer: `HVX threads: 4   Accel execute: 681,386 μs   Samples: 100`

- [ ] **Step 6: Verify detail-mode adds DRAM(R) and VTCM Hit columns**

Run: `uv run python docs/design/perf/console_mockup.py --op-tracing detail`
Expected: Same as Step 5 plus two additional columns and a second footer line with DRAM/VTCM totals.

- [ ] **Step 7: Verify --top-k override**

Run: `uv run python docs/design/perf/console_mockup.py --op-tracing basic --top-k 10`
Expected: 10 rows; final Cum % is 83.6%; summary reads `Top 10 instances account for 83.6%`.

- [ ] **Step 8: Verify --top-k > N falls back to all-N phrasing**

Run: `uv run python docs/design/perf/console_mockup.py --op-tracing basic --top-k 999`
Expected: 20 rows (all templates); summary reads `All 20 instances account for 100.0% of execute time`.

- [ ] **Step 9: Verify --iterations 1 triggers the single-sample footer note**

Run: `uv run python docs/design/perf/console_mockup.py --op-tracing basic --iterations 1`
Expected: Table renders with `Avg == p90` for each row, and the dim italic note `Note: p90 reflects single-sample data; increase --iterations for meaningful percentiles.` appears below the footer.

- [ ] **Step 10: Commit**

```bash
git add docs/design/perf/console_mockup.py
git commit -m "feat(perf-mockup): add --op-tracing/--top-k/--iterations CLI with hard-error precondition"
```

---

## Task 9: Update the module docstring to document the new flags

**Files:**
- Modify: `docs/design/perf/console_mockup.py:6-83` (the module docstring)

- [ ] **Step 1: Replace the existing `Run: uv run python docs/design/perf/console_mockup.py` line (around line 15) with a richer set of examples**

Find:

```python
Run: uv run python docs/design/perf/console_mockup.py
```

Replace with:

```python
Run examples:

    # Original 3-phase animation (no op-tracing):
    uv run python docs/design/perf/console_mockup.py

    # 3-phase animation + Phase 4 operator tracing:
    uv run python docs/design/perf/console_mockup.py --op-tracing basic
    uv run python docs/design/perf/console_mockup.py --op-tracing detail
    uv run python docs/design/perf/console_mockup.py --op-tracing basic --top-k 10
    uv run python docs/design/perf/console_mockup.py --op-tracing basic --iterations 1

    # --top-k without --op-tracing is rejected with exit 2.
```

- [ ] **Step 2: Append a "Contract D — Op-tracing" section to the docstring, just before the closing `"""` (around line 82)**

Insert before the closing triple-quote of the module docstring:

```text
Contract D — Op-tracing per-instance schema:

    {
        "node_name": str,                # full ONNX node path
        "op_type": str,                  # e.g. "Conv2d", "MatMul", "Add"
        "sample_durations_us": list[float],   # per-iteration durations
        "dram_read_bytes": int | None,
        "vtcm_hit_ratio": float | None,  # 0..1
        # derived:
        "avg_us":   sum(samples) / len(samples)
        "total_us": sum(samples)
        "p90_us":   inclusive 90th percentile of samples (== avg when n==1)
    }

The op-tracing report is gated on --op-tracing. Sort key is avg_us
descending, with node_name as a deterministic tiebreaker. The "Cum %"
column is a running sum of per-instance percent-of-total down the table.
```

- [ ] **Step 3: Verify the script still runs**

Run: `uv run python docs/design/perf/console_mockup.py --op-tracing basic`
Expected: Same Phase-4 output as Task 8 Step 5.

- [ ] **Step 4: Commit**

```bash
git add docs/design/perf/console_mockup.py
git commit -m "docs(perf-mockup): document --op-tracing flags and per-instance contract"
```

---

## Task 10: Final five-scenario verification pass

This task is purely visual verification — no code changes. Run every scenario in a fresh terminal and confirm each matches its expected output. If any scenario fails, return to the relevant earlier task before proceeding.

- [ ] **Scenario 1: No flags — legacy 3-phase mockup unchanged**

Run: `uv run python docs/design/perf/console_mockup.py`
Expected: Pre-bench header → Live monitor (3-second animation, NPU/CPU/GPU chart, progress bar) → Post-bench summary (Latency table, Throughput, Hardware block, Save footer). **No** Phase 4.

- [ ] **Scenario 2: `--op-tracing basic`**

Run: `uv run python docs/design/perf/console_mockup.py --op-tracing basic`
Expected: All of Scenario 1 plus, after the save footer:
- Rule line: `──── Operator Tracing (basic, 100 samples) ────`
- Table titled `Top 5 Operator Instances by Avg Duration  (timings in μs)`
- Row 1: `1   /resnet/encoder/stage.3/block.0/Conv_…   Conv2d   1,270.4   127,037.8   18.6%   18.6%   1,386.4`
- Row 5: `5   /resnet/encoder/stage.3/block.2/Add_2     Add       426.1    42,607.8    6.3%   62.4%     463.8`
- Summary: `Top 5 instances account for 62.4% of execute time (424,919.6 / 681,386.1 μs accumulated)`
- Footer: `HVX threads: 4   Accel execute: 681,386 μs   Samples: 100`

- [ ] **Scenario 3: `--op-tracing detail`**

Run: `uv run python docs/design/perf/console_mockup.py --op-tracing detail`
Expected: Same as Scenario 2 plus two extra columns:
- Row 1 ends with `   18.0 KB   94.2%`
- Row 5 ends with `   4.0 KB   99.0%`
- Footer is two dim lines: `HVX threads: 4   Inference: 687,518 μs   Execute: 681,386 μs   Utilization: 91.3%` and `DRAM read total: 76.1 KB   VTCM read total: 608.6 KB   Peak VTCM alloc: 1.8 MB`

- [ ] **Scenario 4: `--op-tracing basic --top-k 10`**

Run: `uv run python docs/design/perf/console_mockup.py --op-tracing basic --top-k 10`
Expected: Title shows `Top 10 ...`, table has 10 rows, final Cum % is 83.6%. Summary line says `Top 10 instances account for 83.6% of execute time (569,833.8 / 681,386.1 μs accumulated)`.

- [ ] **Scenario 5: Hard-error — `--top-k` without `--op-tracing`**

Run: `uv run python docs/design/perf/console_mockup.py --top-k 7; echo "EXIT=$?"`
Expected:
```
Error: --top-k requires --op-tracing to be set.
EXIT=2
```
No 3-phase output. No Phase 4. Process exits with code 2 before any rendering.

- [ ] **Scenario 6: Single-sample degenerate case — `--iterations 1`**

Run: `uv run python docs/design/perf/console_mockup.py --op-tracing basic --iterations 1`
Expected: Table renders with `Avg == p90` for every row. Below the footer, a dim italic note: `Note: p90 reflects single-sample data; increase --iterations for meaningful percentiles.`

- [ ] **Final commit (only if you fixed anything during verification)**

If all six scenarios pass cleanly, no commit is needed for Task 10.

If you fixed something, commit with:

```bash
git add docs/design/perf/console_mockup.py
git commit -m "fix(perf-mockup): address verification finding in op-tracing section"
```

---

## Done

After Task 10 passes all six scenarios, the mockup supports `--op-tracing` and is ready for review against the production `wmk perf` implementation that will follow.

The visual output of this mockup is the canonical specification: any divergence between the implementation in `modelkit/optracing/report.py` and the strings produced by this script is a bug in the implementation, not the mockup.

## Out of Scope

The following are **not** addressed by this plan and should be raised separately if needed:

- **Real op-tracing integration with the live `wmk perf` command** — this plan only covers the design mockup. The implementation work in `modelkit/optracing/` is a separate effort that consumes this mockup as its UX spec.
- **Pytest-based tests for the mockup** — visual inspection is sufficient for a design mockup. If tests are wanted, that's a follow-up.
- **JSON serialization of the op-tracing block** — Contract A (`<model>_perf.json`) does not yet include op-tracing data. If/when added, schema design is a separate plan.
- **Op-type aggregation report** — explicitly rejected during design ("Top-K op INSTANCES"). Do not add op-type rollup tables.
- **Additional percentiles (p50/p95/p99)** — explicitly trimmed to "avg + p90 only".
- **Multiple-QHAS-runs detail-mode strategy** — the underlying CSV captures per-sample data in both modes, so a single QHAS run is sufficient. No re-architecture of the profiler run loop.
