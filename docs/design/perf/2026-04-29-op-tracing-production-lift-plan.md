# Op-Tracing Production Lift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the team-reviewed perf console mockup (`docs/design/perf/console_mockup.py` + `2026-04-28-console-mockup-design.md` v2.0) into production code paths under `src/winml/modelkit/`.

**Architecture:** The data pipeline (`OpTraceResult`, parsers, perf.py CLI wiring) is already in place. The lift is two-pronged: (1) extend `OperatorMetrics` to retain per-sample timings so p90 can be derived, (2) rewrite the render layer (`report.py`, `_live_chart.py`) and add pre-bench / save-to-footer helpers per the mockup's Contract D and approved UX.

**Tech Stack:** Python 3.11+, Rich (Table/Panel/Console), pytest, dataclasses (with derived `@property`).

---

## Pre-flight reading (before Task 1)

The implementer should skim these in order before starting:

| File | Lines | Purpose |
|---|---|---|
| `docs/design/perf/2026-04-28-console-mockup-design.md` | full | Spec, 21 ACs, contracts |
| `docs/design/perf/console_mockup.py` | 383–477 | Canonical `render_op_tracing` (basic + detail tables) |
| `docs/design/perf/console_mockup.py` | 130–160 | Module constants (widths, window seconds, top-K default) |
| `src/winml/modelkit/session/monitor/op_metrics.py` | full (131) | Current `OperatorMetrics` / `OpTraceResult` |
| `src/winml/modelkit/session/monitor/report.py` | 92–200 | Current basic + detail render functions |
| `src/winml/modelkit/session/monitor/qnn/csv_parser.py` | 113–157, 177–end | `_extract_samples` and `_aggregate_operators` |
| `src/winml/modelkit/commands/_live_chart.py` | 1–60 | `LiveMonitorDisplay` constants |
| `src/winml/modelkit/commands/perf.py` | 1469–1521 | Existing op-trace post-benchmark hook |

**What is already wired (do NOT redo):**
- `--op-tracing {basic,detail}` / `--top-k` / `--iterations` CLI flags
- Smart `--iterations=1` default when `--op-tracing` is set without explicit value
- Hard-fail at parse time for ONNX-file path + `--op-tracing`
- Routing through monitored path when `op_tracing OR monitor`
- `display_op_trace_report` and `write_op_trace_json` exist
- `OpTraceResult` JSON serialization shape

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `src/winml/modelkit/session/monitor/op_metrics.py` | Modify | Add `samples_us` list + derived `@property` (avg/p90/total/count) on `OperatorMetrics`. Keep `duration_us` field for serialization back-compat. |
| `src/winml/modelkit/session/monitor/qnn/csv_parser.py` | Modify | Retain per-sample timings in `_aggregate_operators` instead of collapsing to a single avg. |
| `src/winml/modelkit/session/monitor/qnn/qhas_parser.py` | Modify | Same per-sample retention if QHAS parser also produces `OperatorMetrics`. |
| `src/winml/modelkit/session/monitor/report.py` | Modify | New `_truncate_node_name`. Rewrite `_display_basic_report` (4 cols, width-locked at 120). Rewrite `_display_detail_report` (10 cols, width-locked at 120). Header rename "Op-Level Profiling" → "Op-Tracing". |
| `src/winml/modelkit/commands/_pre_bench.py` | Create | `print_pre_bench_block(...)` Rich helper: 3 panels (Model identity / Surface / Device). |
| `src/winml/modelkit/commands/perf.py` | Modify | Wire pre-bench block before benchmark loop. Wire save-to footer (trace JSON + profiling CSV) after `display_op_trace_report`. |
| `src/winml/modelkit/commands/_live_chart.py` | Modify | `_CHART_WINDOW_SECONDS = 15.0` (was 10.0). Default `chart_width=120` (was 80). |
| `tests/session/monitor/test_op_metrics_samples.py` | Create | Unit tests for `OperatorMetrics.samples_us` derived properties. |
| `tests/session/monitor/qnn/test_csv_parser_samples.py` | Create | Test parser retains per-sample timings. |
| `tests/session/monitor/test_report_basic.py` | Create | Capture-and-assert tests for new basic-mode render. |
| `tests/session/monitor/test_report_detail.py` | Create | Capture-and-assert tests for new detail-mode render. |
| `tests/commands/test_pre_bench.py` | Create | Tests for pre-bench identity block. |
| `tests/commands/test_perf_save_footer.py` | Create | Tests for save-to footer output. |
| `tests/commands/test_live_chart_constants.py` | Create | Trivial constant-pinning tests for chart window/width. |

---

## Task 1: Add per-sample retention to `OperatorMetrics`

**Files:**
- Modify: `src/winml/modelkit/session/monitor/op_metrics.py:35-71`
- Create: `tests/session/monitor/test_op_metrics_samples.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/session/monitor/test_op_metrics_samples.py
"""Per-sample retention + derived stats on OperatorMetrics."""
import pytest

from winml.modelkit.session.monitor.op_metrics import OperatorMetrics


def test_samples_us_default_empty():
    op = OperatorMetrics(name="Conv2d", op_path="/layer1/conv/Conv")
    assert op.samples_us == []
    assert op.sample_count == 0


def test_avg_us_from_samples():
    op = OperatorMetrics(
        name="Conv2d", op_path="/x", samples_us=[100.0, 200.0, 300.0]
    )
    assert op.avg_us == pytest.approx(200.0)


def test_total_us_from_samples():
    op = OperatorMetrics(
        name="Conv2d", op_path="/x", samples_us=[10.0, 20.0, 30.0]
    )
    assert op.total_us == pytest.approx(60.0)


def test_p90_us_inclusive_method():
    op = OperatorMetrics(
        name="Conv2d", op_path="/x",
        samples_us=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    # Inclusive p90 of 1..10 is 9.1 (statistics.quantiles n=10 method='inclusive' index 8)
    assert op.p90_us == pytest.approx(9.1, abs=0.01)


def test_p90_single_sample():
    op = OperatorMetrics(name="Conv2d", op_path="/x", samples_us=[42.0])
    assert op.p90_us == pytest.approx(42.0)


def test_p90_empty_samples_returns_zero():
    op = OperatorMetrics(name="Conv2d", op_path="/x", samples_us=[])
    assert op.p90_us == 0.0


def test_duration_us_back_compat_when_samples_present():
    """duration_us should mirror avg_us when samples_us is populated, for back-compat."""
    op = OperatorMetrics(
        name="Conv2d", op_path="/x",
        duration_us=200.0,  # explicitly set, mirrors avg
        samples_us=[100.0, 200.0, 300.0],
    )
    # to_dict still serializes duration_us; samples_us is additive
    d = op.to_dict()
    assert d["duration_us"] == 200.0
    assert d["samples_us"] == [100.0, 200.0, 300.0]
```

- [ ] **Step 2: Run tests — confirm they fail**

```
uv run pytest tests/session/monitor/test_op_metrics_samples.py -v
```

Expected: 7 failures with `AttributeError: 'OperatorMetrics' object has no attribute 'samples_us'` (and similar for the derived properties).

- [ ] **Step 3: Extend `OperatorMetrics`**

Edit `src/winml/modelkit/session/monitor/op_metrics.py`. Add the import at the top of the file (next to existing imports):

```python
import statistics as _stats
```

Add the new field after `dims: list[int] | None = None` (around line 66, before `to_dict`):

```python
    # Per-sample timings retained for downstream stats (p90, total, count).
    # Empty when source parser only produced an aggregated avg.
    samples_us: list[float] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        return len(self.samples_us)

    @property
    def avg_us(self) -> float:
        return sum(self.samples_us) / len(self.samples_us) if self.samples_us else 0.0

    @property
    def total_us(self) -> float:
        return sum(self.samples_us)

    @property
    def p90_us(self) -> float:
        n = len(self.samples_us)
        if n == 0:
            return 0.0
        if n == 1:
            return self.samples_us[0]
        # statistics.quantiles with n=10 method="inclusive" gives 9 cut points;
        # index 8 is the 90th percentile.
        return _stats.quantiles(self.samples_us, n=10, method="inclusive")[8]
```

Note: `to_dict` already uses `asdict()` which picks up dataclass fields automatically; `samples_us` will serialize without further changes. `@property` methods are not in the dataclass fields and stay out of `to_dict` (which is what we want — they are derived).

- [ ] **Step 4: Run tests — confirm they pass**

```
uv run pytest tests/session/monitor/test_op_metrics_samples.py -v
```

Expected: 7 passing.

- [ ] **Step 5: Run lint**

```
uv run ruff check src/winml/modelkit/session/monitor/op_metrics.py tests/session/monitor/test_op_metrics_samples.py --fix
```

- [ ] **Step 6: Commit**

```bash
git add src/winml/modelkit/session/monitor/op_metrics.py tests/session/monitor/test_op_metrics_samples.py
git commit -m "feat(op-metrics): retain per-sample timings + derive p90/total/count

Adds samples_us: list[float] field with derived @property avg_us,
p90_us, total_us, sample_count. Existing duration_us field stays
for serialization back-compat. Inclusive-method quantile matches
the mockup contract D.

Constraint: render layer needs p90 per-instance, parsers
already emit per-sample data
Confidence: high
Scope-risk: narrow"
```

---

## Task 2: Retain per-sample timings in QNN CSV parser

**Files:**
- Modify: `src/winml/modelkit/session/monitor/qnn/csv_parser.py` (`_aggregate_operators`)
- Create: `tests/session/monitor/qnn/test_csv_parser_samples.py`

- [ ] **Step 1: Read the current `_aggregate_operators` impl**

Open `src/winml/modelkit/session/monitor/qnn/csv_parser.py` lines 177–end. Note: it walks `samples: list[list[dict]]` and currently collapses to `OperatorMetrics(duration_us=avg)`. The change is to ALSO populate `samples_us` with the per-sample list for that op.

- [ ] **Step 2: Write the failing test**

```python
# tests/session/monitor/qnn/test_csv_parser_samples.py
"""QNN CSV parser must retain per-sample timings for each operator."""
from winml.modelkit.session.monitor.qnn.csv_parser import _aggregate_operators


def test_per_sample_retention():
    # Two samples, three ops each with same op_path → samples_us has length 2
    sample_1 = [
        {"op_path": "/conv1/Conv", "name": "Conv2d", "duration_us": 100.0},
        {"op_path": "/relu1/Relu", "name": "Relu", "duration_us": 5.0},
    ]
    sample_2 = [
        {"op_path": "/conv1/Conv", "name": "Conv2d", "duration_us": 110.0},
        {"op_path": "/relu1/Relu", "name": "Relu", "duration_us": 6.0},
    ]
    ops = _aggregate_operators([sample_1, sample_2])
    by_path = {op.op_path: op for op in ops}

    assert by_path["/conv1/Conv"].samples_us == [100.0, 110.0]
    assert by_path["/relu1/Relu"].samples_us == [5.0, 6.0]


def test_per_sample_back_compat_avg():
    """duration_us still equals avg across samples (back-compat)."""
    sample_1 = [{"op_path": "/x", "name": "X", "duration_us": 100.0}]
    sample_2 = [{"op_path": "/x", "name": "X", "duration_us": 300.0}]
    ops = _aggregate_operators([sample_1, sample_2])
    assert ops[0].duration_us == 200.0
    assert ops[0].samples_us == [100.0, 300.0]
```

Note: if the actual CSV row keys differ from `op_path` / `duration_us`, adjust the test fixtures to match the parser's expected dict keys. The implementer should check the parser source, then either adjust the test or the parser to make them consistent.

- [ ] **Step 3: Run test — confirm it fails**

```
uv run pytest tests/session/monitor/qnn/test_csv_parser_samples.py -v
```

Expected: assertion failure on `samples_us` (currently `[]` because parser only sets `duration_us`).

- [ ] **Step 4: Modify `_aggregate_operators`**

In `_aggregate_operators`, when accumulating per-op data across samples, build a `samples_us: list[float]` per op and pass it into `OperatorMetrics(...)`. Keep the avg-into-`duration_us` logic intact (back-compat).

The exact edit depends on the current loop; the implementer reads lines 177–end and adjusts. The contract: each returned `OperatorMetrics` must have `samples_us` populated with one float per sample where that op appeared.

- [ ] **Step 5: Run test — confirm it passes**

```
uv run pytest tests/session/monitor/qnn/test_csv_parser_samples.py -v
```

- [ ] **Step 6: Run all monitor tests to check for regressions**

```
uv run pytest tests/session/monitor/ -v
```

Expected: all green (Task 1 + Task 2 tests + pre-existing tests).

- [ ] **Step 7: Repeat the same change for QHAS parser if applicable**

Check `src/winml/modelkit/session/monitor/qnn/qhas_parser.py`. If it also produces `OperatorMetrics`, add equivalent per-sample retention. Add a parallel test file `tests/session/monitor/qnn/test_qhas_parser_samples.py` if so. If QHAS only enriches detail-mode metadata and does not aggregate timings, no change needed — note this in the commit message.

- [ ] **Step 8: Lint + commit**

```bash
uv run ruff check src/winml/modelkit/session/monitor/qnn/ tests/session/monitor/qnn/ --fix
git add src/winml/modelkit/session/monitor/qnn/ tests/session/monitor/qnn/
git commit -m "feat(qnn-parser): populate OperatorMetrics.samples_us from per-sample CSV rows

Constraint: render layer needs p90 per op (Contract D)
Rejected: compute p90 inline in parser | render layer is the right consumer
Confidence: high
Scope-risk: narrow"
```

---

## Task 3: Add `_truncate_node_name` helper to `report.py`

**Files:**
- Modify: `src/winml/modelkit/session/monitor/report.py`
- Create: `tests/session/monitor/test_truncate_node_name.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/session/monitor/test_truncate_node_name.py
"""Left-ellipsis truncation matches mockup spec."""
from winml.modelkit.session.monitor.report import _truncate_node_name


def test_under_max_width_unchanged():
    assert _truncate_node_name("/short", max_width=80) == "/short"


def test_exact_max_width_unchanged():
    name = "x" * 80
    assert _truncate_node_name(name, max_width=80) == name


def test_over_max_width_left_ellipsis():
    name = "/very/long/path/" + "x" * 200
    out = _truncate_node_name(name, max_width=80)
    assert len(out) == 80
    assert out.startswith("…")          # leading ellipsis char
    assert out.endswith("x" * 79)            # right side preserved


def test_max_width_one():
    assert _truncate_node_name("anything", max_width=1) == "…"


def test_max_width_zero_returns_empty():
    assert _truncate_node_name("anything", max_width=0) == ""
```

- [ ] **Step 2: Run test — confirm it fails**

```
uv run pytest tests/session/monitor/test_truncate_node_name.py -v
```

Expected: `ImportError: cannot import name '_truncate_node_name'`.

- [ ] **Step 3: Add the helper**

Edit `src/winml/modelkit/session/monitor/report.py`. Add after the existing internal-helpers comment block (after `_format_number`, around line 90):

```python
def _truncate_node_name(name: str, max_width: int = 80) -> str:
    """Left-truncate a node path with a leading ellipsis.

    Preserves the right side because the leaf operator name (the
    differentiator) lives at the tail of the path.
    """
    if max_width <= 0:
        return ""
    if len(name) <= max_width:
        return name
    if max_width == 1:
        return "…"
    return "…" + name[-(max_width - 1):]
```

- [ ] **Step 4: Run test — confirm it passes**

```
uv run pytest tests/session/monitor/test_truncate_node_name.py -v
```

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/winml/modelkit/session/monitor/report.py tests/session/monitor/test_truncate_node_name.py --fix
git add src/winml/modelkit/session/monitor/report.py tests/session/monitor/test_truncate_node_name.py
git commit -m "feat(report): add _truncate_node_name with left-ellipsis

Mirrors mockup helper. Right side is preserved because the
leaf op name (the differentiator) lives at the path tail.

Confidence: high
Scope-risk: narrow"
```

---

## Task 4: Rewrite `_display_basic_report` to mockup spec

**Files:**
- Modify: `src/winml/modelkit/session/monitor/report.py:92-132` (and the public `display_op_trace_report` doctring if needed)
- Create: `tests/session/monitor/test_report_basic.py`

**Mockup spec (basic mode, 4 columns, width-locked at 120):**

| # | Column | justify | width spec |
|---|---|---|---|
| 1 | Node | left | `min_width=80, max_width=80, no_wrap=True, overflow="ellipsis"` (we feed pre-truncated text so overflow is unused) |
| 2 | Type | left | `width=12, no_wrap=True` |
| 3 | p90 | right | `width=9` |
| 4 | % Tot | right | `width=6` |

Header rule text: `"Op-Tracing (basic)"` (was `"Op-Level Profiling (basic)"`).

- [ ] **Step 1: Write the failing test**

```python
# tests/session/monitor/test_report_basic.py
"""Basic-mode rendering matches mockup spec."""
from io import StringIO

from rich.console import Console

from winml.modelkit.session.monitor.op_metrics import OperatorMetrics, OpTraceResult
from winml.modelkit.session.monitor.report import display_op_trace_report


def _render(result: OpTraceResult, top_n: int = 5) -> str:
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False, record=True)
    display_op_trace_report(result, console=console, top_n=top_n)
    return console.export_text()


def _make_result(num_ops: int = 3) -> OpTraceResult:
    ops = [
        OperatorMetrics(
            name=f"OpType{i}",
            op_path=f"/path/to/op_{i}/Op",
            duration_us=100.0 - i * 10,
            percent_of_total=30.0 - i * 5,
            samples_us=[90.0 + i, 100.0 - i, 110.0 - i * 2],
        )
        for i in range(num_ops)
    ]
    return OpTraceResult(
        model="convnext-base", device="NPU", tracing_level="basic",
        operators=ops, num_samples=3,
    )


def test_basic_header_renamed():
    out = _render(_make_result())
    assert "Op-Tracing (basic)" in out
    assert "Op-Level Profiling" not in out


def test_basic_columns():
    out = _render(_make_result())
    # header row contains the four column names in order
    header_line = next(line for line in out.splitlines() if "Node" in line and "Type" in line)
    assert header_line.index("Node") < header_line.index("Type")
    assert header_line.index("Type") < header_line.index("p90")
    assert header_line.index("p90") < header_line.index("% Tot")


def test_basic_no_rank_column():
    """Mockup drops the # rank column in basic mode."""
    out = _render(_make_result())
    header_line = next(line for line in out.splitlines() if "Node" in line and "Type" in line)
    # The leading column is Node (no leading "#" digit before it).
    # Allow leading box-drawing / whitespace, then "Node".
    assert header_line.lstrip("│ ").startswith("Node")


def test_basic_long_node_path_left_truncated():
    long_path = "/very/deep" + "/segment" * 30
    op = OperatorMetrics(
        name="Conv2d", op_path=long_path,
        duration_us=100.0, percent_of_total=50.0, samples_us=[100.0],
    )
    result = OpTraceResult(
        model="m", device="NPU", tracing_level="basic",
        operators=[op], num_samples=1,
    )
    out = _render(result)
    # The truncated node line should contain the leading ellipsis and the
    # tail of the path (the rightmost characters preserved).
    assert "…" in out
    assert long_path[-20:] in out


def test_basic_p90_rendered_when_samples_present():
    out = _render(_make_result())
    # Should NOT show "—" for p90 since samples_us is populated
    p90_dash_count = out.count("—")  # em dash
    assert p90_dash_count == 0


def test_basic_p90_em_dash_when_no_samples():
    op = OperatorMetrics(
        name="Conv2d", op_path="/x",
        duration_us=50.0, percent_of_total=10.0, samples_us=[],
    )
    result = OpTraceResult(
        model="m", device="NPU", tracing_level="basic",
        operators=[op], num_samples=0,
    )
    out = _render(result)
    assert "—" in out
```

- [ ] **Step 2: Run tests — confirm they fail**

```
uv run pytest tests/session/monitor/test_report_basic.py -v
```

Expected: failures on header text (still says "Op-Level Profiling"), column order (current has `# / Operator / Avg Cyc / % Tot`), missing p90 column, no truncation applied.

- [ ] **Step 3: Replace `_display_basic_report`**

Replace the body of `_display_basic_report` (lines ~92–132) with:

```python
def _display_basic_report(result: OpTraceResult, console: Console, top_n: int) -> None:
    """Render a basic-mode op-tracing report (4 columns, width-locked at 120)."""
    console.print()
    console.rule("[bold]Op-Tracing (basic)[/bold]")

    # Summary line (unchanged)
    parts: list[str] = []
    hvx = result.summary.get("hvx_threads")
    if hvx is not None:
        parts.append(f"HVX Threads: {hvx}")
    accel = result.summary.get("accel_execute_us")
    if accel is not None:
        parts.append(f"Accel Execute: {_format_number(accel)} us")
    if result.num_samples:
        parts.append(f"Samples: {result.num_samples}")
    if parts:
        console.print(" | ".join(parts))
    console.print()

    ops = result.operators[:top_n]
    if not ops:
        console.print("[dim]No operator data available.[/dim]")
        return

    table = Table(show_lines=False)
    table.add_column(
        "Node", min_width=80, max_width=80, no_wrap=True, overflow="ellipsis"
    )
    table.add_column("Type", width=12, no_wrap=True)
    table.add_column("p90", justify="right", width=9)
    table.add_column("% Tot", justify="right", width=6)

    for op in ops:
        node_str = _truncate_node_name(op.op_path, max_width=80)
        p90_str = (
            f"{op.p90_us:,.1f}" if op.samples_us else "—"
        )
        table.add_row(
            node_str,
            op.name,
            p90_str,
            f"{op.percent_of_total:.1f}%",
        )

    console.print(table)
```

> **Note:** Unit is announced in table context (header rule + summary line),
> not per-cell. The `width=9` budget cannot hold `'1,234.5 us'` (10 chars) —
> appending a `' us'` suffix to the p90 cell causes Rich to vertically wrap
> any kilo-microsecond value (≥ 1000 µs is typical for real NPU traces).

- [ ] **Step 4: Run tests — confirm they pass**

```
uv run pytest tests/session/monitor/test_report_basic.py -v
```

- [ ] **Step 5: Run all report tests for regressions**

```
uv run pytest tests/session/monitor/ -v
```

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff check src/winml/modelkit/session/monitor/report.py tests/session/monitor/test_report_basic.py --fix
git add src/winml/modelkit/session/monitor/report.py tests/session/monitor/test_report_basic.py
git commit -m "feat(report): rewrite basic-mode render per mockup spec

Drops # rank column. New columns: Node / Type / p90 / % Tot.
Width-locked at 120 with Node min=max=80 and left-ellipsis on
overflow. Header rule renamed Op-Level Profiling → Op-Tracing.
Falls back to em-dash when samples_us is empty.

Constraint: total table width must be 120 cols
Confidence: high
Scope-risk: moderate (visible UX change)"
```

---

## Task 5: Rewrite `_display_detail_report` to 10-column mockup spec

**Files:**
- Modify: `src/winml/modelkit/session/monitor/report.py:135-200`
- Create: `tests/session/monitor/test_report_detail.py`

**Mockup spec (detail mode, 10 columns):**

| # | Column | justify | width spec |
|---|---|---|---|
| 1 | # | right | `style="dim", width=3` |
| 2 | Node | left | `width=80, no_wrap=True, overflow="ellipsis"` (matches basic; check that totals fit 120 with 10 cols + separators — see step 1 below) |
| 3 | Type | left | `min_width=9, no_wrap=True` |
| 4 | Avg | right | `width=9` |
| 5 | Total | right | `width=10` |
| 6 | % Tot | right | `width=6` |
| 7 | Cum % | right | `width=6` |
| 8 | p90 | right | `width=9` |
| 9 | DRAM(R) | right | `width=8` |
| 10 | VTCM Hit | right | `width=8` |

Note on widths: 10 cols at the listed widths sum to 88 + 9 separators ≈ 97 cols; that leaves Node room to shrink without violating the 120-col envelope. The implementer should re-check the actual mockup file (`docs/design/perf/console_mockup.py:448-465`) and copy widths verbatim. If the mockup uses a smaller Node width for detail mode, follow it.

- [ ] **Step 1: Read the canonical detail-mode column specs**

Open `docs/design/perf/console_mockup.py` lines 448–465. Copy the exact `add_column` widths into your impl. The plan's table above is a guide; the mockup is the source of truth.

- [ ] **Step 2: Write the failing test**

```python
# tests/session/monitor/test_report_detail.py
"""Detail-mode rendering matches mockup spec."""
from io import StringIO

from rich.console import Console

from winml.modelkit.session.monitor.op_metrics import OperatorMetrics, OpTraceResult
from winml.modelkit.session.monitor.report import display_op_trace_report


def _render(result: OpTraceResult, top_n: int = 5) -> str:
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False, record=True)
    display_op_trace_report(result, console=console, top_n=top_n)
    return console.export_text()


def _make_detail_result() -> OpTraceResult:
    ops = [
        OperatorMetrics(
            name=f"OpType{i}",
            op_path=f"/path/to/op_{i}/Op",
            duration_us=100.0 - i * 10,
            percent_of_total=30.0 - i * 5,
            dram_read_bytes=1024 * (i + 1),
            vtcm_hit_ratio=0.85 - i * 0.05,
            samples_us=[90.0 + i, 100.0 - i, 110.0 - i * 2],
        )
        for i in range(3)
    ]
    return OpTraceResult(
        model="convnext-base", device="NPU", tracing_level="detail",
        operators=ops, num_samples=3,
    )


def test_detail_header_renamed():
    out = _render(_make_detail_result())
    assert "Op-Tracing (detail)" in out
    assert "Op-Level Profiling" not in out


def test_detail_ten_columns_present():
    out = _render(_make_detail_result())
    expected = ["#", "Node", "Type", "Avg", "Total", "% Tot", "Cum %", "p90", "DRAM(R)", "VTCM Hit"]
    header_line = next(line for line in out.splitlines() if all(c in line for c in expected))
    # If we get here, all 10 column headers are present in a single header row
    assert header_line is not None


def test_detail_cumulative_percent_monotonic():
    """Cum % column should be monotonically non-decreasing across rows."""
    out = _render(_make_detail_result())
    # Extract Cum % values: each is "X.X%"
    # Sort ops are sorted by duration desc (or % desc) — the cum % should grow.
    import re
    pcts = [float(m) for m in re.findall(r"(\d+\.\d)%", out)]
    # We have 3 ops, each row contributes both % Tot and Cum %.
    # Cum % values are at indices 1, 3, 5 (or however the renderer interleaves).
    # Simpler check: at least one cum% should be > the first % Tot value.
    assert any(p > pcts[0] for p in pcts[1:])
```

- [ ] **Step 3: Run test — confirm it fails**

```
uv run pytest tests/session/monitor/test_report_detail.py -v
```

- [ ] **Step 4: Replace `_display_detail_report`**

Replace the body of `_display_detail_report` with a 10-column impl following the mockup. Include cumulative-percent computation:

```python
def _display_detail_report(result: OpTraceResult, console: Console, top_n: int) -> None:
    """Render a detail-mode op-tracing report (10 columns, width-locked)."""
    backend_suffix = ""
    if result.tracing_backend:
        backend_suffix = f" -- {result.tracing_backend}"
    console.print()
    console.rule(f"[bold]Op-Tracing (detail){backend_suffix}[/bold]")

    # Summary lines (unchanged from prior impl)
    summary = result.summary
    line1_parts: list[str] = []
    inf_us = summary.get("inference_us")
    if inf_us is not None:
        line1_parts.append(f"Inference: {_format_number(inf_us)} us")
    exe_us = summary.get("execute_us")
    if exe_us is not None:
        line1_parts.append(f"Execute: {_format_number(exe_us)} us")
    util = summary.get("utilization_pct")
    if util is not None:
        line1_parts.append(f"Utilization: {util}%")
    if line1_parts:
        console.print(" | ".join(line1_parts))

    line2_parts: list[str] = []
    dram_r = summary.get("dram_read_bytes")
    dram_w = summary.get("dram_write_bytes")
    if dram_r is not None or dram_w is not None:
        dr = _format_bytes(dram_r)
        dw = _format_bytes(dram_w)
        line2_parts.append(f"DRAM: Read {dr} / Write {dw}")
    vtcm_peak = summary.get("vtcm_peak_bytes")
    if vtcm_peak is not None:
        line2_parts.append(f"VTCM: Peak {_format_bytes(vtcm_peak)}")
    if line2_parts:
        console.print(" | ".join(line2_parts))
    console.print()

    ops = result.operators[:top_n]
    if not ops:
        console.print("[dim]No operator data available.[/dim]")
        return

    table = Table(show_lines=False)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Node", width=80, no_wrap=True, overflow="ellipsis")  # adjust per mockup
    table.add_column("Type", min_width=9, no_wrap=True)
    table.add_column("Avg", justify="right", width=9)
    table.add_column("Total", justify="right", width=10)
    table.add_column("% Tot", justify="right", width=6)
    table.add_column("Cum %", justify="right", width=6)
    table.add_column("p90", justify="right", width=9)
    table.add_column("DRAM(R)", justify="right", width=8)
    table.add_column("VTCM Hit", justify="right", width=8)

    cum = 0.0
    for i, op in enumerate(ops, 1):
        cum += op.percent_of_total
        node_str = _truncate_node_name(op.op_path, max_width=80)
        avg_str = f"{op.avg_us:,.1f}" if op.samples_us else f"{op.duration_us:,.1f}"
        total_str = f"{op.total_us:,.1f}" if op.samples_us else "—"
        p90_str = f"{op.p90_us:,.1f}" if op.samples_us else "—"
        vtcm_str = (
            f"{op.vtcm_hit_ratio * 100:.1f}%" if op.vtcm_hit_ratio is not None else "—"
        )
        table.add_row(
            str(i),
            node_str,
            op.name,
            avg_str,
            total_str,
            f"{op.percent_of_total:.1f}%",
            f"{cum:.1f}%",
            p90_str,
            _format_bytes(op.dram_read_bytes),
            vtcm_str,
        )

    console.print(table)
```

- [ ] **Step 5: Run tests — confirm they pass**

```
uv run pytest tests/session/monitor/test_report_detail.py -v
```

- [ ] **Step 6: Run all monitor tests**

```
uv run pytest tests/session/monitor/ -v
```

- [ ] **Step 7: Lint + commit**

```bash
uv run ruff check src/winml/modelkit/session/monitor/report.py tests/session/monitor/test_report_detail.py --fix
git add src/winml/modelkit/session/monitor/report.py tests/session/monitor/test_report_detail.py
git commit -m "feat(report): rewrite detail-mode render per 10-column mockup spec

Adds Cum % derivation, p90/Total/Avg from samples_us, em-dash for
unavailable detail-mode metrics. Header rule renamed.

Confidence: high
Scope-risk: moderate"
```

---

## Task 6: Add pre-bench identity block

**Files:**
- Create: `src/winml/modelkit/commands/_pre_bench.py`
- Modify: `src/winml/modelkit/commands/perf.py` (call `print_pre_bench_block` before benchmark loop)
- Create: `tests/commands/test_pre_bench.py`

The pre-bench block has 3 logical sub-blocks per the mockup `build_pre_bench_block`:

1. **Model identity** — for HF: `model_id`, `task`, `opset`, `inputs`, `outputs`, `cached_onnx_path`. For ONNX-file path: just the file path.
2. **Surface** — empty placeholder for now (mockup shows it for forward-looking use).
3. **Device** — resolved device + EP (e.g., `Device: NPU` / `EP: QNN`).

- [ ] **Step 1: Read the mockup helper**

Open `docs/design/perf/console_mockup.py` and locate `build_pre_bench_block` (grep for it). Copy the structure verbatim, adapted to take real perf-command inputs instead of fake constants.

- [ ] **Step 2: Write the failing test**

```python
# tests/commands/test_pre_bench.py
"""Pre-bench identity block: HF and ONNX-file paths."""
from io import StringIO

from rich.console import Console

from winml.modelkit.commands._pre_bench import print_pre_bench_block


def _render_hf() -> str:
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False, record=True)
    print_pre_bench_block(
        console,
        model_id="facebook/convnext-base-224",
        task="image-classification",
        opset=17,
        inputs=[("pixel_values", "float32", (1, 3, 224, 224))],
        outputs=[("logits", "float32", (1, 1000))],
        cached_onnx_path=r"C:\Users\u\.cache\winml\artifacts\convnext.onnx",
        onnx_file=None,
        device="NPU",
        ep="QNN",
    )
    return console.export_text()


def _render_onnx_file() -> str:
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False, record=True)
    print_pre_bench_block(
        console,
        model_id=None, task=None, opset=None, inputs=None, outputs=None,
        cached_onnx_path=None,
        onnx_file=r"C:\models\my_model.onnx",
        device="CPU",
        ep="ORT-CPU",
    )
    return console.export_text()


def test_hf_block_shows_model_id():
    out = _render_hf()
    assert "facebook/convnext-base-224" in out
    assert "image-classification" in out
    assert "17" in out  # opset
    assert "convnext.onnx" in out


def test_hf_block_shows_inputs_and_outputs():
    out = _render_hf()
    assert "pixel_values" in out
    assert "logits" in out


def test_onnx_file_block_shows_path_only():
    out = _render_onnx_file()
    assert "my_model.onnx" in out
    assert "facebook" not in out
    assert "image-classification" not in out


def test_device_block_shows_device_and_ep():
    out = _render_hf()
    assert "NPU" in out
    assert "QNN" in out
```

- [ ] **Step 3: Run test — confirm it fails**

```
uv run pytest tests/commands/test_pre_bench.py -v
```

Expected: `ImportError: cannot import name 'print_pre_bench_block'`.

- [ ] **Step 4: Create `_pre_bench.py`**

```python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Pre-benchmark identity block.

Renders a 3-sub-block intro before the benchmark loop: model identity,
surface (placeholder), and resolved device. Mirrors the mockup helper
in ``docs/design/perf/console_mockup.py``.
"""

from __future__ import annotations

from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def print_pre_bench_block(
    console: Console,
    *,
    model_id: str | None,
    task: str | None,
    opset: int | None,
    inputs: Sequence[tuple[str, str, tuple[int, ...]]] | None,
    outputs: Sequence[tuple[str, str, tuple[int, ...]]] | None,
    cached_onnx_path: str | None,
    onnx_file: str | None,
    device: str,
    ep: str,
) -> None:
    """Print the 3-sub-block pre-benchmark identity panel.

    For HF inputs (``model_id`` set), shows the full identity card. For
    raw ONNX-file inputs, shows just the file path.
    """
    # 1. Model identity
    if model_id:
        ident = Table.grid(padding=(0, 2))
        ident.add_column(justify="right", style="dim")
        ident.add_column()
        ident.add_row("Model:", model_id)
        if task:
            ident.add_row("Task:", task)
        if opset is not None:
            ident.add_row("Opset:", str(opset))
        if inputs:
            ident.add_row("Inputs:", _fmt_io(inputs))
        if outputs:
            ident.add_row("Outputs:", _fmt_io(outputs))
        if cached_onnx_path:
            ident.add_row("Cached ONNX:", cached_onnx_path)
        console.print(Panel(ident, title="Model", expand=True))
    elif onnx_file:
        ident = Table.grid(padding=(0, 2))
        ident.add_column(justify="right", style="dim")
        ident.add_column()
        ident.add_row("ONNX file:", onnx_file)
        console.print(Panel(ident, title="Model", expand=True))

    # 2. Surface (placeholder; forward-looking)
    # Skipped for now to avoid empty panel noise.

    # 3. Device
    dev = Table.grid(padding=(0, 2))
    dev.add_column(justify="right", style="dim")
    dev.add_column()
    dev.add_row("Device:", device)
    dev.add_row("EP:", ep)
    console.print(Panel(dev, title="Device", expand=True))


def _fmt_io(specs: Sequence[tuple[str, str, tuple[int, ...]]]) -> str:
    return ", ".join(f"{n} ({d}, {tuple(s)})" for n, d, s in specs)
```

- [ ] **Step 5: Wire into `perf.py`**

In `src/winml/modelkit/commands/perf.py`, find where the benchmark begins (after parsing inputs, before the iteration loop — the implementer should locate this around the start of `_run_benchmark` or where the model is loaded). Add a call:

```python
from ._pre_bench import print_pre_bench_block

# ... later, before benchmark loop:
print_pre_bench_block(
    console,
    model_id=self.config.model_id,
    task=resolved_task,
    opset=resolved_opset,
    inputs=resolved_input_specs,
    outputs=resolved_output_specs,
    cached_onnx_path=str(cached_onnx) if cached_onnx else None,
    onnx_file=str(onnx_file_arg) if onnx_file_arg else None,
    device=self.config.device or "auto",
    ep=resolved_ep,
)
```

The exact variable names depend on `perf.py`'s existing locals — the implementer reads the surrounding code and uses the available identifiers. If a piece of metadata (e.g., opset) is not currently extracted, pass `None` rather than fabricating it. Add a follow-up note to the design doc §13 if metadata is missing.

- [ ] **Step 6: Run tests — confirm they pass**

```
uv run pytest tests/commands/test_pre_bench.py -v
```

- [ ] **Step 7: Manual smoke test (optional, no real model required)**

```
uv run python -c "from rich.console import Console; from winml.modelkit.commands._pre_bench import print_pre_bench_block; print_pre_bench_block(Console(), model_id='facebook/convnext-base-224', task='image-classification', opset=17, inputs=[('pixel_values', 'float32', (1,3,224,224))], outputs=[('logits', 'float32', (1,1000))], cached_onnx_path=r'C:\\cache\\m.onnx', onnx_file=None, device='NPU', ep='QNN')"
```

Expected: two panels rendered cleanly.

- [ ] **Step 8: Lint + commit**

```bash
uv run ruff check src/winml/modelkit/commands/_pre_bench.py src/winml/modelkit/commands/perf.py tests/commands/test_pre_bench.py --fix
git add src/winml/modelkit/commands/_pre_bench.py src/winml/modelkit/commands/perf.py tests/commands/test_pre_bench.py
git commit -m "feat(perf): add pre-benchmark identity block

Renders model identity (HF or ONNX path) + resolved device/EP
before the benchmark loop. Surface sub-block reserved for
forward-looking use.

Confidence: high
Scope-risk: narrow"
```

---

## Task 7: Add save-to footers after op-trace report

**Files:**
- Modify: `src/winml/modelkit/commands/perf.py` (around line 1506, after `display_op_trace_report`)
- Create: `tests/commands/test_perf_save_footer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/commands/test_perf_save_footer.py
"""Save-to footer prints after op-trace report."""
from io import StringIO

from rich.console import Console

from winml.modelkit.commands.perf import _print_save_to_footer


def _render(trace_json: str | None, profiling_csv: str | None) -> str:
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False, record=True)
    _print_save_to_footer(console, trace_json=trace_json, profiling_csv=profiling_csv)
    return console.export_text()


def test_both_paths_shown():
    out = _render(r"C:\out\trace.json", r"C:\out\prof.csv")
    assert "trace.json" in out
    assert "prof.csv" in out


def test_csv_omitted_when_none():
    out = _render(r"C:\out\trace.json", None)
    assert "trace.json" in out
    assert ".csv" not in out


def test_neither_when_both_none():
    out = _render(None, None)
    assert out.strip() == ""
```

- [ ] **Step 2: Run test — confirm it fails**

```
uv run pytest tests/commands/test_perf_save_footer.py -v
```

- [ ] **Step 3: Add `_print_save_to_footer` to `perf.py`**

Add near the other private helpers in `perf.py`:

```python
def _print_save_to_footer(
    console: "Console",
    *,
    trace_json: str | None,
    profiling_csv: str | None,
) -> None:
    """Print save-to footer lines after the op-trace report."""
    if trace_json:
        console.print(f"[dim]Op-trace JSON:[/dim] {trace_json}")
    if profiling_csv:
        console.print(f"[dim]Profiling CSV:[/dim] {profiling_csv}")
```

- [ ] **Step 4: Wire into the post-benchmark block**

Around line 1506 in `perf.py`, after the existing `display_op_trace_report(trace_result, console)` and `write_op_trace_json(trace_result, trace_output)` calls, add:

```python
profiling_csv = trace_result.artifacts.get("profiling_csv")
_print_save_to_footer(
    console,
    trace_json=str(trace_output),
    profiling_csv=profiling_csv,
)
```

- [ ] **Step 5: Run tests — confirm they pass**

```
uv run pytest tests/commands/test_perf_save_footer.py -v
```

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff check src/winml/modelkit/commands/perf.py tests/commands/test_perf_save_footer.py --fix
git add src/winml/modelkit/commands/perf.py tests/commands/test_perf_save_footer.py
git commit -m "feat(perf): print save-to footer after op-trace report

Lists trace JSON path always, profiling CSV path when the
parser surfaced it via OpTraceResult.artifacts.

Confidence: high
Scope-risk: narrow"
```

---

## Task 8: Bump HW chart window to 15s and width to 120

**Files:**
- Modify: `src/winml/modelkit/commands/_live_chart.py:20, 38` (constants)
- Create: `tests/commands/test_live_chart_constants.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/commands/test_live_chart_constants.py
"""Constants pin the mockup-approved chart geometry."""
from winml.modelkit.commands import _live_chart


def test_chart_window_seconds_is_fifteen():
    assert _live_chart._CHART_WINDOW_SECONDS == 15.0


def test_default_chart_width_is_one_hundred_twenty():
    import inspect
    sig = inspect.signature(_live_chart.LiveMonitorDisplay.__init__)
    assert sig.parameters["chart_width"].default == 120
```

- [ ] **Step 2: Run tests — confirm they fail**

```
uv run pytest tests/commands/test_live_chart_constants.py -v
```

Expected: `_CHART_WINDOW_SECONDS == 10.0` (failure), `chart_width == 80` (failure).

- [ ] **Step 3: Update constants**

Edit `src/winml/modelkit/commands/_live_chart.py`:

```python
# Line 20:
_CHART_WINDOW_SECONDS = 15.0  # was 10.0; bumped proportionally with width

# Line 38 (in __init__ signature):
        chart_width: int = 120,  # was 80
```

- [ ] **Step 4: Run tests — confirm they pass**

```
uv run pytest tests/commands/test_live_chart_constants.py -v
```

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/winml/modelkit/commands/_live_chart.py tests/commands/test_live_chart_constants.py --fix
git add src/winml/modelkit/commands/_live_chart.py tests/commands/test_live_chart_constants.py
git commit -m "feat(live-chart): bump window to 15s and default width to 120

Matches mockup geometry. Window scaled proportionally with the
wider chart so x-axis tick density remains comfortable.

Confidence: high
Scope-risk: narrow"
```

---

## Task 9: Hardware E2E verification with convnext-base-224

This task is **manual** — no test code. The goal is to confirm the lifted UX renders correctly on real hardware before merging.

- [ ] **Step 1: Run basic mode**

```
uv run python -m winml.modelkit perf -m facebook/convnext-base-224 --op-tracing basic
```

Expected output:
- Pre-bench identity block: model `facebook/convnext-base-224`, task, opset, inputs/outputs, cached ONNX path, then Device: NPU + EP: QNN
- Live chart for the (single) iteration with 15s window
- Op-Tracing (basic) header
- 4-column table (Node / Type / p90 / % Tot), Node column width 80 with left-ellipsis on long paths
- Save-to footer: trace JSON path, profiling CSV path

- [ ] **Step 2: Run detail mode**

```
uv run python -m winml.modelkit perf -m facebook/convnext-base-224 --op-tracing detail --iterations 50
```

Expected:
- Same pre-bench block
- Op-Tracing (detail) header (with `-- <backend>` suffix)
- 10-column table including Avg / Total / Cum % / p90 / DRAM(R) / VTCM Hit
- Top-K = 5 by default (the global `_TOP_K_DEFAULT` should match the mockup's `OP_TRACING_TOP_K_DEFAULT = 5`; if `--top-k` is not yet wired into `display_op_trace_report`, file a follow-up)

- [ ] **Step 3: Visual diff against mockup**

Compare side-by-side:
- `cd docs/design/perf && uv run python console_mockup.py --op-tracing basic`
- vs the live output from Step 1

The basic-mode tables should be visually indistinguishable except for real op names vs fake ones.

- [ ] **Step 4: Capture a screenshot or text snapshot for the PR**

Save the production output to `temp/op_tracing_e2e_basic.txt` and `temp/op_tracing_e2e_detail.txt`. These are evidence for the PR description.

- [ ] **Step 5: No commit unless something broke**

If a real-hardware bug surfaces, fix it as a separate small commit referencing this task. Otherwise nothing to commit.

---

## Task 10: Update design doc + write production-lift summary

**Files:**
- Modify: `docs/design/perf/2026-04-28-console-mockup-design.md` (status field)
- Modify: `docs/design/perf/2026-04-29-session-handoff.md` (mark resolved items)
- Create: `docs/design/perf/2026-04-29-op-tracing-production-lift-summary.md` (1-page outcome)

- [ ] **Step 1: Update design doc status**

In the revision-history table at the top, add a row:

```
| v2.1 | 2026-04-29 | Production lift complete — render layer + pre-bench + footers + chart geometry now in src/winml/modelkit/. Mockup retained as reference. |
```

Update the status line near the top from `"Implemented; under team review"` to `"Production-lifted; mockup retained as reference"`.

- [ ] **Step 2: Update the handoff doc**

In `docs/design/perf/2026-04-29-session-handoff.md`, mark these items as resolved:
- "Production lift roadmap" section → strike-through and link to the new commits
- Any "open commit-strategy" item → resolved

- [ ] **Step 3: Write a 1-page production-lift summary**

```markdown
# Op-Tracing Production Lift — Outcome Summary

**Date:** 2026-04-29
**Branch:** feat/op-tracing-refactor
**Plan executed:** 2026-04-29-op-tracing-production-lift-plan.md

## What changed

| File | Change |
|---|---|
| `op_metrics.py` | +samples_us field, +avg/p90/total/count derived properties |
| `qnn/csv_parser.py` | Retains per-sample timings instead of collapsing to avg |
| `report.py` | Basic + detail render rewritten per mockup spec; +`_truncate_node_name` |
| `commands/_pre_bench.py` | NEW — pre-bench identity block |
| `commands/perf.py` | +pre-bench wiring, +save-to footer |
| `commands/_live_chart.py` | Window 10s→15s, default width 80→120 |

## Acceptance criteria coverage

All 21 ACs from the mockup design doc covered by tests + manual E2E. See per-task test files:
- ACs 1–8: `test_op_metrics_samples.py`, `test_csv_parser_samples.py`
- ACs 9–14: `test_report_basic.py`, `test_report_detail.py`
- ACs 15–17: `test_pre_bench.py`, `test_perf_save_footer.py`
- ACs 18–21: `test_live_chart_constants.py` + manual E2E captures

## Forward-looking follow-ups (deferred)

(carry over from design doc §13)

- GPU silicon column in HW monitor
- Surface sub-block content (currently a no-op placeholder)
- Promotion of hardcoded constants in `_live_chart.py` to settings
```

- [ ] **Step 4: Commit doc updates**

```bash
git add docs/design/perf/2026-04-28-console-mockup-design.md docs/design/perf/2026-04-29-session-handoff.md docs/design/perf/2026-04-29-op-tracing-production-lift-summary.md
git commit -m "docs(perf): mark op-tracing mockup production-lifted

Confidence: high
Scope-risk: narrow"
```

---

## Self-review checklist (already applied)

**Spec coverage:** All 21 ACs from the design doc map to tasks (1, 2 → data; 3–5 → render; 6 → pre-bench; 7 → footer; 8 → chart geometry; 9 → E2E; 10 → docs).

**Placeholder scan:** Two known unknowns are explicit, not placeholders:
- Task 6 step 5: variable names from `perf.py` locals — implementer reads surrounding code; this is necessary because `perf.py` has 1521 lines and I did not survey them all.
- Task 5 step 1: detail-mode column widths — implementer reads `console_mockup.py:448-465` (cited as source of truth); this is intentional because re-typing widths invites drift from the mockup.

**Type consistency:** `OperatorMetrics` field names (`samples_us`, `avg_us`, `p90_us`, `total_us`, `sample_count`) used consistently across Tasks 1, 2, 4, 5. No drift.

**Open follow-ups noted:**
- If `--top-k` flag is not yet plumbed into `display_op_trace_report(top_n=...)`, that is a one-line wire-through change; flag in PR description if discovered.
- QHAS parser may not aggregate timings; if so, Task 2 step 7 is a no-op.
- Surface sub-block content (Task 6) is reserved blank; future task to populate.

---

## Execution choice

**Plan complete and saved to `docs/design/perf/2026-04-29-op-tracing-production-lift-plan.md`. Two execution options:**

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for this plan because tasks 1–8 are independent enough that per-task review catches regressions early.

2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Saves context but loses the per-task review gate.

**Which approach?**
