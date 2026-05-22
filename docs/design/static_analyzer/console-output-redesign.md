> Note: This design doc was written before implementation. Some details (file names, column labels) may differ from the final implementation in modelkit/commands/analyze.py.

# Static Analyzer Console Output Redesign

> Date: 2026-03-18 | Branch: `mvp/analyzer`
> Status: Draft
> Mockup: `docs/design/static_analyzer/console_mockup.py`

---

## 1. Problem Statement

The static analyzer's console output has several limitations:

1. **No real-time progress** — analysis can take 30+ seconds with no visual feedback
2. **Per-instance data lost** — `RuntimeChecker` checks each node individually, but
   `OutputAggregator` collapses results to per-type flat lists, losing instance counts
3. **Verbose flag is dead code** — `console_writer.py` accepts `verbose` but never uses it
4. **Default log level hides progress** — WARNING (30) suppresses INFO progress messages
5. **NullHandler missing** — library consumers get "no handler found" warnings

## 2. Scope

### In Scope

- Logging infrastructure fixes (default level, NullHandler, verbosity plumbing)
- New per-instance classification field on `EPSupport`
- Callback API on `analyze()` for streaming per-node results
- New stacked bar console visualization with `rich.live.Live`
- CLI integration wiring callback to Rich Live display

### Out of Scope

- Changes to `RuntimeChecker` core logic (already per-node)
- Changes to JSON output format (backward compat)
- New CLI flags beyond existing `-v`/`-q`
- Pattern detection or information engine changes

## 3. Design

### 3.1 Logging Infrastructure

#### 3.1.1 Default Level: WARNING -> INFO

**File**: `modelkit/utils/logging.py`

Change the base level so progress messages show by default:

```
-q       → ERROR (40)      errors only (quiet / scripting)
(default)→ INFO (20)       progress messages visible
-v       → DEBUG (10)      developer tracing
```

Formula change: `max(DEBUG, INFO - verbosity * 10)`. Quiet override stays `ERROR`.

Since `-v` and `-vv` both clamp to DEBUG with this formula, we effectively have
3 useful levels (ERROR, INFO, DEBUG). This is sufficient — the gap between INFO
and DEBUG can be addressed later with VERBOSE=15 if needed.

#### 3.1.2 Fix Verbosity Plumbing

**File**: `modelkit/commands/static-analyzer.py`

```python
# Before (broken — hits deprecated bool compat path):
configure_logging(verbose=verbose, quiet=quiet)

# After (uses int verbosity correctly):
configure_logging(verbosity=verbose, quiet=quiet)
```

Same fix needed in `compile.py` and `quantize.py`. Note: these commands currently use
`--verbose` as a `bool` flag (`is_flag=True`), not a count. To get the full verbosity
range, their `--verbose` option must also be changed to `count=True` to match
`static-analyzer.py`. Until then, `verbosity=True` coerces to `verbosity=1` (INFO),
which is acceptable as a first step.

#### 3.1.3 NullHandler (Done)

Already added to `modelkit/__init__.py:31`. All child loggers covered.

#### 3.1.4 Third-Party Logger Suppression

At DEBUG level, suppress noisy third-party loggers:

```python
if log_level <= logging.DEBUG:
    for name in ("onnx", "onnxruntime", "transformers", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
```

### 3.2 Data Model — Per-Instance Counts

**File**: `modelkit/static_analyzer/models/output.py`

Add new field to `EPSupport`:

```python
class EPSupport(BaseModel):
    # ... existing fields ...

    # NEW: per-instance classification counts
    instance_counts: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description=(
            "Per-operator-type instance counts by support level. "
            "e.g., {'Conv': {'white': 53}, 'Add': {'white': 12, 'gray': 5, 'black': 1}}"
        ),
    )
```

Uses string keys (not SupportLevel enum) for JSON serialization simplicity.

**File**: `modelkit/static_analyzer/core/output_aggregator.py`

In `build_ep_support()`, count per-instance results.

Operator `pattern_id` format is always `OP/{domain}/{op_name}` (three segments,
e.g., `OP/ai.onnx/Conv`). Extract display name via a shared helper to avoid
duplicating the parse logic:

```python
def _display_name(pattern_id: str) -> str:
    """Extract operator display name from pattern_id ('OP/ai.onnx/Conv' -> 'Conv')."""
    return pattern_id.split("/")[-1]

instance_counts: dict[str, dict[str, int]] = {}
for pattern_runtime in check_results:
    display_name = _display_name(pattern_runtime.pattern_id)
    level = pattern_runtime.result.classification.value  # e.g., "white"

    if display_name not in instance_counts:
        instance_counts[display_name] = {}
    instance_counts[display_name][level] = (
        instance_counts[display_name].get(level, 0) + 1
    )
```

Existing `classification` field preserved unchanged for backward compat.

**Scope limitation**: `instance_counts` covers operator-level nodes only, not subgraph
patterns. Subgraph results (`subgraph_runtime_check_result`) are passed to the
information engine but not included in `check_results` flowing to `aggregate()`.
This is intentional — subgraph patterns are displayed in the information/actions
section, not the operator bar chart.

### 3.3 Callback API for Streaming Results

Three callbacks provide lifecycle hooks for the analysis pipeline. All are
optional (`None` by default) — existing callers are unaffected.

```python
def analyze(
    self,
    model_path: str,
    ep: str | None = None,
    ...,
    on_node_result=None,   # (PatternRuntime) -> None
    on_ep_start=None,      # (ep_name: str, operator_counts: dict[str, int]) -> None
) -> AnalysisResult:
```

#### Callback signatures and lifecycle

```
analyze_from_proto()
  │
  ├─ Step 1: PatternExtractor.summary()
  │    → extracts operator_counts: {"Conv": 53, "Relu": 49, ...}
  │    → this is MODEL-LEVEL, same for all EPs
  │
  ├─ Step 2: For each EP:
  │    ├─ on_ep_start(ep_name, operator_counts)     ← EP begins
  │    │    The command uses operator_counts to build pending rows
  │    │    (dim table with all ops listed, ░░░ placeholder bars)
  │    │
  │    ├─ RuntimeChecker.op_support() loop:
  │    │    for node in graph.node:
  │    │        result = query.run_for_node(node)
  │    │        on_node_result(result)               ← per-node result
  │    │        The command updates the row for this op type
  │    │
  │    └─ (information engine runs, no callbacks)
  │
  └─ Step 3: OutputAggregator → return AnalysisResult
```

#### Expected Live display behavior

1. **on_ep_start fires**: Table appears with ALL op types as dim/pending rows
   (we know the ops from `operator_counts`). Each row shows `░░░` bars.
2. **on_node_result fires per-node**: As each node is checked, its op type row
   accumulates counts. When the row's count reaches the total for that type,
   the row transitions from dim to colored (icon + S/P/U + stacked bar).
3. **All nodes checked**: Table shows "Complete" with all rows filled in.
4. **Next EP**: Reset counts, show new pending table, repeat.

#### Threading through the API

**`analyzer.py`**: `analyze()` → `analyze_from_proto()` → forwards both callbacks.

**`analyze_from_proto()`**: Calls `on_ep_start(current_ep, operator_counts)` before
each `RuntimeChecker.summary()` call, where `operator_counts` comes from
`extraction_result["summary"].operator_counts`.

**`runtime_checker.py`**: `summary()` → `op_support()` → invokes `on_node_result`
per-node. tqdm is disabled when `on_node_result` is provided.

#### Design constraints
- Callbacks are `None` by default — zero impact on existing callers
- Callbacks receive plain data (no UI dependency)
- Analyzer API has no Rich dependency — stays a pure library
- `on_node_result` replaces tqdm's role for progress visualization
- `operator_counts` is passed on every `on_ep_start` call (same dict, no cost)

### 3.4 Console Writer — Stacked Bar Visualization

**File**: `modelkit/static_analyzer/console_writer.py`

New visualization replacing the current operator table + classification sections.

#### Layout

```
              📊 ONNX Static Analysis — QNNExecutionProvider  ✅ Complete
 Op Type                       Analyze
 🟢 Conv (53)                  53/0/0          ████████████████████████████████████████
 🟢 Relu (53)                  53/0/0          ████████████████████████████████████████
 🟡 MatMul (25)                20/5/0          ███████████████████
 🔴 Add (18)                   12/5/1          ██████████████
 🔵 Reshape (10)               8/0/0/2         ████████
 🟡 Erf (8)                    0/8/0           ██████
 🔴 Resize (3)                 0/0/3           ██
 TOTAL (231)                   203/19/4/5      ████████████████████████████████████████
```

#### Column Specification

| Column | Content | Width |
|--------|---------|-------|
| Op Type | `{icon} {op_name} ({total})` | 28 chars |
| Analyze | `W/G/B[/U]` colored counts | 14 chars |
| (untitled) | Stacked bar, variable width | Proportional to count |

#### Icon Semantics (Worst-Case Indicator)

| Icon | Condition | Meaning |
|------|-----------|---------|
| 🟢 | All instances WHITE | Fully supported |
| 🟡 | Any GRAY (no BLACK) | Partial support |
| 🔴 | Any BLACK | Unsupported instances |
| 🔵 | Any UNKNOWN (no GRAY/BLACK) | Unknown support |

#### Bar Rendering

- Width proportional to `op_total / max_op_total * MAX_BAR_WIDTH`
- Segments colored: green (white), yellow (gray), red (black), dim gray (unknown)
- Each segment gets at least 1 char if count > 0. To guarantee this, the total
  bar width must be `max(bar_width, num_nonzero_segments)` so no segment is
  clamped to 0 by the proportional width calculation.
- Built with `rich.text.Text` per-segment styling

#### Analyze Column Format

`W/G/B` with colored digits. Fourth value `/U` appended only when unknown > 0.

```
53/0/0       → green "53", dim "0", dim "0"
12/5/1       → green "12", yellow "5", red "1"
8/0/0/2      → green "8", dim "0", dim "0", gray "2"
```

### 3.5 CLI Integration

**File**: `modelkit/commands/static-analyzer.py`

```python
from rich.live import Live

# Build initial empty table
console = Console(stderr=True)

# Wire callback to Live display
counts = {}
# build_analysis_table must handle empty data gracefully (no crash on max() of empty seq)
with Live(build_analysis_table(counts), console=console, refresh_per_second=4) as live:
    def on_result(pattern_runtime):
        # Accumulate per-instance counts
        op = pattern_runtime.pattern_id.split("/")[-1]
        level = pattern_runtime.result.classification.value
        counts.setdefault(op, {}).setdefault(level, 0)
        counts[op][level] += 1
        live.update(build_analysis_table(counts))

    result = analyzer.analyze(
        model_path=model,
        ep=ep_normalized,
        on_node_result=on_result,
        ...
    )

# After Live exits: show remaining sections
# (header, pattern summary, information items, footer)
display_post_analysis(result.output)
```

**Quiet mode** (`-q`): Skip Live display entirely, pass `on_node_result=None`.

### 3.6 Separation of Concerns

```
┌─────────────────────────────────┐
│  analyzer.py (LIBRARY)          │
│  - logging.getLogger(__name__)  │
│  - callback(PatternRuntime)     │
│  - No Rich, no UI              │
└──────────┬──────────────────────┘
           │ callback with plain data
┌──────────▼──────────────────────┐
│  static-analyzer.py (command)   │
│  - configure_logging()          │
│  - Rich Live display            │
│  - Wires callback to UI         │
└─────────────────────────────────┘
```

## 4. Testing Strategy

- **Unit tests**: `build_stacked_bar()`, `worst_level_icon()`, `build_analyzed_text()` — pure functions, no Rich Live needed
- **Data model tests**: Verify `instance_counts` populated correctly by OutputAggregator
- **Callback tests**: Mock callback, verify it receives correct PatternRuntime per node
- **Integration test**: Full `analyze()` with callback, verify counts match final result
- **Console writer**: Capture Rich output to string buffer, verify table structure

## 5. Migration / Backward Compatibility

- Existing `EPSupport.classification` field unchanged — JSON output stays the same
- `instance_counts` defaults to empty dict — old serialized data still valid
- `on_node_result=None` default — all existing callers work without changes
- Default log level change (WARNING → INFO) affects all commands — intentional, aligns with pip convention

## 6. Files Changed

| File | Change |
|------|--------|
| `modelkit/utils/logging.py` | Default level WARNING→INFO, third-party suppression |
| `modelkit/commands/static-analyzer.py` | Fix verbosity plumbing, Rich Live integration |
| `modelkit/commands/compile.py` | Fix verbosity plumbing (`verbosity=verbose`) |
| `modelkit/commands/quantize.py` | Fix verbosity plumbing (`verbosity=verbose`) |
| `modelkit/static_analyzer/models/output.py` | Add `instance_counts` field to `EPSupport` |
| `modelkit/static_analyzer/core/output_aggregator.py` | Populate `instance_counts` |
| `modelkit/static_analyzer/analyzer.py` | Add `on_node_result` callback parameter |
| `modelkit/static_analyzer/core/runtime_checker.py` | Invoke callback per-node |
| `modelkit/static_analyzer/console_writer.py` | New stacked bar table, remove old sections |
| `modelkit/__init__.py` | NullHandler (already done, no further changes needed) |
