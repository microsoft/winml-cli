# src/winml/modelkit/optracing/qnn/csv_parser.py (DELETED)

## TL;DR
This file is removed. The QNN basic-mode profiling CSV parser has been relocated verbatim (and slightly extended) into the private `session/monitor/qnn/_internal.py` module. The public entrypoint `parse_qnn_profiling_csv()` is re-exported from `session/monitor/qnn/__init__.py`.

## Diff metrics
- Lines deleted: 227
- Status: DELETED

## What this file did (pre-state)
Parsed the seven-column CSV emitted by QNN EP in basic profiling mode into a structured `{metadata, operators, samples}` dict. It:
- Read CSV rows with `csv.DictReader`.
- Extracted ROOT-level metadata (HVX thread count, accelerator execute cycles/us) — first occurrence only.
- Identified sample boundaries via ROOT `"Accelerator (execute) time (cycles)"` rows.
- Parsed NODE SUB-EVENT rows (CYCLES unit) into per-operator entries via the `_OP_PATTERN` regex (`(.+?)(?:\s+|:)OpId_(\d+)\s*\(cycles\)`).
- Stripped the QNN compiler's `_token_\d+(?:_\d+)?` suffix via `_TOKEN_SUFFIX`.
- Aggregated operators across samples by `op_id`, averaging cycles, sorting descending.

## Public symbols (pre-deletion)
- `_OP_PATTERN: re.Pattern` — module-private regex for `(name, op_id, cycles)` extraction from the Event Identifier column.
- `_TOKEN_SUFFIX: re.Pattern` — module-private regex for stripping QNN compiler-injected token suffixes.
- `parse_qnn_profiling_csv(csv_path) -> dict[str, Any]` — public parser; the single user-facing API.
- `_read_csv`, `_extract_metadata`, `_extract_samples`, `_parse_node_event`, `_aggregate_operators` — private helpers.

## Where the functionality moved
| Pre-state symbol | Where it lives now |
|---|---|
| `parse_qnn_profiling_csv` | `src/winml/modelkit/session/monitor/qnn/_internal.py` (and re-exported from `session/monitor/qnn/__init__.py`). Same signature; same return shape with one addition: each aggregated operator now also carries a `samples_cycles: list[int]` field so downstream layers can derive p90/total/count from per-sample data. |
| `_OP_PATTERN` | `session/monitor/qnn/_internal.py` — same regex literal. |
| `_TOKEN_SUFFIX` | `session/monitor/qnn/_internal.py` — same regex. **Promoted to a cross-module import:** `QNNMonitor._heuristic_op_type` imports it from `_internal` so the CSV path and the heuristic op-type fallback share strip semantics. |
| `_read_csv`, `_extract_metadata`, `_extract_samples`, `_aggregate_operators` | `session/monitor/qnn/_internal.py` — same logic. `_aggregate_operators` extended to also build `samples_cycles[oid]` (per-sample cycle list keyed by op_id) and attach it to each aggregated entry. |
| `_parse_node_event` | `session/monitor/qnn/_internal.py`. **Behavior extended:** now also returns `op_path` (the cleaned, full event-id string after token strip) in addition to `name` (the leaf segment). This is the v2.4 split needed to keep the `op_path` key consistent with the QHAS path so the L1 ONNX-op-type lookup in `QNNMonitor._resolve_op_type` works. |
| (new in `_internal.py`) `_split_op_event_id` | New helper for the CSV-only path: returns `(op_type, op_path)` where the op_type is the trailing slash-delimited leaf. Documents the warning that the leaf is the *ONNX op symbol*, not the QNN op type — only a best-effort fallback when QHAS isn't available. |

## Net behavior change
- The aggregated-operator dict gained a `samples_cycles: list[int]` field. Downstream consumers (`QNNMonitor`) convert this to microseconds via the ROOT-level `cycle_to_us` ratio and surface it as `OperatorMetrics.samples_us`, enabling p90/total/count metrics.
- The `_parse_node_event` return shape gained `op_path` so the CSV path can drive the same L1 ONNX-lookup → L3 heuristic → L4 raw resolver chain as the QHAS path.
- The module is now private (`_internal.py`) per the v2.4 information-hiding contract; only `qnn_monitor.py` is supposed to import non-`_`-prefixed names from it, and an architecture regression test (`tests.unit.architecture.test_qnn_imports`) enforces this.

## Risks
- No public `optracing.qnn.csv_parser` module import path remains. Out-of-tree callers must switch to `from winml.modelkit.session.monitor.qnn import parse_qnn_profiling_csv`.
- The information-hiding boundary means *direct* imports from `session.monitor.qnn._internal` will trip the architecture regression test (with a CLAUDE.md-sanctioned exception for `_`-prefixed names like `_TOKEN_SUFFIX`).
