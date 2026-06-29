# src/winml/modelkit/optracing/qnn/csv_parser.py (DELETED)

## TL;DR
This file is removed. It parsed QNN basic-mode profiling CSV files into per-sample operator cycle counts. The function `parse_qnn_profiling_csv` is **relocated to `session/monitor/qnn/_internal.py`** (same signature, same return shape), and the per-sample lists are now used to populate the new `OperatorMetrics.samples_us` field.

## Diff metrics
- Lines deleted: 227
- Status: DELETED

## What this file did (pre-state)
Parsed the seven-column CSV that QNN EP emits in basic profiling mode (`Msg Timestamp, Message, Time, Unit of Measurement, Timing Source, Event Level, Event Identifier`):

- Walked `ROOT` rows to extract aggregate metadata (HVX thread count, accelerator execute cycles/us).
- Walked `NODE SUB-EVENT` rows to extract per-operator cycle counts.
- Detected sample boundaries via `ROOT "Accelerator (execute) time (cycles)"` rows.
- Parsed the `Event Identifier` regex `(.+?)(?:\s+|:)OpId_(\d+)\s*\(cycles\)` to split operator name from `OpId`.
- Stripped `_token_\d+(?:_\d+)?` suffixes injected by the QNN compiler.
- Averaged cycle counts across samples, keyed by `op_id` to keep identically-named ops distinct, then sorted descending by cycles.

## Public symbols (pre-deletion)
- `parse_qnn_profiling_csv(csv_path: str | Path) -> dict[str, Any]` — returns `{metadata, operators, samples}`.
- Module-level regex constants `_OP_PATTERN`, `_TOKEN_SUFFIX`.
- Private helpers: `_read_csv`, `_extract_metadata`, `_extract_samples`, `_parse_node_event`, `_aggregate_operators`.

## Where the functionality moved
| Pre-state symbol | Where it lives now |
|---|---|
| `parse_qnn_profiling_csv(csv_path)` | **`session.monitor.qnn._internal.parse_qnn_profiling_csv`** — same name, same return shape (`{metadata, operators, samples}`). Now called from `QNNMonitor._parse_csv_artifacts` during `__exit__`. |
| `_OP_PATTERN` regex | Retained in `qnn/_internal.py` (same pattern). |
| `_TOKEN_SUFFIX` regex | Retained in `qnn/_internal.py` (same pattern). |
| `_read_csv`, `_extract_metadata`, `_extract_samples`, `_parse_node_event`, `_aggregate_operators` | All retained as `_`-prefixed helpers in `qnn/_internal.py`. A new `_split_op_event_id` helper was added alongside them. A new `_require(d, key, context)` helper was added for stricter key validation. |

## Net behavior change
- The CSV parser is functionally unchanged — the regex, sample-boundary detection, and aggregation logic are preserved verbatim. The relocation is primarily a packaging move (`optracing/qnn/` → `session/monitor/qnn/`).
- The new `_internal.py` co-locates CSV and QHAS parsing in a single private module (previously two files in the same package).
- Downstream consumption changed: the parsed per-sample data now populates `OperatorMetrics.samples_us` rather than being averaged into a single `cycles` field at parse time.

## Risks
- Any caller that imported `from winml.modelkit.optracing.qnn.csv_parser import parse_qnn_profiling_csv` will fail. The new location is private (`qnn._internal`) by intent — third parties should consume the `OpTraceResult` produced by `QNNMonitor`, not the raw parser.
