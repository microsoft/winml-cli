# src/winml/modelkit/optracing/qnn/qhas_parser.py (DELETED)

## TL;DR
This file is removed. It parsed QNN Hardware Acceleration Summary (QHAS) JSON artifacts into a normalised summary + per-operator list (with cycles → microseconds conversion, VTCM hit ratios, and dominant-path durations). The function `parse_qhas` is **relocated to `session/monitor/qnn/_internal.py`** with the same signature and return shape.

## Diff metrics
- Lines deleted: 113
- Status: DELETED

## What this file did (pre-state)
Transformed deserialised QHAS JSON (produced by the QNN profile viewer) into a structured dict consumable by the detail-mode op-tracing reporter:

- Extracted `data.htp_overall_summary` into a flat summary dict (time, cycles, DRAM/VTCM totals, utilization, etc.).
- Derived a cycle-to-microsecond conversion factor from `summary["time_us"] / summary["timeline_cycles"]`.
- Walked `data.qnn_op_instances_nodes` rows, applying the cycle-to-us factor to each op's cycles, dominant-path cycles.
- Computed VTCM hit ratio as `vtcm_read / (vtcm_read + dram_read)` (None when both zero).

## Public symbols (pre-deletion)
- `parse_qhas(qhas_data: dict) -> dict` — returns `{"summary": {...}, "operators": [...]}`.
- Private helpers: `_extract_summary`, `_transform_op`, `_vtcm_ratio`.

## Where the functionality moved
| Pre-state symbol | Where it lives now |
|---|---|
| `parse_qhas(qhas_data)` | **`session.monitor.qnn._internal.parse_qhas`** — same name, same `{summary, operators}` return shape. Called from `QNNMonitor._try_qhas` during `__exit__`. |
| `_extract_summary`, `_transform_op`, `_vtcm_ratio` | All retained as `_`-prefixed helpers in `qnn/_internal.py`. A new `_require(d, key, context)` helper wraps strict-key access with a contextful KeyError message used across both QHAS and CSV parsers. |

## Net behavior change
- The QHAS parser is functionally unchanged. The relocation is a packaging move; the math (cycle-to-us conversion, VTCM hit ratio formula) is identical.
- Co-located with the CSV parser in `_internal.py` rather than living in a sibling module, reflecting that both parsers now serve the single `QNNMonitor` rather than separate `OpTracer` subclasses.

## Risks
- Any caller that imported `from winml.modelkit.optracing.qnn.qhas_parser import parse_qhas` will fail. The new path is intentionally private (`qnn._internal`); third parties should consume the typed `OpTraceResult` produced by `QNNMonitor`.
- The QHAS-input schema expectations (presence of `data.htp_overall_summary`, `data.qnn_op_instances_nodes`, the field names like `time_us`, `timeline_cycles`, `percent_active_cycles`, `qnn_op_type`) are unchanged — anything currently working will keep working under the new home.
