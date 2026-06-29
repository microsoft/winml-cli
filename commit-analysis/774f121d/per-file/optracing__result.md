# src/winml/modelkit/optracing/result.py (DELETED)

## TL;DR
This file is removed. It defined the `OperatorMetrics` and `OpTraceResult` dataclasses — the structured shape of all op-tracing output. The dataclasses are **relocated almost verbatim** to `session/monitor/op_metrics.py`, with additive fields (`status`, `error`, `samples_us`) and a new `TraceStatus` `Literal` alias.

## Diff metrics
- Lines deleted: 99
- Status: DELETED

## What this file did (pre-state)
Defined the two dataclasses that the whole op-tracing pipeline produced and consumed:
- `OperatorMetrics` — per-op metrics: identity (`name`, `op_path`, `op_id`), temporal localization (`start_time_us`, `duration_us`, `percent_of_total`), roofline (`hardware_time_us`, `memory_time_us`, `dominant_path_us`), DMA traffic (`dram_*`, `vtcm_*`), and cache efficiency (`vtcm_hit_ratio`).
- `OpTraceResult` — top-level result: metadata (`model`, `device`, `tracing_level`, `ep`, `tracing_backend`, `timestamp`, `num_samples`), `operators: list[OperatorMetrics]`, plus `summary`, `statistics`, and `artifacts` dicts.

Both dataclasses provided `to_dict()`; `OpTraceResult` also provided `to_json(indent)`.

## Public symbols (pre-deletion)
- `OperatorMetrics` (dataclass) with `to_dict()`.
- `OpTraceResult` (dataclass) with `to_dict()` and `to_json(indent: int = 2)`.

## Where the functionality moved
| Pre-state symbol | Where it lives now |
|---|---|
| `OperatorMetrics` | **`session.monitor.op_metrics.OperatorMetrics`** — same fields, plus new `samples_us: list[float]` and three derived properties (`sample_count`, `avg_us`, `total_us`, `p90_us`) for retaining per-sample timings. The `name` field's docstring was expanded to spell out the v2.4 fallback chain (L1 ONNX op_type → L2 EP-authoritative → L3 heuristic split → L4 raw op_path). |
| `OpTraceResult` | **`session.monitor.op_metrics.OpTraceResult`** — same fields, plus additive `status: TraceStatus = "ok"` and `error: str \| None = None`. `model: str` widened to `model: str \| None`. `to_dict()` now adds top-level `status` / `error` keys. `datetime.timezone.utc` swapped for `datetime.UTC`. |
| (new) `TraceStatus` | New `Literal["ok", "no_data", "parse_failed", "basic_fallback", "not_run"]` alias declared in the new module — enforces a closed status vocabulary. |

## Net behavior change
- The schema is **additive-only on `OpTraceResult`**: the `metadata`/`summary`/`operators`/`statistics`/`artifacts` nested structure produced by `to_dict()` is unchanged; new top-level `status` and `error` keys are appended. Older JSON consumers that ignore unknown keys continue to work.
- `OperatorMetrics.samples_us` is new — populated by the new QNN CSV parser with per-iteration cycle samples so downstream code can compute `p90`, `total`, `avg` without re-parsing. Pre-state, only the averaged duration was carried.
- `model` becoming nullable allows monitors to construct a partial result before the session knows its model path.

## Risks
- Any caller that did `from winml.modelkit.optracing import OpTraceResult` (or `from ...result`) gets `ImportError`. New path: `from ...session.monitor.op_metrics import OpTraceResult`.
- Code that pattern-matched on `OpTraceResult.model: str` (assuming non-None) must now handle `None`.
- Anything that imported `from datetime import timezone` and replicated the `datetime.now(timezone.utc).isoformat()` default expression still works at runtime; the swap to `UTC` in the new module is cosmetic.
