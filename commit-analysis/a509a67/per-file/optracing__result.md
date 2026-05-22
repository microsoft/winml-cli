# src/winml/modelkit/optracing/result.py (DELETED)

## TL;DR
This file is removed. The `OperatorMetrics` and `OpTraceResult` dataclasses have been relocated to `src/winml/modelkit/session/monitor/op_metrics.py` and extended with per-sample statistics (`samples_us` + `avg_us` / `total_us` / `p90_us` / `sample_count` properties) and lifecycle reporting (`status: TraceStatus` and `error: str | None`). The `model` field is now `str | None` (was `str`).

## Diff metrics
- Lines deleted: 99
- Status: DELETED

## What this file did (pre-state)
Defined the structured profiling output schema used by `OpTracer.run()` to return results:
- `OperatorMetrics` dataclass with per-op fields organized into priorities P0 (temporal: `start_time_us`, `duration_us`, `percent_of_total`), P1 (roofline: `hardware_time_us`, `memory_time_us`, `dominant_path_us`), P2 (DMA: DRAM/VTCM read/write bytes), P3 (cache: `vtcm_hit_ratio`), plus identity (`name`, `op_path`, `op_id`) and context (`num_htp_ops`, `data_type`, `dims`) fields. `to_dict()` via `dataclasses.asdict`.
- `OpTraceResult` dataclass aggregating `model`, `device`, `tracing_level`, `operators: list[OperatorMetrics]`, EP metadata, ISO-8601 timestamp, sample count, `summary`, `statistics`, and `artifacts` dicts. `to_dict()` emits a nested `metadata` block; `to_json(indent=2)` wraps `json.dumps`.

## Public symbols (pre-deletion)
- `OperatorMetrics` — `@dataclass`, 16 fields, `to_dict()`.
- `OpTraceResult` — `@dataclass`, 11 fields, `to_dict()` + `to_json(indent=2)`.

## Where the functionality moved
| Pre-state symbol | Where it lives now |
|---|---|
| `OperatorMetrics` dataclass | `src/winml/modelkit/session/monitor/op_metrics.py` — same identity / P0 / P1 / P2 / P3 / context fields. **Added:** `samples_us: list[float] = field(default_factory=list)` (per-sample timings; empty when source parser produced aggregated avg only). |
| `OperatorMetrics.to_dict()` | Same — `dataclasses.asdict(self)`. Now also serializes the new `samples_us` list. |
| (new) `OperatorMetrics.sample_count` `@property` | New. `len(self.samples_us)`. |
| (new) `OperatorMetrics.avg_us` `@property` | New. Mean of `samples_us`, `0.0` when empty. |
| (new) `OperatorMetrics.total_us` `@property` | New. Sum of `samples_us`, `0.0` when empty. |
| (new) `OperatorMetrics.p90_us` `@property` | New. Inclusive 90th-percentile via `statistics.quantiles(..., n=10, method="inclusive")[8]`; returns `0.0` for empty list and `samples_us[0]` for single-element list (special-cased because `quantiles` requires `n ≥ 2`). |
| `OpTraceResult.model: str` | Now `model: str | None` — allows the monitor to construct a result without knowing the model name (the session owns that). |
| `OpTraceResult.device`, `tracing_level`, `operators`, `ep`, `tracing_backend`, `timestamp`, `num_samples`, `summary`, `statistics`, `artifacts` | Unchanged. |
| (new) `OpTraceResult.status: TraceStatus` | New, default `"ok"`. Closed set: `"ok"`, `"no_data"`, `"parse_failed"`, `"basic_fallback"`, `"not_run"`. Lets `QNNMonitor` report parse-time failures without raising. |
| (new) `OpTraceResult.error: str | None` | New, default `None`. Populated when `status == "parse_failed"`. |
| (new) `TraceStatus` `Literal` alias | New module-level type alias for the closed set above. Enforced statically by mypy/ruff; at runtime it is a plain `str` so JSON serialization is unaffected. |
| `OpTraceResult.to_dict()` | Same nested `metadata` / flat-arrays schema; **additive** top-level `status` and `error` keys appended. Existing consumers of `metadata`/`summary`/`operators`/`statistics`/`artifacts` are unaffected. |
| `OpTraceResult.to_json(indent=2)` | Unchanged. |

## Net behavior change
- The dataclass schema is a superset of the pre-state schema: every old field is present with the same name and type, except `model` widened from `str` to `str | None`.
- The JSON `to_dict()` output is a superset: nested `metadata`/`summary`/`operators`/`statistics`/`artifacts` are unchanged, plus two additive top-level keys (`status`, `error`). Existing consumers that key into the nested blocks won't notice; consumers that did `dict.keys()` introspection will see two extra keys.
- `OperatorMetrics` now exposes p90 / total / sample-count derived stats directly on the dataclass, eliminating the need for callers to do `statistics.quantiles` themselves.

## Risks
- `OpTraceResult.model: str` → `str | None` is technically a widening that breaks `Optional`-strict consumers (e.g. callers doing `result.model.lower()` without a `None` check). Production consumers in this repo (`report.py`) don't dereference `model`, but external code may.
- Reading the `samples_us` list on a result produced by the old CSV path returns an empty list (the legacy parser only emitted aggregated averages). Callers must guard against empty input or rely on the `0.0` fallback on the derived properties.
- The closed `TraceStatus` set is statically enforced — out-of-tree code that constructs `OpTraceResult(status="custom_state")` will fail type-check but not runtime.
