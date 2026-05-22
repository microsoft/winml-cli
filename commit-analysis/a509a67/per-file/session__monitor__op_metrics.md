# src/winml/modelkit/session/monitor/op_metrics.py

## TL;DR
New file defining the structured profiling output schema: `OperatorMetrics` dataclass (per-op metrics with computed `avg_us`/`total_us`/`p90_us` properties), the `OpTraceResult` aggregate dataclass, and the `TraceStatus` Literal type alias for the closed set of trace lifecycle states. Relocated from the deleted `optracing/result.py` and extended with `status` / `error` fields for failure reporting.

## Diff metrics
- Lines added: 168
- Lines removed: 0
- New file (relocated content; the old `optracing/result.py` was deleted)

## Role before vs after
- **Before:** Logically equivalent dataclass lived in `src/winml/modelkit/optracing/result.py` (now deleted). No `status` / `error` lifecycle fields; the closed `TraceStatus` literal alias did not exist.
- **After:** Canonical home for the op-tracing data schema under `session/monitor/`. Adds top-level `status: TraceStatus` and `error: str | None` for explicit failure reporting. Both dataclasses are JSON-serializable via `to_dict()` / `to_json()`.

## Symbol-level changes
- **`TraceStatus`** — added (new)
  - `Literal["ok", "no_data", "parse_failed", "basic_fallback", "not_run"]` — closed set for `OpTraceResult.status`. Enforced statically by mypy/ruff; at runtime it is a plain `str` so JSON serialization is unaffected.
- **`OperatorMetrics`** — added (new dataclass)
  - Identity: `name`, `op_path`, `op_id`. Per the inline comment, `name` is the resolved op type chosen by QNNMonitor's L1→L4 fallback chain (ONNX `node.op_type` → EP-authoritative QHAS `qnn_op_type` → heuristic leaf-split → raw `op_path`).
  - P0 (temporal): `start_time_us`, `duration_us`, `percent_of_total`.
  - P1 (roofline, detail-only): `hardware_time_us`, `memory_time_us`, `dominant_path_us`.
  - P2 (DMA, detail-only per-op): `dram_read_bytes`, `dram_write_bytes`, `vtcm_read_bytes`, `vtcm_write_bytes`.
  - P3 (cache, detail-only derived): `vtcm_hit_ratio`.
  - Context: `num_htp_ops`, `data_type`, `dims`.
  - `samples_us: list[float]` — per-sample timings; empty when parser only produced aggregated avg.
  - `@property sample_count` — `len(self.samples_us)`.
  - `@property avg_us` — mean of samples, `0.0` when empty.
  - `@property total_us` — sum of samples, `0.0` when empty.
  - `@property p90_us` — inclusive 90th-percentile via `statistics.quantiles(..., n=10, method="inclusive")[8]`; returns `0.0` for empty list and `samples_us[0]` for single-element list (special-cased because `quantiles` requires n ≥ 2).
  - `to_dict()` — `dataclasses.asdict(self)`, preserves `None` for unavailable fields.
- **`OpTraceResult`** — added (new dataclass)
  - Required: `model: str | None`, `device: str`, `tracing_level: str` ("basic" or "detail"), `operators: list[OperatorMetrics]`.
  - Metadata: `ep: str = ""`, `tracing_backend: str = ""`, `timestamp` (default factory: `datetime.now(timezone.utc).isoformat()`), `num_samples: int = 0`.
  - Aggregates: `summary: dict[str, Any]`, `statistics: dict[str, dict[str, float]]`, `artifacts: dict[str, str]`.
  - Lifecycle (new vs. relocated source): `status: TraceStatus = "ok"`, `error: str | None = None`.
  - `to_dict()` — emits a nested `metadata` block plus flat `summary`/`operators`/`statistics`/`artifacts`, then additive top-level `status` and `error` keys.
  - `to_json(indent=2)` — `json.dumps(self.to_dict(), indent=indent)`.

## Behavior / contract changes
- Defines the cross-module data contract for op-tracing output. Anyone setting `monitor._result` must populate an `OpTraceResult`; anyone consuming `monitor.result` reads this dataclass.
- `status` defaults to `"ok"`. The `"not_run"` value is documented as the state pre-`__exit__` but the default is `"ok"`, so a monitor that fails to set status explicitly will misreport. (Subclasses are expected to either initialize to `"not_run"` and flip on success, or explicitly assign one of the closed values.)
- `to_dict()` is "additive": `status` and `error` are appended at top level alongside the nested `metadata` block — older consumers parsing only `metadata`/`summary`/`operators` are unaffected.
- `p90_us` semantics chosen to be inclusive (matches numpy/Excel default); the single-sample short-circuit avoids a `statistics.StatisticsError`.
- `avg_us` is computed from `samples_us` only — it does **not** fall back to `duration_us`. Renderers in `report.py` explicitly handle this by displaying `duration_us` when `samples_us` is empty.

## Cross-file impact
- **Used by which modules:** `report.py` (display/serialize), `ep_monitor.py` (forward-ref TYPE_CHECKING import for `EPMonitor.result` typing), `qnn_monitor.py` and other concrete op-tracing monitors (populate via `self._result`), `commands/perf.py` and `eval/evaluate.py` (consume).
- **Depends on which modules:** stdlib only — `json`, `statistics`, `dataclasses`, `datetime`, `typing`.

## Risks / subtleties
- `samples_us: list[float] = field(default_factory=list)` — mutable default handled correctly via `default_factory`, fine.
- `p90_us` calls `_stats.quantiles(..., n=10, method="inclusive")[8]` — index 8 is the 9th of 9 cut points, which is the 90th percentile. Documented inline. Note: for n=2 inputs this is still valid; the n==1 short-circuit is the only special case.
- `timestamp` uses `datetime.now(timezone.utc).isoformat()` — UTC, no `Z` suffix, includes microseconds; downstream consumers parsing should be ISO-8601 tolerant.
- `to_dict` does not preserve declaration-order parity with the dataclass — it builds a hand-crafted nested dict. Round-trip via `asdict()` then back through `OpTraceResult(**)` is **not** what happens; consumers see the nested `metadata` shape.
- `TraceStatus` is statically-enforced only. Runtime assignment of an arbitrary string would pass through `to_dict()` unchanged — there is no runtime validation.
- `OperatorMetrics.name` semantics are non-obvious: it is the resolved op *type*, not a unique identifier. The unique key is `op_path`. The fallback chain comment is the only documentation of the L1→L4 resolution rule and lives in the dataclass field comment, not in a class-level docstring.

## Open questions / TODOs surfaced
- `OpTraceResult.status` default is `"ok"`, but the `TraceStatus` docstring describes `"not_run"` as the pre-`__exit__` state. Should the default be `"not_run"` to enforce explicit transition? Currently any monitor that constructs an empty result and never sets status reports it as successful.
- No `from_dict` / `from_json` deserializer — round-trip is one-way only. Consumers that need to re-read written JSON files have to parse manually.
- `OperatorMetrics.name` resolution is delegated to `QNNMonitor._resolve_op_type` but the field is generic. For non-QNN op-tracing monitors (none yet) the contract is informal.
