# src/winml/modelkit/session/monitor/op_metrics.py

## TL;DR
New file (168 lines) defining the structured profiling-output schema: `OperatorMetrics` dataclass, `OpTraceResult` aggregate dataclass, and a new `TraceStatus` `Literal` alias for the closed set of trace lifecycle states. Relocated from the deleted `optracing/result.py` and extended with `status` / `error` fields for failure reporting, computed `avg_us` / `total_us` / `p90_us` properties on `OperatorMetrics`, a `samples_us` list, and an inline comment documenting the L1→L4 op-type fallback chain.

## Diff metrics
- Lines added: 168
- Lines removed: 0
- New file; predecessor at `src/winml/modelkit/optracing/result.py` (99 lines) was deleted in the same commit.

## Role before vs after
- **Before:** `optracing/result.py` (99 lines) defined `OperatorMetrics` and `OpTraceResult` with the same identity/temporal/roofline/DMA/cache fields. No per-sample `samples_us`, no computed `avg_us`/`p90_us`/`total_us` properties, no `status`/`error` lifecycle fields, no `TraceStatus` alias. `OperatorMetrics.name` was documented narrowly as "QNN op type ('Conv2d', 'LayerNorm')".
- **After:** Canonical home for the op-tracing data schema under `session/monitor/`. Adds:
  - `samples_us: list[float]` per-op timings + derived `sample_count` / `avg_us` / `total_us` / `p90_us` properties.
  - Top-level `status: TraceStatus` (`"ok" | "no_data" | "parse_failed" | "basic_fallback" | "not_run"`) and `error: str | None`.
  - Docstring expansion on `OperatorMetrics.name` documenting the L1→L4 resolution chain (ONNX `node.op_type` → EP-authoritative QHAS `qnn_op_type` → heuristic leaf-split → raw `op_path`).
  - `model: str | None` (was non-optional `str`) — allows results from synthesized inputs without a file-backed model.

## Symbol-level changes
- **`TraceStatus`** — added (new)
  - `Literal["ok", "no_data", "parse_failed", "basic_fallback", "not_run"]`. Statically enforced; at runtime it's a plain `str` so JSON serialization is unaffected.
- **`OperatorMetrics`** — modified (relocated and extended)
  - Identity: `name`, `op_path`, `op_id` — unchanged field set; docstring rewritten for the new vocabulary.
  - P0/P1/P2/P3 fields: identical to the deleted version.
  - **New** `samples_us: list[float] = field(default_factory=list)`.
  - **New properties:** `sample_count` (= `len(self.samples_us)`), `avg_us` (mean, 0.0 when empty), `total_us` (sum), `p90_us` (inclusive 90-th percentile via `statistics.quantiles(..., n=10, method="inclusive")[8]`; special-cased for n=0 and n=1).
  - `to_dict()` unchanged (`asdict(self)`).
- **`OpTraceResult`** — modified (relocated and extended)
  - `model: str | None` (was `str`).
  - Existing metadata / summary / statistics / artifacts fields preserved.
  - **New** `status: TraceStatus = "ok"`, `error: str | None = None`.
  - `to_dict()` extended to emit additive top-level `status` and `error` keys.
  - `to_json(indent=2)` unchanged.
- **Imports** — `datetime.UTC` replaces `datetime.timezone.utc`; `Literal` added; `statistics` aliased as `_stats`.

## Behavior / contract changes
- Defines the cross-module data contract for op-tracing output. Anyone setting `monitor._result` must populate an `OpTraceResult`; anyone consuming `monitor.result` reads this dataclass.
- `status` defaults to `"ok"`. The `"not_run"` value is documented as the state pre-`__exit__` but the default is `"ok"`, so a monitor that fails to set status explicitly will misreport (silently). Subclasses must either initialize to `"not_run"` and flip on success, or explicitly assign one of the closed values.
- `to_dict()` is additive: older consumers parsing only `metadata`/`summary`/`operators`/`statistics`/`artifacts` are unaffected by the new top-level `status`/`error` keys.
- `OperatorMetrics.avg_us` is computed from `samples_us` only — does **not** fall back to `duration_us`. Renderers in `report.py` are expected to display `duration_us` when `samples_us` is empty.

## Cross-file impact
- **Used by which modules:** `monitor/report.py` (display/serialize), `monitor/ep_monitor.py` (forward-ref `TYPE_CHECKING` import for `WinMLEPMonitor.result` typing), `monitor/qnn_monitor.py` (populates via `self._result`), `commands/perf.py` and `commands/eval.py` (consume).
- **Depends on which modules:** stdlib only — `json`, `statistics`, `dataclasses`, `datetime`, `typing`.

## Risks / subtleties
- `p90_us` uses `statistics.quantiles(..., n=10, method="inclusive")[8]` — index 8 is the 9th of 9 cut points, the 90-th percentile. Documented inline. Special-cased for n=0/1 because `quantiles` requires n ≥ 2.
- `timestamp = datetime.now(UTC).isoformat()` — UTC, no `Z` suffix, includes microseconds.
- `to_dict` is **not** `asdict(self)` — it builds a hand-crafted nested dict (`metadata` block + flat `summary`/`operators`/`statistics`/`artifacts` + additive `status`/`error`). Round-trip via `OpTraceResult(**d)` is not what happens.
- `TraceStatus` is statically enforced only — runtime assignment of an arbitrary string passes through `to_dict()` unchanged.
- `OperatorMetrics.name` semantics changed silently: it's now the resolved op *type* (not unique), with `op_path` as the unique key. The L1→L4 fallback chain comment lives in a field comment, not a class-level docstring.

## Open questions / TODOs surfaced
- `OpTraceResult.status` default is `"ok"`, but the `TraceStatus` docstring describes `"not_run"` as the pre-`__exit__` state. Should the default be `"not_run"` so a monitor must explicitly transition to `"ok"`?
- No `from_dict` / `from_json` deserializer — round-trip is one-way only.
- `OperatorMetrics.name` resolution is delegated to `QNNMonitor._resolve_op_type` but the field is generic across monitors. For non-QNN op-tracing monitors (none yet exist) the contract is informal.

## Simplification opportunities
- **Asymmetry — `OperatorMetrics.to_dict()` is `asdict(self)` but `OpTraceResult.to_dict()` is hand-crafted.** This means `OpTraceResult.to_dict()` does not automatically pick up new fields when the dataclass grows — every new field needs a manual entry. Either (a) make both use `asdict` (simpler, looser schema) or (b) make both hand-crafted (stricter schema, more code). The current mix is a maintenance trap.
- The "additive" placement of `status`/`error` at the top level of the dict rather than inside `metadata` is documented as a back-compat decision; once internal consumers are updated, moving them under `metadata` would normalize the schema. Hard-break refactor opportunity per `MEMORY.md`.
- Inline comment on `OperatorMetrics.name` referencing `QNNMonitor._resolve_op_type` couples the schema to the QNN implementation. If a second op-tracing monitor materializes, this comment is the first thing that breaks.
- `samples_us: list[float] = field(default_factory=list)` + four derived properties (`sample_count`/`avg_us`/`total_us`/`p90_us`) is the right encapsulation, but downstream `report.py` is documented as handling `samples_us == []` by falling back to `duration_us`. If `avg_us` returned `duration_us` when samples are empty, that fallback could be inlined here and removed from `report.py` — one less place for the empty-samples special case to leak.
