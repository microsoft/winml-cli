# Review: `src/winml/modelkit/session/monitor/op_metrics.py`

**Status:** new file (relocated from `optracing/result.py`)
**Lines added/removed:** 168+ / 0-
**Diff command:** `git diff 1bea4cf..HEAD -- src/winml/modelkit/session/monitor/op_metrics.py`

## 1. Purpose of this file

Defines the two canonical output dataclasses for op-tracing: `OperatorMetrics` (per-operator profiling fields) and `OpTraceResult` (the top-level container). Also defines `TraceStatus`, a `Literal` type alias for the closed set of status strings. This file was relocated from `optracing/result.py` and extended with `status`/`error` fields and nullable `model` support per PRD SC-6 and FR-6.

## 2. Changes summary

- Relocated from `optracing/result.py` to `session/monitor/op_metrics.py`.
- Added `TraceStatus` literal alias with five values: `"ok"`, `"no_data"`, `"parse_failed"`, `"basic_fallback"`, `"not_run"`.
- Extended `OperatorMetrics` with `samples_us: list[float]` and computed properties `sample_count`, `avg_us`, `total_us`, `p90_us`.
- Relaxed `OpTraceResult.model: str` to `str | None` (SC per PRD SC-6 / FR-6).
- Added `status: TraceStatus = "ok"` and `error: str | None = None` fields to `OpTraceResult`.
- `OpTraceResult.to_dict()` extended with additive top-level `"status"` and `"error"` keys.
- Added `OpTraceResult.to_json()`.

## 3. Per-symbol review

### `TraceStatus`

- **Role:** Closed set of status values for `OpTraceResult.status`.
- **Signature:** `TraceStatus = Literal["ok", "no_data", "parse_failed", "basic_fallback", "not_run"]`
- **Behavior:** Statically enforced by mypy/ruff; at runtime `status` is still a plain `str`. The five values cover: clean parse, missing artifact, corrupt artifact, forced basic degradation, and pre-exit state.
- **Risks / concerns:** `"not_run"` is defined in the alias but `OpTraceResult.__init__` defaults to `"ok"` — a freshly constructed result without calling `__exit__` appears to have status `"ok"` rather than `"not_run"`. `QNNMonitor` sets `self._result = None` before `__exit__`, so accessing `monitor.result` pre-exit returns `None` (not an `OpTraceResult`), avoiding the confusion in practice. Still, if a consumer ever constructs an `OpTraceResult()` directly (e.g. in a test) and queries it before populating it, `"ok"` is misleading.
- **Tests:** `test_op_metrics.py::test_status_default_is_ok`, `test_status_can_be_set`, `test_trace_status_alias_importable`.

---

### `OperatorMetrics`

- **Role:** Per-operator profiling record.
- **Signature:** `@dataclass class OperatorMetrics:`
- **Behavior:** Stores operator identity (`name`, `op_path`, `op_id`), temporal localization (`duration_us`, `percent_of_total`), roofline fields (P1, detail-only), DMA traffic (P2, detail-only), cache efficiency (P3, derived), and per-sample timing list (`samples_us`). Properties compute `avg_us`, `total_us`, `p90_us`, and `sample_count` on demand.
- **Invariants:** `name` is the resolved op type, sourced by the v2.4 fallback chain (L1 ONNX, L2 EP-authoritative, L3 heuristic, L4 raw path). `op_path` is the framework path after `_token_N` strip. `samples_us` is empty when the source parser produced only an aggregated average (e.g. the QHAS path).
- **Risks / concerns:** `name` comment at `op_metrics.py:42-49` describes the four-layer fallback chain but is inside the class body as an inline comment attached to the field — it will not appear in `help(OperatorMetrics)` or tool-generated docs. Should be a docstring on the field or a class-level note. `samples_us` being empty vs populated creates a two-mode rendering contract (report.py uses `if op.samples_us` guards extensively) that is not enforced by the dataclass itself.
- **Tests:** `test_op_metrics.py`, `test_op_metrics_samples.py`.

---

### `OperatorMetrics.p90_us` (property)

- **Role:** Inclusive 90th-percentile latency of the per-sample timing list.
- **Signature:** `@property def p90_us(self) -> float:`
- **Behavior:** Uses `statistics.quantiles(n=10, method="inclusive")[8]` for `len >= 2`, returns the single sample directly for `len == 1`, and returns `0.0` for empty.
- **Invariants:** `statistics.quantiles` requires at least 2 data points when `n=10`; the `n == 1` branch handles the single-sample edge case.
- **Risks / concerns:** No risk; the implementation matches the spec comment in the code. The `[8]` index (90th percentile from 9 quantiles) is correct.
- **Tests:** `test_op_metrics_samples.py`.

---

### `OpTraceResult`

- **Role:** Top-level op-tracing result container.
- **Signature:** `@dataclass class OpTraceResult:`
- **Behavior:** Holds metadata, per-operator list, model-level summary, multi-sample statistics, raw artifact paths, and status/error. `to_dict()` serializes to the nested schema expected by `display_op_trace_report` and `write_op_trace_json`; it adds `"status"` and `"error"` at the top level additively.
- **Invariants:** The existing nested schema (`metadata`, `summary`, `operators`, `statistics`, `artifacts`) is preserved unchanged per PRD OOS-4 / FR-6. `model` is `str | None` — `None` serializes to JSON `null` cleanly via `json.dumps`.
- **Risks / concerns:** `statistics: dict[str, dict[str, float]]` and `summary: dict[str, Any]` are both plain dicts with no schema enforcement. As QHAS adds new metrics, callers may rely on specific keys that are absent in basic mode — no validation guards this. This is an acceptable tradeoff (schema-free for now, can add a typed summary class later).
- **Tests:** `test_op_metrics.py::test_to_dict_preserves_nested_schema`, `test_to_dict_adds_status_and_error_at_top_level`, `test_to_json_round_trip`, `test_model_field_accepts_none`.

---

### `OpTraceResult.to_dict()`

- **Role:** Serialization to the canonical nested dict structure.
- **Behavior:** Returns a two-level nested dict with additive `"status"` and `"error"` keys at the top level. Calls `op.to_dict()` for each operator (uses `dataclasses.asdict`).
- **Risks / concerns:** `dataclasses.asdict` recursively converts all nested dataclasses, including `OperatorMetrics`. This is correct, but if `OperatorMetrics` gains a non-serializable field (e.g. a `datetime` object) in the future, `asdict` will include it and `json.dumps` will raise `TypeError`. Acceptable at current complexity level.
- **Tests:** `test_op_metrics.py::test_to_dict_preserves_nested_schema`, `test_operator_metrics_to_dict_preserved`.

---

### `OpTraceResult.to_json()`

- **Role:** Convenience wrapper over `json.dumps(self.to_dict(), indent=2)`.
- **Behavior:** Produces a 2-space-indented JSON string. No custom encoder — relies on the standard types in `to_dict()` being JSON-serializable.
- **Tests:** `test_op_metrics.py::test_to_json_round_trip`.

## 4. Cross-cutting concerns

**Spec drift:** Matches PRD SC-6, FR-6 exactly. The `"not_run"` status value is defined in `TraceStatus` but the default `"ok"` on `OpTraceResult.__init__` means a freshly-constructed result has `status="ok"` rather than `"not_run"` — this is a minor deviation from the `TraceStatus` docstring intent (which lists `"not_run"` for the pre-`__exit__` state). In practice, the `QNNMonitor` pre-exit state is surfaced as `result = None`, not `OpTraceResult(status="not_run")`, so the value is unused.

**Information-hiding contract:** This module is public API. No internal imports from `qnn/_internal.py`. Used by `qnn_monitor.py`, `report.py`, `commands/perf.py`, and the session `__init__.py` re-export.

**Deferred work:** No TODO markers.

**EPDevice / ep_name:** Not referenced.

## 5. Confidence level

**High.** The dataclasses are straightforward, the status extension is additive, and the two-mode `samples_us` convention (populated vs empty) is well-documented inline. Test coverage is thorough.

## 6. Verbatim risk inventory

| Severity | Location | Description |
|----------|----------|-------------|
| Low | `op_metrics.py:122` | `OpTraceResult` defaults `status="ok"` rather than `"not_run"`, meaning a freshly-constructed, un-exited result appears successful. In practice `monitor.result` is `None` pre-exit, so this is not observable on the hot path — but a caller who constructs `OpTraceResult()` directly in a test and queries `status` before populating it will get a misleading `"ok"`. |
| Low | `op_metrics.py:42-49` | The four-layer fallback chain for `OperatorMetrics.name` is documented as inline comments on the field, not as part of the class docstring. This won't be surfaced by `help()` or autodoc tools. |
| Info | `op_metrics.py:133` | `statistics` field is `dict[str, dict[str, float]]` with no schema; key availability differs between basic and detail mode, and callers have no way to detect this statically. |
