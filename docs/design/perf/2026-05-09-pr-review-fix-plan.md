# Op-Tracing PR Review — Fix Plan

**Date:** 2026-05-09
**Branch:** feat/op-tracing-refactor
**Source:** comprehensive PR review (5 specialized reviewers) on commit 2c1b8232 + downstream
**Aggregate findings:** 6 Critical + 12 Important + Strengths
**Fact-check:** 5 confirmed, 0 false positives, 2 nuanced (one with two parts, both real but with caveats)

## Summary table

| ID    | Verdict     | Severity   | One-line                                                                 |
| ----- | ----------- | ---------- | ------------------------------------------------------------------------ |
| CRIT-1| CONFIRMED   | Blocker    | FR-14 ONNX-primary lookup is silently inert in QHAS detail mode          |
| CRIT-2| NUANCED     | Minor      | Empty-string short-circuit possible but trigger is rare in practice      |
| CRIT-3| CONFIRMED   | Important  | parse_existing_artifacts diverges from __exit__ on parse-failure contract|
| CRIT-4| CONFIRMED   | Important  | Architecture regression test under-enforces (two real gaps)              |
| CRIT-5| CONFIRMED   | Minor      | OperatorMetrics.name docstring stale; duration_us / avg_us dual-source   |
| CRIT-6| CONFIRMED   | Important  | _monitor_to_json_dict has zero direct tests + no error containment       |
| I-9   | CONFIRMED   | Blocker    | Detail-mode summary keys mismatch — summary line silently empty          |

## Confirmed issues (with evidence)

### CRIT-1: FR-14 ONNX-primary lookup is silently inert in QHAS detail mode

**Severity**: Blocker (correctness — silently disables a documented v2.4 invariant)
**Files**:
- `src/winml/modelkit/session/monitor/qnn/_internal.py:381` (raw `op_path` storage)
- `src/winml/modelkit/session/monitor/qnn_monitor.py:542` (lookup site)
- `src/winml/modelkit/session/session.py:490` (clean-key map producer)

**Reviewer source**: aggregate (CRIT-1)

**The bug**:
The CSV path strips `_TOKEN_SUFFIX` from `op_path` before storing it. The QHAS path does NOT — `_transform_op` stores `op_path = op["qnn_op"]` raw with the `_token_N_M` suffix intact. Production `_build_op_type_map` produces clean ONNX node names (no token suffix). Therefore in production detail mode, L1 lookup ALWAYS misses on QHAS rows, and `qnn_op_type` (e.g. `Conv2d`) wins instead of the ONNX `op_type` (e.g. `Conv`). The Phase-2 FR-14 contract ("ONNX has the last word") is silently never active in detail mode.

**Evidence**:

CSV path strips before storing (`qnn/_internal.py:225-227`):
```python
# Strip _token_\d+ suffixes inserted by the QNN compiler.
cleaned = _TOKEN_SUFFIX.sub("", raw_name)
name, op_path = _split_op_event_id(cleaned)
```
Both `name` and `op_path` are derived from the cleaned (stripped) string.

QHAS path stores raw (`qnn/_internal.py:379-381`):
```python
return {
    "name": op["qnn_op_type"],
    "op_path": op["qnn_op"],   # <-- raw, NO token strip
    ...
}
```

Lookup site (`qnn_monitor.py:540-542`):
```python
operators = [
    OperatorMetrics(
        name=self._resolve_op_type(op["op_path"], ep_authoritative=op["name"]),
        ...
```
Passes the token-bearing `op_path` as the L1 lookup key.

L1 implementation (`qnn_monitor.py:306-307`):
```python
if op_path in self._onnx_op_types:
    return self._onnx_op_types[op_path]
```

Production map producer (`session.py:489-490`):
```python
model = _onnx.load(str(onnx_path), load_external_data=False)
return {n.name: n.op_type for n in model.graph.node if n.name}
```
Keys are clean ONNX `node.name`, with no `_token_N_M` suffix.

QHAS fixture confirms 8 of 10 paths carry `_token_N_M` (`tests/unit/session/monitor/qnn/fixtures/qhas_resnet50.json`):
```
/resnet/embedder/embedder/convolution/Conv_token_1_2
/resnet/embedder/pooler/MaxPool_token_7_2
Transpose_token_328_2
...
```

The unit test `test_qhas_path_uses_onnx_op_type_when_map_populated`
(`tests/unit/session/monitor/test_qnn_monitor.py:588-641`) injects
`{first_path: "Conv"}` where `first_path` is the raw token-bearing
QHAS string. The test passes because the artificial fixture key
matches the artificial lookup key — but this construction is NOT what
production ever passes through `_build_op_type_map`. The test masks
the production-realistic miss.

Note: `_TOKEN_SUFFIX = r"_token_\d+(?:_\d+)?"` does NOT match the
`Add_3` shape (paths like `/resnet/encoder/stages.0/layers.0/Add_3`).
For those, ONNX `node.name` typically also lacks the `_3` suffix in
typical exports — so even after fixing the `_token_*` mismatch, the
`Add_3`-style paths are still a residual case. Acknowledge in fix.

**Fix**:

Choose one of two approaches; option A is preferred because it keeps the strip semantics co-located with the parser:

**Option A (preferred)**: strip in `_transform_op`. The strip is idempotent on already-clean strings (`_TOKEN_SUFFIX.sub("", "Conv")` returns `"Conv"`).
```python
# qnn/_internal.py::_transform_op
return {
    "name": op["qnn_op_type"],
    "op_path": _TOKEN_SUFFIX.sub("", op["qnn_op"]),
    ...
}
```

**Option B**: strip at lookup time inside `_resolve_op_type`.
```python
# qnn_monitor.py::_resolve_op_type
def _resolve_op_type(self, op_path: str, ep_authoritative: str | None = None) -> str:
    cleaned = _TOKEN_SUFFIX.sub("", op_path)
    if cleaned in self._onnx_op_types:
        return self._onnx_op_types[cleaned]
    ...
```

Option A also preserves the cleaned `op_path` as the rendered framework
path (matches the CSV path's behavior so the two modes have parity).

**Test additions**:

1. New test in `tests/unit/session/monitor/test_qnn_monitor.py` that
   simulates the production wiring: build a clean-name `onnx_op_types`
   map (no `_token_*` suffix in keys) and assert L1 wins for
   token-suffixed QHAS rows.
2. Strengthen `test_qhas_path_uses_onnx_op_type_when_map_populated` to
   inject the *cleaned* path as the dict key:
   ```python
   from winml.modelkit.session.monitor.qnn._internal import _TOKEN_SUFFIX
   first_path_clean = _TOKEN_SUFFIX.sub("", parsed["operators"][0]["op_path"])
   monitor.set_onnx_op_types({first_path_clean: "Conv"})
   ```
3. Document in the test docstring that production keys are always
   clean (no token suffix), to lock the contract.
4. Acknowledge `_3`-style residuals as out-of-scope for the
   `_TOKEN_SUFFIX` regex; track via I-list if it ever bites.

**Estimated effort**: S (2-3 lines of code, 1-2 new tests, 1 strengthened test)

---

### CRIT-3: parse_existing_artifacts error contract diverges from __exit__

**Severity**: Important (API consistency — surprises offline-analysis callers)
**File**: `src/winml/modelkit/session/monitor/qnn_monitor.py:251-263, 367-385`

**Reviewer source**: aggregate (CRIT-3)

**The bug**:
`__exit__` wraps `_parse_artifacts` in try/except and produces an
`OpTraceResult(status="parse_failed", error=str(exc))` for any
parse-time exception. `parse_existing_artifacts` calls
`_parse_artifacts` directly with no wrapper, so the same exception
propagates to the caller as a raw `Exception`. Two different error
contracts for the same parsing logic — callers using offline analysis
(e.g. CI post-processing) need to add their own try/except wrapper
specifically for this entrypoint.

**Evidence**:

`__exit__` wraps (lines 251-263):
```python
def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    """Parse whatever artifacts are on disk. Never suppresses caller exceptions."""
    try:
        self._result = self._parse_artifacts()
    except Exception as exc:
        logger.warning("QNNMonitor: artifact parse failed: %s", exc)
        self._result = self._make_failure_result(status="parse_failed", error=str(exc))
```

`parse_existing_artifacts` does not (lines 379-385):
```python
qhas_path = artifacts.get("qhas")
result = instance._parse_artifacts(qhas_override=Path(qhas_path) if qhas_path else None)
# M-2 carry-forward: leave the constructed instance internally
# consistent so callers that hold onto it (e.g. via a wrapper) see
# the parsed result via the typed accessor instead of None.
instance._result = result
return result
```

**Fix**:
Wrap the parse call in `parse_existing_artifacts` with the same
try/except pattern, returning a `parse_failed` `OpTraceResult` on
exception. Extract the wrapping into a private helper to enforce a
single error-handling implementation:

```python
def _parse_artifacts_safe(self, qhas_override: Path | None = None) -> OpTraceResult:
    try:
        return self._parse_artifacts(qhas_override=qhas_override)
    except Exception as exc:
        logger.warning("QNNMonitor: artifact parse failed: %s", exc)
        return self._make_failure_result(status="parse_failed", error=str(exc))
```

Then both `__exit__` and `parse_existing_artifacts` call
`_parse_artifacts_safe`. Single source of truth.

**Test additions**:
- Test that `parse_existing_artifacts` returns
  `OpTraceResult(status="parse_failed")` (rather than raising) when
  given a corrupt CSV / QHAS JSON.

**Estimated effort**: S

---

### CRIT-4: Architecture regression test under-enforces (two gaps)

**Severity**: Important (defense-in-depth — the test exists but doesn't enforce what its docstring claims)
**File**: `tests/unit/architecture/test_qnn_imports.py`

**Reviewer source**: aggregate (CRIT-4)

#### Gap 1: scope is `src/` only — test files reach in for non-`_`-prefixed names

**Evidence**:
Scope (line 44):
```python
src_root = pathlib.Path(__file__).parents[3] / "src" / "winml" / "modelkit"
```

Tests at `test_qhas_parser.py:10` and `test_csv_parser.py:9` directly
import `parse_qhas` and `parse_qnn_profiling_csv` (no `_` prefix —
look like public API) from `qnn._internal`:
```
$ grep -rn "from.*qnn._internal" tests/
tests/unit/session/monitor/qnn/test_qhas_parser.py:10:    from winml.modelkit.session.monitor.qnn._internal import parse_qhas
tests/unit/session/monitor/qnn/test_csv_parser.py:9:     from winml.modelkit.session.monitor.qnn._internal import parse_qnn_profiling_csv
tests/unit/session/monitor/qnn/test_csv_parser_samples.py:19: from winml.modelkit.session.monitor.qnn._internal import _aggregate_operators
tests/unit/session/monitor/qnn/test_event_id_split.py:23: from winml.modelkit.session.monitor.qnn._internal import _split_op_event_id
tests/unit/session/monitor/test_qnn_monitor.py:543: from winml.modelkit.session.monitor.qnn._internal import parse_qhas
tests/unit/session/monitor/test_qnn_monitor.py:605: from winml.modelkit.session.monitor.qnn._internal import parse_qhas
tests/unit/session/monitor/test_qnn_monitor_parse_existing.py:51: from winml.modelkit.session.monitor.qnn._internal import parse_qnn_profiling_csv
```

Per `CLAUDE.md` (project rules), test code may import from internal
submodules **only for `_`-prefixed private symbols**.
`parse_qhas` and `parse_qnn_profiling_csv` are non-prefixed public
names living in a private module, then imported by tests. This is the
combination CLAUDE.md does NOT bless. The architecture test's `src/`
scope misses it.

#### Gap 2: `from .qnn import _internal` form is missed by the AST detector

**Evidence**:
Detector (lines 25-39):
```python
def _is_internal_import(node: ast.AST) -> bool:
    if (
        isinstance(node, ast.ImportFrom)
        and node.module is not None
        and (node.module.endswith("qnn._internal") or node.module.endswith(".qnn._internal"))
    ):
        return True
    if isinstance(node, ast.Import):
        return any(alias.name.endswith("qnn._internal") for alias in node.names)
    return False
```

AST verification:
```
'from .qnn import _internal':            ImportFrom module='qnn'   level=1 names=['_internal']
'from .qnn._internal import parse_qhas': ImportFrom module='qnn._internal' level=1 names=['parse_qhas']
```

`from .qnn import _internal` parses as
`ImportFrom(module="qnn", names=[alias(name="_internal")])`.
The detector's `node.module.endswith("qnn._internal")` is `False`
(module is just `"qnn"`); the `ast.Import` branch only fires for
`import X` form, not `from X import Y`. Consequently any rogue module
that wrote `from .qnn import _internal` (and accessed
`_internal.parse_qhas`) would slip through the detector.

**Fix**:

For Gap 1, expand scope to also walk `tests/`. The architecture test
should accept `_`-prefixed-name imports from `_internal` (per
CLAUDE.md) but flag non-prefixed names. Two options:
1. Re-export the public names through `qnn/__init__.py`, then ban
   direct `_internal` imports for the non-prefixed names. Cleanest
   fix because it preserves the information-hiding boundary while
   giving tests a stable surface.
2. Leave the names where they are but rename to `_parse_qhas` /
   `_parse_qnn_profiling_csv`. Forces the prefix-matches-privacy
   convention.

Recommend option 1: it removes the architectural smell without
changing the parser API.

For Gap 2, extend the detector to also flag `ImportFrom` whose name
list contains `_internal`:
```python
def _is_internal_import(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        # from .qnn._internal import X
        if node.module is not None and (
            node.module.endswith("qnn._internal")
            or node.module.endswith(".qnn._internal")
        ):
            return True
        # from .qnn import _internal
        if node.module is not None and (
            node.module == "qnn"
            or node.module.endswith(".qnn")
            or node.module.endswith("qnn")
        ):
            if any(alias.name == "_internal" for alias in node.names):
                return True
    if isinstance(node, ast.Import):
        return any(alias.name.endswith("qnn._internal") for alias in node.names)
    return False
```

**Test additions**:
- Add a synthetic-AST unit test (no filesystem) that feeds each
  malformed-import string to `_is_internal_import` and asserts the
  detector returns `True`. Include `from .qnn import _internal`,
  `from winml.modelkit.session.monitor.qnn import _internal`, and
  `import winml.modelkit.session.monitor.qnn._internal`.
- After expanding scope to `tests/`, expect the test to fail until
  the public-name re-export from `qnn/__init__.py` is in place.

**Estimated effort**: M (re-export + import rewrite + detector
strengthen + new tests)

---

### CRIT-5 (Part A): OperatorMetrics.name docstring is stale

**Severity**: Minor (correctness — but only the docstring; behaviour is correct)
**File**: `src/winml/modelkit/session/monitor/op_metrics.py:41`

**Reviewer source**: aggregate (CRIT-5)

**The bug**:
Line 41 says:
```python
name: str  # QNN op type ("Conv2d", "LayerNorm")
```
After v2.4 FR-14, `name` is the **resolved** op type from
`_resolve_op_type`. When the ONNX map is populated and the path
matches, `name` is the ONNX `op_type` (e.g. `"Conv"`, `"Add"`,
`"MaxPool"`) — a different vocabulary from the QNN op type. The
docstring still claims it's the QNN op type, which is wrong post-v2.4.

**Fix**:
Update the comment to reflect the v2.4 contract:
```python
name: str  # Resolved op type. Vocabulary depends on the v2.4 fallback
           # chain in QNNMonitor._resolve_op_type:
           # L1: ONNX node.op_type (e.g. "Conv", "Add", "MaxPool")
           # L2: EP-authoritative (e.g. QHAS qnn_op_type "Conv2d")
           # L3: heuristic leaf-split
           # L4: raw op_path
op_path: str  # Framework path ("/layer1/conv/Conv")
```

**Test additions**: none required — docstring fix.

**Estimated effort**: S

---

### CRIT-5 (Part B): duration_us / avg_us dual-source-of-truth

**Severity**: Minor (structural smell — currently in sync but no enforced invariant)
**Files**:
- `src/winml/modelkit/session/monitor/op_metrics.py:47, 79-81`
- `src/winml/modelkit/session/monitor/report.py:236`

**The smell**:
`OperatorMetrics.duration_us` is a stored field (legacy aggregate),
`avg_us` is a property derived from `samples_us`. The two are
populated from different code paths:
- CSV path: `duration_us = op["cycles"] * cycle_to_us`,
  `samples_us = [c * cycle_to_us for c in samples_cycles]`
- QHAS path: `duration_us = op["duration_us"]` (QHAS-aggregate),
  `samples_us = [op["duration_us"]]` (single-element)

There is no enforced invariant tying them. The renderer
(`report.py:236`) treats them as equivalent:
```python
avg_str = f"{op.avg_us:,.1f}" if op.samples_us else f"{op.duration_us:,.1f}"
```
Currently they happen to be equal in both paths, so this works — but
no test pins the invariant. A future refactor that diverges them will
silently change rendered output depending on whether `samples_us` is
populated.

**Fix** (defer to follow-up; document the intent here):

Either:
1. Make `duration_us` a derived property like `avg_us`. Removes the
   storage and the divergence risk.
2. Add an assertion in `__post_init__` that
   `samples_us == [] or abs(duration_us - sum(samples_us)/len(samples_us)) < 1e-6`.
3. Leave the code, add a regression test that pins
   `duration_us == avg_us` in both CSV and QHAS paths.

Recommend option 3 for this PR (lowest-risk), follow up with option 1
in a separate refactor.

**Test additions**:
- Test pinning `duration_us == avg_us` for ops produced by both
  CSV and QHAS paths.

**Estimated effort**: S (test only); follow-up M for option 1 refactor.

---

### CRIT-6 (Part A): _monitor_to_json_dict has zero direct tests

**Severity**: Important (coverage gap — three branches uncovered)
**File**: `src/winml/modelkit/commands/perf.py:55-79`

**Reviewer source**: aggregate (CRIT-6)

**Evidence**:
```
$ grep -rn "_monitor_to_json_dict" tests/
(no results)
```
The dispatch helper at lines 55-79 has three branches (typed `result`,
transitional `to_dict()`, fallthrough `{}`) and zero direct unit
tests. Coverage is incidental at best (whatever the higher-level CLI
tests happen to exercise).

**Fix**:
Add a unit test module
`tests/unit/commands/test_monitor_to_json_dispatch.py` that exercises
each branch with a fake monitor:
1. Op-tracing monitor (`result` returns an `OpTraceResult` with
   non-empty operators) — assert `to_dict` carries the nested schema.
2. Proof-of-execution monitor (`result` is `None`, has `to_dict()`)
   — assert it falls through to the transitional path.
3. Null monitor (`result` is `None`, no `to_dict()`) — assert it
   returns `{}`.

**Test additions**: see fix.

**Estimated effort**: S

### CRIT-6 (Part B): _monitor_to_json_dict has no error containment

**Severity**: Important (CLI robustness — runaway exception during JSON write)
**File**: `src/winml/modelkit/commands/perf.py:55-79`

**Evidence**:
No try/except around `result.to_dict()` or `monitor.to_dict()`
calls. Both calls dispatch into monitor-owned code; a regression in
any monitor's serializer would crash the entire `wmk perf` invocation
mid-output.

**Fix**:
Wrap both dispatch points and degrade gracefully on exception.
Logging at WARNING preserves diagnosability:

```python
def _monitor_to_json_dict(monitor: EPMonitor) -> dict[str, Any]:
    try:
        result = monitor.result
        if result is not None:
            return result.to_dict()
        if hasattr(monitor, "to_dict"):
            return monitor.to_dict()
    except Exception as exc:
        logger.warning(
            "Monitor JSON serialization failed for %s: %s",
            type(monitor).__name__,
            exc,
        )
        return {"error": f"monitor_serialization_failed: {exc}"}
    return {}
```

**Test additions**:
- Test that a monitor whose `to_dict()` raises still returns a dict
  (with an `error` key) rather than propagating.

**Estimated effort**: S

---

### I-9: Detail-mode summary keys mismatch

**Severity**: Blocker (correctness — summary line silently empty in production)
**Files**:
- `src/winml/modelkit/session/monitor/qnn/_internal.py:330-345` (parser produces these keys)
- `src/winml/modelkit/session/monitor/report.py:173-194` (renderer reads different keys)

**Reviewer source**: aggregate (I-9)

**The bug**:
Complete key disjoint between parser output and renderer input. The
detail-mode summary line(s) silently render empty for real production
data, even when QHAS produced a fully-populated summary.

**Evidence**:

Parser produces (`qnn/_internal.py:330-345`):
```python
return {
    "time_us": raw["time_us"],
    "graph_execute_us": raw["graph_execute_us"],
    ...
    "total_dram_read": raw["total_dram_read"],
    "total_dram_write": raw["total_dram_write"],
    ...
    "peak_vtcm_alloc": raw["peak_vtcm_alloc"],
    ...
}
```

Renderer reads (`report.py:173-194`):
```python
inf_us = summary.get("inference_us")     # parser produces "time_us"
exe_us = summary.get("execute_us")       # parser produces "graph_execute_us"
util   = summary.get("utilization_pct")  # MATCH (parser produces "utilization_pct")
dram_r = summary.get("dram_read_bytes")  # parser produces "total_dram_read"
dram_w = summary.get("dram_write_bytes") # parser produces "total_dram_write"
vtcm_peak = summary.get("vtcm_peak_bytes") # parser produces "peak_vtcm_alloc"
```

Five of six summary fields silently render empty. (Only
`utilization_pct` matches.)

**Fix**:

Pick one schema and make both ends use it. The renderer's names
(`inference_us`, `execute_us`, `dram_read_bytes`, `dram_write_bytes`,
`vtcm_peak_bytes`) are more user-facing — units are explicit, prefix
indicates aggregate. Suggest renaming in the parser (single source of
truth, no semver concerns since `_internal` is private):

```python
# qnn/_internal.py::_extract_summary
return {
    "inference_us": raw["time_us"],
    "execute_us": raw["graph_execute_us"],
    "inf_per_s": raw["inf_per_s"],
    "timeline_cycles": raw["timeline_cycles"],
    "utilization_pct": raw["percent_utilization"],
    "dram_read_bytes": raw["total_dram_read"],
    "dram_write_bytes": raw["total_dram_write"],
    "vtcm_read_bytes": raw["total_vtcm_read"],
    "vtcm_write_bytes": raw["total_vtcm_write"],
    "vtcm_peak_bytes": raw["peak_vtcm_alloc"],
    "qnn_nodes": raw["qnn_nodes"],
    "htp_nodes": raw["htp_nodes"],
    "unique_qnn_ops": raw["unique_qnn_ops"],
    "unique_htp_ops": raw["unique_htp_ops"],
}
```

**Test additions**:
- Snapshot test of `_extract_summary` against a fixture, asserting
  the exact key set the renderer expects.
- End-to-end test rendering a QHAS-derived `OpTraceResult` and
  asserting the summary lines contain non-empty `Inference:`,
  `Execute:`, `DRAM:`, `VTCM:` substrings.

**Estimated effort**: S (rename keys, add tests)

---

## Nuanced findings

### CRIT-2: Empty-string short-circuit in _resolve_op_type chain

**Verdict**: NUANCED — bug is real but trigger is rare in practice.

**Severity if confirmed**: Minor

**Files**:
- `src/winml/modelkit/session/session.py:490` (filter only `n.name`)
- `src/winml/modelkit/session/monitor/qnn_monitor.py:306` (membership-not-truthy check)

**Reviewer source**: aggregate (CRIT-2)

**Evidence**:
`_build_op_type_map` filters only `n.name`, not `n.op_type`:
```python
return {n.name: n.op_type for n in model.graph.node if n.name}
```
`_resolve_op_type` uses `in` (membership), not truthy check:
```python
if op_path in self._onnx_op_types:
    return self._onnx_op_types[op_path]
```

Confirmed at the protobuf level: ONNX allows `op_type=""` in the
NodeProto schema (the field is required by spec but accepted by
`onnx.load` without explicit checker). So an empty-`op_type` node
produced by some upstream tool would propagate `""` as
`OperatorMetrics.name`.

**Why nuanced**:
- Real ONNX checker rejects empty `op_type`. Models that pass
  `onnx.checker.check_model` cannot trigger this.
- Any reasonable ONNX exporter (Optimum, torch.onnx.export, manual
  graph builders) sets `op_type` to a real op name.
- The trigger requires a malformed ONNX file — itself an integrity
  problem the user would notice via other failures.

**Fix** (defensive, low-cost):
Add the truthy check in both places:

```python
# session.py
return {n.name: n.op_type for n in model.graph.node if n.name and n.op_type}
```

```python
# qnn_monitor.py::_resolve_op_type
mapped = self._onnx_op_types.get(op_path)
if mapped:  # truthy: not None, not empty string
    return mapped
if ep_authoritative:
    return ep_authoritative
return self._heuristic_op_type(op_path) or op_path
```

The combined defense (both filter and truthy check) means the
empty-string can never propagate even if the filter misses.

**Test additions**:
- Synthetic test: build an `onnx_op_types` dict with a `""` value
  and assert `_resolve_op_type` falls through to L2 / L3 / L4.

**Estimated effort**: S

---

## Rejected findings (false positives)

None. All seven findings are real bugs or real architectural smells.
Two (CRIT-2 and CRIT-5 Part B) are scoped down to nuanced/structural
rather than active regressions.

---

## Recommended fix order

Priority-ranked. P1 fixes are blockers; merge cannot land without
them. P2 fixes are important but not user-visible regressions in
common paths. P3 fixes are cleanups.

**P1 — Blockers (must fix before merge)**:
1. **CRIT-1** — FR-14 silently inert in QHAS detail mode.
   Highest-priority because the v2.4 design intent is being
   silently violated; tests pass but production behaviour is wrong.
2. **I-9** — Detail-mode summary keys mismatch. Highly visible to
   users running `wmk perf --op-tracing detail`; the summary line
   silently shows empty, which is a worse UX than no summary at all
   because it implies the data is missing.

**P2 — Important (should fix before merge)**:
3. **CRIT-3** — `parse_existing_artifacts` divergent error contract.
   Trivial fix; not user-visible today but will surface as soon as
   anyone uses offline analysis on a real corrupt artifact.
4. **CRIT-6 Part B** — `_monitor_to_json_dict` no error containment.
   Defense-in-depth; matters when CLI is automated in CI.
5. **CRIT-4** — Architecture test under-enforces. Trivial fix; lets
   us claim the boundary is enforced.
6. **CRIT-6 Part A** — `_monitor_to_json_dict` no direct tests.
   Coverage gap; matters less if Part B is fixed but still wanted.

**P3 — Cleanup (optional this PR / can carry-forward)**:
7. **CRIT-5 Part A** — Stale docstring. Comment fix.
8. **CRIT-5 Part B** — `duration_us / avg_us` dual-source.
   Add a regression test only; structural refactor (option 1) goes
   to follow-up.
9. **CRIT-2** — Empty-`op_type` short-circuit. Defensive only;
   trigger is unlikely.

---

## Bundling recommendation

Package commits to keep semantics clean and review boundaries crisp.

- **Commit A (P1: correctness blockers)**:
  - CRIT-1 fix — strip `_TOKEN_SUFFIX` in `_transform_op` (or in
    `_resolve_op_type`); strengthen the existing test to use
    production-realistic clean keys; add a new test that simulates
    the production wiring end-to-end.
  - I-9 fix — rename keys in `_extract_summary`; add the snapshot
    test.
  - These two fixes are independent but both about
    "parser/renderer contract drift" — landing them together makes
    the design intent obvious in the diff.

- **Commit B (P2: API + defense-in-depth)**:
  - CRIT-3 fix — extract `_parse_artifacts_safe`, route both
    `__exit__` and `parse_existing_artifacts` through it.
  - CRIT-6 Part B fix — try/except wrapper.
  - Both are small error-handling consolidations.

- **Commit C (P2: architecture enforcement)**:
  - CRIT-4 Gap 2 fix — extend AST detector.
  - CRIT-4 Gap 1 fix — re-export public names through
    `qnn/__init__.py`; rewrite test imports; expand detector scope
    to `tests/`.
  - Both architectural; landing together keeps the boundary
    consistent.

- **Commit D (P2: coverage)**:
  - CRIT-6 Part A — three direct tests for `_monitor_to_json_dict`.

- **Commit E (P3: cleanup)**:
  - CRIT-5 Part A — docstring fix.
  - CRIT-5 Part B — regression test only (`duration_us == avg_us`).
  - CRIT-2 — defensive truthy filter and check.

---

## Carry-forward to follow-up PRs

- **CRIT-5 Part B refactor (option 1)** — make `duration_us` a
  derived property, remove the storage. Removes the divergence risk
  permanently. Defer to a focused refactor PR because it touches
  more sites than this fix-plan covers.
- **`_3`-style residual paths** — `_TOKEN_SUFFIX` doesn't cover the
  trailing-`_N` shape that QHAS occasionally emits for non-token
  paths (e.g. `Add_3`). Track via I-list; only re-engage if a
  realistic ONNX model is reported with this pattern in production.
- **Public re-exports for parser API** — if option 1 of CRIT-4 Gap
  1 is taken (re-export through `qnn/__init__.py`), then the
  parser's surface becomes part of the documented public API. May
  warrant a follow-up to add docstrings and `__all__` to
  `qnn/__init__.py`.
