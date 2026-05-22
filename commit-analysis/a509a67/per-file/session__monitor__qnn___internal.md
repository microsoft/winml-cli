# src/winml/modelkit/session/monitor/qnn/_internal.py

## TL;DR
New private parser module consolidating the previous `qnn/csv_parser.py` + `qnn/qhas_parser.py` into one `_`-prefixed file per the v2.4 simplification (OQ-1 (b)). Architecture regression test at `tests/unit/architecture/test_qnn_imports.py` enforces information hiding (PRD NFR-8): only `_`-prefixed names may be imported from here, and only `qnn_monitor` is allowed the public-name import. **Key hardening: 19 bare `dict[key]` accesses are now routed through a `_require(d, key, context)` helper that raises a named `KeyError` with the missing key in the message, so SDK schema drift surfaces verbatim in the `basic_fallback` warning log** (commit body rounds this as "18 bare dict[key] accesses").

## Diff metrics
- Lines added / removed: **+443 / 0**
- New / modified: **new file**

## Role before vs after
- **Before**: Two separately public-ish files (`qnn/csv_parser.py`, `qnn/qhas_parser.py`) lived under `optracing/`, with their helpers importable by anyone.
- **After**: One private file; only `_`-prefixed names cross the package wall (CLAUDE.md testing exception); two public functions (`parse_qhas`, `parse_qnn_profiling_csv`) are re-exported through `qnn/__init__.py`. The private regex `_TOKEN_SUFFIX` is imported into `qnn_monitor.py` to share strip semantics with `_heuristic_op_type` (CLAUDE.md `_`-name exception).

## Symbol-level changes
Module-level:
- `_OP_PATTERN = re.compile(r"(.+?)(?:\s+|:)OpId_(\d+)\s*\(cycles\)")` — Event-Identifier parser for the CSV path.
- `_TOKEN_SUFFIX = re.compile(r"_token_\d+(?:_\d+)?")` — strips the QNN compiler's `_token_N` / `_token_N_M` suffix; **shared with `qnn_monitor.QNNMonitor._heuristic_op_type` so both paths use the same strip semantics**.

CSV-side helpers (basic-mode profiling):
- `_split_op_event_id(event_id) -> (op_type, op_path)` — heuristic split; whitespace strip + trailing-`/` fallback documented in docstring.
- `parse_qnn_profiling_csv(csv_path) -> dict` (public, re-exported) — returns `{metadata, operators, samples}`.
- `_read_csv(csv_path) -> list[dict[str, str]]`
- `_extract_metadata(rows) -> dict` — captures the **first** ROOT-row occurrence of `hvx_threads`, `accel_execute_cycles`, `accel_execute_us`.
- `_extract_samples(rows) -> list[list[dict]]` — splits NODE SUB-EVENT rows at each ROOT `Accelerator (execute) time (cycles)` boundary.
- `_parse_node_event(event_id, time_val) -> dict | None`
- `_aggregate_operators(samples) -> list[dict]` — averages cycles per `op_id`, attaches `samples_cycles` list, sorts desc; **leaves cycle-to-µs conversion to caller** because the ratio lives in ROOT metadata.

QHAS-side helpers (detail-mode roofline):
- **`_require(d: dict, key: str, context: str) -> Any`** — exact signature. Raises `KeyError(f"Required QHAS field {key!r} is missing in {context}")` when `key not in d`.
- `parse_qhas(qhas_data) -> dict` (public, re-exported) — returns `{summary, operators}`.
- `_extract_summary(data) -> dict` — renames raw QHAS keys (`time_us`, `graph_execute_us`, `total_dram_read`, ...) to user-facing renderer vocabulary (`inference_us`, `execute_us`, `dram_read_bytes`, ...) so `monitor.report._display_detail_report` reads them directly.
- `_transform_op(op, cycle_to_us) -> dict` — converts cycles → µs, computes `dominant_path_us`, surfaces DRAM/VTCM byte counters, strips `_TOKEN_SUFFIX` from `op_path` so QHAS path keys match clean ONNX `node.name` keys (production map keys are clean; QHAS `qnn_op` carries the suffix — without this strip the FR-14 L1 ONNX-primary lookup would be silently inert in detail mode).
- `_vtcm_ratio(op) -> float | None` — returns `None` when there's no read traffic.

## Behavior / contract changes — `_require` rollout (the called-out hardening)
**Exact `_require` helper signature:**
```python
def _require(d: dict, key: str, context: str) -> Any:
    if key not in d:
        raise KeyError(f"Required QHAS field {key!r} is missing in {context}")
    return d[key]
```

**Inventory of `_require` call sites (19 total, by call site count; commit body says 18):**

| # | Site | Key | Context label |
|---|---|---|---|
| 1 | `parse_qhas` | `"data"` | `"QHAS root"` |
| 2-15 | `_extract_summary` (context = `"htp_overall_summary row"`) | `"time_us"`, `"graph_execute_us"`, `"inf_per_s"`, `"timeline_cycles"`, `"percent_utilization"`, `"total_dram_read"`, `"total_dram_write"`, `"total_vtcm_read"`, `"total_vtcm_write"`, `"peak_vtcm_alloc"`, `"qnn_nodes"`, `"htp_nodes"`, `"unique_qnn_ops"`, `"unique_htp_ops"` | 14 keys |
| 16-19 | `_transform_op` (context = `"qnn_op_instances_nodes entry"`) | `"cycles"`, `"qnn_op_type"`, `"qnn_op"`, `"percent_active_cycles"` | 4 keys |

(Counted via Grep: 19 call sites + 1 definition. Commit body's "18" likely excludes the outermost `"data"` access in `parse_qhas`.)

Behavior change: previously each of these was a bare `d[k]` that raised a plain `KeyError: 'time_us'` (or whichever key). The outer `_try_qhas` catches `except Exception` and logs `"QHAS JSON parse failed: %s"` — so a schema drift would surface as `'time_us'` in the log with no indication of which row or which transform. Now the log carries `"Required QHAS field 'time_us' is missing in htp_overall_summary row"` — both the key and the structural context.

Other behavior aspects worth flagging:
- **`_extract_metadata` captures first occurrence only**: per docstring, the initial inference sample's metadata wins. Subsequent samples are ignored; this is by design but undocumented before.
- **`_aggregate_operators` keys by `op_id`**, not by name — identically-named ops at different graph positions are kept separate.
- **`_transform_op` strips `_token_*` from `op_path`**: critical correctness fix — without this, L1 ONNX-primary lookup in detail mode silently misses every row because production ONNX map keys are clean (`/encoder/conv1/Conv`) but QHAS `qnn_op` carries the compiler-injected suffix (`/encoder/conv1/Conv_token_1_2`). The strip is idempotent on already-clean strings.
- **`_split_op_event_id` warning in docstring**: leaf segment is the **ONNX op symbol** (`"Conv"`, `"Add"`), NOT the **QNN op type** (`"Conv2d"`, `"ElementWiseAdd"`). When the authoritative QNN op type is available (QHAS `qnn_op_type`), use it directly.

## Cross-file impact
- `qnn_monitor.py` imports `_TOKEN_SUFFIX` (private, the `_` exception) plus `parse_qhas`, `parse_qnn_profiling_csv` (public, via `from .qnn._internal import ...`).
- `qnn/__init__.py` re-exports the two public parsers.
- `tests/unit/architecture/test_qnn_imports.py` (PRD NFR-8) is the regression-test guard.
- Renamed QHAS summary fields ripple to `monitor/report.py::_display_detail_report` (renderer is the source of truth for user-facing names per `_extract_summary` docstring).

## Risks / subtleties
- `_require` raises `KeyError`, which `_try_qhas` catches as `Exception` — surfacing requires reading the WARNING log line. If anyone wraps `parse_qhas` outside `_try_qhas` they should preserve this catch-broadly-log-loudly pattern.
- `_require(d, k, ctx)` is annotated `d: dict` (not `Mapping`) — mypy-strict callers passing a dict-like (e.g. `mappingproxy`) might lint-warn.
- The "first occurrence wins" rule in `_extract_metadata` is silent: a QNN SDK that someday emits per-sample-varying metadata would have all but the first sample's values silently discarded.
- `_aggregate_operators` average across samples is arithmetic mean over `op_id`; outliers in a small `num_samples` skew it more than median would. Not a regression (matches legacy behaviour) but worth noting because `samples_cycles` is exposed downstream for p90 derivation — callers wanting robust central tendency should compute from the per-sample list themselves.
- `_TOKEN_SUFFIX` and `_OP_PATTERN` are compiled module-scope; safe across threads but module reload during pytest can produce duplicate regex objects (minor, not a correctness issue).

## Open questions / TODOs surfaced
- Per PRD-NFR8 the architecture regression test needs a `qnn_monitor`-specific whitelist for its non-`_` import of public names from `_internal`; the docstring at `qnn/__init__.py` describes the rule but the precise test allow-list isn't visible from this file.
- The count discrepancy (commit-body "18" vs actual 19 call sites) — recommend updating one or the other so review history matches code.
- The QHAS-key → renderer-key renaming (`time_us → inference_us`, etc.) is one-way and undocumented at the renderer side; if anyone needs to introspect raw QHAS fields they have to know both vocabularies.
- `_require` always raises `KeyError`; consider raising a typed `QhasSchemaError` so callers downstream of `parse_qhas` can disambiguate schema drift from caller error.
