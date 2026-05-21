# src/winml/modelkit/optracing/qnn/qhas_parser.py (DELETED)

## TL;DR
This file is removed. The QHAS JSON parser is relocated into the private `session/monitor/qnn/_internal.py` module alongside the CSV parser, with hardened error reporting (named-key `KeyError` via a new `_require` helper) and a refactored summary vocabulary that aligns with the renderer.

## Diff metrics
- Lines deleted: 113
- Status: DELETED

## What this file did (pre-state)
Parsed a QNN Hardware Acceleration Summary (QHAS) JSON dict into a `{summary, operators}` structure. It:
- Extracted the HTP overall summary row from `data.htp_overall_summary.data[0]` into a flat dict (`time_us`, `graph_execute_us`, `inf_per_s`, `timeline_cycles`, `percent_utilization`, `total_dram_read`, `total_dram_write`, `total_vtcm_read`, `total_vtcm_write`, `peak_vtcm_alloc`, `qnn_nodes`, `htp_nodes`, `unique_qnn_ops`, `unique_htp_ops`).
- Derived a `cycle_to_us` factor from `time_us / timeline_cycles`.
- For each entry in `data.qnn_op_instances_nodes.data`, built a per-op dict with `name`, `op_path`, `op_type`, `cycles`, `duration_us`, `percent_of_total`, `dominant_path_us`, `num_htp_ops`, DRAM read/write, VTCM read/write, and a derived `vtcm_hit_ratio = vtcm_read / (vtcm_read + dram_read)`.
- Treated `op["qnn_op"]` as both `name` and `op_path` (note: this conflated the framework path with the op type).

## Public symbols (pre-deletion)
- `parse_qhas(qhas_data: dict) -> dict` — the public entrypoint.
- `_extract_summary`, `_transform_op`, `_vtcm_ratio` — private helpers.

## Where the functionality moved
| Pre-state symbol | Where it lives now |
|---|---|
| `parse_qhas` | `src/winml/modelkit/session/monitor/qnn/_internal.py` (and re-exported from `session/monitor/qnn/__init__.py`). Same signature; same return shape `{summary, operators}` but with **renamed summary keys** (see below). Now uses the named-`KeyError` helper `_require(d, key, context)` so SDK schema drift surfaces the offending field name in the log instead of an opaque `KeyError`. |
| `_extract_summary` | `session/monitor/qnn/_internal.py`. **Summary keys renamed** to the user-facing renderer vocabulary so `session/monitor/report.py::_display_detail_report` reads them directly:<br>• `time_us` → `inference_us`<br>• `graph_execute_us` → `execute_us`<br>• `percent_utilization` → `utilization_pct`<br>• `total_dram_read` → `dram_read_bytes`<br>• `total_dram_write` → `dram_write_bytes`<br>• `total_vtcm_read` → `vtcm_read_bytes`<br>• `total_vtcm_write` → `vtcm_write_bytes`<br>• `peak_vtcm_alloc` → `vtcm_peak_bytes`<br>Other keys (`inf_per_s`, `timeline_cycles`, `qnn_nodes`, `htp_nodes`, `unique_qnn_ops`, `unique_htp_ops`) carry through unchanged. |
| `_transform_op` | `session/monitor/qnn/_internal.py`. **Behavior changed:**<br>• `name` is now `op["qnn_op_type"]` (the authoritative QNN op type, e.g. `"Conv2d"`) — *not* `op["qnn_op"]`. The docstring explicitly calls out that leaf-splitting `qnn_op` was wrong because the leaf is the ONNX op symbol (`"Conv"`), a different vocabulary from the QNN op type (`"Conv2d"`).<br>• `op_path` is now `_TOKEN_SUFFIX.sub("", op["qnn_op"])` — the QNN-compiler-injected `_token_\d+(?:_\d+)?` suffix is stripped so QHAS paths match the CSV path's strip semantics *and* match the clean ONNX `node.name` keys produced by `WinMLSession._build_op_type_map`. Without the strip, FR-14 L1 ONNX-primary lookup would silently miss in detail mode.<br>• The `op_type` field is removed from the returned dict (it's now redundant with `name`).<br>• Cycle-to-us factor is now derived from `summary["inference_us"] / timeline_cycles` (renamed from `time_us`). |
| `_vtcm_ratio` | `session/monitor/qnn/_internal.py` — unchanged. |
| (new) `_require(d, key, context) -> Any` | New helper. Raises a named `KeyError(f"Required QHAS field {key!r} is missing in {context}")` so the outer `QNNMonitor._try_qhas` `except Exception` handler logs *which* field went missing rather than a bare `KeyError: 'time_us'`. Makes SDK schema drift diagnosable. |

## Net behavior change
- Summary keys are renamed to the renderer's `_us` / `_bytes` / `_pct` vocabulary; downstream consumers reading the summary dict need the new keys.
- The `name` field on each per-op dict is the QHAS-authoritative `qnn_op_type`, not the framework-path leaf. This is the L2 layer of the v2.4 op-type fallback chain in `QNNMonitor._resolve_op_type` (L1=ONNX, L2=this, L3=heuristic, L4=raw).
- `op_path` is stripped of the QNN compiler's token suffix so the L1 ONNX-primary lookup in `QNNMonitor._resolve_op_type` actually hits — production map keys are clean (`/encoder/conv1/Conv`) but raw QHAS `qnn_op` carries `_token_1_2`, so without this strip L1 would silently miss and L2 would always win.
- The `op_type` field is gone from the returned dict (callers should read `name`).
- Missing-field errors are now diagnosable: schema drift surfaces the offending field name in WARNING log output (then falls back to basic mode), instead of bubbling an opaque `KeyError` that gets caught and dropped.

## Risks
- Any out-of-tree caller that consumed the *old* summary keys (`time_us`, `total_dram_read`, `peak_vtcm_alloc`, etc.) will silently see `KeyError` or `None` when reading the new dict. The renderer was updated in lockstep, but external readers need migration.
- Any out-of-tree caller that read the `op_type` field on the per-op dict will need to switch to `name`.
- Callers that depended on `op_path == qnn_op` (verbatim, with `_token_*` suffix) will get a different string — the stripped form. Downstream lookups against this string need to use the cleaned vocabulary.
