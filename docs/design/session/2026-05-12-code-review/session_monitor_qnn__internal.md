# Review: `src/winml/modelkit/session/monitor/qnn/_internal.py`

**Status:** new file (consolidates deleted `csv_parser.py` + `qhas_parser.py`)
**Lines added/removed:** 427+ / 0-
**Diff command:** `git diff 1bea4cf..HEAD -- src/winml/modelkit/session/monitor/qnn/_internal.py`

## 1. Purpose of this file

Private implementation module for QNN op-tracing parsers. Consolidates the previously public `qnn/csv_parser.py` and `qnn/qhas_parser.py` into a single private submodule per the v2.4 simplification (spec §3.2 / coreloop §4.3 / OQ-1 resolution option b). Exports `parse_qnn_profiling_csv`, `parse_qhas`, and `_TOKEN_SUFFIX` (the last of which is consumed by `QNNMonitor._heuristic_op_type()`). All other helpers are module-private.

## 2. Changes summary

- New file; consolidation of `csv_parser.py` and `qhas_parser.py`.
- Added `_TOKEN_SUFFIX` as a named module-level constant (previously inline in the CSV parser).
- `_split_op_event_id` is now module-private (was public in `csv_parser.py`).
- `_parse_node_event` now strips `_TOKEN_SUFFIX` before calling `_split_op_event_id`, matching the v2.4 strip-then-lookup requirement (FR-15).
- `_aggregate_operators` now carries `samples_cycles` per-op for downstream `samples_us` conversion.
- `_transform_op` now strips `_TOKEN_SUFFIX` from `qnn_op` (the QHAS framework path) to enable FR-14 L1 ONNX lookup in detail mode.
- `_extract_metadata` uses `first-occurrence` capture (captures only the first ROOT sample's metadata, which is correct for a multi-sample CSV).
- `parse_qhas` now requires a structured `{"data": {...}}` wrapper (hard key access at line 311).

## 3. Per-symbol review

### `_OP_PATTERN`

- **Role:** Regex for parsing the CSV Event Identifier field into `(raw_name, op_id)`.
- **Value:** `re.compile(r"(.+?)(?:\s+|:)OpId_(\d+)\s*\(cycles\)")`
- **Behavior:** Captures everything before the first `OpId_N` separator (lazy `+?`) as group 1, and the numeric ID as group 2.
- **Risks / concerns:** The lazy `(.+?)` relies on the separator token (`\s+` or `:`) being present. Event IDs without an `OpId_N` component are silently dropped by `_parse_node_event` (returns `None` when `_OP_PATTERN.match(event_id)` is `None`). This is the correct behavior — non-node events don't carry an OpId. However, if QNN EP changes the event format (e.g. adds a new separator character), the match fails silently and all nodes in that sample are dropped. No warning is emitted.
- **Tests:** `tests/unit/session/monitor/qnn/test_csv_parser.py`.

---

### `_TOKEN_SUFFIX`

- **Role:** Regex for stripping the `_token_N` or `_token_N_M` suffix injected by the QNN compiler.
- **Value:** `re.compile(r"_token_\d+(?:_\d+)?")`
- **Behavior:** Idempotent on already-clean strings. Exported for use by `QNNMonitor._heuristic_op_type()` to ensure the strip semantics are shared between the CSV and the heuristic path.
- **Invariants:** FR-15 — runtime instance suffixes must be stripped before ONNX lookup. Shared regex prevents divergence.
- **Tests:** `test_qnn_monitor_resolve.py::test_heuristic_token_suffix_stripped_before_split`.

---

### `_split_op_event_id()`

- **Role:** Heuristic leaf-split for bare op-type recovery on the CSV-only path.
- **Signature:** `def _split_op_event_id(event_id: str) -> tuple[str, str]:`
- **Behavior:** Returns `(op_type, op_path)`. For bare event IDs (no `/`), both elements equal the trimmed input. For path-style IDs, `op_type` is the leaf segment (trimmed), `op_path` is the trimmed full string. For trailing-slash inputs, falls back to the full input for `op_type` to avoid returning empty string.
- **Risks / concerns:** The docstring warning is accurate and important: the leaf segment is the ONNX op symbol (`"Conv"`, `"Add"`), NOT the QNN op type (`"Conv2d"`, `"ElementWiseAdd"`). The heuristic is L3 in the fallback chain — it fires only when L1 (ONNX map) and L2 (EP-authoritative) both miss. This is explicitly documented. The risk is a consumer of the CSV path that skips L1/L2 injection seeing QNN op symbols where they expect QNN op types in the Type column — but since `_split_op_event_id` is private, its consumers are limited to `_parse_node_event` which is also private.
- **Tests:** `tests/unit/session/monitor/qnn/test_event_id_split.py`.

---

### `parse_qnn_profiling_csv()`

- **Role:** Top-level CSV parser. Returns `{"metadata": {...}, "operators": [...], "samples": [[...]]}`.
- **Signature:** `def parse_qnn_profiling_csv(csv_path: str | Path) -> dict[str, Any]:`
- **Behavior:** Reads the CSV, extracts metadata (first occurrence of ROOT fields), extracts per-sample operator lists, aggregates across samples, and returns. The `operators` list is sorted by average cycles descending.
- **Invariants:** `metadata["num_samples"]` reflects the actual number of sample boundaries found. `operators[n]["samples_cycles"]` carries per-sample cycle lists for downstream `samples_us` conversion.
- **Risks / concerns:**
  1. `_read_csv` opens the file with `encoding="utf-8"`. If the QNN EP produces a CSV with a different encoding (e.g. UTF-16 on some Windows locales), this raises `UnicodeDecodeError` which propagates up to `_parse_artifacts_safe()` and becomes `status="parse_failed"`. This is acceptable — the parse-failure contract handles it.
  2. `csv.DictReader` silently skips rows where the header count doesn't match the row count. A malformed CSV row (e.g. an extra comma in an operator name) would be silently misaligned. Not a current risk for machine-generated QNN CSVs.
- **Tests:** `tests/unit/session/monitor/qnn/test_csv_parser.py`, `test_csv_parser_samples.py`.

---

### `_extract_metadata()`

- **Role:** Extract ROOT-level metadata from CSV rows (first occurrence only).
- **Behavior:** Captures `hvx_threads`, `accel_execute_cycles`, `accel_execute_us` from the first ROOT row matching each field. Multi-sample CSVs repeat these ROOT rows; `None` guards on each field ensure only the first is captured.
- **Risks / concerns:** `"Accelerator (execute) time"` (no `"(cycles)"` suffix) is the key for `accel_execute_us`. If QNN EP ever changes this string (e.g. drops the space before the unit or adds parentheses around `US`), the field is silently `None` → defaults to `0` → `cycle_to_us = 0.0` → all `duration_us` values are `0.0`. No warning is emitted for missing metadata fields.
- **Tests:** `test_csv_parser.py::test_parse_csv_metadata`.

---

### `_extract_samples()`

- **Role:** Parse per-sample operator lists from NODE SUB-EVENT rows.
- **Behavior:** Sample boundaries are ROOT rows with `Accelerator (execute) time (cycles)` + `CYCLES` unit. Collects NODE SUB-EVENT rows with CYCLES unit into the current sample. Flushes the last sample after the loop.
- **Risks / concerns:** If the CSV begins with NODE SUB-EVENT rows before the first ROOT boundary (malformed or truncated CSV), `current_sample is None` and those rows are silently dropped. This would produce `samples = []` and `operators = []` — effectively `status="no_data"` behavior but without the explicit status, since the CSV file IS present. The result would have `status="ok"` with an empty operator list. This is a silent data loss scenario.
- **Tests:** `test_csv_parser.py::test_parse_csv_multi_sample`, `test_csv_parser_samples.py`.

---

### `_aggregate_operators()`

- **Role:** Average cycles across samples, attach per-sample lists, sort by average cycles descending.
- **Behavior:** Keyed by `op_id` — operators with the same `op_id` in different positions are kept separate (correct for QNN which assigns unique IDs per node). Returns a list with `samples_cycles` per operator.
- **Risks / concerns:** If the same `op_id` appears in some samples but not others (e.g. an operator is skipped on certain passes due to QNN's early-exit optimization), the average is over the samples where it appeared (`counts[oid]`), not over the total number of samples. This is correct behavior but means `len(samples_cycles[oid]) < num_samples` for such operators. Downstream `p90` / `total` are consistent (they use the per-sample list), but `duration_us` from `avg_cycles * cycle_to_us` does not account for the skipped samples. Documented implicitly by `samples_cycles` length difference.
- **Tests:** `test_csv_parser_samples.py`.

---

### `parse_qhas()`

- **Role:** Top-level QHAS parser. Returns `{"summary": {...}, "operators": [...]}`.
- **Signature:** `def parse_qhas(qhas_data: dict) -> dict:`
- **Behavior:** Accesses `qhas_data["data"]` (hard key) then calls `_extract_summary` and `_transform_op` for each raw op.
- **Risks / concerns:**
  1. **Hard key access at line 311:** `data = qhas_data["data"]` raises `KeyError` if `qhas_data` lacks the `"data"` key. There is no `.get("data", {})` guard. The caller (`QNNMonitor._try_qhas`) wraps the call in `try/except Exception` which converts this to a `logger.warning` + `(None, None, None)` return — so it is safe at the call site. However, if `parse_qhas` is called directly (via the `qnn/__init__.py` public re-export), the caller must handle `KeyError` themselves. This is an API contract gap that should be documented on the function.
  2. **`_extract_summary` hard key accesses at lines 342-355:** The 14 `raw["key"]` accesses in `_extract_summary` all raise `KeyError` if a key is absent in the QHAS JSON. If QNN SDK changes the JSON schema (e.g. renames `"time_us"` to `"inference_time_us"`), all 14 keys raise in sequence from the first missing one. The failure propagates up to the `try/except` in `_try_qhas`, resulting in `status="basic_fallback"` with a warning log. This is safe but produces a degraded result without a clear error message about which key was missing.
  3. **`_transform_op` hard key accesses at lines 365, 402-403:** `op["cycles"]`, `op["qnn_op_type"]`, `op["qnn_op"]` are hard accesses. Same schema-fragility concern.
- **Tests:** `tests/unit/session/monitor/qnn/test_qhas_parser.py`.

---

### `_transform_op()`

- **Role:** Transform a single `qnn_op_instances_nodes` entry into a normalized dict.
- **Behavior:** Converts cycles to microseconds, computes VTCM hit ratio, strips `_TOKEN_SUFFIX` from `qnn_op` to produce the clean `op_path`. The `qnn_op_type` value (e.g. `"Conv2d"`) is placed in `"name"` as the L2 EP-authoritative source for the resolver chain.
- **Invariants:** FR-15 — `_TOKEN_SUFFIX` strip on `qnn_op` ensures QHAS path keys match the CSV strip semantics and match ONNX `node.name` keys for L1 lookup.
- **Risks / concerns:** The `_TOKEN_SUFFIX.sub("", op["qnn_op"])` strip is applied to the full path. If a node's base name contains `_token_` as a legitimate substring (e.g. a model layer named `attention_token_embedding`), the suffix is incorrectly stripped from the middle. The regex `_token_\d+(?:_\d+)?` requires digits after `_token_`, so a plain `_token_embedding` would NOT be stripped. However `_token_123` embedded mid-path (e.g. `embedding_token_1/Conv`) would be stripped to `embedding/Conv`, corrupting the path. This is an edge case: the QNN compiler convention is to inject this suffix at the END of the path, not in the middle. In practice this is not a risk.
- **Tests:** `test_qhas_parser.py`.

---

### `_vtcm_ratio()`

- **Role:** Compute VTCM hit ratio: `vtcm_read / (vtcm_read + dram_read)`.
- **Behavior:** Returns `None` when both are zero (no read traffic). Returns a value in `[0.0, 1.0]`.
- **Tests:** Exercised via `test_qhas_parser.py`.

## 4. Cross-cutting concerns

**Spec drift:**
- The strip of `_TOKEN_SUFFIX` from `qnn_op` in `_transform_op` (line 403) is the v2.4 fix documented in spec §3.5 and coreloop §4.3. Without this strip, L1 ONNX lookup in detail mode is silently inert because production map keys are clean (`/encoder/conv1/Conv`) but QHAS `qnn_op` carries the suffix (`/encoder/conv1/Conv_token_1_2`). This fix is correctly applied.
- `_extract_metadata` uses first-occurrence capture — this matches the spec's intent of capturing the initial inference sample's metadata for the `cycle_to_us` ratio. However, if the first sample's metadata is atypical (e.g. a warmup pass with different cycle counts than the steady-state run), the cycle-to-us conversion ratio would be skewed. The design chooses simplicity over perfect accuracy.

**Information-hiding contract:** `_internal.py` is imported only by:
1. `qnn_monitor.py` (line 30): `from .qnn._internal import _TOKEN_SUFFIX, parse_qhas, parse_qnn_profiling_csv`
2. Tests using the `_`-prefixed exception: `test_csv_parser_samples.py:19`, `test_event_id_split.py:23`, `test_qnn_monitor.py:616, 680`.

All of these are sanctioned by the architecture test's rules. Verified by grep — no other `src/` file imports from `_internal`.

**Deferred work:** No TODO markers. The consolidation of `csv_parser.py` and `qhas_parser.py` into one file is complete.

**EPDevice / ep_name:** Not referenced. This module is pure parsing logic with no ORT runtime dependencies.

## 5. Confidence level

**Medium-High.** The CSV and QHAS parsers are well-structured and the strip semantics are correctly unified. The main risks are:

1. Hard key access in `parse_qhas` / `_extract_summary` / `_transform_op` — KeyError on schema change is caught by the `_try_qhas` try/except, but the error message doesn't identify which key was missing.
2. Silent data loss when CSV begins before the first ROOT boundary (pre-boundary rows silently dropped).
3. The `_OP_PATTERN` match failure produces a silent drop (no warning per unmatched row).

## 6. Verbatim risk inventory

| Severity | Location | Description |
|----------|----------|-------------|
| Medium | `_internal.py:311` | `data = qhas_data["data"]` — hard key access raises `KeyError` if the QHAS JSON schema changes. Caught by `_try_qhas` try/except at runtime, but the resulting `logger.warning("QHAS JSON parse failed: %s")` doesn't name the missing key. External callers of `parse_qhas()` (via the public `qnn/__init__.py` re-export) must handle `KeyError` themselves — this is not documented on the function. |
| Medium | `_internal.py:342-355` | Fourteen hard key accesses in `_extract_summary`. A single schema change (e.g. QNN SDK renames `"time_us"`) converts all QHAS parsing to `status="basic_fallback"` with no actionable error identifying the missing key. Consider `.get()` with explicit `None` fallback and a separate validation pass. |
| Medium | `_internal.py:402-403` | `op["qnn_op_type"]` and `op["qnn_op"]` hard access in `_transform_op`. Same schema-fragility as above. A future QNN SDK that renames these fields silently degrades all detail-mode profiles to `status="basic_fallback"`. |
| Low | `_internal.py:167-185` | If the CSV begins with NODE SUB-EVENT rows before the first ROOT `Accelerator (execute) time (cycles)` row (truncated or malformed file), those rows are silently dropped and `samples = []`. The result has `status="ok"` with an empty operator list rather than `status="no_data"`. |
| Low | `_internal.py:233-235` | `_parse_node_event` returns `None` silently for any event ID that doesn't match `_OP_PATTERN`. There is no per-unmatched-row warning, so if QNN EP changes the event ID format, all operators are silently dropped. |
| Info | `_internal.py:133-134` | First-occurrence metadata capture means the `cycle_to_us` ratio is computed from the first sample's ROOT metadata. For multi-sample traces where warm-up passes have different timing characteristics than steady-state, this ratio may slightly skew per-operator microsecond values. |
