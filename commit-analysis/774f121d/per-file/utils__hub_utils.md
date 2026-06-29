# src/winml/modelkit/utils/hub_utils.py

## TL;DR
Mechanical Python 3.11 modernization: replaces `from datetime import datetime, timezone` (and its `timezone.utc` use) with `from datetime import UTC`. Three substitutions in `inject_hub_metadata` (the lazy import + two `datetime.now(timezone.utc).isoformat()` call sites). Same project-wide sweep applied to `core/time_utils.py`, `telemetry/library/exporter.py`, `serve/manager.py`. Zero behavior change.

## Diff metrics
- Lines: +3 / -3 (net 0)
- Hunks: 3 (one module-level import; one lazy import inside `inject_hub_metadata`; two `datetime.now(...)` call sites)
- Symbols touched: 0 (only the imported names)

## Role before vs after
Unchanged. Still the HuggingFace Hub metadata injection helper. The module-level docstring describes its single responsibility (injecting `hf_*` metadata properties into ONNX models), and that responsibility is unchanged.

## Symbol-level changes
- Added: top-of-file `from datetime import UTC` (new module-level import).
- Modified: lazy import inside `inject_hub_metadata` — `from datetime import datetime, timezone` → `from datetime import datetime`. The `UTC` name is captured from the module-level import.
- Modified: two call sites in `inject_hub_metadata`:
  1. `add_prop("hf_export_timestamp", datetime.now(timezone.utc).isoformat())` → `datetime.now(UTC).isoformat()`.
  2. `onnx_model.doc_string` template — `datetime.now(timezone.utc).isoformat()` → `datetime.now(UTC).isoformat()`.

## Behavior / contract changes
- None. `datetime.UTC` is the public alias for `datetime.timezone.utc` (since Python 3.11). The functions return identical values.
- The `add_prop("hf_export_timestamp", ...)` value is still an ISO 8601 datetime string with `+00:00` suffix. No `.replace("+00:00", "Z")` here — the suffix stays as-is, unlike `core/time_utils.format_timestamp_iso`.

## Cross-file impact
- Part of the project-wide UTC modernization. Equivalent diff in `core/time_utils.py`, `telemetry/library/exporter.py`, `serve/manager.py`.
- No consumer imported `timezone` from this module.

## Risks / subtleties
- The lazy `from datetime import datetime, timezone` inside `inject_hub_metadata` was originally there to make the function's `datetime`-usage optional (deferring the import until call). The new code still has a lazy `from datetime import datetime` but pulls `UTC` from module scope — meaning the module-level import is now eager. Marginal: `datetime` is in the stdlib and zero-cost to import.

## Simplification opportunities
- The redundant lazy `from datetime import datetime` inside `inject_hub_metadata` could be dropped now that `UTC` already requires the module-level import. Tiny readability win.
- Both `datetime.now(UTC).isoformat()` call sites duplicate the same computation. A one-liner helper (`_now_iso()`) would compress. Marginal.

## Open questions / TODOs surfaced
- None. The change is mechanical and obviously correct.
