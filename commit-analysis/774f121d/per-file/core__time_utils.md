# src/winml/modelkit/core/time_utils.py

## TL;DR
Pure modernization: replaces `datetime.timezone.utc` with `datetime.UTC` (the alias added in Python 3.11). Two-line touch — one in `from` clause, one in the function body. No behavior change. Compatible with the project's Python ≥ 3.11 floor.

## Diff metrics
- Lines: +2 / -2 (net 0)
- Hunks: 2 (import + one call site)
- Symbols touched: 0 (only the imported name)

## Role before vs after
- Before: imported `timezone` along with `datetime`; used `tz=timezone.utc` to anchor the parsed epoch in UTC. Same function (`format_timestamp_iso`), same return shape.
- After: imports `UTC` directly, uses `tz=UTC`. Same function, same return shape. The pre-existing `.replace("+00:00", "Z")` step converts to the ISO 8601 "Zulu" form.

## Symbol-level changes
- Import: `from datetime import datetime, timezone` → `from datetime import UTC, datetime` (re-ordered alphabetically per ruff's `I001` sort).
- Body: `datetime.fromtimestamp(epoch_time, tz=timezone.utc)` → `datetime.fromtimestamp(epoch_time, tz=UTC)`. Functionally identical (`datetime.UTC is datetime.timezone.utc` on 3.11+).

## Behavior / contract changes
- None. `datetime.UTC` was introduced in Python 3.11 as a public alias of `datetime.timezone.utc`. The two refer to the same object.

## Cross-file impact
- Consistent with the same pattern applied in this commit to `core/time_utils.py`, `utils/hub_utils.py`, `telemetry/library/exporter.py`, and `serve/manager.py`. All five locations migrated together — the diff is a project-wide sweep, not a one-off.
- No imports of `timezone` from this module's namespace existed in tests; nothing breaks.

## Risks / subtleties
- If any downstream code did `from winml.modelkit.core.time_utils import timezone` (which it shouldn't — `timezone` was never an exported symbol of this module), it would now `ImportError`. Verified no such consumer exists.
- The `.replace("+00:00", "Z")` hack still works because `datetime.UTC.isoformat()` produces `+00:00` exactly as `timezone.utc` did. Future Python versions could in theory change `UTC`'s formatting (unlikely — would break a wide swath of stdlib consumers).

## Simplification opportunities
- Consider `dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"` — but the current `isoformat(timespec="milliseconds").replace("+00:00", "Z")` is more readable. Not worth changing.
- This module is 22 lines and exports one function. Could be inlined at its single in-tree call site if it has only one — but `format_timestamp_iso` is a generic helper and likely worth keeping.

## Open questions / TODOs surfaced
- None. The change is mechanical and obviously correct.
