# src/winml/modelkit/telemetry/library/exporter.py

## TL;DR
Mechanical Python 3.11 modernization: replaces `from datetime import datetime, timezone` (and its `timezone.utc` use) with `from datetime import UTC, datetime`. One call site (`_ns_to_datetime`). Same project-wide sweep applied to `core/time_utils.py`, `utils/hub_utils.py`, `serve/manager.py`. Zero behavior change.

## Diff metrics
- Lines: +2 / -2 (net 0)
- Hunks: 2 (one import; one call site)
- Symbols touched: 0

## Role before vs after
Unchanged. `_ns_to_datetime` still translates nanosecond-precision ORT log timestamps to UTC `datetime` instances for the OneCollector exporter payload.

## Symbol-level changes
- Import: `from datetime import datetime, timezone` → `from datetime import UTC, datetime` (re-ordered alphabetically per ruff's `I001`).
- Body: `datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc)` → `datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=UTC)`.

## Behavior / contract changes
- None. `datetime.UTC is datetime.timezone.utc` on Python 3.11+.

## Cross-file impact
- None. Internal helper of the exporter; not exported.

## Risks / subtleties
- None. Mechanical change.

## Simplification opportunities
- The `_ns_to_datetime` function name is precise; no further compression.
- `ts_ns / 1_000_000_000` could use `ts_ns * 1e-9` for marginal readability, but float-division is fine and more obviously precise here. Not a real change.

## Open questions / TODOs surfaced
- None.
