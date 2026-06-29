# src/winml/modelkit/serve/manager.py

## TL;DR
One-character Python 3.11 modernization: `datetime.timezone.utc` → `datetime.UTC` inside the `_fmt_monotonic` formatting helper. Tracks the same UTC sweep as `core/time_utils.py`, `utils/hub_utils.py`, `telemetry/library/exporter.py`.

## Diff metrics
- Lines: +1 / -1 (net 0)
- Hunks: 1 (single `datetime.now(...)` call site)
- Symbols touched: 0

## Role before vs after
Unchanged. `_fmt_monotonic` still translates a `time.monotonic()` timestamp to an approximate ISO 8601 wall-clock string for display.

## Symbol-level changes
- `datetime.datetime.now(tz=datetime.timezone.utc)` → `datetime.datetime.now(tz=datetime.UTC)`. The fully-qualified `datetime.*` form is used because the file imports `datetime` as a module (not `from datetime import datetime`). Same on the `UTC` side.

## Behavior / contract changes
- None. `datetime.UTC is datetime.timezone.utc` on Python 3.11+. Same wall-clock value returned.

## Cross-file impact
- None. Internal helper of `ModelSlotManager`; not exported.

## Risks / subtleties
- None. Mechanical change.
- The fully-qualified `datetime.UTC` form is correct for the file's import style (`import datetime`).

## Simplification opportunities
- The whole module could `from datetime import datetime, timedelta, UTC` and shorten call sites, but that's a broader style change not warranted by this commit.
- `_fmt_monotonic` itself is essentially `(now - elapsed).isoformat()` — could be a one-liner. Marginal.

## Open questions / TODOs surfaced
- None.
