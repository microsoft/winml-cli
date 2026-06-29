# src/winml/modelkit/export/htp/metadata_writer.py

## TL;DR
Two-line stdlib modernization: `from datetime import datetime, timezone` -> `from datetime import UTC, datetime`, and `tz=timezone.utc` -> `tz=UTC` at the single use-site inside `_write_default`. Pure Python 3.11+ idiom cleanup, no semantic change. Unrelated to the v2.9 session refactor; bundled into the squash.

## Diff metrics
- 2 insertions / 2 deletions (net 0), two hunks: import line and the `datetime.fromtimestamp` call at line 56.
- File: `src/winml/modelkit/export/htp/metadata_writer.py`.

## Role before vs after
Role unchanged. `MetadataWriter` is still the JSON metadata writer using `HTPMetadataBuilder`. `_write_default` still produces an ISO-8601 millisecond-precision timestamp with a trailing `Z` to mark UTC.

## Symbol-level changes
- Module-level import: `timezone` no longer imported; `UTC` imported instead.
- `MetadataWriter._write_default(export_step, data)`:
  - Pre: `datetime.fromtimestamp(epoch_time, tz=timezone.utc)`.
  - Post: `datetime.fromtimestamp(epoch_time, tz=UTC)`.
  - Surrounding `.isoformat(timespec="milliseconds").replace("+00:00", "Z")` chain unchanged.
- No other symbol changed.

## Behavior / contract changes
- None at runtime. `datetime.UTC` (added in Python 3.11) **is** `datetime.timezone.utc` — they refer to the same singleton object. The produced string is byte-identical.
- The trailing `.replace("+00:00", "Z")` still does its job: `isoformat` on a UTC-aware datetime emits `+00:00` regardless of which alias was used to construct the tzinfo.

## Cross-file impact
- None. The exported `MetadataWriter` API surface and its emitted JSON schema are bit-for-bit unchanged.
- The companion change in `export/htp/monitor.py` (this commit) is unrelated to this edit but lives in the same package.

## Risks / subtleties
- Requires Python 3.11+. As with the StrEnum changes in `onnx/domains.py` and `pattern/models.py`, this hard-fails on 3.10 imports. The project's `requires-python` is presumed `>=3.11`.
- A separate `from ...core.time_utils import format_timestamp_iso` is also imported (line 21) — there is now **two** code paths producing the same kind of ISO timestamp in this file. See Simplification.

## Open questions / TODOs
- None surfaced.

## Simplification opportunities
- The `datetime.fromtimestamp(epoch_time, tz=UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")` chain in `_write_default` looks like a duplicate of what `core.time_utils.format_timestamp_iso` (already imported at line 21) is supposed to do. If `format_timestamp_iso` accepts an epoch float, this whole expression could collapse to `format_timestamp_iso(epoch_time)`. Worth a quick check — would remove the `datetime` / `UTC` imports from this module entirely. Real cleanup opportunity that this commit did not take.
- The `if epoch_time:` guard treats `0.0` (epoch start) as "no timestamp" — fine for production but a latent gotcha. Out of scope.
