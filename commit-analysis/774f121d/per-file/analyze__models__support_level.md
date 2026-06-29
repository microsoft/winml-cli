# src/winml/modelkit/analyze/models/support_level.py

## TL;DR

Python 3.11 `StrEnum` modernization. `SupportLevel` migrated from `class SupportLevel(str, Enum):` to `class SupportLevel(StrEnum):`. No member additions or renames. Identical semantics for equality and JSON serialization. Pure cleanup.

## Diff metrics

- Lines changed: +2 / -2 (4 total)
- Imports: `from enum import Enum` → `from enum import StrEnum`
- Class declaration: `SupportLevel(str, Enum)` → `SupportLevel(StrEnum)`
- Member values: unchanged (`SUPPORTED = "supported"`, ... — diff shows the first; remainder presumed identical)

## Role before vs after

Unchanged. `SupportLevel` is the support-classification enum used across the analyze layer to record whether an op / pattern / node is `SUPPORTED`, `UNSUPPORTED`, or whatever else the enum encodes. The downstream contract (JSON-safe string values comparable to bare strings) is preserved by `StrEnum`.

## Symbol-level changes

### Module imports

- `from enum import Enum` → `from enum import StrEnum`.

### `SupportLevel`

- Declaration only: `class SupportLevel(str, Enum):` → `class SupportLevel(StrEnum):`.
- Members unchanged.

## Behavior / contract changes

No semantic difference for `==`, hashing, JSON, or membership-in-list checks.

One inherited footgun from the `StrEnum` migration applies here too: `str(SupportLevel.SUPPORTED)` now returns `"supported"` instead of `"SupportLevel.SUPPORTED"`. Any code that depended on the qualified form when formatting/logging would silently change. Low-probability break, but worth `grep`-ing log statements.

## Cross-file impact

- The analyze layer reads `SupportLevel` members and compares them to string literals throughout the support-classification pipeline — unchanged.
- `runtime_checks.py` got the same `StrEnum` migration in this commit, confirming a sweep.

## Risks / subtleties

- **Python 3.11 floor required.** `StrEnum` is 3.11+. Same caveat as `runtime_checks.py`.
- **`str(...)` repr difference.** Identical to the `runtime_checks.py` note — low probability of breakage in practice.

## Open questions / TODOs surfaced

- Same as `runtime_checks.py`: should the small enum modules in `analyze/models/` be consolidated into a single `enums.py`? Four files for four tiny enums is over-decomposition.
- Are members of `SupportLevel` written to any persisted artifact (JSON manifest, cache key, build provenance) where the repr change could matter? Unlikely — `StrEnum` serializes to the value string just like `(str, Enum)` did — but `model_dump_json` style serializers from third-party libs sometimes differ.

## Simplification opportunities

- This **is** the simplification: `StrEnum` is the canonical replacement.
- Further opportunity: the four enum classes spread across `runtime_checks.py`, `support_level.py`, `ihv_type.py`, and any sibling files could collapse into one module. Each enum has 2-5 members and minimal lifecycle of its own — the split adds import noise without isolating change.
