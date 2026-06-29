# src/winml/modelkit/pattern/models.py

## TL;DR
Same modernization as `onnx/domains.py`: `from enum import Enum` -> `from enum import StrEnum` and `class PatternType(str, Enum)` -> `class PatternType(StrEnum)`. Two-line, zero-semantic-change cleanup. Unrelated to the v2.9 session refactor; bundled into the squash.

## Diff metrics
- 2 insertions / 2 deletions (net 0), two hunks: the import line and the class header.
- No symbol added, removed, renamed, or re-signed.

## Role before vs after
Role unchanged. `PatternType` is still the two-member `"operator"` / `"subgraph"` enum used in the Pydantic discriminator on `Pattern.pattern_type` and validated against the `pattern_id` prefix.

## Symbol-level changes
- `PatternType`:
  - Pre: `class PatternType(str, Enum)`.
  - Post: `class PatternType(StrEnum)`.
  - Members `OPERATOR = "operator"`, `SUBGRAPH = "subgraph"` unchanged.
- `Pattern`, `SubgraphPattern`, `OperatorPattern` Pydantic models — untouched.

## Behavior / contract changes
- Same subtle `str(PatternType.OPERATOR)` shift as `ONNXDomain`: legacy `(str, Enum)` returned `"PatternType.OPERATOR"` from `str()` by default; `StrEnum` returns `"operator"`. f-string interpolation always returned the value. JSON serialization via Pydantic also already returned the value.
- Equality with bare strings (`PatternType.OPERATOR == "operator"`) still holds.
- The `field_validator("pattern_type")` on `Pattern` compares enum members directly (`v == PatternType.OPERATOR`), not strings — unaffected.

## Cross-file impact
- Re-exported through `pattern/__init__.py` (verified consumers in `modelkit.pattern` and `modelkit.analyze`).
- Any external code stringifying members for display sees the cleaner value — typically a fix, not a break.
- No session-package coupling; this file does not import from `..session`.

## Risks / subtleties
- The Pydantic model defaults `pattern_type: PatternType = Field(default=PatternType.SUBGRAPH, ...)` still serialize identically — Pydantic uses `.value` for str-mixed enums.
- Requires Python 3.11+ (StrEnum was added in 3.11).

## Open questions / TODOs
- None surfaced.

## Simplification opportunities
- None remaining here — the change *is* the simplification. The file is already minimal.
