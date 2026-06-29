# src/winml/modelkit/analyze/models/information.py

## TL;DR
Same `(str, Enum)` → `StrEnum` modernisation as `ihv_type.py`, applied to `ActionLevel`.
Two-line touch: import + class declaration. Pydantic / serialisation / equality semantics
preserved; `str(ActionLevel.REQUIRED)` format changes from `"ActionLevel.REQUIRED"` to
`"required"` (the canonical `StrEnum` behaviour).

## Diff metrics
- 2 insertions / 2 deletions (net 0).
- One import line, one class declaration.
- No new members, no signature changes.

## Role before vs after
Before: `from enum import Enum` + `class ActionLevel(str, Enum):` — legacy idiom for
string-valued enum.

After: `from enum import StrEnum` + `class ActionLevel(StrEnum):` — canonical Python 3.11+
form. Zero member-level changes (`REQUIRED`, `RECOMMENDED`, etc. all unchanged).

## Symbol-level changes
- `ActionLevel` class base: `(str, Enum)` → `(StrEnum)`.
- Module import: `Enum` → `StrEnum`.
- `Information` Pydantic model unchanged.

## Behavior / contract changes
- **`str(ActionLevel.REQUIRED)` format changes** from `"ActionLevel.REQUIRED"` to
  `"required"`. Logging / f-strings now print the value directly. Pydantic's JSON
  serialisation of `ActionLevel` fields is unchanged (it serialises by `.value` either way).
- Equality with literals, isinstance against `str`, and `.value` access — all unchanged.

## Cross-file impact
- Part of the 3-file `StrEnum` sweep alongside `ihv_type.py` and `onnx_model.py` in this
  commit. No caller signature changes required.
- `from .runtime_checks import PatternRuntime` and `from .support_level import SupportLevel`
  still import the legacy `(str, Enum)` versions — codebase is now mixed-idiom.

## Risks / subtleties
- `str(ActionLevel.X)` output change is the only observable delta. Verify no log/test
  asserts on the old "ClassName.MEMBER" string form.
- Python >= 3.11 required for `StrEnum` (must match `pyproject.toml` floor).

## Open questions / TODOs surfaced
- `PatternRuntime` and `SupportLevel` (imported in this file) still use the legacy form —
  follow-up sweep candidate.

## Simplification opportunities
- Same as `ihv_type.py`: sweep remaining `(str, Enum)` to `StrEnum` for consistency.
