# src/winml/modelkit/analyze/models/onnx_model.py

## TL;DR
Same `(str, Enum)` → `StrEnum` modernisation, applied to `ModelTag`. Two-line touch: import +
class declaration. The `ONNXModel` Pydantic model and `ModelTag` member set are unchanged;
only the base-class spelling moves to the canonical Python 3.11+ form.

## Diff metrics
- 2 insertions / 2 deletions (net 0).
- One import, one class declaration.
- No members, no fields, no validators touched.

## Role before vs after
Before: `from enum import Enum` + `class ModelTag(str, Enum):` — legacy string-enum idiom
for the model-level tag set stored in `ONNXModel.model_tags`.

After: `from enum import StrEnum` + `class ModelTag(StrEnum):` — canonical Python 3.11+
form. Members like `INFERRED_SHAPES`, `EXTERNAL_DATA`, etc. are unchanged.

## Symbol-level changes
- `ModelTag` class base: `(str, Enum)` → `(StrEnum)`.
- Module import: `Enum` → `StrEnum`.
- `ONNXModel` Pydantic model (the actual consumer of `ModelTag`) unchanged.
- `from ...onnx import ONNXDomain` is preserved; that enum is *not* part of this sweep
  (still uses `(str, Enum)` in `onnx/domains.py`).

## Behavior / contract changes
- **`str(ModelTag.INFERRED_SHAPES)` format changes** from `"ModelTag.INFERRED_SHAPES"` to
  the member's `.value`. Pydantic JSON serialisation of `ONNXModel.model_tags` is
  unaffected (pydantic serialises by `.value`). Comparisons against literal strings,
  isinstance against `str`, and `.value` access are all unchanged.

## Cross-file impact
- Final file in the 3-file `StrEnum` sweep this commit (`ihv_type.py`, `information.py`,
  `onnx_model.py`).
- The sibling `ONNXDomain` enum imported here (`onnx/domains.py`) was *not* migrated —
  inconsistency worth flagging.

## Risks / subtleties
- `str(ModelTag.X)` output change is the only observable delta. The pre-commit form
  embedded the class name; downstream logging or debug-print consumers may notice. No test
  asserts on the legacy `"ModelTag.X"` form (verified by grep absence).
- Python >= 3.11 floor required.

## Open questions / TODOs surfaced
- `ONNXDomain` (imported here from `...onnx`) is still `(str, Enum)`. If the sweep
  intention is "all string-enums become `StrEnum`", complete it.

## Simplification opportunities
- Finish the sweep across `ONNXDomain`, `PatternRuntime`, `SupportLevel`, `pattern.models`
  enums, and `telemetry.deviceid` enums for a consistent codebase. Otherwise the project is
  permanently mixed-idiom and new contributors will copy whichever pattern they see first.
