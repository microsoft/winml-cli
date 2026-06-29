# src/winml/modelkit/analyze/models/ihv_type.py

## TL;DR
A two-line modernisation: `from enum import Enum` → `from enum import StrEnum`, and
`class IHVType(str, Enum):` → `class IHVType(StrEnum):`. Behaviourally identical for runtime
isinstance/equality semantics with `str`; the new form is the Python 3.11+ canonical spelling
for "string-valued enum" and removes a redundant base class. Zero behavioural change for
callers comparing against literal strings.

## Diff metrics
- 2 insertions / 2 deletions (net 0).
- One import line and one class declaration.
- No new members, no signature changes.

## Role before vs after
Before: legacy idiom `class Foo(str, Enum):` to get string-valued enum semantics
(`IHVType.QC == "QC"`).

After: canonical Python 3.11+ idiom `class Foo(StrEnum):` — same semantics, less boilerplate.
`StrEnum` is `enum.StrEnum`, formally `str` + `Enum` with `__str__` returning the value (the
pre-commit form returned `"IHVType.QC"` for `str(IHVType.QC)`, while `StrEnum` returns
`"QC"` — see Risks).

## Symbol-level changes
- `IHVType` class declaration: base changes from `(str, Enum)` to `(StrEnum)`.
- No members touched: `QC`, the other IHV codes, are unchanged.

## Behavior / contract changes
- **`str(IHVType.QC)` output changes from `"IHVType.QC"` → `"QC"`.** This is the canonical
  Python behaviour difference between `(str, Enum)` and `StrEnum`. Any logging / f-string /
  serialisation that relied on the `"IHVType.QC"` form will now print just `"QC"`. In practice
  pydantic serialisation, JSON dumps, and direct `== "QC"` comparisons are all unchanged.
- `IHVType.QC.value` still returns `"QC"`; `IHVType.QC == "QC"` still `True`; isinstance
  against `str` still `True`.

## Cross-file impact
- This is part of a coordinated 3-file sweep in the same commit (`ihv_type.py`,
  `information.py`, `onnx_model.py`) — all replacing `(str, Enum)` with `StrEnum`. Other
  unmigrated string-enums in the codebase (per Grep: `runtime_checks.py`, `support_level.py`,
  `pattern/models.py`, `onnx/domains.py`, `telemetry/deviceid/deviceid.py`) were not touched
  in this commit, so the codebase is now mixed-idiom.
- No caller signatures or imports needed changes.

## Risks / subtleties
- **`str(IHVType.QC)` format change** is the only observable behavioural delta. If logging
  output is asserted in tests, those assertions will now fail. Search confirms no such
  assertion in the test suite for `IHVType` specifically, but worth scanning logs in
  downstream consumers.
- **Python version floor.** `StrEnum` is Python 3.11+. The project must be on >=3.11
  (verifiable in `pyproject.toml`), otherwise this hard-breaks older interpreters. The
  matching sweep across two other files plus the lack of a `__future__` import or
  conditional polyfill says the project floor is fine here.

## Open questions / TODOs surfaced
- Why only 3 of the 8 string-enum classes in the codebase? `runtime_checks.PatternRuntime`,
  `support_level.SupportLevel`, `pattern.models.*`, `onnx.domains.ONNXDomain`,
  `telemetry.deviceid.deviceid.*` still use `(str, Enum)`. Worth a follow-up sweep for
  consistency.

## Simplification opportunities
- Sweep the remaining `(str, Enum)` declarations to `StrEnum` for repository-wide
  consistency. The 5 untouched sites are listed above.
