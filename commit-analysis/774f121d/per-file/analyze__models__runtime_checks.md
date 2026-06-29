# src/winml/modelkit/analyze/models/runtime_checks.py

## TL;DR

Python 3.11 `StrEnum` cleanup. The two enum classes in this file (`NodeTag`, `AlternativeType`) switched from the legacy `class X(str, Enum):` multi-inheritance idiom to `class X(StrEnum):`. Pure modernization — semantics, members, values, and the public API are identical. Zero behavioral change.

## Diff metrics

- Lines changed: +3 / -3 (6 total)
- Imports: `from enum import Enum` → `from enum import StrEnum`
- Class declarations updated: `NodeTag`, `AlternativeType`
- No new members, no removed members, no renamed values

## Role before vs after

Role is unchanged. This file defines the small enum taxonomy used by the analyze layer:

- `NodeTag` — node classification tags (`ALL_INPUTS_CONSTANT`, `MISSING_SHAPE_INFERENCE`).
- `AlternativeType` — relationship type between alternative patterns (`EQUIVALENT`, ...).

Both still expose `str`-comparable values for downstream JSON serialization and equality checks against bare strings — `StrEnum` guarantees this contract just as the `(str, Enum)` mix-in did.

## Symbol-level changes

### Module imports

- Removed `from enum import Enum`.
- Added `from enum import StrEnum`.

### `NodeTag`

- Declaration: `class NodeTag(str, Enum):` → `class NodeTag(StrEnum):`.
- Members unchanged: `ALL_INPUTS_CONSTANT = "all_inputs_constant"`, `MISSING_SHAPE_INFERENCE = "missing_shape_inference"`.

### `AlternativeType`

- Declaration: `class AlternativeType(str, Enum):` → `class AlternativeType(StrEnum):`.
- Member shown in the diff: `EQUIVALENT = "equivalent"` (others not in diff, presumed unchanged).

## Behavior / contract changes

None at the value-level. `StrEnum` (PEP 663 / Python 3.11+) is the canonical replacement for `(str, Enum)` multi-inheritance. It produces members that are still string instances and compare equal to their `_value_`.

One subtle difference: `StrEnum` forces `str(member)` to return the **value** (e.g. `"all_inputs_constant"`), whereas plain `(str, Enum)` returned `"NodeTag.ALL_INPUTS_CONSTANT"`. Code that does `f"{tag}"` or `str(tag)` now sees the value string. Any test comparing `str(NodeTag.ALL_INPUTS_CONSTANT) == "NodeTag.ALL_INPUTS_CONSTANT"` would break. The commit body does not call this out.

## Cross-file impact

- Consumers serialize these enums to JSON / compare with raw strings — unchanged behavior because `StrEnum` members still `==` their string values.
- Anyone formatting these enums into log lines or human-readable output now gets the shorter value-only repr. Worth a quick `grep` of analyze logs.

## Risks / subtleties

- **`str(member)` repr change.** As noted above — this is the standard footgun when migrating from `(str, Enum)` to `StrEnum`. Probability of breakage is low (the codebase rarely stringifies enums directly), but a hidden test asserting the old format would silently break.
- **Python 3.11 floor.** `StrEnum` is Python 3.11+. If the project's `pyproject.toml` floor is below 3.11, this is a build-time break. (Almost certainly 3.11+ given other recent commits — but the floor isn't verified here.)

## Open questions / TODOs surfaced

- Are there any `str(NodeTag.X)` or `repr(NodeTag.X)` usages anywhere that depend on the old `(str, Enum)` format? A grep would settle it.
- Should the rest of the analyze enum family (`IHVType`, `SupportLevel`) get the same treatment? `support_level.py` was migrated in this same commit — confirming this is a sweep, not a one-off.

## Simplification opportunities

- This change **is** a simplification. The `(str, Enum)` idiom was always a workaround for the lack of a first-class string enum in older Python; `StrEnum` is the canonical replacement.
- A future opportunity: collapse `NodeTag`/`AlternativeType`/`IHVType`/`SupportLevel` into a single `enums.py` module if they share lifecycle, instead of one file per enum. Currently four files for four tiny enums is over-decomposition.
