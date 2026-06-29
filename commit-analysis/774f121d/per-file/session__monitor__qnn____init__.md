# src/winml/modelkit/session/monitor/qnn/__init__.py

## TL;DR
New 27-line public-surface module for the relocated `qnn` subpackage. Re-exports exactly two parser functions (`parse_qhas`, `parse_qnn_profiling_csv`) from the private `._internal` module and documents the v2.0.1 information-hiding contract that the architecture regression test (`tests/unit/architecture/test_qnn_imports.py`) enforces with AST scans across both `src/` and `tests/`.

## Diff metrics
- Lines added / removed: **+27 / 0**
- New / modified: **new file** (created in this squash; the predecessor relocate commit `14b7c3e5` had a stub `__init__.py` of only 5 lines, replaced wholesale here)

## Role before vs after
- **Before**: Old `optracing/qnn/` package was deleted in `5f86d9e8`. The intermediate relocate commit (`14b7c3e5`) created a 5-line `__init__.py` in the new location; this final squash supersedes it with a curated 27-line public surface documenting the boundary.
- **After**: The sole sanctioned import path for external CSV/QHAS parsing is `from winml.modelkit.session.monitor.qnn import parse_qhas, parse_qnn_profiling_csv`. The `viewer` sibling is **not** re-exported — it's reached via the fully-qualified `winml.modelkit.session.monitor.qnn.viewer` path (verified: `qnn_monitor.py:31` does `from .qnn.viewer import find_qnn_sdk, run_qhas_viewer`).

## Symbol-level changes
- Re-export: `from ._internal import parse_qhas, parse_qnn_profiling_csv`
- `__all__ = ["parse_qhas", "parse_qnn_profiling_csv"]`
- No re-export of any name from `.viewer` despite the docstring naming it as a sibling module.

## Behavior / contract changes
- Establishes the information-hiding boundary in code: external callers see exactly two functions; every CSV/QHAS internal helper (e.g. `_OP_PATTERN`, `_TOKEN_SUFFIX`, `_split_op_event_id`, `_aggregate_operators`) stays private. The architecture test (`test_no_external_imports_of_qnn_internal`) AST-walks both trees and allows the `_`-prefixed exception only for test files importing private *function* names — not the `_internal` module itself.
- The docstring states "only `qnn_monitor` is allowed to import non-`_`-prefixed names from `._internal`"; the architecture test enforces this exactly by whitelisting `qnn_monitor.__file__` (line 114 of the test) as the sole src-tree skip.

## Cross-file impact
- Consumers should `from winml.modelkit.session.monitor.qnn import parse_qhas, parse_qnn_profiling_csv` (absolute, per tests/ rule in CLAUDE.md).
- `qnn_monitor.py:30` is the only intra-package consumer of `_internal`-private names and mixes one underscore name with two public ones — this is why the architecture test whitelists the file by path rather than by name-prefix.
- Regression test: `tests/unit/architecture/test_qnn_imports.py` (PRD NFR-8) — includes synthetic-AST tests that guard the detector itself against future shape regressions.

## Risks / subtleties
- `viewer.py` is not `_`-prefixed and not re-exported here, which is the inconsistent middle ground flagged on the a509a67 baseline. The squash did not resolve it — `viewer.py` remains a public-by-omission sibling that `qnn_monitor.py` reaches into via `from .qnn.viewer import ...`.
- The architecture test's allow-list is one file (`qnn_monitor.py`); if a future caller in the source tree legitimately needs the parsers, it must import them via the package `__init__` (the docstring rule) rather than reaching into `_internal`. There is no positive list of valid public-API consumers — only the negative rule.
- The docstring references "spec v2.0.1" but the squash message references "v2.4 information-hiding contract" (consistent with the prior a509a67 doc). The two version numbers in the docstring vs. the test commentary aren't aligned.

## Open questions / TODOs
- Should `viewer.py` be renamed `_viewer.py` to match `_internal.py`'s privacy stance? Or should `find_qnn_sdk` / `run_qhas_viewer` be re-exported from this `__init__.py` to bring the viewer under the same curated public surface?
- Spec version numbers in the docstring (v2.0.1) vs. squash commentary (v2.4) should be reconciled.

## Simplification opportunities
- **Could `_internal` just be the package contents?** The package has exactly two non-`_` modules (`_internal`, `viewer`). If the only public parser surface lives in `_internal`, the indirection of "private module + re-export init" is a heavyweight wrapper for two re-exports policed by an AST test. An alternative shape: put `parse_qhas` / `parse_qnn_profiling_csv` directly in `__init__.py` (with private helpers in a sibling `_helpers.py`), drop the re-export, and the architecture test reduces to "no imports from `qnn._helpers`".
- **Drop `_internal` and let `_`-prefixed names in `__init__.py` carry the privacy contract** — relies on the existing CLAUDE.md private-prefix convention without needing a separate AST architecture test (other modules in the repo do this).
- The docstring is 17 of 27 lines (63%). For a two-symbol re-export this is heavy; the boundary is already enforced by the architecture test and the `__all__` list, so most of the prose could move into the test docstring (where the rule is actually enforced).
