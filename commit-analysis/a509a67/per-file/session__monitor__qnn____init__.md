# src/winml/modelkit/session/monitor/qnn/__init__.py

## TL;DR
New 27-line public-surface module for the `qnn` subpackage. Exports exactly two parser functions (`parse_qhas`, `parse_qnn_profiling_csv`) lifted from the private `._internal` module, and documents the information-hiding contract that the architecture regression test (`tests.unit.architecture.test_qnn_imports`) enforces.

## Diff metrics
- Lines added / removed: **+27 / 0**
- New / modified: **new file**

## Role before vs after
- **Before**: No `session/monitor/qnn/` package existed; equivalent CSV/QHAS parsers lived as public-ish modules `qnn/csv_parser.py` and `qnn/qhas_parser.py` under the old `optracing/` tree (deleted by this commit per spec v2.4 simplification / OQ-1 (b)).
- **After**: Single curated public surface. Two parsers are re-exported; the regex internal (`_TOKEN_SUFFIX`) is intentionally **not** re-exported even though `qnn_monitor.py` imports it directly from `._internal` (the underscore prefix is the CLAUDE.md private-internals testing exception that the architecture test honours).

## Symbol-level changes
- Re-export: `from ._internal import parse_qhas, parse_qnn_profiling_csv`
- `__all__ = ["parse_qhas", "parse_qnn_profiling_csv"]`
- No symbols re-exported from `.viewer` — its functions (`find_qnn_sdk`, `run_qhas_viewer`, `run_basic_viewer`) are imported directly by `qnn_monitor.py` via `from .qnn.viewer import ...`, which is allowed because `viewer.py` is not name-prefixed `_`.

## Behavior / contract changes
- Establishes the v2.4 information-hiding boundary in code: external consumers (tests, downstream) get exactly two functions from `winml.modelkit.session.monitor.qnn`. The CSV/QHAS internal helpers (`_OP_PATTERN`, `_TOKEN_SUFFIX`, `_split_op_event_id`, `_read_csv`, `_extract_metadata`, `_extract_samples`, `_parse_node_event`, `_aggregate_operators`, `_require`, `_extract_summary`, `_transform_op`, `_vtcm_ratio`) are private and reachable only via `_`-prefixed names (which the architecture regression test explicitly allows for testing).
- Per docstring: "only `qnn_monitor` is allowed to import non-`_`-prefixed names from `._internal`" — a code-review-enforced exception layered on top of the architecture-test guard.

## Cross-file impact
- Consumers should `from winml.modelkit.session.monitor.qnn import parse_qhas, parse_qnn_profiling_csv` (absolute, per tests/ rule in CLAUDE.md).
- `qnn_monitor.py` is the sole intra-package consumer of `_internal`-private names.
- Regression test: `tests/unit/architecture/test_qnn_imports.py` (PRD NFR-8) — flags any non-`_` import of `_internal` from anywhere except `qnn_monitor.py`.

## Risks / subtleties
- The "only qnn_monitor" exception is documented but the test (per docstring) catches all non-`_`-prefixed imports from `_internal` — qnn_monitor's `from .qnn._internal import _TOKEN_SUFFIX, parse_qhas, parse_qnn_profiling_csv` mixes one `_`-prefixed name with two public ones, so the test must specifically whitelist `qnn_monitor` as a caller (not just the underscore-name exception).
- `viewer` is described in the module docstring as a sibling of `_internal` but is not re-exported here. Anyone needing the QHAS viewer shell-out must reach into `qnn.viewer` directly — that's a documented `qnn/`-internal coupling, but it's not a `_`-prefixed module so it counts as public-by-omission. Could surface unintentionally.

## Open questions / TODOs surfaced
- Should `viewer` be marked `_viewer` (matching `_internal`'s privacy stance) and re-exported here, or left as a public sibling? Current state is the inconsistent middle ground: `viewer.py` is public-import-able from `winml.modelkit.session.monitor.qnn.viewer` but undocumented at the package surface.
- The architecture regression test's exact allow-list for the `qnn_monitor → _internal` exception isn't documented here — only the negative rule. Recommend mirroring the test's positive whitelist in this docstring.
