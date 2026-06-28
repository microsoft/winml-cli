# Review: `src/winml/modelkit/session/monitor/qnn/__init__.py`

**Status:** new file (replaces deleted `csv_parser.py` + `qhas_parser.py` public surfaces)
**Lines added/removed:** 27+ / 0-
**Diff command:** `git diff 1bea4cf..HEAD -- src/winml/modelkit/session/monitor/qnn/__init__.py`

## 1. Purpose of this file

Public surface of the `session/monitor/qnn/` sub-package. Re-exports `parse_qhas` and `parse_qnn_profiling_csv` from the private `_internal` module so that the `qnn/` package has a stable public API (usable by tests that exercise parsers directly), while keeping the internal helpers private. This balances information hiding with testability.

## 2. Changes summary

- New file replacing the previously deleted `qnn/__init__.py` (which was empty after removing `csv_parser.py` and `qhas_parser.py`).
- Re-exports `parse_qhas` and `parse_qnn_profiling_csv` from `._internal`.
- Defines `__all__ = ["parse_qhas", "parse_qnn_profiling_csv"]`.
- Docstring references the information-hiding contract and links to the architecture regression test.

## 3. Per-symbol review

### `parse_qhas` (re-export)

- **Role:** Re-export of `_internal.parse_qhas` at the package boundary.
- **Behavior:** No logic in this file — pure re-export.
- **Risks / concerns:** Making `parse_qhas` part of `__all__` means it is "public" at the `qnn` package level. External consumers (outside `session/monitor/`) CAN import it as `from winml.modelkit.session.monitor.qnn import parse_qhas`. The architecture test at `tests/unit/architecture/test_qnn_imports.py` catches direct imports from `qnn._internal` but does NOT restrict imports of the re-exported public names from `qnn`. This is intentional — the architecture test's purpose is to enforce that `_internal`'s private implementation is not directly reached; the public re-exports are the sanctioned API surface.
- **Tests:** `tests/unit/session/monitor/qnn/test_csv_parser.py`, `tests/unit/session/monitor/qnn/test_qhas_parser.py` (which still exist and import through the public API, i.e. `from winml.modelkit.session.monitor.qnn import parse_qnn_profiling_csv`).

---

### `parse_qnn_profiling_csv` (re-export)

- **Role:** Re-export of `_internal.parse_qnn_profiling_csv`.
- **Behavior:** Same as `parse_qhas` above.
- **Tests:** `tests/unit/session/monitor/qnn/test_csv_parser.py`.

---

### `__all__`

- **Role:** Defines the public API for `from winml.modelkit.session.monitor.qnn import *`.
- **Value:** `["parse_qhas", "parse_qnn_profiling_csv"]`
- **Risks / concerns:** The private `_TOKEN_SUFFIX` regex and `_split_op_event_id` helper in `_internal.py` are not re-exported. This is correct — they are implementation details. `qnn_monitor.py` imports `_TOKEN_SUFFIX` directly from `._internal`, which is the only allowed importer per the architecture contract. Tests that test `_split_op_event_id` directly import it from `._internal` using the `_`-prefixed exception clause in the architecture test.

## 4. Cross-cutting concerns

**Spec drift:** PRD §10.4 says "Add (v2.3, conditional on option b) `qnn/_internal.py` — NEW private submodule. Imported only by `qnn_monitor.py`; no public exports." The `__init__.py` re-exports `parse_qhas` and `parse_qnn_profiling_csv` as a public surface for test convenience. The docstring explains this: "Tests and downstream consumers should import via this `__init__.py` to preserve the information-hiding boundary." This is a minor pragmatic deviation from "no public exports" — the re-exports exist specifically so tests don't need to violate the `_internal` boundary. The architecture test explicitly permits `_`-prefixed function imports from `_internal` for testing private internals (per the `CLAUDE.md` exception); the non-`_`-prefixed functions are accessible via the `__init__.py` re-export path, which is the cleaner approach.

**Information-hiding contract:** Verified. Only `qnn_monitor.py` imports non-`_`-prefixed names directly from `._internal` (confirmed by grep showing `from .qnn._internal import _TOKEN_SUFFIX, parse_qhas, parse_qnn_profiling_csv` only in `qnn_monitor.py:30`). The `__init__.py` re-exports provide a stable public path that the architecture test does not block.

**Deferred work:** No TODO markers.

**EPDevice / ep_name:** Not referenced.

## 5. Confidence level

**High.** The file is minimal and the design is sound. The only nuance is the tension between "no public exports" (spec) and "testability without violating `_internal` boundary" (practice). The current approach (re-export in `__init__.py`) is the right resolution.

## 6. Verbatim risk inventory

| Severity | Location | Description |
|----------|----------|-------------|
| Info | `__init__.py:24` | Re-exporting `parse_qhas` and `parse_qnn_profiling_csv` makes them accessible to any consumer outside `qnn_monitor.py`. The architecture test guards the `_internal` path but not the public `qnn.*` path. External consumers who import these directly bypass `QNNMonitor._parse_artifacts_safe()` and its parse-failure contract — they would need to handle exceptions themselves. Document this in the function docstrings. |
