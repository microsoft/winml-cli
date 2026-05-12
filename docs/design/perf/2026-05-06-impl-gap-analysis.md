# Op-Tracing Refactor — Implementation Gap Analysis

**Date:** 2026-05-06
**Status:** Research artifact — research only, no code changes
**Branch:** `feat/op-tracing-refactor`
**Anchor commits:** `b77043a1` (QHAS authoritative `qnn_op_type`) + `56549546` (`--top-k` CLI default and wiring)
**Companion docs:**
- `docs/design/perf/2026-05-03-op-trace-parser-interface-spec.md` v1.2 — focused architectural spec for the `OpTraceParser` ABC, the four-layer fallback chain, and the ONNX op-type lookup map
- `docs/design/session/monitor/1_prd.md` v2.3 — module PRD (SC-7/-8/-9, FR-13–FR-17, NFR-8, C-7)
- `docs/design/session/monitor/2_coreloop.md` v2.3 — module coreloop (§0.5, §2.1, §3.1, §4.1.5, §4.3, §4.5, §8)

Audience: the engineer who will execute the next implementation phase. Cross-references the v2.3 PRD / coreloop and v1.2 spec against the current code on `feat/op-tracing-refactor` post-`b77043a1` and `56549546`.

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. What's already done](#2-whats-already-done-post-b77043a1--56549546)
- [3. What's missing — by design requirement](#3-whats-missing--by-design-requirement)
  - [3.1 Missing classes / modules](#31-missing-classes--modules)
  - [3.2 Missing modifications to existing classes](#32-missing-modifications-to-existing-classes)
  - [3.3 Files to delete](#33-files-to-delete)
  - [3.4 Tests to add / modify / delete](#34-tests-to-add--modify--delete)
- [4. Open questions blocking implementation](#4-open-questions-blocking-implementation)
- [5. Suggested implementation order](#5-suggested-implementation-order)
- [6. Risks and gotchas](#6-risks-and-gotchas)
- [7. References](#7-references)

---

## 1. Executive Summary

The v2.3 design adds **eight new requirements** beyond the v2.2 baseline (SC-7, SC-8, SC-9, FR-13, FR-14, FR-15, FR-16, FR-17, NFR-8, C-7). Of those, **three are fully met** today (FR-15 token-strip mechanics for ONNX-name bridging are present in CSV path; FR-16 verbatim ONNX naming is honoured because there is no translation table; C-7 is honoured by virtue of nothing translating). **Five are not implemented**: the `OpTraceParser` ABC does not exist; `QNNMonitor` is single-inheritance; `WinMLSession.perf()` does not build or inject the ONNX op-type map; `qnn/csv_parser.py` and `qnn/qhas_parser.py` are still public-ish modules imported from outside the QNN package; the architecture regression test does not exist. **One is partially met**: FR-14 (ONNX as primary source) — the fallback machinery is morally present, but Layer 1 (ONNX lookup) is structurally absent.

The v2.2 baseline (SC-1 through SC-6, FR-1 through FR-12, NFR-1 through NFR-7, C-1 through C-6) is fully implemented and behaviourally validated; the post-`b77043a1` fixes locked down QHAS authoritative behaviour.

Remaining implementation scope: **moderate**. The work decomposes into ~10 small TDD-style commits (see §5) covering one new ABC, one `WinMLSession` static helper, one inheritance change on `QNNMonitor`, two file deletions, one test directory restructure, and one architecture regression test. No risk of behavioural regression in the v2.2 surface because every step preserves the existing `OperatorMetrics` arithmetic verbatim — only the source of `OperatorMetrics.name` changes, and only when an ONNX file is reachable at session setup.

---

## 2. What's already done (post `b77043a1` + `56549546`)

Itemised list of design requirements that are met by the current code. Each item cites the SC/FR/NFR/C identifier from the PRD or the section number from the coreloop and the file:line range that satisfies it.

| Status | Identifier | What | File:Line |
|---|---|---|---|
| ✓ | SC-1 / FR-1 | Op-tracing works with both `onnxruntime-qnn` and `onnxruntime-windowsml` via `add_provider_for_devices` and WinML EP registry initialisation | `src/winml/modelkit/session/monitor/qnn_monitor.py:138-179`; `src/winml/modelkit/session/ep_registry.py` |
| ✓ | SC-2 / FR-3 / FR-4 | `QNNProfiler`, `OpTracer`, `optracing/base.py`, `optracing/registry.py` removed; `optracing/` package no longer exists in tree | (deleted; nothing to cite) |
| ✓ | SC-3 / FR-8 | `QNNMonitor.is_available()` checks both bundled and WinML-registered paths via `ensure_initialized()` | `src/winml/modelkit/session/monitor/qnn_monitor.py:138-179` |
| ✓ | SC-4 / FR-7 | Standalone profiling via `WinMLSession` + `QNNMonitor` primitives works; no helper class needed | (verified by integration tests; see `tests/unit/session/test_perf_monitor.py`) |
| ✓ | SC-6 / OOS-4 | `display_op_trace_report` and `write_op_trace_json` consume `OpTraceResult`; not modified by the refactor | `src/winml/modelkit/session/monitor/report.py:17-47` |
| ✓ | FR-2 | Op-tracing attaches via `session.perf(monitor=...)`; singular monitor parameter | `src/winml/modelkit/session/session.py:589-677` |
| ✓ | FR-5 | Two profiling levels exposed (`basic`/`detail`); QHAS unavailable → `status="basic_fallback"` | `src/winml/modelkit/session/monitor/qnn_monitor.py:329-338` |
| ✓ | FR-6 | `QNNMonitor.result` is `OpTraceResult \| None`; `to_dict()` delegates; nested schema preserved with additive `status`/`error`/`not_run`; `model: str \| None` relaxation applied | `src/winml/modelkit/session/monitor/op_metrics.py:105-159`; `src/winml/modelkit/session/monitor/qnn_monitor.py:264-281` |
| ✓ | FR-10 | `EPMonitor` ABC has `get_session_options()`, `get_provider_options()`, `requires_session_teardown` defaults; `is_available()`, `__enter__`/`__exit__`, `to_dict()` mandatory | `src/winml/modelkit/session/monitor/ep_monitor.py:22-110` |
| ✓ | FR-11 | Monitor instantiation via explicit `_resolve_ep_monitor` dispatch; no factory or registry | `src/winml/modelkit/commands/perf.py` (CLI dispatch) |
| ✓ | FR-12 / C-5 | No `os.chdir`; `_find_schematic` glob-fallback with mtime gating | `src/winml/modelkit/session/monitor/qnn_monitor.py:422-455` |
| ✓ | NFR-2 | `status="no_data"` on missing CSV; `status="parse_failed"` on parse error; `error` populated; never empty success | `src/winml/modelkit/session/monitor/qnn_monitor.py:246-298, 457-470` |
| ✓ | NFR-3 | Auto-reset logs at WARNING with monitor type | `src/winml/modelkit/session/session.py:635-641` |
| ✓ | NFR-4 | Idempotent hooks; paths produced at `__init__` (`_csv_path` resolved once) | `src/winml/modelkit/session/monitor/qnn_monitor.py:104-117` |
| ✓ | NFR-5 | `__exit__` returns implicit `None`; never suppresses caller exception; `sys.exc_info()` captured at session-perf exit | `src/winml/modelkit/session/monitor/qnn_monitor.py:246-258`; `src/winml/modelkit/session/session.py:660-677` |
| ✓ | NFR-6 | No process-global state; `_init_winml_eps_once` extracted to `ep_registry.ensure_initialized()` | `src/winml/modelkit/session/ep_registry.py` |
| ✓ | C-2 | Load-bearing teardown ordering (session reset → gc → monitor `__exit__`) | `src/winml/modelkit/session/session.py:660-677` |
| ✓ | C-3 | `profiling_level` and `profiling_file_path` not user-overridable; explicit assignment after `dict.update(extra)` | `src/winml/modelkit/session/monitor/qnn_monitor.py:230-234` |
| ✓ | T1 (op-tracing lift) | `OperatorMetrics.samples_us` field + derived `sample_count`/`avg_us`/`total_us`/`p90_us` properties | `src/winml/modelkit/session/monitor/op_metrics.py:69-98` |
| ✓ | T2 (op-tracing lift) | CSV parser retains per-sample timings via `samples_cycles` | `src/winml/modelkit/session/monitor/qnn/csv_parser.py:226-283`; converted to `samples_us` at `qnn_monitor.py:316` |
| ✓ | post-T9-fix-2-revise | QHAS detail-mode `name` is the authoritative `qhas.qnn_op_type` (no leaf-split) | `src/winml/modelkit/session/monitor/qnn/qhas_parser.py:96-99`; consumed at `qnn_monitor.py:404-405` |
| ✓ | post-T9-fix-2-revise | CSV-path leaf-split heuristic with `.strip()` safety | `src/winml/modelkit/session/monitor/qnn/csv_parser.py:39-77` (`_split_op_event_id`) |
| ✓ | post-T9-fix-1 | `--top-k` CLI flag wired through; default `5` per mockup spec; usage error when `--top-k` without `--op-tracing` | `src/winml/modelkit/commands/perf.py:1268-1367, 1554-1557` |
| ✓ (partial) | FR-15 | `_TOKEN_SUFFIX` regex strip is present and applied during CSV node parsing — produces a cleaned `op_path` already today | `src/winml/modelkit/session/monitor/qnn/csv_parser.py:36, 220` |
| ✓ | FR-16 / C-7 | No translation tables anywhere in the parser, monitor, render layer or downstream consumers — verbatim today by absence of any translator | (verified by inspection; nothing to cite) |

---

## 3. What's missing — by design requirement

This section walks through every NEW v2.3 design item and assesses current state.

### 3.1 Missing classes / modules

| Item | Target file | Current state |
|---|---|---|
| `OpTraceParser` ABC | `src/winml/modelkit/session/monitor/op_trace_parser.py` (new file; final placement per spec §10.1 Q3 — may co-locate with `ep_monitor.py` if preferred) | Absent. The class does not exist anywhere on the branch. |
| `qnn/_internal.py` (option b, recommended) | `src/winml/modelkit/session/monitor/qnn/_internal.py` (new file) | Absent. CSV/QHAS helpers currently live in two public-named modules (see §3.3). |
| Architecture regression test | `tests/unit/architecture/test_qnn_imports.py` (or equivalent) — directory does not exist either | Absent. There is no `tests/unit/architecture/` directory. |

The new ABC's required surface (per spec §3.1 / coreloop §4.1.5):

```python
class OpTraceParser(ABC):
    def __init__(self, onnx_op_types: dict[str, str] | None = None) -> None: ...
    def set_onnx_op_types(self, onnx_op_types: dict[str, str]) -> None: ...    # concrete
    @abstractmethod
    def parse_basic(self, artifacts: dict[str, Path]) -> list[OperatorMetrics]: ...
    @abstractmethod
    def parse_detail(self, artifacts: dict[str, Path]) -> list[OperatorMetrics]: ...
    @abstractmethod
    def supported_levels(self) -> set[Literal["basic", "detail"]]: ...
    def _resolve_op_type(self, op_path: str, ep_authoritative: str | None = None) -> str: ...   # template method, NOT abstract
    def _heuristic_op_type(self, op_path: str) -> str | None: ...               # default returns None
```

### 3.2 Missing modifications to existing classes

| Item | Current state | Target state | Anchor |
|---|---|---|---|
| `QNNMonitor` inheritance | `class QNNMonitor(EPMonitor):` at `qnn_monitor.py:51` | `class QNNMonitor(EPMonitor, OpTraceParser):` (multiple inheritance, monitor IS the parser) | Coreloop §4.3, spec §3.2 |
| `QNNMonitor.parse_basic` | Absent | Method present; wraps current CSV→`OperatorMetrics` list-comp at `qnn_monitor.py:309-319`; `name=self._resolve_op_type(op["op_path"], ep_authoritative=None)` instead of inline `op["name"]` | Coreloop §4.3, spec §3.2 |
| `QNNMonitor.parse_detail` | Absent | Method present; wraps current QHAS→`OperatorMetrics` list-comp at `qnn_monitor.py:403-419`; `name=self._resolve_op_type(op["op_path"], ep_authoritative=op["qnn_op_type"])` | Coreloop §4.3, spec §3.2 |
| `QNNMonitor.supported_levels` | Absent | Returns `{"basic", "detail"}` | Coreloop §4.3 |
| `QNNMonitor._heuristic_op_type` | Absent (logic exists today inside `csv_parser._parse_node_event` via `_TOKEN_SUFFIX.sub(...)` + `_split_op_event_id`) | Method present on `QNNMonitor`; strips `_TOKEN_SUFFIX`, leaf-splits on `/`, returns `None` for empty (chain falls through to L4) | Coreloop §4.3, spec §3.2 |
| `QNNMonitor.__init__` | Calls `EPMonitor` implicitly via Python default; no explicit super-init dispatch | Calls `EPMonitor.__init__(self)` and `OpTraceParser.__init__(self, onnx_op_types=onnx_op_types)` explicitly; new optional `onnx_op_types` parameter (per SC-9) | Coreloop §4.3, spec §3.2 |
| `QNNMonitor._parse_artifacts` | Inline mode dispatch (`if self._level == "detail":` branch at `qnn_monitor.py:329-338`) plus inline list-comps for both branches | Thin wrapper that dispatches to `self.parse_basic(...)` or `self.parse_detail(...)` and wraps the resulting `list[OperatorMetrics]` into an `OpTraceResult` with summary/status/artifacts | Coreloop §4.3 |
| `WinMLSession._build_op_type_map` | Absent | `@staticmethod` (recommended placement per OQ-5); accepts `Path \| None`; calls `onnx.load(path, load_external_data=False)`; returns `{n.name: n.op_type for n in model.graph.node if n.name}`; returns `{}` on `None`, missing, or unparseable input | Coreloop §4.5, spec §6.1 |
| `WinMLSession.perf().__enter__` ONNX-map injection | Absent | After session-options/provider-options merge but before `mon.__enter__()`: `if isinstance(mon, OpTraceParser) and self._onnx_path is not None: mon.set_onnx_op_types(self._build_op_type_map(self._onnx_path))` | Coreloop §4.5, §3.1 |
| `OperatorMetrics.name` docstring | `# QNN op type ("Conv2d", "LayerNorm")` at `op_metrics.py:41` | Per spec §4.1: "Op type. Sourced from ONNX `node.op_type` when the model graph is available; falls back to EP-specific labels (e.g. QNN's `qnn_op_type`) when the graph lookup misses. Use ONNX naming verbatim — no translation tables." Plus example vocabularies. | Spec §4.1 |

### 3.3 Files to delete

| File | Current size | External callers (grep results) | What happens to helpers |
|---|---|---|---|
| `src/winml/modelkit/session/monitor/qnn/csv_parser.py` | 283 lines | Imported by `qnn_monitor.py:30` (allowed) and three test files: `tests/unit/session/monitor/qnn/test_csv_parser.py:9`, `test_csv_parser_samples.py:19`, `test_event_id_split.py:23` | All five helpers (`parse_qnn_profiling_csv`, `_read_csv`, `_extract_metadata`, `_extract_samples`, `_parse_node_event`, `_aggregate_operators`, `_split_op_event_id`, `_TOKEN_SUFFIX`) move to `qnn/_internal.py` (option b, recommended) or onto `QNNMonitor` as private methods (option a). The CSV-side `_token_N` strip moves into `_heuristic_op_type` plus the private CSV-reading path. |
| `src/winml/modelkit/session/monitor/qnn/qhas_parser.py` | 122 lines | Imported by `qnn_monitor.py:31` (allowed) and two test files: `tests/unit/session/monitor/qnn/test_qhas_parser.py:10` and `tests/unit/session/monitor/test_qnn_monitor.py:549` (the latter is a non-architectural cross-cut — see §6) | `parse_qhas`, `_extract_summary`, `_transform_op`, `_vtcm_ratio` move to `qnn/_internal.py` (option b) or onto `QNNMonitor` (option a). The "`name = qnn_op_type`" rule moves into the parser's `_resolve_op_type(op_path, ep_authoritative=qnn_op_type)` call (Layer 2). |
| `src/winml/modelkit/session/monitor/qnn/__init__.py` | 5 lines (docstring only, no exports) | Imported only as a package marker | Delete if option (a) and no other public surface remains in `qnn/`; keep as private package marker if option (b). `qnn/viewer.py` is unchanged (per coreloop §2.1.1) and remains importable from `qnn_monitor.py`. |

### 3.4 Tests to add / modify / delete

#### New tests (must add)

| Test file | Layer | Asserts |
|---|---|---|
| `tests/unit/session/monitor/test_op_trace_parser.py` | Layer 1 (ABC unit) | `_resolve_op_type` walks chain across (L1 hit/miss) × (L2 hit/None) × (L3 hit/None); ABC cannot be instantiated; `set_onnx_op_types` idempotent (last value wins); empty heuristic treated as L4 fall-through |
| `tests/unit/session/monitor/test_qnn_monitor.py::test_parse_basic_uses_onnx_lookup` | Layer 2 (parser-method unit) | Inject `{"_qnn_event": "Conv"}` + small CSV → `OperatorMetrics(name="Conv")` |
| `tests/unit/session/monitor/test_qnn_monitor.py::test_parse_basic_falls_back_to_heuristic` | Layer 2 | Inject `{}` + CSV containing `/encoder/conv1/Conv_token_1_2` → `OperatorMetrics(name="Conv", op_path="/encoder/conv1/Conv")` (cleaned) |
| `tests/unit/session/monitor/test_qnn_monitor.py::test_parse_detail_falls_back_to_qhas` | Layer 2 | Inject `{}`; QHAS fixture with `qnn_op_type="ElementWiseAdd"` → `OperatorMetrics(name="ElementWiseAdd")` via L2 |
| `tests/unit/session/monitor/test_qnn_monitor.py::test_parse_detail_onnx_wins_over_qhas` | Layer 2 | Inject map with `op_path → "Add"`; QHAS row with `qnn_op_type="ElementWiseAdd"` for same path → `OperatorMetrics(name="Add")` (ONNX wins) |
| `tests/unit/session/monitor/test_qnn_monitor.py::test_heuristic_strips_token_suffix` | Layer 3 (heuristic unit) | `monitor._heuristic_op_type("/encoder/conv1/Conv_token_1_2") == "Conv"` |
| `tests/unit/session/monitor/test_qnn_monitor.py::test_constructor_accepts_onnx_op_types` | Layer 2 | `QNNMonitor(level="basic", onnx_op_types={"a": "Conv"})._onnx_op_types == {"a": "Conv"}` |
| `tests/unit/session/test_session.py::test_build_op_type_map_resnet50` | Layer 4 (integration) | `WinMLSession._build_op_type_map(<resnet50.onnx>)` returns non-empty dict whose keys include known node names |
| `tests/unit/session/test_session.py::test_build_op_type_map_handles_failures` | Layer 4 | `_build_op_type_map(None)`, `_build_op_type_map(missing_path)`, `_build_op_type_map(corrupt.onnx)` all return `{}` without raising |
| `tests/unit/session/test_perf_monitor_integration.py::test_onnx_map_injected_into_op_trace_parser` | Layer 4 | With a fake monitor implementing both ABCs, `session.perf().__enter__` calls `set_onnx_op_types(non_empty)` BEFORE `mon.__enter__()` |
| `tests/unit/session/test_perf_monitor_integration.py::test_onnx_map_skipped_for_lifecycle_only_monitor` | Layer 4 | With a `VitisAIMonitor`-style mock that does NOT implement `OpTraceParser`, `set_onnx_op_types` is never called |
| `tests/unit/architecture/test_qnn_imports.py` | Layer 5 (architecture) | AST scan over `src/winml/modelkit/` (excluding `session/monitor/qnn/`) and `tests/`: no `import` / `from` statement references `qnn.csv_parser`, `qnn.qhas_parser`, or `qnn._internal`. The single permitted importer of `qnn/_internal.py` is `qnn_monitor.py`. |

(See coreloop §8.2 / §8.3 / §8.3.1 and PRD §10.5 for the full canonical test list.)

#### Existing tests to delete (architectural debt — coverage migrates to integration tests above)

| File | Size | Rationale |
|---|---|---|
| `tests/unit/session/monitor/qnn/test_csv_parser.py` | 52 lines | Imports `parse_qnn_profiling_csv` (now private). Coverage migrates to `test_qnn_monitor.py::test_parse_basic_*` integration tests using the same CSV fixtures. |
| `tests/unit/session/monitor/qnn/test_csv_parser_samples.py` | 79 lines | Imports `_aggregate_operators` (now private). Coverage migrates to integration tests on `QNNMonitor.parse_basic`. |
| `tests/unit/session/monitor/qnn/test_event_id_split.py` | 108 lines | Imports `_split_op_event_id` (now private). Coverage migrates to `_heuristic_op_type` unit tests on `QNNMonitor`. |
| `tests/unit/session/monitor/qnn/test_qhas_parser.py` | 106 lines | Imports `parse_qhas` (now private). Coverage migrates to `test_qnn_monitor.py::test_parse_detail_*` integration tests using the same QHAS fixtures. |

#### Existing tests to refactor (non-architectural cross-cuts)

| File:line | Current | Target |
|---|---|---|
| `tests/unit/session/monitor/test_qnn_monitor.py:549` | `from winml.modelkit.session.monitor.qnn.qhas_parser import parse_qhas` (used to mirror `_try_qhas` parsing in a fixture-based assertion) | Refactor to call `QNNMonitor.parse_detail({"qhas": fixture_path})` directly. The test asserts post-`b77043a1` QHAS authoritative behaviour; preserve that intent under the new entry point. |

#### Fixtures (no changes)

`tests/unit/session/monitor/qnn/fixtures/optrace_resnet50.csv` and `qhas_resnet50.json` are reused by the new integration tests. No move or rename.

---

## 4. Open questions blocking implementation

These are spec §10.1 Q1/Q2/Q3, mirrored as PRD §9 OQ-3/-4/-5. Each needs a one-line decision before implementation starts.

### OQ-3: Helper placement — option (a) vs option (b)

**Question:** Where do CSV/QHAS/`_TOKEN_SUFFIX` helpers live after the public modules are deleted?

- **Option (a):** Fold all helpers as private methods on `QNNMonitor`. Single file ~1000+ lines, one read.
- **Option (b):** Move ~378 lines into a private sibling submodule `qnn/_internal.py`, imported only by `qnn_monitor.py`. No public exports.

**Spec recommendation:** Option (b) — keeps `qnn_monitor.py` focused on lifecycle (~700 lines) while still satisfying information hiding (rule is "no module *outside the QNN package* imports parsing internals"). Submodule privacy enforced by convention (`_internal.py` filename, `_`-prefixed function names, no `__init__.py` re-exports).

**What changes if user picks (a) instead:** Step 5 of the migration (move helpers) becomes "fold helpers as `_`-prefixed methods on `QNNMonitor`"; `qnn/__init__.py` is deleted (no package needed); `qnn/viewer.py` is also folded (or moved up a level if option (a) eliminates the package entirely). Architecture test enforces the same invariant either way.

### OQ-4: Architecture-test mechanism — AST scan vs lint rule

**Question:** How is the "no external imports of QNN parsing internals" rule enforced?

- **Option (a):** Python AST `import` / `import-from` scan, materialised as a single self-contained pytest test (`tests/unit/architecture/test_qnn_imports.py`).
- **Option (b):** `ruff`'s `flake8-tidy-imports` `banned-module-level-imports` rule.
- **Option (c):** `mypy` plugin.

**Spec recommendation:** Option (a) — AST scan in pytest. Self-contained; no project-config changes; trivial to extend when future EPs (TensorRT, OpenVINO) need the same rule.

**What changes if user picks (b) instead:** The pytest file goes away; instead `pyproject.toml` gets a `[tool.ruff.lint.flake8-tidy-imports.banned-api]` block listing `winml.modelkit.session.monitor.qnn.csv_parser`, `qnn.qhas_parser`, `qnn._internal`. Slightly faster (lint is faster than pytest) but couples the rule to `ruff` and won't catch dynamic `importlib` usage.

### OQ-5: `_build_op_type_map` placement — staticmethod vs free function

**Question:** Where does the ONNX op-type map builder live?

- **Option (a):** `@staticmethod` on `WinMLSession`. Trivially testable as `WinMLSession._build_op_type_map(...)`.
- **Option (b):** Free function in a new module (e.g. `session/perf/op_type_map.py`).

**Spec recommendation:** Option (a) — keep as `@staticmethod` on `WinMLSession`. No state, trivially testable, doesn't introduce a new module for one stateless function.

**What changes if user picks (b) instead:** New file `session/perf/op_type_map.py` (or similar); `WinMLSession.perf()` imports the function and calls it directly; tests target the free function. Arguably cleaner separation but adds a file with no real benefit; spec §10.1 Q3 explicitly recommends against.

---

## 5. Suggested implementation order

Dependency-ordered task list. Each task is small enough to be a single TDD-style commit. Each ends with `uv run pytest tests/` green.

| Step | Task | Size | Depends on | Tests needed |
|---|---|---|---|---|
| 1 | Add `OpTraceParser` ABC at `session/monitor/op_trace_parser.py`. Includes `__init__`, `set_onnx_op_types`, `_resolve_op_type` (concrete template method), `_heuristic_op_type` (default returns `None`), `parse_basic` / `parse_detail` / `supported_levels` (abstract). | S | — | New `tests/unit/session/monitor/test_op_trace_parser.py`: ABC instantiation raises; `_resolve_op_type` walks all (L1 hit/miss) × (L2 hit/None) × (L3 hit/None) combinations; `set_onnx_op_types` idempotent; empty heuristic treated as L4 fall-through. |
| 2 | Add `WinMLSession._build_op_type_map(onnx_path)` static helper. Returns `{}` on `None`, missing, or unparseable input. | S | — | New `test_session.py::test_build_op_type_map_resnet50` (fixture asserts non-empty + known names); `test_build_op_type_map_handles_failures` (None / missing / corrupt → `{}` no raise). |
| 3 | Wire `WinMLSession.perf()` to call `set_onnx_op_types(self._build_op_type_map(self._onnx_path))` when `isinstance(monitor, OpTraceParser)`. Injection MUST happen before `monitor.__enter__()`. | S | 1, 2 | New `test_perf_monitor_integration.py::test_onnx_map_injected_into_op_trace_parser` (fake monitor implementing both ABCs); `test_onnx_map_skipped_for_lifecycle_only_monitor` (mock not implementing `OpTraceParser`). |
| 4 | Refactor `QNNMonitor` to multiple inheritance: `class QNNMonitor(EPMonitor, OpTraceParser)`. Add `parse_basic`, `parse_detail`, `supported_levels`, `_heuristic_op_type`. At this point `parse_basic` / `parse_detail` wrap the **existing** `qnn/csv_parser.py` / `qnn/qhas_parser.py` imports without behavioural change. `name=` is now resolved via `_resolve_op_type(op_path, ep_authoritative=...)`. Update `__init__` to call `EPMonitor.__init__(self)` and `OpTraceParser.__init__(self, onnx_op_types=onnx_op_types)` explicitly; add `onnx_op_types` constructor parameter. | M | 1 | All existing `test_qnn_monitor.py` tests pass unchanged; new `test_constructor_accepts_onnx_op_types`; new `test_parse_basic_uses_onnx_lookup`; `test_parse_detail_falls_back_to_qhas`; `test_parse_detail_onnx_wins_over_qhas`; `test_heuristic_strips_token_suffix`. |
| 5 | Move `_TOKEN_SUFFIX` regex and leaf-split heuristic out of `csv_parser.py` and into `QNNMonitor._heuristic_op_type` plus the private CSV-reading helpers. The strip MUST happen BEFORE the ONNX lookup in `parse_basic` (same call order as today's `_parse_node_event`). | S | 4 | `test_heuristic_strips_token_suffix` covers it; existing CSV tests still green. |
| 6 | Per OQ-3 resolution: move CSV / QHAS reading primitives into `qnn/_internal.py` (option b) or onto `QNNMonitor` (option a). Update `qnn_monitor.py` imports to private location. | M | 4, 5 | Existing integration tests on `QNNMonitor.parse_basic` / `parse_detail` continue to pass. |
| 7 | Delete `qnn/csv_parser.py` and `qnn/qhas_parser.py` as public modules. Empty `qnn/__init__.py` (option b) or delete (option a). Refactor `tests/unit/session/monitor/test_qnn_monitor.py:549` to call `QNNMonitor.parse_detail` instead of `parse_qhas`. Delete the four legacy unit-test files (`test_csv_parser.py`, `test_csv_parser_samples.py`, `test_event_id_split.py`, `test_qhas_parser.py`). | S | 6 | Pytest still green; coverage report shows no regression on the QNN module (coverage migrated to integration tests added in steps 1, 4, 5). |
| 8 | Update `OperatorMetrics.name` docstring per spec §4.1: ONNX-primary, EP-fallback examples, no translation tables. | S | none | None — pure docstring change. |
| 9 | Per OQ-4 resolution: add architecture regression test `tests/unit/architecture/test_qnn_imports.py` (AST scan, recommended). The test fails any future commit that re-exposes a private QNN parsing helper. | S | 7 | Self-asserting; should pass on first run. |
| 10 | Hardware E2E re-run with convnext (or a real model that exercises both QHAS-glue and ONNX-mappable nodes) to validate the full chain end-to-end. Smoke-test ONNX-resolved Type column. | M | 1–9 | Hardware-gated; manual confirmation. |

Steps 1–3 are independent of 4 and can run in either order. Steps 4–9 are strictly sequential. Step 10 is hardware-gated and follows the rest.

---

## 6. Risks and gotchas

Things that could derail the implementation:

### 6.1 MRO order on multiple inheritance

`class QNNMonitor(EPMonitor, OpTraceParser)` puts `EPMonitor` first in the MRO. Both ABCs have `__init__`, but `OpTraceParser.__init__` accepts `onnx_op_types`. The recommended pattern (per coreloop §4.3) is **explicit base-class init** rather than `super().__init__()` chaining:

```python
EPMonitor.__init__(self)
OpTraceParser.__init__(self, onnx_op_types=onnx_op_types)
```

Verify after implementation: a unit test that constructs `QNNMonitor()` and asserts both `_onnx_op_types == {}` AND that the `EPMonitor`-side state (e.g. nothing today, but defensively check `_entered = False`, `_result = None`) is correctly initialised. Subtle bugs here are silent: `super().__init__()` calls only `EPMonitor.__init__` and leaves `_onnx_op_types` unset, breaking `_resolve_op_type` with `AttributeError` only when the chain is exercised.

### 6.2 `_onnx_path` is always set, but may point at a compiled EPContext model

`WinMLSession.__init__` raises `FileNotFoundError` if `onnx_path` is missing (`session.py:191-193`), so `self._onnx_path` is never `None`. Good — the spec's `_build_op_type_map(None) -> {}` defence is for tests, not for the production path.

**However:** if `_onnx_path` points at an already-compiled EPContext model (`*_ctx.onnx`), `onnx.load(path, load_external_data=False)` will succeed but the graph contents are EPContext nodes that reference an external compiled bundle, not the original model topology. The map will key on `EPContext` op nodes rather than the user's `Conv`/`LayerNormalization` etc., and Layer 1 will miss for almost every QNN event ID — the chain falls through to L2/L3/L4 just like today.

This is **not a correctness bug** (chain is monotonic in quality), but it's an observability surprise: a user running on a cached EPContext path will see no behavioural change from v2.2 even though the new code is wired up. Mitigation: log at DEBUG inside `_build_op_type_map` how many entries the map carries (`logger.debug("ONNX op-type map: %d entries from %s", len(m), p)` per spec §6.3); a near-zero count on a cached path is the diagnostic signal.

### 6.3 Token-strip ordering — must run BEFORE the ONNX lookup

The current CSV path strips `_TOKEN_SUFFIX` inside `_parse_node_event` (`csv_parser.py:220`), BEFORE leaf-split. After the refactor, `parse_basic` walks rows → builds `op_path` → calls `_resolve_op_type(op_path, None)`. **The strip MUST happen before `op_path` is constructed**, not inside `_resolve_op_type` or `_heuristic_op_type` only.

Concretely: the cleaned form (post-strip) is what gets stored on `OperatorMetrics.op_path` (per FR-15) AND what gets used as the L1 lookup key. Without the strip, every path-style event ID misses ONNX (because `node.name` does not carry `_token_N`) and the chain always falls through to fallbacks — defeating the v2.3 ONNX-primary contract.

Recommended call order in `parse_basic`:

```
raw_event_id → _TOKEN_SUFFIX.sub("", raw_event_id) → cleaned_op_path
  → _resolve_op_type(cleaned_op_path, ep_authoritative=None)
```

The `_heuristic_op_type` regex re-application is idempotent belt-and-braces (per coreloop §4.3 "On token-suffix stripping, point 2") in case a caller passes a still-suffixed string; it doesn't substitute for stripping at construction time.

### 6.4 ONNX-vs-QHAS precedence: spec wants ONNX to win even when QHAS is authoritative

The current QHAS detail-mode path writes `qhas.qnn_op_type` directly into `OperatorMetrics.name` (`qnn_monitor.py:404-405`, `qhas_parser.py:96-99`). That's post-`b77043a1` correct behaviour for v2.2.

After the refactor, `parse_detail` calls `self._resolve_op_type(op["op_path"], ep_authoritative=op["qnn_op_type"])`. The order is **L1 ONNX first, L2 QHAS-`qnn_op_type` second**. Per spec §3.2 and §5 acceptance criterion P-8: "When the ONNX map has an entry for a node that QHAS also names, the rendered Type column shows the ONNX value."

This is an **intentional behaviour change**: detail-mode rows that today show `Conv2d` (QNN vocabulary) will show `Conv` (ONNX vocabulary) after the refactor when the ONNX map covers them. Verify with `test_parse_detail_onnx_wins_over_qhas` (canonical test in coreloop §8.2). The post-`b77043a1` test at `test_qnn_monitor.py:549` that pins `Conv2d` for the QHAS path will need to be updated: when an ONNX map is injected and contains the matching node, the test should now assert the ONNX value (`Conv`); the assertion that "QHAS wins when no ONNX map" is still true and tests it via the `onnx_op_types={}` path.

Glue-op rows (compiler-inserted, no ONNX equivalent — e.g. `_qnn_compiler_glue_Add_3` → `qnn_op_type="ElementWiseAdd"`) continue to show `ElementWiseAdd` because Layer 1 misses. This is the case-2 walkthrough in spec §3.4.4.

### 6.5 `OpTraceResult` preservation — list[OperatorMetrics] needs wrapping

The new `parse_basic` / `parse_detail` return `list[OperatorMetrics]`. The existing `display_op_trace_report` and `write_op_trace_json` consume `OpTraceResult` (per OOS-4, unchanged). The wrap is the responsibility of `_parse_artifacts`, which dispatches to `parse_*` and then constructs the `OpTraceResult` with the right `summary` / `status` / `artifacts`:

```python
def _parse_artifacts(self) -> OpTraceResult:
    if self._level == "detail":
        ops = self.parse_detail({"qhas": self._qhas_path})
        artifacts = {"qhas": str(self._qhas_path), "csv": str(self._csv_path)}
    else:
        ops = self.parse_basic({"csv": self._csv_path})
        artifacts = {"csv": str(self._csv_path)}
    if not ops:
        return self._make_failure_result(status="no_data", error=None)
    return OpTraceResult(..., operators=ops, summary=self._build_summary(ops), ...)
```

Watch for: today's `_parse_artifacts` reads `meta = parsed.get("metadata", {})` to drive `summary` (`qnn_monitor.py:301, 321-325`). After the refactor, the CSV-side metadata extraction has to live somewhere accessible to `_parse_artifacts` — either as a separate private call (e.g. `_read_csv_metadata`) or stashed on the monitor during `parse_basic` (mutable state on the parser, which the spec calls "stateless across calls" — discouraged). Recommendation: keep `summary` derivation inside `_parse_artifacts` by calling a private CSV-metadata extractor on the same path; do not bleed CSV metadata into `OperatorMetrics`. Verified intent: spec §3.2 sketch uses `meta` only inside the parser's private methods (e.g. `_compute_cycle_ratio(meta)`); the wrapper `_parse_artifacts` builds summary independently.

Detail-mode summary already comes from `parse_qhas` (today, `qnn_monitor.py:330-334`); after the refactor that summary moves into `parse_detail`'s closure or a paired private helper.

### 6.6 Existing monitor-level summary fields cross the `parse_*` boundary

`_parse_artifacts` populates `OpTraceResult.num_samples` from CSV metadata (`qnn_monitor.py:348`) and `OpTraceResult.summary` with `hvx_threads`, `accel_execute_cycles`, `accel_execute_us` (CSV) or QHAS overall summary (`qnn_monitor.py:321-334`). These fields are **not** on `OperatorMetrics` — they're EP-wide / model-wide telemetry. After the refactor, they cannot be returned by `parse_basic` / `parse_detail` (which return `list[OperatorMetrics]`).

Two viable shapes:

1. **Keep summary derivation inside `_parse_artifacts`** — call private metadata-reading helpers (`_read_csv_metadata(csv_path)` / `_read_qhas_summary(qhas_path)`) from `_parse_artifacts` independently of `parse_*`. Slightly redundant (reads CSV/QHAS twice) but cleanly separates ABC contract from monitor-internal needs.
2. **Augment `parse_*` return types** — return `(list[OperatorMetrics], summary_dict)` tuples. Breaks the ABC signature and forces every future EP to deal with summary; rejected.

Recommendation: shape (1). The redundancy is one extra small JSON parse / CSV scan; not a hot path.

---

## 7. References

- **Spec:** `docs/design/perf/2026-05-03-op-trace-parser-interface-spec.md` v1.2 (716 lines)
- **Module PRD:** `docs/design/session/monitor/1_prd.md` v2.3 (358 lines)
- **Module coreloop:** `docs/design/session/monitor/2_coreloop.md` v2.3 (1210 lines)
- **Production-lift summary:** `docs/design/perf/2026-05-01-op-tracing-production-lift-summary.md`
- **Last impl commit:** `b77043a1` (QHAS authoritative)
- **Top-K fix:** `56549546`
- **Cardinal Rule #1:** project `CLAUDE.md` — "Never hardcode model architecture names, node/operator names, input/output tensor names, layer naming patterns, or any model-specific logic."
