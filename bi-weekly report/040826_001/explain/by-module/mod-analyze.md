# Module: analyze
**Path**: `src/winml/modelkit/analyze/`
**Period**: 2026-03-23 to 2026-04-08

## 1. Module Overview
The `analyze` module provides static analysis of ONNX models against execution provider support rules. It includes a runtime checker (checking operator support per EP), pattern matching, information engine, and console output formatting.

## 2. Files Changed This Period
| File | PRs | Summary |
|------|-----|---------|
| `core/runtime_checker.py` | #19, #210, #221, #49 | Subgraph result overwrite logic; QNN pattern rule support; one-line bug fix; import cleanup |
| `core/runtime_checker_query.py` | #15, #19, #28, #39, #49, #244 | Large batch update in #15; subgraph query handling (#19); class rename (#28); import cleanup (#39, #49); symbolic axis guard (#244) |
| `runtime_checker/check_ops.py` | #15, #39, #198, #23, #235 | Major update in #15; import cleanup (#39); stale path fixes (#198); normalization fix (#23); refactored into op_utils (#235) |
| `runtime_checker/result_processor.py` | #15, #22, #39, #204, #23 | Import cleanup; p1 result handling (#22); normalization consistency (#23); minor update (#204) |
| `runtime_checker/case_runner.py` | #15, #39, #198 | New file in #15; import cleanup (#39); stale path fix (#198) |
| `utils/op_utils.py` | #235 | New utility file extracted from check_ops.py (+97 lines) |
| `console_writer.py` | #15, #200 | Batch update (#15); charmap/Unicode fix (#200) |
| `models/` | #15, #28, #47 | Batch update (#15); class renames (#28); public API expansion (#47) |
| `pattern/check_patterns.py` | #39, #47, #198 | Import cleanup; public API expansion; stale path fix |
| `__init__.py` | #47 | ~15 new public symbols exported |

## 3. Net Change Summary
- Subgraph-level support results now correctly overwrite op-level results in the runtime checker, fixing incorrect support classification for models with subgraph patterns.
- The `check_ops.py` logic for WinML registration was refactored: operator utility helpers extracted to the new `utils/op_utils.py`, reducing check_ops.py by ~102 lines.
- `input_constraints` normalization before case signature computation prevents duplicate cache entries.
- QNN pattern rules are now evaluated in the runtime checker alongside the existing op-level rules.
- Windows cp1252 charmap errors in `console_writer.py` fixed by removing `legacy_windows=False` and replacing non-ASCII symbols.
- The public API (`__init__.py`) was expanded with ~15 new exported symbols: `IHVType`, `SupportLevel`, `Action`, `ActionItem`, `ActionLevel`, `Information`, `EPSupport`, `ModelStats`, `ONNXModel`, `ONNXOp`, `RuntimeCheckRule`, `RuntimeTestResult`, `AlternativeType`, `ModelTag`, `RuleLoader`, `infer_ihv_from_ep_name`.

## 4. New APIs/Functions Added
| Symbol | Description |
|--------|-------------|
| `analyze.utils.op_utils` | New module with operator utility helpers extracted from check_ops.py |
| `IHVType`, `SupportLevel`, `Action`, `ActionItem`, `ActionLevel` | Exported from `analyze/__init__.py` (#47) |
| `Information`, `EPSupport`, `ModelStats`, `ONNXModel`, `ONNXOp` | Exported from `analyze/__init__.py` (#47) |
| `RuntimeCheckRule`, `RuntimeTestResult`, `AlternativeType`, `ModelTag`, `RuleLoader` | Exported from `analyze/__init__.py` (#47) |
