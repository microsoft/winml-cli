# Module: optim
**Path**: `src/winml/modelkit/optim/`
**Period**: 2026-03-23 to 2026-04-08

## 1. Module Overview
The `optim` module provides the graph optimization pipeline for ONNX models, including fusion pipes, graph rewriting, surgery pipes, and capability checking. It exposes the `Optimizer` class as the primary entry point.

## 2. Files Changed This Period
| File | PRs | Summary |
|------|-----|---------|
| `pipes/fusion.py` | #198, #237 | Relative import fix (#198); .onnx.data cleanup (#237) |
| `pipes/graph.py` | #15, #198, #237 | Batch update (#15); relative import fix (#198); .onnx.data cleanup (+11/-5) (#237) |
| `pipes/rewrite_rules.py` | #45, #198, #246 | Import fix (#45, #198); ASCII arrow fix for cp1252 (#246) |
| `pipes/rewrite.py` | #198 | Relative import fix |
| `capabilities/graph.py` | #246 | ASCII arrow fix |
| `capabilities/layernorm.py` | #246 | ASCII arrow fix |
| `capabilities/surgery.py` | #246 | ASCII arrow fix |
| `api.py` | #198, #212 | Import fix (#198); guard added for SAM2 double-optimize (#212) |
| `optimizer.py` | #15 | Minor update |
| `__init__.py` | #45 | `Optimizer` exported |

## 3. Net Change Summary
- `.onnx.data` external data files are now cleaned up alongside temporary `.onnx` files in both `fusion.py` and `graph.py`, preventing temp directory accumulation for large models.
- `Optimizer` was added to `optim/__init__.py` so test and consumer code can import it from the package level without reaching into internal submodules.
- Non-ASCII Unicode arrows in capability description strings were replaced with ASCII `->` to prevent cp1252 encoding errors on Windows terminals in `winml optimize --help`.
- ~35 absolute import paths in optim source files were converted to relative imports as part of the import cleanup series.

## 4. New APIs/Functions Added
| Symbol | Description |
|--------|-------------|
| `Optimizer` | Exported from `optim/__init__.py` (#45) |
