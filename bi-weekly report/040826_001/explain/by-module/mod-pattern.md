# Module: pattern
**Path**: `src/winml/modelkit/pattern/`
**Period**: 2026-03-23 to 2026-04-08

## 1. Module Overview
The `pattern` module implements ONNX operator pattern matching and rewriting, and provides input generators (`op_input_gen/`) for generating valid operator inputs during runtime checking. It includes patterns for attention, GELU, GeMM, LayerNorm, RMSNorm, and transpose fusion.

## 2. Files Changed This Period
| File | PRs | Summary |
|------|-----|---------|
| `op_input_gen/op_input_gen.py` | #15, #17, #39, #47, #198, #23 | Large updates across multiple PRs: qdq_types rename (#17), import cleanup (#39, #47, #198), normalization before case signature (#23) |
| `op_input_gen/resize_input_generator.py` | #15, #204, #236 | New file in #15; QDQ extension (#204, #236) |
| `op_input_gen/slice_input_generator.py` | #15, #22, #244 | Batch update (#15); p1 coverage (#22); symbolic axis fix (#244) |
| `op_input_gen/unary_input_generator.py` | #15, #236 | Batch update; QDQ support added (#236) |
| `op_input_gen/conv_input_generator.py` | #17, #204, #39 | qdq_types rename (#17); QDQ extension (#204); import cleanup (#39) |
| `op_input_gen/matmul_input_generator.py` | #17, #204, #39 | qdq_types rename (#17); return fix (#204); import cleanup (#39) |
| `op_input_gen/normalization_input_generator.py` | #17, #39 | qdq_types rename; import cleanup |
| `op_input_gen/indexing_input_generator.py` | #22, #244 | p1 coverage (#22); Split n_outputs fix (#244) |
| `op_input_gen/squeeze_input_generator.py` | #22 | p1 coverage |
| `op_input_gen/qdq_gen.py` | #15, #39, #246 | Batch update; import cleanup; print→logger.debug (#246) |
| `op_input_gen/variadic_input_generator.py` | #15 | Major update (+228/-x) |
| `op_input_gen/shape_input_generator.py` | #15 | New file (+8 lines) |
| `attention_patterns.py` | #15, #19, #39, #198 | Batch update; registration name unification (#19); import cleanup |
| `gelu_patterns.py` | #15, #19, #39, #198 | Batch update; registration name unification; import cleanup |
| `base.py` | #15, #19, #39, #198 | Batch update; name unification; import cleanup |
| `transpose_patterns.py` | #15, #19, #39, #198 | Batch update; registration name unification; import cleanup |
| `layernorm_patterns.py` | #15, #19, #39, #198 | Batch update; import cleanup |
| `rmsnorm_patterns.py` | #15, #19, #39, #198 | Batch update; import cleanup |
| `gemm_patterns.py` | #15, #19, #39, #198 | Batch update; import cleanup |
| `__init__.py` | #15, #49 | Batch update; converted to relative imports (#49) |
| `rules/default.json` | #15, #19 | Pattern rule updates |

## 3. Net Change Summary
- `support_weight` and `support_activation` in `QDQParameterConfig` were replaced by a unified `qdq_types` field in PR #17, affecting 4 input generators.
- Three additional input generators (`resize`, `unary`, `indexing`) received QDQ type support in PRs #204 and #236.
- Symbolic dimension names in data shapes no longer cause crashes in `slice_input_generator.py` or `indexing_input_generator.py` (PR #244).
- `input_constraints` are now normalized before computing case signatures in `op_input_gen.py`, preventing incorrect cache behavior.
- `qdq_gen.py` debug `print()` calls replaced with `logger.debug()`, eliminating stdout pollution during `winml analyze`.
- All pattern `__init__.py` imports converted to relative paths; test imports converted to package-level.

## 4. New APIs/Functions Added
| Symbol | Description |
|--------|-------------|
| `op_input_gen/resize_input_generator.py` | New input generator for Resize op |
| `op_input_gen/shape_input_generator.py` | New input generator for Shape op |
