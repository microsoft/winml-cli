# PR-07: Overwrite Op Results with Subgraph Results (#19)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `166a453` |
| Date | 2026-03-31 |
| Author | Charles Zhang (chinazhangchao) |
| PR Number | #19 |
| Files Changed | 15 |
| Insertions | +391 |
| Deletions | -119 |

## Summary
Fixed the runtime checker's result aggregation so that subgraph-level support results overwrite the parent op-level result when a subgraph match is found. Previously, a subgraph pattern match could be shadowed by the op's own baseline result, causing incorrect support classification. Also unified pattern registration naming across all pattern files (attention, gelu, gemm, layernorm, rmsnorm, transpose) and updated an OpenVINO NPU rule zip. Companion test updates ensure the new precedence logic is exercised.

## Files Changed
- `src/winml/modelkit/analyze/core/runtime_checker.py` — subgraph result overwrite logic (+36 lines)
- `src/winml/modelkit/analyze/core/runtime_checker_query.py` — query-level changes for subgraph handling (+364/-119)
- `src/winml/modelkit/pattern/` — attention, gelu, gemm, layernorm, rmsnorm, transpose pattern files — unified registration names
- `src/winml/modelkit/pattern/rules/default.json` — rule name update
- `tests/unit/analyze/pattern/test_rewrite_pipe.py` — updated tests
- `tests/unit/optim/pipes/test_pipe_rewrite.py` — updated tests
