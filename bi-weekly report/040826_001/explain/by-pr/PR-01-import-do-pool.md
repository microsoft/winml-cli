# PR-01: Import do_pool directly (#13)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `4f60333` |
| Date | 2026-03-26 |
| Author | Zhipeng Wang (timenick) |
| PR Number | #13 |
| Files Changed | 1 |
| Insertions | +12 |
| Deletions | -23 |

## Summary
Replaced a locally copied `_do_pool` function in `sam.py` with an inline import from `transformers.models.sam2.modeling_sam2.do_pool`, using a `try/except` fallback to a local implementation. This eliminates code duplication and guards against future internal API changes in HuggingFace Transformers by falling back gracefully if the upstream symbol is unavailable.

## Files Changed
- `src/winml/modelkit/models/hf/sam.py` — removed copied function body, added inline import with fallback
