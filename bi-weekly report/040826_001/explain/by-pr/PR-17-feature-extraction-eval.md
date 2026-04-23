# PR-17: Implement Feature Extraction Evaluation (#190)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `f92b313` |
| Date | 2026-04-03 |
| Author | Zhenchao Ni (zhenchaoni) |
| PR Number | #190 |
| Files Changed | 16 |
| Insertions | +1,237 |
| Deletions | -36 |

## Summary
Implemented feature extraction evaluation support end-to-end. Added `FeatureExtractionEvaluator` using Spearman rank correlation as the similarity metric between WinML and PyTorch baseline feature embeddings. Added `WinMLModelForFeatureExtraction` model class. Extended `evaluate.py` to dispatch to the new evaluator, updated `datasets/__init__.py` and `eval/__init__.py` exports, and added `SpearmanCorrelation` metric to `eval/metrics/`. Expanded the e2e eval script baseline cache and model test set with feature extraction model entries.

## Files Changed
- `src/winml/modelkit/eval/feature_extraction_evaluator.py` — new evaluator (+149)
- `src/winml/modelkit/eval/metrics/spearman_correlation.py` — new metric (+57)
- `src/winml/modelkit/models/winml/feature_extraction.py` — new WinML model class (+57)
- `src/winml/modelkit/eval/evaluate.py` — dispatch integration (+18)
- `src/winml/modelkit/eval/base_evaluator.py` — minor update (+8/-8)
- `scripts/e2e_eval/utils/accuracy.py` — feature extraction accuracy support (+28/-10)
- `scripts/e2e_eval/run_eval.py` — feature extraction dispatch (+29/-4)
- `scripts/e2e_eval/testsets/models_with_acc.json` — +306 new model entries
- `tests/unit/eval/test_feature_extraction_evaluator.py` — new tests (+259)
- `tests/unit/models/auto/test_feature_extraction.py` — new tests (+139)
