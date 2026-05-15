# Response eval — 20260511-053000

## Overall

| Metric | with_skill | baseline | Δ |
|---|---|---|---|
| Pass rate | **97.3%** (36/37) | 62.2% (23/37) | +35.1pp |

## Per case

| Case | with_skill | baseline | Details |
|---|---|---|---|
| `eval-is-model-supported` | 6/6 | 3/6 | [comparison](eval-is-model-supported/comparison.md) |
| `eval-llm-out-of-scope` | 5/5 | 4/5 | [comparison](eval-llm-out-of-scope/comparison.md) |
| `eval-npu-vs-cpu-comparison` | 7/7 | 3/7 | [comparison](eval-npu-vs-cpu-comparison/comparison.md) |
| `eval-optimize-failure-recovery` | 6/6 | 4/6 | [comparison](eval-optimize-failure-recovery/comparison.md) |
| `eval-ryzen-ai-quick-benchmark` | 5/6 ⚠ | 3/6 | [comparison](eval-ryzen-ai-quick-benchmark/comparison.md) |
| `eval-snapdragon-resnet-build` | 7/7 | 6/7 | [comparison](eval-snapdragon-resnet-build/comparison.md) |

Cases marked ⚠ have failing assertions — open the linked `comparison.md` for response text and per-assertion evidence.
