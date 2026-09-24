# QNN / GPU EP Optimization Sweep — Catalog Models

Generated: 2026-06-25T19:44:53  
EP: `qnn` / device: `gpu`  
Protocol: screen 200 iters, full 300×3 sessions + 2 confirm  

## Per-Model Results

| Model | Baseline p50 | Best p50 | Best config | Gain% | Verdict | Notes |
|-------|-------------|----------|-------------|-------|---------|-------|
| `BAAI/bge-small-en-v1.5` | 52.6 ms | N/A | N/A () | N/A | — | h8: screen bench failed; h9: BUILD_FAIL; h10: BUILD_FAIL; h1 |
| `apple/mobilevit-small` | 18.0 ms | N/A | N/A () | N/A | — | h8: screen bench failed; h10: screen bench failed |
| `deepset/roberta-base-squad2` | 99.5 ms | N/A | N/A () | N/A | — | h6: BUILD_FAIL; h7: BUILD_FAIL; h8: BUILD_FAIL; h9: BUILD_FA |
| `deepset/tinyroberta-squad2` | 51.2 ms | N/A | N/A () | N/A | — | h8: screen bench failed; h10: screen bench failed |
| `facebook/dinov2-small` | 26.4 ms | 22.0 ms | h12 (opset 17 + transpose_optimizer) | 16.7% | — | h8: screen bench failed; h10: screen bench failed |
| `microsoft/rad-dino` | 321.3 ms | N/A | N/A () | N/A | — | h8: screen bench failed; h10: screen bench failed |
| `microsoft/resnet-18` | 6.8 ms | 6.3 ms | h12 (opset 17 + transpose_optimizer) | 8.4% | — | none |
| `microsoft/swinv2-tiny-patch4-window16-256` | 57.5 ms | 57.5 ms | h0 (baseline FP32 (no quant, no compile)) | 0.0% | BASELINE_IS_BEST | none |
| `monologg/koelectra-small-v2-distilled-korquad-384` | 35.5 ms | 34.5 ms | h13 (no optimization (analyzer auto-optimization disabled, --no-analyze)) | 2.9% | BASELINE_IS_BEST | none |
| `nvidia/segformer-b1-finetuned-ade-512-512` | 116.8 ms | 90.3 ms | h13 (no optimization (analyzer auto-optimization disabled, --no-analyze)) | -1.9% | KEEP | none |
| `openai/clip-vit-base-patch32` | 15.7 ms | 15.5 ms | h2 (opset 19) | 1.1% | BASELINE_IS_BEST | h5: build failed; h6: skipped (disk pressure, free=2.8GB); h |
| `sentence-transformers/all-MiniLM-L6-v2` | 27.9 ms | N/A | N/A () | N/A | — | h2: BUILD_FAIL; h3: BUILD_FAIL; h4: BUILD_FAIL; h5: BUILD_FA |
| `sentence-transformers/all-mpnet-base-v2` | 108.7 ms | 104.7 ms | h13 (no optimization (analyzer auto-optimization disabled, --no-analyze)) | 3.6% | BASELINE_IS_BEST | none |

## Cross-Model Finding Checks

| Model | gpu-006 |
|---|---|
| `BAAI/bge-small-en-v1.5` | — |
| `apple/mobilevit-small` | — |
| `deepset/roberta-base-squad2` | — |
| `deepset/tinyroberta-squad2` | — |
| `facebook/dinov2-small` | — |
| `microsoft/rad-dino` | — |
| `microsoft/resnet-18` | — |
| `microsoft/swinv2-tiny-patch4-window16-256` | N/A (h1 not OK) |
| `monologg/koelectra-small-v2-distilled-korquad-384` | N/A (h1 not OK) |
| `nvidia/segformer-b1-finetuned-ade-512-512` | N/A (h1 not OK) |
| `openai/clip-vit-base-patch32` | N/A (h1 not OK) |
| `sentence-transformers/all-MiniLM-L6-v2` | — |
| `sentence-transformers/all-mpnet-base-v2` | N/A (h1 not OK) |
