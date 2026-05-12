# QNN (Qualcomm) GPU Test Report

## Summary

- **Models tested**: 56
- **Configs tested**: 192
- **Perf pass rate**: 162/192 (84%)
- **Eval pass rate**: 105/192 (55%)
- **Non-pass results**: 53 errors, 65 timeouts

## Notes

- This report is a concise summary generated from current result artifacts under this folder.
- Main timeout cluster is in `zero-shot-image-classification` eval workloads (multiple CLIP/SigLIP variants).
- Latest resumed run completed with: `PASS=27, FAIL=3, TIMEOUT=2, SKIP=160`.

## Result Files

- Perf pass files: `*_perf.json`
- Eval pass files: `*_eval.json`
- Error files: `*.error.txt`
- Timeout files: `*.timeout`

## Top Eval Non-Pass Models (by file count)

- BAAI/bge-small-en-v1.5: 6
- google/siglip-so400m-patch14-384: 3
- joeddav/xlm-roberta-large-xnli: 3
- laion/CLIP-ViT-B-32-laion2B-s34B-b79K: 3
- google-bert/bert-base-multilingual-uncased: 3
- google-bert/bert-base-uncased: 3
- google/siglip-base-patch16-224: 3
- laion/CLIP-ViT-H-14-laion2B-s32B-b79K: 3
- openai/clip-vit-large-patch14: 3
- openai/clip-vit-large-patch14-336: 3
- w11wo/indonesian-roberta-base-posp-tagger: 3
- microsoft/table-transformer-detection: 3
