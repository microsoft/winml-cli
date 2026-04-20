# AMD VitisAI Test Results

**Date:** April 20, 2026  
**EP:** VitisAI (`--ep vitisai`)  
**Device:** NPU  
**Precisions tested:** w8a8, w8a16, fp16  
**Pipeline:** `winml config` → `winml build` → `winml perf` (100 iterations) → `winml eval`

## Summary

- **Config:** 42/42 PASS
- **Build:** 42/42 produced ONNX output (compile step failed in all cases; w8a8/w8a16 → quantized.onnx, fp16 → optimized.onnx)
- **w8a8:** 12/14 fully passed (perf + eval), 2 eval-only failures
- **w8a16:** 0/14 passed — perf failed on all models
- **fp16:** 12/14 fully passed (perf + eval), 2 eval-only failures

## Detailed Results

| # | Model | Task | Precision | Config | Build | Perf | Eval | Logs |
|---|-------|------|-----------|--------|-------|------|------|------|
| 1 | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | feature-extraction | w8a8 | PASS | quantized (compile failed) | PASS | PASS | [logs](BAAI_bge-base-en-v1.5_feature-extraction/w8a8/) |
| 2 | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | feature-extraction | w8a16 | PASS | quantized (compile failed) | FAIL | FAIL | [logs](BAAI_bge-base-en-v1.5_feature-extraction/w8a16/) |
| 3 | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | feature-extraction | fp16 | PASS | optimized (quant failed) | PASS | PASS | [logs](BAAI_bge-base-en-v1.5_feature-extraction/fp16/) |
| 4 | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | sentence-similarity | w8a8 | PASS | quantized (compile failed) | PASS | PASS | [logs](BAAI_bge-base-en-v1.5_sentence-similarity/w8a8/) |
| 5 | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | sentence-similarity | w8a16 | PASS | quantized (compile failed) | FAIL | FAIL | [logs](BAAI_bge-base-en-v1.5_sentence-similarity/w8a16/) |
| 6 | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | sentence-similarity | fp16 | PASS | optimized (quant failed) | PASS | PASS | [logs](BAAI_bge-base-en-v1.5_sentence-similarity/fp16/) |
| 7 | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | feature-extraction | w8a8 | PASS | quantized (compile failed) | PASS | PASS | [logs](BAAI_bge-small-en-v1.5_feature-extraction/w8a8/) |
| 8 | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | feature-extraction | w8a16 | PASS | quantized (compile failed) | FAIL | FAIL | [logs](BAAI_bge-small-en-v1.5_feature-extraction/w8a16/) |
| 9 | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | feature-extraction | fp16 | PASS | optimized (quant failed) | PASS | PASS | [logs](BAAI_bge-small-en-v1.5_feature-extraction/fp16/) |
| 10 | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | sentence-similarity | w8a8 | PASS | quantized (compile failed) | PASS | PASS | [logs](BAAI_bge-small-en-v1.5_sentence-similarity/w8a8/) |
| 11 | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | sentence-similarity | w8a16 | PASS | quantized (compile failed) | FAIL | FAIL | [logs](BAAI_bge-small-en-v1.5_sentence-similarity/w8a16/) |
| 12 | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | sentence-similarity | fp16 | PASS | optimized (quant failed) | PASS | PASS | [logs](BAAI_bge-small-en-v1.5_sentence-similarity/fp16/) |
| 13 | [Babelscape/wikineural-multilingual-ner](https://huggingface.co/Babelscape/wikineural-multilingual-ner) | token-classification | w8a8 | PASS | quantized (compile failed) | PASS | PASS | [logs](Babelscape_wikineural-multilingual-ner_token-classification/w8a8/) |
| 14 | [Babelscape/wikineural-multilingual-ner](https://huggingface.co/Babelscape/wikineural-multilingual-ner) | token-classification | w8a16 | PASS | quantized (compile failed) | FAIL | FAIL | [logs](Babelscape_wikineural-multilingual-ner_token-classification/w8a16/) |
| 15 | [Babelscape/wikineural-multilingual-ner](https://huggingface.co/Babelscape/wikineural-multilingual-ner) | token-classification | fp16 | PASS | optimized (quant failed) | PASS | PASS | [logs](Babelscape_wikineural-multilingual-ner_token-classification/fp16/) |
| 16 | [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) | token-classification | w8a8 | PASS | quantized (compile failed) | PASS | PASS | [logs](dslim_bert-base-NER_token-classification/w8a8/) |
| 17 | [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) | token-classification | w8a16 | PASS | quantized (compile failed) | FAIL | FAIL | [logs](dslim_bert-base-NER_token-classification/w8a16/) |
| 18 | [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) | token-classification | fp16 | PASS | optimized (quant failed) | PASS | PASS | [logs](dslim_bert-base-NER_token-classification/fp16/) |
| 19 | [facebook/convnext-tiny-224](https://huggingface.co/facebook/convnext-tiny-224) | image-classification | w8a8 | PASS | quantized (compile failed) | PASS | PASS | [logs](facebook_convnext-tiny-224_image-classification/w8a8/) |
| 20 | [facebook/convnext-tiny-224](https://huggingface.co/facebook/convnext-tiny-224) | image-classification | w8a16 | PASS | quantized (compile failed) | FAIL | FAIL | [logs](facebook_convnext-tiny-224_image-classification/w8a16/) |
| 21 | [facebook/convnext-tiny-224](https://huggingface.co/facebook/convnext-tiny-224) | image-classification | fp16 | PASS | optimized (quant failed) | PASS | PASS | [logs](facebook_convnext-tiny-224_image-classification/fp16/) |
| 22 | [google-bert/bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased) | feature-extraction | w8a8 | PASS | quantized (compile failed) | PASS | PASS | [logs](google-bert_bert-base-multilingual-cased_feature-extraction/w8a8/) |
| 23 | [google-bert/bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased) | feature-extraction | w8a16 | PASS | quantized (compile failed) | FAIL | FAIL | [logs](google-bert_bert-base-multilingual-cased_feature-extraction/w8a16/) |
| 24 | [google-bert/bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased) | feature-extraction | fp16 | PASS | optimized (quant failed) | PASS | PASS | [logs](google-bert_bert-base-multilingual-cased_feature-extraction/fp16/) |
| 25 | [Intel/bert-base-uncased-mrpc](https://huggingface.co/Intel/bert-base-uncased-mrpc) | feature-extraction | w8a8 | PASS | quantized (compile failed) | PASS | PASS | [logs](Intel_bert-base-uncased-mrpc_feature-extraction/w8a8/) |
| 26 | [Intel/bert-base-uncased-mrpc](https://huggingface.co/Intel/bert-base-uncased-mrpc) | feature-extraction | w8a16 | PASS | quantized (compile failed) | FAIL | FAIL | [logs](Intel_bert-base-uncased-mrpc_feature-extraction/w8a16/) |
| 27 | [Intel/bert-base-uncased-mrpc](https://huggingface.co/Intel/bert-base-uncased-mrpc) | feature-extraction | fp16 | PASS | optimized (quant failed) | PASS | PASS | [logs](Intel_bert-base-uncased-mrpc_feature-extraction/fp16/) |
| 28 | [Intel/bert-base-uncased-mrpc](https://huggingface.co/Intel/bert-base-uncased-mrpc) | text-classification | w8a8 | PASS | quantized (compile failed) | PASS | PASS | [logs](Intel_bert-base-uncased-mrpc_text-classification/w8a8/) |
| 29 | [Intel/bert-base-uncased-mrpc](https://huggingface.co/Intel/bert-base-uncased-mrpc) | text-classification | w8a16 | PASS | quantized (compile failed) | FAIL | FAIL | [logs](Intel_bert-base-uncased-mrpc_text-classification/w8a16/) |
| 30 | [Intel/bert-base-uncased-mrpc](https://huggingface.co/Intel/bert-base-uncased-mrpc) | text-classification | fp16 | PASS | optimized (quant failed) | PASS | PASS | [logs](Intel_bert-base-uncased-mrpc_text-classification/fp16/) |
| 31 | [microsoft/table-transformer-detection](https://huggingface.co/microsoft/table-transformer-detection) | object-detection | w8a8 | PASS | quantized (compile failed) | PASS | **FAIL** | [logs](microsoft_table-transformer-detection_object-detection/w8a8/) |
| 32 | [microsoft/table-transformer-detection](https://huggingface.co/microsoft/table-transformer-detection) | object-detection | w8a16 | PASS | quantized (compile failed) | FAIL | FAIL | [logs](microsoft_table-transformer-detection_object-detection/w8a16/) |
| 33 | [microsoft/table-transformer-detection](https://huggingface.co/microsoft/table-transformer-detection) | object-detection | fp16 | PASS | optimized (quant failed) | PASS | **FAIL** | [logs](microsoft_table-transformer-detection_object-detection/fp16/) |
| 34 | [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) | text-classification | w8a8 | PASS | quantized (compile failed) | PASS | **FAIL** | [logs](ProsusAI_finbert_text-classification/w8a8/) |
| 35 | [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) | text-classification | w8a16 | PASS | quantized (compile failed) | FAIL | FAIL | [logs](ProsusAI_finbert_text-classification/w8a16/) |
| 36 | [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) | text-classification | fp16 | PASS | optimized (quant failed) | PASS | **FAIL** | [logs](ProsusAI_finbert_text-classification/fp16/) |
| 37 | [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | feature-extraction | w8a8 | PASS | quantized (compile failed) | PASS | PASS | [logs](sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2_feature-extraction/w8a8/) |
| 38 | [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | feature-extraction | w8a16 | PASS | quantized (compile failed) | FAIL | FAIL | [logs](sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2_feature-extraction/w8a16/) |
| 39 | [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | feature-extraction | fp16 | PASS | optimized (quant failed) | PASS | PASS | [logs](sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2_feature-extraction/fp16/) |
| 40 | [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | sentence-similarity | w8a8 | PASS | quantized (compile failed) | PASS | PASS | [logs](sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2_sentence-similarity/w8a8/) |
| 41 | [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | sentence-similarity | w8a16 | PASS | quantized (compile failed) | FAIL | FAIL | [logs](sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2_sentence-similarity/w8a16/) |
| 42 | [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | sentence-similarity | fp16 | PASS | optimized (quant failed) | PASS | PASS | [logs](sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2_sentence-similarity/fp16/) |

## Key Findings

1. **w8a16 is completely broken on VitisAI** — all 14 model+task combinations failed at perf. This precision mode appears unsupported by the VitisAI EP.

2. **w8a8 and fp16 work well** — 12/14 model+task combinations pass both perf and eval with these precisions.

3. **Compile always fails** — the VitisAI compile step failed for all 42 tests. For w8a8/w8a16, the pipeline stopped at `quantized.onnx`; for fp16, it stopped at `optimized.onnx` (quantization also failed since fp16 doesn't apply to VitisAI quantization).

4. **Two models have eval failures across all precisions:**
   - **microsoft/table-transformer-detection** (object-detection) — perf passes but eval fails on w8a8 and fp16
   - **ProsusAI/finbert** (text-classification) — perf passes but eval fails on w8a8 and fp16

## Notes

- Each test folder contains: `commands.txt`, `config.json`, `build/`, `config_log.txt`, `build_log.txt`, `perf_log.txt`, `eval_log.txt`, `perf.json` (if passed), `eval.json` (if passed), and `status.txt`.
- Perf was run with `--ep vitisai --iterations 100`.
- Eval was run with `--device npu`.
