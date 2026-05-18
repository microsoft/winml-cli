# DML (GPU) Test Report

## Summary

- **Models tested**: 63
- **Configs tested**: 63
- **Perf pass rate**: 18/63 (29%)
- **Eval pass rate**: 13/63 (21%)
- **Non-pass results**: 22 errors, 1 timeouts

## Results

| Model | Task | Config | Perf | Eval |
|------|------|------|------|------|
| BAAI/bge-base-en-v1.5 | feature-extraction | [config](BAAI_bge-base-en-v1.5/feature-extraction_config.json) | [9.66ms, 103.5sps](BAAI_bge-base-en-v1.5/feature-extraction_perf.json) | [cosine_spearman=89.2759](BAAI_bge-base-en-v1.5/feature-extraction_eval.json) |
|  | sentence-similarity | [config](BAAI_bge-base-en-v1.5/sentence-similarity_config.json) | [9.66ms, 103.5sps](BAAI_bge-base-en-v1.5/sentence-similarity_perf.json) | [cosine_spearman=89.2759](BAAI_bge-base-en-v1.5/sentence-similarity_eval.json) |
| BAAI/bge-large-en-v1.5 | sentence-similarity | [config](BAAI_bge-large-en-v1.5/sentence-similarity_config.json) | [53.96ms, 18.5sps](BAAI_bge-large-en-v1.5/sentence-similarity_perf.json) | [cosine_spearman=90.3971](BAAI_bge-large-en-v1.5/sentence-similarity_eval.json) |
| BAAI/bge-small-en-v1.5 | feature-extraction | [config](BAAI_bge-small-en-v1.5/feature-extraction_config.json) | [4.96ms, 201.8sps](BAAI_bge-small-en-v1.5/feature-extraction_perf.json) | [cosine_spearman=88.3662](BAAI_bge-small-en-v1.5/feature-extraction_eval.json) |
|  | sentence-similarity | [config](BAAI_bge-small-en-v1.5/sentence-similarity_config.json) | [4.97ms, 201.0sps](BAAI_bge-small-en-v1.5/sentence-similarity_perf.json) | [cosine_spearman=88.3662](BAAI_bge-small-en-v1.5/sentence-similarity_eval.json) |
| Babelscape/wikineural-multilingual-ner | token-classification | [config](Babelscape_wikineural-multilingual-ner/token-classification_config.json) | [7.17ms, 139.4sps](Babelscape_wikineural-multilingual-ner/token-classification_perf.json) | [overall_precision=0.8343 overall_recall=0.7966 overall_f1=0.8150 overall_accuracy=0.9673](Babelscape_wikineural-multilingual-ner/token-classification_eval.json) |
| cardiffnlp/twitter-roberta-base-sentiment-latest | text-classification | [config](cardiffnlp_twitter-roberta-base-sentiment-latest/text-classification_config.json) | [7.62ms, 131.3sps](cardiffnlp_twitter-roberta-base-sentiment-latest/text-classification_perf.json) | [accuracy=0.8100](cardiffnlp_twitter-roberta-base-sentiment-latest/text-classification_eval.json) |
| dbmdz/bert-large-cased-finetuned-conll03-english | token-classification | [config](dbmdz_bert-large-cased-finetuned-conll03-english/token-classification_config.json) | [20.53ms, 48.7sps](dbmdz_bert-large-cased-finetuned-conll03-english/token-classification_perf.json) | [overall_precision=0.9888 overall_recall=0.9944 overall_f1=0.9915 overall_accuracy=0.8837](dbmdz_bert-large-cased-finetuned-conll03-english/token-classification_eval.json) |
| deepset/bert-large-uncased-whole-word-masking-squad2 | question-answering | [config](deepset_bert-large-uncased-whole-word-masking-squad2/question-answering_config.json) | [31.97ms, 31.3sps](deepset_bert-large-uncased-whole-word-masking-squad2/question-answering_perf.json) | [exact=88.0000 f1=89.8500 total=100 HasAns_exact=87.5000 HasAns_f1=91.3542 HasAns_total=48 NoAns_exact=88.4615 NoAns_f1=88.4615 NoAns_total=52 best_exact=88.0000 best_exact_thresh=0.9955 best_f1=89.8500 best_f1_thresh=0.9955](deepset_bert-large-uncased-whole-word-masking-squad2/question-answering_eval.json) |
| deepset/roberta-base-squad2 | question-answering | [config](deepset_roberta-base-squad2/question-answering_config.json) | [7.26ms, 137.7sps](deepset_roberta-base-squad2/question-answering_perf.json) | [exact=79.0000 f1=82.3008 total=100 HasAns_exact=79.1667 HasAns_f1=86.0433 HasAns_total=48 NoAns_exact=78.8462 NoAns_f1=78.8462 NoAns_total=52 best_exact=79.0000 best_exact_thresh=0.9872 best_f1=82.3008 best_f1_thresh=0.9872](deepset_roberta-base-squad2/question-answering_eval.json) |
| deepset/tinyroberta-squad2 | question-answering | [config](deepset_tinyroberta-squad2/question-answering_config.json) | [4.81ms, 207.9sps](deepset_tinyroberta-squad2/question-answering_perf.json) | [exact=80.0000 f1=82.2500 total=100 HasAns_exact=79.1667 HasAns_f1=83.8542 HasAns_total=48 NoAns_exact=80.7692 NoAns_f1=80.7692 NoAns_total=52 best_exact=80.0000 best_exact_thresh=0.9948 best_f1=82.2500 best_f1_thresh=0.9948](deepset_tinyroberta-squad2/question-answering_eval.json) |
| dslim/bert-base-NER | token-classification | [config](dslim_bert-base-NER/token-classification_config.json) | [7.21ms, 138.7sps](dslim_bert-base-NER/token-classification_perf.json) | [overall_precision=0.9773 overall_recall=0.9718 overall_f1=0.9745 overall_accuracy=0.9967](dslim_bert-base-NER/token-classification_eval.json) |
| facebook/convnext-tiny-224 | image-classification | [config](facebook_convnext-tiny-224/image-classification_config.json) | [3.37ms, 296.4sps](facebook_convnext-tiny-224/image-classification_perf.json) | FAIL |
| facebook/dino-vitb16 | image-feature-extraction | [config](facebook_dino-vitb16/image-feature-extraction_config.json) | FAIL | FAIL |
| facebook/dino-vits16 | image-feature-extraction | [config](facebook_dino-vits16/image-feature-extraction_config.json) | [3.33ms, 300.1sps](facebook_dino-vits16/image-feature-extraction_perf.json) | FAIL |
| facebook/dinov2-base | image-feature-extraction | [config](facebook_dinov2-base/image-feature-extraction_config.json) | [5.90ms, 169.4sps](facebook_dinov2-base/image-feature-extraction_perf.json) | FAIL |
| facebook/dinov2-large | image-feature-extraction | [config](facebook_dinov2-large/image-feature-extraction_config.json) | [12.79ms, 78.2sps](facebook_dinov2-large/image-feature-extraction_perf.json) | FAIL |
| facebook/dinov2-small | image-feature-extraction | [config](facebook_dinov2-small/image-feature-extraction_config.json) | [3.57ms, 279.8sps](facebook_dinov2-small/image-feature-extraction_perf.json) | FAIL |
| FacebookAI/roberta-base | fill-mask | [config](FacebookAI_roberta-base/fill-mask_config.json) | [165.18ms, 6.0sps](FacebookAI_roberta-base/fill-mask_perf.json) | [pseudo_perplexity=5.2706 nll=1.6621](FacebookAI_roberta-base/fill-mask_eval.json) |
| FacebookAI/roberta-large | fill-mask | [config](FacebookAI_roberta-large/fill-mask_config.json) | FAIL | FAIL |
| FacebookAI/xlm-roberta-base | fill-mask | [config](FacebookAI_xlm-roberta-base/fill-mask_config.json) | FAIL | FAIL |
| FacebookAI/xlm-roberta-large | fill-mask | [config](FacebookAI_xlm-roberta-large/fill-mask_config.json) | FAIL | FAIL |
| google-bert/bert-base-multilingual-cased | feature-extraction | [config](google-bert_bert-base-multilingual-cased/feature-extraction_config.json) | FAIL | FAIL |
| google-bert/bert-base-multilingual-uncased | fill-mask | [config](google-bert_bert-base-multilingual-uncased/fill-mask_config.json) | FAIL | FAIL |
| google-bert/bert-base-uncased | fill-mask | [config](google-bert_bert-base-uncased/fill-mask_config.json) | FAIL | FAIL |
| google-bert/bert-large-uncased-whole-word-masking-finetuned-squad | question-answering | [config](google-bert_bert-large-uncased-whole-word-masking-finetuned-squad/question-answering_config.json) | FAIL | FAIL |
| google/vit-base-patch16-224 | image-classification | [config](google_vit-base-patch16-224/image-classification_config.json) | FAIL | FAIL |
| google/vit-base-patch16-224-in21k | image-feature-extraction | [config](google_vit-base-patch16-224-in21k/image-feature-extraction_config.json) | FAIL | FAIL |
| Intel/bert-base-uncased-mrpc | feature-extraction | [config](Intel_bert-base-uncased-mrpc/feature-extraction_config.json) | FAIL | FAIL |
|  | text-classification | [config](Intel_bert-base-uncased-mrpc/text-classification_config.json) | FAIL | FAIL |
| joeddav/xlm-roberta-large-xnli | zero-shot-classification | [config](joeddav_xlm-roberta-large-xnli/zero-shot-classification_config.json) | FAIL | FAIL |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | feature-extraction | [config](laion_CLIP-ViT-B-32-laion2B-s34B-b79K/feature-extraction_config.json) | FAIL | FAIL |
|  | zero-shot-image-classification | [config](laion_CLIP-ViT-B-32-laion2B-s34B-b79K/zero-shot-image-classification_config.json) | FAIL | FAIL |
| laion/CLIP-ViT-H-14-laion2B-s32B-b79K | zero-shot-image-classification | [config](laion_CLIP-ViT-H-14-laion2B-s32B-b79K/zero-shot-image-classification_config.json) | FAIL | FAIL |
| mattmdjaga/segformer_b2_clothes | image-segmentation | [config](mattmdjaga_segformer_b2_clothes/image-segmentation_config.json) | — | — |
| microsoft/rad-dino | image-feature-extraction | [config](microsoft_rad-dino/image-feature-extraction_config.json) | — | — |
| microsoft/resnet-50 | image-classification | [config](microsoft_resnet-50/image-classification_config.json) | — | — |
| microsoft/swin-large-patch4-window7-224 | image-classification | [config](microsoft_swin-large-patch4-window7-224/image-classification_config.json) | — | — |
| microsoft/table-transformer-detection | object-detection | [config](microsoft_table-transformer-detection/object-detection_config.json) | — | — |
| microsoft/trocr-base-handwritten | image-to-text | [config](microsoft_trocr-base-handwritten/image-to-text_config.json) | — | — |
| microsoft/trocr-base-printed | image-to-text | [config](microsoft_trocr-base-printed/image-to-text_config.json) | — | — |
| microsoft/trocr-large-handwritten | image-to-text | [config](microsoft_trocr-large-handwritten/image-to-text_config.json) | — | — |
| microsoft/trocr-large-printed | image-to-text | [config](microsoft_trocr-large-printed/image-to-text_config.json) | — | — |
| nvidia/segformer-b1-finetuned-ade-512-512 | image-segmentation | [config](nvidia_segformer-b1-finetuned-ade-512-512/image-segmentation_config.json) | — | — |
| nvidia/segformer-b2-finetuned-ade-512-512 | image-segmentation | [config](nvidia_segformer-b2-finetuned-ade-512-512/image-segmentation_config.json) | — | — |
| nvidia/segformer-b5-finetuned-ade-640-640 | image-segmentation | [config](nvidia_segformer-b5-finetuned-ade-640-640/image-segmentation_config.json) | — | — |
| openai/clip-vit-base-patch16 | feature-extraction | [config](openai_clip-vit-base-patch16/feature-extraction_config.json) | — | — |
|  | zero-shot-image-classification | [config](openai_clip-vit-base-patch16/zero-shot-image-classification_config.json) | — | — |
| openai/clip-vit-base-patch32 | feature-extraction | [config](openai_clip-vit-base-patch32/feature-extraction_config.json) | — | — |
|  | zero-shot-image-classification | [config](openai_clip-vit-base-patch32/zero-shot-image-classification_config.json) | — | — |
| openai/clip-vit-large-patch14 | zero-shot-image-classification | [config](openai_clip-vit-large-patch14/zero-shot-image-classification_config.json) | — | — |
| openai/clip-vit-large-patch14-336 | zero-shot-image-classification | [config](openai_clip-vit-large-patch14-336/zero-shot-image-classification_config.json) | — | — |
| patrickjohncyh/fashion-clip | zero-shot-image-classification | [config](patrickjohncyh_fashion-clip/zero-shot-image-classification_config.json) | — | — |
| ProsusAI/finbert | text-classification | [config](ProsusAI_finbert/text-classification_config.json) | — | — |
| rizvandwiki/gender-classification | image-classification | [config](rizvandwiki_gender-classification/image-classification_config.json) | — | — |
| Salesforce/blip-image-captioning-base | image-to-text | [config](Salesforce_blip-image-captioning-base/image-to-text_config.json) | — | — |
| sentence-transformers/all-MiniLM-L6-v2 | feature-extraction | [config](sentence-transformers_all-MiniLM-L6-v2/feature-extraction_config.json) | — | — |
|  | sentence-similarity | [config](sentence-transformers_all-MiniLM-L6-v2/sentence-similarity_config.json) | — | — |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | feature-extraction | [config](sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2/feature-extraction_config.json) | — | — |
|  | sentence-similarity | [config](sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2/sentence-similarity_config.json) | — | — |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | sentence-similarity | [config](sentence-transformers_paraphrase-multilingual-mpnet-base-v2/sentence-similarity_config.json) | — | — |
| StanfordAIMI/dinov2-base-xray-224 | image-feature-extraction | [config](StanfordAIMI_dinov2-base-xray-224/image-feature-extraction_config.json) | — | — |
| w11wo/indonesian-roberta-base-posp-tagger | token-classification | [config](w11wo_indonesian-roberta-base-posp-tagger/token-classification_config.json) | — | — |
