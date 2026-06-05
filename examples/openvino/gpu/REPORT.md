# OpenVINO (Intel, GPU) Report

## Summary

Counts canonical `(model, task)` pairs from `scripts/e2e_eval/testsets/example_model_tasks.txt`.

- (Model, Task): 77
- Configs: 77
- Eval Pass: 29/77 (38%)

## Results

| Model | Task | Precision | Config | Eval |
|---|---|---|---|---|
| AdamCodd/vit-base-nsfw-detector | image-classification |  | [config](./AdamCodd_vit-base-nsfw-detector/image-classification_config.json) | — |
| ahotrod/electra_large_discriminator_squad2_512 | question-answering |  | [config](./ahotrod_electra_large_discriminator_squad2_512/question-answering_config.json) | — |
| amunchet/rorshark-vit-base | image-classification |  | [config](./amunchet_rorshark-vit-base/image-classification_config.json) | — |
| apple/mobilevit-small | image-classification |  | [config](./apple_mobilevit-small/image-classification_config.json) | — |
| BAAI/bge-base-en-v1.5 | feature-extraction |  | [config](./BAAI_bge-base-en-v1.5/feature-extraction_config.json) | — |
|  | sentence-similarity |  | [config](./BAAI_bge-base-en-v1.5/sentence-similarity_config.json) | — |
| BAAI/bge-large-en-v1.5 | sentence-similarity |  | [config](./BAAI_bge-large-en-v1.5/sentence-similarity_config.json) | cosine_spearman=90.3971 ([metric](./BAAI_bge-large-en-v1.5/sentence-similarity_eval_result.json)) |
| BAAI/bge-m3 | feature-extraction |  | [config](./BAAI_bge-m3/feature-extraction_config.json) | — |
|  | sentence-similarity |  | [config](./BAAI_bge-m3/sentence-similarity_config.json) | — |
| BAAI/bge-reranker-v2-m3 | text-classification |  | [config](./BAAI_bge-reranker-v2-m3/text-classification_config.json) | — |
| BAAI/bge-small-en-v1.5 | feature-extraction |  | [config](./BAAI_bge-small-en-v1.5/feature-extraction_config.json) | cosine_spearman=88.3662 ([metric](./BAAI_bge-small-en-v1.5/feature-extraction_eval_result.json)) |
|  | sentence-similarity |  | [config](./BAAI_bge-small-en-v1.5/sentence-similarity_config.json) | cosine_spearman=88.3662 ([metric](./BAAI_bge-small-en-v1.5/sentence-similarity_eval_result.json)) |
| Babelscape/wikineural-multilingual-ner | token-classification |  | [config](./Babelscape_wikineural-multilingual-ner/token-classification_config.json) | — |
| cardiffnlp/twitter-roberta-base-sentiment-latest | text-classification |  | [config](./cardiffnlp_twitter-roberta-base-sentiment-latest/text-classification_config.json) | accuracy=0.8100 ([metric](./cardiffnlp_twitter-roberta-base-sentiment-latest/text-classification_eval_result.json)) |
| cross-encoder/ms-marco-MiniLM-L4-v2 | text-classification |  | [config](./cross-encoder_ms-marco-MiniLM-L4-v2/text-classification_config.json) | — |
| cross-encoder/ms-marco-MiniLM-L6-v2 | text-classification |  | [config](./cross-encoder_ms-marco-MiniLM-L6-v2/text-classification_config.json) | — |
| deepset/bert-large-uncased-whole-word-masking-squad2 | question-answering |  | [config](./deepset_bert-large-uncased-whole-word-masking-squad2/question-answering_config.json) | exact=88.0000, f1=89.8500, total=100, HasAns_exact=87.5000, HasAns_f1=91.3542, HasAns_total=48, NoAns_exact=88.4615, NoAns_f1=88.4615, NoAns_total=52, best_exact=88.0000, best_exact_thresh=0.9955, best_f1=89.8500, best_f1_thresh=0.9955 ([metric](./deepset_bert-large-uncased-whole-word-masking-squad2/question-answering_eval_result.json)) |
| deepset/roberta-base-squad2 | question-answering |  | [config](./deepset_roberta-base-squad2/question-answering_config.json) | exact=79.0000, f1=82.3008, total=100, HasAns_exact=79.1667, HasAns_f1=86.0433, HasAns_total=48, NoAns_exact=78.8462, NoAns_f1=78.8462, NoAns_total=52, best_exact=79.0000, best_exact_thresh=0.9873, best_f1=82.3008, best_f1_thresh=0.9873 ([metric](./deepset_roberta-base-squad2/question-answering_eval_result.json)) |
| deepset/tinyroberta-squad2 | question-answering |  | [config](./deepset_tinyroberta-squad2/question-answering_config.json) | exact=80.0000, f1=82.2500, total=100, HasAns_exact=79.1667, HasAns_f1=83.8542, HasAns_total=48, NoAns_exact=80.7692, NoAns_f1=80.7692, NoAns_total=52, best_exact=80.0000, best_exact_thresh=0.9948, best_f1=82.2500, best_f1_thresh=0.9948 ([metric](./deepset_tinyroberta-squad2/question-answering_eval_result.json)) |
| dima806/fairface_age_image_detection | image-classification |  | [config](./dima806_fairface_age_image_detection/image-classification_config.json) | — |
| distilbert/distilbert-base-cased-distilled-squad | question-answering |  | [config](./distilbert_distilbert-base-cased-distilled-squad/question-answering_config.json) | — |
| distilbert/distilbert-base-uncased | fill-mask |  | [config](./distilbert_distilbert-base-uncased/fill-mask_config.json) | — |
| distilbert/distilbert-base-uncased-distilled-squad | question-answering |  | [config](./distilbert_distilbert-base-uncased-distilled-squad/question-answering_config.json) | — |
| distilbert/distilbert-base-uncased-finetuned-sst-2-english | text-classification |  | [config](./distilbert_distilbert-base-uncased-finetuned-sst-2-english/text-classification_config.json) | — |
| dslim/bert-base-NER | token-classification |  | [config](./dslim_bert-base-NER/token-classification_config.json) | — |
| facebook/convnext-tiny-224 | image-classification |  | [config](./facebook_convnext-tiny-224/image-classification_config.json) | — |
| facebook/dino-vitb16 | image-feature-extraction |  | [config](./facebook_dino-vitb16/image-feature-extraction_config.json) | knn_top1_accuracy=83.2000, knn_top5_accuracy=95.5000 ([metric](./facebook_dino-vitb16/image-feature-extraction_eval_result.json)) |
| facebook/dino-vits16 | image-feature-extraction |  | [config](./facebook_dino-vits16/image-feature-extraction_config.json) | knn_top1_accuracy=79.7000, knn_top5_accuracy=94.5000 ([metric](./facebook_dino-vits16/image-feature-extraction_eval_result.json)) |
| facebook/dinov2-base | image-feature-extraction |  | [config](./facebook_dinov2-base/image-feature-extraction_config.json) | knn_top1_accuracy=89.1000, knn_top5_accuracy=97.8000 ([metric](./facebook_dinov2-base/image-feature-extraction_eval_result.json)) |
| facebook/dinov2-large | image-feature-extraction |  | [config](./facebook_dinov2-large/image-feature-extraction_config.json) | knn_top1_accuracy=91.0000, knn_top5_accuracy=97.7000 ([metric](./facebook_dinov2-large/image-feature-extraction_eval_result.json)) |
| facebook/dinov2-small | image-feature-extraction |  | [config](./facebook_dinov2-small/image-feature-extraction_config.json) | knn_top1_accuracy=85.5000, knn_top5_accuracy=96.6000 ([metric](./facebook_dinov2-small/image-feature-extraction_eval_result.json)) |
| FacebookAI/roberta-base | fill-mask |  | [config](./FacebookAI_roberta-base/fill-mask_config.json) | pseudo_perplexity=5.2705, nll=1.6621 ([metric](./FacebookAI_roberta-base/fill-mask_eval_result.json)) |
| FacebookAI/xlm-roberta-base | fill-mask |  | [config](./FacebookAI_xlm-roberta-base/fill-mask_config.json) | pseudo_perplexity=3.6438, nll=1.2930 ([metric](./FacebookAI_xlm-roberta-base/fill-mask_eval_result.json)) |
| Falconsai/nsfw_image_detection | image-classification |  | [config](./Falconsai_nsfw_image_detection/image-classification_config.json) | — |
| google-bert/bert-base-multilingual-cased | feature-extraction |  | [config](./google-bert_bert-base-multilingual-cased/feature-extraction_config.json) | cosine_spearman=41.0034 ([metric](./google-bert_bert-base-multilingual-cased/feature-extraction_eval_result.json)) |
|  | fill-mask |  | [config](./google-bert_bert-base-multilingual-cased/fill-mask_config.json) | — |
|  | masked-lm |  | [config](./google-bert_bert-base-multilingual-cased/masked-lm_config.json) | — |
| google-bert/bert-base-multilingual-uncased | fill-mask |  | [config](./google-bert_bert-base-multilingual-uncased/fill-mask_config.json) | pseudo_perplexity=4.4644, nll=1.4961 ([metric](./google-bert_bert-base-multilingual-uncased/fill-mask_eval_result.json)) |
| google-bert/bert-base-uncased | fill-mask |  | [config](./google-bert_bert-base-uncased/fill-mask_config.json) | pseudo_perplexity=4.2980, nll=1.4582 ([metric](./google-bert_bert-base-uncased/fill-mask_eval_result.json)) |
| google/vit-base-patch16-224 | image-classification |  | [config](./google_vit-base-patch16-224/image-classification_config.json) | accuracy=0.7000 ([metric](./google_vit-base-patch16-224/image-classification_eval_result.json)) |
| google/vit-base-patch16-224-in21k | image-feature-extraction |  | [config](./google_vit-base-patch16-224-in21k/image-feature-extraction_config.json) | knn_top1_accuracy=91.9000, knn_top5_accuracy=97.0000 ([metric](./google_vit-base-patch16-224-in21k/image-feature-extraction_eval_result.json)) |
| hustvl/yolos-small | object-detection |  | [config](./hustvl_yolos-small/object-detection_config.json) | — |
| Intel/bert-base-uncased-mrpc | feature-extraction |  | [config](./Intel_bert-base-uncased-mrpc/feature-extraction_config.json) | — |
|  | text-classification |  | [config](./Intel_bert-base-uncased-mrpc/text-classification_config.json) | — |
| Intel/dpt-hybrid-midas | depth-estimation |  | [config](./Intel_dpt-hybrid-midas/depth-estimation_config.json) | — |
| Isotonic/distilbert_finetuned_ai4privacy_v2 | token-classification |  | [config](./Isotonic_distilbert_finetuned_ai4privacy_v2/token-classification_config.json) | — |
| Jean-Baptiste/camembert-ner-with-dates | token-classification |  | [config](./Jean-Baptiste_camembert-ner-with-dates/token-classification_config.json) | — |
| kredor/punctuate-all | token-classification |  | [config](./kredor_punctuate-all/token-classification_config.json) | — |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | feature-extraction |  | [config](./laion_CLIP-ViT-B-32-laion2B-s34B-b79K/feature-extraction_config.json) | cosine_spearman=75.4619 ([metric](./laion_CLIP-ViT-B-32-laion2B-s34B-b79K/feature-extraction_eval_result.json)) |
|  | zero-shot-image-classification |  | [config](./laion_CLIP-ViT-B-32-laion2B-s34B-b79K/zero-shot-image-classification_config.json) | — |
| lxyuan/distilbert-base-multilingual-cased-sentiments-student | zero-shot-classification |  | [config](./lxyuan_distilbert-base-multilingual-cased-sentiments-student/zero-shot-classification_config.json) | — |
| microsoft/beit-base-patch16-224-pt22k-ft22k | image-classification |  | [config](./microsoft_beit-base-patch16-224-pt22k-ft22k/image-classification_config.json) | — |
| microsoft/rad-dino | image-feature-extraction |  | [config](./microsoft_rad-dino/image-feature-extraction_config.json) | knn_top1_accuracy=94.6735, knn_top5_accuracy=99.3127 ([metric](./microsoft_rad-dino/image-feature-extraction_eval_result.json)) |
| microsoft/resnet-18 | image-classification |  | [config](./microsoft_resnet-18/image-classification_config.json) | — |
| microsoft/swin-large-patch4-window7-224 | image-classification |  | [config](./microsoft_swin-large-patch4-window7-224/image-classification_config.json) | — |
| microsoft/swinv2-tiny-patch4-window16-256 | image-classification |  | [config](./microsoft_swinv2-tiny-patch4-window16-256/image-classification_config.json) | — |
| monologg/koelectra-small-v2-distilled-korquad-384 | question-answering |  | [config](./monologg_koelectra-small-v2-distilled-korquad-384/question-answering_config.json) | — |
| openai/clip-vit-base-patch16 | feature-extraction |  | [config](./openai_clip-vit-base-patch16/feature-extraction_config.json) | cosine_spearman=68.3395 ([metric](./openai_clip-vit-base-patch16/feature-extraction_eval_result.json)) |
| openai/clip-vit-base-patch32 | feature-extraction |  | [config](./openai_clip-vit-base-patch32/feature-extraction_config.json) | cosine_spearman=63.8115 ([metric](./openai_clip-vit-base-patch32/feature-extraction_eval_result.json)) |
|  | zero-shot-image-classification |  | [config](./openai_clip-vit-base-patch32/zero-shot-image-classification_config_image-encoder.json) | top1_accuracy=58.3000, top5_accuracy=84.0000 ([metric](./openai_clip-vit-base-patch32/zero-shot-image-classification_eval_result.json)) |
| patrickjohncyh/fashion-clip | zero-shot-image-classification |  | [config](./patrickjohncyh_fashion-clip/zero-shot-image-classification_config_image-encoder.json) | — |
| ProsusAI/finbert | text-classification |  | [config](./ProsusAI_finbert/text-classification_config.json) | — |
| rizvandwiki/gender-classification | image-classification |  | [config](./rizvandwiki_gender-classification/image-classification_config.json) | FAIL |
| sentence-transformers/all-MiniLM-L6-v2 | feature-extraction |  | [config](./sentence-transformers_all-MiniLM-L6-v2/feature-extraction_config.json) | cosine_spearman=81.2943 ([metric](./sentence-transformers_all-MiniLM-L6-v2/feature-extraction_eval_result.json)) |
|  | sentence-similarity |  | [config](./sentence-transformers_all-MiniLM-L6-v2/sentence-similarity_config.json) | cosine_spearman=81.2943 ([metric](./sentence-transformers_all-MiniLM-L6-v2/sentence-similarity_eval_result.json)) |
| sentence-transformers/all-mpnet-base-v2 | feature-extraction |  | [config](./sentence-transformers_all-mpnet-base-v2/feature-extraction_config.json) | — |
|  | fill-mask |  | [config](./sentence-transformers_all-mpnet-base-v2/fill-mask_config.json) | — |
|  | sentence-similarity |  | [config](./sentence-transformers_all-mpnet-base-v2/sentence-similarity_config.json) | — |
| sentence-transformers/multi-qa-mpnet-base-dot-v1 | feature-extraction |  | [config](./sentence-transformers_multi-qa-mpnet-base-dot-v1/feature-extraction_config.json) | — |
|  | fill-mask |  | [config](./sentence-transformers_multi-qa-mpnet-base-dot-v1/fill-mask_config.json) | — |
|  | sentence-similarity |  | [config](./sentence-transformers_multi-qa-mpnet-base-dot-v1/sentence-similarity_config.json) | — |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | feature-extraction |  | [config](./sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2/feature-extraction_config.json) | cosine_spearman=84.5004 ([metric](./sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2/feature-extraction_eval_result.json)) |
|  | sentence-similarity |  | [config](./sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2/sentence-similarity_config.json) | cosine_spearman=84.5004 ([metric](./sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2/sentence-similarity_eval_result.json)) |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | sentence-similarity |  | [config](./sentence-transformers_paraphrase-multilingual-mpnet-base-v2/sentence-similarity_config.json) | cosine_spearman=86.0325 ([metric](./sentence-transformers_paraphrase-multilingual-mpnet-base-v2/sentence-similarity_eval_result.json)) |
| tau/splinter-base | question-answering |  | [config](./tau_splinter-base/question-answering_config.json) | — |
| valentinafeve/yolos-fashionpedia | object-detection |  | [config](./valentinafeve_yolos-fashionpedia/object-detection_config.json) | — |
| w11wo/indonesian-roberta-base-posp-tagger | token-classification |  | [config](./w11wo_indonesian-roberta-base-posp-tagger/token-classification_config.json) | FAIL |
