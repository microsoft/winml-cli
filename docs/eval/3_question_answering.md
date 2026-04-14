# Question Answering Evaluation — Design Spec

## Overview

The QA evaluator measures extractive question answering accuracy by comparing ONNX model predictions against a PyTorch baseline on SQuAD-format datasets. It wraps the HuggingFace `evaluate` library's `QuestionAnsweringEvaluator` and handles WinML-specific concerns (static ONNX shapes, SQuAD v2 auto-detection, optional input filtering).

**Evaluator class**: `WinMLQuestionAnsweringEvaluator` (extends `WinMLEvaluator`)
**Model class**: `WinMLModelForQuestionAnswering` (extends `WinMLPreTrainedModel`)

## HF Pipeline Output

The HuggingFace `QuestionAnsweringPipeline` performs extractive QA — it finds the answer span within the given context, rather than generating new text.

Given a question and a context passage, the pipeline tokenizes both, feeds them to the model which produces `start_logits` and `end_logits`, then extracts the highest-scoring text span from the context. For example, given question `"In what country is Normandy located?"` and a context containing `"...Normandy, a region in France."`, the model's logits peak at the token positions for "France", and the pipeline returns `{"answer": "France", "score": 0.73, "start": 94, "end": 100}`.

For SQuAD v2, the pipeline additionally computes `no_answer_probability` — when this exceeds a threshold, the predicted answer is empty (`""`).

## Metrics and Datasets

### SQuAD v1

Every question has an answer span in the context. The predicted answer text is compared against the reference answer(s) using two metrics:

- **Exact Match (EM)**: Both texts are normalized (lowercase, strip articles/punctuation/whitespace), then compared. EM = 1 if the normalized prediction matches any reference answer exactly.
- **F1**: Both texts are split into word tokens, then bag-of-words precision and recall are computed. F1 = 2 * precision * recall / (precision + recall). When multiple reference answers exist, F1 is computed against each and the maximum is taken. This gives partial credit for partial matches — e.g., predicting "artificial intelligence" when the reference is "artificial intelligence and machine learning" yields F1 = 0.57 rather than 0.

### SQuAD v2

Extends v1 by adding unanswerable questions (~50% of the dataset), where the correct answer is empty. The `squad_v2` metric additionally handles no-answer predictions: if the model's `no_answer_probability` exceeds a threshold, the predicted answer is empty and compared accordingly. The metric reports both `HasAns` and `NoAns` breakdowns alongside the overall scores.

### F1 as the primary metric

F1 is used as the primary accuracy metric (`wmk_metric_key: "f1"`) because it provides partial credit and is more stable than EM. Like other accuracy metrics (classification accuracy, mAP), F1 is compared relatively between the quantized ONNX model and the PyTorch baseline — a relative delta under 5% is PASS, 5–10% is AT_RISK, over 10% is REGRESSION.

### Datasets

| Dataset | Format | Description |
|---------|--------|-------------|
| `rajpurkar/squad` (v1) | Answerable only | 10,570 validation samples |
| `rajpurkar/squad_v2` (v2) | Answerable + unanswerable | 11,873 validation samples |
| `KorQuAD/squad_kor_v1` | Korean, answerable only | 5,774 validation samples |

All share the same schema: `question`, `context`, `id`, `answers: {text: [str], answer_start: [int]}`.

## Implementation

### Evaluator: `WinMLQuestionAnsweringEvaluator`

- Delegates evaluation to HF `evaluator("question-answering")`
- **SQuAD v2 auto-detection**: `compute()` inspects the dataset via `is_squad_v2_format()` to check for unanswerable questions, then passes the matching `metric` (`"squad"` or `"squad_v2"`) and `squad_v2_format` flag to the HF evaluator
- Without this detection, HF defaults to the `squad` metric and crashes on v2 datasets because the pipeline produces `no_answer_probability` that the v1 metric doesn't expect
- `prepare_pipeline()` configures tokenizer padding to `max_length` matching the ONNX model's static sequence length (e.g., 512)

### Model: `WinMLModelForQuestionAnswering`

- Thin inference wrapper accepting `input_ids`, `attention_mask`, and `token_type_ids` from the HF pipeline
- Returns `QuestionAnsweringModelOutput` with `start_logits` and `end_logits`
- **Filters `token_type_ids`**: some architectures (e.g., RoBERTa, DeBERTa) do not include `token_type_ids` as an ONNX input because they use position-based segment encoding instead of explicit segment embeddings, but the HF tokenizer always produces and passes it
- Checks `self.io_config["input_names"]` and only forwards `token_type_ids` to the ONNX session if the model actually accepts it, preventing an ORT input mismatch error
