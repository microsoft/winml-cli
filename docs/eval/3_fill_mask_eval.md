# Fill-Mask Evaluator Design

## 1. Overview

Fill-mask models (BERT, RoBERTa, DistilBERT, etc.) are **Masked Language Models (MLM)**. Given a sentence with one or more tokens replaced by `[MASK]`, the model predicts the original token.

**Example:**

```
Input:  "The cat [MASK] on the mat."
Output: "The cat sat on the mat."   (predicted token: "sat", score: 0.82)
```

## 2. Usage

```bash
uv run winml eval \
    -m ~/.cache/winml/artifacts/google-bert_bert-base-uncased/mask_e7ec673175d3b94d_model.onnx \
    --model-id google-bert/bert-base-uncased \
    --device npu \
    --task fill-mask \
    --dataset Salesforce/wikitext \
    --dataset-name wikitext-2-raw-v1 \
    --split test \
    --samples 1000 \
    --column input_column=text
```

Output:

```
╭───────────────────────────────────────────╮
│ Evaluation: google-bert/bert-base-uncased │
╰───────────────────────────────────────────╯

Task:       fill-mask
Device:     npu
Dataset:    Salesforce/wikitext
Samples:    1000

┏━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric        ┃  Value ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ cross_entropy │ 2.1625 │
└───────────────┴────────┘
```

## 3. Metric

**Cross-entropy** is the standard loss metric for Masked Language Modeling (MLM) evaluation.

For each masked token, cross-entropy measures how well the model predicts the original token:

$$\text{CE} = -\frac{1}{N}\sum_{i=1}^{N} \log P(t_i \mid \text{context})$$

where $N$ is the total number of masked tokens across all samples, and $P(t_i \mid \text{context})$ is the model's predicted probability for the correct token $t_i$.

**Why cross-entropy?**

- **It is the MLM loss function.** Cross-entropy is the exact objective that masked language models are trained to minimize. Evaluating with the same loss gives a direct measure of how well the model performs at its core task — no proxy metric or downstream adaptation needed.

- **More informative than accuracy.** A top-1 accuracy metric only checks whether the correct token is the model's No.1 prediction (right or wrong, binary). Cross-entropy captures the full probability distribution — a model that assigns 40% to the correct token scores much better than one that assigns 5%, even if both get it "wrong" by top-1. This makes CE more sensitive to quality differences, especially when comparing ONNX-quantized models against PyTorch baselines.

- **Linear and comparable.** CE is a continuous value where smaller is better. A delta of 0.1 CE between two models has consistent meaning regardless of the absolute value. This makes relative comparison straightforward — e.g., "the ONNX model's CE is 2.8% higher than the PyTorch baseline" — which is exactly what the e2e evaluation system needs for pass/regression verdicts.

## 4. Implementation

The evaluator processes each text sample in four steps:

1. **Tokenize** — Convert text to token IDs with the model's tokenizer. Pad to the ONNX model's fixed sequence length if required.

2. **Mask** — Use HuggingFace `DataCollatorForLanguageModeling` to randomly replace 15% of tokens following the standard MLM protocol (80% → `[MASK]`, 10% → random token, 10% → unchanged). A fixed seed ensures reproducible masking across runs. The original token IDs at masked positions become the labels.

3. **Infer** — Pass the masked `input_ids` and `attention_mask` directly to the model (no HF pipeline). The model returns logits of shape `[seq_len, vocab_size]`.

4. **Score** — `CrossEntropyMetric` accumulates the cross-entropy loss at masked positions (`labels != -100`) across all samples, then computes the mean CE per token.

```
┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌─────────────────┐
│ Tokenize │───>│ Mask 15%     │───>│ Model     │───>│ CrossEntropy    │
│          │    │ (seed=42)    │    │ inference │    │ Metric          │
│ text →   │    │ input_ids →  │    │ logits    │    │ CE              │
│ tokens   │    │ masked_ids + │    │           │    │                 │
│          │    │ labels       │    │           │    │                 │
└──────────┘    └──────────────┘    └───────────┘    └─────────────────┘
```

## 5. Evaluation results
| Model | ONNX | Baseline | Delta (lower better) | Verdict |
|---|---|---|---|---|
| bert-base-uncased | 2.1625 | 2.1032 | +2.8% | PASS |
| xlm-roberta-base | 1.5680 | 1.5272 | +2.7% | PASS |
| xlm-roberta-large | 1.2023 | 1.1999 | +0.2% | PASS |
| roberta-base | 1.9361 | 1.9061 | +1.6% | PASS |
| roberta-large | 1.9178 | 1.9265 | -0.5% | PASS |
| bert-base-multilingual-uncased | 1.8312 | 1.8057 | +1.4% | PASS |
| bert-base-multilingual-cased | 1.9570 | 1.8814 | +4.0% | PASS |
| **distilbert-base-uncased** | **8.5174** | **2.3718** | **+259.1%** | **REGRESSION** |