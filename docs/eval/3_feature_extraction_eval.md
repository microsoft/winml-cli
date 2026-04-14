# Feature Extraction / Sentence Similarity Evaluator -> Design

**Version**: 1.0
**Date**: 2026-03-31
**Status**: Implemented

---

## 1. Overview

The feature extraction evaluator measures how well an ONNX sentence embedding model preserves semantic similarity after quantization and deployment. Given a pair of sentences, the model encodes each into a fixed-length embedding vector, and the evaluator computes how well the cosine similarity between embeddings correlates with human-judged similarity scores.

The primary metric is **Spearman's rank correlation (cosine_spearman)**, which measures whether the model ranks sentence pairs in the same order as humans -> without requiring the similarity scores to match exactly. This is the standard metric used by MTEB (Massive Text Embedding Benchmark) and sentence-transformers for evaluating embedding models.

Both `feature-extraction` and `sentence-similarity` tasks use this evaluator. The two task names are aliases -> they share the same inference model class (`WinMLModelForFeatureExtraction`), the same HF pipeline (`feature-extraction`), and the same evaluation logic. The distinction exists because HuggingFace Hub tags models under different task labels, but ModelKit treats them identically.

This evaluator extends the existing `wmk eval` framework (see [3_design.md](3_design.md)) with embedding-specific logic for mean pooling, cosine similarity computation, and Spearman correlation via `torchmetrics.regression.SpearmanCorrCoef`.

---

## 2. Schemas

### 2.1 I/O Schema

#### Input -> Column Mapping

Sentence similarity datasets have paired sentences with a continuous similarity score. The evaluator uses `columns_mapping` to locate these fields.

| Key | Default | Description |
|---|---|---|
| `input_column_1` | `"sentence1"` | Column containing the first sentence of each pair |
| `input_column_2` | `"sentence2"` | Column containing the second sentence of each pair |
| `score_column` | `"score"` | Column containing the ground truth similarity score |

**CLI usage**:
```bash
wmk eval -m model.onnx --model-id sentence-transformers/all-MiniLM-L6-v2 \
    --task feature-extraction \
    --dataset mteb/stsbenchmark-sts --split test --samples 1000 \
    --column input_column_1=sentence1 \
    --column input_column_2=sentence2 \
    --column score_column=score
```

#### Output -> Evaluation Result

```json
{
  "model_id": "sentence-transformers/all-MiniLM-L6-v2",
  "model_path": "~/.cache/winml/artifacts/.../feat_..._model.onnx",
  "task": "feature-extraction",
  "device": "npu",
  "dataset": {
    "path": "mteb/stsbenchmark-sts",
    "split": "test",
    "samples": 1000,
    "shuffle": true,
    "seed": 42,
    "columns_mapping": {
      "input_column_1": "sentence1",
      "input_column_2": "sentence2",
      "score_column": "score"
    }
  },
  "metrics": {
    "cosine_spearman": 82.05
  }
}
```

| Metric Key | Range | Description |
|---|---|---|
| `cosine_spearman` | [-100, 100] | **Primary metric** -> Spearman rank correlation x 100 between predicted cosine similarities and ground truth scores |

- **100** -> Perfect agreement: the model ranks all sentence pairs in exactly the same order as human annotators.
- **0** -> No correlation: the model's similarity rankings are unrelated to human judgment (random ordering).
- **-100** -> Perfect disagreement: the model ranks pairs in the exact opposite order of human judgment (pairs humans rate as most similar, the model rates as least similar).

In practice, a well-trained sentence embedding model scores 75-90 on STS-B. A score below 50 typically indicates the model was not trained for semantic similarity (e.g., a classification-only BERT). Negative scores are extremely rare and would indicate a fundamentally broken model.

### 2.2 Dataset Ground Truth Schema

The default dataset is **STS-B** (`mteb/stsbenchmark-sts`), a widely used benchmark for sentence similarity. Each row contains a sentence pair with a human-annotated similarity score:

```python
# One dataset row
{
    "sentence1": "A girl is styling her hair.",
    "sentence2": "A girl is brushing her hair.",
    "score": 2.5,   # human similarity score (0 = unrelated, 5 = equivalent)
}
```

| Field | Type | Description |
|---|---|---|
| `sentence1` | `str` | First sentence of the pair |
| `sentence2` | `str` | Second sentence of the pair |
| `score` | `float` | Human-annotated similarity score, typically in [0, 5] for STS-B |

The score scale varies by dataset (STS-B uses 0-5). Since Spearman correlation operates on ranks, the evaluator is scale-agnostic -> it works with any monotonic score range.

### 2.3 Model Output Schema

The HuggingFace `feature-extraction` pipeline returns a nested list of token-level embeddings:

```python
# Pipeline output for a single sentence
pipe("A girl is styling her hair.")
# -> [[[0.12, -0.34, ...], [0.05, 0.22, ...], ...]]
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#     shape: [1, seq_len, hidden_dim]
#     e.g., [1, 512, 384] for all-MiniLM-L6-v2
```

| Field | Shape | Description |
|---|---|---|
| Output | `[1, seq_len, hidden_dim]` | Token-level embeddings, padded to model's fixed sequence length |

This output is then mean-pooled (section 3.1) into a single sentence embedding of shape `[hidden_dim]`.

---

## 3. Spearman Correlation Metric

### 3.1 From Model Output to Sentence Embedding

A sentence embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`) takes a text string as input and produces **token-level embeddings** -> one vector per token in the sequence:

$$\text{Model}(\text{"A cat sat on the mat"}) \rightarrow \mathbf{H} \in \mathbb{R}^{L \times d}$$

where $L$ is the sequence length (padded to a fixed size, e.g., 512) and $d$ is the hidden dimension (e.g., 384).

To obtain a single **sentence embedding** $\mathbf{s} \in \mathbb{R}^{d}$, we apply **attention-mask-weighted mean pooling**. This averages only over real tokens, excluding padding positions:

$$\mathbf{s} = \frac{\sum_{i=1}^{L} m_i \cdot \mathbf{h}_i}{\sum_{i=1}^{L} m_i}$$

where $m_i \in \{0, 1\}$ is the attention mask (1 for real tokens, 0 for padding) and $\mathbf{h}_i$ is the token embedding at position $i$.

> **Why mask-weighted pooling?** For ONNX models with fixed input shapes (e.g., `[1, 512]`), short sentences are padded to the full sequence length. Without masking, a 6-word sentence padded to 512 tokens would have 98% of the mean dominated by padding embeddings -> producing a near-identical vector for every input. Mask-weighted pooling ensures only meaningful tokens contribute.

### 3.2 Cosine Similarity Between Sentence Pairs

Given two sentences encoded as embeddings $\mathbf{s}_1$ and $\mathbf{s}_2$, their cosine similarity measures the angle between the vectors:

$$\text{cos\_sim}(\mathbf{s}_1, \mathbf{s}_2) = \frac{\mathbf{s}_1 \cdot \mathbf{s}_2}{\|\mathbf{s}_1\| \cdot \|\mathbf{s}_2\|}$$

The result is in $[-1, 1]$: 1 means identical direction (semantically identical), 0 means orthogonal (unrelated), -1 means opposite.

### 3.3 Spearman's Rank Correlation

Spearman's correlation measures whether two variables have the same **rank order**, regardless of their actual magnitudes. This is crucial for evaluating embedding models because:

- We don't care if the model outputs cosine similarity 0.85 vs. 0.90 for a "highly similar" pair
- We care that the model **ranks** "highly similar" pairs above "somewhat similar" pairs above "unrelated" pairs
- A quantized ONNX model may shift all similarity scores slightly, but as long as the ranking is preserved, the model is functionally equivalent

**Worked example:**

Consider 5 sentence pairs with human similarity scores (0-5 scale) and model-predicted cosine similarities:

| Pair | Human Score | Cosine Sim | Human Rank | Model Rank |
|---|---|---|---|---|
| "A dog runs" / "A puppy is running" | 4.8 | 0.91 | 1 | 1 |
| "A cat sleeps" / "The kitten naps" | 4.2 | 0.85 | 2 | 2 |
| "A man eats" / "A dog runs" | 1.0 | 0.32 | 4 | 4 |
| "Sun is shining" / "It is raining" | 0.5 | 0.18 | 5 | 5 |
| "Cars on road" / "Traffic in the city" | 3.5 | 0.74 | 3 | 3 |

In this example, the model ranks all pairs in perfect agreement with human judgment -> Spearman correlation = 1.0. Note that the actual cosine values (0.18-0.91) don't need to match the human scores (0.5-4.8); only the **rank order** matters.

If quantization caused the model to swap the ranking of two pairs (e.g., it ranked "Cars on road" above "A cat sleeps"), the Spearman correlation would drop, indicating quality degradation from quantization.

Formally, Spearman's correlation is computed as Pearson's correlation on the rank-transformed data:

$$\rho = \frac{\text{cov}(R_X, R_Y)}{\sigma_{R_X} \cdot \sigma_{R_Y}}$$

where $R_X$ and $R_Y$ are the rank orderings of the predicted cosine similarities and the ground truth scores respectively.

The metric is reported as **cosine_spearman** in the range $[-100, 100]$, following the MTEB convention (Spearman $\rho$ x 100). A value of 80+ is typical for well-performing sentence embedding models on STS-B.

---

## 4. Design Details

### 4.1 Evaluation Flow

The evaluator follows a four-step flow:

```
1. Load Dataset    ->  Load HF dataset, shuffle, sample N pairs
3. Prepare Pipeline ->  Create HF feature-extraction pipeline, configure padding
4. Encode & Compare ->  For each pair: embed both sentences, compute cosine similarity
5. Compute Metric  ->  Spearman rank correlation between cosine sims and ground truth scores
```

Steps 3 happens per-sample in a single loop. After iterating through all pairs, the collected cosine similarities and ground truth scores are passed to `SpearmanCorrelationMetric.compute()`.

### 4.2 Prepare Pipeline

The evaluator creates a HuggingFace `feature-extraction` pipeline and configures it for fixed-shape ONNX inference:

1. **Pipeline task**: Always uses `"feature-extraction"` -> `"sentence-similarity"` is not a valid HF pipeline task, so both `feature-extraction` and `sentence-similarity` evaluator tasks use the same pipeline.
2. **Tokenizer padding**: Reads the ONNX model's input shape from `io_config` (e.g., `[1, 512]`) and sets `padding="max_length"`, `max_length=512`, `truncation=True` on the pipeline's preprocessing parameters. This ensures every input is padded/truncated to the model's fixed sequence length.

### 4.3 Encode & Compare

For each sentence pair in the dataset:

1. **Encode sentence 1**: `pipe(sentence1)` -> token embeddings `[seq_len, hidden_dim]`
2. **Mean pool**: Apply attention-mask-weighted mean pooling -> sentence embedding `[hidden_dim]`
3. **Encode sentence 2**: Same as above
4. **Cosine similarity**: Compute `cos_sim(emb1, emb2)` -> scalar in [-1, 1]

The cosine similarity and ground truth score are collected for all pairs.

### 4.4 Compute Metric

After processing all pairs, the evaluator computes Spearman's rank correlation:

```python
SpearmanCorrelationMetric().compute(cosine_similarities, ground_truth_scores)
# -> {"cosine_spearman": 82.05}
```

Internally, this uses `torchmetrics.regression.SpearmanCorrCoef`, which computes Pearson's correlation on the rank-transformed inputs. The result is multiplied by 100 to match the MTEB reporting convention.

---

## 5. Design Decisions

### DD-001: Use Spearman Rank Correlation as Primary Metric

**Decision**: Use Spearman's rank correlation (cosine_spearman) as the primary evaluation metric, following the MTEB standard.

**Rationale**: Sentence embedding evaluation cares about **relative ranking**, not absolute similarity values. Spearman correlation captures this: if a quantized model shifts all cosine similarities by a constant offset but preserves the ranking, Spearman stays at 1.0, correctly indicating no quality loss. This is exactly the property we need when comparing FP32 PyTorch baselines against quantized ONNX models on NPU.

**Alternatives considered**: Pearson correlation (sensitive to non-linear distortions in similarity scores), cosine similarity MAE (penalizes constant offsets that don't affect usability).

### DD-002: Attention-Mask-Weighted Mean Pooling

**Decision**: Use attention-mask-weighted mean pooling to convert token embeddings into sentence embeddings, rather than simple mean or CLS token extraction.

**Rationale**: ONNX models have fixed input shapes (e.g., `[1, 512]`). A sentence with 8 real tokens is padded with 504 padding tokens. Without masking:
- Simple mean: 98% of the average comes from padding embeddings -> nearly identical vectors for all inputs
- CLS token: some models (e.g., BERT) use CLS, but sentence-transformers models are trained with mean pooling

Mask-weighted mean pooling is the standard approach used by sentence-transformers and matches the training-time pooling for these models. It produces embeddings consistent with what the model was optimized for.

### DD-003: Use `feature-extraction` Pipeline for Both Tasks

**Decision**: Both `feature-extraction` and `sentence-similarity` tasks use the HuggingFace `feature-extraction` pipeline.

**Rationale**: `"sentence-similarity"` is not a valid HuggingFace pipeline task name. The HF `feature-extraction` pipeline returns raw token embeddings, which is exactly what we need for mean pooling. The `sentence-similarity` task is an alias that routes to the same evaluator and model class (`WinMLModelForFeatureExtraction`), differing only in the HuggingFace model ID resolution at export time.

### DD-004: Default Dataset -> STS-B (mteb/stsbenchmark-sts)

**Decision**: Use STS-B as the default evaluation dataset for feature-extraction and sentence-similarity tasks.

**Rationale**: STS-B (Semantic Textual Similarity Benchmark) is the most widely used benchmark for sentence embedding evaluation. It appears in MTEB, sentence-transformers, and virtually all embedding model papers. Using STS-B as default means:
- Results are directly comparable to published model card numbers
- The dataset is small (1,379 test samples) and evaluates quickly
- It provides human-annotated similarity scores with good inter-annotator agreement

### DD-005: Report Metric in 0-100 Scale

**Decision**: Report `cosine_spearman` as Spearman's $\rho$ x 100 (e.g., 82.05 instead of 0.8205).

**Rationale**: This follows the MTEB convention used in leaderboards, model cards, and papers. Values in the 0-100 range are easier to compare and discuss (e.g., "the model scores 82 on STS-B" vs. "the model scores 0.82").
