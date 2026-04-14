# Eval Module - Product Requirements Document

**Version**: 2.0  
**Date**: 2026-03-04  
**Status**: Implemented  

---

## 1. Executive Summary

### 1.1 Purpose

The eval module (`wmk eval`) measures model inference quality metrics on real
datasets. After a model goes through export, optimize, quantize, and compile,
eval verifies that model quality has not degraded significantly.

### 1.2 Problem Statement

When deploying models through the ModelKit pipeline (export → optimize →
quantize → compile), there is no built-in way to verify that the output model
produces correct results. A model may successfully run inference but return
incorrect predictions due to quantization errors, optimization bugs, or
compilation issues.

### 1.3 Core Value Proposition

| Value | Description |
|-------|-------------|
| **Quality Verification** | Measure accuracy/metrics after pipeline transformations |
| **Regression Detection** | Compare metrics before and after optimization/quantization |
| **Deployment Confidence** | Confirm the deployed model meets quality thresholds |

---

## 2. Requirements

### 2.1 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Evaluate a model on a HuggingFace dataset with task-specific metrics | P0 |
| FR-2 | Support HF model ID as input (build + evaluate in one step) | P0 |
| FR-3 | Support ONNX model file as input with HF model ID for config | P0 |
| FR-4 | Auto-detect task from model config | P0 |
| FR-5 | Auto-resolve preprocessors from model ID | P0 |
| FR-6 | Support configurable dataset, split, and sample count | P0 |
| FR-7 | Output results as JSON for automation | P0 |
| FR-8 | Support ONNX model without HF model ID (local config files) | P1 |
| FR-9 | Explicit metric override (`--metric`) | P1 |
| FR-10 | Support object-detection and image-segmentation evaluation | P2 |

### 2.2 Supported Tasks

| Task | Default Metric | Priority |
|------|----------------|----------|
| `image-classification` | accuracy | P0 |
| `object-detection` | mAP | P0 |
| `image-segmentation` | mIoU | P0 |
| `text-classification` | accuracy | P0 |
| `token-classification` | seqeval | P0 |

### 2.3 CLI Interface

```
wmk eval --model-id microsoft/resnet-50 --dataset ILSVRC/imagenet-1k
wmk eval -m model.onnx --model-id microsoft/resnet-50 --dataset ILSVRC/imagenet-1k
```

### 2.4 Python API

```python
from modelkit.eval import WinMLEvaluationConfig, evaluate

result = evaluate(WinMLEvaluationConfig(
    model_id="microsoft/resnet-50",
    dataset="ILSVRC/imagenet-1k",
    samples=100,
))
```

---

## 3. Success Criteria

| Metric | Target |
|--------|--------|
| Eval completes for image-classification models | Verified with resnet-50 |
| Task-generic implementation (no task-specific code) | Achieved |
| Consistent results with HF reference implementation | Verified (70% on 10 ImageNet samples matches HF) |

---