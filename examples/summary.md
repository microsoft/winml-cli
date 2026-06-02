# Example Configs Test Summary

## Overview

Count basis is canonical `(model, task)` pairs from `scripts/e2e_eval/testsets/models_57.txt`.

| EP | (Model, Task) | Configs | Eval Pass | Eval Fail | Eval Timeout | Report |
|----|---------------|---------|-----------|-----------|--------------|--------|
| AMD (VitisAI, NPU) - fp16 | 57 | 57 | 41/57 (72%) | 14/57 (25%) | 2/57 (4%) | [Report](vitisai/npu/REPORT.md) |
| AMD (VitisAI, NPU) - w8a16 | 57 | 57 | 33/57 (58%) | 14/57 (25%) | 2/57 (4%) | [Report](vitisai/npu/REPORT.md) |
| AMD (VitisAI, NPU) - w8a8 | 57 | 57 | 33/57 (58%) | 15/57 (26%) | 1/57 (2%) | [Report](vitisai/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - fp16 | 57 | 57 | 40/57 (70%) | 15/57 (26%) | 2/57 (4%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - w8a16 | 28 | 28 | 27/28 (96%) | 1/28 (4%) | 0/28 (0%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - w8a8 | 28 | 28 | 27/28 (96%) | 1/28 (4%) | 0/28 (0%) | [Report](qnn/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - fp16 | 56 | 56 | 25/56 (45%) | 2/56 (4%) | 0/56 (0%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - w8a16 | 27 | 27 | 25/27 (93%) | 2/27 (7%) | 0/27 (0%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - w8a8 | 27 | 27 | 25/27 (93%) | 2/27 (7%) | 0/27 (0%) | [Report](openvino/npu/REPORT.md) |
| QNN (Qualcomm, GPU) | 57 | 57 | 38/57 (67%) | 19/57 (33%) | 0/57 (0%) | [Report](qnn/gpu/REPORT.md) |
| OpenVINO (Intel, CPU) | 57 | 57 | 33/57 (58%) | 11/57 (19%) | 3/57 (5%) | [Report](openvino/cpu/REPORT.md) |
| OpenVINO (Intel, GPU) | 57 | 57 | 38/57 (67%) | 11/57 (19%) | 0/57 (0%) | [Report](openvino/gpu/REPORT.md) |
| DML (GPU) | 57 | 57 | 45/57 (79%) | 8/57 (14%) | 4/57 (7%) | [Report](dml/gpu/REPORT.md) |
| MLAS (CPU) | 57 | 57 | 44/57 (77%) | 12/57 (21%) | 1/57 (2%) | [Report](mlas/cpu/REPORT.md) |
| NVIDIA TensorRT RTX (GPU) | 57 | 57 | 24/57 (42%) | 21/57 (37%) | 0/57 (0%) | [Report](nv_tensorrt_rtx/gpu/REPORT.md) |
