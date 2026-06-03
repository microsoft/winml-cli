# Example Configs Test Summary

## Overview

Count basis is canonical `(model, task)` pairs from `scripts/e2e_eval/testsets/models_57.txt`.

| EP | (Model, Task) | Configs | Eval Pass | Eval Fail | Eval Timeout | Report |
|----|---------------|---------|-----------|-----------|--------------|--------|
| AMD (VitisAI, NPU) - fp16 | 56 | 56 | 46/56 (82%) | 10/56 (18%) | 0/56 (0%) | [Report](vitisai/npu/REPORT.md) |
| AMD (VitisAI, NPU) - w8a16 | 56 | 56 | 45/56 (80%) | 10/56 (18%) | 1/56 (2%) | [Report](vitisai/npu/REPORT.md) |
| AMD (VitisAI, NPU) - w8a8 | 56 | 56 | 44/56 (79%) | 12/56 (21%) | 0/56 (0%) | [Report](vitisai/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - fp16 | 57 | 57 | 46/57 (81%) | 10/57 (18%) | 0/57 (0%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - w8a16 | 57 | 57 | 47/57 (82%) | 10/57 (18%) | 0/57 (0%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - w8a8 | 57 | 57 | 47/57 (82%) | 9/57 (16%) | 0/57 (0%) | [Report](qnn/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - fp16 | 55 | 55 | 24/55 (44%) | 2/55 (4%) | 0/55 (0%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - w8a16 | 26 | 26 | 24/26 (92%) | 2/26 (8%) | 0/26 (0%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - w8a8 | 26 | 26 | 24/26 (92%) | 2/26 (8%) | 0/26 (0%) | [Report](openvino/npu/REPORT.md) |
| QNN (Qualcomm, GPU) | 56 | 56 | 37/56 (66%) | 19/56 (34%) | 0/56 (0%) | [Report](qnn/gpu/REPORT.md) |
| OpenVINO (Intel, CPU) | 57 | 57 | 40/57 (70%) | 12/57 (21%) | 4/57 (7%) | [Report](openvino/cpu/REPORT.md) |
| OpenVINO (Intel, GPU) | 57 | 57 | 44/57 (77%) | 12/57 (21%) | 0/57 (0%) | [Report](openvino/gpu/REPORT.md) |
| DML (GPU) | 56 | 56 | 45/56 (80%) | 8/56 (14%) | 3/56 (5%) | [Report](dml/gpu/REPORT.md) |
| MLAS (CPU) | 56 | 56 | 43/56 (77%) | 12/56 (21%) | 1/56 (2%) | [Report](mlas/cpu/REPORT.md) |
| NVIDIA TensorRT RTX (GPU) | 56 | 56 | 44/56 (79%) | 12/56 (21%) | 0/56 (0%) | [Report](nv_tensorrt_rtx/gpu/REPORT.md) |
