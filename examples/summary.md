# Example Configs Test Summary

## Overview

Count basis is canonical `(model, task)` pairs from `scripts/e2e_eval/testsets/models_57.txt`.

| EP | (Model, Task) | Configs | Eval Pass | Report |
|----|---------------|---------|-----------|--------|
| AMD (VitisAI, NPU) - fp16 | 56 | 56 | 46/56 (82%) | [Report](vitisai/npu/REPORT.md) |
| AMD (VitisAI, NPU) - w8a16 | 56 | 56 | 45/56 (80%) | [Report](vitisai/npu/REPORT.md) |
| AMD (VitisAI, NPU) - w8a8 | 56 | 56 | 44/56 (79%) | [Report](vitisai/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - fp16 | 57 | 57 | 46/57 (81%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - w8a16 | 57 | 57 | 47/57 (82%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - w8a8 | 57 | 57 | 47/57 (82%) | [Report](qnn/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - fp16 | 55 | 55 | 24/55 (44%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - w8a16 | 26 | 26 | 24/26 (92%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - w8a8 | 26 | 26 | 24/26 (92%) | [Report](openvino/npu/REPORT.md) |
| QNN (Qualcomm, GPU) | 56 | 56 | 37/56 (66%) | [Report](qnn/gpu/REPORT.md) |
| OpenVINO (Intel, CPU) | 57 | 57 | 40/57 (70%) | [Report](openvino/cpu/REPORT.md) |
| OpenVINO (Intel, GPU) | 57 | 57 | 44/57 (77%) | [Report](openvino/gpu/REPORT.md) |
| DML (GPU) | 56 | 56 | 45/56 (80%) | [Report](dml/gpu/REPORT.md) |
| MLAS (CPU) | 56 | 56 | 43/56 (77%) | [Report](mlas/cpu/REPORT.md) |
| NVIDIA TensorRT RTX (GPU) | 56 | 56 | 44/56 (79%) | [Report](nv_tensorrt_rtx/gpu/REPORT.md) |
