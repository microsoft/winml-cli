# Example Configs Test Summary

## Overview

Count basis is canonical `(model, task)` pairs from `scripts/e2e_eval/testsets/example_model_tasks.txt`.

| EP | (Model, Task) | Configs | Eval Pass | Report |
|----|---------------|---------|-----------|--------|
| AMD (VitisAI, NPU) - fp16 | 77 | 77 | 48/77 (62%) | [Report](vitisai/npu/REPORT.md) |
| AMD (VitisAI, NPU) - w8a16 | 77 | 77 | 47/77 (61%) | [Report](vitisai/npu/REPORT.md) |
| AMD (VitisAI, NPU) - w8a8 | 77 | 77 | 48/77 (62%) | [Report](vitisai/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - fp16 | 77 | 77 | 48/77 (62%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - w8a16 | 77 | 77 | 49/77 (64%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - w8a8 | 77 | 77 | 49/77 (64%) | [Report](qnn/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - fp16 | 77 | 77 | 44/77 (57%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - w8a16 | 77 | 77 | 45/77 (58%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - w8a8 | 77 | 77 | 43/77 (56%) | [Report](openvino/npu/REPORT.md) |
| QNN (Qualcomm, GPU) | 77 | 77 | 21/77 (27%) | [Report](qnn/gpu/REPORT.md) |
| OpenVINO (Intel, CPU) | 77 | 77 | 27/77 (35%) | [Report](openvino/cpu/REPORT.md) |
| OpenVINO (Intel, GPU) | 77 | 77 | 29/77 (38%) | [Report](openvino/gpu/REPORT.md) |
| DML (GPU) | 77 | 77 | 27/77 (35%) | [Report](dml/gpu/REPORT.md) |
| MLAS (CPU) | 77 | 77 | 30/77 (39%) | [Report](mlas/cpu/REPORT.md) |
| NVIDIA TensorRT RTX (GPU) | 77 | 77 | 31/77 (40%) | [Report](nv_tensorrt_rtx/gpu/REPORT.md) |
