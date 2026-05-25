# Example Configs Test Summary

## Overview

| EP | Models | Configs | Perf Pass | Eval Pass | Report |
|----|--------|---------|-----------|-----------|--------|
| AMD (VitisAI, NPU) - fp16 | 48 | 56 | 0/56 (0%) | 26/56 (46%) | [Report](vitisai/npu/REPORT.md) |
| AMD (VitisAI, NPU) - w8a16 | 48 | 56 | 30/56 (54%) | 26/56 (46%) | [Report](vitisai/npu/REPORT.md) |
| AMD (VitisAI, NPU) - w8a8 | 48 | 56 | 0/56 (0%) | 26/56 (46%) | [Report](vitisai/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - fp16 | 55 | 63 | 62/63 (98%) | 47/63 (75%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - w8a16 | 55 | 63 | 63/63 (100%) | 48/63 (76%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - w8a8 | 55 | 63 | 63/63 (100%) | 48/63 (76%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, GPU) | - | - | - | - | - |
| OpenVINO (Intel, NPU) | - | - | - | - | - |
| OpenVINO (Intel, CPU) | - | - | - | - | - |
| OpenVINO (Intel, GPU) | - | - | - | - | - |
| DML (GPU) | 55 | 63 | 54/63 (86%) | 38/63 (60%) | [Report](dml/gpu/REPORT.md) |
| MLAS (CPU) | 55 | 63 | 61/63 (97%) | 46/63 (73%) | [Report](mlas/cpu/REPORT.md) |
| NVIDIA TensorRT RTX (GPU) | 55 | 63 | 51/63 (81%) | 57/63 (90%) | [Report](nv_tensorrt_rtx/gpu/REPORT.md) |