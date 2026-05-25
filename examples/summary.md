# Example Configs Test Summary

## Overview

| EP | Models | Configs | Perf Pass | Eval Pass | Report |
|----|--------|---------|-----------|-----------|--------|
| AMD (VitisAI, NPU) | 48 | 168 | 30/168 (18%) | 78/168 (46%) | [Report](vitisai/npu/REPORT.md) |
| QNN (Qualcomm, NPU) | 55 | 189 | 188/189 (99%) | 143/189 (76%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, GPU) | - | - | - | - | - |
| OpenVINO (Intel, NPU) | - | - | - | - | - |
| OpenVINO (Intel, CPU) | - | - | - | - | - |
| OpenVINO (Intel, GPU) | - | - | - | - | - |
| DML (GPU) | 55 | 63 | 54/63 (86%) | 38/63 (60%) | [Report](dml/gpu/REPORT.md) |
| MLAS (CPU) | 55 | 63 | 61/63 (97%) | 46/63 (73%) | [Report](mlas/cpu/REPORT.md) |
| NVIDIA TensorRT RTX (GPU) | 55 | 63 | 51/63 (81%) | 57/63 (90%) | [Report](nv_tensorrt_rtx/gpu/REPORT.md) |