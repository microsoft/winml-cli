# Example Configs Test Summary

## Overview

| EP | Models | Configs | Perf Pass | Eval Pass | Report |
|----|--------|---------|-----------|-----------|--------|
| AMD (VitisAI, NPU) - w8a16 | 48 | 56 | 30/56 (54%) | 26/56 (46%) | [Report](vitisai/npu/REPORT.md) |
| AMD (VitisAI, NPU) - w8a8 | 48 | 56 | 33/56 (59%) | 29/56 (52%) | [Report](vitisai/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - fp16 | 55 | 63 | 62/63 (98%) | 51/63 (81%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - w8a16 | 55 | 63 | 63/63 (100%) | 53/63 (84%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - w8a8 | 55 | 63 | 63/63 (100%) | 53/63 (84%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, GPU) | 55 | 63 | 28/63 (44%) | 35/63 (56%) | [Report](qnn/gpu/REPORT.md) |
| OpenVINO (Intel, NPU) - fp16 | 48 | 56 | 55/56 (98%) | 49/56 (88%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - w8a16 | 48 | 56 | 55/56 (98%) | 51/56 (91%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - w8a8 | 48 | 56 | 55/56 (98%) | 51/56 (91%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, CPU) | 69 | 82 | 0/82 (0%) | 49/82 (60%) | [Report](openvino/cpu/REPORT.md) |
| OpenVINO (Intel, GPU) | 71 | 84 | 0/84 (0%) | 56/84 (67%) | [Report](openvino/gpu/REPORT.md) |
| DML (GPU) | 55 | 63 | 57/63 (90%) | 43/63 (68%) | [Report](dml/gpu/REPORT.md) |
| MLAS (CPU) | 55 | 63 | 61/63 (97%) | 47/63 (75%) | [Report](mlas/cpu/REPORT.md) |
| NVIDIA TensorRT RTX (GPU) | 55 | 63 | 51/63 (81%) | 57/63 (90%) | [Report](nv_tensorrt_rtx/gpu/REPORT.md) |