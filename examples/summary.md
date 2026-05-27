# Example Configs Test Summary

## Overview

| EP | Models | Configs | Perf Pass | Eval Pass | Report |
|----|--------|---------|-----------|-----------|--------|
| AMD (VitisAI, NPU) - w8a16 | 48 | 56 | 30/56 (54%) | 26/56 (46%) | [Report](vitisai/npu/REPORT.md) |
| AMD (VitisAI, NPU) - w8a8 | 48 | 56 | 33/56 (59%) | 29/56 (52%) | [Report](vitisai/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - fp16 | 77 | 90 | 62/90 (69%) | 62/90 (69%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - w8a16 | 55 | 63 | 63/63 (100%) | 53/63 (84%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - w8a8 | 55 | 63 | 63/63 (100%) | 53/63 (84%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, GPU) | 71 | 84 | 28/84 (33%) | 48/84 (57%) | [Report](qnn/gpu/REPORT.md) |
| OpenVINO (Intel, NPU) - fp16 | 67 | 80 | 55/80 (69%) | 49/80 (61%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - w8a16 | 48 | 56 | 55/56 (98%) | 51/56 (91%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - w8a8 | 48 | 56 | 55/56 (98%) | 51/56 (91%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, CPU) | 69 | 82 | 80/82 (98%) | 49/82 (60%) | [Report](openvino/cpu/REPORT.md) |
| OpenVINO (Intel, GPU) | 71 | 84 | 78/84 (93%) | 56/84 (67%) | [Report](openvino/gpu/REPORT.md) |
| DML (GPU) | 62 | 73 | 57/73 (78%) | 43/73 (59%) | [Report](dml/gpu/REPORT.md) |
| MLAS (CPU) | 65 | 76 | 61/76 (80%) | 54/76 (71%) | [Report](mlas/cpu/REPORT.md) |
| NVIDIA TensorRT RTX (GPU) | 69 | 80 | 51/80 (64%) | 57/80 (71%) | [Report](nv_tensorrt_rtx/gpu/REPORT.md) |