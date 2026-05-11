# Example Configs Test Summary

## Overview

| EP | Models | Perf Pass | Eval Pass | Report |
|----|--------|-----------|-----------|--------|
| AMD (VitisAI, NPU) | 48 | 47/48 (97%) | 44/48 (91%) | [Report](vitisai/npu/REPORT.md) |
| QNN (Qualcomm, NPU) | 56 | 183/192 (95%) | 161/192 (84%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, GPU) | 56 | 162/192 (84%) | 105/192 (55%) | [Report](qnn/gpu/REPORT.md) |
| OpenVINO (Intel, NPU) | 48 | 48/48 (100%) | 45/48 (93%) | [Report](openvino/npu/REPORT.md) |
| MLAS (CPU) | 46 | 46/46 (100%) | 36/46 (78%) | [Report](mlas/cpu/REPORT.md) |
| NVIDIA TensorRT RTX (GPU) | 64 | 192/192 (100%) | 177/192 (92%) | [Report](nv_tensorrt_rtx/gpu/REPORT.md) |
