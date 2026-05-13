# Example Configs Test Summary

## Overview

| EP | Models | Perf Pass | Eval Pass | Report |
|----|--------|-----------|-----------|--------|
| AMD (VitisAI, NPU) | 64 | 56/64 (87%) | 49/64 (76%) | [Report](vitisai/npu/REPORT.md) |
| QNN (Qualcomm, NPU) | 56 | 183/192 (95%) | 161/192 (84%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, GPU) | 56 | 162/192 (84%) | 105/192 (55%) | [Report](qnn/gpu/REPORT.md) |
| OpenVINO (Intel, NPU) | 64 | 64/64 (100%) | 52/64 (81%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, CPU) | 64 | 192/192 (100%) | 133/192 (69%) | [Report](openvino/cpu/REPORT.md) |
| MLAS (CPU) | 46 | 46/46 (100%) | 36/46 (78%) | [Report](mlas/cpu/REPORT.md) |
| NVIDIA TensorRT RTX (GPU) | 64 | 192/192 (100%) | 177/192 (92%) | [Report](nv_tensorrt_rtx/gpu/REPORT.md) |
