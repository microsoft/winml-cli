# Example Configs Test Summary

## Overview

| EP | Models | Configs | Perf Pass | Eval Pass | Report |
|----|--------|---------|-----------|-----------|--------|
| AMD (VitisAI, NPU) | 56 | 192 | 140/192 (73%) | 151/192 (79%) | [Report](vitisai/npu/REPORT.md) |
| QNN (Qualcomm, NPU) | 63 | 213 | 202/213 (95%) | 168/213 (79%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, GPU) | 63 | 213 | 162/213 (76%) | 105/213 (49%) | [Report](qnn/gpu/REPORT.md) |
| OpenVINO (Intel, NPU) | 56 | 192 | 144/192 (75%) | 156/192 (81%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, CPU) | 56 | 192 | 192/192 (100%) | 133/192 (69%) | [Report](openvino/cpu/REPORT.md) |
| MLAS (CPU) | 46 | 152 | 152/152 (100%) | 122/152 (80%) | [Report](mlas/cpu/REPORT.md) |
| NVIDIA TensorRT RTX (GPU) | 55 | 63 | 62/63 (98%) | 58/63 (92%) | [Report](nv_tensorrt_rtx/gpu/REPORT.md) |

All pass rates above are config-based (`*_config.json` as denominator).
