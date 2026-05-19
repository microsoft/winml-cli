# Example Configs Test Summary

## Overview

| EP | Models | Configs | Perf Pass | Eval Pass | Report |
|----|--------|---------|-----------|-----------|--------|
| AMD (VitisAI, NPU) | 56 | 192 | 140/192 (73%) | 151/192 (79%) | [Report](vitisai/npu/REPORT.md) |
| QNN (Qualcomm, NPU) | 63 | 213 | 202/213 (95%) | 168/213 (79%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, GPU) | 62 | 63 | 45/63 (71%) | 31/63 (49%) | [Report](qnn/gpu/REPORT.md) |
| OpenVINO (Intel, NPU) | 56 | 192 | 144/192 (75%) | 156/192 (81%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, CPU) | 55 | 63 | 62/63 (98%) | 46/63 (73%) | [Report](openvino/cpu/REPORT.md) |
| OpenVINO (Intel, GPU) | 55 | 63 | 62/63 (98%) | 53/63 (84%) | [Report](openvino/gpu/REPORT.md) |
| DML (GPU) | 63 | 63 | 18/63 (29%) | 13/63 (21%) | [Report](dml/gpu/REPORT.md) |
| MLAS (CPU) | 55 | 63 | 62/63 (98%) | 58/63 (92%) | [Report](mlas/cpu/REPORT.md) |
| NVIDIA TensorRT RTX (GPU) | 55 | 63 | 62/63 (98%) | 58/63 (92%) | [Report](nv_tensorrt_rtx/gpu/REPORT.md) |

All pass rates above are config-based (`*_config.json` as denominator).
