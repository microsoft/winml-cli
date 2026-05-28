# Example Configs Test Summary

## Overview

| EP | Models | Configs | Perf Pass | Eval Pass | Report |
|----|--------|---------|-----------|-----------|--------|
| AMD (VitisAI, NPU) - fp16 | 78 | 92 | 71/92 (77%) | 47/92 (51%) | [Report](vitisai/npu/REPORT.md) |
| AMD (VitisAI, NPU) - w8a16 | 78 | 92 | 63/92 (68%) | 39/92 (42%) | [Report](vitisai/npu/REPORT.md) |
| AMD (VitisAI, NPU) - w8a8 | 78 | 92 | 60/92 (65%) | 42/92 (46%) | [Report](vitisai/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - fp16 | 78 | 92 | 90/92 (98%) | 65/92 (71%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - w8a16 | 55 | 63 | 63/63 (100%) | 53/63 (84%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - w8a8 | 55 | 63 | 63/63 (100%) | 53/63 (84%) | [Report](qnn/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - fp16 | 71 | 85 | 84/85 (99%) | 49/85 (58%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - w8a16 | 48 | 56 | 55/56 (98%) | 51/56 (91%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - w8a8 | 48 | 56 | 55/56 (98%) | 51/56 (91%) | [Report](openvino/npu/REPORT.md) |
| QNN (Qualcomm, GPU) | 78 | 92 | 57/92 (62%) | 54/92 (59%) | [Report](qnn/gpu/REPORT.md) |
| OpenVINO (Intel, CPU) | 78 | 92 | 91/92 (99%) | 49/92 (53%) | [Report](openvino/cpu/REPORT.md) |
| OpenVINO (Intel, GPU) | 78 | 92 | 87/92 (95%) | 56/92 (61%) | [Report](openvino/gpu/REPORT.md) |
| DML (GPU) | 78 | 92 | 86/92 (93%) | 72/92 (78%) | [Report](dml/gpu/REPORT.md) |
| MLAS (CPU) | 78 | 92 | 89/92 (97%) | 73/92 (79%) | [Report](mlas/cpu/REPORT.md) |
| NVIDIA TensorRT RTX (GPU) | 78 | 92 | 80/92 (87%) | 57/92 (62%) | [Report](nv_tensorrt_rtx/gpu/REPORT.md) |
