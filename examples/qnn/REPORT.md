# QNN (Qualcomm) Test Report

## Hardware Overview

| Hardware | Models | Configs | Perf Pass | Eval Pass | Report |
|---|---:|---:|---:|---:|---|
| NPU | 56 | 192 | 183/192 (95%) | 161/192 (84%) | [NPU Report](npu/REPORT.md) |
| GPU | 56 | 192 | 162/192 (84%) | 105/192 (55%) | [GPU Report](gpu/REPORT.md) |

## Notes

- NPU and GPU reports are generated from current artifacts in each hardware folder.
- Detailed per-model/per-task rows remain in hardware-specific reports.
