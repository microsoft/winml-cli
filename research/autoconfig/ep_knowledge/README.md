# Per-EP Empirical Knowledge Base

Each JSON file stores empirical findings for one EP/device combination.

## ⚠️ CRITICAL EPISTEMICS

These findings are **observational hypotheses, not ground truth**. They were derived
from a small number of experiments on a single model (ConvNext-tiny) on a single device
(Snapdragon X Elite CRD). Every finding carries a `confidence` field and a `falsified_by`
field. Before using a finding to prune a search space, check:

1. **Is the model architecture similar?** (ConvNext ≠ BERT ≠ ResNet)
2. **Is the hardware the same?** (X Elite CRD ≠ X Plus ≠ X1E-80-100)
3. **Is the ORT/QNN SDK version the same?**
4. **Is the mechanism confirmed?** (see `mechanism_confirmed` field)

**Dialectical rule**: A finding that prunes a search dimension must be re-enabled
if a new experiment on a new model/hardware contradicts it. Findings degrade over time
as ORT and QNN SDK versions change.

## Files
- `qnn_npu.json` — QNN HTP (NPU) EP findings
- `qnn_gpu.json` — QNN GPU EP findings
- `dml.json`     — DirectML EP findings
- `cpu.json`     — CPU EP findings
