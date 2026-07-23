# WinML CLI Naming Convention

This document defines the naming rules for the WinML CLI codebase. All new code and refactored code must follow these conventions.

## 1. Acronyms in Class Names

Domain acronyms in PascalCase class names **retain their uppercase form**, except for two-letter abbreviations used as generic prefixes.

### Canonical Acronym Table

| Acronym | Meaning | Class Casing | Example |
|---------|---------|--------------|---------|
| ONNX | Open Neural Network Exchange | `ONNX` | `ONNXStaticAnalyzer`, `ONNXLoader` |
| EP | Execution Provider | `EP` | `EPChecker`, `EPConfig`, `WinMLEPMonitor` |
| QDQ | Quantize-Dequantize | `QDQ` | `QDQParameterConfig`, `QDQGenerator` |
| QNN | Qualcomm Neural Network | `QNN` | `QNNMonitor` |
| Op | Operator (2-letter prefix) | `Op` | `OpUnsupportedError` |
| IO | Input/Output | `IO` | `IOConfigInfo` |
| HTP | Hexagon Tensor Processor | `HTP` | `HTPConfig`, `HTPExporter`, `HTPMetadataBuilder` |

### Why `Op` Not `OP`

Two-letter acronyms used as **class name prefixes** use PascalCase:

- `OPUnsupported` reads ambiguously as three tokens (O-P-Unsupported)
- `OpUnsupported` reads clearly as two tokens (Op-Unsupported)
- Consistent with conventions like `Id` vs `ID`

All-caps is acceptable in **constants** (e.g., `SUPPORTED_OPS`).

### Canonical Execution Provider Names

Execution providers appear mainly in constants, EP-name strings, and config keys rather than as class prefixes. Each EP has a fixed canonical short name (used in our code) and an ORT full name (the `*ExecutionProvider` symbol).

| Short name | ORT full name | Device | Vendor / Notes |
|------------|---------------|--------|----------------|
| `CPU` | `CPUExecutionProvider` | CPU | Default fallback. |
| `CUDA` | `CUDAExecutionProvider` | GPU | NVIDIA. All caps. |
| `DML` | `DmlExecutionProvider` | GPU | DirectML. Use `DML` in our code; do not write `DirectML` as the EP name. |
| `MIGraphX` | `MIGraphXExecutionProvider` | GPU | AMD. Exact casing (mixed case). |
| `NvTensorRTRTX` | `NvTensorRTRTXExecutionProvider` | GPU | NVIDIA TensorRT-RTX. Exact casing; do not shorten to `TensorRT`. |
| `OpenVINO` | `OpenVINOExecutionProvider` | CPU / GPU / NPU | Intel. Exact casing. Alias: `ov`. |
| `QNN` | `QNNExecutionProvider` | NPU | Qualcomm. All caps. |
| `VitisAI` | `VitisAIExecutionProvider` | NPU | AMD Ryzen AI. Exact casing. Alias: `vitis`. |

### Other Canonical Identifiers

| Token | Meaning | Notes |
|-------|---------|-------|
| `HF_` | HuggingFace (constant/variable prefix) | e.g., `HF_MODEL_CLASS_MAPPING`, `HF_TASK_DEFAULTS`. Not used as a class prefix. |

## 2. Module and Package Names

Follow PEP 8: all lowercase with underscores.

```
correct:   onnx_op.py, ep_checker.py, qdq_fix.py
wrong:     OnnxOp.py, EP_Checker.py
```

## 3. Function and Method Names

Snake_case, lowercase.

```
correct:   normalize_ep_name(), generate_build_config()
wrong:     normalizeEPName(), GenerateBuildConfig()
```

## 4. Constants

UPPER_CASE with underscores.

```
correct:   SUPPORTED_EPS, EP_ALIASES, DEVICE_TO_DEVICE_TYPE
wrong:     supportedEps, ep_aliases
```

## 5. Directory Abbreviation Policy

The codebase uses a mix of abbreviated and full directory names. The established names are frozen — do not rename existing directories for consistency alone. For **new** directories, prefer full names unless the abbreviation is widely recognized in the domain (e.g., `optim`, `eval`, `quant`).

| Established Abbreviation | Full Form |
|---|---|
| `optim` | optimization |
| `quant` | quantization |
| `eval` | evaluation |
| `sysinfo` | system information |
| `optracing` | operator tracing |

## 6. Avoid Name Collisions Across Hierarchy

Do not reuse a parent or sibling package name at a deeper level. When creating new subpackages, verify the name does not already exist elsewhere in the tree.

Known collisions to be aware of:

| Name | Locations | Issue |
|---|---|---|
| `winml` | top-level namespace, `modelkit/winml.py`, `models/winml/` | 3-level collision |
| `core` | `modelkit/core/`, `analyze/core/` | same name, different content |
| `models` | `modelkit/models/`, `analyze/models/` | ML models vs data models |
| `utils` | `modelkit/utils/`, `analyze/utils/` | no shared content |
| `pattern` | `modelkit/pattern/`, `analyze/pattern/` | active vs near-empty |
| `inspect` | `modelkit/inspect/` | shadows Python stdlib |
