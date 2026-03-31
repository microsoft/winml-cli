# ModelKit Naming Convention

This document defines the naming rules for the ModelKit codebase. All new code and refactored code must follow these conventions.

## 1. Acronyms in Class Names

Domain acronyms in PascalCase class names **retain their uppercase form**, except for two-letter abbreviations used as generic prefixes.

### Canonical Acronym Table

| Acronym | Meaning | Class Casing | Example |
|---------|---------|--------------|---------|
| ONNX | Open Neural Network Exchange | `ONNX` | `ONNXStaticAnalyzer`, `ONNXLoader` |
| EP | Execution Provider | `EP` | `EPChecker`, `EPConfig`, `EPMonitor` |
| QDQ | Quantize-Dequantize | `QDQ` | `QDQParameterConfig`, `QDQGenerator` |
| QNN | Qualcomm Neural Network | `QNN` | `QNNMonitor` |
| Op | Operator (2-letter prefix) | `Op` | `OpUnsupportedError` |
| IO | Input/Output | `IO` | `IOConfigInfo` |
| HF | HuggingFace | `HF` | `HF_MODEL_CLASS_MAPPING` |
| HTP | Hexagon Tensor Processor | `HTP` | (directory/module level) |

### Why `Op` Not `OP`

Two-letter acronyms used as **class name prefixes** use PascalCase:

- `OPUnsupported` reads ambiguously as three tokens (O-P-Unsupported)
- `OpUnsupported` reads clearly as two tokens (Op-Unsupported)
- Consistent with conventions like `Id` vs `ID`

All-caps is acceptable in **constants** (e.g., `SUPPORTED_OPS`).

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

## 7. Current Violations

The following classes violate the acronym naming rules and should be renamed:

| Current | Correct | File |
|---|---|---|
| `OnnxOP` | `ONNXOp` | `src/winml/modelkit/analyze/models/onnx_op.py` |
| `OnnxConfigNotFoundError` | `ONNXConfigNotFoundError` | `src/winml/modelkit/export/io.py` |
| `OnnxModelOutput` | `ONNXModelOutput` | `src/winml/modelkit/export/htp/metadata_builder.py` |
| `EpContextNodeChecker` | `EPContextNodeChecker` | `src/winml/modelkit/analyze/core/node_checkers/ep_context_node_checker.py` |
| `EpPackage` | `EPPackage` | `src/winml/modelkit/sysinfo/software.py` |
| `QdqFixResult` | `QDQFixResult` | `src/winml/modelkit/quant/qdq_fix.py` |
| `OPOptionalInputSupportError` | `OpOptionalInputSupportError` | `src/winml/modelkit/analyze/exceptions.py` |
| `OPLackOfRequiredInformationError` | `OpLackOfRequiredInformationError` | `src/winml/modelkit/analyze/exceptions.py` |
| `OPUnsupportedError` | `OpUnsupportedError` | `src/winml/modelkit/analyze/exceptions.py` |
