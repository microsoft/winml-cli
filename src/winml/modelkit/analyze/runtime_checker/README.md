# Runtime checker

## Overview
The runtime checker tests ONNX operators on various execution providers (EPs) to validate their support across different input combinations, data types, and attributes.

## Example Usage
The following command will run `Reshape` op on certain input cases on QNN EP, to test if it's
runnable under different conditions. Compile test will check if it runs on NPU, and run test will
check if it runs without throwing exception (either on NPU or by falling back to CPU).

```powershell
python -m winml.modelkit.static_analyzer.runtime_checker.run_reshape_qnn_example reshape_qnn_results.json --opset 22 --quick --validate_inputs

# Arguments:
#   output_path:        Required (unless --validate_inputs) - Path where test results JSON will be saved
#   --opset:            Optional - ONNX opset version to use (default: 17)
#   --quick:            Optional - Use example generator for quick testing with smaller input set
#   --validate_inputs:  Optional - Validate input combinations, before EP testing
```

The output is like:
```json
{
    "check_results": [
        {
            "type_vars": {
                "T_Reshape": "BOOL"
            },
            "input_constraints": {
                "data": {
                    "type": "shape",
                    "shape": [2, 3, 2, 2]
                },
                "shape": {
                    "type": "value",
                    "value": [2, 3, 2, 1, 2],
                    "dtype": "int64"
                }
            },
            "attrs": {
                "allowzero": 0
            },
            "input_is_constant": {
                "data": true,
                "shape": false
            },
            "check_result": {
                "compile": {
                    "result": {
                        "success": false,
                        "reason": "[ONNXRuntimeError] : 1 : FAIL : graph_partitioner.cc:816 onnxruntime::CreateEpContextModel Unable to compile any nodes. Check that the session EPs support compilation and can execute at least one subgraph in the model."
                    },
                    "stdout": "Adding QNNExecutionProvider for OrtHardwareDeviceType.NPU\n",
                    "stderr": ""
                },
                "run": {
                    "result": {
                        "success": true,
                        "reason": null
                    },
                    "stdout": "Adding QNNExecutionProvider for OrtHardwareDeviceType.NPU\nRun outputs: [array([[[[[False, False]],\n\n         [[False, False]]],\n\n\n        [[[False, False]],\n\n         [[False, False]]],\n\n\n        [[[False, False]],\n\n         [[False, False]]]],\n\n\n\n       [[[[False, False]],\n\n         [[False, False]]],\n\n\n        [[[False, False]],\n\n         [[False, False]]],\n\n\n        [[[False, False]],\n\n         [[False, False]]]]])]\n",
                    "stderr": ""
                }
            }
        },
        ...
    ],
    "sys_info": {<INFO OF TEST ENVIRONMENT>}
}
```

## Developing New OpInputGenerator Subclasses

When creating a new `OpInputGenerator` subclass to test a different ONNX operator, you MUST first gather information:
1. Read the doc of the op, to understand what are the valid inputs of this op like:
```python
# Note that we use fixed opset22 for docs, but the opset used during check may differ.
help(onnxscript.opset22.Reshape) # replace `Reshape` with the op you are working on
```
2. Read @op_input_gen/op_input_gen.py, especially class `OpInputGenerator` to understand how it works and how it should be subclassed; check also `ExampleReshapeInputGenerator` for a brief example.

Next, you MUST implement the following in your subclass:
1. **`op_name`**: Class attribute specifying the operator name
2. **`get_finite_attribute_sets()`**: Return dict of attributes with finite value sets
3. **`get_input_and_infinite_attribute_combinations()`**: Return list of input constraint combinations

Finally, validate the input combinations you wrote as in the next subsection.

### Validating Input Combinations

Before committing a new OpInputGenerator subclass, use the `validate_inputs()` method to ensure all input combinations are valid:

```python
from winml.modelkit.static_analyzer.runtime_checker.op_input_gen import YourOpInputGenerator
from winml.modelkit.static_analyzer.onnx_opset import ONNXDomain

# Get schema for your operator (e.g., opset version 17, default domain)
schema = ONNXDomain.AI_ONNX.get_op_schema("YourOp", 17)
gen = YourOpInputGenerator(schema)

# This will raise ValueError if any input combination is invalid
assert gen.validate_inputs()
```

**What it checks**:
- Each input combination must be valid for at least one combination of attributes and TypeVars
- An input combination is considered valid if it runs successfully for at least one combination of attribute and TypeVar.


**Example output**:
```
Validating input combination: {'data': {'type': 'shape', 'shape': [2, 3]}, ...}
Validating input combination: {'data': {'type': 'shape', 'shape': [4, 5, 6]}, ...}
ValueError: Found invalid input combinations: [{'data': {'type': 'shape', 'shape': [1, 0]}, ...}]
```

This validation step helps catch issues early during development before running expensive EP testing.