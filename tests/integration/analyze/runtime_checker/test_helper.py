# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from winml.modelkit.analyze.runtime_checker.ep_checker import (
    EPChecker,
)
from winml.modelkit.onnx import ONNXDomain
from winml.modelkit.pattern.op_input_gen import (
    ExampleReshapeInputGenerator,
    get_runtime_checker_op,
)
from winml.modelkit.sysinfo import SysInfo


def compare_objs(obj1: Any, obj2: Any, ignored_keys: Sequence[str] | None = None) -> bool:
    """Compare two objects, ignoring specified keys."""
    if ignored_keys is None:
        ignored_keys = []

    def _remove_ignored(d: Any) -> Any:
        if isinstance(d, dict):
            return {k: _remove_ignored(v) for k, v in d.items() if k not in ignored_keys}
        if isinstance(d, list):
            return [_remove_ignored(item) for item in d]
        return d

    obj1_clean = _remove_ignored(obj1)
    obj2_clean = _remove_ignored(obj2)
    for item1, item2 in zip(obj1_clean["check_results"], obj2_clean["check_results"], strict=True):
        if item1 != item2:
            print("------------Difference found------------:")
            print("obj1 item:", item1)
            print("obj2 item:", item2)

    return bool(obj1_clean == obj2_clean)


def reshape_quick_helper(
    ep_checker: EPChecker,
    truth_file: Path,
) -> None:
    opset_name = "opset22"
    opset_version = 22
    sys_info = SysInfo().to_dict()

    print(f"Testing Reshape op with {opset_name}")
    # Get the OpSchema for Reshape operator at opset version 22
    schema = ONNXDomain.AI_ONNX.get_op_schema("Reshape", opset_version)
    gen = ExampleReshapeInputGenerator(schema)

    test_results_iter = gen.check_on_ep(
        ep_checker,
        capture_output=True,
    )
    result = {"check_results": list(test_results_iter), "sys_info": sys_info}
    with truth_file.open() as f:
        truth_object = json.load(f)
    same = compare_objs(
        truth_object,
        result,
        ignored_keys=["stderr", "stdout", "reason", "sys_info", "min_max"],
    )
    if not same:
        with (truth_file.with_suffix(".actual.json")).open("w") as f:
            json.dump(result, f, indent=2)
    assert same, f"Results differ from truth file: {truth_file}"


def op_quick_helper(
    op_name: str,
    ep_checker: EPChecker,
    truth_file: Path,
    opset: int = 22,
) -> None:
    """Helper to test a single operator against ground truth.

    Args:
        op_name: Name of the operator to test (e.g., "Not", "Abs")
        ep_checker: EPChecker instance to use for testing
        truth_file: Path to ground truth JSON file
        opset: ONNX opset version to use (default: 22)
    """
    opset_name = f"opset{opset}"
    sys_info = SysInfo().to_dict()

    print(f"Testing {op_name} op with {opset_name}")

    # Get the generator class for this operator from registry
    generator_class = get_runtime_checker_op(op_name)
    # Get the OpSchema for the operator at specified opset version
    schema = ONNXDomain.AI_ONNX.get_op_schema(op_name, opset)
    gen = generator_class(schema)

    test_results = gen.check_on_ep(
        ep_checker,
        capture_output=True,
    )
    result = {"check_results": test_results, "sys_info": sys_info}

    with truth_file.open() as f:
        truth_object = json.load(f)

    assert compare_objs(
        truth_object,
        result,
        ignored_keys=["stderr", "stdout", "reason", "sys_info", "min_max"],
    )


def should_run_ep_test(ep_name: str, device_type, skip_message: str | None = None) -> bool:
    """Determine if EP test should run."""
    # Run if hardware is available
    try:
        from winml.modelkit import winml
        winml.register_execution_providers(ort=True)
        import onnxruntime as ort
        
        ep_devices = ort.get_ep_devices()
        for ep_device in ep_devices:
            if ep_device.ep_name == ep_name and ep_device.device.type == device_type:
                return True
    except Exception:
        pass

    if skip_message:
        pytest.skip(skip_message)
    return False
