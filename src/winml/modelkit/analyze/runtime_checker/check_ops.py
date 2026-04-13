# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test ONNX operators on QNN execution provider.

This script tests ONNX operators on the QNN execution provider,
generating test results for each specified operator.

Usage:
    Test specific operators:
        python -m winml.modelkit.analyze.runtime_checker.check_ops --ops Abs Relu Sigmoid

    Test all registered operators:
        python -m winml.modelkit.analyze.runtime_checker.check_ops --all_ops
"""

from pathlib import Path
from typing import Any

import onnxruntime as ort
from onnx.defs import SchemaError

from ... import winml
from ...onnx import ONNXDomain
from ...pattern.op_input_gen import (
    OpInputGenerator,
    get_registered_operators,
    get_runtime_checker_op,
)
from ...pattern.op_input_gen.qdq_gen import QDQGenerator
from ...sysinfo import SysInfo
from ...utils import constants
from ..utils import CheckResultWriter
from ..utils.model_utils import get_op_since_version
from .ep_checker import EPChecker


# Register WinML EPs at module level before any ORT session is created.
# This must stay at the top of the file so EPs are available for all downstream usage.
winml.register_execution_providers(ort=True)


def check_ops(
    ep_checker: EPChecker,
    ops: list[str],
    opset_version: int,
    opset_domain: str,
    version_until: int | None = None,
    validate_inputs: bool = False,
    output_dir: str | Path = ".",
    model_output_dir: str | Path | None = None,
    n_cases: int | None = None,
    save_failed_model: bool = False,
    save_model: bool = False,
    rerun_failed: bool = False,
    delta_only: bool = False,
    use_qdq: bool = False,
    dry_run: bool = False,
    dynamic_axis_mode: str = "none",
    not_run_start_id: int = 1,
    case_index: str | list[str] | None = None,
):
    """Run operators on execution provider.

    Args:
        ops: List of operator names to test (e.g., ["Abs", "Relu", "Sigmoid"])
        opset_version: ONNX opset version to use
        opset_domain: ONNX opset domain (e.g., "ai.onnx", "com.microsoft")
        validate_inputs: Whether to validate input combinations before testing
        output_dir: Output directory for test results JSON files (default: current directory)
        model_output_dir: Directory to save generated ONNX models.
            Defaults to output_dir/saved_models.
        n_cases: If not None, only run the first n_cases test cases for each operator.
                 If n_cases is greater than total cases, run all cases.
        save_failed_model: If True, save the ONNX model when a test case fails.
        rerun_failed: If True, rerun failed cases (compile or run failed).
        delta_only: If True, only run new test cases not in existing results.
        dry_run: If True, skip compile/run execution and emit check_result with reason "not_run".
        dynamic_axis_mode: Dynamic axis testing mode for input generators.
        not_run_start_id: Initial id used for not_run placeholder reasons (not_run_<id>).
        case_index: Optional hashed signature(s) to filter to specific test cases.
    """
    sys_info = SysInfo().to_dict()
    domain = ONNXDomain.from_str(opset_domain)

    qdq_gen = QDQGenerator(1, ONNXDomain.COM_MICROSOFT) if use_qdq else None

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Directory to stash saved ONNX models (separate from json
    # outputs); creation is lazy in generators.
    model_output_dir = (
        Path(model_output_dir) if model_output_dir is not None else output_dir / "saved_models"
    )

    # Test each operator
    for op_name in ops:
        # Get the generator class for this operator from registry
        generator_class = get_runtime_checker_op(op_name)

        current_opset_version = opset_version
        while True:
            opset_name = f"opset{current_opset_version}"
            print(f"\n{'=' * 80}")
            print(f"Testing {op_name} operator with {opset_domain}:{opset_name}")
            print(f"{'=' * 80}\n")

            # Get the schema for this operator
            try:
                schema = domain.get_op_schema(op_name, current_opset_version)
            except SchemaError as e:
                # Schema not found - print error and skip
                print(f"Skipping {op_name}: {e}")
                break

            gen: OpInputGenerator = generator_class(
                schema,
                qdq_generator=qdq_gen,
                dynamic_axis_mode=dynamic_axis_mode,
                # onnx_types_to_check=["FLOAT"]  # use this to limit data types for debugging
            )

            # Validate inputs if requested
            if validate_inputs:
                print(f"Validating input combinations for {op_name}...")
                gen.validate_inputs()
                print(f"Validation passed for {op_name}\n")

            # Prepare output file
            since_version = get_op_since_version(op_name, current_opset_version, opset_domain)
            device = constants.DEVICE_TYPE_TO_DEVICE[ep_checker.device_type]
            qdq_suffix = "_qdq" if use_qdq else ""
            output_filename = (
                f"{op_name}_{ep_checker.ep_name}_{device}"
                f"_{opset_domain}_opset{since_version}"
                f"{qdq_suffix}.json"
            )
            output_path = output_dir / output_filename

            # Use writer as context manager (auto-flushes on exit)
            with CheckResultWriter(
                output_path,
                sys_info,
                save_per_cases=None if dry_run else 100,
                rerun_failed=rerun_failed,
                delta_only=delta_only,
                not_run_start_id=not_run_start_id,
                filter_case_index=case_index,
            ) as writer:
                # Run tests on execution provider
                print(f"Running {op_name} tests on {ep_checker.ep_name}...")
                if n_cases is not None:
                    print(f"Limiting to first {n_cases} test cases")

                check_results_iter = gen.check_on_ep(
                    ep_checker,
                    capture_output=True,
                    n_cases=n_cases,
                    skip_cases=0,
                    save_failed_model=save_failed_model,
                    save_model=save_model,
                    model_output_dir=model_output_dir,
                    skip_signature_fn=writer.should_skip_case,
                    # Also yield skipped cases (with skip
                    # marker) to maintain order
                    yield_skipped=True,
                    dry_run=dry_run,
                )

                # Process results in generator order - reuse existing or run new
                run_count = 0
                reused_count = 0
                skipped_count = 0
                for result in check_results_iter:
                    if result.get("_skipped"):
                        skipped_count += 1
                        if writer.reuse_existing_result(result):
                            reused_count += 1
                    else:
                        writer.append_result(result)
                        run_count += 1

                dropped_count = writer.get_dropped_count()
                duplicate_skipped = writer.get_duplicate_skipped_count()
                print(
                    f"Ran {run_count} test cases, reused "
                    f"{reused_count} existing cases, "
                    f"dropped {dropped_count} obsolete "
                    f"cases, duplicates skipped "
                    f"{duplicate_skipped}, skipped "
                    f"{skipped_count}."
                )

                # Finalize to clear unused signatures before final flush
                writer.finalize()

                check_results = writer.results

            print(f"\nResults saved to: {output_path}")
            print(f"Total test cases: {len(check_results)}")

            if version_until is not None:
                if since_version <= 1:
                    break
                if since_version <= version_until:
                    print(
                        f"opset_version {since_version} is "
                        f"already <= version_until "
                        f"{version_until}, stopping further"
                        f" testing for {op_name}."
                    )
                    break
                current_opset_version = since_version - 1
            else:
                break


# don't use EPChecker directly as there is a bug with pytest in subprocess
class OpenVINONPUChecker(EPChecker):
    """OpenVINO NPU execution provider checker wrapper for pytest compatibility."""

    def __init__(self, device_type: ort.OrtHardwareDeviceType) -> None:
        """Initialize OpenVINO NPU checker."""
        super().__init__(ep_name="OpenVINOExecutionProvider", device_type=device_type)


# don't use EPChecker directly as there is a bug with pytest in subprocess
class QNNNPUChecker(EPChecker):
    """QNN NPU execution provider checker wrapper for pytest compatibility."""

    def __init__(self, device_type: ort.OrtHardwareDeviceType) -> None:
        """Initialize QNN NPU checker."""
        super().__init__(ep_name="QNNExecutionProvider", device_type=device_type)


class VitisAIChecker(EPChecker):
    """VitisAI execution provider checker wrapper for pytest compatibility."""

    def __init__(self, device_type: ort.OrtHardwareDeviceType) -> None:
        if device_type != ort.OrtHardwareDeviceType.NPU:
            raise ValueError("VitisAIExecutionProvider only supports NPU device type")
        """Initialize VitisAI checker."""
        super().__init__(ep_name="VitisAIExecutionProvider", device_type=device_type)


class MIGraphXChecker(EPChecker):
    """MIGraphX execution provider checker wrapper for pytest compatibility."""

    def __init__(self, device_type: ort.OrtHardwareDeviceType) -> None:
        if device_type != ort.OrtHardwareDeviceType.GPU:
            raise ValueError("MIGraphXExecutionProvider only supports GPU device type")
        """Initialize MIGraphX checker."""
        super().__init__(ep_name="MIGraphXExecutionProvider", device_type=device_type)


class RTXChecker(EPChecker):
    """NVIDIA TensorRT RTX execution provider checker wrapper for pytest compatibility."""

    def __init__(self, device_type: ort.OrtHardwareDeviceType) -> None:
        if device_type != ort.OrtHardwareDeviceType.GPU:
            raise ValueError("NvTensorRTRTXExecutionProvider only supports GPU device type")
        """Initialize RTX checker."""
        super().__init__(
            ep_name="NvTensorRTRTXExecutionProvider", device_type=ort.OrtHardwareDeviceType.GPU
        )


def get_ep_checker(ep_name: str, device: str) -> EPChecker:
    """Get EPChecker for given execution provider name.

    Args:
        ep_name: Execution provider name (e.g., "QNNExecutionProvider")

    Returns:
        EPChecker corresponding to the execution provider.

    Raises:
        ValueError: If the execution provider name is not supported.
    """
    device_type = constants.DEVICE_TO_DEVICE_TYPE[device]
    ep_name_to_checker = {
        "QNNExecutionProvider": QNNNPUChecker,
        "OpenVINOExecutionProvider": OpenVINONPUChecker,
        "VitisAIExecutionProvider": VitisAIChecker,
        "MIGraphXExecutionProvider": MIGraphXChecker,
        "NvTensorRTRTXExecutionProvider": RTXChecker,
        # Add other EPChecker subclasses here as needed
    }
    if ep_name not in ep_name_to_checker:
        raise ValueError(
            f"Unsupported execution provider: {ep_name}. "
            f"Available: {', '.join(ep_name_to_checker.keys())}"
        )
    return ep_name_to_checker[ep_name](device_type=device_type)


def build_parser():
    """Build argument parser for check_ops-style commands."""
    import argparse

    parser = argparse.ArgumentParser(description="Test ONNX operators on execution provider")

    # Get available operators from registry
    available_ops = get_registered_operators()

    # Create mutually exclusive group for --ops and --all_ops
    ops_group = parser.add_mutually_exclusive_group(required=True)
    ops_group.add_argument(
        "--ops",
        type=str,
        nargs="+",
        choices=available_ops,
        help=(
            f"Operator names to test (e.g., Abs Relu Sigmoid). "
            f"Available: {', '.join(available_ops)}"
        ),
    )
    ops_group.add_argument(
        "--all_ops",
        action="store_true",
        help="Test all registered operators",
    )
    parser.add_argument(
        "--ep",
        type=str,
        required=True,
        choices=[
            "QNNExecutionProvider",
            "OpenVINOExecutionProvider",
            "VitisAIExecutionProvider",
            "MIGraphXExecutionProvider",
            "NvTensorRTRTXExecutionProvider",
        ],
        help=(
            "Execution Provider names to test. "
            "Available: QNNExecutionProvider, "
            "OpenVINOExecutionProvider, "
            "VitisAIExecutionProvider, "
            "MIGraphXExecutionProvider, "
            "NvTensorRTRTXExecutionProvider"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="NPU",
        choices=["CPU", "GPU", "NPU"],
        help=("Target device type (CPU, GPU, NPU). "),
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        required=True,
        help="ONNX opset version to use",
    )
    parser.add_argument(
        "--opset_domain",
        type=str,
        required=True,
        help="ONNX opset domain (e.g., 'ai.onnx', 'com.microsoft')",
    )
    parser.add_argument(
        "--validate_inputs",
        action="store_true",
        help="Validate input combinations before testing",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for test results JSON files (default: current directory)",
    )
    parser.add_argument(
        "--n_cases",
        type=int,
        default=None,
        help="Limit number of test cases per operator (default: run all cases)",
    )
    parser.add_argument(
        "--save_failed_model",
        action="store_true",
        help="Save the model for compile failed test cases",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the model for all test cases",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--rerun_failed",
        action="store_true",
        help=(
            "Rerun only failed cases (compile failed or run failed). "
            "Mutually exclusive with --delta_only and --case_index."
        ),
    )
    mode_group.add_argument(
        "--delta_only",
        action="store_true",
        help=(
            "Only run new test cases that do not exist in the existing results file. "
            "Mutually exclusive with --rerun_failed and --case_index."
        ),
    )
    mode_group.add_argument(
        "--case_index",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Only process cases matching these case_index hashes. "
            "Mutually exclusive with --rerun_failed and --delta_only."
        ),
    )
    parser.add_argument(
        "--use_qdq",
        action="store_true",
        help="Use QDQ input generation for testing",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run without executing",
    )
    parser.add_argument(
        "--not_run_start_id",
        type=int,
        default=1,
        help="Initial id for dry-run reason placeholder sequence (not_run_<id>).",
    )
    parser.add_argument(
        "--with_dynamic",
        action="store_true",
        help="Also test with first axis as dynamic (axis 0) for non-constant, non-scalar inputs",
    )
    parser.add_argument(
        "--version_until",
        type=int,
        default=None,
        # For example, version_until=13, opset_version=20, opset has 11,12,15 will test 12 and 15
        help=(
            "Test each distinct operator schema version "
            "down to the first one <= this, up to the "
            "specified opset_version."
        ),
    )
    return parser


def run_from_args(args: Any) -> None:
    """Run check_ops from parsed CLI args."""
    available_ops = get_registered_operators()
    ops_to_check = available_ops if args.all_ops else args.ops
    ep_checker = get_ep_checker(args.ep, device=args.device)
    check_ops(
        ep_checker,
        ops=ops_to_check,
        opset_version=args.opset_version,
        opset_domain=args.opset_domain,
        version_until=args.version_until,
        validate_inputs=args.validate_inputs,
        output_dir=args.output_dir,
        n_cases=args.n_cases,
        save_failed_model=args.save_failed_model,
        save_model=args.save_model,
        rerun_failed=args.rerun_failed,
        delta_only=args.delta_only,
        use_qdq=args.use_qdq,
        dry_run=args.dry_run,
        dynamic_axis_mode="first_axis_dynamic" if args.with_dynamic else "none",
        not_run_start_id=args.not_run_start_id,
        case_index=args.case_index,
    )


def parse_and_check() -> None:
    """Main entry point for command-line execution."""
    parser = build_parser()
    args = parser.parse_args()
    run_from_args(args)


if __name__ == "__main__":
    parse_and_check()
