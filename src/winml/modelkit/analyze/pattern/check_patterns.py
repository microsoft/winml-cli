# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test ONNX patterns on execution providers.

This script tests ONNX patterns (subgraph patterns like Gelu, MatMulAdd)
on execution providers, generating test results for each specified pattern.

Usage:
    Test specific patterns:
        python -m modelkit.analyze.pattern.check_patterns --patterns Gelu MatMulAdd

    Test all registered patterns:
        python -m modelkit.analyze.pattern.check_patterns --all_patterns
"""

import json
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
from google.protobuf import json_format

from winml.modelkit.onnx.domains import ONNXDomain
from winml.modelkit.pattern.base import (
    PatternInputGenerator,
    get_pattern_input_generator,
    get_registered_pattern_input_generators,
)

from ... import winml
from ...sysinfo import SysInfo
from ...utils import constants
from ..runtime_checker.ep_checker import EPChecker


winml.register_execution_providers(ort=True)


class CheckResultWriter:
    """Writer for test results that supports continuation from existing files."""

    def __init__(
        self,
        file_path: str | Path,
        sys_info: dict[str, Any],
        save_per_cases: int = 20,
        continue_from_existing: bool = False,
    ) -> None:
        """Initialize the writer.

        Args:
            file_path: Path to the output JSON file
            sys_info: System information dictionary (constant during run)
            save_per_cases: Number of results to accumulate before saving to file
            continue_from_existing: If True, read existing file and continue from there.
                                   If False, start fresh (ignore existing file).
        """
        self.file_path = Path(file_path)
        self.sys_info = sys_info
        self.save_per_cases = save_per_cases
        self.skip_cases = 0
        self.results: list[dict[str, Any]] = []
        self.pending_count = 0

        # Only read existing file if continuing
        if continue_from_existing and self.file_path.exists():
            with self.file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if "check_results" in data:
                    self.results = data["check_results"]
                    self.skip_cases = len(self.results)
                    print(
                        f"Found existing file with "
                        f"{self.skip_cases} test cases. "
                        f"Will continue from there."
                    )

    def get_skip_cases(self) -> int:
        """Get the number of cases to skip when continuing."""
        return self.skip_cases

    def append_result(self, result: dict[str, Any]) -> None:
        """Append a test result and save to file periodically.

        Args:
            result: Test result dictionary
        """
        self.results.append(result)
        self.pending_count += 1

        # Save to file once per save_per_cases results
        if self.pending_count >= self.save_per_cases:
            self._save()
            self.pending_count = 0

    def __enter__(self) -> "CheckResultWriter":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and flush pending results."""
        self.flush()

    def flush(self) -> None:
        """Force save any pending results to file."""
        if self.pending_count > 0:
            self._save()
            self.pending_count = 0

    def _save(self) -> None:
        """Save results to file."""
        output_data = {
            "check_results": self.results,
            "sys_info": self.sys_info,
        }

        def json_default(obj: Any) -> Any:
            if isinstance(obj, onnx.TensorProto):
                return json.loads(json_format.MessageToJson(obj))
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with self.file_path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(output_data, f, indent=2, default=json_default)
        print(f"Saved {len(self.results)} results to {self.file_path}")


def check_patterns(
    ep_checker: EPChecker,
    patterns: list[str],
    validate_inputs: bool = False,
    output_dir: str | Path = ".",
    n_cases: int | None = None,
    continue_from_existing: bool = False,
    save_failed_model: bool = False,
    opset_mapping: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run patterns on execution provider and return results.

    Args:
        ep_checker: EPChecker instance for the execution provider.
        patterns: List of pattern names to test (e.g., ["Gelu", "MatMulAdd"])
        validate_inputs: Whether to validate input combinations before testing
        output_dir: Output directory for test results JSON files (default: current directory)
        n_cases: If not None, only run the first n_cases test cases for each pattern.
                 If n_cases is greater than total cases, run all cases.
        continue_from_existing: If True, continue from existing result file by skipping
                                already completed test cases.
        save_failed_model: If True, save the model for compile failed test cases.
        opset_mapping: Required dict mapping domain strings to opset versions,
                       e.g., {"ai.onnx": 17, "com.microsoft": 1}.
                       Used for ONNX model generation.

    Returns:
        Dictionary mapping pattern names to their test results:
        {
            "Gelu": {"check_results": [...], "sys_info": {...}, "output_path": "..."},
            "MatMulAdd": {"check_results": [...], "sys_info": {...}, "output_path": "..."},
            ...
        }
    """
    sys_info = SysInfo().to_dict()

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store results for all patterns
    all_results: dict[str, dict[str, Any]] = {}

    # Convert opset_mapping to ONNXDomain keys if provided
    domain_versions = {
        ONNXDomain.from_str(domain): version for domain, version in opset_mapping.items()
    }

    # Test each pattern
    for pattern_name in patterns:
        print(f"\n{'=' * 80}")
        print(f"Testing {pattern_name} pattern")
        print(f"{'=' * 80}\n")

        # Get the PatternInputGenerator class from registry and instantiate with domain_versions
        generator_class = get_pattern_input_generator(pattern_name)
        gen: PatternInputGenerator = generator_class(domain_versions=domain_versions)

        print(f"Using domain versions: {domain_versions}")

        # Validate inputs if requested
        if validate_inputs:
            print(f"Validating input combinations for {pattern_name}...")
            gen.validate_inputs()
            print(f"Validation passed for {pattern_name}\n")

        # Build opset suffix for filename using ai.onnx version only
        # (com.microsoft opset is always 1, so we omit it for brevity)
        # If ai.onnx not present, fall back to the first domain in the map
        opset_suffix = ""
        ai_onnx_version = domain_versions.get(ONNXDomain.AI_ONNX)
        if ai_onnx_version is not None:
            # NEVER remove domain name from suffix
            opset_suffix = f"_{ONNXDomain.AI_ONNX.name}_opset{ai_onnx_version}"
        else:
            # Use first domain in the map as fallback
            first_domain, first_version = next(iter(domain_versions.items()))
            opset_suffix = f"_{first_domain.value}_opset{first_version}"

        # Prepare output file
        device = constants.DEVICE_TYPE_TO_DEVICE[ep_checker.device_type]
        output_filename = f"{pattern_name}_{ep_checker.ep_name}_{device}{opset_suffix}.json"
        output_path = output_dir / output_filename

        # Use writer as context manager (auto-flushes on exit)
        with CheckResultWriter(
            output_path,
            sys_info,
            continue_from_existing=continue_from_existing,
        ) as writer:
            skip_cases = writer.get_skip_cases()

            # Run tests on execution provider
            print(f"Running {pattern_name} tests on {ep_checker.ep_name}...")
            if skip_cases > 0:
                print(f"Continuing from existing results, skipping {skip_cases} test cases")
            if n_cases is not None:
                print(f"Limiting to first {n_cases} test cases")

            check_results_iter = gen.check_on_ep(
                ep_checker,
                capture_output=True,
                n_cases=n_cases,
                skip_cases=skip_cases,
                save_failed_model=save_failed_model,
            )

            # Process results using writer
            for result in check_results_iter:
                writer.append_result(result)

            check_results = writer.results

        print(f"\nResults saved to: {output_path}")
        print(f"Total test cases: {len(check_results)}")

        # Store results for return
        all_results[pattern_name] = {
            "check_results": check_results,
            "sys_info": sys_info,
            "output_path": str(output_path),
        }

    return all_results


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


def get_ep_checker(ep_name: str, device: str) -> EPChecker:
    """Get EPChecker for given execution provider name.

    Args:
        ep_name: Execution provider name (e.g., "QNNExecutionProvider")
        device: Target device type (CPU, GPU, NPU)

    Returns:
        EPChecker corresponding to the execution provider.

    Raises:
        ValueError: If the execution provider name is not supported.
    """
    device_type = constants.DEVICE_TO_DEVICE_TYPE[device]
    ep_name_to_checker: dict[str, type[EPChecker]] = {
        "QNNExecutionProvider": QNNNPUChecker,
        "OpenVINOExecutionProvider": OpenVINONPUChecker,
        # Add other EPChecker subclasses here as needed
    }
    if ep_name not in ep_name_to_checker:
        raise ValueError(
            f"Unsupported execution provider: {ep_name}. "
            f"Available: QNNExecutionProvider, "
            f"OpenVINOExecutionProvider"
        )
    return ep_name_to_checker[ep_name](device_type=device_type)


def parse_and_check() -> None:
    """Main entry point for command-line execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Test ONNX patterns on execution provider")

    # Get available patterns from registry
    available_patterns = get_registered_pattern_input_generators()

    # Create mutually exclusive group for --patterns and --all_patterns
    patterns_group = parser.add_mutually_exclusive_group(required=True)
    patterns_group.add_argument(
        "--patterns",
        type=str,
        nargs="+",
        choices=available_patterns,
        help=(
            f"Pattern names to test (e.g., Gelu MatMulAdd). "
            f"Available: {', '.join(available_patterns)}"
        ),
    )
    patterns_group.add_argument(
        "--all_patterns",
        action="store_true",
        help="Test all registered patterns",
    )
    parser.add_argument(
        "--ep",
        type=str,
        required=True,
        choices=["QNNExecutionProvider", "OpenVINOExecutionProvider"],
        help=(
            "Execution Provider names to test. "
            "Available: QNNExecutionProvider, OpenVINOExecutionProvider"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="NPU",
        choices=["CPU", "GPU", "NPU"],
        help="Target device type (CPU, GPU, NPU).",
    )
    parser.add_argument(
        "--opset_mapping",
        type=str,
        nargs="+",
        required=True,
        help=("Domain:version pairs for ONNX opset versions, e.g., ai.onnx:17 com.microsoft:1"),
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
        help="Limit number of test cases per pattern (default: run all cases)",
    )
    parser.add_argument(
        "--continue",
        action="store_true",
        dest="continue_from_existing",
        help="Continue from existing result file by skipping already completed test cases",
    )
    parser.add_argument(
        "--save_failed_model",
        action="store_true",
        help="Save the model for compile failed test cases",
    )
    args = parser.parse_args()

    # Parse opset_mapping from "domain:version" pairs
    opset_mapping = None
    if args.opset_mapping:
        opset_mapping = {}
        for pair in args.opset_mapping:
            domain, version = pair.split(":")
            opset_mapping[domain] = int(version)

    # Determine which patterns to test
    patterns_to_check = available_patterns if args.all_patterns else args.patterns
    ep_checker = get_ep_checker(args.ep, device=args.device)

    # Run the tests
    check_patterns(
        ep_checker,
        patterns=patterns_to_check,
        validate_inputs=args.validate_inputs,
        output_dir=args.output_dir,
        n_cases=args.n_cases,
        continue_from_existing=args.continue_from_existing,
        save_failed_model=args.save_failed_model,
        opset_mapping=opset_mapping,
    )


if __name__ == "__main__":
    parse_and_check()
