# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test ONNX patterns on execution providers.

This script tests ONNX patterns (subgraph patterns like Gelu, MatMulAdd)
on execution providers, generating test results for each specified pattern.

Usage:
    Test specific patterns:
        python -m winml.modelkit.analyze.pattern.check_patterns --patterns Gelu MatMulAdd

    Test all registered patterns:
        python -m winml.modelkit.analyze.pattern.check_patterns --all_patterns
"""

from pathlib import Path
from typing import Any

import onnxruntime as ort

from ... import winml
from ...onnx import ONNXDomain
from ...pattern.base import (
    PatternInputGenerator,
    get_pattern_input_generator,
    get_registered_pattern_input_generators,
)
from ...sysinfo import SysInfo
from ...utils import constants
from ..runtime_checker.ep_checker import EPChecker
from ..utils import CheckResultWriter


winml.register_execution_providers(ort=True)


def check_patterns(
    ep_checker: EPChecker,
    patterns: list[str],
    validate_inputs: bool = False,
    output_dir: str | Path = ".",
    n_cases: int | None = None,
    save_failed_model: bool = False,
    rerun_failed: bool = False,
    delta_only: bool = False,
    dry_run: bool = False,
    not_run_start_id: int = 1,
    case_index: str | list[str] | None = None,
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
        save_failed_model: If True, save the model for compile failed test cases.
        rerun_failed: If True, rerun failed cases (compile or run failed).
        delta_only: If True, only run new test cases not in existing results.
        dry_run: If True, skip compile/run execution and emit check_result with reason "not_run".
        not_run_start_id: Initial id used for not_run placeholder reasons (not_run_<id>).
        case_index: Optional hashed signature(s) to filter to specific test cases.
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

    if opset_mapping is None:
        raise ValueError("opset_mapping must be provided for pattern model generation")

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
            save_per_cases=None if dry_run else 20,
            rerun_failed=rerun_failed,
            delta_only=delta_only,
            not_run_start_id=not_run_start_id,
            filter_case_index=case_index,
        ) as writer:
            # Run tests on execution provider
            print(f"Running {pattern_name} tests on {ep_checker.ep_name}...")
            if n_cases is not None:
                print(f"Limiting to first {n_cases} test cases")

            check_results_iter = gen.check_on_ep(
                ep_checker,
                capture_output=True,
                n_cases=n_cases,
                skip_cases=0,
                save_failed_model=save_failed_model,
                skip_signature_fn=writer.should_skip_case,
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
    ep_name_to_checker: dict[str, Any] = {
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


def build_parser():
    """Build argument parser for check_patterns-style commands."""
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
        # CARVE-OUT: This subprocess tool intentionally supports only a curated subset of
        # NPU EPs. VitisAI and future NPU EPs are excluded because this pattern-checking
        # tool has not been validated against them. Do NOT derive from eps_for_device("npu")
        # or EP_DEVICE_SPECS — this is an explicit opt-in allowlist, not catalog drift.
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

    opset_group = parser.add_mutually_exclusive_group(required=True)
    opset_group.add_argument(
        "--opset_mapping",
        type=str,
        nargs="+",
        help=(
            "Domain:version pairs for ONNX opset versions, "
            "e.g., ai.onnx:17 com.microsoft:1"
        ),
    )
    opset_group.add_argument(
        "--opset_version",
        type=int,
        help=(
            "ONNX opset version to use together with --opset_domain. "
            "If used without --opset_mapping, com.microsoft:1 is added automatically."
        ),
    )
    parser.add_argument(
        "--opset_domain",
        type=str,
        default=ONNXDomain.AI_ONNX.value,
        help=(
            "ONNX opset domain to use with --opset_version "
            f"(default: {ONNXDomain.AI_ONNX.value})"
        ),
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
        "--dry_run",
        action="store_true",
        help="Skip compile/run execution and emit check_result with reason 'not_run'",
    )
    parser.add_argument(
        "--not_run_start_id",
        type=int,
        default=1,
        help="Initial id used for not_run placeholder reasons (not_run_<id>) (default: 1)",
    )
    parser.add_argument(
        "--save_failed_model",
        action="store_true",
        help="Save the model for compile failed test cases",
    )
    return parser


def _parse_opset_mapping(args: Any) -> dict[str, int]:
    """Parse opset mapping from CLI args.

    Supports either:
    - --opset_mapping domain:version [domain:version ...]
    - --opset_version + optional --opset_domain
    """
    if args.opset_mapping:
        opset_mapping: dict[str, int] = {}
        for pair in args.opset_mapping:
            if ":" not in pair:
                raise ValueError(
                    "Invalid --opset_mapping value "
                    f"'{pair}'. Expected format: domain:version"
                )
            domain, version_text = pair.split(":", 1)
            if not domain:
                raise ValueError(f"Invalid --opset_mapping value '{pair}': empty domain")
            try:
                opset_mapping[domain] = int(version_text)
            except ValueError as exc:
                raise ValueError(
                    "Invalid --opset_mapping value "
                    f"'{pair}'. Version must be an integer"
                ) from exc
        return opset_mapping

    if args.opset_version is None:
        raise ValueError("Either --opset_mapping or --opset_version must be provided")

    opset_mapping = {args.opset_domain: int(args.opset_version)}

    # Keep compatibility with existing pattern generators that expect this domain.
    if args.opset_domain != ONNXDomain.COM_MICROSOFT.value:
        opset_mapping.setdefault(ONNXDomain.COM_MICROSOFT.value, 1)

    return opset_mapping


def run_from_args(args: Any) -> None:
    """Run check_patterns from parsed CLI args."""
    available_patterns = get_registered_pattern_input_generators()

    # Determine which patterns to test
    patterns_to_check = available_patterns if args.all_patterns else args.patterns
    ep_checker = get_ep_checker(args.ep, device=args.device)

    # Parse opset mapping from either mapping pairs or opset_domain/opset_version
    opset_mapping = _parse_opset_mapping(args)

    # Run the tests
    check_patterns(
        ep_checker,
        patterns=patterns_to_check,
        validate_inputs=args.validate_inputs,
        output_dir=args.output_dir,
        n_cases=args.n_cases,
        save_failed_model=args.save_failed_model,
        rerun_failed=args.rerun_failed,
        delta_only=args.delta_only,
        dry_run=args.dry_run,
        not_run_start_id=args.not_run_start_id,
        case_index=args.case_index,
        opset_mapping=opset_mapping,
    )


def parse_and_check() -> None:
    """Main entry point for command-line execution."""
    parser = build_parser()
    args = parser.parse_args()
    run_from_args(args)


if __name__ == "__main__":
    parse_and_check()
