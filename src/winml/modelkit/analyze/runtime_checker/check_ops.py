"""Test ONNX operators on QNN execution provider.

This script tests ONNX operators on the QNN execution provider,
generating test results for each specified operator.

Usage:
    Test specific operators:
        python -m modelkit.analyze.runtime_checker.check_qnn_ops --ops Abs Relu Sigmoid

    Test all registered operators:
        python -m modelkit.analyze.runtime_checker.check_qnn_ops --all_ops
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from google.protobuf import json_format
from onnx.defs import SchemaError

from ... import winml
from ...sysinfo import SysInfo
from ...utils import constants
from winml.modelkit.onnx.domains import ONNXDomain
from ..utils.model_utils import get_op_since_version
from .ep_checker import EPChecker
from winml.modelkit.pattern.op_input_gen import (
    OpInputGenerator,
    get_registered_operators,
    get_runtime_checker_op,
)
from winml.modelkit.pattern.op_input_gen.qdq_gen import QDQGenerator


winml.register_execution_providers(ort=True)

RERUN_ERROR_FILE = "need_rerun_errors.json"


def _get_rerun_error_config_path() -> Path | None:
    """Locate rerun error config file next to this script."""

    config_path = Path(__file__).resolve().parent / RERUN_ERROR_FILE
    return config_path if config_path.exists() else None


def _load_rerun_error_patterns(config_path: Path) -> list[re.Pattern[str]]:
    """Load regex patterns from rerun error config file.

    Expected format: JSON array of regex strings, e.g. ["Unable to compile", "timeout"].
    """

    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # best-effort load
        print(f"Failed to load rerun error config {config_path}: {exc}")
        return []

    if not isinstance(data, list):
        print(f"Rerun error config {config_path} must be a JSON array of regex strings; got {type(data).__name__}.")
        return []

    raw_patterns = [p for p in data if isinstance(p, str)]
    if len(raw_patterns) != len(data):
        print(f"Rerun error config {config_path} contains non-string entries; only string patterns are used.")

    compiled: list[re.Pattern[str]] = []
    for pattern in raw_patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            print(f"Skip invalid rerun error pattern {pattern!r}: {exc}")
    return compiled


def _compute_case_signature(result: dict) -> str:
    """Compute a signature for a test case based on its content.

    The signature is used to match test cases across different runs,
    allowing delta detection when the input generator changes.

    Args:
        result: Test result dictionary containing type_vars, attrs, input_constraints, etc.

    Returns:
        A string signature that uniquely identifies the test case.
    """
    # Extract the key fields that define a test case
    sig_parts = []

    def _safe_dump(obj: Any) -> str:
        def _default(o: Any):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.generic):
                return o.item()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        return json.dumps(obj, sort_keys=True, default=_default)

    # Type variables (e.g., T=FLOAT)
    if "type_vars" in result:
        type_vars = result["type_vars"]
        sig_parts.append(f"types:{_safe_dump(type_vars)}")

    # Attributes
    if "attrs" in result:
        attrs = result["attrs"]
        sig_parts.append(f"attrs:{_safe_dump(attrs)}")

    # Input constraints (shapes/values)
    if "input_constraints" in result:
        constraints = result["input_constraints"]
        sig_parts.append(f"inputs:{_safe_dump(constraints)}")

    # Input is constant flags
    if "input_is_constant" in result:
        is_const = result["input_is_constant"]
        sig_parts.append(f"const:{_safe_dump(is_const)}")

    return "|".join(sig_parts)


class CheckResultWriter:
    """Writer for test results that supports continuation from existing files."""

    def __init__(
        self,
        file_path: str | Path,
        sys_info: dict[str, Any],
        save_per_cases: int = 100,
        rerun_failed: bool = False,
        rerun_failed_with_filter: bool = False,
        delta_only: bool = False,
    ) -> None:
        """Initialize the writer.

        Args:
            file_path: Path to the output JSON file
            sys_info: System information dictionary (constant during run)
            save_per_cases: Number of results to accumulate before saving to file
            rerun_failed: If True, rerun failed cases (compile or run failed).
            delta_only: If True, only run new test cases not in existing results.
        """
        self.file_path = Path(file_path)
        self.sys_info = sys_info
        self.save_per_cases = save_per_cases
        self.results = []
        self.pending_count = 0
        self.rerun_failed = rerun_failed
        self.rerun_failed_with_filter = rerun_failed_with_filter
        self.delta_only = delta_only
        self.rerun_error_patterns: list[re.Pattern[str]] = []
        self.rerun_error_config_path: Path | None = None
        self.existing_signatures: dict[str, dict] = {}  # Signature -> existing result
        self.used_signatures: set[str] = set()  # Signatures already added to results
        self.failed_signatures: set[str] = set()  # Signatures of failed cases
        self.save_count = 0
        self.save_events: list[dict[str, Any]] = []
        self.save_log_path = os.getenv("CHECK_OPS_SAVE_LOG")

        if rerun_failed_with_filter:
            self.rerun_error_config_path = _get_rerun_error_config_path()
            if self.rerun_error_config_path:
                self.rerun_error_patterns = _load_rerun_error_patterns(self.rerun_error_config_path)
                if self.rerun_error_patterns:
                    print(
                        f"Loaded {len(self.rerun_error_patterns)} rerun error pattern(s) from {self.rerun_error_config_path}"
                    )
                else:
                    print(
                        f"Rerun error config {self.rerun_error_config_path} found but no valid patterns loaded; rerun_failed_with_filter will rerun 0 cases."
                    )
            else:
                print("No need_rerun_errors.json found next to check_ops.py; rerun_failed_with_filter will rerun 0 cases.")

        # Read existing file if rerun_failed or delta_only is set
        if (rerun_failed or delta_only) and self.file_path.exists():
            with self.file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if "check_results" in data:
                    existing_results = data["check_results"]
                    # Build signature map for all existing results
                    for result in existing_results:
                        sig = _compute_case_signature(result)
                        self.existing_signatures[sig] = result

                        # Track failed cases if rerun_failed is set
                        if rerun_failed:
                            check_result = result.get("check_result", {})
                            compile_success = (
                                check_result.get("compile", {})
                                .get("result", {})
                                .get("success", False)
                            )
                            run_success = (
                                check_result.get("run", {}).get("result", {}).get("success", False)
                            )
                            if not (compile_success and run_success):
                                if not self.rerun_failed_with_filter or self._should_rerun_failure(check_result):
                                    self.failed_signatures.add(sig)

                    rerun_filter_suffix = ""
                    if rerun_failed and self.rerun_failed_with_filter:
                        rerun_filter_suffix = (
                            f" matching {len(self.rerun_error_patterns)} error pattern(s) from {self.rerun_error_config_path}"
                            if self.rerun_error_patterns
                            else " with filter but no valid patterns (0 cases will be rerun)"
                        )

                    # Print status
                    if rerun_failed and delta_only:
                        print(
                            f"Found existing file with {len(existing_results)} test cases. "
                            f"Will rerun {len(self.failed_signatures)} failed cases{rerun_filter_suffix} and run new cases only."
                        )
                    elif rerun_failed:
                        print(
                            f"Found existing file with {len(existing_results)} test cases. "
                            f"Will rerun {len(self.failed_signatures)} failed cases{rerun_filter_suffix}."
                        )
                    elif delta_only:
                        print(
                            f"Found existing file with {len(existing_results)} test cases. "
                            f"Will run new cases only (delta mode)."
                        )

    def has_existing_results(self) -> bool:
        """Check if we have existing results to work with."""
        return len(self.existing_signatures) > 0

    def should_skip_case(self, result: dict) -> bool:
        """Check if a case should be skipped based on its signature.

        Logic:
        - New cases (not in existing): skip unless delta_only is set
        - Failed cases (in existing): skip unless rerun_failed is set
        - Passed cases (in existing): always skip (when this method is called)

        Args:
            result: The test case result (before running check_result)

        Returns:
            True if the case should be skipped.
        """
        sig = _compute_case_signature(result)

        # New case - only run if delta_only is set
        if sig not in self.existing_signatures:
            return not self.delta_only

        # Failed case - only run if rerun_failed is set
        if sig in self.failed_signatures:
            return not self.rerun_failed

        # Passed case - always skip
        return True

    def append_result(self, result: dict[str, Any]) -> None:
        """Append a test result and save to file periodically.

        Args:
            result: Test result dictionary
        """
        self.results.append(result)
        self._increment_pending_and_maybe_save()

    def reuse_existing_result(self, result: dict) -> bool:
        """Reuse an existing result for a skipped case.

        This does NOT trigger periodic saves since reused results already exist
        in the file and intermediate saves preserve remaining existing cases.

        Args:
            result: The skipped result (with _skipped marker)

        Returns:
            True if existing result was found and reused, False otherwise.
        """
        sig = _compute_case_signature(result)
        existing_result = self.existing_signatures.get(sig)
        if existing_result:
            self.results.append(existing_result)
            self.used_signatures.add(sig)
            return True
        return False

    def _increment_pending_and_maybe_save(self) -> None:
        """Increment pending count and save if threshold reached."""
        self.pending_count += 1
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
        # Always save if we have results (handles both normal and rerun_failed modes)
        if self.results:
            self._save()
            self.pending_count = 0

    def get_dropped_count(self) -> int:
        """Get the number of existing cases that were dropped (not in current generator output)."""
        if not self.existing_signatures:
            return 0
        return len(set(self.existing_signatures.keys()) - self.used_signatures)

    def _should_rerun_failure(self, check_result: dict[str, Any]) -> bool:
        """Decide whether a failed case should be rerun based on error patterns."""

        if not self.rerun_failed_with_filter:
            return True
        if not self.rerun_error_patterns:
            return False

        compile_result = check_result.get("compile", {}).get("result", {})
        run_result = check_result.get("run", {}).get("result", {})
        return self._reason_matches(compile_result) or self._reason_matches(run_result)

    def _reason_matches(self, result: dict[str, Any]) -> bool:
        if result.get("success") is True:
            return False
        reason = result.get("reason")
        if not isinstance(reason, str):
            return False
        return any(pattern.search(reason) for pattern in self.rerun_error_patterns)

    def finalize(self) -> None:
        """Finalize the writer after iteration is complete.

        This clears unused existing signatures so they won't be saved in the final flush.
        Call this after processing all cases from the generator.
        """
        self.existing_signatures.clear()

    def _save(self) -> None:
        """Save results to file.

        During iteration, appends remaining existing cases to prevent data loss on crash.
        After finalize() is called, only saves the accumulated results.
        """
        results_to_save = self.results

        # Append remaining existing cases to prevent data loss during iteration
        if self.existing_signatures:
            remaining_sigs = set(self.existing_signatures.keys()) - self.used_signatures
            if remaining_sigs:
                remaining_results = [self.existing_signatures[sig] for sig in remaining_sigs]
                results_to_save = self.results + remaining_results

        output_data = {
            "check_results": results_to_save,
            "sys_info": self.sys_info,
        }

        def json_default(obj):
            if isinstance(obj, onnx.TensorProto):
                return json.loads(json_format.MessageToJson(obj))
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, float) and (obj != obj):  # NaN check: NaN != NaN
                return None  # Convert NaN to null
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with self.file_path.open("w", encoding="utf-8", newline="\n") as f:
            start = time.perf_counter()
            json.dump(output_data, f, indent=2, default=json_default, allow_nan=False)
            duration = time.perf_counter() - start
        self.save_count += 1
        self.save_events.append(
            {
                "index": self.save_count,
                "duration_sec": duration,
                "results_in_batch": len(self.results),
                "pending_after_save": self.pending_count,
                "path": str(self.file_path),
            }
        )
        self._write_save_log(duration)
        print(
            f"Saved batch #{self.save_count} ({len(self.results)} results, pending {self.pending_count}) to {self.file_path} in {duration:.2f}s"
        )

    def _write_save_log(self, duration: float) -> None:
        if not self.save_log_path:
            return
        try:
            path = Path(self.save_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            line = (
                f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{self.file_path.stem},{self.save_count},{duration:.2f},{len(self.results)},{self.pending_count},{self.file_path}"
            )
            with path.open("a", encoding="utf-8") as log_file:
                log_file.write(line + "\n")
        except Exception:
            # Best-effort logging; do not break main flow
            pass


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
    rerun_failed_with_filter: bool = False,
    delta_only: bool = False,
    use_qdq: bool = False,
    dry_run: bool = False,
    dynamic_axis_mode: str = "none",
):
    """Run operators on execution provider.

    Args:
        ops: List of operator names to test (e.g., ["Abs", "Relu", "Sigmoid"])
        opset_version: ONNX opset version to use
        opset_domain: ONNX opset domain (e.g., "ai.onnx", "com.microsoft")
        validate_inputs: Whether to validate input combinations before testing
        output_dir: Output directory for test results JSON files (default: current directory)
        model_output_dir: Directory to save generated ONNX models. Defaults to output_dir/saved_models.
        n_cases: If not None, only run the first n_cases test cases for each operator.
                 If n_cases is greater than total cases, run all cases.
        save_failed_model: If True, save the ONNX model when a test case fails.
        rerun_failed: If True, rerun failed cases (compile or run failed).
        delta_only: If True, only run new test cases not in existing results.
                    Can be combined with rerun_failed.
        dynamic_axis_mode: Dynamic axis testing mode for input generators.
    """
    sys_info = SysInfo().to_dict()
    domain = ONNXDomain.from_str(opset_domain)

    qdq_gen = QDQGenerator(1, ONNXDomain.COM_MICROSOFT) if use_qdq else None

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Directory to stash saved ONNX models (separate from json outputs); creation is lazy in generators.
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
            output_filename = f"{op_name}_{ep_checker.ep_name}_{constants.DEVICE_TYPE_TO_DEVICE[ep_checker.device_type]}_{opset_domain}_opset{since_version}{'_qdq' if use_qdq else ''}.json"
            output_path = output_dir / output_filename

            # Use writer as context manager (auto-flushes on exit)
            with CheckResultWriter(
                output_path,
                sys_info,
                rerun_failed=rerun_failed or rerun_failed_with_filter,
                rerun_failed_with_filter=rerun_failed_with_filter,
                delta_only=delta_only,
            ) as writer:
                # Run tests on execution provider
                print(f"Running {op_name} tests on {ep_checker.ep_name}...")
                if n_cases is not None:
                    print(f"Limiting to first {n_cases} test cases")

                if writer.has_existing_results():
                    # We have existing results - use skip logic
                    check_results_iter = gen.check_on_ep(
                        ep_checker,
                        capture_output=True,
                        n_cases=n_cases,
                        skip_cases=0,
                        save_failed_model=save_failed_model,
                        save_model=save_model,
                        model_output_dir=model_output_dir,
                        skip_signature_fn=writer.should_skip_case,
                        yield_skipped=True,  # Also yield skipped cases (with skip marker) to maintain order
                        dry_run=dry_run,
                    )

                    # Process results in generator order - reuse existing or run new
                    run_count = 0
                    reused_count = 0
                    for result in check_results_iter:
                        if result.get("_skipped"):
                            # This case was skipped - reuse existing result
                            if writer.reuse_existing_result(result):
                                reused_count += 1
                        else:
                            # Case was actually run (new or rerun)
                            writer.append_result(result)
                            run_count += 1

                    dropped_count = writer.get_dropped_count()
                    print(
                        f"Ran {run_count} test cases, reused {reused_count} existing cases, dropped {dropped_count} obsolete cases."
                    )

                    # Finalize to clear unused signatures before final flush
                    writer.finalize()
                else:
                    check_results_iter = gen.check_on_ep(
                        ep_checker,
                        capture_output=True,
                        n_cases=n_cases,
                        save_failed_model=save_failed_model,
                        save_model=save_model,
                        model_output_dir=model_output_dir,
                        dry_run=dry_run,
                    )

                    # Process results using writer
                    for result in check_results_iter:
                        writer.append_result(result)

                check_results = writer.results

                if writer.save_events:
                    durations = ", ".join(f"{evt['duration_sec']:.2f}s" for evt in writer.save_events)
                    print(
                        f"Save stats for {op_name}: {writer.save_count} saves | durations: {durations}"
                    )
                else:
                    print(f"Save stats for {op_name}: 0 saves (no results written)")

            print(f"\nResults saved to: {output_path}")
            print(f"Total test cases: {len(check_results)}")

            if version_until is not None:
                if since_version <= 1:
                    break
                if since_version <= version_until:
                    print(
                        f"opset_version {since_version} is already <= version_until {version_until}, stopping further testing for {op_name}."
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
            f"Unsupported execution provider: {ep_name}. Available: QNNExecutionProvider, OpenVINOExecutionProvider, VitisAIExecutionProvider, MIGraphXExecutionProvider, NvTensorRTRTXExecutionProvider"
        )
    return ep_name_to_checker[ep_name](device_type=device_type)


def parse_and_check() -> None:
    """Main entry point for command-line execution."""
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
            "Available: QNNExecutionProvider, OpenVINOExecutionProvider, VitisAIExecutionProvider, MIGraphXExecutionProvider, NvTensorRTRTXExecutionProvider"
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
    rerun_group = parser.add_mutually_exclusive_group()
    rerun_group.add_argument(
        "--rerun_failed",
        action="store_true",
        help="Rerun all failed cases (compile failed or run failed). Can be combined with --delta_only.",
    )
    rerun_group.add_argument(
        "--rerun_failed_with_filter",
        action="store_true",
        help="Rerun failed cases only when their error reason matches patterns in need_rerun_errors.json.",
    )
    parser.add_argument(
        "--delta_only",
        action="store_true",
        help="Only run new test cases that don't exist in the existing results file. Can be combined with --rerun_failed.",
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
        "--with_dynamic",
        action="store_true",
        help="Also test with first axis as dynamic (axis 0) for non-constant, non-scalar inputs",
    )
    parser.add_argument(
        "--version_until",
        type=int,
        default=None,
        # For example, version_until=13, opset_version=20, opset has 11,12,15 will test 12 and 15
        help="Test each distinct operator schema version down to the first one <= this, up to the specified opset_version.",
    )
    args = parser.parse_args()

    # Determine which operators to test
    ops_to_check = available_ops if args.all_ops else args.ops
    ep_checker = get_ep_checker(args.ep, device=args.device)
    # Run the tests
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
        rerun_failed_with_filter=args.rerun_failed_with_filter,
        delta_only=args.delta_only,
        use_qdq=args.use_qdq,
        dry_run=args.dry_run,
        dynamic_axis_mode="first_axis_dynamic" if args.with_dynamic else "none",
    )


if __name__ == "__main__":
    parse_and_check()
