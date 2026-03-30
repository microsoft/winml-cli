# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test ONNX operators on QNN execution provider.

This script tests ONNX operators on the QNN execution provider,
generating test results for each specified operator.

Usage:
    Test specific operators:
        python -m modelkit.analyze.runtime_checker.check_qnn_ops --ops Abs Relu Sigmoid

    Test all registered operators:
        python -m modelkit.analyze.runtime_checker.check_qnn_ops --all_ops
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from google.protobuf import json_format
from onnx.defs import SchemaError

from winml.modelkit.onnx.domains import ONNXDomain
from winml.modelkit.pattern.op_input_gen import (
    OpInputGenerator,
    get_registered_operators,
    get_runtime_checker_op,
)
from winml.modelkit.pattern.op_input_gen.qdq_gen import QDQGenerator

from ... import winml
from ...sysinfo import SysInfo
from ...utils import constants
from ..utils.model_utils import get_op_since_version
from .ep_checker import EPChecker


winml.register_execution_providers(ort=True)


def _compute_case_signature(case: dict, *, namespace: str) -> str:
    """Compute a signature for a test case based on its content.

    The signature is used to match test cases across different runs,
    allowing delta detection when the input generator changes.

    Args:
        case: Test case dictionary containing type_vars, attrs, input_constraints, etc.

    Returns:
        A string signature that uniquely identifies the test case.
    """
    # Extract the key fields that define a test case
    sig_parts = []

    if namespace:
        # Namespacing keeps case_index stable per output file when signatures collide across files
        sig_parts.append(f"ns:{namespace}")

    def _safe_dump(obj: Any) -> str:
        def _default(o: Any):
            if isinstance(o, onnx.TensorProto):
                return json.loads(json_format.MessageToJson(o))
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.generic):
                return o.item()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        return json.dumps(obj, sort_keys=True, default=_default)

    def _is_empty_top_level(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (dict, list, tuple, set)):
            return len(value) == 0
        return False

    # Type variables (e.g., T=FLOAT)
    if "type_vars" in case:
        type_vars = case["type_vars"]
        sig_parts.append(f"types:{_safe_dump(type_vars)}")

    # Attributes
    if "attrs" in case:
        attrs = case["attrs"]
        if not _is_empty_top_level(attrs):
            sig_parts.append(f"attrs:{_safe_dump(attrs)}")

    # Input constraints (shapes/values)
    if "input_constraints" in case:
        constraints = case["input_constraints"]
        sig_parts.append(f"inputs:{_safe_dump(constraints)}")

    # Input is constant flags
    if "input_is_constant" in case:
        is_const = case["input_is_constant"]
        sig_parts.append(f"const:{_safe_dump(is_const)}")

    # Dynamic axes configuration
    if "dynamic_axes" in case:
        dynamic_axes = case["dynamic_axes"]
        if not _is_empty_top_level(dynamic_axes):
            sig_parts.append(f"dynamic:{_safe_dump(dynamic_axes)}")

    # QDQ configuration: include only when present to keep non-QDQ signatures stable.
    if "qdq_types" in case:
        qdq_types = case["qdq_types"]
        if not _is_empty_top_level(qdq_types):
            sig_parts.append(f"qdq:{_safe_dump(qdq_types)}")

    return "|".join(sig_parts)


def _hash_case_signature(signature: str) -> str:
    """Return a stable hash value for a case signature."""
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


class CheckResultWriter:
    """Writer for test results that supports continuation from existing files."""

    def __init__(
        self,
        file_path: str | Path,
        sys_info: dict[str, Any],
        save_per_cases: int | None = 100,
        rerun_failed: bool = False,
        delta_only: bool = False,
        not_run_start_id: int = 1,
        filter_case_index: str | list[str] | None = None,
    ) -> None:
        """Initialize the writer.

        Args:
            file_path: Path to the output JSON file
            sys_info: System information dictionary (constant during run)
            save_per_cases: Number of results to accumulate before saving to file.
                If None, disable periodic saves and save only on flush/finalization.
            rerun_failed: If True, rerun failed cases (compile or run failed).
            delta_only: If True, only run new test cases not in existing results.
        """
        self.file_path = Path(file_path)
        self.sys_info = sys_info
        self.save_per_cases = save_per_cases
        self.results = []
        self.pending_count = 0
        self.rerun_failed = rerun_failed
        self.delta_only = delta_only
        self.existing_signatures: dict[str, dict] = {}  # Signature -> existing result
        self.used_signatures: set[str] = set()  # Signatures already added to results
        self.output_signatures: set[str] = set()  # Signatures already emitted to output
        self.failed_signatures: set[str] = set()  # Signatures of failed cases
        self.duplicate_skipped_count = 0
        self._next_not_run_id = not_run_start_id
        self.case_namespace = (
            self.file_path.stem
        )  # File name without extension for case_index namespace
        if filter_case_index is None:
            self.filter_case_indices: list[str] | None = None
            self._filter_case_index_set: set[str] | None = None
        elif isinstance(filter_case_index, str):
            self.filter_case_indices = [filter_case_index]
            self._filter_case_index_set = {filter_case_index}
        else:
            self.filter_case_indices = list(filter_case_index)
            self._filter_case_index_set = set(self.filter_case_indices)

        # filter_case_index, delta_only, rerun_failed are mutually exclusive
        mode_count = int(self.filter_case_indices is not None) + int(delta_only) + int(rerun_failed)
        if mode_count > 1:
            raise ValueError("filter_case_index, delta_only, rerun_failed cannot be used together")

        if self.file_path.exists():
            with self.file_path.open("r", encoding="utf-8") as f:
                raw = f.read()

            # Treat an empty file as "no existing results" instead of failing the run.
            data = {} if not raw.strip() else json.loads(raw)
            if "check_results" in data:
                existing_cases = data["check_results"]
                # Build signature map for existing results that actually ran
                for case in existing_cases:
                    if self._contains_not_run_reason(case):
                        continue

                    sig = _compute_case_signature(case, namespace=self.case_namespace)
                    self.existing_signatures[sig] = case

                    check_result = case.get("check_result", {})
                    compile_success = (
                        check_result.get("compile", {}).get("result", {}).get("success", False)
                    )
                    run_success = (
                        check_result.get("run", {}).get("result", {}).get("success", False)
                    )
                    if not (compile_success and run_success):
                        self.failed_signatures.add(sig)

    def has_existing_results(self) -> bool:
        """Check if we have existing results to work with."""
        return len(self.existing_signatures) > 0

    def should_skip_case(self, case: dict) -> bool:
        """Check if a case should be skipped based on its signature.

        Args:
            case: The test case (before running check_result)

        Returns:
            True if the case should be skipped.
        """
        sig = _compute_case_signature(case, namespace=self.case_namespace)
        if self.filter_case_indices is not None:
            assert self._filter_case_index_set is not None
            return _hash_case_signature(sig) not in self._filter_case_index_set

        if self.delta_only:
            # Only run brand-new cases; skip anything we already have
            return sig in self.existing_signatures

        if self.rerun_failed:
            # Only rerun known failed cases; skip successes and any new cases
            return sig not in self.failed_signatures

        # Default: run everything
        return False

    def append_result(self, case: dict[str, Any]) -> None:
        """Append a test result and save to file periodically.

        Args:
            case: Test case dictionary
        """
        sig = _compute_case_signature(case, namespace=self.case_namespace)
        if sig in self.output_signatures:
            self.duplicate_skipped_count += 1
            return

        self._assign_not_run_ids(case)
        self._set_case_index_signature(case)
        self.results.append(case)
        self.output_signatures.add(sig)
        self._increment_pending_and_maybe_save()

    def reuse_existing_result(self, case: dict) -> bool:
        """Reuse an existing result for a skipped case.

        This does NOT trigger periodic saves since reused results already exist
        in the file and intermediate saves preserve remaining existing cases.

        Args:
            case: The skipped case (with _skipped marker)

        Returns:
            True if existing result was found and reused, False otherwise.
        """
        sig = _compute_case_signature(case, namespace=self.case_namespace)
        if self.filter_case_indices is not None:
            assert self._filter_case_index_set is not None
            if _hash_case_signature(sig) not in self._filter_case_index_set:
                return False

        existing_case = self.existing_signatures.get(sig)
        if existing_case:
            self.used_signatures.add(sig)
            if sig in self.output_signatures:
                self.duplicate_skipped_count += 1
                return False  # duplicate reuse should not count as reused
            self._set_case_index_signature(existing_case)
            self.results.append(existing_case)
            self.output_signatures.add(sig)
            return True
        return False

    def _set_case_index_signature(self, case: dict[str, Any]) -> None:
        """Set case_index to a stable hash derived from normalized signature."""
        signature = _compute_case_signature(case, namespace=self.case_namespace)
        case["case_index"] = _hash_case_signature(signature)

    def _contains_not_run_reason(self, case: dict[str, Any]) -> bool:
        """Check whether compile/run reason contains a not_run placeholder."""
        check_result = case.get("check_result")
        if not isinstance(check_result, dict):
            return False

        for stage in ("compile", "run"):
            stage_result = check_result.get(stage)
            if not isinstance(stage_result, dict):
                continue
            payload = stage_result.get("result")
            if not isinstance(payload, dict):
                continue
            reason = payload.get("reason")
            if isinstance(reason, str) and reason.startswith("not_run"):
                return True
        return False

    def _assign_not_run_ids(self, case: dict[str, Any]) -> None:
        """Assign a single sequential not_run id per case, shared by compile/run reasons.

        If either compile or run has a not_run reason, both stages share the same
        auto-incremented id. This keeps the pair logically tied together instead
        of consuming two separate ids for one case.
        """
        check_result = case.get("check_result")
        if not isinstance(check_result, dict):
            return

        not_run_payloads: list[dict[str, Any]] = []
        for stage in ("compile", "run"):
            stage_result = check_result.get(stage)
            if not isinstance(stage_result, dict):
                continue
            payload = stage_result.get("result")
            if not isinstance(payload, dict):
                continue
            reason = payload.get("reason")
            if isinstance(reason, str) and reason.startswith("not_run"):
                not_run_payloads.append(payload)

        if not not_run_payloads:
            return

        new_id = self._next_not_run_id
        self._next_not_run_id += 1
        for payload in not_run_payloads:
            payload["reason"] = f"not_run_{new_id}"

    def _increment_pending_and_maybe_save(self) -> None:
        """Increment pending count and save if threshold reached."""
        self.pending_count += 1
        if self.save_per_cases is not None and self.pending_count >= self.save_per_cases:
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

    def get_duplicate_skipped_count(self) -> int:
        """Get the number of duplicate-signature cases skipped from output."""
        return self.duplicate_skipped_count

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

        # Append remaining existing cases to prevent data loss during iteration.
        # When filtering by case_index, avoid re-emitting unrelated cases.
        if self.existing_signatures and self.filter_case_indices is None:
            remaining_sigs = set(self.existing_signatures.keys()) - self.used_signatures
            if remaining_sigs:
                remaining_results = [self.existing_signatures[sig] for sig in remaining_sigs]
                results_to_save = self.results + remaining_results

        output_data = {
            "check_results": results_to_save,
            # NOTE: Intentionally do not persist sys_info.
            # This file may be updated across multiple runs, and different cases in
            # the same output can come from different run environments/sys_info.
            # Persisting one top-level sys_info would be misleading.
            # "sys_info": self.sys_info,
        }

        for item in output_data["check_results"]:
            if isinstance(item, dict):
                self._set_case_index_signature(item)

        # Sort results by case_index to keep deterministic ordering before writing
        output_data["check_results"] = sorted(
            output_data["check_results"],
            key=lambda x: x.get("case_index", "") if isinstance(x, dict) else "",
        )

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

        # Decide final save path: when filtering a single case, write to a _case_ suffixed file
        save_path = self.file_path
        if self.filter_case_indices:
            primary = self.filter_case_indices[0]
            case_suffix = primary[:12]
            if len(self.filter_case_indices) > 1:
                case_suffix = f"{case_suffix}_plus{len(self.filter_case_indices) - 1}"
            save_path = self.file_path.with_name(
                f"{self.file_path.stem}_cases_{case_suffix}{self.file_path.suffix}"
            )

        # Force CRLF in output JSON to align with consumer expectations
        with save_path.open("w", encoding="utf-8", newline="\r\n") as f:
            json.dump(output_data, f, indent=2, default=json_default, allow_nan=False)


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
