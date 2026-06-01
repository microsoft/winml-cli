# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Utility functions for operator check result management."""

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from google.protobuf import json_format

from ...pattern.op_input_gen import normalize_constraint_dict


def load_case_indices_from_conflict_file(conflict_file: str | Path) -> list[str]:
    """Load case_index values from the 2nd CSV column of a conflict file.

    The expected layout is compatible with conflict CSVs where column 1 is
    groupid and column 2 is case_index.
    """
    conflict_path = Path(conflict_file).expanduser()
    if not conflict_path.is_absolute():
        raise ValueError("--conflict_file must be an absolute path")
    if not conflict_path.exists():
        raise FileNotFoundError(f"Conflict file not found: {conflict_path}")

    case_indices: list[str] = []
    with conflict_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) < 2:
                raise ValueError(
                    f"Conflict file row {row_idx} has fewer than 2 columns: {conflict_path}"
                )

            case_value = str(row[1]).strip()
            if not case_value:
                continue

            # Skip header row like: groupid,case_index,...
            if row_idx == 1 and case_value.lower() == "case_index":
                continue

            case_indices.append(case_value)

    case_indices = list(dict.fromkeys(case_indices))
    if not case_indices:
        raise ValueError(f"No case_index values found in conflict file: {conflict_path}")
    return case_indices


def compute_case_signature(case: dict, *, namespace: str) -> str:
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
        def _default(o: Any) -> Any:
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
        constraints = {
            k: normalize_constraint_dict(v) if isinstance(v, dict) else v
            for k, v in case["input_constraints"].items()
        }
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


def hash_case_signature(signature: str) -> str:
    """Return a stable hash value for a case signature."""
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def build_namespace_without_ep_device(namespace: str) -> str:
    """Build a namespace that is stable across EP/device variants of the same file stem."""
    if not namespace:
        return namespace

    stem = namespace[:-4] if namespace.endswith("_qdq") else namespace
    match = re.match(r"^(?P<prefix>.+)_([^_]+)_(CPU|GPU|NPU)_([^_]+)_opset(?P<opset>\d+)$", stem)
    if match is None:
        return namespace

    prefix = match.group("prefix")
    domain = match.group(4)
    opset = match.group("opset")
    ep_device_agnostic = f"{prefix}_{domain}_opset{opset}"
    if namespace.endswith("_qdq"):
        ep_device_agnostic = f"{ep_device_agnostic}_qdq"
    return ep_device_agnostic


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
        self.results: list[dict[str, Any]] = []
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
        self.case_namespace_ignore_ep_device = build_namespace_without_ep_device(
            self.case_namespace
        )
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

                    sig = compute_case_signature(case, namespace=self.case_namespace)
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
        sig = compute_case_signature(case, namespace=self.case_namespace)
        if self.filter_case_indices is not None:
            assert self._filter_case_index_set is not None
            return hash_case_signature(sig) not in self._filter_case_index_set

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
        sig = compute_case_signature(case, namespace=self.case_namespace)
        if sig in self.output_signatures:
            self.duplicate_skipped_count += 1
            return

        self._assign_not_run_ids(case)
        self._set_case_index_signatures(case)
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
        sig = compute_case_signature(case, namespace=self.case_namespace)
        if self.filter_case_indices is not None:
            assert self._filter_case_index_set is not None
            if hash_case_signature(sig) not in self._filter_case_index_set:
                return False

        existing_case = self.existing_signatures.get(sig)
        if existing_case:
            self.used_signatures.add(sig)
            if sig in self.output_signatures:
                self.duplicate_skipped_count += 1
                return False  # duplicate reuse should not count as reused
            # Re-apply current serialization to input_constraints so reused cases
            # get the same compact same_value format that a fresh run would produce.
            # The skipped case (from the generator) already went through to_dict()
            # via iter(), so its input_constraints is up-to-date.
            if "input_constraints" in case:
                existing_case["input_constraints"] = case["input_constraints"]
            # Ensure reused cases keep the current model payload contract.
            if isinstance(case.get("model_bytes_b64"), str):
                existing_case["model_bytes_b64"] = case["model_bytes_b64"]
            self._set_case_index_signatures(existing_case)
            self.results.append(existing_case)
            self.output_signatures.add(sig)
            return True
        return False

    def _set_case_index_signatures(self, case: dict[str, Any]) -> None:
        """Set both case index signatures derived from normalized case content."""
        signature = compute_case_signature(case, namespace=self.case_namespace)
        case["case_index"] = hash_case_signature(signature)

        signature_ignore_ep_device = compute_case_signature(
            case,
            namespace=self.case_namespace_ignore_ep_device,
        )
        case["case_index_ignore_ep_device"] = hash_case_signature(signature_ignore_ep_device)

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
                self._set_case_index_signatures(item)

        # Sort results by case_index to keep deterministic ordering before writing
        output_data["check_results"] = sorted(
            output_data["check_results"],
            key=lambda x: x.get("case_index", "") if isinstance(x, dict) else "",
        )

        def json_default(obj: Any) -> Any:
            if isinstance(obj, onnx.TensorProto):
                return json.loads(json_format.MessageToJson(obj))
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, float) and math.isnan(obj):
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
