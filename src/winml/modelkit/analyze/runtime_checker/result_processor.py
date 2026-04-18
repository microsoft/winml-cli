# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from colorama import Fore, Style
from onnx.defs import SchemaError, onnx_opset_version

from ...onnx import ONNXDomain
from ...pattern.base import get_pattern_input_generator
from ...pattern.op_input_gen import (
    OpInputGenerator,
    get_runtime_checker_op,
    normalize_constraint_dict,
)
from ..utils.model_utils import get_op_since_version, make_hashable
from ..utils.rule_loader import get_runtime_rules_search_dirs


def _get_input_constraint_types(
    check_results: list[dict[str, Any]],
) -> dict[str, str]:
    """Determine the constraint type for each input from non-None constraints.

    Scans all check results to find the constraint type (shape/value/variadic)
    used for each input when it is provided (not None).

    Args:
        check_results: List of check result items

    Returns:
        Dict mapping input_name to constraint type ("shape", "value", or "variadic")
    """
    input_constraint_types: dict[str, str] = {}
    for item in check_results:
        for input_name, constraint in item["input_constraints"].items():
            if (
                input_name not in item["attrs"]
                and constraint is not None
                and input_name not in input_constraint_types
            ):
                input_constraint_types[input_name] = constraint["type"]
    return input_constraint_types


def _get_all_attr_names(check_results: list[dict[str, Any]]) -> set[str]:
    """Collect all attribute names across all check results.

    Args:
        check_results: List of check result items

    Returns:
        Set of all attribute names found
    """
    all_attrs: set[str] = set()
    for item in check_results:
        all_attrs.update(item["attrs"].keys())
    return all_attrs


def item_to_row(
    item: dict[str, Any],
    input_constraint_types: dict[str, str] | None = None,
    all_attr_names: set[str] | None = None,
    replace_float_with_dummy: bool = True,
    use_qdq: bool = False,
) -> dict[str, Any]:
    """Convert a check result item to a flat dictionary row.

    Args:
        item: Check result item containing type_vars, input_is_constant,
              attrs, input_constraints, and optionally check_result
        input_constraint_types: Optional dict mapping input names to their
              constraint types ("shape", "value", "variadic"). Used to ensure
              consistent property naming when optional inputs are None.
        all_attr_names: Optional set of all attribute names across all check results.
              Used to ensure consistent property naming when attributes are omitted.
        use_qdq: when True, fix missing input_is_constant by setting to False

    Returns:
        Flat dictionary with all properties as keys
    """
    res = {}
    if "case_index" in item:
        res["case_index"] = item["case_index"]

    # properties
    if "check_result" in item:
        compile_result = item["check_result"]["compile"]["result"]
        run_result = item["check_result"]["run"]["result"]
        res["compile_run_success"] = (
            compile_result["success"],
            run_result["success"],
        )
        compile_reason = compile_result.get("reason")
        run_reason = run_result.get("reason")
        res["compile_reason"] = compile_reason
        res["run_reason"] = run_reason
        res["has_not_run_placeholder_reason"] = (
            isinstance(compile_reason, str) and compile_reason.startswith("not_run")
        ) or (isinstance(run_reason, str) and run_reason.startswith("not_run"))
    # common properties
    res.update(item["type_vars"])
    # TODO: add _dyanmic_axes and _is_fixed_shape for QDQ?
    dynamic_axes = item.get("dynamic_axes", {})

    def set_properties_for_dynamic_axes(input_name: str, is_constant: bool):
        if "__" not in input_name:  # skip variadic inputs
            axes = dynamic_axes.get(input_name, ())
            res[f"{input_name}_is_constant"] = is_constant
            res[f"{input_name}_is_fixed_shape"] = len(axes) == 0
            res[f"{input_name}_dynamic_axes"] = tuple(axes)

    if "input_is_constant" in item:
        for input_name, is_constant in item["input_is_constant"].items():
            set_properties_for_dynamic_axes(input_name, is_constant)
    for attr, value in item["attrs"].items():
        res[f"attr_{attr}"] = value
        res[f"attr_{attr}_is_none"] = value is None
    for input_name, constraint in item["input_constraints"].items():
        if input_name not in item["attrs"]:
            # Handle optional inputs that are None (not provided)
            if constraint is None:
                res[f"{input_name}_is_constant"] = True
                res[f"{input_name}_is_fixed_shape"] = True
                res[f"{input_name}_dynamic_axes"] = ()
                res[f"{input_name}_is_none"] = True
                # Use the constraint type from non-None cases to ensure consistent keys
                constraint_type = input_constraint_types[input_name]
                if constraint_type == "shape":
                    res[f"{input_name}_shape"] = None
                else:  # value or variadic
                    res[f"{input_name}_value"] = None
            elif constraint["type"] == "variadic":
                res[f"{input_name}_shape"] = tuple(
                    element["shape"] if element["type"] == "shape" else None
                    for element in constraint["elements"]
                )
                res[f"{input_name}_value"] = tuple(
                    normalize_constraint_dict(element)["value"]
                    if element["type"] == "value"
                    else None
                    for element in constraint["elements"]
                )
                if use_qdq:
                    res[f"{input_name}_is_constant"] = tuple(
                        item.get("input_is_constant", {}).get(f"{input_name}__{idx}", False)
                        for idx in range(len(constraint["elements"]))
                    )
                else:
                    res[f"{input_name}_is_constant"] = tuple(
                        item["input_is_constant"][f"{input_name}__{idx}"]
                        for idx in range(len(constraint["elements"]))
                    )
                res[f"{input_name}_is_fixed_shape"] = tuple(
                    len(dynamic_axes.get(f"{input_name}__{idx}", ())) == 0
                    for idx in range(len(constraint["elements"]))
                )
                res[f"{input_name}_dynamic_axes"] = tuple(
                    tuple(dynamic_axes.get(f"{input_name}__{idx}", ()))
                    for idx in range(len(constraint["elements"]))
                )
                res[f"{input_name}_is_none"] = False
            elif constraint["type"] == "shape":
                res[f"{input_name}_shape"] = constraint["shape"]
                res[f"{input_name}_is_none"] = False
            else:  # value
                res[f"{input_name}_value"] = normalize_constraint_dict(constraint)["value"]
                res[f"{input_name}_is_none"] = False

            if use_qdq and f"{input_name}_is_constant" not in res:
                set_properties_for_dynamic_axes(input_name, False)

    # Handle inputs that are omitted from this item's input_constraints
    # (present in other test cases but not in this one)
    if input_constraint_types:
        for input_name, constraint_type in input_constraint_types.items():
            if input_name not in item["input_constraints"] and input_name not in item["attrs"]:
                # Treat omitted input same as None constraint
                res[f"{input_name}_is_constant"] = True
                res[f"{input_name}_is_fixed_shape"] = True
                res[f"{input_name}_dynamic_axes"] = ()
                res[f"{input_name}_is_none"] = True
                if constraint_type == "shape":
                    res[f"{input_name}_shape"] = None
                elif constraint_type in ("value", "variadic"):
                    res[f"{input_name}_value"] = None

    # Handle attributes that are omitted from this item's attrs
    # (present in other test cases but not in this one)
    if all_attr_names:
        for attr_name in all_attr_names:
            if attr_name not in item["attrs"]:
                res[f"attr_{attr_name}"] = None
                res[f"attr_{attr_name}_is_none"] = True

    if "qdq_types" in item:
        for input_name, qdq_type in item["qdq_types"].items():
            res[f"QDQ_{input_name}"] = qdq_type

    # convert lists to tuples with float replacement, to make hashable and avoid comparing float
    res = {
        k: make_hashable(v, replace_float_with_dummy=replace_float_with_dummy)
        for k, v in res.items()
    }
    return res


def _format_rule_signature(group_cols: list[str], group_key: Any) -> str:
    """Build a readable signature for a conflict group."""
    if not group_cols:
        return "all_rows"
    key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
    parts = []
    for col, value in zip(group_cols, key_tuple, strict=False):
        value_repr = "NA" if pd.isna(value) else repr(value)
        parts.append(f"{col}={value_repr}")
    return ";".join(parts)


def check_df_consistent(
    df: pd.DataFrame,
    op_name: str,
    result_col: str,
    ignored_cols: list[str],
    op_version: int,
    device: str,
    ep_name: str,
    op_domain: str,
    is_qdq: bool = False,
) -> bool:
    """Check if DataFrame has consistent results for same property combinations.

    Verifies that the same combination of finite properties always produces
    the same result. If any combination has conflicting results, raises an error.

    Args:
        df: DataFrame containing test results
        result_col: Column name containing the result to check for consistency
        ignored_cols: Columns to ignore in consistency check (typically infinite properties)
        op_version: Operator version/opset to include in conflict CSV filename
        device: Device name to include in conflict CSV filename

    Returns:
        True if consistent

    Raises:
        ValueError: If conflicts are found (same properties, different results)
    """
    placeholder_col = "has_not_run_placeholder_reason"

    excluded_group_cols = set(ignored_cols)
    excluded_group_cols.add(result_col)
    if placeholder_col in df.columns:
        excluded_group_cols.add(placeholder_col)

    group_cols = [c for c in df.columns if c not in excluded_group_cols]
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]

    conflict_details: list[pd.DataFrame] = []
    rule_counter = 1
    for group_key, group_df in grouped:
        eval_df = group_df
        if placeholder_col in group_df.columns:
            eval_df = group_df[group_df[placeholder_col].isna() | (group_df[placeholder_col] == "")]

        # If all rows are placeholders, this group should not trigger conflicts.
        if eval_df.empty:
            continue

        unique_results = set(eval_df[result_col].tolist())
        if len(unique_results) > 1:
            cols_to_show = group_cols + ignored_cols + [result_col]
            cols_to_show = [c for c in cols_to_show if c in group_df.columns]
            # Add a deterministic signature so rows from the same
            # rule candidate can be grouped visually.
            conflict_df = group_df.loc[:, cols_to_show].copy()
            conflict_df.insert(0, "rule_index", rule_counter)
            conflict_df["rule_signature"] = _format_rule_signature(group_cols, group_key)
            conflict_details.append(conflict_df)
            rule_counter += 1

    if not conflict_details:
        return True

    details_data_frame = pd.concat(conflict_details, ignore_index=False)
    ordered_cols = ["rule_index"]
    if "case_index" in details_data_frame.columns:
        ordered_cols.append("case_index")
    # keep other columns except the signature, then place signature last
    ordered_cols.extend(
        [c for c in details_data_frame.columns if c not in ordered_cols and c != "rule_signature"]
    )
    ordered_cols.append("rule_signature")
    details_data_frame = details_data_frame.loc[:, ordered_cols]
    domain_str = op_domain if op_domain else "ai.onnx"
    filename_parts = [op_name, ep_name, device, domain_str, f"opset{op_version}"]
    if is_qdq:
        filename_parts.append("qdq")
    conflict_dir = Path("conflicts")
    conflict_dir.mkdir(parents=True, exist_ok=True)
    conflict_filename = conflict_dir / ("_".join(filename_parts) + "_conflicts.csv")
    details_data_frame.to_csv(conflict_filename, index=False)

    raise ValueError(
        f"Found groups with multiple {result_col} values, "
        f"consider adding more derived properties to "
        f"distinguish them, save conflicts result to "
        f"{op_name}_conflicts.csv\n\n"
    )


def np_to_python_value(value: Any) -> Any:
    """Convert numpy types to Python native types.

    Args:
        value: Value to convert (may be numpy type or Python type)

    Returns:
        Python native type equivalent
    """
    if isinstance(value, np.generic):
        return value.item()
    return value


def extract_single_negative_rules(
    df: pd.DataFrame, result_col: str, ignored_cols: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """Extract single negative rules from DataFrame.

    A negative rule identifies property values that always lead to failure.
    For each column, find values where ALL occurrences have failed results.

    Args:
        df: DataFrame containing test results
        result_col: Column name containing the result (success/failure)
        ignored_cols: Columns to ignore (typically infinite properties like shapes/values)

    Returns:
        Dictionary mapping column names to lists of failing values with counts
    """
    if result_col not in df.columns:
        raise KeyError(f"{result_col} is not a dataframe column")

    target_cols = [c for c in df.columns if c not in ignored_cols and c != result_col]
    if not target_cols:
        return {}

    n_results = df[result_col][0].__len__()  # type: ignore[attr-defined]
    assert n_results == 2
    all_negative_rules = []
    all_failed = []
    for i in range(n_results):
        results = df[result_col].apply(lambda x, _i=i: x[_i])
        all_failed.append(np_to_python_value(results.eq(False).all()))
        negative_rules = {}
        for col in target_cols:
            failing_values = []
            for value in df[col].unique():
                mask = df[col].isna() if pd.isna(value) else df[col] == value
                if mask.any() and results[mask].eq(False).all():
                    failing_values.append(
                        {"value": np_to_python_value(value), "row_count": int(mask.sum())}
                    )
            if failing_values:
                negative_rules[col] = failing_values
        all_negative_rules.append(negative_rules)

    return all_negative_rules, all_failed


def build_op_query_negative_rules_and_table(
    check_results: list[dict[str, Any]],
    input_generator: OpInputGenerator,
    use_qdq: bool,
    op_version: int,
    device: str,
    ep_name: str,
    op_domain: str,
    # schema: OpSchema,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build negative rules from check results for a specific operator.

    Args:
        check_results: List of check result items from runtime checker
        input_generator: OpInputGenerator object for the operator

    Returns:
        Tuple of (negative_rules_dict, dataframe):
        - negative_rules_dict: Dictionary containing operator name and negative rules
        - dataframe: DataFrame with all test results and properties
    """
    op_name = input_generator.op_name
    if not check_results:
        return {"op_name": op_name, "negative_rules": {}}, pd.DataFrame()

    # Convert items to rows

    # Pre-compute constraint types from non-None constraints for consistent property naming
    input_constraint_types = _get_input_constraint_types(check_results)
    # Pre-compute all attribute names for consistent property naming
    all_attr_names = _get_all_attr_names(check_results)

    def get_row(item: dict[str, Any]) -> dict[str, Any]:
        """Convert item to row with derived properties if available."""
        row = item_to_row(
            item,
            input_constraint_types,
            all_attr_names,
            input_generator.replace_float_with_dummy_in_query,
            use_qdq=use_qdq,
        )
        try:
            row = input_generator.derive_properties(row)
        except NotImplementedError:
            pass
        return row

    rows = [get_row(item) for item in check_results]

    # Create DataFrame and replace NaN with None
    df = pd.DataFrame(rows, dtype=object)
    df = df.replace({np.nan: None})

    # Auto-detect infinite properties (those ending with _shape or _value)
    # These represent unbounded input spaces that should not be used for negative rules
    infinite_properties = input_generator.get_infinite_property_names()
    internal_reason_cols = [
        "compile_reason",
        "run_reason",
        "has_not_run_placeholder_reason",
        "case_index",
    ]
    consistency_ignored = [*infinite_properties, *internal_reason_cols]
    assert check_df_consistent(
        df,
        op_name,
        "compile_run_success",
        consistency_ignored,
        op_version=op_version,
        device=device,
        ep_name=ep_name,
        op_domain=op_domain,
        is_qdq=use_qdq,
    )

    # Internal reason columns are only for consistency filtering and must not be
    # exported to tables/rules, otherwise downstream matcher treats them as
    # required condition keys.
    export_df = df.drop(columns=internal_reason_cols, errors="ignore")

    negative_rules, all_failed = extract_single_negative_rules(
        export_df, "compile_run_success", infinite_properties
    )
    names = ["compile", "run"]

    negative_rules_dict = {
        "op_name": op_name,
        "negative_rules": dict(zip(names, negative_rules, strict=False)),
        "all_failed": dict(zip(names, all_failed, strict=False)),
        "total_row_count": len(export_df),
    }

    return negative_rules_dict, export_df


def _parse_filename(filename: str) -> tuple[str, str, str, str, int, bool]:
    """Parse operator name, EP name, domain, opset, and QDQ flag from filename.

    Expected filename format:
    - <op_name>_<ep_name>_<device>_<domain>_opset<number>[_qdq].json

    Args:
        filename: Name of the JSON file (without extension)

    Returns:
        Tuple of (op_domain, op_name, ep_name, device, opset_version, is_qdq)
    """
    import re

    # Extract opset number from filename (e.g., "opset17" or "opset17_qdq")
    opset_match = re.search(r"_opset(\d+)(?:_qdq)?$", filename)
    assert opset_match is not None, f"Could not extract opset from filename: {filename}"

    is_qdq = filename.endswith("_qdq")
    opset_version = int(opset_match.group(1))
    # Remove the opset suffix to get the rest
    filename_without_opset = filename[: opset_match.start()]
    parts = filename_without_opset.split("_")

    assert len(parts) == 4, (
        f"Filename must have op_name, ep_name, device, and domain parts: {filename}"
    )

    # Format: op_name_ep_name_device_domain
    # First part is op_name, second is ep_name, third is device, last is domain
    op_name = parts[0]
    ep_name = parts[1]
    device = parts[2]
    op_domain = parts[3]

    # Normalize domain name: ai.onnx -> empty string (for consistency with ONNX standard)
    if op_domain == "ai.onnx":
        op_domain = ""

    return op_domain, op_name, ep_name, device, opset_version, is_qdq


def get_opset_version_range(op_name: str, start_opset_version: int, op_domain: str) -> list[int]:
    """Get the range of opset versions that use the same op schema version.

    Given an op_name and a starting opset version, determines all consecutive opset
    versions that use the same since_version of the operator. This is useful when
    updating rules: e.g., if Slice has versions 1, 10, 11, 13, and start_opset_version=11,
    the since_version is 11 and the next version is 13, so we return [11, 12].

    Args:
        op_name: Name of the ONNX operator (e.g., "Slice")
        start_opset_version: The starting opset version
        op_domain: The domain of the operator (empty string for ai.onnx)

    Returns:
        List of consecutive opset versions sharing the same op schema version
    """
    max_opset = onnx_opset_version()
    base_since = get_op_since_version(op_name, start_opset_version, op_domain)

    versions = []
    for v in range(start_opset_version, max_opset + 1):
        try:
            since = get_op_since_version(op_name, v, op_domain)
        except SchemaError:
            break
        if since == base_since:
            versions.append(v)
        else:
            break

    return versions


if __name__ == "__main__":
    import argparse
    import sys
    import traceback

    parser = argparse.ArgumentParser(
        description="Process runtime checker results and generate negative rules"
    )
    parser.add_argument("input_dir", type=str, help="Input directory containing JSON result files")
    parser.add_argument(
        "--opset_version", type=int, required=True, help="Opset version for the ONNX operators"
    )
    parser.add_argument(
        "--opset_domain",
        type=str,
        required=True,
        help="Opset domain for the ONNX operators (e.g., 'ai.onnx', 'com.microsoft')",
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for negative rules (defaults to input_dir)"
    )
    parser.add_argument(
        "-uz",
        "--update-zip",
        action="store_true",
        help="Zip rule files and update the zip rules files in modelkit/analyze/rules",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="When updating zip, keep existing files not in the new output; "
        "if a file exists in both, merge JSON dicts "
        "(new values override old keys, old-only keys are preserved) and sort.",
    )
    parser.add_argument(
        "--rules-dir",
        type=str,
        default=None,
        help="Directory where rule zip files are written when --update-zip is set. "
        "Defaults to the runtime_check_rules folder relative to this script "
        "(../rules/runtime_check_rules).",
    )
    parser.add_argument(
        "-range",
        "--opset_range_ref_op",
        type=str,
        default=None,
        help="Reference operator name or end opset version number. "
        "When a number N is provided, processes all opset versions in "
        "[--opset_version, N] (inclusive). "
        "When an operator name is provided, computes the range of opset versions "
        "that share the same since_version for this op, starting from --opset_version. "
        "Example: --opset_range_ref_op 12 --opset_version 11 processes versions 11-12. "
        "Example: --opset_range_ref_op Slice --opset_version 11 processes versions 11-12 "
        "since Slice has since_versions 1, 10, 11, 13.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize the opset_domain (ai.onnx -> empty string for ONNX standard)
    target_domain = "" if args.opset_domain == "ai.onnx" else args.opset_domain
    domain_str_for_filename = args.opset_domain  # Keep original for filename matching

    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        exit(1)

    # Extract unique (op_name, ep_name, device, is_qdq)
    # combinations from filenames for the target domain
    # Filename format: <op_name>_<ep_name>_<device>_<domain>_opset<N>[_qdq].json
    import re

    op_info_set: set[tuple[str, str, str, bool]] = set()
    for json_file in json_files:
        is_qdq = json_file.stem.endswith("_qdq")
        # Remove opset suffix to get base info
        opset_match = re.search(r"_opset(\d+)(?:_qdq)?$", json_file.stem)
        if opset_match:
            filename_without_opset = json_file.stem[: opset_match.start()]
            parts = filename_without_opset.split("_")
            if len(parts) == 4:
                op_name, ep_name, device, file_domain = parts[:4]
                # Only include operators from the target domain
                if file_domain == domain_str_for_filename:
                    op_info_set.add((op_name, ep_name, device, is_qdq))

    print(f"Found {len(op_info_set)} unique operators to process for domain '{args.opset_domain}'")

    # Determine which opset versions to process
    if args.opset_range_ref_op:
        if args.opset_range_ref_op.isdigit():
            end_opset = int(args.opset_range_ref_op)
            opset_versions_to_process = list(range(args.opset_version, end_opset + 1))
            print(f"Numeric range: will process opset versions {opset_versions_to_process}")
        else:
            opset_versions_to_process = get_opset_version_range(
                args.opset_range_ref_op, args.opset_version, target_domain
            )
            print(
                f"Reference op '{args.opset_range_ref_op}' "
                f"with opset_version {args.opset_version}: "
                f"will process opset versions {opset_versions_to_process}"
            )
    else:
        opset_versions_to_process = [args.opset_version]

    qdq_generator = None
    if any(is_qdq for _, _, _, is_qdq in op_info_set):
        from ...pattern.op_input_gen.qdq_gen import QDQGenerator

        qdq_generator = QDQGenerator(1, ONNXDomain.COM_MICROSOFT)

    for current_opset_version in opset_versions_to_process:
        if len(opset_versions_to_process) > 1:
            print(f"\n{'=' * 60}")
            print(f"Processing opset version {current_opset_version}")
            print(f"{'=' * 60}")

        # Group results by (EP, device, domain, opset, is_qdq)
        results_by_ep_domain_opset: dict[tuple[str, str, str, int, bool], dict[str, Any]] = {}
        tables_by_ep_domain_opset: dict[tuple[str, str, str, int, bool], dict[str, Any]] = {}
        table_columns_by_ep_domain_opset: dict[
            tuple[str, str, str, int, bool], dict[str, list[str]]
        ] = {}

        for op_name, ep_name, device, is_qdq in sorted(op_info_set):
            # Get the since_version for this operator based on
            # the current opset_version. Handle Op and Pattern.
            # TODO: build a since_version list for
            # PatternSchemas based on since_version of
            # included ops
            try:
                since_version = get_op_since_version(op_name, current_opset_version, target_domain)
            except SchemaError:
                since_version = current_opset_version

            # Build the expected filename with since_version
            qdq_suffix = "_qdq" if is_qdq else ""
            expected_filename = (
                f"{op_name}_{ep_name}_{device}"
                f"_{domain_str_for_filename}"
                f"_opset{since_version}{qdq_suffix}.json"
            )
            json_file = input_dir / expected_filename
            print(f"Processing {expected_filename}...", end=" ")

            if not json_file.exists():
                print(f"{Fore.YELLOW}SKIPPED: File not found. {Style.RESET_ALL}")
                continue

            if json_file.stat().st_size == 0:
                print(f"{Fore.YELLOW}SKIPPED: Empty JSON file. {Style.RESET_ALL}")
                continue

            try:
                with open(json_file, encoding="utf-8") as f:  # noqa: PTH123
                    data = json.load(f)

                op_domain, op_name, ep_name, device, opset_version, is_qdq = _parse_filename(
                    json_file.stem
                )

                check_results = data.get("check_results", [])

                if not check_results:
                    print(f"{Fore.RED}Error: No check_results found, skipping{Style.RESET_ALL}")
                    continue

                # Build negative rules and get DataFrame
                domain = ONNXDomain.from_str(op_domain)
                try:
                    schema = domain.get_op_schema(op_name, opset_version)
                    input_generator = get_runtime_checker_op(op_name)(
                        schema, qdq_generator=qdq_generator if is_qdq else None
                    )
                except SchemaError:
                    # pattern case
                    # TODO: if a pattern depends on multiple
                    # domains, the filename currently contains
                    # only AI_ONNX; need to recover all domains
                    domain_versions = {
                        op_domain: opset_version,
                        ONNXDomain.COM_MICROSOFT: 1,  # safeguard
                    }
                    input_generator = get_pattern_input_generator(op_name)(domain_versions)

                op_negative_rules, df = build_op_query_negative_rules_and_table(
                    check_results,
                    input_generator,
                    use_qdq=is_qdq,
                    op_version=opset_version,
                    device=device,
                    ep_name=ep_name,
                    op_domain=op_domain,
                )

                # Group by (EP, domain, current opset_version, is_qdq)
                key = (ep_name, device, target_domain, current_opset_version, is_qdq)
                if key not in results_by_ep_domain_opset:
                    results_by_ep_domain_opset[key] = {}
                    tables_by_ep_domain_opset[key] = {}
                    table_columns_by_ep_domain_opset[key] = {}

                results_by_ep_domain_opset[key][op_name] = op_negative_rules

                # Convert DataFrame to JSON-serializable format
                tables_by_ep_domain_opset[key][op_name] = df.to_dict()
                table_columns_by_ep_domain_opset[key][op_name] = [
                    col_name
                    for col_name in df.columns.to_list()
                    if col_name != "compile_run_success"
                ]

                print(f"OK ({len(check_results)} results)")

            except Exception as e:
                print(f"{Fore.RED}ERROR: {e}{Style.RESET_ALL}")
                traceback.print_exc()
                sys.exit(1)

        zip_group = {}
        # Save negative rules
        for (
            ep_name,
            device,
            op_domain,
            opset_version,
            is_qdq,
        ), op_results in results_by_ep_domain_opset.items():
            # Create domain-specific filename
            domain_str = op_domain if op_domain else "ai.onnx"
            qdq_suffix = "_qdq" if is_qdq else ""
            output_file = output_dir / (
                f"{ep_name}_{device}_{domain_str}"
                f"_opset{opset_version}"
                f"_negative_rules{qdq_suffix}.json"
            )

            with open(output_file, "w", encoding="utf-8", newline="\n") as f:  # noqa: PTH123
                json.dump(dict(sorted(op_results.items())), f, indent=2)

            print(f"\nSaved {len(op_results)} operators to {output_file}")
            zip_group.setdefault(f"{ep_name}_{device}", []).append(output_file)

        # Save tables
        for (
            ep_name,
            device,
            op_domain,
            opset_version,
            is_qdq,
        ), op_tables in tables_by_ep_domain_opset.items():
            # Create domain-specific filename
            domain_str = op_domain if op_domain else "ai.onnx"
            qdq_suffix = "_qdq" if is_qdq else ""
            output_file = (
                output_dir
                / f"{ep_name}_{device}_{domain_str}_opset{opset_version}_tables{qdq_suffix}.json"
            )

            with open(output_file, "w", encoding="utf-8", newline="\n") as f:  # noqa: PTH123
                json.dump(dict(sorted(op_tables.items())), f, indent=2)

            print(f"Saved {len(op_tables)} operator tables to {output_file}")
            zip_group.setdefault(f"{ep_name}_{device}", []).append(output_file)

        # Save table column names
        for (
            ep_name,
            device,
            op_domain,
            opset_version,
            is_qdq,
        ), op_columns in table_columns_by_ep_domain_opset.items():
            domain_str = op_domain if op_domain else "ai.onnx"
            qdq_suffix = "_qdq" if is_qdq else ""
            output_file = output_dir / (
                f"{ep_name}_{device}_{domain_str}"
                f"_opset{opset_version}_table_columns{qdq_suffix}.json"
            )

            with open(output_file, "w", encoding="utf-8", newline="\n") as f:  # noqa: PTH123
                json.dump(dict(sorted(op_columns.items())), f, indent=2)

            print(f"Saved {len(op_columns)} operator table column sets to {output_file}")
            zip_group.setdefault(f"{ep_name}_{device}", []).append(output_file)

        print(
            f"\nProcessing complete! Generated "
            f"{len(results_by_ep_domain_opset)} "
            f"negative rule file(s) "
            f"and {len(tables_by_ep_domain_opset)} table file(s), "
            f"plus {len(table_columns_by_ep_domain_opset)} table-column file(s)."
        )

        if args.update_zip:
            rules_dir = (
                Path(args.rules_dir) if args.rules_dir else get_runtime_rules_search_dirs()[0]
            )
            for group_name, file_list in zip_group.items():
                rule_zip_path = (
                    rules_dir
                    / f"{group_name}_{domain_str_for_filename}_opset{current_opset_version}.zip"
                )

                # In append mode, load existing zip entries to preserve files not being updated
                existing_content: dict[str, bytes] = {}
                if args.append and rule_zip_path.exists():
                    with zipfile.ZipFile(rule_zip_path, mode="r") as existing_zf:
                        for name in existing_zf.namelist():
                            existing_content[name] = existing_zf.read(name)

                new_arcnames = {Path(f).name for f in file_list}

                with zipfile.ZipFile(
                    rule_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED
                ) as rule_zf:
                    # Keep existing entries not covered by the new output
                    for name, data in existing_content.items():
                        if name not in new_arcnames:
                            rule_zf.writestr(name, data)

                    for filename in file_list:
                        arcname = Path(filename).name
                        if args.append and arcname in existing_content:
                            # Merge: old dict updated with new dict, then sort
                            old_dict = json.loads(existing_content[arcname])
                            with open(filename, encoding="utf-8") as f:  # noqa: PTH123
                                new_dict = json.load(f)
                            merged = dict(sorted({**old_dict, **new_dict}.items()))
                            rule_zf.writestr(arcname, json.dumps(merged, indent=2))
                        else:
                            rule_zf.write(filename, arcname=arcname)

                print(
                    f"Rule zip file {group_name}"
                    f"_{domain_str_for_filename}"
                    f"_opset{current_opset_version}.zip "
                    f"updated with {len(file_list)} files."
                )
