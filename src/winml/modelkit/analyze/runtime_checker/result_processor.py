# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from onnx.defs import SchemaError, onnx_opset_version

from ...onnx import ONNXDomain
from ...pattern.base import get_pattern_input_generator
from ...pattern.op_input_gen import (
    OpInputGenerator,
    get_runtime_checker_op,
    normalize_constraint_dict,
)
from ..utils.model_utils import (
    encode_rule_condition_value_for_parquet,
    get_op_since_version,
    make_hashable,
)
from ..utils.rule_loader import get_runtime_rules_search_dirs


# Snapshot metadata keys used in generated rule artifacts.
SNAPSHOT_TYPE_KEY = "__snapshot_type__"
SNAPSHOT_TYPE_DELTA = "delta_v1"
SNAPSHOT_BASE_OPSET_KEY = "__base_opset__"
SNAPSHOT_CURRENT_OPSET_KEY = "__current_opset__"
SNAPSHOT_CHANGED_KEY = "__changed__"
SNAPSHOT_DELETED_KEY = "__deleted__"


def _sorted_dict_by_key(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow key-sorted dict for stable JSON output."""
    return dict(sorted(payload.items()))


def _build_snapshot_payload(
    current_payload: dict[str, Any],
    current_opset: int,
    previous_payload: dict[str, Any] | None,
    previous_opset: int | None,
) -> dict[str, Any]:
    """Build either a full snapshot (first version) or a delta snapshot.

    Full snapshots keep backward compatibility with existing plain-dict format.
    Delta snapshots store only changed/deleted operators relative to the previous opset.
    """
    if previous_payload is None or previous_opset is None:
        return _sorted_dict_by_key(current_payload)

    changed = {
        op_name: value
        for op_name, value in current_payload.items()
        if op_name not in previous_payload or previous_payload[op_name] != value
    }
    deleted = sorted(op_name for op_name in previous_payload if op_name not in current_payload)

    return {
        SNAPSHOT_TYPE_KEY: SNAPSHOT_TYPE_DELTA,
        SNAPSHOT_BASE_OPSET_KEY: previous_opset,
        SNAPSHOT_CURRENT_OPSET_KEY: current_opset,
        SNAPSHOT_CHANGED_KEY: _sorted_dict_by_key(changed),
        SNAPSHOT_DELETED_KEY: deleted,
    }


def _is_delta_snapshot_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get(SNAPSHOT_TYPE_KEY) == SNAPSHOT_TYPE_DELTA


def _can_append_merge(existing_payload: Any, new_payload: Any) -> bool:
    """Whether append-mode shallow dict merge is safe for these payloads."""
    return (
        isinstance(existing_payload, dict)
        and isinstance(new_payload, dict)
        and not _is_delta_snapshot_payload(existing_payload)
        and not _is_delta_snapshot_payload(new_payload)
    )


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


def _parse_requested_domains(domains_arg: str) -> list[str]:
    """Parse and validate --domains values."""
    requested_domains = [part.strip() for part in domains_arg.split(",") if part.strip()]
    if not requested_domains:
        requested_domains = ["ai.onnx", "com.microsoft"]

    domains_to_process: list[str] = []
    for requested in requested_domains:
        normalized = requested.lower()
        if normalized == "ai.onnx":
            mapped_domain = "ai.onnx"
        elif normalized == "com.microsoft":
            mapped_domain = "com.microsoft"
        else:
            print(
                f"Ignoring unsupported domain '{requested}'. "
                "Supported values: ai.onnx, com.microsoft"
            )
            continue

        if mapped_domain not in domains_to_process:
            domains_to_process.append(mapped_domain)

    return domains_to_process


def _build_op_rule_dataframe(
    check_results: list[dict[str, Any]],
    input_generator: OpInputGenerator,
    use_qdq: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Build per-case rule rows for one op file.

    Returns:
        tuple[pd.DataFrame, list[str]]:
            - DataFrame with per-case properties and compile/run outputs
            - Infinite property names that should be ignored for matching
    """
    input_constraint_types = _get_input_constraint_types(check_results)
    all_attr_names = _get_all_attr_names(check_results)

    def get_row(item: dict[str, Any]) -> dict[str, Any]:
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
            # Some OpInputGenerator implementations do not provide derived
            # properties; keep the base row unchanged in that case.
            row = row
        return row

    rows = [get_row(item) for item in check_results]
    df = pd.DataFrame(rows, dtype=object).replace({np.nan: None})
    infinite_properties = input_generator.get_infinite_property_names()

    return df, infinite_properties


def _deduplicate_rule_rows(
    df: pd.DataFrame,
    condition_cols: list[str],
    output_cols: list[str],
    compare_output_cols: list[str] | None = None,
    case_index_col: str = "case_index",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Deduplicate rows by condition columns and detect conflicts.

    A conflict means the same condition set maps to multiple values
    in compare_output_cols.

    When compare_output_cols is None, output_cols is used for both
    conflict detection and conflict display.
    """
    if df.empty:
        return df.copy(), None

    compare_cols = compare_output_cols if compare_output_cols is not None else output_cols
    if not compare_cols:
        raise ValueError("compare_output_cols cannot be empty")

    dedup_rows: list[pd.Series] = []
    conflict_parts: list[pd.DataFrame] = []
    conflict_group_id = 0

    grouped: Any
    if condition_cols:
        grouped = df.groupby(condition_cols, dropna=False, sort=False)
    else:
        grouped = [((), df)]

    for _, group_df in grouped:
        unique_outputs = group_df.loc[:, compare_cols].drop_duplicates(ignore_index=True)
        if len(unique_outputs) > 1:
            conflict_group_id += 1

            cols_to_show = [case_index_col, *condition_cols, *output_cols]
            cols_to_show = [c for c in cols_to_show if c in group_df.columns]
            conflict_group_df = group_df.loc[:, cols_to_show].copy()

            if case_index_col not in conflict_group_df.columns:
                conflict_group_df.insert(0, case_index_col, list(group_df.index))

            conflict_group_df.insert(0, "groupid", conflict_group_id)
            conflict_parts.append(conflict_group_df)
            continue

        row = group_df.iloc[0].copy()
        row["rule_row_count"] = int(len(group_df))
        dedup_rows.append(row)

    dedup_df = pd.DataFrame(dedup_rows, dtype=object)
    conflict_df = pd.concat(conflict_parts, ignore_index=True) if conflict_parts else None
    return dedup_df, conflict_df


def _json_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert dataframe records to JSON-safe Python objects."""
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records", force_ascii=False))


def _encode_condition_columns_for_parquet(df: pd.DataFrame, condition_cols: list[str]) -> pd.DataFrame:
    """Return a parquet-write dataframe with condition columns encoded as strings."""
    if df.empty or not condition_cols:
        return df.copy()

    encoded_df = df.copy()
    for col in condition_cols:
        if col in encoded_df.columns:
            raw = encoded_df[col].to_numpy()
            encoded_df[col] = [encode_rule_condition_value_for_parquet(v) for v in raw]
    return encoded_df


if __name__ == "__main__":
    import argparse
    import sys
    import traceback

    parser = argparse.ArgumentParser(
        description=(
            "Process runtime checker per-op result files and generate "
            "deduplicated per-op rule artifacts."
        )
    )
    parser.add_argument("input_dir", type=str, help="Input directory containing JSON result files")
    parser.add_argument(
        "--output-dir",
        type=str,
        help=(
            "Output directory for per-op JSON rule files "
            "(defaults to input_dir)."
        ),
    )
    parser.add_argument(
        "--rules-dir",
        type=str,
        default=None,
        help=(
            "Directory where per-op parquet rule files are written. "
            "Defaults to first entry from rule search dirs."
        ),
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="ai.onnx,com.microsoft",
        help="Comma-separated domains to process: ai.onnx,com.microsoft",
    )
    parser.add_argument(
        "--stop-on-conflict",
        action="store_true",
        help=(
            "Stop immediately when the first per-op conflict is detected. "
            "Useful for auto-resolve workflows that process one conflict file per round."
        ),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        exit(1)

    domains_to_process = _parse_requested_domains(args.domains)
    if not domains_to_process:
        print("No valid domains selected to process.")
        exit(1)

    selected_domain_set = set(domains_to_process)
    parquet_dir = Path(args.rules_dir) if args.rules_dir else get_runtime_rules_search_dirs()[0]
    parquet_dir.mkdir(parents=True, exist_ok=True)

    qdq_generator = None
    processed = 0
    skipped = 0
    conflict_skipped = 0

    output_cols = ["compile_run_success"]
    compare_output_cols = ["compile_run_success"]
    internal_cols = ["has_not_run_placeholder_reason", "compile_reason", "run_reason"]

    for json_file in sorted(json_files):
        try:
            op_domain, op_name, ep_name, device, opset_version, is_qdq = _parse_filename(
                json_file.stem
            )
        except AssertionError:
            skipped += 1
            continue

        domain_str = op_domain if op_domain else "ai.onnx"
        if domain_str not in selected_domain_set:
            skipped += 1
            continue

        if json_file.stat().st_size == 0:
            print(f"SKIPPED empty file: {json_file.name}")
            skipped += 1
            continue

        output_json = output_dir / json_file.name
        parquet_file = parquet_dir / f"{json_file.stem}.parquet"

        print(f"Processing {json_file.name} ...", end=" ")

        try:
            with open(json_file, encoding="utf-8") as f:  # noqa: PTH123
                data = json.load(f)

            check_results = data.get("check_results", [])
            if not check_results:
                print("SKIPPED (no check_results)")
                skipped += 1
                continue

            domain = ONNXDomain.from_str(op_domain)
            if is_qdq and qdq_generator is None:
                from ...pattern.op_input_gen.qdq_gen import QDQGenerator

                qdq_generator = QDQGenerator(1, ONNXDomain.COM_MICROSOFT)

            try:
                schema = domain.get_op_schema(op_name, opset_version)
                input_generator = get_runtime_checker_op(op_name, domain=op_domain)(
                    schema, qdq_generator=qdq_generator if is_qdq else None
                )
            except SchemaError:
                domain_versions = {
                    op_domain: opset_version,
                    ONNXDomain.COM_MICROSOFT: 1,
                }
                input_generator = get_pattern_input_generator(op_name)(domain_versions)

            rule_df, infinite_properties = _build_op_rule_dataframe(
                check_results, input_generator, use_qdq=is_qdq
            )
            rule_df = rule_df.drop(columns=internal_cols, errors="ignore")

            condition_cols = [
                c
                for c in rule_df.columns
                if c not in output_cols and c not in infinite_properties and c != "case_index"
            ]

            dedup_df, conflict_df = _deduplicate_rule_rows(
                rule_df,
                condition_cols,
                output_cols,
                compare_output_cols=compare_output_cols,
            )

            if conflict_df is not None and not conflict_df.empty:
                # Conflict is detected before any rule artifacts are written for this file,
                # so skip only this op file and continue processing remaining files.
                conflict_dir = output_dir / "conflicts"
                conflict_dir.mkdir(parents=True, exist_ok=True)
                conflict_file = conflict_dir / f"{json_file.stem}_conflicts.csv"
                conflict_df.to_csv(conflict_file, index=False)
                conflict_skipped += 1
                skipped += 1
                print(f"CONFLICT (saved: {conflict_file})")

                if args.stop_on_conflict:
                    sys.exit(2)

                continue

            parquet_df = dedup_df.loc[:, [*condition_cols, *output_cols]].copy()

            parquet_write_df = _encode_condition_columns_for_parquet(
                parquet_df,
                condition_cols,
            )

            json_payload = {
                "format": "per_op_rules_v1",
                "op_name": op_name,
                "ep_name": ep_name,
                "device": device,
                "domain": domain_str,
                "opset_version": opset_version,
                "is_qdq": is_qdq,
                "condition_columns": condition_cols,
                "rows": _json_safe_records(parquet_df),
            }

            with open(output_json, "w", encoding="utf-8", newline="\n") as f:  # noqa: PTH123
                json.dump(json_payload, f, indent=2, ensure_ascii=False)

            try:
                parquet_write_df.to_parquet(parquet_file, index=False, compression="snappy")
            except Exception as e:
                raise RuntimeError(
                    "Failed to write parquet file. Ensure a parquet engine "
                    "(for example pyarrow) is installed."
                ) from e

            processed += 1
            print(f"OK ({len(check_results)} cases -> {len(parquet_df)} dedup rules)")

        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            sys.exit(1)

    print(
        "Completed per-op rule generation. "
        f"processed={processed}, skipped={skipped}, conflict_skipped={conflict_skipped}, "
        f"output_dir={output_dir}"
    )
    print(f"Parquet rule files written to: {parquet_dir}")

    if conflict_skipped > 0:
        # Exit with a dedicated code so batch scripts can treat conflicts differently
        # from hard failures while still detecting that conflicts occurred.
        sys.exit(2)
