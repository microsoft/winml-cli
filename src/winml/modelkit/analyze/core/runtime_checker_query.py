# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""RuntimeCheckerQuery - Query runtime database for pattern support."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import numpy_helper, shape_inference

from ...onnx import (
    ONNXDomain,
    SupportedONNXType,
    infer_onnx_shapes,
    remove_optional_from_type_annotation,
)
from ...pattern.base import (
    get_pattern_input_generator,
    get_registered_pattern_input_generators,
)
from ...pattern.match import PatternMatchResult
from ...pattern.op_input_gen import (
    get_runtime_checker_op,
)
from ..exceptions import (
    OpLackOfRequiredInformationError,
    OpOptionalInputSupportError,
    OpUnsupportedError,
)
from ..models.runtime_checks import NodeTag, PatternAlternative, PatternRuntime, RuntimeTestResult
from ..runtime_checker.ep_checker import EPChecker
from ..runtime_checker.runner import ResilientRunner
from ..utils.model_utils import (
    collect_initializers,
    collect_valueinfo_dict,
    dtype_from_tensorproto_enum,
    get_attribute_proto_value,
    get_op_input_properties,
    make_hashable,
    node_to_pattern_match,
    shape_and_dtype_from_valueinfo,
)
from ..utils.table_utils import build_table_df
from .node_checkers.base import NodeChecker
from .node_checkers.registry import NodeCheckerRegistry


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from winml.modelkit.pattern.match import PatternMatchResult

    from .node_checkers.base import NodeChecker

# Centralized key for attaching debug details to error/info payloads
EG_RULE_DEBUG_DETAILS_KEY = "__debug_details"
EG_RULE_ERROR_KEY = "__error"


class _PseudoNode:
    """Lightweight stand-in for onnx.NodeProto used only for logging in _check_negative_rules."""

    __slots__ = ("name", "op_type")

    def __init__(self, op_type: str) -> None:
        self.op_type = op_type
        self.name = op_type


def _make_pseudo_node(pattern_name: str) -> _PseudoNode:
    """Create a pseudo-node for pattern-level negative rule checks."""
    return _PseudoNode(pattern_name)


def query_table_exact_match(df: pd.DataFrame, query: dict[str, Any]) -> pd.DataFrame:
    """Query DataFrame with exact match.

    Args:
        df: DataFrame to query
        query: Dictionary of column -> value to match

    Returns:
        Filtered DataFrame with matching rows
    """
    mask = pd.Series(True, index=df.index)
    for col, value in query.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in dataframe.")
        if value is None:
            mask &= df[col].isna()
        else:
            mask &= df[col] == value
    return df.loc[mask]


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize DataFrame for consistent querying.

    Apply make_hashable to convert lists/dicts to tuples.
    """
    for col in df.columns:
        # Make values hashable (lists -> tuples, floats -> DUMMY_FLOAT)
        df[col] = df[col].apply(make_hashable)
    return df


class LazyDomainTables:
    """Lazy-loading wrapper that reads the table file from a zip on first operator access."""

    def __init__(self, zip_path: Path, file_name: str) -> None:
        """Initialize with zip path and file name — no I/O performed.

        Args:
            zip_path: Path to the zip archive containing the table file
            file_name: Name of the JSON table file inside the zip
        """
        self._zip_path = zip_path
        self._file_name = file_name
        self._raw_data: dict[str, Any] = {}
        self._loaded_tables: dict[str, pd.DataFrame] = {}
        self._loaded: bool = False

    def _ensure_loaded(self) -> None:
        """Load the table file from the zip archive on first access."""
        if self._loaded:
            return
        self._loaded = True
        import zipfile

        if not self._zip_path.exists():
            logger.warning(
                "Rule zip not found: %s. "
                "Run 'uv run python scripts/download_rules.py' to download rule files.",
                self._zip_path,
            )
            return
        try:
            with zipfile.ZipFile(self._zip_path, "r") as zf:
                if self._file_name in zf.namelist():
                    self._raw_data = json.loads(zf.read(self._file_name).decode("utf-8"))
                else:
                    logger.debug(f"Table file not found in zip: {self._file_name}")
        except Exception as e:
            logger.debug(f"Failed to load table file {self._file_name}: {e}")

    def __getitem__(self, key: str) -> pd.DataFrame:
        """Get table for operator, loading from zip if needed."""
        if key not in self._loaded_tables:
            self._ensure_loaded()
            if key not in self._raw_data:
                raise KeyError(f"Operator '{key}' not found in tables")
            self._loaded_tables[key] = _sanitize_df(build_table_df(self._raw_data[key]))
            del self._raw_data[key]
        return self._loaded_tables[key]

    def __contains__(self, key: str) -> bool:
        """Check if operator exists in tables."""
        if key in self._loaded_tables:
            return True
        self._ensure_loaded()
        return key in self._raw_data

    def get(self, key: str, default: pd.DataFrame | None = None) -> pd.DataFrame | None:
        """Get table for operator with default fallback."""
        try:
            return self[key]
        except KeyError:
            return default


class _LazyNegRules(dict):  # type: ignore[type-arg]
    """dict[str, Any] subclass that loads negative rules from a zip on first access.

    Filters the loaded JSON to either operator rules or pattern rules.
    When set_error_on_missing=True and the zip/file is missing, the dict is
    populated with EG_RULE_ERROR_KEY / EG_RULE_DEBUG_DETAILS_KEY entries so
    callers can detect and report the missing-rules condition.
    """

    def __init__(
        self,
        zip_path: Path,
        rule_file: str,
        registered_patterns: set[str],
        patterns_only: bool = False,
        set_error_on_missing: bool = False,
    ) -> None:
        super().__init__()
        self._zip_path = zip_path
        self._rule_file = rule_file
        self._registered_patterns = registered_patterns
        self._patterns_only = patterns_only
        self._set_error_on_missing = set_error_on_missing
        self._loaded: bool = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._load()

    def _load(self) -> None:
        import zipfile

        if not self._zip_path.exists():
            if self._set_error_on_missing:
                self[EG_RULE_ERROR_KEY] = "rules_zip_not_found"
                self[EG_RULE_DEBUG_DETAILS_KEY] = str(self._zip_path)
                logger.warning(
                    "Rule zip file not found: %s. "
                    "Run 'uv run python scripts/download_rules.py' to download rule files.",
                    self._zip_path,
                )
            return

        with zipfile.ZipFile(self._zip_path, "r") as zf:
            if self._rule_file not in zf.namelist():
                if self._set_error_on_missing:
                    self[EG_RULE_ERROR_KEY] = "negative_rule_file_not_found"
                    self[EG_RULE_DEBUG_DETAILS_KEY] = str(self._rule_file)
                    logger.warning(f"Negative rule file not found: {self._rule_file}")
                else:
                    logger.debug(f"Negative rule file not found: {self._rule_file}")
                return

            raw: dict[str, Any] = json.loads(zf.read(self._rule_file).decode("utf-8"))

        filtered: dict[str, Any] = {
            key: value
            for key, value in raw.items()
            if (key in self._registered_patterns or "Pattern" in key) == self._patterns_only
        }
        self.update(_sanitize_domain_neg_rules(filtered))

        if self._patterns_only and filtered:
            logger.info(
                f"Loaded {len(filtered)} pattern rules from {self._rule_file}: "
                f"{list(filtered.keys())}"
            )

    def __getitem__(self, key: str) -> Any:
        self._ensure_loaded()
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self._ensure_loaded()
        return super().__contains__(key)

    def __iter__(self):
        self._ensure_loaded()
        return super().__iter__()

    def __len__(self) -> int:
        self._ensure_loaded()
        return super().__len__()

    def get(self, key: Any, default: Any = None) -> Any:  # type: ignore[override]
        self._ensure_loaded()
        return super().get(key, default)

    def keys(self):  # type: ignore[override]
        self._ensure_loaded()
        return super().keys()

    def values(self):  # type: ignore[override]
        self._ensure_loaded()
        return super().values()

    def items(self):  # type: ignore[override]
        self._ensure_loaded()
        return super().items()


def _sanitize_domain_neg_rules(neg_rules: dict[str, Any]) -> dict[str, Any]:
    """Sanitize negative rules by applying _make_hashable to invalid values."""
    for op_rules in neg_rules.values():
        for rule_type in ["compile", "run"]:
            for value_list in op_rules["negative_rules"][rule_type].values():
                for value_dict in value_list:
                    value_dict["value"] = make_hashable(value_dict["value"])
    return neg_rules


def _format_list_preview(items: Any, max_items: int = 10) -> list[Any]:
    """Return a preview list with at most `max_items` elements, appending '...more...' if truncated.

    Args:
        items: Iterable or sequence to preview.
        max_items: Maximum number of items to display before adding '...more...'.

    Returns:
        A list containing up to `max_items` items from `items`. If the original
        collection has more than `max_items`, the last element will be '...' to
        indicate more items exist.
    """
    try:
        lst = list(items)
    except TypeError:
        # If not iterable, just wrap as single-element list
        lst = [items]

    if len(lst) > max_items:
        return [*lst[:max_items], "...more..."]
    return lst


def _normalize_table_zip_path(path_like: str | Path) -> str:
    """Normalize table zip path for debug output.

    - Resolve '..' segments when possible.
    - If the path includes a workspace folder marker (e.g., ModelKit),
      return a path starting from that marker.
    """
    p = Path(path_like)
    try:
        p = p.resolve(strict=False)
    except Exception:
        pass

    parts = list(p.parts)
    for marker in ("ModelKit",):
        if marker in parts:
            idx = parts.index(marker)
            return "\\".join(parts[idx:])

    return str(p)


def _build_rules_not_found_debug_details(
    domain_rules: dict[str, Any],
    default_debug_details: dict[str, Any],
    table_zip_path: str,
    table_file: str,
) -> dict[str, Any]:
    """Build a normalized debug_details payload for rules-not-found paths."""
    raw_details = domain_rules.get(EG_RULE_DEBUG_DETAILS_KEY, default_debug_details)
    if isinstance(raw_details, dict):
        details = dict(raw_details)
    else:
        details = {"raw_debug_details": raw_details}

    details["table_zip_path"] = table_zip_path
    details["table_file"] = table_file
    return details


class QDQTypeInfo:
    """Store type annotation and domain information for QDQ nodes."""

    def __init__(self, type_annotation: str, domain: ONNXDomain):
        self.type_annotation = type_annotation
        self.domain = domain

    def __repr__(self) -> str:
        return f"QDQTypeInfo(type={self.type_annotation}, domain={self.domain.name})"

    def __str__(self) -> str:
        return self.__repr__()


def _normalize_type_var_annotation(type_value: str) -> str:
    """Normalize a type-var value to the runtime table annotation format."""
    try:
        return SupportedONNXType.from_onnx_type(type_value).annotation
    except ValueError:
        return SupportedONNXType.normalize_annotation(type_value)


def _get_pattern_type_var_conditions(
    pattern_match: PatternMatchResult,
    gen: Any,
) -> dict[str, str]:
    """Build normalized type-var conditions for a pattern generator."""
    conditions: dict[str, str] = {}
    type_var_suffix = f"_{gen.op_name}"

    for type_var_name, dtypes_to_test in gen.type_var_dtypes_to_test.items():
        base_type_var_name = (
            type_var_name[: -len(type_var_suffix)]
            if type_var_name.endswith(type_var_suffix)
            else type_var_name
        )
        matched_type = pattern_match.type_param_to_type.get(base_type_var_name)

        if matched_type is not None:
            conditions[type_var_name] = _normalize_type_var_annotation(matched_type)
        elif dtypes_to_test:
            conditions[type_var_name] = dtypes_to_test[0].annotation

    return conditions


def _get_qdq_query_conditions_for_node(
    node: onnx.NodeProto,
    schema: onnx.defs.OpSchema,
    input_to_dq: dict[str, QDQTypeInfo],
    output_to_q: dict[str, QDQTypeInfo],
) -> dict[str, Any]:
    """Extract qdq query conditions for runtime checking of an ONNX node.

    Check all inputs and outputs of the node to see if they are quantized via QDQ pattern.
    For each quantized input/output, add corresponding conditions:
        QDQ_{var_name_in_schema} = type_annotation

    If any input/output is quantized, non-quantized inputs/outputs are also recorded:
        QDQ_{var_name_in_schema} = None

    If input/output is in schema but not in node, explicitly set to None.

    Args:
        node: ONNX node to analyze.
        schema: ONNX schema for the node's op_type.
        input_to_dq: Maps DQ output name -> QDQTypeInfo (for nodes consuming DQ outputs).
        output_to_q: Maps Q input name -> QDQTypeInfo (for nodes producing Q inputs).

    Returns:
        Dict of QDQ conditions for runtime check query. Empty if no QDQ quantization.
    """
    qdq_inputs: dict[str, str | None] = {}
    qdq_outputs: dict[str, str | None] = {}

    # Check inputs against schema.
    # Variadic inputs are collapsed to the schema base name (e.g. "inputs") using
    # the type of the first quantized variadic tensor as the representative value,
    # matching the stored rule columns produced by the generator.
    node_input_idx = 0
    for schema_input in schema.inputs:
        is_variadic = schema_input.option == onnx.defs.OpSchema.FormalParameterOption.Variadic
        if is_variadic:
            # Consume all remaining node inputs; record the first quantized type found.
            representative_type: str | None = None
            any_provided = False
            while node_input_idx < len(node.input):
                tensor_name = node.input[node_input_idx]
                if tensor_name:
                    any_provided = True
                    if tensor_name in input_to_dq and representative_type is None:
                        representative_type = input_to_dq[tensor_name].type_annotation
                node_input_idx += 1
            if any_provided:
                qdq_inputs[schema_input.name] = representative_type
        else:
            tensor_name = node.input[node_input_idx] if node_input_idx < len(node.input) else ""
            if tensor_name:
                qdq_inputs[schema_input.name] = (
                    input_to_dq[tensor_name].type_annotation if tensor_name in input_to_dq else None
                )
            elif schema_input.option == onnx.defs.OpSchema.FormalParameterOption.Optional:
                # Optional input not provided - explicitly set to None
                qdq_inputs[schema_input.name] = None
            node_input_idx += 1

    # Check outputs against schema
    for idx, schema_output in enumerate(schema.outputs):
        tensor_name = node.output[idx] if idx < len(node.output) else ""
        if tensor_name:
            if tensor_name in output_to_q:
                qdq_outputs[schema_output.name] = output_to_q[tensor_name].type_annotation
            else:
                qdq_outputs[schema_output.name] = None

    # Only return conditions if at least one input/output is quantized
    has_qdq = any(v is not None for v in qdq_inputs.values()) or any(
        v is not None for v in qdq_outputs.values()
    )

    if not has_qdq:
        return {}

    conditions: dict[str, Any] = {}
    for name, type_annotation in qdq_inputs.items():
        conditions[f"QDQ_{name}"] = type_annotation
    for name, type_annotation in qdq_outputs.items():
        conditions[f"QDQ_{name}"] = type_annotation

    logger.debug("Node %s QDQ conditions: %s", node.op_type, conditions)
    return conditions


def get_query_conditions_for_node(
    node: onnx.NodeProto,
    opset_version: int,
    valueinfo: dict,
    initializers: dict,
    constants: dict,
    domain: ONNXDomain,
    input_to_dq: dict[str, QDQTypeInfo],
    output_to_q: dict[str, QDQTypeInfo],
    dynamic_axis_strict_mode: bool = False,
) -> tuple[dict[str, Any], list[str], bool]:
    """Extract query conditions for runtime checking of an ONNX node.

    Args:
        node: ONNX node to analyze.
        opset_version: ONNX opset version.
        valueinfo: Dict mapping tensor names to ValueInfoProto.
        initializers: Dict mapping initializer names to TensorProto.
        constants: Dict mapping constant node output names to TensorProto.
        domain: ONNX domain of the node.
        input_to_dq: Maps DQ output name -> QDQTypeInfo (for nodes consuming DQ outputs).
        output_to_q: Maps Q input name -> QDQTypeInfo (for nodes producing Q inputs).
        dynamic_axis_strict_mode: If False (default), maps any dynamic axes to (0,)
            for matching against first_axis test data. If True, preserves exact indices.

    Returns:
        Tuple of (conditions, infinite_properties, is_qdq):
        - conditions: Dict of property conditions for runtime check query.
        - infinite_properties: List of property names with infinite value ranges.
        - is_qdq: True if node has QDQ quantization on inputs or outputs.
    """
    conditions = {}
    schema = domain.get_op_schema(node.op_type, opset_version)
    input_names, variadic_input_name, attribute_names, type_annotations = get_op_input_properties(
        schema
    )

    # Build set of optional input names from schema
    optional_input_names = {
        inp.name
        for inp in schema.inputs
        if inp.option == onnx.defs.OpSchema.FormalParameterOption.Optional
    }

    for a in node.attribute:
        if a is None:
            raise OpOptionalInputSupportError(
                f"Node {node.op_type} has optional attribute. "
                f"Expected attribute names: {_format_list_preview(attribute_names)}"
            )
        if a.name in attribute_names:
            column_name = f"attr_{a.name}"
            attr_value = get_attribute_proto_value(a)
            conditions[column_name] = attr_value
            conditions[f"{column_name}_is_none"] = attr_value is None

    # create runtime checker op
    try:
        runtime_checker_op = get_runtime_checker_op(node.op_type)(schema)
    except KeyError:
        raise OpUnsupportedError(f"Node {node.op_type} is not supported") from None
    type_vars = {}

    # fill missing attrs with default values; set None for optional attrs without defaults
    for k, v in schema.attributes.items():
        column_name = f"attr_{k}"
        if column_name not in conditions:
            if v.default_value.name:
                attr_value = get_attribute_proto_value(v.default_value)
                conditions[column_name] = attr_value
                conditions[f"{column_name}_is_none"] = attr_value is None
            elif not v.required:
                conditions[column_name] = None
                conditions[f"{column_name}_is_none"] = True
            else:
                logger.warning(
                    "Node %s (name: %s): required attribute '%s' missing and has no default",
                    node.op_type,
                    node.name,
                    k,
                )

    # # TODO: add values for optional inputs
    # assert len(node.input) >= len(input_names), (
    #     f"Node {node.op_type} has fewer inputs "
    #     f"({len(node.input)}) than expected ({len(input_names)})"
    # )

    def _compute_dynamic_axes(shape: tuple | None, is_constant: bool) -> tuple[int, ...]:
        """Compute dynamic axis indices from a shape.

        Constant inputs are always fixed shape. For non-constant inputs,
        detect dimensions that are None, string (symbolic), or negative.
        """
        if is_constant or shape is None:
            return ()
        dyn = tuple(
            i
            for i, s in enumerate(shape)
            if s is None or isinstance(s, str) or (isinstance(s, int) and s < 0)
        )
        # Non-strict mode: map any dynamic axes to (0,) for database matching
        if not dynamic_axis_strict_mode and len(dyn) > 0:
            dyn = (0,)
        return dyn

    def update_conditions_(
        cond: dict,
        input_name: str,
        is_variadic: bool,
        is_constant: bool,
        shape: tuple | None = None,
        value: tuple | None = None,
    ):
        dyn_axes = _compute_dynamic_axes(shape, is_constant)
        if is_variadic:
            cond[f"{input_name}_is_constant"] = (
                *cond.get(f"{input_name}_is_constant", ()),
                is_constant,
            )
            cond[f"{input_name}_is_fixed_shape"] = (
                *cond.get(f"{input_name}_is_fixed_shape", ()),
                len(dyn_axes) == 0,
            )
            cond[f"{input_name}_dynamic_axes"] = (
                *cond.get(f"{input_name}_dynamic_axes", ()),
                dyn_axes,
            )
            cond[f"{input_name}_shape"] = (*cond.get(f"{input_name}_shape", ()), shape)
            cond[f"{input_name}_value"] = (*cond.get(f"{input_name}_value", ()), value)
        else:
            cond[f"{input_name}_is_constant"] = is_constant
            cond[f"{input_name}_is_fixed_shape"] = len(dyn_axes) == 0
            cond[f"{input_name}_dynamic_axes"] = dyn_axes
            # Always set shape, even if None (for quantized models with incomplete valueinfo)
            cond[f"{input_name}_shape"] = shape
            # Always set value, even if None
            cond[f"{input_name}_value"] = value

    if variadic_input_name is not None:
        # Calculate number of variadic inputs
        # variadic_input_name is NOT included in input_names, so we need:
        # n_variadic_inputs = total_inputs - non_variadic_inputs
        n_variadic_inputs = max(len(node.input) - len(input_names), 0) if variadic_input_name else 0
        input_names += [variadic_input_name] * n_variadic_inputs
        # Get input constraint types for consistent property naming when optional inputs are None

    # Iterate through all expected inputs (including optional ones)
    for idx, input_name in enumerate(input_names):
        is_variadic = input_name == variadic_input_name
        type_annotation = remove_optional_from_type_annotation(type_annotations[input_name])

        # Get the input name from node.input at this position
        # For optional inputs, node.input may have fewer entries or empty strings
        inp_name = node.input[idx] if idx < len(node.input) else ""

        # Handle optional inputs that are not provided (empty string or missing)
        if not inp_name:
            # Check if this is an optional input using schema
            if input_name in optional_input_names:
                # Mark as optional/undefined - is_constant is True
                # since the value is known (None/not provided)
                logger.warning(
                    "Node %s (name: %s): input '%s' is optional"
                    " and not provided, setting value to None",
                    node.op_type,
                    node.name,
                    input_name,
                )
                conditions[f"{input_name}_is_constant"] = True
                conditions[f"{input_name}_is_fixed_shape"] = True
                conditions[f"{input_name}_dynamic_axes"] = ()
                conditions[f"{input_name}_is_none"] = True
                conditions[f"{input_name}_shape"] = None
                conditions[f"{input_name}_value"] = None
                continue
            # Required input is missing - this is an error
            raise OpOptionalInputSupportError(
                f"Node {node.op_type} missing required input {input_name}"
            )

        if inp_name in initializers:
            init = initializers[inp_name]
            arr = numpy_helper.to_array(init)
            update_conditions_(
                conditions, input_name, is_variadic, True, arr.shape, make_hashable(arr)
            )
            conditions[f"{input_name}_is_none"] = False

            # Add type_vars info for initializers
            dtype = dtype_from_tensorproto_enum(init.data_type)
            if type_annotation in runtime_checker_op.type_var_dtypes_to_test:
                assert type_annotation not in type_vars or type_vars[type_annotation] == dtype, (
                    f"Inconsistent dtype for type annotation "
                    f"{type_annotation}: "
                    f"{type_vars[type_annotation]} vs {dtype}"
                )
                type_vars[type_annotation] = dtype
        elif inp_name in constants:
            # Handle Constant node inputs
            const_tensor = constants[inp_name]
            arr = numpy_helper.to_array(const_tensor)
            update_conditions_(
                conditions, input_name, is_variadic, True, arr.shape, make_hashable(arr)
            )
            conditions[f"{input_name}_is_none"] = False

            # Add type_vars info for constants
            dtype = dtype_from_tensorproto_enum(const_tensor.data_type)
            if type_annotation in runtime_checker_op.type_var_dtypes_to_test:
                assert type_annotation not in type_vars or type_vars[type_annotation] == dtype, (
                    f"Inconsistent dtype for type annotation "
                    f"{type_annotation}: "
                    f"{type_vars[type_annotation]} vs {dtype}"
                )
                type_vars[type_annotation] = dtype
        else:
            vi = valueinfo.get(inp_name)
            shape, dtype = (None, None)
            if vi is not None:
                shape, dtype = shape_and_dtype_from_valueinfo(vi)
            else:
                # Input is provided but valueinfo not found
                # This commonly happens in quantized models where DequantizeLinear outputs
                # are not properly captured by shape inference
                raise OpLackOfRequiredInformationError(
                    f"Node {node.op_type} (name: "
                    f"{node.name}): Input '{inp_name}' "
                    f"(parameter '{input_name}') not found "
                    f"in valueinfo - model may have "
                    f"incomplete shape information "
                    f"(common in quantized models)"
                )

            if type_annotation in runtime_checker_op.type_var_dtypes_to_test:
                assert type_annotation not in type_vars or type_vars[type_annotation] == dtype, (
                    f"Inconsistent dtype for type annotation "
                    f"{type_annotation}: "
                    f"{type_vars[type_annotation]} vs {dtype}"
                )
                type_vars[type_annotation] = dtype

            is_constant = False  # QDQ doesn't care about constant status
            update_conditions_(conditions, input_name, is_variadic, is_constant, shape, None)
            conditions[f"{input_name}_is_none"] = False

    conditions["n_outputs"] = len(node.output)

    # Try to derive properties, but catch errors for incomplete/invalid model information
    try:
        conditions = runtime_checker_op.derive_properties(conditions)
        conditions.pop("n_outputs", None)
    except (KeyError, TypeError, IndexError) as e:
        # KeyError: missing required property (e.g., 'input_value', 'input_shape')
        # TypeError: invalid property value (e.g., None when expecting iterable)
        # IndexError: accessing empty shape/array (e.g., shape[-1] on empty tuple)
        raise OpLackOfRequiredInformationError(
            f"Node {node.op_type} (name: {node.name}): "
            f"Incomplete model information for "
            f"derive_properties: {e}"
        ) from e

    for k, v in runtime_checker_op.type_var_dtypes_to_test.items():
        if k not in type_vars:
            type_vars[k] = v[0].annotation  # use first dtype as default
    conditions.update(type_vars)

    qdq_conditions = _get_qdq_query_conditions_for_node(node, schema, input_to_dq, output_to_q)
    conditions.update(qdq_conditions)
    is_qdq = bool(qdq_conditions)

    conditions = {k: make_hashable(v) for k, v in conditions.items()}

    return conditions, runtime_checker_op.get_infinite_property_names(), is_qdq


def get_query_conditions_for_pattern(
    pattern_match: PatternMatchResult,
    pattern_name: str,
    opset_versions: dict[ONNXDomain, int],
    dynamic_axis_strict_mode: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Extract query conditions for runtime checking of a subgraph pattern.

    Builds the same conditions format as get_query_conditions_for_node but
    from PatternMatchResult fields (type variables, input infos, attributes).

    Args:
        pattern_match: PatternMatchResult containing match details.
        pattern_name: Pattern variant name (e.g., "ReshapeTransposeReshapeLowDim").
        opset_versions: Dict mapping ONNXDomain to opset version.
        dynamic_axis_strict_mode: If False (default), maps any dynamic axes to (0,).

    Returns:
        Tuple of (conditions, infinite_properties):
        - conditions: Dict of property conditions for runtime check query.
        - infinite_properties: List of property names with infinite value ranges.
    """
    conditions: dict[str, Any] = {}
    gen = None
    infinite_properties: list[str] = []

    def _compute_dynamic_axes(shape: tuple | None, is_constant: bool) -> tuple[int, ...]:
        if is_constant or shape is None:
            return ()
        dyn = tuple(
            i
            for i, s in enumerate(shape)
            if s is None or isinstance(s, str) or (isinstance(s, int) and s < 0)
        )
        if not dynamic_axis_strict_mode and len(dyn) > 0:
            dyn = (0,)
        return dyn

    # Type variables (e.g., T_ReshapeTransposeReshapePattern -> "FLOAT")
    conditions.update(pattern_match.type_param_to_type)

    try:
        gen_class = get_pattern_input_generator(pattern_name)
        gen = gen_class(dict(opset_versions))
        conditions.update(_get_pattern_type_var_conditions(pattern_match, gen))
    except KeyError as e:
        logger.debug("Could not load pattern input generator for '%s': %s", pattern_name, e)

    # Input properties from input_infos
    for input_name, info in pattern_match.input_infos.items():
        dyn_axes = _compute_dynamic_axes(info.shape, info.is_constant)
        conditions[f"{input_name}_is_constant"] = info.is_constant
        conditions[f"{input_name}_is_fixed_shape"] = len(dyn_axes) == 0
        conditions[f"{input_name}_dynamic_axes"] = dyn_axes
        conditions[f"{input_name}_shape"] = info.shape
        conditions[f"{input_name}_value"] = (
            make_hashable(info.value) if info.value is not None else None
        )
        conditions[f"{input_name}_is_none"] = False

    # Attributes (with attr_ prefix)
    for attr_name, attr_value in pattern_match.attributes.items():
        conditions[f"attr_{attr_name}"] = attr_value
        conditions[f"attr_{attr_name}_is_none"] = attr_value is None

    conditions["n_outputs"] = len(pattern_match.skeleton_match_result.pattern.get_schema().outputs)

    # Derive additional properties via pattern input generator
    if gen is not None:
        try:
            conditions = gen.derive_properties(conditions)
            conditions.pop("n_outputs", None)
            infinite_properties = gen.get_infinite_property_names()
        except Exception as e:
            logger.debug("Could not derive properties for pattern '%s': %s", pattern_name, e)

    conditions = {k: make_hashable(v) for k, v in conditions.items()}
    return conditions, infinite_properties


class RuntimeCheckerQuery:
    """Query runtime database for pattern support (placeholder implementation)."""

    def __init__(
        self,
        model_proto: onnx.ModelProto,
        ep_name: str,
        device_type: str,
        dynamic_axis_strict_mode: bool = False,
    ) -> None:
        """Initialize runtime checker query.

        Args:
            model_proto: ONNX model proto
            ep_name: Execution provider name
            device_type: Device type (e.g., "CPU", "GPU", "NPU")
            dynamic_axis_strict_mode: If False (default), maps any dynamic axes to (0,)
                for matching against first_axis test data. If True, preserves exact
                dynamic axis indices.
        """
        self.dynamic_axis_strict_mode = dynamic_axis_strict_mode
        # Try shape inference: standard ONNX first, then symbolic (onnxruntime)
        try:
            # First apply standard ONNX shape inference
            self.model_proto = shape_inference.infer_shapes(model_proto)

            # Then try to enhance with symbolic shape inference
            # if available which supports Microsoft domain
            try:
                from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

                self.model_proto = SymbolicShapeInference.infer_shapes(self.model_proto)
            except Exception as e:
                # If symbolic shape inference fails, continue with standard inference result
                logger.debug(
                    f"Symbolic shape inference not available or "
                    f"failed: {e}. Using standard ONNX shape "
                    f"inference result."
                )
        except Exception as e:
            # If standard shape inference fails, use original model
            logger.warning(f"Shape inference failed: {e}. Using original model.")
            self.model_proto = model_proto

        self.ep_name = ep_name
        self.device_type = device_type
        self.valueinfo = collect_valueinfo_dict(self.model_proto)
        self.initializers = collect_initializers(self.model_proto)
        self._collect_qdq_types()
        # Store opset versions by domain in a dictionary
        self.opset_versions = ONNXDomain.get_model_domain_opset_versions(self.model_proto)

        # Collect Constant nodes: map output name -> TensorProto
        self.constants = {}
        for node in self.model_proto.graph.node:
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value":
                        self.constants[node.output[0]] = attr.t
                        break

        # Alternatives support not yet implemented
        self.alternatives: list[PatternAlternative] = []

        # Lazy-initialized EP checker for local fallback
        self._ep_checker: EPChecker | None = None
        self._ep_available_locally: bool | None = None

        # Instantiate registered node checkers from the registry
        self.node_checkers: list[NodeChecker] = [
            checker_class() for checker_class in NodeCheckerRegistry.get_all_checkers()
        ]

        # To avoid logging the same failed node multiple times
        self._failed_nodes_logged: set[Any] = set()
        # Cache of nodes that have been run locally for quick lookup
        self._local_run_nodes: dict[Any, RuntimeTestResult] = {}

        # Register per-domain lazy rule objects — no file I/O occurs here.
        registered_patterns = set(get_registered_pattern_input_generators())
        self.ep_neg_rules: dict[ONNXDomain, dict] = {}
        self.ep_neg_rules_qdq: dict[ONNXDomain, dict] = {}
        self.pattern_neg_rules: dict[ONNXDomain, dict] = {}
        self.pattern_neg_rules_qdq: dict[ONNXDomain, dict] = {}
        self.df_tables: dict[ONNXDomain, LazyDomainTables] = {}
        self.df_tables_qdq: dict[ONNXDomain, LazyDomainTables] = {}

        for domain, opset_version in self.opset_versions.items():
            file_prefix = domain.name
            rule_zip_path = (
                Path(__file__)
                .parent.joinpath(
                    f"../rules/runtime_check_rules/{self.ep_name}_{self.device_type}_{file_prefix}_opset{opset_version}.zip"
                )
                .resolve(strict=False)
            )
            base = f"{self.ep_name}_{self.device_type}_{file_prefix}_opset{opset_version}"
            rule_file = f"{base}_negative_rules.json"
            qdq_rule_file = f"{base}_negative_rules_qdq.json"
            table_file = f"{base}_tables.json"
            qdq_table_file = f"{base}_tables_qdq.json"

            self.ep_neg_rules[domain] = _LazyNegRules(
                rule_zip_path, rule_file, registered_patterns, set_error_on_missing=True
            )
            self.pattern_neg_rules[domain] = _LazyNegRules(
                rule_zip_path, rule_file, registered_patterns, patterns_only=True
            )
            self.ep_neg_rules_qdq[domain] = _LazyNegRules(
                rule_zip_path, qdq_rule_file, registered_patterns
            )
            self.pattern_neg_rules_qdq[domain] = _LazyNegRules(
                rule_zip_path, qdq_rule_file, registered_patterns, patterns_only=True
            )
            self.df_tables[domain] = LazyDomainTables(rule_zip_path, table_file)
            self.df_tables_qdq[domain] = LazyDomainTables(rule_zip_path, qdq_table_file)

    def _collect_qdq_types(self) -> None:
        """Collect QDQ types from the model.

        Maps input names to DequantizeLinear output types and output names to QuantizeLinear types.
        - input_to_dq_type: Maps DQ output name -> dtype (DQ output feeds into other nodes as input)
        - output_to_q_type: Maps Q input name -> dtype (other nodes' outputs feed into Q as input)
        """
        self.input_to_dq_type: dict[str, QDQTypeInfo] = {}
        self.output_to_q_type: dict[str, QDQTypeInfo] = {}

        for node in self.model_proto.graph.node:
            if node.op_type == "DequantizeLinear" and node.output and node.input:
                output_name = node.output[0]
                x_name = node.input[0]
                dtype = None
                # Try to get dtype from valueinfo first
                vi = self.valueinfo.get(x_name)
                if vi is not None:
                    _, dtype = shape_and_dtype_from_valueinfo(vi)
                # Fall back to initializers if not in valueinfo
                if dtype is None:
                    init = self.initializers.get(x_name)
                    if init is not None:
                        dtype = dtype_from_tensorproto_enum(init.data_type)
                if dtype is not None:
                    domain = ONNXDomain.from_str(node.domain)
                    self.input_to_dq_type[output_name] = QDQTypeInfo(
                        type_annotation=dtype,
                        domain=domain,
                    )

            elif node.op_type == "QuantizeLinear" and node.input and node.output:
                input_name = node.input[0]
                y_name = node.output[0]
                dtype = None
                # Try to get dtype from valueinfo first
                vi = self.valueinfo.get(y_name)
                if vi is not None:
                    _, dtype = shape_and_dtype_from_valueinfo(vi)
                # Fall back to initializers if not in valueinfo
                if dtype is None:
                    init = self.initializers.get(y_name)
                    if init is not None:
                        dtype = dtype_from_tensorproto_enum(init.data_type)
                if dtype is not None:
                    domain = ONNXDomain.from_str(node.domain)
                    self.output_to_q_type[input_name] = QDQTypeInfo(
                        type_annotation=dtype,
                        domain=domain,
                    )
        logger.debug("Collected input_to_dq_type: %s", self.input_to_dq_type)
        logger.debug("Collected output_to_q_type: %s", self.output_to_q_type)

    def _is_ep_available_locally(self) -> bool:
        """Check if the target EP is available on the local machine.

        Returns:
            True if the EP+device combination is available locally.
        """
        if self._ep_available_locally is not None:
            return self._ep_available_locally

        from ... import winml

        winml.register_execution_providers(ort=True)

        from ...utils.constants import DEVICE_TO_DEVICE_TYPE

        device_type_enum = DEVICE_TO_DEVICE_TYPE.get(self.device_type)
        if device_type_enum is None:
            self._ep_available_locally = False
            return False

        try:
            ep_devices = ort.get_ep_devices()
            self._ep_available_locally = any(
                ep_dev.ep_name == self.ep_name and ep_dev.device.type == device_type_enum
                for ep_dev in ep_devices
            )
        except Exception as e:
            logger.debug("Failed to query EP devices: %s", e)
            self._ep_available_locally = False

        return self._ep_available_locally

    def _get_ep_checker(self) -> EPChecker:
        """Get or create an EPChecker instance for local runtime checks.

        Returns:
            EPChecker instance configured for the target EP+device.
        """
        if self._ep_checker is None:
            from ...utils.constants import DEVICE_TO_DEVICE_TYPE

            self._ep_checker = EPChecker(
                ep_name=self.ep_name,
                device_type=DEVICE_TO_DEVICE_TYPE[self.device_type],
            )
        return self._ep_checker

    def _build_single_node_model(
        self, node: onnx.NodeProto, op_domain: ONNXDomain, opset_version: int
    ) -> onnx.ModelProto:
        """Build a standalone ONNX model containing a single node.

        Extracts the node along with its input/output value info and initializers
        from the parent model to create a self-contained single-node model.

        Args:
            node: The ONNX node to extract.
            op_domain: The domain of the node's operator.
            opset_version: The opset version to use.

        Returns:
            A standalone ONNX ModelProto containing only this node.

        Raises:
            ValueError: If required input information is missing.
        """
        graph_inputs: list[onnx.ValueInfoProto] = []
        graph_initializers: list[onnx.TensorProto] = []

        for inp_name in node.input:
            if not inp_name:
                continue
            if inp_name in self.initializers:
                graph_initializers.append(self.initializers[inp_name])
            elif inp_name in self.constants:
                # Convert Constant node output to initializer
                graph_initializers.append(self.constants[inp_name])
            else:
                vi = self.valueinfo.get(inp_name)
                if vi is not None:
                    graph_inputs.append(vi)
                else:
                    raise ValueError(
                        f"Input '{inp_name}' for node '{node.name}' ({node.op_type}) "
                        f"not found in valueinfo or initializers"
                    )

        graph_outputs: list[onnx.ValueInfoProto] = []
        for out_name in node.output:
            if not out_name:
                continue
            vi = self.valueinfo.get(out_name)
            if vi is not None:
                graph_outputs.append(vi)
            else:
                # Create output with unknown type/shape as fallback
                graph_outputs.append(
                    onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.UNDEFINED, None)
                )

        graph = onnx.helper.make_graph(
            [node],
            f"single_node_{node.op_type}",
            graph_inputs,
            graph_outputs,
            initializer=graph_initializers,
        )

        # Build opset imports
        domain_str = op_domain.schema_domain
        is_default_domain = domain_str == "" or domain_str == ONNXDomain.AI_ONNX.value
        effective_version = max(opset_version, 7) if is_default_domain else opset_version
        opset_imports = [onnx.helper.make_opsetid(domain_str, effective_version)]

        # If node uses a non-default domain, also add the default domain
        if not is_default_domain:
            default_opset = self.opset_versions.get(ONNXDomain.AI_ONNX, 17)
            opset_imports.append(onnx.helper.make_opsetid("", max(default_opset, 7)))

        model = onnx.helper.make_model(graph, opset_imports=opset_imports)

        try:
            model = infer_onnx_shapes(model)
        except Exception as e:
            logger.debug("Shape inference failed for single-node model: %s", e)

        return model

    def _generate_node_inputs(self, node: onnx.NodeProto) -> dict[str, np.ndarray]:
        """Generate dummy input data for a single-node model.

        Creates numpy arrays with appropriate shapes and dtypes based on the
        node's input value info. Initializer/constant inputs are excluded since
        they are embedded in the model.

        Args:
            node: The ONNX node to generate inputs for.

        Returns:
            Dict mapping input names to numpy arrays.

        Raises:
            ValueError: If dtype or shape information is missing for an input.
        """
        input_feed: dict[str, np.ndarray] = {}
        default_dim_size = 2  # Replace dynamic/unknown dims with this size

        for inp_name in node.input:
            if not inp_name:
                continue
            # Skip initializers and constants - they are embedded in the model
            if inp_name in self.initializers or inp_name in self.constants:
                continue

            vi = self.valueinfo.get(inp_name)
            if vi is None:
                raise ValueError(
                    f"Input '{inp_name}' for node '{node.name}' ({node.op_type}) "
                    f"not found in valueinfo"
                )

            shape, dtype_str = shape_and_dtype_from_valueinfo(vi)
            if dtype_str is None:
                raise ValueError(
                    f"Input '{inp_name}' for node '{node.name}' ({node.op_type}) "
                    f"has no dtype information"
                )

            # Convert dtype string to numpy dtype
            np_dtype = SupportedONNXType.from_annotation(dtype_str).np_type

            if shape is None:
                # No shape info at all - use a simple 1D array
                concrete_shape = (default_dim_size,)
            else:
                # Replace dynamic dimensions (strings or None) with default size
                concrete_shape = tuple(
                    d if isinstance(d, int) and d > 0 else default_dim_size for d in shape
                )

            input_feed[inp_name] = np.zeros(concrete_shape, dtype=np_dtype)

        return input_feed

    def _try_local_ep_check(
        self,
        node: onnx.NodeProto,
        op_domain: ONNXDomain,
        opset_version: int,
        pattern_match: PatternMatchResult,
        node_tags: list[NodeTag],
        fallback_reason: str,
        save_node_types: set[str] | None = None,
        conditions: Any | None = None,
    ) -> PatternRuntime | None:
        """Attempt to compile and run a node locally when rules are not found.

        If the local machine supports the target EP, builds a single-node model
        and runs compile/run checks using EPChecker.

        Args:
            node: The ONNX node to check.
            op_domain: The domain of the node's operator.
            opset_version: The opset version to use.
            pattern_match: Pattern match result for the node.
            node_tags: Collected tags for the node.
            fallback_reason: The original reason rules were not found.
            save_node_types: Set of node types to save (e.g., {"partial", "unsupported"}).
            conditions: Conditions for the local check.

        Returns:
            PatternRuntime with local check results, or None if local check
            is not possible (EP not available, model build fails, etc.).
        """
        if not self._is_ep_available_locally():
            logger.debug(
                "EP '%s' on device '%s' not available locally, skipping local check for %s (%s)",
                self.ep_name,
                self.device_type,
                node.name,
                node.op_type,
            )
            return None

        if conditions is not None and conditions in self._local_run_nodes:
            return PatternRuntime(
                pattern_id=pattern_match.pattern.pattern_id,
                result=self._local_run_nodes[conditions],
                alternatives=self.alternatives,
                pattern_match=pattern_match,
            )

        try:
            model = self._build_single_node_model(node, op_domain, opset_version)
            input_feed = self._generate_node_inputs(node)
        except Exception as e:
            logger.warning(
                "Failed to build single-node model for local EP check on %s (%s): %s",
                node.name,
                node.op_type,
                e,
            )
            return None

        model_bytes = model.SerializeToString()
        ep_checker = self._get_ep_checker()

        compile_success = False
        run_success = False
        reasons: list[str] = []

        try:
            with ResilientRunner(capture_output=True, timeout_sec=60) as runner:
                compile_result = runner.run(ep_checker.check_compile, model_bytes, input_feed)
            compile_success = compile_result["result"]["success"]
            if not compile_success:
                reasons.append(
                    f"compile_failed: {compile_result['result'].get('reason', 'unknown')}"
                )
        except Exception as e:
            logger.warning(
                "Local EP compile check failed for %s (%s): %s",
                node.name,
                node.op_type,
                e,
            )
            reasons.append(f"compile_exception: {e}")

        try:
            with ResilientRunner(capture_output=True, timeout_sec=60) as runner:
                run_result = runner.run(ep_checker.check_run, model_bytes, input_feed)
            run_success = run_result["result"]["success"]
            if not run_success:
                reasons.append(f"run_failed: {run_result['result'].get('reason', 'unknown')}")
        except Exception as e:
            logger.warning(
                "Local EP run check failed for %s (%s): %s",
                node.name,
                node.op_type,
                e,
            )
            reasons.append(f"run_exception: {e}")

        reason_str = f"local_ep_check ({fallback_reason})"
        if reasons:
            reason_str += ": " + "; ".join(reasons)

        logger.info(
            "Local EP check for %s (%s): compile=%s, run=%s",
            node.name,
            node.op_type,
            compile_success,
            run_success,
        )

        if not compile_success:
            _save_types = save_node_types or set()
            is_unsupported = "unsupported" in _save_types and not run_success
            is_partial = "partial" in _save_types and run_success
            if is_unsupported or is_partial:
                self._save_failed_node(
                    node,
                    model,
                    conditions,
                    name_suffix="unsupported" if is_unsupported else "partial",
                )

        result = RuntimeTestResult(
            compile=compile_success,
            run=run_success,
            no_data=False,
            reason=reason_str,
            node_tags=node_tags,
            debug_details={
                "source": "local_ep_check",
                "fallback_reason": fallback_reason,
                "op_type": node.op_type,
                "node_name": node.name,
                "domain": str(op_domain),
                "opset_version": opset_version,
                "table_zip_path": "",
                "table_file": "",
            },
        )

        if conditions is not None:
            self._local_run_nodes[conditions] = result

        return PatternRuntime(
            pattern_id=pattern_match.pattern.pattern_id,
            result=result,
            alternatives=self.alternatives,
            pattern_match=pattern_match,
        )

    def _detect_missing_shape_info(self, node: onnx.NodeProto) -> list[str]:
        """Detect inputs of a node that have missing shape information.

        Args:
            node: ONNX node to check

        Returns:
            List of input names with missing or incomplete shape information
        """
        missing_inputs = []
        for inp_name in node.input:
            # Skip empty inputs (unprovided optional inputs)
            if not inp_name:
                continue

            # Skip initializers and constants - they always have shape
            if inp_name in self.initializers or inp_name in self.constants:
                continue

            # Check if input has valueinfo with shape
            vi = self.valueinfo.get(inp_name)
            if vi is None:
                # Note: This might be an optional input that wasn't provided with shape info.
                # TODO: Consider schema to distinguish optional vs required inputs
                missing_inputs.append(inp_name)
                continue

            # Use shape_and_dtype_from_valueinfo to extract shape information
            shape, _ = shape_and_dtype_from_valueinfo(vi)

            # Check if shape is missing or contains unknown dimensions
            if shape is None or None in shape:
                missing_inputs.append(inp_name)

        return missing_inputs

    def _collect_node_tags(self, node: onnx.NodeProto) -> list[NodeTag]:
        """Collect all applicable tags for a node.

        Args:
            node: ONNX node to analyze

        Returns:
            List of NodeTag enums describing node properties
        """
        tags: list[NodeTag] = []

        # Non-deterministic operators that should never be constant-folded
        # even if all inputs are constant (they produce random/different outputs)
        non_deterministic_ops = {
            "RandomNormal",
            "RandomNormalLike",
            "RandomUniform",
            "RandomUniformLike",
            "Multinomial",
        }

        # Check if all inputs are constant (excluding non-deterministic ops)
        non_empty_inputs = [inp for inp in node.input if inp]
        if (
            node.op_type not in non_deterministic_ops
            and non_empty_inputs
            and all(inp in self.initializers or inp in self.constants for inp in non_empty_inputs)
        ):
            tags.append(NodeTag.ALL_INPUTS_CONSTANT)

        # Check for missing shape inference
        missing_shape_inputs = self._detect_missing_shape_info(node)
        if missing_shape_inputs:
            tags.append(NodeTag.MISSING_SHAPE_INFERENCE)
            logger.warning(
                "Op %s (%s) has inputs with missing shape info: %s",
                node.name,
                node.op_type,
                missing_shape_inputs,
            )

        return tags

    def _save_failed_node(
        self,
        node: onnx.NodeProto,
        node_model: onnx.ModelProto,
        conditions: Any | None,
        name_suffix: str = "",
    ) -> None:
        if conditions is not None and conditions in self._failed_nodes_logged:
            return  # Skip logging the same failure again

        if conditions is not None:
            self._failed_nodes_logged.add(conditions)

        failed_nodes_dir = Path("failed_nodes")
        failed_nodes_dir.mkdir(parents=True, exist_ok=True)
        safe_name = (
            node.name.replace("/", "_").replace("\\", "_")
            if node.name
            else node.output[0].replace("/", "_").replace("\\", "_")
        )
        # clean up safe_name to avoid invalid characters for filenames on Windows
        safe_name = "".join([c if c.isalnum() or c in "._- " else "_" for c in safe_name])
        if name_suffix:
            safe_name = f"{safe_name}_{name_suffix}"
        model_path = failed_nodes_dir / f"{node.op_type}_{safe_name}.onnx"
        try:
            import onnx

            onnx.save_model(node_model, model_path)
            logger.info("Saved unsupported node to %s", model_path)
        except Exception as e:
            logger.warning("Failed to save node for %s: %s", node.op_type, e)

    def run_for_model_per_op(self) -> dict[str, Any]:
        """Run runtime check for all nodes in model.

        Returns:
            Dict with results for each operator
        """
        # run run_for_nodes for all nodes
        return {}

    def _get_domain_fallback_reason(
        self, target_neg_rules: dict[ONNXDomain, dict], op_domain: ONNXDomain
    ) -> str:
        """Get the fallback reason string from negative rules for a domain."""
        return target_neg_rules.get(op_domain, {}).get(EG_RULE_ERROR_KEY, "rules_not_found")

    def _check_negative_rules(
        self,
        op_neg_rules: dict[str, Any],
        conditions: dict[str, Any],
        node: onnx.NodeProto,
        phase: str,
    ) -> tuple[bool, str]:
        """Check negative rules for a single phase (compile or run).

        Args:
            op_neg_rules: Operator negative rules dict with 'all_failed' and 'negative_rules' keys.
            conditions: Node conditions dict from get_query_conditions_for_node.
            node: The ONNX node being checked.
            phase: "compile" or "run".

        Returns:
            Tuple of (passed, reason_text). passed is False if the op fails this phase.

        Raises:
            OpOptionalInputSupportError: If a required property is missing from conditions.
        """
        if op_neg_rules["all_failed"][phase]:
            return False, f"The op {node.op_type} is not supported by {phase}, "

        passed = True
        reason = ""
        for k, v in op_neg_rules["negative_rules"][phase].items():
            if k not in conditions:
                raise OpOptionalInputSupportError(
                    f"{phase.capitalize()} check for op "
                    f"{node.op_type}: required property "
                    f"'{k}' not found in conditions"
                )
            node_values = conditions[k]
            invalid_values = [iv["value"] for iv in v]
            if node_values in invalid_values:
                logger.warning(
                    "Node %s matched %s negative rule: property"
                    " '%s' has value %s which is in"
                    " invalid values %s",
                    node.op_type,
                    phase,
                    k,
                    node_values,
                    invalid_values,
                )
                passed = False
                reason += f"Value {node_values} is in invalid values {invalid_values}, "
        return passed, reason

    def run_for_node(
        self,
        node: onnx.NodeProto,
        for_debug: bool = False,
        run_unknown_op: bool = True,
        save_node_types: set[str] | None = None,
    ) -> PatternRuntime:
        """Run runtime check for a single node.

        Args:
            node: ONNX node to check.
            for_debug: If True, include detailed debug information in results.
            run_unknown_op: If True, attempt local EP check for unknown ops.
            save_node_types: Set of node types to save (e.g., {"partial", "unsupported"}).

        Returns:
            PatternRuntime with check results.
        """
        pattern_match = node_to_pattern_match(node)

        # Ignore QuantizeLinear and DequantizeLinear ops for now,
        # Q and DQ ops will be tested in quantized ops
        ignored_ops = {
            "OP/ai.onnx/Constant",
            "OP/ai.onnx/QuantizeLinear",
            "OP/ai.onnx/DequantizeLinear",
            "OP/com.microsoft/QuantizeLinear",
            "OP/com.microsoft/DequantizeLinear",
        }
        if pattern_match.pattern.pattern_id in ignored_ops:
            return PatternRuntime(
                pattern_id=pattern_match.pattern.pattern_id,
                result=RuntimeTestResult(run=True, compile=True, no_data=False, debug_details=None),
                alternatives=self.alternatives,
                pattern_match=pattern_match,
            )

        # Collect all tags for this node
        node_tags = self._collect_node_tags(node)

        # If all inputs are constant, short-circuit with success
        if NodeTag.ALL_INPUTS_CONSTANT in node_tags:
            logger.warning("Op %s (%s) has all inputs constant", node.name, node.op_type)
            return PatternRuntime(
                pattern_id=pattern_match.pattern.pattern_id,
                result=RuntimeTestResult(
                    run=True,
                    compile=True,
                    no_data=False,
                    reason="all_inputs_constant",
                    node_tags=node_tags,
                    debug_details=None,
                ),
                alternatives=self.alternatives,
                pattern_match=pattern_match,
            )

        try:
            op_domain = ONNXDomain.from_str(node.domain)
        except ValueError:
            # Unknown domain (e.g., custom ops) — report as no_data
            return PatternRuntime(
                pattern_id=pattern_match.pattern.pattern_id,
                result=RuntimeTestResult(
                    run=False,
                    compile=False,
                    no_data=True,
                    reason=f"unsupported_domain:{node.domain}",
                ),
                alternatives=self.alternatives,
                pattern_match=pattern_match,
            )

        # Determine the opset version based on domain (default to 1 if not in model)
        opset_version = self.opset_versions.get(op_domain, 1)

        # Evaluate custom checkers (before rule-based checks — handles EPContext, etc.)
        for checker in self.node_checkers:
            if checker.can_check(node, op_domain, opset_version):
                return checker.check(
                    node,
                    op_domain,
                    opset_version,
                    pattern_match,
                    self.alternatives,
                    ep_name=self.ep_name,
                )

        # Phase 1: Extract conditions to determine if node is QDQ
        is_qdq = False
        table_zip_path = ""
        table_file = ""

        def get_pattern_id(is_qdq):
            return (
                pattern_match.pattern.pattern_id + " (QDQ)"
                if is_qdq
                else pattern_match.pattern.pattern_id
            )

        try:
            conditions, infinite_properties, is_qdq = get_query_conditions_for_node(
                node,
                opset_version,
                self.valueinfo,
                self.initializers,
                self.constants,
                op_domain,
                self.input_to_dq_type,
                self.output_to_q_type,
                dynamic_axis_strict_mode=self.dynamic_axis_strict_mode,
            )
        except (
            OpOptionalInputSupportError,
            OpLackOfRequiredInformationError,
            OpUnsupportedError,
        ) as e:
            exception_type = type(e).__name__
            logger.error(
                "%s caught for op %s (node: %s): %s",
                exception_type,
                node.op_type,
                node.name,
                str(e),
            )
            return PatternRuntime(
                pattern_id=get_pattern_id(is_qdq),
                result=RuntimeTestResult(
                    compile=False,
                    run=False,
                    no_data=True,
                    reason="optional_input_properties_not_found",
                    node_tags=node_tags,
                    debug_details={
                        "op_type": node.op_type,
                        "node_name": node.name,
                        "error_message": str(e),
                        "table_zip_path": table_zip_path,
                        "table_file": table_file,
                    },
                ),
                alternatives=self.alternatives,
                pattern_match=pattern_match,
            )

        # Phase 2: Select appropriate rules and tables based on QDQ status
        target_neg_rules = self.ep_neg_rules_qdq if is_qdq else self.ep_neg_rules
        target_df_tables = self.df_tables_qdq if is_qdq else self.df_tables
        if op_domain in target_df_tables:
            table_zip_path = _normalize_table_zip_path(
                getattr(target_df_tables[op_domain], "_zip_path", "")
            )
            table_file = str(getattr(target_df_tables[op_domain], "_file_name", ""))

        # Phase 3: Check if op exists in target rules
        if op_domain not in target_neg_rules or node.op_type not in target_neg_rules[op_domain]:
            domain_rules = target_neg_rules.get(op_domain, {})
            default_debug_details = {
                "op_type": node.op_type,
                "domain": str(op_domain),
                "opset_version": opset_version,
            }

            if run_unknown_op:
                fallback_reason = self._get_domain_fallback_reason(target_neg_rules, op_domain)
                local_result = self._try_local_ep_check(
                    node,
                    op_domain,
                    opset_version,
                    pattern_match,
                    node_tags,
                    fallback_reason,
                    save_node_types=save_node_types,
                    # conditions not available when domain/op
                    # rules are missing
                    conditions=None,
                )
                if local_result is not None:
                    return local_result

            return PatternRuntime(
                pattern_id=get_pattern_id(is_qdq),
                result=RuntimeTestResult(
                    run=False,
                    compile=False,
                    no_data=True,
                    reason=domain_rules.get(EG_RULE_ERROR_KEY, "rules_not_found"),
                    debug_details=_build_rules_not_found_debug_details(
                        domain_rules,
                        default_debug_details,
                        table_zip_path,
                        table_file,
                    ),
                    node_tags=node_tags,
                ),
                alternatives=self.alternatives,
                pattern_match=pattern_match,
            )

        # Phase 4: Apply negative rules and table matching
        op_neg_rules = target_neg_rules[op_domain][node.op_type]
        reason = ""

        try:
            compile_result, compile_reason = self._check_negative_rules(
                op_neg_rules, conditions, node, "compile"
            )
            run_result, run_reason = self._check_negative_rules(
                op_neg_rules, conditions, node, "run"
            )
            reason = compile_reason + run_reason

            if compile_result or run_result:
                # Table matching
                if (
                    target_df_tables
                    and op_domain in target_df_tables
                    and node.op_type in target_df_tables[op_domain]
                ):
                    domain_tables = target_df_tables[op_domain]
                    table_zip_path = _normalize_table_zip_path(
                        getattr(domain_tables, "_zip_path", "")
                    )
                    table_file = str(getattr(domain_tables, "_file_name", ""))
                    table_df = domain_tables[node.op_type]
                    match_keys = [
                        item
                        for item in table_df.columns.to_list()
                        if item not in infinite_properties
                    ]
                    match_keys.remove("compile_run_success")
                    filter_v = {}
                    for k in match_keys:
                        if k in conditions:
                            filter_v[k] = conditions[k]
                        else:
                            avail = _format_list_preview(conditions.keys())
                            raise OpOptionalInputSupportError(
                                f"Match key '{k}' not found "
                                f"in conditions for op "
                                f"{node.op_type} (domain: "
                                f"{op_domain}). It should be "
                                f"an optional input. Available"
                                f" keys: {avail}"
                            )

                    ret = query_table_exact_match(table_df, filter_v)
                    if not ret.empty:
                        compile_result = ret.iloc[0]["compile_run_success"][0]
                        run_result = ret.iloc[0]["compile_run_success"][1]
                    else:
                        debug_details = None
                        if for_debug:
                            debug_steps: list[dict[str, Any]] = []
                            current_df = table_df
                            for col, value in filter_v.items():
                                rows_before = len(current_df)
                                if col in current_df.columns:
                                    current_df = current_df[current_df[col] == value]
                                rows_after = len(current_df)
                                debug_steps.append(
                                    {
                                        "column": col,
                                        "value": value,
                                        "rows_before": rows_before,
                                        "rows_after": rows_after,
                                    }
                                )
                            debug_details = {
                                "type": "properties_not_found",
                                "total_rows": len(table_df),
                                "table_zip_path": table_zip_path,
                                "table_file": table_file,
                                "steps": debug_steps,
                            }

                        pid = get_pattern_id(is_qdq)
                        logger.info(
                            f"Negative rules check passed, "
                            f"but properties combination not "
                            f"found for {pid} ({node.name}):"
                            f" {filter_v}"
                        )

                        if run_unknown_op:
                            fallback_reason = self._get_domain_fallback_reason(
                                target_neg_rules, op_domain
                            )
                            local_result = self._try_local_ep_check(
                                node,
                                op_domain,
                                opset_version,
                                pattern_match,
                                node_tags,
                                fallback_reason,
                                conditions=make_hashable(filter_v),
                            )
                            if local_result is not None:
                                return local_result

                        return PatternRuntime(
                            pattern_id=get_pattern_id(is_qdq),
                            result=RuntimeTestResult(
                                compile=False,
                                run=False,
                                no_data=True,
                                reason="properties_not_found",
                                filter=str(filter_v),
                                node_tags=node_tags,
                                debug_details=debug_details,
                            ),
                            alternatives=self.alternatives,
                            pattern_match=pattern_match,
                        )
                else:  # no table data
                    if run_unknown_op:
                        fallback_reason = self._get_domain_fallback_reason(
                            target_neg_rules, op_domain
                        )
                        local_result = self._try_local_ep_check(
                            node,
                            op_domain,
                            opset_version,
                            pattern_match,
                            node_tags,
                            fallback_reason,
                            conditions=None,
                        )
                        if local_result is not None:
                            return local_result

                    table_source = "qdq" if is_qdq else "non_qdq"
                    has_tables_dict = bool(target_df_tables)
                    has_domain_tables = bool(has_tables_dict and op_domain in target_df_tables)
                    has_op_table = bool(
                        has_domain_tables and node.op_type in target_df_tables[op_domain]
                    )
                    available_ops_sample: list[str] = []
                    if has_domain_tables:
                        domain_tables = target_df_tables[op_domain]
                        # Expose a small sample of known ops to help debug missing tables
                        raw_keys = list(getattr(domain_tables, "_raw_data", {}).keys())
                        loaded_keys = list(getattr(domain_tables, "_loaded_tables", {}).keys())
                        available_ops_sample = sorted(set(raw_keys + loaded_keys))[:10]
                    table_zip_path = (
                        _normalize_table_zip_path(
                            getattr(target_df_tables[op_domain], "_zip_path", "")
                        )
                        if has_domain_tables
                        else ""
                    )
                    table_file = (
                        str(getattr(target_df_tables[op_domain], "_file_name", ""))
                        if has_domain_tables
                        else ""
                    )
                    return PatternRuntime(
                        pattern_id=get_pattern_id(is_qdq),
                        result=RuntimeTestResult(
                            compile=False,
                            run=False,
                            no_data=True,
                            reason="no_table_data",
                            node_tags=node_tags,
                            debug_details={
                                "ep": self.ep_name,
                                "device": self.device_type,
                                "domain": str(op_domain),
                                "op_type": node.op_type,
                                "opset_version": opset_version,
                                "table_source": table_source,
                                "has_tables_dict": has_tables_dict,
                                "has_domain_tables": has_domain_tables,
                                "has_op_table": has_op_table,
                                "table_zip_path": table_zip_path,
                                "table_file": table_file,
                                "available_ops_sample": available_ops_sample,
                            },
                        ),
                        alternatives=self.alternatives,
                        pattern_match=pattern_match,
                    )
        except (OpOptionalInputSupportError, OpLackOfRequiredInformationError) as e:
            exception_type = type(e).__name__
            logger.error(
                "%s caught for op %s (node: %s): %s",
                exception_type,
                node.op_type,
                node.name,
                str(e),
            )

            tags_for_exception = node_tags.copy() if node_tags else []

            return PatternRuntime(
                pattern_id=get_pattern_id(is_qdq),
                result=RuntimeTestResult(
                    compile=False,
                    run=False,
                    no_data=True,
                    reason="optional_input_properties_not_found",
                    node_tags=tags_for_exception,
                    debug_details={
                        "op_type": node.op_type,
                        "node_name": node.name,
                        "error_message": str(e),
                        "table_zip_path": table_zip_path,
                        "table_file": table_file,
                    },
                ),
                alternatives=self.alternatives,
                pattern_match=pattern_match,
            )

        if not compile_result:
            _save_types = save_node_types or set()
            is_unsupported = "unsupported" in _save_types and not run_result
            is_partial = "partial" in _save_types and run_result
            if is_unsupported or is_partial:
                node_model = self._build_single_node_model(node, op_domain, opset_version)
                # TODO: Need to use match_keys to filter conditions
                self._save_failed_node(
                    node,
                    node_model,
                    make_hashable(conditions),
                    name_suffix="unsupported" if is_unsupported else "partial",
                )

        return PatternRuntime(
            pattern_id=get_pattern_id(is_qdq),
            result=RuntimeTestResult(
                compile=compile_result,
                run=run_result,
                reason=reason.strip().rstrip(","),
                no_data=False,
                node_tags=node_tags,
                debug_details=None,
            ),
            alternatives=self.alternatives,
            pattern_match=pattern_match,
        )

    def run_for_subgraph(
        self,
        pattern_match: PatternMatchResult,
        run_unknown_op: bool = True,
    ) -> PatternRuntime:
        """Run runtime check for subgraph pattern.

        Strategy:
        1. Extract conditions from pattern match (type vars, inputs, attributes)
        2. Look up pattern in pattern_neg_rules by variant name
        3. Apply _check_negative_rules for compile/run phases
        4. Do table matching from df_tables
        5. Fallback to checking individual operators if no pattern rules found

        Args:
            pattern_match: PatternMatchResult containing pattern information
            run_unknown_op: If True, attempt local EP check for unknown patterns.

        Returns:
            PatternRuntime with check results
        """
        pattern_id = pattern_match.pattern.pattern_id
        pattern_name = pattern_match.pattern.__class__.__name__

        # Step 1: Look up pattern in pattern_neg_rules by direct key match
        found_domain: ONNXDomain | None = None
        pattern_rules: dict[str, Any] | None = None
        for domain, patterns in self.pattern_neg_rules.items():
            if pattern_name in patterns:
                found_domain = domain
                pattern_rules = patterns[pattern_name]
                break

        # Step 2: If no pattern rules found, fallback to individual node checking
        if pattern_rules is None:
            logger.info(
                "No pattern-level rules found for '%s', checking individual operators",
                pattern_name,
            )
            return self._run_for_subgraph_per_node(pattern_match, pattern_name, run_unknown_op)

        logger.info("Found pattern-level rules for '%s' in database", pattern_name)

        # Step 3: Extract conditions from PatternMatchResult
        try:
            conditions, infinite_properties = get_query_conditions_for_pattern(
                pattern_match,
                pattern_name,
                self.opset_versions,
                dynamic_axis_strict_mode=self.dynamic_axis_strict_mode,
            )
        except Exception as e:
            logger.error("Failed to extract conditions for pattern '%s': %s", pattern_name, e)
            return PatternRuntime(
                pattern_id=pattern_id,
                result=RuntimeTestResult(
                    compile=False,
                    run=False,
                    no_data=True,
                    reason="pattern_conditions_extraction_failed",
                    debug_details={
                        "pattern_name": pattern_name,
                        "error_message": str(e),
                    },
                ),
                alternatives=self.alternatives,
                pattern_match=pattern_match,
            )

        # Step 4: Resolve table metadata
        assert found_domain is not None
        target_df_tables = self.df_tables
        table_zip_path = ""
        table_file = ""
        if found_domain in target_df_tables:
            table_zip_path = _normalize_table_zip_path(
                getattr(target_df_tables[found_domain], "_zip_path", "")
            )
            table_file = str(getattr(target_df_tables[found_domain], "_file_name", ""))

        # Step 5: Apply negative rules (same as run_for_node Phase 4)
        reason = ""
        try:
            compile_result, compile_reason = self._check_negative_rules(
                pattern_rules, conditions, _make_pseudo_node(pattern_name), "compile"
            )
            run_result, run_reason = self._check_negative_rules(
                pattern_rules, conditions, _make_pseudo_node(pattern_name), "run"
            )
            reason = compile_reason + run_reason

            if compile_result or run_result:
                # Step 6: Table matching
                if (
                    target_df_tables
                    and found_domain in target_df_tables
                    and pattern_name in target_df_tables[found_domain]
                ):
                    domain_tables = target_df_tables[found_domain]
                    table_zip_path = _normalize_table_zip_path(
                        getattr(domain_tables, "_zip_path", "")
                    )
                    table_file = str(getattr(domain_tables, "_file_name", ""))
                    table_df = domain_tables[pattern_name]
                    match_keys = [
                        item
                        for item in table_df.columns.to_list()
                        if item not in infinite_properties
                    ]
                    match_keys.remove("compile_run_success")
                    filter_v: dict[str, Any] = {}
                    for k in match_keys:
                        if k in conditions:
                            filter_v[k] = conditions[k]
                        else:
                            raise OpOptionalInputSupportError(
                                f"Match key '{k}' not found in "
                                f"conditions for pattern {pattern_name}. "
                                f"Available: {_format_list_preview(conditions.keys())}"
                            )

                    ret = query_table_exact_match(table_df, filter_v)
                    if not ret.empty:
                        compile_result = ret.iloc[0]["compile_run_success"][0]
                        run_result = ret.iloc[0]["compile_run_success"][1]
                    else:
                        logger.info(
                            "Negative rules passed but properties not found for %s: %s",
                            pattern_name,
                            filter_v,
                        )

                        # Fallback to per-node check
                        return self._run_for_subgraph_per_node(
                            pattern_match, pattern_name, run_unknown_op
                        )
                else:
                    # No table data — fallback to per-node check
                    return self._run_for_subgraph_per_node(
                        pattern_match, pattern_name, run_unknown_op
                    )

        except OpOptionalInputSupportError as e:
            logger.error("OpOptionalInputSupportError for pattern '%s': %s", pattern_name, e)
            return PatternRuntime(
                pattern_id=pattern_id,
                result=RuntimeTestResult(
                    compile=False,
                    run=False,
                    no_data=True,
                    reason="optional_input_properties_not_found",
                    debug_details={
                        "pattern_name": pattern_name,
                        "error_message": str(e),
                        "table_zip_path": table_zip_path,
                        "table_file": table_file,
                    },
                ),
                alternatives=self.alternatives,
                pattern_match=pattern_match,
            )

        return PatternRuntime(
            pattern_id=pattern_id,
            result=RuntimeTestResult(
                compile=compile_result,
                run=run_result,
                reason=reason.strip().rstrip(","),
                no_data=False,
                debug_details=None,
            ),
            alternatives=self.alternatives,
            pattern_match=pattern_match,
        )

    def _run_for_subgraph_per_node(
        self,
        pattern_match: PatternMatchResult,
        pattern_name: str,
        run_unknown_op: bool,
    ) -> PatternRuntime:
        """Fallback: check each operator in the pattern individually.

        Args:
            pattern_match: PatternMatchResult containing pattern information.
            pattern_name: Pattern variant name.
            run_unknown_op: If True, attempt local EP check for unknown ops.

        Returns:
            PatternRuntime with aggregated results from individual node checks.
        """
        pattern_id = pattern_match.pattern.pattern_id

        if (
            not hasattr(pattern_match, "skeleton_match_result")
            or pattern_match.skeleton_match_result is None
        ):
            logger.warning(
                f"Pattern '{pattern_id}' has no "
                f"skeleton_match_result, cannot check "
                f"individual nodes"
            )
            return PatternRuntime(
                pattern_id=pattern_id,
                result=RuntimeTestResult(
                    compile=False,
                    run=False,
                    no_data=True,
                    reason=(
                        f"Pattern '{pattern_name}' not "
                        f"found in database and has no "
                        f"matched nodes to check"
                    ),
                ),
                alternatives=self.alternatives,
                pattern_match=pattern_match,
            )

        matched_nodes = pattern_match.skeleton_match_result.matched_nodes

        if not matched_nodes:
            logger.warning("Pattern '%s' has no matched nodes", pattern_id)
            return PatternRuntime(
                pattern_id=pattern_id,
                result=RuntimeTestResult(
                    compile=False,
                    run=False,
                    no_data=True,
                    reason=f"Pattern '{pattern_name}' has no nodes to check",
                ),
                alternatives=self.alternatives,
                pattern_match=pattern_match,
            )

        # Check runtime support for each node in the pattern
        node_results: list[PatternRuntime] = []
        for node in matched_nodes:
            node_result = self.run_for_node(node, run_unknown_op=run_unknown_op)
            node_results.append(node_result)

        # Aggregate results: pattern is supported only if ALL nodes are supported
        all_compile = all(r.result.compile for r in node_results)
        all_run = all(r.result.run for r in node_results)
        any_no_data = any(r.result.no_data for r in node_results)

        # Collect failure reasons
        failed_nodes = [
            f"{r.pattern_id}: {r.result.reason}"
            for r in node_results
            if not r.result.compile or not r.result.run
        ]

        no_data_nodes = [r.pattern_id for r in node_results if r.result.no_data]

        if all_compile and all_run and not any_no_data:
            return PatternRuntime(
                pattern_id=pattern_id,
                result=RuntimeTestResult(
                    compile=True,
                    run=True,
                    no_data=False,
                    reason=(
                        f"Pattern '{pattern_name}' fully "
                        f"supported: all "
                        f"{len(node_results)} operators "
                        f"supported"
                    ),
                ),
                alternatives=self.alternatives,
                pattern_match=pattern_match,
            )

        if any_no_data:
            return PatternRuntime(
                pattern_id=pattern_id,
                result=RuntimeTestResult(
                    compile=False,
                    run=False,
                    no_data=True,
                    reason=(
                        f"Pattern '{pattern_name}' status "
                        f"unknown: no data for operators "
                        f"{', '.join(no_data_nodes[:3])}"
                        f"{'...' if len(no_data_nodes) > 3 else ''}"
                    ),
                ),
                alternatives=self.alternatives,
                pattern_match=pattern_match,
            )

        failure_summary = "; ".join(failed_nodes[:3])
        if len(failed_nodes) > 3:
            failure_summary += f" (and {len(failed_nodes) - 3} more)"

        return PatternRuntime(
            pattern_id=pattern_id,
            result=RuntimeTestResult(
                compile=all_compile,
                run=all_run,
                no_data=False,
                reason=f"Pattern '{pattern_name}' has unsupported operators: {failure_summary}",
            ),
            alternatives=self.alternatives,
            pattern_match=pattern_match,
        )
