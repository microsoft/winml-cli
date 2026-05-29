# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""RuntimeCheckerQuery - Query runtime database for pattern support."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
import pandas as pd
from onnx import numpy_helper

from ...onnx import (
    ONNXDomain,
    SupportedONNXType,
    infer_onnx_shapes,
    remove_optional_from_type_annotation,
)
from ...onnx.external_data import try_load_external_initializer_array
from ...pattern.base import (
    get_pattern_input_generator,
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
    encode_rule_condition_value_for_parquet,
    get_attribute_proto_value,
    get_op_input_properties,
    get_op_since_version,
    make_hashable,
    node_to_pattern_match,
    shape_and_dtype_from_valueinfo,
)
from ..utils.node_key_utils import build_node_key_by_node_id, resolve_stable_node_key
from ..utils.rule_loader import resolve_rule_parquet_path
from ..utils.timing_utils import make_timing_logger
from .node_checkers.base import NodeChecker
from .node_checkers.registry import NodeCheckerRegistry


logger = logging.getLogger(__name__)

_log_timing = make_timing_logger(logger)

_LOG_COLOR_LIGHT_CYAN = "\033[96m"
_LOG_COLOR_GREEN = "\033[92m"
_LOG_COLOR_RESET = "\033[0m"

if TYPE_CHECKING:
    from winml.modelkit.pattern.match import PatternMatchResult

    from ...utils.constants import EPName
    from .node_checkers.base import NodeChecker


QDQ_SUFFIX = " (QDQ)"


def _elapsed_ms(start_time: float) -> int:
    """Return elapsed milliseconds from `start_time` to now."""
    return int((time.perf_counter() - start_time) * 1000)


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
        # Bypass pandas .apply() overhead — iterate directly over the numpy array.
        raw = df[col].to_numpy()
        df[col] = [make_hashable(v) for v in raw]
    return df


@dataclass
class _ParquetConditionTree:
    """Pre-built condition tree for fast parquet row lookup."""

    condition_columns: list[str]
    root: dict[Any, Any]
    default_row_position: int | None = None


_TREE_ROW_POSITION = object()


def _extract_condition_columns(table_df: pd.DataFrame) -> list[str]:
    """Extract rule-condition columns from a parquet table."""
    output_cols = {
        "row_index",
        "rule_key",
        "compile_run_success",
        "compile_reason",
        "run_reason",
        "rule_row_count",
    }
    return [col for col in table_df.columns if col not in output_cols]


def _build_condition_tree(table_df: pd.DataFrame | None) -> _ParquetConditionTree | None:
    """Build a condition-column tree for constant-time row lookup."""
    if table_df is None:
        return None

    condition_columns = _extract_condition_columns(table_df)
    if table_df.empty:
        return _ParquetConditionTree(condition_columns=condition_columns, root={})

    if not condition_columns:
        return _ParquetConditionTree(
            condition_columns=condition_columns,
            root={},
            default_row_position=0,
        )

    root: dict[Any, Any] = {}
    duplicate_count = 0

    for position, (_, row) in enumerate(table_df.iterrows()):
        node = root
        for col in condition_columns:
            value = row[col]
            child = node.get(value)
            if not isinstance(child, dict):
                child = {}
                node[value] = child
            node = child

        if _TREE_ROW_POSITION in node:
            duplicate_count += 1
            continue
        node[_TREE_ROW_POSITION] = position

    if duplicate_count:
        logger.warning(
            "Found %d duplicate condition rows in parquet table; first occurrence wins.",
            duplicate_count,
        )

    return _ParquetConditionTree(condition_columns=condition_columns, root=root)


def _lookup_row_position_in_condition_tree(
    tree: _ParquetConditionTree | None,
    query_conditions: dict[str, Any],
) -> int | None:
    """Lookup row position by traversing condition tree with query values."""
    if tree is None:
        return None

    if not tree.condition_columns:
        return tree.default_row_position

    node: Any = tree.root
    for col in tree.condition_columns:
        if col not in query_conditions:
            return None
        if not isinstance(node, dict):
            return None
        node = node.get(query_conditions[col])
        if node is None:
            return None

    if not isinstance(node, dict):
        return None
    row_position = node.get(_TREE_ROW_POSITION)
    if isinstance(row_position, int):
        return row_position
    return None


@dataclass
class _ParquetTableCacheEntry:
    """Thread-safe global parquet cache entry."""

    is_loading: bool
    table_df: pd.DataFrame | None = None


_PARQUET_TABLE_GLOBAL_CACHE: dict[
    tuple[str, tuple[int, int] | None],
    _ParquetTableCacheEntry,
] = {}
_PARQUET_TABLE_GLOBAL_CACHE_COND = threading.Condition()


def _clear_global_parquet_table_cache() -> None:
    """Clear module-level parquet cache (used in tests)."""
    with _PARQUET_TABLE_GLOBAL_CACHE_COND:
        _PARQUET_TABLE_GLOBAL_CACHE.clear()
        _PARQUET_TABLE_GLOBAL_CACHE_COND.notify_all()


def _supports_ansi_log_color() -> bool:
    """Return True when ANSI colorized log text should be emitted."""
    if os.environ.get("NO_COLOR"):
        return False
    stream = getattr(sys, "stdout", None)
    return bool(stream and hasattr(stream, "isatty") and stream.isatty())


def _colorize_log_text(text: str, color_code: str) -> str:
    """Wrap log text with ANSI color when terminal supports it."""
    if not _supports_ansi_log_color():
        return text
    return f"{color_code}{text}{_LOG_COLOR_RESET}"


def _log_parquet_cache_hit(parquet_path: Path, scope: str) -> None:
    """Emit a light-blue log entry for parquet cache hits."""
    logger.info(
        _colorize_log_text(
            f"[parquet-cache] {parquet_path.name} hit cache ({scope})",
            _LOG_COLOR_LIGHT_CYAN,
        )
    )


def _log_parquet_load(parquet_path: Path) -> None:
    """Emit a green log entry when parquet is loaded from disk."""
    logger.info(_colorize_log_text(f"[parquet-cache] Load {parquet_path.name}", _LOG_COLOR_GREEN))


def _build_global_parquet_cache_key(parquet_path: Path) -> tuple[str, tuple[int, int] | None]:
    """Build global cache key from normalized path + file signature."""
    try:
        resolved = parquet_path.resolve(strict=False)
    except Exception:
        resolved = parquet_path

    try:
        stat = parquet_path.stat()
        signature: tuple[int, int] | None = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        signature = None

    return str(resolved).casefold(), signature


def _read_and_sanitize_parquet_table(parquet_path: Path) -> pd.DataFrame | None:
    """Read parquet table from disk and sanitize it for query matching."""
    if not parquet_path.exists():
        return None

    try:
        table_df = pd.read_parquet(parquet_path)
        table_df = table_df.replace({np.nan: None})
        return _sanitize_df(table_df)
    except Exception as e:
        logger.warning("Failed to read parquet rules %s: %s", parquet_path, e)
        return None


def _get_or_load_parquet_table_global(parquet_path: Path) -> pd.DataFrame | None:
    """Get parquet table from global cache, loading once per file signature."""
    global_cache_key = _build_global_parquet_cache_key(parquet_path)

    with _PARQUET_TABLE_GLOBAL_CACHE_COND:
        while True:
            existing = _PARQUET_TABLE_GLOBAL_CACHE.get(global_cache_key)
            if existing is None:
                _log_parquet_load(parquet_path)
                _PARQUET_TABLE_GLOBAL_CACHE[global_cache_key] = _ParquetTableCacheEntry(
                    is_loading=True,
                )
                break

            if not existing.is_loading:
                _log_parquet_cache_hit(parquet_path, scope="global")
                return existing.table_df

            _PARQUET_TABLE_GLOBAL_CACHE_COND.wait()

    table_df: pd.DataFrame | None = None
    try:
        table_df = _read_and_sanitize_parquet_table(parquet_path)
    finally:
        with _PARQUET_TABLE_GLOBAL_CACHE_COND:
            _PARQUET_TABLE_GLOBAL_CACHE[global_cache_key] = _ParquetTableCacheEntry(
                is_loading=False,
                table_df=table_df,
            )
            _PARQUET_TABLE_GLOBAL_CACHE_COND.notify_all()

    return table_df


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


def _build_table_filter_conditions(
    conditions: dict[str, Any],
    column_names: list[str],
    infinite_properties: list[str],
    error_context: str,
) -> dict[str, Any]:
    """Filter query conditions down to the columns used for table matching."""
    filter_conditions: dict[str, Any] = {}

    for key in column_names:
        if key not in infinite_properties:
            if key not in conditions:
                raise OpOptionalInputSupportError(
                    f"Match key '{key}' not found in conditions for {error_context}. "
                    f"Available: {_format_list_preview(conditions.keys())}"
                )
            filter_conditions[key] = conditions[key]

    return filter_conditions


def _build_query_signature(
    column_names: list[str],
    filter_conditions: dict[str, Any],
) -> tuple[Any, ...]:
    """Build query signature for result cache key.

    The signature must reflect only the columns used by the actual table filter.
    Columns omitted from ``filter_conditions`` (for example infinite properties)
    are intentionally excluded to avoid KeyError and keep cache keys aligned with
    query behavior.
    """
    return tuple(filter_conditions[col] for col in column_names if col in filter_conditions)


def _normalize_table_path(path_like: str | Path) -> str:
    """Normalize table path for debug output.

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


def _build_rules_debug_parquet_candidates(parquet_path: Path) -> list[Path]:
    """Build the canonical rules_debug parquet path mapped from a rule parquet."""
    parquet_name = parquet_path.name
    parent_dir = parquet_path.parent
    grandparent_dir = parent_dir.parent

    if grandparent_dir.name.lower() in {"rules", "runtime_check_rules"}:
        return [grandparent_dir.parent / "rules_debug" / parent_dir.name / parquet_name]

    return [grandparent_dir / "rules_debug" / parent_dir.name / parquet_name]


def _normalize_case_indices(value: Any) -> list[str] | None:
    """Normalize case_indices parquet payload to list[str]."""
    parsed: Any = None
    if isinstance(value, str):
        if not value:
            return None
        try:
            parsed = json.loads(value)
        except Exception:
            return None
    elif isinstance(value, (list, tuple)):
        parsed = list(value)
    else:
        return None

    if not isinstance(parsed, list):
        return None

    normalized: list[str] = []
    for item in parsed:
        if item in (None, ""):
            continue
        normalized.append(str(item))

    return normalized


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
    model_path: str | Path | None = None,
    model_base_dir: str | None = None,
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
        model_path: Optional path to the source ONNX model. Used to resolve
            external tensor sidecar files.
        model_base_dir: Backward-compatible directory hint for resolving
            external tensor data when model_path is unavailable.
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

    resolved_model_path: Path | None = None
    resolved_model_base_dir: str | None = None
    if model_path is not None:
        resolved_model_path = Path(model_path).resolve(strict=False)
        resolved_model_base_dir = str(resolved_model_path.parent)
    elif model_base_dir is not None:
        resolved_base = Path(model_base_dir).resolve(strict=False)
        resolved_model_base_dir = str(resolved_base)
        # Build a synthetic model path so helper logic can resolve sidecar files
        # relative to the provided base directory.
        resolved_model_path = resolved_base / "__model__.onnx"

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
        runtime_checker_op = get_runtime_checker_op(node.op_type, domain=domain.value)(schema)
    except KeyError:
        raise OpUnsupportedError(f"Node {node.op_type} is not supported") from None
    type_vars: dict[str, Any] = {}

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

    def _compute_dynamic_axes(
        shape: tuple[Any, ...] | list[Any] | None,
        is_constant: bool,
    ) -> tuple[int, ...]:
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
        shape: tuple[Any, ...] | list[Any] | None = None,
        value: Any | None = None,
    ) -> None:
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

    def _tensor_to_array_with_fallback(tensor: onnx.TensorProto) -> np.ndarray:
        try:
            return numpy_helper.to_array(tensor, base_dir=resolved_model_base_dir or "")
        except Exception as exc:
            if tensor.data_location != onnx.TensorProto.EXTERNAL:
                raise

            # Try bounded external-data loading first when we have model directory context.
            external_arr = try_load_external_initializer_array(tensor, resolved_model_path)
            if external_arr is not None:
                return external_arr

            try:
                np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
            except Exception:
                np_dtype = np.dtype(np.float32)

            shape = tuple(int(d) for d in tensor.dims)
            logger.warning(
                "External tensor data for '%s' could not be loaded from '%s': %s. "
                "Using zeros fallback with shape=%s dtype=%s.",
                tensor.name or "<unnamed>",
                resolved_model_base_dir or ".",
                exc,
                shape,
                np_dtype,
            )
            return np.zeros(shape, dtype=np_dtype)

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
                logger.debug(
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

            # External data initializers may be graph-only stubs without payload.
            # Keep shape information, but do not mark them constant unless sidecar
            # data can be materialized.
            if init.data_location == onnx.TensorProto.EXTERNAL and not init.raw_data:
                external_arr = try_load_external_initializer_array(init, resolved_model_path)
                if external_arr is not None:
                    update_conditions_(
                        conditions,
                        input_name,
                        is_variadic,
                        True,
                        external_arr.shape,
                        external_arr,
                    )
                else:
                    shape = tuple(init.dims) if init.dims is not None else None
                    update_conditions_(conditions, input_name, is_variadic, False, shape, None)
            else:
                arr = _tensor_to_array_with_fallback(init)
                update_conditions_(conditions, input_name, is_variadic, True, arr.shape, arr)
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
            arr = _tensor_to_array_with_fallback(const_tensor)
            update_conditions_(conditions, input_name, is_variadic, True, arr.shape, arr)
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
            shape_seq: list | tuple[int, ...] | None = None
            dtype = None
            if vi is not None:
                shape_seq, dtype = shape_and_dtype_from_valueinfo(vi)
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
            update_conditions_(conditions, input_name, is_variadic, is_constant, shape_seq, None)
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

    for tvar_name, dtypes in runtime_checker_op.type_var_dtypes_to_test.items():
        if tvar_name not in type_vars:
            type_vars[tvar_name] = dtypes[0].annotation  # use first dtype as default
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
        conditions[f"{input_name}_value"] = info.value
        conditions[f"{input_name}_is_none"] = False

    # Attributes (with attr_ prefix)
    for attr_name, attr_value in pattern_match.attributes.items():
        conditions[f"attr_{attr_name}"] = attr_value
        conditions[f"attr_{attr_name}_is_none"] = attr_value is None

    pattern_obj = pattern_match.skeleton_match_result.pattern
    assert hasattr(pattern_obj, "get_schema"), (
        f"Pattern {type(pattern_obj).__name__} does not provide get_schema()"
    )
    conditions["n_outputs"] = len(pattern_obj.get_schema().outputs)

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
        ep_name: EPName,
        device_type: str,
        model_path: str | Path | None = None,
        dynamic_axis_strict_mode: bool = False,
        node_key_by_node_id: dict[int, str] | None = None,
    ) -> None:
        """Initialize runtime checker query.

        Args:
            model_proto: ONNX model proto
            ep_name: Execution provider name
            device_type: Device type (e.g., "CPU", "GPU", "NPU")
            model_path: Optional source ONNX model path used to resolve
                external tensor data.
            dynamic_axis_strict_mode: If False (default), maps any dynamic axes to (0,)
                for matching against first_axis test data. If True, preserves exact
                dynamic axis indices.
            node_key_by_node_id: Optional sidecar map from id(node) to stable node key.
        """
        self.model_path = str(Path(model_path).resolve(strict=False)) if model_path else None
        self.model_base_dir = str(Path(self.model_path).parent) if self.model_path else None
        self.dynamic_axis_strict_mode = dynamic_axis_strict_mode

        inferred_model: onnx.ModelProto = model_proto
        # Try shape inference: standard ONNX first, then symbolic (onnxruntime)
        try:
            # Standard ONNX shape inference — uses temp file for models
            # with external data (avoids silent empty-graph result).
            standard_inferred = infer_onnx_shapes(model_proto)
            if standard_inferred is not None:
                inferred_model = standard_inferred

            # Then try to enhance with symbolic shape inference
            # if available which supports Microsoft domain
            try:
                from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

                symbolic_inferred = SymbolicShapeInference.infer_shapes(inferred_model)
                if symbolic_inferred is not None:
                    inferred_model = symbolic_inferred
            except Exception as e:
                # If symbolic shape inference fails, continue with standard inference result
                logger.debug(
                    f"Symbolic shape inference not available or "
                    f"failed: {e}. Using standard ONNX shape "
                    f"inference result."
                )
        except Exception as e:
            # If standard shape inference fails, use original model
            logger.debug(f"Shape inference failed: {e}. Using original model.")

        self.model_proto: onnx.ModelProto = inferred_model
        # Keep stable Python wrapper references for graph nodes so id(node)
        # mappings do not accidentally collide with new transient wrappers.
        self._graph_nodes: list[onnx.NodeProto] = list(self.model_proto.graph.node)
        if node_key_by_node_id is not None:
            self._node_key_by_node_id = dict(node_key_by_node_id)
        else:
            self._node_key_by_node_id = build_node_key_by_node_id(self._graph_nodes)

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
        # Cache of node results keyed by hashable table_filter_conditions
        self._node_result_cache: dict[Any, PatternRuntime] = {}
        # Per-op parquet rule table cache keyed by (op, domain, since, qdq)
        self._parquet_rule_table_cache: dict[tuple[str, str, int, bool], pd.DataFrame | None] = {}
        self._parquet_condition_tree_cache: dict[
            tuple[str, str, int, bool],
            _ParquetConditionTree | None,
        ] = {}
        # since_version cache keyed by (op, domain, model_opset)
        self._since_version_cache: dict[tuple[str, str, int], int] = {}
        # rules_debug cache keyed by resolved parquet path (casefold)
        self._rules_debug_case_index_rows_cache: dict[str, list[list[str] | None] | None] = {}
        self._rules_debug_file_cache: dict[str, str | None] = {}

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
        from ...utils.constants import DEVICE_TO_DEVICE_TYPE

        device_type_enum = DEVICE_TO_DEVICE_TYPE.get(self.device_type)
        if device_type_enum is None:
            self._ep_available_locally = False
            return False

        try:
            ep_devices = winml.get_registered_ep_devices()
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

    @staticmethod
    def _clone_node_proto(node: onnx.NodeProto) -> onnx.NodeProto:
        """Clone a node proto so extracted test models do not reuse graph objects."""
        cloned = onnx.NodeProto()
        cloned.CopyFrom(node)
        return cloned

    def _find_producer_node(self, tensor_name: str) -> onnx.NodeProto | None:
        """Return the node that produces a tensor, if any."""
        if not tensor_name:
            return None

        for candidate in self.model_proto.graph.node:
            if tensor_name in candidate.output:
                return candidate
        return None

    def _find_consumer_nodes(self, tensor_name: str) -> list[onnx.NodeProto]:
        """Return nodes that consume a tensor."""
        if not tensor_name:
            return []

        return [
            candidate for candidate in self.model_proto.graph.node if tensor_name in candidate.input
        ]

    def _build_opset_imports(
        self,
        nodes: list[onnx.NodeProto],
        fallback_op_domain: ONNXDomain,
        fallback_opset_version: int,
    ) -> list[onnx.OperatorSetIdProto]:
        """Build opset imports for an extracted runtime-test model."""
        opset_imports: list[onnx.OperatorSetIdProto] = []
        added_domains: set[str] = set()
        saw_non_default_domain = False

        def add_domain(domain_str: str, version: int) -> None:
            canonical_domain = "" if domain_str in {"", ONNXDomain.AI_ONNX.value} else domain_str
            if canonical_domain in added_domains:
                return

            added_domains.add(canonical_domain)
            effective_version = max(version, 7) if canonical_domain == "" else version
            opset_imports.append(onnx.helper.make_opsetid(canonical_domain, effective_version))

        for included_node in nodes:
            raw_domain = included_node.domain or ""
            try:
                node_domain = ONNXDomain.from_str(raw_domain)
                add_domain(node_domain.schema_domain, self.opset_versions.get(node_domain, 1))
                saw_non_default_domain = saw_non_default_domain or node_domain != ONNXDomain.AI_ONNX
            except ValueError:
                add_domain(raw_domain, 1)
                saw_non_default_domain = saw_non_default_domain or bool(raw_domain)

        if not opset_imports:
            add_domain(fallback_op_domain.schema_domain, fallback_opset_version)
            saw_non_default_domain = fallback_op_domain != ONNXDomain.AI_ONNX

        if saw_non_default_domain and "" not in added_domains:
            default_opset = self.opset_versions.get(ONNXDomain.AI_ONNX, 17)
            add_domain("", default_opset)

        return opset_imports

    def _build_runtime_test_model(
        self,
        node: onnx.NodeProto,
        op_domain: ONNXDomain,
        opset_version: int,
        include_adjacent_qdq: bool = False,
    ) -> onnx.ModelProto:
        """Build the model used for local EP fallback and failed-node artifacts.

        For QDQ operators, include adjacent DequantizeLinear/QuantizeLinear nodes
        so local runtime checks preserve the quantized context around the target op.
        """
        if not include_adjacent_qdq:
            return self._build_single_node_model(node, op_domain, opset_version)

        graph_inputs: list[onnx.ValueInfoProto] = []
        graph_initializers: list[onnx.TensorProto] = []
        graph_outputs: list[onnx.ValueInfoProto] = []
        pre_nodes: list[onnx.NodeProto] = []
        post_nodes: list[onnx.NodeProto] = []
        seen_inputs: set[str] = set()
        seen_initializers: set[str] = set()
        seen_outputs: set[str] = set()
        seen_pre_nodes: set[str] = set()
        seen_post_nodes: set[str] = set()

        def add_graph_source(name: str) -> None:
            if not name:
                return

            if name in self.initializers:
                init = self.initializers[name]
                if init.data_location == onnx.TensorProto.EXTERNAL:
                    if name in seen_inputs:
                        return

                    vi = self.valueinfo.get(name)
                    if vi is not None:
                        graph_inputs.append(vi)
                    else:
                        graph_inputs.append(
                            onnx.helper.make_tensor_value_info(
                                name,
                                init.data_type,
                                list(init.dims),
                            )
                        )
                    seen_inputs.add(name)
                else:
                    if name not in seen_initializers:
                        graph_initializers.append(init)
                        seen_initializers.add(name)
                return

            if name in self.constants:
                if name not in seen_initializers:
                    graph_initializers.append(self.constants[name])
                    seen_initializers.add(name)
                return

            vi = self.valueinfo.get(name)
            if vi is None:
                raise ValueError(f"Tensor '{name}' not found in valueinfo or initializers")
            if name not in seen_inputs:
                graph_inputs.append(vi)
                seen_inputs.add(name)

        def add_graph_output(name: str) -> None:
            if not name or name in seen_outputs:
                return

            vi = self.valueinfo.get(name)
            if vi is not None:
                graph_outputs.append(vi)
            else:
                graph_outputs.append(
                    onnx.helper.make_tensor_value_info(name, onnx.TensorProto.UNDEFINED, None)
                )
            seen_outputs.add(name)

        for inp_name in node.input:
            if not inp_name:
                continue

            producer = self._find_producer_node(inp_name)
            if producer is not None and producer.op_type == "DequantizeLinear":
                producer_key = producer.name or "|".join(producer.output)
                if producer_key not in seen_pre_nodes:
                    pre_nodes.append(self._clone_node_proto(producer))
                    seen_pre_nodes.add(producer_key)
                for producer_input in producer.input:
                    add_graph_source(producer_input)
                continue

            add_graph_source(inp_name)

        for out_name in node.output:
            if not out_name:
                continue

            quantize_consumers = [
                consumer
                for consumer in self._find_consumer_nodes(out_name)
                if consumer.op_type == "QuantizeLinear"
                and consumer.input
                and consumer.input[0] == out_name
            ]
            if quantize_consumers:
                for consumer in quantize_consumers:
                    consumer_key = consumer.name or "|".join(consumer.output)
                    if consumer_key not in seen_post_nodes:
                        post_nodes.append(self._clone_node_proto(consumer))
                        seen_post_nodes.add(consumer_key)
                    for consumer_input in consumer.input[1:]:
                        add_graph_source(consumer_input)
                    for consumer_output in consumer.output:
                        add_graph_output(consumer_output)
                continue

            add_graph_output(out_name)

        nodes = [*pre_nodes, self._clone_node_proto(node), *post_nodes]
        graph = onnx.helper.make_graph(
            nodes,
            f"runtime_test_{node.op_type}",
            graph_inputs,
            graph_outputs,
            initializer=graph_initializers,
        )

        model = onnx.helper.make_model(
            graph,
            opset_imports=self._build_opset_imports(nodes, op_domain, opset_version),
        )

        try:
            model = infer_onnx_shapes(model)
        except Exception as e:
            logger.debug("Shape inference failed for runtime-test model: %s", e)

        return model

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
                init = self.initializers[inp_name]
                # External-data initializers may not have accessible sidecar files
                # in graph-only model scenarios; expose them as runtime inputs.
                if init.data_location == onnx.TensorProto.EXTERNAL:
                    vi = self.valueinfo.get(inp_name)
                    if vi is not None:
                        graph_inputs.append(vi)
                    else:
                        graph_inputs.append(
                            onnx.helper.make_tensor_value_info(
                                inp_name,
                                init.data_type,
                                list(init.dims),
                            )
                        )
                else:
                    graph_initializers.append(init)
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

    def _generate_model_inputs(self, model: onnx.ModelProto) -> dict[str, np.ndarray]:
        """Generate dummy input data for a runtime-test model."""
        input_feed: dict[str, np.ndarray] = {}
        default_dim_size = 2  # Replace dynamic/unknown dims with this size
        initializer_names = {initializer.name for initializer in model.graph.initializer}

        for graph_input in model.graph.input:
            if graph_input.name in initializer_names:
                continue

            shape, dtype_str = shape_and_dtype_from_valueinfo(graph_input)
            if dtype_str is None:
                raise ValueError(f"Input '{graph_input.name}' has no dtype information")

            np_dtype = SupportedONNXType.from_annotation(dtype_str).np_type

            concrete_shape: tuple[int, ...]
            if shape is None:
                concrete_shape = (default_dim_size,)
            else:
                concrete_shape = tuple(
                    dim if isinstance(dim, int) and dim > 0 else default_dim_size for dim in shape
                )

            input_feed[graph_input.name] = np.zeros(concrete_shape, dtype=np_dtype)

        return input_feed

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
            # Skip regular initializers/constants - they are embedded in the model.
            # External-data initializers are modeled as runtime inputs.
            if inp_name in self.initializers:
                init = self.initializers[inp_name]
                if init.data_location != onnx.TensorProto.EXTERNAL:
                    continue

                try:
                    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
                except Exception:
                    np_dtype = np.dtype(np.float32)

                shape = tuple(int(d) for d in init.dims)
                input_feed[inp_name] = np.zeros(shape, dtype=np_dtype)
                continue

            if inp_name in self.constants:
                continue

            vi = self.valueinfo.get(inp_name)
            if vi is None:
                raise ValueError(
                    f"Input '{inp_name}' for node '{node.name}' ({node.op_type}) "
                    f"not found in valueinfo"
                )

            vi_shape, dtype_str = shape_and_dtype_from_valueinfo(vi)
            if dtype_str is None:
                raise ValueError(
                    f"Input '{inp_name}' for node '{node.name}' ({node.op_type}) "
                    f"has no dtype information"
                )

            # Convert dtype string to numpy dtype
            np_dtype = SupportedONNXType.from_annotation(dtype_str).np_type

            concrete_shape: tuple[int, ...]
            if vi_shape is None:
                # No shape info at all - use a simple 1D array
                concrete_shape = (default_dim_size,)
            else:
                # Replace dynamic dimensions (strings or None) with default size
                concrete_shape = tuple(
                    d if isinstance(d, int) and d > 0 else default_dim_size for d in vi_shape
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
        for_debug: bool = False,
        include_adjacent_qdq: bool = False,
        save_node_types: set[str] | None = None,
        conditions: Any | None = None,
    ) -> PatternRuntime | None:
        """Attempt to compile and run a node locally when rules are not found.

        If the local machine supports the target EP, builds a runtime-test model
        and runs compile/run checks using EPChecker.

        Args:
            node: The ONNX node to check.
            op_domain: The domain of the node's operator.
            opset_version: The opset version to use.
            pattern_match: Pattern match result for the node.
            node_tags: Collected tags for the node.
            fallback_reason: The original reason rules were not found.
            for_debug: Whether to include debug_details in result payload.
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
            model = self._build_runtime_test_model(
                node,
                op_domain,
                opset_version,
                include_adjacent_qdq=include_adjacent_qdq,
            )
            input_feed = self._generate_model_inputs(model)
        except Exception as e:
            logger.debug(
                "Failed to build runtime-test model for local EP check on %s (%s): %s",
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
            debug_details=(
                {
                    "source": "local_ep_check",
                    "fallback_reason": fallback_reason,
                    "op_type": node.op_type,
                    "node_name": node.name,
                    "domain": str(op_domain),
                    "opset_version": opset_version,
                    "table_path": "",
                    "table_file": "",
                }
                if for_debug
                else None
            ),
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

    def _maybe_save_failed_node_result(
        self,
        node: onnx.NodeProto,
        op_domain: ONNXDomain,
        opset_version: int,
        result: RuntimeTestResult,
        cache_key: Any,
        include_adjacent_qdq: bool = False,
        save_node_types: set[str] | None = None,
    ) -> None:
        """Save unsupported or partial node models without re-running result computation."""
        if result.no_data or result.compile:
            return

        save_types = save_node_types or set()
        is_unsupported = "unsupported" in save_types and not result.run
        is_partial = "partial" in save_types and result.run
        if not (is_unsupported or is_partial):
            return

        node_model = self._build_runtime_test_model(
            node,
            op_domain,
            opset_version,
            include_adjacent_qdq=include_adjacent_qdq,
        )
        self._save_failed_node(
            node,
            node_model,
            cache_key,
            name_suffix="unsupported" if is_unsupported else "partial",
        )

    def _get_op_since_version_cached(
        self,
        op_name: str,
        op_domain: ONNXDomain,
        model_opset_version: int,
    ) -> int:
        """Get and cache schema since_version for op/domain/opset."""
        cache_key = (op_name, op_domain.value, model_opset_version)
        cached = self._since_version_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            since_version = get_op_since_version(op_name, model_opset_version, op_domain.value)
        except Exception:
            since_version = model_opset_version

        self._since_version_cache[cache_key] = since_version
        return since_version

    def _load_parquet_rule_table(
        self,
        op_name: str,
        op_domain: ONNXDomain,
        op_since_version: int,
        is_qdq: bool,
    ) -> tuple[pd.DataFrame | None, Path, _ParquetConditionTree | None]:
        """Load per-op parquet rule table with cache.

        Returns:
            tuple[pd.DataFrame | None, Path, _ParquetConditionTree | None]:
                Loaded dataframe when available, otherwise None,
                the resolved parquet path used for lookup,
                and optional pre-built condition tree.
        """
        parquet_name = (
            f"{op_name}_{self.ep_name}_{self.device_type.upper()}_{op_domain.name}"
            f"_opset{op_since_version}{'_qdq' if is_qdq else ''}.parquet"
        )
        parquet_path = resolve_rule_parquet_path(parquet_name)

        cache_key = (op_name, op_domain.value, op_since_version, is_qdq)
        if cache_key in self._parquet_rule_table_cache:
            _log_parquet_cache_hit(parquet_path, scope="instance")
            return (
                self._parquet_rule_table_cache[cache_key],
                parquet_path,
                self._parquet_condition_tree_cache.get(cache_key),
            )

        table_df = _get_or_load_parquet_table_global(parquet_path)
        condition_tree = _build_condition_tree(table_df)
        self._parquet_rule_table_cache[cache_key] = table_df
        self._parquet_condition_tree_cache[cache_key] = condition_tree
        return table_df, parquet_path, condition_tree

    def _load_rules_debug_case_indices_by_row_position(
        self,
        parquet_path: Path,
        row_position: int | None,
    ) -> tuple[list[str] | None, str | None]:
        """Load case_index list from rules_debug parquet mapping using row position."""
        if row_position is None or row_position < 0:
            return None, None

        parquet_cache_key = str(parquet_path.resolve(strict=False)).casefold()

        if parquet_cache_key not in self._rules_debug_case_index_rows_cache:
            debug_parquet_path: Path | None = None
            for candidate in _build_rules_debug_parquet_candidates(parquet_path):
                if candidate.exists():
                    debug_parquet_path = candidate
                    break

            self._rules_debug_file_cache[parquet_cache_key] = (
                _normalize_table_path(debug_parquet_path) if debug_parquet_path is not None else None
            )

            if debug_parquet_path is None:
                self._rules_debug_case_index_rows_cache[parquet_cache_key] = None
            else:
                try:
                    debug_df = pd.read_parquet(debug_parquet_path)
                except Exception as e:
                    logger.debug(
                        "Failed to load rules_debug parquet from %s: %s",
                        debug_parquet_path,
                        e,
                    )
                    self._rules_debug_case_index_rows_cache[parquet_cache_key] = None
                else:
                    debug_df = debug_df.replace({np.nan: None})
                    case_rows: list[list[str] | None] = []
                    for _, row in debug_df.iterrows():
                        case_rows.append(_normalize_case_indices(row.get("case_indices_json")))

                    self._rules_debug_case_index_rows_cache[parquet_cache_key] = case_rows

        rules_debug_file = self._rules_debug_file_cache.get(parquet_cache_key)
        cached_rows = self._rules_debug_case_index_rows_cache.get(parquet_cache_key)
        if not isinstance(cached_rows, list):
            return None, rules_debug_file
        if row_position >= len(cached_rows):
            return None, rules_debug_file

        return cached_rows[row_position], rules_debug_file

    def _run_for_node_with_parquet_rules(
        self,
        node: onnx.NodeProto,
        op_domain: ONNXDomain,
        opset_version: int,
        conditions: dict[str, Any],
        infinite_properties: list[str],
        is_qdq: bool,
        node_tags: list[NodeTag],
        pattern_match: PatternMatchResult,
        pattern_id: str,
        for_debug: bool,
        run_unknown_op: bool,
        save_node_types: set[str] | None,
    ) -> PatternRuntime:
        """Run parquet-based per-op matching for a node."""
        total_start = time.perf_counter()
        since_version_ms: int | None = None
        load_table_ms: int | None = None
        build_filter_ms: int | None = None
        cache_lookup_ms: int | None = None
        row_lookup_ms: int | None = None
        local_fallback_ms: int | None = None
        maybe_save_ms: int | None = None

        def _finish(result: PatternRuntime, outcome: str, **extra: Any) -> PatternRuntime:
            _log_timing(
                "run_for_node.parquet",
                op=node.op_type,
                node=node.name or "<unnamed>",
                ep=self.ep_name,
                device=self.device_type,
                is_qdq=is_qdq,
                outcome=outcome,
                total_ms=_elapsed_ms(total_start),
                since_version_ms=since_version_ms,
                load_table_ms=load_table_ms,
                build_filter_ms=build_filter_ms,
                cache_lookup_ms=cache_lookup_ms,
                row_lookup_ms=row_lookup_ms,
                local_fallback_ms=local_fallback_ms,
                maybe_save_ms=maybe_save_ms,
                **extra,
            )
            return result

        since_version_start = time.perf_counter()
        op_since_version = self._get_op_since_version_cached(node.op_type, op_domain, opset_version)
        since_version_ms = _elapsed_ms(since_version_start)

        load_table_start = time.perf_counter()
        table_df, parquet_path, condition_tree = self._load_parquet_rule_table(
            node.op_type,
            op_domain,
            op_since_version,
            is_qdq,
        )
        load_table_ms = _elapsed_ms(load_table_start)
        parquet_file = parquet_path.name
        parquet_path_norm = _normalize_table_path(parquet_path)

        if table_df is None:
            if run_unknown_op:
                local_fallback_start = time.perf_counter()
                local_result = self._try_local_ep_check(
                    node,
                    op_domain,
                    opset_version,
                    pattern_match,
                    node_tags,
                    "rules_not_found",
                    for_debug=for_debug,
                    include_adjacent_qdq=is_qdq,
                    conditions=None,
                    save_node_types=save_node_types,
                )
                local_fallback_ms = _elapsed_ms(local_fallback_start)
                if local_result is not None:
                    return _finish(
                        local_result,
                        outcome="rules_not_found_local_fallback",
                        table_file=parquet_file,
                        op_since_version=op_since_version,
                        compile=local_result.result.compile,
                        run=local_result.result.run,
                        no_data=local_result.result.no_data,
                    )

            return _finish(
                PatternRuntime(
                    pattern_id=pattern_id,
                    result=RuntimeTestResult(
                        compile=False,
                        run=False,
                        no_data=True,
                        reason="rules_not_found",
                        node_tags=node_tags,
                        debug_details=(
                            {
                                "op_type": node.op_type,
                                "domain": str(op_domain),
                                "opset_version": opset_version,
                                "table_path": parquet_path_norm,
                                "table_file": parquet_file,
                                "op_since_version": op_since_version,
                            }
                            if for_debug
                            else None
                        ),
                    ),
                    alternatives=self.alternatives,
                    pattern_match=pattern_match,
                ),
                outcome="rules_not_found",
                table_file=parquet_file,
                op_since_version=op_since_version,
            )

        op_columns = condition_tree.condition_columns if condition_tree is not None else []
        build_filter_start = time.perf_counter()
        table_filter_conditions = _build_table_filter_conditions(
            conditions,
            op_columns,
            infinite_properties,
            f"op {node.op_type} (domain: {op_domain})",
        )
        parquet_filter_conditions = {
            k: encode_rule_condition_value_for_parquet(v)
            for k, v in table_filter_conditions.items()
        }
        # Defensive code: with the latest result_processor pipeline, parquet condition
        # columns are already filtered to exclude infinite properties, so every
        # op_column should normally be present here. We still build the signature
        # from available keys only to avoid KeyError if mixed/legacy artifacts appear.
        query_signature = _build_query_signature(op_columns, parquet_filter_conditions)
        build_filter_ms = _elapsed_ms(build_filter_start)

        cache_key = (node.op_type, op_domain.value, op_since_version, is_qdq, query_signature)
        cache_lookup_start = time.perf_counter()
        if cache_key in self._node_result_cache:
            cache_lookup_ms = _elapsed_ms(cache_lookup_start)
            cached = self._node_result_cache[cache_key]
            return _finish(
                PatternRuntime(
                    pattern_id=pattern_id,
                    result=cached.result,
                    alternatives=self.alternatives,
                    pattern_match=pattern_match,
                ),
                outcome="result_cache_hit",
                table_file=parquet_file,
                op_since_version=op_since_version,
                query_signature_size=len(query_signature),
            )
        cache_lookup_ms = _elapsed_ms(cache_lookup_start)

        matched_row = None
        row_lookup_start = time.perf_counter()
        row_position = _lookup_row_position_in_condition_tree(
            condition_tree,
            parquet_filter_conditions,
        )
        tree_hit = row_position is not None
        if row_position is not None:
            matched_row = table_df.iloc[row_position]
        else:
            ret = query_table_exact_match(table_df, parquet_filter_conditions)
            if not ret.empty:
                matched_row = ret.iloc[0]
        row_lookup_ms = _elapsed_ms(row_lookup_start)

        if matched_row is None:
            debug_details = None
            if for_debug:
                debug_steps: list[dict[str, Any]] = []
                current_df = table_df
                for col, value in parquet_filter_conditions.items():
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
                    "table_path": parquet_path_norm,
                    "table_file": parquet_file,
                    "op_since_version": op_since_version,
                    "lookup_columns": op_columns,
                    "query_signature": query_signature,
                }
                debug_details["steps"] = debug_steps

            if run_unknown_op:
                local_fallback_start = time.perf_counter()
                local_result = self._try_local_ep_check(
                    node,
                    op_domain,
                    opset_version,
                    pattern_match,
                    node_tags,
                    "properties_not_found",
                    for_debug=for_debug,
                    include_adjacent_qdq=is_qdq,
                    conditions=cache_key,
                    save_node_types=save_node_types,
                )
                local_fallback_ms = _elapsed_ms(local_fallback_start)
                if local_result is not None:
                    self._node_result_cache[cache_key] = local_result
                    return _finish(
                        local_result,
                        outcome="properties_not_found_local_fallback",
                        table_file=parquet_file,
                        op_since_version=op_since_version,
                        tree_hit=tree_hit,
                        compile=local_result.result.compile,
                        run=local_result.result.run,
                        no_data=local_result.result.no_data,
                    )

            result = RuntimeTestResult(
                compile=False,
                run=False,
                no_data=True,
                reason="properties_not_found",
                filter=str(table_filter_conditions),
                node_tags=node_tags,
                debug_details=debug_details,
            )
            pattern_runtime = PatternRuntime(
                pattern_id=pattern_id,
                result=result,
                alternatives=self.alternatives,
                pattern_match=pattern_match,
            )
            self._node_result_cache[cache_key] = pattern_runtime
            return _finish(
                pattern_runtime,
                outcome="properties_not_found",
                table_file=parquet_file,
                op_since_version=op_since_version,
                tree_hit=tree_hit,
                query_signature_size=len(query_signature),
            )

        row = matched_row
        compile_run = row.get("compile_run_success", (False, False))
        compile_result = bool(compile_run[0])
        run_result = bool(compile_run[1])

        matched_row_position: int | None = row_position
        if matched_row_position is None:
            try:
                matched_row_position = int(row.name)
            except Exception:
                matched_row_position = None

        matched_case_indices_raw, rules_debug_file = (
            self._load_rules_debug_case_indices_by_row_position(parquet_path, matched_row_position)
        )
        matched_case_indices: list[str] | None = None
        if isinstance(matched_case_indices_raw, list):
            matched_case_indices = matched_case_indices_raw

        reason_text = ""
        if not compile_result or not run_result:
            if matched_row_position is not None:
                reason_text = f"matched_rule={parquet_file}#row={matched_row_position}"
            else:
                reason_text = f"matched_rule={parquet_file}"

        debug_payload: dict[str, Any] | None = None
        if for_debug:
            debug_payload = {
                "table_path": parquet_path_norm,
                "table_file": parquet_file,
                "op_since_version": op_since_version,
                "lookup_columns": op_columns,
                "query_signature": query_signature,
                "matched_rule_row": matched_row_position,
                "rules_debug_file": rules_debug_file,
                "matched_case_indices": matched_case_indices,
                "matched_first_case_index": (
                    matched_case_indices[0] if matched_case_indices else None
                ),
                "matched_case_count": (
                    len(matched_case_indices) if matched_case_indices is not None else None
                ),
            }

        result = RuntimeTestResult(
            compile=compile_result,
            run=run_result,
            reason=reason_text,
            no_data=False,
            node_tags=node_tags,
            debug_details=debug_payload,
        )

        maybe_save_start = time.perf_counter()
        self._maybe_save_failed_node_result(
            node,
            op_domain,
            opset_version,
            result,
            cache_key,
            include_adjacent_qdq=is_qdq,
            save_node_types=save_node_types,
        )
        maybe_save_ms = _elapsed_ms(maybe_save_start)

        pattern_runtime = PatternRuntime(
            pattern_id=pattern_id,
            result=result,
            alternatives=self.alternatives,
            pattern_match=pattern_match,
        )
        self._node_result_cache[cache_key] = pattern_runtime
        return _finish(
            pattern_runtime,
            outcome="matched",
            table_file=parquet_file,
            op_since_version=op_since_version,
            tree_hit=tree_hit,
            query_signature_size=len(query_signature),
            compile=compile_result,
            run=run_result,
        )

    def run_for_node(
        self,
        node: onnx.NodeProto,
        for_debug: bool = False,
        run_unknown_op: bool = False,
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
        total_start = time.perf_counter()
        pattern_match_ms: int | None = None
        collect_tags_ms: int | None = None
        domain_resolve_ms: int | None = None
        custom_checker_ms: int | None = None
        custom_checker_name: str | None = None
        conditions_ms: int | None = None
        parquet_rules_ms: int | None = None
        node_key = resolve_stable_node_key(
            node,
            node_key_by_node_id=self._node_key_by_node_id,
            graph_nodes=self._graph_nodes,
            unknown_unnamed_error=(
                "Cannot resolve stable key for unnamed node outside "
                "RuntimeCheckerQuery model graph."
            ),
        )

        pattern_match_start = time.perf_counter()
        pattern_match = node_to_pattern_match(node, node_key)
        pattern_match_ms = _elapsed_ms(pattern_match_start)

        def _finish(result: PatternRuntime, outcome: str) -> PatternRuntime:
            _log_timing(
                "run_for_node",
                op=node.op_type,
                node=node_key,
                ep=self.ep_name,
                device=self.device_type,
                pattern_id=result.pattern_id,
                outcome=outcome,
                total_ms=_elapsed_ms(total_start),
                pattern_match_ms=pattern_match_ms,
                collect_tags_ms=collect_tags_ms,
                domain_resolve_ms=domain_resolve_ms,
                custom_checker_ms=custom_checker_ms,
                custom_checker=custom_checker_name,
                conditions_ms=conditions_ms,
                parquet_rules_ms=parquet_rules_ms,
                compile=result.result.compile,
                run=result.result.run,
                no_data=result.result.no_data,
                reason=result.result.reason or "",
            )
            return result

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
            return _finish(
                PatternRuntime(
                    pattern_id=pattern_match.pattern.pattern_id,
                    result=RuntimeTestResult(
                        run=True,
                        compile=True,
                        no_data=False,
                        debug_details=None,
                    ),
                    alternatives=self.alternatives,
                    pattern_match=pattern_match,
                ),
                outcome="ignored_op",
            )

        # Collect all tags for this node
        collect_tags_start = time.perf_counter()
        node_tags = self._collect_node_tags(node)
        collect_tags_ms = _elapsed_ms(collect_tags_start)

        # If all inputs are constant, short-circuit with success
        if NodeTag.ALL_INPUTS_CONSTANT in node_tags:
            logger.warning("Op %s (%s) has all inputs constant", node.name, node.op_type)
            return _finish(
                PatternRuntime(
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
                ),
                outcome="all_inputs_constant",
            )

        domain_resolve_start = time.perf_counter()
        try:
            op_domain = ONNXDomain.from_str(node.domain)
        except ValueError:
            domain_resolve_ms = _elapsed_ms(domain_resolve_start)
            # Unknown domain (e.g., custom ops) — report as no_data
            return _finish(
                PatternRuntime(
                    pattern_id=pattern_match.pattern.pattern_id,
                    result=RuntimeTestResult(
                        run=False,
                        compile=False,
                        no_data=True,
                        reason=f"unsupported_domain:{node.domain}",
                        debug_details=None,
                    ),
                    alternatives=self.alternatives,
                    pattern_match=pattern_match,
                ),
                outcome="unsupported_domain",
            )
        domain_resolve_ms = _elapsed_ms(domain_resolve_start)

        # Determine the opset version based on domain (default to 1 if not in model)
        opset_version = self.opset_versions.get(op_domain, 1)

        # Evaluate custom checkers (before rule-based checks — handles EPContext, etc.)
        custom_checker_start = time.perf_counter()
        for checker in self.node_checkers:
            if checker.can_check(node, op_domain, opset_version):
                custom_checker_ms = _elapsed_ms(custom_checker_start)
                custom_checker_name = checker.__class__.__name__
                return _finish(
                    checker.check(
                        node,
                        op_domain,
                        opset_version,
                        pattern_match,
                        self.alternatives,
                        ep_name=self.ep_name,
                    ),
                    outcome="custom_checker",
                )
        custom_checker_ms = _elapsed_ms(custom_checker_start)

        # Phase 1: Extract conditions to determine if node is QDQ
        is_qdq = False

        def get_pattern_id(is_qdq: bool) -> str:
            return (
                pattern_match.pattern.pattern_id + QDQ_SUFFIX
                if is_qdq
                else pattern_match.pattern.pattern_id
            )

        conditions_start = time.perf_counter()
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
                model_path=self.model_path,
                dynamic_axis_strict_mode=self.dynamic_axis_strict_mode,
            )
        except (
            OpOptionalInputSupportError,
            OpLackOfRequiredInformationError,
            OpUnsupportedError,
        ) as e:
            conditions_ms = _elapsed_ms(conditions_start)
            exception_type = type(e).__name__
            logger.error(
                "%s caught for op %s (node: %s): %s",
                exception_type,
                node.op_type,
                node.name,
                str(e),
            )
            return _finish(
                PatternRuntime(
                    pattern_id=get_pattern_id(is_qdq),
                    result=RuntimeTestResult(
                        compile=False,
                        run=False,
                        no_data=True,
                        reason="optional_input_properties_not_found",
                        node_tags=node_tags,
                        debug_details=(
                            {
                                "op_type": node.op_type,
                                "node_name": node.name,
                                "error_message": str(e),
                                "table_path": "",
                                "table_file": "",
                            }
                            if for_debug
                            else None
                        ),
                    ),
                    alternatives=self.alternatives,
                    pattern_match=pattern_match,
                ),
                outcome="conditions_error",
            )
        conditions_ms = _elapsed_ms(conditions_start)

        pattern_id = get_pattern_id(is_qdq)
        parquet_rules_start = time.perf_counter()
        final_result = self._run_for_node_with_parquet_rules(
            node,
            op_domain,
            opset_version,
            conditions,
            infinite_properties,
            is_qdq,
            node_tags,
            pattern_match,
            pattern_id,
            for_debug,
            run_unknown_op,
            save_node_types,
        )
        parquet_rules_ms = _elapsed_ms(parquet_rules_start)
        return _finish(final_result, outcome="parquet_rules")

    def run_for_subgraph(
        self,
        pattern_match: PatternMatchResult,
        run_unknown_op: bool = False,
    ) -> PatternRuntime:
        """Run runtime check for subgraph pattern via per-node checks."""
        pattern_name = pattern_match.pattern.__class__.__name__
        logger.info(
            "Pattern-level aggregated rules are removed; checking individual operators for '%s'",
            pattern_name,
        )
        return self._run_for_subgraph_per_node(
            pattern_match,
            pattern_name,
            run_unknown_op,
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
                    debug_details=None,
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
                    debug_details=None,
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
                    debug_details=None,
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
                    debug_details=None,
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
                debug_details=None,
            ),
            alternatives=self.alternatives,
            pattern_match=pattern_match,
        )
