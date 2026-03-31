# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import base64
import hashlib
import itertools
import json
import time
import zlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
from colorama import Fore, Style
from onnx.defs import OpSchema

from winml.modelkit.pattern.utils import get_op_input_properties

from ...onnx import ONNXDomain, SupportedONNXType
from .qdq_gen import QDQGenerator


if TYPE_CHECKING:
    from winml.modelkit.analyze.runtime_checker.ep_checker import EPChecker
    from winml.modelkit.analyze.runtime_checker.runner import ResilientRunner


def model_bytes_to_b64(model_bytes: bytes) -> str:
    """Compress and base64-encode ONNX model bytes for JSON storage."""
    return base64.b64encode(zlib.compress(model_bytes)).decode("utf-8")


def model_from_b64(model_b64: str) -> bytes:
    """Decode and decompress base64 payload back to raw ONNX model bytes."""
    return zlib.decompress(base64.b64decode(model_b64))


# Registry for operator input generators
_OP_INPUT_GENERATOR_REGISTRY: dict[str, type["OpInputGenerator"]] = {}

# Error message pattern for QNN EP setup failure
_QNN_EP_SETUP_FAILURE_PATTERN = "Failed to setup so cleaning up"


def _check_qnn_ep_setup_failure(result: dict) -> None:
    """Check for critical QNN EP setup failure and raise if detected.

    Args:
        result: Result dict from runner.run() containing stdout/stderr

    Raises:
        RuntimeError: If QNN EP setup failure is detected
    """
    if result.get("stderr") and _QNN_EP_SETUP_FAILURE_PATTERN in result["stderr"]:
        raise RuntimeError(
            "CRITICAL ERROR: QNN EP setup failure detected. "
            "Please restart your computer and try again."
        )


def register_runtime_checker_op(cls: type["OpInputGenerator"]) -> type["OpInputGenerator"]:
    """Decorator to register an OpInputGenerator class by its op_name.

    Usage:
        @register_runtime_checker_op
        class AbsInputGenerator(OpInputGenerator):
            op_name = "Abs"
            ...
    """
    if not hasattr(cls, "op_name"):
        raise ValueError(
            f"Class {cls.__name__} must have an 'op_name' class attribute to be registered"
        )
    op_name = cls.op_name
    if op_name in _OP_INPUT_GENERATOR_REGISTRY:
        raise ValueError(
            f"Operator '{op_name}' is already registered by "
            f"{_OP_INPUT_GENERATOR_REGISTRY[op_name].__name__}"
        )
    _OP_INPUT_GENERATOR_REGISTRY[op_name] = cls
    return cls


def get_runtime_checker_op(op_name: str) -> type["OpInputGenerator"]:
    """Get registered OpInputGenerator class by operator name.

    Args:
        op_name: The ONNX operator name (e.g., "Abs", "Relu")

    Returns:
        The OpInputGenerator class for the specified operator

    Raises:
        KeyError: If no generator is registered for the operator
    """
    if op_name not in _OP_INPUT_GENERATOR_REGISTRY:
        raise KeyError(
            f"No OpInputGenerator registered for operator '{op_name}'. "
            f"Available operators: {sorted(_OP_INPUT_GENERATOR_REGISTRY.keys())}"
        )
    return _OP_INPUT_GENERATOR_REGISTRY[op_name]


def get_registered_operators() -> list[str]:
    """Get list of all registered operator names.

    Returns:
        Sorted list of registered operator names
    """
    return sorted(_OP_INPUT_GENERATOR_REGISTRY.keys())


class InputConstraint(ABC):
    """Abstract base class for op input constraints.

    Note that attributes shall not use InputConstraint, and their values
    must be directly specified in input combinations.
    """

    @abstractmethod
    def get_value(self, type_annotation: str) -> Any:
        """Get the value for this constraint."""
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert constraint to JSON-serializable dictionary."""
        ...


class InputValueConstraint(InputConstraint):
    """Constraint on the value of the input tensor.

    get_value() will return the specified value, ignoring type_annotation.
    Use InputValueConstraint(None) for optional inputs that are omitted in the input.
    """

    def __init__(self, value: Any) -> None:
        self.value = value

    def get_value(self, type_annotation: str = "") -> Any:
        """Return the constraint value."""
        # TODO: current workaround is to cast numpy arrays to the correct type
        # based on type_annotation; maybe use covariant typing for InputValueConstraint
        # to handle this cleanly
        if isinstance(self.value, np.ndarray):
            np_dtype = SupportedONNXType.from_annotation(type_annotation).np_type
            return self.value.astype(np_dtype)
        return self.value

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        if isinstance(self.value, np.ndarray):
            flat = self.value.ravel()
            if flat.size > 0 and np.all(flat == flat[0]):
                return {
                    "type": "value",
                    "same_value": flat[0].item(),
                    "same_value_shape": list(self.value.shape),
                    "dtype": str(self.value.dtype),
                }
            return {
                "type": "value",
                "value": self.value.tolist(),  # Nested list structure reflects shape
                "dtype": str(self.value.dtype),
            }
        # Check if value is JSON-serializable (primitives)
        if isinstance(self.value, (dict, list, str, int, float, bool, type(None))):
            return {
                "type": "value",
                "value": self.value,
            }
        # Unsupported type
        msg = (
            f"Cannot serialize InputValueConstraint with value type "
            f"{type(self.value).__name__}. "
            f"Supported types: np.ndarray, dict, list, str, int, float, bool, None"
        )
        raise TypeError(msg)


def normalize_constraint_dict(c: dict) -> dict:
    """Expand same_value/same_value_shape back to the canonical value list form.

    Converts the compact representation produced by InputValueConstraint.to_dict()
    when all values are equal, back to the full nested value list. Use this when
    consuming serialized constraint dicts to ensure consistent handling regardless
    of which representation was saved.
    """
    if "same_value" in c and "same_value_shape" in c:
        normalized = {k: v for k, v in c.items() if k not in ("same_value", "same_value_shape")}
        dtype = np.dtype(c["dtype"]) if "dtype" in c else None
        normalized["value"] = np.full(c["same_value_shape"], c["same_value"], dtype=dtype).tolist()

        return normalized
    return c


class InputShapeConstraint(InputConstraint):
    """Constraint on shape of the input tensor.

    get_value() will return a tensor with the specified shape and type_annotation.
    """

    def __init__(self, shape: Sequence[int], min_max: tuple[Any, Any] | None = None) -> None:
        self.shape = shape
        self.min_max = min_max
        if min_max is not None:
            min_val, max_val = min_max
            assert min_val is not None, "min_val must be specified in min_max"
            assert max_val is not None, "max_val must be specified in min_max"
            assert min_val <= max_val, "min_val must be less than or equal to max_val in min_max"

    def get_value(self, type_annotation: str) -> Any:
        """Generate a tensor with the specified shape and type."""
        # random values may cause runtime errors when running an op
        np_dtype = SupportedONNXType.from_annotation(type_annotation).np_type
        seed_material = json.dumps(
            {
                "shape": list(self.shape),
                "min_max": self.min_max,
                "type": str(np_dtype),
            },
            sort_keys=True,
        ).encode("utf-8")
        seed_int = int(hashlib.sha256(seed_material).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed_int)

        if np_dtype == np.bool_:
            return rng.choice([True, False], size=self.shape)

        # Use min_max if provided (skip for bool)
        if self.min_max is not None:
            min_val, max_val = self.min_max
            if np.issubdtype(np_dtype, np.integer):
                return rng.integers(min_val, max_val + 1, size=self.shape, dtype=np_dtype)
            return rng.uniform(min_val, max_val, size=self.shape).astype(np_dtype)

        # Default behavior when min_max is None
        if np.issubdtype(np_dtype, np.integer):
            return rng.integers(0, 2, size=self.shape, dtype=np_dtype)
        return rng.random(self.shape).astype(np_dtype)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "type": "shape",
            "shape": list(self.shape),
            "min_max": self.min_max,
        }


class VariadicInputConstraint(InputConstraint):
    """Constraint for variadic inputs (list of tensors)."""

    def __init__(self, element_constraints: list[InputConstraint]) -> None:
        self.element_constraints = element_constraints

    def get_value(self, type_annotation: str) -> Any:
        """Generate a list of tensors based on the element constraints."""
        return [
            constraint.get_value(type_annotation=type_annotation)
            for constraint in self.element_constraints
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "type": "variadic",
            "elements": [constraint.to_dict() for constraint in self.element_constraints],
        }


class QDQParameterConfig:
    """Configuration for QDQ parameter support on operator inputs."""

    def __init__(
        self,
        support_weight: bool = False,
        support_activation: bool = False,
        support_non_qdq: bool = False,
        qdq_types: list[SupportedONNXType] | None = None,
    ):
        """Initialize QDQParameterConfig.

        qdq_types, when set, specifies the exact quantization types to yield for this parameter.
        qdq_types is yielded as a distinct combination.
        """
        self.support_weight = support_weight
        self.support_activation = support_activation
        self.support_non_qdq = support_non_qdq
        self.qdq_types = qdq_types
        assert (
            self.support_activation
            or self.support_weight
            or self.support_non_qdq
            or self.qdq_types is not None
        ), (
            "At least one of support_weight, support_activation, "
            "support_non_qdq, or qdq_types must be set"
        )


class OpInputGenerator(ABC):
    """Base class for generating test inputs for ONNX operators."""

    op_name: str  # op_name must be defined in subclasses specific to a given op
    expand_optionals: bool = (
        True  # Can be overridden by subclasses to disable optional input/attribute expansion
    )
    replace_float_with_dummy_in_query: bool = True

    type_vars_key = "type_vars"
    # A dictionary with schema_name to QDQ dtype
    qdq_types_key = "qdq_types"

    def __init__(
        self,
        schema: OpSchema,
        onnx_types_to_check: Sequence[str] | None = None,
        qdq_generator: QDQGenerator | None = None,
        dynamic_axis_mode: str = "none",
        runner: "ResilientRunner | None" = None,
        ep_checker: "EPChecker | None" = None,
    ) -> None:
        """Initialize OpInputGenerator with an ONNX OpSchema.

        Args:
            schema: ONNX OpSchema for the operator
            onnx_types_to_check: Optional list of ONNX type annotations to test.
                                If None, all supported types are tested.
            qdq_generator: Optional QDQ generator for quantized model generation.
            dynamic_axis_mode: Controls dynamic axis testing. One of:
                - "none": No dynamic axes (default, preserves existing behavior).
                - "first_axis_dynamic": Test with first axis (axis 0) as dynamic
                  for all non-constant, non-scalar inputs.
                - "first_axis_combinations": Iterate over
                  {fixed_shape|first_axis_is_dynamic} for each
                  non-constant and non-scalar input.
                  Not yet implemented.
                - "all_axes_combinations": Iterate over all axis
                  subsets for each non-constant and non-scalar
                  input. Not yet implemented.
            runner: Optional ResilientRunner for EP runtime checking. Injected by static_analyzer.
            ep_checker: Optional EPChecker for EP validation. Injected by static_analyzer.
        """
        assert dynamic_axis_mode in (
            "none",
            "first_axis_dynamic",
            "first_axis_combinations",
            "all_axes_combinations",
        )
        if dynamic_axis_mode in ("first_axis_combinations", "all_axes_combinations"):
            raise NotImplementedError(
                f"dynamic_axis_mode='{dynamic_axis_mode}' is not yet implemented"
            )
        self.dynamic_axis_mode = dynamic_axis_mode

        self.schema = schema
        self.qdq_generator = qdq_generator

        # Validate that schema matches op_name
        # for patterns, one schema may correspond to multiple patterns so the check is skipped
        if isinstance(self.schema, OpSchema):
            assert self.schema.name == self.op_name, (
                f"Schema name '{self.schema.name}' does not match op_name '{self.op_name}'"
            )

        self.onnx_types_to_check = (
            {SupportedONNXType.from_annotation(t).onnx_type for t in onnx_types_to_check}
            if onnx_types_to_check is not None
            else {x.onnx_type for x in SupportedONNXType}
        )

        output_only_type_vars = {x.type_str for x in self.schema.outputs} - {
            x.type_str for x in self.schema.inputs
        }

        self.type_var_dtypes_to_test = {
            # legacy compatibility: adding _op_name suffix
            f"{constraint.type_param_str}_{self.op_name}": list(
                map(
                    SupportedONNXType.from_onnx_type,
                    filter(lambda x: x in self.onnx_types_to_check, constraint.allowed_type_strs),
                )
            )
            for constraint in self.schema.type_constraints
            if constraint.type_param_str not in output_only_type_vars
        }

        self.type_vars_with_unique_dtypes = {
            f"{constraint.type_param_str}_{self.op_name}": SupportedONNXType.from_onnx_type(
                constraint.allowed_type_strs[0]
            )
            for constraint in self.schema.type_constraints
            if len(constraint.allowed_type_strs) == 1
        }

        (
            self.op_input_names,
            self.op_variadic_input_name,
            self.op_attribute_names,
            self.type_annotations,
        ) = get_op_input_properties(self.schema)

        # Identify optional inputs from schema
        self.optional_input_names = [
            input_param.name
            for input_param in self.schema.inputs
            if input_param.option == OpSchema.FormalParameterOption.Optional
        ]

        # Identify optional attributes without default values from schema
        # These are attributes that can be omitted but have no explicit default in the schema
        # (the runtime infers them from input shapes or uses implicit defaults)
        self.optional_attrs_without_defaults = [
            attr_name
            for attr_name in self.schema.attributes
            if not self.schema.attributes[attr_name].required
            and not self._attr_has_default(self.schema.attributes[attr_name])
        ]

        # Constructor-injected dependencies for EP runtime checking (optional)
        self._runner = runner
        self._ep_checker = ep_checker

    @staticmethod
    def _attr_has_default(attr_info: Any) -> bool:
        """Check if an attribute has an explicit default value in the schema.

        Args:
            attr_info: Attribute info from schema.attributes[attr_name]

        Returns:
            True if the attribute has a non-empty default value
        """
        default_val = attr_info.default_value
        return default_val.ByteSize() > 0 if default_val else False

    def _type_var_combination_iter(self) -> Any:
        options = [
            [
                (name, value.annotation) for value in dtypes
            ]  # legacy compatibility: using annotation of dtype
            for name, dtypes in self.type_var_dtypes_to_test.items()
        ]
        for type_var_comb in itertools.product(*options):
            yield dict(type_var_comb)

    def _apply_type_var_combination(
        self, type_annotation: str, type_var_comb: dict[str, str]
    ) -> str:
        for type_var, dtype in type_var_comb.items():
            type_annotation = type_annotation.replace(type_var, dtype)
        return type_annotation

    def _finite_attribute_combination_iter(self) -> Any:
        finite_attribute_sets = self.get_finite_attribute_sets()
        options = [
            [(name, value) for value in values] for name, values in finite_attribute_sets.items()
        ]
        for attr_comb in itertools.product(*options):
            # Omit attributes with None value to simulate them being not provided
            yield {k: v for k, v in attr_comb if v is not None}

    def _optional_input_combination_iter(self, input_comb: dict[str, InputConstraint]) -> Any:
        """Iterate over combinations of optional inputs being provided or None.

        For each optional input present in input_comb, generates combinations where
        that input is either provided (original value) or None (not provided).

        Args:
            input_comb: Original input combination dict

        Yields:
            Modified input_comb dicts with optional inputs set to None in various combinations
        """
        # If expand_optionals is disabled, just yield the original
        if not self.expand_optionals:
            yield input_comb
            return

        # Find optional inputs that are present in this input_comb
        optional_inputs_in_comb = [name for name in self.optional_input_names if name in input_comb]

        if not optional_inputs_in_comb:
            # No optional inputs in this combination, just yield the original
            yield input_comb
            return

        # Generate all combinations: for each optional input, either keep value or set to None
        # Options: [(name, True), (name, False)] where True = keep value, False = set to None
        options = [[(name, True), (name, False)] for name in optional_inputs_in_comb]

        for comb in itertools.product(*options):
            use_value_map = dict(comb)
            modified_input_comb = {}
            for k, v in input_comb.items():
                if k in use_value_map:
                    if use_value_map[k]:
                        # Keep the original value
                        modified_input_comb[k] = v
                    else:
                        # Set to None (optional input not provided)
                        modified_input_comb[k] = None
                else:
                    # Non-optional input, keep as is
                    modified_input_comb[k] = v
            yield modified_input_comb

    def _optional_attr_combination_iter(self, input_comb: dict[str, InputConstraint]) -> Any:
        """Iterate over combinations of optional attributes being provided or omitted.

        Only covers attributes without defaults.

        For each optional attribute without a default value that is present in input_comb,
        generates combinations where that attribute is either provided (original value) or omitted.

        Args:
            input_comb: Original input combination dict (may contain attributes)

        Yields:
            Modified input_comb dicts with optional attrs removed in various combinations
        """
        # If expand_optionals is disabled, just yield the original
        if not self.expand_optionals:
            yield input_comb
            return

        # Find optional attrs without defaults that are present in this input_comb
        optional_attrs_in_comb = [
            name for name in self.optional_attrs_without_defaults if name in input_comb
        ]

        if not optional_attrs_in_comb:
            # No optional attrs in this combination, just yield the original
            yield input_comb
            return

        # Generate all combinations: for each optional attr, either keep value or omit
        # Options: [(name, True), (name, False)] where True = keep value, False = omit
        options = [[(name, True), (name, False)] for name in optional_attrs_in_comb]

        for comb in itertools.product(*options):
            use_value_map = dict(comb)
            modified_input_comb = {}
            for k, v in input_comb.items():
                if k in use_value_map:
                    if use_value_map[k]:
                        # Keep the original value
                        modified_input_comb[k] = v
                    # else: omit this attribute entirely (don't add to dict)
                else:
                    # Non-optional attr or input, keep as is
                    modified_input_comb[k] = v
            yield modified_input_comb

    def filter_kwargs_by_opset(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Filter kwargs to only those supported by the operator schema.

        This is useful when validating inputs by creating ONNX models,
        as some kwargs may not be supported in certain schema versions.
        """
        keys_supported = set(self.op_input_names + self.op_attribute_names)

        # For variadic inputs, also include expanded names like "inputs__0", "inputs__1", etc.
        if self.op_variadic_input_name is not None:
            variadic_prefix = f"{self.op_variadic_input_name}__"
            return {
                k: v
                for k, v in kwargs.items()
                if k in keys_supported
                or k.startswith(variadic_prefix)
                or k == self.op_variadic_input_name  # for both kwargs and input_constraints
            }

        return {k: v for k, v in kwargs.items() if k in keys_supported}

    def _is_input_key(self, key: str) -> bool:
        """Check if a key is an input (including expanded variadic inputs).

        Args:
            key: The key to check

        Returns:
            True if the key is a regular input or an expanded variadic input
        """
        if key in self.op_input_names:
            return True
        if self.op_variadic_input_name is not None:
            # Treat the base variadic name (e.g., "inputs") as an input before expansion
            if key == self.op_variadic_input_name:
                return True
            variadic_prefix = f"{self.op_variadic_input_name}__"
            if key.startswith(variadic_prefix):
                return True
        return False

    def _iter_constant_combinations(self, kwargs: dict[str, Any]) -> Any:
        """Iterate over different is_constant configurations for inputs.

        Yields all valid combinations of which inputs should be constants vs graph inputs.
        At least one input must be non-constant.

        Args:
            kwargs: Operator inputs and attributes

        Yields:
            Dict mapping input_name -> is_constant (bool) or None (for optional inputs not provided)
        """
        # Use _is_input_key to identify inputs (including expanded variadic inputs)
        # For None values (optional inputs not provided), set is_constant to True
        none_inputs = {k: True for k, v in kwargs.items() if self._is_input_key(k) and v is None}
        options = [
            [(k, True), (k, False)]
            for k, v in kwargs.items()
            if self._is_input_key(k) and v is not None
        ]
        for comb in itertools.product(*options):
            is_constant_map = dict(comb)
            # At least one input must be non-constant
            if not all(is_constant_map.values()):
                # Add None inputs to the map
                is_constant_map.update(none_inputs)
                yield is_constant_map

    def _iter_should_qdq_combinations(
        self, kwargs: dict[str, Any], qdq_config: dict[str, QDQParameterConfig]
    ) -> Any:
        """Iterate over different QDQ configurations for inputs and outputs.

        Yields all valid combinations of which inputs/outputs should be quantized with QDQ.
        Only applicable if qdq_generator is provided.

        Args:
            kwargs: Operator inputs and attributes
        Yields:
            Flat dict mapping input/output schema names to whether to apply QDQ (bool)

        For inputs,
        - If support_weight is True or support_activation is True, yield a true
          (iter_qdq_combinations will handle the actual type checks with is_constant_map).
        - If support_non_qdq is True, yield a false

        For outputs,
        - If not configed in qdq_config or support_activation is True, yield a true
        - If support_non_qdq is True, yield a false
        """
        # Expand variadic config entries the same way iter_qdq_combinations does,
        # so input name lookups are consistent.
        expanded_config = dict(qdq_config)
        if (
            self.op_variadic_input_name is not None
            and self.op_variadic_input_name in expanded_config
        ):
            variadic_config = expanded_config.pop(self.op_variadic_input_name)
            variadic_prefix = f"{self.op_variadic_input_name}__"
            for k in kwargs:
                if k.startswith(variadic_prefix):
                    expanded_config[k] = variadic_config

        schema_input_names: set[str] = set(self.op_input_names)
        if self.op_variadic_input_name is not None:
            variadic_prefix = f"{self.op_variadic_input_name}__"
            schema_input_names.update(k for k in kwargs if k.startswith(variadic_prefix))

        # For each input in the config, collect the possible should_qdq values.
        input_names: list[str] = []
        input_option_lists: list[list[bool | SupportedONNXType]] = []
        for input_name, config in expanded_config.items():
            if input_name not in schema_input_names:
                continue  # output name present in qdq_config; handled below
            options: list[bool | SupportedONNXType] = []
            if config.support_activation or config.support_weight:
                options.append(True)
            if config.qdq_types is not None:
                options.extend(config.qdq_types)
            if config.support_non_qdq:
                options.append(False)
            input_names.append(input_name)
            input_option_lists.append(options)

        # For each schema output, collect the possible should_qdq values.
        output_names: list[str] = []
        output_option_lists: list[list[bool]] = []
        for output in self.schema.outputs:
            output_name = output.name
            output_config = qdq_config.get(output_name)
            options = []
            if output_config is None or output_config.support_activation:
                options.append(True)
            if output_config is not None and output_config.support_non_qdq:
                options.append(False)
            output_names.append(output_name)
            output_option_lists.append(options)

        all_names = input_names + output_names
        all_option_lists = input_option_lists + output_option_lists

        if not all_names:
            yield {}
            return

        for combo in itertools.product(*all_option_lists):
            yield dict(zip(all_names, combo, strict=False))

    def _create_model(
        self,
        kwargs: dict[str, Any],
        is_constant_map: dict[str, bool],
        output_dtypes: list[str],
        qdq_types: dict[str, SupportedONNXType | None] | None = None,
        dynamic_axes: dict[str, tuple[int, ...]] | None = None,
    ) -> onnx.ModelProto:
        """Create ONNX model with specified constant/non-constant configuration.

        Args:
            kwargs: Operator inputs and attributes
            is_constant_map: Dict mapping input_name -> is_constant (bool)
            output_dtypes: List of output dtype annotations
            qdq_types: Optional QDQ types for quantized model generation
            dynamic_axes: Optional dict mapping input_name -> tuple of axis indices
                         to mark as dynamic (unknown) dimensions in the ONNX model.
                         Only applied to non-constant graph inputs.

        Returns:
            ONNX ModelProto
        """
        # Separate inputs and attributes by name (use _is_input_key to handle variadic inputs)
        input_kwargs = {k: v for k, v in kwargs.items() if self._is_input_key(k)}
        attr_kwargs = {k: v for k, v in kwargs.items() if k in self.op_attribute_names}

        # ONNX helper cannot infer attribute element type from empty iterables.
        # For optional attrs without defaults (e.g., Squeeze axes in older opsets),
        # treat empty values as omitted attributes.
        for attr_name in self.optional_attrs_without_defaults:
            if attr_name not in attr_kwargs:
                continue
            attr_value = attr_kwargs[attr_name]
            is_empty_array = isinstance(attr_value, np.ndarray) and attr_value.size == 0
            is_empty_sequence = isinstance(attr_value, (list, tuple)) and len(attr_value) == 0
            if is_empty_array or is_empty_sequence:
                attr_kwargs.pop(attr_name)

        assert all(v is not None for v in attr_kwargs.values()), "Attributes cannot be None"

        # Build graph components
        graph_inputs = []
        initializers = []
        node_inputs = []
        input_dq_nodes = []  # DequantizeLinear nodes for inputs
        output_q_nodes = []  # QuantizeLinear nodes for outputs

        # Build the list of input names to iterate over, expanding variadic inputs
        input_names_to_process = self.op_input_names.copy()
        if self.op_variadic_input_name is not None:
            variadic_prefix = f"{self.op_variadic_input_name}__"
            variadic_keys = sorted(
                [k for k in input_kwargs if k.startswith(variadic_prefix)],
                key=lambda x: int(x.split("__")[1]),
            )
            input_names_to_process = self.op_input_names + variadic_keys

        # Iterate over input names in order to maintain correct positional ordering.
        # For optional inputs that are None, use empty string "" as placeholder.
        # This is required by ONNX spec for operators with multiple optional inputs.
        for input_name in input_names_to_process:
            input_value = input_kwargs.get(input_name)
            if input_value is None:
                # Optional input not provided - use empty string placeholder
                node_inputs.append("")
                continue
            input_shape = list(input_value.shape)
            if is_constant_map[input_name]:
                # Constant input -> create initializer
                if (
                    qdq_types is not None
                    and input_name in qdq_types
                    and qdq_types[input_name] is not None
                ):
                    # For QDQ: create quantized initializer with DequantizeLinear
                    quant_type = qdq_types[input_name]
                    dq_output_name = f"{input_name}_dq"
                    scale_name = f"{input_name}_scale"
                    zp_name = f"{input_name}_zero_point"

                    # Create scale and zero_point initializers
                    scale_tensor = onnx.helper.make_tensor(
                        name=scale_name,
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[],
                        vals=[1.0],
                    )
                    zp_tensor = onnx.helper.make_tensor(
                        name=zp_name,
                        data_type=quant_type.tensor_proto_type,
                        dims=[],
                        vals=[0],
                    )
                    # Quantize the weight data
                    quantized_data = input_value.astype(quant_type.np_type)
                    weight_tensor = onnx.helper.make_tensor(
                        name=input_name,
                        data_type=quant_type.tensor_proto_type,
                        dims=input_shape,
                        vals=quantized_data.flatten().tolist(),
                    )
                    initializers.extend([weight_tensor, scale_tensor, zp_tensor])

                    # Create DequantizeLinear node for weight
                    dq_node = onnx.helper.make_node(
                        "DequantizeLinear",
                        inputs=[input_name, scale_name, zp_name],
                        outputs=[dq_output_name],
                        domain=self.qdq_generator.domain.value,
                    )
                    input_dq_nodes.append(dq_node)
                    node_inputs.append(dq_output_name)
                else:
                    tensor = onnx.helper.make_tensor(
                        name=input_name,
                        data_type=SupportedONNXType.from_np_type(
                            input_value.dtype
                        ).tensor_proto_type,
                        dims=input_shape,
                        vals=input_value.flatten().tolist(),
                    )
                    initializers.append(tensor)
                    node_inputs.append(input_name)
            else:
                # Non-constant input -> create graph input
                # Apply dynamic axes to non-constant graph inputs
                if dynamic_axes and input_name in dynamic_axes:
                    for axis_idx in dynamic_axes[input_name]:
                        if axis_idx < len(input_shape):
                            input_shape[axis_idx] = -1  # ONNX unknown dimension

                if (
                    qdq_types is not None
                    and input_name in qdq_types
                    and qdq_types[input_name] is not None
                ):
                    # For QDQ: create quantized graph input with DequantizeLinear
                    quant_type = qdq_types[input_name]
                    dq_output_name = f"{input_name}_dq"
                    scale_name = f"{input_name}_scale"
                    zp_name = f"{input_name}_zero_point"

                    # Graph input is quantized type
                    input_info = onnx.helper.make_tensor_value_info(
                        input_name,
                        quant_type.tensor_proto_type,
                        input_shape,
                    )
                    graph_inputs.append(input_info)

                    # Create scale and zero_point initializers
                    scale_tensor = onnx.helper.make_tensor(
                        name=scale_name,
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[],
                        vals=[1.0],
                    )
                    zp_tensor = onnx.helper.make_tensor(
                        name=zp_name,
                        data_type=quant_type.tensor_proto_type,
                        dims=[],
                        vals=[0],
                    )
                    initializers.extend([scale_tensor, zp_tensor])

                    # Create DequantizeLinear node for activation input
                    dq_node = onnx.helper.make_node(
                        "DequantizeLinear",
                        inputs=[input_name, scale_name, zp_name],
                        outputs=[dq_output_name],
                        domain=self.qdq_generator.domain.value,
                    )
                    input_dq_nodes.append(dq_node)
                    node_inputs.append(dq_output_name)
                else:
                    input_info = onnx.helper.make_tensor_value_info(
                        input_name,
                        SupportedONNXType.from_np_type(input_value.dtype).tensor_proto_type,
                        input_shape,
                    )
                    graph_inputs.append(input_info)
                    node_inputs.append(input_name)

        # Strip trailing empty strings from node_inputs (not needed for trailing optional inputs)
        while node_inputs and node_inputs[-1] == "":
            node_inputs.pop()

        # Create outputs
        graph_outputs = []
        output_names = []
        op_output_names = []  # Actual outputs from the operator node

        for idx, dtype in enumerate(output_dtypes):
            output_dtype = SupportedONNXType.from_annotation(dtype).tensor_proto_type
            # Get schema output name for this index
            schema_output_name = (
                self.schema.outputs[idx].name if idx < len(self.schema.outputs) else None
            )
            if (
                qdq_types is not None
                and schema_output_name is not None
                and schema_output_name in qdq_types
                and qdq_types[schema_output_name] is not None
            ):
                # For QDQ: operator outputs to intermediate, then Q to final output
                quant_type = qdq_types[schema_output_name]
                op_output_name = f"op_output_{idx}"
                final_output_name = f"output_{idx}"
                scale_name = f"output_{idx}_scale"
                zp_name = f"output_{idx}_zero_point"

                op_output_names.append(op_output_name)

                # Create scale and zero_point initializers for output Q
                scale_tensor = onnx.helper.make_tensor(
                    name=scale_name,
                    data_type=onnx.TensorProto.FLOAT,
                    dims=[],
                    vals=[1.0],
                )
                zp_tensor = onnx.helper.make_tensor(
                    name=zp_name,
                    data_type=quant_type.tensor_proto_type,
                    dims=[],
                    vals=[0],
                )
                initializers.extend([scale_tensor, zp_tensor])

                # QuantizeLinear: float output -> quantized
                q_node = onnx.helper.make_node(
                    "QuantizeLinear",
                    inputs=[op_output_name, scale_name, zp_name],
                    outputs=[final_output_name],
                    domain=self.qdq_generator.domain.value,
                )
                output_q_nodes.append(q_node)

                # Graph output is quantized type
                output_info = onnx.helper.make_tensor_value_info(
                    final_output_name, quant_type.tensor_proto_type, None
                )
                graph_outputs.append(output_info)
                output_names.append(final_output_name)
            else:
                output_name = f"output_{idx}"
                output_info = onnx.helper.make_tensor_value_info(output_name, output_dtype, None)
                graph_outputs.append(output_info)
                output_names.append(output_name)
                op_output_names.append(output_name)

        # Create the operator node
        node = onnx.helper.make_node(
            self.op_name,
            inputs=node_inputs,
            outputs=op_output_names,
            domain=self.schema.domain,
            **attr_kwargs,
        )

        # Build node list: DQ nodes for inputs -> main op -> Q nodes for outputs
        all_nodes = [*input_dq_nodes, node, *output_q_nodes] if qdq_types is not None else [node]

        # Create the graph
        graph = onnx.helper.make_graph(
            all_nodes,
            f"{self.op_name}_graph",
            graph_inputs,
            graph_outputs,
            initializer=initializers,
        )

        # Create opset imports
        is_ai_onnx_domain = (
            self.schema.domain == "" or self.schema.domain == ONNXDomain.AI_ONNX.value
        )
        # ONNX Runtime only *guarantees* support for models
        # stamped with opset version 7 or above for opset
        # domain 'ai.onnx'.
        schema_version = (
            max(self.schema.since_version, 7) if is_ai_onnx_domain else self.schema.since_version
        )
        # TODO: use self.schema.since_version or some input opset version?
        if qdq_types is not None and self.qdq_generator is not None:
            # Add default ONNX domain for Q/DQ ops if using a different domain
            if self.qdq_generator.domain.value == self.schema.domain or (
                self.qdq_generator.domain == ONNXDomain.AI_ONNX and is_ai_onnx_domain
            ):
                since_version = (
                    schema_version
                    if schema_version > self.qdq_generator.opset_version
                    else self.qdq_generator.opset_version
                )
                opset_imports = [onnx.helper.make_opsetid(self.schema.domain, since_version)]
            else:
                opset_imports = [
                    onnx.helper.make_opsetid(self.schema.domain, schema_version),
                    onnx.helper.make_opsetid(
                        self.qdq_generator.domain.value, self.qdq_generator.opset_version
                    ),
                ]
        else:
            opset_imports = [onnx.helper.make_opsetid(self.schema.domain, schema_version)]

        # Create the model
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)

        # Infer shapes
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Shape inference failed: {e}. {Style.RESET_ALL}")

        # Check model validity
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Model validation failed: {e}. {Style.RESET_ALL}")
            # Continue anyway - some runnable models may fail
            # this check, e.g. Unsqueeze and Split

        return model

    def _build_dynamic_axes_variants(
        self, kwargs: dict[str, Any], is_constant_map: dict[str, bool]
    ) -> list[dict[str, tuple[int, ...]]]:
        """Build list of dynamic_axes dicts for the current constant configuration.

        For "none": returns [{}] (fixed shapes only).
        For "first_axis_dynamic": returns [{}, dynamic_axes_dict] where
        dynamic_axes_dict marks axis 0 as dynamic for all non-constant inputs
        whose shape has ndim > 0 (i.e., excludes scalars and constant inputs).

        Args:
            kwargs: Operator inputs and attributes (values are numpy arrays or scalars).
            is_constant_map: Dict mapping input_name -> is_constant (bool).

        Returns:
            List of dynamic_axes dicts to iterate over.
        """
        # Fixed-shape case is always included
        variants: list[dict[str, tuple[int, ...]]] = [{}]

        if self.dynamic_axis_mode == "first_axis_dynamic":
            dynamic_axes: dict[str, tuple[int, ...]] = {}
            for input_name, is_constant in is_constant_map.items():
                # Skip constant inputs — their shapes are baked into initializers
                if is_constant:
                    continue
                input_value = kwargs.get(input_name)
                if input_value is None:
                    continue
                # Skip scalars (ndim == 0)
                if hasattr(input_value, "shape") and len(input_value.shape) > 0:
                    dynamic_axes[input_name] = (0,)
            if dynamic_axes:
                variants.append(dynamic_axes)

        return variants

    def iter_const_and_dynamic_models(self, kwargs: dict[str, Any], tags: dict[str, Any]) -> Any:
        """Iterate over ONNX models with different constant and dynamic axis configurations.

        Yields a tuple (onnx_model, final_tags).
        Dynamic axes are only applied to non-constant graph inputs.
        """
        qdq_config = self.get_qdq_config()
        qdq_tested_types: set[tuple[tuple[str, str | None], ...]] = set()
        for is_constant_map in self._iter_constant_combinations(kwargs):
            dynamic_axes_variants = self._build_dynamic_axes_variants(kwargs, is_constant_map)
            for dynamic_axes in dynamic_axes_variants:
                # Inject dynamic_axes into tags for downstream consumers
                final_tags = tags.copy()
                final_tags["dynamic_axes"] = dynamic_axes

                if self.qdq_generator is None:
                    # qdq_generator will set input_is_constant
                    # when parameters are not supported for
                    # quantization, so only set it here when not
                    # using qdq_generator
                    final_tags["input_is_constant"] = is_constant_map
                    output_dtypes = self.infer_output_types(kwargs, final_tags)
                    model = self._create_model(
                        kwargs,
                        is_constant_map,
                        output_dtypes,
                        dynamic_axes=dynamic_axes,
                    )

                    yield model, final_tags
                else:
                    for should_qdq_map in self._iter_should_qdq_combinations(kwargs, qdq_config):
                        # We iterate after constant combination
                        # but not iterate weight and activation
                        # types directly
                        # because input could support both
                        for model, qdq_final_tags in self.iter_qdq_combinations(
                            kwargs,
                            final_tags,
                            is_constant_map,
                            should_qdq_map,
                            qdq_config,
                            qdq_tested_types,
                        ):
                            yield model, qdq_final_tags

    def iter_qdq_combinations(
        self,
        kwargs: dict[str, Any],
        tags: dict[str, Any],
        is_constant_map: dict[str, bool],
        should_qdq_map: dict[str, bool],
        qdq_config: dict[str, QDQParameterConfig],
        qdq_tested_types: set[tuple[tuple[str, str | None], ...]],
    ) -> Any:
        """Iterate over different QDQ combinations.

        From self.get_qdq_config(), first check if is_constant_map is supported by the config
        Then check if each type are supported
        Then iterate over all combinations of weight and activation types
        Add the QD type of each input and D type of each output to the tags with key "qdq_types"

        Optimization: Only iterate over weight types if there are constant inputs that need them,
        and only iterate over activation types if there are non-constant inputs or outputs.

        Yields:
            Tuple of (onnx_model, final_tags) where final_tags includes qdq_types
        """
        if self.qdq_generator is None:
            return

        if qdq_config is None:
            return

        # Expand qdq_config for variadic inputs: replace the base variadic key
        # (e.g. "inputs") with the individually expanded keys present in kwargs
        # (e.g. "inputs__0", "inputs__1"), so all downstream lookups work uniformly.
        if self.op_variadic_input_name is not None and self.op_variadic_input_name in qdq_config:
            variadic_config = qdq_config[self.op_variadic_input_name]
            variadic_prefix = f"{self.op_variadic_input_name}__"
            qdq_config = {k: v for k, v in qdq_config.items() if k != self.op_variadic_input_name}
            for k in kwargs:
                if k.startswith(variadic_prefix):
                    qdq_config[k] = variadic_config

        # should_qdq_map is a flat dict: {schema_name: bool} for both inputs and outputs.
        # Absent keys mean "use qdq_config default" (same as before).

        # Check if is_constant_map is compatible with QDQ config
        # For each input, check if it's constant (weight) or non-constant (activation)
        # and verify the config supports that mode
        # All inputs must be explicitly defined in qdq_config
        needs_weight_iteration = False

        for input_name, is_constant in is_constant_map.items():
            if input_name not in qdq_config:
                # Input not in QDQ config - skip this combination
                # All inputs must be explicitly defined in qdq_config
                return

            # should_qdq_map says False → treat as pass-through, skip all QDQ checks.
            if should_qdq_map.get(input_name) is False:
                continue

            config = qdq_config[input_name]
            should_val = should_qdq_map.get(input_name)

            if isinstance(should_val, SupportedONNXType):
                # Specific type from qdq_types — always quantize with this type, skip
                # weight/activation mode checks since the type is fully determined.
                pass
            else:
                # should_val is True (from support_weight or support_activation)
                # If neither flag is set, treat as pass-through.
                if not config.support_weight and not config.support_activation:
                    continue
                if is_constant and not config.support_weight:
                    # Config doesn't support this input as weight
                    return
                if not is_constant and not config.support_activation:
                    # Config doesn't support this input as activation
                    return
                # Only need weight iteration when type is not already determined
                if is_constant and config.support_weight:
                    needs_weight_iteration = True

        # Step 3: Validate input types against qdq_generator.SUPPORT_DQ_OUTPUT_TYPES
        # DQ nodes output the type that the operator expects as input; the original input
        # types must therefore be in SUPPORT_DQ_OUTPUT_TYPES (e.g. float32).
        for input_name in qdq_config:
            if kwargs.get(input_name) is None:
                continue  # Optional input not provided, skip
            # For expanded variadic keys (e.g. "inputs__0"), fall back to the base
            # variadic name for type_annotations lookup since the schema only tracks
            # the base name.
            ta_key = (
                self.op_variadic_input_name
                if self.op_variadic_input_name is not None
                and input_name.startswith(f"{self.op_variadic_input_name}__")
                else input_name
            )
            if ta_key not in self.type_annotations:
                raise ValueError(f"Input '{input_name}' not found in type annotations")
            config = qdq_config[input_name]
            should_val = should_qdq_map.get(input_name)
            if should_val is False:
                continue  # should_qdq_map says no DQ for this input, skip type check
            if (
                not config.support_activation
                and not config.support_weight
                and not isinstance(should_val, SupportedONNXType)
            ):
                continue  # This input is not quantized, skip type check
            type_template = self.type_annotations[ta_key]
            annotation = self._apply_type_var_combination(type_template, tags[self.type_vars_key])
            try:
                onnx_type = SupportedONNXType.from_annotation(annotation).onnx_type
            except ValueError:
                return
            if onnx_type not in self.qdq_generator.SUPPORT_DQ_OUTPUT_TYPES:
                return

        # Validate output types against qdq_generator.SUPPORTED_Q_INPUT_TYPES
        # Q nodes take the operator output as their float input; the original output types
        # must therefore be in SUPPORTED_Q_INPUT_TYPES (e.g. float32).
        # Skip validation for outputs explicitly configured as no-Q (empty QDQParameterConfig).
        for idx, output_annotation in enumerate(self.infer_output_types(kwargs, tags)):
            schema_output_name = (
                self.schema.outputs[idx].name if idx < len(self.schema.outputs) else None
            )
            if schema_output_name is not None and should_qdq_map.get(schema_output_name) is False:
                continue  # should_qdq_map says no Q for this output, skip type validation
            if schema_output_name is not None and schema_output_name in qdq_config:
                output_config = qdq_config[schema_output_name]
                if not output_config.support_weight and not output_config.support_activation:
                    # No Q node for this output
                    # (support_non_qdq only), skip validation
                    continue
            try:
                onnx_type = SupportedONNXType.from_annotation(output_annotation).onnx_type
            except ValueError:
                return
            if onnx_type not in self.qdq_generator.SUPPORTED_Q_INPUT_TYPES:
                return

        # Build type lists to iterate over
        # If we don't need iteration, use a single placeholder to ensure we run once
        weight_types_to_iterate = (
            self.qdq_generator.weight_onnx_types
            if needs_weight_iteration
            else [None]  # Placeholder - won't be used
        )

        # Iterate over weight and activation type combinations
        for weight_onnx_type in weight_types_to_iterate:
            for activation_onnx_type in self.qdq_generator.activation_onnx_types:
                # Build qdq_types mapping for each input
                qdq_types: dict[str, SupportedONNXType | None] = {}
                new_constant_map: dict[str, bool] = {}

                for input_name, is_constant in is_constant_map.items():
                    # Set None for optional inputs not provided
                    if kwargs.get(input_name) is None:
                        qdq_types[input_name] = None
                        continue

                    # should_qdq_map says False → treat as pass-through regardless of config.
                    if should_qdq_map.get(input_name) is False:
                        qdq_types[input_name] = None  # No DQ per should_qdq_map
                        new_constant_map[input_name] = is_constant
                        continue

                    config = qdq_config[input_name]
                    should_val = should_qdq_map.get(input_name)
                    if (
                        not config.support_activation
                        and not config.support_weight
                        and not isinstance(should_val, SupportedONNXType)
                    ):
                        qdq_types[input_name] = None  # Pass-through input (support_non_qdq only)
                        new_constant_map[input_name] = (
                            is_constant  # Pass-through input, keep original constant setting
                        )
                        continue

                    if isinstance(should_val, SupportedONNXType):
                        qdq_types[input_name] = should_val
                    elif is_constant:
                        qdq_types[input_name] = SupportedONNXType.from_onnx_type(weight_onnx_type)
                    else:
                        qdq_types[input_name] = SupportedONNXType.from_onnx_type(
                            activation_onnx_type
                        )

                # Infer output types for this combination first
                output_dtypes = self.infer_output_types(kwargs, tags)

                # Store output quantization type only for outputs that will be created
                # Outputs use activation type
                output_type = (
                    SupportedONNXType.from_onnx_type(activation_onnx_type)
                    if activation_onnx_type is not None
                    else None
                )
                if output_type is not None:
                    for idx in range(len(output_dtypes)):
                        if idx < len(self.schema.outputs):
                            output_name = self.schema.outputs[idx].name
                            if should_qdq_map.get(output_name) is False:
                                qdq_types[output_name] = None  # No Q per should_qdq_map
                            else:
                                output_config = qdq_config.get(output_name)
                                if (
                                    output_config is not None
                                    and not output_config.support_weight
                                    and not output_config.support_activation
                                ):
                                    qdq_types[output_name] = None  # support_non_qdq only
                                else:
                                    qdq_types[output_name] = output_type

                # Skip combinations where no QDQ node is applied at all — those are
                # plain (non-quantized) models already covered by the base iter() loop.
                if all(v is None for v in qdq_types.values()):
                    continue

                # Deduplicate based on actual qdq_types AND pass-through constant map.
                # Two combinations are distinct if either their quantization types differ
                # OR their pass-through (non-quantized) is_constant settings differ,
                # since the latter changes whether a pass-through input is a graph
                # input or an initializer in the generated model.
                qdq_types_key = tuple(
                    (k, v.onnx_type if v is not None else None)
                    for k, v in sorted(qdq_types.items())
                ) + tuple(sorted(new_constant_map.items()))
                if qdq_types_key in qdq_tested_types:
                    continue
                qdq_tested_types.add(qdq_types_key)

                # Create model with QDQ nodes
                model = self._create_model(
                    kwargs,
                    is_constant_map,
                    output_dtypes,
                    qdq_types=qdq_types,
                    dynamic_axes=tags["dynamic_axes"],
                )

                # Build final tags
                final_tags = tags.copy()
                # Collapse variadic expanded keys back to the base schema name in the
                # stored tags. e.g. {"inputs__0": float32, "inputs__1": float32} →
                # {"inputs": float32}, keeping the type from inputs__0 as representative.
                if self.op_variadic_input_name is not None:
                    variadic_prefix = f"{self.op_variadic_input_name}__"
                    first_key = f"{self.op_variadic_input_name}__0"
                    collapsed = {
                        k: v for k, v in qdq_types.items() if not k.startswith(variadic_prefix)
                    }
                    if first_key in qdq_types:
                        collapsed[self.op_variadic_input_name] = qdq_types[first_key]
                    tags_qdq_types = collapsed
                else:
                    tags_qdq_types = qdq_types
                final_tags[self.qdq_types_key] = {
                    k: v.annotation if v is not None else None for k, v in tags_qdq_types.items()
                }
                if new_constant_map:
                    final_tags["input_is_constant"] = new_constant_map
                print(
                    "Yielding QDQ model with types:",
                    final_tags[self.qdq_types_key],
                    should_qdq_map,
                    new_constant_map,
                )
                yield model, final_tags

    def validate_inputs(self) -> None:
        """Validate that all input combinations are valid for the operator.

        An input combination is considered valid if it runs successfully for at
        least one combination of attribute and TypeVar.
        This method is intended to be called during development of new
        OpInputGenerator subclasses to ensure correctness of input combinations
        before code checkin.
        Note that the correctness is ensured by the iteration order of
        self.iter(), which yields cases of same input combination altogether.
        """
        # TODO: check none of the inputs or attributes are omitted
        invalid_inputs = []
        current_inputs = None
        has_succeeded = False
        for kwargs, tags in self.iter():
            if tags["input_constraints"] != current_inputs:
                if current_inputs is not None and not has_succeeded:
                    invalid_inputs.append(current_inputs)
                current_inputs = tags["input_constraints"]
                has_succeeded = False
                print("Validating input combination:", current_inputs)
            if has_succeeded:
                continue
            try:
                _ = self._run_op_on_cpu(kwargs, tags)
            except Exception:
                # print(f"{Fore.RED}  Input combination failed on: {kwargs}.{Style.RESET_ALL}")
                # print(f"{Fore.RED}  Exception: {e}{Style.RESET_ALL}")
                # Intentionally silent - checking if any combo works for this input
                continue
            else:
                has_succeeded = True

        if current_inputs is not None and not has_succeeded:
            invalid_inputs.append(current_inputs)
        if len(invalid_inputs) > 0:
            raise ValueError(f"Found invalid input combinations: {invalid_inputs}")

    def check_on_ep(
        self,
        ep_checker: "EPChecker",
        capture_output: bool = True,
        n_cases: int | None = None,
        skip_cases: int = 0,
        save_failed_model: bool = False,
        save_model: bool = False,
        model_output_dir: str | Path | None = None,
        skip_signature_fn: Callable[[dict], bool] | None = None,
        yield_skipped: bool = False,
        dry_run: bool = False,
    ) -> Any:
        """Test the given OpInputGenerator by generating ONNX models.

        ep_checker: EPChecker instance to use for testing, providing
        check_compile and check_run methods.
        capture_output: if True, capture stdout/stderr from runner subprocess.
        n_cases: if not None, only run the first n_cases test cases.
                 If n_cases is greater than total cases, run all cases.
        skip_cases: number of test cases to skip before starting to run tests.
        skip_signature_fn: if not None, a function that takes
            a result dict and returns True if the case should
            be skipped (for delta or rerun mode).
        yield_skipped: if True, also yield skipped cases with a "_skipped" marker.
                       This allows the caller to reuse existing results in order.

        Yields:
            Test result dictionaries one at a time.
        """
        if self.qdq_generator is not None and self.get_qdq_config() is None:
            print(
                f"{Fore.YELLOW}Warning: QDQ generator is "
                f"set but no QDQ config is defined. "
                f"Skipping QDQ generation."
                f"{Style.RESET_ALL}"
            )
            return

        # Resolve where to stash any saved ONNX models; defer creation unless saving
        save_dir: Path | None = None

        # TODO: parallel and/or distributed execution of `check_compile`/`check_run`
        cases_skipped = 0
        from winml.modelkit.analyze.runtime_checker.runner import ResilientRunner

        with ResilientRunner(capture_output=capture_output, timeout_sec=60) as runner:
            for case_idx, (kwargs, tags) in enumerate(self.iter()):
                # Check if we've reached the case limit
                if n_cases is not None and case_idx >= n_cases:
                    print(f"Reached n_cases limit ({n_cases}), stopping checks.")
                    break
                # validate input by running on CPU
                kwargs_summary = {
                    k: f"ndarray(shape={v.shape}, dtype={v.dtype})"
                    if isinstance(v, np.ndarray) and v.ndim > 1
                    else v
                    for k, v in kwargs.items()
                }
                print("Running", kwargs_summary)
                try:
                    _ = dry_run or self._run_op_on_cpu(kwargs, tags)
                except Exception as e:
                    print("Skipping invalid input causing with exception:", e)
                    continue

                for onnx_model, final_tags in self.iter_const_and_dynamic_models(kwargs, tags):
                    # Check if we should skip this case based on signature (delta/rerun mode)
                    if skip_signature_fn is not None:
                        if skip_signature_fn(final_tags):
                            if yield_skipped:
                                # Yield skipped case with marker so caller can reuse existing result
                                final_tags["_skipped"] = True
                                yield final_tags
                            continue
                    # Check if we need to skip this case
                    elif cases_skipped < skip_cases:
                        cases_skipped += 1
                        continue

                    model_bytes = onnx_model.SerializeToString()
                    if dry_run:
                        # For dry_run, stash the model bytes in base64 for JSON output/replay.
                        final_tags["model_bytes_b64"] = model_bytes_to_b64(model_bytes)

                    qdq_types = final_tags.get(self.qdq_types_key, None)
                    ep_checker_inputs = self.create_input_dict(kwargs, qdq_types=qdq_types)

                    def _dry_run_result() -> dict[str, Any]:
                        return {
                            "result": {
                                "success": True,
                                "reason": "not_run",
                            },
                            "stdout": "not run",
                            "stderr": "not run",
                        }

                    compile_result = (
                        runner.run(ep_checker.check_compile, model_bytes, ep_checker_inputs)
                        if not dry_run
                        else _dry_run_result()
                    )
                    _check_qnn_ep_setup_failure(compile_result)

                    # TODO: if compilation succeeded, maybe skip run test?
                    if compile_result["result"]["success"]:
                        print(f"{Fore.GREEN}Compilation test passed.{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Compilation test failed.{Style.RESET_ALL}")
                    if (
                        not compile_result["result"]["success"] and save_failed_model
                    ) or save_model:
                        if save_dir is None:
                            save_dir = (
                                Path(model_output_dir)
                                if model_output_dir is not None
                                else Path.cwd()
                            )
                            save_dir.mkdir(parents=True, exist_ok=True)
                        import re

                        # replace 'value': [1.0, 1.0, 1.0, 1.0] to 'value': [..]
                        pattern = r"('value':\s*\[)[^\]]*,[^\]]*(\])"
                        replacement = r"\1..\2"
                        file_name_tags = str(
                            {k: v for k, v in final_tags.items() if k != "model_bytes_b64"}
                        )
                        for key in final_tags:
                            file_name_tags = file_name_tags.replace(f"'{key}': ", "")
                        file_name_tags = (
                            file_name_tags.replace(", 'min_max': None", "")
                            .replace("'type': 'shape', ", "")
                            .replace("'type': 'value', ", "")
                            .replace("'dtype': ", "")
                            .replace("False", "F")
                            .replace("True", "T")
                            .replace("None", "N")
                            .replace("INT", "I")
                            .replace("FLOAT", "F")
                            .replace("float", "f")
                        )
                        file_name_tags = re.sub(pattern, replacement, file_name_tags)
                        file_name_tags = (
                            file_name_tags.replace(" ", "")
                            .replace(":", "_")  # to differentiate number like -1
                            .replace("'", "")
                            .replace('"', "")
                            .replace("}", "")
                            .replace("{", "")
                        )

                        file_name_tags = file_name_tags[:180]
                        # add timestamp to filename to avoid collision
                        while True:
                            timestamp = int(time.time() * 1000)
                            candidate = save_dir / f"{file_name_tags}-{timestamp}.onnx"
                            if not candidate.exists():
                                onnx_path = candidate
                                break

                        # Add final_tags as ONNX metadata
                        def _json_default(o: Any):
                            if isinstance(o, np.ndarray):
                                return o.tolist()
                            if isinstance(o, np.generic):
                                return o.item()
                            raise TypeError(
                                f"Object of type {o.__class__.__name__} is not JSON serializable"
                            )

                        meta = onnx_model.metadata_props.add()
                        meta.key = "final_tags"
                        meta.value = json.dumps(final_tags, default=_json_default)
                        onnx.save(onnx_model, onnx_path)
                        print(f"Saved model to {onnx_path}. {Style.RESET_ALL}")
                    run_result = (
                        runner.run(ep_checker.check_run, model_bytes, ep_checker_inputs)
                        if not dry_run
                        else _dry_run_result()
                    )
                    _check_qnn_ep_setup_failure(run_result)

                    if run_result["result"]["success"]:
                        print(f"{Fore.GREEN}Run test passed.{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Run test failed.{Style.RESET_ALL}")
                    final_tags["check_result"] = {
                        "compile": compile_result,
                        "run": run_result,
                    }
                    print(final_tags["check_result"])
                    yield final_tags

    @abstractmethod
    def get_finite_attribute_sets(self) -> dict[str, list[Any]]:
        """Returns a dict {attribute_name: [possible_values]}.

        Whether an attribute allows finite input value set (and what those
        values are) can be inferred from the schema.attributes.
        """

    @abstractmethod
    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Returns a list of dicts: {input_or_attribute_name: constraint_or_value}.

        Representing a combination of inputs to test the op with.
        Any attribute with finite input value set shall not appear in the dicts,
        as those will be handled separately by iterating over the finite
        attribute sets.
        For each input name: input_constraint_or_value is an instance of
        InputConstraint, which can be either InputValueConstraint or
        InputShapeConstraint.
        For each attribute name: input_constraint_or_value is the actual
        attribute value to use in this combination.
        This list must contain only valid combinations in that they are
        meaningful to the op being tested, and will not cause runtime errors due
        to invalid input shapes or values.
        The returned list must must cover different input dimensions (dimension
        means number of axes specifically here), since the op support may vary
        with input dimensions. The total dimension of each of the inputs shall
        not exceed 6, and size of each axis shall not exceed 6.
        In addition, the return list should contain combinations with from
        smallest to biggest possible dimensions.
        """

    def iter(self) -> Any:
        """Iterate over all input and attribute combinations for testing.

        Iteration order is input combinations -> optional attr combinations
        -> optional input combinations -> finite attribute combinations -> TypeVar combinations.
        This order is important for validate_inputs() to work correctly.
        Make sure to update validate_inputs() if the order is changed.
        """
        input_combinations = self.get_input_and_infinite_attribute_combinations()
        for input_comb in input_combinations:
            # Generate combinations where optional attrs
            # (without defaults) are either provided or omitted
            for optional_attr_comb in self._optional_attr_combination_iter(input_comb):
                # Generate combinations where optional inputs are either provided or None
                for optional_input_comb in self._optional_input_combination_iter(
                    optional_attr_comb
                ):
                    for attr_comb in self._finite_attribute_combination_iter():
                        for type_var_comb in self._type_var_combination_iter():
                            applied_type_annotations = {
                                name: self._apply_type_var_combination(
                                    type_annotation, type_var_comb
                                )
                                for name, type_annotation in self.type_annotations.items()
                            }
                            input_constraints = {
                                k: v.to_dict() if isinstance(v, InputConstraint) else v
                                for k, v in optional_input_comb.items()
                                if self._is_input_key(k)
                            }
                            applied_input_comb = {
                                k: (
                                    v.get_value(type_annotation=applied_type_annotations[k])
                                    if isinstance(v, InputConstraint)
                                    else v
                                )
                                for k, v in optional_input_comb.items()
                            }
                            kwargs = {**attr_comb, **applied_input_comb}
                            # Expand variadic inputs to key-value
                            # pairs, and normalize kv order
                            # to inputs, variadic inputs, attributes
                            if self.op_variadic_input_name is not None:
                                variadic_input = kwargs.pop(self.op_variadic_input_name)
                                for idx, tensor in enumerate(variadic_input):
                                    kwargs[f"{self.op_variadic_input_name}__{idx}"] = tensor
                                variadic_keys = [
                                    f"{self.op_variadic_input_name}__{i}"
                                    for i in range(len(variadic_input))
                                ]
                                normalized_key_order = (
                                    self.op_input_names + variadic_keys + self.op_attribute_names
                                )
                            else:
                                normalized_key_order = self.op_input_names + self.op_attribute_names
                            # Normalize kwargs key order; keep None values for optional inputs
                            kwargs = {k: kwargs[k] for k in normalized_key_order if k in kwargs}

                            attrs = {
                                k: v for k, v in kwargs.items() if k in self.op_attribute_names
                            }
                            tags = {
                                self.type_vars_key: type_var_comb,
                                "input_constraints": self.filter_kwargs_by_opset(input_constraints),
                                "attrs": attrs,
                            }
                            yield self.filter_kwargs_by_opset(kwargs), tags

        # TODO: check completeness of inputs+attributes, check no redeundancy

    def _run_op_on_cpu(self, kwargs: dict[str, Any], tags: dict[str, Any]) -> Any:
        """Run the operator on CPU with the given kwargs.

        This method creates an ONNX model and executes it using ONNX Runtime
        to validate that the input combination is valid.

        Args:
            kwargs: Operator inputs and attributes

        Returns:
            Output from running the model

        Raises:
            Exception: If model creation or execution fails
        """
        import onnxruntime as ort

        # Create validation model with all inputs as non-constant
        # Use _is_input_key to identify inputs (including expanded variadic inputs)
        input_kwargs = {k: v for k, v in kwargs.items() if self._is_input_key(k)}
        is_constant_map = dict.fromkeys(input_kwargs.keys(), False)
        output_dtypes = self.infer_output_types(kwargs, tags)
        if len(output_dtypes) == 0:
            raise ValueError("Cannot infer output types for the given inputs and type vars.")
        qdq_types = tags.get(self.qdq_types_key)
        model = self._create_model(kwargs, is_constant_map, output_dtypes, qdq_types=qdq_types)

        # Create inference session
        sess = ort.InferenceSession(
            model.SerializeToString(),
        )

        input_dict = {k: v for k, v in kwargs.items() if k not in self.op_attribute_names}
        input_dict = self.create_input_dict(input_dict, qdq_types=qdq_types)

        # Run inference
        return sess.run(None, input_dict)

    def create_input_dict(
        self, kwargs: dict[str, Any], qdq_types: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create input dictionary for inference, optionally quantizing inputs for QDQ models.

        For QDQ models, non-constant inputs (graph inputs) need to be quantized to match
        the expected input type of the ONNX model (since graph inputs are quantized types
        that get dequantized by DequantizeLinear nodes).

        Args:
            kwargs: Dictionary of input name -> numpy array (non-constant inputs only)
            qdq_types: Optional QDQ type mapping. Keys are input/output names from schema,
                       values are SupportedONNXType annotations (str) indicating quantization type.

        Returns:
            Dictionary of input name -> numpy array, with quantized types if QDQ is enabled.
        """
        if qdq_types is None:
            # No QDQ - return inputs as-is, filtering out None values
            return {k: v for k, v in kwargs.items() if v is not None}

        input_dict = {}
        for input_name, input_value in kwargs.items():
            if input_value is None:
                # Skip optional inputs not provided
                continue

            # For expanded variadic keys (e.g. "inputs__0"), fall back to the base
            # schema name (e.g. "inputs") which is what collapsed qdq_types stores.
            qdq_key = input_name
            if input_name not in qdq_types and self.op_variadic_input_name is not None:
                variadic_prefix = f"{self.op_variadic_input_name}__"
                if input_name.startswith(variadic_prefix):
                    qdq_key = self.op_variadic_input_name

            if qdq_types.get(qdq_key):
                # Get the quantization type for this input
                quant_type_annotation = qdq_types[qdq_key]
                if isinstance(quant_type_annotation, str):
                    quant_type = SupportedONNXType.from_annotation(quant_type_annotation)
                else:
                    # Already a SupportedONNXType
                    quant_type = quant_type_annotation

                # Quantize the input data to the expected type
                # Using simple cast since scale=1.0, zero_point=0
                input_dict[input_name] = input_value.astype(quant_type.np_type)
            else:
                # Input not in QDQ config - keep original type
                input_dict[input_name] = input_value

        return input_dict

    def derive_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """Derive additional properties based on given properties.

        This method can be overridden by subclasses to derive additional
        properties based on the given properties.
        The derived properties will be added to the properties dict.
        To call in input processing.
        """
        raise NotImplementedError("derive_properties() not implemented for this OpInputGenerator")

    def get_infinite_property_names(self) -> list[str]:
        """Get list of attribute names and input names that have infinite value sets.

        To call in result processing. To be overridden by subclasses.

        Returns:
            List of attribute and input names with infinite value sets.
        """
        return []

    def get_qdq_config(self) -> dict[str, QDQParameterConfig] | None:
        """Get QDQ configuration for the op.

        If returns None, QDQ generation is not supported.
        If returns a dict, the keys are input names in schema,
        and the values indicate the input could be quantized as
        weight or activation.
        - as weight: the input is from initializer
        - as activation: the input is not from initializer
        - if the config has qdq_types, it indicates the input could be quantized as those types.
          Overwrite the default list
        If input names are not in the dict
        - if the input name is optional, then when the value is actually provided, we will not
          generate the model because we don't know what it is supported
        - if the input name is required, then we will not generate the model because we don't
          know what it is supported
        """
        return None

    def infer_output_types(
        self,
        kwargs: dict[str, Any],
        tags: dict[str, Any],
        required_outputs_only: bool = True,
    ) -> list[str]:
        """Infer ALL output types from operator kwargs and type variable assignments.

        if qdq_types_key in tags, then the output types will be inferred based on the QDQ types.

        Args:
            kwargs: Operator input arguments
            tags: Tags containing type_vars and other metadata
            required_outputs_only: If True, only infer types for required outputs.
                                   Optional outputs are skipped.

        Returns:
            List of output dtypes as ONNX type annotation strings (one per output).
        """
        output_dtypes = []
        for output in self.schema.outputs:
            is_optional = output.option == OpSchema.FormalParameterOption.Optional
            if is_optional and required_outputs_only:
                # Optional output - skip if only required outputs are needed
                continue
            type_var_key = output.type_str
            # legacy compatibility: adding _op_name suffix
            type_var_key_with_op_name = f"{type_var_key}_{self.op_name}"
            # Resolve the type using type variable assignments
            if type_var_key_with_op_name in tags[self.type_vars_key]:
                # Type variable resolved from input types
                annotation = tags[self.type_vars_key][type_var_key_with_op_name]
            elif type_var_key_with_op_name in self.type_vars_with_unique_dtypes:
                annotation = self.type_vars_with_unique_dtypes[type_var_key_with_op_name].annotation
            else:
                annotation = SupportedONNXType.from_onnx_type(type_var_key).annotation
            # TODO: decide what to do with optional outputs
            # in a general way - currently we add them to outputs
            # NOTE: variadic output not handled in this general
            # method, as the number of such outputs is unknown
            output_dtypes.append(annotation)

        return output_dtypes


class ExampleReshapeInputGenerator(OpInputGenerator):
    """Example input generator for the Reshape operator."""

    op_name = "Reshape"

    def get_finite_attribute_sets(self) -> dict[str, list[Any]]:
        """Return finite attribute sets for Reshape."""
        return {"allowzero": [0, 1]}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for Reshape."""
        return [
            {
                "data": InputShapeConstraint((2, 3, 2, 2)),
                "shape": InputValueConstraint(np.array([2, 3, 2, 1, 2], dtype=np.int64)),
            },
            {
                "data": InputShapeConstraint((2, 3, 4)),
                "shape": InputValueConstraint(np.array([6, 4], dtype=np.int64)),
            },
            {
                "data": InputShapeConstraint((5, 1, 2)),
                "shape": InputValueConstraint(np.array([10, 1, 1], dtype=np.int64)),
            },
        ]
