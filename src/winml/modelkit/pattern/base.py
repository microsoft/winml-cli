# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Pattern matching system for ONNX models.

This module provides a flexible framework for matching and validating subgraph patterns
in ONNX models. It supports both topology-based matching and semantic validation.

Architecture
------------

The pattern matching system consists of several key components:

1. **Pattern Definition** (Pattern, Skeleton, PatternSchema):
   - Pattern: Abstract base class defining what to match and how to validate
   - Skeleton: Topology specification (nodes, domains, edges, inputs, outputs)
   - PatternSchema: Schema describing inputs, outputs, type constraints

2. **Matching Process** (PatternMatcher):
   - Topology matching: Find subgraphs matching the skeleton structure
   - Domain validation: Check each node belongs to the correct ONNX domain
   - Semantic validation: Check semantic constraints (constants, types, etc.)
   - Result building: Create PatternMatchResult with full metadata

3. **Result Objects** (PatternMatchResult, InputInfo):
   - PatternMatchResult: Complete match information with validation
   - InputInfo: Metadata for each input (shape, value, is_constant)

Type System
-----------

Uses centralized type conversions from `modelkit.onnx.dtypes.SupportedONNXType`:
- ONNX type strings ('tensor(float)') ↔ numpy dtypes (np.float32) ↔ TensorProto types (1)
- Automatic type inference from model using shape inference
- Type-aware constant validation

Performance Optimizations
-------------------------

PatternMatcher builds lookup dictionaries during initialization for O(1) access:
- tensor_shapes: tensor_name → shape tuple
- tensor_types: tensor_name → ONNX type string
- tensor_values: tensor_name → numpy array (for constants/initializers)
- constant_and_initializer_names: set of constant tensor names
- node_lookup: node_name → node object
- producer_lookup: tensor_name → (producer_name, slot, op_type)

These are built once from:
- Graph inputs/outputs (with shape inference)
- Value_info (intermediate tensors)
- Initializers
- Constant nodes

Usage Example
-------------

```python
import onnx
from winml.modelkit.pattern import GeluPattern, PatternMatcher

# Load model
model = onnx.load("model.onnx")

# Create matcher and register pattern
matcher = PatternMatcher(model)
gelu = GeluPattern()
matcher.register_pattern(gelu)

# Match and validate
results = matcher.match()  # Returns list[PatternMatchResult]

for result in results:
    # Access matched structure
    print(f"Matched nodes: {result.skeleton_match_result.matched_nodes}")
    print(f"Inputs: {result.skeleton_match_result.inputs}")
    print(f"Output: {result.skeleton_match_result.output}")

    # Access semantic info
    print(f"Schema mappings: {result.schema_input_to_value}")
    print(f"Type mappings: {result.type_param_to_type}")

    # Access input metadata
    for name, info in result.input_infos.items():
        print(f"{name}: shape={info.shape}, constant={info.is_constant}")
```

Implementing New Patterns
--------------------------

To add a new pattern, subclass Pattern and implement:

1. `get_skeleton()`: Define the topology (nodes, edges, virtual inputs)
2. `get_schema()`: Define the pattern's schema (inputs, outputs, types)
3. `get_internal_constants_and_attributes()`: Return internal constants and attributes
4. `check_skeleton_result()`: (Optional) Override to add pattern-specific validation
5. `get_onnx_model()`: (Optional) Generate ONNX model for the pattern

Example:
```python
class MyPattern(Pattern):
    def get_skeleton(self) -> Skeleton:
        return Skeleton(
            node_op_types=['Op1', 'Op2'],
            node_domains=[ONNXDomain.AI_ONNX, ONNXDomain.AI_ONNX],  # Optional, defaults to AI_ONNX
            edges=[
                (-1, 0, 0, 0),  # virtual input -1 → Op1
                (0, 0, 1, 0),   # Op1 output → Op2
            ],
            exit_nodes=[1],
            n_inputs=1,
        )

    def get_internal_constants_and_attributes(
        self, inputs, attributes, is_constant_map, domain_versions
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[str, Any]]:
        # Return internal constants and attributes
        internal_constants = [...]  # (node_idx, slot, value)
        internal_attributes = {...}  # attribute constraints
        return internal_constants, internal_attributes

    def check_skeleton_result(self, result):
        # Call base implementation first for constant/attribute validation
        pattern_result = super().check_skeleton_result(result)
        if pattern_result is None:
            return None

        # Add pattern-specific validation here...

        # Return the base result or modify as needed
        return pattern_result

    def get_schema(self) -> PatternSchema:
        return PatternSchema(
            name='MyPattern',
            doc='Pattern description',
            type_constraints=[...],
            inputs=[...],
            outputs=[...],
        )
```

Important Notes
---------------

- matched_nodes contains ONLY actual nodes (no virtual inputs)
- Virtual inputs are stored separately in SkeletonMatchResult.inputs
- node_domains defaults to AI_ONNX for all nodes if not specified
- Domain matching ensures nodes belong to the correct ONNX domain (ai.onnx, com.microsoft, etc.)
- Shape inference is run automatically during PatternMatcher initialization
- Type conversions use SupportedONNXType for consistency
- Constant validation should use type-aware dtypes from inferred types
- Helper methods _infer_type_mapping() and _build_input_infos() are available
- The base check_skeleton_result() validates constants and
  attributes from get_internal_constants_and_attributes()
"""


# TODO: currently assuming one single output
# TODO: add "hyper-variables" to support offline constant folding
# (evaluated before in graph building stage, which will not be
# reflected in the onnx model)
# TODO: a common onnx model util class to ease different model-related queries

import itertools as it
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import ModelProto, numpy_helper
from onnx.defs import OpSchema

from ..onnx import ONNXDomain, SupportedONNXType, check_onnx_model, infer_onnx_shapes
from .match import InputInfo, PatternMatchResult, SkeletonMatchResult
from .op_input_gen import InputShapeConstraint
from .op_input_gen.op_input_gen import OpInputGenerator
from .utils import get_attribute_proto_value, make_hashable


logger = logging.getLogger(__name__)

# ModelTag string constants (inlined to avoid analyze module dependency)
_MODEL_TAG_INVALID_PATTERN_MATCHER_MODEL = "invalid_pattern_matcher_model"
_MODEL_TAG_MISSING_NODE_NAMES = "missing_node_names"


class PatternMismatchedError(Exception):
    """Exception raised when a skeleton match fails to validate as a pattern.

    This exception is raised during _infer_schema_attributes or other validation
    steps when the matched skeleton cannot be validated as the target pattern
    (e.g., required constants are not available, attributes cannot be extracted).

    The check_skeleton_result method catches this exception and returns None,
    indicating that the skeleton match does not satisfy the pattern constraints.
    """


class InvalidPatternMatcherModelError(Exception):
    """Exception raised when a model is invalid for pattern matching.

    This exception is raised during PatternMatcher initialization when the model
    does not meet the requirements for pattern matching, such as nodes with empty names.

    Attributes:
        error_tag: Associated ModelTag for this exception type
    """

    def __init__(self, message: str = "", error_tag: str | None = None):
        """Initialize the exception.

        Args:
            message: Error message.
            error_tag: ModelTag to associate with this exception.
                      Defaults to ModelTag.INVALID_PATTERN_MATCHER_MODEL if not provided.
        """
        super().__init__(message)
        self._error_tag = (
            error_tag if error_tag is not None else _MODEL_TAG_INVALID_PATTERN_MATCHER_MODEL
        )

    @property
    def error_tag(self) -> str:
        """Return the ModelTag associated with this exception."""
        return self._error_tag


# Registry for PatternInputGenerator classes
_PATTERN_INPUT_GENERATOR_REGISTRY: dict[str, type["PatternInputGenerator"]] = {}


def register_pattern_input_generator(
    cls: type["PatternInputGenerator"],
) -> type["PatternInputGenerator"]:
    """Decorator to register a PatternInputGenerator class by its registration_name.

    Usage:
        @register_pattern_input_generator
        class GeluPatternInputGenerator(PatternInputGenerator):
            registration_name = "Gelu1Pattern"
            ...
    """
    if not hasattr(cls, "registration_name"):
        raise ValueError(
            f"Class {cls.__name__} must have a 'registration_name' class attribute to be registered"
        )
    registration_name = cls.registration_name

    if registration_name in _PATTERN_INPUT_GENERATOR_REGISTRY:
        raise ValueError(
            f"Pattern '{registration_name}' is already registered by "
            f"{_PATTERN_INPUT_GENERATOR_REGISTRY[registration_name].__name__}"
        )
    _PATTERN_INPUT_GENERATOR_REGISTRY[registration_name] = cls
    return cls


def get_pattern_input_generator(registration_name: str) -> type["PatternInputGenerator"]:
    """Get registered PatternInputGenerator class by pattern name.

    Args:
        registration_name: The pattern registration name (e.g., "Gelu", "MatMulAdd")

    Returns:
        The PatternInputGenerator class for the specified pattern

    Raises:
        KeyError: If no PatternInputGenerator is registered for the pattern
    """
    if registration_name not in _PATTERN_INPUT_GENERATOR_REGISTRY:
        raise KeyError(
            f"No PatternInputGenerator registered for '{registration_name}'. "
            f"Available patterns: {sorted(_PATTERN_INPUT_GENERATOR_REGISTRY.keys())}"
        )
    return _PATTERN_INPUT_GENERATOR_REGISTRY[registration_name]


def get_registered_pattern_input_generators() -> list[str]:
    """Get list of all registered pattern names.

    Returns:
        Sorted list of registered pattern names
    """
    return sorted(_PATTERN_INPUT_GENERATOR_REGISTRY.keys())


def _merge_mappings(mappings: list[dict[int, str]]) -> dict[int, str] | None:
    """Merge multiple node mappings, checking for conflicts.

    Args:
        mappings: List of mappings from subgraph node index to model node name.

    Returns:
        Merged mapping if all mappings are compatible, None if conflicts exist.
    """
    merged: dict[int, str] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if key in merged:
                if merged[key] != value:
                    return None
            else:
                merged[key] = value
    return merged


def opschema_to_pattern_schema(op_schema: OpSchema) -> "PatternSchema":
    """Convert an ONNX OpSchema to a PatternSchema.

    This function transforms the ONNX operator schema into the pattern schema
    format used by the pattern matching system. It preserves inputs, outputs,
    type constraints, and attributes.

    Args:
        op_schema: The ONNX OpSchema to convert.

    Returns:
        PatternSchema with the same inputs, outputs, type constraints, and attributes.

    Example:
        >>> from onnx.defs import get_schema
        >>> relu_schema = get_schema("Relu", 14)
        >>> pattern_schema = opschema_to_pattern_schema(relu_schema)
        >>> pattern_schema.name
        'Relu'
    """
    # Convert type constraints - OpSchema.TypeConstraintParam is already the right type
    type_constraints = list(op_schema.type_constraints)

    # Convert inputs - OpSchema.FormalParameter is already the right type
    inputs = list(op_schema.inputs)

    # Convert outputs - OpSchema.FormalParameter is already the right type
    outputs = list(op_schema.outputs)

    # Convert attributes - already a dict[str, OpSchema.Attribute]
    attributes = dict(op_schema.attributes)

    return PatternSchema(
        name=f"{op_schema.name}Pattern",
        doc=op_schema.doc or "",
        inputs=inputs,
        outputs=outputs,
        type_constraints=type_constraints,
        attributes=attributes,
    )


def make_single_op_pattern(
    op_schema: OpSchema,
) -> tuple["PatternSchema", type["Pattern"]]:
    """Create a Pattern class for a single ONNX operator.

    This factory function generates a Pattern subclass that matches a single
    operator node. The generated pattern:
    - Has a skeleton with one node of the specified op type and domain (derived from op_schema)
    - Has edges connecting each virtual input to the node's input slots
    - Has no internal constants (returns empty list)
    - Returns schema attributes as-is for internal attributes

    Args:
        op_schema: The ONNX OpSchema defining the operator. The domain is derived from
                  op_schema.domain (empty string means ai.onnx).

    Returns:
        A tuple of (PatternSchema, Pattern subclass) for the single operator.
        The PatternSchema has name "<OpName>Pattern" (e.g., "GeluPattern").

    Example:
        >>> from onnx.defs import get_schema
        >>> relu_schema = get_schema("Relu", 14)
        >>> relu_pattern_schema, ReluPattern = make_single_op_pattern(relu_schema)
        >>> relu_pattern_schema.name
        'ReluPattern'
        >>> pattern = ReluPattern()
        >>> pattern.get_schema().name
        'ReluPattern'
    """
    # Generate PatternSchema from OpSchema
    pattern_schema = opschema_to_pattern_schema(op_schema)

    # Derive domain from OpSchema
    domain = ONNXDomain.from_str(op_schema.domain)

    # Count the number of inputs (excluding variadic which we handle separately)
    # For single op patterns, each input becomes a virtual input
    n_inputs = len(op_schema.inputs)

    # Build edges: each virtual input connects to the corresponding input slot
    # Virtual inputs are -1, -2, -3, ... (we use -(i+1) for input i)
    # Virtual input -(i+1) connects with slot 0 to node 0's input slot i
    edges: list[tuple[int, int, int, int]] = [(-(i + 1), 0, 0, i) for i in range(n_inputs)]

    # Create the Pattern subclass dynamically
    class SingleOpPattern(Pattern):
        """Pattern for a single ONNX operator."""

        _op_schema = op_schema
        _domain = domain
        _pattern_schema = pattern_schema
        _n_inputs = n_inputs
        _edges = edges

        def get_skeleton(self) -> "Skeleton":
            """Return the skeleton structure for the single op pattern."""
            return Skeleton(
                node_op_types=[self._op_schema.name],
                node_domains=[self._domain],
                edges=self._edges,
                exit_nodes=[0],
                n_inputs=self._n_inputs,
            )

        def get_internal_constants_and_attributes(
            self,
            inputs: dict[str, np.ndarray],
            attributes: dict[str, Any],
            is_constant_map: dict[str, bool],
            domain_versions: dict[ONNXDomain, int],
        ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
            """Return internal constants and attributes for the single op pattern.

            Single op patterns have no internal constants. For internal attributes,
            the schema attributes are returned as-is (mapped to node 0).

            Args:
                inputs: Dictionary mapping input names to numpy array values.
                attributes: Dictionary of attribute values for the pattern.
                is_constant_map: Dict mapping input_name -> is_constant (bool).
                domain_versions: Dict mapping ONNXDomain to opset version.

            Returns:
                Tuple of (empty list for constants, attributes mapped to node 0).
            """
            # No internal constants for single op patterns
            internal_constants: list[tuple[int, int, np.ndarray]] = []

            # Map schema attributes to node 0
            internal_attributes: dict[tuple[int, str], Any] = {}
            for attr_name, attr_value in attributes.items():
                internal_attributes[(0, attr_name)] = attr_value

            return internal_constants, internal_attributes

        def get_schema(self) -> "PatternSchema":
            """Return the schema definition for the single op pattern."""
            return self._pattern_schema

    # Set a meaningful class name
    SingleOpPattern.__name__ = f"Single{op_schema.name}Pattern"
    SingleOpPattern.__qualname__ = f"Single{op_schema.name}Pattern"

    return pattern_schema, SingleOpPattern


@dataclass
class EdgeInfo:
    """Information about an edge (tensor) in the ONNX graph.

    Attributes:
        edge_name: Name of the tensor (edge).
        src_name: Name of the producer node.
        src_slot: Output slot of the producer node.
        src_op_type: Op type of the producer node.
        dst_name: Name of the consumer node.
        dst_slot: Input slot of the consumer node.
        dst_op_type: Op type of the consumer node.
    """

    edge_name: str
    src_name: str
    src_slot: int
    src_op_type: str
    dst_name: str
    dst_slot: int
    dst_op_type: str


@dataclass
class PatternSchema:
    """Schema definition for a pattern, similar to OpSchema but without domain and version.

    Attributes:
        name: Name of the pattern.
        doc: Documentation string describing the pattern.
        inputs: List of input formal parameters.
        outputs: List of output formal parameters.
        type_constraints: List of type constraint parameters.
        attributes: Dict mapping attribute name to attribute definition.
    """

    name: str
    doc: str
    inputs: list[OpSchema.FormalParameter]
    outputs: list[OpSchema.FormalParameter]
    type_constraints: list[OpSchema.TypeConstraintParam] = field(default_factory=list)
    attributes: dict[str, OpSchema.Attribute] = field(default_factory=dict)


class Pattern(ABC):
    """Abstract base class for patterns used in static analysis of onnx models."""

    # This class is intentend to be the rough equivalent of onnx
    # schema or OpInputGenerator for general patterns, both op
    # and subgraphs.

    @property
    def pattern_id(self) -> str:
        """Return the pattern identifier for this pattern.

        Default implementation returns "SUBGRAPH/<schema_name>".
        Subclasses can override to provide custom pattern IDs.

        Returns:
            Pattern identifier string (e.g., "SUBGRAPH/Gelu1")
        """
        schema = self.get_schema()
        return f"SUBGRAPH/{schema.name}"

    @abstractmethod
    def get_skeleton(self) -> "Skeleton":
        """Return the skeleton structure for this pattern.

        Returns:
            Skeleton defining the pattern topology.
        """

    def check_skeleton_result(
        self, skeleton_match_result: "SkeletonMatchResult"
    ) -> "PatternMatchResult | None":
        """Check skeleton match result and return PatternMatchResult if valid, None otherwise.

        This base implementation validates:
        - Internal constant constraints (via get_internal_constants_and_attributes)
        - Internal attribute constraints for inner nodes

        Subclasses can override this method and call super() first to leverage
        base validation, then add pattern-specific checks if the result is not None.

        Args:
            skeleton_match_result: The skeleton match result to validate.

        Returns:
            PatternMatchResult if validation passes, None otherwise.
        """
        try:
            return self._check_skeleton_result_impl(skeleton_match_result)
        except PatternMismatchedError:
            return None

    def _check_skeleton_result_impl(
        self, skeleton_match_result: "SkeletonMatchResult"
    ) -> "PatternMatchResult | None":
        """Implementation of check_skeleton_result.

        This method contains the actual validation logic. It may raise
        PatternMismatchedError if the skeleton cannot be validated.

        Args:
            skeleton_match_result: The skeleton match result to validate.

        Returns:
            PatternMatchResult if validation passes, None otherwise.

        Raises:
            PatternMismatchedError: If pattern-specific validation fails.
        """
        # First, infer type mapping from actual tensor types in the model
        type_param_to_type = self._infer_type_mapping(skeleton_match_result)

        # Build input infos first (needed for get_internal_constants_and_attributes)
        input_infos = self._build_input_infos(skeleton_match_result)

        # Build inputs dict from input_infos for get_internal_constants_and_attributes
        schema = self.get_schema()
        inputs: dict[str, np.ndarray] = {}
        for name, info in input_infos.items():
            if info.value is not None:
                inputs[name] = info.value
            elif info.shape is not None:
                # Check if shape contains only concrete integer dimensions (not symbolic)
                has_symbolic_dims = any(not isinstance(dim, int) for dim in info.shape)
                if has_symbolic_dims:
                    # Pattern matching invalid for inputs with symbolic/dynamic dimensions
                    return None

                # Get type annotation for this input
                idx = list(input_infos.keys()).index(name)
                type_str = skeleton_match_result.matcher.get_tensor_type_str(
                    skeleton_match_result.inputs[idx]
                )
                if type_str:
                    # Use InputShapeConstraint to create dummy value
                    type_annotation = SupportedONNXType.from_onnx_type(type_str).annotation
                    inputs[name] = InputShapeConstraint(info.shape).get_value(type_annotation)

        # Build is_constant_map from input_infos
        is_constant_map = {name: info.is_constant for name, info in input_infos.items()}

        # Infer schema-level attributes from the matched pattern
        schema_attributes = self._infer_schema_attributes(skeleton_match_result)

        # Get domain versions from matcher for opset-aware validation
        domain_versions = skeleton_match_result.matcher.domain_versions

        # Get internal constants and attributes for validation
        constant_constraints, attribute_constraints = self.get_internal_constants_and_attributes(
            inputs, schema_attributes, is_constant_map, domain_versions
        )

        # Validate constant constraints
        is_valid = skeleton_match_result.matcher._check_constant_constraints(
            skeleton_match_result.matched_nodes,
            constant_constraints,
        )

        if not is_valid:
            return None

        # Validate attribute constraints for inner nodes
        for (node_idx, attr_name), expected_value in attribute_constraints.items():
            node = skeleton_match_result.matched_nodes[node_idx]

            # Find the attribute in the node
            attr_found = False
            for attr in node.attribute:
                if attr.name == attr_name:
                    attr_found = True
                    # Use get_attribute_proto_value to extract actual value
                    # and _make_hashable to normalize expected value for comparison
                    actual_value = get_attribute_proto_value(attr, replace_float_with_dummy=False)
                    normalized_expected = make_hashable(
                        expected_value, replace_float_with_dummy=False
                    )
                    if actual_value != normalized_expected:
                        return None
                    break

            if not attr_found:
                # Attribute not found in node - check if expected_value matches the default
                skeleton = self.get_skeleton()
                node_domain = skeleton.node_domains[node_idx]
                op_type = skeleton.node_op_types[node_idx]
                opset_versions = ONNXDomain.get_model_domain_opset_versions(
                    skeleton_match_result.model
                )
                opset_version = opset_versions[node_domain]
                op_schema = node_domain.get_op_schema(op_type, opset_version)

                # Check if attribute has a default value in the schema
                if attr_name not in op_schema.attributes:
                    return None

                attr_def = op_schema.attributes[attr_name]
                if not attr_def.default_value.name:
                    # No default value defined
                    return None

                # Get the default value and compare with expected
                default_value = get_attribute_proto_value(
                    attr_def.default_value, replace_float_with_dummy=False
                )
                normalized_expected = make_hashable(expected_value, replace_float_with_dummy=False)
                if default_value != normalized_expected:
                    return None

        # Build schema input mapping
        schema_input_to_value = {}
        for idx, input_param in enumerate(schema.inputs):
            if idx < len(skeleton_match_result.inputs):
                schema_input_to_value[input_param.name] = skeleton_match_result.inputs[idx]

        # Build schema output mapping
        schema_output_to_value = {}
        if schema.outputs and skeleton_match_result.output:
            schema_output_to_value[schema.outputs[0].name] = skeleton_match_result.output

        return PatternMatchResult(
            skeleton_match_result=skeleton_match_result,
            schema_input_to_value=schema_input_to_value,
            schema_output_to_value=schema_output_to_value,
            type_param_to_type=type_param_to_type,
            attributes=schema_attributes,
            input_infos=input_infos,
        )

    @abstractmethod
    def get_schema(self) -> PatternSchema:
        """Return the schema definition for this pattern."""

    @abstractmethod
    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """Return internal constants and attributes for this pattern.

        Args:
            inputs: Dictionary mapping input names to numpy array values.
            attributes: Dictionary of attribute values for the pattern.
            is_constant_map: Dict mapping input_name -> is_constant (bool).
            domain_versions: Dict mapping ONNXDomain to opset version.

        Returns:
            Tuple of (internal_constants, internal_attributes):
            - internal_constants: List of (node_idx, input_slot, value) tuples
              - node_idx: Index of the node in the skeleton
              - input_slot: Input slot index on that node
              - value: Expected numpy array value with appropriate dtype
            - internal_attributes: Dictionary mapping (node_idx, attr_name) to expected values
        """

    def get_onnx_model(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        output_dtypes: list[str],
        domain_versions: dict[ONNXDomain, int],
        prefix: str = "",
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
    ) -> ModelProto:
        """Generate a standalone ONNX model for this pattern.

        Creates an ONNX graph from the skeleton with:
        - Nodes from skeleton topology
        - Internal constants as initializers
        - External inputs as graph inputs or initializers based on is_constant_map
        - Proper type annotations matching output_dtypes

        Args:
            inputs: Dictionary mapping input names (from schema.inputs) to numpy array values.
            attributes: Attribute values for the pattern.
            is_constant_map: Dict mapping input_name -> is_constant (bool). If True, the input
                            is created as an initializer; otherwise as a graph input.
            output_dtypes: List of ONNX type strings for outputs (e.g., ['tensor(float)']).
            domain_versions: Dict mapping ONNXDomain to opset version for model opset imports.
            prefix: Optional prefix to apply to all names in the graph.
            input_names: Optional list of input names. If provided, must match the number of
                        inputs in the schema. If not provided, uses schema input names.
            output_names: Optional list of output names. If provided, must match the number of
                         outputs in the schema. If not provided, uses generated output names.

        Returns:
            ONNX ModelProto representing this pattern as a standalone model.
        """
        from onnx import helper, numpy_helper

        skeleton = self.get_skeleton()
        schema = self.get_schema()

        # Validate input_names and output_names if provided
        if input_names is None:
            input_names = [input_param.name for input_param in schema.inputs]
        else:
            assert len(input_names) == len(schema.inputs), (
                f"input_names length ({len(input_names)}) must match "
                f"schema inputs length ({len(schema.inputs)})"
            )
        if output_names is None:
            output_names = [output_param.name for output_param in schema.outputs]
        else:
            assert len(output_names) == len(schema.outputs), (
                f"output_names length ({len(output_names)}) must match "
                f"schema outputs length ({len(schema.outputs)})"
            )

        internal_constants, internal_attributes = self.get_internal_constants_and_attributes(
            inputs, attributes, is_constant_map, domain_versions
        )

        # Build mapping of (node_idx, slot) -> constant value
        constant_map: dict[tuple[int, int], np.ndarray] = {
            (node_idx, slot): value for node_idx, slot, value in internal_constants
        }

        # Create graph inputs and initializers
        graph_inputs = []
        initializers = []

        # Create initializers for internal constants
        # These are constants that are NOT virtual inputs but are internal to the pattern
        constant_tensor_names: dict[tuple[int, int], str] = {}
        for (node_idx, slot), constant_value in constant_map.items():
            tensor_name = f"{prefix}const_{node_idx}_{slot}"
            tensor_proto = numpy_helper.from_array(constant_value, tensor_name)
            initializers.append(tensor_proto)
            constant_tensor_names[(node_idx, slot)] = tensor_name

        # Map virtual inputs to tensor names
        virtual_input_names = {}
        for idx, input_param in enumerate(schema.inputs):
            virtual_idx = -(idx + 1)  # -1, -2, -3, ...
            # no prefix for input names
            tensor_name = input_names[idx]
            virtual_input_names[virtual_idx] = tensor_name

            # Get the type for this input from the input's dtype
            input_value = inputs.get(input_param.name)
            if input_value is not None:
                elem_type = SupportedONNXType.from_np_type(input_value.dtype).tensor_proto_type
                input_shape = list(input_value.shape)
            else:
                # Fallback: try to infer from output_dtypes (assuming all same type)
                elem_type = (
                    SupportedONNXType.from_onnx_type(output_dtypes[0]).tensor_proto_type
                    if output_dtypes
                    else 1
                )
                input_shape = None

            # Check if this input is a constant (based on is_constant_map)
            if is_constant_map[input_param.name]:
                assert input_value is not None, (
                    f"Constant input {input_param.name} must have a value"
                )
                # Create initializer for constant input
                tensor_proto = numpy_helper.from_array(input_value, tensor_name)
                initializers.append(tensor_proto)
                # Also add as graph input for ONNX compatibility
                input_tensor = helper.make_tensor_value_info(
                    tensor_name,
                    elem_type,
                    input_shape,
                )
            else:
                # Create graph input for non-constant input
                input_tensor = helper.make_tensor_value_info(
                    tensor_name,
                    elem_type,
                    input_shape if input_value is not None else [None],
                )
                graph_inputs.append(input_tensor)

        # Create nodes
        nodes = []
        node_output_names = {}  # node_idx -> output_name

        for node_idx in range(skeleton.n_nodes):
            op_type = skeleton.node_op_types[node_idx]
            node_name = f"{prefix}node_{node_idx}_{op_type}"

            # Build input list for this node
            input_slots = skeleton.node_input_slots[node_idx]

            # Build inputs in slot order
            # Need to handle both edges (from input_slots) and constants (from constant_map)
            # First, find all slots mentioned in either input_slots or constant_map
            all_slots = set(input_slots.keys())
            for const_node_idx, const_slot in constant_map:
                if const_node_idx == node_idx:
                    all_slots.add(const_slot)

            max_slot = -1 if not all_slots else max(all_slots)

            # Check for contiguous slots
            expected_slots = set(range(max_slot + 1))
            if all_slots != expected_slots:
                missing_slots = expected_slots - all_slots
                raise ValueError(
                    f"Node {node_idx} ({op_type}) has non-contiguous input slots. "
                    f"Missing slots: {sorted(missing_slots)}"
                )

            node_inputs = []
            for slot in range(max_slot + 1):
                # Check if this slot is a constant
                if (node_idx, slot) in constant_tensor_names:
                    input_name = constant_tensor_names[(node_idx, slot)]
                elif slot in input_slots:
                    src, _src_slot = input_slots[slot]
                    input_name = virtual_input_names[src] if src < 0 else node_output_names[src]
                else:
                    raise ValueError(f"Node {node_idx} ({op_type}) missing input at slot {slot}")

                node_inputs.append(input_name)

            # Create output name
            # For exit nodes with custom output_names, use the provided name
            if node_idx in skeleton.exit_nodes:
                output_name = output_names[skeleton.exit_nodes.index(node_idx)]
            else:
                output_name = f"{prefix}{node_name}_out"
            node_output_names[node_idx] = output_name

            # Collect attributes for this node from internal_attributes
            node_attrs = {}
            for (attr_node_idx, attr_name), attr_value in internal_attributes.items():
                if attr_node_idx == node_idx:
                    node_attrs[attr_name] = attr_value

            # Create node
            # Use schema_domain property which returns "" for ai.onnx, actual domain for others
            node = helper.make_node(
                op_type,
                inputs=node_inputs,
                outputs=[output_name],
                name=node_name,
                domain=skeleton.node_domains[node_idx].schema_domain,
                **node_attrs,
            )
            nodes.append(node)

        # Create graph output (from exit nodes)
        graph_outputs = []
        for output_idx, exit_node_idx in enumerate(skeleton.exit_nodes):
            output_name = node_output_names[exit_node_idx]

            # Get the type for this output from output_dtypes
            elem_type = SupportedONNXType.from_onnx_type(
                output_dtypes[output_idx]
            ).tensor_proto_type

            output_tensor = helper.make_tensor_value_info(output_name, elem_type, None)
            graph_outputs.append(output_tensor)

        # Create graph
        graph = helper.make_graph(
            nodes=nodes,
            name=f"{prefix}{schema.name}_graph",
            inputs=graph_inputs,
            outputs=graph_outputs,
            initializer=initializers,
        )

        # Create opset imports from domain_versions parameter
        opset_imports = []
        for domain, version in domain_versions.items():
            opset_imports.append(helper.make_opsetid(domain.schema_domain, version))

        # Create model
        model = helper.make_model(
            graph,
            producer_name="modelkit-pattern-generator",
            opset_imports=opset_imports,
        )
        # Set IR version to 11 for compatibility with older onnxruntime versions
        model.ir_version = 11

        try:
            model = infer_onnx_shapes(model)
        except Exception:
            pass

        return model

    def _infer_type_mapping(self, skeleton_match_result: "SkeletonMatchResult") -> dict[str, str]:
        """Infer type parameter mapping from actual tensor types in the model.

        Args:
            skeleton_match_result: The skeleton match result containing inputs.

        Returns:
            Dictionary mapping type parameters (e.g., 'T') to actual types (e.g., 'tensor(float)').
        """
        schema = self.get_schema()
        type_param_to_type = {}

        for idx, input_param in enumerate(schema.inputs):
            if idx < len(skeleton_match_result.inputs):
                tensor_name = skeleton_match_result.inputs[idx]
                actual_type = skeleton_match_result.matcher.get_tensor_type_str(tensor_name)
                if actual_type and input_param.type_str:
                    type_param_to_type[input_param.type_str] = actual_type

        return type_param_to_type

    def _build_input_infos(
        self, skeleton_match_result: "SkeletonMatchResult"
    ) -> dict[str, InputInfo]:
        """Build InputInfo objects for all inputs in the pattern match.

        Args:
            skeleton_match_result: The skeleton match result containing inputs.

        Returns:
            Dictionary mapping schema input names to InputInfo objects.
        """
        schema = self.get_schema()
        input_infos = {}
        matcher = skeleton_match_result.matcher

        for idx, input_param in enumerate(schema.inputs):
            if idx < len(skeleton_match_result.inputs):
                tensor_name = skeleton_match_result.inputs[idx]

                # Get shape from model
                shape = matcher.get_tensor_shape(tensor_name)

                # Get value if it's a constant or initializer
                value = matcher.tensor_values.get(tensor_name)

                # Determine if it's a constant
                is_constant = tensor_name in matcher.constant_and_initializer_names

                input_infos[input_param.name] = InputInfo(
                    name=input_param.name,
                    shape=shape,
                    value=value,
                    is_constant=is_constant,
                )

        return input_infos

    def _infer_schema_attributes(
        self, skeleton_match_result: "SkeletonMatchResult"
    ) -> dict[str, Any]:
        """Infer schema-level attributes from a matched pattern.

        Override this method in subclasses to extract pattern-level attributes
        from the matched nodes. The returned attributes are passed to
        get_internal_constants_and_attributes for validation and model generation.

        Args:
            skeleton_match_result: The skeleton match result containing matched nodes.

        Returns:
            Dictionary mapping attribute names to their values.
            Default implementation returns an empty dict.
        """
        return {}

    # TODO: derive_properties, get_ignored_properties


class PatternInputGenerator(OpInputGenerator):
    """Input generator that wraps a Pattern for runtime checking."""

    pattern: Pattern = None
    registration_name: str

    def __init__(
        self,
        domain_versions: dict[ONNXDomain, int],
        onnx_types_to_check: list[str] | None = None,
    ) -> None:
        """Initialize PatternInputGenerator with a Pattern instance.

        Args:
            domain_versions: Dict mapping ONNXDomain to opset version for model generation.
            onnx_types_to_check: Optional list of ONNX type annotations to test.
                                If None, all supported types from type constraints are tested.
        """
        assert self.pattern is not None, "Pattern instance must be defined in subclass"
        self.domain_versions = domain_versions
        schema = self.pattern.get_schema()
        self.op_name = schema.name  # compatibility with OpInputGenerator
        super().__init__(schema, onnx_types_to_check)

    def _create_model(
        self,
        kwargs: dict[str, Any],
        is_constant_map: dict[str, bool],
        output_dtypes: list[str],
        qdq_types: dict[str, Any] | None = None,
        dynamic_axes: dict[str, tuple[int, ...]] | None = None,
    ) -> onnx.ModelProto:
        """Create ONNX model using the pattern's get_onnx_model method.

        Args:
            kwargs: Pattern inputs and attributes (input_name -> value)
            is_constant_map: Dict mapping input_name -> is_constant (bool)
            output_dtypes: List of output dtype strings in annotation format (e.g., 'FLOAT')
            qdq_types: Optional QDQ types (unused for patterns, accepted for API compatibility)

        Returns:
            ONNX ModelProto
        """
        # Separate inputs and attributes
        input_kwargs = {k: v for k, v in kwargs.items() if self._is_input_key(k)}
        attr_kwargs = {k: v for k, v in kwargs.items() if not self._is_input_key(k)}

        # Convert annotation format to onnx_type format for get_onnx_model
        onnx_type_dtypes = [
            SupportedONNXType.from_annotation(dtype).onnx_type for dtype in output_dtypes
        ]

        # Use the pattern's get_onnx_model method
        return self.pattern.get_onnx_model(
            inputs=input_kwargs,
            attributes=attr_kwargs,
            is_constant_map=is_constant_map,
            output_dtypes=onnx_type_dtypes,
            domain_versions=self.domain_versions,
        )


@dataclass
class Skeleton:
    """Representation of a pattern skeleton used in skeleton matching.

    This class captures the topology and constraints of a subgraph pattern
    to be matched in ONNX models.

    Attributes:
        node_op_types: List of operator types for each node in the subgraph
                      (e.g., ['Div', 'Erf', 'Add']).
        node_domains: List of ONNX domains for each node. Defaults to AI_ONNX for all nodes
                     if not specified. Must match length of node_op_types if provided.
        edges: Edge connections as (src, src_slot, dst, dst_slot) tuples.
               src < 0 indicates subgraph input (virtual nodes: -1, -2, ...).
               src >= 0 indicates connection from another node in the subgraph.
        exit_nodes: List of node indices that produce outputs of the subgraph, in output order.
        n_inputs: Number of inputs to the subgraph pattern.
        constant_input_slots: Constraints on constant inputs. Maps node_idx -> list of alternative
                             constraint dicts, where each dict maps slot -> value checker.
        node_input_slots: Derived mapping of input slots for each node.
                         Maps node_idx -> {dst_slot: (src, src_slot)}.
                         This is automatically built from edges in __post_init__.
    """

    # Node topology
    node_op_types: list[str]

    # ONNX domain for each node (defaults to AI_ONNX if not specified)
    node_domains: list[ONNXDomain]

    # Edge connections: (src, src_slot, dst, dst_slot)
    # src < 0 indicates subgraph input (virtual nodes: -1, -2, ...)
    # src >= 0 indicates connection from another node in the subgraph
    edges: list[tuple[int, int, int, int]]

    # Exit nodes that produce outputs of the subgraph
    exit_nodes: list[int]

    # Number of inputs to the subgraph
    n_inputs: int

    # Constant input constraints: node_idx -> [{slot: ValueChecker}, ...]
    # Each node can have multiple alternative constraint sets
    # constant_input_slots: dict[int, list[dict[int, Any]]] = field(default_factory=dict)

    # Derived field: mapping of input slots for each node
    # node_idx -> {dst_slot: (src, src_slot)}
    # This is built from edges in __post_init__
    node_input_slots: list[dict[int, tuple[int, int]]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Build node_input_slots from edges after initialization."""
        self.n_nodes = len(self.node_op_types)

        if len(self.node_domains) != self.n_nodes:
            raise ValueError(
                f"Length of node_domains ({len(self.node_domains)}) must match "
                f"length of node_op_types ({self.n_nodes})"
            )

        self.node_input_slots = [{} for _ in range(self.n_nodes)]
        for src, src_slot, dst, dst_slot in self.edges:
            self.node_input_slots[dst][dst_slot] = (src, src_slot)


@dataclass
class EdgePartialMatchResult:
    """Partial matching result for an edge during pattern matching.

    Attributes:
        edge_name: Name of the edge (tensor) in the model.
        node_mapping: Mapping from subgraph node index to model node name.
                     For virtual input nodes (negative indices), maps to
                     edge names instead of node names.
    """

    edge_name: str
    node_mapping: dict[int, str] = field(default_factory=dict)


class PatternMatcher:
    """Matcher of patterns in an ONNX model against pattern skeletons.

    This class performs efficient pattern matching by:
    1. Running shape inference on the model during initialization
    2. Building O(1) lookup dictionaries for tensor properties
    3. Matching skeleton topology against the graph
    4. Validating matches using pattern-specific constraints

    Performance Optimizations
    -------------------------
    All lookup dictionaries are built once during __init__ for O(1) access:
    - tensor_shapes/tensor_types: Shape and type info for all tensors
    - tensor_values: Numpy arrays for constants and initializers
    - constant_and_initializer_names: Fast constant checking
    - node_lookup: Direct node access by name
    - producer_lookup: Fast producer queries
    - edge_info_by_name: Edge topology information

    These eliminate repeated graph traversals during pattern matching.

    Type System Integration
    -----------------------
    Uses SupportedONNXType from winml.modelkit.onnx.dtypes for all type conversions.
    This ensures consistency across the codebase and avoids duplicate type maps.

    Usage
    -----
    matcher = PatternMatcher(model)
    matcher.register_pattern(GeluPattern())
    results = matcher.match()  # Returns list[PatternMatchResult]
    """

    def __init__(
        self,
        onnx_model: ModelProto,
        raise_on_invalid_model: bool = True,
        model_path: str | Path | None = None,
    ) -> None:
        """Initialize the pattern matcher with an ONNX model.

        This performs shape inference and builds all lookup dictionaries upfront
        for efficient O(1) access during pattern matching.

        Args:
            onnx_model: The ONNX model to search for patterns in.
            raise_on_invalid_model: Whether to validate the model and log issues.
            model_path: Optional path to the source ONNX model on disk. When provided,
                external-data initializers whose payloads are not loaded into the
                proto will be lazily resolved from sidecar files (subject to the
                size limits in the runtime-checker query helper) so that
                value-based pattern constraints can still be evaluated.
        """
        # Run shape inference to populate value_info with type information
        # This is critical for type inference and shape lookups

        self.model = infer_onnx_shapes(onnx_model)
        self.graph = self.model.graph
        self.model_path = Path(model_path) if model_path is not None else None

        # Registered patterns: pattern class name -> pattern instance
        self.patterns: dict[str, Pattern] = {}

        # Build lookup structures for efficient pattern matching
        # Maps edge name -> consumer node name -> EdgeInfo
        self.edge_info_by_name: dict[str, dict[str, EdgeInfo]] = defaultdict(dict)

        # Maps tensor name -> (producer node name, output slot, op type)
        self.producer_lookup: dict[str, tuple[str, int, str]] = {}

        # Set of constant and initializer tensor names
        self.constant_and_initializer_names: set[str] = set()

        # Maps tensor name -> numpy array value (for constants/initializers)
        self.tensor_values: dict[str, np.ndarray] = {}

        # Maps tensor name -> shape tuple
        self.tensor_shapes: dict[str, tuple] = {}

        # Maps tensor name -> type string (e.g., 'tensor(float)')
        self.tensor_types: dict[str, str] = {}

        # Maps node name -> node object
        self.node_lookup: dict[str, Any] = {}

        # Set of graph output tensor names
        self.graph_output_names: set[str] = set()

        # Get domain versions from the model's opset imports
        self.domain_versions = ONNXDomain.get_model_domain_opset_versions(onnx_model)

        if raise_on_invalid_model:
            try:
                check_onnx_model(self.model)
            except onnx.checker.ValidationError as e:
                logger.debug("Model failed ONNX checker validation (non-fatal): %s", e)
            # Warn about nodes with empty names; they get auto-generated names
            # (node_{idx}) in _build_lookups and are added to all lookup structures,
            # so pattern matching proceeds normally via tensor connectivity.
            nodes_with_empty_names = [
                (idx, node.op_type) for idx, node in enumerate(self.graph.node) if not node.name
            ]
            if nodes_with_empty_names:
                node_details = ", ".join(
                    f"node {idx} ({op_type})" for idx, op_type in nodes_with_empty_names[:10]
                )
                if len(nodes_with_empty_names) > 10:
                    node_details += f", ... and {len(nodes_with_empty_names) - 10} more"
                logger.info(
                    "Model has %d nodes with empty names (%s). "
                    "These nodes are assigned auto-generated names (node_<idx>) "
                    "and participate in pattern matching normally via tensor connectivity.",
                    len(nodes_with_empty_names),
                    node_details,
                )

        # Build the lookup structures
        self._build_lookups()

    def _register_producer(
        self, tensor_name: str, src_name: str, src_slot: int, src_op_type: str
    ) -> None:
        """Register a producer for a tensor.

        Args:
            tensor_name: Name of the tensor.
            src_name: Name of the producer node.
            src_slot: Output slot of the producer.
            src_op_type: Op type of the producer.
        """
        if tensor_name:
            self.producer_lookup[tensor_name] = (src_name, src_slot, src_op_type)

    def _build_lookups(self) -> None:
        """Build all lookup structures from the ONNX graph.

        This method is called once during initialization to build O(1) lookup
        dictionaries for:
        1. Tensor shapes and types (from inputs, outputs, value_info, initializers, constants)
        2. Tensor values (from initializers and Constant nodes)
        3. Node lookup (fast access by name)
        4. Producer lookup (which node/initializer produces each tensor)
        5. Edge information (topology with src/dst connections)
        6. Constant/initializer tracking (fast is_constant checks)

        Important: Constant nodes are processed to extract their shape, type, and value.
        Shape inference must be run before this (done in __init__).
        """
        for value_info in self.graph.input:
            self._register_producer(value_info.name, value_info.name, 0, "GraphInput")

        # Build tensor values from initializers
        for initializer in self.graph.initializer:
            self.producer_lookup.setdefault(initializer.name, (initializer.name, 0, "Initializer"))
            if not initializer.name:
                continue
            if initializer.data_location == onnx.TensorProto.EXTERNAL and not initializer.raw_data:
                # External data not loaded — try resolving the sidecar file lazily
                # via the runtime-checker query helper. Lazy import avoids the
                # circular dependency between pattern.base and analyze.core.
                from ..analyze.core.runtime_checker_query import (
                    try_load_external_initializer_array,
                )

                external_arr = try_load_external_initializer_array(initializer, self.model_path)
                if external_arr is not None:
                    self.tensor_values[initializer.name] = external_arr
                else:
                    # Pattern matching still works via graph topology; only
                    # value-based constraint checks will be skipped for this tensor.
                    logger.debug(
                        "Skipping tensor value for '%s': external data not loaded",
                        initializer.name,
                    )
            else:
                self.tensor_values[initializer.name] = numpy_helper.to_array(initializer)

        for node_idx, node in enumerate(self.graph.node):
            node_name = node.name or f"node_{node_idx}"

            # Build node lookup
            self.node_lookup[node_name] = node

            # Extract constant values, shapes, and types from Constant nodes
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value":
                        tensor_proto = attr.t
                        for output_name in node.output:
                            if output_name:
                                self.tensor_values[output_name] = numpy_helper.to_array(
                                    tensor_proto
                                )
                                # Add shape and type for constant
                                self.tensor_shapes[output_name] = tuple(tensor_proto.dims)
                                self.tensor_types[output_name] = self._elem_type_to_str(
                                    tensor_proto.data_type
                                )

            for out_idx, output_name in enumerate(node.output):
                self._register_producer(output_name, node_name, out_idx, node.op_type)
            for in_idx, input_name in enumerate(node.input):
                if not input_name:
                    continue
                producer = self.producer_lookup.get(input_name)
                if not producer:
                    continue
                src_name, src_slot, src_op_type = producer
                self.edge_info_by_name[input_name][node_name] = EdgeInfo(
                    edge_name=input_name,
                    src_name=src_name,
                    src_slot=src_slot,
                    src_op_type=src_op_type,
                    dst_name=node_name,
                    dst_slot=in_idx,
                    dst_op_type=node.op_type,
                )

        for out_idx, output_info in enumerate(self.graph.output):
            if not output_info.name:
                continue
            # Track graph output tensor names
            self.graph_output_names.add(output_info.name)
            producer = self.producer_lookup.get(output_info.name)
            if not producer:
                continue
            src_name, src_slot, src_op_type = producer
            self.edge_info_by_name[output_info.name][output_info.name] = EdgeInfo(
                edge_name=output_info.name,
                src_name=src_name,
                src_slot=src_slot,
                src_op_type=src_op_type,
                dst_name=output_info.name,
                dst_slot=out_idx,
                dst_op_type="GraphOutput",
            )

        self.constant_and_initializer_names = {
            init.name for init in self.graph.initializer if init.name
        }
        self.constant_and_initializer_names.update(
            output_name
            for node in self.graph.node
            if node.op_type == "Constant"
            for output_name in node.output
            if output_name
        )

        # Build tensor shapes and types lookup for O(1) access
        # This eliminates repeated graph traversals during pattern matching
        # Sources: graph inputs, outputs, value_info,
        # initializers (shapes added above for constants)

        # From graph inputs
        for value_info in self.graph.input:
            if value_info.name and value_info.type.HasField("tensor_type"):
                tensor_type = value_info.type.tensor_type
                if tensor_type.HasField("shape"):
                    self.tensor_shapes[value_info.name] = tuple(
                        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                        for dim in tensor_type.shape.dim
                    )
                elem_type = tensor_type.elem_type
                self.tensor_types[value_info.name] = self._elem_type_to_str(elem_type)

        # From graph outputs
        for value_info in self.graph.output:
            if value_info.name and value_info.type.HasField("tensor_type"):
                tensor_type = value_info.type.tensor_type
                if tensor_type.HasField("shape"):
                    self.tensor_shapes[value_info.name] = tuple(
                        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                        for dim in tensor_type.shape.dim
                    )
                elem_type = tensor_type.elem_type
                self.tensor_types[value_info.name] = self._elem_type_to_str(elem_type)

        # From value_info (intermediate tensors)
        for value_info in self.graph.value_info:
            if value_info.name and value_info.type.HasField("tensor_type"):
                tensor_type = value_info.type.tensor_type
                if tensor_type.HasField("shape"):
                    self.tensor_shapes[value_info.name] = tuple(
                        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                        for dim in tensor_type.shape.dim
                    )
                elem_type = tensor_type.elem_type
                self.tensor_types[value_info.name] = self._elem_type_to_str(elem_type)

        # From initializers
        for initializer in self.graph.initializer:
            if initializer.name:
                self.tensor_shapes[initializer.name] = tuple(initializer.dims)
                self.tensor_types[initializer.name] = self._elem_type_to_str(initializer.data_type)

    def get_tensor_shape(self, tensor_name: str) -> tuple | None:
        """Get the shape of a tensor in the ONNX graph.

        Args:
            tensor_name: Name of the tensor.

        Returns:
            Tuple representing the shape if available, None otherwise.
            Symbolic dimensions are represented as strings.
        """
        return self.tensor_shapes.get(tensor_name)

    def get_tensor_type_str(self, tensor_name: str) -> str:
        """Get the type string for a tensor in the ONNX graph.

        Args:
            tensor_name: Name of the tensor.

        Returns:
            Type string in ONNX format (e.g., 'tensor(float)', 'tensor(int64)').
            Returns empty string if type cannot be determined.
        """
        return self.tensor_types.get(tensor_name, "")

    def _elem_type_to_str(self, elem_type: int) -> str:
        """Convert ONNX element type to string format.

        Args:
            elem_type: ONNX TensorProto.DataType value.

        Returns:
            Type string in format 'tensor(type_name)'.
        """
        try:
            return SupportedONNXType.from_tensor_proto_type(elem_type).onnx_type
        except ValueError:
            # Fallback for unsupported types (e.g., STRING, COMPLEX64, BFLOAT16)
            return f"tensor(unknown_{elem_type})"

    def _get_node_domain(self, node: Any) -> ONNXDomain:
        """Get the ONNX domain of a node.

        Args:
            node: The ONNX node object.

        Returns:
            ONNXDomain enum value. Defaults to AI_ONNX if domain is not specified.
        """
        return ONNXDomain.from_str(node.domain)

    def _check_constant_constraints(
        self,
        matched_nodes: list[str],
        constant_constraints: list[tuple[int, int, np.ndarray]],
    ) -> bool:
        """Check constant value constraints for a skeleton match.

        Args:
            matched_nodes: List of matched node names (actual nodes only, no virtual inputs).
            constant_constraints: List of (node_idx, slot, expected_value) tuples.

        Returns:
            True if all constant constraints are satisfied.
        """
        # TODO: some constant values may be different but sematically equivalent
        # (e.g. [a, b] and [-1, b] for "shape" input of Reshape)
        # how to handle these cases in a most general way?
        # Check each constant constraint
        for node_idx, slot, expected_value in constant_constraints:
            # matched_nodes contains NodeProto objects (no virtual inputs)
            node = matched_nodes[node_idx]

            # Get input tensor name at slot
            if slot >= len(node.input):
                return False
            input_tensor_name = node.input[slot]

            # Get tensor value
            if input_tensor_name not in self.tensor_values:
                return False
            actual_value = self.tensor_values[input_tensor_name]

            # Compare shape and dtype
            if actual_value.shape != expected_value.shape:
                return False
            if actual_value.dtype != expected_value.dtype:
                return False

            # Compare values (with tolerance for floating point)
            if np.issubdtype(actual_value.dtype, np.floating):
                if not np.allclose(actual_value, expected_value):
                    return False
            else:
                if not (actual_value == expected_value).all():
                    return False

        return True

    def _compute_removable(self, matched_nodes: list[str], skeleton_output: str) -> bool:
        """Determine if the skeleton nodes can be safely removed.

        A skeleton is removable iff none of the intermediate tensors (outputs of
        skeleton nodes, excluding the final skeleton output) are consumed by nodes
        outside the skeleton or are graph outputs. The skeleton output is exempt
        because it will be replaced by an equivalent subgraph.

        Args:
            matched_nodes: List of matched node names in the skeleton.
            skeleton_output: The output tensor name of the skeleton (exempt from check).

        Returns:
            True if removable, False otherwise.
        """
        # matched_nodes is list of node name strings
        matched_node_names = set(matched_nodes)

        # Check each matched node's outputs
        for node_name in matched_nodes:
            node = self.node_lookup[node_name]
            for output_tensor in node.output:
                if not output_tensor:
                    continue

                # Skip the skeleton's final output (it will be replaced)
                if output_tensor == skeleton_output:
                    continue

                # If intermediate tensor is a graph output, not removable
                if output_tensor in self.graph_output_names:
                    return False

                # Check all consumers of this tensor
                consumers = self.edge_info_by_name.get(output_tensor, {})
                for consumer_node_name in consumers:
                    # If consumer is outside the skeleton, not removable
                    if consumer_node_name not in matched_node_names:
                        return False

        return True

    def register_pattern(self, pattern: Pattern) -> None:
        """Register a pattern to search for.

        Args:
            pattern: The pattern to register.
        """
        pattern_class_name = pattern.__class__.__name__
        self.patterns[pattern_class_name] = pattern

    def match(self) -> list[PatternMatchResult]:
        """Match registered patterns against the ONNX graph with validation.

        This method performs both skeleton matching and validation of matched results
        using each pattern's check_skeleton_result method.

        Returns:
            List of validated pattern match results found in the graph.
        """
        skeleton_results = self.match_skeleton()

        # Validate each result using pattern's check_skeleton_result
        validated_results = []
        for result in skeleton_results:
            pattern_match_result = result.pattern.check_skeleton_result(result)
            if pattern_match_result is not None:
                validated_results.append(pattern_match_result)

        return validated_results

    def match_skeleton(self) -> list[SkeletonMatchResult]:
        """Match registered patterns against the ONNX graph.

        Returns:
            List of skeleton match results found in the graph.
        """
        all_results: list[SkeletonMatchResult] = []

        # Match each registered pattern
        for pattern in self.patterns.values():
            skeleton = pattern.get_skeleton()
            results = self._match_single_skeleton(pattern, skeleton)
            all_results.extend(results)

        return all_results

    def _match_single_skeleton(
        self, pattern: Pattern, skeleton: Skeleton
    ) -> list[SkeletonMatchResult]:
        """Match a single skeleton pattern in the graph.

        Args:
            pattern: The pattern being matched.
            skeleton: The skeleton structure to match.

        Returns:
            List of skeleton match results for this pattern.
        """
        n_nodes = skeleton.n_nodes
        node_op_types = skeleton.node_op_types
        node_domains = skeleton.node_domains
        node_input_slots = skeleton.node_input_slots
        exit_nodes = skeleton.exit_nodes

        # Track partial matching results for each edge
        # edge_name -> list of partial matching results
        edge_partial_matching_results: dict[str, list[EdgePartialMatchResult]] = defaultdict(list)

        # Results found for this pattern
        skeleton_results: list[SkeletonMatchResult] = []

        # Touch graph inputs in edge_partial_matching_results
        for graph_input in self.graph.input:
            if graph_input.name:
                edge_partial_matching_results[graph_input.name] = []

        for _idx, node in enumerate(self.graph.node):
            # touch output edges
            for out_edge in node.output:
                edge_partial_matching_results[out_edge] = []
            # print(node.op_type, node.name)
            # print("  inputs:", list(node.input))

            for subgraph_node in range(n_nodes):
                # assuming the current node matches idx in subgraph
                if node_op_types[subgraph_node] != node.op_type or node_domains[
                    subgraph_node
                ] != self._get_node_domain(node):
                    continue
                input_slots = node_input_slots[subgraph_node]
                src_slot_matched = True
                for dst_slot, input_edge in enumerate(node.input):
                    edge_info = self.edge_info_by_name[input_edge][node.name]
                    if dst_slot in input_slots:
                        # check 1: src_slot match
                        src, src_slot = input_slots[dst_slot]
                        if src >= 0 and edge_info.src_slot != src_slot:
                            src_slot_matched = False
                            break
                    # else:
                    # print("TODO: input slot unspecified: ")
                if not src_slot_matched:
                    continue

                # check 2: filter the partial match results that match src node index in subgraph
                dst_slot_partial_mappings = []
                # src_node_matched = True
                for dst_slot, input_edge in enumerate(node.input):
                    edge_info = self.edge_info_by_name[input_edge][node.name]
                    if dst_slot in input_slots:
                        src, src_slot = input_slots[dst_slot]
                        if src < 0:
                            mapping = {src: input_edge}
                            dst_slot_partial_mappings.append([mapping])
                        else:
                            if input_edge not in edge_partial_matching_results:
                                assert input_edge in self.constant_and_initializer_names, (
                                    f"Edge {input_edge} not in "
                                    f"partial matching results or "
                                    f"constants/initializers"
                                )
                                # cannot match non-constant/
                                # initializer edges, since src >= 0
                                dst_slot_partial_mappings.append([])
                                # src_node_matched = False
                                continue
                            src_matched_mappings = [
                                partial_mapping.node_mapping.copy()
                                for partial_mapping in edge_partial_matching_results[input_edge]
                                if (
                                    src in partial_mapping.node_mapping
                                    and edge_info.src_name == partial_mapping.node_mapping[src]
                                )
                            ]
                            dst_slot_partial_mappings.append(src_matched_mappings)
                    else:
                        continue  # edge not specified in subgraph, skipping adding mapping

                # if not src_node_matched:
                #     continue

                assert len(dst_slot_partial_mappings) > 0, (
                    "dst_slot_partial_mappings should not be empty"
                )

                # check 3: the mappings must be compatible
                valid_merged_mappings = []
                for mapping_combination in it.product(*dst_slot_partial_mappings):
                    merged_mapping = _merge_mappings(mapping_combination)
                    if merged_mapping is not None:
                        # valid mapping
                        merged_mapping[subgraph_node] = node.name
                        valid_merged_mappings.append(merged_mapping)
                # TODO: attach partial result to node of edge?
                for out_edge in node.output:
                    for valid_mapping in valid_merged_mappings:
                        edge_partial_matching_results[out_edge].append(
                            EdgePartialMatchResult(
                                edge_name=out_edge,
                                node_mapping=valid_mapping,
                            )
                        )

                if valid_merged_mappings and subgraph_node in exit_nodes:
                    # print(f"    (is exit node)")
                    for valid_mapping in valid_merged_mappings:
                        # print(valid_mapping)

                        # Extract inputs (virtual nodes -1, -2, -3, ... in that order)
                        inputs = [valid_mapping[-i] for i in range(1, skeleton.n_inputs + 1)]

                        # Get output from exit node (assuming
                        # single exit node with output slot 0)
                        exit_node_name = valid_mapping[subgraph_node]
                        exit_node = self.node_lookup[exit_node_name]
                        output = exit_node.output[0]

                        # Compute matched_nodes for removability check
                        matched_node_names = [valid_mapping[i] for i in range(n_nodes)]
                        removable = self._compute_removable(matched_node_names, output)

                        # Convert node names to NodeProto objects
                        matched_nodes_list = [self.node_lookup[name] for name in matched_node_names]

                        skeleton_results.append(
                            SkeletonMatchResult(
                                pattern=pattern,
                                matched_nodes=matched_nodes_list,
                                matcher=self,
                                inputs=inputs,
                                output=output,
                                removable=removable,
                            )
                        )

        return skeleton_results


class PatternRewriter:
    """Rewrites matched patterns in an ONNX model by replacing them with new patterns.

    This class takes an ONNX model and a list of pattern match results, then replaces
    matched subgraphs with new patterns while maintaining graph topology and integrity.

    Usage
    -----
    ```python
    # Load model and find patterns to replace
    model = onnx.load("model.onnx")
    matcher = PatternMatcher(model)
    matcher.register_pattern(MatMulAddPattern())
    results = matcher.match()

    # Rewrite MatMulAdd patterns with ReshapeGemmReshape
    rewriter = PatternRewriter(model)
    new_model = rewriter.rewrite([(results, ReshapeGemmReshapePattern)])
    ```

    Notes:
    -----
    - Only patterns marked as removable will be rewritten; others are skipped with a warning.
    - The new pattern must be compatible with the matched pattern's inputs/outputs.
    - Topological order is preserved by inserting new nodes at the index of the first deleted node.
    """

    def __init__(self, onnx_model: ModelProto) -> None:
        """Initialize the pattern rewriter with an ONNX model.

        Args:
            onnx_model: The ONNX model to rewrite.
        """
        self.model = onnx_model

        # Get domain versions from the model's opset imports
        self.domain_versions = ONNXDomain.get_model_domain_opset_versions(onnx_model)
        self.domain_versions[ONNXDomain.COM_MICROSOFT] = 1  # safeguard

    def _remove_unused_constants(self, model: ModelProto) -> None:
        """Remove unused constants and initializers from the model after rewriting.

        This cleans up constants/initializers that are no longer referenced by any node
        after pattern rewrites.

        Args:
            model: The ONNX model to clean up (modified in place).
        """
        graph = model.graph

        # Build set of tensor names used as inputs by nodes
        input_name_to_nodes: dict[str, list] = {}
        for node in graph.node:
            for input_name in node.input:
                if input_name:  # Could be empty when optional
                    if input_name not in input_name_to_nodes:
                        input_name_to_nodes[input_name] = [node]
                    else:
                        input_name_to_nodes[input_name].append(node)

        # Also check graph outputs
        graph_output_names = {output.name for output in graph.output}

        # Remove unused Constant nodes
        unused_nodes = [
            node
            for node in graph.node
            if (
                node.op_type == "Constant"
                and node.output[0] not in graph_output_names
                and node.output[0] not in input_name_to_nodes
            )
        ]

        for node in unused_nodes:
            graph.node.remove(node)

        # Remove unused initializers
        unused_initializers = [
            initializer
            for initializer in graph.initializer
            if (
                initializer.name not in input_name_to_nodes
                and initializer.name not in graph_output_names
            )
        ]

        for initializer in unused_initializers:
            graph.initializer.remove(initializer)
            # Also remove from graph.input if present
            for graph_input in list(graph.input):
                if graph_input.name == initializer.name:
                    graph.input.remove(graph_input)
                    break

    def rewrite(
        self,
        pattern_match_results: list[tuple[list[PatternMatchResult], type[Pattern]]],
    ) -> ModelProto:
        """Rewrite matched patterns in the model with new patterns.

        Args:
            pattern_match_results: List of tuples, each containing:
                - List of PatternMatchResult to replace
                - Pattern class to use for replacement

        Returns:
            New ONNX ModelProto with patterns rewritten.
        """
        import copy
        import warnings

        from onnx import helper

        # Deep copy the model to avoid modifying the original
        new_model = copy.deepcopy(self.model)
        graph = new_model.graph

        # Track which nodes have been deleted to avoid double deletion
        deleted_node_names: set[str] = set()

        rewrite_counter = 0

        # Assert non-overlap across all matches
        all_matched_node_names = [
            node_name
            for match_results, _ in pattern_match_results
            for match_result in match_results
            for node_name in match_result.skeleton_match_result.matched_node_names
        ]
        assert len(all_matched_node_names) == len(set(all_matched_node_names)), (
            "Overlapping nodes found in pattern matches to rewrite. "
            "Each node can only be matched by one pattern."
        )
        for match_results, new_pattern_class in pattern_match_results:
            for match_result in match_results:
                # Rebuild node_name_to_idx before each rewrite to handle index changes
                # This avoids issues if matches within a group are in bad order
                # TODO: avoid rebuiling by using
                # dict[node_name, linked_list_node]; this would
                # reduce theoretical time complexity but we need
                # careful implementation to make it actually faster
                node_name_to_idx: dict[str, int] = {}
                for idx, node in enumerate(graph.node):
                    assert node.name is not None, "All nodes must have names for rewriting."
                    node_name_to_idx[node.name] = idx
                skeleton_match = match_result.skeleton_match_result

                # Check if removable
                if not skeleton_match.removable:
                    warnings.warn(
                        f"Skipping non-removable pattern match for "
                        f"{skeleton_match.pattern.__class__.__name__} with nodes "
                        f"{skeleton_match.matched_node_names}. Intermediate tensors may be used "
                        f"by nodes outside the pattern.",
                        stacklevel=2,
                    )
                    continue

                # Check for already deleted nodes
                already_deleted = [
                    n for n in skeleton_match.matched_node_names if n in deleted_node_names
                ]
                if already_deleted:
                    warnings.warn(
                        f"Skipping pattern match with already deleted nodes: {already_deleted}",
                        stacklevel=2,
                    )
                    continue

                # Create the new pattern instance
                new_pattern = new_pattern_class()
                assert skeleton_match.pattern.get_schema() == new_pattern.get_schema(), (
                    f"New pattern {new_pattern_class.__name__} schema does not match "
                    f"the matched pattern {skeleton_match.pattern.__class__.__name__} schema."
                )

                # Build inputs dict from match result
                inputs: dict[str, np.ndarray] = {}
                is_constant_map: dict[str, bool] = {}
                schema = new_pattern.get_schema()

                for input_param in schema.inputs:
                    input_name = input_param.name
                    if input_name in match_result.input_infos:
                        info = match_result.input_infos[input_name]
                        is_constant_map[input_name] = info.is_constant
                        if info.value is not None:
                            inputs[input_name] = info.value
                        elif info.shape is not None:
                            # Create a dummy array with the shape for internal constant computation
                            dtype_str = match_result.type_param_to_type[input_param.type_str]
                            np_dtype = SupportedONNXType.from_onnx_type(dtype_str).np_type
                            # For shape computation, create a zero array
                            inputs[input_name] = np.zeros(info.shape, dtype=np_dtype)
                        else:
                            # No shape info, skip this input
                            is_constant_map[input_name] = False
                    else:
                        is_constant_map[input_name] = False

                # Get output dtype from the matched pattern
                output_dtypes = []
                for output_param in schema.outputs:
                    type_str = match_result.type_param_to_type[output_param.type_str]
                    output_dtypes.append(type_str)

                # Get input and output names from the matched pattern
                # These need to match the existing graph tensors
                input_names = skeleton_match.inputs
                output_names = [skeleton_match.output]

                # Create prefix for new nodes
                prefix = f"Rewrite_{new_pattern_class.__name__}_{rewrite_counter}_"
                rewrite_counter += 1

                # Generate the new subgraph model
                try:
                    new_subgraph_model = new_pattern.get_onnx_model(
                        inputs=inputs,
                        attributes=match_result.attributes,
                        is_constant_map=is_constant_map,
                        output_dtypes=output_dtypes,
                        domain_versions=self.domain_versions,
                        prefix=prefix,
                        input_names=input_names,
                        output_names=output_names,
                    )
                except PatternMismatchedError as e:
                    logger.debug("Skipping rewrite %s: %s", new_pattern_class.__name__, e)
                    continue

                # Find insertion point: position of last matched node after deletions
                # Since original graph is topologically sorted,
                # last matched node is after all input producers
                matched_indices = [node_name_to_idx[n] for n in skeleton_match.matched_node_names]
                max_matched_idx = max(matched_indices)
                insert_idx = max_matched_idx - (len(matched_indices) - 1)

                # Delete matched nodes from the graph (reverse order to avoid index shifting)
                for idx in sorted(matched_indices, reverse=True):
                    del graph.node[idx]

                # Mark nodes as deleted
                deleted_node_names.update(skeleton_match.matched_node_names)

                # Insert new nodes at the computed position
                new_nodes = list(new_subgraph_model.graph.node)
                for i, new_node in enumerate(new_nodes):
                    graph.node.insert(insert_idx + i, new_node)

                # Append new initializers (constants) from the new subgraph
                for initializer in new_subgraph_model.graph.initializer:
                    # Check if initializer already exists (by name)
                    existing_names = {init.name for init in graph.initializer}
                    if initializer.name not in existing_names:
                        graph.initializer.append(initializer)

        # Add any missing opset imports to the model
        existing_opset_domains = {op.domain for op in new_model.opset_import}
        for domain, version in self.domain_versions.items():
            schema_domain = domain.schema_domain
            if schema_domain not in existing_opset_domains:
                from onnx import helper

                new_model.opset_import.append(helper.make_opsetid(schema_domain, version))

        # Remove unused constants after all rewrites
        self._remove_unused_constants(new_model)

        # Run shape inference on the final model
        try:
            new_model = infer_onnx_shapes(new_model)
        except Exception as e:
            warnings.warn(
                f"Shape inference failed on rewritten model: {e}",
                stacklevel=2,
            )

        # Check model validity
        try:
            check_onnx_model(new_model)
        except Exception as e:
            warnings.warn(
                f"Model validation failed on rewritten model: {e}",
                stacklevel=2,
            )
            # Continue anyway - some runnable models may fail this check

        return new_model
