# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any

import numpy as np

from ..onnx import ONNXDomain
from .base import (
    Pattern,
    PatternInputGenerator,
    PatternSchema,
    Skeleton,
    make_single_op_pattern,
    register_pattern_input_generator,
)
from .op_input_gen import get_runtime_checker_op


# TODO: Add and Mul are commutative, support matching either
# input order; currently assuming all const inputs are the second
# input, and shortcut input is the first one of Mul.

# Schema for GELU pattern
_GELU_SCHEMA, SingleGeluPattern = make_single_op_pattern(
    ONNXDomain.COM_MICROSOFT.get_op_schema("Gelu", 1)
)


class Gelu1Pattern(Pattern):
    """Pattern definition for GELU (Gaussian Error Linear Unit) activation.

    GELU is computed as: x * 0.5 * (1 + erf(x / sqrt(2)))
    This translates to the following node topology:
    - Div: x / sqrt(2)
    - Erf: erf(...)
    - Add: ... + 1
    - Mul: x * ...
    - Mul: ... * 0.5
    """

    def get_skeleton(self) -> Skeleton:
        """Return the skeleton structure for GELU pattern.

        Returns:
            Skeleton defining the GELU computation graph topology.
        """
        # GELU pattern: x * 0.5 * (1 + erf(x / sqrt(2)))
        # Node indices: 0=Div, 1=Erf, 2=Add, 3=Mul, 4=Mul
        node_op_types = ["Div", "Erf", "Add", "Mul", "Mul"]
        node_domains = [ONNXDomain.AI_ONNX] * len(node_op_types)

        # Edges: (src, src_slot, dst, dst_slot)
        # -1 represents the input to the subgraph
        edges = [
            (-1, 0, 0, 0),  # input -> Div[0]
            (0, 0, 1, 0),  # Div -> Erf[0]
            (1, 0, 2, 0),  # Erf -> Add[0]
            (-1, 0, 3, 0),  # input -> Mul[0]
            (2, 0, 4, 1),  # Add -> Mul[1] (second input)
            (3, 0, 4, 0),  # Mul -> Mul[0]
        ]

        # Exit node that produces the final output
        exit_nodes = [4]

        return Skeleton(
            node_op_types=node_op_types,
            node_domains=node_domains,
            edges=edges,
            exit_nodes=exit_nodes,
            n_inputs=1,
        )

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """Return internal constants and attributes for GELU pattern.

        GELU requires specific constant values:
        - Node 0 (Div) slot 1: sqrt(2.0) ≈ 1.4142135
        - Node 2 (Add) slot 1: 1.0
        - Node 4 (Mul) slot 1: 0.5

        Args:
            inputs: Dictionary mapping input names to numpy array values.
            attributes: Dictionary of attribute values for the pattern.
            is_constant_map: Dict mapping input_name -> is_constant (bool).
            domain_versions: Dict mapping ONNXDomain to opset version.

        Returns:
            Tuple of (internal_constants, internal_attributes).
        """
        # Determine the numpy dtype from the first input
        # Default to float32 if type cannot be determined
        dtype = np.float32
        if "X" in inputs and inputs["X"] is not None:
            dtype = inputs["X"].dtype

        # Create constant constraints with the correct dtype
        internal_constants = [
            (0, 1, np.sqrt(2.0).astype(dtype)),
            (2, 1, np.array(1.0, dtype=dtype)),
            (3, 1, np.array(0.5, dtype=dtype)),
        ]

        # GELU has no internal attributes
        internal_attributes: dict[tuple[int, str], Any] = {}

        return internal_constants, internal_attributes

    def get_schema(self) -> PatternSchema:
        """Return the schema definition for GELU pattern.

        Returns:
            PatternSchema defining the GELU pattern's input/output types.
        """
        return _GELU_SCHEMA


@register_pattern_input_generator
class Gelu1PatternInputGenerator(PatternInputGenerator, get_runtime_checker_op("Gelu")):  # type: ignore[misc]  # dynamic base class (runtime-checker op)
    """Input generator for GELU activation pattern variant 1."""

    pattern = Gelu1Pattern()
    registration_name = "Gelu1Pattern"


class Gelu2Pattern(Pattern):
    """Pattern definition for GELU (Gaussian Error Linear Unit) activation - Variant 2.

    GELU is computed as: (1 + erf(x / sqrt(2))) * 0.5 * x
    This translates to the following node topology:
                   +------------------------------------+
                   |                                    |
                   |                                    v
                [root] --> Div -----> Erf  --> Add --> Mul -->Mul -->
                          (B=1.4142...)       (1)            (0.5)

    - Div: x / sqrt(2)
    - Erf: erf(...)
    - Add: ... + 1
    - Mul: ... * 0.5
    - Mul: ... * x
    """

    def get_skeleton(self) -> Skeleton:
        """Return the skeleton structure for GELU pattern variant 2.

        Returns:
            Skeleton defining the GELU computation graph topology.
        """
        # GELU pattern: (1 + erf(x / sqrt(2))) * 0.5 * x
        # Node indices: 0=Div, 1=Erf, 2=Add, 3=Mul, 4=Mul
        node_op_types = ["Div", "Erf", "Add", "Mul", "Mul"]
        node_domains = [ONNXDomain.AI_ONNX] * len(node_op_types)

        # Edges: (src, src_slot, dst, dst_slot)
        # -1 represents the input to the subgraph
        edges = [
            (-1, 0, 0, 0),  # input -> Div[0]
            (0, 0, 1, 0),  # Div -> Erf[0]
            (1, 0, 2, 0),  # Erf -> Add[0]
            (2, 0, 3, 1),  # Add -> Mul[0] (node 3)
            (3, 0, 4, 0),  # Mul (node 3) -> Mul[0] (node 4)
            (-1, 0, 3, 0),  # input -> Mul[1] (node 3, second input)
        ]

        # Exit node that produces the final output
        exit_nodes = [4]

        return Skeleton(
            node_op_types=node_op_types,
            node_domains=node_domains,
            edges=edges,
            exit_nodes=exit_nodes,
            n_inputs=1,
        )

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """Return internal constants and attributes for GELU pattern variant 2.

        Args:
            inputs: Dictionary mapping input names to numpy array values.
            attributes: Dictionary of attribute values for the pattern.
            is_constant_map: Dict mapping input_name -> is_constant (bool).
            domain_versions: Dict mapping ONNXDomain to opset version.

        Returns:
            Tuple of (internal_constants, internal_attributes).
        """
        dtype = np.float32
        if "X" in inputs and inputs["X"] is not None:
            dtype = inputs["X"].dtype

        internal_constants = [
            (0, 1, np.sqrt(2.0).astype(dtype)),
            (2, 1, np.array(1.0, dtype=dtype)),
            (4, 1, np.array(0.5, dtype=dtype)),
        ]

        internal_attributes: dict[tuple[int, str], Any] = {}

        return internal_constants, internal_attributes

    def get_schema(self) -> PatternSchema:
        """Return the schema definition for GELU pattern.

        Returns:
            PatternSchema defining the GELU pattern's input/output types.
        """
        return _GELU_SCHEMA


@register_pattern_input_generator
class Gelu2PatternInputGenerator(PatternInputGenerator, get_runtime_checker_op("Gelu")):  # type: ignore[misc]  # dynamic base class (runtime-checker op)
    """Input generator for GELU activation pattern variant 2."""

    pattern = Gelu2Pattern()
    registration_name = "Gelu2Pattern"


class Gelu3Pattern(Pattern):
    """Pattern definition for GELU (Gaussian Error Linear Unit) activation - Variant 3.

    GELU is computed as: 0.5 * (1 + erf(x / sqrt(2))) * x
    This translates to the following node topology:
                   +------------------------------------------+
                   |                                          |
                   |                                          v
                [root] --> Div -----> Erf  --> Add --> Mul -->Mul
                          (B=1.4142...)       (B=1)   (B=0.5)

    - Div: x / sqrt(2)
    - Erf: erf(...)
    - Add: ... + 1
    - Mul: 0.5 * ...  (with 0.5 as second input per convention)
    - Mul: ... * x
    """

    def get_skeleton(self) -> Skeleton:
        """Return the skeleton structure for GELU pattern variant 3.

        Returns:
            Skeleton defining the GELU computation graph topology.
        """
        # GELU pattern: 0.5 * (1 + erf(x / sqrt(2))) * x
        # Node indices: 0=Div, 1=Erf, 2=Add, 3=Mul, 4=Mul
        node_op_types = ["Div", "Erf", "Add", "Mul", "Mul"]
        node_domains = [ONNXDomain.AI_ONNX] * len(node_op_types)

        # Edges: (src, src_slot, dst, dst_slot)
        # -1 represents the input to the subgraph
        # In this variant, Add output goes to Mul[0] slot 0, 0.5 to slot 1
        # Then that Mul's output goes to next Mul[0], and input goes to slot 1
        edges = [
            (-1, 0, 0, 0),  # input -> Div[0]
            (0, 0, 1, 0),  # Div -> Erf[0]
            (1, 0, 2, 0),  # Erf -> Add[0]
            (2, 0, 3, 0),  # Add -> Mul[0] (node 3, first input)
            (3, 0, 4, 1),  # Mul (node 3) -> Mul[0] (node 4, first input)
            (-1, 0, 4, 0),  # input -> Mul[1] (node 4, second input)
        ]

        # Exit node that produces the final output
        exit_nodes = [4]

        return Skeleton(
            node_op_types=node_op_types,
            node_domains=node_domains,
            edges=edges,
            exit_nodes=exit_nodes,
            n_inputs=1,
        )

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """Return internal constants and attributes for GELU pattern variant 3.

        GELU requires specific constant values:
        - Node 0 (Div) slot 1: sqrt(2.0) ≈ 1.4142135
        - Node 2 (Add) slot 1: 1.0
        - Node 3 (Mul) slot 1: 0.5

        Args:
            inputs: Dictionary mapping input names to numpy array values.
            attributes: Dictionary of attribute values for the pattern.
            is_constant_map: Dict mapping input_name -> is_constant (bool).
            domain_versions: Dict mapping ONNXDomain to opset version.

        Returns:
            Tuple of (internal_constants, internal_attributes).
        """
        dtype = np.float32
        if "X" in inputs and inputs["X"] is not None:
            dtype = inputs["X"].dtype

        internal_constants = [
            (0, 1, np.sqrt(2.0).astype(dtype)),
            (2, 1, np.array(1.0, dtype=dtype)),
            (3, 1, np.array(0.5, dtype=dtype)),
        ]

        internal_attributes: dict[tuple[int, str], Any] = {}

        return internal_constants, internal_attributes

    def get_schema(self) -> PatternSchema:
        """Return the schema definition for GELU pattern.

        Returns:
            PatternSchema defining the GELU pattern's input/output types.
        """
        return _GELU_SCHEMA


@register_pattern_input_generator
class Gelu3PatternInputGenerator(PatternInputGenerator, get_runtime_checker_op("Gelu")):  # type: ignore[misc]  # dynamic base class (runtime-checker op)
    """Input generator for GELU activation pattern variant 3."""

    pattern = Gelu3Pattern()
    registration_name = "Gelu3Pattern"


class Gelu4Pattern(Pattern):
    """Pattern definition for GELU (Gaussian Error Linear Unit) activation - Variant 4.

    GELU is computed as: (1 + erf(x * (1/sqrt(2)))) * 0.5 * x
    This variant uses Mul with 1/sqrt(2) instead of Div with sqrt(2).
    This translates to the following node topology:
                   +----------------------------------------------+
                   |                                              |
                   |                                              v
                [root] --> Mul -----> Erf    -->   Add --> Mul -->Mul
                           (B=0.7071067690849304)  (B=1)  (B=0.5)

    - Mul: x * (1/sqrt(2)) ≈ x * 0.7071067690849304
    - Erf: erf(...)
    - Add: ... + 1
    - Mul: ... * 0.5
    - Mul: ... * x
    """

    def get_skeleton(self) -> Skeleton:
        """Return the skeleton structure for GELU pattern variant 4.

        Returns:
            Skeleton defining the GELU computation graph topology.
        """
        # GELU pattern: (1 + erf(x * 0.7071...)) * 0.5 * x
        # Node indices: 0=Mul, 1=Erf, 2=Add, 3=Mul, 4=Mul
        node_op_types = ["Mul", "Erf", "Add", "Mul", "Mul"]
        node_domains = [ONNXDomain.AI_ONNX] * len(node_op_types)

        # Edges: (src, src_slot, dst, dst_slot)
        # -1 represents the input to the subgraph
        edges = [
            (-1, 0, 0, 0),  # input -> Mul[0] (node 0, first input)
            (0, 0, 1, 0),  # Mul (node 0) -> Erf[0]
            (1, 0, 2, 0),  # Erf -> Add[0]
            (2, 0, 3, 0),  # Add -> Mul[0] (node 3, first input)
            (3, 0, 4, 1),  # Mul (node 3) -> Mul[0] (node 4, first input)
            (-1, 0, 4, 0),  # input -> Mul[1] (node 4, second input)
        ]

        # Exit node that produces the final output
        exit_nodes = [4]

        return Skeleton(
            node_op_types=node_op_types,
            node_domains=node_domains,
            edges=edges,
            exit_nodes=exit_nodes,
            n_inputs=1,
        )

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """Return internal constants and attributes for GELU pattern variant 4.

        GELU requires specific constant values:
        - Node 0 (Mul) slot 1: 1/sqrt(2.0) ≈ 0.7071067690849304
        - Node 2 (Add) slot 1: 1.0
        - Node 3 (Mul) slot 1: 0.5

        Args:
            inputs: Dictionary mapping input names to numpy array values.
            attributes: Dictionary of attribute values for the pattern.
            is_constant_map: Dict mapping input_name -> is_constant (bool).
            domain_versions: Dict mapping ONNXDomain to opset version.

        Returns:
            Tuple of (internal_constants, internal_attributes).
        """
        dtype = np.float32
        if "X" in inputs and inputs["X"] is not None:
            dtype = inputs["X"].dtype

        # Use the specific constant value from the pattern: 0.7071067690849304
        # This is 1/sqrt(2) with specific precision
        internal_constants = [
            (0, 1, np.sqrt(0.5).astype(dtype)),
            (2, 1, np.array(1.0, dtype=dtype)),
            (3, 1, np.array(0.5, dtype=dtype)),
        ]

        internal_attributes: dict[tuple[int, str], Any] = {}

        return internal_constants, internal_attributes

    def get_schema(self) -> PatternSchema:
        """Return the schema definition for GELU pattern.

        Returns:
            PatternSchema defining the GELU pattern's input/output types.
        """
        return _GELU_SCHEMA


@register_pattern_input_generator
class Gelu4PatternInputGenerator(PatternInputGenerator, get_runtime_checker_op("Gelu")):  # type: ignore[misc]  # dynamic base class (runtime-checker op)
    """Input generator for GELU activation pattern variant 4."""

    pattern = Gelu4Pattern()
    registration_name = "Gelu4Pattern"


@register_pattern_input_generator
class SingleGeluPatternInputGenerator(PatternInputGenerator, get_runtime_checker_op("Gelu")):  # type: ignore[misc]  # dynamic base class (runtime-checker op)
    """Input generator for native com.microsoft.Gelu single-op pattern."""

    pattern = SingleGeluPattern()
    registration_name = "SingleGeluPattern"
