# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Conv2DInplaceLinear patterns for Qualcomm NPU optimization.

Converts MatMul+Add (linear layers) to 1x1 Conv2D for Qualcomm HTP via
PatternRewriter. Based on Qualcomm AI Hub SAM model patches:
    https://github.com/quic/ai-hub-models/blob/main/qai_hub_models/models/sam/model_patches.py

These patterns share the MatMulAdd schema (A, B, C -> Y) so that
PatternRewriter can directly rewrite MatMulAddPattern matches into Conv2D
subgraphs.

Variants:
    - 4D: Transpose(A) -> Conv(A', B', C) -> Transpose(Y')
    - 3D: Transpose(A) -> Unsqueeze(A') -> Conv(A'', B', C) -> Squeeze -> Transpose(Y')
    - 2D: Reshape(A) -> Conv(A', B', C) -> Reshape(Y')

Each variant checks input dimension in get_internal_constants_and_attributes
and raises PatternMismatchedError if the input rank is incompatible,
so that PatternRewriter can skip infeasible rewrites gracefully.
"""

from typing import Any

import numpy as np

from ..onnx import ONNXDomain
from .base import (
    Pattern,
    PatternInputGenerator,
    PatternMismatchedError,
    PatternSchema,
    Skeleton,
    register_pattern_input_generator,
)
from .gemm_patterns import _MATMUL_ADD_SCHEMA
from .match import PatternMatchResult, SkeletonMatchResult
from .op_input_gen import InputShapeConstraint


def _weight_reshape_constant(
    inputs: dict[str, np.ndarray],
) -> np.ndarray | None:
    """Compute the Reshape target for the weight path.

    B is (in_f, out_f) in MatMulAdd convention.
    After Transpose([1,0]) -> (out_f, in_f).
    Reshape -> (out_f, in_f, 1, 1) for Conv2D.
    """
    if "B" in inputs and inputs["B"] is not None:
        b = inputs["B"]
        if b.ndim != 2:
            return None
        in_f, out_f = b.shape
        return np.array([out_f, in_f, 1, 1], dtype=np.int64)
    return None


def _check_bc_shapes(
    pattern_result: PatternMatchResult | None,
) -> PatternMatchResult | None:
    """Validate B is 2D and C is 1D."""
    if pattern_result is None:
        return None
    input_infos = pattern_result.input_infos
    if "B" in input_infos:
        b_shape = input_infos["B"].shape
        if b_shape is not None and len(b_shape) != 2:
            return None
    if "C" in input_infos:
        c_shape = input_infos["C"].shape
        if c_shape is not None and len(c_shape) != 1:
            return None
    return pattern_result


class Conv2DInplaceLinear4DPattern(Pattern):
    """Transpose -> Conv(1x1) -> Transpose for 4D NHWC inputs.

    Skeleton (5 nodes):
        A -> Transpose(NHWC->NCHW) -> Conv -> Transpose(NCHW->NHWC) -> Y
        B -> Transpose([1,0]) -> Reshape([out_f,in_f,1,1]) -> Conv[1]
        C -> Conv[2]
    """

    def get_skeleton(self) -> Skeleton:
        """Return skeleton for 4D pattern with weight transformation."""
        node_op_types = ["Transpose", "Transpose", "Reshape", "Conv", "Transpose"]
        node_domains = [ONNXDomain.AI_ONNX] * len(node_op_types)

        edges = [
            (-1, 0, 0, 0),  # A -> Transpose_data[0]
            (-2, 0, 1, 0),  # B -> Transpose_weight[0]
            (1, 0, 2, 0),   # Transpose_weight -> Reshape_weight[0]
            (0, 0, 3, 0),   # Transpose_data -> Conv[0]
            (2, 0, 3, 1),   # Reshape_weight -> Conv[1]
            (-3, 0, 3, 2),  # C -> Conv[2]
            (3, 0, 4, 0),   # Conv -> Transpose_out[0]
        ]

        return Skeleton(
            node_op_types=node_op_types,
            node_domains=node_domains,
            edges=edges,
            exit_nodes=[4],
            n_inputs=3,
        )

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """Return weight reshape shape and node attributes.

        Raises PatternMismatchedError if A is not 4D.
        """
        if "A" in inputs and inputs["A"] is not None and inputs["A"].ndim != 4:
            raise PatternMismatchedError(
                f"Conv2DInplaceLinear4D requires 4D input A, got {inputs['A'].ndim}D"
            )

        internal_constants: list[tuple[int, int, np.ndarray]] = []

        w_shape = _weight_reshape_constant(inputs)
        if w_shape is not None:
            internal_constants.append((2, 1, w_shape))

        internal_attributes: dict[tuple[int, str], Any] = {
            (0, "perm"): [0, 3, 1, 2],   # NHWC -> NCHW
            (1, "perm"): [1, 0],          # Transpose weight
            (3, "kernel_shape"): [1, 1],
            (4, "perm"): [0, 2, 3, 1],   # NCHW -> NHWC
        }
        return internal_constants, internal_attributes

    def get_schema(self) -> PatternSchema:
        """Return shared MatMulAdd schema."""
        return _MATMUL_ADD_SCHEMA

    def check_skeleton_result(
        self, skeleton_match_result: SkeletonMatchResult,
    ) -> PatternMatchResult | None:
        """Validate shape constraints."""
        return _check_bc_shapes(super().check_skeleton_result(skeleton_match_result))


class Conv2DInplaceLinear3DPattern(Pattern):
    """Transpose -> Unsqueeze -> Conv(1x1) -> Squeeze -> Transpose for 3D inputs.

    Skeleton (7 nodes):
        A -> Transpose -> Unsqueeze -> Conv -> Squeeze -> Transpose -> Y
        B -> Transpose([1,0]) -> Reshape([out_f,in_f,1,1]) -> Conv[1]
        C -> Conv[2]
    """

    def get_skeleton(self) -> Skeleton:
        """Return skeleton for 3D pattern with weight transformation."""
        node_op_types = [
            "Transpose",  # 0: data (N,seq,in_f) -> (N,in_f,seq)
            "Unsqueeze",  # 1: add spatial dim
            "Transpose",  # 2: weight transpose
            "Reshape",    # 3: weight reshape to 4D
            "Conv",       # 4: 1x1 conv
            "Squeeze",    # 5: remove spatial dim
            "Transpose",  # 6: data back (N,out_f,seq) -> (N,seq,out_f)
        ]
        node_domains = [ONNXDomain.AI_ONNX] * len(node_op_types)

        edges = [
            (-1, 0, 0, 0),  # A -> Transpose_data[0]
            (0, 0, 1, 0),   # Transpose_data -> Unsqueeze[0]
            (-2, 0, 2, 0),  # B -> Transpose_weight[0]
            (2, 0, 3, 0),   # Transpose_weight -> Reshape_weight[0]
            (1, 0, 4, 0),   # Unsqueeze -> Conv[0]
            (3, 0, 4, 1),   # Reshape_weight -> Conv[1]
            (-3, 0, 4, 2),  # C -> Conv[2]
            (4, 0, 5, 0),   # Conv -> Squeeze[0]
            (5, 0, 6, 0),   # Squeeze -> Transpose_out[0]
        ]

        return Skeleton(
            node_op_types=node_op_types,
            node_domains=node_domains,
            edges=edges,
            exit_nodes=[6],
            n_inputs=3,
        )

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """Return internal constants and attributes.

        Raises PatternMismatchedError if A is not 3D.
        """
        if "A" in inputs and inputs["A"] is not None and inputs["A"].ndim != 3:
            raise PatternMismatchedError(
                f"Conv2DInplaceLinear3D requires 3D input A, got {inputs['A'].ndim}D"
            )

        internal_constants: list[tuple[int, int, np.ndarray]] = [
            (1, 1, np.array([-1], dtype=np.int64)),  # Unsqueeze axes
            (5, 1, np.array([-1], dtype=np.int64)),  # Squeeze axes
        ]

        w_shape = _weight_reshape_constant(inputs)
        if w_shape is not None:
            internal_constants.append((3, 1, w_shape))

        internal_attributes: dict[tuple[int, str], Any] = {
            (0, "perm"): [0, 2, 1],
            (2, "perm"): [1, 0],
            (4, "kernel_shape"): [1, 1],
            (6, "perm"): [0, 2, 1],
        }
        return internal_constants, internal_attributes

    def get_schema(self) -> PatternSchema:
        """Return shared MatMulAdd schema."""
        return _MATMUL_ADD_SCHEMA

    def check_skeleton_result(
        self, skeleton_match_result: SkeletonMatchResult,
    ) -> PatternMatchResult | None:
        """Validate shape constraints."""
        return _check_bc_shapes(super().check_skeleton_result(skeleton_match_result))


class Conv2DInplaceLinear2DPattern(Pattern):
    """Reshape -> Conv(1x1) -> Reshape for 2D inputs (Gemm replacement).

    Skeleton (5 nodes):
        A -> Reshape([0,-1,1,1]) -> Conv -> Reshape([0,-1]) -> Y
        B -> Transpose([1,0]) -> Reshape([out_f,in_f,1,1]) -> Conv[1]
        C -> Conv[2]
    """

    def get_skeleton(self) -> Skeleton:
        """Return skeleton for 2D pattern with weight transformation."""
        node_op_types = ["Reshape", "Transpose", "Reshape", "Conv", "Reshape"]
        node_domains = [ONNXDomain.AI_ONNX] * len(node_op_types)

        edges = [
            (-1, 0, 0, 0),  # A -> Reshape_data[0]
            (-2, 0, 1, 0),  # B -> Transpose_weight[0]
            (1, 0, 2, 0),   # Transpose_weight -> Reshape_weight[0]
            (0, 0, 3, 0),   # Reshape_data -> Conv[0]
            (2, 0, 3, 1),   # Reshape_weight -> Conv[1]
            (-3, 0, 3, 2),  # C -> Conv[2]
            (3, 0, 4, 0),   # Conv -> Reshape_out[0]
        ]

        return Skeleton(
            node_op_types=node_op_types,
            node_domains=node_domains,
            edges=edges,
            exit_nodes=[4],
            n_inputs=3,
        )

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """Return reshape shape constants and Conv kernel_shape attribute.

        Raises PatternMismatchedError if A is not 2D.
        """
        if "A" in inputs and inputs["A"] is not None and inputs["A"].ndim != 2:
            raise PatternMismatchedError(
                f"Conv2DInplaceLinear2D requires 2D input A, got {inputs['A'].ndim}D"
            )

        internal_constants: list[tuple[int, int, np.ndarray]] = [
            (0, 1, np.array([0, -1, 1, 1], dtype=np.int64)),
            (4, 1, np.array([0, -1], dtype=np.int64)),
        ]

        w_shape = _weight_reshape_constant(inputs)
        if w_shape is not None:
            internal_constants.append((2, 1, w_shape))

        internal_attributes: dict[tuple[int, str], Any] = {
            (1, "perm"): [1, 0],
            (3, "kernel_shape"): [1, 1],
        }
        return internal_constants, internal_attributes

    def get_schema(self) -> PatternSchema:
        """Return shared MatMulAdd schema."""
        return _MATMUL_ADD_SCHEMA

    def check_skeleton_result(
        self, skeleton_match_result: SkeletonMatchResult,
    ) -> PatternMatchResult | None:
        """Validate shape constraints."""
        return _check_bc_shapes(super().check_skeleton_result(skeleton_match_result))


# ---------------------------------------------------------------------------
# Input generators
# ---------------------------------------------------------------------------


class Conv2DInplaceLinearInputGeneratorBase(PatternInputGenerator):
    """Shared input generator for Conv2DInplaceLinear patterns.

    Subclasses override ``_get_a_shapes`` for topology-appropriate A shapes.
    """

    def _get_a_shapes(self) -> list[tuple[int, ...]]:
        """Return A shapes compatible with this pattern's topology."""
        raise NotImplementedError

    def get_finite_attribute_sets(self) -> dict[str, list[Any]]:
        """No finite attribute sets."""
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, Any]]:
        """Generate input shape combinations for A, B, and C."""
        in_f = 4
        out_f = 8
        b_shape = (in_f, out_f)
        c_shape = (out_f,)

        return [
            {
                "A": InputShapeConstraint(a_shape),
                "B": InputShapeConstraint(b_shape),
                "C": InputShapeConstraint(c_shape),
            }
            for a_shape in self._get_a_shapes()
        ]

    def derive_properties(self, properties: dict) -> dict:
        """Add input dimension to properties."""
        item = properties.copy()
        item["A_dim"] = len(item["A_shape"])
        item["B_dim"] = len(item["B_shape"])
        if "C_shape" in item:
            item["C_dim"] = len(item["C_shape"])
        else:
            item["C_dim"] = 0
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return shape property names."""
        return ["A_shape", "B_shape", "C_shape"]


@register_pattern_input_generator
class Conv2DInplaceLinear4DPatternInputGenerator(
    Conv2DInplaceLinearInputGeneratorBase,
):
    """Input generator for 4D Conv2DInplaceLinear pattern."""

    pattern = Conv2DInplaceLinear4DPattern()
    registration_name = "Conv2DInplaceLinear4D"

    def _get_a_shapes(self) -> list[tuple[int, ...]]:
        """4D NHWC shapes."""
        in_f = 4
        return [(1, 8, 8, in_f), (2, 4, 4, in_f)]


@register_pattern_input_generator
class Conv2DInplaceLinear3DPatternInputGenerator(
    Conv2DInplaceLinearInputGeneratorBase,
):
    """Input generator for 3D Conv2DInplaceLinear pattern."""

    pattern = Conv2DInplaceLinear3DPattern()
    registration_name = "Conv2DInplaceLinear3D"

    def _get_a_shapes(self) -> list[tuple[int, ...]]:
        """3D shapes (batch, seq, features)."""
        in_f = 4
        return [(1, 10, in_f), (2, 5, in_f)]


@register_pattern_input_generator
class Conv2DInplaceLinear2DPatternInputGenerator(
    Conv2DInplaceLinearInputGeneratorBase,
):
    """Input generator for 2D Conv2DInplaceLinear pattern."""

    pattern = Conv2DInplaceLinear2DPattern()
    registration_name = "Conv2DInplaceLinear2D"

    def _get_a_shapes(self) -> list[tuple[int, ...]]:
        """2D shapes."""
        in_f = 4
        return [(1, in_f), (4, in_f)]
