# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Pattern + rewrite for batched MatMul with a constant operand on OpenVINO GPU.

OpenVINO GPU's oneDNN gemm cannot select an implementation for a batched
(rank >= 3) MatMul where an operand is a compile-time constant. The identical
gemm with a dynamic operand, and 2D constant gemm, both compile fine. Models
whose batched MatMul weights fold to constants (e.g. transformer disentangled
attention position terms) therefore fail to compile on OpenVINO GPU with:

    [GPU] Failed to select implementation for ... type: gemm

The rewrite makes the constant operand runtime-valued without changing numerics:
it routes the constant through ``Add(const, zero)`` where ``zero`` is a ``[1]``
runtime tensor derived from the MatMul's *own dynamic operand*
(``Reshape([-1]) -> Slice([0:1]) -> Sub(elem, elem) == 0``). Because ``zero`` is
data-dependent, OpenVINO's constant folder cannot collapse the ``Add`` back into
a packed gemm weight, yet ``+ 0`` leaves the values unchanged and the single
batched MatMul is preserved (no per-head decomposition, no perf regression).

Deriving ``zero`` from the dynamic operand (rather than a graph input) keeps the
replacement *local*: the rewriter only wires the target's nodes to the matched
subgraph's own boundary tensors. The two MatMul operands share a dtype (ONNX
MatMul requires it), so ``zero`` automatically matches the constant's dtype with
no Cast.

The source pattern matches a bare ``MatMul`` and validates the constant-operand
structure in ``check_skeleton_result``; it deliberately does **not** call the
base implementation, which rejects matches whose non-constant input has
symbolic/dynamic dimensions — exactly the activation operand here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from onnx.defs import OpSchema

from ..onnx import ONNXDomain
from .base import Pattern, PatternSchema, Skeleton
from .match import PatternMatchResult


if TYPE_CHECKING:
    from onnx import ModelProto

    from .match import SkeletonMatchResult


# Minimum operand rank that triggers the OpenVINO GPU gemm impl-selection failure.
_MIN_BATCHED_RANK = 3


# Source and target share this schema so PatternRewriter's schema-equality
# assertion holds (a MatMul: two same-typed operands -> one output).
_BATCHED_CONST_MATMUL_SCHEMA = PatternSchema(
    name="BatchedConstMatMulPattern",
    doc=(
        "Batched (rank >= 3) MatMul with exactly one constant operand.\n"
        "Computes Y = MatMul(A, B) where one of A/B is a compile-time constant "
        "of rank >= 3 and the other is runtime-valued. Targeted by the untie "
        "rewrite for OpenVINO GPU, whose oneDNN gemm cannot select an "
        "implementation for this shape."
    ),
    type_constraints=[
        OpSchema.TypeConstraintParam(
            type_param_str="T",
            allowed_type_strs=[
                "tensor(float16)",
                "tensor(float)",
                "tensor(double)",
            ],
            description="Constrain operands and output to float tensors.",
        )
    ],
    inputs=[
        OpSchema.FormalParameter(
            name="A",
            type_str="T",
            description="First MatMul operand (constant or runtime).",
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
        ),
        OpSchema.FormalParameter(
            name="B",
            type_str="T",
            description="Second MatMul operand (constant or runtime).",
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="Y",
            type_str="T",
            description="MatMul output.",
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
        )
    ],
)


class BatchedConstMatMulPattern(Pattern):
    """Source: a MatMul with exactly one rank->=3 constant operand."""

    def get_skeleton(self) -> Skeleton:
        """Return a single-MatMul skeleton with two virtual inputs."""
        return Skeleton(
            node_op_types=["MatMul"],
            node_domains=[ONNXDomain.AI_ONNX],
            edges=[
                (-1, 0, 0, 0),  # input A -> MatMul[0]
                (-2, 0, 0, 1),  # input B -> MatMul[1]
            ],
            exit_nodes=[0],
            n_inputs=2,
        )

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """No internal constants or attributes for a bare MatMul."""
        return [], {}

    def get_schema(self) -> PatternSchema:
        """Return the shared batched-const-MatMul schema."""
        return _BATCHED_CONST_MATMUL_SCHEMA

    def check_skeleton_result(
        self, skeleton_match_result: SkeletonMatchResult
    ) -> PatternMatchResult | None:
        """Accept only a MatMul with exactly one rank->=3 constant operand.

        This does not call ``super().check_skeleton_result``: the base
        implementation rejects matches whose non-constant input carries
        symbolic/dynamic dimensions, which is precisely the activation operand
        of a transformer MatMul. The structural predicate here depends only on
        the constant operand's rank, so symbolic activation dims are fine.
        """
        input_infos = self._build_input_infos(skeleton_match_result)

        const_names = [name for name, info in input_infos.items() if info.is_constant]
        # Exactly one constant operand: all-constant MatMuls fold away entirely
        # and never reach gemm impl selection; zero-constant MatMuls are unaffected.
        if len(const_names) != 1:
            return None

        const_info = input_infos[const_names[0]]
        rank = _operand_rank(const_info)
        if rank is None or rank < _MIN_BATCHED_RANK:
            return None

        # Build the PatternMatchResult directly (mirrors the tail of the base
        # implementation, minus the symbolic-dim rejection).
        schema = self.get_schema()
        type_param_to_type = self._infer_type_mapping(skeleton_match_result)
        schema_input_to_value = {
            param.name: skeleton_match_result.inputs[idx]
            for idx, param in enumerate(schema.inputs)
            if idx < len(skeleton_match_result.inputs)
        }
        schema_output_to_value = {}
        if schema.outputs and skeleton_match_result.output:
            schema_output_to_value[schema.outputs[0].name] = skeleton_match_result.output

        return PatternMatchResult(
            skeleton_match_result=skeleton_match_result,
            schema_input_to_value=schema_input_to_value,
            schema_output_to_value=schema_output_to_value,
            type_param_to_type=type_param_to_type,
            attributes={},
            input_infos=input_infos,
        )


class UntiedBatchedConstMatMulPattern(Pattern):
    """Target: MatMul with the constant operand routed through ``Add(const, zero)``.

    ``zero`` is a ``[1]`` runtime tensor derived from the dynamic operand, so the
    rewrite stays local and OpenVINO's constant folder cannot repack the operand
    into a gemm weight. ``get_onnx_model`` is overridden to emit the subgraph,
    since which operand is constant is only known at rewrite time.
    """

    def get_skeleton(self) -> Skeleton:
        """Return a representative skeleton (unused by the rewriter).

        The replacement is built in :meth:`get_onnx_model`; this skeleton exists
        only to satisfy the abstract base and documents the canonical RHS-constant
        topology ``Reshape -> Slice -> Sub -> Add -> MatMul``.
        """
        return Skeleton(
            node_op_types=["Reshape", "Slice", "Sub", "Add", "MatMul"],
            node_domains=[ONNXDomain.AI_ONNX] * 5,
            edges=[
                (-1, 0, 0, 0),  # dynamic A -> Reshape[0]
                (0, 0, 1, 0),  # Reshape -> Slice[0]
                (1, 0, 2, 0),  # Slice -> Sub[0]
                (1, 0, 2, 1),  # Slice -> Sub[1]
                (-2, 0, 3, 0),  # constant B -> Add[0]
                (2, 0, 3, 1),  # Sub(zero) -> Add[1]
                (-1, 0, 4, 0),  # dynamic A -> MatMul[0]
                (3, 0, 4, 1),  # Add(untied B) -> MatMul[1]
            ],
            exit_nodes=[4],
            n_inputs=2,
        )

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """No declarative constants; the subgraph is built in get_onnx_model."""
        return [], {}

    def get_schema(self) -> PatternSchema:
        """Return the shared batched-const-MatMul schema."""
        return _BATCHED_CONST_MATMUL_SCHEMA

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
        """Emit ``MatMul(dyn, Add(const, zero(dyn)))`` preserving operand order.

        The constant operand (per ``is_constant_map``) is routed through
        ``Add(const, zero)``; ``zero`` is a ``[1]`` runtime tensor built from the
        dynamic operand. Operand slots are preserved so the MatMul semantics are
        unchanged.
        """
        from onnx import helper, numpy_helper

        schema = self.get_schema()
        if input_names is None:
            input_names = [param.name for param in schema.inputs]
        if output_names is None:
            output_names = [param.name for param in schema.outputs]

        # Identify which operand is the constant (schema order: A=0, B=1).
        param_names = [param.name for param in schema.inputs]
        const_idx = 0 if is_constant_map.get(param_names[0]) else 1
        dyn_idx = 1 - const_idx
        const_name = input_names[const_idx]
        dyn_name = input_names[dyn_idx]
        out_name = output_names[0]

        nodes = []
        initializers = []

        def _init(arr: np.ndarray, name: str) -> str:
            initializers.append(numpy_helper.from_array(arr, name))
            return name

        neg1 = _init(np.array([-1], dtype=np.int64), f"{prefix}neg1")
        starts = _init(np.array([0], dtype=np.int64), f"{prefix}slice_starts")
        ends = _init(np.array([1], dtype=np.int64), f"{prefix}slice_ends")
        axis = _init(np.array([0], dtype=np.int64), f"{prefix}slice_axis")

        flat = f"{prefix}flat"
        elem = f"{prefix}elem"
        zero = f"{prefix}zero"
        untied = f"{prefix}untied"

        # flat = Reshape(dyn, [-1]); elem = Slice(flat, [0:1]); zero = elem - elem.
        # dyn and const share a dtype (ONNX MatMul), so zero needs no Cast.
        nodes.append(helper.make_node("Reshape", [dyn_name, neg1], [flat], name=f"{prefix}Reshape"))
        nodes.append(
            helper.make_node("Slice", [flat, starts, ends, axis], [elem], name=f"{prefix}Slice")
        )
        nodes.append(helper.make_node("Sub", [elem, elem], [zero], name=f"{prefix}Sub"))
        nodes.append(helper.make_node("Add", [const_name, zero], [untied], name=f"{prefix}Add"))

        # Preserve original operand order in the rebuilt MatMul.
        mm_inputs = [untied, dyn_name] if const_idx == 0 else [dyn_name, untied]
        nodes.append(helper.make_node("MatMul", mm_inputs, [out_name], name=f"{prefix}MatMul"))

        opset_imports = [
            helper.make_opsetid(domain.schema_domain, version)
            for domain, version in domain_versions.items()
        ]
        graph = helper.make_graph(
            nodes=nodes,
            name=f"{prefix}untied_batched_const_matmul",
            inputs=[],
            outputs=[],
            initializer=initializers,
        )
        model = helper.make_model(
            graph, producer_name="winmlcli-pattern-generator", opset_imports=opset_imports
        )
        model.ir_version = 11
        return model


def _operand_rank(info: Any) -> int | None:
    """Return an operand's rank from its InputInfo (shape, else constant value)."""
    if info.shape is not None:
        return len(info.shape)
    if info.value is not None:
        return int(info.value.ndim)
    return None
