# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""RMSNorm to LpNormalization fusion.

Detects decomposed RMSNorm subgraphs and replaces them with
LpNormalization(p=2) + weight-adjusted Mul. This produces a standard
ONNX op (LpNormalization) instead of a Microsoft-specific op
(SimplifiedLayerNormalization), making it compatible with QNN EP.

The RMSNorm pattern (after ORT graph optimization):

    root --------------------------------------+
     |                                         |
     v                                         v
    Pow(2) -> ReduceMean -> Add(e) -> Sqrt -> Div(root, sqrt) -> Mul(weight)

Replacement:

    root -> LpNormalization(p=2, axis=-1) -> Mul(weight * sqrt(N))

The weight is multiplied by sqrt(N) (hidden_size) to compensate for the
difference between L2 norm and RMS norm: RMS(x) = L2(x) / sqrt(N).

Reference: Olive RMSNormToL2Norm in olive/passes/onnx/graph_surgeries.py
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from onnx import helper, numpy_helper
from onnxruntime.transformers.fusion_base import Fusion


if TYPE_CHECKING:
    from onnx import NodeProto
    from onnxruntime.transformers.onnx_model import OnnxModel


logger = logging.getLogger(__name__)


class FusionRMSNorm(Fusion):  # type: ignore[misc]  # ORT Fusion base ships no stubs (Any)
    """Replace decomposed RMSNorm with LpNormalization(p=2).

    Searches backward from Mul nodes (the final weight scaling) to find
    the full RMSNorm pattern. Uses ORT's match_parent_path for graph
    traversal.
    """

    def __init__(self, model: OnnxModel) -> None:
        super().__init__(model, "LpNormalization", "Mul", "FuseRMSNorm")

    def fuse(
        self,
        node: NodeProto,
        input_name_to_nodes: dict[str, list[NodeProto]],
        output_name_to_node: dict[str, NodeProto],
    ) -> None:
        """Match RMSNorm pattern and replace with LpNormalization."""
        # The weight Mul: Mul(weight_initializer, normalized_output)
        weight_index = self._find_weight_input(node)
        if weight_index is None:
            return

        # Walk backward: Mul <- Div <- Sqrt <- Add <- ReduceMean <- Pow
        path = self.model.match_parent_path(
            node,
            ["Div", "Sqrt", "Add", "ReduceMean", "Pow"],
            [None, None, 0, None, None],
            output_name_to_node,
        )
        if path is None:
            return

        div_node, sqrt_node, add_node, reduce_mean, pow_node = path

        # Validate Pow exponent is 2.0
        if not self.model.has_constant_input(pow_node, 2.0):
            return

        # Get root input (the tensor being normalized)
        root_input = pow_node.input[0]

        # Validate root feeds into Div (skip connection)
        if root_input not in div_node.input:
            return

        # Get weight tensor and hidden size
        weight_name = node.input[weight_index]
        weight_init = self.model.get_initializer(weight_name)
        if weight_init is None:
            return
        weight_array = numpy_helper.to_array(weight_init)
        hidden_size = weight_array.shape[-1]

        # Collect all nodes to remove
        all_nodes = [node, div_node, sqrt_node, add_node, reduce_mean, pow_node]

        # Safety check
        if not self.model.is_safe_to_fuse_nodes(
            all_nodes,
            [node.output[0]],
            input_name_to_nodes,
            output_name_to_node,
        ):
            return

        # -- Create replacement --

        # 1. LpNormalization node
        l2norm_name = self.model.create_node_name("LpNormalization", "RMSNorm_")
        l2norm_output = f"{l2norm_name}_output_0"

        # Note: LpNormalization has no epsilon parameter. The original
        # Add(eps) for numerical stability is dropped. This is acceptable:
        # 1. LpNormalization implementations have internal epsilon guards
        # 2. Transformer hidden states are empirically never all-zero
        # 3. Matches Olive RMSNormToL2Norm approach
        l2norm_node = helper.make_node(
            "LpNormalization",
            inputs=[root_input],
            outputs=[l2norm_output],
            name=l2norm_name,
            axis=-1,
            p=2,
        )

        # 2. Adjust weight: weight * sqrt(N)
        sqrt_n = np.sqrt(hidden_size).astype(weight_array.dtype)
        adjusted_weight = weight_array * sqrt_n

        # If all weights are 1.0 (rotated models), collapse to scalar
        if np.allclose(weight_array, 1.0):
            adjusted_weight = np.array([sqrt_n], dtype=weight_array.dtype)

        adjusted_weight_name = f"{weight_name}_l2norm_adjusted"
        adjusted_tensor = numpy_helper.from_array(
            adjusted_weight, name=adjusted_weight_name
        )
        self.model.add_initializer(adjusted_tensor)

        # 3. New Mul with adjusted weight
        new_mul_name = self.model.create_node_name("Mul", "RMSNorm_Scale_")
        new_mul = helper.make_node(
            "Mul",
            inputs=[l2norm_output, adjusted_weight_name],
            outputs=[node.output[0]],
            name=new_mul_name,
        )

        # -- Apply --
        self.nodes_to_remove.extend(all_nodes)
        self.nodes_to_add.extend([l2norm_node, new_mul])
        self.node_name_to_graph_name[l2norm_name] = self.this_graph_name
        self.node_name_to_graph_name[new_mul_name] = self.this_graph_name

        self.increase_counter("RMSNorm")

    def _find_weight_input(self, mul_node: NodeProto) -> int | None:
        """Find which Mul input is an initializer weight. Returns index or None."""
        for i, inp in enumerate(mul_node.input):
            if self.model.get_initializer(inp) is not None:
                return i
        return None
