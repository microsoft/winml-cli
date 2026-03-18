"""Generate ORT Graph Optimization Test Patterns (v2 - Modular).

Creates a single ONNX model with pattern types using a universal template.
This is the modular version that imports builders from submodules.

Template Structure:
X ──┬─→ Pattern → Pattern ──┐
    └─→ Pattern ────────────┴─→ Add → Y

This tests:
1. CSE: First Pattern on both paths is identical → one eliminated
2. Consecutive fusion: Top path has 2 chained patterns
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import onnx
from onnx import TensorProto, helper

# Activation patterns
from .builders.activation import (
    bias_dropout_builder,
    relu_clip_builder,
)

# Attention patterns (existing module)
# NOTE: AttentionFusion (ORT's real pattern) CANNOT use PatternTemplate because:
# - Requires complete ModelProto via create_ort_attention_fusion_model
# - Requires 2 inputs (hidden states + int32 mask) - PatternTemplate only supports 1 input
# - Uses complex mask processing chain (Unsqueeze→Cast→Sub→Mul)
# - Uses Div for scaling (not Mul) to match ORT's pattern
# AttentionFusion is tested in test_pipe_graph_isolated.py using BUILDER_REGISTRY.
from .builders.attention import (
    multi_head_attention_builder,
    rotary_embeddings_builder,
)
from .builders.conv import (
    conv_activation_builder as conv_activation_fusion_builder,
)
from .builders.conv import (
    conv_add_activation_builder as conv_add_activation_fusion_builder,
)

# Conv patterns (existing module, some additions needed)
from .builders.conv import (
    conv_add_fusion_builder,
    conv_add_relu_builder,
    conv_bn_builder,
    nchwc_transformer_builder,
    pad_conv_builder,
)
from .builders.conv import (
    conv_mul_builder as conv_mul_fusion_builder,
)

# =============================================================================
# Import all builders from submodules
# =============================================================================
# Core patterns (NEW module needed)
from .builders.core import (
    constant_folding_builder,
    cse_builder,
    identity_relu_builder,
    reshape_builder,
)

# Elimination patterns (existing module)
from .builders.elimination import (
    concat_slice_elimination_builder,
    reshape_elimination_builder,
    slice_elimination_builder,
    unsqueeze_elimination_builder,
)

# GELU patterns (NEW module needed)
from .builders.gelu import (
    bias_gelu_builder,
    fast_gelu_builder,
    gelu_approximation_builder,
    quick_gelu_builder,
)

# Gemm patterns (existing module, name mappings)
from .builders.gemm import (
    gemm_activation_builder,
)
from .builders.gemm import (
    gemm_sum_builder as gemm_sum_fusion_builder,
)
from .builders.gemm import (
    gemm_transpose_builder as gemm_transpose_fusion_builder,
)
from .builders.layernorm import (
    bias_skip_layernorm_builder as bias_skip_layer_norm_builder,
)
from .builders.layernorm import (
    simplified_layernorm_builder as simplified_layer_norm_builder,
)

# LayerNorm patterns (existing module, some name mappings)
from .builders.layernorm import (
    skip_layernorm_builder as skip_layer_norm_builder,
)

# MatMul patterns (existing module, some additions needed)
from .builders.matmul import (
    dynamic_quantize_matmul_builder,
    matmul_activation_builder,
    matmul_add_relu_builder,
    matmul_bn_builder,
    matmul_scale_builder,
    matmul_transpose_builder,
)

# Misc patterns (existing module, additions needed)
from .builders.misc import (
    concat_slice_builder,
    gather_split_builder,
    noop_elimination_builder,
    qdq_pairs_builder,
    reduce_softmax_builder,
    softmax_builder,
    transpose_chain_builder,
)
from .builders.misc import (
    gather_to_slice_builder as gather_slice_builder,
)
from .builders.misc import (
    not_where_builder as not_where_fusion_builder,
)


if TYPE_CHECKING:
    from collections.abc import Callable


def make_compatible_model(
    graph: onnx.GraphProto,
    opset_version: int = 17,
    include_ms_domain: bool = True,
) -> onnx.ModelProto:
    """Create model with IR version compatible with ORT.

    Uses opset 17 by default to support LayerNormalization op (added in opset 17),
    which is required for SkipLayerNormFusion testing.

    Args:
        graph: The ONNX graph to wrap in a model.
        opset_version: ONNX opset version (default: 17).
        include_ms_domain: Include com.microsoft domain for fused ops like Attention.
    """
    opset_imports = [helper.make_opsetid("", opset_version)]

    # Add com.microsoft domain for contrib operators (Attention, EmbedLayerNormalization, etc.)
    if include_ms_domain:
        opset_imports.append(helper.make_opsetid("com.microsoft", 1))

    model = helper.make_model(graph, opset_imports=opset_imports)
    model.ir_version = 8
    return model


class PatternTemplate:
    """Universal container for all patterns.

    Creates the structure:
    X ──┬─→ Pattern → Pattern ──┐
        └─→ Pattern ────────────┴─→ Add → Y
    """

    def __init__(
        self,
        prefix: str,
        x_shape: tuple,
        pattern_builder: Callable,
        extra_inputs: list | None = None,
    ):
        """
        Args:
            prefix: Unique prefix for this pattern instance
            x_shape: Shape of input tensor X
            pattern_builder: Function that builds pattern nodes
            extra_inputs: Additional inputs for special patterns
        """
        self.prefix = prefix
        self.x_shape = x_shape
        self.pattern_builder = pattern_builder
        self.extra_inputs = extra_inputs or []

    def build(self) -> tuple[list, list, list, list]:
        """Build the pattern and return (nodes, inputs, outputs, initializers)."""
        input_name = f"{self.prefix}X"
        output_name = f"{self.prefix}Y"
        inter1 = f"{self.prefix}inter1"
        inter2 = f"{self.prefix}inter2"
        inter3 = f"{self.prefix}inter3"

        nodes = []
        initializers = []

        # Top path: X → Pattern → inter1 → Pattern → inter2
        top_path_1 = self.pattern_builder(input_name, inter1, f"{self.prefix}top1_", initializers)
        top_path_2 = self.pattern_builder(inter1, inter2, f"{self.prefix}top2_", initializers)
        nodes.extend(top_path_1)
        nodes.extend(top_path_2)

        # Bottom path: X → Pattern → inter3 (CSE candidate with top_path_1)
        bottom_path = self.pattern_builder(input_name, inter3, f"{self.prefix}bot_", initializers)
        nodes.extend(bottom_path)

        # Merge: inter2 + inter3 → Y
        nodes.append(
            helper.make_node(
                "Add",
                [inter2, inter3],
                [output_name],
                name=f"{self.prefix}merge",
            )
        )

        # Create input/output value infos
        inputs = [
            helper.make_tensor_value_info(input_name, TensorProto.FLOAT, list(self.x_shape))
        ]
        outputs = [
            helper.make_tensor_value_info(output_name, TensorProto.FLOAT, list(self.x_shape))
        ]

        return nodes, inputs, outputs, initializers


# =============================================================================
# PATTERN REGISTRY
# =============================================================================
# Maps pattern name → (builder_function, input_shape)

PATTERN_REGISTRY = {
    # Phase 1: Core patterns
    "identity": (identity_relu_builder, (1, 64)),
    "constfold": (constant_folding_builder, (1, 64)),
    "cse": (cse_builder, (1, 64)),
    "convbn": (conv_bn_builder, (1, 16, 32, 32)),
    "convaddrelu": (conv_add_relu_builder, (1, 16, 32, 32)),
    "matmuladdrelu": (matmul_add_relu_builder, (1, 64)),
    "reshape": (reshape_builder, (1, 64)),
    "biasgelu": (bias_gelu_builder, (1, 64)),
    "skiplayernorm": (skip_layer_norm_builder, (1, 4, 64)),  # 3D for SkipLayerNormFusion
    # Phase 2: Activation and optimizer patterns
    "softmax": (softmax_builder, (1, 64)),
    "reluclip": (relu_clip_builder, (1, 64)),
    "matmulact": (matmul_activation_builder, (1, 64)),
    "transpose": (transpose_chain_builder, (64, 64)),
    "simpln": (simplified_layer_norm_builder, (1, 64)),
    "reducesoftmax": (reduce_softmax_builder, (1, 64)),
    # Phase 3: Specialized patterns
    "gemmact": (gemm_activation_builder, (1, 64)),
    "gatherslice": (gather_slice_builder, (1, 64)),
    "padconv": (pad_conv_builder, (1, 64)),
    "qdqpairs": (qdq_pairs_builder, (1, 64)),
    # NOTE: embedln removed from PATTERN_REGISTRY - EmbedLayerNormFusion has complex
    # requirements (integer input_ids, Gather ops, Attention node following) that
    # don't fit the PatternTemplate chain model. The pattern changes shape through
    # Flatten/Gather ops which breaks chaining. Test separately if needed.
    # "embedln": (embed_layer_norm_builder, (1, 64)),
    # Phase 3: GEMM and Conv variant patterns
    "gemmsum": (gemm_sum_fusion_builder, (1, 64)),
    "gemmtrans": (gemm_transpose_fusion_builder, (64, 64)),
    "convmul": (conv_mul_fusion_builder, (1, 16, 32, 32)),
    "convadd": (conv_add_fusion_builder, (1, 16, 32, 32)),  # ConvAddFusion
    "convact": (conv_activation_fusion_builder, (1, 16, 32, 32)),
    "convaddact": (conv_add_activation_fusion_builder, (1, 16, 32, 32)),
    # Phase 3: MatMul variant patterns
    "matmulbn": (matmul_bn_builder, (1, 64)),
    "matmulscale": (matmul_scale_builder, (1, 64)),
    "matmultrans": (matmul_transpose_builder, (64, 64)),
    "dynquant": (dynamic_quantize_matmul_builder, (1, 64)),
    # Phase 4: GELU variant patterns
    "fastgelu": (fast_gelu_builder, (1, 64)),
    "quickgelu": (quick_gelu_builder, (1, 64)),
    "geluapprox": (gelu_approximation_builder, (1, 64)),
    "biasdropout": (bias_dropout_builder, (1, 64)),
    # Phase 4: Layout patterns
    "nchwc": (nchwc_transformer_builder, (1, 64)),
    # Phase 4: Misc patterns
    "notwhere": (not_where_fusion_builder, (1, 64)),
    "noop": (noop_elimination_builder, (1, 64)),
    # Phase 4: Attention-related patterns
    # NOTE: AttentionFusion REMOVED from PATTERN_REGISTRY - it requires:
    # - Complete ModelProto (not nodes) via create_ort_attention_fusion_model
    # - 2 inputs (hidden states + int32 mask) - PatternTemplate only supports 1 input
    # - Complex mask chain (Unsqueeze→Cast→Sub→Mul) that can't be chained
    # - Div for scaling (not Mul) to trigger ORT's pattern matching
    # AttentionFusion is tested in test_pipe_graph_isolated.py using BUILDER_REGISTRY.
    # "attention": (attention_fusion_builder, (1, 4, 64)),  # DOES NOT trigger ORT fusion!
    #
    # The patterns below are structural tests only, they may not trigger actual fusions:
    "mha": (multi_head_attention_builder, (1, 64)),
    "rotary": (rotary_embeddings_builder, (1, 64)),
    "biasskiln": (bias_skip_layer_norm_builder, (1, 64)),
    # Phase 5: Gather and Slice patterns
    "gathersplit": (gather_split_builder, (1, 2)),
    "concatslice": (concat_slice_builder, (1, 64)),
    # Phase 6: Elimination patterns (using builders from elimination.py)
    "sliceelim": (slice_elimination_builder, (1, 64)),
    "unsqueezeelim": (unsqueeze_elimination_builder, (1, 64)),
    "reshapeelim": (reshape_elimination_builder, (1, 64)),
    "concatsliceelim": (concat_slice_elimination_builder, (1, 64)),
}


def create_all_patterns_model() -> onnx.ModelProto:
    """Create ONNX model with all patterns using universal template."""
    all_nodes = []
    all_inputs = []
    all_outputs = []
    all_initializers = []

    for idx, (pattern_name, (builder, shape)) in enumerate(PATTERN_REGISTRY.items(), start=1):
        prefix = f"p{idx:02d}_{pattern_name}_"

        template = PatternTemplate(prefix, shape, builder)
        nodes, inputs, outputs, initializers = template.build()

        all_nodes.extend(nodes)
        all_inputs.extend(inputs)
        all_outputs.extend(outputs)
        all_initializers.extend(initializers)

    # ==========================================================================
    # SPECIAL PATTERNS: EmbedLayerNorm - DISABLED
    # ==========================================================================
    # EmbedLayerNormFusion CANNOT be tested in isolation because:
    #
    # ORT's EmbedLayerNormFusion has two code paths:
    # 1. FuseSubGraphDistilBert: Requires complex position embedding subgraph
    #    (Shape->Expand->Gather pattern from transformers library export)
    # 2. FuseSubGraph (full BERT): Requires segment embeddings (3 Gathers)
    #
    # A simple Gather(pos_embed_table, pos_ids) pattern doesn't match either path:
    # - FuseSubGraphDistilBert calls MatchPositionEmbeddingSubgraphsFromGather
    #   which expects complex Shape/Expand/Gather subgraphs
    # - FuseSubGraph requires segment embeddings which we don't have
    #
    # As a result, SkipLayerNormFusion intercepts the Add->LayerNorm pattern
    # before EmbedLayerNormFusion can process it.
    #
    # EmbedLayerNormFusion is now tested via embed_layer_norm_builder() in
    # test_pipe_graph_isolated.py using BUILDER_REGISTRY with model_factory.
    # The builder returns a complete ModelProto based on ORT's Format 5 (DistilBERT).

    graph = helper.make_graph(
        all_nodes,
        "ort_graph_optim_patterns",
        all_inputs,
        all_outputs,
        initializer=all_initializers,
    )
    return make_compatible_model(graph)


def main():
    output_dir = Path(__file__).parent.parent.parent.parent / "temp" / "ort_test_patterns"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ort_graph_optim_all_patterns.onnx"

    print("=" * 70)
    print("ORT GRAPH OPTIMIZATION TEST PATTERN GENERATOR (v2)")
    print("=" * 70)

    # Create model
    model = create_all_patterns_model()

    # Validate
    onnx.checker.check_model(model)

    # Save
    onnx.save(model, str(output_path))

    # Print stats
    print(f"\nSaved: {output_path}")
    print(f"Total nodes: {len(model.graph.node)}")
    print(f"Total inputs: {len(model.graph.input)}")
    print(f"Total outputs: {len(model.graph.output)}")
    print(f"Total initializers: {len(model.graph.initializer)}")
    print(f"Total patterns: {len(PATTERN_REGISTRY)}")


if __name__ == "__main__":
    main()
