"""LayerNorm Pattern Builders for ORT Graph Optimization Testing.

This module provides corrected builders for LayerNorm fusion patterns that match
ORT's expected graph structures based on analysis of:
- onnxruntime/core/optimizer/layer_norm_fusion.cc
- onnxruntime/core/optimizer/skip_layer_norm_fusion.cc
- onnxruntime/core/optimizer/embed_layer_norm_fusion.cc

Key patterns:
1. decomposed_layernorm_builder: Full decomposed LayerNorm for LayerNormFusionL2
2. simplified_layernorm_builder: Variance-only SimplifiedLayerNormFusion
3. skip_layernorm_builder: Add + LayerNorm for SkipLayerNormFusion (Format 3)
4. bias_skip_layernorm_builder: Bias + Skip + LayerNorm (Format 1/2)
5. embed_layer_norm_builder: Returns ModelProto for EmbedLayerNormFusion (Format 5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from onnx import TensorProto, helper, numpy_helper


if TYPE_CHECKING:
    import onnx


def decomposed_layernorm_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build decomposed LayerNorm pattern for LayerNormFusionL2.

    Creates the exact pattern ORT's LayerNormFusion (L2) expects:

        +---------------------+
        |                     |
        |                     v
    X --+--> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                             |                                               ^
                             |                                               |
                             +-----------------------------------------------+

    Based on layer_norm_fusion.cc lines 81-139.

    Critical requirements:
    - ReduceMean axes must be consecutive and include last dim (e.g., [-1])
    - Both ReduceMean nodes must have same axes
    - Sub output connects to both Pow and Div
    - Scale (gamma) shape must be 1D matching normalized axis
    - Bias (beta) shape must be 1D matching normalized axis
    - Pow exponent must be 2.0
    """
    hidden_size = 64

    # Scale (gamma) - 1D along last axis
    initializers.append(
        numpy_helper.from_array(np.ones(hidden_size, dtype=np.float32), f"{prefix}gamma")
    )
    # Bias (beta) - 1D along last axis
    initializers.append(
        numpy_helper.from_array(np.zeros(hidden_size, dtype=np.float32), f"{prefix}beta")
    )
    # Epsilon for numerical stability
    initializers.append(numpy_helper.from_array(np.array([1e-5], dtype=np.float32), f"{prefix}eps"))
    # Exponent for Pow (must be 2.0)
    initializers.append(numpy_helper.from_array(np.array([2.0], dtype=np.float32), f"{prefix}two"))

    return [
        # Step 1: Compute mean of input along last axis
        helper.make_node(
            "ReduceMean",
            [input_name],
            [f"{prefix}mean"],
            name=f"{prefix}reducemean1",
            axes=[-1],
            keepdims=1,
        ),
        # Step 2: Subtract mean from input (center the data)
        helper.make_node(
            "Sub",
            [input_name, f"{prefix}mean"],
            [f"{prefix}centered"],
            name=f"{prefix}sub",
        ),
        # Step 3: Square the centered values
        helper.make_node(
            "Pow",
            [f"{prefix}centered", f"{prefix}two"],
            [f"{prefix}squared"],
            name=f"{prefix}pow",
        ),
        # Step 4: Compute variance (mean of squared values)
        helper.make_node(
            "ReduceMean",
            [f"{prefix}squared"],
            [f"{prefix}var"],
            name=f"{prefix}reducemean2",
            axes=[-1],
            keepdims=1,
        ),
        # Step 5: Add epsilon for numerical stability
        helper.make_node(
            "Add",
            [f"{prefix}var", f"{prefix}eps"],
            [f"{prefix}var_eps"],
            name=f"{prefix}add_eps",
        ),
        # Step 6: Compute standard deviation
        helper.make_node(
            "Sqrt",
            [f"{prefix}var_eps"],
            [f"{prefix}std"],
            name=f"{prefix}sqrt",
        ),
        # Step 7: Normalize by dividing centered by std
        # NOTE: Sub output connects here (important for fusion pattern matching)
        helper.make_node(
            "Div",
            [f"{prefix}centered", f"{prefix}std"],
            [f"{prefix}normalized"],
            name=f"{prefix}div",
        ),
        # Step 8: Scale by gamma
        helper.make_node(
            "Mul",
            [f"{prefix}normalized", f"{prefix}gamma"],
            [f"{prefix}scaled"],
            name=f"{prefix}mul_gamma",
        ),
        # Step 9: Shift by beta
        helper.make_node(
            "Add",
            [f"{prefix}scaled", f"{prefix}beta"],
            [output_name],
            name=f"{prefix}add_beta",
        ),
    ]


def simplified_layernorm_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build SimplifiedLayerNorm pattern (variance-only, no mean subtraction).

    Creates the pattern for SimplifiedLayerNormFusion:

    X --+--> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul
        |                                            ^
        |                                            |
        +--------------------------------------------+

    Based on layer_norm_fusion.cc lines 517-539.

    This is RMS normalization - normalizes by root mean square without centering.
    No bias term (just scale), used in some transformer variants.
    """
    hidden_size = 64

    # Scale (gamma) - 1D along last axis
    initializers.append(
        numpy_helper.from_array(np.ones(hidden_size, dtype=np.float32), f"{prefix}gamma")
    )
    # Epsilon
    initializers.append(numpy_helper.from_array(np.array([1e-5], dtype=np.float32), f"{prefix}eps"))
    # Exponent for Pow
    initializers.append(numpy_helper.from_array(np.array([2.0], dtype=np.float32), f"{prefix}two"))

    return [
        # Step 1: Square the input
        helper.make_node(
            "Pow",
            [input_name, f"{prefix}two"],
            [f"{prefix}squared"],
            name=f"{prefix}pow",
        ),
        # Step 2: Compute mean of squared values (variance without centering)
        helper.make_node(
            "ReduceMean",
            [f"{prefix}squared"],
            [f"{prefix}var"],
            name=f"{prefix}reducemean",
            axes=[-1],
            keepdims=1,
        ),
        # Step 3: Add epsilon
        helper.make_node(
            "Add",
            [f"{prefix}var", f"{prefix}eps"],
            [f"{prefix}var_eps"],
            name=f"{prefix}add_eps",
        ),
        # Step 4: Compute RMS (root mean square)
        helper.make_node(
            "Sqrt",
            [f"{prefix}var_eps"],
            [f"{prefix}rms"],
            name=f"{prefix}sqrt",
        ),
        # Step 5: Normalize by dividing input by RMS
        # NOTE: Input connects directly here (important for pattern matching)
        helper.make_node(
            "Div",
            [input_name, f"{prefix}rms"],
            [f"{prefix}normalized"],
            name=f"{prefix}div",
        ),
        # Step 6: Scale by gamma
        helper.make_node(
            "Mul",
            [f"{prefix}normalized", f"{prefix}gamma"],
            [output_name],
            name=f"{prefix}mul_gamma",
        ),
    ]


def skip_layernorm_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    r"""Build SkipLayerNorm pattern: sublayer + Add + LayerNormalization.

    Creates Format 3 pattern for SkipLayerNormFusion:

          [Sub1]   [Sub2]
             \       /
              \     /
               \   /
                Add1
                 |
         LayerNormalization

    Based on skip_layer_norm_fusion.cc lines 146-153.

    Critical requirements from CheckFirstAdd():
    - Both Add inputs must be 3D tensors [batch, seq, hidden]
    - Both inputs must have same dimensions
    - Uses native LayerNormalization op (opset 17)
    """
    hidden_size = 64

    # MatMul weight for sublayer (creates dynamic skip input)
    rng = np.random.RandomState(hash(prefix) % (2**32))
    initializers.append(
        numpy_helper.from_array(
            rng.randn(hidden_size, hidden_size).astype(np.float32) * 0.1,
            f"{prefix}sublayer_weight",
        )
    )
    # LayerNorm parameters (1D along last axis)
    initializers.append(
        numpy_helper.from_array(np.ones(hidden_size, dtype=np.float32), f"{prefix}gamma")
    )
    initializers.append(
        numpy_helper.from_array(np.zeros(hidden_size, dtype=np.float32), f"{prefix}beta")
    )

    return [
        # Sublayer operation (simulates FFN/attention) - creates DYNAMIC skip input
        helper.make_node(
            "MatMul",
            [input_name, f"{prefix}sublayer_weight"],
            [f"{prefix}sublayer_out"],
            name=f"{prefix}sublayer_matmul",
        ),
        # Residual Add: BOTH inputs are dynamic 3D tensors (required for SkipLayerNormFusion)
        helper.make_node(
            "Add",
            [input_name, f"{prefix}sublayer_out"],  # input + sublayer_out (both dynamic 3D)
            [f"{prefix}skip_added"],
            name=f"{prefix}add_skip",
        ),
        # Native LayerNormalization op (opset 17) - REQUIRED for SkipLayerNormFusion
        helper.make_node(
            "LayerNormalization",
            [f"{prefix}skip_added", f"{prefix}gamma", f"{prefix}beta"],
            [output_name],
            name=f"{prefix}layernorm",
            axis=-1,
            epsilon=1e-5,
        ),
    ]


def bias_skip_layernorm_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    r"""Build BiasSkipLayerNorm pattern with dynamic skip tensor.

    Creates Format 1 pattern for SkipLayerNormFusion:

        [Sub1]  C    [Sub2]
            \  /     /
            Add2    /
               \   /
                Add1
                 |
         LayerNormalization

    Based on skip_layer_norm_fusion.cc lines 127-135.

    Critical requirements from CheckSecondAdd():
    - Add2 first input (Sub1) must be 3D tensor
    - Add2 second input (C/bias) must be 1D constant
    - Add1 inputs must be 3D with same dimensions
    - Uses native LayerNormalization op (opset 17)

    IMPORTANT: Skip (Sub2) must be a DYNAMIC tensor (graph input), not constant!
    This is the key fix - the original used a constant initializer for skip.
    """
    hidden_size = 64
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Bias for Add2 (1D constant) - this is the "C" in the pattern
    initializers.append(
        numpy_helper.from_array(rng.randn(hidden_size).astype(np.float32) * 0.1, f"{prefix}bias")
    )
    # LayerNorm parameters (1D along last axis)
    initializers.append(
        numpy_helper.from_array(np.ones(hidden_size, dtype=np.float32), f"{prefix}gamma")
    )
    initializers.append(
        numpy_helper.from_array(np.zeros(hidden_size, dtype=np.float32), f"{prefix}beta")
    )
    # MatMul weight to create dynamic skip tensor
    initializers.append(
        numpy_helper.from_array(
            rng.randn(hidden_size, hidden_size).astype(np.float32) * 0.1,
            f"{prefix}skip_weight",
        )
    )

    return [
        # Create dynamic skip tensor (Sub2) via MatMul - NOT a constant!
        helper.make_node(
            "MatMul",
            [input_name, f"{prefix}skip_weight"],
            [f"{prefix}skip_tensor"],
            name=f"{prefix}skip_matmul",
        ),
        # Add2: Add bias to input (Sub1 + C)
        # First input is 3D, second input is 1D constant
        helper.make_node(
            "Add",
            [input_name, f"{prefix}bias"],
            [f"{prefix}biased"],
            name=f"{prefix}add_bias",
        ),
        # Add1: Add skip connection (Add2_output + Sub2)
        # Both inputs are 3D tensors with same dimensions
        helper.make_node(
            "Add",
            [f"{prefix}biased", f"{prefix}skip_tensor"],
            [f"{prefix}skip_added"],
            name=f"{prefix}add_skip",
        ),
        # Native LayerNormalization op (opset 17)
        helper.make_node(
            "LayerNormalization",
            [f"{prefix}skip_added", f"{prefix}gamma", f"{prefix}beta"],
            [output_name],
            name=f"{prefix}layernorm",
            axis=-1,
            epsilon=1e-5,
        ),
    ]


def embed_layer_norm_builder() -> onnx.ModelProto:
    """Create ONNX model that matches ORT's EmbedLayerNormFusion pattern (Format 5).

    This builder creates the DistilBERT-style EmbedLayerNorm pattern that ORT's
    EmbedLayerNormFusion optimizer expects. Based on ORT's embed_layer_norm_gen.py
    GenerateModel5 function.

    Format 5 Key Characteristics:
    - Pre-computed position embeddings as constant initializer (simplest pattern)
    - 3 INT64 inputs: input_ids, segment_ids, input_mask
    - Word + Position + Segment embeddings -> LayerNorm -> Attention
    - 11 nodes total before optimization

    Pattern Structure:
        input_ids (INT64) --> Gather(word_embed) ----+
                                                      |
        pos_gather_out (constant, pre-computed) -----+--> Add
                                                      |      |
        segment_ids (INT64) --> Gather(seg_embed) ---+       |
                                                              |
                                                         LayerNorm
                                                              |
                                           com.microsoft:Attention
                                                              |
                                                         MatMul + Add
                                                              |
                                                         Add (skip)
                                                              |
                                                           output

    Based on onnxruntime/test/testdata/transform/fusion/embed_layer_norm_gen.py

    Returns:
        ONNX ModelProto that triggers EmbedLayerNormFusion optimization
    """

    # Dimensions matching ORT test pattern
    batch_size = 2
    hidden_size = 4
    attention_heads = 2
    sequence_length = 3

    # Build nodes exactly matching ORT's GenerateModel5
    nodes = [
        # 1. Word embedding lookup
        helper.make_node(
            "Gather",
            ["word_embed", "input_ids"],
            ["word_gather_out"],
            "word_gather",
            axis=0,
        ),
        # 2. Add word + pre-computed position embeddings
        helper.make_node(
            "Add",
            ["word_gather_out", "pos_gather_out"],
            ["word_add_pos_out"],
            "word_add_pos",
        ),
        # 3. Segment embedding lookup
        helper.make_node(
            "Gather",
            ["seg_embed", "segment_ids"],
            ["seg_gather_out"],
            "seg_gather",
            axis=0,
        ),
        # 4. Add segment embeddings
        helper.make_node(
            "Add",
            ["word_add_pos_out", "seg_gather_out"],
            ["add3_out"],
            "add3",
        ),
        # 5. LayerNormalization
        helper.make_node(
            "LayerNormalization",
            ["add3_out", "layer_norm_weight", "layer_norm_bias"],
            ["layernorm_out"],
            "layernorm",
            axis=-1,
            epsilon=9.999999747378752e-06,
        ),
        # 6. Cast mask to INT32 for ReduceSum
        helper.make_node(
            "Cast",
            ["input_mask"],
            ["mask_cast_out"],
            "mask_cast",
            to=6,  # INT32
        ),
        # 7. ReduceSum for mask index (opset 13 style)
        helper.make_node(
            "ReduceSum",
            ["mask_cast_out", "axes_1"],
            ["mask_index_out"],
            "mask_index",
            keepdims=0,
        ),
        # 8. com.microsoft:Attention node
        helper.make_node(
            "Attention",
            ["layernorm_out", "qkv_weights", "qkv_bias", "mask_index_out"],
            ["att_out"],
            "att",
            domain="com.microsoft",
            num_heads=attention_heads,
        ),
        # 9. Output projection MatMul
        helper.make_node(
            "MatMul",
            ["att_out", "matmul_weight"],
            ["matmul_out"],
            "matmul",
        ),
        # 10. Add bias
        helper.make_node(
            "Add",
            ["matmul_out", "add_bias"],
            ["add_out"],
            "add",
        ),
        # 11. Skip connection
        helper.make_node(
            "Add",
            ["add_out", "layernorm_out"],
            ["add2_out"],
            "add2",
        ),
    ]

    # QKV weights for Attention
    qkv_weights = [1.0] * hidden_size * (3 * hidden_size)

    # Initializers matching ORT's pattern
    initializers = [
        # Word embedding table [vocab_size=2, hidden_size=4]
        helper.make_tensor(
            "word_embed",
            TensorProto.FLOAT,
            [2, hidden_size],
            [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        ),
        # Pre-computed position embeddings [batch, seq, hidden]
        # This is the key difference in Format 5 - positions are pre-computed
        helper.make_tensor(
            "pos_gather_out",
            TensorProto.FLOAT,
            [batch_size, sequence_length, hidden_size],
            [
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 8.0, 7.0, 6.0,
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 8.0, 7.0, 6.0,
            ],
        ),
        # Segment embedding table [vocab_size=2, hidden_size=4]
        helper.make_tensor(
            "seg_embed",
            TensorProto.FLOAT,
            [2, hidden_size],
            [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        ),
        # LayerNorm weights
        helper.make_tensor(
            "layer_norm_weight",
            TensorProto.FLOAT,
            [hidden_size],
            [1.0, 2.0, 3.0, 4.0],
        ),
        # LayerNorm bias
        helper.make_tensor(
            "layer_norm_bias",
            TensorProto.FLOAT,
            [hidden_size],
            [0.1, 0.2, 0.3, 0.4],
        ),
        # QKV weights for Attention [hidden, 3*hidden]
        helper.make_tensor(
            "qkv_weights",
            TensorProto.FLOAT,
            [hidden_size, 3 * hidden_size],
            qkv_weights,
        ),
        # QKV bias [3*hidden]
        helper.make_tensor(
            "qkv_bias",
            TensorProto.FLOAT,
            [3 * hidden_size],
            [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
        ),
        # Output projection weight [hidden, hidden]
        helper.make_tensor(
            "matmul_weight",
            TensorProto.FLOAT,
            [hidden_size, hidden_size],
            [
                1.0, 2.0, 3.0, 4.0,
                1.0, 2.0, 3.0, 4.0,
                1.0, 2.0, 3.0, 4.0,
                1.0, 2.0, 3.0, 4.0,
            ],
        ),
        # Add bias [hidden]
        helper.make_tensor(
            "add_bias",
            TensorProto.FLOAT,
            [hidden_size],
            [0.1, 0.2, 0.3, 0.4],
        ),
        # Axes for ReduceSum (opset 13)
        helper.make_tensor(
            "axes_1",
            TensorProto.INT64,
            [1],
            [1],
        ),
    ]

    # Create graph with proper inputs/outputs
    graph = helper.make_graph(
        nodes,
        "EmbedLayerNorm_format5",
        [
            # 3 INT64 inputs
            helper.make_tensor_value_info(
                "input_ids", TensorProto.INT64, [batch_size, sequence_length]
            ),
            helper.make_tensor_value_info(
                "segment_ids", TensorProto.INT64, [batch_size, sequence_length]
            ),
            helper.make_tensor_value_info(
                "input_mask", TensorProto.INT64, [batch_size, sequence_length]
            ),
        ],
        [
            helper.make_tensor_value_info(
                "add2_out", TensorProto.FLOAT, [batch_size, sequence_length, hidden_size]
            ),
        ],
        initializers,
    )

    # Create model with com.microsoft domain for Attention op
    opset_imports = [
        helper.make_opsetid("", 13),
        helper.make_opsetid("com.microsoft", 1),
    ]

    model = helper.make_model(graph, opset_imports=opset_imports)
    model.ir_version = 8

    return model
