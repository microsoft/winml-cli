# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Copyright (c) 2025 ModelKit Authors
# SPDX-License-Identifier: Apache-2.0
"""Pattern builder modules for ORT optimization tests.

Available builders:

Activation patterns (activation.py):
- bias_softmax_builder: Add + Softmax (CUDA-only optimizer)
- bias_dropout_builder: Add + Dropout (training-only optimizer)
- relu_clip_builder: ReLU + Clip -> Relu6

Attention patterns (attention.py):
- attention_builder: BERT-style self-attention (NOTE: AttentionFusion NOT handled by GraphPipe - TBD)
- multi_head_attention_builder: Multi-head attention for FusionPipe
- rotary_embeddings_builder: RoPE pattern for RotaryEmbedding

Conv patterns (conv.py):
- conv_bn_builder: Conv + BatchNorm fusion
- conv_add_relu_builder: Conv + Add + ReLU fusion
- conv_activation_builder: Conv + Activation fusion
- conv_mul_builder: Conv + Mul fusion
- conv_add_activation_builder: Conv + Add + Activation fusion
- conv_add_fusion_builder: Conv(no bias) + Add(1D bias) fusion
- nchwc_transformer_builder: NCHWc layout transformation
- nhwc_transformer_builder: NHWC layout transformation
- pad_conv_builder: Pad + Conv fusion

Core patterns (core.py):
- identity_relu_builder: Identity -> Relu pattern

Elimination patterns (elimination.py):
- slice_elimination_builder: Identity slice (starts=0, ends=INT64_MAX)
- unsqueeze_elimination_builder: Unsqueeze on constant initializer
- reshape_elimination_builder: Contiguous reshape fusion
- expand_elimination_builder: Expand to same shape elimination
- concat_slice_elimination_builder: Concat + exact slice extraction

Gelu patterns (gelu.py):
- gelu_fusion_builder: Decomposed GELU for GeluFusionL2
- bias_gelu_builder: Add + GELU for BiasGeluFusion
- fast_gelu_builder: Tanh approximation for FastGeluFusion
- quick_gelu_builder: Sigmoid approximation for QuickGeluFusion
- gelu_approximation_builder: Standard GELU for GeluApproximation

LayerNorm patterns (layernorm.py):
- decomposed_layernorm_builder: Full decomposed LayerNorm for LayerNormFusionL2
- simplified_layernorm_builder: Variance-only SimplifiedLayerNormFusion
- skip_layernorm_builder: Add + LayerNorm for SkipLayerNormFusion (Format 3)
- bias_skip_layernorm_builder: Bias + Skip + LayerNorm (Format 1/2)

Misc patterns (misc.py):
- gather_slice_to_split_builder: Multiple Gather -> Split fusion
- gather_to_slice_builder: Gather -> Slice conversion
- not_where_builder: Not + Where fusion
- qdq_pairs_builder: Double QDQ pairs removal
- pad_fusion_builder: Pad + Conv fusion
- softmax_builder: Simple Softmax pattern
- transpose_chain_builder: Double transpose cancellation
- reduce_softmax_builder: ReduceSum + Softmax pattern
- noop_elimination_builder: Add(x,0), Mul(x,1), Sub(x,0) elimination
- gather_split_builder: Multiple Gather -> Split + Squeeze
- concat_slice_builder: Split -> Concat -> Slice elimination
"""

from .activation import (
    bias_dropout_builder,
    bias_softmax_builder,
    relu_clip_builder,
)
from .attention import (
    attention_builder,
    multi_head_attention_builder,
    rotary_embeddings_builder,
)
from .conv import (
    conv_activation_builder,
    conv_add_activation_builder,
    conv_add_fusion_builder,
    conv_add_relu_builder,
    conv_bn_builder,
    conv_mul_builder,
    nchwc_transformer_builder,
    nhwc_transformer_builder,
    pad_conv_builder,
)
from .core import (
    identity_relu_builder,
)
from .elimination import (
    concat_slice_elimination_builder,
    expand_elimination_builder,
    reshape_elimination_builder,
    slice_elimination_builder,
    unsqueeze_elimination_builder,
)
from .gelu import (
    bias_gelu_builder,
    fast_gelu_builder,
    gelu_approximation_builder,
    gelu_fusion_builder,
    quick_gelu_builder,
)
from .layernorm import (
    bias_skip_layernorm_builder,
    decomposed_layernorm_builder,
    simplified_layernorm_builder,
    skip_layernorm_builder,
)
from .misc import (
    concat_slice_builder,
    gather_slice_to_split_builder,
    gather_split_builder,
    gather_to_slice_builder,
    noop_elimination_builder,
    not_where_builder,
    pad_fusion_builder,
    qdq_pairs_builder,
    reduce_softmax_builder,
    softmax_builder,
    transpose_chain_builder,
)


__all__ = [
    # Attention patterns
    "attention_builder",
    # Activation patterns
    "bias_dropout_builder",
    # Gelu patterns
    "bias_gelu_builder",
    # LayerNorm patterns
    "bias_skip_layernorm_builder",
    "bias_softmax_builder",
    # Misc patterns
    "concat_slice_builder",
    # Elimination patterns
    "concat_slice_elimination_builder",
    # Conv patterns
    "conv_activation_builder",
    "conv_add_activation_builder",
    "conv_add_fusion_builder",
    "conv_add_relu_builder",
    "conv_bn_builder",
    "conv_mul_builder",
    "decomposed_layernorm_builder",
    "expand_elimination_builder",
    "fast_gelu_builder",
    "gather_slice_to_split_builder",
    "gather_split_builder",
    "gather_to_slice_builder",
    "gelu_approximation_builder",
    "gelu_fusion_builder",
    # Core patterns
    "identity_relu_builder",
    "multi_head_attention_builder",
    "nchwc_transformer_builder",
    "nhwc_transformer_builder",
    "noop_elimination_builder",
    "not_where_builder",
    "pad_conv_builder",
    "pad_fusion_builder",
    "qdq_pairs_builder",
    "quick_gelu_builder",
    "reduce_softmax_builder",
    "relu_clip_builder",
    "reshape_elimination_builder",
    "rotary_embeddings_builder",
    "simplified_layernorm_builder",
    "skip_layernorm_builder",
    "slice_elimination_builder",
    "softmax_builder",
    "transpose_chain_builder",
    "unsqueeze_elimination_builder",
]
