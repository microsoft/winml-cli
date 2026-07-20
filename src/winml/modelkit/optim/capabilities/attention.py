# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Attention mechanism fusion capabilities.

This module defines attention mechanism fusion optimizations. These optimizations
detect and fuse multi-head attention patterns into efficient fused operations,
improving performance for transformer-based architectures.

Attention mechanisms are fundamental building blocks in transformer architectures.
These fusions detect common attention computation patterns and replace them with
optimized fused operations.
"""

from __future__ import annotations

from ..registry import BoolCapability, CapabilityCategory


# Attention fusion - fuses general attention patterns
ATTENTION_FUSION = BoolCapability(
    name="attention-fusion",
    ort_name="AttentionFusion",
    description="Fuse attention computation patterns into optimized operations",
    category=CapabilityCategory.ATTENTION,
    default=False,
)

# Packed QKV fusion - packs the Q, K, V projections of self-attention into a
# single MatMul, producing a packed-QKV input to the fused Attention op.
# Primarily used by Stable Diffusion's UNet self-attention blocks. Requires the
# base attention-fusion to be enabled (it modifies the shape of the produced
# Attention/MultiHeadAttention node).
PACKED_QKV_FUSION = BoolCapability(
    name="packed-qkv-fusion",
    ort_name="PackedQKVFusion",  # FusionOptions attr: enable_packed_qkv
    description="Pack Q/K/V projections into a single MatMul for self-attention (SD UNet)",
    category=CapabilityCategory.ATTENTION,
    default=False,
    depends_on=("attention-fusion",),
)

# Packed KV fusion - packs the K, V projections of cross-attention into a
# single MatMul (Q comes from a different source — image latents — and is left
# as a separate MatMul). Primarily used by Stable Diffusion's UNet
# cross-attention blocks that consume text embeddings. Requires the base
# attention-fusion to be enabled.
PACKED_KV_FUSION = BoolCapability(
    name="packed-kv-fusion",
    ort_name="PackedKVFusion",  # FusionOptions attr: enable_packed_kv
    description="Pack K/V projections into a single MatMul for cross-attention (SD UNet)",
    category=CapabilityCategory.ATTENTION,
    default=False,
    depends_on=("attention-fusion",),
)

# NOTE: MultiHeadAttention was removed - this is an OUTPUT NODE TYPE, not an optimizer.
# AttentionFusion (above) creates MultiHeadAttention or Attention nodes as output.
# Verified against ort_optimizer_inventory.md - no "MultiHeadAttention" optimizer exists.

# NOTE: Group Query Attention fusion is NOT currently supported by ONNX Runtime.
# The attribute "enable_group_query_attention" does not exist in FusionOptions.
# This capability is commented out to prevent runtime errors.
# If ORT adds GQA support in the future, uncomment and verify the fusion_attr.
#
# GROUP_QUERY_ATTENTION_FUSION = BoolCapability(
#     name="group-query-attention-fusion",
#     ort_name="GroupQueryAttentionFusion",
#     description="Fuse Group Query Attention (GQA) patterns used in Llama and Mistral models",
#     category=CapabilityCategory.ATTENTION,
#     default=False,
#     depends_on=("attention-fusion",),
# )
