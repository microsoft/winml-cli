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
