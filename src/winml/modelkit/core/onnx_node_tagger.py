# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNX Node Tagger with Corrected Priority System.

This module implements the corrected tagging strategy:
1. NO HARDCODED LOGIC - all model/operation info extracted dynamically
2. Operation types only added if on the supportedlist
3. Operation-based fallback is optional
4. Root fallback always uses clean root module tag

CARDINAL RULES:
- MUST-001: NO HARDCODED LOGIC - works with any model
- MUST-002: TORCH.NN FILTERING - only supportedlist operations
- MUST-003: UNIVERSAL DESIGN - architecture-agnostic
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar


if TYPE_CHECKING:
    import onnx


class ONNXNodeTagger:
    """Universal ONNX node tagger using hierarchy data and priority system.

    NO HARDCODED LOGIC - extracts all model information dynamically.
    """

    # Supportedlist operation types that can be added to tags
    # These are fundamental ONNX operations that provide semantic value
    SUPPORTEDLIST_OPERATIONS: ClassVar[set[str]] = {
        "MatMul",
        "Gemm",  # Matrix operations
        "LayerNormalization",  # Normalization
        "Softmax",
        "Gelu",
        "Relu",  # Activations
        "Add",
        "Mul",  # Basic arithmetic
        "Gather",
        "Embedding",  # Lookup operations
    }

    def __init__(
        self, hierarchy_data: dict[str, dict], enable_operation_fallback: bool = False
    ) -> None:
        """Initialize ONNX node tagger.

        Args:
            hierarchy_data: Module hierarchy from TracingHierarchyBuilder
            enable_operation_fallback: Whether to use operation-based fallback (PRIORITY 3)
        """
        self.hierarchy_data = hierarchy_data
        self.enable_operation_fallback = enable_operation_fallback

        # Extract model root dynamically (NO HARDCODED)
        self.model_root_tag = self._extract_model_root_tag()

        # Pre-compute scope lookup for efficiency
        self.scope_to_tag: dict[str, str] = {
            module_name: module_info["traced_tag"]
            for module_name, module_info in hierarchy_data.items()
        }

    def _extract_model_root_tag(self) -> str:
        """Extract model root tag dynamically from hierarchy data.

        NO HARDCODED LOGIC - works with any model.
        """
        if not self.hierarchy_data:
            return "/UnknownModel"

        # Find the shortest tag (closest to root)
        all_tags = [
            info["traced_tag"] for info in self.hierarchy_data.values() if info.get("traced_tag")
        ]

        if not all_tags:
            return "/UnknownModel"

        # Extract root from shortest tag path
        shortest_tag = min(all_tags, key=len)
        # Root is the first component: "/BertModel/..." -> "/BertModel"
        root_parts = shortest_tag.strip("/").split("/")
        return f"/{root_parts[0]}" if root_parts else "/UnknownModel"

    def bucketize_nodes_by_scope(
        self, onnx_model: onnx.ModelProto
    ) -> dict[str, list[onnx.NodeProto]]:
        """Bucketize ONNX nodes by their scope names.

        Nodes without scope belong to root module.
        """
        scope_buckets = defaultdict(list)

        for node in onnx_model.graph.node:
            scope_name = self._extract_scope_from_node(node)
            scope_buckets[scope_name].append(node)

        return dict(scope_buckets)

    def _extract_scope_from_node(self, node: onnx.NodeProto) -> str:
        """Extract scope name from ONNX node.

        Examples:
            "/embeddings/word_embeddings/Gather" → "embeddings.word_embeddings"
            "/encoder/layer.0/attention/self/query/MatMul" →
                "encoder.layer.0.attention.self.query"
            "/encoder/stages.0/layers.0/layer/layer.0/convolution/Conv" →
                "encoder.stages.0.layers.0.layer.0.convolution"
            "/Softmax_123" → "__root__"
            "MatMul" → "__root__"
        """
        node_name = node.name or ""

        # Handle empty or non-scoped names
        if not node_name or not node_name.startswith("/"):
            return "__root__"

        # Parse: "/scope/path/OperationType" -> ["scope", "path", "OperationType"]
        parts = node_name.strip("/").split("/")

        # Single component means no scope
        if len(parts) <= 1:
            return "__root__"

        # Extract scope (everything except operation)
        scope_parts = parts[:-1]

        # Handle ResNet-style double layer pattern: "layer/layer.0" -> "layer.0"
        # This happens when ONNX export creates redundant path components
        cleaned_parts = []
        i = 0
        while i < len(scope_parts):
            if (
                i < len(scope_parts) - 1
                and scope_parts[i] == "layer"
                and scope_parts[i + 1].startswith("layer.")
            ):
                # Skip the redundant "layer" and use "layer.N" directly
                cleaned_parts.append(scope_parts[i + 1])
                i += 2
            else:
                cleaned_parts.append(scope_parts[i])
                i += 1

        return ".".join(cleaned_parts) if cleaned_parts else "__root__"

    def tag_all_nodes(self, onnx_model: onnx.ModelProto) -> dict[str, str]:
        """Tag all ONNX nodes using the 4-priority system.

        Returns:
            Dictionary mapping node names to hierarchy tags (NO EMPTY TAGS)
        """
        # Step 1: Bucketize nodes by scope
        scope_buckets = self.bucketize_nodes_by_scope(onnx_model)

        # Step 2: Tag each bucket using priority system
        tagged_nodes = {}

        for scope_name, nodes in scope_buckets.items():
            if scope_name == "__root__":
                # Root nodes always get model root tag
                tag = self.model_root_tag
            else:
                # Apply priority system for scoped nodes
                tag = self._find_tag_for_scope(scope_name)

            # Assign tag to all nodes in bucket
            for node in nodes:
                node_name = node.name or f"{node.op_type}_{id(node)}"
                tagged_nodes[node_name] = tag

        # Verify no empty tags
        for node_name, tag in tagged_nodes.items():
            assert tag, f"Empty tag generated for node {node_name}"
            assert tag.strip(), f"Whitespace-only tag generated for node {node_name}"
            assert tag.startswith("/"), f"Invalid tag format: {tag}"

        return tagged_nodes

    def _find_tag_for_scope(self, scope_name: str) -> str:
        """Find best tag for scope using 4-priority system.

        GUARANTEED to return non-empty tag.
        """
        # PRIORITY 1: Direct scope matching (highest accuracy)
        if scope_name in self.scope_to_tag:
            return self.scope_to_tag[scope_name]

        # PRIORITY 2: Execution context matching (parent scope)
        parent_tag = self._find_parent_scope_tag(scope_name)
        if parent_tag:
            return parent_tag

        # PRIORITY 3: Operation-based fallback (OPTIONAL)
        if self.enable_operation_fallback:
            operation_tag = self._find_operation_based_tag(scope_name)
            if operation_tag:
                return operation_tag

        # PRIORITY 4: Root fallback (NEVER EMPTY)
        return self.model_root_tag

    def _find_parent_scope_tag(self, scope_name: str) -> str | None:
        """Find parent scope tag by walking up the hierarchy.

        Example:
            scope_name = "encoder.layer.0.attention.self.unknown"

            Check: "encoder.layer.0.attention.self" → found!

            Return: corresponding tag
        """
        scope_parts = scope_name.split(".")

        # Walk up the hierarchy (remove one component at a time)
        for i in range(len(scope_parts) - 1, 0, -1):
            parent_scope = ".".join(scope_parts[:i])
            if parent_scope in self.scope_to_tag:
                return self.scope_to_tag[parent_scope]

        return None

    def _find_operation_based_tag(self, scope_name: str) -> str | None:
        """Find tag using operation-based similarity.

        Only used if enable_operation_fallback=True.
        Prefers shorter matches (more general) when scores are equal.
        """
        # Find most similar scope by prefix matching
        best_match = None
        best_score = 0

        scope_parts = scope_name.split(".")

        for hierarchy_scope in self.scope_to_tag:
            hierarchy_parts = hierarchy_scope.split(".")

            # Calculate common prefix length
            common_length = 0
            for a, b in zip(scope_parts, hierarchy_parts, strict=False):
                if a == b:
                    common_length += 1
                else:
                    break

            # Prefer matches with higher score, or shorter paths when score is equal
            if common_length > best_score or (
                common_length == best_score
                and (best_match is None or len(hierarchy_scope) < len(best_match))
            ):
                best_score = common_length
                best_match = hierarchy_scope

        # Return if we found a reasonable match (at least 1 common component)
        if best_match and best_score > 0:
            return self.scope_to_tag[best_match]

        return None

    def get_tagging_statistics(self, onnx_model: onnx.ModelProto) -> dict[str, int]:
        """Get statistics about the tagging process."""
        scope_buckets = self.bucketize_nodes_by_scope(onnx_model)

        stats = {
            "total_nodes": len(onnx_model.graph.node),
            "root_nodes": len(scope_buckets.get("__root__", [])),
            "scoped_nodes": sum(
                len(nodes) for scope, nodes in scope_buckets.items() if scope != "__root__"
            ),
            "unique_scopes": len([s for s in scope_buckets if s != "__root__"]),
            "direct_matches": 0,
            "parent_matches": 0,
            "operation_matches": 0,
            "root_fallbacks": 0,
        }

        # Count match types
        for scope_name, nodes in scope_buckets.items():
            if scope_name == "__root__":
                stats["root_fallbacks"] += len(nodes)
            elif scope_name in self.scope_to_tag:
                stats["direct_matches"] += len(nodes)
            elif self._find_parent_scope_tag(scope_name):
                stats["parent_matches"] += len(nodes)
            elif self.enable_operation_fallback and self._find_operation_based_tag(scope_name):
                stats["operation_matches"] += len(nodes)
            else:
                stats["root_fallbacks"] += len(nodes)

        return stats


def create_node_tagger_from_hierarchy(
    hierarchy_data: dict[str, dict], enable_operation_fallback: bool = False
) -> ONNXNodeTagger:
    """Factory function to create ONNX node tagger from hierarchy data.

    Args:
        hierarchy_data: Output from TracingHierarchyBuilder
        enable_operation_fallback: Enable PRIORITY 3 operation-based fallback

    Returns:
        Configured ONNXNodeTagger instance
    """
    return ONNXNodeTagger(hierarchy_data, enable_operation_fallback)
