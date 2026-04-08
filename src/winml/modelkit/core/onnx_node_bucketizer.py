# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNX Node Bucketization by Scope Name.

This demonstrates how to group ONNX nodes by their scope names and
handle nodes with no scope (assign to root module).
"""

from collections import defaultdict

import onnx


def bucketize_onnx_nodes_by_scope(
    onnx_model: onnx.ModelProto,
) -> dict[str, list[onnx.NodeProto]]:
    """Bucketize ONNX nodes by their scope names.

    Args:
        onnx_model: ONNX model to analyze

    Returns:
        Dictionary mapping scope names to lists of nodes
    """
    scope_buckets = defaultdict(list)

    for node in onnx_model.graph.node:
        scope_name = extract_scope_from_node(node)
        scope_buckets[scope_name].append(node)

    return dict(scope_buckets)


def extract_scope_from_node(node: onnx.NodeProto) -> str:
    """Extract scope name from ONNX node.

    Examples:
        "/embeddings/word_embeddings/Gather" → "embeddings.word_embeddings"
        "/encoder/layer.0/attention/self/query/MatMul" → "encoder.layer.0.attention.self.query"
        "/Softmax_123" → "__root__" (no scope)
        "MatMul" → "__root__" (no scope)

    Returns:
        Scope name as dotted path, or "__root__" for nodes without scope
    """
    node_name = node.name or ""

    # Handle empty node names
    if not node_name:
        return "__root__"

    # Handle root-level operations (no leading slash or single component)
    if not node_name.startswith("/"):
        return "__root__"

    # Parse structured node name: "/scope/path/OperationType"
    parts = node_name.strip("/").split("/")

    # Single component means no scope (e.g., "/Gather_3")
    if len(parts) <= 1:
        return "__root__"

    # Extract scope path (everything except the last operation part)
    scope_parts = parts[:-1]  # Remove operation name
    scope_name = ".".join(scope_parts)  # Convert to dotted notation

    return scope_name if scope_name else "__root__"


def demonstrate_bucketization():
    """Demonstrate the bucketization process with examples."""
    # Example ONNX node names and their expected scopes
    example_nodes = [
        ("/embeddings/word_embeddings/Gather", "embeddings.word_embeddings"),
        ("/embeddings/LayerNorm/Add", "embeddings"),  # LayerNorm under embeddings
        (
            "/encoder/layer.0/attention/self/query/MatMul",
            "encoder.layer.0.attention.self.query",
        ),
        ("/encoder/layer.0/attention/self/MatMul", "encoder.layer.0.attention.self"),
        (
            "/encoder/layer.0/attention/output/dense/Gemm",
            "encoder.layer.0.attention.output.dense",
        ),
        ("/pooler/dense/Tanh", "pooler.dense"),
        ("/Softmax_123", "__root__"),  # No scope
        ("MatMul_456", "__root__"),  # No scope
        ("/Constant_789", "__root__"),  # Root-level constant
    ]

    print("🗂️ ONNX Node Scope Bucketization Examples:")
    print("=" * 60)

    scope_buckets = defaultdict(list)

    for node_name, expected_scope in example_nodes:
        # Create mock node
        mock_node = type(
            "MockNode",
            (),
            {"name": node_name, "op_type": node_name.split("/")[-1].split("_")[0]},
        )()

        # Extract scope
        actual_scope = extract_scope_from_node(mock_node)

        # Add to bucket
        scope_buckets[actual_scope].append(mock_node)

        # Verify expectation
        status = "✅" if actual_scope == expected_scope else "❌"
        print(f"{status} {node_name:50} -> {actual_scope}")

    print("\n📊 Scope Buckets:")
    print("-" * 30)
    for scope_name, nodes in scope_buckets.items():
        print(f"{scope_name:30} ({len(nodes)} nodes)")
        for node in nodes:
            print(f"  └─ {node.name}")

    return scope_buckets


if __name__ == "__main__":
    demonstrate_bucketization()
