"""Comprehensive tests for modelkit.core.node_metadata module.

This module tests the node-level metadata system for ONNX models.
Metadata is stored as custom attributes on ONNX nodes with the 'winml.' prefix.

Test Categories:
- Unit Tests (UT-001 to UT-017): Individual method testing
- Smoke Tests (SM-001 to SM-005): Quick validation of basic functionality
- Sanity Tests (SN-001 to SN-004): End-to-end validation with real ONNX models
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from winml.modelkit.core.node_metadata import (
    NodeMetadata,
    add_metadata_to_node,
    get_metadata_from_node,
    get_optimization_summary,
    mark_fused_node,
    query_fused_nodes,
    query_nodes_by_origin,
    set_origin_for_graph,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_node() -> onnx.NodeProto:
    """Create a simple ONNX node for testing."""
    return helper.make_node(
        "MatMul",
        inputs=["A", "B"],
        outputs=["C"],
        name="MatMul_0",
    )


@pytest.fixture
def simple_graph() -> onnx.GraphProto:
    """Create a simple ONNX graph with multiple nodes."""
    nodes = [
        helper.make_node("MatMul", ["A", "B"], ["C"], name="MatMul_0"),
        helper.make_node("Add", ["C", "D"], ["E"], name="Add_0"),
        helper.make_node("Relu", ["E"], ["F"], name="Relu_0"),
    ]
    return helper.make_graph(
        nodes,
        "test_graph",
        inputs=[
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 4]),
            helper.make_tensor_value_info("D", TensorProto.FLOAT, [1, 4]),
        ],
        outputs=[
            helper.make_tensor_value_info("F", TensorProto.FLOAT, [1, 4]),
        ],
    )


@pytest.fixture
def simple_model(simple_graph: onnx.GraphProto) -> onnx.ModelProto:
    """Create a simple ONNX model."""
    return helper.make_model(simple_graph, opset_imports=[helper.make_opsetid("", 17)])


# ============================================================================
# Unit Tests - NodeMetadata Class (UT-001 to UT-010)
# ============================================================================


def test_node_metadata_init_required_fields() -> None:
    """UT-001: Create NodeMetadata with only required fields (name, origin)."""
    metadata = NodeMetadata(name="MatMul_0", origin="export")

    assert metadata.name == "MatMul_0"
    assert metadata.origin == "export"
    assert metadata.hierarchy_tag is None
    assert metadata.hierarchy_depth is None
    assert metadata.semantic_type is None
    assert metadata.semantic_layer_id is None
    assert metadata.optim_applied == []
    assert metadata.optim_sources == []


def test_node_metadata_init_all_fields() -> None:
    """UT-002: Create NodeMetadata with all optional fields populated."""
    metadata = NodeMetadata(
        name="FusedAttention_0",
        origin="optimize",
        hierarchy_tag="/BertModel/BertEncoder/BertLayer.0/BertAttention",
        hierarchy_depth=5,
        semantic_type="attention/self",
        semantic_layer_id="0",
        optim_applied=["attention-fusion", "gelu-fusion"],
        optim_sources=["MatMul_Q", "MatMul_K", "MatMul_V"],
    )

    assert metadata.name == "FusedAttention_0"
    assert metadata.origin == "optimize"
    assert metadata.hierarchy_tag == "/BertModel/BertEncoder/BertLayer.0/BertAttention"
    assert metadata.hierarchy_depth == 5
    assert metadata.semantic_type == "attention/self"
    assert metadata.semantic_layer_id == "0"
    assert metadata.optim_applied == ["attention-fusion", "gelu-fusion"]
    assert metadata.optim_sources == ["MatMul_Q", "MatMul_K", "MatMul_V"]


def test_node_metadata_default_values() -> None:
    """UT-003: Verify default values for optional fields (None, empty lists)."""
    metadata = NodeMetadata(name="Add_0", origin="export")

    # Optional hierarchy fields should be None
    assert metadata.hierarchy_tag is None
    assert metadata.hierarchy_depth is None

    # Optional semantic fields should be None
    assert metadata.semantic_type is None
    assert metadata.semantic_layer_id is None

    # Optional optimization lists should be empty
    assert isinstance(metadata.optim_applied, list)
    assert len(metadata.optim_applied) == 0
    assert isinstance(metadata.optim_sources, list)
    assert len(metadata.optim_sources) == 0


def test_to_attributes_required_only() -> None:
    """UT-004: Convert metadata with only required fields to ONNX attributes."""
    metadata = NodeMetadata(name="MatMul_0", origin="export")
    attrs = metadata.to_attributes()

    # Should only have 2 attributes for required fields
    assert len(attrs) == 2

    # Extract attribute dict for easier testing
    attr_dict = {attr.name: attr.s.decode() for attr in attrs}

    assert attr_dict["winml.node.name"] == "MatMul_0"
    assert attr_dict["winml.node.origin"] == "export"


def test_to_attributes_all_fields() -> None:
    """UT-005: Convert metadata with all fields to ONNX attributes."""
    metadata = NodeMetadata(
        name="FusedAttention_0",
        origin="optimize",
        hierarchy_tag="/BertModel/BertAttention",
        hierarchy_depth=5,
        semantic_type="attention/query",
        semantic_layer_id="0",
        optim_applied=["attention-fusion", "gelu-fusion"],
        optim_sources=["MatMul_Q", "MatMul_K"],
    )
    attrs = metadata.to_attributes()

    # Should have 8 attributes (all fields populated)
    assert len(attrs) == 8

    # Extract attribute dict
    attr_dict = {attr.name: attr.s.decode() for attr in attrs}

    assert attr_dict["winml.node.name"] == "FusedAttention_0"
    assert attr_dict["winml.node.origin"] == "optimize"
    assert attr_dict["winml.hierarchy.tag"] == "/BertModel/BertAttention"
    assert attr_dict["winml.hierarchy.depth"] == "5"
    assert attr_dict["winml.semantic.type"] == "attention/query"
    assert attr_dict["winml.semantic.layer_id"] == "0"
    assert attr_dict["winml.optim.applied"] == "attention-fusion,gelu-fusion"
    assert attr_dict["winml.optim.sources"] == "MatMul_Q,MatMul_K"


def test_to_attributes_list_serialization() -> None:
    """UT-006: Verify optim_applied and optim_sources serialize as comma-separated."""
    metadata = NodeMetadata(
        name="FusedNode",
        origin="optimize",
        optim_applied=["fusion1", "fusion2", "fusion3"],
        optim_sources=["NodeA", "NodeB", "NodeC", "NodeD"],
    )
    attrs = metadata.to_attributes()

    attr_dict = {attr.name: attr.s.decode() for attr in attrs}

    # Lists should be comma-separated
    assert attr_dict["winml.optim.applied"] == "fusion1,fusion2,fusion3"
    assert attr_dict["winml.optim.sources"] == "NodeA,NodeB,NodeC,NodeD"


def test_from_node_with_metadata(simple_node: onnx.NodeProto) -> None:
    """UT-007: Extract metadata from node that has winml.* attributes."""
    # Add metadata attributes to the node
    simple_node.attribute.extend(
        [
            helper.make_attribute("winml.node.name", "MatMul_0"),
            helper.make_attribute("winml.node.origin", "export"),
            helper.make_attribute("winml.hierarchy.tag", "/Model/Layer"),
            helper.make_attribute("winml.hierarchy.depth", "3"),
            helper.make_attribute("winml.semantic.type", "attention/query"),
            helper.make_attribute("winml.semantic.layer_id", "2"),
            helper.make_attribute("winml.optim.applied", "fusion1,fusion2"),
            helper.make_attribute("winml.optim.sources", "NodeA,NodeB"),
        ]
    )

    metadata = NodeMetadata.from_node(simple_node)

    assert metadata is not None
    assert metadata.name == "MatMul_0"
    assert metadata.origin == "export"
    assert metadata.hierarchy_tag == "/Model/Layer"
    assert metadata.hierarchy_depth == 3
    assert metadata.semantic_type == "attention/query"
    assert metadata.semantic_layer_id == "2"
    assert metadata.optim_applied == ["fusion1", "fusion2"]
    assert metadata.optim_sources == ["NodeA", "NodeB"]


def test_from_node_without_metadata(simple_node: onnx.NodeProto) -> None:
    """UT-008: Return None for node without winml.* attributes."""
    metadata = NodeMetadata.from_node(simple_node)
    assert metadata is None


def test_from_node_partial_metadata(simple_node: onnx.NodeProto) -> None:
    """UT-009: Extract metadata when only some winml.* attributes present."""
    # Add only some metadata attributes
    simple_node.attribute.extend(
        [
            helper.make_attribute("winml.node.name", "MatMul_0"),
            helper.make_attribute("winml.node.origin", "optimize"),
            helper.make_attribute("winml.semantic.type", "feedforward"),
        ]
    )

    metadata = NodeMetadata.from_node(simple_node)

    assert metadata is not None
    assert metadata.name == "MatMul_0"
    assert metadata.origin == "optimize"
    assert metadata.semantic_type == "feedforward"
    # Other fields should have defaults
    assert metadata.hierarchy_tag is None
    assert metadata.hierarchy_depth is None
    assert metadata.semantic_layer_id is None
    assert metadata.optim_applied == []
    assert metadata.optim_sources == []


def test_roundtrip_metadata(simple_node: onnx.NodeProto) -> None:
    """UT-010: to_attributes → add to node → from_node produces equivalent metadata."""
    original = NodeMetadata(
        name="MatMul_0",
        origin="optimize",
        hierarchy_tag="/Model/Layer",
        hierarchy_depth=4,
        semantic_type="attention/value",
        semantic_layer_id="1",
        optim_applied=["fusion-a", "fusion-b"],
        optim_sources=["Node1", "Node2", "Node3"],
    )

    # Convert to attributes and add to node
    simple_node.attribute.extend(original.to_attributes())

    # Extract back from node
    recovered = NodeMetadata.from_node(simple_node)

    assert recovered is not None
    assert recovered.name == original.name
    assert recovered.origin == original.origin
    assert recovered.hierarchy_tag == original.hierarchy_tag
    assert recovered.hierarchy_depth == original.hierarchy_depth
    assert recovered.semantic_type == original.semantic_type
    assert recovered.semantic_layer_id == original.semantic_layer_id
    assert recovered.optim_applied == original.optim_applied
    assert recovered.optim_sources == original.optim_sources


# ============================================================================
# Unit Tests - Utility Functions (UT-011 to UT-017)
# ============================================================================


def test_add_metadata_to_node(simple_node: onnx.NodeProto) -> None:
    """UT-011: Add metadata to node, verify attributes present."""
    metadata = NodeMetadata(
        name="MatMul_0",
        origin="export",
        semantic_type="attention",
    )

    add_metadata_to_node(simple_node, metadata)

    # Verify attributes were added
    attr_dict = {attr.name: attr.s.decode() for attr in simple_node.attribute}

    assert "winml.node.name" in attr_dict
    assert attr_dict["winml.node.name"] == "MatMul_0"
    assert attr_dict["winml.node.origin"] == "export"
    assert attr_dict["winml.semantic.type"] == "attention"


def test_add_metadata_overwrites_existing(simple_node: onnx.NodeProto) -> None:
    """UT-012: Adding metadata replaces existing winml.* attributes."""
    # Add initial metadata
    simple_node.attribute.extend(
        [
            helper.make_attribute("winml.node.name", "OldName"),
            helper.make_attribute("winml.node.origin", "old_origin"),
            helper.make_attribute("winml.semantic.type", "old_type"),
            helper.make_attribute("other.attribute", "should_remain"),
        ]
    )

    # Add new metadata
    new_metadata = NodeMetadata(
        name="NewName",
        origin="new_origin",
        hierarchy_tag="/New/Path",
    )
    add_metadata_to_node(simple_node, new_metadata)

    # Extract attributes
    attr_dict = {attr.name: attr.s.decode() for attr in simple_node.attribute}

    # New metadata should be present
    assert attr_dict["winml.node.name"] == "NewName"
    assert attr_dict["winml.node.origin"] == "new_origin"
    assert attr_dict["winml.hierarchy.tag"] == "/New/Path"

    # Old winml.semantic.type should be gone
    assert "winml.semantic.type" not in attr_dict

    # Non-winml attribute should remain
    assert attr_dict["other.attribute"] == "should_remain"


def test_get_metadata_from_node(simple_node: onnx.NodeProto) -> None:
    """UT-013: Retrieve metadata from node with attributes."""
    simple_node.attribute.extend(
        [
            helper.make_attribute("winml.node.name", "MatMul_0"),
            helper.make_attribute("winml.node.origin", "quantize"),
        ]
    )

    metadata = get_metadata_from_node(simple_node)

    assert metadata is not None
    assert metadata.name == "MatMul_0"
    assert metadata.origin == "quantize"


def test_set_origin_for_graph(simple_graph: onnx.GraphProto) -> None:
    """UT-014: Set origin for all nodes in a graph."""
    set_origin_for_graph(simple_graph, origin="export")

    # Verify all nodes have the origin set
    for node in simple_graph.node:
        metadata = get_metadata_from_node(node)
        assert metadata is not None
        assert metadata.origin == "export"
        assert metadata.name == node.name


def test_set_origin_no_overwrite(simple_graph: onnx.GraphProto) -> None:
    """UT-015: set_origin_for_graph with overwrite=False skips existing."""
    # Add metadata to first node
    first_node = simple_graph.node[0]
    existing_metadata = NodeMetadata(
        name=first_node.name,
        origin="existing_origin",
        hierarchy_tag="/Existing/Path",
    )
    add_metadata_to_node(first_node, existing_metadata)

    # Set origin for all nodes with overwrite=False
    set_origin_for_graph(simple_graph, origin="new_origin", overwrite=False)

    # First node should keep existing origin
    first_metadata = get_metadata_from_node(simple_graph.node[0])
    assert first_metadata is not None
    assert first_metadata.origin == "existing_origin"
    assert first_metadata.hierarchy_tag == "/Existing/Path"

    # Other nodes should have new origin
    for node in simple_graph.node[1:]:
        metadata = get_metadata_from_node(node)
        assert metadata is not None
        assert metadata.origin == "new_origin"


def test_mark_fused_node(simple_node: onnx.NodeProto) -> None:
    """UT-016: Mark node as fusion result with sources and optimization."""
    mark_fused_node(
        simple_node,
        source_nodes=["MatMul_1", "Add_2", "Softmax_3"],
        optimization="attention-fusion",
    )

    metadata = get_metadata_from_node(simple_node)

    assert metadata is not None
    assert metadata.origin == "optimize"
    assert metadata.optim_sources == ["MatMul_1", "Add_2", "Softmax_3"]
    assert "attention-fusion" in metadata.optim_applied


def test_mark_fused_node_accumulates_optimizations(simple_node: onnx.NodeProto) -> None:
    """UT-017: Multiple mark_fused_node calls accumulate optim_applied."""
    # First fusion
    mark_fused_node(
        simple_node,
        source_nodes=["Node1", "Node2"],
        optimization="fusion-pass-1",
    )

    # Second fusion (different optimization)
    mark_fused_node(
        simple_node,
        source_nodes=["Node3", "Node4"],
        optimization="fusion-pass-2",
    )

    metadata = get_metadata_from_node(simple_node)

    assert metadata is not None
    assert metadata.origin == "optimize"
    # Both optimizations should be present
    assert "fusion-pass-1" in metadata.optim_applied
    assert "fusion-pass-2" in metadata.optim_applied
    # Latest sources should be set
    assert metadata.optim_sources == ["Node3", "Node4"]


# ============================================================================
# Smoke Tests (SM-001 to SM-005)
# ============================================================================


def test_smoke_create_metadata() -> None:
    """SM-001: Create NodeMetadata, convert to attributes, verify count."""
    metadata = NodeMetadata(
        name="TestNode",
        origin="export",
        semantic_type="attention",
    )

    attrs = metadata.to_attributes()

    # Should have 3 attributes: name, origin, semantic_type
    assert len(attrs) == 3
    assert all(isinstance(attr, onnx.AttributeProto) for attr in attrs)


def test_smoke_add_to_onnx_node() -> None:
    """SM-002: Create ONNX node, add metadata, verify node is valid."""
    node = helper.make_node("Relu", ["X"], ["Y"], name="Relu_0")
    metadata = NodeMetadata(name="Relu_0", origin="optimize")

    add_metadata_to_node(node, metadata)

    # Verify node is still valid
    assert node.name == "Relu_0"
    assert node.op_type == "Relu"
    assert len(node.attribute) > 0

    # Verify metadata extraction works
    recovered = get_metadata_from_node(node)
    assert recovered is not None
    assert recovered.name == "Relu_0"


def test_smoke_query_by_origin(simple_graph: onnx.GraphProto) -> None:
    """SM-003: Create graph with multiple origins, query by origin."""
    # Set different origins
    add_metadata_to_node(simple_graph.node[0], NodeMetadata(name="MatMul_0", origin="export"))
    add_metadata_to_node(simple_graph.node[1], NodeMetadata(name="Add_0", origin="optimize"))
    add_metadata_to_node(simple_graph.node[2], NodeMetadata(name="Relu_0", origin="export"))

    # Create model for query
    model = helper.make_model(simple_graph, opset_imports=[helper.make_opsetid("", 17)])

    export_nodes = query_nodes_by_origin(model, "export")
    optimize_nodes = query_nodes_by_origin(model, "optimize")

    assert set(export_nodes) == {"MatMul_0", "Relu_0"}
    assert optimize_nodes == ["Add_0"]


def test_smoke_query_fused_nodes(simple_graph: onnx.GraphProto) -> None:
    """SM-004: Create graph with fused nodes, query fusions."""
    # Mark first node as fused
    mark_fused_node(
        simple_graph.node[0],
        source_nodes=["OrigA", "OrigB", "OrigC"],
        optimization="fusion-1",
    )

    # Mark second node as fused
    mark_fused_node(simple_graph.node[1], source_nodes=["OrigD", "OrigE"], optimization="fusion-2")

    # Create model for query
    model = helper.make_model(simple_graph, opset_imports=[helper.make_opsetid("", 17)])

    fused_nodes = query_fused_nodes(model)

    assert "MatMul_0" in fused_nodes
    assert fused_nodes["MatMul_0"] == ["OrigA", "OrigB", "OrigC"]
    assert "Add_0" in fused_nodes
    assert fused_nodes["Add_0"] == ["OrigD", "OrigE"]
    assert "Relu_0" not in fused_nodes  # Not marked as fused


def test_smoke_optimization_summary(simple_graph: onnx.GraphProto) -> None:
    """SM-005: Create graph with optimizations, get summary."""
    # Add optimizations to nodes
    mark_fused_node(simple_graph.node[0], ["A", "B"], optimization="attention-fusion")
    mark_fused_node(simple_graph.node[1], ["C", "D"], optimization="gelu-fusion")
    mark_fused_node(simple_graph.node[2], ["E", "F"], optimization="attention-fusion")

    # Create model for query
    model = helper.make_model(simple_graph, opset_imports=[helper.make_opsetid("", 17)])

    summary = get_optimization_summary(model)

    assert summary["attention-fusion"] == 2
    assert summary["gelu-fusion"] == 1


# ============================================================================
# Sanity Tests (SN-001 to SN-004)
# ============================================================================


def test_sanity_metadata_survives_save_load(
    simple_model: onnx.ModelProto,
) -> None:
    """SN-001: Add metadata → save ONNX → load ONNX → metadata intact."""
    # Add metadata to all nodes
    set_origin_for_graph(simple_model.graph, origin="export")

    # Add additional metadata to first node
    first_node = simple_model.graph.node[0]
    metadata = NodeMetadata(
        name=first_node.name,
        origin="export",
        hierarchy_tag="/Model/Layer1",
        semantic_type="attention/query",
        optim_applied=["opt1", "opt2"],
    )
    add_metadata_to_node(first_node, metadata)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        temp_path = Path(tmp.name)

    try:
        onnx.save(simple_model, str(temp_path))

        # Load model back
        loaded_model = onnx.load(str(temp_path))

        # Verify metadata survived
        loaded_first_node = loaded_model.graph.node[0]
        loaded_metadata = get_metadata_from_node(loaded_first_node)

        assert loaded_metadata is not None
        assert loaded_metadata.name == first_node.name
        assert loaded_metadata.origin == "export"
        assert loaded_metadata.hierarchy_tag == "/Model/Layer1"
        assert loaded_metadata.semantic_type == "attention/query"
        assert loaded_metadata.optim_applied == ["opt1", "opt2"]
    finally:
        temp_path.unlink(missing_ok=True)


def test_sanity_metadata_with_real_model(simple_model: onnx.ModelProto) -> None:
    """SN-002: Apply metadata to exported model, verify all nodes tagged."""
    # Simulate export process
    set_origin_for_graph(simple_model.graph, origin="export")

    # Add hierarchy metadata to nodes
    for i, node in enumerate(simple_model.graph.node):
        existing = get_metadata_from_node(node)
        assert existing is not None

        enriched = NodeMetadata(
            name=existing.name,
            origin=existing.origin,
            hierarchy_tag=f"/Model/Layer{i}",
            hierarchy_depth=i + 1,
        )
        add_metadata_to_node(node, enriched)

    # Verify all nodes have complete metadata
    for i, node in enumerate(simple_model.graph.node):
        metadata = get_metadata_from_node(node)
        assert metadata is not None
        assert metadata.origin == "export"
        assert metadata.hierarchy_tag == f"/Model/Layer{i}"
        assert metadata.hierarchy_depth == i + 1


def test_sanity_multiple_origins_in_graph(simple_graph: onnx.GraphProto) -> None:
    """SN-003: Graph with export, optimize, quantize origins coexist."""
    # Set different origins for different nodes
    add_metadata_to_node(
        simple_graph.node[0],
        NodeMetadata(name="MatMul_0", origin="export", hierarchy_tag="/Layer1"),
    )
    add_metadata_to_node(
        simple_graph.node[1],
        NodeMetadata(
            name="Add_0",
            origin="optimize",
            optim_applied=["fusion"],
            optim_sources=["OrigAdd"],
        ),
    )
    add_metadata_to_node(
        simple_graph.node[2],
        NodeMetadata(name="Relu_0", origin="quantize", semantic_type="activation"),
    )

    # Verify all metadata coexists
    metadata_0 = get_metadata_from_node(simple_graph.node[0])
    assert metadata_0 is not None
    assert metadata_0.origin == "export"
    assert metadata_0.hierarchy_tag == "/Layer1"

    metadata_1 = get_metadata_from_node(simple_graph.node[1])
    assert metadata_1 is not None
    assert metadata_1.origin == "optimize"
    assert "fusion" in metadata_1.optim_applied

    metadata_2 = get_metadata_from_node(simple_graph.node[2])
    assert metadata_2 is not None
    assert metadata_2.origin == "quantize"
    assert metadata_2.semantic_type == "activation"


def test_sanity_semantic_type_parsing() -> None:
    """SN-004: Parse 'attention/query' format correctly."""
    # Test various semantic type formats
    test_cases = [
        ("attention", "attention"),
        ("attention/query", "attention/query"),
        ("attention/key", "attention/key"),
        ("attention/value", "attention/value"),
        ("feedforward/up", "feedforward/up"),
        ("feedforward/down", "feedforward/down"),
        ("layernorm", "layernorm"),
        ("embedding", "embedding"),
    ]

    for input_type, expected_type in test_cases:
        metadata = NodeMetadata(name="TestNode", origin="export", semantic_type=input_type)

        attrs = metadata.to_attributes()
        node = helper.make_node("Identity", ["X"], ["Y"], name="TestNode")
        node.attribute.extend(attrs)

        recovered = NodeMetadata.from_node(node)
        assert recovered is not None
        assert recovered.semantic_type == expected_type, f"Failed for input: {input_type}"
