# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for universal ONNX metadata capture/restore.

Uses realistic ONNX model fixtures with hierarchy tags, winml.* attributes,
and model-level metadata to verify end-to-end metadata preservation through
destructive graph transformations.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from winml.modelkit.onnx.metadata import (
    MetadataSnapshot,
    NodeMetadataEntry,
    capture_metadata,
    restore_metadata,
)


# ---------------------------------------------------------------------------
# Fixtures: realistic tagged model (like a mini CNN after export)
# ---------------------------------------------------------------------------


def _tag_node(
    node: onnx.NodeProto,
    hierarchy_tag: str,
    *,
    origin: str = "export",
) -> None:
    """Add full winml metadata to a node (metadata_props + attributes).

    This mirrors what the HTP exporter does in production.
    """
    depth = len([p for p in hierarchy_tag.split("/") if p])

    # metadata_props (what the exporter embeds)
    node.metadata_props.add(key="winml.hierarchy.tag", value=hierarchy_tag)
    node.metadata_props.add(key="winml.hierarchy.depth", value=str(depth))

    # winml.* attributes (from node_metadata system)
    node.attribute.append(helper.make_attribute("winml.node.name", node.name))
    node.attribute.append(helper.make_attribute("winml.node.origin", origin))
    node.attribute.append(helper.make_attribute("winml.hierarchy.tag", hierarchy_tag))


@pytest.fixture()
def tagged_cnn_model() -> onnx.ModelProto:
    """A realistic mini-CNN with full winml metadata on every node.

    Topology (mimics a ResNet bottleneck block):
        X -> Conv1 -> Relu1 -> Conv2 -> Relu2 -> Conv3 -> Add(X_skip) -> Y

    Each node has:
    - metadata_props: winml.hierarchy.tag, winml.hierarchy.depth
    - attributes: winml.node.name, winml.node.origin, winml.hierarchy.tag

    The model also has graph-level metadata:
    - winml.io.inputs, winml.io.outputs
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64, 56, 56])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64, 56, 56])

    # Initializers (weights + biases for each conv)
    inits = []
    for i in range(1, 4):
        w = onnx.numpy_helper.from_array(
            np.random.randn(64, 64, 3, 3).astype(np.float32),
            name=f"conv{i}_weight",
        )
        inits.append(w)

    # Nodes with hierarchy tags
    conv1 = helper.make_node(
        "Conv",
        ["X", "conv1_weight"],
        ["conv1_out"],
        name="/model/block/conv1/Conv",
        pads=[1, 1, 1, 1],
    )
    _tag_node(conv1, "/Model/Block/Conv1")

    relu1 = helper.make_node(
        "Relu",
        ["conv1_out"],
        ["relu1_out"],
        name="/model/block/relu1/Relu",
    )
    _tag_node(relu1, "/Model/Block/Relu1")

    conv2 = helper.make_node(
        "Conv",
        ["relu1_out", "conv2_weight"],
        ["conv2_out"],
        name="/model/block/conv2/Conv",
        pads=[1, 1, 1, 1],
    )
    _tag_node(conv2, "/Model/Block/Conv2")

    relu2 = helper.make_node(
        "Relu",
        ["conv2_out"],
        ["relu2_out"],
        name="/model/block/relu2/Relu",
    )
    _tag_node(relu2, "/Model/Block/Relu2")

    conv3 = helper.make_node(
        "Conv",
        ["relu2_out", "conv3_weight"],
        ["conv3_out"],
        name="/model/block/conv3/Conv",
        pads=[1, 1, 1, 1],
    )
    _tag_node(conv3, "/Model/Block/Conv3")

    add = helper.make_node(
        "Add",
        ["conv3_out", "X"],
        ["Y"],
        name="/model/block/Add",
    )
    _tag_node(add, "/Model/Block/Add")

    graph = helper.make_graph(
        [conv1, relu1, conv2, relu2, conv3, add],
        "tagged_cnn",
        [x],
        [y],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    # Graph-level metadata
    model.metadata_props.add(
        key="winml.io.inputs",
        value='[{"name":"X","dtype":"float32","shape":[1,64,56,56],"value_range":[0,1]}]',
    )
    model.metadata_props.add(
        key="winml.io.outputs",
        value='[{"name":"Y","dtype":"float32","shape":[1,64,56,56]}]',
    )

    return model


def _make_plain_model() -> onnx.ModelProto:
    """Build model with no custom metadata."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["X"], ["Y"], name="relu")
    graph = helper.make_graph([node], "plain", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def _strip_all_metadata(model: onnx.ModelProto) -> None:
    """Strip all custom metadata to simulate ORT graph rebuild."""
    del model.metadata_props[:]
    for node in model.graph.node:
        del node.metadata_props[:]
        to_remove = [a for a in node.attribute if a.name.startswith("winml.")]
        for a in to_remove:
            node.attribute.remove(a)


# ---------------------------------------------------------------------------
# Tests: capture_metadata
# ---------------------------------------------------------------------------


class TestCaptureMetadata:
    def test_captures_model_props(self, tagged_cnn_model):
        snapshot = capture_metadata(tagged_cnn_model)
        keys = [k for k, _ in snapshot.model_props]
        assert "winml.io.inputs" in keys
        assert "winml.io.outputs" in keys

    def test_captures_node_metadata_props(self, tagged_cnn_model):
        snapshot = capture_metadata(tagged_cnn_model)
        conv1 = snapshot.nodes["/model/block/conv1/Conv"]
        assert ("winml.hierarchy.tag", "/Model/Block/Conv1") in conv1.props
        assert ("winml.hierarchy.depth", "3") in conv1.props

    def test_captures_winml_attributes(self, tagged_cnn_model):
        snapshot = capture_metadata(tagged_cnn_model)
        conv1 = snapshot.nodes["/model/block/conv1/Conv"]
        attr_names = [name for name, _, _ in conv1.attrs]
        assert "winml.node.origin" in attr_names
        assert "winml.node.name" in attr_names
        assert "winml.hierarchy.tag" in attr_names

    def test_captures_all_6_nodes(self, tagged_cnn_model):
        snapshot = capture_metadata(tagged_cnn_model)
        assert snapshot.node_count == 6

    def test_plain_model_returns_empty_snapshot(self):
        model = _make_plain_model()
        snapshot = capture_metadata(model)
        assert snapshot.node_count == 0
        assert snapshot.model_prop_count == 0


# ---------------------------------------------------------------------------
# Tests: restore_metadata
# ---------------------------------------------------------------------------


class TestRestoreMetadata:
    def test_restores_model_props(self, tagged_cnn_model):
        snapshot = capture_metadata(tagged_cnn_model)
        _strip_all_metadata(tagged_cnn_model)

        result = restore_metadata(tagged_cnn_model, snapshot)
        keys = {p.key for p in tagged_cnn_model.metadata_props}
        assert "winml.io.inputs" in keys
        assert result.model_props_restored == 2

    def test_restores_node_props(self, tagged_cnn_model):
        snapshot = capture_metadata(tagged_cnn_model)
        _strip_all_metadata(tagged_cnn_model)

        result = restore_metadata(tagged_cnn_model, snapshot)
        conv1 = next(n for n in tagged_cnn_model.graph.node if n.name == "/model/block/conv1/Conv")
        props = {p.key: p.value for p in conv1.metadata_props}
        assert props["winml.hierarchy.tag"] == "/Model/Block/Conv1"
        assert result.nodes_restored == 6

    def test_restores_winml_attributes(self, tagged_cnn_model):
        snapshot = capture_metadata(tagged_cnn_model)
        _strip_all_metadata(tagged_cnn_model)

        restore_metadata(tagged_cnn_model, snapshot)
        add_node = next(n for n in tagged_cnn_model.graph.node if n.name == "/model/block/Add")
        attr_map = {a.name: a.s.decode() for a in add_node.attribute if a.name.startswith("winml.")}
        assert attr_map["winml.node.origin"] == "export"
        assert attr_map["winml.hierarchy.tag"] == "/Model/Block/Add"

    def test_does_not_duplicate(self, tagged_cnn_model):
        """Restore on already-tagged model doesn't duplicate."""
        snapshot = capture_metadata(tagged_cnn_model)
        restore_metadata(tagged_cnn_model, snapshot)

        conv1 = next(n for n in tagged_cnn_model.graph.node if n.name == "/model/block/conv1/Conv")
        tag_count = sum(1 for p in conv1.metadata_props if p.key == "winml.hierarchy.tag")
        assert tag_count == 1

    def test_unmatched_nodes_skipped(self):
        model = _make_plain_model()
        snapshot = MetadataSnapshot(
            nodes={"/nonexistent": NodeMetadataEntry(props=[("k", "v")])},
        )
        result = restore_metadata(model, snapshot)
        assert result.nodes_restored == 0

    def test_empty_snapshot(self, tagged_cnn_model):
        result = restore_metadata(tagged_cnn_model, MetadataSnapshot())
        assert result.nodes_restored == 0


# ---------------------------------------------------------------------------
# E2E: capture → destructive op → restore
# ---------------------------------------------------------------------------


class TestMetadataPreservationE2E:
    """End-to-end tests simulating real pipeline scenarios."""

    def test_survives_full_strip_and_restore(self, tagged_cnn_model):
        """capture → strip everything → restore → verify all metadata intact."""
        snapshot = capture_metadata(tagged_cnn_model)

        # Verify everything is present before strip
        assert snapshot.node_count == 6
        assert snapshot.model_prop_count == 2

        _strip_all_metadata(tagged_cnn_model)

        # Verify stripped
        assert len(tagged_cnn_model.metadata_props) == 0
        for n in tagged_cnn_model.graph.node:
            assert len(n.metadata_props) == 0
            assert not any(a.name.startswith("winml.") for a in n.attribute)

        # Restore
        result = restore_metadata(tagged_cnn_model, snapshot)
        assert result.nodes_restored == 6
        assert result.model_props_restored == 2

        # Verify every node got its metadata back
        for node in tagged_cnn_model.graph.node:
            props = {p.key for p in node.metadata_props}
            assert "winml.hierarchy.tag" in props, f"{node.name} missing tag"
            assert "winml.hierarchy.depth" in props, f"{node.name} missing depth"
            attrs = {a.name for a in node.attribute if a.name.startswith("winml.")}
            assert "winml.node.origin" in attrs, f"{node.name} missing origin attr"

    def test_survives_shape_inference(self, tagged_cnn_model):
        """capture → onnx shape inference (creates new model) → restore."""
        snapshot = capture_metadata(tagged_cnn_model)

        # Shape inference creates a new ModelProto (may strip metadata)
        new_model = onnx.shape_inference.infer_shapes(
            tagged_cnn_model,
            strict_mode=False,
        )

        restore_metadata(new_model, snapshot)

        # Verify tags restored on new model
        for node in new_model.graph.node:
            props = {p.key for p in node.metadata_props}
            assert "winml.hierarchy.tag" in props, f"{node.name} missing tag"

    def test_survives_node_removal(self, tagged_cnn_model):
        """Simulate ORT removing some nodes (like Relu absorption into QLinearConv).

        Capture on 6-node model, strip 2 Relu nodes, restore → 4 nodes get tags.
        """
        snapshot = capture_metadata(tagged_cnn_model)

        # Remove Relu nodes (simulates what ORT quantize does: Conv+Relu → QLinearConv)
        graph = tagged_cnn_model.graph
        kept = [n for n in graph.node if n.op_type != "Relu"]
        del graph.node[:]
        graph.node.extend(kept)

        # Strip metadata from remaining nodes
        _strip_all_metadata(tagged_cnn_model)

        result = restore_metadata(tagged_cnn_model, snapshot)

        # 4 non-Relu nodes should be restored (Conv1, Conv2, Conv3, Add)
        assert result.nodes_restored == 4
        assert result.nodes_total == 4

    def test_new_nodes_not_affected(self, tagged_cnn_model):
        """New nodes (like synthetic QDQ nodes) don't get tags from old snapshot."""
        snapshot = capture_metadata(tagged_cnn_model)
        _strip_all_metadata(tagged_cnn_model)

        # Add a synthetic node that wasn't in original
        new_node = helper.make_node(
            "QuantizeLinear",
            ["X", "scale", "zp"],
            ["Q_out"],
            name="input_QuantizeLinear",
        )
        tagged_cnn_model.graph.node.append(new_node)

        result = restore_metadata(tagged_cnn_model, snapshot)

        # Original 6 restored, new node untouched
        assert result.nodes_restored == 6

        q_node = next(n for n in tagged_cnn_model.graph.node if n.name == "input_QuantizeLinear")
        assert len(q_node.metadata_props) == 0

    def test_model_level_io_metadata_survives(self, tagged_cnn_model):
        """winml.io.inputs/outputs restored after strip."""
        snapshot = capture_metadata(tagged_cnn_model)
        _strip_all_metadata(tagged_cnn_model)

        restore_metadata(tagged_cnn_model, snapshot)

        meta = {p.key: p.value for p in tagged_cnn_model.metadata_props}
        assert "winml.io.inputs" in meta
        assert '"pixel_values"' not in meta["winml.io.inputs"]  # our fixture uses "X"
        assert '"X"' in meta["winml.io.inputs"]
