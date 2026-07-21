# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for DynamoMetadataTagger.

The dynamo ONNX exporter (``torch.onnx.export(dynamo=True)``) records the
originating module hierarchy on each node's ``metadata_props`` as ``repr()``
of two aligned Python lists:

- ``pkg.torch.onnx.name_scopes``: cumulative module paths, first entry ``""``
  (root), last entry the node's own name.
- ``pkg.torch.onnx.class_hierarchy``: parallel fully-qualified class names,
  last entry the aten op target.

These tests build synthetic ONNX nodes carrying that exact metadata format and
assert the tagger reproduces the ``/Root/Child.N/Leaf`` hierarchy-tag contract.
"""

from __future__ import annotations

import onnx
from onnx import helper

from winml.modelkit.core.onnx_node_tagger import DynamoMetadataTagger


NAME_SCOPES_KEY = "pkg.torch.onnx.name_scopes"
CLASS_HIERARCHY_KEY = "pkg.torch.onnx.class_hierarchy"


def _make_node(
    op_type: str,
    name: str,
    *,
    name_scopes: list[str] | None = None,
    class_hierarchy: list[str] | None = None,
    raw_name_scopes: str | None = None,
    raw_class_hierarchy: str | None = None,
) -> onnx.NodeProto:
    """Build an ONNX node with dynamo-style metadata_props.

    ``name_scopes``/``class_hierarchy`` are serialized with ``repr`` exactly as
    torch does. ``raw_*`` overrides let a test inject malformed strings.
    """
    node = helper.make_node(op_type, inputs=["x"], outputs=["y"], name=name)
    if raw_name_scopes is not None:
        node.metadata_props.append(
            onnx.StringStringEntryProto(key=NAME_SCOPES_KEY, value=raw_name_scopes)
        )
    elif name_scopes is not None:
        node.metadata_props.append(
            onnx.StringStringEntryProto(key=NAME_SCOPES_KEY, value=repr(name_scopes))
        )
    if raw_class_hierarchy is not None:
        node.metadata_props.append(
            onnx.StringStringEntryProto(key=CLASS_HIERARCHY_KEY, value=raw_class_hierarchy)
        )
    elif class_hierarchy is not None:
        node.metadata_props.append(
            onnx.StringStringEntryProto(key=CLASS_HIERARCHY_KEY, value=repr(class_hierarchy))
        )
    return node


def _make_model(nodes: list[onnx.NodeProto]) -> onnx.ModelProto:
    """Wrap nodes in a minimal ModelProto (tagger only iterates graph.node)."""
    graph = helper.make_graph(nodes, "g", inputs=[], outputs=[])
    return helper.make_model(graph)


def _resnet_like_nodes() -> list[onnx.NodeProto]:
    """Two nodes mirroring the empirically verified dynamo output shape."""
    return [
        _make_node(
            "Gemm",
            "node_gemm",
            name_scopes=["", "blocks.0", "blocks.0.lin", "linear"],
            class_hierarchy=[
                "pkg.Net",
                "pkg.Blk",
                "torch.nn.modules.linear.Linear",
                "aten.linear.default",
            ],
        ),
        _make_node(
            "Relu",
            "node_relu",
            name_scopes=["", "blocks.0", "blocks.0.act", "relu"],
            class_hierarchy=[
                "pkg.Net",
                "pkg.Blk",
                "torch.nn.modules.activation.ReLU",
                "aten.relu.default",
            ],
        ),
    ]


class TestDynamoMetadataTagger:
    """Tag generation from dynamo node metadata."""

    def test_tags_indexed_and_named_modules(self) -> None:
        model = _make_model(_resnet_like_nodes())
        tags = DynamoMetadataTagger().tag_all_nodes(model)

        # Digit scope component ("blocks.0") folds into "Blk.0"; named module
        # ("blocks.0.lin" -> Linear) uses the class short-name only.
        assert tags["node_gemm"] == "/Net/Blk.0/Linear"
        assert tags["node_relu"] == "/Net/Blk.0/ReLU"

    def test_model_root_tag_from_first_class(self) -> None:
        tagger = DynamoMetadataTagger()
        tagger.tag_all_nodes(_make_model(_resnet_like_nodes()))
        assert tagger.model_root_tag == "/Net"

    def test_tags_are_never_empty(self) -> None:
        model = _make_model(_resnet_like_nodes())
        tags = DynamoMetadataTagger().tag_all_nodes(model)
        for tag in tags.values():
            assert tag
            assert tag.strip()
            assert tag.startswith("/")

    def test_missing_metadata_falls_back_to_root(self) -> None:
        nodes = _resnet_like_nodes()
        # A node with no dynamo metadata at all (e.g. a fused/optimized node).
        nodes.append(_make_node("Add", "node_bare"))
        model = _make_model(nodes)

        tags = DynamoMetadataTagger().tag_all_nodes(model)
        # Falls back to the derived model root, never empty.
        assert tags["node_bare"] == "/Net"

    def test_malformed_metadata_does_not_raise(self) -> None:
        nodes = _resnet_like_nodes()
        nodes.append(
            _make_node(
                "Add",
                "node_bad",
                raw_name_scopes="not-a-list",
                raw_class_hierarchy="[unclosed",
            )
        )
        model = _make_model(nodes)

        tags = DynamoMetadataTagger().tag_all_nodes(model)
        assert tags["node_bad"] == "/Net"

    def test_no_metadata_anywhere_uses_unknown_root(self) -> None:
        model = _make_model([_make_node("Add", "n0"), _make_node("Mul", "n1")])
        tagger = DynamoMetadataTagger()
        tags = tagger.tag_all_nodes(model)

        assert tagger.model_root_tag == "/UnknownModel"
        assert tags["n0"] == "/UnknownModel"
        assert tags["n1"] == "/UnknownModel"

    def test_root_only_node_tags_to_root(self) -> None:
        # A node whose only module level is the root model itself.
        node = _make_node(
            "Gemm",
            "node_root",
            name_scopes=["", "linear"],
            class_hierarchy=["pkg.Net", "aten.linear.default"],
        )
        tags = DynamoMetadataTagger().tag_all_nodes(_make_model([node]))
        assert tags["node_root"] == "/Net"

    def test_unnamed_node_key_uses_optype_fallback(self) -> None:
        node = _make_node(
            "Gemm",
            "",  # unnamed
            name_scopes=["", "blocks.0", "blocks.0.lin", "linear"],
            class_hierarchy=[
                "pkg.Net",
                "pkg.Blk",
                "torch.nn.modules.linear.Linear",
                "aten.linear.default",
            ],
        )
        tags = DynamoMetadataTagger().tag_all_nodes(_make_model([node]))
        # Key mirrors exporter convention: "<op_type>_<id>".
        (only_key,) = tags.keys()
        assert only_key.startswith("Gemm_")
        assert tags[only_key] == "/Net/Blk.0/Linear"

    def test_get_tagging_statistics(self) -> None:
        nodes = _resnet_like_nodes()
        nodes.append(_make_node("Add", "node_bare"))  # root fallback
        model = _make_model(nodes)

        stats = DynamoMetadataTagger().get_tagging_statistics(model)
        assert stats["total_nodes"] == 3
        assert stats["direct_matches"] == 2
        assert stats["scoped_nodes"] == 2
        assert stats["root_fallbacks"] == 1
        assert stats["unique_scopes"] == 2
        # Keys required by the shared console/report/metadata writers exist.
        for key in ("root_nodes", "parent_matches", "operation_matches"):
            assert key in stats
