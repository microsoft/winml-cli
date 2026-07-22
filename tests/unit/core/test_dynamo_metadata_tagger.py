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

from onnx import ModelProto, NodeProto, StringStringEntryProto, helper

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
) -> NodeProto:
    """Build an ONNX node with dynamo-style metadata_props.

    ``name_scopes``/``class_hierarchy`` are serialized with ``repr`` exactly as
    torch does. ``raw_*`` overrides let a test inject malformed strings.
    """
    node = helper.make_node(op_type, inputs=["x"], outputs=["y"], name=name)
    if raw_name_scopes is not None:
        node.metadata_props.append(
            StringStringEntryProto(key=NAME_SCOPES_KEY, value=raw_name_scopes)
        )
    elif name_scopes is not None:
        node.metadata_props.append(
            StringStringEntryProto(key=NAME_SCOPES_KEY, value=repr(name_scopes))
        )
    if raw_class_hierarchy is not None:
        node.metadata_props.append(
            StringStringEntryProto(key=CLASS_HIERARCHY_KEY, value=raw_class_hierarchy)
        )
    elif class_hierarchy is not None:
        node.metadata_props.append(
            StringStringEntryProto(key=CLASS_HIERARCHY_KEY, value=repr(class_hierarchy))
        )
    return node


def _make_model(nodes: list[NodeProto]) -> ModelProto:
    """Wrap nodes in a minimal ModelProto (tagger only iterates graph.node)."""
    graph = helper.make_graph(nodes, "g", inputs=[], outputs=[])
    return helper.make_model(graph)


def _resnet_like_nodes() -> list[NodeProto]:
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

        # Indexed scope component ("blocks.0") folds into "Blk.0"; named modules
        # ("blocks.0.lin" -> Linear, "blocks.0.act" -> ReLU) fold their local
        # attribute name in the same way so same-class siblings stay distinct.
        assert tags["node_gemm"] == "/Net/Blk.0/Linear.lin"
        assert tags["node_relu"] == "/Net/Blk.0/ReLU.act"

    def test_same_class_named_siblings_stay_distinct(self) -> None:
        # Attention query/key/value are all torch.nn.Linear; without folding the
        # local attribute name they would collapse to a single "/.../Linear" tag
        # and could no longer be benchmarked or scoped independently.
        siblings = ("query", "key", "value")
        nodes = [
            _make_node(
                "MatMul",
                f"node_{name}",
                name_scopes=["", "attn", f"attn.{name}", "linear"],
                class_hierarchy=[
                    "pkg.Net",
                    "pkg.Attention",
                    "torch.nn.modules.linear.Linear",
                    "aten.linear.default",
                ],
            )
            for name in siblings
        ]
        tags = DynamoMetadataTagger().tag_all_nodes(_make_model(nodes))

        assert tags["node_query"] == "/Net/Attention.attn/Linear.query"
        assert tags["node_key"] == "/Net/Attention.attn/Linear.key"
        assert tags["node_value"] == "/Net/Attention.attn/Linear.value"
        assert len({tags["node_query"], tags["node_key"], tags["node_value"]}) == 3

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
        assert tags[only_key] == "/Net/Blk.0/Linear.lin"

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


def _attention_nodes() -> list[NodeProto]:
    """Query/key/value nodes for one attention block (all torch.nn.Linear)."""
    return [
        _make_node(
            "MatMul",
            f"node_{name}",
            name_scopes=["", "blocks.0", "blocks.0.attn", f"blocks.0.attn.{name}", "linear"],
            class_hierarchy=[
                "pkg.Net",
                "pkg.Blk",
                "pkg.Attention",
                "torch.nn.modules.linear.Linear",
                "aten.linear.default",
            ],
        )
        for name in ("query", "key", "value")
    ]


class TestBuildModuleHierarchy:
    """Reconstruction of the flat module hierarchy from dynamo node metadata."""

    def test_root_module_keyed_by_empty_string(self) -> None:
        hierarchy = DynamoMetadataTagger().build_module_hierarchy(_make_model(_resnet_like_nodes()))
        assert "" in hierarchy
        assert hierarchy[""]["class_name"] == "Net"
        assert hierarchy[""]["traced_tag"] == "/Net"

    def test_cumulative_scope_paths_and_tags(self) -> None:
        hierarchy = DynamoMetadataTagger().build_module_hierarchy(_make_model(_resnet_like_nodes()))
        # Each module level is keyed by its cumulative dotted scope path with the
        # matching class name and full hierarchy tag.
        assert hierarchy["blocks.0"]["class_name"] == "Blk"
        assert hierarchy["blocks.0"]["traced_tag"] == "/Net/Blk.0"
        assert hierarchy["blocks.0.lin"]["class_name"] == "Linear"
        assert hierarchy["blocks.0.lin"]["traced_tag"] == "/Net/Blk.0/Linear.lin"
        assert hierarchy["blocks.0.act"]["traced_tag"] == "/Net/Blk.0/ReLU.act"

    def test_same_class_siblings_get_distinct_entries(self) -> None:
        hierarchy = DynamoMetadataTagger().build_module_hierarchy(_make_model(_attention_nodes()))
        for name in ("query", "key", "value"):
            key = f"blocks.0.attn.{name}"
            assert hierarchy[key]["class_name"] == "Linear"
            assert hierarchy[key]["traced_tag"] == f"/Net/Blk.0/Attention.attn/Linear.{name}"
        # The shared attention parent is recorded once.
        assert hierarchy["blocks.0.attn"]["class_name"] == "Attention"

    def test_execution_order_is_unique_per_module(self) -> None:
        hierarchy = DynamoMetadataTagger().build_module_hierarchy(_make_model(_attention_nodes()))
        orders = [info["execution_order"] for info in hierarchy.values()]
        assert all(isinstance(o, int) for o in orders)
        assert len(orders) == len(set(orders))

    def test_nodes_without_metadata_are_skipped(self) -> None:
        nodes = _resnet_like_nodes()
        nodes.append(_make_node("Add", "node_bare"))  # no metadata
        nodes.append(
            _make_node(
                "Add",
                "node_bad",
                raw_name_scopes="not-a-list",
                raw_class_hierarchy="[unclosed",
            )
        )
        hierarchy = DynamoMetadataTagger().build_module_hierarchy(_make_model(nodes))
        # Only the real module scopes are present; malformed nodes contribute
        # nothing and do not raise.
        assert set(hierarchy) == {"", "blocks.0", "blocks.0.lin", "blocks.0.act"}

    def test_empty_graph_returns_empty_hierarchy(self) -> None:
        hierarchy = DynamoMetadataTagger().build_module_hierarchy(_make_model([]))
        assert hierarchy == {}
