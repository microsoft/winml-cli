# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for onnx/utils.py — strip_node_attrs."""

from __future__ import annotations

from onnx import ModelProto, NodeProto, TensorProto, helper

from winml.modelkit.onnx import strip_node_attrs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gqa_model(attr_dict: dict[str, int]) -> ModelProto:
    """Build a minimal model with a single com.microsoft::GroupQueryAttention node.

    Uses explicit ``make_attribute`` calls so attr names match what PyTorch's
    TorchScript ONNX exporter produces (no ``_i`` suffix).
    """
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64, 512])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64, 512])
    node = NodeProto()
    node.op_type = "GroupQueryAttention"
    node.domain = "com.microsoft"
    node.input.append("x")
    node.output.append("y")
    for name, value in attr_dict.items():
        attr = helper.make_attribute(name, value)
        node.attribute.append(attr)
    graph = helper.make_graph([node], "gqa_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("com.microsoft", 1)])


def _attr_names(model: ModelProto) -> set[str]:
    return {a.name for n in model.graph.node for a in n.attribute}


# ---------------------------------------------------------------------------
# strip_node_attrs
# ---------------------------------------------------------------------------


def test_strip_removes_extra_attrs():
    """Attributes not in keep_attrs are removed from matching nodes."""
    model = _make_gqa_model(
        {
            "do_rotary": 1,
            "num_heads": 16,
            "kv_num_heads": 8,
            "local_window_size": -1,
            "smooth_softmax": -1,
        }
    )
    keep = frozenset({"do_rotary", "num_heads", "kv_num_heads"})
    result = strip_node_attrs(model, "GroupQueryAttention", keep, domain="com.microsoft")
    remaining = _attr_names(result)
    assert remaining == keep


def test_strip_keep_attrs_preserved():
    """Attributes listed in keep_attrs survive stripping."""
    model = _make_gqa_model({"do_rotary": 1, "num_heads": 16, "kv_num_heads": 8})
    keep = frozenset({"do_rotary", "num_heads", "kv_num_heads"})
    strip_node_attrs(model, "GroupQueryAttention", keep, domain="com.microsoft")
    remaining = _attr_names(model)
    assert "do_rotary" in remaining
    assert "num_heads" in remaining
    assert "kv_num_heads" in remaining


def test_strip_no_matching_nodes_is_noop():
    """strip_node_attrs is a no-op when no nodes match op_type."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    result = strip_node_attrs(model, "GroupQueryAttention", frozenset(), domain="com.microsoft")
    assert result is model  # same object returned


def test_strip_domain_mismatch_is_noop():
    """Nodes with a different domain are not modified."""
    model = _make_gqa_model({"do_rotary_i": 1, "extra_i": 0})
    before = _attr_names(model)
    # Pass wrong domain — nothing should be removed
    strip_node_attrs(model, "GroupQueryAttention", frozenset({"do_rotary"}), domain="wrong.domain")
    assert _attr_names(model) == before


def test_strip_keep_all_attrs():
    """When keep_attrs contains all attr names, nothing is removed."""
    model = _make_gqa_model({"do_rotary": 1, "num_heads": 16})
    keep = frozenset({"do_rotary", "num_heads"})
    strip_node_attrs(model, "GroupQueryAttention", keep, domain="com.microsoft")
    assert _attr_names(model) == keep


def test_strip_empty_keep_attrs_removes_all():
    """An empty keep_attrs set removes every attribute from matching nodes."""
    model = _make_gqa_model({"do_rotary": 1, "num_heads": 16})
    strip_node_attrs(model, "GroupQueryAttention", frozenset(), domain="com.microsoft")
    assert _attr_names(model) == set()


def test_strip_returns_same_model_object():
    """strip_node_attrs mutates in-place and returns the same object."""
    model = _make_gqa_model({"do_rotary": 1})
    result = strip_node_attrs(
        model, "GroupQueryAttention", frozenset({"do_rotary"}), domain="com.microsoft"
    )
    assert result is model


def test_strip_multiple_gqa_nodes():
    """All matching nodes in a multi-node graph are stripped."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64, 512])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64, 512])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 64, 512])

    def _gqa_node(name: str, inp: str, out: str) -> NodeProto:
        node = NodeProto()
        node.op_type = "GroupQueryAttention"
        node.domain = "com.microsoft"
        node.name = name
        node.input.append(inp)
        node.output.append(out)
        for attr_name, value in [("do_rotary", 1), ("num_heads", 16), ("local_window_size", -1)]:
            node.attribute.append(helper.make_attribute(attr_name, value))
        return node

    graph = helper.make_graph(
        [_gqa_node("gqa0", "x", "y"), _gqa_node("gqa1", "y", "z")],
        "multi_gqa",
        [x],
        [z],
        value_info=[y],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("com.microsoft", 1)])
    keep = frozenset({"do_rotary", "num_heads"})
    strip_node_attrs(model, "GroupQueryAttention", keep, domain="com.microsoft")
    for node in model.graph.node:
        assert {a.name for a in node.attribute} == keep
