# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for Qwen3 genai ONNX graph passes (strip_gqa_default_attrs)."""

from __future__ import annotations

from onnx import ModelProto, NodeProto, TensorProto, helper

from winml.modelkit.models.hf.qwen3 import strip_gqa_default_attrs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gqa_model(attr_dict: dict[str, int], *, domain: str = "com.microsoft") -> ModelProto:
    """Build a minimal model with a single GroupQueryAttention node."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64, 512])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64, 512])
    node = NodeProto()
    node.op_type = "GroupQueryAttention"
    node.domain = domain
    node.input.append("x")
    node.output.append("y")
    for name, value in attr_dict.items():
        node.attribute.append(helper.make_attribute(name, value))
    graph = helper.make_graph([node], "gqa_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid(domain, 1)])


def _attr_names(model: ModelProto) -> set[str]:
    return {a.name for n in model.graph.node for a in n.attribute}


# ---------------------------------------------------------------------------
# strip_gqa_default_attrs
# ---------------------------------------------------------------------------


def test_strip_gqa_keeps_only_required_attrs():
    """Exporter-injected extras are removed; the required Qwen3 attrs remain."""
    model = _make_gqa_model(
        {
            "do_rotary": 1,
            "num_heads": 16,
            "kv_num_heads": 8,
            "local_window_size": -1,
            "smooth_softmax": -1,
            "k_quant_type": 0,
            "v_quant_type": 0,
        }
    )
    strip_gqa_default_attrs(model)
    assert _attr_names(model) == {"do_rotary", "num_heads", "kv_num_heads"}


def test_strip_gqa_noop_when_only_required_present():
    """When only the required attrs are present, nothing is removed."""
    model = _make_gqa_model({"do_rotary": 1, "num_heads": 16, "kv_num_heads": 8})
    strip_gqa_default_attrs(model)
    assert _attr_names(model) == {"do_rotary", "num_heads", "kv_num_heads"}


def test_strip_gqa_returns_same_model_object():
    """The pass mutates in-place and returns the same proto for chaining."""
    model = _make_gqa_model({"do_rotary": 1})
    assert strip_gqa_default_attrs(model) is model


def test_strip_gqa_ignores_default_domain_nodes():
    """Only com.microsoft::GroupQueryAttention nodes are targeted."""
    model = _make_gqa_model({"do_rotary": 1, "local_window_size": -1}, domain="")
    strip_gqa_default_attrs(model)
    assert _attr_names(model) == {"do_rotary", "local_window_size"}
