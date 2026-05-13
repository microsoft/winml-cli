# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for DocConstraintChecker stable node key behavior."""

from __future__ import annotations

import pytest
from onnx import TensorProto, helper

from winml.modelkit.analyze.core.doc_constraint_checker import DocConstraintChecker


def test_doc_constraint_checker_rejects_unknown_unnamed_node(monkeypatch: pytest.MonkeyPatch):
    """Unknown unnamed nodes should raise instead of using node_obj fallback."""

    monkeypatch.setattr(DocConstraintChecker, "_load_mapping_config", lambda self: {})
    monkeypatch.setattr(DocConstraintChecker, "_load_constraints", lambda self: {})

    node = helper.make_node("Add", ["a", "b"], ["c"], name="add_node")
    input_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1])
    input_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1])
    output_c = helper.make_tensor_value_info("c", TensorProto.FLOAT, [1])
    graph = helper.make_graph([node], "doc_checker_graph", [input_a, input_b], [output_c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    checker = DocConstraintChecker(
        model,
        ep_name="CPUExecutionProvider",
        device_type="CPU",
        skip_shape_inference=True,
    )

    unknown_unnamed_node = helper.make_node("Relu", ["x"], ["y"])
    with pytest.raises(KeyError, match="unnamed node outside DocConstraintChecker model graph"):
        checker._get_stable_node_key(unknown_unnamed_node)
