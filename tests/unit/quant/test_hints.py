# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for optimizer-to-quantizer region hints."""

from __future__ import annotations

import json

import numpy as np
import pytest
from onnx import (
    GraphProto,
    ModelProto,
    StringStringEntryProto,
    TensorProto,
    checker,
    helper,
    numpy_helper,
)

from winml.modelkit.quant.hints import (
    QUANTIZATION_REGION_HINT_KEY,
    QuantizationHintError,
    canonicalize_quantization_region_hints,
)


_REMOVED_NODE_NAMES = {
    "branch0_output_q",
    "branch0_output_dq",
    "branch1_output_q",
    "branch1_output_dq",
}


def _scalar(value: object, dtype: np.dtype, name: str) -> TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)


def _make_hinted_qdq_model() -> ModelProto:
    initializers = [
        _scalar(0.1, np.dtype(np.float32), "input_scale"),
        _scalar(128, np.dtype(np.uint8), "input_zero_point"),
        _scalar(0.05, np.dtype(np.float32), "weight_scale"),
        _scalar(120, np.dtype(np.uint8), "weight_zero_point"),
        _scalar(0.005, np.dtype(np.float32), "bias_scale"),
        _scalar(0, np.dtype(np.int32), "bias_zero_point"),
        _scalar(0.2, np.dtype(np.float32), "output_scale"),
        _scalar(127, np.dtype(np.uint8), "output_zero_point"),
        numpy_helper.from_array(np.full((2, 2, 1), 121, dtype=np.uint8), "weight0_q"),
        numpy_helper.from_array(np.full((2, 2, 1), 122, dtype=np.uint8), "weight1_q"),
        numpy_helper.from_array(np.array([1, 2], dtype=np.int32), "bias0_q"),
        numpy_helper.from_array(np.array([3, 4], dtype=np.int32), "bias1_q"),
        numpy_helper.from_array(np.ones((1, 4, 1), dtype=np.float32), "other_weight"),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["input", "input_scale", "input_zero_point"],
            ["input_q"],
            name="input_q",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["input_q", "input_scale", "input_zero_point"],
            ["input_dq"],
            name="input_dq",
        ),
        helper.make_node(
            "Split", ["input_dq"], ["branch0_input", "branch1_input"], name="split", axis=1
        ),
        helper.make_node(
            "DequantizeLinear",
            ["weight0_q", "weight_scale", "weight_zero_point"],
            ["weight0_dq"],
            name="weight0_dq",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["bias0_q", "bias_scale", "bias_zero_point"],
            ["bias0_dq"],
            name="bias0_dq",
        ),
        helper.make_node(
            "Conv",
            ["branch0_input", "weight0_dq", "bias0_dq"],
            ["branch0_output"],
            name="branch0",
        ),
        helper.make_node(
            "QuantizeLinear",
            ["branch0_output", "output_scale", "output_zero_point"],
            ["branch0_output_q_value"],
            name="branch0_output_q",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["branch0_output_q_value", "output_scale", "output_zero_point"],
            ["branch0_output_dq_value"],
            name="branch0_output_dq",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["weight1_q", "weight_scale", "weight_zero_point"],
            ["weight1_dq"],
            name="weight1_dq",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["bias1_q", "bias_scale", "bias_zero_point"],
            ["bias1_dq"],
            name="bias1_dq",
        ),
        helper.make_node(
            "Conv",
            ["branch1_input", "weight1_dq", "bias1_dq"],
            ["branch1_output"],
            name="branch1",
        ),
        helper.make_node(
            "QuantizeLinear",
            ["branch1_output", "output_scale", "output_zero_point"],
            ["branch1_output_q_value"],
            name="branch1_output_q",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["branch1_output_q_value", "output_scale", "output_zero_point"],
            ["branch1_output_dq_value"],
            name="branch1_output_dq",
        ),
        helper.make_node(
            "Concat",
            ["branch0_output_dq_value", "branch1_output_dq_value"],
            ["concat_output"],
            name="concat",
            axis=1,
        ),
        helper.make_node(
            "QuantizeLinear",
            ["concat_output", "output_scale", "output_zero_point"],
            ["concat_output_q"],
            name="concat_output_q",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["concat_output_q", "output_scale", "output_zero_point"],
            ["output"],
            name="concat_output_dq",
        ),
        helper.make_node(
            "Conv",
            ["input_dq", "other_weight"],
            ["other_conv_output"],
            name="other_conv",
        ),
        helper.make_node(
            "QuantizeLinear",
            ["other_conv_output", "output_scale", "output_zero_point"],
            ["other_output_q"],
            name="other_output_q",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["other_output_q", "output_scale", "output_zero_point"],
            ["other_output"],
            name="other_output_dq",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "hinted_qdq",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 3])],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 3]),
            helper.make_tensor_value_info("other_output", TensorProto.FLOAT, [1, 1, 3]),
        ],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.metadata_props.add(key="keep.me", value="untouched")
    model.metadata_props.add(
        key=QUANTIZATION_REGION_HINT_KEY,
        value=json.dumps(
            {
                "version": 1,
                "regions": [
                    {
                        "kind": "conv_concat",
                        "branches": ["branch0", "branch1"],
                        "concat": "concat",
                    }
                ],
            }
        ),
    )
    checker.check_model(model, full_check=True)
    return model


def _hint_entry(model: ModelProto) -> StringStringEntryProto:
    return next(item for item in model.metadata_props if item.key == QUANTIZATION_REGION_HINT_KEY)


class TestCanonicalizeQuantizationRegionHints:
    def test_removes_only_hinted_branch_output_qdq(self) -> None:
        model = _make_hinted_qdq_model()
        original = model.SerializeToString()
        original_nodes = {node.name for node in model.graph.node}
        original_initializers = [
            initializer.SerializeToString() for initializer in model.graph.initializer
        ]

        result = canonicalize_quantization_region_hints(model)

        assert model.SerializeToString() == original
        checker.check_model(result, full_check=True)
        assert {node.name for node in result.graph.node} == original_nodes - _REMOVED_NODE_NAMES
        assert [
            initializer.SerializeToString() for initializer in result.graph.initializer
        ] == original_initializers
        concat = next(node for node in result.graph.node if node.name == "concat")
        assert list(concat.input) == ["branch0_output", "branch1_output"]
        assert {item.key: item.value for item in result.metadata_props} == {"keep.me": "untouched"}
        assert {node.name for node in result.graph.node if node.op_type == "QuantizeLinear"} == {
            "input_q",
            "concat_output_q",
            "other_output_q",
        }

    def test_no_hint_returns_original_model(self) -> None:
        model = _make_hinted_qdq_model()
        model.metadata_props.remove(_hint_entry(model))

        assert canonicalize_quantization_region_hints(model) is model

    @pytest.mark.parametrize(
        "payload",
        [
            "not-json",
            json.dumps({"version": 2, "regions": []}),
            json.dumps({"version": 1, "regions": []}),
            json.dumps(
                {
                    "version": 1,
                    "regions": [
                        {"kind": "unknown", "branches": ["branch0", "branch1"], "concat": "concat"}
                    ],
                }
            ),
        ],
    )
    def test_invalid_hint_fails_without_mutating_input(self, payload: str) -> None:
        model = _make_hinted_qdq_model()
        _hint_entry(model).value = payload
        original = model.SerializeToString()

        with pytest.raises(QuantizationHintError):
            canonicalize_quantization_region_hints(model)

        assert model.SerializeToString() == original

    def test_stale_branch_name_fails_without_mutating_input(self) -> None:
        model = _make_hinted_qdq_model()
        next(node for node in model.graph.node if node.name == "branch1").name = "renamed_branch"
        original = model.SerializeToString()

        with pytest.raises(QuantizationHintError, match="branch1"):
            canonicalize_quantization_region_hints(model)

        assert model.SerializeToString() == original

    def test_branch_q_fanout_fails_without_mutating_input(self) -> None:
        model = _make_hinted_qdq_model()
        model.graph.node.append(
            helper.make_node(
                "Identity",
                ["branch0_output_q_value"],
                ["fanout_output"],
                name="fanout",
            )
        )
        original = model.SerializeToString()

        with pytest.raises(QuantizationHintError, match="consumer"):
            canonicalize_quantization_region_hints(model)

        assert model.SerializeToString() == original

    def test_q_graph_output_fails_without_mutating_input(self) -> None:
        model = _make_hinted_qdq_model()
        model.graph.output.append(
            helper.make_tensor_value_info(
                "branch0_output_q_value",
                TensorProto.UINT8,
                [1, 2, 3],
            )
        )
        original = model.SerializeToString()

        with pytest.raises(QuantizationHintError, match="graph output"):
            canonicalize_quantization_region_hints(model)

        assert model.SerializeToString() == original

    def test_branch_dq_fanout_fails_without_mutating_input(self) -> None:
        model = _make_hinted_qdq_model()
        model.graph.node.append(
            helper.make_node(
                "Identity",
                ["branch0_output_dq_value"],
                ["fanout_output"],
                name="fanout",
            )
        )
        original = model.SerializeToString()

        with pytest.raises(QuantizationHintError, match="consumer"):
            canonicalize_quantization_region_hints(model)

        assert model.SerializeToString() == original

    def test_dq_graph_output_fails_without_mutating_input(self) -> None:
        model = _make_hinted_qdq_model()
        model.graph.output.append(
            helper.make_tensor_value_info(
                "branch0_output_dq_value",
                TensorProto.FLOAT,
                [1, 2, 3],
            )
        )
        original = model.SerializeToString()

        with pytest.raises(QuantizationHintError, match="graph output"):
            canonicalize_quantization_region_hints(model)

        assert model.SerializeToString() == original

    @pytest.mark.parametrize(
        ("captured_name", "data_type"),
        [
            ("branch0_output_q_value", TensorProto.UINT8),
            ("branch0_output_dq_value", TensorProto.FLOAT),
        ],
    )
    def test_nested_capture_fails_without_mutating_input(
        self,
        captured_name: str,
        data_type: int,
    ) -> None:
        model = _make_hinted_qdq_model()
        model.graph.initializer.append(_scalar(True, np.dtype(np.bool_), "condition"))

        def branch_graph(name: str) -> GraphProto:
            output_name = f"{name}_output"
            return helper.make_graph(
                [
                    helper.make_node(
                        "Identity",
                        [captured_name],
                        [output_name],
                        name=f"{name}_identity",
                    )
                ],
                name,
                [],
                [helper.make_tensor_value_info(output_name, data_type, [1, 2, 3])],
            )

        model.graph.node.append(
            helper.make_node(
                "If",
                ["condition"],
                ["captured_output"],
                name="capture_nested_value",
                then_branch=branch_graph("then_branch"),
                else_branch=branch_graph("else_branch"),
            )
        )
        model.graph.output.append(
            helper.make_tensor_value_info("captured_output", data_type, [1, 2, 3])
        )
        checker.check_model(model, full_check=True)
        original = model.SerializeToString()

        with pytest.raises(QuantizationHintError, match="consumer"):
            canonicalize_quantization_region_hints(model)

        assert model.SerializeToString() == original

    def test_duplicate_hinted_branch_name_fails_without_mutating_input(self) -> None:
        model = _make_hinted_qdq_model()
        _hint_entry(model).value = json.dumps(
            {
                "version": 1,
                "regions": [
                    {
                        "kind": "conv_concat",
                        "branches": ["branch0", "branch0"],
                        "concat": "concat",
                    }
                ],
            }
        )
        original = model.SerializeToString()

        with pytest.raises(QuantizationHintError, match="Invalid conv_concat"):
            canonicalize_quantization_region_hints(model)

        assert model.SerializeToString() == original

    def test_direct_branch_to_concat_fails_without_mutating_input(self) -> None:
        model = _make_hinted_qdq_model()
        removed = {"branch0_output_q", "branch0_output_dq"}
        kept = [node for node in model.graph.node if node.name not in removed]
        del model.graph.node[:]
        model.graph.node.extend(kept)
        concat = next(node for node in model.graph.node if node.name == "concat")
        concat.input[0] = "branch0_output"
        original = model.SerializeToString()

        with pytest.raises(QuantizationHintError, match="consumer is Concat"):
            canonicalize_quantization_region_hints(model)

        assert model.SerializeToString() == original

    def test_branch_output_used_as_q_parameter_fails_without_mutating_input(self) -> None:
        model = _make_hinted_qdq_model()
        q_node = next(node for node in model.graph.node if node.name == "branch0_output_q")
        q_node.input[0] = "branch0_input"
        q_node.input[1] = "branch0_output"
        original = model.SerializeToString()

        with pytest.raises(QuantizationHintError, match="data input"):
            canonicalize_quantization_region_hints(model)

        assert model.SerializeToString() == original

    def test_q_output_used_as_dq_parameter_fails_without_mutating_input(self) -> None:
        model = _make_hinted_qdq_model()
        dq_node = next(node for node in model.graph.node if node.name == "branch0_output_dq")
        dq_node.input[0] = "input_q"
        dq_node.input[2] = "branch0_output_q_value"
        original = model.SerializeToString()

        with pytest.raises(QuantizationHintError, match="data input"):
            canonicalize_quantization_region_hints(model)

        assert model.SerializeToString() == original

    @pytest.mark.parametrize(
        "node_name",
        ["branch0", "concat", "branch0_output_q", "branch0_output_dq"],
    )
    def test_unknown_operator_domain_fails_without_mutating_input(self, node_name: str) -> None:
        model = _make_hinted_qdq_model()
        node = next(node for node in model.graph.node if node.name == node_name)
        node.domain = "example.custom"
        model.opset_import.append(helper.make_opsetid("example.custom", 1))
        original = model.SerializeToString()

        with pytest.raises(QuantizationHintError, match="domain"):
            canonicalize_quantization_region_hints(model)

        assert model.SerializeToString() == original

    def test_microsoft_qdq_domain_is_supported(self) -> None:
        model = _make_hinted_qdq_model()
        for node in model.graph.node:
            if node.name in _REMOVED_NODE_NAMES:
                node.domain = "com.microsoft"
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))

        result = canonicalize_quantization_region_hints(model)

        assert {node.name for node in result.graph.node}.isdisjoint(_REMOVED_NODE_NAMES)
