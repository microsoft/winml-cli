# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for runtime checker query helper functions."""

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from winml.modelkit.analyze.core import runtime_checker_query as runtime_checker_query_module
from winml.modelkit.analyze.core.runtime_checker_query import (
    RuntimeCheckerQuery,
    _build_query_signature,
    _build_table_filter_conditions,
    get_query_conditions_for_node,
    node_to_pattern_match,
    try_load_external_initializer_array,
)
from winml.modelkit.analyze.exceptions import OpOptionalInputSupportError
from winml.modelkit.analyze.utils.model_utils import DUMMY_FLOAT
from winml.modelkit.onnx import ONNXDomain


class TestBuildTableFilterConditions:
    """Test table filter condition extraction for runtime rule lookups."""

    def test_returns_conditions_for_all_present_columns(self):
        """Returns only the requested table columns when all are present."""
        conditions = {
            "input_shape": (1, 3, 224, 224),
            "attr_axis": 1,
            "unused_key": "ignored",
        }

        result = _build_table_filter_conditions(
            conditions=conditions,
            column_names=["input_shape", "attr_axis"],
            infinite_properties=[],
            error_context="op Add",
        )

        assert result == {
            "input_shape": (1, 3, 224, 224),
            "attr_axis": 1,
        }

    def test_raises_when_required_column_is_missing(self):
        """Raises a descriptive error when a required table column is unavailable."""
        conditions = {"input_shape": (1, 3, 224, 224)}

        with pytest.raises(OpOptionalInputSupportError) as exc_info:
            _build_table_filter_conditions(
                conditions=conditions,
                column_names=["input_shape", "attr_axis"],
                infinite_properties=[],
                error_context="op Add",
            )

        assert str(exc_info.value) == (
            "Match key 'attr_axis' not found in conditions for op Add. Available: ['input_shape']"
        )

    def test_skips_columns_listed_in_infinite_properties(self):
        """Columns marked infinite are skipped even when absent from conditions."""
        conditions = {
            "input_shape": (1, 3, 224, 224),
            "attr_axis": 1,
        }

        result = _build_table_filter_conditions(
            conditions=conditions,
            column_names=["input_shape", "attr_axis", "input_value"],
            infinite_properties=["input_value"],
            error_context="op Add",
        )

        assert result == {
            "input_shape": (1, 3, 224, 224),
            "attr_axis": 1,
        }

    def test_returns_empty_dict_for_empty_column_names(self):
        """Returns an empty mapping when no table columns are requested."""
        result = _build_table_filter_conditions(
            conditions={"input_shape": (1, 3, 224, 224)},
            column_names=[],
            infinite_properties=[],
            error_context="op Add",
        )

        assert result == {}


class TestBuildQuerySignature:
    """Test query-signature construction for result-cache keys."""

    def test_skips_columns_absent_from_filter_conditions(self):
        """Columns not used by table filter (for example infinite properties) are ignored."""
        signature = _build_query_signature(
            column_names=["T_Add", "A_shape", "input_dim"],
            filter_conditions={
                "T_Add": "FLOAT",
                "input_dim": 4,
            },
        )

        assert signature == ("FLOAT", 4)

    def test_preserves_column_order_for_present_entries(self):
        """Signature order follows table-column order after dropping missing columns."""
        signature = _build_query_signature(
            column_names=["input_dim", "T_Add", "A_shape"],
            filter_conditions={
                "T_Add": "FLOAT",
                "input_dim": 4,
            },
        )

        assert signature == (4, "FLOAT")


class TestGetQueryConditionsForNode:
    """Test condition extraction for runtime rule lookups."""

    def test_external_initializer_without_payload_is_not_marked_constant(self):
        """External-data initializers without loaded values keep shape but not constant status."""
        node = helper.make_node("Add", ["weight", "input"], ["output"], name="add_node")
        input_value_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])

        external_initializer = onnx.TensorProto()
        external_initializer.name = "weight"
        external_initializer.data_type = TensorProto.FLOAT
        external_initializer.dims.extend([2])
        external_initializer.data_location = TensorProto.EXTERNAL
        external_initializer.external_data.add(key="location", value="weight.bin")

        conditions, infinite_properties, is_qdq = get_query_conditions_for_node(
            node=node,
            opset_version=17,
            valueinfo={"input": input_value_info},
            initializers={"weight": external_initializer},
            constants={},
            domain=ONNXDomain.AI_ONNX,
            input_to_dq={},
            output_to_q={},
        )

        assert conditions["A_is_constant"] is False
        assert conditions["A_shape"] == (2,)
        assert conditions["A_value"] is None
        assert conditions["A_is_fixed_shape"] is True
        assert conditions["A_dynamic_axes"] == ()
        assert conditions["A_is_none"] is False
        assert infinite_properties == ["A_shape", "B_shape"]
        assert is_qdq is False

    def test_external_initializer_sidecar_is_loaded_when_model_path_is_available(
        self,
        tmp_path: Path,
    ) -> None:
        """Small external-data initializers can be resolved from the model sidecar."""
        weight = onnx.numpy_helper.from_array(
            np.array([1.5, -2.0], dtype=np.float32),
            name="weight",
        )
        node = helper.make_node("Add", ["weight", "input"], ["output"], name="add_node")
        input_value_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])
        output_value_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])
        graph = helper.make_graph(
            [node],
            "external_initializer_graph",
            [input_value_info],
            [output_value_info],
            initializer=[weight],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model_path = tmp_path / "external_initializer.onnx"
        onnx.save_model(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.bin",
            size_threshold=0,
        )
        graph_only_model = onnx.load(str(model_path), load_external_data=False)

        conditions, infinite_properties, is_qdq = get_query_conditions_for_node(
            node=node,
            opset_version=17,
            valueinfo={"input": input_value_info},
            initializers={"weight": graph_only_model.graph.initializer[0]},
            constants={},
            domain=ONNXDomain.AI_ONNX,
            input_to_dq={},
            output_to_q={},
            model_path=model_path,
        )

        assert conditions["A_is_constant"] is True
        assert conditions["A_shape"] == (2,)
        assert conditions["A_value"] == (DUMMY_FLOAT, DUMMY_FLOAT)
        assert conditions["A_is_none"] is False
        assert infinite_properties == ["A_shape", "B_shape"]
        assert is_qdq is False

    def test_try_load_external_initializer_array_returns_plain_ndarray(
        self,
        tmp_path: Path,
    ) -> None:
        """Loaded sidecar tensors are copied into memory and do not retain file handles."""
        weight = onnx.numpy_helper.from_array(
            np.array([[1.5, -2.0]], dtype=np.float32),
            name="weight",
        )
        node = helper.make_node("Identity", ["weight"], ["output"], name="identity_node")
        output_value_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
        graph = helper.make_graph(
            [node],
            "external_initializer_graph",
            [],
            [output_value_info],
            initializer=[weight],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model_path = tmp_path / "external_initializer.onnx"
        onnx.save_model(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.bin",
            size_threshold=0,
        )
        graph_only_model = onnx.load(str(model_path), load_external_data=False)

        loaded = try_load_external_initializer_array(
            graph_only_model.graph.initializer[0],
            model_path,
        )

        assert loaded is not None
        assert isinstance(loaded, np.ndarray)
        assert not isinstance(loaded, np.memmap)
        assert not isinstance(getattr(loaded, "base", None), np.memmap)
        assert np.array_equal(loaded, np.array([[1.5, -2.0]], dtype=np.float32))

        sidecar_path = tmp_path / "weights.bin"
        renamed_sidecar_path = tmp_path / "weights-renamed.bin"
        sidecar_path.rename(renamed_sidecar_path)
        assert renamed_sidecar_path.exists()


class TestLocalEPFallback:
    """Test local EP fallback helpers for single-node execution."""

    def test_local_ep_check_feeds_promoted_external_initializer(self, monkeypatch):
        """Promoted external-data initializers are included in the local EP input feed."""
        node = helper.make_node("Add", ["weight", "input"], ["output"], name="add_node")
        input_value_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])
        output_value_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])

        external_initializer = onnx.TensorProto()
        external_initializer.name = "weight"
        external_initializer.data_type = TensorProto.FLOAT
        external_initializer.dims.extend([2])
        external_initializer.data_location = TensorProto.EXTERNAL
        external_initializer.external_data.add(key="location", value="weight.bin")

        graph = helper.make_graph(
            [node],
            "external_initializer_graph",
            [input_value_info],
            [output_value_info],
            initializer=[external_initializer],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        query = RuntimeCheckerQuery(model, ep_name="CPUExecutionProvider", device_type="CPU")

        captured_calls = []

        class FakeRunner:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def run(self, fn, *args):
                return {"result": fn(*args), "stdout": "", "stderr": ""}

        class FakeEPChecker:
            def check_compile(self, model_bytes, input_feed):
                captured_calls.append(("compile", model_bytes, input_feed))
                return {"success": True}

            def check_run(self, model_bytes, input_feed):
                captured_calls.append(("run", model_bytes, input_feed))
                return {"success": True}

        monkeypatch.setattr(runtime_checker_query_module, "ResilientRunner", FakeRunner)
        monkeypatch.setattr(RuntimeCheckerQuery, "_is_ep_available_locally", lambda self: True)
        monkeypatch.setattr(
            RuntimeCheckerQuery,
            "_get_ep_checker",
            lambda self: FakeEPChecker(),
        )

        result = query._try_local_ep_check(
            node=node,
            op_domain=ONNXDomain.AI_ONNX,
            opset_version=17,
            pattern_match=node_to_pattern_match(node),
            node_tags=[],
            fallback_reason="rules_not_found",
        )

        assert result is not None
        assert result.result.compile is True
        assert result.result.run is True
        assert [phase for phase, _, _ in captured_calls] == ["compile", "run"]

        for _, model_bytes, input_feed in captured_calls:
            assert set(input_feed) == {"weight", "input"}
            assert input_feed["weight"].shape == (2,)
            assert input_feed["weight"].dtype == np.float32
            assert input_feed["input"].shape == (2,)

            single_node_model = onnx.ModelProto()
            single_node_model.ParseFromString(model_bytes)
            assert {vi.name for vi in single_node_model.graph.input} == {"weight", "input"}
            assert {init.name for init in single_node_model.graph.initializer} == set()
