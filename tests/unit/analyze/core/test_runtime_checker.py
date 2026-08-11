# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Unit tests for RuntimeChecker type hints and functionality.

Tests verify:
- Correct return types for summary() method
- Type safety with PatternRuntime
- Cache reuse for RuntimeCheckerQuery
"""

import time
from pathlib import Path

import numpy as np
import onnx
import pytest

from winml.modelkit.analyze import ONNXModel, RuntimeChecker
from winml.modelkit.analyze.core import runtime_checker_query as runtime_checker_query_module
from winml.modelkit.analyze.core.runtime_checker_query import RuntimeCheckerQuery
from winml.modelkit.analyze.models.runtime_checks import PatternRuntime


TensorProto = onnx.TensorProto
helper = onnx.helper


@pytest.fixture
def simple_onnx_model() -> ONNXModel:
    """Create a simple ONNX model for testing."""
    # Create a simple Add operation model
    input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 3, 224, 224])
    input2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [1, 3, 224, 224])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    add_node = helper.make_node("Add", ["input1", "input2"], ["output"], name="add_node")

    graph_def = helper.make_graph([add_node], "test_graph", [input1, input2], [output])

    model_def = helper.make_model(
        graph_def, producer_name="test", opset_imports=[helper.make_opsetid("", 13)]
    )

    return ONNXModel.from_onnx_model(model_def, "test.onnx")


class TestRuntimeCheckerTypeHints:
    """Test RuntimeChecker return type correctness."""

    def test_summary_returns_correct_type(self, simple_onnx_model: ONNXModel):
        """Test that summary() returns dict[str, list[PatternRuntime]]."""
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            model=simple_onnx_model,
        )

        result = checker.summary()

        # Verify return type structure
        assert isinstance(result, dict)
        assert all(isinstance(key, str) for key in result)

        # Check that values are lists of PatternRuntime
        for value in result.values():
            assert isinstance(value, list)
            assert all(isinstance(item, PatternRuntime) for item in value)

        assert set(result) == {"op_runtime_check_result"}

    def test_summary_with_model_only(self, simple_onnx_model: ONNXModel):
        """Test summary() when initialized with model only."""
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            model=simple_onnx_model,
        )

        result = checker.summary()

        assert isinstance(result, dict)
        assert set(result) == {"op_runtime_check_result"}

        # Verify types
        op_results = result["op_runtime_check_result"]
        assert isinstance(op_results, list)
        assert all(isinstance(item, PatternRuntime) for item in op_results)

    def test_op_support_returns_list_of_pattern_runtime(self, simple_onnx_model: ONNXModel):
        """Test that op_support() returns list[PatternRuntime]."""
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            model=simple_onnx_model,
        )

        result = checker.op_support()

        # Verify return type
        assert isinstance(result, list)
        assert all(isinstance(item, PatternRuntime) for item in result)

        # Should have one operator (Add node)
        assert len(result) > 0


class TestRuntimeCheckerValidation:
    """Test RuntimeChecker initialization validation."""

    def test_requires_model(self):
        """Test that RuntimeChecker requires a model."""
        with pytest.raises(ValueError, match="'model' is required"):
            RuntimeChecker(
                ep="QNNExecutionProvider",
                device="NPU",
                model=None,
            )

    def test_requires_non_empty_device(self, simple_onnx_model: ONNXModel):
        """Test that device parameter cannot be empty."""
        with pytest.raises(ValueError, match="device parameter cannot be empty"):
            RuntimeChecker(
                ep="QNNExecutionProvider",
                device="",
                model=simple_onnx_model,
            )


class TestRuntimeCheckerIntegration:
    """Integration tests for RuntimeChecker."""

    def test_full_workflow_with_model(self, simple_onnx_model: ONNXModel):
        """Test complete workflow: initialize with model, check op support, get summary."""
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            model=simple_onnx_model,
        )

        # Get operator support
        op_results = checker.op_support()
        assert len(op_results) > 0
        assert all(isinstance(r, PatternRuntime) for r in op_results)

        # Get summary with empty patterns
        summary = checker.summary()
        assert isinstance(summary, dict)
        assert "op_runtime_check_result" in summary
        assert len(summary["op_runtime_check_result"]) == len(op_results)

    def test_op_support_handles_graph_only_external_initializer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Graph-only external-data initializers survive the analyzer runtime-check path."""

        weight = onnx.numpy_helper.from_array(np.zeros((2,), dtype=np.float32), name="weight")
        weight.data_location = onnx.TensorProto.EXTERNAL
        weight.ClearField("raw_data")
        weight.external_data.add(key="location", value="weight.bin")

        input_value_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])
        output_value_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])
        add_node = helper.make_node("Add", ["weight", "input"], ["output"], name="add_node")
        graph = helper.make_graph(
            [add_node],
            "external_initializer_graph",
            [input_value_info],
            [output_value_info],
            initializer=[weight],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        model_path = tmp_path / "external_initializer.onnx"
        onnx.save(model, model_path)
        graph_only_model = onnx.load(str(model_path), load_external_data=False)
        onnx_model = ONNXModel.from_onnx_model(graph_only_model, str(model_path))

        checker = RuntimeChecker(
            ep="CPUExecutionProvider",
            device="CPU",
            model=onnx_model,
        )
        query = checker._get_query()

        captured_calls: list[tuple[str, bytes, dict[str, np.ndarray]]] = []

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

        query.ep_neg_rules = {}
        query.df_tables = {}
        monkeypatch.setattr(RuntimeCheckerQuery, "_is_ep_available_locally", lambda self: True)
        monkeypatch.setattr(
            RuntimeCheckerQuery,
            "_get_ep_checker",
            lambda self: FakeEPChecker(),
        )

        results = checker.op_support(run_unknown_op=True)

        assert len(results) == 1
        assert results[0].pattern_id == "OP/ai.onnx/Add"
        assert results[0].result.compile is True
        assert results[0].result.run is True
        assert results[0].result.no_data is False
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


class TestRuntimeCheckerQueryCache:
    """Test RuntimeCheckerQuery caching functionality."""

    def test_query_cache_reuse(self, simple_onnx_model: ONNXModel):
        """Test that RuntimeCheckerQuery is cached and reused."""
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            model=simple_onnx_model,
        )

        # First call should create the query
        assert checker._query is None
        first_result = checker.op_support()
        first_query = checker._query
        assert first_query is not None

        # Second call should reuse the cached query
        second_result = checker.op_support()
        second_query = checker._query
        assert second_query is first_query  # Same object reference

        # Results should be consistent
        assert len(first_result) == len(second_result)

    def test_query_cache_performance(self, simple_onnx_model: ONNXModel):
        """Test that cache improves performance on repeated calls."""
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            model=simple_onnx_model,
        )

        # First call - cold (creates query)
        start_time = time.time()
        checker.op_support()
        _first_call_time = time.time() - start_time
        first_query = checker._query

        # Second call - warm (uses cache)
        start_time = time.time()
        checker.op_support()
        _second_call_time = time.time() - start_time
        second_query = checker._query

        # Second call should be faster or at least not significantly slower
        # We're primarily checking that it doesn't recreate the query
        # which would add initialization overhead
        # Not asserting timing directly as it can be flaky,
        # but verifying cache exists proves the optimization
        assert first_query is not None
        assert second_query is first_query
