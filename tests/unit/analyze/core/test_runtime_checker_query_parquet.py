# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for per-op parquet runtime rule lookup in RuntimeCheckerQuery."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from onnx import TensorProto, helper

import winml.modelkit.analyze.core.runtime_checker_query as runtime_checker_query_module
from winml.modelkit.analyze.core.runtime_checker_query import RuntimeCheckerQuery
from winml.modelkit.analyze.utils import encode_rule_condition_value_for_parquet


if TYPE_CHECKING:
    from pathlib import Path


def _build_add_model(opset: int = 13):
    """Build a minimal ONNX model with one Add node."""
    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
    input_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 4])
    output_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    add_node = helper.make_node("Add", ["A", "B"], ["Y"], name="add_node")
    graph = helper.make_graph([add_node], "add_graph", [input_a, input_b], [output_y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def _write_parquet_rules(
    rules_dir: Path,
    compile_run_success: tuple[bool, bool] = (True, False),
    extra_columns: dict[str, object] | None = None,
) -> Path:
    """Write parquet artifact equivalent to one Add rule row."""
    parquet_path = (
        rules_dir
        / "QNNExecutionProvider_NPU"
        / "Add_QNNExecutionProvider_NPU_ai.onnx_opset13.parquet"
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, object] = {
        "T_Add": encode_rule_condition_value_for_parquet("FLOAT"),
        "input_dim": encode_rule_condition_value_for_parquet(4),
        "compile_run_success": compile_run_success,
    }
    if extra_columns:
        row.update(extra_columns)
    rule_df = pd.DataFrame([row])
    rule_df.to_parquet(parquet_path, index=False)
    return parquet_path


def _write_legacy_parquet_rules_with_row_index(rules_dir: Path) -> Path:
    """Write legacy parquet artifact that still includes row_index column."""
    parquet_path = (
        rules_dir
        / "QNNExecutionProvider_NPU"
        / "Add_QNNExecutionProvider_NPU_ai.onnx_opset13.parquet"
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    rule_df = pd.DataFrame(
        [
            {
                "row_index": "legacy-row-index",
                "T_Add": encode_rule_condition_value_for_parquet("FLOAT"),
                "input_dim": encode_rule_condition_value_for_parquet(4),
                "compile_run_success": (True, False),
            }
        ]
    )
    rule_df.to_parquet(parquet_path, index=False)
    return parquet_path


@pytest.fixture
def patched_query_conditions(monkeypatch: pytest.MonkeyPatch):
    """Patch condition extraction to stable deterministic values for this test."""

    def _fake_get_query_conditions_for_node(*args, **kwargs):
        del args, kwargs
        return (
            {
                "T_Add": "FLOAT",
                "input_dim": 4,
            },
            [],
            False,
        )

    monkeypatch.setattr(
        runtime_checker_query_module,
        "get_query_conditions_for_node",
        _fake_get_query_conditions_for_node,
    )


@pytest.fixture(autouse=True)
def clear_global_parquet_cache():
    """Ensure each test starts with a clean global parquet cache."""
    runtime_checker_query_module._clear_global_parquet_table_cache()
    yield
    runtime_checker_query_module._clear_global_parquet_table_cache()


@pytest.fixture(autouse=True)
def clear_debug_rules_env(monkeypatch: pytest.MonkeyPatch):
    """Prevent host env debug rules dir from leaking into tests."""
    monkeypatch.delenv("WINMLCLI_RULES_DIR_FOR_DEBUG", raising=False)


class TestRuntimeCheckerQueryParquet:
    """Validate parquet runtime rule lookup."""

    def test_parquet_lookup_returns_expected_result(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patched_query_conditions,
    ):
        """Per-op parquet lookup should return compile/run values from matched parquet row."""
        del patched_query_conditions

        monkeypatch.setenv("WINMLCLI_RULES_DIR", str(tmp_path))
        _write_parquet_rules(tmp_path)

        model = _build_add_model()
        node = model.graph.node[0]

        query_parquet = RuntimeCheckerQuery(model, "QNNExecutionProvider", "NPU")
        query_parquet.node_checkers = []
        result_parquet = query_parquet.run_for_node(node, for_debug=True, run_unknown_op=False)

        assert result_parquet.result.no_data is False
        assert result_parquet.result.compile is True
        assert result_parquet.result.run is False
        assert str(result_parquet.result.debug_details.get("table_file", "")).endswith(".parquet")

    def test_parquet_lookup_omits_debug_details_without_for_debug(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patched_query_conditions,
    ):
        """debug_details should be omitted unless for_debug is explicitly enabled."""
        del patched_query_conditions

        monkeypatch.setenv("WINMLCLI_RULES_DIR", str(tmp_path))
        _write_parquet_rules(tmp_path)

        model = _build_add_model()
        node = model.graph.node[0]

        query_parquet = RuntimeCheckerQuery(model, "QNNExecutionProvider", "NPU")
        query_parquet.node_checkers = []
        result_parquet = query_parquet.run_for_node(node, for_debug=False, run_unknown_op=False)

        assert result_parquet.result.no_data is False
        assert result_parquet.result.compile is True
        assert result_parquet.result.run is False
        assert result_parquet.result.debug_details is None

    def test_rules_not_found_reports_expected_table_path_and_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patched_query_conditions,
    ):
        """Missing parquet should still report the expected lookup path and file name."""
        del patched_query_conditions

        monkeypatch.setenv("WINMLCLI_RULES_DIR", str(tmp_path))

        model = _build_add_model()
        node = model.graph.node[0]

        query_parquet = RuntimeCheckerQuery(model, "QNNExecutionProvider", "NPU")
        query_parquet.node_checkers = []
        result = query_parquet.run_for_node(node, for_debug=True, run_unknown_op=False)

        assert result.result.no_data is True
        assert result.result.reason == "rules_not_found"

        debug_details = result.result.debug_details
        assert isinstance(debug_details, dict)

        expected_file = "Add_QNNExecutionProvider_NPU_ai.onnx_opset13.parquet"
        expected_suffix = f"QNNExecutionProvider_NPU/{expected_file}"

        assert debug_details.get("table_file") == expected_file
        table_path = str(debug_details.get("table_path", "")).replace("\\", "/")
        assert table_path.endswith(expected_suffix)

    def test_parquet_lookup_prefers_debug_dir_when_for_debug(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patched_query_conditions,
    ):
        """for_debug should resolve rules from WINMLCLI_RULES_DIR_FOR_DEBUG first."""
        del patched_query_conditions

        base_rules_dir = tmp_path / "rules"
        debug_rules_dir = tmp_path / "rules_debug"
        base_rules_dir.mkdir(parents=True, exist_ok=True)
        debug_rules_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setenv("WINMLCLI_RULES_DIR", str(base_rules_dir))
        monkeypatch.setenv("WINMLCLI_RULES_DIR_FOR_DEBUG", str(debug_rules_dir))

        _write_parquet_rules(base_rules_dir, compile_run_success=(True, False))
        _write_parquet_rules(
            debug_rules_dir,
            compile_run_success=(False, True),
            extra_columns={"case_indices": ["case_42", "case_43"]},
        )

        model = _build_add_model()
        node = model.graph.node[0]

        query_parquet = RuntimeCheckerQuery(model, "QNNExecutionProvider", "NPU")
        query_parquet.node_checkers = []
        result_parquet = query_parquet.run_for_node(node, for_debug=True, run_unknown_op=False)

        assert result_parquet.result.no_data is False
        assert result_parquet.result.compile is False
        assert result_parquet.result.run is True
        debug_details = result_parquet.result.debug_details
        assert isinstance(debug_details, dict)
        assert "rules_debug" in str(debug_details.get("table_path", ""))
        assert debug_details.get("case_indices") == ("case_42", "case_43")

    def test_parquet_global_cache_reused_across_instances(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patched_query_conditions,
    ):
        """Multiple RuntimeCheckerQuery instances should share the module-level parquet cache."""
        del patched_query_conditions

        monkeypatch.setenv("WINMLCLI_RULES_DIR", str(tmp_path))
        _write_parquet_rules(tmp_path)

        read_count = 0
        original_read_parquet = runtime_checker_query_module.pd.read_parquet

        def _counting_read_parquet(*args, **kwargs):
            nonlocal read_count
            read_count += 1
            return original_read_parquet(*args, **kwargs)

        monkeypatch.setattr(runtime_checker_query_module.pd, "read_parquet", _counting_read_parquet)

        model_a = _build_add_model()
        node_a = model_a.graph.node[0]
        query_a = RuntimeCheckerQuery(model_a, "QNNExecutionProvider", "NPU")
        query_a.node_checkers = []

        model_b = _build_add_model()
        node_b = model_b.graph.node[0]
        query_b = RuntimeCheckerQuery(model_b, "QNNExecutionProvider", "NPU")
        query_b.node_checkers = []

        result_a = query_a.run_for_node(node_a, for_debug=True, run_unknown_op=False)
        result_b = query_b.run_for_node(node_b, for_debug=True, run_unknown_op=False)

        assert result_a.result.no_data is False
        assert result_b.result.no_data is False
        assert read_count == 1

    def test_parquet_global_cache_waits_for_inflight_load(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patched_query_conditions,
    ):
        """Concurrent requests should wait for the same in-flight parquet load."""
        del patched_query_conditions

        monkeypatch.setenv("WINMLCLI_RULES_DIR", str(tmp_path))
        _write_parquet_rules(tmp_path)

        load_started = threading.Event()
        continue_load = threading.Event()
        read_count = 0
        original_read_parquet = runtime_checker_query_module.pd.read_parquet

        def _blocking_read_parquet(*args, **kwargs):
            nonlocal read_count
            read_count += 1
            load_started.set()
            continue_load.wait(timeout=5)
            return original_read_parquet(*args, **kwargs)

        monkeypatch.setattr(runtime_checker_query_module.pd, "read_parquet", _blocking_read_parquet)

        model_a = _build_add_model()
        node_a = model_a.graph.node[0]
        query_a = RuntimeCheckerQuery(model_a, "QNNExecutionProvider", "NPU")
        query_a.node_checkers = []

        model_b = _build_add_model()
        node_b = model_b.graph.node[0]
        query_b = RuntimeCheckerQuery(model_b, "QNNExecutionProvider", "NPU")
        query_b.node_checkers = []

        errors: list[Exception] = []
        results: dict[str, object] = {}

        def _run_a():
            try:
                results["a"] = query_a.run_for_node(node_a, for_debug=True, run_unknown_op=False)
            except Exception as exc:
                errors.append(exc)

        def _run_b():
            try:
                results["b"] = query_b.run_for_node(node_b, for_debug=True, run_unknown_op=False)
            except Exception as exc:
                errors.append(exc)

        thread_a = threading.Thread(target=_run_a)
        thread_b = threading.Thread(target=_run_b)

        thread_a.start()
        assert load_started.wait(timeout=5)

        thread_b.start()
        continue_load.set()

        thread_a.join(timeout=10)
        thread_b.join(timeout=10)

        assert not errors
        assert "a" in results and "b" in results
        assert read_count == 1

    def test_parquet_condition_tree_lookup_avoids_dataframe_scan(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patched_query_conditions,
    ):
        """Condition-tree lookup should avoid DataFrame exact-match scan."""
        del patched_query_conditions

        monkeypatch.setenv("WINMLCLI_RULES_DIR", str(tmp_path))
        _write_parquet_rules(tmp_path)

        def _unexpected_scan(*args, **kwargs):
            del args, kwargs
            raise AssertionError(
                "query_table_exact_match should not be called for row_index lookup"
            )

        monkeypatch.setattr(
            runtime_checker_query_module,
            "query_table_exact_match",
            _unexpected_scan,
        )

        model = _build_add_model()
        node = model.graph.node[0]

        query_parquet = RuntimeCheckerQuery(model, "QNNExecutionProvider", "NPU")
        query_parquet.node_checkers = []
        result = query_parquet.run_for_node(node, for_debug=True, run_unknown_op=False)

        assert result.result.no_data is False
        assert result.result.compile is True
        assert result.result.run is False

    def test_parquet_legacy_table_with_row_index_still_matches(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patched_query_conditions,
    ):
        """Legacy tables with row_index should still match via condition columns."""
        del patched_query_conditions

        monkeypatch.setenv("WINMLCLI_RULES_DIR", str(tmp_path))
        _write_legacy_parquet_rules_with_row_index(tmp_path)

        model = _build_add_model()
        node = model.graph.node[0]

        query_parquet = RuntimeCheckerQuery(model, "QNNExecutionProvider", "NPU")
        query_parquet.node_checkers = []
        result = query_parquet.run_for_node(node, for_debug=True, run_unknown_op=False)

        assert result.result.no_data is False
        assert result.result.compile is True
        assert result.result.run is False
