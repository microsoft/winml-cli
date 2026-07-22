# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for HTPExporter export statistics correctness."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from onnx import TensorProto, helper, save

from winml.modelkit.export.htp import HTPExporter


if TYPE_CHECKING:
    from pathlib import Path


class TestHTPExporterTaggedNodesStats:
    """tagged_nodes, empty_tags, and coverage must be 0 when embed_hierarchy_attributes=False."""

    def test_all_stats_zero_when_hierarchy_disabled(self) -> None:
        exporter = HTPExporter(embed_hierarchy_attributes=False)
        exporter._node_tagger = MagicMock()
        exporter._node_tagger.tag_all_nodes.return_value = {
            "node1": "/Model/Layer1",
            "node2": "/Model/Layer2",
            "node3": "/Model/Layer3",
        }
        exporter._node_tagger.get_tagging_statistics.return_value = {}

        mock_model = MagicMock()
        mock_model.graph.node = [MagicMock() for _ in range(5)]

        exporter._apply_hierarchy_tags(mock_model)

        assert exporter._export_stats["tagged_nodes"] == 0
        assert exporter._export_stats["coverage_percentage"] == 0.0
        assert exporter._export_stats["empty_tags"] == 0

    def test_stats_populated_when_hierarchy_enabled(self) -> None:
        """Control: stats are populated normally when embedding is enabled."""
        exporter = HTPExporter(embed_hierarchy_attributes=True)
        exporter._node_tagger = MagicMock()
        exporter._node_tagger.tag_all_nodes.return_value = {
            "n1": "/t1",
            "n2": "/t2",
        }
        exporter._node_tagger.get_tagging_statistics.return_value = {}

        mock_model = MagicMock()
        mock_model.graph.node = [MagicMock() for _ in range(4)]

        exporter._apply_hierarchy_tags(mock_model)

        assert exporter._export_stats["tagged_nodes"] == 2
        assert exporter._export_stats["coverage_percentage"] == 50.0
        assert exporter._export_stats["empty_tags"] == 0


class TestHTPExporterReadDefaultOpset:
    """_read_default_opset reports the produced model's ai.onnx opset.

    The dynamo exporter may not honor a lower requested opset (torch's dynamo op
    set has a minimum of 18), so the exporter records the opset actually present
    in the file instead of echoing the requested value.
    """

    @staticmethod
    def _make_model(tmp_path: Path, opset_imports: list) -> str:
        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(graph, opset_imports=opset_imports)
        path = tmp_path / "m.onnx"
        save(model, str(path))
        return str(path)

    def test_reads_default_domain_opset(self, tmp_path: Path) -> None:
        path = self._make_model(tmp_path, [helper.make_opsetid("", 18)])
        assert HTPExporter._read_default_opset(path) == 18

    def test_reads_ai_onnx_domain_opset(self, tmp_path: Path) -> None:
        path = self._make_model(tmp_path, [helper.make_opsetid("ai.onnx", 17)])
        assert HTPExporter._read_default_opset(path) == 17

    def test_prefers_default_domain_over_custom(self, tmp_path: Path) -> None:
        path = self._make_model(
            tmp_path,
            [helper.make_opsetid("com.microsoft", 1), helper.make_opsetid("", 18)],
        )
        assert HTPExporter._read_default_opset(path) == 18

    def test_returns_none_without_default_domain(self, tmp_path: Path) -> None:
        path = self._make_model(tmp_path, [helper.make_opsetid("com.microsoft", 1)])
        assert HTPExporter._read_default_opset(path) is None

    def test_returns_none_for_unreadable_path(self, tmp_path: Path) -> None:
        assert HTPExporter._read_default_opset(str(tmp_path / "missing.onnx")) is None
