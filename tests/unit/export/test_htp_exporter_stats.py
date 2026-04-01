# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for HTPExporter export statistics correctness."""

from __future__ import annotations

from unittest.mock import MagicMock

from winml.modelkit.export.htp.exporter import HTPExporter


class TestHTPExporterTaggedNodesStats:
    """Bug D: tagged_nodes and coverage must be 0 when embed_hierarchy_attributes=False."""

    def test_tagged_nodes_zero_when_hierarchy_disabled(self) -> None:
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

    def test_coverage_zero_when_hierarchy_disabled(self) -> None:
        exporter = HTPExporter(embed_hierarchy_attributes=False)
        exporter._node_tagger = MagicMock()
        exporter._node_tagger.tag_all_nodes.return_value = {
            "n1": "/t1",
            "n2": "/t2",
        }
        exporter._node_tagger.get_tagging_statistics.return_value = {}

        mock_model = MagicMock()
        mock_model.graph.node = [MagicMock() for _ in range(4)]

        exporter._apply_hierarchy_tags(mock_model)

        assert exporter._export_stats["coverage_percentage"] == 0.0

    def test_tagged_nodes_nonzero_when_hierarchy_enabled(self) -> None:
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
