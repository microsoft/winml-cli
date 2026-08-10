# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for PatternExtractor."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import onnx
import pytest
from onnx import TensorProto, helper

from winml.modelkit.analyze import ModelStats, ONNXModel, PatternExtractor
from winml.modelkit.analyze.models.runtime_checks import RuntimeTestResult
from winml.modelkit.pattern import SubgraphPattern


@pytest.fixture
def simple_model_proto() -> onnx.ModelProto:
    """Create a simple ONNX model proto for testing."""
    input1 = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    # Create a simple graph with Conv and Relu
    conv_node = helper.make_node("Conv", ["input", "weight"], ["conv_out"], name="conv")
    relu_node = helper.make_node("Relu", ["conv_out"], ["output"], name="relu")

    graph_def = helper.make_graph([conv_node, relu_node], "test_graph", [input1], [output])
    return helper.make_model(
        graph_def, producer_name="test", opset_imports=[helper.make_opsetid("", 13)]
    )


@pytest.fixture
def simple_onnx_model(simple_model_proto: onnx.ModelProto) -> ONNXModel:
    """Create a simple ONNXModel for testing."""
    return ONNXModel.from_onnx_model(simple_model_proto, "test.onnx")


@pytest.fixture
def mock_subgraph_pattern() -> SubgraphPattern:
    """Create a mock SubgraphPattern for testing."""
    return SubgraphPattern(
        pattern_id="SUBGRAPH/TestPattern",
        pattern_name="TestPattern",
        operators=["Conv", "Relu"],
        node_topology={
            "conv_node": "Conv",
            "relu_node": "Relu",
        },
        edge_topology=[("conv_node", "relu_node")],
    )


class TestPatternExtractorInit:
    """Tests for PatternExtractor initialization."""

    def test_init_with_valid_model(self, simple_onnx_model: ONNXModel) -> None:
        """Test initialization with valid ONNXModel."""
        extractor = PatternExtractor(simple_onnx_model)
        assert extractor.model == simple_onnx_model

    def test_init_with_invalid_model_raises_type_error(self) -> None:
        """Test initialization with non-ONNXModel raises TypeError."""
        with pytest.raises(TypeError, match="Expected ONNXModel"):
            PatternExtractor("not_a_model")  # type: ignore[arg-type]

    def test_init_with_none_raises_type_error(self) -> None:
        """Test initialization with None raises TypeError."""
        with pytest.raises(TypeError, match="Expected ONNXModel"):
            PatternExtractor(None)  # type: ignore[arg-type]


class TestPatternExtractorModelProperty:
    """Tests for model property."""

    def test_model_property_returns_model(self, simple_onnx_model: ONNXModel) -> None:
        """Test model property returns the ONNXModel."""
        extractor = PatternExtractor(simple_onnx_model)
        assert extractor.model is simple_onnx_model


class TestPatternExtractorSummary:
    """Tests for summary method."""

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_summary_returns_dict_with_expected_keys(
        self, mock_config_cls: MagicMock, simple_onnx_model: ONNXModel
    ) -> None:
        """Test summary returns dict with 'summary' and 'subgraph_patterns' keys."""
        # Mock RuleLoader to return empty pattern list
        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = []
        mock_config_cls.return_value = mock_config

        extractor = PatternExtractor(simple_onnx_model)
        result = extractor.summary()

        assert isinstance(result, dict)
        assert "summary" in result
        assert "subgraph_patterns" in result

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_summary_metadata_is_model_metadata(
        self, mock_config_cls: MagicMock, simple_onnx_model: ONNXModel
    ) -> None:
        """Test summary 'summary' key contains ModelStats."""
        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = []
        mock_config_cls.return_value = mock_config

        extractor = PatternExtractor(simple_onnx_model)
        result = extractor.summary()

        assert isinstance(result["summary"], ModelStats)
        assert result["summary"].model_path == "test.onnx"

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_summary_subgraph_patterns_is_list(
        self, mock_config_cls: MagicMock, simple_onnx_model: ONNXModel
    ) -> None:
        """Test summary 'subgraph_patterns' key contains list."""
        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = []
        mock_config_cls.return_value = mock_config

        extractor = PatternExtractor(simple_onnx_model)
        result = extractor.summary()

        assert isinstance(result["subgraph_patterns"], list)

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_summary_includes_detected_pattern_count(
        self, mock_config_cls: MagicMock, simple_onnx_model: ONNXModel
    ) -> None:
        """Test summary metadata includes correct detected_pattern_count."""
        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = []
        mock_config_cls.return_value = mock_config

        extractor = PatternExtractor(simple_onnx_model)
        result = extractor.summary(ep="QNNExecutionProvider")

        # Since no patterns are matched, the selected EP has an empty count mapping.
        assert result["summary"].detected_pattern_count == {
            "QNNExecutionProvider": {}
        }

    def test_summary_reports_pattern_check_supported_with_local_fallback(
        self,
        simple_onnx_model: ONNXModel,
    ) -> None:
        """Local probing keeps pattern progress active without parquet rules."""
        extractor = PatternExtractor(simple_onnx_model)
        on_pattern_query_start = MagicMock()

        with patch.object(
            PatternExtractor,
            "_is_valid_parquet_lookup_target",
            return_value=False,
        ):
            result = extractor.summary(
                ep="DmlExecutionProvider",
                device="gpu",
                on_pattern_query_start=on_pattern_query_start,
                local_pattern_checker=MagicMock(),
            )

        assert result["parquet_lookup_supported"] is False
        on_pattern_query_start.assert_called_once_with({}, True)


class TestPatternExtractorModelSummary:
    """Tests for model_summary method."""

    def test_model_summary_returns_metadata(self, simple_onnx_model: ONNXModel) -> None:
        """Test model_summary returns ModelStats."""
        extractor = PatternExtractor(simple_onnx_model)
        metadata = extractor.model_summary()

        assert isinstance(metadata, ModelStats)
        assert metadata.model_path == "test.onnx"
        assert metadata.opset_version == 13

    def test_model_summary_with_pattern_count(self, simple_onnx_model: ONNXModel) -> None:
        """Test model_summary includes detected_pattern_count."""
        extractor = PatternExtractor(simple_onnx_model)
        pattern_count_dict = {
            "QNNExecutionProvider": {"SUBGRAPH/GELU_Erf": 5}
        }
        metadata = extractor.model_summary(detected_pattern_count=pattern_count_dict)

        assert metadata.detected_pattern_count == pattern_count_dict

    def test_model_summary_default_pattern_count_is_zero(
        self, simple_onnx_model: ONNXModel
    ) -> None:
        """Test model_summary default detected_pattern_count is empty dict."""
        extractor = PatternExtractor(simple_onnx_model)
        metadata = extractor.model_summary()

        assert metadata.detected_pattern_count == {}

    def test_model_summary_includes_operator_counts(self, simple_onnx_model: ONNXModel) -> None:
        """Test model_summary includes operator statistics."""
        extractor = PatternExtractor(simple_onnx_model)
        metadata = extractor.model_summary()

        assert metadata.total_operators == 2
        assert metadata.unique_operator_types == 2
        assert "Conv" in metadata.operator_counts
        assert "Relu" in metadata.operator_counts


class TestPatternExtractorAlternativeSelection:
    """Tests for merge-prep alternative sorting/filtering helpers."""

    @staticmethod
    def _make_candidate(
        *,
        pattern_id: str,
        pattern_class: str,
        is_alternative: bool,
        status: str,
        compile_ok: bool | None,
        run_ok: bool | None,
    ) -> dict[str, object]:
        return {
            "pattern_class": pattern_class,
            "pattern_id": pattern_id,
            "is_alternative": is_alternative,
            "status": status,
            "mismatch_error": None,
            "compile": compile_ok,
            "run": run_ok,
            "row_count": 1,
            "table_file": "dummy.parquet",
            "table_path": "dummy.parquet",
            "domain": "ai.onnx",
            "opset_version": 17,
            "compile_true_rows": int(bool(compile_ok)),
            "run_true_rows": int(bool(run_ok)),
            "case_indices": None,
            "query_condition_count": 0,
            "query_condition_keys": [],
            "debug_details": None,
        }

    def test_select_and_filter_prefers_status_before_priority(
        self,
        simple_onnx_model: ONNXModel,
    ) -> None:
        """supported status wins even when its priority value is larger."""
        extractor = PatternExtractor(simple_onnx_model)

        alternatives_meta = [
            {"pattern_to_id": "SUBGRAPH/AltA", "pattern_class": "AltA", "priority": 1},
            {"pattern_to_id": "SUBGRAPH/AltB", "pattern_class": "AltB", "priority": 2},
        ]
        candidate_results = [
            self._make_candidate(
                pattern_id="SUBGRAPH/Base",
                pattern_class="BasePattern",
                is_alternative=False,
                status="ok",
                compile_ok=True,
                run_ok=True,
            ),
            self._make_candidate(
                pattern_id="SUBGRAPH/AltA",
                pattern_class="AltA",
                is_alternative=True,
                status="ok",
                compile_ok=False,
                run_ok=True,
            ),
            self._make_candidate(
                pattern_id="SUBGRAPH/AltB",
                pattern_class="AltB",
                is_alternative=True,
                status="ok",
                compile_ok=True,
                run_ok=True,
            ),
        ]

        selected_alternatives, filtered_candidates = extractor._select_and_filter_alternatives(
            alternatives_meta=alternatives_meta,
            candidate_results=candidate_results,  # type: ignore[arg-type]
        )

        assert len(selected_alternatives) == 1
        assert selected_alternatives[0]["pattern_to_id"] == "SUBGRAPH/AltB"

        alternative_candidates = [
            candidate for candidate in filtered_candidates if candidate["is_alternative"]
        ]
        assert len(filtered_candidates) == 2
        assert len(alternative_candidates) == 1
        assert alternative_candidates[0]["pattern_id"] == "SUBGRAPH/AltB"

    def test_select_and_filter_uses_priority_as_tiebreaker(
        self,
        simple_onnx_model: ONNXModel,
    ) -> None:
        """When statuses tie, lower priority value is selected."""
        extractor = PatternExtractor(simple_onnx_model)

        alternatives_meta = [
            {"pattern_to_id": "SUBGRAPH/AltA", "pattern_class": "AltA", "priority": 2},
            {"pattern_to_id": "SUBGRAPH/AltB", "pattern_class": "AltB", "priority": 1},
        ]
        candidate_results = [
            self._make_candidate(
                pattern_id="SUBGRAPH/Base",
                pattern_class="BasePattern",
                is_alternative=False,
                status="ok",
                compile_ok=True,
                run_ok=True,
            ),
            self._make_candidate(
                pattern_id="SUBGRAPH/AltA",
                pattern_class="AltA",
                is_alternative=True,
                status="ok",
                compile_ok=True,
                run_ok=True,
            ),
            self._make_candidate(
                pattern_id="SUBGRAPH/AltB",
                pattern_class="AltB",
                is_alternative=True,
                status="ok",
                compile_ok=True,
                run_ok=True,
            ),
        ]

        selected_alternatives, filtered_candidates = extractor._select_and_filter_alternatives(
            alternatives_meta=alternatives_meta,
            candidate_results=candidate_results,  # type: ignore[arg-type]
        )

        assert len(selected_alternatives) == 1
        assert selected_alternatives[0]["pattern_to_id"] == "SUBGRAPH/AltB"
        assert len(filtered_candidates) == 2
        assert filtered_candidates[1]["pattern_id"] == "SUBGRAPH/AltB"

    def test_select_and_filter_drops_selected_unsupported_alternative(
        self,
        simple_onnx_model: ONNXModel,
    ) -> None:
        """If the selected best-ranked alternative is unsupported, remove alternatives."""
        extractor = PatternExtractor(simple_onnx_model)

        alternatives_meta = [
            {
                "pattern_to_id": "SUBGRAPH/AltUnsupported",
                "pattern_class": "AltUnsupported",
                "priority": 1,
            },
            {"pattern_to_id": "SUBGRAPH/AltUnknown", "pattern_class": "AltUnknown", "priority": 1},
        ]
        candidate_results = [
            self._make_candidate(
                pattern_id="SUBGRAPH/Base",
                pattern_class="BasePattern",
                is_alternative=False,
                status="ok",
                compile_ok=True,
                run_ok=True,
            ),
            self._make_candidate(
                pattern_id="SUBGRAPH/AltUnsupported",
                pattern_class="AltUnsupported",
                is_alternative=True,
                status="ok",
                compile_ok=False,
                run_ok=False,
            ),
            self._make_candidate(
                pattern_id="SUBGRAPH/AltUnknown",
                pattern_class="AltUnknown",
                is_alternative=True,
                status="table_not_found",
                compile_ok=None,
                run_ok=None,
            ),
        ]

        selected_alternatives, filtered_candidates = extractor._select_and_filter_alternatives(
            alternatives_meta=alternatives_meta,
            candidate_results=candidate_results,  # type: ignore[arg-type]
        )

        assert selected_alternatives == []
        assert len(filtered_candidates) == 1
        assert filtered_candidates[0]["is_alternative"] is False

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_merge_prep_locally_checks_only_base_pattern_when_table_is_unavailable(
        self,
        mock_config_cls: MagicMock,
        simple_onnx_model: ONNXModel,
    ) -> None:
        PatternExtractor._MERGE_PREP_CACHE.clear()
        extractor = PatternExtractor(simple_onnx_model)

        pattern_obj = MagicMock()
        pattern_obj.pattern_id = "SUBGRAPH/Base"
        pattern_match = MagicMock()
        pattern_match.pattern = pattern_obj
        pattern_match.match_id = "match_1"
        pattern_match.matched_node_keys = ["node_a", "node_b"]

        mock_config = MagicMock()
        mock_config.get_alternatives.return_value = [
            SimpleNamespace(
                pattern_to_id="SUBGRAPH/Alt",
                pattern_class="AltPattern",
                priority=1,
                enabled=True,
                module=None,
                action_items=None,
                details=None,
                reason=None,
            )
        ]
        mock_config_cls.return_value = mock_config
        local_checker = MagicMock(
            return_value=RuntimeTestResult(compile=True, run=True)
        )

        with (
            patch.object(
                PatternExtractor,
                "_is_valid_parquet_lookup_target",
                return_value=False,
            ),
            patch.object(
                PatternExtractor,
                "_probe_candidate_pattern_mismatch",
                return_value=(False, None),
            ),
            patch.object(
                PatternExtractor,
                "_domain_and_target_opset_for_pattern",
                return_value=("ai.onnx", 13),
            ),
        ):
            entries = extractor._build_merge_prep_metadata(
                subgraph_patterns_by_source={
                    "default": {"BasePattern": [pattern_match]}
                },
                model_signature="sig_local",
                ep="QNNExecutionProvider",
                device="NPU",
                for_debug=True,
                local_pattern_checker=local_checker,
            )

        local_checker.assert_called_once_with(pattern_match, "table_not_found", True)
        assert entries[0]["support_status"] == "supported"
        assert entries[0]["candidates"][0]["status"] == "local_ep_check"
        assert entries[0]["candidates"][0]["compile"] is True
        assert entries[0]["candidates"][0]["run"] is True
        assert entries[0]["candidates"][1]["is_alternative"] is True
        assert entries[0]["candidates"][1]["status"] == "table_not_found"
        PatternExtractor._MERGE_PREP_CACHE.clear()

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_merge_prep_uses_cache_after_first_build(
        self,
        mock_config_cls: MagicMock,
        simple_onnx_model: ONNXModel,
    ) -> None:
        """Second call with same cache key should reuse cached merge-prep entries."""
        PatternExtractor._MERGE_PREP_CACHE.clear()
        extractor = PatternExtractor(simple_onnx_model)

        pattern_obj = MagicMock()
        pattern_obj.pattern_id = "SUBGRAPH/Base"

        pattern_match = MagicMock()
        pattern_match.pattern = pattern_obj
        pattern_match.match_id = "match_1"
        pattern_match.matched_node_keys = ["node_a", "node_b"]
        pattern_match.input_infos = {}
        pattern_match.attributes = {}

        subgraph_patterns_by_source = {
            "default": {
                "BasePattern": [pattern_match],
            }
        }

        mock_config = MagicMock()
        mock_config.get_alternatives.return_value = [
            SimpleNamespace(
                pattern_to_id="SUBGRAPH/Alt",
                pattern_class="AltPattern",
                priority=1,
                enabled=True,
                module=None,
                action_items=None,
                details=None,
                reason=None,
            )
        ]
        mock_config_cls.return_value = mock_config

        with (
            patch.object(
                PatternExtractor,
                "_is_valid_parquet_lookup_target",
                return_value=True,
            ),
            patch.object(
                PatternExtractor,
                "_probe_candidate_pattern_mismatch",
                return_value=(False, None),
            ),
            patch.object(
                PatternExtractor,
                "_domain_and_target_opset_for_pattern",
                return_value=("ai.onnx", 13),
            ),
            patch.object(
                PatternExtractor,
                "_resolve_pattern_rule_table",
                return_value=(Path("dummy.parquet"), "ai.onnx", 13),
            ),
            patch.object(
                PatternExtractor,
                "_query_pattern_rule_compile_run_for_match",
                return_value=("ok", True, True, 1, 1, 1, None, 0, [], None),
            ) as mock_query,
        ):
            first = extractor._build_merge_prep_metadata(
                subgraph_patterns_by_source=subgraph_patterns_by_source,
                model_signature="sig_1",
                ep="QNNExecutionProvider",
                device="NPU",
                for_debug=True,
            )
            assert mock_query.call_count == 2

            second = extractor._build_merge_prep_metadata(
                subgraph_patterns_by_source=subgraph_patterns_by_source,
                model_signature="sig_1",
                ep="QNNExecutionProvider",
                device="NPU",
                for_debug=True,
            )
            assert mock_query.call_count == 2

        assert first == second
        assert first is not second
        PatternExtractor._MERGE_PREP_CACHE.clear()

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_merge_prep_stops_after_first_supported_alternative_by_priority(
        self,
        mock_config_cls: MagicMock,
        simple_onnx_model: ONNXModel,
    ) -> None:
        """Alternative probing stops once the first priority-ordered supported option is found."""
        PatternExtractor._MERGE_PREP_CACHE.clear()
        extractor = PatternExtractor(simple_onnx_model)

        pattern_obj = MagicMock()
        pattern_obj.pattern_id = "SUBGRAPH/Base"

        pattern_match = MagicMock()
        pattern_match.pattern = pattern_obj
        pattern_match.match_id = "match_short_circuit"
        pattern_match.matched_node_keys = ["node_short_a", "node_short_b"]
        pattern_match.input_infos = {}
        pattern_match.attributes = {}

        subgraph_patterns_by_source = {
            "default": {
                "BasePattern": [pattern_match],
            }
        }

        mock_config = MagicMock()
        mock_config.get_alternatives.return_value = [
            SimpleNamespace(
                pattern_to_id="SUBGRAPH/AltLowPriority",
                pattern_class="AltLowPriority",
                priority=2,
                enabled=True,
                module=None,
                action_items=None,
                details=None,
                reason=None,
            ),
            SimpleNamespace(
                pattern_to_id="SUBGRAPH/AltHighPriority",
                pattern_class="AltHighPriority",
                priority=1,
                enabled=True,
                module=None,
                action_items=[
                    {
                        "type": "GraphOptimization",
                        "optimization_options": {"matmul_add_fusion": True},
                    }
                ],
                details="Use the selected graph optimization.",
                reason="The selected alternative is supported.",
            ),
        ]
        mock_config_cls.return_value = mock_config

        def query_side_effect(
            **kwargs: object,
        ) -> tuple[
            str,
            bool | None,
            bool | None,
            int,
            int,
            int,
            list[object] | None,
            int,
            list[str],
            dict[str, object] | None,
        ]:
            candidate_name = str(kwargs["candidate_pattern_name"])
            if candidate_name == "AltHighPriority":
                return ("ok", True, True, 1, 1, 1, None, 0, [], None)
            return ("ok", False, False, 1, 0, 0, None, 0, [], None)

        with (
            patch.object(
                PatternExtractor,
                "_is_valid_parquet_lookup_target",
                return_value=True,
            ),
            patch.object(
                PatternExtractor,
                "_probe_candidate_pattern_mismatch",
                return_value=(False, None),
            ),
            patch.object(
                PatternExtractor,
                "_domain_and_target_opset_for_pattern",
                return_value=("ai.onnx", 13),
            ),
            patch.object(
                PatternExtractor,
                "_resolve_pattern_rule_table",
                return_value=(Path("dummy.parquet"), "ai.onnx", 13),
            ),
            patch.object(
                PatternExtractor,
                "_query_pattern_rule_compile_run_for_match",
                side_effect=query_side_effect,
            ) as mock_query,
        ):
            entries = extractor._build_merge_prep_metadata(
                subgraph_patterns_by_source=subgraph_patterns_by_source,
                model_signature="sig_short_circuit",
                ep="QNNExecutionProvider",
                device="NPU",
                for_debug=True,
            )

        queried_candidates = [
            str(call.kwargs["candidate_pattern_name"]) for call in mock_query.call_args_list
        ]
        assert queried_candidates == ["MagicMock", "AltHighPriority"]
        assert entries[0]["alternatives"][0]["pattern_to_id"] == "SUBGRAPH/AltHighPriority"
        assert entries[0]["alternatives"][0]["action_items"] == [
            {
                "type": "GraphOptimization",
                "optimization_options": {"matmul_add_fusion": True},
            }
        ]
        assert entries[0]["alternatives"][0]["details"] == (
            "Use the selected graph optimization."
        )
        PatternExtractor._MERGE_PREP_CACHE.clear()


class TestPatternExtractorEPDedup:
    """Tests for EP-priority dedup and EP-scoped dedup cache."""

    @staticmethod
    def _make_match(node_keys: list[str]) -> MagicMock:
        match = MagicMock()
        match.matched_node_keys = node_keys
        return match

    def test_ep_priority_dedup_prefers_ep_source_over_default(
        self,
        simple_onnx_model: ONNXModel,
    ) -> None:
        """EP source should be traversed first and win on overlapping node keys."""
        PatternExtractor._DEDUPED_MATCH_CACHE.clear()
        extractor = PatternExtractor(simple_onnx_model)

        ep_match = self._make_match(["shared_node", "ep_only_node"])
        default_overlap = self._make_match(["shared_node", "default_only_node"])
        default_unique = self._make_match(["default_unique_node"])

        grouped = {
            "default": {"DemoPattern": [default_overlap, default_unique]},
            "qnn": {"DemoPattern": [ep_match]},
        }

        deduped_grouped, deduped_flat = extractor._dedup_grouped_matches_for_ep(
            subgraph_patterns_by_source=grouped,  # type: ignore[arg-type]
            sources=["default", "qnn"],
            model_signature="sig_ep_priority",
            ep="QNNExecutionProvider",
        )

        assert deduped_flat == [ep_match, default_unique]
        assert deduped_grouped["qnn"]["DemoPattern"] == [ep_match]
        assert deduped_grouped["default"]["DemoPattern"] == [default_unique]
        PatternExtractor._DEDUPED_MATCH_CACHE.clear()

    def test_ep_dedup_cache_reused_for_same_ep(
        self,
        simple_onnx_model: ONNXModel,
    ) -> None:
        """Dedup cache key should ignore device and reuse by model+EP."""
        PatternExtractor._DEDUPED_MATCH_CACHE.clear()
        extractor = PatternExtractor(simple_onnx_model)

        first_match = self._make_match(["node_a"])
        grouped_first = {
            "default": {"DemoPattern": [first_match]},
            "qnn": {"DemoPattern": []},
        }

        first_grouped, first_flat = extractor._dedup_grouped_matches_for_ep(
            subgraph_patterns_by_source=grouped_first,  # type: ignore[arg-type]
            sources=["default", "qnn"],
            model_signature="sig_ep_cache",
            ep="QNNExecutionProvider",
        )

        second_match = self._make_match(["node_b"])
        grouped_second = {
            "default": {"DemoPattern": [second_match]},
            "qnn": {"DemoPattern": []},
        }

        second_grouped, second_flat = extractor._dedup_grouped_matches_for_ep(
            subgraph_patterns_by_source=grouped_second,  # type: ignore[arg-type]
            sources=["default", "qnn"],
            model_signature="sig_ep_cache",
            ep="QNNExecutionProvider",
        )

        assert first_flat == [first_match]
        assert second_flat == [first_match]
        assert first_grouped == second_grouped
        PatternExtractor._DEDUPED_MATCH_CACHE.clear()


class TestPatternExtractorIntegration:
    """Integration tests for PatternExtractor."""

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_full_workflow(
        self,
        mock_config_cls: MagicMock,
        simple_onnx_model: ONNXModel,
        mock_subgraph_pattern: SubgraphPattern,
    ) -> None:
        """Test complete workflow from initialization to summary."""
        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = [mock_subgraph_pattern]
        mock_config_cls.return_value = mock_config

        # Initialize extractor
        extractor = PatternExtractor(simple_onnx_model)

        # Generate summary
        result = extractor.summary()

        # Verify result structure
        assert "summary" in result
        assert "subgraph_patterns" in result
        assert isinstance(result["summary"], ModelStats)
        assert isinstance(result["subgraph_patterns"], list)

        # Verify metadata
        assert result["summary"].model_path == "test.onnx"
        assert result["summary"].total_operators == 2

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_workflow_with_multiple_patterns(
        self, mock_config_cls: MagicMock, simple_onnx_model: ONNXModel
    ) -> None:
        """Test workflow with multiple pattern definitions."""
        pattern1 = SubgraphPattern(
            pattern_id="SUBGRAPH/Pattern1",
            pattern_name="Pattern1",
            operators=["Conv"],
            node_topology={"conv": "Conv"},
            edge_topology=[],
        )
        pattern2 = SubgraphPattern(
            pattern_id="SUBGRAPH/Pattern2",
            pattern_name="Pattern2",
            operators=["Relu"],
            node_topology={"relu": "Relu"},
            edge_topology=[],
        )

        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = [pattern1, pattern2]
        mock_config_cls.return_value = mock_config

        extractor = PatternExtractor(simple_onnx_model)

        # Summary should run end-to-end with multiple pattern definitions
        result = extractor.summary()
        assert isinstance(result["subgraph_patterns"], list)
