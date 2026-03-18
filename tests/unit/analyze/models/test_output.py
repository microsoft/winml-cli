# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Unit tests for AnalysisOutput Pydantic validation.

Tests verify:
- AnalysisOutput aggregation constraints
- ModelStats validation (total_operators must equal sum of operator_counts)
- EPSupport validation
- model_dump_json serialization
"""

import json

import pytest
from pydantic import ValidationError

from winml.modelkit.analyze.models.ihv_type import IHVType
from winml.modelkit.analyze.models.output import (
    AnalysisOutput,
    EPSupport,
    ModelStats,
)
from winml.modelkit.analyze.models.support_level import SupportLevel


class TestModelStatsValidation:
    """Test ModelStats validation rules."""

    def test_valid_model_metadata(self):
        """Test that valid ModelStats is accepted."""
        metadata = ModelStats(
            model_path="/path/to/model.onnx",
            opset_version=14,
            total_operators=10,
            operator_counts={"Conv": 5, "Relu": 5},
            unique_operator_types=2,
            detected_pattern_count={},
        )

        assert metadata.model_path == "/path/to/model.onnx"
        assert metadata.opset_version == 14
        assert metadata.total_operators == 10

    def test_producer_info_optional(self):
        """Test that producer_name and producer_version are optional."""
        metadata = ModelStats(
            model_path="/path/to/model.onnx",
            opset_version=14,
            total_operators=5,
            operator_counts={"Conv": 5},
            unique_operator_types=1,
            detected_pattern_count={},
        )
        assert metadata.producer_name is None
        assert metadata.producer_version is None

    def test_total_operators_equals_sum_of_counts(self):
        """Test that total_operators must equal sum of operator_counts."""
        # Valid: total matches sum
        metadata = ModelStats(
            model_path="/test.onnx",
            opset_version=13,
            total_operators=10,
            operator_counts={"Conv": 3, "Relu": 5, "MaxPool": 2},
            unique_operator_types=3,
            detected_pattern_count={},
        )
        assert metadata.total_operators == 10
        assert sum(metadata.operator_counts.values()) == 10

        # Invalid: total doesn't match sum
        with pytest.raises(
            ValidationError,
            match=r"total_operators .* must equal sum of operator_counts",
        ):
            ModelStats(
                model_path="/test.onnx",
                opset_version=13,
                total_operators=10,
                operator_counts={"Conv": 3, "Relu": 5, "MaxPool": 1},  # Sum is 9
                unique_operator_types=3,
                detected_pattern_count={},
            )

    def test_empty_operator_counts(self):
        """Test that empty operator_counts requires total_operators=0."""
        # Valid: empty counts, total=0
        metadata = ModelStats(
            model_path="/test.onnx",
            opset_version=13,
            total_operators=0,
            operator_counts={},
            unique_operator_types=0,
            detected_pattern_count={},
        )
        assert metadata.total_operators == 0
        assert len(metadata.operator_counts) == 0

        # Invalid: empty counts but total > 0
        with pytest.raises(
            ValidationError,
            match=r"total_operators .* must equal sum of operator_counts",
        ):
            ModelStats(
                model_path="/test.onnx",
                opset_version=13,
                total_operators=5,
                operator_counts={},
                unique_operator_types=0,
                detected_pattern_count={},
            )


class TestEPSupportValidation:
    """Test EPSupport validation rules."""

    def test_valid_ihv_support(self):
        """Test that valid EPSupport is accepted."""
        support = EPSupport(
            ihv_type=IHVType.QC,
            ep_type="QNNExecutionProvider",
            runtime_support=True,
            classification={
                SupportLevel.WHITE: ["Conv_0"],
                SupportLevel.GRAY: [],
                SupportLevel.BLACK: [],
            },
            has_errors=False,
                    has_warnings=False,
        )

        assert support.ihv_type == IHVType.QC
        assert support.runtime_support is True
        assert len(support.classification[SupportLevel.WHITE]) == 1

    def test_optional_version_fields(self):
        """Test that ep_version and driver_version are optional."""
        support = EPSupport(
            ihv_type=IHVType.INTEL,
            ep_type="OpenVINOExecutionProvider",
            runtime_support=True,
            classification={
                SupportLevel.WHITE: [],
                SupportLevel.GRAY: [],
                SupportLevel.BLACK: [],
            },
            has_errors=False,
                    has_warnings=False,
        )
        assert support.ep_version is None
        assert support.driver_version is None


class TestAnalysisOutputValidation:
    """Test AnalysisOutput validation rules."""

    def test_valid_analysis_output(self):
        """Test that valid AnalysisOutput is accepted."""
        output = AnalysisOutput(
            analyzer_version="1.0.0",
            metadata=ModelStats(
                model_path="/models/resnet50.onnx",
                opset_version=13,
                total_operators=50,
                operator_counts={"Conv": 20, "Relu": 20, "MaxPool": 5, "Gemm": 5},
                unique_operator_types=4,
                detected_pattern_count={},
            ),
            results=[
                EPSupport(
                    ihv_type=IHVType.QC,
                    ep_type="QNNExecutionProvider",
                    runtime_support=True,
                    classification={
                        SupportLevel.WHITE: [],
                        SupportLevel.GRAY: [],
                        SupportLevel.BLACK: [],
                    },
                    has_errors=False,
                    has_warnings=False,
                )
            ],
        )

        assert output.metadata.model_path == "/models/resnet50.onnx"
        assert len(output.results) == 1
        assert output.analyzer_version == "1.0.0"

    def test_unique_ihv_types_validation(self):
        """Test that IHV types must be unique in results."""
        # Valid: unique IHV types
        output = AnalysisOutput(
            analyzer_version="1.0.0",
            metadata=ModelStats(
                model_path="/test.onnx",
                opset_version=13,
                total_operators=1,
                operator_counts={"Conv": 1},
                unique_operator_types=1,
                detected_pattern_count={},
            ),
            results=[
                EPSupport(
                    ihv_type=IHVType.QC,
                    ep_type="QNNExecutionProvider",
                    runtime_support=True,
                    classification={
                        SupportLevel.WHITE: [],
                        SupportLevel.GRAY: [],
                        SupportLevel.BLACK: [],
                    },
                    has_errors=False,
                    has_warnings=False,
                ),
                EPSupport(
                    ihv_type=IHVType.INTEL,
                    ep_type="OpenVINOExecutionProvider",
                    runtime_support=True,
                    classification={
                        SupportLevel.WHITE: [],
                        SupportLevel.GRAY: [],
                        SupportLevel.BLACK: [],
                    },
                    has_errors=False,
                    has_warnings=False,
                ),
            ],
        )
        assert len(output.results) == 2

        # Invalid: duplicate IHV types
        with pytest.raises(ValidationError, match="Duplicate IHV types found"):
            AnalysisOutput(
                analyzer_version="1.0.0",
                metadata=ModelStats(
                    model_path="/test.onnx",
                    opset_version=13,
                    total_operators=1,
                    operator_counts={"Conv": 1},
                    unique_operator_types=1,
                    detected_pattern_count={},
                ),
                results=[
                    EPSupport(
                        ihv_type=IHVType.QC,
                        ep_type="QNNExecutionProvider",
                        runtime_support=True,
                        classification={
                            SupportLevel.WHITE: [],
                            SupportLevel.GRAY: [],
                            SupportLevel.BLACK: [],
                        },
                        has_errors=False,
                    has_warnings=False,
                    ),
                    EPSupport(
                        ihv_type=IHVType.QC,  # Duplicate
                        ep_type="QNNExecutionProvider",
                        runtime_support=True,
                        classification={
                            SupportLevel.WHITE: [],
                            SupportLevel.GRAY: [],
                            SupportLevel.BLACK: [],
                        },
                        has_errors=False,
                    has_warnings=False,
                    ),
                ],
            )

    def test_max_three_ihv_results(self):
        """Test that results list has max 3 items."""
        # Valid: 3 results
        output = AnalysisOutput(
            analyzer_version="1.0.0",
            metadata=ModelStats(
                model_path="/test.onnx",
                opset_version=13,
                total_operators=1,
                operator_counts={"Conv": 1},
                unique_operator_types=1,
                detected_pattern_count={},
            ),
            results=[
                EPSupport(
                    ihv_type=IHVType.QC,
                    ep_type="QNNExecutionProvider",
                    runtime_support=True,
                    classification={
                        SupportLevel.WHITE: [],
                        SupportLevel.GRAY: [],
                        SupportLevel.BLACK: [],
                    },
                    has_errors=False,
                    has_warnings=False,
                ),
                EPSupport(
                    ihv_type=IHVType.INTEL,
                    ep_type="OpenVINOExecutionProvider",
                    runtime_support=True,
                    classification={
                        SupportLevel.WHITE: [],
                        SupportLevel.GRAY: [],
                        SupportLevel.BLACK: [],
                    },
                    has_errors=False,
                    has_warnings=False,
                ),
                EPSupport(
                    ihv_type=IHVType.AMD,
                    ep_type="ACEExecutionProvider",
                    runtime_support=True,
                    classification={
                        SupportLevel.WHITE: [],
                        SupportLevel.GRAY: [],
                        SupportLevel.BLACK: [],
                    },
                    has_errors=False,
                    has_warnings=False,
                ),
            ],
        )
        assert len(output.results) == 3

    def test_model_dump_json_serialization(self):
        """Test that model_dump_json() produces valid JSON."""
        output = AnalysisOutput(
            analyzer_version="1.0.0",
            metadata=ModelStats(
                model_path="/models/test.onnx",
                opset_version=14,
                total_operators=3,
                operator_counts={"Conv": 1, "Relu": 1, "MaxPool": 1},
                unique_operator_types=3,
                detected_pattern_count={},
            ),
            results=[
                EPSupport(
                    ihv_type=IHVType.QC,
                    ep_type="QNNExecutionProvider",
                    runtime_support=True,
                    classification={
                        SupportLevel.WHITE: [],
                        SupportLevel.GRAY: [],
                        SupportLevel.BLACK: [],
                    },
                    has_errors=False,
                    has_warnings=False,
                )
            ],
        )

        json_output = output.model_dump_json(indent=2)

        # Validate that it's valid JSON
        parsed = json.loads(json_output)

        assert parsed["metadata"]["model_path"] == "/models/test.onnx"
        assert parsed["metadata"]["opset_version"] == 14
        assert parsed["metadata"]["total_operators"] == 3
        assert parsed["metadata"]["operator_counts"]["Conv"] == 1
        assert parsed["analyzer_version"] == "1.0.0"

    def test_comprehensive_output_with_all_fields(self):
        """Test AnalysisOutput with all fields populated."""
        from winml.modelkit.pattern.models import SubgraphPattern
        from winml.modelkit.analyze.models.information import Information

        # Create the subgraph pattern
        _subgraph_pattern = SubgraphPattern(
            pattern_id="SUBGRAPH/GELU",
            pattern_name="GELU",
            node_topology={"div": "Div", "erf": "Erf"},
            edge_topology=[("div", "erf")],
        )

        output = AnalysisOutput(
            analyzer_version="1.0.0",
            metadata=ModelStats(
                model_path="/models/comprehensive.onnx",
                opset_version=15,
                producer_name="test_producer",
                producer_version="1.0",
                total_operators=100,
                operator_counts={
                    "Conv": 30,
                    "Relu": 30,
                    "MaxPool": 10,
                    "Gemm": 20,
                    "Softmax": 10,
                },
                unique_operator_types=5,
                detected_pattern_count={"SUBGRAPH/GELU_Erf": 1},
            ),
            results=[
                EPSupport(
                    ihv_type=IHVType.QC,
                    ep_type="QNNExecutionProvider",
                    ep_version="1.0.0",
                    driver_version="2.0.0",
                    runtime_support=True,
                    has_errors=False,
                    has_warnings=False,
                    classification={
                        SupportLevel.WHITE: ["Conv_0"],
                        SupportLevel.GRAY: [],
                        SupportLevel.BLACK: [],
                    },
                    information=[
                        Information(
                            action=None,
                            explanation=("Replace GELU subgraph with optimized implementation"),
                            pattern_id="SUBGRAPH/GELU",
                        )
                    ],
                )
            ],
        )

        assert output.metadata.producer_name == "test_producer"
        assert output.metadata.total_operators == 100
        assert len(output.results) == 1
        assert len(output.results[0].information) == 1
        assert sum(output.metadata.detected_pattern_count.values()) == 1

        # Validate JSON serialization
        json_output = output.model_dump_json()
        parsed = json.loads(json_output)

        assert "metadata" in parsed
        assert "results" in parsed
        assert "analyzer_version" in parsed
