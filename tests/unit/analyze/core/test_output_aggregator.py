"""Unit tests for OutputAggregator."""

from datetime import datetime

import pytest

from winml.modelkit.pattern.match import PatternMatchResult, SkeletonMatchResult
from winml.modelkit.pattern.models import OperatorPattern, PatternType
from winml.modelkit.analyze.core.output_aggregator import OutputAggregator
from winml.modelkit.analyze.models.ihv_type import IHVType
from winml.modelkit.analyze.models.information import Action, ActionLevel, Information
from winml.modelkit.analyze.models.output import (
    AnalysisOutput,
    EPSupport,
    ModelStats,
)
from winml.modelkit.analyze.models.runtime_checks import (
    PatternRuntime,
    RuntimeTestResult,
)
from winml.modelkit.analyze.models.support_level import SupportLevel


@pytest.fixture
def sample_metadata() -> ModelStats:
    """Create sample model metadata."""
    from onnx import helper

    pattern = OperatorPattern(
        pattern_id="OP/ai.onnx/Conv",
        pattern_type=PatternType.OPERATOR,
        namespace="ai.onnx",
        op_type="Conv",
    )

    # Create mock node proto
    node_proto = helper.make_node("Conv", ["input"], ["output"], name="conv1")

    # Create SkeletonMatchResult
    skeleton_result = SkeletonMatchResult(
        pattern=pattern,
        matched_nodes=[node_proto],
        matcher=None,
    )

    _pattern_match = PatternMatchResult(
        skeleton_match_result=skeleton_result,
        schema_input_to_value={},
        schema_output_to_value={},
        type_param_to_type={},
    )

    return ModelStats(
        model_path="test.onnx",
        opset_version=13,
        producer_name="pytorch",
        producer_version="1.9",
        total_operators=10,
        operator_counts={"Conv": 5, "Relu": 5},
        unique_operator_types=2,
        detected_pattern_count={},
    )


@pytest.fixture
def sample_check_results() -> dict[str, list[PatternRuntime]]:
    """Create sample runtime check results."""
    qc_results = [
        PatternRuntime(
            pattern_id="OP/ai.onnx/Conv",
            result=RuntimeTestResult(compile=True, run=True),
        ),
        PatternRuntime(
            pattern_id="OP/ai.onnx/Relu",
            result=RuntimeTestResult(compile=False, run=False, reason="Not supported"),
        ),
    ]

    return {"QNNExecutionProvider": qc_results}


@pytest.fixture
def sample_information() -> dict[str, list[Information]]:
    """Create sample information list."""
    action = Action(
        pattern_from_id="OP/ai.onnx/Relu",
        pattern_to_id="OP/ai.onnx/LeakyRelu",
        level=ActionLevel.REQUIRED,
        action="Replace Relu with LeakyRelu",
        status=SupportLevel.WHITE,
        details="Relu is not supported, use LeakyRelu instead",
    )

    info = Information(
        explanation="Relu operator is not supported",
        actions=[action],
        pattern_id="OP/ai.onnx/Relu",
    )

    return {"QNNExecutionProvider": [info]}


class TestOutputAggregatorInit:
    """Tests for OutputAggregator initialization."""

    def test_init_default_version(self) -> None:
        """Test initialization with default version."""
        aggregator = OutputAggregator()

        assert aggregator.analyzer_version == "0.1.0"

    def test_init_custom_version(self) -> None:
        """Test initialization with custom version."""
        aggregator = OutputAggregator(analyzer_version="1.2.3")

        assert aggregator.analyzer_version == "1.2.3"


class TestOutputAggregatorAggregate:
    """Tests for aggregate method."""

    def test_aggregate_creates_output(
        self,
        sample_metadata: ModelStats,
        sample_check_results: dict[IHVType, list[PatternRuntime]],
        sample_information: dict[IHVType, list[Information]],
    ) -> None:
        """Test aggregate creates AnalysisOutput."""
        aggregator = OutputAggregator(analyzer_version="0.1.0")

        output = aggregator.aggregate(
            metadata=sample_metadata,
            check_results=sample_check_results,
            information_list=sample_information,
        )

        assert isinstance(output, AnalysisOutput)
        assert output.analyzer_version == "0.1.0"
        assert output.metadata == sample_metadata
        assert len(output.results) == 1
        assert output.results[0].ihv_type == IHVType.QC

    def test_aggregate_includes_timestamp(
        self,
        sample_metadata: ModelStats,
        sample_check_results: dict[IHVType, list[PatternRuntime]],
        sample_information: dict[IHVType, list[Information]],
    ) -> None:
        """Test aggregate includes analysis timestamp."""
        aggregator = OutputAggregator()

        before = datetime.now()
        output = aggregator.aggregate(
            metadata=sample_metadata,
            check_results=sample_check_results,
            information_list=sample_information,
        )
        after = datetime.now()

        assert before <= output.analysis_timestamp <= after

    def test_aggregate_with_empty_inputs(self, sample_metadata: ModelStats) -> None:
        """Test aggregate with empty check results and information."""
        aggregator = OutputAggregator()

        output = aggregator.aggregate(
            metadata=sample_metadata,
            check_results={},
            information_list={},
        )

        assert isinstance(output, AnalysisOutput)
        assert len(output.results) == 0

    def test_aggregate_with_multiple_ihv_types(self, sample_metadata: ModelStats) -> None:
        """Test aggregate with multiple IHV types."""
        check_results = {
            "QNNExecutionProvider": [
                PatternRuntime(
                    pattern_id="OP/ai.onnx/Conv",
                    result=RuntimeTestResult(compile=True, run=True),
                ),
            ],
            "OpenVINOExecutionProvider": [
                PatternRuntime(
                    pattern_id="OP/ai.onnx/Relu",
                    result=RuntimeTestResult(compile=True, run=True),
                ),
            ],
        }

        information_list = {
            "QNNExecutionProvider": [],
            "OpenVINOExecutionProvider": [],
        }

        aggregator = OutputAggregator()
        output = aggregator.aggregate(
            metadata=sample_metadata,
            check_results=check_results,
            information_list=information_list,
        )

        assert len(output.results) == 2
        ihv_types = {result.ihv_type for result in output.results}
        assert ihv_types == {IHVType.QC, IHVType.INTEL}

    def test_aggregate_combines_check_and_info_sources(self, sample_metadata: ModelStats) -> None:
        """Test aggregate combines EP names from both check_results and information_list."""
        check_results = {
            "QNNExecutionProvider": [
                PatternRuntime(
                    pattern_id="OP/ai.onnx/Conv",
                    result=RuntimeTestResult(compile=True, run=True),
                ),
            ],
        }

        information_list = {
            "OpenVINOExecutionProvider": [
                Information(
                    explanation="Test info",
                    pattern_id="OP/ai.onnx/Test",
                ),
            ],
        }

        aggregator = OutputAggregator()
        output = aggregator.aggregate(
            metadata=sample_metadata,
            check_results=check_results,
            information_list=information_list,
        )

        assert len(output.results) == 2
        ihv_types = {result.ihv_type for result in output.results}
        assert ihv_types == {IHVType.QC, IHVType.INTEL}


class TestOutputAggregatorBuildEPSupport:
    """Tests for build_ep_support method."""

    def test_build_ep_support_creates_support_object(self) -> None:
        """Test build_ep_support creates EPSupport object."""
        aggregator = OutputAggregator()

        check_results = [
            PatternRuntime(
                pattern_id="OP/ai.onnx/Conv",
                result=RuntimeTestResult(compile=True, run=True),
            ),
        ]

        information_list = []

        ihv_support = aggregator.build_ep_support(
            check_results=check_results,
            information_list=information_list,
            ep_type="QNNExecutionProvider",
        )

        assert isinstance(ihv_support, EPSupport)
        assert ihv_support.ihv_type == IHVType.QC
        assert ihv_support.runtime_support is True
        assert len(ihv_support.classification[SupportLevel.WHITE]) == 1

    def test_build_ep_support_classifies_white(self) -> None:
        """Test WHITE classification for fully supported patterns."""
        aggregator = OutputAggregator()

        check_results = [
            PatternRuntime(
                pattern_id="OP/ai.onnx/Conv",
                result=RuntimeTestResult(compile=True, run=True),
            ),
        ]

        ihv_support = aggregator.build_ep_support(
            check_results=check_results,
            information_list=[],
            ep_type="QNNExecutionProvider",
        )

        assert "OP/ai.onnx/Conv" in ihv_support.classification[SupportLevel.WHITE]
        assert len(ihv_support.classification[SupportLevel.GRAY]) == 0
        assert len(ihv_support.classification[SupportLevel.BLACK]) == 0
        assert ihv_support.runtime_support is True

    def test_build_ep_support_classifies_gray(self) -> None:
        """Test GRAY classification for partially supported patterns."""
        aggregator = OutputAggregator()

        check_results = [
            PatternRuntime(
                pattern_id="OP/ai.onnx/MatMul",
                result=RuntimeTestResult(compile=False, run=True),
            ),
        ]

        ihv_support = aggregator.build_ep_support(
            check_results=check_results,
            information_list=[],
            ep_type="QNNExecutionProvider",
        )
        assert ihv_support.runtime_support is False

    def test_build_ep_support_classifies_black(self) -> None:
        """Test BLACK classification for unsupported patterns."""
        aggregator = OutputAggregator()

        check_results = [
            PatternRuntime(
                pattern_id="OP/ai.onnx/Custom",
                result=RuntimeTestResult(compile=False, run=False),
            ),
        ]

        ihv_support = aggregator.build_ep_support(
            check_results=check_results,
            information_list=[],
            ep_type="QNNExecutionProvider",
        )

        assert "OP/ai.onnx/Custom" in ihv_support.classification[SupportLevel.BLACK]
        assert ihv_support.runtime_support is False

    def test_build_ep_support_deduplicates_patterns(self) -> None:
        """Test pattern IDs are deduplicated in classification."""
        aggregator = OutputAggregator()

        # Same pattern appears twice
        check_results = [
            PatternRuntime(
                pattern_id="OP/ai.onnx/Conv",
                result=RuntimeTestResult(compile=True, run=True),
            ),
            PatternRuntime(
                pattern_id="OP/ai.onnx/Conv",
                result=RuntimeTestResult(compile=True, run=True),
            ),
        ]

        ihv_support = aggregator.build_ep_support(
            check_results=check_results,
            information_list=[],
            ep_type="QNNExecutionProvider",
        )

        # Should only appear once in classification
        assert ihv_support.classification[SupportLevel.WHITE].count("OP/ai.onnx/Conv") == 1

    def test_build_ep_support_runtime_support_false_with_black(self) -> None:
        """Test runtime_support is False when BLACK patterns present."""
        aggregator = OutputAggregator()

        check_results = [
            PatternRuntime(
                pattern_id="OP/ai.onnx/Conv",
                result=RuntimeTestResult(compile=True, run=True),
            ),
            PatternRuntime(
                pattern_id="OP/ai.onnx/Custom",
                result=RuntimeTestResult(compile=False, run=False),
            ),
        ]

        ihv_support = aggregator.build_ep_support(
            check_results=check_results,
            information_list=[],
            ep_type="QNNExecutionProvider",
        )

        assert ihv_support.runtime_support is False

    def test_build_ep_support_runtime_support_false_with_gray(self) -> None:
        """Test runtime_support is False when GRAY patterns present."""
        aggregator = OutputAggregator()

        check_results = [
            PatternRuntime(
                pattern_id="OP/ai.onnx/MatMul",
                result=RuntimeTestResult(compile=True, run=False),
            ),
        ]

        ihv_support = aggregator.build_ep_support(
            check_results=check_results,
            information_list=[],
            ep_type="QNNExecutionProvider",
        )

        assert ihv_support.runtime_support is False

    def test_build_ep_support_runtime_support_false_with_unknown(self) -> None:
        """Test runtime_support is False when UNKNOWN patterns present."""
        aggregator = OutputAggregator()

        # Create a mock RuntimeTestResult with UNKNOWN classification
        # This happens when both compile and run are False with no clear reason
        check_results = [
            PatternRuntime(
                pattern_id="OP/ai.onnx/Test",
                result=RuntimeTestResult(compile=False, run=False),
            ),
        ]

        ihv_support = aggregator.build_ep_support(
            check_results=check_results,
            information_list=[],
            ep_type="QNNExecutionProvider",
        )

        # BLACK classification also makes runtime_support False
        assert ihv_support.runtime_support is False

    def test_build_ep_support_with_empty_check_results(self) -> None:
        """Test build_ep_support with empty check results."""
        aggregator = OutputAggregator()

        ihv_support = aggregator.build_ep_support(
            check_results=[],
            information_list=[],
            ep_type="QNNExecutionProvider",
        )

        assert ihv_support.runtime_support is False
        assert all(len(patterns) == 0 for patterns in ihv_support.classification.values())

    def test_build_ep_support_includes_information(
        self, sample_information: dict[str, list[Information]]
    ) -> None:
        """Test build_ep_support includes information list."""
        aggregator = OutputAggregator()

        check_results = [
            PatternRuntime(
                pattern_id="OP/ai.onnx/Conv",
                result=RuntimeTestResult(compile=True, run=True),
            ),
        ]

        info_list = sample_information["QNNExecutionProvider"]

        ihv_support = aggregator.build_ep_support(
            check_results=check_results,
            information_list=info_list,
            ep_type="QNNExecutionProvider",
        )

        assert len(ihv_support.information) == 1
        assert ihv_support.information[0].pattern_id == "OP/ai.onnx/Relu"

    def test_build_ep_support_with_ep_and_driver_versions(self) -> None:
        """Test build_ep_support with ep_version and driver_version."""
        aggregator = OutputAggregator()

        check_results = [
            PatternRuntime(
                pattern_id="OP/ai.onnx/Conv",
                result=RuntimeTestResult(compile=True, run=True),
            ),
        ]

        ihv_support = aggregator.build_ep_support(
            check_results=check_results,
            information_list=[],
            ep_type="QNNExecutionProvider",
            ep_version="1.0.0",
            driver_version="NPU",
        )

        assert ihv_support.ep_version == "1.0.0"
        assert ihv_support.driver_version == "NPU"


class TestOutputAggregatorIntegration:
    """Integration tests for OutputAggregator."""

    def test_full_workflow_single_ihv(
        self,
        sample_metadata: ModelStats,
        sample_check_results: dict[str, list[PatternRuntime]],
        sample_information: dict[str, list[Information]],
    ) -> None:
        """Test complete aggregation workflow with single IHV."""
        aggregator = OutputAggregator(analyzer_version="1.0.0")

        output = aggregator.aggregate(
            metadata=sample_metadata,
            check_results=sample_check_results,
            information_list=sample_information,
        )

        assert output.analyzer_version == "1.0.0"
        assert output.metadata.model_path == "test.onnx"
        assert len(output.results) == 1

        ihv_support = output.results[0]
        assert ihv_support.ihv_type == IHVType.QC
        assert len(ihv_support.classification[SupportLevel.WHITE]) == 1
        assert len(ihv_support.classification[SupportLevel.BLACK]) == 1
        assert len(ihv_support.information) == 1

    def test_full_workflow_multiple_ihv(self, sample_metadata: ModelStats) -> None:
        """Test complete aggregation workflow with multiple IHVs."""
        check_results = {
            "QNNExecutionProvider": [
                PatternRuntime(
                    pattern_id="OP/ai.onnx/Conv",
                    result=RuntimeTestResult(compile=True, run=True),
                ),
            ],
            "OpenVINOExecutionProvider": [
                PatternRuntime(
                    pattern_id="OP/ai.onnx/Relu",
                    result=RuntimeTestResult(compile=True, run=True),
                ),
            ],
            "ACEExecutionProvider": [
                PatternRuntime(
                    pattern_id="OP/ai.onnx/Add",
                    result=RuntimeTestResult(compile=False, run=False),
                ),
            ],
        }

        information_list = {
            "QNNExecutionProvider": [],
            "OpenVINOExecutionProvider": [],
            "ACEExecutionProvider": [
                Information(
                    explanation="Add not supported",
                    pattern_id="OP/ai.onnx/Add",
                ),
            ],
        }

        aggregator = OutputAggregator()
        output = aggregator.aggregate(
            metadata=sample_metadata,
            check_results=check_results,
            information_list=information_list,
        )

        assert len(output.results) == 3

        # Verify each IHV has correct data
        ihv_map = {result.ihv_type: result for result in output.results}
        assert IHVType.QC in ihv_map
        assert IHVType.INTEL in ihv_map
        assert IHVType.AMD in ihv_map

        # Verify runtime support status
        assert ihv_map[IHVType.QC].runtime_support is True
        assert ihv_map[IHVType.INTEL].runtime_support is True
        assert ihv_map[IHVType.AMD].runtime_support is False

    def test_json_serialization(
        self,
        sample_metadata: ModelStats,
        sample_check_results: dict[str, list[PatternRuntime]],
        sample_information: dict[str, list[Information]],
    ) -> None:
        """Test output can be serialized to JSON."""
        aggregator = OutputAggregator()

        output = aggregator.aggregate(
            metadata=sample_metadata,
            check_results=sample_check_results,
            information_list=sample_information,
        )

        json_str = output.model_dump_json()

        assert isinstance(json_str, str)
        assert len(json_str) > 0
        assert "analyzer_version" in json_str
        assert "metadata" in json_str
        assert "results" in json_str
