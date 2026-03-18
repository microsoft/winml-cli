"""Unit tests for InformationEngine."""

import onnx
import pytest

from winml.modelkit.analyze.core.information_engine import InformationEngine
from winml.modelkit.analyze.models.information import ActionLevel, Information
from winml.modelkit.analyze.models.onnx_model import ONNXModel
from winml.modelkit.analyze.models.runtime_checks import (
    AlternativeType,
    PatternAlternative,
    PatternRuntime,
    RuntimeTestResult,
)
from winml.modelkit.analyze.models.support_level import SupportLevel


@pytest.fixture
def simple_model() -> ONNXModel:
    """Create a simple ONNX model for testing."""
    # Create a minimal ONNX model with one input and one output
    input_tensor = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
    )
    output_tensor = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 1000])

    # Create a simple node (e.g., Identity)
    node = onnx.helper.make_node("Identity", ["input"], ["output"], name="identity_node")

    # Create the graph
    graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])

    # Create the model
    model_proto = onnx.helper.make_model(graph, producer_name="test")
    model_proto.opset_import[0].version = 13  # Use version 13 to pass validation

    return ONNXModel.from_onnx_model(model_proto, "test_model.onnx")


@pytest.fixture
def test_device() -> str:
    """Return test device string."""
    return "NPU"


@pytest.fixture
def white_runtime_result() -> PatternRuntime:
    """Create WHITE (fully supported) runtime result."""
    return PatternRuntime(
        pattern_id="OP/ai.onnx/Conv",
        result=RuntimeTestResult(compile=True, run=True),
        alternatives=[],
    )


@pytest.fixture
def gray_runtime_result() -> PatternRuntime:
    """Create GRAY (partial support) runtime result."""
    return PatternRuntime(
        pattern_id="OP/ai.onnx/MatMul",
        result=RuntimeTestResult(compile=False, run=True, reason="Performance issues"),
        alternatives=[],
    )


@pytest.fixture
def black_runtime_result() -> PatternRuntime:
    """Create BLACK (not supported) runtime result."""
    return PatternRuntime(
        pattern_id="OP/ai.onnx/Custom",
        result=RuntimeTestResult(compile=False, run=False, reason="Unsupported op"),
        alternatives=[],
    )


@pytest.fixture
def black_with_white_alternative() -> PatternRuntime:
    """Create BLACK runtime result with WHITE alternative."""
    alternative = PatternAlternative(
        pattern_id="OP/ai.onnx/Conv",
        result=RuntimeTestResult(compile=True, run=True),
        alternative_type=AlternativeType.EQUIVALENT,
    )
    return PatternRuntime(
        pattern_id="SUBGRAPH/old_pattern",
        result=RuntimeTestResult(compile=False, run=False, reason="Not supported"),
        alternatives=[alternative],
    )


@pytest.fixture
def black_with_gray_alternative() -> PatternRuntime:
    """Create BLACK runtime result with GRAY alternative."""
    alternative = PatternAlternative(
        pattern_id="OP/ai.onnx/Alt",
        result=RuntimeTestResult(compile=False, run=True),  # GRAY: compile=False, run=True
        alternative_type=AlternativeType.APPROXIMATION,
    )
    return PatternRuntime(
        pattern_id="SUBGRAPH/unsupported",
        result=RuntimeTestResult(compile=False, run=False),
        alternatives=[alternative],
    )


@pytest.fixture
def gray_with_white_alternative() -> PatternRuntime:
    """Create GRAY runtime result with WHITE alternative."""
    alternative = PatternAlternative(
        pattern_id="SUBGRAPH/optimized",
        result=RuntimeTestResult(compile=True, run=True),
        alternative_type=AlternativeType.EQUIVALENT,
    )
    return PatternRuntime(
        pattern_id="SUBGRAPH/unoptimized",
        result=RuntimeTestResult(compile=False, run=True),  # GRAY: compile=False, run=True
        alternatives=[alternative],
    )


class TestInformationEngineInit:
    """Tests for InformationEngine initialization."""

    def test_init_with_op_results(
        self, white_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test initialization with operator results."""
        engine = InformationEngine(
            op_runtime_results=[white_runtime_result],
            subgraph_runtime_results=[],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        assert len(engine.op_runtime_results) == 1
        assert len(engine.subgraph_runtime_results) == 0
        assert engine._ep == "QNNExecutionProvider"

    def test_init_with_subgraph_results(
        self, black_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test initialization with subgraph results."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[black_runtime_result],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        assert len(engine.op_runtime_results) == 0
        assert len(engine.subgraph_runtime_results) == 1

    def test_init_with_both_results(
        self,
        white_runtime_result: PatternRuntime,
        black_runtime_result: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test initialization with both operator and subgraph results."""
        engine = InformationEngine(
            op_runtime_results=[white_runtime_result],
            subgraph_runtime_results=[black_runtime_result],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        assert len(engine.op_runtime_results) == 1
        assert len(engine.subgraph_runtime_results) == 1

    def test_init_with_empty_results_raises_error(
        self, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test initialization with empty results and no model raises ValueError."""
        with pytest.raises(
            ValueError,
            match="At least one of op_runtime_results, subgraph_runtime_results, or model must be provided",
        ):
            InformationEngine(
                op_runtime_results=[],
                subgraph_runtime_results=[],
                ep="QNNExecutionProvider",
                model=None,
                device=test_device,
            )

    def test_init_with_empty_results_and_model_succeeds(
        self, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test initialization with empty results but valid model succeeds."""
        # Should not raise - model validators can still run
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        assert len(engine.op_runtime_results) == 0
        assert len(engine.subgraph_runtime_results) == 0


class TestInformationEngineProperties:
    """Tests for InformationEngine properties."""

    def test_op_runtime_results_property(
        self, white_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test op_runtime_results property."""
        engine = InformationEngine(
            op_runtime_results=[white_runtime_result],
            subgraph_runtime_results=[],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        results = engine.op_runtime_results
        assert len(results) == 1
        assert results[0] == white_runtime_result

    def test_subgraph_runtime_results_property(
        self, black_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test subgraph_runtime_results property."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[black_runtime_result],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        results = engine.subgraph_runtime_results
        assert len(results) == 1
        assert results[0] == black_runtime_result


class TestInformationEngineSummary:
    """Tests for summary method."""

    def test_summary_returns_information_list(
        self,
        white_runtime_result: PatternRuntime,
        black_runtime_result: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test summary returns list of Information objects."""
        engine = InformationEngine(
            op_runtime_results=[white_runtime_result],
            subgraph_runtime_results=[black_runtime_result],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        result = engine.summary()

        assert isinstance(result, list)
        assert all(isinstance(info, Information) for info in result)

    def test_summary_combines_op_and_pattern_info(
        self,
        black_runtime_result: PatternRuntime,
        black_with_white_alternative: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test summary combines operator and pattern information."""
        engine = InformationEngine(
            op_runtime_results=[black_runtime_result],
            subgraph_runtime_results=[black_with_white_alternative],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        result = engine.summary()

        # Should have information for both black operator and black pattern with alternative
        assert len(result) >= 2


class TestInformationEngineCheckSingleOps:
    """Tests for _check_single_ops method."""

    def test_white_pattern_no_action(
        self, white_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test WHITE pattern generates no information."""
        engine = InformationEngine(
            op_runtime_results=[white_runtime_result],
            subgraph_runtime_results=[],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info_list = engine._check_single_ops()

        # WHITE patterns should not generate information
        assert len(info_list) == 0

    def test_gray_pattern_optional_action(
        self, gray_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test GRAY pattern generates optional action."""
        engine = InformationEngine(
            op_runtime_results=[gray_runtime_result],
            subgraph_runtime_results=[],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info_list = engine._check_single_ops()

        assert len(info_list) == 1
        info = info_list[0]
        assert info.pattern_id == "OP/ai.onnx/MatMul"
        assert (
            "is not supported" in info.explanation.lower()
            or "partial support" in info.explanation.lower()
        )

    def test_black_pattern_required_action(
        self, black_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test BLACK pattern generates required action."""
        engine = InformationEngine(
            op_runtime_results=[black_runtime_result],
            subgraph_runtime_results=[],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info_list = engine._check_single_ops()

        assert len(info_list) == 1
        info = info_list[0]
        assert info.pattern_id == "OP/ai.onnx/Custom"
        assert "not supported" in info.explanation.lower()
        assert info.actions is not None
        assert len(info.actions) == 1
        assert info.actions[0].level == ActionLevel.REQUIRED

    def test_skip_patterns_with_alternatives(
        self,
        black_with_white_alternative: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test patterns with alternatives are skipped in _check_single_ops."""
        engine = InformationEngine(
            op_runtime_results=[black_with_white_alternative],
            subgraph_runtime_results=[],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info_list = engine._check_single_ops()

        # Should be empty because pattern has alternatives
        assert len(info_list) == 0

    def test_skip_empty_pattern_id(self, simple_model: ONNXModel, test_device: str) -> None:
        """Test patterns with empty pattern_id are skipped."""
        invalid_result = PatternRuntime(
            pattern_id="",
            result=RuntimeTestResult(compile=False, run=False),
            alternatives=[],
        )

        engine = InformationEngine(
            op_runtime_results=[invalid_result],
            subgraph_runtime_results=[],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info_list = engine._check_single_ops()

        assert len(info_list) == 0


class TestInformationEngineCheckPatterns:
    """Tests for _check_patterns method."""

    def test_patterns_with_alternatives_processed(
        self,
        black_with_white_alternative: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test patterns with alternatives are processed."""
        engine = InformationEngine(
            op_runtime_results=[black_with_white_alternative],
            subgraph_runtime_results=[],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info_list = engine._check_patterns()

        assert len(info_list) == 1
        info = info_list[0]
        assert info.pattern_id == "SUBGRAPH/old_pattern"

    def test_subgraph_patterns_processed(
        self,
        black_with_white_alternative: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test subgraph patterns are processed."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[black_with_white_alternative],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info_list = engine._check_patterns()

        assert len(info_list) == 1

    def test_white_pattern_with_no_alternatives_skipped(
        self, white_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test WHITE pattern with no alternatives is skipped."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[white_runtime_result],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info_list = engine._check_patterns()

        assert len(info_list) == 0


class TestInformationEngineProcessPatternWithAlternatives:
    """Tests for _process_pattern_with_alternatives method."""

    def test_black_to_white_alternative_required_action(
        self,
        black_with_white_alternative: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test BLACK to WHITE alternative generates required action."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[black_with_white_alternative],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info = engine._process_pattern_with_alternatives(black_with_white_alternative)

        assert info is not None
        assert info.pattern_id == "SUBGRAPH/old_pattern"
        assert "not supported" in info.explanation.lower()
        assert info.actions is not None
        assert len(info.actions) == 1
        assert info.actions[0].level == ActionLevel.REQUIRED
        assert info.actions[0].pattern_to_id == "OP/ai.onnx/Conv"

    def test_black_to_gray_alternative_required_action(
        self, black_with_gray_alternative: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test BLACK to GRAY alternative generates required action."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[black_with_gray_alternative],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info = engine._process_pattern_with_alternatives(black_with_gray_alternative)

        assert info is not None
        assert info.actions is not None
        assert len(info.actions) == 1
        # BLACK → GRAY is still REQUIRED (improvement from not working to partial support)
        assert info.actions[0].level == ActionLevel.REQUIRED
        assert (
            "partial support" in info.actions[0].details.lower()
            or "gray" in info.actions[0].details.lower()
        )

    def test_gray_to_white_alternative_optional_action(
        self, gray_with_white_alternative: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test GRAY to WHITE alternative generates required action (improvement to full support)."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[gray_with_white_alternative],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info = engine._process_pattern_with_alternatives(gray_with_white_alternative)

        assert info is not None
        assert info.actions is not None
        assert len(info.actions) == 1
        # GRAY → WHITE is REQUIRED (improvement from partial to full support)
        assert info.actions[0].level == ActionLevel.REQUIRED
        assert info.actions[0].status == SupportLevel.WHITE

    def test_white_with_no_alternatives_returns_none(
        self, white_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test WHITE pattern with no alternatives returns None."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[white_runtime_result],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info = engine._process_pattern_with_alternatives(white_runtime_result)

        assert info is None


class TestInformationEngineExtractActions:
    """Tests for _extract_actions method."""

    def test_extract_actions_black_to_white(
        self,
        black_with_white_alternative: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test extracting actions from BLACK to WHITE alternative."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[black_with_white_alternative],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        actions = engine._extract_actions(black_with_white_alternative)

        assert len(actions) == 1
        action = actions[0]
        assert action.pattern_from_id == "SUBGRAPH/old_pattern"
        assert action.pattern_to_id == "OP/ai.onnx/Conv"
        assert action.level == ActionLevel.REQUIRED
        assert action.status == SupportLevel.WHITE
        assert "equivalent" in action.details.lower()

    def test_extract_actions_gray_to_white(
        self, gray_with_white_alternative: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test extracting actions from GRAY to WHITE alternative."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[gray_with_white_alternative],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        actions = engine._extract_actions(gray_with_white_alternative)

        assert len(actions) == 1
        action = actions[0]
        # GRAY → WHITE is REQUIRED (improvement)
        assert action.level == ActionLevel.REQUIRED
        assert action.status == SupportLevel.WHITE

    def test_extract_actions_no_improvement_alternatives(
        self, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test alternatives with no improvement are skipped."""
        # BLACK pattern with BLACK alternative (no improvement)
        alternative = PatternAlternative(
            pattern_id="OP/ai.onnx/Alt",
            result=RuntimeTestResult(compile=False, run=False),
            alternative_type=AlternativeType.EQUIVALENT,
        )
        runtime = PatternRuntime(
            pattern_id="OP/ai.onnx/Test",
            result=RuntimeTestResult(compile=False, run=False),
            alternatives=[alternative],
        )

        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[runtime],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        actions = engine._extract_actions(runtime)

        # Should create a warning action since no viable alternatives
        assert len(actions) == 1
        assert actions[0].level == ActionLevel.WARNING

    def test_extract_actions_black_with_no_alternatives(
        self, black_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test BLACK pattern with no alternatives generates warning."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[black_runtime_result],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        actions = engine._extract_actions(black_runtime_result)

        assert len(actions) == 1
        assert actions[0].level == ActionLevel.WARNING
        assert (
            "no alternatives are available" in actions[0].details.lower()
            or "no supported alternatives" in actions[0].details.lower()
        )

    def test_extract_actions_skip_invalid_alternative(
        self, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test invalid alternatives are skipped."""
        # Alternative with empty pattern_id
        invalid_alt = PatternAlternative(
            pattern_id="",
            result=RuntimeTestResult(compile=True, run=True),
            alternative_type=AlternativeType.EQUIVALENT,
        )
        runtime = PatternRuntime(
            pattern_id="OP/ai.onnx/Test",
            result=RuntimeTestResult(compile=False, run=False),
            alternatives=[invalid_alt],
        )

        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[runtime],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        actions = engine._extract_actions(runtime)

        # Should generate warning since no valid alternatives
        assert len(actions) == 1
        assert actions[0].level == ActionLevel.WARNING


class TestInformationEngineIntegration:
    """Integration tests for InformationEngine."""

    def test_full_workflow_mixed_results(
        self,
        white_runtime_result: PatternRuntime,
        gray_runtime_result: PatternRuntime,
        black_runtime_result: PatternRuntime,
        black_with_white_alternative: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test complete workflow with mixed runtime results."""
        engine = InformationEngine(
            op_runtime_results=[
                white_runtime_result,
                gray_runtime_result,
                black_runtime_result,
            ],
            subgraph_runtime_results=[black_with_white_alternative],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info_list = engine.summary()

        # Should have information for:
        # - GRAY operator (optional action)
        # - BLACK operator (required action)
        # - BLACK subgraph with alternative (required action)
        # WHITE operator should not generate information
        assert len(info_list) >= 3

        # Verify all information objects are valid
        for info in info_list:
            assert isinstance(info, Information)
            assert info.explanation
            assert info.pattern_id

    def test_all_white_patterns_no_information(
        self, white_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test all WHITE patterns generate no information."""
        engine = InformationEngine(
            op_runtime_results=[white_runtime_result],
            subgraph_runtime_results=[white_runtime_result],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info_list = engine.summary()

        assert len(info_list) == 0

    def test_all_black_patterns_generate_actions(
        self, black_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test all BLACK patterns generate required or warning actions."""
        engine = InformationEngine(
            op_runtime_results=[black_runtime_result],
            subgraph_runtime_results=[black_runtime_result],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info_list = engine.summary()

        assert len(info_list) == 2
        for info in info_list:
            assert info.actions is not None
            # BLACK patterns without alternatives generate WARNING or REQUIRED actions
            assert any(
                action.level in (ActionLevel.REQUIRED, ActionLevel.WARNING)
                for action in info.actions
            )
