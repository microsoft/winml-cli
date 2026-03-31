# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for InformationEngine."""

import onnx
import pytest

from winml.modelkit.analyze import (
    ActionLevel,
    AlternativeType,
    Information,
    InformationEngine,
    ONNXModel,
    RuntimeTestResult,
    SupportLevel,
)
from winml.modelkit.analyze.models.runtime_checks import (  # Testing internal implementation
    PatternAlternative,
    PatternRuntime,
)


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
def supported_runtime_result() -> PatternRuntime:
    """Create SUPPORTED (fully supported) runtime result."""
    return PatternRuntime(
        pattern_id="OP/ai.onnx/Conv",
        result=RuntimeTestResult(compile=True, run=True),
        alternatives=[],
    )


@pytest.fixture
def partial_runtime_result() -> PatternRuntime:
    """Create PARTIAL (partial support) runtime result."""
    return PatternRuntime(
        pattern_id="OP/ai.onnx/MatMul",
        result=RuntimeTestResult(compile=False, run=True, reason="Performance issues"),
        alternatives=[],
    )


@pytest.fixture
def unsupported_runtime_result() -> PatternRuntime:
    """Create UNSUPPORTED (not supported) runtime result."""
    return PatternRuntime(
        pattern_id="OP/ai.onnx/Custom",
        result=RuntimeTestResult(compile=False, run=False, reason="Unsupported op"),
        alternatives=[],
    )


@pytest.fixture
def unsupported_with_supported_alternative() -> PatternRuntime:
    """Create UNSUPPORTED runtime result with SUPPORTED alternative."""
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
def unsupported_with_partial_alternative() -> PatternRuntime:
    """Create UNSUPPORTED runtime result with PARTIAL alternative."""
    alternative = PatternAlternative(
        pattern_id="OP/ai.onnx/Alt",
        result=RuntimeTestResult(compile=False, run=True),  # PARTIAL: compile=False, run=True
        alternative_type=AlternativeType.APPROXIMATION,
    )
    return PatternRuntime(
        pattern_id="SUBGRAPH/unsupported",
        result=RuntimeTestResult(compile=False, run=False),
        alternatives=[alternative],
    )


@pytest.fixture
def partial_with_supported_alternative() -> PatternRuntime:
    """Create PARTIAL runtime result with SUPPORTED alternative."""
    alternative = PatternAlternative(
        pattern_id="SUBGRAPH/optimized",
        result=RuntimeTestResult(compile=True, run=True),
        alternative_type=AlternativeType.EQUIVALENT,
    )
    return PatternRuntime(
        pattern_id="SUBGRAPH/unoptimized",
        result=RuntimeTestResult(compile=False, run=True),  # PARTIAL: compile=False, run=True
        alternatives=[alternative],
    )


class TestInformationEngineInit:
    """Tests for InformationEngine initialization."""

    def test_init_with_op_results(
        self, supported_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test initialization with operator results."""
        engine = InformationEngine(
            op_runtime_results=[supported_runtime_result],
            subgraph_runtime_results=[],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        assert len(engine.op_runtime_results) == 1
        assert len(engine.subgraph_runtime_results) == 0
        assert engine._ep == "QNNExecutionProvider"

    def test_init_with_subgraph_results(
        self, unsupported_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test initialization with subgraph results."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[unsupported_runtime_result],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        assert len(engine.op_runtime_results) == 0
        assert len(engine.subgraph_runtime_results) == 1

    def test_init_with_both_results(
        self,
        supported_runtime_result: PatternRuntime,
        unsupported_runtime_result: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test initialization with both operator and subgraph results."""
        engine = InformationEngine(
            op_runtime_results=[supported_runtime_result],
            subgraph_runtime_results=[unsupported_runtime_result],
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
            match=(
                "At least one of op_runtime_results, "
                "subgraph_runtime_results, or model "
                "must be provided"
            ),
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
        self, supported_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test op_runtime_results property."""
        engine = InformationEngine(
            op_runtime_results=[supported_runtime_result],
            subgraph_runtime_results=[],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        results = engine.op_runtime_results
        assert len(results) == 1
        assert results[0] == supported_runtime_result

    def test_subgraph_runtime_results_property(
        self, unsupported_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test subgraph_runtime_results property."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[unsupported_runtime_result],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        results = engine.subgraph_runtime_results
        assert len(results) == 1
        assert results[0] == unsupported_runtime_result


class TestInformationEngineSummary:
    """Tests for summary method."""

    def test_summary_returns_information_list(
        self,
        supported_runtime_result: PatternRuntime,
        unsupported_runtime_result: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test summary returns list of Information objects."""
        engine = InformationEngine(
            op_runtime_results=[supported_runtime_result],
            subgraph_runtime_results=[unsupported_runtime_result],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        result = engine.summary()

        assert isinstance(result, list)
        assert all(isinstance(info, Information) for info in result)

    def test_summary_combines_op_and_pattern_info(
        self,
        unsupported_runtime_result: PatternRuntime,
        unsupported_with_supported_alternative: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test summary combines operator and pattern information."""
        engine = InformationEngine(
            op_runtime_results=[unsupported_runtime_result],
            subgraph_runtime_results=[unsupported_with_supported_alternative],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        result = engine.summary()

        # Should have information for both unsupported operator
        # and unsupported pattern with alternative
        assert len(result) >= 2


class TestInformationEngineCheckSingleOps:
    """Tests for _check_single_ops method."""

    def test_supported_pattern_no_action(
        self, supported_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test SUPPORTED pattern generates no information."""
        engine = InformationEngine(
            op_runtime_results=[supported_runtime_result],
            subgraph_runtime_results=[],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info_list = engine._check_single_ops()

        # Supported patterns should not generate information
        assert len(info_list) == 0

    def test_partial_pattern_optional_action(
        self, partial_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test PARTIAL pattern generates optional action."""
        engine = InformationEngine(
            op_runtime_results=[partial_runtime_result],
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

    def test_unsupported_pattern_required_action(
        self, unsupported_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test UNSUPPORTED pattern generates required action."""
        engine = InformationEngine(
            op_runtime_results=[unsupported_runtime_result],
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
        unsupported_with_supported_alternative: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test patterns with alternatives are skipped in _check_single_ops."""
        engine = InformationEngine(
            op_runtime_results=[unsupported_with_supported_alternative],
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
        unsupported_with_supported_alternative: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test patterns with alternatives are processed."""
        engine = InformationEngine(
            op_runtime_results=[unsupported_with_supported_alternative],
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
        unsupported_with_supported_alternative: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test subgraph patterns are processed."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[unsupported_with_supported_alternative],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info_list = engine._check_patterns()

        assert len(info_list) == 1

    def test_supported_pattern_with_no_alternatives_skipped(
        self, supported_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test SUPPORTED pattern with no alternatives is skipped."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[supported_runtime_result],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info_list = engine._check_patterns()

        assert len(info_list) == 0


class TestInformationEngineProcessPatternWithAlternatives:
    """Tests for _process_pattern_with_alternatives method."""

    def test_unsupported_to_supported_alternative_required_action(
        self,
        unsupported_with_supported_alternative: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test UNSUPPORTED to SUPPORTED alternative generates required action."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[unsupported_with_supported_alternative],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info = engine._process_pattern_with_alternatives(unsupported_with_supported_alternative)

        assert info is not None
        assert info.pattern_id == "SUBGRAPH/old_pattern"
        assert "not supported" in info.explanation.lower()
        assert info.actions is not None
        assert len(info.actions) == 1
        assert info.actions[0].level == ActionLevel.REQUIRED
        assert info.actions[0].pattern_to_id == "OP/ai.onnx/Conv"

    def test_unsupported_to_partial_alternative_required_action(
        self,
        unsupported_with_partial_alternative: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test UNSUPPORTED to PARTIAL alternative generates required action."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[unsupported_with_partial_alternative],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info = engine._process_pattern_with_alternatives(unsupported_with_partial_alternative)

        assert info is not None
        assert info.actions is not None
        assert len(info.actions) == 1
        # UNSUPPORTED -> PARTIAL is still REQUIRED (improvement from not working to partial support)
        assert info.actions[0].level == ActionLevel.REQUIRED
        assert (
            "partial support" in info.actions[0].details.lower()
            or "partial" in info.actions[0].details.lower()
        )

    def test_partial_to_supported_alternative_optional_action(
        self,
        partial_with_supported_alternative: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test PARTIAL to SUPPORTED alternative generates required action."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[partial_with_supported_alternative],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info = engine._process_pattern_with_alternatives(partial_with_supported_alternative)

        assert info is not None
        assert info.actions is not None
        assert len(info.actions) == 1
        # PARTIAL -> SUPPORTED is REQUIRED (improvement from partial to full support)
        assert info.actions[0].level == ActionLevel.REQUIRED
        assert info.actions[0].status == SupportLevel.SUPPORTED

    def test_supported_with_no_alternatives_returns_none(
        self, supported_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test SUPPORTED pattern with no alternatives returns None."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[supported_runtime_result],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info = engine._process_pattern_with_alternatives(supported_runtime_result)

        assert info is None


class TestInformationEngineExtractActions:
    """Tests for _extract_actions method."""

    def test_extract_actions_unsupported_to_supported(
        self,
        unsupported_with_supported_alternative: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test extracting actions from UNSUPPORTED to SUPPORTED alternative."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[unsupported_with_supported_alternative],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        actions = engine._extract_actions(unsupported_with_supported_alternative)

        assert len(actions) == 1
        action = actions[0]
        assert action.pattern_from_id == "SUBGRAPH/old_pattern"
        assert action.pattern_to_id == "OP/ai.onnx/Conv"
        assert action.level == ActionLevel.REQUIRED
        assert action.status == SupportLevel.SUPPORTED
        assert "equivalent" in action.details.lower()

    def test_extract_actions_partial_to_supported(
        self,
        partial_with_supported_alternative: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test extracting actions from PARTIAL to SUPPORTED alternative."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[partial_with_supported_alternative],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        actions = engine._extract_actions(partial_with_supported_alternative)

        assert len(actions) == 1
        action = actions[0]
        # PARTIAL -> SUPPORTED is REQUIRED (improvement)
        assert action.level == ActionLevel.REQUIRED
        assert action.status == SupportLevel.SUPPORTED

    def test_extract_actions_no_improvement_alternatives(
        self, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test alternatives with no improvement are skipped."""
        # UNSUPPORTED pattern with UNSUPPORTED alternative (no improvement)
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

    def test_extract_actions_unsupported_with_no_alternatives(
        self, unsupported_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test UNSUPPORTED pattern with no alternatives generates warning."""
        engine = InformationEngine(
            op_runtime_results=[],
            subgraph_runtime_results=[unsupported_runtime_result],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        actions = engine._extract_actions(unsupported_runtime_result)

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
        supported_runtime_result: PatternRuntime,
        partial_runtime_result: PatternRuntime,
        unsupported_runtime_result: PatternRuntime,
        unsupported_with_supported_alternative: PatternRuntime,
        simple_model: ONNXModel,
        test_device: str,
    ) -> None:
        """Test complete workflow with mixed runtime results."""
        engine = InformationEngine(
            op_runtime_results=[
                supported_runtime_result,
                partial_runtime_result,
                unsupported_runtime_result,
            ],
            subgraph_runtime_results=[unsupported_with_supported_alternative],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info_list = engine.summary()

        # Should have information for:
        # - PARTIAL operator (optional action)
        # - UNSUPPORTED operator (required action)
        # - UNSUPPORTED subgraph with alternative (required action)
        # SUPPORTED operator should not generate information
        assert len(info_list) >= 3

        # Verify all information objects are valid
        for info in info_list:
            assert isinstance(info, Information)
            assert info.explanation
            assert info.pattern_id

    def test_all_supported_patterns_no_information(
        self, supported_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test all SUPPORTED patterns generate no information."""
        engine = InformationEngine(
            op_runtime_results=[supported_runtime_result],
            subgraph_runtime_results=[supported_runtime_result],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info_list = engine.summary()

        assert len(info_list) == 0

    def test_all_unsupported_patterns_generate_actions(
        self, unsupported_runtime_result: PatternRuntime, simple_model: ONNXModel, test_device: str
    ) -> None:
        """Test all UNSUPPORTED patterns generate required or warning actions."""
        engine = InformationEngine(
            op_runtime_results=[unsupported_runtime_result],
            subgraph_runtime_results=[unsupported_runtime_result],
            ep="QNNExecutionProvider",
            model=simple_model,
            device=test_device,
        )

        info_list = engine.summary()

        assert len(info_list) == 2
        for info in info_list:
            assert info.actions is not None
            # UNSUPPORTED patterns without alternatives generate WARNING or REQUIRED actions
            assert any(
                action.level in (ActionLevel.REQUIRED, ActionLevel.WARNING)
                for action in info.actions
            )
