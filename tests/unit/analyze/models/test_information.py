"""
Unit tests for Information and Action Pydantic validation.

Tests verify:
- Action model with pattern transformation details
- Action priority level (required, optional, warning)
- Information model with optional action field
- Information UUID generation
"""

from uuid import UUID

import pytest
from pydantic import ValidationError

from winml.modelkit.analyze.models.information import Action, ActionLevel, Information
from winml.modelkit.analyze.models.support_level import SupportLevel


class TestActionValidation:
    """Test Action model validation rules."""

    def test_valid_action_creation(self):
        """Test creating a valid Action."""
        action = Action(
            pattern_from_id="SUBGRAPH/GELU_Erf",
            pattern_to_id="SUBGRAPH/GELU_Tanh",
            level=ActionLevel.REQUIRED,
            status=SupportLevel.WHITE,
            details="Use Tanh-based GELU for better hardware support",
        )

        assert action.pattern_from_id == "SUBGRAPH/GELU_Erf"
        assert action.pattern_to_id == "SUBGRAPH/GELU_Tanh"
        assert action.level == ActionLevel.REQUIRED
        assert action.status == SupportLevel.WHITE

    def test_action_level_enum_values(self):
        """Test that ActionLevel enum values are accepted."""
        for level in [ActionLevel.REQUIRED, ActionLevel.OPTIONAL, ActionLevel.WARNING]:
            action = Action(
                pattern_from_id="OP/ai.onnx/Conv",
                pattern_to_id="OP/com.microsoft/FusedConv",
                level=level,
                status=SupportLevel.WHITE,
                details="Test details",
            )
            assert action.level == level

    def test_support_level_enum_values(self):
        """Test that SupportLevel enum values are accepted for status."""
        for status in [SupportLevel.WHITE, SupportLevel.GRAY, SupportLevel.BLACK]:
            action = Action(
                pattern_from_id="OP/ai.onnx/Conv",
                pattern_to_id="OP/com.microsoft/FusedConv",
                level=ActionLevel.REQUIRED,
                status=status,
                details="Test details",
            )
            assert action.status == status

    def test_all_fields_required(self):
        """Test that only required Action fields must be provided."""
        # level and status are optional now, so this should succeed
        action = Action(
            pattern_from_id="OP/ai.onnx/Conv",
            pattern_to_id="OP/com.microsoft/FusedConv",
            details="Test details",
        )
        assert action.level is None
        assert action.status is None

        # Missing required field pattern_from_id
        with pytest.raises(ValidationError):
            Action(
                pattern_to_id="OP/com.microsoft/FusedConv",
                details="Test details",
            )

        # Missing required field pattern_to_id
        with pytest.raises(ValidationError):
            Action(
                pattern_from_id="OP/ai.onnx/Conv",
                details="Test details",
            )


class TestInformationValidation:
    """Test Information model validation rules."""

    def test_information_id_generated_as_uuid(self):
        """Test that Information_id is auto-generated as UUID."""
        info = Information(
            explanation="Use FusedConv for better performance",
            pattern_id="OP/ai.onnx/Conv",
        )

        # Validate that Information_id is a valid UUID
        assert isinstance(info.Information_id, str)
        UUID(info.Information_id)  # Should not raise ValueError

    def test_information_with_actions(self):
        """Test Information with actions populated."""
        action = Action(
            pattern_from_id="SUBGRAPH/GELU_Erf",
            pattern_to_id="SUBGRAPH/GELU_Tanh",
            level=ActionLevel.REQUIRED,
            status=SupportLevel.WHITE,
            details="Tanh-based GELU has better hardware support",
        )

        info = Information(
            actions=[action],
            explanation="GELU Erf-based pattern is not supported",
            pattern_id="SUBGRAPH/GELU_Erf",
        )

        assert info.actions is not None
        assert len(info.actions) == 1
        assert info.actions[0].pattern_from_id == "SUBGRAPH/GELU_Erf"
        assert info.actions[0].pattern_to_id == "SUBGRAPH/GELU_Tanh"
        assert info.pattern_id == "SUBGRAPH/GELU_Erf"

    def test_information_without_actions(self):
        """Test Information without actions (informational only)."""
        info = Information(
            explanation="This pattern may have performance implications",
            pattern_id="OP/ai.onnx/Conv",
        )

        assert info.actions is None
        assert info.explanation == "This pattern may have performance implications"

    def test_pattern_id_optional(self):
        """Test that pattern_id field is optional."""
        # Without pattern_id (general information)
        info = Information(
            explanation="General optimization recommendation",
        )
        assert info.pattern_id is None

        # With pattern_id (specific to a pattern)
        info_with_pattern = Information(
            explanation="Pattern-specific recommendation",
            pattern_id="OP/ai.onnx/Conv",
        )
        assert info_with_pattern.pattern_id == "OP/ai.onnx/Conv"

    def test_explanation_required(self):
        """Test that explanation field is required."""
        with pytest.raises(ValidationError):
            Information(
                pattern_id="OP/ai.onnx/Conv",
            )

    def test_complete_information_example(self):
        """Test a complete Information with all fields."""
        action = Action(
            pattern_from_id="OP/ai.onnx/Conv",
            pattern_to_id="OP/com.microsoft/FusedConv",
            level=ActionLevel.REQUIRED,
            status=SupportLevel.WHITE,
            details=("FusedConv combines convolution and activation for better performance"),
        )

        info = Information(
            actions=[action],
            explanation="Standard Conv operator is not optimized for this hardware",
            pattern_id="OP/ai.onnx/Conv",
        )

        assert info.actions[0].level == ActionLevel.REQUIRED
        assert info.pattern_id == "OP/ai.onnx/Conv"
        assert "FusedConv" in info.actions[0].details


class TestInformationIntegrationScenarios:
    """Test Information integration scenarios."""

    def test_required_action_whitelist(self):
        """Test required action that improves to whitelist."""
        action = Action(
            pattern_from_id="SUBGRAPH/GELU_Erf",
            pattern_to_id="OP/ai.onnx/Gelu",
            level=ActionLevel.REQUIRED,
            status=SupportLevel.WHITE,
            details="Native Gelu operator is fully supported",
        )

        info = Information(
            actions=[action],
            explanation="Erf-based GELU pattern is blacklisted on this hardware",
            pattern_id="SUBGRAPH/GELU_Erf",
        )

        assert info.actions[0].level == ActionLevel.REQUIRED
        assert info.actions[0].status == SupportLevel.WHITE

    def test_optional_action_performance_hint(self):
        """Test optional action for performance optimization."""
        action = Action(
            pattern_from_id="OP/ai.onnx/Conv",
            pattern_to_id="OP/com.microsoft/FusedConv",
            level=ActionLevel.OPTIONAL,
            status=SupportLevel.WHITE,
            details="FusedConv can improve inference speed by 20%",
        )

        info = Information(
            actions=[action],
            explanation="Conv operator works but can be optimized",
            pattern_id="OP/ai.onnx/Conv",
        )

        assert info.actions[0].level == ActionLevel.OPTIONAL
        assert info.actions[0].status == SupportLevel.WHITE

    def test_warning_action_graylist(self):
        """Test warning action for potential issues."""
        action = Action(
            pattern_from_id="OP/ai.onnx/DynamicQuantizeLinear",
            pattern_to_id="OP/ai.onnx/QuantizeLinear",
            level=ActionLevel.WARNING,
            status=SupportLevel.GRAY,
            details="Dynamic quantization may have accuracy issues on this hardware",
        )

        info = Information(
            actions=[action],
            explanation="Dynamic quantization operators have limited support",
            pattern_id="OP/ai.onnx/DynamicQuantizeLinear",
        )

        assert info.actions[0].level == ActionLevel.WARNING
        assert info.actions[0].status == SupportLevel.GRAY

    def test_multiple_pattern_transformation(self):
        """Test action involving complex pattern transformation."""
        action = Action(
            pattern_from_id="SUBGRAPH/LayerNormalization",
            pattern_to_id="OP/ai.onnx/LayerNormalization",
            level=ActionLevel.REQUIRED,
            status=SupportLevel.WHITE,
            details="Native LayerNormalization operator provides better performance",
        )

        info = Information(
            actions=[action],
            explanation="Decomposed LayerNorm pattern has poor performance",
            pattern_id="SUBGRAPH/LayerNormalization",
        )

        assert info.actions[0].pattern_from_id.startswith("SUBGRAPH/")
        assert info.actions[0].pattern_to_id.startswith("OP/")
        assert info.actions[0].level == ActionLevel.REQUIRED

    def test_informational_only_no_actions(self):
        """Test purely informational message without actions."""
        info = Information(
            explanation=("This model contains operators that are experimental on this hardware"),
            pattern_id="OP/ai.onnx/ExperimentalOp",
        )

        assert info.actions is None
        assert "experimental" in info.explanation.lower()

    def test_general_information_no_specific_pattern(self):
        """Test general information not tied to specific pattern."""
        info = Information(
            explanation=("Consider updating to the latest driver version for optimal performance"),
        )

        assert info.pattern_id is None
        assert info.actions is None
        assert "driver version" in info.explanation.lower()

    def test_multiple_actions_in_information(self):
        """Test Information with multiple actions."""
        action1 = Action(
            pattern_from_id="SUBGRAPH/GELU_Erf",
            pattern_to_id="OP/ai.onnx/Gelu",
            level=ActionLevel.REQUIRED,
            status=SupportLevel.WHITE,
            details="Native Gelu operator is fully supported",
        )

        action2 = Action(
            pattern_from_id="SUBGRAPH/GELU_Erf",
            pattern_to_id="SUBGRAPH/GELU_Tanh",
            level=ActionLevel.OPTIONAL,
            status=SupportLevel.WHITE,
            details="Tanh approximation also works well",
        )

        info = Information(
            actions=[action1, action2],
            explanation="Multiple options available for GELU pattern",
            pattern_id="SUBGRAPH/GELU_Erf",
        )

        assert len(info.actions) == 2
        assert info.actions[0].level == ActionLevel.REQUIRED
        assert info.actions[1].level == ActionLevel.OPTIONAL
        assert all(a.pattern_from_id == "SUBGRAPH/GELU_Erf" for a in info.actions)
