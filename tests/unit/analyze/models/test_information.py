# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

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

from winml.modelkit.analyze.models.information import Action, ActionItem, ActionLevel, Information
from winml.modelkit.analyze.models.support_level import SupportLevel


class TestActionValidation:
    """Test Action model validation rules."""

    def test_valid_action_creation(self):
        """Test creating a valid Action."""
        action = Action(
            pattern_from_id="SUBGRAPH/GELU_Erf",
            pattern_to_id="SUBGRAPH/GELU_Tanh",
            level=ActionLevel.REQUIRED,
            status=SupportLevel.SUPPORTED,
            details="Use Tanh-based GELU for better hardware support",
        )

        assert action.pattern_from_id == "SUBGRAPH/GELU_Erf"
        assert action.pattern_to_id == "SUBGRAPH/GELU_Tanh"
        assert action.level == ActionLevel.REQUIRED
        assert action.status == SupportLevel.SUPPORTED

    def test_action_level_enum_values(self):
        """Test that ActionLevel enum values are accepted."""
        for level in [ActionLevel.REQUIRED, ActionLevel.OPTIONAL, ActionLevel.WARNING]:
            action = Action(
                pattern_from_id="OP/ai.onnx/Conv",
                pattern_to_id="OP/com.microsoft/FusedConv",
                level=level,
                status=SupportLevel.SUPPORTED,
                details="Test details",
            )
            assert action.level == level

    def test_support_level_enum_values(self):
        """Test that SupportLevel enum values are accepted for status."""
        for status in [SupportLevel.SUPPORTED, SupportLevel.PARTIAL, SupportLevel.UNSUPPORTED]:
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
            status=SupportLevel.SUPPORTED,
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
            status=SupportLevel.SUPPORTED,
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

    def test_required_action_supportedlist(self):
        """Test required action that improves to supportedlist."""
        action = Action(
            pattern_from_id="SUBGRAPH/GELU_Erf",
            pattern_to_id="OP/ai.onnx/Gelu",
            level=ActionLevel.REQUIRED,
            status=SupportLevel.SUPPORTED,
            details="Native Gelu operator is fully supported",
        )

        info = Information(
            actions=[action],
            explanation="Erf-based GELU pattern is unsupported on this hardware",
            pattern_id="SUBGRAPH/GELU_Erf",
        )

        assert info.actions[0].level == ActionLevel.REQUIRED
        assert info.actions[0].status == SupportLevel.SUPPORTED

    def test_optional_action_performance_hint(self):
        """Test optional action for performance optimization."""
        action = Action(
            pattern_from_id="OP/ai.onnx/Conv",
            pattern_to_id="OP/com.microsoft/FusedConv",
            level=ActionLevel.OPTIONAL,
            status=SupportLevel.SUPPORTED,
            details="FusedConv can improve inference speed by 20%",
        )

        info = Information(
            actions=[action],
            explanation="Conv operator works but can be optimized",
            pattern_id="OP/ai.onnx/Conv",
        )

        assert info.actions[0].level == ActionLevel.OPTIONAL
        assert info.actions[0].status == SupportLevel.SUPPORTED

    def test_warning_action_partiallist(self):
        """Test warning action for potential issues."""
        action = Action(
            pattern_from_id="OP/ai.onnx/DynamicQuantizeLinear",
            pattern_to_id="OP/ai.onnx/QuantizeLinear",
            level=ActionLevel.WARNING,
            status=SupportLevel.PARTIAL,
            details="Dynamic quantization may have accuracy issues on this hardware",
        )

        info = Information(
            actions=[action],
            explanation="Dynamic quantization operators have limited support",
            pattern_id="OP/ai.onnx/DynamicQuantizeLinear",
        )

        assert info.actions[0].level == ActionLevel.WARNING
        assert info.actions[0].status == SupportLevel.PARTIAL

    def test_multiple_pattern_transformation(self):
        """Test action involving complex pattern transformation."""
        action = Action(
            pattern_from_id="SUBGRAPH/LayerNormalization",
            pattern_to_id="OP/ai.onnx/LayerNormalization",
            level=ActionLevel.REQUIRED,
            status=SupportLevel.SUPPORTED,
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
            status=SupportLevel.SUPPORTED,
            details="Native Gelu operator is fully supported",
        )

        action2 = Action(
            pattern_from_id="SUBGRAPH/GELU_Erf",
            pattern_to_id="SUBGRAPH/GELU_Tanh",
            level=ActionLevel.OPTIONAL,
            status=SupportLevel.SUPPORTED,
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


class TestActionItemModelRewrite:
    """Tests for ModelRewrite ActionItem type with flag_name."""

    def test_model_rewrite_action_item_with_optimization_options(self):
        """Test creating a ModelRewrite ActionItem using optimization_options."""
        item = ActionItem(
            type="ModelRewrite",
            optimization_options={"highdimRTR-lowdimRTR": True},
        )

        assert item.type == "ModelRewrite"
        assert item.optimization_options == {"highdimRTR-lowdimRTR": True}

    def test_regular_action_item_optimization_options(self):
        """Test that GraphOptimization ActionItem uses optimization_options."""
        item = ActionItem(
            type="GraphOptimization",
            optimization_options={"gelu_fusion": True},
        )

        assert item.type == "GraphOptimization"
        assert item.optimization_options == {"gelu_fusion": True}

    def test_reshape_transpose_reshape_action_in_information(self):
        """Test ReshapeTransposeReshapeOverlyHighDimPattern info includes a ModelRewrite action."""
        rewrite_item = ActionItem(
            type="ModelRewrite",
            optimization_options={"highdimRTR-lowdimRTR": True},
        )
        action = Action(
            pattern_from_id="SUBGRAPH/ReshapeTransposeReshapeOverlyHighDimPattern",
            pattern_to_id="SUBGRAPH/ReshapeTransposeReshapeLowDimPattern",
            action_items=[rewrite_item],
            details="Merge Reshape-Transpose-Reshape into a single operator.",
        )
        info = Information(
            actions=[action],
            explanation="The Reshape-Transpose-Reshape sequence can be merged.",
            pattern_id="SUBGRAPH/ReshapeTransposeReshapeOverlyHighDimPattern",
        )

        assert info.pattern_id == "SUBGRAPH/ReshapeTransposeReshapeOverlyHighDimPattern"
        assert len(info.actions) == 1
        assert len(info.actions[0].action_items) == 1
        item = info.actions[0].action_items[0]
        assert item.type == "ModelRewrite"
        assert item.optimization_options == {"highdimRTR-lowdimRTR": True}

    def test_default_information_json_has_reshape_transpose_reshape_entry(self):
        """Test default_information.json includes the ReshapeTransposeReshape entry."""
        import json
        from pathlib import Path

        rules_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "src"
            / "winml"
            / "modelkit"
            / "analyze"
            / "rules"
            / "information_rules"
            / "default_information.json"
        )
        data = json.loads(rules_path.read_text(encoding="utf-8"))
        pattern_ids = [entry["pattern_id"] for entry in data if "pattern_id" in entry]
        assert "SUBGRAPH/ReshapeTransposeReshapeOverlyHighDimPattern" in pattern_ids

        target_id = "SUBGRAPH/ReshapeTransposeReshapeOverlyHighDimPattern"
        entry = next(e for e in data if e.get("pattern_id") == target_id)
        assert entry["enabled"] is True
        action_items = entry["actions"][0]["action_items"]
        assert len(action_items) == 1
        assert action_items[0]["type"] == "GraphOptimization"
        assert action_items[0]["optimization_options"] == {"highdimRTR-lowdimRTR": True}

    def test_qc_information_json_has_transpose_attention_entry(self):
        """Test that qc_information.json has the QC-specific TransposeAttentionPattern entry."""
        import json
        from pathlib import Path

        rules_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "src"
            / "winml"
            / "modelkit"
            / "analyze"
            / "rules"
            / "information_rules"
            / "qc_information.json"
        )
        data = json.loads(rules_path.read_text(encoding="utf-8"))
        pattern_ids = {e["pattern_id"] for e in data if "pattern_id" in e}
        assert "SUBGRAPH/TransposeAttentionPattern" in pattern_ids

        target_id = "SUBGRAPH/TransposeAttentionPattern"
        entry = next(e for e in data if e.get("pattern_id") == target_id)
        assert entry["enabled"] is True
        action_items = entry["actions"][0]["action_items"]
        assert len(action_items) == 1
        assert action_items[0]["type"] == "GraphOptimization"
        assert action_items[0]["optimization_options"] == {"attention-expandedattention": True}

    def test_default_information_json_does_not_have_transpose_attention_entry(self):
        """Test that TransposeAttentionPattern is NOT in default_information.json (QC-specific)."""
        import json
        from pathlib import Path

        rules_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "src"
            / "winml"
            / "modelkit"
            / "analyze"
            / "rules"
            / "information_rules"
            / "default_information.json"
        )
        data = json.loads(rules_path.read_text(encoding="utf-8"))
        pattern_ids = {e["pattern_id"] for e in data if "pattern_id" in e}
        assert "SUBGRAPH/TransposeAttentionPattern" not in pattern_ids
