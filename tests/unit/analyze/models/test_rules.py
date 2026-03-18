"""
Unit tests for RuntimeCheckRule and RuntimeTestResult Pydantic validation.

Tests verify:
- RuntimeTestResult classification computation from compile and run status
- RuntimeCheckRule pattern_id validation
- RuntimeCheckRule UUID generation
- Optional fields (type_vars, attributes, version constraints)
"""

from uuid import UUID

import pytest
from pydantic import ValidationError

from winml.modelkit.analyze.models.ihv_type import IHVType
from winml.modelkit.analyze.models.runtime_checks import (
    RuntimeCheckRule,
    RuntimeTestResult,
)
from winml.modelkit.analyze.models.support_level import SupportLevel


class TestRuntimeTestResultValidation:
    """Test RuntimeTestResult nested model validation."""

    def test_classification_whitelist_compile_true_run_true(self):
        """Test that compile=True and run=True gives whitelist classification."""
        result = RuntimeTestResult(
            compile=True,
            run=True,
        )
        assert result.classification == SupportLevel.WHITE

    def test_classification_graylist_compile_true_run_false(self):
        """Test that compile=False and run=True gives graylist classification."""
        result = RuntimeTestResult(
            compile=False,
            run=True,
        )
        assert result.classification == SupportLevel.GRAY

    def test_classification_blacklist_compile_false(self):
        """Test that compile=False and run=False gives blacklist classification."""
        # compile=False, run=False
        result1 = RuntimeTestResult(
            compile=False,
            run=False,
        )
        assert result1.classification == SupportLevel.BLACK

        # compile=False, run=True gives GRAY (fallback to CPU scenario)
        result2 = RuntimeTestResult(
            compile=False,
            run=True,
        )
        assert result2.classification == SupportLevel.GRAY

    def test_reason_optional(self):
        """Test that reason field is optional."""
        # Without reason
        result = RuntimeTestResult(
            compile=True,
            run=True,
        )
        assert result.reason is None

        # With reason
        result_with_reason = RuntimeTestResult(
            compile=False,
            run=False,
            reason="Custom operator not supported",
        )
        assert result_with_reason.reason == "Custom operator not supported"


class TestRuntimeCheckRuleValidation:
    """Test RuntimeCheckRule validation rules."""

    def test_rule_id_generated_as_uuid(self):
        """Test that rule_id is auto-generated as UUID."""
        rule = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            test_result=RuntimeTestResult(
                compile=True,
                run=True,
            ),
        )

        # Validate that rule_id is a valid UUID
        assert isinstance(rule.rule_id, str)
        UUID(rule.rule_id)  # Should not raise ValueError

    def test_valid_pattern_id_formats(self):
        """Test that valid pattern_id formats are accepted."""
        # Operator pattern
        rule_op = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.INTEL,
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule_op.pattern_id == "OP/ai.onnx/Conv"

        # Subgraph pattern
        rule_subgraph = RuntimeCheckRule(
            pattern_id="SUBGRAPH/GELU",
            ihv_type=IHVType.AMD,
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule_subgraph.pattern_id == "SUBGRAPH/GELU"

    def test_invalid_pattern_id_format(self):
        """Test that invalid pattern_id formats are rejected."""
        with pytest.raises(ValidationError):
            RuntimeCheckRule(
                pattern_id="INVALID/Conv",
                ihv_type=IHVType.QC,
                test_result=RuntimeTestResult(compile=True, run=True),
            )

    def test_ihv_type_enum_values(self):
        """Test that IHVType enum values are accepted."""
        for ihv in [IHVType.QC, IHVType.INTEL, IHVType.AMD]:
            rule = RuntimeCheckRule(
                pattern_id="OP/ai.onnx/Conv",
                ihv_type=ihv,
                test_result=RuntimeTestResult(compile=True, run=True),
            )
            assert rule.ihv_type == ihv

    def test_version_constraints_optional(self):
        """Test that ep_version and driver_version are optional."""
        # Without versions
        rule = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule.ep_version is None
        assert rule.driver_version is None

        # With wildcard versions
        rule_wildcard = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            ep_version="*",
            driver_version="*",
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule_wildcard.ep_version == "*"
        assert rule_wildcard.driver_version == "*"

        # With specific versions
        rule_specific = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.INTEL,
            ep_version="2023.3",
            driver_version="31.0.101.5522",
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule_specific.ep_version == "2023.3"
        assert rule_specific.driver_version == "31.0.101.5522"

    def test_namespace_optional(self):
        """Test that namespace field is optional."""
        rule = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule.namespace is None

        rule_with_namespace = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            namespace="ai.onnx",
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule_with_namespace.namespace == "ai.onnx"

    def test_op_version_optional_with_minimum_value(self):
        """Test that op_version is optional and must be >= 1 if provided."""
        # Without op_version
        rule = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule.op_version is None

        # With valid op_version
        rule_with_version = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            op_version=13,
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule_with_version.op_version == 13

        # Invalid: op_version < 1
        with pytest.raises(ValidationError):
            RuntimeCheckRule(
                pattern_id="OP/ai.onnx/Conv",
                ihv_type=IHVType.QC,
                op_version=0,
                test_result=RuntimeTestResult(compile=True, run=True),
            )

    def test_type_vars_optional(self):
        """Test that type_vars field is optional."""
        # Without type_vars
        rule = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule.type_vars is None

        # With type_vars
        rule_with_vars = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            type_vars={"T": "float32"},
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule_with_vars.type_vars == {"T": "float32"}

        # With wildcard type_vars
        rule_wildcard = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            type_vars={"T": "*"},
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule_wildcard.type_vars == {"T": "*"}

    def test_attributes_optional(self):
        """Test that attributes field is optional."""
        # Without attributes
        rule = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule.attributes is None

        # With attributes
        rule_with_attrs = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            attributes={"kernel_shape": "[3, 3]", "pads": "[1, 1, 1, 1]"},
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule_with_attrs.attributes["kernel_shape"] == "[3, 3]"

        # With wildcard attributes
        rule_wildcard = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            attributes={"kernel_shape": "*", "pads": "*"},
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule_wildcard.attributes["kernel_shape"] == "*"

    def test_input_shapes_optional(self):
        """Test that input_shapes field is optional."""
        # Without input_shapes
        rule = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule.input_shapes is None

        # With input_shapes
        rule_with_shapes = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            input_shapes={"X": [1, 3, 224, 224], "W": [64, 3, 3, 3]},
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule_with_shapes.input_shapes["X"] == [1, 3, 224, 224]

    def test_input_is_constant_optional(self):
        """Test that input_is_constant field is optional."""
        # Without input_is_constant
        rule = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule.input_is_constant is None

        # With input_is_constant
        rule_with_const = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            input_is_constant={"W": True, "X": False},
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule_with_const.input_is_constant["W"] is True
        assert rule_with_const.input_is_constant["X"] is False

    def test_alternatives_optional(self):
        """Test that alternatives field is optional."""
        # Without alternatives
        rule = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            test_result=RuntimeTestResult(compile=True, run=True),
        )
        assert rule.alternatives is None

        # With alternatives
        from winml.modelkit.analyze.models.runtime_checks import AlternativeType

        rule_with_alts = RuntimeCheckRule(
            pattern_id="SUBGRAPH/GELU_Erf",
            ihv_type=IHVType.QC,
            alternatives=[
                {"SUBGRAPH/GELU_Tanh": AlternativeType.APPROXIMATION},
                {"OP/ai.onnx/Gelu": AlternativeType.EQUIVALENT},
            ],
            test_result=RuntimeTestResult(compile=False, run=False),
        )
        assert rule_with_alts.alternatives is not None
        assert len(rule_with_alts.alternatives) == 2

    def test_complete_rule_example(self):
        """Test a complete rule with all fields populated."""
        rule = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Conv",
            ihv_type=IHVType.QC,
            ep_version="2.0",
            driver_version="1.0.0",
            namespace="ai.onnx",
            op_version=13,
            type_vars={"T": "float32"},
            attributes={"kernel_shape": "[3, 3]", "pads": "[1, 1, 1, 1]"},
            input_shapes={"X": [1, 3, 224, 224]},
            input_is_constant={"W": True},
            test_result=RuntimeTestResult(
                compile=True,
                run=True,
                reason="Successfully tested on device",
            ),
        )

        assert rule.pattern_id == "OP/ai.onnx/Conv"
        assert rule.ihv_type == IHVType.QC
        assert rule.test_result.classification == SupportLevel.WHITE
        assert rule.namespace == "ai.onnx"
        assert rule.op_version == 13


class TestRuntimeCheckRuleIntegration:
    """Test RuntimeCheckRule integration scenarios."""

    def test_whitelist_rule(self):
        """Test creating a whitelist rule."""
        rule = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/Relu",
            ihv_type=IHVType.QC,
            ep_version="*",
            driver_version="*",
            test_result=RuntimeTestResult(compile=True, run=True),
        )

        assert rule.test_result.classification == SupportLevel.WHITE

    def test_graylist_rule(self):
        """Test creating a graylist rule (compiles but doesn't run)."""
        rule = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/CustomOp",
            ihv_type=IHVType.INTEL,
            test_result=RuntimeTestResult(
                compile=False,
                run=True,
                reason="Op runs but with precision issues",
            ),
        )

        assert rule.test_result.classification == SupportLevel.GRAY

    def test_blacklist_rule(self):
        """Test creating a blacklist rule (doesn't compile)."""
        rule = RuntimeCheckRule(
            pattern_id="OP/ai.onnx/UnsupportedOp",
            ihv_type=IHVType.AMD,
            test_result=RuntimeTestResult(
                compile=False,
                run=False,
                reason="Operator not supported by backend",
            ),
        )

        assert rule.test_result.classification == SupportLevel.BLACK
