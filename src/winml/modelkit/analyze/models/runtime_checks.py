# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Rule entities - RuntimeRule and PatternMatch."""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .ihv_type import IHVType
from .support_level import SupportLevel


class NodeTag(str, Enum):
    """Node tag enum for classifying nodes based on their properties."""

    ALL_INPUTS_CONSTANT = "all_inputs_constant"
    MISSING_SHAPE_INFERENCE = "missing_shape_inference"


class AlternativeType(str, Enum):
    """Alternative pattern relationship type enum."""

    EQUIVALENT = "equivalent"
    APPROXIMATION = "approximation"
    QDQ = "QDQ"


class RuntimeTestResult(BaseModel):
    """Runtime test results for a pattern.

    Attributes:
        compile: Whether compilation succeeds
        run: Whether execution succeeds
        reason: Failure reason (optional)
        no_data: Whether runtime data is unavailable (optional)
        node_tags: Dict mapping node name to list of tag strings for classification (optional)
        classification: Support level (computed from compile and run)
    """

    compile: bool = Field(..., description="Whether compilation succeeds")
    run: bool = Field(..., description="Whether execution succeeds")
    reason: str | None = Field(default=None, description="Failure reason")
    filter: str | None = Field(default=None, description="Filter applied during fuzzing matching")
    no_data: bool = Field(default=False, description="Whether runtime data is unavailable")
    node_tags: list[NodeTag] = Field(
        default_factory=list, description="List of NodeTag enums for classifying this node"
    )
    debug_details: Any | None = Field(
        None, description="Optional debug information for runtime checks"
    )

    @property
    def classification(self) -> SupportLevel:
        """Compute classification from compile and run status.

        Returns:
            SupportLevel.UNKNOWN if no_data=True
            SupportLevel.WHITE if compile=True and run=True
            SupportLevel.GRAY if compile=False and run=True
            SupportLevel.BLACK if compile=False and run=False
        """
        if self.no_data:
            return SupportLevel.UNKNOWN
        if self.compile and self.run:
            return SupportLevel.WHITE
        if not self.compile and self.run:
            return SupportLevel.GRAY
        return SupportLevel.BLACK


class PatternAlternative(BaseModel):
    """Alternative pattern with runtime result.

    Attributes:
        pattern_id: Alternative pattern identifier
        result: Runtime test result for this alternative
        alternative_type: Type of alternative relationship
    """

    pattern_id: str = Field(..., description="Alternative pattern identifier")
    result: RuntimeTestResult = Field(..., description="Runtime test result")
    alternative_type: AlternativeType = Field(..., description="Type of alternative relationship")


class PatternRuntime(BaseModel):
    """Runtime execution result for a pattern with alternatives.

    Attributes:
        pattern_id: Original pattern identifier
        result: Runtime test result for the original pattern
        alternatives: List of alternative patterns with their results
        pattern_match: The PatternMatch object for this runtime check
    """

    pattern_id: str = Field(..., description="Pattern identifier")
    result: RuntimeTestResult = Field(..., description="Runtime test result")
    alternatives: list[PatternAlternative] = Field(
        default_factory=list, description="Alternative patterns with results"
    )
    pattern_match: Any = Field(
        default=None, description="The PatternMatch object for this runtime check"
    )


class RuntimeCheckRule(BaseModel):
    """Represents IHV-specific operator/pattern support validation rule.

    Attributes:
        rule_id: Unique rule identifier (UUID)
        pattern_id: Links to OperatorPattern or SubgraphPattern
        alternatives: Dict of alternate op/subgraph patterns with their
            relationship types (optional)
        ihv_type: Target execution provider
        ep_version: EP version constraint (optional, "*" for any)
        driver_version: Driver version constraint (optional, "*" for any)
        namespace: ONNX namespace (optional for subgraph patterns)
        op_version: Opset version (optional)
        type_vars: Type constraints with wildcard support (optional)
        attributes: Attribute constraints with wildcard support (optional)
        input_shapes: Input shape constraints (optional)
        input_is_constant: Whether inputs must be constants (optional)
        test_result: Validation results
    """

    rule_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique rule ID")
    pattern_id: str = Field(..., pattern=r"^(OP|SUBGRAPH)/.+", description="Pattern identifier")
    alternatives: list[dict[str, AlternativeType]] | None = Field(
        default=None,
        description=(
            "Alternative patterns with relationship types (equivalent, approximation, QDQ)"
        ),
    )
    ihv_type: IHVType = Field(..., description="Target IHV type")
    ep_version: str | None = Field(default=None, description="EP version constraint")
    driver_version: str | None = Field(default=None, description="Driver version constraint")
    namespace: str | None = Field(default=None, description="ONNX namespace")
    op_version: int | None = Field(default=None, ge=1, description="Opset version")
    type_vars: dict[str, str] | None = Field(default=None, description="Type constraints")
    attributes: dict[str, str] | None = Field(default=None, description="Attribute constraints")
    input_shapes: dict[str, list[int]] | None = Field(
        default=None, description="Input shape constraints"
    )
    input_is_constant: dict[str, bool] | None = Field(
        default=None, description="Constant input requirements"
    )
    test_result: RuntimeTestResult = Field(..., description="Validation results")
