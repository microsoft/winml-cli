# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Output entity - AnalysisOutput for JSON serialization."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .ihv_type import IHVType
from .information import Information
from .support_level import SupportLevel


if TYPE_CHECKING:
    from .onnx_model import ONNXModel


class RuntimeDebugSummaryEntry(BaseModel):
    """Aggregated runtime debug details for a single node_stable_key."""

    case_indices: list[Any] | None = Field(
        default=None,
        description="Matched case indices from rule table.",
    )
    table_path: str | None = Field(
        default=None,
        description="Absolute or normalized path to the source table.",
    )
    table_file: str | None = Field(
        default=None,
        description="Source table filename.",
    )


class EPSupport(BaseModel):
    """Execution Provider support information.

    Attributes:
        ihv_type: IHV type (QC, Intel, AMD, NVIDIA)
        ep_type: Execution Provider name (e.g., 'QNNExecutionProvider')
        device_type: Device type (e.g., 'CPU', 'GPU', 'NPU')
        ep_version: Execution provider version (optional)
        driver_version: Driver version (optional)
        runtime_support: Runtime support status
        has_errors: True if unsupported patterns exist (blocking errors)
        has_warnings: True if partial patterns exist (warnings/optimizations)
        classification: Operator classification by support level
        information: List of information
    """

    ihv_type: IHVType = Field(..., description="IHV type")
    ep_type: str = Field(..., description="Execution Provider name (e.g., QNNExecutionProvider)")
    device_type: str | None = Field(default=None, description="Device type (e.g., CPU, GPU, NPU)")
    ep_version: str | None = Field(default=None, description="Execution provider version")
    driver_version: str | None = Field(default=None, description="Driver version")
    runtime_support: bool = Field(..., description="Runtime support status")
    has_errors: bool = Field(
        ..., description="True if unsupported patterns exist (blocking errors)"
    )
    has_warnings: bool = Field(
        ..., description="True if partial patterns exist (warnings/optimizations)"
    )
    classification: dict[SupportLevel, list[str]] = Field(
        ...,
        description=(
            "Operator classification by support level, "
            "the list[str] will contain ONNXOp's display name"
        ),
    )
    information: list[Information] = Field(
        default_factory=list, description="Available information"
    )
    runtime_debug_details_summary: (
        dict[str, list[str] | dict[str, RuntimeDebugSummaryEntry]] | None
    ) = Field(
        default=None,
        description=(
            "Optional runtime debug summary grouped by support level. "
            "The 'unknown' level is a list of node_stable_key values (no case "
            "data); 'supported'/'partial'/'unsupported' map node_stable_key to "
            "detail entries."
        ),
    )


class ModelStats(BaseModel):
    """Model metadata and analysis statistics.

    Attributes:
        model_path: Analyzed model file path
        opset_version: ONNX opset version
        producer_name: Model producer identifier (optional)
        producer_version: Producer version (optional)
        total_operators: Total number of operator nodes
        operator_counts: Operator type frequency map
        unique_operator_types: Number of unique operator types
        detected_pattern_count: Total patterns detected
    """

    model_path: str = Field(..., description="Analyzed model path")
    opset_version: int = Field(..., ge=1, description="ONNX opset version")
    producer_name: str | None = Field(default=None, description="Model producer")
    producer_version: str | None = Field(default=None, description="Producer version")
    total_operators: int = Field(..., ge=0, description="Total operator count")
    operator_counts: dict[str, int] = Field(..., description="Operator type frequencies")
    unique_operator_types: int = Field(..., ge=0, description="Unique operator types")
    detected_pattern_count: dict[str, int] = Field(
        default_factory=dict,
        description="Pattern ID to count mapping (e.g., {'SUBGRAPH/GELU_Erf': 18})",
    )

    @model_validator(mode="after")
    def validate_total_matches_sum(self) -> ModelStats:
        """Validate total_operators equals sum of operator_counts values."""
        counts_sum = sum(self.operator_counts.values())
        if self.total_operators != counts_sum:
            raise ValueError(
                f"total_operators {self.total_operators} must equal "
                f"sum of operator_counts {counts_sum}"
            )
        return self


class AnalysisOutput(BaseModel):
    """Aggregated analysis results for JSON serialization.

    Attributes:
        analysis_timestamp: When analysis ran
        metadata: Model metadata and statistics
        results: Analysis results
    """

    analysis_timestamp: datetime = Field(
        default_factory=datetime.now, description="Analysis timestamp"
    )
    metadata: ModelStats = Field(..., description="Model metadata and statistics")
    results: list[EPSupport] = Field(..., description="Execution Provider support results")

    @field_validator("results")
    @classmethod
    def validate_ep_types_unique(cls, v: list[EPSupport]) -> list[EPSupport]:
        """Validate that EP types are unique in the list."""
        ep_types = [item.ep_type for item in v]
        if len(ep_types) != len(set(ep_types)):
            raise ValueError(f"Duplicate EP types found: {ep_types}")
        return v

    def model_dump_json(self, **kwargs: object) -> str:
        """Serialize to JSON with datetime handling."""
        return super().model_dump_json(exclude_none=True, **kwargs)  # type: ignore[arg-type]


def extract_model_stats(
    model: ONNXModel,
    detected_pattern_count: dict[str, int] | None = None,
) -> ModelStats:
    """Extract metadata from ONNXModel for analysis output.

    Args:
        model: ONNXModel instance to extract metadata from
        detected_pattern_count: Pattern ID to count mapping (default: empty dict)

    Returns:
        ModelStats object with model statistics
    """
    # Get the model proto to count operators
    model_proto = model.get_model()
    graph = model_proto.graph

    # Count operators
    operator_counts = Counter(node.op_type for node in graph.node)
    total_operators = sum(operator_counts.values())
    unique_operator_types = len(operator_counts)

    return ModelStats(
        model_path=model.model_path,
        opset_version=model.opset_version,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        total_operators=total_operators,
        operator_counts=dict(operator_counts),
        unique_operator_types=unique_operator_types,
        detected_pattern_count=detected_pattern_count if detected_pattern_count else {},
    )
