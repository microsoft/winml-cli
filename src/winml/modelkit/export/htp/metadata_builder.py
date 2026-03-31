# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Metadata Builder for HTP Exporter.

This module provides a clean, template-based approach to building metadata
using dataclasses and the builder pattern for better maintainability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import __spec_version__ as htp_version


@dataclass
class ExportContext:
    """Export session context information."""

    timestamp: str = ""  # Will be set from ExportData
    strategy: str = "htp"
    version: str = htp_version
    exporter: str = "HTPExporter"
    embed_hierarchy_attributes: bool = True
    export_time_seconds: float = 0.0


@dataclass
class ModelInfo:
    """Model information."""

    name_or_path: str
    class_name: str
    framework: str = "transformers"
    total_modules: int = 0
    total_parameters: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with renamed fields."""
        return {
            "name_or_path": self.name_or_path,
            "class_name": self.class_name,
            "framework": self.framework,
            "total_modules": self.total_modules,
            "total_parameters": self.total_parameters,
        }


@dataclass
class TracingInfo:
    """Tracing execution information."""

    builder: str = "TracingHierarchyBuilder"
    modules_traced: int = 0
    execution_steps: int = 0
    model_type: str | None = None
    task: str | None = None
    inputs: dict[str, dict[str, Any]] | None = None
    outputs: list[str] | None = None


@dataclass
class TaggingCoverage:
    """Tagging coverage statistics."""

    total_onnx_nodes: int = 0
    tagged_nodes: int = 0
    coverage_percentage: float = 0.0
    empty_tags: int = 0


@dataclass
class TaggingInfo:
    """Node tagging information."""

    tagged_nodes: dict[str, str] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    coverage: TaggingCoverage = field(default_factory=TaggingCoverage)


@dataclass
class ONNXModelOutput:
    """ONNX model output information."""

    path: str
    size_mb: float = 0.0
    opset_version: int = 17
    output_names: list[str] | None = None


@dataclass
class OutputFiles:
    """Output file information."""

    onnx_model: ONNXModelOutput
    metadata: dict[str, str] = field(default_factory=dict)
    report: dict[str, str] = field(default_factory=dict)


@dataclass
class ExportStep:
    """Export process step information."""

    status: str = "pending"
    details: dict[str, Any] | None = None


@dataclass
class QualityGuarantees:
    """Export quality guarantees."""

    no_hardcoded_logic: bool = True
    universal_module_tracking: str = "TracingHierarchyBuilder"
    empty_tags_guarantee: int = 0
    coverage_guarantee: str = "0.0%"
    optimum_compatible: bool = True


@dataclass
class ExportReport:
    """Export process report."""

    export_time_seconds: float = 0.0
    steps: dict[str, Any] = field(default_factory=dict)
    quality_guarantees: QualityGuarantees = field(default_factory=QualityGuarantees)


@dataclass
class Statistics:
    """Export statistics summary."""

    export_time: float = 0.0
    hierarchy_modules: int = 0
    traced_modules: int = 0
    onnx_nodes: int = 0
    tagged_nodes: int = 0
    empty_tags: int = 0
    coverage_percentage: float = 0.0
    module_types: list[str] = field(default_factory=list)


class HTPMetadataBuilder:
    """Builder for HTP metadata using clean architecture patterns.

    This builder provides a fluent interface for constructing metadata
    step by step, ensuring all required fields are properly set.
    """

    def __init__(self) -> None:
        """Initialize the builder with default values."""
        self._export_context = ExportContext()
        self._model_info: ModelInfo | None = None
        self._tracing_info = TracingInfo()
        self._modules: dict[str, dict[str, Any]] = {}
        self._tagging_info = TaggingInfo()
        self._output_files: OutputFiles | None = None
        self._export_report = ExportReport()
        self._statistics = Statistics()

    def with_export_context(
        self,
        strategy: str = "htp",
        version: str | None = None,
        exporter: str = "HTPExporter",
        embed_hierarchy_attributes: bool = True,
        export_time_seconds: float = 0.0,
    ) -> HTPMetadataBuilder:
        """Set export context information."""
        self._export_context = ExportContext(
            strategy=strategy,
            version=version if version is not None else htp_version,
            exporter=exporter,
            embed_hierarchy_attributes=embed_hierarchy_attributes,
            export_time_seconds=export_time_seconds,
        )
        return self

    def with_model_info(
        self,
        name_or_path: str,
        class_name: str,
        total_modules: int,
        total_parameters: int,
        framework: str = "transformers",
    ) -> HTPMetadataBuilder:
        """Set model information."""
        self._model_info = ModelInfo(
            name_or_path=name_or_path,
            class_name=class_name,
            framework=framework,
            total_modules=total_modules,
            total_parameters=total_parameters,
        )
        return self

    def with_tracing_info(
        self,
        modules_traced: int,
        execution_steps: int,
        model_type: str | None = None,
        task: str | None = None,
        inputs: dict[str, dict[str, Any]] | None = None,
        outputs: list[str] | None = None,
    ) -> HTPMetadataBuilder:
        """Set tracing information."""
        self._tracing_info = TracingInfo(
            modules_traced=modules_traced,
            execution_steps=execution_steps,
            model_type=model_type,
            task=task,
            inputs=inputs,
            outputs=outputs,
        )
        return self

    def with_modules(self, modules: dict[str, dict[str, Any]]) -> HTPMetadataBuilder:
        """Set module hierarchy data."""
        self._modules = modules
        return self

    def with_tagging_info(
        self,
        tagged_nodes: dict[str, str],
        statistics: dict[str, Any],
        total_onnx_nodes: int,
        tagged_nodes_count: int,
        coverage_percentage: float,
        empty_tags: int,
    ) -> HTPMetadataBuilder:
        """Set tagging information."""
        self._tagging_info = TaggingInfo(
            tagged_nodes=tagged_nodes,
            statistics=statistics,
            coverage=TaggingCoverage(
                total_onnx_nodes=total_onnx_nodes,
                tagged_nodes=tagged_nodes_count,
                coverage_percentage=coverage_percentage,
                empty_tags=empty_tags,
            ),
        )
        return self

    def with_output_files(
        self,
        onnx_path: str,
        onnx_size_mb: float,
        metadata_path: str,
        opset_version: int = 17,
        output_names: list[str] | None = None,
        report_path: str | None = None,
    ) -> HTPMetadataBuilder:
        """Set output file information."""
        self._output_files = OutputFiles(
            onnx_model=ONNXModelOutput(
                path=Path(onnx_path).name,
                size_mb=onnx_size_mb,
                opset_version=opset_version,
                output_names=output_names,
            ),
            metadata={"path": Path(metadata_path).name},
        )

        # Add report file if provided
        if report_path:
            self._output_files.report = {"path": Path(report_path).name}

        return self

    def with_export_report(
        self,
        export_time_seconds: float,
        steps: dict[str, Any],
        empty_tags_guarantee: int,
        coverage_percentage: float,
    ) -> HTPMetadataBuilder:
        """Set export report information."""
        self._export_report = ExportReport(
            export_time_seconds=export_time_seconds,
            steps=steps,
            quality_guarantees=QualityGuarantees(
                empty_tags_guarantee=empty_tags_guarantee,
                coverage_guarantee=f"{coverage_percentage:.1f}%",
            ),
        )
        return self

    def with_statistics(
        self,
        export_time: float,
        hierarchy_modules: int,
        traced_modules: int,
        onnx_nodes: int,
        tagged_nodes: int,
        empty_tags: int,
        coverage_percentage: float,
        module_types: list[str],
    ) -> HTPMetadataBuilder:
        """Set statistics summary."""
        self._statistics = Statistics(
            export_time=export_time,
            hierarchy_modules=hierarchy_modules,
            traced_modules=traced_modules,
            onnx_nodes=onnx_nodes,
            tagged_nodes=tagged_nodes,
            empty_tags=empty_tags,
            coverage_percentage=coverage_percentage,
            module_types=module_types,
        )
        return self

    def build(self) -> dict[str, Any]:
        """Build the final metadata dictionary.

        Returns:
            Complete metadata dictionary ready for JSON serialization.

        Raises:
            ValueError: If required fields are missing.
        """
        if self._model_info is None:
            raise ValueError("Model info is required")

        if self._output_files is None:
            raise ValueError("Output files info is required")

        # Build the metadata dictionary in the correct order
        # Updated order: tracing moved after model and before modules
        metadata = {
            "export_context": asdict(self._export_context),
            "model": self._model_info.to_dict(),
            "tracing": asdict(self._tracing_info),  # Moved after model
            "modules": self._modules,
            "nodes": self._tagging_info.tagged_nodes,  # Nodes at root level per schema
            "outputs": {
                "onnx_model": asdict(self._output_files.onnx_model),
                "metadata": self._output_files.metadata,
                "report": self._output_files.report if self._output_files.report else None,
            },
            "report": {"steps": self._export_report.steps},
            "statistics": asdict(self._statistics),
        }

        # Clean up None values
        return self._clean_dict(metadata)

    def build_minimal(self, error: str | None = None) -> dict[str, Any]:
        """Build minimal valid metadata for error cases.

        This method creates a minimal metadata structure that can be used
        when the full build() method fails. It ensures there's always valid
        metadata output even in error scenarios.

        Args:
            error: Optional error message to include in metadata

        Returns:
            Minimal metadata dictionary with export context and optional error
        """
        import time

        from ...core.time_utils import format_timestamp_iso

        # Create minimal export context with defaults
        minimal_context = ExportContext(
            timestamp=format_timestamp_iso(time.time()),
            strategy="htp",
            # version will use the default htp_version from the dataclass
        )

        result = {"export_context": asdict(minimal_context)}

        if error:
            result["error"] = error

        return result

    def _clean_dict(self, d: dict[str, Any]) -> dict[str, Any]:
        """Remove None values from dictionary recursively."""
        cleaned = {}
        for key, value in d.items():
            if value is not None:
                if isinstance(value, dict):
                    cleaned_value = self._clean_dict(value)
                    if cleaned_value:  # Only add non-empty dicts
                        cleaned[key] = cleaned_value
                else:
                    cleaned[key] = value
        return cleaned
