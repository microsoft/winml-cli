"""WinMLExportConfig - ONNX Export Configuration.

Configuration class for ONNX export with static batch defaults for QNN compatibility.

IMPORTANT: This module follows CARDINAL RULE #1 - NO HARDCODED MODEL LOGIC.
Model-specific configurations should use presets which override these defaults.
"""

from __future__ import annotations

import logging
from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING, Any, Literal

# InputTensorSpec and OutputTensorSpec live in modelkit.onnx.io (canonical home).
from ..onnx import InputTensorSpec, OutputTensorSpec


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ..loader.config import WinMLLoaderConfig


logger = logging.getLogger(__name__)


@dataclass
class WinMLExportConfig:
    """Configuration for ONNX export with static batch defaults for QNN compatibility.

    Key Features:
    - Default batch_size=1 to ensure MatMulAddFusion works (prevents BiasGelu)
    - Optional input/output tensor specifications for explicit control
    - Controlled dynamic_axes to prevent dynamic batch issues
    - Opset 17 for modern operator support
    - Optional hierarchy preservation via HTP (Phase 2)

    Input/Output Tensor Specifications:
    - input_tensors: List of InputTensorSpec for explicit input definitions
    - output_tensors: List of OutputTensorSpec for explicit output definitions
    - Both are optional - when None, tensors are inferred from the model

    Hierarchy Preservation (HTP Integration):
    - enable_hierarchy_tags: Add module hierarchy tags to ONNX nodes for debugging
    - clean_onnx: Remove hierarchy tags after export for deployment
    - hierarchy_tag_format: Tag detail level (full, module_only)

    Example:
        # Basic config with defaults
        config = WinMLExportConfig()

        # With explicit input/output tensors
        config = WinMLExportConfig(
            input_tensors=[
                InputTensorSpec(name="pixel_values", dtype="float32", shape=(1, 3, 224, 224))
            ],
            output_tensors=[
                OutputTensorSpec(name="logits")
            ],
        )

        # From preset dict
        config = WinMLExportConfig.from_dict({
            "opset_version": 17,
            "batch_size": 1,
            "input_tensors": [{"name": "pixel_values", "shape": [1, 3, 224, 224]}],
        })
    """

    opset_version: int = 17
    batch_size: int = 1  # Static batch for QNN compatibility

    # Input/output tensor specifications (optional)
    input_tensors: list[InputTensorSpec] | None = None
    output_tensors: list[OutputTensorSpec] | None = None

    # Dynamic axes for ONNX export (optional)
    dynamic_axes: dict[str, dict[int, str]] | None = None

    # Export behavior
    export_params: bool = True
    do_constant_folding: bool = True
    verbose: bool = False
    dynamo: bool = False  # Use TorchScript exporter by default (dynamo=True for PyTorch 2.x)

    # Phase 2: Hierarchy Preservation Options
    enable_hierarchy_tags: bool = True  # Enable HTP hierarchy tagging by default
    clean_onnx: bool = False  # Remove hierarchy tags for deployment
    hierarchy_tag_format: Literal["full", "module_only"] = "full"

    # Backward compatibility: legacy init-only parameters
    # (converted to input_tensors/output_tensors in __post_init__)
    # Named with underscore suffix to avoid conflict with properties
    input_shape_: InitVar[tuple[int, ...] | None] = None
    input_names_: InitVar[list[str] | None] = None
    output_names_: InitVar[list[str] | None] = None

    def __post_init__(
        self,
        input_shape_: tuple[int, ...] | None,
        input_names_: list[str] | None,
        output_names_: list[str] | None,
    ) -> None:
        """Validate configuration after initialization."""
        # Handle legacy parameters - convert to input_tensors/output_tensors if needed
        if input_shape_ is not None and self.input_tensors is None:
            # Convert legacy input_shape to input_tensors
            first_input_name = "input"
            if input_names_:
                first_input_name = input_names_[0]
            self.input_tensors = [InputTensorSpec(name=first_input_name, shape=input_shape_)]
            # Add remaining input names if provided
            if input_names_ and len(input_names_) > 1:
                for name in input_names_[1:]:
                    self.input_tensors.append(InputTensorSpec(name=name))

        if output_names_ is not None and self.output_tensors is None:
            # Convert legacy output_names to output_tensors
            self.output_tensors = [OutputTensorSpec(name=name) for name in output_names_]

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.opset_version < 11:
            logger.warning("opset_version %s is very old, consider using 17+", self.opset_version)

        # Validate input tensor shapes match batch_size if specified
        if self.input_tensors:
            for spec in self.input_tensors:
                if spec.shape and len(spec.shape) > 0 and spec.shape[0] != self.batch_size:
                    logger.warning(
                        "Input tensor '%s' shape[0]=%s doesn't match batch_size=%s",
                        spec.name or "unnamed",
                        spec.shape[0],
                        self.batch_size,
                    )

        # Warn if dynamic batch is detected
        if self.dynamic_axes:
            for input_name, axes in self.dynamic_axes.items():
                if 0 in axes:  # Batch dimension is dynamic
                    logger.warning(
                        "Dynamic batch detected for input '%s'. "
                        "This prevents MatMulAddFusion and causes BiasGelu!",
                        input_name,
                    )

        # Validate hierarchy preservation options
        valid_formats = ["full", "module_only"]
        if self.hierarchy_tag_format not in valid_formats:
            raise ValueError(
                f"Invalid hierarchy_tag_format '{self.hierarchy_tag_format}'. "
                f"Must be one of {valid_formats}"
            )

        # Warn if conflicting hierarchy options
        if self.clean_onnx and not self.enable_hierarchy_tags:
            logger.warning("clean_onnx=True has no effect when enable_hierarchy_tags=False")

    def get_input_names(self) -> list[str]:
        """Get list of input tensor names.

        Returns:
            List of input names, or empty list if not specified.
        """
        if not self.input_tensors:
            return []
        return [spec.name for spec in self.input_tensors if spec.name]

    def get_output_names(self) -> list[str]:
        """Get list of output tensor names.

        Returns:
            List of output names, or empty list if not specified.
        """
        if not self.output_tensors:
            return []
        return [spec.name for spec in self.output_tensors if spec.name]

    def get_input_shape(self, name: str | None = None) -> tuple[int, ...] | None:
        """Get shape for a specific input tensor or the first input.

        Args:
            name: Input tensor name. If None, returns first input's shape.

        Returns:
            Tensor shape tuple, or None if not specified.
        """
        if not self.input_tensors:
            return None

        if name is None:
            # Return first input's shape
            return self.input_tensors[0].shape if self.input_tensors else None

        for spec in self.input_tensors:
            if spec.name == name:
                return spec.shape
        return None

    # Backward compatibility properties for exporter
    @property
    def input_names(self) -> list[str]:
        """Backward-compatible property for input tensor names."""
        return self.get_input_names()

    @property
    def output_names(self) -> list[str]:
        """Backward-compatible property for output tensor names."""
        return self.get_output_names()

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Backward-compatible property for first input tensor shape.

        Returns default (1, 3, 224, 224) if no input tensors specified.
        """
        shape = self.get_input_shape()
        return shape if shape else (self.batch_size, 3, 224, 224)

    def generate_dummy_inputs(self) -> Any:
        """Generate dummy input tensors from input_tensors specs.

        Returns:
            dict[str, torch.Tensor] mapping input names to tensors.
            Int types filled with ones, float types with random [0, 1].

        Raises:
            ValueError: If input_tensors is empty or has no valid specs.
        """
        if not self.input_tensors:
            raise ValueError("input_tensors must be populated to generate dummy inputs.")
        inputs = {t.name: t.to_tensor() for t in self.input_tensors if t.name and t.shape}
        if not inputs:
            raise ValueError("No valid InputTensorSpec found (need name + shape).")
        return inputs

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "opset_version": self.opset_version,
            "batch_size": self.batch_size,
            "export_params": self.export_params,
            "do_constant_folding": self.do_constant_folding,
            "verbose": self.verbose,
            "dynamo": self.dynamo,
            "enable_hierarchy_tags": self.enable_hierarchy_tags,
            "clean_onnx": self.clean_onnx,
            "hierarchy_tag_format": self.hierarchy_tag_format,
        }

        if self.input_tensors:
            result["input_tensors"] = [spec.to_dict() for spec in self.input_tensors]

        if self.output_tensors:
            result["output_tensors"] = [spec.to_dict() for spec in self.output_tensors]

        if self.dynamic_axes:
            result["dynamic_axes"] = self.dynamic_axes

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WinMLExportConfig:
        """Create from dictionary, ignoring unknown fields.

        Args:
            data: Configuration dictionary.

        Returns:
            WinMLExportConfig instance.
        """
        # Parse input_tensors if present
        input_tensors = None
        raw_inputs = data.get("input_tensors")
        if raw_inputs:
            input_tensors = [
                InputTensorSpec.from_dict(spec) if isinstance(spec, dict) else spec
                for spec in raw_inputs
            ]

        # Parse output_tensors if present
        output_tensors = None
        raw_outputs = data.get("output_tensors")
        if raw_outputs:
            output_tensors = [
                OutputTensorSpec.from_dict(spec) if isinstance(spec, dict) else spec
                for spec in raw_outputs
            ]

        return cls(
            opset_version=data.get("opset_version", 17),
            batch_size=data.get("batch_size", 1),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            dynamic_axes=data.get("dynamic_axes"),
            export_params=data.get("export_params", True),
            do_constant_folding=data.get("do_constant_folding", True),
            verbose=data.get("verbose", False),
            dynamo=data.get("dynamo", False),
            enable_hierarchy_tags=data.get("enable_hierarchy_tags", True),
            clean_onnx=data.get("clean_onnx", False),
            hierarchy_tag_format=data.get("hierarchy_tag_format", "full"),
        )


def _resolve_export_config_from_specs(
    model_type: str,
    task: str,
    hf_config: PretrainedConfig,
    *,
    library_name: str = "transformers",
    model_id: str | None = None,
    batch_size: int = 1,
    int_dtype: str = "int32",
    float_dtype: str = "fp32",
    **shape_kwargs: Any,
) -> WinMLExportConfig:
    """Low-level: resolve export config from pre-resolved model specs.

    Wraps ``resolve_io_specs()`` and builds a ``WinMLExportConfig``
    with properly typed ``InputTensorSpec`` / ``OutputTensorSpec`` entries.

    Requires caller to have already resolved model_type, task, hf_config
    via ``resolve_loader_config()``.

    Args:
        model_type: Model type for OnnxConfig lookup (may be sub-model type).
        task: Resolved task name (e.g., "feature-extraction").
        hf_config: HF config for dimensions (may be sub-config for multimodal).
        library_name: Source library (default: "transformers").
        model_id: HF model ID for preprocessor_config.json (correct image sizes).
        batch_size: Batch size for input shapes (default: 1 for QNN compatibility).
        int_dtype: Integer dtype for text inputs (default: "int32").
        float_dtype: Float dtype for vision inputs (default: "fp32").
        **shape_kwargs: Shape overrides (sequence_length, height, width, etc.).

    Returns:
        WinMLExportConfig with input_tensors and output_tensors populated.

    Raises:
        ValueError: If no OnnxConfig is registered for the model_type/task.
    """
    from .io import resolve_io_specs as _resolve_io_specs

    io_specs = _resolve_io_specs(
        model_type,
        task,
        hf_config,
        library_name=library_name,
        model_id=model_id,
        batch_size=batch_size,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
        **shape_kwargs,
    )

    # Build input tensor specs
    input_tensors = None
    if io_specs.get("input_names"):
        input_names = io_specs["input_names"]
        input_shapes = io_specs.get("input_shapes", [None] * len(input_names))
        input_dtypes = io_specs.get("input_dtypes", [None] * len(input_names))

        if len(input_shapes) != len(input_names) or len(input_dtypes) != len(input_names):
            logger.warning(
                "I/O spec length mismatch: names=%d, shapes=%d, dtypes=%d. Using available data.",
                len(input_names),
                len(input_shapes),
                len(input_dtypes),
            )

        value_ranges = io_specs.get("value_ranges", {})

        input_tensors = [
            InputTensorSpec(
                name=name,
                shape=shape,
                dtype=dtype,
                value_range=value_ranges.get(name),
            )
            for name, shape, dtype in zip(input_names, input_shapes, input_dtypes, strict=False)
        ]

    output_tensors = None
    if io_specs.get("output_names"):
        output_tensors = [OutputTensorSpec(name=name) for name in io_specs["output_names"]]

    return WinMLExportConfig(
        input_tensors=input_tensors,
        output_tensors=output_tensors,
    )


def resolve_export_config(
    model_id: str | None = None,
    *,
    task: str | None = None,
    model_class: str | None = None,
    model_type: str | None = None,
    shape_config: dict | None = None,
    library_name: str = "transformers",
    trust_remote_code: bool = False,
) -> tuple[WinMLExportConfig, WinMLLoaderConfig]:
    """Resolve export and loader config for a HuggingFace model.

    Combines loader resolution (task, model_type, hf_config) with
    export I/O specification (input_tensors, output_tensors) in a
    single call.

    Args:
        model_id: HuggingFace model ID or local path.
        task: Task name (auto-detected if None).
        model_class: Explicit model class name.
        model_type: Explicit model type override.
        shape_config: Shape overrides (sequence_length, height, width).
        library_name: Source library (default: "transformers").
        trust_remote_code: Whether to trust remote code.

    Returns:
        (WinMLExportConfig, WinMLLoaderConfig)
    """
    from ..loader import resolve_loader_config

    loader_config, hf_config, _ = resolve_loader_config(
        model_id,
        task=task,
        model_class=model_class,
        model_type=model_type,
        trust_remote_code=trust_remote_code,
        library_name=library_name,
    )

    export_config = _resolve_export_config_from_specs(
        model_type=loader_config.model_type,
        task=loader_config.task,
        hf_config=hf_config,
        library_name=library_name,
        model_id=model_id,
        batch_size=WinMLExportConfig().batch_size,
        **(shape_config or {}),
    )

    return export_config, loader_config


__all__ = [
    "InputTensorSpec",
    "OutputTensorSpec",
    "WinMLExportConfig",
    "resolve_export_config",
]
