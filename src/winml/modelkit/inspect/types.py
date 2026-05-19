# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Data structures for the inspect command."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SupportLevel(Enum):
    """Support level for each component."""

    SUPPORTED = "supported"  # Explicit config exists in WinML CLI
    DEFAULT = "default"  # Using framework defaults (TasksManager/Optimum)
    GENERIC = "generic"  # Using generic fallback class
    UNSUPPORTED = "unsupported"  # No viable path found


@dataclass
class TensorInfo:
    """Information about a tensor."""

    name: str
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    shape_desc: str | None = None  # Human-readable shape like "[B, 3, 224, 224]"
    dynamic_axes: dict[int, str] | None = None  # {0: "batch", 1: "sequence"}
    value_range: tuple[float, float] | None = None  # e.g., (0.0, 1.0) for pixel values


@dataclass
class LoaderInfo:
    """Information about loader configuration."""

    hf_model_class: str  # e.g., "CLIPTextModelWithProjection"
    hf_model_class_source: str  # "HF_MODEL_CLASS_MAPPING" | "HF_TASK_DEFAULTS" | "TasksManager"
    support_level: SupportLevel


@dataclass
class ExporterInfo:
    """Information about exporter configuration."""

    onnx_config_class: str | None  # e.g., "CLIPTextModelIOConfig"
    onnx_config_source: str  # "MODEL_BUILD_CONFIGS" | "TasksManager" | "none"
    support_level: SupportLevel
    input_tensors: list[TensorInfo] = field(default_factory=list)
    output_tensors: list[TensorInfo] = field(default_factory=list)
    opset_version: int = 17


@dataclass
class WinMLInfo:
    """Information about WinML inference class."""

    winml_class: str  # e.g., "WinMLModelForImageClassification"
    winml_class_source: str  # "WINML_MODEL_CLASS_MAPPING" | "TASK_TO_WINML_CLASS" | "Generic"
    support_level: SupportLevel


@dataclass
class ModuleInfo:
    """Information about a single HF module in the hierarchy."""

    name: str  # e.g., "text_model.encoder.layers.0"
    class_name: str  # e.g., "CLIPEncoderLayer"
    module_path: str  # Full module path for display
    depth: int  # Nesting depth (0 = root)
    num_parameters: int = 0  # Number of parameters in this module
    children: list["ModuleInfo"] = field(default_factory=list)


@dataclass
class HierarchyInfo:
    """Information about the HF module hierarchy."""

    root_class: str  # e.g., "CLIPTextModelWithProjection"
    total_parameters: int  # Total model parameters
    hf_modules: list[ModuleInfo]  # List of HF-specific modules (tree structure)
    hf_module_count: int  # Count of HF modules
    nn_module_count: int  # Count of filtered torch.nn modules (for reference)


@dataclass
class ProcessorInfo:
    """Information about data processing classes."""

    processor_class: str | None = None  # e.g., "CLIPProcessor"
    tokenizer_class: str | None = None  # e.g., "CLIPTokenizerFast"
    image_processor_class: str | None = None  # e.g., "CLIPImageProcessor"
    feature_extractor_class: str | None = None  # e.g., "Wav2Vec2FeatureExtractor"
    # Source tracking for transparency (e.g., ResNet -> ConvNextImageProcessorFast)
    processor_source: str | None = None  # "hub_config" | "auto_class"
    image_processor_source: str | None = None
    feature_extractor_source: str | None = None
    tokenizer_source: str | None = None


@dataclass
class IOConfigInfo:
    """Input/Output configuration from model config."""

    # Text-related
    max_position_embeddings: int | None = None  # Max sequence length
    vocab_size: int | None = None

    # Vision-related
    image_size: int | tuple[int, int] | None = None
    patch_size: int | None = None
    num_channels: int | None = None

    # Audio-related
    sampling_rate: int | None = None

    # General
    hidden_size: int | None = None
    hidden_sizes: list[int] | None = None  # Per-stage hidden dims (e.g., ResNet)

    # Extra attrs discovered dynamically from OnnxConfig
    extra: dict[str, Any] | None = None


@dataclass
class CacheStageInfo:
    """Information about a cached pipeline stage."""

    stage: str  # "export" | "optimize" | "quantize" | "compile"
    cached: bool  # Whether this stage is cached
    path: str | None = None  # Path to cached file
    size_mb: float | None = None  # Size in MB
    created: str | None = None  # Creation timestamp


@dataclass
class CacheInfo:
    """Information about cached artifacts for a model."""

    cache_dir: str  # Cache directory path
    stages: list[CacheStageInfo]  # Info for each pipeline stage
    total_cached: int = 0  # Number of cached stages
    total_size_mb: float = 0.0  # Total size of cached artifacts


@dataclass
class InspectResult:
    """Complete inspection result for a model."""

    # Model identification
    model_id: str
    model_type: str
    architectures: list[str]
    task: str
    task_source: str  # "TasksManager" | "HF_MODEL_CLASS_MAPPING" | "explicit"

    # Component info
    loader: LoaderInfo
    exporter: ExporterInfo
    winml: WinMLInfo

    # Overall support status
    overall_support: SupportLevel
    support_notes: list[str] = field(default_factory=list)

    # Raw configs (for verbose/JSON output)
    build_config: dict[str, Any] | None = None

    # Module hierarchy (only populated with --hierarchy flag)
    hierarchy: HierarchyInfo | None = None

    # Cache information
    cache: CacheInfo | None = None

    # Processor information
    processor: ProcessorInfo | None = None

    # IO configuration from model config
    io_config: IOConfigInfo | None = None
