"""Resolution logic for inspect command.

Leverages existing loader, export, and models modules - NO NEW CONFIG LOGIC.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..loader.task import (
    HF_TASK_DEFAULTS,
    _detect_task_from_config,
    _get_custom_model_class,
)
from ..models import (
    HF_MODEL_CLASS_MAPPING,
    MODEL_BUILD_CONFIGS,
    TASK_TO_WINML_CLASS,
    WINML_MODEL_CLASS_MAPPING,
)
from .types import (
    CacheInfo,
    CacheStageInfo,
    ExporterInfo,
    IOConfigInfo,
    LoaderInfo,
    ProcessorInfo,
    SupportLevel,
    TensorInfo,
    WinMLInfo,
)


if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PretrainedConfig

    from ..config import WinMLBuildConfig

logger = logging.getLogger(__name__)

# Mapping from pipeline stage verbs to the filenames build_hf_model() produces.
# "export" is omitted because its stage name equals its filename — the
# .get(stage, stage) fallback handles it.  Used only in the legacy
# filename-scanning path; manifest-based resolution reads filenames directly.
_STAGE_TO_FILENAME = {
    "optimize": "optimized",
    "quantize": "quantized",
    "compile": "compiled",
}


def _get_known_tasks() -> set[str]:
    """Collect all known task strings from internal mappings and TasksManager.

    Returns:
        Set of known task strings.
    """
    tasks: set[str] = set()

    # From HF_MODEL_CLASS_MAPPING values (task part of each (model_type, task) key)
    for _model_type, task in HF_MODEL_CLASS_MAPPING:
        tasks.add(task)

    # From HF_TASK_DEFAULTS keys
    tasks.update(HF_TASK_DEFAULTS.keys())

    # From optimum TasksManager if available
    try:
        from optimum.exporters.tasks import TasksManager

        if hasattr(TasksManager, "_TASKS_TO_LIBRARY"):
            tasks.update(TasksManager._TASKS_TO_LIBRARY.keys())
        if hasattr(TasksManager, "_TASKS_TO_AUTOMODELS"):
            tasks.update(TasksManager._TASKS_TO_AUTOMODELS.keys())
    except Exception:
        pass

    return tasks


def validate_task(task: str) -> None:
    """Validate that a task string is a known task.

    Args:
        task: Task string to validate.

    Raises:
        ValueError: If the task is not recognized.
    """
    known = _get_known_tasks()
    if task not in known:
        sorted_tasks = sorted(known)
        raise ValueError(
            f"Unknown task '{task}'. Known tasks: {', '.join(sorted_tasks)}"
        )


def detect_task(config: PretrainedConfig) -> tuple[str, str]:
    """Detect task from HF config.

    Args:
        config: HuggingFace PretrainedConfig

    Returns:
        Tuple of (task_name, detection_source)
    """
    model_type = getattr(config, "model_type", "unknown")
    model_type_normalized = model_type.lower().replace("_", "-")

    # Check if we have explicit mapping for this model_type
    for mt, task in HF_MODEL_CLASS_MAPPING:
        if mt == model_type_normalized:
            return task, "HF_MODEL_CLASS_MAPPING"

    # Use TasksManager detection
    try:
        task = _detect_task_from_config(config)
        return task, "TasksManager"
    except ValueError:
        pass

    # Fallback to task defaults
    if HF_TASK_DEFAULTS:
        first_task = next(iter(HF_TASK_DEFAULTS.keys()))
        return first_task, "HF_TASK_DEFAULTS"

    return "unknown", "none"


def resolve_loader(model_type: str, task: str) -> LoaderInfo:
    """Resolve loader configuration for a model.

    Uses _get_custom_model_class() from loader/task.py which looks up
    MODEL_CLASS_MAPPING for (model_type, task) overrides.

    Args:
        model_type: HuggingFace model type (e.g., "clip")
        task: Canonical task name (e.g., "feature-extraction")

    Returns:
        LoaderInfo with class name, source, and support level
    """
    model_type_normalized = model_type.lower().replace("_", "-")

    # Use existing _get_custom_model_class() which does the lookup
    model_class = _get_custom_model_class(model_type_normalized, task)

    if model_class:
        # Determine source - check which mapping it came from
        key = (model_type_normalized, task)
        if key in HF_MODEL_CLASS_MAPPING:
            return LoaderInfo(
                hf_model_class=model_class.__name__,
                hf_model_class_source="MODEL_CLASS_MAPPING",
                support_level=SupportLevel.SUPPORTED,
            )
        if task in HF_TASK_DEFAULTS:
            return LoaderInfo(
                hf_model_class=model_class.__name__,
                hf_model_class_source="HF_TASK_DEFAULTS",
                support_level=SupportLevel.DEFAULT,
            )

    # Fallback to TasksManager default
    return LoaderInfo(
        hf_model_class="Auto (TasksManager)",
        hf_model_class_source="TasksManager",
        support_level=SupportLevel.DEFAULT,
    )


def _extract_tensor_specs_from_onnx_config(
    onnx_config_cls,
    hf_config: PretrainedConfig,
) -> tuple[list[TensorInfo], list[TensorInfo]]:
    """Extract tensor specifications from an ONNX config class.

    Uses the ONNX config's generate_dummy_inputs() to get actual tensor shapes,
    and the inputs/outputs properties for dynamic axes information.

    Args:
        onnx_config_cls: ONNX config constructor (may be functools.partial)
        hf_config: HuggingFace PretrainedConfig for shape bounds

    Returns:
        Tuple of (input_tensors, output_tensors)
    """
    input_tensors: list[TensorInfo] = []
    output_tensors: list[TensorInfo] = []

    try:
        # Instantiate ONNX config with HF config
        onnx_config = onnx_config_cls(hf_config)

        # Generate dummy inputs to get actual shapes
        dummy_inputs: dict = {}
        try:
            dummy_inputs = onnx_config.generate_dummy_inputs(framework="pt")
        except Exception as e:
            logger.debug("Failed to generate dummy inputs: %s", e)

        # Helper to convert shape to description with dynamic axis markers
        def shape_to_desc(
            shape: tuple | list | None, dynamic_axes: dict[int, str]
        ) -> str:
            """Convert tensor shape to human-readable string with dynamic markers."""
            if shape is None:
                # Fallback: use dynamic axes only
                parts = []
                for _idx, axis_name in sorted(dynamic_axes.items()):
                    if "batch" in axis_name.lower():
                        parts.append("B")
                    else:
                        parts.append(axis_name)
                return f"[{', '.join(parts)}]" if parts else "[]"

            parts = []
            for i, dim in enumerate(shape):
                if i in dynamic_axes:
                    axis_name = dynamic_axes[i].lower()
                    if "batch" in axis_name:
                        parts.append("B")
                    elif "sequence" in axis_name:
                        parts.append("S")
                    elif "height" in axis_name or "width" in axis_name:
                        parts.append(str(dim))  # Use actual size
                    else:
                        parts.append(str(dim))
                else:
                    parts.append(str(dim))
            return f"[{', '.join(parts)}]"

        # Standard input dtypes based on tensor name patterns
        def infer_dtype(name: str) -> str:
            name_lower = name.lower()
            if "ids" in name_lower or "label" in name_lower:
                return "int64"
            if "mask" in name_lower and "pixel" not in name_lower:
                return "int64"
            return "float32"

        # Process inputs - use actual shapes from dummy inputs
        if hasattr(onnx_config, "inputs"):
            for name, axes in onnx_config.inputs.items():
                shape = None
                if name in dummy_inputs:
                    shape = tuple(dummy_inputs[name].shape)
                shape_desc = shape_to_desc(shape, axes)
                dtype = infer_dtype(name)
                input_tensors.append(
                    TensorInfo(
                        name=name,
                        dtype=dtype,
                        shape_desc=shape_desc,
                        dynamic_axes=dict(axes),
                    )
                )

        # Process outputs - we don't have actual shapes, use dynamic axes
        if hasattr(onnx_config, "outputs"):
            for name, axes in onnx_config.outputs.items():
                shape_desc = shape_to_desc(None, axes)
                output_tensors.append(
                    TensorInfo(
                        name=name,
                        shape_desc=shape_desc,
                        dynamic_axes=dict(axes),
                    )
                )

    except Exception as e:
        logger.debug("Failed to extract tensor specs from ONNX config: %s", e)

    return input_tensors, output_tensors


def resolve_exporter(
    model_type: str,
    task: str,
    hf_config: PretrainedConfig | None = None,
) -> ExporterInfo:
    """Resolve exporter configuration for a model.

    Uses MODEL_BUILD_CONFIGS registry from models/__init__.py.

    Args:
        model_type: HuggingFace model type (e.g., "clip")
        task: Canonical task name
        hf_config: Optional HuggingFace config for extracting tensor shapes

    Returns:
        ExporterInfo with ONNX config, tensors, and support level
    """
    model_type_normalized = model_type.lower().replace("_", "-")

    # Check MODEL_BUILD_CONFIGS for predefined config
    if model_type_normalized in MODEL_BUILD_CONFIGS:
        config: WinMLBuildConfig = MODEL_BUILD_CONFIGS[model_type_normalized]
        export_config = config.export

        # Extract input tensors
        input_tensors: list[TensorInfo] = []
        if export_config.input_tensors:
            input_tensors.extend(
                TensorInfo(
                    name=spec.name or "unknown",
                    dtype=spec.dtype,
                    shape=spec.shape,
                )
                for spec in export_config.input_tensors
            )

        # Extract output tensors
        output_tensors: list[TensorInfo] = []
        if export_config.output_tensors:
            output_tensors.extend(
                TensorInfo(name=spec.name or "unknown")
                for spec in export_config.output_tensors
            )

        return ExporterInfo(
            onnx_config_class=f"{model_type_normalized.upper()}IOConfig",
            onnx_config_source="MODEL_BUILD_CONFIGS",
            support_level=SupportLevel.SUPPORTED,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            opset_version=export_config.opset_version,
        )

    # Check if TasksManager supports this model_type
    try:
        # Import model_configs to trigger registration of ONNX configs via decorators
        import optimum.exporters.onnx.model_configs  # noqa: F401
        from optimum.exporters.tasks import TasksManager

        # TasksManager uses underscores (sam2_video), not hyphens (sam2-video)
        # Use original model_type for TasksManager lookup
        onnx_config_cls = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            model_type=model_type,
            task=task,
            library_name="transformers",
        )
        if onnx_config_cls:
            # Handle functools.partial returned by TasksManager
            import functools

            if isinstance(onnx_config_cls, functools.partial):
                config_name = onnx_config_cls.func.__name__
            else:
                config_name = onnx_config_cls.__name__

            # Extract tensor specs from ONNX config if HF config is available
            input_tensors: list[TensorInfo] = []
            output_tensors: list[TensorInfo] = []

            if hf_config is not None:
                input_tensors, output_tensors = _extract_tensor_specs_from_onnx_config(
                    onnx_config_cls, hf_config
                )

            return ExporterInfo(
                onnx_config_class=config_name,
                onnx_config_source="TasksManager",
                support_level=SupportLevel.DEFAULT,
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                opset_version=17,
            )
    except Exception as e:
        logger.debug("TasksManager lookup failed for %s/%s: %s", model_type, task, e)

    # Unsupported
    return ExporterInfo(
        onnx_config_class=None,
        onnx_config_source="none",
        support_level=SupportLevel.UNSUPPORTED,
        input_tensors=[],
        output_tensors=[],
        opset_version=17,
    )


def resolve_winml(model_type: str, task: str) -> WinMLInfo:
    """Resolve WinML inference class for a model.

    Uses the three-level mapping from models/winml/__init__.py:
    1. WINML_MODEL_CLASS_MAPPING (specialized)
    2. TASK_TO_WINML_CLASS (by task)
    3. WinMLModelForGenericTask (fallback)

    Args:
        model_type: HuggingFace model type (e.g., "clip")
        task: Canonical task name

    Returns:
        WinMLInfo with class name, source, and support level
    """
    model_type_normalized = model_type.lower().replace("_", "-")

    # Level 1: Check WINML_MODEL_CLASS_MAPPING (specialized)
    key = (model_type_normalized, task)
    if key in WINML_MODEL_CLASS_MAPPING:
        return WinMLInfo(
            winml_class=WINML_MODEL_CLASS_MAPPING[key],
            winml_class_source="WINML_MODEL_CLASS_MAPPING",
            support_level=SupportLevel.SUPPORTED,
        )

    # Level 2: Check TASK_TO_WINML_CLASS (by task)
    if task in TASK_TO_WINML_CLASS:
        return WinMLInfo(
            winml_class=TASK_TO_WINML_CLASS[task],
            winml_class_source="TASK_TO_WINML_CLASS",
            support_level=SupportLevel.DEFAULT,
        )

    # Level 3: Generic fallback
    return WinMLInfo(
        winml_class="WinMLModelForGenericTask",
        winml_class_source="Generic",
        support_level=SupportLevel.GENERIC,
    )


def compile_support_status(
    loader: LoaderInfo,
    exporter: ExporterInfo,
    winml: WinMLInfo,
) -> tuple[SupportLevel, list[str]]:
    """Compile overall support status from component statuses.

    Args:
        loader: LoaderInfo
        exporter: ExporterInfo
        winml: WinMLInfo

    Returns:
        Tuple of (overall_support_level, support_notes)
    """
    notes: list[str] = []

    # Collect notes for non-optimal components
    if loader.support_level == SupportLevel.UNSUPPORTED:
        notes.append("Loader: Model class not found in registry")
    elif loader.support_level == SupportLevel.DEFAULT:
        notes.append("Loader: Using TasksManager defaults")

    if exporter.support_level == SupportLevel.UNSUPPORTED:
        notes.append("Exporter: No ONNX config available")
    elif exporter.support_level == SupportLevel.DEFAULT:
        notes.append("Exporter: Using TasksManager defaults")

    if winml.support_level == SupportLevel.GENERIC:
        notes.append("WinML: Using generic inference class")
    elif winml.support_level == SupportLevel.DEFAULT:
        notes.append("WinML: Using task-based class")

    # Determine overall status
    levels = [loader.support_level, exporter.support_level, winml.support_level]

    if SupportLevel.UNSUPPORTED in levels:
        return SupportLevel.UNSUPPORTED, notes
    if all(level == SupportLevel.SUPPORTED for level in levels):
        return SupportLevel.SUPPORTED, notes
    return SupportLevel.DEFAULT, notes


def get_build_config(model_type: str) -> dict | None:
    """Get the full build config for a model type.

    Args:
        model_type: HuggingFace model type (e.g., "clip")

    Returns:
        Build config as dict, or None if not found
    """
    model_type_normalized = model_type.lower().replace("_", "-")

    if model_type_normalized in MODEL_BUILD_CONFIGS:
        config = MODEL_BUILD_CONFIGS[model_type_normalized]
        return config.to_dict()

    return None


def resolve_cache(model_id: str) -> CacheInfo:
    """Resolve cache status for a model.

    Uses build manifests as the primary resolution path when available,
    falling back to filename-scanning for pre-manifest builds.

    Args:
        model_id: HuggingFace model identifier (e.g., "openai/clip-vit-base-patch32")

    Returns:
        CacheInfo with status for each pipeline stage
    """
    import json

    from ..cache import get_cache_dir, get_model_dir

    cache_dir = get_cache_dir()
    model_dir = get_model_dir(model_id, cache_dir=cache_dir)

    stages: list[CacheStageInfo] = []
    total_cached = 0
    total_size_mb = 0.0

    # Pipeline stages to check
    pipeline_stages = ["export", "optimize", "quantize", "compile"]

    # -------------------------------------------------------------------------
    # PRIMARY: Manifest-based resolution
    # -------------------------------------------------------------------------
    if model_dir.exists():
        manifests = list(model_dir.glob("*build_manifest.json"))
        if manifests:
            # Use the most recent manifest (by mtime) when multiple variants exist
            manifest_path = max(manifests, key=lambda p: p.stat().st_mtime)
            try:
                manifest = json.loads(manifest_path.read_text())
                manifest_stages = {s["name"]: s for s in manifest.get("stages", [])}

                for stage in pipeline_stages:
                    ms = manifest_stages.get(stage)
                    if ms and ms.get("status") == "completed":
                        filename = ms.get("filename")
                        artifact = model_dir / filename if filename else None
                        size_bytes = (
                            artifact.stat().st_size
                            if artifact and artifact.exists()
                            else 0
                        )
                        stage_info = CacheStageInfo(
                            stage=stage,
                            cached=True,
                            path=str(artifact) if artifact else None,
                            size_mb=round(size_bytes / (1024 * 1024), 2),
                        )
                        total_cached += 1
                        total_size_mb += stage_info.size_mb or 0.0
                    else:
                        stage_info = CacheStageInfo(stage=stage, cached=False)

                    stages.append(stage_info)

                return CacheInfo(
                    cache_dir=str(cache_dir),
                    stages=stages,
                    total_cached=total_cached,
                    total_size_mb=round(total_size_mb, 2),
                )
            except (json.JSONDecodeError, KeyError, OSError) as exc:
                logger.debug("Failed to read manifest %s: %s", manifest_path, exc)
                # Fall through to filename scanning
                stages = []
                total_cached = 0
                total_size_mb = 0.0

    # -------------------------------------------------------------------------
    # FALLBACK: Filename-scanning for pre-manifest builds
    # -------------------------------------------------------------------------
    cached_files: dict[str, Path] = {}
    if model_dir.exists():
        for f in model_dir.glob("*.onnx"):
            # Parse stage from filename: {cache_key}_{stage}.onnx
            stem = f.stem
            last_sep = stem.rfind("_")
            if last_sep > 0:
                stage_name = stem[last_sep + 1:]
                cached_files[stage_name] = f

    for stage in pipeline_stages:
        # Map stage names to the filenames build_hf_model produces
        stage_file = cached_files.get(stage) or cached_files.get(
            _STAGE_TO_FILENAME.get(stage, stage)
        )
        if stage_file and stage_file.exists():
            size_bytes = stage_file.stat().st_size
            stage_info = CacheStageInfo(
                stage=stage,
                cached=True,
                path=str(stage_file),
                size_mb=round(size_bytes / (1024 * 1024), 2),
            )
            total_cached += 1
            total_size_mb += stage_info.size_mb or 0.0
        else:
            stage_info = CacheStageInfo(stage=stage, cached=False)

        stages.append(stage_info)

    return CacheInfo(
        cache_dir=str(cache_dir),
        stages=stages,
        total_cached=total_cached,
        total_size_mb=round(total_size_mb, 2),
    )


def resolve_io_config(config: PretrainedConfig) -> IOConfigInfo:
    """Extract IO configuration from HuggingFace config.

    Extracts IO-related configuration values from a PretrainedConfig object.
    For multimodal models (like CLIP), also checks nested configs (text_config,
    vision_config) to gather all relevant settings.

    Args:
        config: HuggingFace PretrainedConfig object

    Returns:
        IOConfigInfo with extracted configuration values
    """
    # Helper to get attribute from config or nested configs
    def get_config_attr(
        attr_name: str,
        nested_configs: list[str] | None = None,
    ) -> int | tuple[int, int] | None:
        """Get attribute from main config or nested configs.

        Args:
            attr_name: Attribute name to look for
            nested_configs: List of nested config names to check (e.g., ["text_config"])

        Returns:
            Attribute value or None if not found
        """
        # First check the main config
        value = getattr(config, attr_name, None)
        if value is not None:
            return value

        # Check nested configs if provided
        if nested_configs:
            for nested_name in nested_configs:
                nested_config = getattr(config, nested_name, None)
                if nested_config is not None:
                    value = getattr(nested_config, attr_name, None)
                    if value is not None:
                        return value

        return None

    # Text-related attributes - check main and text_config
    max_position_embeddings = get_config_attr(
        "max_position_embeddings", ["text_config"]
    )
    vocab_size = get_config_attr("vocab_size", ["text_config"])

    # Vision-related attributes - check main and vision_config
    image_size = get_config_attr("image_size", ["vision_config"])
    patch_size = get_config_attr("patch_size", ["vision_config"])
    num_channels = get_config_attr("num_channels", ["vision_config"])

    # Audio-related attributes - check main and audio_config
    sampling_rate = get_config_attr("sampling_rate", ["audio_config"])

    # General attributes - check main config only
    hidden_size = get_config_attr("hidden_size", ["text_config", "vision_config"])

    return IOConfigInfo(
        max_position_embeddings=max_position_embeddings,
        vocab_size=vocab_size,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        sampling_rate=sampling_rate,
        hidden_size=hidden_size,
    )


def resolve_processor(model_id: str) -> ProcessorInfo:
    """Resolve data processing classes for a HuggingFace model.

    Detects the processor/tokenizer/image_processor/feature_extractor classes
    associated with a model. Uses a multi-strategy approach:

    1. First tries to fetch config files from HuggingFace Hub without downloading
       the full model (fast, no dependencies)
    2. Uses Auto classes to fill in any missing information that wasn't found
       in the config files

    Args:
        model_id: HuggingFace model identifier (e.g., "openai/clip-vit-base-patch32")

    Returns:
        ProcessorInfo with detected class names for each processor type
    """
    processor_class: str | None = None
    tokenizer_class: str | None = None
    image_processor_class: str | None = None
    feature_extractor_class: str | None = None

    # Strategy 1: Try to get class names from config files via HuggingFace Hub API
    # This is fast and doesn't require downloading/instantiating processors
    try:
        processor_class, tokenizer_class, image_processor_class, feature_extractor_class = (
            _resolve_processor_from_hub_configs(model_id)
        )
    except Exception as e:
        logger.debug("Failed to resolve processors from hub configs: %s", e)

    # Strategy 2: Use Auto classes to fill in any missing information
    # This approach actually loads the processors, so it's slower but more reliable
    try:
        (
            auto_processor,
            auto_tokenizer,
            auto_image_processor,
            auto_feature_extractor,
        ) = _resolve_processor_from_auto_classes(model_id)

        # Fill in missing values from auto classes
        if processor_class is None:
            processor_class = auto_processor
        if tokenizer_class is None:
            tokenizer_class = auto_tokenizer
        if image_processor_class is None:
            image_processor_class = auto_image_processor
        if feature_extractor_class is None:
            feature_extractor_class = auto_feature_extractor
    except Exception as e:
        logger.debug("Failed to resolve processors from auto classes: %s", e)

    return ProcessorInfo(
        processor_class=processor_class,
        tokenizer_class=tokenizer_class,
        image_processor_class=image_processor_class,
        feature_extractor_class=feature_extractor_class,
    )


def _resolve_processor_from_hub_configs(
    model_id: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Resolve processor classes by fetching config files from HuggingFace Hub.

    This approach is fast because it only downloads small JSON config files,
    not the full model weights or processor files.

    Args:
        model_id: HuggingFace model identifier

    Returns:
        Tuple of (processor_class, tokenizer_class, image_processor_class, feature_extractor_class)
    """
    import json
    from pathlib import Path

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    processor_class: str | None = None
    tokenizer_class: str | None = None
    image_processor_class: str | None = None
    feature_extractor_class: str | None = None

    # Try to download and parse preprocessor_config.json
    # This file contains image_processor_type or processor_class
    try:
        preprocessor_config_path = hf_hub_download(
            repo_id=model_id,
            filename="preprocessor_config.json",
        )
        with Path(preprocessor_config_path).open(encoding="utf-8") as f:
            preprocessor_config = json.load(f)

        # Check for processor_class (multimodal models like CLIP)
        if "processor_class" in preprocessor_config:
            processor_class = preprocessor_config["processor_class"]

        # Check for image_processor_type (vision models)
        if "image_processor_type" in preprocessor_config:
            image_processor_class = preprocessor_config["image_processor_type"]

        # Check for feature_extractor_type (audio/legacy vision models)
        if "feature_extractor_type" in preprocessor_config:
            feature_extractor_class = preprocessor_config["feature_extractor_type"]

    except (EntryNotFoundError, RepositoryNotFoundError, OSError):
        logger.debug("preprocessor_config.json not found for %s", model_id)
    except json.JSONDecodeError as e:
        logger.debug("Failed to parse preprocessor_config.json for %s: %s", model_id, e)

    # Try to download and parse tokenizer_config.json
    # This file contains tokenizer_class
    try:
        tokenizer_config_path = hf_hub_download(
            repo_id=model_id,
            filename="tokenizer_config.json",
        )
        with Path(tokenizer_config_path).open(encoding="utf-8") as f:
            tokenizer_config = json.load(f)

        # Check for tokenizer_class
        if "tokenizer_class" in tokenizer_config:
            tokenizer_class = tokenizer_config["tokenizer_class"]

    except (EntryNotFoundError, RepositoryNotFoundError, OSError):
        logger.debug("tokenizer_config.json not found for %s", model_id)
    except json.JSONDecodeError as e:
        logger.debug("Failed to parse tokenizer_config.json for %s: %s", model_id, e)

    return processor_class, tokenizer_class, image_processor_class, feature_extractor_class


def _resolve_processor_from_auto_classes(
    model_id: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Resolve processor classes by instantiating HuggingFace Auto classes.

    This is a fallback approach that actually loads the processors. It's slower
    but more reliable for models with non-standard configurations.

    Args:
        model_id: HuggingFace model identifier

    Returns:
        Tuple of (processor_class, tokenizer_class, image_processor_class, feature_extractor_class)
    """
    processor_class: str | None = None
    tokenizer_class: str | None = None
    image_processor_class: str | None = None
    feature_extractor_class: str | None = None

    # Try AutoProcessor first - for multimodal models
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        processor_class = type(processor).__name__

        # AutoProcessor may wrap tokenizer and image_processor
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            tokenizer_class = type(processor.tokenizer).__name__

        if hasattr(processor, "image_processor") and processor.image_processor is not None:
            image_processor_class = type(processor.image_processor).__name__

        # Some older models use feature_extractor instead of image_processor
        if hasattr(processor, "feature_extractor") and processor.feature_extractor is not None:
            feature_extractor_class = type(processor.feature_extractor).__name__

    except Exception as e:
        logger.debug("AutoProcessor failed for %s: %s", model_id, e)

    # Try AutoTokenizer if we don't have tokenizer yet
    if tokenizer_class is None:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer_class = type(tokenizer).__name__
        except Exception as e:
            logger.debug("AutoTokenizer failed for %s: %s", model_id, e)

    # Try AutoImageProcessor if we don't have image_processor yet
    if image_processor_class is None:
        try:
            from transformers import AutoImageProcessor

            image_processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
            image_processor_class = type(image_processor).__name__
        except Exception as e:
            logger.debug("AutoImageProcessor failed for %s: %s", model_id, e)

    # Try AutoFeatureExtractor if we don't have feature_extractor yet
    if feature_extractor_class is None:
        try:
            from transformers import AutoFeatureExtractor

            feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            feature_extractor_class = type(feature_extractor).__name__
        except Exception as e:
            logger.debug("AutoFeatureExtractor failed for %s: %s", model_id, e)

    return processor_class, tokenizer_class, image_processor_class, feature_extractor_class
