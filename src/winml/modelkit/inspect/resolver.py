# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Resolution logic for inspect command.

Leverages existing loader, export, and models modules - NO NEW CONFIG LOGIC.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NamedTuple

from ..loader.resolution import _get_custom_model_class
from ..loader.task import (
    COMPOSITE_TASKS,
    HF_TASK_DEFAULTS,
    KNOWN_TASKS,
    resolve_optimum_library,
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
    CompositeInfo,
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


def get_known_tasks() -> set[str]:
    """Return the canonical set of task names recognized by inspect.

    Combines the hand-coded :data:`KNOWN_TASKS` with locally registered tasks
    so any future entries in :data:`HF_TASK_DEFAULTS` or
    :data:`HF_MODEL_CLASS_MAPPING` are picked up automatically. Does not
    import ``optimum.exporters`` — that import costs ~10s due to its
    transitive ``transformers`` import and would make ``--list-tasks`` slow.

    Note on the dual path:
        ``winml inspect --list-tasks`` deliberately bypasses this helper and
        reads :data:`KNOWN_TASKS` directly. Going through ``..inspect.resolver``
        would import ``..models`` (which transitively imports ``transformers``)
        and re-introduce the latency this module's hand-coded constant exists
        to avoid. The two paths therefore see slightly different sets:

        * ``--list-tasks``     ->  :data:`KNOWN_TASKS`
        * ``validate_task()``  ->  ``KNOWN_TASKS`` plus :data:`COMPOSITE_TASKS`
                                  plus ``HF_TASK_DEFAULTS`` keys plus
                                  ``HF_MODEL_CLASS_MAPPING`` task entries

        ``tests/unit/loader/test_known_tasks.py`` asserts ``KNOWN_TASKS``
        is a superset of the local registries, so anything ``validate_task``
        accepts also appears in ``--list-tasks``. Drift is a CI failure, not
        a silent break.
    """
    tasks: set[str] = set(KNOWN_TASKS)
    tasks.update(COMPOSITE_TASKS)
    tasks.update(HF_TASK_DEFAULTS.keys())
    tasks.update(task for _, task in HF_MODEL_CLASS_MAPPING if task is not None)
    return tasks


def validate_task(task: str) -> None:
    """Validate that a task string is a known task.

    Args:
        task: Task string to validate.

    Raises:
        ValueError: If the task is not recognized.
    """
    known = get_known_tasks()
    if task not in known:
        sorted_tasks = sorted(known)
        raise ValueError(f"Unknown task '{task}'. Known tasks: {', '.join(sorted_tasks)}")


def resolve_loader(model_type: str, task: str) -> LoaderInfo:
    """Resolve loader configuration for a model.

    Uses _get_custom_model_class() from loader/resolution.py which looks up
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


def _shape_to_desc(shape: tuple | list | None, dynamic_axes: dict[int, str]) -> str:
    """Convert tensor shape to human-readable string with dynamic markers.

    Dynamic axes are shown as the concrete value from dummy inputs,
    distinguishable from static dims by context (batch → "B").
    For non-batch dynamic dims (sequence, height, width), shows the
    concrete value since that's what the model actually uses for export.

    Fixes D-3 from #247: uses axis names directly, no hardcoded abbreviations.
    """
    if shape is None:
        parts = []
        for _idx, axis_name in sorted(dynamic_axes.items()):
            if axis_name.lower() in ("batch", "batch_size"):
                parts.append("B")
            else:
                parts.append(axis_name)
        return f"[{', '.join(parts)}]" if parts else "[]"

    parts = []
    for i, dim in enumerate(shape):
        if i in dynamic_axes:
            axis_name = dynamic_axes[i]
            if axis_name.lower() in ("batch", "batch_size"):
                parts.append("B")
            else:
                # Show concrete value — this is the export shape from
                # preprocessor_config or shape_config, not a placeholder
                parts.append(str(dim))
        else:
            parts.append(str(dim))
    return f"[{', '.join(parts)}]"


def build_tensor_infos_from_io_specs(
    io_specs: dict,
) -> tuple[list[TensorInfo], list[TensorInfo]]:
    """Convert resolve_io_specs() output to TensorInfo lists.

    Single conversion point from config's I/O spec format to inspect's
    TensorInfo dataclass. Eliminates the duplicated extraction logic
    that previously lived in _extract_tensor_specs_from_onnx_config.

    Args:
        io_specs: Dict returned by export/io.py resolve_io_specs()

    Returns:
        Tuple of (input_tensors, output_tensors)
    """
    input_tensors: list[TensorInfo] = []
    output_tensors: list[TensorInfo] = []

    input_names = io_specs.get("input_names", [])
    input_shapes = io_specs.get("input_shapes", [])
    input_dtypes = io_specs.get("input_dtypes", [])
    inputs_axes = io_specs.get("inputs", {})
    value_ranges = io_specs.get("value_ranges", {})

    for i, name in enumerate(input_names):
        shape = input_shapes[i] if i < len(input_shapes) else None
        dtype = input_dtypes[i] if i < len(input_dtypes) else None
        axes = inputs_axes.get(name, {})
        vr = value_ranges.get(name)

        shape_desc = _shape_to_desc(shape, axes) if shape else None

        input_tensors.append(
            TensorInfo(
                name=name,
                dtype=dtype,
                shape=shape,
                shape_desc=shape_desc,
                dynamic_axes=dict(axes) if axes else None,
                value_range=vr,
            )
        )

    output_names = io_specs.get("output_names", [])
    outputs_axes = io_specs.get("outputs", {})

    for name in output_names:
        axes = outputs_axes.get(name, {})
        shape_desc = _shape_to_desc(None, axes) if axes else None
        output_tensors.append(
            TensorInfo(
                name=name,
                shape_desc=shape_desc,
                dynamic_axes=dict(axes) if axes else None,
            )
        )

    return input_tensors, output_tensors


def resolve_exporter(
    model_type: str,
    task: str,
    hf_config: PretrainedConfig | None = None,
    *,
    model_id: str | None = None,
) -> ExporterInfo:
    """Resolve exporter configuration for a model.

    Uses MODEL_BUILD_CONFIGS registry, then falls back to
    export/io.py resolve_io_specs() for I/O extraction. This ensures
    inspect and config share the same battle-tested I/O extraction path,
    including correct image sizes from preprocessor_config.json.

    Args:
        model_type: HuggingFace model type (e.g., "clip")
        task: Canonical task name
        hf_config: Optional HuggingFace config for extracting tensor shapes
        model_id: Optional HuggingFace model ID for preprocessor_config.json
                  (needed for correct image sizes on models like ResNet)

    Returns:
        ExporterInfo with ONNX config, tensors, and support level
    """
    model_type_normalized = model_type.lower().replace("_", "-")

    # Check MODEL_BUILD_CONFIGS for predefined config
    if model_type_normalized in MODEL_BUILD_CONFIGS:
        config: WinMLBuildConfig = MODEL_BUILD_CONFIGS[model_type_normalized]
        # MODEL_BUILD_CONFIGS entries are HF export configs; export is None only on
        # the direct-ONNX build path, which never reaches this registry lookup.
        export_config = config.export
        if export_config is None:
            raise ValueError(
                f"MODEL_BUILD_CONFIGS entry for {model_type_normalized!r} is missing an "
                "export config (export is None only on the direct-ONNX build path)."
            )

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
                TensorInfo(name=spec.name or "unknown") for spec in export_config.output_tensors
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

        # TasksManager expects Optimum-canonical task names
        from ..loader import to_optimum_task

        # TasksManager uses underscores (sam2_video), not hyphens (sam2-video)
        # Use original model_type for TasksManager lookup
        onnx_config_cls = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            model_type=model_type,
            task=to_optimum_task(task),
            library_name=resolve_optimum_library(model_type),
        )
        if onnx_config_cls:
            # Handle functools.partial returned by TasksManager
            import functools

            if isinstance(onnx_config_cls, functools.partial):
                config_name = onnx_config_cls.func.__name__
            else:
                config_name = onnx_config_cls.__name__

            # Extract tensor specs via resolve_io_specs (shared with config command)
            input_tensors = []
            output_tensors = []

            if hf_config is not None:
                try:
                    from ..export.io import resolve_io_specs

                    io_specs = resolve_io_specs(
                        model_type=model_type,
                        task=task,
                        hf_config=hf_config,
                        model_id=model_id,
                    )
                    input_tensors, output_tensors = build_tensor_infos_from_io_specs(io_specs)
                except Exception as e:
                    logger.debug("resolve_io_specs failed for %s/%s: %s", model_type, task, e)

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
    from ..utils.manifest import MANIFEST_FILENAME, WinMLManifest

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
        manifest_paths = sorted(
            model_dir.glob(f"*{MANIFEST_FILENAME}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if manifest_paths:
            try:
                manifest = WinMLManifest.load(manifest_paths[0])
                manifest_stages = {s.name: s for s in manifest.stages}

                for stage in pipeline_stages:
                    ms = manifest_stages.get(stage)
                    if ms and ms.status == "completed":
                        filename = ms.filename
                        artifact = model_dir / filename if filename else None
                        size_bytes = (
                            artifact.stat().st_size if artifact and artifact.exists() else 0
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
                logger.debug("Failed to read manifest %s: %s", manifest_paths[0], exc)
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
                stage_name = stem[last_sep + 1 :]
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


def resolve_composite_info(
    model_type: str, detected_components: dict[str, str] | None = None
) -> CompositeInfo | None:
    """Composite pipeline structure for a *resolved* model, or ``None``.

    Returns ``None`` unless the model's resolved task bridges to a composite — i.e.
    ``detected_components`` (from :attr:`TaskResolution.composite`) is set. This scopes
    the composite view to genuine multi-component / non-runnable-half exports (a seq2seq
    decoder, etc.) and avoids flagging every model_type that merely *could* serve a
    composite pipeline (e.g. a CLIP inspected for plain feature-extraction).

    ``pipeline_tasks`` (the higher-level pipelines the model_type serves) come from the
    live composite registry; ``components`` is the resolver's detected breakdown.
    """
    # `is None` (not falsy): only the un-set case means "composite path not taken".
    # An empty dict would be a genuine (if currently unproduced) composite with no
    # component breakdown — the pipeline_tasks guard below still protects rendering.
    if detected_components is None:
        return None

    from ..loader import composite_pipeline_tasks

    pipeline_tasks = composite_pipeline_tasks(model_type)
    if not pipeline_tasks:
        # Registry-divergence guard: the resolver detected a composite but the registry
        # yields no pipeline tasks for this model_type (e.g. a future registration the
        # WinMLCompositeModel filter excludes). A CompositeInfo with empty pipeline_tasks
        # would render a broken "[composite]" Task row, so treat it as non-composite.
        return None

    return CompositeInfo(
        pipeline_tasks=pipeline_tasks,
        components=dict(detected_components),
    )


def _find_nested_configs(config: PretrainedConfig) -> list:
    """Discover all nested PretrainedConfig objects dynamically.

    Walks config attributes to find nested configs without hardcoding
    names like "text_config", "vision_config", etc. Fixes D-2 and D-5
    from #247.

    Args:
        config: HuggingFace PretrainedConfig object

    Returns:
        List of nested PretrainedConfig instances
    """
    from transformers import PretrainedConfig

    nested = []
    for attr_name in vars(config):
        if attr_name.startswith("_"):
            continue
        try:
            val = getattr(config, attr_name)
            if isinstance(val, PretrainedConfig):
                nested.append(val)
        except Exception:
            continue
    return nested


def _discover_io_attrs_from_onnx_config(
    model_type: str,
    task: str,
    hf_config: PretrainedConfig,
) -> set[str]:
    """Discover IO-relevant config attributes from OnnxConfig.

    Instead of hardcoding which config attributes to show, we read the
    uppercase class attrs on NormalizedConfig subclasses. These define
    the canonical attribute mapping for each model type, e.g.:

        NormalizedTextConfig.VOCAB_SIZE = "vocab_size"
        NormalizedVisionConfig.IMAGE_SIZE = "image_size"

    We also scan DUMMY_INPUT_GENERATOR_CLASSES for additional attrs
    referenced via normalized_config.xxx in generator __init__ code.

    Returns:
        Set of config attribute names relevant to I/O for this model.
    """
    import inspect
    import re

    attrs: set[str] = set()
    try:
        from ..export.io import _get_onnx_config

        onnx_config = _get_onnx_config(model_type, task, hf_config)

        # Primary: enumerate uppercase attrs on NormalizedConfig class.
        # These ARE the canonical IO attribute mapping (e.g., VOCAB_SIZE="vocab_size").
        nc = getattr(onnx_config, "_normalized_config", None)
        if nc is not None:
            for attr_name in dir(type(nc)):
                if attr_name.isupper() and not attr_name.startswith("_"):
                    # The value is the actual config attr name (e.g., "vocab_size")
                    val = getattr(type(nc), attr_name)
                    if isinstance(val, str):
                        # Handle dotted paths like "text_config.hidden_size"
                        leaf = val.split(".")[-1]
                        # Skip structural pointers (nested config references)
                        if not leaf.endswith("_config"):
                            attrs.add(leaf)

        # Secondary: scan generator __init__ for additional normalized_config refs
        for gen_cls in getattr(onnx_config, "DUMMY_INPUT_GENERATOR_CLASSES", []):
            try:
                src = inspect.getsource(gen_cls.__init__)
            except (TypeError, OSError):
                continue
            refs = re.findall(r"normalized_config\.(\w+)", src)
            attrs.update(r for r in refs if r != "has_attribute")
    except Exception as e:
        logger.debug("Failed to discover IO attrs from OnnxConfig: %s", e)

    return attrs


def resolve_io_config(
    config: PretrainedConfig,
    *,
    model_id: str | None = None,
    model_type: str | None = None,
    task: str | None = None,
) -> IOConfigInfo:
    """Extract IO configuration from HuggingFace config.

    Dynamically discovers which config attributes matter for I/O by
    inspecting OnnxConfig's NormalizedConfig and input generators.
    Falls back to a universal set of well-known attrs if OnnxConfig
    lookup fails. No hardcoded model-specific attribute names.

    Args:
        config: HuggingFace PretrainedConfig object
        model_id: Optional HF model ID for preprocessor_config.json fallback
        model_type: HF model type for OnnxConfig lookup
        task: Task name for OnnxConfig lookup

    Returns:
        IOConfigInfo with extracted configuration values
    """
    # Dynamically discover nested configs (fixes D-2: no hardcoded names)
    nested_configs = _find_nested_configs(config)

    def get_config_attr(
        attr_name: str,
    ) -> Any:
        """Get attribute from main config or any nested config.

        Returns ``Any``: HF config attributes are dynamically typed (int, tuple,
        list, str, ...), so each call site narrows to its target field type.
        """
        value = getattr(config, attr_name, None)
        if value is not None:
            return value

        for nested in nested_configs:
            value = getattr(nested, attr_name, None)
            if value is not None:
                return value

        return None

    # Step 1: Discover which attrs the OnnxConfig actually uses
    io_attrs: set[str] = set()
    if model_type and task:
        io_attrs = _discover_io_attrs_from_onnx_config(
            model_type,
            task,
            config,
        )

    # Step 2: Always include universal well-known IO attrs that Optimum's
    # NormalizedConfig classes reference. These are framework conventions,
    # not model-specific — they appear in NormalizedTextConfig,
    # NormalizedVisionConfig, NormalizedSeq2SeqConfig, etc.
    universal_io_attrs = {
        "max_position_embeddings",
        "vocab_size",
        "image_size",
        "patch_size",
        "num_channels",
        "input_size",
        "sampling_rate",
        "hidden_size",
        "hidden_sizes",
    }
    io_attrs.update(universal_io_attrs)

    # Step 3: Look up each discovered attr
    max_position_embeddings = get_config_attr("max_position_embeddings")
    vocab_size = get_config_attr("vocab_size")
    image_size = get_config_attr("image_size")
    patch_size = get_config_attr("patch_size")
    num_channels = get_config_attr("num_channels")
    sampling_rate = get_config_attr("sampling_rate")
    hidden_size = get_config_attr("hidden_size")
    hidden_sizes = get_config_attr("hidden_sizes")

    # Step 4: Collect any extra attrs discovered from OnnxConfig
    # that aren't in our dataclass fields
    known_fields = {
        "max_position_embeddings",
        "vocab_size",
        "image_size",
        "patch_size",
        "num_channels",
        "sampling_rate",
        "hidden_size",
        "hidden_sizes",
    }
    extra: dict[str, int | str | list | None] = {}
    for attr in io_attrs - known_fields:
        val = get_config_attr(attr)
        if val is not None:
            extra[attr] = val

    # Step 5: Fallback — read image_size from a preprocessor-style dict
    # (preprocessor_config.json on the hub, or synthesized from a nested
    # dict on hf_config such as TimmWrapperConfig.pretrained_cfg) when the
    # top-level HF config lacks image_size.
    if image_size is None and model_id is not None:
        try:
            from ..export.io import _populate_image_size_from_preprocessor

            shape_kwargs: dict = {}
            _populate_image_size_from_preprocessor(model_id, shape_kwargs, config)
            if "height" in shape_kwargs:
                h, w = shape_kwargs["height"], shape_kwargs["width"]
                image_size = h if h == w else (h, w)
        except Exception as e:
            logger.debug("Failed to get image_size from preprocessor: %s", e)

    return IOConfigInfo(
        max_position_embeddings=max_position_embeddings,
        vocab_size=vocab_size,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        sampling_rate=sampling_rate,
        hidden_size=hidden_size,
        hidden_sizes=hidden_sizes,
        extra=extra if extra else None,
    )


def resolve_processor(
    model_id: str,
    model_type: str | None = None,
) -> ProcessorInfo:
    """Resolve data processing classes for a HuggingFace model.

    Detects the processor/tokenizer/image_processor/feature_extractor classes
    associated with a model. Uses a multi-strategy approach:

    0. Check HF's IMAGE_PROCESSOR_MAPPING_NAMES for model_type-specific mapping
    1. Fetch config files from HuggingFace Hub (fast, no model download)
    2. Use Auto classes to fill in any remaining gaps

    Args:
        model_id: HuggingFace model identifier (e.g., "openai/clip-vit-base-patch32")
        model_type: HuggingFace model type (e.g., "resnet") for registry lookup

    Returns:
        ProcessorInfo with detected class names for each processor type
    """
    processor_class: str | None = None
    tokenizer_class: str | None = None
    image_processor_class: str | None = None
    feature_extractor_class: str | None = None
    # Source tracking
    processor_source: str | None = None
    tokenizer_source: str | None = None
    image_processor_source: str | None = None
    feature_extractor_source: str | None = None

    # Strategy 0: Check HF registry for the canonical image processor class
    # for this model_type. This is authoritative — HF maps model types to
    # their processor classes (e.g., resnet → ConvNextImageProcessor).
    if model_type is not None:
        try:
            from transformers.models.auto.image_processing_auto import (
                IMAGE_PROCESSOR_MAPPING_NAMES,
            )

            mapping = IMAGE_PROCESSOR_MAPPING_NAMES.get(model_type)
            if mapping:
                # mapping is (SlowProcessor, FastProcessor) or a string
                image_processor_class = mapping[0] if isinstance(mapping, tuple) else mapping
                image_processor_source = "hf_registry"
        except Exception as e:
            logger.debug("Registry lookup failed for %s: %s", model_type, e)

    # Strategy 1: Try to get class names from config files via HuggingFace Hub API
    # This is fast and doesn't require downloading/instantiating processors
    # NOTE: These JSON keys (processor_class, image_processor_type, etc.) are
    # standard HuggingFace config conventions, not model-specific hardcoding.
    has_preprocessor_config = True
    try:
        hub_result = _resolve_processor_from_hub_configs(model_id)
        if hub_result.processor_class and processor_class is None:
            processor_class = hub_result.processor_class
            processor_source = "hub_config"
        if hub_result.tokenizer_class and tokenizer_class is None:
            tokenizer_class = hub_result.tokenizer_class
            tokenizer_source = "hub_config"
        if hub_result.image_processor_class and image_processor_class is None:
            image_processor_class = hub_result.image_processor_class
            image_processor_source = "hub_config"
        if hub_result.feature_extractor_class and feature_extractor_class is None:
            feature_extractor_class = hub_result.feature_extractor_class
            feature_extractor_source = "hub_config"
        has_preprocessor_config = hub_result.has_preprocessor_config
    except Exception as e:
        logger.debug("Failed to resolve processors from hub configs: %s", e)

    # Strategy 2: Use Auto classes to fill in any missing information.
    # Skip entirely when Strategies 0 + 1 already populated every field —
    # each Auto* instantiation does its own HF Hub I/O plus class init
    # (AutoProcessor and AutoFeatureExtractor are several seconds each).
    #
    # When ``preprocessor_config.json`` is missing on the hub, the model
    # has neither an image processor nor a feature extractor; skip those
    # two Auto* round-trips (they would each spend ~1s confirming a 404).
    need_processor = processor_class is None
    need_tokenizer = tokenizer_class is None
    need_image_processor = image_processor_class is None and has_preprocessor_config
    need_feature_extractor = feature_extractor_class is None and has_preprocessor_config

    if need_processor or need_tokenizer or need_image_processor or need_feature_extractor:
        try:
            (
                auto_processor,
                auto_tokenizer,
                auto_image_processor,
                auto_feature_extractor,
            ) = _resolve_processor_from_auto_classes(
                model_id,
                try_processor=need_processor,
                try_tokenizer=need_tokenizer,
                try_image_processor=need_image_processor,
                try_feature_extractor=need_feature_extractor,
            )

            # Fill in missing values from auto classes
            if need_processor and auto_processor:
                processor_class = auto_processor
                processor_source = "auto_class"
            if need_tokenizer and auto_tokenizer:
                tokenizer_class = auto_tokenizer
                tokenizer_source = "auto_class"
            if need_image_processor and auto_image_processor:
                image_processor_class = auto_image_processor
                image_processor_source = "auto_class"
            if need_feature_extractor and auto_feature_extractor:
                feature_extractor_class = auto_feature_extractor
                feature_extractor_source = "auto_class"
        except Exception as e:
            logger.debug("Failed to resolve processors from auto classes: %s", e)

    return ProcessorInfo(
        processor_class=processor_class,
        tokenizer_class=tokenizer_class,
        image_processor_class=image_processor_class,
        feature_extractor_class=feature_extractor_class,
        processor_source=processor_source,
        tokenizer_source=tokenizer_source,
        image_processor_source=image_processor_source,
        feature_extractor_source=feature_extractor_source,
    )


class _HubConfigResult(NamedTuple):
    """Result of ``_resolve_processor_from_hub_configs``.

    A NamedTuple rather than a plain tuple so the trailing boolean cannot be
    silently swapped with the four ``str | None`` fields at the call site.
    """

    processor_class: str | None
    tokenizer_class: str | None
    image_processor_class: str | None
    feature_extractor_class: str | None
    has_preprocessor_config: bool


def _resolve_processor_from_hub_configs(model_id: str) -> _HubConfigResult:
    """Resolve processor classes by fetching config files from HuggingFace Hub.

    This approach is fast because it only downloads small JSON config files,
    not the full model weights or processor files.

    Args:
        model_id: HuggingFace model identifier

    Returns:
        A ``_HubConfigResult`` whose ``has_preprocessor_config`` reports
        whether ``preprocessor_config.json`` actually exists on the hub —
        the authoritative signal that the model has no image processor or
        feature extractor, so the caller can skip the corresponding
        ``AutoImageProcessor`` / ``AutoFeatureExtractor`` round-trips
        (which would each spend ~1s confirming a 404 on text-only models).
    """
    import json
    from pathlib import Path

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    processor_class: str | None = None
    tokenizer_class: str | None = None
    image_processor_class: str | None = None
    feature_extractor_class: str | None = None
    has_preprocessor_config = False

    # Try to download and parse preprocessor_config.json
    # This file contains image_processor_type or processor_class
    try:
        preprocessor_config_path = hf_hub_download(
            repo_id=model_id,
            filename="preprocessor_config.json",
        )
        # Set the flag as soon as the file exists on the hub, *before* parsing.
        # A corrupt JSON is still proof that the model ships preprocessor
        # config — fall back to Auto* lookups rather than declaring the model
        # text-only and silently dropping its image/feature processor.
        has_preprocessor_config = True
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

    return _HubConfigResult(
        processor_class=processor_class,
        tokenizer_class=tokenizer_class,
        image_processor_class=image_processor_class,
        feature_extractor_class=feature_extractor_class,
        has_preprocessor_config=has_preprocessor_config,
    )


def _is_tokenizer_class_name(name: str) -> bool:
    """Heuristic: does this transformers class name look like a tokenizer?

    Tokenizer classes follow the ``*Tokenizer`` / ``*TokenizerFast`` naming
    convention (e.g. ``RobertaTokenizer``, ``BertTokenizerFast``). Used to
    detect when ``AutoProcessor.from_pretrained`` returned a leaf tokenizer
    rather than a multimodal ``ProcessorMixin`` wrapper.
    """
    return name.endswith(("Tokenizer", "TokenizerFast"))


def _is_image_processor_class_name(name: str) -> bool:
    """Heuristic: does this transformers class name look like an image processor?"""
    return name.endswith(("ImageProcessor", "ImageProcessorFast"))


def _is_feature_extractor_class_name(name: str) -> bool:
    """Heuristic: does this transformers class name look like a feature extractor?"""
    return name.endswith("FeatureExtractor")


def _resolve_processor_from_auto_classes(
    model_id: str,
    *,
    try_processor: bool = True,
    try_tokenizer: bool = True,
    try_image_processor: bool = True,
    try_feature_extractor: bool = True,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Resolve processor classes by instantiating HuggingFace Auto classes.

    This is a fallback approach that actually loads the processors. It's slower
    but more reliable for models with non-standard configurations.

    Args:
        model_id: HuggingFace model identifier
        try_processor: When False, skip ``AutoProcessor.from_pretrained``.
            AutoProcessor is the most expensive single call (several seconds
            even on warm cache), so callers that already know all four classes
            should pass False for every flag to skip Strategy 2 entirely.
        try_tokenizer: When False, skip ``AutoTokenizer.from_pretrained``.
        try_image_processor: When False, skip ``AutoImageProcessor.from_pretrained``.
        try_feature_extractor: When False, skip ``AutoFeatureExtractor.from_pretrained``.

    Returns:
        Tuple of (processor_class, tokenizer_class, image_processor_class, feature_extractor_class).
        Each element is None when the corresponding lookup was skipped or failed.
    """
    processor_class: str | None = None
    tokenizer_class: str | None = None
    image_processor_class: str | None = None
    feature_extractor_class: str | None = None

    # Try AutoProcessor only when the processor class itself is needed.
    # AutoProcessor.from_pretrained is the single most expensive Auto* call
    # (~3.5s warm). When the caller only needs sub-pieces (tokenizer /
    # image_processor / feature_extractor) we fall through to the standalone
    # Auto* calls below, which are individually cheaper. AutoProcessor can
    # still fill those sub-fields as a side effect on success — that's
    # preserved here when ``try_processor`` is True.
    if try_processor:
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            processor_class = type(processor).__name__

            # AutoProcessor may wrap tokenizer / image_processor / feature_extractor
            # as a multimodal `ProcessorMixin`.  For single-modality models it
            # often returns the leaf class directly (e.g. RoBERTa →
            # `RobertaTokenizerFast`), which has none of those attributes.
            # Pattern-match the returned class name so the standalone Auto*
            # calls below can be skipped — otherwise we pay for a second,
            # redundant load (~2s for AutoTokenizer on warm cache).
            wrapped_tokenizer = getattr(processor, "tokenizer", None)
            wrapped_image_processor = getattr(processor, "image_processor", None)
            wrapped_feature_extractor = getattr(processor, "feature_extractor", None)

            if try_tokenizer and wrapped_tokenizer is not None:
                tokenizer_class = type(wrapped_tokenizer).__name__
            elif try_tokenizer and _is_tokenizer_class_name(processor_class):
                tokenizer_class = processor_class

            if try_image_processor and wrapped_image_processor is not None:
                image_processor_class = type(wrapped_image_processor).__name__
            elif try_image_processor and _is_image_processor_class_name(processor_class):
                image_processor_class = processor_class

            if try_feature_extractor and wrapped_feature_extractor is not None:
                feature_extractor_class = type(wrapped_feature_extractor).__name__
            elif try_feature_extractor and _is_feature_extractor_class_name(processor_class):
                feature_extractor_class = processor_class

        except Exception as e:
            logger.debug("AutoProcessor failed for %s: %s", model_id, e)

    # Try AutoTokenizer if we don't have tokenizer yet
    if try_tokenizer and tokenizer_class is None:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer_class = type(tokenizer).__name__
        except Exception as e:
            logger.debug("AutoTokenizer failed for %s: %s", model_id, e)

    # Try AutoImageProcessor if we don't have image_processor yet
    if try_image_processor and image_processor_class is None:
        try:
            from transformers import AutoImageProcessor

            image_processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
            image_processor_class = type(image_processor).__name__
        except Exception as e:
            logger.debug("AutoImageProcessor failed for %s: %s", model_id, e)

    # Try AutoFeatureExtractor if we don't have feature_extractor yet
    if try_feature_extractor and feature_extractor_class is None:
        try:
            from transformers import AutoFeatureExtractor

            feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            feature_extractor_class = type(feature_extractor).__name__
        except Exception as e:
            logger.debug("AutoFeatureExtractor failed for %s: %s", model_id, e)

    return processor_class, tokenizer_class, image_processor_class, feature_extractor_class
