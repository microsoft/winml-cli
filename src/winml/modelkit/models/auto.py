# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLAutoModel - Factory class for automatic model selection.

Implements the from_pretrained() pattern with two-level task mapping.
Delegates the build pipeline to ``build_hf_model()`` or ``build_onnx_model()``
from ``modelkit.build``.

Design Principles
-----------------
1. FACTORY PATTERN: WinMLAutoModel orchestrates pipeline, task-specific classes are thin wrappers
2. CONFIG-DRIVEN: All pipeline behavior controlled by WinMLBuildConfig, no hardcoded logic
3. SEPARATION OF CONCERNS: WinMLAutoModel does NOT parse config internals - passes config to
   each module and lets the module decide behavior
4. OPTIONAL STAGES: config.quant = None skips quantization, config.compile = None skips compile
5. CACHE-FIRST: Cache check happens BEFORE build (skip on cache hit)
6. ARTIFACT FILES: All stages produce artifact files in cache directory
7. ONNX PATH: If model_id ends with .onnx and the file exists, uses build_onnx_model() directly

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..cache import get_cache_dir, get_cache_key, get_model_dir
from ..loader import load_hf_model
from ..loader.task import get_task_abbrev

# Import task mapping from winml/ subpackage
from .winml import get_supported_tasks, get_winml_class


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ..config import WinMLBuildConfig
    from .winml.base import WinMLPreTrainedModel

logger = logging.getLogger(__name__)


# =============================================================================
# WinMLAutoModel Factory
# =============================================================================


class WinMLAutoModel:
    """Factory class for automatic WinML model selection.

    This is a FACTORY - it is NOT instantiable. Use from_pretrained().

    Design Principles:
        1. FACTORY PATTERN: Orchestrates pipeline, does NOT do inference
        2. CONFIG-DRIVEN: All behavior controlled by WinMLBuildConfig
        3. SEPARATION OF CONCERNS: Does NOT parse config internals - passes
           config to each module and lets the module decide behavior
        4. OPTIONAL STAGES: config.quant = None skips quantization,
           config.compile = None skips compilation
        5. CACHE-FIRST: Check cache BEFORE build, skip on hit

    Pipeline:
        HF model: Load → Export to ONNX → Optimize → [Quantize] → [Compile]
        ONNX file: Optimize → [Quantize] → [Compile]
        → Return inference-ready WinMLPreTrainedModel subclass

    Example:
        >>> from winml.modelkit import WinMLAutoModel
        >>> # From HuggingFace model
        >>> model = WinMLAutoModel.from_pretrained("microsoft/resnet-50")
        >>> # Returns WinMLModelForImageClassification (inference-ready)
        >>>
        >>> # From pre-exported ONNX file (auto-generates config)
        >>> model = WinMLAutoModel.from_onnx("model.onnx", device="npu")
        >>>
        >>> # Or via from_pretrained (delegates to from_onnx)
        >>> model = WinMLAutoModel.from_pretrained("model.onnx", config=my_config)
        >>>
        >>> # Use forward() for inference
        >>> output = model.forward(pixel_values=images)
        >>> # Or use __call__
        >>> output = model(pixel_values=images)
        >>>
        >>> # Use to() for device placement
        >>> model.to("npu")
    """

    def __init__(self) -> None:
        raise OSError(
            "WinMLAutoModel is designed to be instantiated using the "
            "`WinMLAutoModel.from_pretrained(model_id)` class method."
        )

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str | Path | dict[str, str | Path],
        *,
        task: str | None = None,
        config: WinMLBuildConfig | None = None,
        device: str = "auto",
        precision: str = "auto",
        ep: str | None = None,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        force_rebuild: bool = False,
        skip_build: bool = False,
        session_options: Any | None = None,
        hf_config: PretrainedConfig | None = None,
        **kwargs: Any,
    ) -> WinMLPreTrainedModel | WinMLCompositeModel:  # noqa: F821
        """Build from a pre-exported ONNX file.

        Runs optimize -> [quantize] -> [compile] via ``build_onnx_model()``.
        If *config* is None, auto-generates via ``generate_build_config(onnx_path=...)``.

        Args:
            onnx_path: Path to existing ONNX model file.
            task: Task name. Optional for ONNX builds (not needed for build pipeline).
            config: Build config. If None, auto-generated with device/precision resolution.
            device: Target device ("auto", "npu", "gpu", "cpu").
            precision: Target precision ("auto", "fp32", "fp16", "int8").
            ep: Explicit execution provider.
            cache_dir: Override cache directory.
            use_cache: Whether to use persistent cache.
            force_rebuild: Force rebuild even if cached.
            hf_config: HF ``PretrainedConfig`` for composite (dict) dispatch only.
                Required when ``onnx_path`` is a dict so the composite registry
                lookup can resolve ``(model_type, task)``. Ignored for single-file
                builds.
            **kwargs: Forwarded to ``build_onnx_model()``.

        Returns:
            WinMLPreTrainedModel inference wrapper.
        """
        if isinstance(onnx_path, dict):
            from .winml.composite_model import WinMLCompositeModel

            return WinMLCompositeModel.from_onnx(
                onnx_path,
                task=task,
                hf_config=hf_config,
                device=device,
                precision=precision,
                ep=ep,
                cache_dir=cache_dir,
                use_cache=use_cache,
                force_rebuild=force_rebuild,
                skip_build=skip_build,
                session_options=session_options,
                **kwargs,
            )

        onnx_path = Path(onnx_path)
        if not onnx_path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}. Provide a valid path to an existing ONNX file."
            )
        logger.info("Loading WinML model from ONNX: %s", onnx_path)

        # Always generate config with device/precision resolution.
        # If user provides config, treat it as an override (merged on top).
        from ..config import generate_onnx_build_config

        config = generate_onnx_build_config(
            onnx_path,
            task=task,
            device=device,
            precision=precision,
            ep=ep,
            override=config,
        )

        # Resolve task from explicit arg or generated config
        resolved_task = task or (config.loader.task if config.loader else None)

        # Skip build for compiled models or explicit skip.
        # Check is_compiled_onnx directly — don't rely on config shape alone
        # because auto+auto also produces quant=None, compile=None for raw models.
        from ..onnx import is_compiled_onnx

        if skip_build or is_compiled_onnx(onnx_path):
            logger.info("Skipping build (compiled model or explicit skip). Using original ONNX.")
            # TODO: run analyze_onnx for validation/lint
            # Use hf_config.model_type so WINML_MODEL_CLASS_MAPPING's
            # specialised entries (e.g. ESRGAN) are picked when the caller
            # knows the model_type; falls back to the generic task class
            # otherwise.
            model_type = hf_config.model_type if hf_config is not None else None
            winml_class = get_winml_class(model_type, resolved_task)
            return winml_class(
                onnx_path=onnx_path,
                config=hf_config,
                device=device,
                session_options=session_options,
                ep=ep,
            )

        # Resolve output directory
        if use_cache:
            cache_dir_path = get_cache_dir(override=cache_dir)
            output_dir = get_model_dir(onnx_path.stem, cache_dir=cache_dir_path)
        else:
            import tempfile

            cache_dir_path = Path(tempfile.mkdtemp(prefix="winml_"))
            output_dir = cache_dir_path
            force_rebuild = True
            logger.info("Cache disabled -- using temp directory: %s", output_dir)

        # Build: optimize → [quantize] → [compile]
        from ..build import build_onnx_model

        result = build_onnx_model(
            onnx_path=onnx_path,
            config=config,
            output_dir=output_dir,
            rebuild=force_rebuild,
            ep=ep,
            device=device,
            **kwargs,
        )

        # Wrap in inference model (task-specific or generic fallback).
        # When the caller supplies hf_config, pick the specialised class
        # registered for its ``model_type`` and propagate the config.
        model_type = hf_config.model_type if hf_config is not None else None
        winml_class = get_winml_class(model_type, resolved_task)
        logger.info("Creating inference wrapper: %s", winml_class.__name__)

        return winml_class(
            onnx_path=result.final_onnx_path,
            config=hf_config,
            device=device,
            session_options=session_options,
            ep=ep,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path,
        *,
        task: str | None = None,
        config: WinMLBuildConfig | None = None,
        device: str = "auto",
        precision: str = "auto",
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        force_rebuild: bool = False,
        trust_remote_code: bool = False,
        shape_config: dict | None = None,
        **kwargs: Any,
    ) -> WinMLPreTrainedModel:
        """Load appropriate WinML model based on task detection.

        Supports two input modes:

        **HF model path** (default): Runs the full pipeline --
        CONFIG -> LOAD -> BUILD (export -> optimize -> [quantize] -> [compile]) -> RUNTIME.

        **ONNX file path**: If ``model_id_or_path`` ends with ``.onnx`` and the
        file exists, skips HF loading and export, and runs optimize -> [quantize]
        -> [compile] directly via ``build_onnx_model()``. Requires ``config`` with
        ``loader.task`` set (task cannot be auto-detected from a bare ONNX file).

        Args:
            model_id_or_path: HF model ID, local path, or path to .onnx file.
            task: Explicit task name. If None, auto-detected from config.
            config: WinMLBuildConfig for pipeline configuration.
                Required when model_id_or_path is an ONNX file.
            device: Target device ("auto", "npu", "gpu", "cpu").
                "auto" detects available hardware (NPU > GPU > CPU).
            precision: Target precision ("auto", "fp32", "fp16", "int8", "int16").
                "auto" selects based on device (npu->int8, gpu->fp16, cpu->fp16).
            cache_dir: Directory for caching. If None, uses default cache dir.
            use_cache: If True (default), use persistent cache directory.
                If False, build in a temp directory and always rebuild.
            force_rebuild: If True, rebuild even if cached model exists.
            trust_remote_code: Whether to trust remote code in HF models
            shape_config: Shape overrides passed to generate_build_config().
                Valid keys -- text: sequence_length; vision: height, width;
                audio: feature_size, nb_max_frames, audio_sequence_length.
            **kwargs: Additional arguments

        Returns:
            WinMLPreTrainedModel subclass (e.g., WinMLModelForImageClassification)
            with forward(), to(), and __call__() methods for HF compatibility.

        Raises:
            ValueError: If task cannot be detected or is not supported, or if
                an ONNX file is given without a config containing loader.task.
        """
        model_id = str(model_id_or_path)  # Ensure string for Path inputs
        logger.info("Loading WinML model from: %s", model_id)

        # =====================================================================
        # ONNX FAST PATH -- skip HF loading and export when given an .onnx file
        # =====================================================================
        onnx_file = Path(model_id)
        if onnx_file.suffix == ".onnx" and onnx_file.exists():
            return cls.from_onnx(
                onnx_path=onnx_file,
                task=task,
                config=config,
                device=device,
                precision=precision,
                ep=kwargs.pop("ep", None),
                cache_dir=cache_dir,
                use_cache=use_cache,
                force_rebuild=force_rebuild,
                **kwargs,
            )

        # =====================================================================
        # COMPOSITE MODEL CHECK — delegate to WinMLCompositeModel.from_pretrained
        # when (model_type, task) is a registered composite (e.g., T5 translation,
        # Qwen text-generation).  AutoConfig is lightweight (~config.json only).
        # The registry probe (AutoConfig.from_pretrained) is gated on whether
        # `task` appears in any registered composite entry, avoiding an
        # unconditional network/disk round-trip for every non-composite call.
        # =====================================================================
        if task is not None:
            from .winml.composite_model import COMPOSITE_MODEL_REGISTRY

            _known_composite_tasks = {t for (_, t) in COMPOSITE_MODEL_REGISTRY}
            if task in _known_composite_tasks:
                from transformers import AutoConfig

                _hf_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
                _model_type = getattr(_hf_cfg, "model_type", None)
            else:
                _model_type = None

            if _model_type is not None and (_model_type, task) in COMPOSITE_MODEL_REGISTRY:
                from .winml.composite_model import WinMLCompositeModel

                return WinMLCompositeModel.from_pretrained(
                    model_id,
                    task,
                    device=device,
                    use_cache=use_cache,
                    force_rebuild=force_rebuild,
                    trust_remote_code=trust_remote_code,
                    shape_config=shape_config,
                    precision=precision,
                    config=config,
                    cache_dir=cache_dir,
                    **kwargs,
                )

        # =====================================================================
        # [1] CONFIG PHASE - Generate complete config with I/O specs (Lightweight, ~2s)
        # =====================================================================
        from ..config import generate_hf_build_config

        # Device/precision resolution is handled inside generate_hf_build_config().
        # When config is provided, it merges as Tier-1 override on top of defaults.
        build_config = generate_hf_build_config(
            model_id,
            task=task,
            override=config,
            shape_config=shape_config,
            device=device,
            precision=precision,
            trust_remote_code=trust_remote_code,
            ep=kwargs.get("ep"),
        )

        resolved_task = build_config.loader.task
        logger.debug("Generated config with task: %s", resolved_task)

        # =====================================================================
        # [2] LOAD PHASE - Load HF model with weights (Heavyweight, ~30-60s)
        # =====================================================================
        effective_trust = trust_remote_code or (
            build_config.loader.trust_remote_code if build_config.loader else False
        )
        pytorch_model, hf_config, _ = load_hf_model(
            model_name_or_path=model_id,
            task=resolved_task,
            trust_remote_code=effective_trust,
        )
        model_type = getattr(hf_config, "model_type", "unknown")
        logger.debug("Model type: %s, task: %s", model_type, resolved_task)

        config = build_config
        task = resolved_task

        # =====================================================================
        # [3] CACHE + BUILD PHASE -- delegate to build_hf_model()
        # =====================================================================
        if use_cache:
            cache_dir_path = get_cache_dir(override=cache_dir)
        else:
            # No cache -- use temp directory, always rebuild
            import tempfile

            cache_dir_path = Path(tempfile.mkdtemp(prefix="winml_"))
            force_rebuild = True
            logger.info("Cache disabled -- using temp directory: %s", cache_dir_path)

        # Compute cache_key and output_dir via shared cache module
        cache_key = get_cache_key(get_task_abbrev(task), config.generate_cache_key())
        output_dir = get_model_dir(model_id, cache_dir=cache_dir_path)

        from ..build import build_hf_model

        # Pass resolved EP so the static analyzer targets only this EP
        resolved_ep = config.compile.ep_config.provider if config.compile is not None else None
        result = build_hf_model(
            config=config,
            output_dir=output_dir,
            model_id=model_id,
            pytorch_model=pytorch_model,
            rebuild=force_rebuild,
            trust_remote_code=trust_remote_code,
            cache_key=cache_key,
            ep=resolved_ep,
            device=device,
        )
        onnx_path = result.final_onnx_path

        # =====================================================================
        # [4] RUNTIME PHASE - Return inference wrapper
        # =====================================================================
        winml_class = get_winml_class(model_type, task)
        logger.info("Creating inference wrapper: %s", winml_class.__name__)

        model = winml_class(
            onnx_path=onnx_path,
            config=hf_config,  # HF PretrainedConfig for pipeline compatibility
            device=device,  # pass user's original device string; WinMLSession handles "auto"
            ep=resolved_ep,
        )
        model._build_config = config  # resolved build config (task, quant, compile)
        return model

    @classmethod
    def supported_tasks(cls) -> list[str]:
        """Get list of supported tasks."""
        return get_supported_tasks()
