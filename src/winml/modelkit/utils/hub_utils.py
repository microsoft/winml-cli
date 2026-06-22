# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""HuggingFace Hub utilities for model detection and configuration loading.

This module provides intelligent detection of HuggingFace Hub models vs local models,
and handles the appropriate metadata storage and configuration loading strategies.
"""

import logging
import re
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


# Local-path indicators used to short-circuit Hub detection. Centralized
# here so every callsite that classifies a model input string applies the
# same rejection rules (existing path, ./ ../ /. ~/ prefixes, Windows
# drive letter). Without this shared helper, each detector tends to
# reimplement only the easiest check (Path.exists) and accept inputs
# like ``./model.onnx`` as Hub references.
_LOCAL_PATH_PREFIXES: tuple[str, ...] = ("./", "../", "/", "~/")
_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _is_local_path(value: str | None) -> bool:
    r"""Return ``True`` if *value* looks like a local filesystem path.

    Heuristic check used to reject local paths before treating an input as
    a Hub identifier. Detects:

    * existing filesystem entries (``Path.exists()``);
    * Unix-style relative/absolute/home prefixes (``./``, ``../``, ``/``, ``~/``);
    * Windows drive-letter absolute paths (``C:\``, ``D:/``).

    ``None`` and empty strings are not local paths.
    """
    if not value:
        return False
    try:
        if Path(value).exists():
            return True
    except (OSError, ValueError):
        # Path may be syntactically invalid on the current platform
        # (e.g. control characters); treat as "not a local path" so the
        # caller can apply Hub-format heuristics instead.
        pass
    if value.startswith(_LOCAL_PATH_PREFIXES):
        return True
    return bool(_WIN_DRIVE_RE.match(value))


def is_hub_model(model_name_or_path: str) -> tuple[bool, dict]:
    """Comprehensive Hub model detection with metadata extraction.

    Args:
        model_name_or_path: Model identifier or path

    Returns:
        Tuple of (is_hub_model, metadata_dict)
    """
    # Local-path rejection (existing path, ./ ../ /. ~/ prefixes, Windows drive)
    if _is_local_path(model_name_or_path):
        return False, {"type": "local", "path": model_name_or_path}

    # Parse potential Hub model format
    # Supports: model-name, org/model, org/model@revision
    hub_pattern = r"^(?:([^/@]+)/)?([^/@]+)(?:@(.+))?$"
    match = re.match(hub_pattern, model_name_or_path)

    if not match:
        return False, {"type": "invalid"}

    org, model, revision = match.groups()
    full_model_id = f"{org}/{model}" if org else model

    # Try to verify with Hub API
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        model_info = api.model_info(full_model_id, revision=revision)

        # Extract comprehensive metadata
        metadata = {
            "type": "hub",
            "model_id": model_info.id,
            "sha": model_info.sha,
            "revision": revision or "main",
            "tags": model_info.tags if hasattr(model_info, "tags") else [],
            "pipeline_tag": model_info.pipeline_tag
            if hasattr(model_info, "pipeline_tag")
            else None,
            "library_name": model_info.library_name
            if hasattr(model_info, "library_name")
            else None,
            "author": model_info.author if hasattr(model_info, "author") else None,
            "last_modified": str(model_info.lastModified)
            if hasattr(model_info, "lastModified")
            else None,
            "private": model_info.private if hasattr(model_info, "private") else False,
            "gated": model_info.gated if hasattr(model_info, "gated") else False,
        }

        # Try to get model card info if available
        try:
            from huggingface_hub import ModelCard

            card = ModelCard.load(full_model_id)
            if hasattr(card.data, "base_model"):
                metadata["base_model"] = card.data.base_model
            if hasattr(card.data, "license"):
                metadata["license"] = card.data.license
            if hasattr(card.data, "language"):
                metadata["language"] = card.data.language
            if hasattr(card.data, "task_categories"):
                metadata["task_categories"] = card.data.task_categories
        except Exception:
            pass

        return True, metadata

    except Exception as e:
        # Could not verify with Hub - might be private or offline
        # Use heuristics to guess
        if len(model_name_or_path.split("/")) <= 2 and "\\" not in model_name_or_path:
            return True, {
                "type": "hub_unverified",
                "model_id": full_model_id,
                "revision": revision or "main",
                "error": str(e),
            }
        return False, {"type": "local", "path": model_name_or_path}


def inject_hub_metadata(onnx_model: Any, model_name_or_path: str, metadata: dict) -> None:
    """Inject HuggingFace Hub metadata into ONNX model.

    Args:
        onnx_model: ONNX model proto
        model_name_or_path: Original model identifier
        metadata: Hub metadata dictionary
    """
    from datetime import datetime, timezone

    # Clear any existing HF metadata
    # We need to remove items by filtering, not reassigning
    hf_props_to_remove = []
    for i, prop in enumerate(onnx_model.metadata_props):
        if prop.key.startswith("hf_"):
            hf_props_to_remove.append(i)

    # Remove in reverse order to maintain indices
    for i in reversed(hf_props_to_remove):
        del onnx_model.metadata_props[i]

    # Add required metadata
    def add_prop(key: str, value: Any) -> None:
        if value is not None:
            prop = onnx_model.metadata_props.add()
            prop.key = key
            prop.value = str(value)

    # Required fields
    add_prop("hf_hub_id", metadata.get("model_id"))
    add_prop("hf_hub_revision", metadata.get("sha", "")[:8])
    add_prop("hf_model_type", "hub")

    # Get ModelExport version
    try:
        from .. import __version__

        export_version = __version__
    except ImportError:
        export_version = "unknown"

    add_prop("hf_export_version", export_version)
    add_prop("hf_export_timestamp", datetime.now(timezone.utc).isoformat())

    # Optional fields
    for key in ["pipeline_tag", "library_name", "base_model", "private", "gated"]:
        if key in metadata:
            add_prop(f"hf_{key}", metadata[key])

    # Producer information
    onnx_model.producer_name = "ModelExport-HTP"
    onnx_model.producer_version = export_version
    onnx_model.domain = "com.modelexport.htp"

    # Add doc string for human readability
    onnx_model.doc_string = (
        f"Exported from HuggingFace model: {metadata.get('model_id')}\n"
        f"Revision: {metadata.get('sha', 'unknown')[:8]}\n"
        f"Export timestamp: {datetime.now(timezone.utc).isoformat()}\n"
        f"ModelExport version: {export_version}"
    )


def save_local_model_configs(model_name_or_path: str, output_dir: Path, metadata: dict) -> None:
    """Save configuration files for local/in-house models.

    Args:
        model_name_or_path: Path to local model
        output_dir: Directory to save configs
        metadata: Local model metadata
    """
    # Check if the path exists first
    if not Path(model_name_or_path).exists():
        logger.info(f"Local model path {model_name_or_path} does not exist, skipping config copy")
        return

    try:
        from transformers import AutoConfig

        # Save config
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.save_pretrained(output_dir)
        logger.info(f"Saved config.json to {output_dir}")

        # Track what components were saved
        components_saved = []

        # Try AutoProcessor (for multimodal)
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(model_name_or_path)
            processor.save_pretrained(output_dir)
            components_saved.append("processor")
        except Exception:
            pass

        # Try AutoTokenizer (for text models) - only if processor wasn't saved
        if "processor" not in components_saved:
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                tokenizer.save_pretrained(output_dir)
                components_saved.append("tokenizer")
            except Exception:
                pass

        # Try AutoImageProcessor (for vision)
        try:
            from transformers import AutoImageProcessor

            image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)
            image_processor.save_pretrained(output_dir)
            components_saved.append("image_processor")
        except Exception:
            pass

        # Try AutoFeatureExtractor (for audio)
        try:
            from transformers import AutoFeatureExtractor

            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
            feature_extractor.save_pretrained(output_dir)
            components_saved.append("feature_extractor")
        except Exception:
            pass

        if components_saved:
            logger.info(f"Saved preprocessing components: {', '.join(components_saved)}")

    except Exception as e:
        logger.warning(f"Could not save config for local model: {e}")
        logger.warning("User will need to provide config manually for inference")


def load_hf_components_from_onnx(onnx_path: str) -> tuple[Any, Any]:
    """Load HuggingFace config and preprocessing components from ONNX.

    Handles both:
    1. Hub models - loads from HF Hub using metadata
    2. Local models - loads from co-located config files

    Args:
        onnx_path: Path to ONNX model

    Returns:
        Tuple of (config, preprocessor)
    """
    from pathlib import Path

    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoProcessor,
        AutoTokenizer,
    )

    from ..onnx import load_onnx

    # Load ONNX model and extract metadata
    onnx_model = load_onnx(onnx_path, validate=False)
    onnx_dir = Path(onnx_path).parent

    # Extract metadata
    metadata = {}
    for prop in onnx_model.metadata_props:
        metadata[prop.key] = prop.value

    model_type = metadata.get("hf_model_type", "unknown")

    if model_type == "hub":
        # Hub model: Load from HuggingFace Hub
        hf_hub_id = metadata.get("hf_hub_id")
        hf_revision = metadata.get("hf_hub_revision")

        if not hf_hub_id:
            raise ValueError("ONNX model marked as Hub model but missing hf_hub_id metadata")

        # Load config from Hub
        config = AutoConfig.from_pretrained(hf_hub_id, revision=hf_revision)

        # Try to load preprocessor from Hub
        preprocessor = None
        # Heterogeneous transformers Auto-loaders sharing a from_pretrained classmethod.
        hub_loaders: list[Any] = [
            AutoProcessor,
            AutoTokenizer,
            AutoImageProcessor,
            AutoFeatureExtractor,
        ]
        for loader_cls in hub_loaders:
            try:
                preprocessor = loader_cls.from_pretrained(hf_hub_id, revision=hf_revision)
                break
            except Exception:
                continue

        return config, preprocessor

    if model_type == "local":
        # Local model: Load from co-located files
        config_path = onnx_dir / "config.json"

        if not config_path.exists():
            raise ValueError(
                f"Local model but config.json not found at {config_path}. "
                "The model may have been moved without its config files."
            )

        # Load config from local file
        config = AutoConfig.from_pretrained(onnx_dir)

        # Try to load preprocessor from local files
        preprocessor = None
        # Heterogeneous transformers Auto-loaders sharing a from_pretrained classmethod.
        local_loaders: list[Any] = [
            AutoProcessor,
            AutoTokenizer,
            AutoImageProcessor,
            AutoFeatureExtractor,
        ]
        for loader_cls in local_loaders:
            try:
                preprocessor = loader_cls.from_pretrained(onnx_dir)
                break
            except Exception:
                continue

        return config, preprocessor

    # Unknown or legacy model
    raise ValueError(
        f"ONNX model has unknown type '{model_type}'. "
        "Was it exported with an older version of ModelExport? "
        "Please re-export the model."
    )
