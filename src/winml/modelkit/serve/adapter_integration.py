# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Adapter integration layer for backward compatibility with InferenceEngine.

Provides a seamless transition path:
- Existing code continues to use engine.predict()
- Internally delegates to the unified Adapter system
- Automatically selects the inference engine (ONNX Runtime or GenAI)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .adapter import AdapterFactory


if TYPE_CHECKING:
    from .schema import PredictionResult


logger = logging.getLogger(__name__)


class AdapterEngineWrapper:
    """Wraps the Adapter system behind an InferenceEngine-compatible interface.

    Transition layer that allows:
    1. Reuse of existing SingleModelManager and API routing
    2. Internal delegation to the unified Adapter design
    3. Gradual migration to a fully Adapter-based architecture

    Usage::

        # Before (ONNX only)
        engine = InferenceEngine()
        engine.load(model_path, task=task, device=device)
        result = engine.predict(image_bytes=data)

        # Now (ONNX or GenAI)
        wrapper = AdapterEngineWrapper()
        wrapper.load_from_manifest(model_path, task=task, device=device)
        result = wrapper.predict(image_bytes=data)
    """

    def __init__(self):
        self.adapter = None
        self.manifest = None
        self.model_path = None
        self.task = None
        self.device = None

    def load_from_manifest(
        self,
        model_path: str | Path,
        task: str | None = None,
        device: str = "auto",
        ep: str | None = None,  # kept for API compat, unused
    ) -> None:
        """Load a model from its build manifest.

        Args:
            model_path: Directory containing build_manifest.json
            task: Task type (read from manifest; this parameter overrides it)
            device: Target device
            ep: Execution provider (kept for backward compat)
        """
        self.model_path = Path(model_path)
        self.device = device

        manifest_file = self.model_path / "build_manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        with manifest_file.open() as f:
            self.manifest = json.load(f)

        if task:
            self.manifest["task"] = task
        self.task = self.manifest.get("task")

        logger.info(f"Loaded manifest for {self.task}")

        self.adapter = AdapterFactory.create(self.manifest, self.model_path, device)

    def predict(
        self,
        image_bytes: bytes | None = None,
        text: str | None = None,
        tensor_inputs: dict[str, list] | None = None,
        top_k: int = 5,
        **kwargs,
    ) -> PredictionResult:
        """Run inference (backward-compatible with InferenceEngine.predict).

        Args:
            image_bytes: Raw image bytes (optional)
            text: Text input (optional)
            tensor_inputs: Pre-processed tensor inputs (optional)
            top_k: Number of top-K results (default 5)
            **kwargs: Extra params forwarded to adapter (max_tokens, temperature, etc.)

        Returns:
            PredictionResult
        """
        if self.adapter is None:
            raise RuntimeError("Engine not loaded. Call load_from_manifest() first.")

        inputs = {
            "image_bytes": image_bytes,
            "text": text,
            "tensor_inputs": tensor_inputs,
            "top_k": top_k,
        }
        inputs.update(kwargs)
        return self.adapter.predict(inputs)

    def unload(self) -> None:
        """Unload the model and release resources."""
        self.adapter = None
        self.manifest = None
        self.model_path = None
        logger.info("Engine unloaded")

    def reload(self) -> None:
        """Reload the model from its original path."""
        if not self.model_path:
            raise RuntimeError("reload() called before load_from_manifest()")
        self.load_from_manifest(self.model_path, task=self.task, device=self.device)


def should_use_adapter_engine(model_path: Path) -> bool:
    """Decide whether to use AdapterEngineWrapper for a model directory.

    Checks:
    1. build_manifest.json exists (new manifest format)
    2. The manifest contains an ``engine`` section

    Args:
        model_path: Model directory

    Returns:
        True if AdapterEngineWrapper should be used, False otherwise.
    """
    manifest_file = Path(model_path) / "build_manifest.json"
    if not manifest_file.exists():
        return False

    try:
        with manifest_file.open() as f:
            manifest = json.load(f)
        return "engine" in manifest
    except Exception as e:
        logger.warning(f"Failed to check manifest: {e}")
        return False
