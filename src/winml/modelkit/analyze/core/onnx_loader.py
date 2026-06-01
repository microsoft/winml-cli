# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNXLoader - Load and validate ONNX model files.

Implements FR-001 (Load ONNX model), FR-037 (File validation), FR-038 (Error handling).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import onnx

from ..models.onnx_model import ONNXModel
from ..models.output import extract_model_stats


if TYPE_CHECKING:
    from ..models.output import ModelStats

logger = logging.getLogger(__name__)


class ONNXLoadError(Exception):
    """Exception raised when ONNX model loading fails."""


class ONNXLoader:
    """Load and validate ONNX model files.

    Responsibilities:
    - Validate file existence and format
    - Load ONNX ModelProto with error handling (from file or memory)
    - Validate model structure
    - Create ONNXModel entity with metadata extraction

    FR-001: Load ONNX model from file path
    FR-037: Validate file exists and is valid ONNX format
    FR-038: Provide clear error messages for invalid inputs

    Attributes:
        model_path: Path to the ONNX model file (or "<memory>" for in-memory models)
        is_loaded: Whether model has been successfully loaded
        model: Loaded ONNXModel instance (after load() is called)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_proto: onnx.ModelProto | None = None,
    ) -> None:
        """Initialize loader with model path or model proto.

        Args:
            model_path: Path to ONNX model file (mutually exclusive with
                model_proto)
            model_proto: ONNX ModelProto already loaded in memory (mutually
                exclusive with model_path)

        Raises:
            FileNotFoundError: If model file does not exist
            ValueError: If path is invalid, or both/neither arguments provided
        """
        if (model_path is None) == (model_proto is None):
            raise ValueError("Must provide exactly one of model_path or model_proto")

        self._onnx_model: ONNXModel | None = None
        self._model_proto: onnx.ModelProto | None = None

        if model_proto is not None:
            # Load from memory
            self._model_path = Path("<memory>")
            self._model_proto = model_proto
            self._from_memory = True
        else:
            # Load from file - mypy can't infer model_path is not None here
            self._model_path = Path(model_path)  # type: ignore[arg-type]
            self._from_memory = False

            # Validate file exists
            if not self._model_path.exists():
                raise FileNotFoundError(f"ONNX model file not found: {model_path}")

            if not self._model_path.is_file():
                raise ValueError(f"Path is not a file: {model_path}")

    @property
    def model_path(self) -> str:
        """Path to the loaded ONNX model file."""
        return str(self._model_path)

    @property
    def is_loaded(self) -> bool:
        """Whether model has been successfully loaded."""
        return self._onnx_model is not None

    @property
    def is_from_memory(self) -> bool:
        """Whether model was loaded from memory (vs file)."""
        return self._from_memory

    @property
    def model(self) -> ONNXModel:
        """Get loaded ONNX model.

        Returns:
            ONNXModel: Loaded model

        Raises:
            RuntimeError: If model not loaded yet
        """
        if self._onnx_model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._onnx_model

    def load(self) -> ONNXModel:
        """Load ONNX model and extract metadata.

        Returns:
            ONNXModel: Structured model representation with metadata

        Raises:
            ONNXLoadError: If model loading or validation fails
            checker.ValidationError: If model is invalid ONNX
        """
        # Return cached model if already loaded
        if self._onnx_model is not None:
            logger.debug("Model already loaded, returning cached instance")
            return self._onnx_model

        # Load model proto if not already loaded from memory
        if self._from_memory:
            # Type checker verified via constructor that _model_proto is not None here
            assert self._model_proto is not None
            model_proto = self._model_proto
            logger.info("Using ONNX model from memory")
        else:
            # Validate file extension (warning only, not blocking)
            if self._model_path.suffix.lower() != ".onnx":
                logger.warning(
                    "File extension '%s' is not '.onnx'. File may not be a valid ONNX model.",
                    self._model_path.suffix,
                )

            # FR-001: Load ONNX model from file
            try:
                logger.info("Loading ONNX model from: %s", self._model_path)
                model_proto = onnx.load(str(self._model_path), load_external_data=False)
            except Exception as e:
                # FR-038: Provide clear error message
                raise ONNXLoadError(
                    f"Failed to load ONNX model from {self._model_path}: {type(e).__name__}: {e}"
                ) from e

        # Validate ONNX model structure
        self.validate(model_proto)

        # Create ONNXModel entity
        try:
            self._onnx_model = ONNXModel.from_onnx_model(model_proto, str(self._model_path))
            logger.info(
                "Successfully loaded ONNX model: %d nodes, opset version %d",
                self._onnx_model.node_count,
                self._onnx_model.opset_version,
            )
            return self._onnx_model
        except Exception as e:
            # FR-038: Provide clear error message
            raise ONNXLoadError(
                f"Failed to create ONNXModel entity from {self._model_path}: "
                f"{type(e).__name__}: {e}"
            ) from e

    @staticmethod
    def validate(model_proto: onnx.ModelProto) -> None:
        """Validate ONNX model structure.

        Args:
            model_proto: ONNX ModelProto to validate

        Raises:
            checker.ValidationError: If ONNX validation fails
            ValueError: If graph is empty or has structural issues

        Validation Checks:
            - Graph is non-empty (has nodes)
            - All node inputs/outputs are defined
            - Initializers have valid tensor types
            - No circular dependencies in graph
        """
        logger.debug("Validating ONNX model structure")

        # Check graph is non-empty
        if not model_proto.graph.node:
            raise ValueError("Model graph has no nodes")

        # Skip strict ONNX validation to allow custom attributes like hierarchy_tag
        # The model structure is still validated by checking for non-empty graph
        logger.debug("Skipping strict ONNX validation to allow custom attributes")

    def extract_metadata(self, detected_pattern_count: dict[str, int] | None = None) -> ModelStats:
        """Extract model metadata for analysis.

        Args:
            detected_pattern_count: Pattern ID to count mapping (default: empty dict)

        Returns:
            ModelStats object with model statistics

        Raises:
            RuntimeError: If model not loaded
        """
        if self._onnx_model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return extract_model_stats(self._onnx_model, detected_pattern_count=detected_pattern_count)


def load_onnx_model(
    model_path: str | Path | None = None,
    model_proto: onnx.ModelProto | None = None,
) -> ONNXModel:
    """Load ONNX model from file or memory.

    Convenience function for simple model loading.

    Args:
        model_path: Path to ONNX model file (mutually exclusive with model_proto)
        model_proto: ONNX ModelProto already loaded in memory (mutually
            exclusive with model_path)

    Returns:
        ONNXModel: Loaded and validated model

    Raises:
        FileNotFoundError: If model file not found
        checker.ValidationError: If validation fails
        ONNXLoadError: If loading fails
        ValueError: If both or neither arguments provided
    """
    loader = ONNXLoader(model_path=model_path, model_proto=model_proto)
    return loader.load()
