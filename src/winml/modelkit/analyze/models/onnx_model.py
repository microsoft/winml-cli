# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNXModel entity - represents loaded ONNX model graph with metadata."""

from __future__ import annotations

from enum import Enum

import onnx
from pydantic import BaseModel, Field, field_validator

from winml.modelkit.onnx.domains import ONNXDomain


class ModelTag(str, Enum):
    """Tags for marking model-level issues and validation states.

    These tags are stored in ONNXModel.model_tags to record various
    model-level problems discovered during analysis.
    """

    # Invalid model for pattern matcher
    INVALID_PATTERN_MATCHER_MODEL = "invalid_pattern_matcher_model"

    # Model has nodes with missing names
    MISSING_NODE_NAMES = "missing_node_names"


class ONNXModel(BaseModel):
    """Represents loaded ONNX model graph with metadata.

    Attributes:
        model_path: Path to ONNX file
        graph: ONNX GraphProto object
        opset_version: ONNX opset version
        producer_name: Model producer identifier (optional)
        producer_version: Producer version (optional)
        nodes: List of operator nodes
        initializers: Model weights/constants
        inputs: Model input specifications
        outputs: Model output specifications
    """

    model_path: str = Field(..., description="Path to ONNX file")
    opset_version: int = Field(..., ge=1, description="ONNX opset version")
    producer_name: str | None = Field(default=None, description="Model producer identifier")
    producer_version: str | None = Field(default=None, description="Producer version")
    node_count: int = Field(..., ge=0, description="Total number of nodes in graph")
    initializer_count: int = Field(..., ge=0, description="Number of initializers")
    input_count: int = Field(..., ge=0, description="Number of inputs")
    output_count: int = Field(..., ge=0, description="Number of outputs")

    # Model-level tags for marking issues and states
    # Maps ModelTag to error message for detailed issue tracking
    model_tags: dict[ModelTag, str] = Field(
        default_factory=dict, description="Model-level issue tags with error messages"
    )

    # Cache for deserialized model to avoid repeated parsing
    _cached_model: onnx.ModelProto | None = None

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @field_validator("opset_version")
    @classmethod
    def validate_opset_version(cls, v: int) -> int:
        """Validate opset version is >= 12 per assumption."""
        if v < 12:
            raise ValueError(f"Opset version {v} < 12 (minimum supported version)")
        return v

    @field_validator("node_count")
    @classmethod
    def validate_non_empty_graph(cls, v: int) -> int:
        """Validate graph is non-empty."""
        if v == 0:
            raise ValueError("Graph must contain at least one node")
        return v

    @classmethod
    def from_onnx_model(cls, model: onnx.ModelProto, model_path: str) -> ONNXModel:
        """Create ONNXModel from onnx.ModelProto.

        Args:
            model: Loaded ONNX ModelProto
            model_path: Path to source ONNX file

        Returns:
            ONNXModel instance
        """
        # Extract opset version
        opset_version = 1
        if model.opset_import:
            for opset in model.opset_import:
                # "" is the historical/common representation
                if opset.domain == "" or opset.domain == ONNXDomain.AI_ONNX:
                    opset_version = opset.version
                    break

        # Extract producer info
        producer_name = model.producer_name if model.producer_name else None
        producer_version = model.producer_version if model.producer_version else None

        # Count graph elements
        graph = model.graph
        node_count = len(graph.node)
        initializer_count = len(graph.initializer)
        input_count = len(graph.input)
        output_count = len(graph.output)

        instance = cls(
            model_path=str(model_path),
            opset_version=opset_version,
            producer_name=producer_name,
            producer_version=producer_version,
            node_count=node_count,
            initializer_count=initializer_count,
            input_count=input_count,
            output_count=output_count,
        )
        # Store ModelProto by reference instead of serializing to bytes
        object.__setattr__(instance, "_cached_model", model)
        return instance

    def get_graph(self) -> onnx.GraphProto:
        """Deserialize and return ONNX graph.

        Returns:
            ONNX GraphProto object

        Note:
            Uses cached deserialized model to avoid repeated parsing.
        """
        return self.get_model().graph

    def get_model(self) -> onnx.ModelProto:
        """Return the cached ONNX ModelProto.

        Returns:
            ONNX ModelProto object

        Raises:
            RuntimeError: If no model is cached (should not happen in normal usage)
        """
        if self._cached_model is None:
            raise RuntimeError(
                "No cached ModelProto available. ONNXModel must be created via from_onnx_model()."
            )
        return self._cached_model
