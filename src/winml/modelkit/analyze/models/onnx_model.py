# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNXModel entity - represents loaded ONNX model graph with metadata."""

from __future__ import annotations

import logging
from enum import Enum

import onnx
from pydantic import BaseModel, Field, field_validator

from ...onnx import ONNXDomain
from ..utils.node_key_utils import (
    build_node_key_by_node_id,
    make_stable_node_key,
    resolve_stable_node_key,
)


logger = logging.getLogger(__name__)


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
    _graph_nodes: list[onnx.NodeProto] | None = None
    _node_key_by_node_id: dict[int, str] | None = None
    _node_by_key: dict[str, onnx.NodeProto] | None = None
    _node_by_name: dict[str, onnx.NodeProto] | None = None

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
        instance._initialize_node_key_index()
        return instance

    def _initialize_node_key_index(self) -> None:
        """Build node sidecar maps for stable key lookup."""
        model = self.get_model()
        graph_nodes = list(model.graph.node)
        node_key_by_node_id = build_node_key_by_node_id(graph_nodes)
        node_by_key: dict[str, onnx.NodeProto] = {}
        node_by_name: dict[str, onnx.NodeProto] = {}

        for index, node in enumerate(graph_nodes):
            stable_key = make_stable_node_key(node, index)
            node_by_key[stable_key] = node
            if node.name:
                if node.name not in node_by_name:
                    node_by_name[node.name] = node
                else:
                    logger.debug(
                        "Duplicate ONNX node.name '%s' encountered; keeping first occurrence "
                        "for get_node_by_name().",
                        node.name,
                    )

        object.__setattr__(self, "_graph_nodes", graph_nodes)
        object.__setattr__(self, "_node_key_by_node_id", node_key_by_node_id)
        object.__setattr__(self, "_node_by_key", node_by_key)
        object.__setattr__(self, "_node_by_name", node_by_name)

    def _ensure_node_key_index(self) -> None:
        """Ensure node sidecar indexes are initialized."""
        if (
            self._graph_nodes is None
            or self._node_key_by_node_id is None
            or self._node_by_key is None
            or self._node_by_name is None
        ):
            self._initialize_node_key_index()

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

    def get_node_key(self, node: onnx.NodeProto) -> str:
        """Get the stable sidecar key for a node."""
        self._ensure_node_key_index()

        node_key_by_node_id = self._node_key_by_node_id
        graph_nodes = self._graph_nodes
        assert node_key_by_node_id is not None
        assert graph_nodes is not None

        return resolve_stable_node_key(
            node,
            node_key_by_node_id=node_key_by_node_id,
            graph_nodes=graph_nodes,
            unknown_unnamed_error=(
                "Cannot resolve stable key for unnamed node outside ONNXModel graph. "
                "Pass a graph node loaded via ONNXModel.from_onnx_model()."
            ),
        )

    def get_node_by_key(self, node_key: str) -> onnx.NodeProto | None:
        """Resolve a stable sidecar key to a node."""
        self._ensure_node_key_index()
        node_by_key = self._node_by_key
        assert node_by_key is not None
        return node_by_key.get(node_key)

    def get_node_by_name(self, node_name: str) -> onnx.NodeProto | None:
        """Resolve original ONNX node name to a node when available."""
        self._ensure_node_key_index()
        node_by_name = self._node_by_name
        assert node_by_name is not None
        return node_by_name.get(node_name)

    def get_node_key_map(self) -> dict[int, str]:
        """Get a copy of node-id to stable-key sidecar mapping."""
        self._ensure_node_key_index()
        node_key_by_node_id = self._node_key_by_node_id
        assert node_key_by_node_id is not None
        return dict(node_key_by_node_id)
