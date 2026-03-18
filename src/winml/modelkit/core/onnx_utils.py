# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# S112: try-except-continue for skipping malformed JSON in node iteration
"""ONNX Model Manipulation Utilities.

This module provides utilities for working with ONNX models, including:
- Model loading and validation
- Node manipulation and metadata injection
- Hierarchy information management
- Model analysis and statistics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnx
import torch

from ..onnx import load_onnx


class ONNXUtils:
    """Utilities for ONNX model manipulation and analysis."""

    @staticmethod
    def load_and_validate(onnx_path: str) -> onnx.ModelProto:
        """Load and validate ONNX model from file.

        Args:
            onnx_path: Path to ONNX model file

        Returns:
            Loaded ONNX model

        Raises:
            FileNotFoundError: If file doesn't exist
            onnx.ValidationError: If model is invalid
        """
        return load_onnx(onnx_path)

    @staticmethod
    def inject_hierarchy_metadata(
        onnx_model: onnx.ModelProto,
        node_tags: dict[str, dict[str, Any]],
        method: str = "unknown",
    ) -> int:
        """Inject hierarchy metadata into ONNX node doc_strings.

        Args:
            onnx_model: ONNX model to modify
            node_tags: Dictionary mapping node names to their tag information
            method: Tagging method name for metadata

        Returns:
            Number of nodes that received hierarchy metadata
        """
        injected_count = 0

        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{hash(str(node))}"

            if node_name in node_tags:
                tag_info = node_tags[node_name]

                # Create hierarchy info for doc_string
                hierarchy_info = {
                    "hierarchy_tags": tag_info.get("tags", []),
                    "hierarchy_path": tag_info.get("primary_path", ""),
                    "hierarchy_method": method,
                    "hierarchy_count": len(tag_info.get("tags", [])),
                }

                # Additional metadata if available
                if "confidence" in tag_info:
                    hierarchy_info["confidence"] = tag_info["confidence"]
                if "source" in tag_info:
                    hierarchy_info["source"] = tag_info["source"]

                # Inject as JSON in doc_string
                node.doc_string = json.dumps(hierarchy_info)
                injected_count += 1

        return injected_count

    @staticmethod
    def extract_hierarchy_metadata(
        onnx_model: onnx.ModelProto,
    ) -> dict[str, dict[str, Any]]:
        """Extract hierarchy metadata from ONNX node doc_strings.

        Args:
            onnx_model: ONNX model to analyze

        Returns:
            Dictionary mapping node names to their hierarchy information
        """
        node_hierarchy = {}

        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{hash(str(node))}"

            if node.doc_string:
                try:
                    hierarchy_info = json.loads(node.doc_string)
                    if isinstance(hierarchy_info, dict) and "hierarchy_tags" in hierarchy_info:
                        node_hierarchy[node_name] = {
                            "tags": hierarchy_info.get("hierarchy_tags", []),
                            "primary_path": hierarchy_info.get("hierarchy_path", ""),
                            "method": hierarchy_info.get("hierarchy_method", "unknown"),
                            "op_type": node.op_type,
                        }

                        # Extract additional metadata
                        if "confidence" in hierarchy_info:
                            node_hierarchy[node_name]["confidence"] = hierarchy_info["confidence"]
                        if "source" in hierarchy_info:
                            node_hierarchy[node_name]["source"] = hierarchy_info["source"]

                except (json.JSONDecodeError, TypeError):
                    # Skip nodes with invalid JSON
                    continue

        return node_hierarchy

    @staticmethod
    def analyze_model_structure(onnx_model: onnx.ModelProto) -> dict[str, Any]:
        """Analyze ONNX model structure and provide statistics.

        Args:
            onnx_model: ONNX model to analyze

        Returns:
            Model structure analysis
        """
        # Count node types
        node_types = {}
        total_nodes = len(onnx_model.graph.node)

        for node in onnx_model.graph.node:
            node_types[node.op_type] = node_types.get(node.op_type, 0) + 1

        # Count inputs/outputs
        inputs = len(onnx_model.graph.input)
        outputs = len(onnx_model.graph.output)

        # Count initializers (parameters)
        initializers = len(onnx_model.graph.initializer)

        # Extract hierarchy statistics
        hierarchy_stats = ONNXUtils._analyze_hierarchy_coverage(onnx_model)

        return {
            "total_nodes": total_nodes,
            "node_types": node_types,
            "inputs": inputs,
            "outputs": outputs,
            "initializers": initializers,
            "hierarchy_coverage": hierarchy_stats,
            "opset_version": onnx_model.opset_import[0].version
            if onnx_model.opset_import
            else None,
        }

    @staticmethod
    def _analyze_hierarchy_coverage(onnx_model: onnx.ModelProto) -> dict[str, Any]:
        """Analyze hierarchy coverage in the model."""
        total_nodes = len(onnx_model.graph.node)
        tagged_nodes = 0
        unique_tags = set()

        for node in onnx_model.graph.node:
            if node.doc_string:
                try:
                    hierarchy_info = json.loads(node.doc_string)
                    if isinstance(hierarchy_info, dict) and "hierarchy_tags" in hierarchy_info:
                        tagged_nodes += 1
                        tags = hierarchy_info.get("hierarchy_tags", [])
                        unique_tags.update(tags)
                except (json.JSONDecodeError, TypeError):
                    continue

        coverage_ratio = tagged_nodes / total_nodes if total_nodes > 0 else 0.0

        return {
            "total_nodes": total_nodes,
            "tagged_nodes": tagged_nodes,
            "untagged_nodes": total_nodes - tagged_nodes,
            "coverage_ratio": coverage_ratio,
            "coverage_percentage": f"{coverage_ratio * 100:.1f}%",
            "unique_hierarchy_paths": len(unique_tags),
        }

    @staticmethod
    def create_sidecar_file(
        onnx_path: str, node_tags: dict[str, dict[str, Any]], metadata: dict[str, Any]
    ) -> str:
        """Create a sidecar JSON file with complete hierarchy information.

        Args:
            onnx_path: Path to ONNX model
            node_tags: Node tag mapping
            metadata: Additional export metadata

        Returns:
            Path to created sidecar file
        """
        sidecar_path = onnx_path.replace(".onnx", "_hierarchy.json")

        sidecar_data = {
            "version": "1.0",
            "model_path": onnx_path,
            "export_method": metadata.get("strategy", "unknown"),
            "node_tags": node_tags,
            "statistics": {
                "total_nodes": len(node_tags),
                "tagged_nodes": len([n for n in node_tags.values() if n.get("tags")]),
                "unique_tags": len(
                    {tag for node in node_tags.values() for tag in node.get("tags", [])}
                ),
            },
            "metadata": metadata,
        }

        from pathlib import Path

        with Path(sidecar_path).open("w") as f:
            json.dump(sidecar_data, f, indent=2)

        return sidecar_path

    @staticmethod
    def validate_hierarchy_consistency(onnx_path: str) -> dict[str, Any]:
        """Validate consistency between ONNX model hierarchy and sidecar file.

        Args:
            onnx_path: Path to ONNX model

        Returns:
            Validation report
        """
        try:
            # Load ONNX model hierarchy
            onnx_model = ONNXUtils.load_and_validate(onnx_path)
            onnx_hierarchy = ONNXUtils.extract_hierarchy_metadata(onnx_model)

            # Load sidecar hierarchy
            sidecar_path = onnx_path.replace(".onnx", "_hierarchy.json")
            if not Path(sidecar_path).exists():
                return {"consistent": False, "error": "Sidecar file not found"}

            with Path(sidecar_path).open() as f:
                sidecar_data = json.load(f)

            sidecar_hierarchy = sidecar_data.get("node_tags", {})

            # Compare hierarchies
            mismatches = []
            onnx_only = set(onnx_hierarchy.keys()) - set(sidecar_hierarchy.keys())
            sidecar_only = set(sidecar_hierarchy.keys()) - set(onnx_hierarchy.keys())

            for node_name in set(onnx_hierarchy.keys()) & set(sidecar_hierarchy.keys()):
                onnx_tags = set(onnx_hierarchy[node_name].get("tags", []))
                sidecar_tags = set(sidecar_hierarchy[node_name].get("tags", []))

                if onnx_tags != sidecar_tags:
                    mismatches.append(
                        {
                            "node": node_name,
                            "onnx_tags": list(onnx_tags),
                            "sidecar_tags": list(sidecar_tags),
                        }
                    )

            is_consistent = len(mismatches) == 0 and len(onnx_only) == 0 and len(sidecar_only) == 0

            return {
                "consistent": is_consistent,
                "total_onnx_nodes": len(onnx_hierarchy),
                "total_sidecar_nodes": len(sidecar_hierarchy),
                "tag_mismatches": mismatches,
                "onnx_only_nodes": list(onnx_only),
                "sidecar_only_nodes": list(sidecar_only),
            }

        except Exception as e:
            return {"consistent": False, "error": str(e)}

    @staticmethod
    def compare_models(onnx_path1: str, onnx_path2: str) -> dict[str, Any]:
        """Compare hierarchy information between two ONNX models.

        Args:
            onnx_path1: Path to first ONNX model
            onnx_path2: Path to second ONNX model

        Returns:
            Comparison report
        """
        try:
            model1 = ONNXUtils.load_and_validate(onnx_path1)
            model2 = ONNXUtils.load_and_validate(onnx_path2)

            hierarchy1 = ONNXUtils.extract_hierarchy_metadata(model1)
            hierarchy2 = ONNXUtils.extract_hierarchy_metadata(model2)

            # Extract unique tags from each model
            tags1 = {tag for node in hierarchy1.values() for tag in node.get("tags", [])}
            tags2 = {tag for node in hierarchy2.values() for tag in node.get("tags", [])}

            return {
                "model1_path": onnx_path1,
                "model2_path": onnx_path2,
                "model1_nodes": len(hierarchy1),
                "model2_nodes": len(hierarchy2),
                "model1_unique_tags": len(tags1),
                "model2_unique_tags": len(tags2),
                "common_tags": list(tags1 & tags2),
                "model1_only_tags": list(tags1 - tags2),
                "model2_only_tags": list(tags2 - tags1),
                "tag_overlap_ratio": len(tags1 & tags2) / max(len(tags1 | tags2), 1),
            }

        except Exception as e:
            return {"error": str(e)}


def infer_output_names(outputs: Any) -> list[str] | None:
    """Infer output names from model outputs for Optimum compatibility.

    This is a universal approach that works with HuggingFace ModelOutput dataclasses
    without hardcoding any specific model output names.

    Args:
        outputs: Model outputs from forward pass

    Returns:
        List of output names if outputs are a dataclass with tensor fields,
        None otherwise (let ONNX export use default names)
    """
    if outputs is None:
        return None

    # Check if outputs are a HuggingFace ModelOutput dataclass
    if hasattr(outputs, "__dataclass_fields__"):
        # Extract field names for simple tensor outputs only
        # Complex outputs (tuples, lists) will use ONNX default names
        output_names = []

        for field_name in outputs.__dataclass_fields__:
            field_value = getattr(outputs, field_name, None)
            if field_value is not None and isinstance(field_value, torch.Tensor):
                output_names.append(field_name)

        # Only return names if we found simple tensor outputs
        # This avoids issues with complex outputs like GPT2's past_key_values
        return output_names if output_names else None

    # For non-dataclass outputs (tuple, tensor, etc.), let ONNX use default names
    return None


def get_io_config(model_proto: onnx.ModelProto) -> dict:
    """Extract I/O configuration from ONNX model.

    Args:
        model_proto: ONNX ModelProto to analyze

    Returns:
        dict with:
            - input_names: list of input tensor names
            - input_shapes: list of input shapes (list of dims, None for dynamic)
            - input_types: list of numpy dtypes for inputs
            - output_names: list of output tensor names
            - output_shapes: list of output shapes
            - output_types: list of numpy dtypes for outputs
    """
    import numpy as np

    # Handle ONNX version compatibility for dtype conversion
    try:
        from onnx.helper import tensor_dtype_to_np_dtype
    except ImportError:
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

        def tensor_dtype_to_np_dtype(tensor_type: int) -> np.dtype:
            return TENSOR_TYPE_TO_NP_TYPE[tensor_type]

    io_config: dict[str, list] = {
        "input_names": [],
        "input_shapes": [],
        "input_types": [],
        "output_names": [],
        "output_shapes": [],
        "output_types": [],
    }

    for prefix, ios in [
        ("input", model_proto.graph.input),
        ("output", model_proto.graph.output),
    ]:
        for io in ios:
            name = io.name
            tensor_type = io.type.tensor_type

            # Handle sequence types (fallback)
            if tensor_type.elem_type == 0 and io.type.HasField("sequence_type"):
                tensor_type = io.type.sequence_type.elem_type.tensor_type

            # Extract dtype
            try:
                dtype = tensor_dtype_to_np_dtype(tensor_type.elem_type)
            except (KeyError, AttributeError):
                dtype = np.float32  # Default fallback

            # Extract shape (None for dynamic dims)
            shape = []
            if tensor_type.HasField("shape"):
                for dim in tensor_type.shape.dim:
                    if dim.HasField("dim_value"):
                        shape.append(dim.dim_value)
                    else:
                        shape.append(None)  # Dynamic dimension

            io_config[f"{prefix}_names"].append(name)
            io_config[f"{prefix}_shapes"].append(shape)
            io_config[f"{prefix}_types"].append(dtype)

    return io_config


def get_epcontext_info(model_or_path: onnx.ModelProto | str | Path) -> dict[str, Any] | None:
    """Extract EPContext information from a compiled ONNX model.

    Returns detailed information about the EPContext nodes including
    the source execution provider, SDK version, and partition names.

    EPContext Attributes (per ORT schema):
        - main_context: 1 if this is the main context, 0 for partitions
        - ep_cache_context: Binary data (embed_mode=1) or file path (embed_mode=0)
        - embed_mode: 1=binary embedded in attribute, 0=external file path
        - ep_sdk_version: SDK version (e.g., "v2.40.0" for QNN)
        - hardware_architecture: Target hardware (e.g., "80+" for NVIDIA Ampere)
        - partition_name: Unique partition identifier with hash
        - source: Source EP ("QNNExecutionProvider", "TensorrtExecutionProvider", etc.)
        - onnx_model_filename: Original ONNX model filename

    Args:
        model_or_path: ONNX ModelProto or path to ONNX file

    Returns:
        Dictionary with EPContext info if model is compiled, None otherwise.
        Keys:
            - is_compiled: True if model contains EPContext
            - ep_contexts: List of EPContext node details
            - source_ep: Primary execution provider (e.g., "QNNExecutionProvider")
            - ep_sdk_version: SDK version used for compilation
            - hardware_architecture: Target hardware architecture
            - embed_mode: 1=embedded binary, 0=external file
            - partition_count: Number of EPContext partitions

    Example:
        >>> info = get_epcontext_info("model_qnn_ctx.onnx")
        >>> info["source_ep"]
        'QNNExecutionProvider'
        >>> info["embed_mode"]
        1
    """
    if isinstance(model_or_path, str | Path):
        model = load_onnx(model_or_path, validate=False)
    else:
        model = model_or_path

    ep_contexts = []
    for node in model.graph.node:
        if node.op_type == "EPContext":
            ctx_info: dict[str, Any] = {
                "name": node.name,
            }
            for attr in node.attribute:
                if attr.name == "source":
                    ctx_info["source"] = attr.s.decode("utf-8") if attr.s else None
                elif attr.name == "ep_sdk_version":
                    ctx_info["ep_sdk_version"] = attr.s.decode("utf-8") if attr.s else None
                elif attr.name == "partition_name":
                    ctx_info["partition_name"] = attr.s.decode("utf-8") if attr.s else None
                elif attr.name == "embed_mode":
                    ctx_info["embed_mode"] = attr.i
                elif attr.name == "main_context":
                    ctx_info["main_context"] = attr.i == 1
                elif attr.name == "hardware_architecture":
                    ctx_info["hardware_architecture"] = attr.s.decode("utf-8") if attr.s else None
                elif attr.name == "onnx_model_filename":
                    ctx_info["onnx_model_filename"] = attr.s.decode("utf-8") if attr.s else None
                elif attr.name == "max_size":
                    ctx_info["max_size"] = attr.i
                elif attr.name == "ep_cache_context" and attr.s and len(attr.s) < 1024:
                    # Only store if likely a file path (short), skip large binary data
                    try:
                        ctx_info["cache_context_path"] = attr.s.decode("utf-8")
                    except UnicodeDecodeError:
                        pass  # Binary data, not a path

            ep_contexts.append(ctx_info)

    if not ep_contexts:
        return None

    # Extract primary info from first (usually main) context
    primary = ep_contexts[0]
    source_ep = primary.get("source")
    ep_sdk_version = primary.get("ep_sdk_version")
    hardware_arch = primary.get("hardware_architecture")
    embed_mode = primary.get("embed_mode", 1)

    return {
        "is_compiled": True,
        "ep_contexts": ep_contexts,
        "source_ep": source_ep,
        "ep_sdk_version": ep_sdk_version,
        "hardware_architecture": hardware_arch,
        "embed_mode": embed_mode,
        "partition_count": len(ep_contexts),
    }
