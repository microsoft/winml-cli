# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Utility functions for reading and manipulating hierarchy tags in ONNX models.

This module provides functions to:
1. Read tags from ONNX node attributes
2. Read tags from sidecar JSON files
3. Validate tag consistency between sources
4. Query and filter operations by tags
"""

import json
from pathlib import Path

import onnx

from ..onnx import load_onnx


def load_tags_from_onnx(onnx_path: str) -> dict[str, dict[str, any]]:
    """Load hierarchy tags from ONNX node attributes and doc_string fields.

    Args:
        onnx_path: Path to ONNX model file

    Returns:
        Dictionary mapping node names to their tag information
    """
    model = load_onnx(onnx_path, validate=False)
    node_tags = {}

    for node in model.graph.node:
        node_name = node.name or f"{node.op_type}_{hash(str(node))}"
        node_info = {"op_type": node.op_type}

        # First, check for hierarchy_tag attribute (HTP format)
        hierarchy_tag = None
        for attr in node.attribute:
            if attr.name == "hierarchy_tag" and attr.type == onnx.AttributeProto.STRING:
                hierarchy_tag = attr.s.decode('utf-8')
                break

        if hierarchy_tag:
            node_info["tags"] = [hierarchy_tag]
            node_info["primary_path"] = hierarchy_tag
            node_info["tag_count"] = 1
            node_info["method"] = "htp"
        else:
            # Fallback: Extract hierarchy information from doc_string (legacy format)
            if node.doc_string:
                try:
                    hierarchy_info = json.loads(node.doc_string)
                    if isinstance(hierarchy_info, dict) and "hierarchy_tags" in hierarchy_info:
                        node_info["tags"] = hierarchy_info.get("hierarchy_tags", [])
                        node_info["primary_path"] = hierarchy_info.get("hierarchy_path", "")
                        node_info["tag_count"] = hierarchy_info.get("hierarchy_count", 0)
                        node_info["method"] = hierarchy_info.get("hierarchy_method", "unknown")
                except (json.JSONDecodeError, TypeError):
                    # Skip nodes with invalid JSON in doc_string
                    pass

        # Only include nodes that have hierarchy tags
        if "tags" in node_info:
            node_tags[node_name] = node_info

    return node_tags


def load_tags_from_sidecar(onnx_path: str) -> dict[str, any]:
    """Load hierarchy tags from sidecar JSON file.

    Args:
        onnx_path: Path to ONNX model file.
            Sidecar assumed to be *_hierarchy.json or *_htp_metadata.json.

    Returns:
        Complete sidecar data including metadata and tag mappings
    """
    # Try HTP metadata format first (new format)
    htp_sidecar_path = Path(onnx_path.replace('.onnx', '_htp_metadata.json'))
    if htp_sidecar_path.exists():
        with htp_sidecar_path.open() as f:
            return json.load(f)

    # Try legacy integrated format
    htp_integrated_path = Path(onnx_path.replace('.onnx', '_htp_integrated_metadata.json'))
    if htp_integrated_path.exists():
        with htp_integrated_path.open() as f:
            return json.load(f)

    # Try legacy hierarchy format
    legacy_sidecar_path = Path(onnx_path.replace('.onnx', '_hierarchy.json'))
    if legacy_sidecar_path.exists():
        with legacy_sidecar_path.open() as f:
            return json.load(f)

    msg = (
        f"Sidecar file not found. Tried: {htp_sidecar_path}, "
        f"{htp_integrated_path}, {legacy_sidecar_path}"
    )
    raise FileNotFoundError(msg)


def validate_tag_consistency(onnx_path: str) -> dict[str, any]:
    """Validate that tags in ONNX attributes match those in sidecar file.

    Args:
        onnx_path: Path to ONNX model file

    Returns:
        Validation report with consistency statistics
    """
    try:
        onnx_tags = load_tags_from_onnx(onnx_path)
        sidecar_data = load_tags_from_sidecar(onnx_path)

        # Handle different sidecar formats
        if "tagged_nodes" in sidecar_data:
            # HTP format: convert to legacy format for comparison
            sidecar_tags = {}
            for node_name, tag in sidecar_data["tagged_nodes"].items():
                if tag:  # Only include non-empty tags
                    sidecar_tags[node_name] = {"tags": [tag]}
        else:
            # Legacy format
            sidecar_tags = sidecar_data.get("node_tags", {})

        # Compare tag consistency
        mismatches = []
        onnx_only = set(onnx_tags.keys()) - set(sidecar_tags.keys())
        sidecar_only = set(sidecar_tags.keys()) - set(onnx_tags.keys())

        for node_name in set(onnx_tags.keys()) & set(sidecar_tags.keys()):
            onnx_node_tags = set(onnx_tags[node_name].get("tags", []))
            sidecar_node_tags = set(sidecar_tags[node_name].get("tags", []))

            if onnx_node_tags != sidecar_node_tags:
                mismatches.append({
                    "node": node_name,
                    "onnx_tags": list(onnx_node_tags),
                    "sidecar_tags": list(sidecar_node_tags)
                })

        return {
            "consistent": len(mismatches) == 0 and len(onnx_only) == 0 and len(sidecar_only) == 0,
            "total_onnx_nodes": len(onnx_tags),
            "total_sidecar_nodes": len(sidecar_tags),
            "tag_mismatches": mismatches,
            "onnx_only_nodes": list(onnx_only),
            "sidecar_only_nodes": list(sidecar_only)
        }

    except Exception as e:
        return {
            "consistent": False,
            "error": str(e)
        }


def query_operations_by_tag(
    onnx_path: str, tag_pattern: str, use_sidecar: bool = True
) -> list[dict[str, any]]:
    """Query operations that match a specific tag pattern.

    Args:
        onnx_path: Path to ONNX model file
        tag_pattern: Tag pattern to match (supports partial matching)
        use_sidecar: Whether to use sidecar file (True) or ONNX attributes (False)

    Returns:
        List of operations matching the tag pattern
    """
    if use_sidecar:
        sidecar_data = load_tags_from_sidecar(onnx_path)

        # Handle different sidecar formats
        if "tagged_nodes" in sidecar_data:
            # HTP format: convert to legacy format for processing
            node_tags = {}
            for node_name, tag in sidecar_data["tagged_nodes"].items():
                if tag:  # Only include non-empty tags
                    node_tags[node_name] = {"tags": [tag]}
        else:
            # Legacy format
            node_tags = sidecar_data.get("node_tags", {})
    else:
        node_tags = load_tags_from_onnx(onnx_path)

    matching_operations = []

    for node_name, node_info in node_tags.items():
        tags = node_info.get("tags", [])

        # Check if any tag matches the pattern
        for tag in tags:
            if tag_pattern in tag:
                matching_operations.append({
                    "node_name": node_name,
                    "op_type": node_info.get("op_type", "unknown"),
                    "matching_tag": tag,
                    "all_tags": tags
                })
                break

    return matching_operations


def get_tag_statistics(onnx_path: str, use_sidecar: bool = True) -> dict[str, any]:
    """Get statistics about tag distribution in the model.

    Args:
        onnx_path: Path to ONNX model file
        use_sidecar: Whether to use sidecar file (True) or ONNX attributes (False)

    Returns:
        Tag distribution statistics
    """
    if use_sidecar:
        try:
            sidecar_data = load_tags_from_sidecar(onnx_path)

            # Check if it has pre-computed tag_statistics (legacy format)
            if "tag_statistics" in sidecar_data:
                return sidecar_data["tag_statistics"]

            # If it's HTP format, compute statistics from tagged_nodes
            if "tagged_nodes" in sidecar_data:
                tagged_nodes = sidecar_data["tagged_nodes"]
                tag_counts = {}
                for tag in tagged_nodes.values():
                    if tag:  # Skip empty tags
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
                return tag_counts

            # Fall back to node_tags format (legacy)
            if "node_tags" in sidecar_data:
                node_tags = sidecar_data["node_tags"]
                tag_counts = {}
                for node_info in node_tags.values():
                    for tag in node_info.get("tags", []):
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
                return tag_counts

        except FileNotFoundError:
            # Fall back to ONNX attributes if sidecar not found
            pass

    # Compute statistics from ONNX attributes
    node_tags = load_tags_from_onnx(onnx_path)
    tag_counts = {}

    for node_info in node_tags.values():
        for tag in node_info.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    return tag_counts


def export_tags_to_csv(onnx_path: str, output_csv: str, use_sidecar: bool = True) -> None:
    """Export tag information to CSV for analysis.

    Args:
        onnx_path: Path to ONNX model file
        output_csv: Path to output CSV file
        use_sidecar: Whether to use sidecar file (True) or ONNX attributes (False)
    """
    import csv

    if use_sidecar:
        sidecar_data = load_tags_from_sidecar(onnx_path)

        # Handle different sidecar formats
        if "tagged_nodes" in sidecar_data:
            # HTP format: convert to legacy format for processing
            node_tags = {}
            for node_name, tag in sidecar_data["tagged_nodes"].items():
                if tag:  # Only include non-empty tags
                    node_tags[node_name] = {"tags": [tag]}
        else:
            # Legacy format
            node_tags = sidecar_data.get("node_tags", {})
    else:
        node_tags = load_tags_from_onnx(onnx_path)

    with Path(output_csv).open('w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Node Name", "Op Type", "Tag Count", "Primary Tag", "All Tags"])

        for node_name, node_info in node_tags.items():
            tags = node_info.get("tags", [])
            op_type = node_info.get("op_type", "unknown")
            primary_tag = tags[0] if tags else ""
            all_tags = "|".join(tags)

            writer.writerow([node_name, op_type, len(tags), primary_tag, all_tags])


def compare_tag_distributions(onnx_path1: str, onnx_path2: str) -> dict[str, any]:
    """Compare tag distributions between two ONNX models.

    Args:
        onnx_path1: Path to first ONNX model
        onnx_path2: Path to second ONNX model

    Returns:
        Comparison report
    """
    stats1 = get_tag_statistics(onnx_path1)
    stats2 = get_tag_statistics(onnx_path2)

    all_tags = set(stats1.keys()) | set(stats2.keys())

    comparison = {
        "model1_path": onnx_path1,
        "model2_path": onnx_path2,
        "tag_differences": [],
        "model1_only_tags": [],
        "model2_only_tags": []
    }

    for tag in all_tags:
        count1 = stats1.get(tag, 0)
        count2 = stats2.get(tag, 0)

        if count1 > 0 and count2 == 0:
            comparison["model1_only_tags"].append(tag)
        elif count1 == 0 and count2 > 0:
            comparison["model2_only_tags"].append(tag)
        elif count1 != count2:
            comparison["tag_differences"].append({
                "tag": tag,
                "model1_count": count1,
                "model2_count": count2,
                "difference": count2 - count1
            })

    return comparison
