#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Atomically add one scoped optimization case and rebuild the lightweight index."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any


CASE_FIELDS = {
    "id",
    "status",
    "scope",
    "observation",
    "mechanism",
    "transformation",
    "expected_evidence",
    "outcome",
    "safety",
    "counterexamples",
    "provenance",
    "generic_review",
    "discovery",
}
DISCOVERY_FIELDS = {"ep", "device", "anchor_ops", "keywords", "lesson"}
SCOPE_FIELDS = {"ep", "device", "graph_requirements"}
PROVENANCE_FIELDS = {"evidence_class", "scope_note"}
GENERIC_REVIEW_FIELDS = {"verdict", "reviewer", "content_sha256"}
STATUS_EVIDENCE_CLASSES = {
    "confirmed": "paired-performance-confirmed",
    "confirmed-performance-provisional-quality": "paired-performance-confirmed-provisional-quality",
    "rejected": "paired-performance-rejected",
    "inconclusive": "paired-performance-inconclusive",
}
RUN_LOCAL_SCOPE_NOTE = (
    "exact model, graph occurrence, toolchain, artifacts, and measurements remain run-local"
)
GENERIC_NUMBERED_TERMS = re.compile(
    r"\b(?:FP16|FP32|INT8|INT16|W8A8|W8A16|Conv1D|Conv2D|Conv3D)\b",
    re.IGNORECASE,
)
GENERIC_DOTTED_TERMS = re.compile(
    r"(?<!\w)(?:e\.g\.|i\.e\.|ai\.onnx|com\.microsoft)(?!\w)", re.IGNORECASE
)
GENERIC_ONE_TERMS = re.compile(
    r"\b(?:one static axis|one axis|step-one bounds|one-to-one)\b", re.IGNORECASE
)


class KnowledgeError(ValueError):
    """Raised when a knowledge case or index violates the compact contract."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KnowledgeError(f"invalid JSON: {path}") from error


def case_content_sha256(case: dict[str, Any]) -> str:
    """Hash canonical case content excluding its independent-review envelope."""
    content = {key: value for key, value in case.items() if key != "generic_review"}
    canonical = (
        json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_generic_text(value: str, location: str) -> None:
    generic_text = GENERIC_DOTTED_TERMS.sub("", value)
    lowered = generic_text.lower()
    if re.search(r"\b[a-z][a-z0-9+.-]*://", generic_text, re.IGNORECASE):
        raise KnowledgeError(f"bundled knowledge must be model-agnostic; {location} is a URL")
    if re.search(r"\b[0-9a-f]{32}\b|\b[0-9a-f]{40}\b|\b[0-9a-f]{64}\b", lowered):
        raise KnowledgeError(f"bundled knowledge must be model-agnostic; {location} embeds a hash")
    if re.search(r"\b(?:sha|hash|commit|revision|rev)\s*[:=#]?\s*[0-9a-f]{7,}\b", lowered):
        raise KnowledgeError(
            f"bundled knowledge must be model-agnostic; {location} embeds a revision"
        )
    if re.search(
        r"(?:[a-z]:[\\/]|(?:^|\s)/[^\s]+|\b[^\s]+\.(?:onnx|ort|pb|data|json|html|csv|txt|log|zip|bin|npz|parquet)\b)",
        generic_text,
        re.IGNORECASE,
    ):
        raise KnowledgeError(
            f"bundled knowledge must be model-agnostic; {location} is a path/artifact"
        )
    if re.search(
        r"\bpattern-\d+\b|\bonnx::|\b[a-z][a-z0-9]*_\d+\b",
        generic_text,
        re.IGNORECASE,
    ):
        raise KnowledgeError(
            f"bundled knowledge must be model-agnostic; {location} is an exact identifier"
        )
    if re.search(
        r"\b[a-z][a-z0-9_]*\.[a-z][a-z0-9_.]*\b",
        generic_text,
        re.IGNORECASE,
    ):
        raise KnowledgeError(
            f"bundled knowledge must be model-agnostic; {location} is a dotted identifier"
        )
    without_generic_one = GENERIC_ONE_TERMS.sub("", generic_text)
    if re.search(
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
        without_generic_one,
        re.IGNORECASE,
    ):
        raise KnowledgeError(
            f"bundled knowledge must be model-agnostic; {location} contains an exact cardinality"
        )
    without_generic_terms = GENERIC_NUMBERED_TERMS.sub("", without_generic_one)
    if re.search(r"\d", without_generic_terms):
        raise KnowledgeError(
            f"bundled knowledge must be model-agnostic; {location} contains run-specific digits"
        )


def _validate_model_agnostic(value: Any, location: str = "case") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_generic_text(str(key), f"{location}.<key>")
            _validate_model_agnostic(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_model_agnostic(child, f"{location}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        raise KnowledgeError(
            f"bundled knowledge must be model-agnostic; {location} is a numeric run value"
        )
    elif isinstance(value, str):
        _validate_generic_text(value, location)


def _validate_case(case: Any) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise KnowledgeError("case must be a JSON object")
    missing = sorted(CASE_FIELDS - case.keys())
    if missing:
        raise KnowledgeError(f"case missing fields: {', '.join(missing)}")
    extra = sorted(case.keys() - CASE_FIELDS)
    if extra:
        raise KnowledgeError(f"case contains extra fields: {', '.join(extra)}")
    content = {key: value for key, value in case.items() if key != "generic_review"}
    _validate_model_agnostic(content)
    case_id = case["id"]
    if not isinstance(case_id, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", case_id):
        raise KnowledgeError("case id must be lowercase kebab-case")
    for field in CASE_FIELDS - {"id"}:
        if case[field] in (None, "", [], {}):
            raise KnowledgeError(f"case field is empty: {field}")
    scope = case["scope"]
    if not isinstance(scope, dict) or set(scope) != SCOPE_FIELDS:
        raise KnowledgeError(
            "bundled knowledge must be model-agnostic; scope must contain only "
            "ep, device, and graph_requirements"
        )
    if not isinstance(scope["graph_requirements"], list) or not scope["graph_requirements"]:
        raise KnowledgeError("scope.graph_requirements must be non-empty")
    provenance = case["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_FIELDS:
        raise KnowledgeError("provenance must contain only evidence_class and scope_note")
    expected_evidence_class = STATUS_EVIDENCE_CLASSES.get(case["status"])
    if expected_evidence_class is None or provenance["evidence_class"] != expected_evidence_class:
        raise KnowledgeError(
            "case status and provenance.evidence_class must identify the same tested outcome"
        )
    if provenance["scope_note"] != RUN_LOCAL_SCOPE_NOTE:
        raise KnowledgeError("provenance.scope_note must preserve exact evidence run-local")
    generic_review = case["generic_review"]
    if not isinstance(generic_review, dict) or set(generic_review) != GENERIC_REVIEW_FIELDS:
        raise KnowledgeError("generic_review has invalid fields")
    if (
        generic_review.get("verdict") != "GENERIC_CASE_APPROVED"
        or generic_review.get("reviewer") != "independent-graph-scout"
    ):
        raise KnowledgeError(
            "generic_review must be GENERIC_CASE_APPROVED by independent-graph-scout"
        )
    if generic_review.get("content_sha256") != case_content_sha256(case):
        raise KnowledgeError("generic_review.content_sha256 does not match case content")
    discovery = case["discovery"]
    if not isinstance(discovery, dict) or set(discovery) != DISCOVERY_FIELDS:
        raise KnowledgeError("case discovery has invalid fields")
    if not isinstance(discovery["anchor_ops"], list) or not discovery["anchor_ops"]:
        raise KnowledgeError("case discovery.anchor_ops must be non-empty")
    if not isinstance(discovery["keywords"], list) or not discovery["keywords"]:
        raise KnowledgeError("case discovery.keywords must be non-empty")
    return case


def _entry(case: dict[str, Any], relative_path: str, digest: str) -> dict[str, Any]:
    discovery = case["discovery"]
    return {
        "id": case["id"],
        "status": case["status"],
        "ep": discovery["ep"],
        "device": discovery["device"],
        "anchor_ops": discovery["anchor_ops"],
        "keywords": discovery["keywords"],
        "lesson": discovery["lesson"],
        "path": relative_path,
        "sha256": digest,
    }


def validate_knowledge(knowledge_root: Path) -> dict[str, Any]:
    """Validate the lightweight index and every hash-bound case it references."""
    knowledge_root = knowledge_root.resolve()
    index_path = knowledge_root / "index.json"
    index = _load_json(index_path)
    if not isinstance(index, dict) or index.get("version") != 1:
        raise KnowledgeError("knowledge index version must be 1")
    if index.get("max_cases_per_round") != 3:
        raise KnowledgeError("max_cases_per_round must be 3")
    entries = index.get("cases")
    if not isinstance(entries, list):
        raise KnowledgeError("knowledge index cases must be a list")

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise KnowledgeError("knowledge index entry must be an object")
        case_id = entry.get("id")
        relative = entry.get("path")
        if not isinstance(case_id, str) or case_id in seen_ids:
            raise KnowledgeError(f"duplicate case id: {case_id}")
        if not isinstance(relative, str) or relative in seen_paths:
            raise KnowledgeError(f"duplicate case path: {relative}")
        seen_ids.add(case_id)
        seen_paths.add(relative)

        case_path = (knowledge_root / relative).resolve()
        if knowledge_root not in case_path.parents or not case_path.is_file():
            raise KnowledgeError(f"invalid case path: {relative}")
        if entry.get("sha256") != sha256_file(case_path):
            raise KnowledgeError(f"case hash mismatch: {case_id}")
        case = _validate_case(_load_json(case_path))
        expected = _entry(case, relative, entry["sha256"])
        if entry != expected:
            raise KnowledgeError(f"case index metadata mismatch: {case_id}")
    return index


def _temp_path(parent: Path, stem: str) -> Path:
    return parent / f".{stem}.{uuid.uuid4().hex}.tmp"


def store_case(case: dict[str, Any], knowledge_root: Path) -> Path:
    """Publish one new case and its index entry as a rollback-safe transaction."""
    case = _validate_case(case)
    knowledge_root = knowledge_root.resolve()
    cases_root = knowledge_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    index_path = knowledge_root / "index.json"
    index = validate_knowledge(knowledge_root)
    case_id = case["id"]
    relative = f"cases/{case_id}.json"
    case_path = knowledge_root / relative
    if case_path.exists() or any(entry["id"] == case_id for entry in index["cases"]):
        raise KnowledgeError(f"case already exists: {case_id}")

    case_bytes = _json_bytes(case)
    digest = hashlib.sha256(case_bytes).hexdigest()
    next_index = dict(index)
    next_index["cases"] = sorted(
        [*index["cases"], _entry(case, relative, digest)],
        key=lambda entry: entry["id"],
    )
    index_bytes = _json_bytes(next_index)
    case_temp = _temp_path(cases_root, case_id)
    index_temp = _temp_path(knowledge_root, "index")
    case_published = False
    index_replaced = False
    previous_index = index_path.read_bytes()
    try:
        case_temp.write_bytes(case_bytes)
        index_temp.write_bytes(index_bytes)
        case_temp.replace(case_path)
        case_published = True
        index_temp.replace(index_path)
        index_replaced = True
        validate_knowledge(knowledge_root)
    except Exception:
        if case_published:
            case_path.unlink(missing_ok=True)
        if index_replaced:
            rollback = _temp_path(knowledge_root, "index-rollback")
            rollback.write_bytes(previous_index)
            rollback.replace(index_path)
        raise
    finally:
        case_temp.unlink(missing_ok=True)
        index_temp.unlink(missing_ok=True)
    return case_path


def main(argv: list[str]) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--content-digest", action="store_true")
    parser.add_argument(
        "--knowledge-root", type=Path, default=Path(__file__).parents[1] / "knowledge"
    )
    args = parser.parse_args(argv)
    record = _load_json(args.record)
    if args.content_digest:
        if not isinstance(record, dict):
            raise KnowledgeError("case must be a JSON object")
        print(case_content_sha256(record))
        return 0
    path = store_case(record, args.knowledge_root)
    print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
