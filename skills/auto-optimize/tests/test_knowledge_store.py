# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Atomic knowledge persistence tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest


if TYPE_CHECKING:
    from types import ModuleType


SKILL_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = SKILL_ROOT / "scripts" / "save_case.py"


@pytest.fixture(scope="module")
def store_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("auto_optimize_save_case", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _case(case_id: str = "new-case") -> dict[str, Any]:
    case = {
        "id": case_id,
        "status": "confirmed",
        "scope": {
            "ep": "QNNExecutionProvider",
            "device": "NPU",
            "graph_requirements": ["static routed affine tail"],
        },
        "observation": "A routed affine tail remains expensive.",
        "mechanism": "A representation change exposes a safe fold.",
        "transformation": {"before": "Slice -> Mul", "after": "Split"},
        "expected_evidence": {"graph": "Mul disappears", "trace": "no added Transpose"},
        "outcome": {
            "verdict": "confirmed",
            "performance": "paired target-device gain exceeded the measured noise floor",
        },
        "safety": ["correctness pass", "paired confidence interval above zero"],
        "counterexamples": ["Scoped to this graph and toolchain."],
        "provenance": {
            "evidence_class": "paired-performance-confirmed",
            "scope_note": (
                "exact model, graph occurrence, toolchain, artifacts, and measurements "
                "remain run-local"
            ),
        },
        "generic_review": {
            "verdict": "GENERIC_CASE_APPROVED",
            "reviewer": "independent-graph-scout",
            "content_sha256": "pending",
        },
        "discovery": {
            "ep": "QNNExecutionProvider",
            "device": "NPU",
            "anchor_ops": ["Slice", "Mul"],
            "keywords": ["route", "affine"],
            "lesson": "Representation can unlock a downstream fold.",
        },
    }
    case["generic_review"]["content_sha256"] = hashlib.sha256(
        (
            json.dumps(
                {key: value for key, value in case.items() if key != "generic_review"},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    return case


def _knowledge_root(tmp_path: Path) -> Path:
    root = tmp_path / "knowledge"
    (root / "cases").mkdir(parents=True)
    (root / "index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "max_cases_per_round": 3,
                "selection": "Scoped analogies only.",
                "cases": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_store_case_updates_case_and_hash_bound_index(
    store_module: ModuleType,
    tmp_path: Path,
) -> None:
    root = _knowledge_root(tmp_path)

    path = store_module.store_case(_case(), root)

    assert path == root / "cases" / "new-case.json"
    index = store_module.validate_knowledge(root)
    assert [entry["id"] for entry in index["cases"]] == ["new-case"]
    assert index["cases"][0]["sha256"] == store_module.sha256_file(path)


def test_duplicate_case_is_rejected_without_mutation(
    store_module: ModuleType,
    tmp_path: Path,
) -> None:
    root = _knowledge_root(tmp_path)
    store_module.store_case(_case(), root)
    before = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }

    with pytest.raises(store_module.KnowledgeError, match="already exists"):
        store_module.store_case(_case(), root)

    after = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    "extra_scope",
    [
        {"identity": "a" * 64},
        {"artifact": "C:/private/source.onnx"},
        {"toolchain_build": "private-build"},
    ],
)
def test_model_specific_case_is_rejected_without_mutation(
    store_module: ModuleType,
    tmp_path: Path,
    extra_scope: dict[str, str],
) -> None:
    root = _knowledge_root(tmp_path)
    case = _case()
    case["scope"].update(extra_scope)
    before = (root / "index.json").read_bytes()

    with pytest.raises(store_module.KnowledgeError, match="model-agnostic"):
        store_module.store_case(case, root)

    assert not list((root / "cases").glob("*.json"))
    assert (root / "index.json").read_bytes() == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("measurement", 1.25),
        ("version", "runtime 2.48.0"),
        ("history", "pattern-123"),
        ("occurrence", "node_mul_19"),
        ("metric", "p50 8.0 ms"),
        ("percentage", "gain 23%"),
        ("protocol", "200 iterations"),
        ("speed", "2.4x speedup"),
        ("hash", f"source hash {'a' * 64} matched"),
        ("commit", "commit abc1234"),
        ("url", "https://example.test/artifact.zip"),
        ("ftp_url", "ftp://example.test/resource"),
        ("other_url", "ssh://example.test/resource"),
        ("artifact", "logs/profile.json"),
        ("other_artifact", "output/model.ort"),
        ("embedded_md5", f"digest {'a' * 32}"),
        ("short_version", "QNN 2.28"),
        ("date_build", "nightly 2026-08-12"),
        ("generated_node", "Conv_123"),
        ("onnx_name", "onnx::MatMul_42"),
        ("dotted_name", "fc.bias"),
        ("embedded_dotted_name", "folded fc.bias into Conv"),
        ("model_like_name", "ResNet50"),
        ("cardinality", "seven affine leaves disappeared"),
        ("single_occurrence", "one affine leaf disappeared"),
    ],
)
def test_run_specific_content_is_rejected_without_mutation(
    store_module: ModuleType,
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    root = _knowledge_root(tmp_path)
    case = _case()
    case["outcome"][field] = value
    before = (root / "index.json").read_bytes()

    with pytest.raises(store_module.KnowledgeError, match="model-agnostic"):
        store_module.store_case(case, root)

    assert not list((root / "cases").glob("*.json"))
    assert (root / "index.json").read_bytes() == before


def test_run_specific_object_key_is_rejected_without_mutation(
    store_module: ModuleType,
    tmp_path: Path,
) -> None:
    root = _knowledge_root(tmp_path)
    case = _case()
    case["outcome"]["Conv_123"] = "removed"
    before = (root / "index.json").read_bytes()

    with pytest.raises(store_module.KnowledgeError, match="model-agnostic"):
        store_module.store_case(case, root)

    assert not list((root / "cases").glob("*.json"))
    assert (root / "index.json").read_bytes() == before


@pytest.mark.parametrize(
    ("case_id", "generic_term"),
    [
        ("generic-fp", "FP16"),
        ("generic-int", "INT8"),
        ("generic-conv", "Conv2D"),
        ("generic-axis", "one static axis"),
        ("generic-step", "step-one bounds"),
        ("generic-sign", "zero or negative values"),
        ("generic-example", "e.g. a static route"),
        ("generic-domain", "ai.onnx standard domain"),
        ("generic-contrib-domain", "com.microsoft operator domain"),
    ],
)
def test_generic_technical_terms_remain_allowed(
    store_module: ModuleType,
    tmp_path: Path,
    case_id: str,
    generic_term: str,
) -> None:
    root = _knowledge_root(tmp_path)
    case = _case(case_id=case_id)
    case["observation"] = f"The reusable graph supports {generic_term}."
    case["generic_review"]["content_sha256"] = store_module.case_content_sha256(case)

    store_module.store_case(case, root)

    assert (root / "cases" / f"{case_id}.json").is_file()


@pytest.mark.parametrize(
    "generic_review",
    [
        {},
        {
            "verdict": "KEEP_RUN_LOCAL",
            "reviewer": "independent-graph-scout",
            "content_sha256": "a" * 64,
        },
        {
            "verdict": "GENERIC_CASE_APPROVED",
            "reviewer": "main-agent",
            "content_sha256": "a" * 64,
        },
    ],
)
def test_missing_or_invalid_independent_generic_review_is_rejected(
    store_module: ModuleType,
    tmp_path: Path,
    generic_review: dict[str, str],
) -> None:
    root = _knowledge_root(tmp_path)
    case = _case()
    case["generic_review"] = generic_review

    with pytest.raises(store_module.KnowledgeError, match="generic_review"):
        store_module.store_case(case, root)


def test_generic_review_digest_must_match_content(
    store_module: ModuleType,
    tmp_path: Path,
) -> None:
    root = _knowledge_root(tmp_path)
    case = _case()
    case["observation"] = "Changed after independent approval."

    with pytest.raises(store_module.KnowledgeError, match="content_sha256"):
        store_module.store_case(case, root)


def test_content_digest_mode_is_read_only(
    store_module: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case = _case()
    record = tmp_path / "candidate.json"
    record.write_text(json.dumps(case), encoding="utf-8")

    assert store_module.main(["--record", str(record), "--content-digest"]) == 0
    assert capsys.readouterr().out.strip() == store_module.case_content_sha256(case)
    assert list(tmp_path.iterdir()) == [record]


@pytest.mark.parametrize("evidence_class", ["unverified idea", "hypothetical mechanism"])
def test_unverified_provenance_is_rejected(
    store_module: ModuleType,
    tmp_path: Path,
    evidence_class: str,
) -> None:
    root = _knowledge_root(tmp_path)
    case = _case()
    case["provenance"]["evidence_class"] = evidence_class
    case["generic_review"]["content_sha256"] = store_module.case_content_sha256(case)

    with pytest.raises(store_module.KnowledgeError, match="evidence_class"):
        store_module.store_case(case, root)


def test_status_and_evidence_class_must_agree(
    store_module: ModuleType,
    tmp_path: Path,
) -> None:
    root = _knowledge_root(tmp_path)
    case = _case()
    case["provenance"]["evidence_class"] = "paired-performance-rejected"
    case["generic_review"]["content_sha256"] = store_module.case_content_sha256(case)

    with pytest.raises(store_module.KnowledgeError, match="status"):
        store_module.store_case(case, root)


def test_unknown_status_is_rejected(
    store_module: ModuleType,
    tmp_path: Path,
) -> None:
    root = _knowledge_root(tmp_path)
    case = _case()
    case["status"] = "untested"
    case["provenance"]["evidence_class"] = None
    case["generic_review"]["content_sha256"] = store_module.case_content_sha256(case)

    with pytest.raises(store_module.KnowledgeError, match="status"):
        store_module.store_case(case, root)


def test_extra_top_level_and_discovery_fields_are_rejected(
    store_module: ModuleType,
    tmp_path: Path,
) -> None:
    root = _knowledge_root(tmp_path)
    case = _case()
    case["private_context"] = "alphabetic codename"
    case["discovery"]["extra"] = "private context"
    case["generic_review"]["content_sha256"] = store_module.case_content_sha256(case)

    with pytest.raises(store_module.KnowledgeError, match=r"fields|discovery"):
        store_module.store_case(case, root)


def test_index_replace_failure_rolls_back_case(
    store_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _knowledge_root(tmp_path)
    before = (root / "index.json").read_bytes()
    real_replace = Path.replace
    calls = 0

    def fail_second_replace(source: Path, destination: str | Path) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected index replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_second_replace)

    with pytest.raises(OSError, match="injected"):
        store_module.store_case(_case(), root)

    assert not (root / "cases" / "new-case.json").exists()
    assert (root / "index.json").read_bytes() == before
