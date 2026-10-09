# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Behavioral tests for optimizer and recipe promotion routing."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest


if TYPE_CHECKING:
    from types import ModuleType


SKILL_ROOT = Path(__file__).resolve().parents[1]
PROMOTION_PATH = SKILL_ROOT / "scripts" / "promotion.py"
FINALIZER_PATH = SKILL_ROOT / "scripts" / "finalize_output.py"
OUTPUT_TEST_PATH = Path(__file__).with_name("test_output_bundle.py")


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _promotion() -> ModuleType:
    assert PROMOTION_PATH.is_file(), "promotion.py must implement the routing contract"
    return _load_module("auto_optimize_promotion", PROMOTION_PATH)


def _context(*, capability_required: Any, durable_source: bool) -> dict[str, Any]:
    return {
        "capability_change": {"required": capability_required},
        "source_identity": {
            "kind": "huggingface" if durable_source else "local",
            "model_id": "microsoft/resnet-50",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "task": "image-classification",
        },
    }


def _routes(classification: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["class"]: record for record in classification["routes"]}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_bundle(tmp_path: Path) -> Path:
    support = _load_module("promotion_output_test_support", OUTPUT_TEST_PATH)
    finalizer = _load_module("promotion_finalize_output", FINALIZER_PATH)
    report, champion, config, companion = support._inputs(tmp_path)
    output = tmp_path / "bundle"
    finalizer.finalize_output(report, champion, config, [companion], output)
    return output


def _make_handoff(
    tmp_path: Path,
    *,
    capability_required: bool,
    durable_source: bool,
) -> tuple[ModuleType, Path, Path]:
    promotion = _promotion()
    bundle = _make_bundle(tmp_path)
    context_path = tmp_path / "promotion-context.json"
    _write_json(
        context_path,
        _context(
            capability_required=capability_required,
            durable_source=durable_source,
        ),
    )
    return promotion, promotion.create_handoff(bundle, context_path), context_path


@pytest.mark.parametrize(
    ("capability_required", "durable_source", "statuses", "order"),
    [
        (False, True, ("NOT_REQUIRED", "ELIGIBLE"), ["recipe"]),
        (True, False, ("ELIGIBLE", "BLOCKED_IDENTITY"), ["optimizer"]),
        (True, True, ("ELIGIBLE", "BLOCKED_ON_OPTIMIZER"), ["optimizer", "recipe"]),
    ],
)
def test_classify_routes_returns_two_owned_routes_in_required_order(
    capability_required: bool,
    durable_source: bool,
    statuses: tuple[str, str],
    order: list[str],
) -> None:
    classification = _promotion().classify_routes(
        _context(
            capability_required=capability_required,
            durable_source=durable_source,
        )
    )

    assert classification == {
        "routes": [
            {
                "class": "optimizer",
                "owner": "auto-optimize",
                "label": "model-opt-by-skill",
                "status": statuses[0],
            },
            {
                "class": "recipe",
                "owner": "adding-model-support",
                "label": "model-scale-by-skill",
                "status": statuses[1],
            },
        ],
        "order": order,
    }


@pytest.mark.parametrize("missing", ["model_id", "revision", "task"])
def test_recipe_requires_complete_durable_huggingface_identity(missing: str) -> None:
    context = _context(capability_required=False, durable_source=True)
    context["source_identity"][missing] = ""

    classification = _promotion().classify_routes(context)

    assert _routes(classification)["recipe"]["status"] == "BLOCKED_IDENTITY"
    assert classification["order"] == []
    non_boolean = _promotion().classify_routes(
        _context(capability_required="true", durable_source=True)
    )
    assert _routes(non_boolean)["optimizer"]["status"] == "NOT_REQUIRED"


def test_create_and_validate_standalone_hash_bound_handoff(tmp_path: Path) -> None:
    promotion = _promotion()
    bundle = _make_bundle(tmp_path)
    context_path = tmp_path / "context.json"
    context = _context(capability_required=False, durable_source=True)
    _write_json(context_path, context)
    manifest_path = bundle / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    report_before = (bundle / "report.json").read_bytes()

    handoff_path = promotion.create_handoff(bundle, context_path)

    assert handoff_path == (tmp_path / "promotion_handoff.json").resolve()
    assert handoff_path.parent == bundle.parent and handoff_path.parent != bundle
    handoff = promotion.validate_handoff(handoff_path)
    assert handoff == {
        "schema": "model-support-promotion-v1",
        "bundle": {
            "path": str(bundle.resolve()),
            "manifest": {
                "size_bytes": manifest_path.stat().st_size,
                "sha256": hashlib.sha256(manifest_before).hexdigest(),
            },
        },
        "context": {
            "path": str(context_path.resolve()),
            "size_bytes": context_path.stat().st_size,
            "sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
        },
        **promotion.classify_routes(context),
    }
    assert manifest_path.read_bytes() == manifest_before
    assert (bundle / "report.json").read_bytes() == report_before
    with pytest.raises(promotion.PromotionError, match="already exists"):
        promotion.create_handoff(bundle, context_path)

    handoff["routes"][0]["owner"] = "adding-model-support"
    _write_json(handoff_path, handoff)
    with pytest.raises(promotion.PromotionError, match="drift"):
        promotion.validate_handoff(handoff_path)


def test_validate_rejects_context_and_manifest_record_tampering(tmp_path: Path) -> None:
    promotion, handoff_path, context_path = _make_handoff(
        tmp_path,
        capability_required=False,
        durable_source=True,
    )
    handoff = promotion.validate_handoff(handoff_path)
    original_context = context_path.read_bytes()
    context_path.write_text("{}", encoding="utf-8")
    with pytest.raises(promotion.PromotionError, match=r"context.*(size|hash)"):
        promotion.validate_handoff(handoff_path)

    context_path.write_bytes(original_context)
    handoff["bundle"]["manifest"]["sha256"] = "0" * 64
    _write_json(handoff_path, handoff)
    with pytest.raises(promotion.PromotionError, match=r"manifest.*hash"):
        promotion.validate_handoff(handoff_path)


def test_optimizer_transition_requirements_preserve_route_ownership(
    tmp_path: Path,
) -> None:
    promotion, handoff_path, _ = _make_handoff(
        tmp_path,
        capability_required=True,
        durable_source=False,
    )
    identities = {
        route["class"]: (route["class"], route["owner"], route["label"])
        for route in promotion.validate_handoff(handoff_path)["routes"]
    }

    with pytest.raises(promotion.PromotionError, match="HTTPS"):
        promotion.update_route(
            handoff_path,
            "optimizer",
            "DRAFT",
            pr_url="http://example.test/pr/1",
        )
    promotion.update_route(
        handoff_path,
        "optimizer",
        "DRAFT",
        pr_url="https://example.test/pr/1",
    )
    with pytest.raises(promotion.PromotionError, match="reviewed SHA"):
        promotion.update_route(
            handoff_path,
            "optimizer",
            "READY_FOR_CHECK_IN",
            reviewed_sha="ABC",
        )
    ready = promotion.update_route(
        handoff_path,
        "optimizer",
        "READY_FOR_CHECK_IN",
        reviewed_sha="a" * 40,
    )
    assert _routes(ready)["optimizer"]["reviewed_sha"] == "a" * 40
    merged = promotion.update_route(handoff_path, "optimizer", "MERGED")
    assert _routes(merged)["optimizer"]["status"] == "MERGED"
    assert {
        route["class"]: (route["class"], route["owner"], route["label"])
        for route in merged["routes"]
    } == identities


def test_recipe_transition_requires_https_pr_url(tmp_path: Path) -> None:
    promotion, handoff_path, _ = _make_handoff(
        tmp_path,
        capability_required=False,
        durable_source=True,
    )
    with pytest.raises(promotion.PromotionError, match="HTTPS"):
        promotion.update_route(
            handoff_path,
            "recipe",
            "IN_PROGRESS",
            pr_url="file:///recipe",
        )
    promotion.update_route(
        handoff_path,
        "recipe",
        "IN_PROGRESS",
        pr_url="https://example.test/recipe/2",
    )
    approved = promotion.update_route(handoff_path, "recipe", "APPROVED")

    assert _routes(approved)["recipe"]["status"] == "APPROVED"
    assert _routes(approved)["recipe"]["owner"] == "adding-model-support"


def test_mixed_recipe_cannot_be_activated_directly(tmp_path: Path) -> None:
    promotion, handoff_path, _ = _make_handoff(
        tmp_path,
        capability_required=True,
        durable_source=True,
    )
    with pytest.raises(promotion.PromotionError, match="blocked on optimizer"):
        promotion.update_route(handoff_path, "recipe", "ELIGIBLE")


def _git(repo: Path, *arguments: str) -> str:
    command = ["git", "-C", str(repo), *arguments]
    run_process = subprocess.run
    result = run_process(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(repo: Path, filename: str, content: str) -> str:
    (repo / filename).write_text(content, encoding="utf-8")
    _git(repo, "add", filename)
    _git(repo, "commit", "-m", filename)
    return _git(repo, "rev-parse", "HEAD")


def test_mixed_optimizer_merge_uses_local_ancestry_and_unlocks_recipe(
    tmp_path: Path,
) -> None:
    promotion, handoff_path, _ = _make_handoff(
        tmp_path,
        capability_required=True,
        durable_source=True,
    )
    repo = tmp_path / "optimizer-repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Promotion Test")
    _git(repo, "config", "user.email", "promotion@example.test")
    reviewed_sha = _commit(repo, "reviewed.txt", "reviewed")
    merged_commit = _commit(repo, "merged.txt", "merged")
    current_main_commit = _commit(repo, "main.txt", "main")
    _git(repo, "update-ref", "refs/remotes/origin/main", current_main_commit)
    promotion.update_route(
        handoff_path,
        "optimizer",
        "DRAFT",
        pr_url="https://example.test/optimizer/3",
    )
    promotion.update_route(
        handoff_path,
        "optimizer",
        "READY_FOR_CHECK_IN",
        reviewed_sha=reviewed_sha,
    )

    with pytest.raises(promotion.PromotionError, match="origin/main"):
        promotion.update_route(
            handoff_path,
            "optimizer",
            "MERGED",
            optimizer_repo=repo,
            merged_commit=merged_commit,
            current_main_commit=reviewed_sha,
        )
    merged = promotion.update_route(
        handoff_path,
        "optimizer",
        "MERGED",
        optimizer_repo=repo,
        merged_commit=merged_commit,
        current_main_commit=current_main_commit,
    )

    assert _routes(merged)["optimizer"]["status"] == "MERGED"
    assert _routes(merged)["recipe"]["status"] == "ELIGIBLE"
    assert _routes(merged)["optimizer"]["merged_commit"] == merged_commit
    assert _routes(merged)["optimizer"]["merge_proof"] == {
        "optimizer_repo": str(repo.resolve()),
        "reviewed_sha": reviewed_sha,
        "merged_commit": merged_commit,
        "current_main_commit": current_main_commit,
    }
    assert promotion.validate_handoff(handoff_path) == merged

    _git(repo, "update-ref", "refs/remotes/origin/main", reviewed_sha)
    with pytest.raises(promotion.PromotionError, match="origin/main"):
        promotion.validate_handoff(handoff_path)


def test_cli_emits_compact_summaries_and_validation_errors(tmp_path: Path) -> None:
    _promotion()
    bundle = _make_bundle(tmp_path)
    context_path = tmp_path / "context.json"
    handoff_path = tmp_path / "handoff.json"
    _write_json(context_path, _context(capability_required=False, durable_source=True))

    create_command = [
        sys.executable,
        str(PROMOTION_PATH),
        "create",
        "--bundle",
        str(bundle),
        "--context",
        str(context_path),
        "--output",
        str(handoff_path),
    ]
    run_process = subprocess.run
    create = run_process(
        create_command,
        capture_output=True,
        text=True,
        check=False,
    )
    assert create.returncode == 0, create.stderr
    assert json.loads(create.stdout)["handoff"] == str(handoff_path.resolve())
    assert ": " not in create.stdout and ", " not in create.stdout

    validate_command = [
        sys.executable,
        str(PROMOTION_PATH),
        "validate",
        "--handoff",
        str(handoff_path),
    ]
    validate = run_process(
        validate_command,
        capture_output=True,
        text=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr
    assert json.loads(validate.stdout)["routes"]["recipe"] == "ELIGIBLE"

    invalid_command = [
        sys.executable,
        str(PROMOTION_PATH),
        "update",
        "--handoff",
        str(handoff_path),
        "--route",
        "recipe",
        "--status",
        "APPROVED",
    ]
    invalid = run_process(
        invalid_command,
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode == 2
    assert invalid.stdout == ""
    assert invalid.stderr and "Traceback" not in invalid.stderr
