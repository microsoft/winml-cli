#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Create and advance standalone optimizer/recipe promotion handoffs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit


if TYPE_CHECKING:
    from types import ModuleType
else:
    ModuleType = type(sys)


HANDOFF_SCHEMA = "model-support-promotion-v1"
HANDOFF_FILENAME = "promotion_handoff.json"
HEX40 = re.compile(r"\A[0-9a-f]{40}\Z")
ROUTE_IDENTITIES = (
    {
        "class": "optimizer",
        "owner": "auto-optimize",
        "label": "model-opt-by-skill",
    },
    {
        "class": "recipe",
        "owner": "adding-model-support",
        "label": "model-scale-by-skill",
    },
)
OPTIMIZER_TRANSITIONS = {
    "ELIGIBLE": "DRAFT",
    "DRAFT": "READY_FOR_CHECK_IN",
    "READY_FOR_CHECK_IN": "MERGED",
}
RECIPE_TRANSITIONS = {
    "ELIGIBLE": "IN_PROGRESS",
    "IN_PROGRESS": "APPROVED",
}


class PromotionError(ValueError):
    """Raised when a promotion handoff or transition is invalid."""


@lru_cache(maxsize=1)
def _finalizer() -> ModuleType:
    path = Path(__file__).with_name("finalize_output.py")
    spec = importlib.util.spec_from_file_location("auto_optimize_finalize_output", path)
    if spec is None or spec.loader is None:
        raise PromotionError(f"cannot load bundle validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PromotionError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise PromotionError(f"{label} must be a JSON object")
    return value


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_record(path: Path, *, include_path: bool) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise PromotionError(f"missing file: {path}") from error
    record: dict[str, Any] = {
        "size_bytes": len(payload),
        "sha256": _sha256(payload),
    }
    if include_path:
        record = {"path": str(path.resolve()), **record}
    return record


def _atomic_write(path: Path, value: dict[str, Any], *, overwrite: bool) -> None:
    if not path.parent.is_dir():
        raise PromotionError(f"output directory does not exist: {path.parent}")
    if not overwrite and path.exists():
        raise PromotionError(f"promotion handoff already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(value))
        if not overwrite and path.exists():
            raise PromotionError(f"promotion handoff already exists: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_durable_huggingface_source(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("kind") != "huggingface":
        return False
    return all(
        isinstance(value.get(field), str) and bool(value[field].strip())
        for field in ("model_id", "revision", "task")
    )


def classify_routes(context: dict[str, object]) -> dict[str, object]:
    """Classify a champion into the optimizer and recipe PR routes."""
    capability = context.get("capability_change")
    optimizer_selected = isinstance(capability, dict) and capability.get("required") is True
    recipe_selected = _is_durable_huggingface_source(context.get("source_identity"))
    optimizer_status = "ELIGIBLE" if optimizer_selected else "NOT_REQUIRED"
    if not recipe_selected:
        recipe_status = "BLOCKED_IDENTITY"
    elif optimizer_selected:
        recipe_status = "BLOCKED_ON_OPTIMIZER"
    else:
        recipe_status = "ELIGIBLE"

    routes = [
        {**ROUTE_IDENTITIES[0], "status": optimizer_status},
        {**ROUTE_IDENTITIES[1], "status": recipe_status},
    ]
    order: list[str] = []
    if optimizer_selected:
        order.append("optimizer")
    if recipe_selected:
        order.append("recipe")
    return {"routes": routes, "order": order}


def _validated_bundle(bundle: Path) -> tuple[Path, dict[str, Any]]:
    bundle = bundle.resolve()
    if not bundle.is_dir():
        raise PromotionError(f"bundle is not a directory: {bundle}")
    try:
        _finalizer().validate_output_bundle(bundle)
    except Exception as error:
        raise PromotionError(f"invalid output bundle: {error}") from error
    return bundle, _file_record(bundle / "manifest.json", include_path=False)


def _absolute_file(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PromotionError(f"{label} path is required")
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise PromotionError(f"{label} path must be an absolute file: {value}")
    return path.resolve()


def _check_record(actual: dict[str, Any], expected: Any, label: str) -> None:
    if not isinstance(expected, dict):
        raise PromotionError(f"{label} record is invalid")
    if expected.get("size_bytes") != actual["size_bytes"]:
        raise PromotionError(f"{label} size mismatch")
    if expected.get("sha256") != actual["sha256"]:
        raise PromotionError(f"{label} hash mismatch")


def _route_map(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 2:
        raise PromotionError("route definition drift")
    routes: dict[str, dict[str, Any]] = {}
    for route in value:
        if not isinstance(route, dict) or route.get("class") not in {
            "optimizer",
            "recipe",
        }:
            raise PromotionError("route definition drift")
        routes[route["class"]] = route
    if set(routes) != {"optimizer", "recipe"}:
        raise PromotionError("route definition drift")
    return routes


def _require_https(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise PromotionError(f"{label} requires an HTTPS URL")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise PromotionError(f"{label} requires an HTTPS URL")
    return value


def _require_sha40(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX40.fullmatch(value) is None:
        raise PromotionError(f"{label} must be a 40-character lowercase reviewed SHA")
    return value


def _validate_routes(handoff: dict[str, Any], classification: dict[str, object]) -> None:
    if handoff.get("order") != classification["order"]:
        raise PromotionError("route order drift")
    actual = _route_map(handoff.get("routes"))
    expected = _route_map(classification["routes"])
    mixed = classification["order"] == ["optimizer", "recipe"]
    allowed_fields = {
        "optimizer": {
            "class",
            "owner",
            "label",
            "status",
            "pr_url",
            "reviewed_sha",
            "merged_commit",
            "merge_proof",
        },
        "recipe": {"class", "owner", "label", "status", "pr_url"},
    }
    allowed_statuses = {
        "optimizer": {
            "NOT_REQUIRED": {"NOT_REQUIRED"},
            "ELIGIBLE": {"ELIGIBLE", "DRAFT", "READY_FOR_CHECK_IN", "MERGED"},
        },
        "recipe": {
            "BLOCKED_IDENTITY": {"BLOCKED_IDENTITY"},
            "BLOCKED_ON_OPTIMIZER": {
                "BLOCKED_ON_OPTIMIZER",
                "ELIGIBLE",
                "IN_PROGRESS",
                "APPROVED",
            },
            "ELIGIBLE": {"ELIGIBLE", "IN_PROGRESS", "APPROVED"},
        },
    }
    for name in ("optimizer", "recipe"):
        route = actual[name]
        baseline = expected[name]
        if set(route) - allowed_fields[name]:
            raise PromotionError(f"{name} route definition drift")
        if any(route.get(field) != baseline[field] for field in ("class", "owner", "label")):
            raise PromotionError(f"{name} route ownership drift")
        initial = baseline["status"]
        if route.get("status") not in allowed_statuses[name][initial]:
            raise PromotionError(f"{name} route status drift")
    optimizer = actual["optimizer"]
    recipe = actual["recipe"]
    if optimizer["status"] in {"DRAFT", "READY_FOR_CHECK_IN", "MERGED"}:
        _require_https(optimizer.get("pr_url"), "optimizer route")
    if optimizer["status"] in {"READY_FOR_CHECK_IN", "MERGED"}:
        _require_sha40(optimizer.get("reviewed_sha"), "optimizer reviewed SHA")
    if recipe["status"] in {"IN_PROGRESS", "APPROVED"}:
        _require_https(recipe.get("pr_url"), "recipe route")
    if mixed and recipe["status"] != "BLOCKED_ON_OPTIMIZER" and optimizer["status"] != "MERGED":
        raise PromotionError("recipe is blocked on optimizer until it is MERGED")
    if mixed and optimizer["status"] == "MERGED":
        proof = optimizer.get("merge_proof")
        if not isinstance(proof, dict) or set(proof) != {
            "optimizer_repo",
            "reviewed_sha",
            "merged_commit",
            "current_main_commit",
        }:
            raise PromotionError("optimizer merge proof is invalid")
        repo_value = proof.get("optimizer_repo")
        if not isinstance(repo_value, str) or not Path(repo_value).is_absolute():
            raise PromotionError("optimizer merge proof repository must be absolute")
        reviewed = _require_sha40(proof.get("reviewed_sha"), "merge proof reviewed SHA")
        merged = _require_sha40(proof.get("merged_commit"), "merge proof merged commit")
        if reviewed != optimizer.get("reviewed_sha") or merged != optimizer.get("merged_commit"):
            raise PromotionError("optimizer merge proof does not match route state")
        _verify_mixed_merge(
            Path(repo_value),
            reviewed,
            merged,
            proof.get("current_main_commit"),
        )


def create_handoff(
    bundle_dir: Path,
    context_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Create a standalone handoff without mutating the validated bundle."""
    bundle, manifest_record = _validated_bundle(Path(bundle_dir))
    context_file = Path(context_path).resolve()
    if not context_file.is_file():
        raise PromotionError(f"missing promotion context: {context_file}")
    context = _load_json(context_file, "promotion context")
    output = (
        Path(output_path).resolve()
        if output_path is not None
        else (bundle.parent / HANDOFF_FILENAME).resolve()
    )
    handoff = {
        "schema": HANDOFF_SCHEMA,
        "bundle": {"path": str(bundle), "manifest": manifest_record},
        "context": _file_record(context_file, include_path=True),
        **classify_routes(context),
    }
    _atomic_write(output, handoff, overwrite=False)
    return output


def validate_handoff(handoff_path: Path) -> dict[str, Any]:
    """Revalidate a handoff, its bundle, its context, and route ownership."""
    path = Path(handoff_path).resolve()
    handoff = _load_json(path, "promotion handoff")
    if handoff.get("schema") != HANDOFF_SCHEMA:
        raise PromotionError("promotion handoff schema mismatch")
    bundle_record = handoff.get("bundle")
    if not isinstance(bundle_record, dict):
        raise PromotionError("bundle record is invalid")
    bundle_value = bundle_record.get("path")
    if not isinstance(bundle_value, str) or not Path(bundle_value).is_absolute():
        raise PromotionError("bundle path must be absolute")
    _bundle, manifest_record = _validated_bundle(Path(bundle_value))
    _check_record(manifest_record, bundle_record.get("manifest"), "manifest")
    context_record = handoff.get("context")
    if not isinstance(context_record, dict):
        raise PromotionError("context record is invalid")
    context_path = _absolute_file(context_record.get("path"), "context")
    _check_record(
        _file_record(context_path, include_path=False),
        context_record,
        "context",
    )
    context = _load_json(context_path, "promotion context")
    _validate_routes(handoff, classify_routes(context))
    return handoff


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    command = ["git", "-C", str(repo), *arguments]
    run_process = subprocess.run
    try:
        return run_process(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise PromotionError(f"cannot inspect optimizer repository: {repo}") from error


def _verify_mixed_merge(
    repo: Path,
    reviewed_sha: str,
    merged_commit: Any,
    current_main_commit: Any,
) -> tuple[str, str]:
    repo = Path(repo).resolve()
    if not repo.is_dir():
        raise PromotionError(f"optimizer repository is not a directory: {repo}")
    merged = _require_sha40(merged_commit, "optimizer merged commit")
    current_main = _require_sha40(current_main_commit, "current-main commit")
    origin_main = _git(repo, "rev-parse", "--verify", "refs/remotes/origin/main^{commit}")
    if origin_main.returncode != 0 or origin_main.stdout.strip() != current_main:
        raise PromotionError("local origin/main does not match current-main commit")
    for commit, label in (
        (reviewed_sha, "reviewed SHA"),
        (merged, "merged commit"),
        (current_main, "current-main commit"),
    ):
        resolved = _git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}")
        if resolved.returncode != 0 or resolved.stdout.strip() != commit:
            raise PromotionError(f"optimizer {label} is not a local commit")
    if _git(repo, "merge-base", "--is-ancestor", reviewed_sha, merged).returncode != 0:
        raise PromotionError("reviewed SHA is not an ancestor of merged commit")
    if _git(repo, "merge-base", "--is-ancestor", merged, current_main).returncode != 0:
        raise PromotionError("merged commit is not an ancestor of current main")
    return merged, current_main


def update_route(
    handoff_path: Path,
    route: str,
    status: str,
    *,
    pr_url: str | None = None,
    reviewed_sha: str | None = None,
    merged_commit: str | None = None,
    optimizer_repo: Path | None = None,
    current_main_commit: str | None = None,
) -> dict[str, Any]:
    """Apply one valid route transition and atomically replace the handoff."""
    path = Path(handoff_path).resolve()
    handoff = validate_handoff(path)
    routes = _route_map(handoff["routes"])
    if route not in routes:
        raise PromotionError("route must be optimizer or recipe")
    state = routes[route]
    current = state["status"]
    transitions = OPTIMIZER_TRANSITIONS if route == "optimizer" else RECIPE_TRANSITIONS
    if current == "BLOCKED_ON_OPTIMIZER":
        raise PromotionError("recipe is blocked on optimizer")
    if transitions.get(current) != status:
        raise PromotionError(f"invalid {route} transition: {current} -> {status}")

    if status in {"DRAFT", "IN_PROGRESS"}:
        state["pr_url"] = _require_https(pr_url, f"{route} {status}")
    if status == "READY_FOR_CHECK_IN":
        _require_https(state.get("pr_url"), "optimizer READY_FOR_CHECK_IN")
        state["reviewed_sha"] = _require_sha40(reviewed_sha, "optimizer reviewed SHA")
    if status == "APPROVED":
        _require_https(state.get("pr_url"), "recipe APPROVED")
    if status == "MERGED" and handoff["order"] == ["optimizer", "recipe"]:
        if optimizer_repo is None:
            raise PromotionError("mixed optimizer MERGED requires optimizer repo")
        stored_reviewed = _require_sha40(state.get("reviewed_sha"), "optimizer reviewed SHA")
        repo = Path(optimizer_repo).resolve()
        verified_merged, verified_main = _verify_mixed_merge(
            repo,
            stored_reviewed,
            merged_commit,
            current_main_commit,
        )
        state["merged_commit"] = verified_merged
        state["merge_proof"] = {
            "optimizer_repo": str(repo),
            "reviewed_sha": stored_reviewed,
            "merged_commit": verified_merged,
            "current_main_commit": verified_main,
        }
        routes["recipe"]["status"] = "ELIGIBLE"
    state["status"] = status
    _validate_routes(
        handoff,
        classify_routes(
            _load_json(
                _absolute_file(handoff["context"]["path"], "context"),
                "promotion context",
            )
        ),
    )
    _atomic_write(path, handoff, overwrite=True)
    return validate_handoff(path)


def _summary(path: Path, handoff: dict[str, Any]) -> dict[str, Any]:
    routes = _route_map(handoff["routes"])
    return {
        "handoff": str(path.resolve()),
        "order": handoff["order"],
        "routes": {name: routes[name]["status"] for name in ("optimizer", "recipe")},
    }


def main(argv: list[str]) -> int:
    """Create, validate, or update a promotion handoff from CLI arguments."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--bundle", required=True, type=Path)
    create.add_argument("--context", required=True, type=Path)
    create.add_argument("--output", type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("--handoff", required=True, type=Path)
    update = commands.add_parser("update")
    update.add_argument("--handoff", required=True, type=Path)
    update.add_argument("--route", required=True)
    update.add_argument("--status", required=True)
    update.add_argument("--pr-url")
    update.add_argument("--reviewed-sha")
    update.add_argument("--merged-commit")
    update.add_argument("--optimizer-repo", type=Path)
    update.add_argument("--current-main-commit")
    args = parser.parse_args(argv)

    try:
        if args.command == "create":
            path = create_handoff(args.bundle, args.context, args.output)
            handoff = validate_handoff(path)
        elif args.command == "validate":
            path = args.handoff.resolve()
            handoff = validate_handoff(path)
        else:
            path = args.handoff.resolve()
            handoff = update_route(
                path,
                args.route,
                args.status,
                pr_url=args.pr_url,
                reviewed_sha=args.reviewed_sha,
                merged_commit=args.merged_commit,
                optimizer_repo=args.optimizer_repo,
                current_main_commit=args.current_main_commit,
            )
    except PromotionError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(_summary(path, handoff), separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
