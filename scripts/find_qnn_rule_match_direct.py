# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Find model matching qnn.json attention rule via direct PatternMatcher.

This script supports two modes:
1) commit-equivalent: only primary class (TransposeAttentionPattern)
2) rule-inclusive: primary + alternatives from qnn.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import onnx


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    src_str = str(src_root)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def _iter_onnx_files(root: Path) -> list[Path]:
    files = list(root.rglob("*.onnx"))
    keywords = ("attention", "transformer", "bert", "gpt", "llm", "qwen", "vit")

    def rank(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        priority = 0 if any(k in name for k in keywords) else 1
        return (priority, str(path).lower())

    return sorted(files, key=rank)


def _load_rule_classes(qnn_json_path: Path, include_alternatives: bool) -> list[tuple[str, str]]:
    cfg = json.loads(qnn_json_path.read_text(encoding="utf-8"))
    rules = cfg.get("SkeletonPatternRules", [])

    target = None
    for rule in rules:
        if rule.get("pattern_id") == "SUBGRAPH/TransposeAttentionPattern":
            target = rule
            break

    if target is None:
        raise RuntimeError("Cannot find SUBGRAPH/TransposeAttentionPattern in qnn.json")

    classes: list[tuple[str, str]] = [
        (target["module"], target["pattern_class"]),
    ]
    if include_alternatives:
        for alt in target.get("alternatives", []):
            classes.append((alt["module"], alt["pattern_class"]))

    # Deduplicate while preserving order.
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for item in classes:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _maybe_candidate_ops(model: onnx.ModelProto, include_alternatives: bool) -> bool:
    op_types = {n.op_type for n in model.graph.node}

    has_transpose_attention = "Transpose" in op_types and "Attention" in op_types
    if has_transpose_attention:
        return True

    if include_alternatives:
        needed = {"Transpose", "Mul", "MatMul", "Add", "Softmax"}
        if needed.issubset(op_types):
            return True

    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Find qnn attention rule matches quickly.")
    parser.add_argument("--models-root", type=Path, default=Path(r"D:\Models"))
    parser.add_argument(
        "--qnn-json",
        type=Path,
        default=Path("src") / "winml" / "modelkit" / "pattern" / "rules" / "qnn.json",
    )
    parser.add_argument(
        "--include-alternatives",
        action="store_true",
        help="Include alternatives from qnn.json in matching.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts") / "find_qnn_rule_match_direct_result.json",
    )
    args = parser.parse_args()

    _ensure_src_on_path()

    from winml.modelkit.pattern.base import InvalidPatternMatcherModelError, PatternMatcher

    files = _iter_onnx_files(args.models_root)
    if args.scan_limit > 0:
        files = files[: args.scan_limit]

    class_specs = _load_rule_classes(args.qnn_json, include_alternatives=args.include_alternatives)
    target_class_names = {class_name for _, class_name in class_specs}

    pattern_instances = []
    for module_name, class_name in class_specs:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        pattern_instances.append(cls())

    t0 = time.perf_counter()
    scanned = 0
    failed = 0
    skipped_prefilter = 0
    invalid_model_for_matcher = 0
    hits: list[dict[str, Any]] = []

    print(
        "[INFO] "
        f"Scanning {len(files)} models, include_alternatives={args.include_alternatives}, "
        f"target_classes={sorted(target_class_names)}"
    )

    for model_path in files:
        scanned += 1
        if args.progress_every > 0 and scanned % args.progress_every == 0:
            print(
                "[PROGRESS] "
                f"scanned={scanned}/{len(files)} failed={failed} "
                f"skipped_prefilter={skipped_prefilter} invalid_model_for_matcher={invalid_model_for_matcher}"
            )

        try:
            model_proto = onnx.load(str(model_path), load_external_data=False)

            if not _maybe_candidate_ops(model_proto, include_alternatives=args.include_alternatives):
                skipped_prefilter += 1
                continue

            try:
                matcher = PatternMatcher(model_proto, model_path=str(model_path))
            except InvalidPatternMatcherModelError:
                invalid_model_for_matcher += 1
                continue

            for pattern in pattern_instances:
                matcher.register_pattern(pattern)

            matches = matcher.match()
            class_hit_count: dict[str, int] = {}
            for match in matches:
                cls_name = match.pattern.__class__.__name__
                if cls_name in target_class_names:
                    class_hit_count[cls_name] = class_hit_count.get(cls_name, 0) + 1

            if class_hit_count:
                hit = {
                    "model_path": str(model_path),
                    "class_hit_count": class_hit_count,
                    "total_hit_count": sum(class_hit_count.values()),
                }
                hits.append(hit)
                print(
                    "[HIT] "
                    f"{model_path} | total_hit_count={hit['total_hit_count']} | class_hit_count={class_hit_count}"
                )
                break

        except Exception:
            failed += 1

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    result = {
        "models_root": str(args.models_root),
        "qnn_json": str(args.qnn_json),
        "include_alternatives": args.include_alternatives,
        "target_classes": sorted(target_class_names),
        "scanned": scanned,
        "failed": failed,
        "skipped_prefilter": skipped_prefilter,
        "invalid_model_for_matcher": invalid_model_for_matcher,
        "hit_count": len(hits),
        "elapsed_ms": elapsed_ms,
        "hits": hits,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        "[SUMMARY] "
        f"scanned={scanned} failed={failed} skipped_prefilter={skipped_prefilter} "
        f"invalid_model_for_matcher={invalid_model_for_matcher} hit_count={len(hits)} elapsed_ms={elapsed_ms}"
    )
    print(f"[SUMMARY] result saved to {args.output}")

    return 0 if hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
