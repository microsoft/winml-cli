# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Directly scan for missing default pattern classes via PatternMatcher.

This script registers only the target classes and searches D:\\Models for
first matching representative per class.
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


TARGET_SPECS = [
    ("winml.modelkit.pattern.gelu_patterns", "Gelu1Pattern"),
    ("winml.modelkit.pattern.gelu_patterns", "Gelu3Pattern"),
    ("winml.modelkit.pattern.gelu_patterns", "Gelu4Pattern"),
    ("winml.modelkit.pattern.layernorm_patterns", "LayerNormalizationPowPattern"),
    ("winml.modelkit.pattern.layernorm_patterns", "LayerNormalizationMulPattern"),
]


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    src_str = str(src_root)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def _iter_onnx_files(root: Path) -> list[Path]:
    files = list(root.rglob("*.onnx"))
    keywords = ("gelu", "bert", "roberta", "vit", "gpt", "llm", "layer", "norm")

    def rank(path: Path) -> tuple[int, str]:
        n = path.name.lower()
        pri = 0 if any(k in n for k in keywords) else 1
        return (pri, str(path).lower())

    return sorted(files, key=rank)


def main() -> int:
    parser = argparse.ArgumentParser(description="Find missing default patterns directly.")
    parser.add_argument("--models-root", type=Path, default=Path(r"D:\Models"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts") / "find_missing_default_patterns_direct_result.json",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--scan-limit", type=int, default=0)
    args = parser.parse_args()

    _ensure_src_on_path()

    from winml.modelkit.pattern.base import InvalidPatternMatcherModelError, PatternMatcher

    pattern_instances = {}
    for mod_name, cls_name in TARGET_SPECS:
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        pattern_instances[cls_name] = cls()

    files = _iter_onnx_files(args.models_root)
    if args.scan_limit > 0:
        files = files[: args.scan_limit]

    pending = set(pattern_instances.keys())
    reps: dict[str, dict[str, Any]] = {}
    scanned = 0
    failed = 0
    invalid = 0

    t0 = time.perf_counter()
    print(f"[INFO] target_classes={sorted(pending)}")
    print(f"[INFO] scanning {len(files)} models under {args.models_root}")

    for model_path in files:
        scanned += 1
        if args.progress_every > 0 and scanned % args.progress_every == 0:
            print(
                "[PROGRESS] "
                f"scanned={scanned}/{len(files)} failed={failed} invalid={invalid} "
                f"found={len(reps)}/{len(pattern_instances)}"
            )

        try:
            model_proto = onnx.load(str(model_path), load_external_data=False)
            try:
                matcher = PatternMatcher(model_proto, model_path=str(model_path))
            except InvalidPatternMatcherModelError:
                invalid += 1
                continue

            for class_name in list(pending):
                matcher.register_pattern(pattern_instances[class_name])

            matches = matcher.match()
            class_count: dict[str, int] = {}
            for match in matches:
                c = match.pattern.__class__.__name__
                class_count[c] = class_count.get(c, 0) + 1

            for class_name in list(pending):
                if class_count.get(class_name, 0) > 0:
                    reps[class_name] = {
                        "model_path": str(model_path),
                        "match_count": class_count[class_name],
                    }
                    pending.remove(class_name)
                    print(
                        "[FOUND] "
                        f"{class_name} -> {model_path} (matches={class_count[class_name]})"
                    )

            if not pending:
                print("[INFO] all target classes found, stopping early")
                break

        except Exception:
            failed += 1

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    result = {
        "models_root": str(args.models_root),
        "target_classes": sorted(pattern_instances.keys()),
        "scanned": scanned,
        "failed": failed,
        "invalid_model_for_matcher": invalid,
        "found_count": len(reps),
        "missing_classes": sorted(list(pending)),
        "elapsed_ms": elapsed_ms,
        "representatives": reps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        "[SUMMARY] "
        f"scanned={scanned} failed={failed} invalid={invalid} "
        f"found={len(reps)}/{len(pattern_instances)} elapsed_ms={elapsed_ms}"
    )
    print(f"[SUMMARY] missing_classes={sorted(list(pending))}")
    print(f"[SUMMARY] result saved to {args.output}")

    return 0 if not pending else 1


if __name__ == "__main__":
    raise SystemExit(main())
