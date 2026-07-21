# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Find one model that matches three target default skeleton patterns together."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import onnx


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    src_str = str(src_root)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def _iter_onnx_files(root: Path) -> list[Path]:
    files = list(root.rglob("*.onnx"))

    # Prioritize transformer-like models for faster hit.
    keywords = ("vit", "bert", "roberta", "transformer", "gpt", "llm")

    def rank(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        priority = 0 if any(k in name for k in keywords) else 1
        return (priority, str(path).lower())

    return sorted(files, key=rank)


def main() -> int:
    parser = argparse.ArgumentParser(description="Find model with 3 target patterns together.")
    parser.add_argument("--models-root", type=Path, default=Path(r"D:\Models"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts") / "find_model_with_three_default_patterns_result.json",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--scan-limit", type=int, default=0)
    args = parser.parse_args()

    _ensure_src_on_path()

    from winml.modelkit.pattern.base import InvalidPatternMatcherModelError, PatternMatcher
    from winml.modelkit.pattern.gelu_patterns import Gelu2Pattern
    from winml.modelkit.pattern.gemm_patterns import MatMulAddPattern
    from winml.modelkit.pattern.transpose_patterns import (
        ReshapeTransposeReshapeOverlyHighDimPattern,
    )

    target_classes = {
        "Gelu2Pattern",
        "MatMulAddPattern",
        "ReshapeTransposeReshapeOverlyHighDimPattern",
    }

    files = _iter_onnx_files(args.models_root)
    if args.scan_limit > 0:
        files = files[: args.scan_limit]

    scanned = 0
    failed = 0
    invalid = 0
    found = None
    t0 = time.perf_counter()

    print(f"[INFO] Scanning {len(files)} ONNX files under {args.models_root}")

    for model_path in files:
        scanned += 1
        if args.progress_every > 0 and scanned % args.progress_every == 0:
            print(
                "[PROGRESS] "
                f"scanned={scanned}/{len(files)} failed={failed} invalid={invalid}"
            )

        try:
            model_proto = onnx.load(str(model_path), load_external_data=False)
            try:
                matcher = PatternMatcher(model_proto, model_path=str(model_path))
            except InvalidPatternMatcherModelError:
                invalid += 1
                continue

            matcher.register_pattern(Gelu2Pattern())
            matcher.register_pattern(MatMulAddPattern())
            matcher.register_pattern(ReshapeTransposeReshapeOverlyHighDimPattern())

            matches = matcher.match()
            class_count: dict[str, int] = {}
            for m in matches:
                class_name = m.pattern.__class__.__name__
                class_count[class_name] = class_count.get(class_name, 0) + 1

            present = {k for k, v in class_count.items() if v > 0}
            if target_classes.issubset(present):
                found = {
                    "model_path": str(model_path),
                    "class_count": class_count,
                }
                print(
                    "[HIT] "
                    f"{model_path} | "
                    f"Gelu2={class_count.get('Gelu2Pattern', 0)} "
                    f"MatMulAdd={class_count.get('MatMulAddPattern', 0)} "
                    "RTRHigh="
                    f"{class_count.get('ReshapeTransposeReshapeOverlyHighDimPattern', 0)}"
                )
                break

        except Exception:
            failed += 1

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    result = {
        "models_root": str(args.models_root),
        "target_classes": sorted(target_classes),
        "scanned": scanned,
        "failed": failed,
        "invalid_model_for_matcher": invalid,
        "found": found,
        "elapsed_ms": elapsed_ms,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if found is None:
        print("[SUMMARY] No model found containing all three target classes together.")
    print(f"[SUMMARY] scanned={scanned} failed={failed} invalid={invalid} elapsed_ms={elapsed_ms}")
    print(f"[SUMMARY] result saved to {args.output}")

    return 0 if found is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
