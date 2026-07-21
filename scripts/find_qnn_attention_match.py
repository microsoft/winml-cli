# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Find ONNX models that match QNN TransposeAttentionPattern quickly.

This script intentionally runs only PatternExtractor.summary(ep="qnn")
and skips any runtime checker logic for speed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import onnx


def _iter_onnx_files(root: Path) -> list[Path]:
    files = list(root.rglob("*.onnx"))

    # Prioritize likely transformer/attention models first for faster first-hit scan.
    keywords = ("attention", "transformer", "bert", "gpt", "llm", "qwen", "vit")

    def rank(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        priority = 0 if any(k in name for k in keywords) else 1
        return (priority, str(path).lower())

    return sorted(files, key=rank)


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    src_str = str(src_root)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan ONNX models and find first hit for QNN TransposeAttentionPattern."
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path(r"D:\Models"),
        help="Root directory that contains ONNX models.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts") / "find_qnn_attention_match_result.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--first-only",
        action="store_true",
        default=True,
        help="Stop scanning after the first hit (default behavior).",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=0,
        help="Optional max number of ONNX files to scan (0 means no limit).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N scanned models.",
    )
    parser.add_argument(
        "--prefilter-attention-op",
        action="store_true",
        default=True,
        help="Only run PatternExtractor on models containing Attention op (default on).",
    )
    parser.add_argument(
        "--verbose-fail",
        action="store_true",
        help="Print failures while scanning.",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Print full traceback for failures.",
    )
    args = parser.parse_args()

    if not args.models_root.exists() or not args.models_root.is_dir():
        print(f"[ERROR] models root not found or not a directory: {args.models_root}")
        return 2

    _ensure_src_on_path()

    from winml.modelkit.analyze.core.onnx_loader import ONNXLoader
    from winml.modelkit.analyze.core.pattern_extractor import PatternExtractor

    all_models = _iter_onnx_files(args.models_root)
    if args.scan_limit > 0:
        all_models = all_models[: args.scan_limit]

    if not all_models:
        print(f"[INFO] No ONNX files found under {args.models_root}")
        return 1

    t0 = time.perf_counter()
    scanned = 0
    failed = 0
    skipped_no_attention = 0
    hits: list[dict[str, Any]] = []

    print(f"[INFO] Scanning {len(all_models)} ONNX files under {args.models_root}")

    for model_path in all_models:
        scanned += 1
        try:
            if args.progress_every > 0 and scanned % args.progress_every == 0:
                print(
                    "[PROGRESS] "
                    f"scanned={scanned}/{len(all_models)} failed={failed} "
                    f"skipped_no_attention={skipped_no_attention}"
                )

            if args.prefilter_attention_op:
                model_proto = onnx.load(str(model_path), load_external_data=False)
                has_attention = any(node.op_type == "Attention" for node in model_proto.graph.node)
                if not has_attention:
                    skipped_no_attention += 1
                    continue

            model = ONNXLoader(model_path=model_path).load()
            extraction = PatternExtractor(model).summary(ep="qnn")

            by_source = extraction.get("subgraph_patterns_by_source", {})
            qnn_group = by_source.get("qnn", {})
            class_matches = qnn_group.get("TransposeAttentionPattern", [])

            if class_matches:
                hit = {
                    "model_path": str(model_path),
                    "match_class": "TransposeAttentionPattern",
                    "match_count": len(class_matches),
                    "total_extract_ms": extraction.get("total_extract_ms", 0),
                    "model_signature": extraction.get("model_signature", ""),
                    "source_stats": extraction.get("source_stats", []),
                }
                hits.append(hit)
                print(
                    "[HIT] "
                    f"{model_path} | matches={hit['match_count']} "
                    f"| extract_ms={hit['total_extract_ms']}"
                )
                if args.first_only:
                    break
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if args.verbose_fail:
                print(f"[FAIL] {model_path} | {type(exc).__name__}: {exc}")
                if args.traceback:
                    traceback.print_exc()

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    result = {
        "models_root": str(args.models_root),
        "scanned": scanned,
        "failed": failed,
        "skipped_no_attention": skipped_no_attention,
        "hit_count": len(hits),
        "elapsed_ms": elapsed_ms,
        "hits": hits,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        "[SUMMARY] "
        f"scanned={scanned} failed={failed} skipped_no_attention={skipped_no_attention} "
        f"hit_count={len(hits)} elapsed_ms={elapsed_ms}"
    )
    print(f"[SUMMARY] result saved to {args.output}")

    return 0 if hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
