# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Find representative models for enabled default skeleton patterns.

This script intentionally uses PatternExtractor.summary() only and does not run
runtime checker logic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    src_str = str(src_root)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def _iter_onnx_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.onnx"), key=lambda p: str(p).lower())


def _load_enabled_default_classes(default_json: Path) -> list[str]:
    cfg = json.loads(default_json.read_text(encoding="utf-8"))
    classes: list[str] = []
    for entry in cfg.get("SkeletonPatternRules", []):
        if entry.get("enabled", False):
            classes.append(entry["pattern_class"])
    return classes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find one representative model for each default skeleton pattern class."
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path(r"D:\Models"),
    )
    parser.add_argument(
        "--default-json",
        type=Path,
        default=Path("src") / "winml" / "modelkit" / "pattern" / "rules" / "default.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts") / "find_default_pattern_representatives_result.json",
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
        "--verbose-fail",
        action="store_true",
    )
    args = parser.parse_args()

    if not args.models_root.exists() or not args.models_root.is_dir():
        print(f"[ERROR] models root invalid: {args.models_root}")
        return 2

    if not args.default_json.exists() or not args.default_json.is_file():
        print(f"[ERROR] default json not found: {args.default_json}")
        return 2

    _ensure_src_on_path()
    from winml.modelkit.analyze.core.onnx_loader import ONNXLoader
    from winml.modelkit.analyze.core.pattern_extractor import PatternExtractor

    target_classes = _load_enabled_default_classes(args.default_json)
    pending = set(target_classes)

    files = _iter_onnx_files(args.models_root)
    if args.scan_limit > 0:
        files = files[: args.scan_limit]

    t0 = time.perf_counter()
    scanned = 0
    failed = 0
    representatives: dict[str, dict[str, Any]] = {}

    print(f"[INFO] target_classes={target_classes}")
    print(f"[INFO] scanning {len(files)} ONNX models under {args.models_root}")

    for model_path in files:
        scanned += 1
        if args.progress_every > 0 and scanned % args.progress_every == 0:
            print(
                "[PROGRESS] "
                f"scanned={scanned}/{len(files)} failed={failed} "
                f"found={len(representatives)}/{len(target_classes)}"
            )

        try:
            model = ONNXLoader(model_path=model_path).load()
            extraction = PatternExtractor(model).summary()
            default_group = extraction.get("subgraph_patterns_by_source", {}).get("default", {})

            for class_name in list(pending):
                matches = default_group.get(class_name, [])
                if not matches:
                    continue

                first_match = matches[0]
                representatives[class_name] = {
                    "model_path": str(model_path),
                    "match_count": len(matches),
                    "pattern_id": first_match.pattern.pattern_id,
                    "total_extract_ms": extraction.get("total_extract_ms", 0),
                }
                pending.remove(class_name)
                print(
                    "[FOUND] "
                    f"{class_name} -> {model_path} (matches={len(matches)})"
                )

            if not pending:
                print("[INFO] all target classes found, stopping early")
                break

        except Exception as exc:  # noqa: BLE001
            failed += 1
            if args.verbose_fail:
                print(f"[FAIL] {model_path} | {type(exc).__name__}: {exc}")

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    result = {
        "models_root": str(args.models_root),
        "default_json": str(args.default_json),
        "target_classes": target_classes,
        "scanned": scanned,
        "failed": failed,
        "found_count": len(representatives),
        "missing_classes": sorted(list(pending)),
        "elapsed_ms": elapsed_ms,
        "representatives": representatives,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        "[SUMMARY] "
        f"scanned={scanned} failed={failed} found={len(representatives)}/{len(target_classes)} "
        f"elapsed_ms={elapsed_ms}"
    )
    print(f"[SUMMARY] missing_classes={sorted(list(pending))}")
    print(f"[SUMMARY] result saved to {args.output}")

    return 0 if not pending else 1


if __name__ == "__main__":
    raise SystemExit(main())
