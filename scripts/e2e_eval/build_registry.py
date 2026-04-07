# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Generate models.json registry from HuggingFace Hub.

Queries HuggingFace Hub API for popular models per task, enriches with
model_type from config.json, assigns priority, and writes models.json.

Usage:
    python scripts/e2e_eval/build_registry.py
    python scripts/e2e_eval/build_registry.py --top-n 5 --output models_small.json
    python scripts/e2e_eval/build_registry.py --stats
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def safe_print(text: str) -> None:
    """Cross-platform safe print (handles Windows Unicode issues)."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


NLP_TASKS = [
    "text-classification",
    "token-classification",
    "question-answering",
    "fill-mask",
    "text-generation",
    "feature-extraction",
    "summarization",
    "translation",
    "zero-shot-classification",
    "sentence-similarity",
]

CV_TASKS = [
    "image-classification",
    "object-detection",
    "image-segmentation",
    "image-feature-extraction",
    "zero-shot-image-classification",
    "depth-estimation",
    "image-to-text",
    "visual-question-answering",
    "document-question-answering",
    "mask-generation",
]

ALL_TASKS = NLP_TASKS + CV_TASKS


def get_models_for_task(task: str, top_n: int) -> list[dict]:
    """Query HF Hub API for top models by downloads for a given task."""
    try:
        from huggingface_hub import list_models
    except ImportError:
        safe_print("  ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        return []

    results = []
    for model_info in list_models(task=task, sort="downloads", direction=-1, limit=top_n):
        last_modified = getattr(model_info, "last_modified", None)
        results.append(
            {
                "model_id": model_info.id,
                "downloads": model_info.downloads or 0,
                "last_modified": last_modified.isoformat() if last_modified else None,
            }
        )
    return results


def get_model_type(model_id: str) -> str | None:
    """Read model_type from config.json via HF Hub API."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    try:
        config_path = Path(hf_hub_download(model_id, "config.json"))
        with config_path.open(encoding="utf-8") as f:
            config = json.load(f)
        return config.get("model_type")
    except Exception:
        return None


def get_model_metadata(model_id: str) -> dict:
    """Fetch last_modified, downloads, and pipeline_tag in one API call."""
    try:
        from huggingface_hub import model_info
    except ImportError:
        return {"last_modified": None, "downloads": 0, "pipeline_tag": ""}

    try:
        info = model_info(model_id)
        last_modified = getattr(info, "last_modified", None)
        return {
            "last_modified": (last_modified.isoformat() if last_modified else None),
            "downloads": getattr(info, "downloads", None) or 0,
            "pipeline_tag": getattr(info, "pipeline_tag", None) or "",
        }
    except Exception:
        return {"last_modified": None, "downloads": 0, "pipeline_tag": ""}


def load_optimum_types() -> set[str]:
    """Get Optimum-supported model_types via code (no CSV dependency).

    Requires: optimum[exporters] installed.
    Note: importing main_export triggers lazy registration of model types.
    """
    try:
        from optimum.exporters.onnx import main_export  # noqa: F401 - triggers lazy loading
        from optimum.exporters.tasks import TasksManager

        types: set[str] = set()
        for model_types in TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES.values():
            types.update(model_types.keys())
        return types
    except ImportError:
        safe_print("  WARNING: optimum not installed. Cannot determine Optimum-supported types.")
        return set()


def load_p0_entries(p0_path: Path) -> list[dict]:
    """Load P0 entries (model + task + group) from P0 source JSON."""
    with p0_path.open(encoding="utf-8") as f:
        entries = json.load(f)
    return [
        {"model": e["model"], "task": e.get("task") or "", "group": e.get("group", "P0")}
        for e in entries
        if "model" in e
    ]


def print_stats(registry_path: Path) -> None:
    """Print registry distribution summary."""
    with registry_path.open(encoding="utf-8") as f:
        entries = json.load(f)

    total = len(entries)
    unique_models = len({e["hf_id"] for e in entries})
    unique_tasks = len({e.get("task", "") for e in entries})

    priority_counts = Counter(e["priority"] for e in entries)
    group_counts = Counter(e["group"] for e in entries)
    task_counts = Counter(e.get("task", "(none)") for e in entries)
    type_counts = Counter(e.get("model_type", "unknown") for e in entries)

    safe_print(f"Registry: {registry_path.name} ({total} entries)")
    safe_print("=" * 60)

    prio_str = " | ".join(f"{k}: {v}" for k, v in sorted(priority_counts.items()))
    safe_print(f"\n  By Priority:    {prio_str}")

    group_str = " | ".join(f"{k}: {v}" for k, v in group_counts.most_common())
    safe_print(f"  By Group:       {group_str}")

    safe_print(f"  Unique Models:  {unique_models} ({total} entries across {unique_tasks} tasks)")

    safe_print("\n  By Task:")
    for task_name, count in task_counts.most_common(10):
        safe_print(f"    {task_name:<40} {count} models")
    if len(task_counts) > 10:
        safe_print(f"    ... ({len(task_counts) - 10} more tasks)")

    safe_print("\n  By Model Type (top 10):")
    for mt, count in type_counts.most_common(10):
        safe_print(f"    {mt:<16} {count} entries")
    if len(type_counts) > 10:
        safe_print(f"    ... ({len(type_counts) - 10} more types)")

    optimum_count = sum(1 for e in entries if e.get("optimum_supported"))
    safe_print(f"\n  Optimum Supported: {optimum_count}/{total}")
    has_update_time = sum(1 for e in entries if e.get("last_update_time"))
    safe_print(f"  Has last_update_time: {has_update_time}/{total}")


def build_registry(
    top_n: int = 10,
    p0_models: set[str] | None = None,
    p0_entries: list[dict] | None = None,
    optimum_types: set[str] | None = None,
) -> list[dict]:
    """Build the registry by querying HF Hub for top models per task.

    After HF query, explicitly merges ALL P0 entries to ensure none are missed
    (P0 models may not appear in top-N results for their task).
    """
    all_entries: list[dict] = []
    seen: set[tuple[str, str]] = set()
    entry_lookup: dict[tuple[str, str], dict] = {}

    # Build lookups from P0 source
    p0_group_lookup: dict[tuple[str, str], str] = {}
    p0_model_group: dict[str, str] = {}
    if p0_entries:
        for p0 in p0_entries:
            p0_group_lookup[(p0["model"], p0["task"])] = p0["group"]
            p0_model_group[p0["model"]] = p0["group"]

    # Phase 1: Query HF Hub for top models per task
    # Soft filter: prioritize Optimum-supported models, then fill remaining slots
    for task in ALL_TASKS:
        safe_print(f"\n  Task: {task}")
        # Fetch extra candidates to allow Optimum-first selection
        candidates = get_models_for_task(task, top_n * 3)

        # Resolve model_type for all candidates (last_modified already in m)
        resolved: list[dict] = []
        for m in candidates:
            model_id = m["model_id"]
            if (model_id, task) in seen:
                continue
            model_type = get_model_type(model_id)
            if model_type is None:
                safe_print(f"    SKIP {model_id} (no config.json / model_type)")
                continue
            resolved.append({**m, "model_type": model_type})

        # Split into Optimum-supported and non-supported
        if optimum_types:
            optimum_models = [m for m in resolved if m["model_type"] in optimum_types]
            non_optimum = [m for m in resolved if m["model_type"] not in optimum_types]
        else:
            optimum_models = resolved
            non_optimum = []

        # Take Optimum-supported first, then fill with non-Optimum
        selected = optimum_models[:top_n]
        remaining = top_n - len(selected)
        if remaining > 0:
            selected += non_optimum[:remaining]

        for m in selected:
            model_id = m["model_id"]
            model_type = m["model_type"]
            key = (model_id, task)
            if key in seen:
                continue

            is_p0 = p0_models and model_id in p0_models
            if is_p0:
                priority = "P0"
                group = p0_group_lookup.get((model_id, task)) or p0_model_group.get(
                    model_id, "AITK"
                )
            else:
                priority = "P1"
                group = "Top200"

            is_optimum = bool(optimum_types and model_type in optimum_types)
            # list_models() may not populate last_modified; fall back to per-model API
            last_modified = m.get("last_modified")
            if not last_modified:
                metadata = get_model_metadata(model_id)
                last_modified = metadata["last_modified"]
            entry = {
                "hf_id": model_id,
                "task": task,
                "model_type": model_type,
                "group": group,
                "priority": priority,
                "downloads": m["downloads"],
                "last_update_time": last_modified,
                "optimum_supported": is_optimum,
            }

            seen.add(key)
            entry_lookup[key] = entry
            all_entries.append(entry)
            opt_tag = "opt" if is_optimum else "non-opt"
            dl = m["downloads"]
            safe_print(f"    [{priority}] {model_id} ({model_type}, {opt_tag}) - {dl} downloads")

    # Phase 2: Merge ALL P0 entries that were not already found via HF query
    # If an entry already exists from Phase 1, promote it to P0 with correct group.
    if p0_entries:
        safe_print("\n  Merging P0 models...")
        for p0 in p0_entries:
            model_id = p0["model"]
            task = p0["task"]
            group = p0["group"]

            # Resolve empty task from HF pipeline_tag (requires API call)
            metadata: dict | None = None
            if not task:
                metadata = get_model_metadata(model_id)
                task = metadata["pipeline_tag"]
                if task:
                    safe_print(f"    Resolved task for {model_id}: {task}")
                else:
                    safe_print(f"    WARNING: {model_id} has no task")

            key = (model_id, task)
            if key in seen:
                # Promote existing entry to P0 with correct group
                existing = entry_lookup[key]
                if existing["priority"] != "P0" or existing["group"] != group:
                    existing["priority"] = "P0"
                    existing["group"] = group
                    safe_print(f"    [P0] {model_id} / {task} — promoted (group={group})")
                continue

            # New P0 entry — fetch metadata if not already loaded
            if metadata is None:
                metadata = get_model_metadata(model_id)

            model_type = get_model_type(model_id) or "unknown"
            is_optimum = bool(optimum_types and model_type in optimum_types)

            entry = {
                "hf_id": model_id,
                "task": task,
                "model_type": model_type,
                "group": group,
                "priority": "P0",
                "downloads": metadata["downloads"],
                "last_update_time": metadata["last_modified"],
                "optimum_supported": is_optimum,
            }

            seen.add(key)
            entry_lookup[key] = entry
            all_entries.append(entry)
            safe_print(f"    [P0] {model_id} / {task} ({model_type}) - merged from P0 source")

    return all_entries


def _generate_static_html(entries: list[dict], output_dir: Path) -> None:
    """Generate a standalone static HTML from models_viewer.html template.

    Reads the template (which uses fetch('./models.json')), replaces
    the async fetch with inline embedded JSON so the file works offline.
    """
    template_path = output_dir / "models_viewer.html"
    if not template_path.exists():
        safe_print("  SKIP static HTML: models_viewer.html not found")
        return

    html = template_path.read_text(encoding="utf-8")
    minified = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))

    # Replace: let DATA = [];  →  let DATA = <inline JSON>;
    before = html
    html = html.replace("let DATA = [];", f"let DATA = {minified};", 1)
    if html == before:
        safe_print("  WARNING: could not embed data (let DATA = []; not found in template)")
        return

    # Replace the async fetch init with a simple render() call
    # since data is already inline
    before = html
    html = re.sub(
        r"// Init: load data from external.*?init\(\);",
        "// Init (static build - data embedded above)\nrender();",
        html,
        flags=re.DOTALL,
    )
    if html == before:
        safe_print("  WARNING: could not replace fetch init (template comment changed?)")

    static_path = output_dir / "models_viewer_static.html"
    static_path.write_text(html, encoding="utf-8")
    safe_print(f"  Static HTML: {static_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate models.json from HuggingFace Hub")
    parser.add_argument("--top-n", type=int, default=10, help="Models per task (default: 10)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "testsets" / "models_all.json",
        help="Output file",
    )
    parser.add_argument(
        "--p0-source",
        type=Path,
        default=Path(__file__).parent / "testsets" / "models_P0.json",
        help="P0 model list for priority assignment",
    )
    parser.add_argument(
        "--no-optimum-filter",
        action="store_true",
        help="Disable Optimum-first soft filter",
    )
    parser.add_argument("--stats", action="store_true", help="Print registry stats and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing")
    args = parser.parse_args()

    if args.stats:
        if not args.output.exists():
            safe_print(f"Registry not found: {args.output}")
            sys.exit(1)
        print_stats(args.output)
        sys.exit(0)

    # Load P0 models
    p0_entries: list[dict] = []
    p0_models: set[str] = set()
    if args.p0_source.exists():
        p0_entries = load_p0_entries(args.p0_source)
        p0_models = {e["model"] for e in p0_entries}
        safe_print(
            f"Loaded {len(p0_models)} P0 model IDs"
            f" ({len(p0_entries)} entries) from {args.p0_source}"
        )

    # Load Optimum types via code
    optimum_types: set[str] | None = None
    if not args.no_optimum_filter:
        loaded = load_optimum_types()
        if loaded:
            optimum_types = loaded
            safe_print(
                f"Loaded {len(optimum_types)} Optimum-supported model types (soft filter enabled)"
            )
        else:
            safe_print("WARNING: Could not load Optimum types. Soft filter disabled.")

    safe_print(f"Building registry: top {args.top_n} models per task, {len(ALL_TASKS)} tasks")

    entries = build_registry(args.top_n, p0_models, p0_entries, optimum_types)

    safe_print(f"\n{'=' * 60}")
    safe_print(f"  Total: {len(entries)} entries")

    if args.dry_run:
        safe_print("  (dry run - not writing file)")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        safe_print(f"  Written to: {args.output}")

        # Generate standalone static HTML with embedded data
        _generate_static_html(entries, args.output.parent)


if __name__ == "__main__":
    main()
