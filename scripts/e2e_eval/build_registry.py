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


HF_TASKS_URL = "https://huggingface.co/api/tasks"


def fetch_all_tasks() -> list[str]:
    """Fetch the list of pipeline tasks from the HuggingFace Hub API."""
    try:
        import requests
    except ImportError:
        safe_print("  ERROR: requests not installed (expected as a HF Hub dependency)")
        return []

    try:
        resp = requests.get(HF_TASKS_URL, timeout=30)
        resp.raise_for_status()
        return sorted(resp.json().keys())
    except Exception as exc:
        safe_print(f"  ERROR fetching tasks from {HF_TASKS_URL}: {exc}")
        return []


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


def load_curated_entries(curated_path: Path) -> list[dict]:
    """Load curated entries (hf_id + task + group + priority) from source JSON."""
    with curated_path.open(encoding="utf-8") as f:
        entries = json.load(f)
    return [
        {
            "hf_id": e["hf_id"],
            "task": e.get("task") or "",
            "group": e.get("group", "P0"),
            "priority": e.get("priority", "P0"),
        }
        for e in entries
        if "hf_id" in e
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
    tasks: list[str],
    top_n: int = 10,
    curated_models: set[str] | None = None,
    curated_entries: list[dict] | None = None,
    optimum_types: set[str] | None = None,
    existing_entries: list[dict] | None = None,
    acc_keys: set[tuple[str, str]] | None = None,
) -> list[dict]:
    """Build the registry by querying HF Hub for top models per task.

    Phase 1: HF top-N query per task.
    Phase 1.5: preserve hf_ids from existing registry not in new top-N
        (refreshed from HF, task = current pipeline_tag).
    Phase 2: merge curated entries (group/priority applied verbatim).
    Phase 3: assign per-task ``order`` field by downloads descending.
    """
    all_entries: list[dict] = []
    seen: set[tuple[str, str]] = set()
    entry_lookup: dict[tuple[str, str], dict] = {}

    # Build lookups from curated source
    curated_group_lookup: dict[tuple[str, str], str] = {}
    curated_model_group: dict[str, str] = {}
    curated_priority_lookup: dict[tuple[str, str], str] = {}
    curated_model_priority: dict[str, str] = {}
    if curated_entries:
        for c in curated_entries:
            curated_group_lookup[(c["hf_id"], c["task"])] = c["group"]
            curated_model_group[c["hf_id"]] = c["group"]
            curated_priority_lookup[(c["hf_id"], c["task"])] = c["priority"]
            curated_model_priority[c["hf_id"]] = c["priority"]

    # Build (hf_id, task) lookup from the previous registry. Phase 1 inherits
    # priority/group from this when re-picking a known entry, so labels are stable
    # across runs. Curated still wins over this; this wins over the new-rule defaults.
    existing_lookup_pt: dict[tuple[str, str], dict] = {}
    if existing_entries:
        for e in existing_entries:
            hf = e.get("hf_id")
            tk = e.get("task")
            if hf and tk is not None:
                existing_lookup_pt[(hf, tk)] = e

    # Phase 1: Query HF Hub for top models per task
    # Soft filter: prioritize Optimum-supported models, then fill remaining slots
    if top_n == 0:
        safe_print("\n  top_n=0 — skipping HF top-N queries")
    for task in (tasks if top_n > 0 else []):
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

            is_optimum = bool(optimum_types and model_type in optimum_types)
            is_curated = curated_models and model_id in curated_models
            existing_pt = existing_lookup_pt.get(key)
            if is_curated:
                priority = curated_priority_lookup.get(
                    (model_id, task)
                ) or curated_model_priority.get(model_id, "P0")
                group = curated_group_lookup.get((model_id, task)) or curated_model_group.get(
                    model_id, "Foundry Toolkit"
                )
            elif existing_pt and "priority" in existing_pt and "group" in existing_pt:
                priority = existing_pt["priority"]
                group = existing_pt["group"]
            elif model_id.startswith("microsoft/"):
                # New non-curated, non-existing Microsoft entry → P2.
                priority = "P2"
                group = "Top200"
            else:
                # New non-curated, non-existing, non-Microsoft entry → P3.
                priority = "P3"
                group = "Top200"
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

    # Phase 1.5: Preserve existing (hf_id, task) entries that Phase 1 did not re-add.
    # All fields are kept verbatim; only ``downloads`` is refreshed so Phase 3 ranking
    # reflects current popularity. Curated Phase 2 can still override group/priority later.
    if existing_entries:
        download_cache: dict[str, int] = {}
        preserved_count = 0
        for e in existing_entries:
            hf_id = e.get("hf_id")
            task = e.get("task", "")
            if not hf_id:
                continue
            key = (hf_id, task)
            if key in seen:
                continue
            preserved_entry = dict(e)
            if top_n > 0:
                if hf_id not in download_cache:
                    download_cache[hf_id] = get_model_metadata(hf_id).get("downloads", 0)
                preserved_entry["downloads"] = download_cache[hf_id]
            if optimum_types:
                model_type = preserved_entry.get("model_type")
                preserved_entry["optimum_supported"] = bool(
                    model_type and model_type in optimum_types
                )
            preserved_entry.pop("order", None)  # Phase 3 will reassign
            seen.add(key)
            entry_lookup[key] = preserved_entry
            all_entries.append(preserved_entry)
            preserved_count += 1
        if preserved_count:
            refresh_note = "downloads refreshed" if top_n > 0 else "downloads kept as-is"
            safe_print(
                f"\n  Preserved {preserved_count} existing entries not in new top-N"
                f" ({refresh_note})"
            )

    # Phase 2: Merge ALL curated entries that were not already found via HF query.
    # If an entry already exists from Phase 1, update it with curated group/priority.
    if curated_entries:
        safe_print("\n  Merging curated models...")
        for c in curated_entries:
            model_id = c["hf_id"]
            task = c["task"]
            group = c["group"]
            priority = c["priority"]

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
                # Update existing entry with curated group/priority
                existing = entry_lookup[key]
                if existing["priority"] != priority or existing["group"] != group:
                    existing["priority"] = priority
                    existing["group"] = group
                    safe_print(f"    [{priority}] {model_id} / {task} — updated (group={group})")
                continue

            # New curated entry — fetch metadata if not already loaded
            if metadata is None:
                metadata = get_model_metadata(model_id)

            model_type = get_model_type(model_id) or "unknown"
            is_optimum = bool(optimum_types and model_type in optimum_types)

            entry = {
                "hf_id": model_id,
                "task": task,
                "model_type": model_type,
                "group": group,
                "priority": priority,
                "downloads": metadata["downloads"],
                "last_update_time": metadata["last_modified"],
                "optimum_supported": is_optimum,
            }

            seen.add(key)
            entry_lookup[key] = entry
            all_entries.append(entry)
            safe_print(
                f"    [{priority}] {model_id} / {task} ({model_type}) - merged from curated source"
            )

    # Phase 3: Rank within each task by downloads (descending).
    task_groups: dict[str, list[dict]] = {}
    for e in all_entries:
        task_groups.setdefault(e["task"], []).append(e)
    for entries_in_task in task_groups.values():
        entries_in_task.sort(key=lambda x: x.get("downloads", 0), reverse=True)
        for idx, e in enumerate(entries_in_task, start=1):
            e["order"] = idx

    # Phase 4: Sync the "acc" tag on every entry from models_with_acc lookup.
    # Other tags are preserved as-is.
    for e in all_entries:
        key = (e["hf_id"], e.get("task", ""))
        tags = [t for t in e.get("tags", []) if t != "acc"]
        if acc_keys and key in acc_keys:
            tags.append("acc")
        if tags:
            e["tags"] = tags
        else:
            e.pop("tags", None)

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
        "--curated-source",
        "-s",
        type=Path,
        default=Path(__file__).parent / "testsets" / "models_curated.json",
        help="Curated model list — group/priority fields are applied verbatim",
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

    # Load curated models
    curated_entries: list[dict] = []
    curated_models: set[str] = set()
    if args.curated_source.exists():
        curated_entries = load_curated_entries(args.curated_source)
        curated_models = {e["hf_id"] for e in curated_entries}
        safe_print(
            f"Loaded {len(curated_models)} curated model IDs"
            f" ({len(curated_entries)} entries) from {args.curated_source}"
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

    tasks = fetch_all_tasks()
    if not tasks:
        safe_print("ERROR: failed to fetch task list from HuggingFace Hub")
        sys.exit(1)
    safe_print(f"Loaded {len(tasks)} pipeline tasks from {HF_TASKS_URL}")

    # Load models_with_acc.json to build the acc tag lookup
    acc_keys: set[tuple[str, str]] = set()
    acc_source = Path(__file__).parent / "testsets" / "models_with_acc.json"
    if acc_source.exists():
        try:
            with acc_source.open(encoding="utf-8") as f:
                acc_entries = json.load(f)
            acc_keys = {(e["hf_id"], e.get("task", "")) for e in acc_entries if "hf_id" in e}
            safe_print(f"Loaded {len(acc_keys)} acc entries from {acc_source.name}")
        except Exception as exc:
            safe_print(f"WARNING: could not load {acc_source.name}: {exc}")

    # Load existing registry (if present) — entries not in new top-N will be preserved
    existing_entries: list[dict] = []
    if args.output.exists():
        try:
            with args.output.open(encoding="utf-8") as f:
                existing_entries = json.load(f)
            safe_print(
                f"Loaded {len(existing_entries)} existing entries from {args.output.name}"
                " (non-top-N will be preserved)"
            )
        except Exception as exc:
            safe_print(f"WARNING: could not load existing registry at {args.output}: {exc}")

    safe_print(f"Building registry: top {args.top_n} models per task, {len(tasks)} tasks")

    entries = build_registry(
        tasks,
        args.top_n,
        curated_models,
        curated_entries,
        optimum_types,
        existing_entries,
        acc_keys,
    )
    entries.sort(key=lambda e: (e["hf_id"], e.get("task", "")))

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
