# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "aiohttp",
#   "datasets>=3.6",
#   "huggingface_hub>=0.26",
#   "numpy",
#   "pandas",
#   "requests",
#   "tiktoken",
# ]
# ///
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Prepare SPEED-Bench prompt subsets for ``winml perf --dataset``.

The Hugging Face copy of SPEED-Bench (``nvidia/SPEED-Bench``) masks many prompts
and points to their source datasets. NVIDIA's standalone ``prepare.py`` rebuilds
those prompts from the sources. This script downloads the NeMo Gym
``prepare.py``, which supports SPEED-Bench v1.1, at a pinned commit, checks its
SHA-256, imports it, and uses it to rebuild only the rows each subset needs.
``winml perf`` then reads only the local files. Each subset is written as a
local dataset folder::

    <output-dir>/<subset>/test.jsonl     id, prompt, category, sub_category, bucket, source
    <output-dir>/<subset>/manifest.json  pins, counts and the recommended winml perf command

Prepared prompts come from third-party datasets under their own licenses. Keep
them out of source control.

Run with ``uv run prepare_speed_bench.py --output-dir <folder>``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any


PREPARE_PY_REPO = "NVIDIA-NeMo/Gym"
PREPARE_PY_COMMIT = "3a9b9eb981ab7a3ea0ebd517cbd4fca1d586768b"
PREPARE_PY_URL = (
    f"https://raw.githubusercontent.com/{PREPARE_PY_REPO}/"
    f"{PREPARE_PY_COMMIT}/benchmarks/speed-bench/prepare.py"
)
PREPARE_PY_SHA256 = "640de89c585d6e7a7ad8a5e09ef3621c3471e459e992848d1387f43d7e47ec8a"
SPEED_BENCH_DATASET = "nvidia/SPEED-Bench"
# Tag v1.1 of the dataset.
SPEED_BENCH_REVISION = "fa4e669bee2a4393c56aa8ed05a7c667897f9590"
PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"

QUALITATIVE = "qualitative"
THROUGHPUT_CONFIGS = (
    "throughput_1k",
    "throughput_2k",
    "throughput_8k",
    "throughput_16k",
    "throughput_32k",
)
CONFIGS = (QUALITATIVE, *THROUGHPUT_CONFIGS)
OUTPUT_COLUMNS = ("id", "prompt", "category", "sub_category", "bucket", "source")
DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1"
COHORT_COLUMNS = "bucket,category"

# Group by category; within a category, alternate between sub-category families
# (the part of ``sub_category`` before the first underscore).
BY_CATEGORY = "category"
# Group by category and sub-category family; within a group, alternate between
# exact ``sub_category`` values.
BY_FAMILY = "family"


@dataclass(frozen=True)
class Part:
    """Rows taken from some configs: ``per_group`` rows per group, or all rows."""

    configs: tuple[str, ...]
    group_by: str
    per_group: int | None


@dataclass(frozen=True)
class Subset:
    """A named prompt selection and the output length it is meant for."""

    description: str
    parts: tuple[Part, ...]
    max_new_tokens: int


SUBSETS: dict[str, Subset] = {
    "mini": Subset(
        "2 qualitative prompts per category, and 2 throughput prompts per entropy "
        "category at 1k, 8k and 16k input tokens.",
        (
            Part((QUALITATIVE,), BY_CATEGORY, 2),
            Part(("throughput_1k", "throughput_8k", "throughput_16k"), BY_CATEGORY, 2),
        ),
        256,
    ),
    "full": Subset(
        "Every qualitative prompt and every throughput prompt from 1k to 32k input tokens.",
        (Part((QUALITATIVE,), BY_CATEGORY, None), Part(THROUGHPUT_CONFIGS, BY_CATEGORY, None)),
        1024,
    ),
    "throughput": Subset(
        "2 throughput prompts per sub-category family at each input length from 1k to 32k.",
        (Part(THROUGHPUT_CONFIGS, BY_FAMILY, 2),),
        256,
    ),
    "long": Subset(
        "2 qualitative prompts per category, and 2 throughput prompts per entropy "
        "category at each input length from 1k to 32k, for long generations.",
        (Part((QUALITATIVE,), BY_CATEGORY, 2), Part(THROUGHPUT_CONFIGS, BY_CATEGORY, 2)),
        4096,
    ),
}


# =============================================================================
# Selection
# =============================================================================


def bucket_of(config: str) -> str:
    """``qualitative`` stays as is; ``throughput_8k`` becomes ``8k``."""
    return config.removeprefix("throughput_")


def family_of(sub_category: str | None) -> str:
    """The part of a sub-category before its first underscore, for example ``code``."""
    return (sub_category or "").split("_", 1)[0]


def stable_key(question_id: str) -> str:
    """Sort key that orders rows the same way on every machine."""
    return hashlib.sha256(question_id.encode("utf-8")).hexdigest()


def interleave(groups: Iterable[Sequence[Any]]) -> list[Any]:
    """Round-robin over the groups: the first item of each, then the second, ..."""
    return [item for batch in zip_longest(*groups) for item in batch if item is not None]


def order_group(rows: Iterable[Mapping[str, Any]], *, by_family: bool) -> list[Mapping[str, Any]]:
    """Alternate between sub-categories (or their families), each in stable hash order."""
    by_value: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        sub_category = row.get("sub_category")
        value = family_of(sub_category) if by_family else (sub_category or "")
        by_value.setdefault(str(value), []).append(row)
    return interleave(
        sorted(by_value[value], key=lambda row: stable_key(row["question_id"]))
        for value in sorted(by_value)
    )


def _group_key(row: Mapping[str, Any], group_by: str) -> tuple[str, ...]:
    if group_by == BY_CATEGORY:
        return (str(row.get("category")),)
    if group_by == BY_FAMILY:
        return (str(row.get("category")), family_of(row.get("sub_category")))
    raise ValueError(f"Unknown grouping '{group_by}'.")


def select_part(
    rows_by_config: Mapping[str, Sequence[Mapping[str, Any]]], part: Part
) -> list[list[tuple[str, Mapping[str, Any]]]]:
    """Return the selected ``(config, row)`` pairs of each group, groups in a fixed order."""
    groups: list[list[tuple[str, Mapping[str, Any]]]] = []
    for config in part.configs:
        by_group: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
        for row in rows_by_config.get(config, ()):
            by_group.setdefault(_group_key(row, part.group_by), []).append(row)
        for key in sorted(by_group):
            ordered = order_group(by_group[key], by_family=part.group_by == BY_CATEGORY)
            chosen = ordered if part.per_group is None else ordered[: part.per_group]
            groups.append([(config, row) for row in chosen])
    return groups


def select_subset(
    rows_by_config: Mapping[str, Sequence[Mapping[str, Any]]], subset: Subset
) -> list[tuple[str, Mapping[str, Any]]]:
    """Select a subset's rows, interleaved across groups so that any prefix is balanced."""
    return interleave(group for part in subset.parts for group in select_part(rows_by_config, part))


# =============================================================================
# Rows
# =============================================================================


def is_masked(row: Mapping[str, Any]) -> bool:
    """Whether a turn still holds the SPEED-Bench placeholder text."""
    turns = row.get("turns") or []
    return any(isinstance(turn, str) and PLACEHOLDER in turn for turn in turns)


def first_turn(turns: Sequence[Any]) -> str:
    """Text of the first user turn: a string, or the content of a message dict."""
    if not turns:
        return ""
    turn = turns[0]
    if isinstance(turn, Mapping):
        return str(turn.get("content") or "")
    return str(turn or "")


def required_sources(row: Mapping[str, Any], source_ids: Iterable[str]) -> list[str]:
    """Source datasets a masked row is rebuilt from, matched as ``prepare.py`` does."""
    if not is_masked(row):
        return []
    source = str(row.get("source") or "")
    return [source_id for source_id in source_ids if source_id in source]


def exclude_sources(
    rows: Sequence[Mapping[str, Any]], excluded: Sequence[str], source_ids: Iterable[str]
) -> tuple[list[Mapping[str, Any]], Counter[str]]:
    """Drop masked rows that need an excluded source; count them per source."""
    source_ids = list(source_ids)
    kept: list[Mapping[str, Any]] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        needed = [source for source in required_sources(row, source_ids) if source in excluded]
        if needed:
            skipped[needed[0]] += 1
        else:
            kept.append(row)
    return kept, skipped


def to_output_row(config: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Map a SPEED-Bench row to the output columns, keeping only the first turn."""
    return {
        "id": str(row["question_id"]),
        "prompt": first_turn(row.get("turns") or []),
        "category": row.get("category"),
        "sub_category": row.get("sub_category"),
        "bucket": bucket_of(config),
        "source": row.get("source"),
    }


def finalize_rows(
    selected: Sequence[tuple[str, Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], int]:
    """Convert rebuilt rows; drop rows that still hold the placeholder or have no prompt."""
    output: list[dict[str, Any]] = []
    dropped = 0
    seen: set[str] = set()
    for config, row in selected:
        if is_masked(row) or not first_turn(row.get("turns") or []).strip():
            dropped += 1
            continue
        out = to_output_row(config, row)
        if out["id"] in seen:
            raise ValueError(f"Prompt id '{out['id']}' appears twice in one subset.")
        seen.add(out["id"])
        output.append(out)
    return output, dropped


# =============================================================================
# Output
# =============================================================================


def recommended_command(dataset_dir: str, max_new_tokens: int) -> str:
    """The ``winml perf`` command suggested for a subset folder."""
    return (
        f"winml perf --endpoint {DEFAULT_ENDPOINT} --dataset {dataset_dir} "
        f"--max-new-tokens {max_new_tokens} --column cohort_column={COHORT_COLUMNS}"
    )


def build_manifest(
    name: str,
    subset: Subset,
    rows: Sequence[Mapping[str, Any]],
    *,
    data_sha256: str,
    dataset_dir: str,
    skipped_sources: Mapping[str, int],
    dropped_unresolved: int,
    versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Describe a written subset: pins, counts, skipped rows and the suggested command."""
    counts: dict[str, dict[str, int]] = {
        column: dict(sorted(Counter(str(row[column]) for row in rows).items()))
        for column in ("bucket", "category")
    }
    counts["bucket_category"] = dict(
        sorted(Counter(f"{row['bucket']}/{row['category']}" for row in rows).items())
    )
    return {
        "subset": name,
        "description": subset.description,
        "row_count": len(rows),
        "columns": list(OUTPUT_COLUMNS),
        "data_file": "test.jsonl",
        "data_sha256": data_sha256,
        "counts": counts,
        "skipped_gated_rows": dict(sorted(skipped_sources.items())),
        "dropped_unresolved_rows": dropped_unresolved,
        "selection": {
            "order": "sha256 of question_id within each group, interleaved across groups",
            "parts": [
                {"configs": list(p.configs), "group_by": p.group_by, "per_group": p.per_group}
                for p in subset.parts
            ],
        },
        "pins": {
            "speed_bench_dataset": SPEED_BENCH_DATASET,
            "speed_bench_revision": SPEED_BENCH_REVISION,
            "prepare_py_repo": PREPARE_PY_REPO,
            "prepare_py_commit": PREPARE_PY_COMMIT,
            "prepare_py_url": PREPARE_PY_URL,
            "prepare_py_sha256": PREPARE_PY_SHA256,
        },
        "versions": dict(versions or {}),
        "recommended_max_new_tokens": subset.max_new_tokens,
        "recommended_command": recommended_command(dataset_dir, subset.max_new_tokens),
        "license_note": (
            "Prompts are rebuilt from third-party datasets; their licenses apply. "
            "Do not commit the prepared data."
        ),
    }


def write_subset(
    out_dir: Path,
    name: str,
    subset: Subset,
    rows: Sequence[Mapping[str, Any]],
    *,
    skipped_sources: Mapping[str, int],
    dropped_unresolved: int,
    versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write ``<out_dir>/<name>/test.jsonl`` and its ``manifest.json``; return the manifest."""
    subset_dir = out_dir / name
    subset_dir.mkdir(parents=True, exist_ok=True)
    data = "".join(
        json.dumps({column: row[column] for column in OUTPUT_COLUMNS}, ensure_ascii=False) + "\n"
        for row in rows
    ).encode("utf-8")
    (subset_dir / "test.jsonl").write_bytes(data)
    manifest = build_manifest(
        name,
        subset,
        rows,
        data_sha256=hashlib.sha256(data).hexdigest(),
        dataset_dir=str(subset_dir),
        skipped_sources=skipped_sources,
        dropped_unresolved=dropped_unresolved,
        versions=versions,
    )
    (subset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


# =============================================================================
# Upstream prepare.py
# =============================================================================


def verify_sha256(data: bytes, expected: str = PREPARE_PY_SHA256) -> None:
    """Raise when ``data`` does not have the pinned SHA-256."""
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"prepare.py has SHA-256 {actual}, expected {expected}. "
            "The file was changed or corrupted, so it was not run."
        )


def load_prepare_module(work_dir: Path) -> Any:
    """Download (once), verify and import the pinned ``prepare.py``."""
    path = work_dir / "prepare.py"
    if not path.is_file():
        work_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading prepare.py from {PREPARE_PY_REPO} commit {PREPARE_PY_COMMIT[:12]}")
        with urllib.request.urlopen(PREPARE_PY_URL, timeout=60) as response:
            data = response.read()
        verify_sha256(data)
        path.write_bytes(data)
    verify_sha256(path.read_bytes())
    spec = importlib.util.spec_from_file_location("speed_bench_prepare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.TURNS_PLACEHOLDER != PLACEHOLDER:
        raise RuntimeError("prepare.py uses a different placeholder text than expected.")
    return module


def _gated_unavailable(source_ids: Iterable[str], force_skip: bool) -> list[str]:
    """Gated Hub datasets that the current login cannot read (all of them with force_skip)."""
    from huggingface_hub import HfApi, get_token

    api = HfApi()
    token = get_token()
    unavailable = []
    for source_id in source_ids:
        if source_id.count("/") != 1:
            continue
        try:
            gated = api.dataset_info(source_id).gated
        except Exception:
            continue
        if not gated:
            continue
        if force_skip or token is None:
            unavailable.append(source_id)
            continue
        try:
            api.auth_check(source_id, repo_type="dataset")
        except Exception:
            unavailable.append(source_id)
    return unavailable


def _resolve(
    module: Any, config: str, meta: Any, selected_ids: set[str]
) -> dict[str, dict[str, Any]]:
    """Rebuild the selected rows of one config with ``prepare.py``."""
    indices = [i for i, qid in enumerate(meta["question_id"]) if qid in selected_ids]
    if not indices:
        return {}
    resolved = module._resolve_external_data(meta.select(indices), config)
    return {row["question_id"]: row for row in resolved}


def _versions() -> dict[str, str]:
    from importlib.metadata import distributions

    wanted = {"datasets", "huggingface-hub", "tiktoken", "numpy", "pandas"}
    found = {}
    for dist in distributions():
        name = dist.metadata["Name"].lower().replace("_", "-")
        if name in wanted:
            found[name] = dist.version
    return dict(sorted(found.items()))


def prepare(out_dir: Path, subset_names: Sequence[str], *, skip_gated: bool) -> list[dict]:
    """Select, rebuild and write the named subsets; return their manifests."""
    import datasets

    module = load_prepare_module(out_dir / ".prepare")
    source_ids = [member.value for member in module.BenchmarkDataset]
    subsets = {name: SUBSETS[name] for name in subset_names}
    configs = [c for c in CONFIGS if any(c in p.configs for s in subsets.values() for p in s.parts)]

    unavailable = _gated_unavailable(source_ids, skip_gated)
    for source_id in unavailable:
        print(
            f"Skipping prompts rebuilt from the gated dataset {source_id}: the current "
            "Hugging Face login cannot read it. To include them, accept its terms on the "
            "Hub, run 'hf auth login', and run this script again."
        )

    metadata: dict[str, Any] = {}
    candidates: dict[str, list[Mapping[str, Any]]] = {}
    skipped: dict[str, Counter[str]] = {}
    for config in configs:
        print(f"Loading {SPEED_BENCH_DATASET} {config} at revision {SPEED_BENCH_REVISION[:12]}")
        meta = datasets.load_dataset(
            SPEED_BENCH_DATASET, config, split="test", revision=SPEED_BENCH_REVISION
        )
        metadata[config] = meta
        candidates[config], skipped[config] = exclude_sources(list(meta), unavailable, source_ids)

    selections = {name: select_subset(candidates, subset) for name, subset in subsets.items()}
    resolved: dict[str, dict[str, dict[str, Any]]] = {}
    for config in configs:
        wanted = {
            row["question_id"] for pairs in selections.values() for c, row in pairs if c == config
        }
        print(f"Rebuilding {len(wanted)} prompts of {config} from their sources")
        resolved[config] = _resolve(module, config, metadata[config], wanted)

    versions = _versions()
    manifests = []
    for name, subset in subsets.items():
        pairs = [(c, resolved[c][row["question_id"]]) for c, row in selections[name]]
        rows, dropped = finalize_rows(pairs)
        skipped_sources: Counter[str] = Counter()
        for config in {c for p in subset.parts for c in p.configs}:
            skipped_sources.update(skipped[config])
        manifests.append(
            write_subset(
                out_dir,
                name,
                subset,
                rows,
                skipped_sources=skipped_sources,
                dropped_unresolved=dropped,
                versions=versions,
            )
        )
    return manifests


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare SPEED-Bench prompt subsets for winml perf --dataset."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder for the subsets.")
    parser.add_argument(
        "--subset",
        action="append",
        choices=sorted(SUBSETS),
        help="Subset to write; repeat for several (default: mini).",
    )
    parser.add_argument(
        "--skip-gated",
        action="store_true",
        help="Skip prompts rebuilt from gated Hub datasets even when the login can read them.",
    )
    args = parser.parse_args(argv)
    names = list(dict.fromkeys(args.subset or ["mini"]))
    manifests = prepare(args.output_dir.resolve(), names, skip_gated=args.skip_gated)

    print()
    print("Prepared subsets")
    print("----------------")
    for manifest in manifests:
        skipped = sum(manifest["skipped_gated_rows"].values())
        print(
            f"{manifest['subset']}: {manifest['row_count']} prompts "
            f"(gated prompts skipped before selection: {skipped}, "
            f"unresolved prompts dropped: {manifest['dropped_unresolved_rows']})"
        )
        print(f"  Run: {manifest['recommended_command']}")
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
