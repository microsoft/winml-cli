# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Build a tiny local grouped reranking dataset from authoritative MS MARCO data.

The fixture is intentionally small: it selects 1-2 real dev queries from the
pinned Hugging Face dataset revision and joins them against the official public
MS MARCO passage-ranking files so ``winml eval --task reranking`` can run fully
offline on a local ``DatasetDict``.

Saved format:
    output/
      dataset_dict.json + Arrow shards via ``DatasetDict.save_to_disk``
      provenance.json

Each saved row contains:
    - ``input``: real query text
    - ``expected_output``: list of positive passage IDs present in candidates
    - ``metadata``: query/group provenance including the pinned HF row index
    - ``candidates``: ordered list of candidate dicts with real passage text

Usage:
    python scripts/e2e_eval/datasets/build_msmarco_reranking_fixture.py --output <dir>
    python scripts/e2e_eval/datasets/build_msmarco_reranking_fixture.py --output <dir> --queries 2 --max-negatives 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


HF_DATASET_ID = "orgrctera/msmarco_passage_ranking"
HF_REVISION = "a7388b9efd4dd4b87a0db91314e5b3f0e4b0d9e6"
HF_PARQUET_RELATIVE_PATH = "data/dev-00000-of-00001.parquet"
HF_PARQUET_URL = (
    "https://huggingface.co/datasets/"
    f"{HF_DATASET_ID}/resolve/{HF_REVISION}/{HF_PARQUET_RELATIVE_PATH}"
)

OFFICIAL_QRELS_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv"
OFFICIAL_TOP1000_URL = (
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.dev.tar.gz"
)
OFFICIAL_QUERIES_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz"

DEFAULT_CACHE = Path.home() / ".cache" / "winml" / "msmarco_reranking_fixture"


@dataclass(frozen=True)
class CandidateRow:
    """One candidate passage row from the official top1000 reranking file."""

    pid: str
    query: str
    passage: str
    rank: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(
    url: str,
    dest: Path,
    *,
    validator: Callable[[Path], None],
    expected_sha256: str | None = None,
) -> Path:
    def is_valid(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            validator(path)
        except Exception:
            return False
        return expected_sha256 is None or _sha256(path) == expected_sha256

    if is_valid(dest):
        return dest
    dest.unlink(missing_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(f"{dest.name}.part")
    partial.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as handle:  # noqa: S310
            shutil.copyfileobj(response, handle)
        if not is_valid(partial):
            raise ValueError(f"Downloaded source failed validation: {url}")
        partial.replace(dest)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return dest


def _validate_parquet(path: Path) -> None:
    import pyarrow.parquet as parquet

    parquet_file = parquet.ParquetFile(path)
    if parquet_file.metadata.num_rows == 0:
        raise ValueError("parquet file has no rows")
    for row_group in range(parquet_file.num_row_groups):
        parquet_file.read_row_group(row_group)


def _validate_qrels(path: Path) -> None:
    row_count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            qid, _unused, pid, relevance = line.split()
            if not qid or not pid:
                raise ValueError("qrels row has an empty identifier")
            int(relevance)
            row_count += 1
    if row_count == 0:
        raise ValueError("qrels file has no rows")


def _validate_tar(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if not members:
            raise ValueError("archive has no file members")
        for member in members:
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read archive member {member.name}")
            while extracted.read(1024 * 1024):
                pass


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"expected JSON list, got {type(value).__name__}")


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"expected JSON object, got {type(value).__name__}")


def _iter_tar_lines(archive_path: Path, preferred_members: Iterable[str]) -> Iterable[str]:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        member = None
        preferred = tuple(preferred_members)
        for candidate in members:
            lower_name = candidate.name.lower()
            if any(token in lower_name for token in preferred):
                member = candidate
                break
        if member is None:
            if not members:
                raise RuntimeError(f"archive {archive_path} has no file members")
            member = members[0]
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"failed to extract {member.name} from {archive_path}")
        for raw_line in extracted:
            yield raw_line.decode("utf-8").rstrip("\n")


def _load_hf_rows(parquet_path: Path) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")
    return [dict(row) for row in dataset]


def _load_queries(archive_path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    for line in _iter_tar_lines(archive_path, preferred_members=("queries.dev", "queries")):
        if not line.strip():
            continue
        qid, query = line.split("\t", 1)
        queries[qid] = query
    return queries


def _load_qrels(qrels_path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    for raw_line in qrels_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        qid, _unused, pid, rel = raw_line.split()
        if int(rel) > 0:
            qrels.setdefault(qid, set()).add(pid)
    return qrels


def _load_top1000(
    archive_path: Path,
    selected_query_ids: set[str],
) -> dict[str, list[CandidateRow]]:
    grouped: dict[str, list[CandidateRow]] = {}
    for line in _iter_tar_lines(archive_path, preferred_members=("top1000.dev", "top1000")):
        if not line.strip():
            continue
        qid, pid, query, passage = line.split("\t", 3)
        if qid not in selected_query_ids:
            continue
        rows = grouped.setdefault(qid, [])
        rows.append(CandidateRow(pid=pid, query=query, passage=passage, rank=len(rows) + 1))
    return grouped


def _select_query_ids(
    hf_rows: list[dict[str, Any]],
    queries: dict[str, str],
    qrels: dict[str, set[str]],
    *,
    max_queries: int,
) -> set[str]:
    selected: set[str] = set()
    for row in hf_rows:
        metadata = _parse_json_object(row["metadata"])
        qid = str(metadata.get("query_id", "")).strip()
        if (
            qid
            and qid in queries
            and qid in qrels
            and str(row["input"]) == queries[qid]
            and {str(value) for value in _parse_json_list(row["expected_output"])} & qrels[qid]
        ):
            selected.add(qid)
            if len(selected) >= max_queries:
                break
    return selected


def _select_rows(
    hf_rows: list[dict[str, Any]],
    queries: dict[str, str],
    qrels: dict[str, set[str]],
    top1000: dict[str, list[CandidateRow]],
    *,
    max_queries: int,
    max_negatives: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_rows: list[dict[str, Any]] = []
    selection_provenance: list[dict[str, Any]] = []

    for row_index, row in enumerate(hf_rows):
        metadata = _parse_json_object(row["metadata"])
        qid = str(metadata.get("query_id", "")).strip()
        if not qid or qid not in queries or qid not in qrels or qid not in top1000:
            continue

        query_text = str(row["input"])
        official_query = queries[qid]
        if official_query != query_text:
            continue

        hf_positive_ids = {str(value) for value in _parse_json_list(row["expected_output"])}
        official_positive_ids = {str(value) for value in qrels[qid]}

        selected_candidates: list[CandidateRow] = []
        selected_positive_ids: list[str] = []
        selected_negative_ids: list[str] = []
        for candidate in top1000[qid]:
            if candidate.pid in hf_positive_ids and candidate.pid in official_positive_ids:
                selected_candidates.append(candidate)
                selected_positive_ids.append(candidate.pid)
            elif candidate.pid not in official_positive_ids and len(selected_negative_ids) < max_negatives:
                selected_candidates.append(candidate)
                selected_negative_ids.append(candidate.pid)
            if selected_positive_ids and len(selected_negative_ids) >= max_negatives:
                break

        if not selected_positive_ids or not selected_negative_ids:
            continue

        candidates = [
            {
                "id": candidate.pid,
                "text": candidate.passage,
                "rank": candidate.rank,
                "relevant": candidate.pid in selected_positive_ids,
            }
            for candidate in selected_candidates
        ]
        metadata_out = {
            **metadata,
            "query_id": qid,
            "source_row_index": row_index,
            "positive_candidate_ids": selected_positive_ids,
            "negative_candidate_ids": selected_negative_ids,
            "selected_candidate_ids": [candidate.pid for candidate in selected_candidates],
            "selection_strategy": "authoritative_top1000_order_with_bounded_negatives",
            "candidate_source": "official_top1000.dev",
        }
        selected_rows.append(
            {
                "input": query_text,
                "expected_output": selected_positive_ids,
                "metadata": metadata_out,
                "candidates": candidates,
            }
        )
        selection_provenance.append(
            {
                "query_id": qid,
                "source_row_index": row_index,
                "hf_expected_output_ids": sorted(hf_positive_ids),
                "official_qrels_positive_ids": sorted(official_positive_ids),
                "selected_positive_candidate_ids": selected_positive_ids,
                "selected_negative_candidate_ids": selected_negative_ids,
                "selected_candidate_ids": [candidate.pid for candidate in selected_candidates],
                "selection_strategy": "authoritative_top1000_order_with_bounded_negatives",
                "candidate_ranks": {candidate.pid: candidate.rank for candidate in selected_candidates},
            }
        )
        if len(selected_rows) >= max_queries:
            break

    if len(selected_rows) < max_queries:
        raise RuntimeError(
            f"Could only materialize {len(selected_rows)} grouped queries; required {max_queries}."
        )
    return selected_rows, selection_provenance


def build_dataset(output_dir: Path, cache_dir: Path, max_queries: int, max_negatives: int) -> Path:
    from datasets import Dataset, DatasetDict

    parquet_path = _download(
        HF_PARQUET_URL,
        cache_dir / "hf" / Path(HF_PARQUET_RELATIVE_PATH).name,
        validator=_validate_parquet,
    )
    qrels_path = _download(
        OFFICIAL_QRELS_URL,
        cache_dir / "official" / "qrels.dev.tsv",
        validator=_validate_qrels,
    )
    top1000_path = _download(
        OFFICIAL_TOP1000_URL,
        cache_dir / "official" / "top1000.dev.tar.gz",
        validator=_validate_tar,
    )
    queries_path = _download(
        OFFICIAL_QUERIES_URL,
        cache_dir / "official" / "queries.tar.gz",
        validator=_validate_tar,
    )

    hf_rows = _load_hf_rows(parquet_path)
    queries = _load_queries(queries_path)
    qrels = _load_qrels(qrels_path)
    selected_query_ids = _select_query_ids(
        hf_rows,
        queries,
        qrels,
        max_queries=max_queries,
    )
    top1000 = _load_top1000(top1000_path, selected_query_ids)

    selected_rows, selection_provenance = _select_rows(
        hf_rows,
        queries,
        qrels,
        top1000,
        max_queries=max_queries,
        max_negatives=max_negatives,
    )

    dataset_dict = DatasetDict({"dev": Dataset.from_list(selected_rows)})
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))

    provenance = {
        "schema": "winml/msmarco-reranking-fixture/2",
        "builder": {
            "script": "scripts/e2e_eval/datasets/build_msmarco_reranking_fixture.py",
            "hf_dataset_id": HF_DATASET_ID,
            "hf_revision": HF_REVISION,
            "max_queries": max_queries,
            "max_negatives": max_negatives,
        },
        "sources": {
            "hf_row_source": {
                "url": HF_PARQUET_URL,
                "relative_path": HF_PARQUET_RELATIVE_PATH,
                "sha256": _sha256(parquet_path),
                "local_path": str(parquet_path),
            },
            "official_queries": {
                "url": OFFICIAL_QUERIES_URL,
                "sha256": _sha256(queries_path),
                "local_path": str(queries_path),
            },
            "official_qrels": {
                "url": OFFICIAL_QRELS_URL,
                "sha256": _sha256(qrels_path),
                "local_path": str(qrels_path),
            },
            "official_top1000": {
                "url": OFFICIAL_TOP1000_URL,
                "sha256": _sha256(top1000_path),
                "local_path": str(top1000_path),
            },
        },
        "selected_rows": selection_provenance,
        "output": {
            "dataset_path": str(output_dir),
            "split": "dev",
            "row_count": len(selected_rows),
        },
    }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a tiny grouped MS MARCO reranking fixture.")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help="Directory for downloaded source artifacts.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=2,
        help="Number of grouped dev queries to materialize.",
    )
    parser.add_argument(
        "--max-negatives",
        type=int,
        default=3,
        help="Maximum real negative candidates to retain per query.",
    )
    args = parser.parse_args()
    provenance_path = build_dataset(
        output_dir=args.output,
        cache_dir=args.cache_dir,
        max_queries=args.queries,
        max_negatives=args.max_negatives,
    )
    print(provenance_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
