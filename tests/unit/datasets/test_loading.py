# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for loading a ``DatasetConfig`` split and selecting its rows.

Datasets are written to ``tmp_path`` as JSONL files or ``save_to_disk``
folders, so no network is needed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from winml.modelkit.datasets import load_split, select_indices
from winml.modelkit.eval import DatasetConfig
from winml.modelkit.utils.eval_utils import DatasetValidationError


if TYPE_CHECKING:
    from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _config(path: Path, **kwargs: Any) -> DatasetConfig:
    return DatasetConfig(path=str(path), **{"split": None, "samples": None, **kwargs})


def test_select_indices_takes_samples_in_order() -> None:
    assert select_indices(5, samples=3, shuffle=False, seed=42) == [0, 1, 2]
    assert select_indices(5, samples=None, shuffle=False, seed=42) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(("samples", "seed"), [(None, 42), (5, 42), (5, 7)])
def test_select_indices_selects_the_rows_eval_selects(samples: int | None, seed: int) -> None:
    import datasets

    rows = [{"row": index} for index in range(20)]
    shuffled = datasets.Dataset.from_list(rows).shuffle(seed=seed)
    expected = [row["row"] for row in shuffled.select(range(samples or len(rows)))]
    assert select_indices(len(rows), samples=samples, shuffle=True, seed=seed) == expected


def test_load_split_of_a_jsonl_folder_records_only_the_folder_name(tmp_path: Path) -> None:
    folder = tmp_path / "parent" / "prompts"
    _write_jsonl(folder / "data.jsonl", [{"prompt": "a"}, {"prompt": "b"}])
    loaded = load_split(_config(folder))
    assert (loaded.source, loaded.display_path, loaded.split) == ("local", "prompts", "train")
    assert len(loaded.dataset) == 2


def test_load_split_of_a_saved_dataset_folder(tmp_path: Path) -> None:
    import datasets

    folder = tmp_path / "saved"
    datasets.Dataset.from_list([{"prompt": "a"}, {"prompt": "b"}]).save_to_disk(str(folder))
    loaded = load_split(_config(folder))
    assert loaded.display_path == "saved"
    assert loaded.dataset["prompt"] == ["a", "b"]


def test_load_split_of_a_single_jsonl_file(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    _write_jsonl(path, [{"prompt": "a"}])
    assert load_split(_config(path)).display_path == "prompts.jsonl"


def test_load_split_needs_split_when_the_folder_has_several(tmp_path: Path) -> None:
    folder = tmp_path / "prompts"
    _write_jsonl(folder / "train.jsonl", [{"prompt": "a"}])
    _write_jsonl(folder / "test.jsonl", [{"prompt": "b"}, {"prompt": "c"}])
    with pytest.raises(DatasetValidationError, match="several splits; choose one with --split"):
        load_split(_config(folder))
    assert load_split(_config(folder, split="test")).dataset["prompt"] == ["b", "c"]
    with pytest.raises(DatasetValidationError, match="no split 'dev'"):
        load_split(_config(folder, split="dev"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"revision": "main"}, "--dataset-revision applies only"),
        ({"name": "config"}, "--dataset-name does not apply to a saved dataset folder"),
    ],
)
def test_load_split_rejects_hub_options_for_a_saved_folder(
    tmp_path: Path, kwargs: dict[str, str], message: str
) -> None:
    import datasets

    folder = tmp_path / "saved"
    datasets.Dataset.from_list([{"prompt": "a"}]).save_to_disk(str(folder))
    with pytest.raises(DatasetValidationError, match=message):
        load_split(_config(folder, **kwargs))
