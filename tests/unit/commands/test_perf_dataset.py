# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for prompt datasets of ``winml perf --endpoint --dataset``.

Datasets are written to ``tmp_path`` as JSONL files or ``save_to_disk``
folders, so no network is needed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from winml.modelkit.commands._perf_dataset import (
    PromptDatasetConfig,
    PromptDatasetError,
    load_prompt_dataset,
    normalize_prompt,
    resolve_columns,
    select_rows,
)


if TYPE_CHECKING:
    from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "question_id": f"q{index}",
            "turns": [f"question {index}", "follow-up"],
            "category": "math" if index % 2 else "coding",
            "level": index % 3,
        }
        for index in range(count)
    ]


_MAPPING = {"input_column": "turns", "id_column": "question_id", "cohort_column": "category,level"}


# ---------------------------------------------------------------------------
# Prompts and columns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "messages"),
    [
        ("hi", [{"role": "user", "content": "hi"}]),
        (["first", "second"], [{"role": "user", "content": "first"}]),
        (
            [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
            [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
        ),
    ],
)
def test_normalize_prompt_shapes(value: object, messages: list[dict[str, Any]]) -> None:
    assert normalize_prompt(value, 0) == messages


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("  ", "is empty"),
        ([], "is an empty list"),
        (None, "is missing"),
        (5, "is a int"),
        (["a", {"role": "user", "content": "b"}], "mixes turns and messages"),
        ([{"role": "user"}], "has a message without a 'role' string and 'content'"),
        ([""], "is empty"),
    ],
)
def test_normalize_prompt_rejects_invalid_prompts_with_row_index(
    value: object, message: str
) -> None:
    with pytest.raises(PromptDatasetError, match=rf"Row 7: the prompt {message}"):
        normalize_prompt(value, 7)


def test_resolve_columns_defaults_and_row_index_fallback() -> None:
    assert resolve_columns({}, ["prompt", "other"]) == ("prompt", None, [])
    assert resolve_columns({}, ["prompt", "id"]) == ("prompt", "id", [])


def test_resolve_columns_maps_and_splits_cohort_columns() -> None:
    columns = ["turns", "question_id", "category", "level"]
    assert resolve_columns(_MAPPING, columns) == ("turns", "question_id", ["category", "level"])


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"prompt_column": "x"}, "Unknown --column key"),
        ({}, "no column 'prompt' \\(input_column\\). Available columns: turns, category"),
        ({"input_column": "turns", "id_column": "uid"}, "no column 'uid' \\(id_column\\)"),
        ({"input_column": "turns", "cohort_column": "category,tier"}, "no column 'tier'"),
        ({"input_column": "turns", "cohort_column": " , "}, "needs at least one column"),
    ],
)
def test_resolve_columns_rejects_unknown_keys_and_missing_columns(
    mapping: dict[str, str], message: str
) -> None:
    with pytest.raises(PromptDatasetError, match=message):
        resolve_columns(mapping, ["turns", "category"])


def test_select_rows_takes_samples_in_order_or_with_a_fixed_shuffle() -> None:
    assert select_rows(5, samples=3, shuffle=False, seed=42) == [0, 1, 2]
    assert select_rows(5, samples=None, shuffle=False, seed=42) == [0, 1, 2, 3, 4]
    shuffled = select_rows(20, samples=None, shuffle=True, seed=42)
    assert sorted(shuffled) == list(range(20))
    assert shuffled != list(range(20))
    assert select_rows(20, samples=5, shuffle=True, seed=42) == shuffled[:5]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_jsonl_folder_maps_columns_and_records_only_the_folder_name(tmp_path: Path) -> None:
    folder = tmp_path / "parent" / "prompts"
    _write_jsonl(folder / "data.jsonl", _rows(4))
    dataset = load_prompt_dataset(
        PromptDatasetConfig(path=str(folder), samples=3, columns_mapping=dict(_MAPPING))
    )

    assert [prompt.prompt_id for prompt in dataset.prompts] == ["q0", "q1", "q2"]
    assert dataset.prompts[1].messages == [{"role": "user", "content": "question 1"}]
    assert dataset.prompts[1].cohorts == {"category": "math", "level": "1"}
    report = dataset.to_dict()
    assert report["source"] == "local"
    assert report["path"] == "prompts"
    assert "parent" not in json.dumps(report)
    assert report["split"] == "train"
    assert report["row_count"] == 4
    assert report["selected_row_count"] == 3
    assert report["cohort_columns"] == ["category", "level"]
    assert report["column_mapping"]["id_column"] == "question_id"


def test_prompts_sha256_identifies_the_selected_prompts(tmp_path: Path) -> None:
    folder = tmp_path / "prompts"
    _write_jsonl(folder / "data.jsonl", _rows(4))

    def digest(**kwargs: Any) -> str:
        config = PromptDatasetConfig(path=str(folder), columns_mapping=dict(_MAPPING), **kwargs)
        return load_prompt_dataset(config).prompts_sha256

    assert digest(samples=3) == digest(samples=3)
    assert digest(samples=3) != digest(samples=2)
    assert digest(shuffle=True) != digest()


def test_load_without_id_column_uses_row_index_after_shuffle(tmp_path: Path) -> None:
    folder = tmp_path / "prompts"
    _write_jsonl(folder / "data.jsonl", [{"prompt": f"p{index}"} for index in range(6)])
    dataset = load_prompt_dataset(PromptDatasetConfig(path=str(folder), shuffle=True))
    expected = select_rows(6, samples=None, shuffle=True, seed=dataset.config.seed)
    assert [prompt.prompt_id for prompt in dataset.prompts] == [str(i) for i in expected]
    assert [prompt.messages[0]["content"] for prompt in dataset.prompts] == [
        f"p{i}" for i in expected
    ]


def test_load_saved_dataset_folder(tmp_path: Path) -> None:
    import datasets

    folder = tmp_path / "saved"
    datasets.Dataset.from_list([{"prompt": "a", "id": 10}, {"prompt": "b", "id": 11}]).save_to_disk(
        str(folder)
    )
    dataset = load_prompt_dataset(PromptDatasetConfig(path=str(folder)))
    assert [prompt.prompt_id for prompt in dataset.prompts] == ["10", "11"]
    assert dataset.to_dict()["path"] == "saved"


def test_load_needs_split_when_the_folder_has_several(tmp_path: Path) -> None:
    folder = tmp_path / "prompts"
    _write_jsonl(folder / "train.jsonl", [{"prompt": "a"}])
    _write_jsonl(folder / "test.jsonl", [{"prompt": "b"}, {"prompt": "c"}])
    with pytest.raises(PromptDatasetError, match="several splits; choose one with --split"):
        load_prompt_dataset(PromptDatasetConfig(path=str(folder)))
    dataset = load_prompt_dataset(PromptDatasetConfig(path=str(folder), split="test"))
    assert [prompt.messages[0]["content"] for prompt in dataset.prompts] == ["b", "c"]
    with pytest.raises(PromptDatasetError, match="no split 'dev'"):
        load_prompt_dataset(PromptDatasetConfig(path=str(folder), split="dev"))


def test_load_single_jsonl_file(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    _write_jsonl(path, [{"prompt": "a"}])
    dataset = load_prompt_dataset(PromptDatasetConfig(path=str(path)))
    assert dataset.to_dict()["path"] == "prompts.jsonl"


def test_load_rejects_repeated_ids_and_empty_prompts(tmp_path: Path) -> None:
    folder = tmp_path / "prompts"
    _write_jsonl(folder / "data.jsonl", [{"prompt": "a", "id": 1}, {"prompt": "b", "id": 1}])
    with pytest.raises(PromptDatasetError, match="Rows 0 and 1 share the prompt id '1'"):
        load_prompt_dataset(PromptDatasetConfig(path=str(folder)))
    _write_jsonl(folder / "data.jsonl", [{"prompt": "a"}, {"prompt": ""}])
    with pytest.raises(PromptDatasetError, match="Row 1: the prompt is empty"):
        load_prompt_dataset(PromptDatasetConfig(path=str(folder)))


def test_load_rejects_revision_for_a_local_path(tmp_path: Path) -> None:
    folder = tmp_path / "prompts"
    _write_jsonl(folder / "data.jsonl", [{"prompt": "a"}])
    with pytest.raises(PromptDatasetError, match="--dataset-revision applies only"):
        load_prompt_dataset(PromptDatasetConfig(path=str(folder), revision="main"))
