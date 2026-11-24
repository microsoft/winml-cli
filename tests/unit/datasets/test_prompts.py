# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for text generation prompts loaded from a dataset.

Datasets are written to ``tmp_path`` as JSONL files, so no network is needed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from winml.modelkit.datasets import load_prompts, select_indices
from winml.modelkit.datasets.prompts import _normalize_prompt, _resolve_columns
from winml.modelkit.eval import DatasetConfig
from winml.modelkit.utils.eval_utils import DatasetValidationError


if TYPE_CHECKING:
    from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _config(path: Path, **kwargs: Any) -> DatasetConfig:
    return DatasetConfig(
        path=str(path), **{"split": None, "samples": None, "shuffle": False, **kwargs}
    )


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
    assert _normalize_prompt(value, 0) == messages


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
    with pytest.raises(DatasetValidationError, match=rf"Row 7: the prompt {message}"):
        _normalize_prompt(value, 7)


def test_resolve_columns_defaults_and_row_index_fallback() -> None:
    assert _resolve_columns({}, ["prompt", "other"]) == ("prompt", None, [])
    assert _resolve_columns({}, ["prompt", "id"]) == ("prompt", "id", [])


def test_resolve_columns_maps_and_splits_cohort_columns() -> None:
    columns = ["turns", "question_id", "category", "level"]
    assert _resolve_columns(_MAPPING, columns) == ("turns", "question_id", ["category", "level"])


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
    with pytest.raises(DatasetValidationError, match=message):
        _resolve_columns(mapping, ["turns", "category"])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_prompts_maps_columns_and_records_only_the_folder_name(tmp_path: Path) -> None:
    folder = tmp_path / "parent" / "prompts"
    _write_jsonl(folder / "data.jsonl", _rows(4))
    prompt_set = load_prompts(_config(folder, samples=3, columns_mapping=dict(_MAPPING)))

    assert [prompt.prompt_id for prompt in prompt_set.prompts] == ["q0", "q1", "q2"]
    assert prompt_set.prompts[1].messages == [{"role": "user", "content": "question 1"}]
    assert prompt_set.prompts[1].cohorts == {"category": "math", "level": "1"}
    report = prompt_set.to_dict()
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
        return load_prompts(
            _config(folder, columns_mapping=dict(_MAPPING), **kwargs)
        ).prompts_sha256

    assert digest(samples=3) == digest(samples=3)
    assert digest(samples=3) != digest(samples=2)
    assert digest(shuffle=True) != digest()


def test_load_prompts_without_id_column_uses_row_index_after_shuffle(tmp_path: Path) -> None:
    folder = tmp_path / "prompts"
    _write_jsonl(folder / "data.jsonl", [{"prompt": f"p{index}"} for index in range(6)])
    prompt_set = load_prompts(_config(folder, shuffle=True))
    expected = select_indices(6, samples=None, shuffle=True, seed=prompt_set.config.seed)
    assert [prompt.prompt_id for prompt in prompt_set.prompts] == [str(i) for i in expected]
    assert [prompt.messages[0]["content"] for prompt in prompt_set.prompts] == [
        f"p{i}" for i in expected
    ]


def test_load_prompts_rejects_repeated_ids_and_empty_prompts(tmp_path: Path) -> None:
    folder = tmp_path / "prompts"
    _write_jsonl(folder / "data.jsonl", [{"prompt": "a", "id": 1}, {"prompt": "b", "id": 1}])
    with pytest.raises(DatasetValidationError, match="Rows 0 and 1 share the prompt id '1'"):
        load_prompts(_config(folder))
    _write_jsonl(folder / "data.jsonl", [{"prompt": "a"}, {"prompt": ""}])
    with pytest.raises(DatasetValidationError, match="Row 1: the prompt is empty"):
        load_prompts(_config(folder))


def test_load_prompts_drops_null_struct_fields_from_messages(tmp_path: Path) -> None:
    import datasets

    def message_rows(*extra: dict[str, Any]) -> list[dict[str, Any]]:
        rows = [
            {"id": "x", "prompt": [{"role": "user", "content": "hi"}]},
            {"id": "y", "prompt": [{"role": "user", "content": "bye"}]},
        ]
        return rows + list(extra)

    # In Parquet every message gets every struct field; an absent one reads back as None.
    plain, wider = tmp_path / "plain.parquet", tmp_path / "wider.parquet"
    datasets.Dataset.from_list(message_rows()).to_parquet(str(plain))
    named = {"id": "z", "prompt": [{"role": "user", "content": "yo", "name": "bob"}]}
    datasets.Dataset.from_list(message_rows(named)).to_parquet(str(wider))

    first = load_prompts(_config(plain, samples=2))
    second = load_prompts(_config(wider, samples=2))
    assert second.prompts[0].messages == [{"role": "user", "content": "hi"}]
    assert first.prompts_sha256 == second.prompts_sha256
    assert load_prompts(_config(wider)).prompts[2].messages[0]["name"] == "bob"


def test_normalize_prompt_rejects_values_that_are_not_json() -> None:
    import datetime

    message = {"role": "user", "content": "hi", "sent": datetime.datetime(2026, 1, 1)}
    with pytest.raises(DatasetValidationError, match="Row 3: the prompt has a message value"):
        _normalize_prompt([message], 3)
