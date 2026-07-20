# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for :class:`PromptDataset` and :class:`PromptRecord`."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from winml.modelkit.datasets import PromptDataset, PromptRecord


if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# PromptRecord dataclass
# =============================================================================


class TestPromptRecord:
    def test_defaults(self) -> None:
        r = PromptRecord(prompt="a photo of a cat")
        assert r.prompt == "a photo of a cat"
        assert r.negative_prompt is None
        assert r.reference_text is None
        assert r.reference_image is None
        assert r.metadata == {}

    def test_to_dict_populates_all_keys(self) -> None:
        r = PromptRecord(
            prompt="hi",
            negative_prompt="lo",
            reference_text="ref",
            reference_image="ref.png",
            metadata={"src": "unit"},
        )
        d = r.to_dict()
        assert set(d) == {
            "prompt",
            "negative_prompt",
            "reference_text",
            "reference_image",
            "metadata",
        }
        assert d["metadata"] == {"src": "unit"}

    def test_metadata_dict_is_copied(self) -> None:
        meta = {"k": 1}
        r = PromptRecord(prompt="hi", metadata=meta)
        d = r.to_dict()
        d["metadata"]["k"] = 99
        # Original metadata is untouched (to_dict copies).
        assert meta == {"k": 1}


# =============================================================================
# Construction & validation
# =============================================================================


class TestConstruction:
    def test_from_list_dicts(self) -> None:
        ds = PromptDataset.from_list([{"prompt": "hello"}, {"prompt": "world"}])
        assert len(ds) == 2
        assert ds[0]["prompt"] == "hello"
        assert ds[1]["prompt"] == "world"

    def test_from_list_records(self) -> None:
        records = [PromptRecord(prompt="a"), PromptRecord(prompt="b")]
        ds = PromptDataset.from_list(records)
        assert len(ds) == 2
        assert ds[0]["prompt"] == "a"

    def test_from_list_mixed(self) -> None:
        records = [PromptRecord(prompt="a"), {"prompt": "b"}]
        ds = PromptDataset.from_list(records)
        assert len(ds) == 2

    def test_optional_fields_survive(self) -> None:
        ds = PromptDataset.from_list(
            [
                {
                    "prompt": "hi",
                    "negative_prompt": "lo",
                    "reference_text": "ref",
                    "reference_image": "img.png",
                    "metadata": {"src": "unit"},
                }
            ]
        )
        sample = ds[0]
        assert sample["negative_prompt"] == "lo"
        assert sample["reference_text"] == "ref"
        assert sample["reference_image"] == "img.png"
        assert sample["metadata"] == {"src": "unit"}

    def test_missing_optional_fields_default_to_none(self) -> None:
        ds = PromptDataset.from_list([{"prompt": "hi"}])
        sample = ds[0]
        assert sample["negative_prompt"] is None
        assert sample["reference_text"] is None
        assert sample["reference_image"] is None
        assert sample["metadata"] == {}

    def test_model_name_optional(self) -> None:
        # Passing no model_name works (task-agnostic dataset).
        ds = PromptDataset.from_list([{"prompt": "hi"}])
        assert ds.model_name is None

    def test_model_name_accepted(self) -> None:
        ds = PromptDataset.from_list([{"prompt": "hi"}], model_name="anything")
        assert ds.model_name == "anything"

    def test_max_samples_truncates(self) -> None:
        ds = PromptDataset.from_list([{"prompt": f"p{i}"} for i in range(10)], max_samples=3)
        assert len(ds) == 3
        assert [ds[i]["prompt"] for i in range(3)] == ["p0", "p1", "p2"]

    def test_max_samples_beyond_size_is_noop(self) -> None:
        ds = PromptDataset.from_list([{"prompt": "a"}, {"prompt": "b"}], max_samples=100)
        assert len(ds) == 2

    def test_dataset_name_stored(self) -> None:
        ds = PromptDataset.from_list([{"prompt": "hi"}], dataset_name="my_corpus")
        assert ds.dataset_name == "my_corpus"

    def test_data_split_stored(self) -> None:
        ds = PromptDataset.from_list([{"prompt": "hi"}], data_split="test")
        assert ds.data_split == "test"

    def test_iteration(self) -> None:
        records = [{"prompt": f"p{i}"} for i in range(4)]
        ds = PromptDataset.from_list(records)
        prompts = [sample["prompt"] for sample in ds]
        assert prompts == ["p0", "p1", "p2", "p3"]


class TestValidation:
    def test_empty_records_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one record"):
            PromptDataset.from_list([])

    def test_missing_prompt_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty string 'prompt'"):
            PromptDataset.from_list([{}])

    def test_empty_prompt_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty string 'prompt'"):
            PromptDataset.from_list([{"prompt": ""}])

    def test_non_string_prompt_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty string 'prompt'"):
            PromptDataset.from_list([{"prompt": 42}])

    def test_unknown_keys_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown keys"):
            PromptDataset.from_list([{"prompt": "hi", "surprise": 1}])

    def test_non_dict_record_rejected(self) -> None:
        with pytest.raises(TypeError, match="dict or PromptRecord"):
            PromptDataset.from_list(["just a string"])  # type: ignore[list-item]

    def test_non_dict_metadata_rejected(self) -> None:
        with pytest.raises(TypeError, match="metadata"):
            PromptDataset.from_list([{"prompt": "hi", "metadata": "nope"}])


# =============================================================================
# from_jsonl
# =============================================================================


class TestFromJsonl:
    def test_basic_load(self, tmp_path: Path) -> None:
        path = tmp_path / "prompts.jsonl"
        lines = [
            {"prompt": "a"},
            {"prompt": "b", "negative_prompt": "c"},
            {"prompt": "d", "metadata": {"k": 1}},
        ]
        path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")

        ds = PromptDataset.from_jsonl(path)
        assert len(ds) == 3
        assert ds[0]["prompt"] == "a"
        assert ds[1]["negative_prompt"] == "c"
        assert ds[2]["metadata"] == {"k": 1}

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "prompts.jsonl"
        path.write_text('{"prompt": "a"}\n\n{"prompt": "b"}\n', encoding="utf-8")
        ds = PromptDataset.from_jsonl(path)
        assert len(ds) == 2

    def test_invalid_json_raises_with_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"prompt": "a"}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match=r"line 2|:2:"):
            PromptDataset.from_jsonl(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            PromptDataset.from_jsonl(tmp_path / "does_not_exist.jsonl")

    def test_dataset_name_defaults_to_path(self, tmp_path: Path) -> None:
        path = tmp_path / "prompts.jsonl"
        path.write_text('{"prompt": "a"}\n', encoding="utf-8")
        ds = PromptDataset.from_jsonl(path)
        assert ds.dataset_name == str(path)

    def test_str_path_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "prompts.jsonl"
        path.write_text('{"prompt": "a"}\n', encoding="utf-8")
        ds = PromptDataset.from_jsonl(str(path))
        assert len(ds) == 1


# =============================================================================
# from_hf (network-free via pre-loaded fake dataset object)
# =============================================================================


class _FakeHFDataset:
    """Minimal stand-in for ``datasets.Dataset`` used to test from_hf
    without invoking ``load_dataset`` (and without the ``datasets``
    library exposing a network-free construction path in tests)."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.column_names = sorted({k for row in rows for k in row})

    def __iter__(self):
        return iter(self._rows)


class TestFromHf:
    def test_basic_mapping(self) -> None:
        fake = _FakeHFDataset(
            [
                {"question": "q1", "answer": "a1"},
                {"question": "q2", "answer": "a2"},
            ]
        )
        ds = PromptDataset.from_hf(
            fake,
            prompt_col="question",
            reference_text_col="answer",
        )
        assert len(ds) == 2
        assert ds[0]["prompt"] == "q1"
        assert ds[0]["reference_text"] == "a1"
        # Column not mapped -> field stays None
        assert ds[0]["negative_prompt"] is None

    def test_metadata_columns_collected(self) -> None:
        fake = _FakeHFDataset([{"p": "hi", "src": "wiki", "difficulty": "easy"}])
        ds = PromptDataset.from_hf(fake, prompt_col="p", metadata_cols=["src", "difficulty"])
        assert ds[0]["metadata"] == {"src": "wiki", "difficulty": "easy"}

    def test_missing_prompt_col_raises(self) -> None:
        fake = _FakeHFDataset([{"other": "value"}])
        with pytest.raises(KeyError, match="prompt"):
            PromptDataset.from_hf(fake, prompt_col="prompt")

    def test_missing_optional_col_raises(self) -> None:
        fake = _FakeHFDataset([{"prompt": "hi"}])
        with pytest.raises(KeyError, match="reference"):
            PromptDataset.from_hf(fake, prompt_col="prompt", reference_text_col="reference")


# =============================================================================
# BaseTaskDataset contract
# =============================================================================


class TestBaseContract:
    def test_len_matches_records(self) -> None:
        ds = PromptDataset.from_list([{"prompt": f"p{i}"} for i in range(7)])
        assert len(ds) == 7

    def test_getitem_returns_dict(self) -> None:
        ds = PromptDataset.from_list([{"prompt": "hi"}])
        sample = ds[0]
        assert isinstance(sample, dict)
        # Returned dict is a copy \u2014 mutating it does not affect the dataset.
        sample["prompt"] = "mutated"
        assert ds[0]["prompt"] == "hi"

    def test_label_col_is_empty_string(self) -> None:
        # PromptDataset is task-agnostic \u2014 no fixed label column.
        ds = PromptDataset.from_list([{"prompt": "hi"}])
        assert ds.label_col == ""
