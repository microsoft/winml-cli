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


# =============================================================================
# Derived-dataset operations: filter / sample / _derive extension hook
# =============================================================================


class TestFilter:
    def test_keeps_matching_records(self) -> None:
        ds = PromptDataset.from_list(
            [
                {"prompt": "short"},
                {"prompt": "a much longer prompt here"},
                {"prompt": "med"},
            ]
        )
        filtered = ds.filter(lambda r: len(r["prompt"]) < 10)
        assert len(filtered) == 2
        assert [r["prompt"] for r in filtered] == ["short", "med"]

    def test_returns_new_instance_source_untouched(self) -> None:
        ds = PromptDataset.from_list([{"prompt": "keep"}, {"prompt": "drop"}])
        filtered = ds.filter(lambda r: r["prompt"] == "keep")
        assert filtered is not ds
        assert len(ds) == 2  # source untouched
        assert len(filtered) == 1

    def test_filter_returns_prompt_dataset(self) -> None:
        ds = PromptDataset.from_list([{"prompt": "keep"}, {"prompt": "drop"}])
        filtered = ds.filter(lambda r: r["prompt"] == "keep")
        assert isinstance(filtered, PromptDataset)

    def test_metadata_inherited_by_default(self) -> None:
        ds = PromptDataset.from_list(
            [{"prompt": "a"}, {"prompt": "b"}],
            dataset_name="my_corpus",
            data_split="test",
        )
        filtered = ds.filter(lambda r: True)
        assert filtered.dataset_name == "my_corpus"
        assert filtered.data_split == "test"

    def test_metadata_override_via_kwargs(self) -> None:
        ds = PromptDataset.from_list([{"prompt": "a"}], dataset_name="orig")
        filtered = ds.filter(lambda r: True, dataset_name="filtered_view")
        assert filtered.dataset_name == "filtered_view"

    def test_predicate_receives_dict_copy_semantics(self) -> None:
        # The predicate sees the record; mutating what it sees must not
        # affect the source dataset.
        ds = PromptDataset.from_list([{"prompt": "hi", "metadata": {"k": 1}}])

        def mutating(record: dict) -> bool:
            record["prompt"] = "MUTATED"
            return True

        _ = ds.filter(mutating)
        assert ds[0]["prompt"] == "hi"

    def test_rejects_all_raises(self) -> None:
        ds = PromptDataset.from_list([{"prompt": "a"}, {"prompt": "b"}])
        # PromptDataset requires >= 1 record; filtering everything out
        # surfaces the base validation error.
        with pytest.raises(ValueError, match="at least one"):
            ds.filter(lambda r: False)

    def test_chainable_with_sample(self) -> None:
        ds = PromptDataset.from_list([{"prompt": f"p{i}"} for i in range(10)])
        result = ds.filter(lambda r: int(r["prompt"][1:]) % 2 == 0).sample(2, seed=1)
        assert len(result) == 2
        for record in result:
            assert int(record["prompt"][1:]) % 2 == 0


class TestSample:
    def test_size_matches_n(self) -> None:
        ds = PromptDataset.from_list([{"prompt": f"p{i}"} for i in range(10)])
        sampled = ds.sample(3, seed=0)
        assert len(sampled) == 3

    def test_seed_reproducible(self) -> None:
        ds = PromptDataset.from_list([{"prompt": f"p{i}"} for i in range(20)])
        a = list(ds.sample(5, seed=42))
        b = list(ds.sample(5, seed=42))
        assert a == b

    def test_different_seeds_differ(self) -> None:
        ds = PromptDataset.from_list([{"prompt": f"p{i}"} for i in range(50)])
        a = list(ds.sample(10, seed=1))
        b = list(ds.sample(10, seed=2))
        # Astronomically unlikely to match with 50-choose-10.
        assert a != b

    def test_full_size_returns_permutation(self) -> None:
        ds = PromptDataset.from_list([{"prompt": f"p{i}"} for i in range(5)])
        sampled = ds.sample(5, seed=0)
        assert len(sampled) == 5
        assert {r["prompt"] for r in sampled} == {"p0", "p1", "p2", "p3", "p4"}

    def test_returns_new_instance_source_untouched(self) -> None:
        ds = PromptDataset.from_list([{"prompt": f"p{i}"} for i in range(4)])
        _ = ds.sample(2, seed=0)
        assert len(ds) == 4

    def test_oversample_rejected(self) -> None:
        ds = PromptDataset.from_list([{"prompt": "a"}, {"prompt": "b"}])
        with pytest.raises(ValueError, match="Cannot sample 5"):
            ds.sample(5, seed=0)

    def test_zero_or_negative_n_rejected(self) -> None:
        ds = PromptDataset.from_list([{"prompt": "a"}])
        with pytest.raises(ValueError, match=">= 1"):
            ds.sample(0)
        with pytest.raises(ValueError, match=">= 1"):
            ds.sample(-1)

    def test_metadata_inherited_by_default(self) -> None:
        ds = PromptDataset.from_list(
            [{"prompt": f"p{i}"} for i in range(5)],
            dataset_name="orig",
            data_split="val",
        )
        sampled = ds.sample(2, seed=0)
        assert sampled.dataset_name == "orig"
        assert sampled.data_split == "val"


class TestDeriveExtensionHook:
    """``_derive`` is the extension point for subclasses adding new derived-
    dataset operations (dedup, group_by, etc.)."""

    def test_subclass_receives_own_type(self) -> None:
        # Contract: derived datasets are instances of ``type(self)``.
        class MyPromptDataset(PromptDataset):
            pass

        ds = MyPromptDataset.from_list([{"prompt": "a"}, {"prompt": "b"}])
        filtered = ds.filter(lambda r: True)
        sampled = ds.sample(1, seed=0)
        assert isinstance(filtered, MyPromptDataset)
        assert isinstance(sampled, MyPromptDataset)

    def test_subclass_can_override_derive_for_extra_metadata(self) -> None:
        class TaggedPromptDataset(PromptDataset):
            def _derive(self, records, **kwargs):
                # Contract: subclasses can inject extra fields safely.
                kwargs.setdefault("dataset_name", f"tagged:{self._dataset_name}")
                return super()._derive(records, **kwargs)

        ds = TaggedPromptDataset.from_list([{"prompt": "a"}, {"prompt": "b"}], dataset_name="base")
        filtered = ds.filter(lambda r: True)
        assert filtered.dataset_name == "tagged:base"
