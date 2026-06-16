# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Universal calibration fallback to RandomDataset.

A task-specific calibration dataset can fail to serve a model in two ways:
1. It cannot be built (e.g. an audio backbone stays ``feature-extraction`` and routes
   to ``TextDataset``, which needs a tokenizer the model lacks).
2. It builds but produces none of the ONNX model's input tensors (text ``input_ids``
   for an audio model that wants ``input_values``).

In both cases — when an ONNX ``model_path`` is known — calibration falls back to
``RandomDataset``, which reads the real input specs straight from the model. This is
modality-agnostic, so it also covers image/video and any future modality.
"""

from __future__ import annotations

import pytest

import winml.modelkit.datasets as ds_mod
from winml.modelkit.datasets import _dataset_produces_any_input, universal_calib_dataset


class _FakeDataset:
    """Single-sample dataset stand-in exposing a fixed field set."""

    def __init__(self, sample: dict) -> None:
        self._sample = sample

    def __getitem__(self, idx: int) -> dict:
        return self._sample

    def __len__(self) -> int:
        return 1


def test_dataset_produces_any_input_detects_overlap() -> None:
    ds = _FakeDataset({"input_ids": 1, "attention_mask": 1})
    assert _dataset_produces_any_input(ds, {"input_ids"}) is True


def test_dataset_produces_any_input_detects_mismatch() -> None:
    ds = _FakeDataset({"input_ids": 1, "attention_mask": 1})
    assert _dataset_produces_any_input(ds, {"input_values"}) is False


def test_dataset_produces_any_input_empty_inputs_is_noop() -> None:
    """An empty model-input set means we cannot judge -> no fallback."""
    ds = _FakeDataset({"input_ids": 1})
    assert _dataset_produces_any_input(ds, set()) is True


def test_modality_mismatch_falls_back_to_random(monkeypatch: pytest.MonkeyPatch) -> None:
    """A text dataset (input_ids) for a model whose ONNX wants input_values has no input
    overlap, so calibration falls back to RandomDataset."""
    text_like = _FakeDataset({"input_ids": [1, 2], "attention_mask": [1, 1]})
    monkeypatch.setattr(
        ds_mod, "_resolve_dataset_class", lambda task: ((lambda **kw: text_like), task)
    )
    sentinel = object()
    monkeypatch.setattr(ds_mod, "RandomDataset", lambda **kw: sentinel)

    result = universal_calib_dataset(
        "model",
        "feature-extraction",
        model_path="model.onnx",
        io_config={"input_values": {"shape": [1, 16000], "dtype": "float32"}},
    )
    assert result is sentinel


def test_construction_failure_falls_back_to_random(monkeypatch: pytest.MonkeyPatch) -> None:
    """A task dataset that fails to build (e.g. audio model has no tokenizer) falls back
    to RandomDataset when an ONNX model_path is available."""

    def _boom(**kw: object) -> object:
        raise RuntimeError("no tokenizer for audio model")

    monkeypatch.setattr(ds_mod, "_resolve_dataset_class", lambda task: (_boom, task))
    sentinel = object()
    monkeypatch.setattr(ds_mod, "RandomDataset", lambda **kw: sentinel)

    result = universal_calib_dataset("model", "feature-extraction", model_path="model.onnx")
    assert result is sentinel


def test_construction_failure_without_model_path_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an ONNX model_path there is nothing to derive random inputs from, so the
    construction error surfaces as RuntimeError rather than a silent fallback."""

    def _boom(**kw: object) -> object:
        raise RuntimeError("dataset load failed")

    monkeypatch.setattr(ds_mod, "_resolve_dataset_class", lambda task: (_boom, task))
    with pytest.raises(RuntimeError, match="Failed to create"):
        universal_calib_dataset("model", "feature-extraction")
