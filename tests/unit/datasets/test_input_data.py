# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for winml.modelkit.datasets.input_data.

Covers the shared ``.npz`` loader (also exercised via ``winml perf``) and the
single-sample :class:`InputDataDataset` used by ``winml eval --mode compare``.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import click
import numpy as np
import pytest
import torch

from winml.modelkit.datasets.input_data import InputDataDataset, load_input_data


class TestLoadInputData:
    _IO: ClassVar[dict] = {
        "input_names": ["pixel_values"],
        "input_shapes": [[None, 3, 8, 8]],
        "input_types": ["float32"],
    }

    def _write_npz(self, tmp_path, **arrays):
        path = tmp_path / "inputs.npz"
        np.savez(path, **arrays)
        return path

    def test_loads_matching_npz(self, tmp_path) -> None:
        path = self._write_npz(tmp_path, pixel_values=np.zeros((2, 3, 8, 8), dtype=np.float32))
        inputs = load_input_data(path, self._IO)
        assert list(inputs) == ["pixel_values"]
        assert inputs["pixel_values"].shape == (2, 3, 8, 8)

    def test_key_mismatch_errors(self, tmp_path) -> None:
        path = self._write_npz(tmp_path, wrong=np.zeros((1, 3, 8, 8), dtype=np.float32))
        with pytest.raises(click.UsageError, match="do not match"):
            load_input_data(path, self._IO)

    def test_dtype_cast_with_warning(self, tmp_path, caplog) -> None:
        io = {"input_names": ["input_ids"], "input_shapes": [[None, 8]], "input_types": ["int32"]}
        path = self._write_npz(tmp_path, input_ids=np.zeros((1, 8), dtype=np.int64))
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.datasets.input_data"):
            inputs = load_input_data(path, io)
        assert inputs["input_ids"].dtype == np.int32
        assert "casting" in caplog.text.lower()

    def test_npy_rejected(self, tmp_path) -> None:
        path = tmp_path / "inputs.npy"
        np.save(path, np.zeros((1, 3, 8, 8), dtype=np.float32))
        with pytest.raises(click.UsageError, match=r"does not support \.npy"):
            load_input_data(path, self._IO)


class TestInputDataDataset:
    _IO: ClassVar[dict] = {
        "input_names": ["x"],
        "input_shapes": [[None, 4]],
        "input_types": ["float32"],
    }

    def _write_npz(self, tmp_path, **arrays):
        path = tmp_path / "inputs.npz"
        np.savez(path, **arrays)
        return path

    def test_single_sample_as_torch_tensors(self, tmp_path) -> None:
        path = self._write_npz(tmp_path, x=np.ones((2, 4), dtype=np.float32))
        ds = InputDataDataset(path, self._IO)

        assert len(ds) == 1
        sample = ds[0]
        assert set(sample) == {"x"}
        assert isinstance(sample["x"], torch.Tensor)
        assert sample["x"].shape == (2, 4)
        assert sample["x"].dtype == torch.float32

    def test_index_out_of_range(self, tmp_path) -> None:
        path = self._write_npz(tmp_path, x=np.ones((1, 4), dtype=np.float32))
        ds = InputDataDataset(path, self._IO)
        with pytest.raises(IndexError):
            _ = ds[1]

    def test_validates_keys_against_io_config(self, tmp_path) -> None:
        path = self._write_npz(tmp_path, wrong=np.ones((1, 4), dtype=np.float32))
        with pytest.raises(click.UsageError, match="do not match"):
            InputDataDataset(path, self._IO)
