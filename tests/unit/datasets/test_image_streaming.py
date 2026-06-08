# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression tests for ImageDataset streaming behavior.

Pins the default-dataset-streaming win (no full bulk download for tiny
calibration sets) and the documented fallback when no max_samples is set.
"""

from __future__ import annotations

import pytest

from winml.modelkit.datasets.image import ImageDataset


def _make_uninitialized(
    *,
    dataset_name: str | None,
    max_samples: int | None,
    config: dict | None = None,
) -> ImageDataset:
    """Build an ImageDataset bypassing __init__ (which triggers HF downloads)."""
    ds = ImageDataset.__new__(ImageDataset)
    ds._model_name = "x"
    ds._dataset_name = dataset_name
    ds._max_samples = max_samples
    ds._data_split = None
    ds._config = config if config is not None else {}
    ds._dataset = None
    ds._metadata = {}
    return ds


class _FakeStreamingDataset:
    """Minimal IterableDataset stand-in capturing shuffle args."""

    def __init__(self) -> None:
        self.shuffle_calls: list[dict] = []
        self.take_calls: list[int] = []
        self.features = None

    def shuffle(self, seed, buffer_size):
        self.shuffle_calls.append({"seed": seed, "buffer_size": buffer_size})
        return self

    def take(self, n):
        self.take_calls.append(n)
        return self

    def __iter__(self):
        return iter([])


class TestDefaultDatasetStreaming:
    def test_default_dataset_enables_streaming(self) -> None:
        ds = _make_uninitialized(dataset_name=None, max_samples=10)
        ds._get_default_dataset()
        assert ds._config.get("streaming") is True
        assert ds._dataset_name == "timm/mini-imagenet"

    def test_custom_dataset_does_not_force_streaming(self) -> None:
        ds = _make_uninitialized(dataset_name="cifar10", max_samples=10)
        ds._get_default_dataset()  # no-op when dataset_name set
        assert ds._config.get("streaming") in (None, False)

    def test_streaming_without_max_samples_degrades_to_bulk(self, monkeypatch) -> None:
        """Documented fallback: streaming=True + max_samples=None => bulk load."""
        ds = _make_uninitialized(
            dataset_name="cifar10",
            max_samples=None,
            config={"streaming": True},
        )

        captured: dict = {}

        class _FakeBulk:
            def __len__(self) -> int:
                return 0

            def shuffle(self, *a, **kw):
                return self

            def select(self, *a, **kw):
                return self

        def fake_load(name, split, streaming, **kwargs):
            captured["streaming"] = streaming
            return _FakeBulk()

        monkeypatch.setattr(
            "winml.modelkit.datasets.image.load_dataset", fake_load
        )
        ds._load_and_sample()
        assert captured["streaming"] is False

    @pytest.mark.parametrize("max_samples", [10, 100, 5000])
    def test_streaming_buffer_is_1000_for_class_diversity(
        self, monkeypatch, max_samples
    ) -> None:
        """Pin the 1000-item reservoir for class diversity on class-ordered streams."""
        ds = _make_uninitialized(
            dataset_name="cifar10",
            max_samples=max_samples,
            config={"streaming": True, "shuffle": True},
        )

        fake = _FakeStreamingDataset()

        def fake_load(name, split, streaming, **kwargs):
            assert streaming is True
            return fake

        monkeypatch.setattr(
            "winml.modelkit.datasets.image.load_dataset", fake_load
        )
        # Bypass ArrowDataset.from_list — fake has no real records.
        monkeypatch.setattr(
            "datasets.Dataset.from_list",
            lambda records, features=None: records,
        )

        ds._load_and_sample()
        assert len(fake.shuffle_calls) == 1
        assert fake.shuffle_calls[0]["buffer_size"] == 1000
        assert fake.take_calls == [max_samples]
