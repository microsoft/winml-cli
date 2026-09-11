# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig
from winml.modelkit.eval.audio_classification_evaluator import (
    WinMLAudioClassificationEvaluator,
    _AudioModelAdapter,
)
from winml.modelkit.utils.eval_utils import DatasetValidationError


if TYPE_CHECKING:
    from pathlib import Path


def _model() -> MagicMock:
    model = MagicMock()
    model.config = SimpleNamespace(
        id2label={0: "real", 1: "fake"},
        label2id={"real": 0, "fake": 1},
    )
    model.io_config = {
        "input_names": ["input_values"],
        "input_shapes": [[1, 16000]],
        "output_names": ["logits"],
        "output_shapes": [[1, 2]],
    }
    return model


def _config(**dataset_kwargs: object) -> WinMLEvaluationConfig:
    return WinMLEvaluationConfig(
        model_id="test/audio-classifier",
        task="audio-classification",
        dataset=DatasetConfig(path="test/audio", shuffle=False, **dataset_kwargs),
    )


def test_dataset_config_roundtrip_preserves_audio_options() -> None:
    config = _config(
        label_mapping={"bonafide": 0, "spoof": 1},
        max_duration_seconds=2.0,
    )

    restored = WinMLEvaluationConfig.from_dict(config.to_dict())

    assert restored.dataset.label_mapping == {"bonafide": 0, "spoof": 1}
    assert restored.dataset.max_duration_seconds == 2.0


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_dataset_config_rejects_invalid_duration(value: float) -> None:
    with pytest.raises(ValueError, match="finite value greater than zero"):
        DatasetConfig(max_duration_seconds=value)


def test_decode_audio_supports_flac_bytes_and_path(tmp_path: Path) -> None:
    import soundfile as sf

    waveform = np.linspace(-0.5, 0.5, 800, dtype=np.float32)
    buffer = BytesIO()
    sf.write(buffer, waveform, 16000, format="FLAC")
    encoded = buffer.getvalue()
    path = tmp_path / "sample.flac"
    path.write_bytes(encoded)

    decoded_bytes, bytes_rate = WinMLAudioClassificationEvaluator._decode_audio(encoded)
    decoded_path, path_rate = WinMLAudioClassificationEvaluator._decode_audio(str(path))

    assert bytes_rate == path_rate == 16000
    np.testing.assert_allclose(decoded_bytes, decoded_path)
    assert decoded_bytes.shape == (800,)


def test_adapter_resamples_stereo_caps_windows_and_averages_logits() -> None:
    feature_extractor = MagicMock()
    feature_extractor.sampling_rate = 16000
    seen: list[np.ndarray] = []

    def encode(waveform: np.ndarray, **_kwargs: object) -> dict[str, torch.Tensor]:
        seen.append(np.asarray(waveform))
        padded = np.pad(waveform, (0, 16000 - len(waveform)))
        return {"input_values": torch.from_numpy(padded[None, :].astype(np.float32))}

    feature_extractor.side_effect = encode
    model = _model()
    model.side_effect = [
        {"logits": torch.tensor([[2.0, 0.0]])},
        {"logits": torch.tensor([[0.0, 4.0]])},
    ]
    config = _config(max_duration_seconds=2.0)
    stereo = np.stack([
        np.linspace(-1.0, 1.0, 44100 * 3, dtype=np.float32),
        np.linspace(1.0, -1.0, 44100 * 3, dtype=np.float32),
    ])

    with patch(
        "transformers.AutoFeatureExtractor.from_pretrained",
        return_value=feature_extractor,
    ):
        adapter = _AudioModelAdapter(config, model)
        logits = adapter({"array": stereo, "sampling_rate": 44100})

    np.testing.assert_allclose(logits, [1.0, 2.0])
    assert [len(window) for window in seen] == [16000, 16000]
    assert adapter.last_window_count == 2
    assert adapter.last_was_truncated is True


def test_prepare_data_preserves_metadata_and_balances_exact_classlabels() -> None:
    from datasets import Audio, ClassLabel, Dataset, Features, Value

    features = Features({
        "audio": {"bytes": Value("binary"), "path": Value("string")},
        "label": ClassLabel(names=["real", "fake", "other"]),
        "archive_member": Value("string"),
    })
    dataset = Dataset.from_dict(
        {
            "audio": [
                {"bytes": b"real", "path": "archive.zip::real.flac"},
                {"bytes": b"fake", "path": "archive.zip::fake.flac"},
                {"bytes": b"other", "path": "archive.zip::other.flac"},
            ],
            "label": [0, 1, 2],
            "archive_member": ["real.flac", "fake.flac", "other.flac"],
        },
        features=features,
    ).cast_column("audio", Audio(decode=False))
    evaluator = WinMLAudioClassificationEvaluator.__new__(WinMLAudioClassificationEvaluator)
    evaluator.config = _config(samples=2)
    evaluator.model = _model()
    evaluator.audio_column = "audio"
    evaluator.label_column = "label"
    evaluator.eligible_count = 0
    evaluator.selected_count = 0
    evaluator.dataset_label_to_model_id = {}
    evaluator.scalar_string_target = False
    evaluator.model_id2label, evaluator.model_label2id = evaluator._model_labels(evaluator.model)

    with patch("datasets.load_dataset", return_value=dataset):
        selected = evaluator.prepare_data()

    assert [sample.model_id for sample in selected] == [0, 1]
    assert [sample["archive_member"] for sample in selected] == ["real.flac", "fake.flac"]
    assert evaluator.eligible_count == 2
    assert evaluator.selected_count == 2


def test_string_targets_require_exact_or_explicit_mapping() -> None:
    from datasets import Dataset

    dataset = Dataset.from_dict({
        "audio": [np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32)],
        "label": ["bonafide", "spoof"],
    })
    evaluator = WinMLAudioClassificationEvaluator.__new__(WinMLAudioClassificationEvaluator)
    evaluator.audio_column = "audio"
    evaluator.label_column = "label"
    evaluator.model = _model()
    evaluator.model_id2label, evaluator.model_label2id = evaluator._model_labels(evaluator.model)
    evaluator.dataset_label_to_model_id = {}
    evaluator.scalar_string_target = False

    with pytest.raises(DatasetValidationError, match="no exact overlap"):
        evaluator._resolve_labels(dataset, DatasetConfig())

    evaluator._resolve_labels(
        dataset,
        DatasetConfig(label_mapping={"bonafide": 0, "spoof": 1}),
    )
    assert evaluator.dataset_label_to_model_id == {"bonafide": 0, "spoof": 1}


def test_compute_reports_metrics_and_bounded_accounting() -> None:
    evaluator = WinMLAudioClassificationEvaluator.__new__(WinMLAudioClassificationEvaluator)
    evaluator.config = _config(samples=2, max_duration_seconds=2.0)
    evaluator.model_id2label = {0: "real", 1: "fake"}
    evaluator.eligible_count = 2
    evaluator.selected_count = 2
    evaluator.audio_column = "audio"
    adapter = MagicMock(side_effect=[np.array([3.0, 1.0]), np.array([0.0, 2.0])])
    adapter.last_window_count = 1
    adapter.last_was_truncated = False
    evaluator.pipe = adapter

    # SimpleNamespace special methods are resolved on the type, so use mappings here.
    evaluator.data = [
        type("Selected", (), {"model_id": 0, "__getitem__": lambda self, key: [0.0]})(),
        type("Selected", (), {"model_id": 1, "__getitem__": lambda self, key: [1.0]})(),
    ]
    metrics = evaluator.compute()

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["requested_samples"] == metrics["processed_samples"] == 2
    assert metrics["inference_windows"] == 2
    assert metrics["rejected_samples"] == 0
    assert metrics["confusion_matrix"] == {"fake": {"fake": 1}, "real": {"real": 1}}


def test_compute_fails_closed_when_a_selected_class_cannot_decode() -> None:
    evaluator = WinMLAudioClassificationEvaluator.__new__(WinMLAudioClassificationEvaluator)
    evaluator.config = _config(samples=2)
    evaluator.model_id2label = {0: "real", 1: "fake"}
    evaluator.eligible_count = 2
    evaluator.selected_count = 2
    evaluator.audio_column = "audio"
    evaluator.data = [
        type("Selected", (), {"model_id": 0, "__getitem__": lambda self, key: [0.0]})(),
        type("Selected", (), {"model_id": 1, "__getitem__": lambda self, key: [1.0]})(),
    ]
    adapter = MagicMock(side_effect=[np.array([3.0, 1.0]), ValueError("bad audio")])
    adapter.last_window_count = 1
    adapter.last_was_truncated = False
    evaluator.pipe = adapter

    with pytest.raises(DatasetValidationError, match="fake=0/1 required"):
        evaluator.compute()
