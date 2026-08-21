# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch
from datasets import (
    Audio,
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    IterableDataset,
    Sequence,
    Value,
)

from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig
from winml.modelkit.eval.audio_classification_evaluator import (
    WinMLAudioClassificationEvaluator,
    _AudioModelAdapter,
)
from winml.modelkit.utils.eval_utils import DatasetValidationError


class _ASTFeatureExtractor:
    sampling_rate = 16_000

    def __call__(self, waveform, *, sampling_rate, return_tensors):
        assert waveform.shape == (16_000,)
        assert sampling_rate == self.sampling_rate
        assert return_tensors == "pt"
        return {"input_values": torch.ones((1, 1024, 128), dtype=torch.float32)}


class _CountingASTModel:
    io_config: ClassVar = {
        "input_names": ["input_values"],
        "input_shapes": [[1, 1024, 128]],
    }
    config = SimpleNamespace(
        id2label={0: "Speech", 1: "Music", 2: "Dog"},
        label2id={"Speech": 0, "Music": 1, "Dog": 2},
    )

    def __init__(self) -> None:
        self.forward_count = 0

    def __call__(self, **inputs):
        assert inputs["input_values"].shape == (1, 1024, 128)
        logits = (
            torch.tensor([[8.0, 7.0, -8.0]])
            if self.forward_count == 0
            else torch.tensor([[-8.0, 7.0, 8.0]])
        )
        self.forward_count += 1
        return {"logits": logits}


class _IdentityWaveformExtractor:
    sampling_rate = 16_000

    def __init__(self) -> None:
        self.last_waveform = None

    def __call__(self, waveform, *, sampling_rate, return_tensors, **_kwargs):
        self.last_waveform = waveform
        assert sampling_rate == self.sampling_rate
        assert return_tensors == "pt"
        return {"input_values": torch.as_tensor(waveform[None, :], dtype=torch.float32)}


class _BinaryModel:
    config = SimpleNamespace(
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    def __init__(self) -> None:
        self.forward_count = 0

    def __call__(self, **inputs):
        score = inputs["input_values"].mean()
        self.forward_count += 1
        return {"logits": torch.stack((-score, score)).reshape(1, 2)}


class _EncodingMustNotRunAudio(Audio):
    def encode_example(self, value):
        raise AssertionError("bounded selection must not re-encode raw audio")


class _RawAudioIterableDataset(IterableDataset):
    def __init__(self, rows, features) -> None:
        self._rows = rows
        self._raw_features = features

    @property
    def features(self):
        return self._raw_features

    @property
    def column_names(self):
        return list(self._raw_features)

    def take(self, count):
        return iter(self._rows[:count])


def test_streaming_raw_audio_is_bounded_without_feature_reencoding() -> None:
    def wav_bytes(value: float) -> bytes:
        buffer = BytesIO()
        sf.write(buffer, np.full(16_000, value, dtype=np.float32), 16_000, format="WAV")
        return buffer.getvalue()

    rows = [
        {"audio": {"bytes": wav_bytes(-0.5), "path": None}, "labels": ["negative"]},
        {"audio": {"bytes": wav_bytes(0.5), "path": None}, "labels": ["positive"]},
    ]
    dataset = _RawAudioIterableDataset(
        rows,
        Features(
            {
                "audio": _EncodingMustNotRunAudio(decode=False),
                "labels": Sequence(Value("string")),
            }
        ),
    )
    config = WinMLEvaluationConfig(
        model_id="example/streaming-audio",
        task="audio-classification",
        runtime="pytorch",
        dataset=DatasetConfig(
            path="example/streaming-audio",
            split="test",
            streaming=True,
            samples=2,
            shuffle=False,
            columns_mapping={"label_column": "labels"},
        ),
    )
    model = _BinaryModel()
    extractor = _IdentityWaveformExtractor()
    decoded_payloads = []
    decode_audio = _AudioModelAdapter._decode_audio

    def track_decode(audio):
        decoded_payloads.append(audio)
        return decode_audio(audio)

    with (
        patch("datasets.load_dataset", return_value=dataset),
        patch(
            "transformers.AutoFeatureExtractor.from_pretrained",
            return_value=extractor,
        ),
        patch.object(_AudioModelAdapter, "_decode_audio", side_effect=track_decode),
    ):
        metrics = WinMLAudioClassificationEvaluator(config, model).compute()

    assert [payload["bytes"] for payload in decoded_payloads] == [
        row["audio"]["bytes"] for row in rows
    ]
    assert model.forward_count == 2
    assert metrics["requested_samples"] == 2
    assert metrics["selected_samples"] == 2
    assert metrics["processed_samples"] == 2
    assert metrics["rejected_samples"] == 0
    assert np.isfinite(metrics["sample_average_precision"])
    assert np.isfinite(metrics["micro_average_precision"])


def test_rank_three_multilabel_uses_one_forward_per_row_and_finite_ap() -> None:
    features = Features(
        {
            "audio": {
                "array": Sequence(Value("float32")),
                "sampling_rate": Value("int32"),
            },
            "labels": Sequence(Value("string")),
            "human_labels": Sequence(Value("string")),
        }
    )
    dataset = Dataset.from_list(
        [
            {
                "audio": {"array": [0.0] * 16_000, "sampling_rate": 16_000},
                "labels": ["/m/speech", "/m/music"],
                "human_labels": ["Speech", "Music"],
            },
            {
                "audio": {"array": [0.0] * 16_000, "sampling_rate": 16_000},
                "labels": ["/m/music", "/m/dog"],
                "human_labels": ["Music", "Dog"],
            },
        ],
        features=features,
    )
    config = WinMLEvaluationConfig(
        model_id="example/ast",
        task="audio-classification",
        dataset=DatasetConfig(
            path="example/audioset",
            split="test",
            samples=2,
            shuffle=False,
            columns_mapping={"label_column": "labels"},
        ),
    )
    model = _CountingASTModel()

    with (
        patch("datasets.load_dataset", return_value=dataset),
        patch(
            "transformers.AutoFeatureExtractor.from_pretrained",
            return_value=_ASTFeatureExtractor(),
        ),
    ):
        metrics = WinMLAudioClassificationEvaluator(config, model).compute()

    assert model.forward_count == 2
    assert metrics["processed_samples"] == 2
    assert np.isfinite(metrics["sample_average_precision"])
    assert np.isfinite(metrics["micro_average_precision"])
    assert 0.0 <= metrics["sample_average_precision"] <= 1.0
    assert 0.0 <= metrics["micro_average_precision"] <= 1.0
    assert metrics["requested_samples"] == 2
    assert metrics["selected_samples"] == 2
    assert metrics["rejected_samples"] == 0


def test_scalar_binary_classlabel_preserves_argmax_accuracy_and_f1() -> None:
    features = Features(
        {
            "audio": {
                "array": Sequence(Value("float32")),
                "sampling_rate": Value("int32"),
            },
            "label": ClassLabel(names=["negative", "positive"]),
        }
    )
    dataset = Dataset.from_list(
        [
            {"audio": {"array": [-1.0] * 8, "sampling_rate": 16_000}, "label": 0},
            {"audio": {"array": [1.0] * 8, "sampling_rate": 16_000}, "label": 1},
        ],
        features=features,
    )
    config = WinMLEvaluationConfig(
        model_id="example/binary-audio",
        task="audio-classification",
        runtime="pytorch",
        dataset=DatasetConfig(path="example/binary", split="test", samples=2, shuffle=False),
    )
    model = _BinaryModel()

    with (
        patch("datasets.load_dataset", return_value=dataset),
        patch(
            "transformers.AutoFeatureExtractor.from_pretrained",
            return_value=_IdentityWaveformExtractor(),
        ),
    ):
        metrics = WinMLAudioClassificationEvaluator(config, model).compute()

    assert model.forward_count == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["represented_classes"] == 2
    assert metrics["class_coverage"] == 1.0


def test_scalar_string_labels_use_explicit_checkpoint_id_mapping() -> None:
    features = Features(
        {
            "audio": {
                "array": Sequence(Value("float32")),
                "sampling_rate": Value("int32"),
            },
            "gender": Value("string"),
        }
    )
    dataset = Dataset.from_list(
        [
            {"audio": {"array": [-1.0] * 8, "sampling_rate": 16_000}, "gender": "female"},
            {"audio": {"array": [1.0] * 8, "sampling_rate": 16_000}, "gender": "male"},
        ],
        features=features,
    )
    config = WinMLEvaluationConfig(
        model_id="example/gender-audio",
        task="audio-classification",
        runtime="pytorch",
        dataset=DatasetConfig(
            path="example/gender-audio",
            split="test",
            samples=2,
            shuffle=False,
            columns_mapping={"label_column": "gender"},
            label_mapping={"female": 0, "male": 1},
        ),
    )
    model = _BinaryModel()

    with (
        patch("datasets.load_dataset", return_value=dataset),
        patch(
            "transformers.AutoFeatureExtractor.from_pretrained",
            return_value=_IdentityWaveformExtractor(),
        ),
    ):
        metrics = WinMLAudioClassificationEvaluator(config, model).compute()

    assert model.forward_count == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


@pytest.mark.parametrize(
    ("label_mapping", "message"),
    [
        (None, "non-empty explicit label mapping"),
        ({}, "non-empty explicit label mapping"),
        ({"female": "0", "male": 1}, "destinations must be checkpoint IDs"),
        ({"female": 0, "male": 0}, "duplicate checkpoint ID destinations"),
        ({"female": 2, "male": 3}, "absent from model.config.id2label"),
        ({"female": 0}, "must cover every checkpoint ID"),
    ],
    ids=[
        "missing",
        "empty",
        "non-integer-destination",
        "duplicate-destination",
        "unknown-destinations",
        "incomplete",
    ],
)
def test_scalar_string_label_mapping_rejects_malformed_mappings(
    label_mapping,
    message,
) -> None:
    features = Features(
        {
            "audio": {
                "array": Sequence(Value("float32")),
                "sampling_rate": Value("int32"),
            },
            "gender": Value("string"),
        }
    )
    dataset = Dataset.from_list(
        [{"audio": {"array": [0.0], "sampling_rate": 16_000}, "gender": "female"}],
        features=features,
    )
    config = WinMLEvaluationConfig(
        model_id="example/gender-audio",
        task="audio-classification",
        dataset=DatasetConfig(
            path="example/gender-audio",
            samples=1,
            shuffle=False,
            columns_mapping={"label_column": "gender"},
            label_mapping=label_mapping,
        ),
    )

    with (
        patch("datasets.load_dataset", return_value=dataset),
        pytest.raises(DatasetValidationError, match=message),
    ):
        WinMLAudioClassificationEvaluator(config, _BinaryModel())


def test_scalar_string_label_mapping_rejects_unmapped_observed_value() -> None:
    features = Features(
        {
            "audio": {
                "array": Sequence(Value("float32")),
                "sampling_rate": Value("int32"),
            },
            "gender": Value("string"),
        }
    )
    dataset = Dataset.from_list(
        [
            {"audio": {"array": [-1.0], "sampling_rate": 16_000}, "gender": "female"},
            {"audio": {"array": [1.0], "sampling_rate": 16_000}, "gender": "unknown"},
        ],
        features=features,
    )
    config = WinMLEvaluationConfig(
        model_id="example/gender-audio",
        task="audio-classification",
        dataset=DatasetConfig(
            path="example/gender-audio",
            samples=2,
            shuffle=False,
            columns_mapping={"label_column": "gender"},
            label_mapping={"female": 0, "male": 1},
        ),
    )
    model = _BinaryModel()

    with (
        patch("datasets.load_dataset", return_value=dataset),
        patch(
            "transformers.AutoFeatureExtractor.from_pretrained",
            return_value=_IdentityWaveformExtractor(),
        ),
        pytest.raises(DatasetValidationError, match=r"'unknown'.*absent"),
    ):
        WinMLAudioClassificationEvaluator(config, model).compute()

    assert model.forward_count == 0


def test_native_hf_model_without_io_config_uses_shared_adapter() -> None:
    config = WinMLEvaluationConfig(
        model_id="example/native-audio",
        task="audio-classification",
        runtime="pytorch",
    )
    model = _BinaryModel()
    extractor = _IdentityWaveformExtractor()

    with patch(
        "transformers.AutoFeatureExtractor.from_pretrained",
        return_value=extractor,
    ):
        adapter = _AudioModelAdapter(config, model)
        logits = adapter.predict_logits(
            {"array": np.ones(8, dtype=np.float32), "sampling_rate": 16_000}
        )

    assert model.forward_count == 1
    assert logits.shape == (2,)


def test_encoded_stereo_audio_is_downmixed_and_resampled(tmp_path) -> None:
    audio_path = tmp_path / "stereo.wav"
    frames = np.column_stack(
        (
            np.full(8_000, 0.25, dtype=np.float32),
            np.full(8_000, 0.75, dtype=np.float32),
        )
    )
    sf.write(audio_path, frames, 8_000)
    config = WinMLEvaluationConfig(model_id="example/audio", task="audio-classification")
    model = _BinaryModel()
    extractor = _IdentityWaveformExtractor()

    with patch(
        "transformers.AutoFeatureExtractor.from_pretrained",
        return_value=extractor,
    ):
        adapter = _AudioModelAdapter(config, model)
        adapter.predict_logits({"bytes": None, "path": str(audio_path)})

    assert extractor.last_waveform.shape == (16_000,)
    np.testing.assert_allclose(extractor.last_waveform[100:-100], 0.5, atol=2e-3)


def test_saved_dataset_dict_selects_requested_split() -> None:
    features = Features(
        {
            "audio": {
                "array": Sequence(Value("float32")),
                "sampling_rate": Value("int32"),
            },
            "label": ClassLabel(names=["negative", "positive"]),
        }
    )
    train = Dataset.from_list(
        [{"audio": {"array": [-1.0], "sampling_rate": 16_000}, "label": 0}],
        features=features,
    )
    test = Dataset.from_list(
        [{"audio": {"array": [1.0], "sampling_rate": 16_000}, "label": 1}],
        features=features,
    )
    config = WinMLEvaluationConfig(
        model_id="example/audio",
        task="audio-classification",
        dataset=DatasetConfig(path="saved-dataset", split="test", samples=1, shuffle=False),
    )

    with (
        patch("pathlib.Path.is_dir", return_value=True),
        patch("datasets.load_from_disk", return_value=DatasetDict(train=train, test=test)),
        patch(
            "transformers.AutoFeatureExtractor.from_pretrained",
            return_value=_IdentityWaveformExtractor(),
        ),
    ):
        evaluator = WinMLAudioClassificationEvaluator(config, _BinaryModel())

    assert evaluator.data[0]["label"] == 1


@pytest.mark.parametrize(
    "label_feature",
    [Value("int64"), Sequence(Sequence(Value("string")))],
    ids=["ambiguous-scalar-integer", "malformed-nested-sequence"],
)
def test_ambiguous_or_malformed_target_schema_fails_closed(label_feature) -> None:
    features = Features(
        {
            "audio": {
                "array": Sequence(Value("float32")),
                "sampling_rate": Value("int32"),
            },
            "label": label_feature,
        }
    )
    label = 0 if isinstance(label_feature, Value) else [["negative"]]
    dataset = Dataset.from_list(
        [{"audio": {"array": [0.0], "sampling_rate": 16_000}, "label": label}],
        features=features,
    )
    config = WinMLEvaluationConfig(
        model_id="example/audio",
        task="audio-classification",
        dataset=DatasetConfig(path="example/audio", samples=1, shuffle=False),
    )

    with (
        patch("datasets.load_dataset", return_value=dataset),
        pytest.raises(DatasetValidationError, match="must be ClassLabel or a sequence"),
    ):
        WinMLAudioClassificationEvaluator(config, _BinaryModel())


def test_unmapped_sequence_target_fails_as_ambiguous() -> None:
    features = Features(
        {
            "audio": {
                "array": Sequence(Value("float32")),
                "sampling_rate": Value("int32"),
            },
            "labels": Sequence(Value("string")),
        }
    )
    dataset = Dataset.from_list(
        [
            {
                "audio": {"array": [0.0], "sampling_rate": 16_000},
                "labels": ["/m/not-a-model-label"],
            }
        ],
        features=features,
    )
    config = WinMLEvaluationConfig(
        model_id="example/audio",
        task="audio-classification",
        dataset=DatasetConfig(
            path="example/audio",
            samples=1,
            shuffle=False,
            columns_mapping={"label_column": "labels"},
        ),
    )

    with (
        patch("datasets.load_dataset", return_value=dataset),
        patch(
            "transformers.AutoFeatureExtractor.from_pretrained",
            return_value=_IdentityWaveformExtractor(),
        ),
        pytest.raises(DatasetValidationError, match="no exact model-label match"),
    ):
        WinMLAudioClassificationEvaluator(config, _BinaryModel()).compute()


def test_registry_schema_and_no_universal_default() -> None:
    from winml.modelkit.eval.evaluate import _DEFAULT_DATASETS, get_evaluator_class
    from winml.modelkit.utils.eval_utils import TASK_SCHEMAS

    assert "audio-classification" in TASK_SCHEMAS
    assert "audio-classification" not in _DEFAULT_DATASETS
    label_schema = next(
        item for item in TASK_SCHEMAS["audio-classification"].columns if item.name == "label_column"
    )
    assert "explicitly mapped scalar string" in label_schema.description
    assert get_evaluator_class(WinMLEvaluationConfig(task="audio-classification")) is (
        WinMLAudioClassificationEvaluator
    )
