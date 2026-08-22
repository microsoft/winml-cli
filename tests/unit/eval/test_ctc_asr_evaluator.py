# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from winml.modelkit.eval.ctc_asr_evaluator import (
    WinMLCTCASREvaluator,
    _configure_processor_language,
    _corpus_error_rate,
    _decode_audio,
    _is_ctc_config,
    _normalize_transcript,
    _RejectedSampleError,
    _resample_audio,
)
from winml.modelkit.utils.eval_utils import TASK_SCHEMAS, DatasetValidationError


class _Tokenizer:
    pad_token_id = 0

    def __init__(self, target_lang: str | None = None, vocab_size: int = 3) -> None:
        self.target_lang = target_lang
        self._vocab = {str(index): index for index in range(vocab_size)}

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def set_target_lang(self, target_lang: str) -> None:
        self.target_lang = target_lang


class _Processor:
    def __init__(self, *, target_lang: str | None = None, vocab_size: int = 3) -> None:
        self.tokenizer = _Tokenizer(target_lang, vocab_size)
        self.feature_extractor = SimpleNamespace(sampling_rate=16_000)
        self.decoded_ids: list[int] = []

    def __call__(self, waveform, **_kwargs):
        return {"input_values": np.asarray(waveform, dtype=np.float32)[None, :]}

    def batch_decode(self, sequences):
        self.decoded_ids = list(sequences[0])
        collapsed: list[int] = []
        previous = None
        for token_id in self.decoded_ids:
            if token_id != 0 and token_id != previous:
                collapsed.append(token_id)
            previous = token_id
        return [" ".join(str(token_id) for token_id in collapsed)]


def _wav_bytes(samples: np.ndarray, sampling_rate: int = 16_000) -> bytes:
    buffer = BytesIO()
    sf.write(buffer, samples, sampling_rate, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def _evaluator(*, input_shape: list[object], processor: _Processor | None = None):
    evaluator = WinMLCTCASREvaluator.__new__(WinMLCTCASREvaluator)
    evaluator.processor = processor or _Processor()
    evaluator.sampling_rate = 16_000
    evaluator.blank_token_id = 0
    evaluator.tokenizer_vocab_size = 3
    evaluator.target_lang = evaluator.processor.tokenizer.target_lang
    evaluator._audio_column = "audio"
    evaluator._transcription_column = "transcription"
    evaluator.model = MagicMock()
    evaluator.model.io_config = {
        "input_names": ["input_values"],
        "input_shapes": [input_shape],
    }
    return evaluator


def test_asr_registry_schema_and_generic_model_route() -> None:
    from winml.modelkit.eval import WinMLEvaluationConfig, get_evaluator_class
    from winml.modelkit.inference import TASK_REGISTRY
    from winml.modelkit.models.winml import get_winml_class
    from winml.modelkit.models.winml.base import WinMLModelForGenericTask

    config = WinMLEvaluationConfig(task="automatic-speech-recognition")
    assert get_evaluator_class(config) is WinMLCTCASREvaluator
    assert "automatic-speech-recognition" in TASK_SCHEMAS
    assert (
        get_winml_class("wav2vec2", "automatic-speech-recognition")
        is WinMLModelForGenericTask
    )
    assert TASK_REGISTRY["automatic-speech-recognition"].user_inputs[0].type == "audio"


def test_asr_default_dataset_is_pinned_and_bounded() -> None:
    from winml.modelkit.eval.evaluate import _DEFAULT_DATASETS

    default = _DEFAULT_DATASETS["automatic-speech-recognition"]
    assert default["path"] == "google/fleurs"
    assert default["name"] == "en_us"
    assert default["revision"] == "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
    assert default["samples"] == 2
    assert default["shuffle"] is False


@pytest.mark.parametrize("architecture", ["Wav2Vec2ForCTC", "HubertForCTC", "AutoModelForCTC"])
def test_ctc_architectures_are_metadata_driven(architecture: str) -> None:
    assert _is_ctc_config(SimpleNamespace(architectures=[architecture]))


def test_non_ctc_asr_fails_closed() -> None:
    assert not _is_ctc_config(SimpleNamespace(architectures=["WhisperForConditionalGeneration"]))

    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig

    model = MagicMock()
    model.config = SimpleNamespace(architectures=["WhisperForConditionalGeneration"])
    config = WinMLEvaluationConfig(
        model_id="org/whisper-model",
        task="automatic-speech-recognition",
        dataset=DatasetConfig(path="dataset"),
    )
    with pytest.raises(ValueError, match=r"only metadata-resolved \*ForCTC"):
        WinMLCTCASREvaluator(config, model)


def test_prepare_data_sorts_by_id_and_caps_rows() -> None:
    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig

    dataset = MagicMock()
    dataset.column_names = ["id", "audio", "transcription"]
    dataset.cast_column.return_value = dataset
    dataset.sort.return_value = dataset
    dataset.select.return_value = "selected"
    dataset.__len__.return_value = 5
    evaluator = WinMLCTCASREvaluator.__new__(WinMLCTCASREvaluator)
    evaluator._audio_column = "audio"
    evaluator.config = WinMLEvaluationConfig(
        task="automatic-speech-recognition",
        dataset=DatasetConfig(path="dataset", samples=2, shuffle=False),
    )

    with patch("datasets.load_dataset", return_value=dataset):
        assert evaluator.prepare_data() == "selected"

    dataset.sort.assert_called_once_with("id")
    selected_range = dataset.select.call_args.args[0]
    assert list(selected_range) == [0, 1]


def test_soundfile_decode_mixes_stereo_to_mono_float32() -> None:
    stereo = np.column_stack(
        [np.linspace(-1, 1, 32, dtype=np.float32), np.linspace(1, -1, 32, dtype=np.float32)]
    )
    waveform, sampling_rate = _decode_audio({"bytes": _wav_bytes(stereo), "path": None})
    assert sampling_rate == 16_000
    assert waveform.shape == (32,)
    assert waveform.dtype == np.float32
    np.testing.assert_allclose(waveform, 0.0, atol=1e-6)


def test_soundfile_decode_rejects_invalid_schema() -> None:
    with pytest.raises(_RejectedSampleError, match="neither bytes nor path"):
        _decode_audio({"bytes": None, "path": None})


def test_resample_uses_processor_rate_only_when_needed() -> None:
    waveform = np.arange(80, dtype=np.float32)
    assert _resample_audio(waveform, 16_000, 16_000) is waveform
    assert _resample_audio(waveform, 8_000, 16_000).shape == (160,)


def test_language_selection_uses_config_and_tokenizer_semantics() -> None:
    processor = _Processor(target_lang="eng")
    active = _configure_processor_language(processor, SimpleNamespace(target_lang="deu"))
    assert active == "deu"
    assert processor.tokenizer.target_lang == "deu"


def test_processor_published_language_is_preserved_without_config_override() -> None:
    processor = _Processor(target_lang="eng")
    assert _configure_processor_language(processor, SimpleNamespace()) == "eng"


def test_static_audio_windows_pad_and_insert_blank_for_processor_ctc_decode() -> None:
    processor = _Processor()
    evaluator = _evaluator(input_shape=[1, 4], processor=processor)
    logits = np.array([[[0, 5, 0], [0, 5, 0]]], dtype=np.float32)
    evaluator.model.side_effect = [
        {"logits": logits},
        {"logits": logits},
    ]
    audio = {"bytes": _wav_bytes(np.arange(6, dtype=np.float32)), "path": None}

    assert evaluator._transcribe(audio) == "1 1"
    assert len(evaluator.model.call_args_list) == 2
    final_input = evaluator.model.call_args_list[1].kwargs["input_values"]
    assert final_input.shape == (1, 4)
    np.testing.assert_array_equal(final_input[0, 2:], np.zeros(2))
    assert processor.decoded_ids == [1, 1, 0, 1, 1]


def test_dynamic_audio_runs_full_utterance_once() -> None:
    evaluator = _evaluator(input_shape=[1, "samples"])
    evaluator.model.return_value = {
        "logits": np.array([[[0, 5, 0], [5, 0, 0], [0, 0, 5]]], dtype=np.float32)
    }
    audio = {"bytes": _wav_bytes(np.arange(9, dtype=np.float32)), "path": None}

    assert evaluator._transcribe(audio) == "1 2"
    assert evaluator.model.call_count == 1
    assert evaluator.model.call_args.kwargs["input_values"].shape == (1, 9)


def test_static_audio_rejects_more_than_64_windows_before_inference() -> None:
    evaluator = _evaluator(input_shape=[1, 1])
    audio = {"bytes": _wav_bytes(np.arange(65, dtype=np.float32)), "path": None}
    with pytest.raises(_RejectedSampleError, match="cap is 64"):
        evaluator._transcribe(audio)
    evaluator.model.assert_not_called()


def test_vocab_mismatch_fails_closed() -> None:
    evaluator = _evaluator(input_shape=[1, "samples"])
    evaluator.model.return_value = {"logits": np.zeros((1, 2, 4), dtype=np.float32)}
    audio = {"bytes": _wav_bytes(np.ones(4, dtype=np.float32)), "path": None}
    with pytest.raises(ValueError, match="does not match active tokenizer"):
        evaluator._transcribe(audio)


def test_transcript_normalization_and_exact_wer_cer() -> None:
    assert _normalize_transcript("  hello\t  world  ") == "hello world"
    assert _corpus_error_rate(["hello world"], ["hello there"], words=True) == 0.5
    assert _corpus_error_rate(["abc"], ["adc"], words=False) == pytest.approx(1 / 3)


def test_compute_preserves_accounting_and_rejection_reasons() -> None:
    evaluator = _evaluator(input_shape=[1, "samples"])
    evaluator.data = [
        {"audio": "good", "transcription": " hello  world "},
        {"audio": "bad", "transcription": "ignored"},
    ]

    def transcribe(value: str) -> str:
        if value == "bad":
            raise _RejectedSampleError("bad audio")
        return "hello world"

    evaluator._transcribe = transcribe
    result = evaluator.compute()
    assert result["wer"] == 0.0
    assert result["cer"] == 0.0
    assert result["processed_samples"] == 1
    assert result["rejected_samples"] == 1
    assert result["rejection_reasons"] == {"bad audio": 1}


def test_compute_fails_closed_when_all_rows_are_rejected() -> None:
    evaluator = _evaluator(input_shape=[1, "samples"])
    evaluator.data = [{"audio": None, "transcription": ""}]
    with pytest.raises(DatasetValidationError, match="processed=0, rejected=1"):
        evaluator.compute()


def test_unexpected_inference_error_propagates() -> None:
    evaluator = _evaluator(input_shape=[1, "samples"])
    evaluator.data = [{"audio": "value", "transcription": "reference"}]
    evaluator._transcribe = MagicMock(side_effect=RuntimeError("session failed"))
    with pytest.raises(RuntimeError, match="session failed"):
        evaluator.compute()
