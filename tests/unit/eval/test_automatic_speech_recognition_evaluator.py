# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for explicit CTC and seq2seq ASR evaluation."""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from winml.modelkit.eval.automatic_speech_recognition_evaluator import (
    WinMLAutomaticSpeechRecognitionEvaluator,
    _asr_mode,
    _word_error_rate,
)
from winml.modelkit.eval.config import DatasetConfig, WinMLEvaluationConfig
from winml.modelkit.utils.eval_utils import TASK_SCHEMAS, DatasetValidationError


def _evaluator(*, seq2seq: bool) -> WinMLAutomaticSpeechRecognitionEvaluator:
    evaluator = WinMLAutomaticSpeechRecognitionEvaluator.__new__(
        WinMLAutomaticSpeechRecognitionEvaluator
    )
    evaluator.config = WinMLEvaluationConfig(
        model_id="test/asr",
        task="automatic-speech-recognition",
        dataset=DatasetConfig(path="test", samples=1),
    )
    evaluator.model = MagicMock()
    evaluator.model.config = SimpleNamespace(is_encoder_decoder=seq2seq)
    evaluator.processor = MagicMock()
    evaluator.mode = "seq2seq" if seq2seq else "ctc"
    evaluator._audio_column = "audio"
    evaluator._label_column = "text"
    evaluator._max_audio_seconds = 30.0
    evaluator._max_new_tokens = 32
    evaluator._language = "english"
    evaluator._generation_task = "transcribe"
    return evaluator


def test_asr_schema_declares_audio_transcript_and_component_roles() -> None:
    schema = TASK_SCHEMAS["automatic-speech-recognition"]
    assert [item.default for item in schema.columns] == ["audio", "text"]
    assert schema.roles == ("encoder", "decoder")


def test_mode_is_selected_from_encoder_decoder_contract() -> None:
    assert _asr_mode(SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=True))) == "seq2seq"
    assert _asr_mode(SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=False))) == "ctc"


def test_word_error_rate() -> None:
    assert _word_error_rate("THE QUICK FOX", "the quick fox") == 0.0
    assert _word_error_rate("the fox", "the quick fox") == pytest.approx(1 / 3)


def test_decode_audio_mixes_resamples_and_caps() -> None:
    evaluator = _evaluator(seq2seq=True)
    evaluator._max_audio_seconds = 0.01
    stereo = np.ones((160, 2), dtype=np.float32)
    with patch(
        "scipy.signal.resample_poly", return_value=np.ones(320, dtype=np.float32)
    ) as resample:
        waveform = evaluator._decode_audio({"array": stereo, "sampling_rate": 8_000})
    resample.assert_called_once()
    assert waveform.shape == (160,)
    assert waveform.dtype == np.float32


def test_decode_audio_reads_raw_bytes() -> None:
    evaluator = _evaluator(seq2seq=True)
    decoded = np.ones(160, dtype=np.float32)
    with patch("soundfile.read", return_value=(decoded, 16_000)) as read_audio:
        waveform = evaluator._decode_audio({"bytes": b"flac", "path": None})

    source = read_audio.call_args.args[0]
    assert isinstance(source, io.BytesIO)
    assert source.read() == b"flac"
    assert np.array_equal(waveform, decoded)


def test_ctc_compute_decodes_frame_logits_and_accounts_exactly() -> None:
    evaluator = _evaluator(seq2seq=False)
    evaluator.data = [{"audio": {"array": np.ones(160), "sampling_rate": 16_000}, "text": "hi"}]
    evaluator.processor.return_value = {"input_values": torch.ones(1, 160)}
    evaluator.processor.batch_decode.return_value = ["hi"]
    evaluator.model.return_value = SimpleNamespace(logits=torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))

    result = evaluator.compute()

    evaluator.model.assert_called_once()
    evaluator.model.generate.assert_not_called()
    assert result == {
        "wer": 0.0,
        "asr_mode": "ctc",
        "requested_samples": 1,
        "processed_samples": 1,
        "skipped_samples": 0,
        "max_audio_seconds": 30.0,
        "max_new_tokens": 0,
        "num_beams": 0,
        "return_sequences": 1,
    }


def test_compute_uses_corpus_word_error_rate() -> None:
    evaluator = _evaluator(seq2seq=False)
    evaluator.config.dataset.samples = 2
    evaluator.data = [
        {"audio": {"array": np.ones(160), "sampling_rate": 16_000}, "text": "one"},
        {
            "audio": {"array": np.ones(160), "sampling_rate": 16_000},
            "text": "two three four",
        },
    ]
    evaluator.processor.return_value = {"input_values": torch.ones(1, 160)}
    evaluator.processor.batch_decode.side_effect = [["wrong"], ["two three four"]]
    evaluator.model.return_value = SimpleNamespace(logits=torch.ones(1, 1, 1))

    result = evaluator.compute()

    assert result["wer"] == 0.25
    assert result["processed_samples"] == 2


def test_seq2seq_compute_uses_bounded_generation() -> None:
    evaluator = _evaluator(seq2seq=True)
    evaluator.data = [{"audio": {"array": np.ones(160), "sampling_rate": 16_000}, "text": "hello"}]
    evaluator.processor.return_value = {"input_features": torch.ones(1, 80, 3000)}
    evaluator.processor.get_decoder_prompt_ids = MagicMock()
    evaluator.processor.batch_decode.return_value = ["hello"]
    evaluator.model.generation_config = SimpleNamespace()
    evaluator.model.generate.return_value = torch.tensor([[1, 2]])

    result = evaluator.compute()

    generation_config = evaluator.model.generate.call_args.kwargs["generation_config"]
    assert generation_config.max_new_tokens == 32
    assert generation_config.num_beams == 1
    assert generation_config.do_sample is False
    assert generation_config.language == "english"
    assert generation_config.task == "transcribe"
    assert generation_config.return_timestamps is False
    assert result["asr_mode"] == "seq2seq"
    assert result["processed_samples"] == 1


def test_compute_fails_closed_on_missing_or_empty_data() -> None:
    evaluator = _evaluator(seq2seq=False)
    evaluator.data = [{"audio": None, "text": "reference"}]
    with pytest.raises(DatasetValidationError, match="row 0"):
        evaluator.compute()

    evaluator.data = []
    with pytest.raises(DatasetValidationError, match="no usable rows"):
        evaluator.compute()
