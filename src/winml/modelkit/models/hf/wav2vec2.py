# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Wav2Vec2 HuggingFace model class routing.

Wav2Vec2 supports several audio heads. Optimum can infer/export the ONNX config,
but its task default for ASR is ambiguous between seq2seq and CTC loaders. Route
CTC ASR checkpoints through ``AutoModelForCTC`` so recipe-free resolution follows
the architecture head instead of the seq2seq fallback.
"""

from __future__ import annotations

from transformers import AutoModelForAudioClassification, AutoModelForCTC


MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("wav2vec2", "audio-classification"): AutoModelForAudioClassification,
    ("wav2vec2", "automatic-speech-recognition"): AutoModelForCTC,
}
