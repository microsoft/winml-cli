# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for graph-contract-driven composite ONNX discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from winml.modelkit.loader import onnx_hub
from winml.modelkit.models.hf.sam import WinMLSAMModel


def _graph(
    name: str,
    *,
    inputs: tuple[tuple[str, str, int], ...],
    outputs: tuple[tuple[str, str, int], ...],
    precision: str,
) -> onnx_hub._GraphContract:
    return onnx_hub._GraphContract(Path(name), inputs, outputs, precision)


def test_resolver_selects_matching_graph_contract_and_falls_back_to_fp32(monkeypatch) -> None:
    graphs = {
        "encoder_fp16.onnx": None,
        "decoder_fp16.onnx": None,
        "encoder.onnx": _graph(
            "encoder.onnx",
            inputs=(("pixels", "tensor(float)", 4),),
            outputs=(("embedding", "tensor(float)", 4), ("position", "tensor(float)", 4)),
            precision="fp32",
        ),
        "decoder.onnx": _graph(
            "decoder.onnx",
            inputs=(
                ("points", "tensor(float)", 4),
                ("labels", "tensor(int64)", 3),
                ("embedding", "tensor(float)", 4),
                ("position", "tensor(float)", 4),
            ),
            outputs=(("scores", "tensor(float)", 3), ("masks", "tensor(float)", 5)),
            precision="fp32",
        ),
    }
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files",
        lambda *args, **kwargs: list(graphs),
    )
    monkeypatch.setattr(
        onnx_hub,
        "resolve_hf_onnx_path",
        lambda model_id, **kwargs: Path(model_id.rsplit("/", 1)[-1]),
    )
    monkeypatch.setattr(onnx_hub, "_inspect_runnable_graph", lambda path: graphs[path.name])

    result = onnx_hub.resolve_hf_onnx_encoder_decoder("org/model", precision="fp16")

    assert result == {
        "image-encoder": Path("encoder.onnx"),
        "prompt-decoder": Path("decoder.onnx"),
    }


def test_sam_published_onnx_absence_preserves_pytorch_fallback(monkeypatch) -> None:
    monkeypatch.setattr("huggingface_hub.list_repo_files", lambda *args, **kwargs: [])

    assert WinMLSAMModel.resolve_pretrained_onnx("facebook/sam-vit-base") is None


def test_sam_malformed_published_onnx_still_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        onnx_hub,
        "resolve_hf_onnx_encoder_decoder",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("ambiguous pair")),
    )

    with pytest.raises(ValueError, match="ambiguous pair"):
        WinMLSAMModel.resolve_pretrained_onnx("org/malformed-sam")
