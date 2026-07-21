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
    has_quantized_weights: bool = False,
    input_shapes: tuple[tuple[object, ...], ...] = (),
    output_shapes: tuple[tuple[object, ...], ...] = (),
) -> onnx_hub._GraphContract:
    return onnx_hub._GraphContract(
        Path(name),
        inputs,
        outputs,
        precision,
        has_quantized_weights,
        input_shapes,
        output_shapes,
    )


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


def test_resolver_discovers_prompt_bearing_encoder_and_embedding_only_decoder(
    monkeypatch,
) -> None:
    graphs = {
        "first.onnx": _graph(
            "first.onnx",
            inputs=(
                ("image", "tensor(float)", 4),
                ("coordinates", "tensor(float)", 3),
                ("labels", "tensor(float)", 2),
            ),
            outputs=(
                ("embedding", "tensor(float)", 4),
                ("sparse", "tensor(float)", 3),
            ),
            precision="fp32",
        ),
        "second.onnx": _graph(
            "second.onnx",
            inputs=(
                ("embedding", "tensor(float)", 4),
                ("sparse", "tensor(float)", 3),
            ),
            outputs=(
                ("mask", "tensor(float)", 4),
                ("quality", "tensor(float)", 2),
            ),
            precision="fp32",
        ),
    }
    monkeypatch.setattr("huggingface_hub.list_repo_files", lambda *args, **kwargs: list(graphs))
    monkeypatch.setattr(
        onnx_hub,
        "resolve_hf_onnx_path",
        lambda model_id, **kwargs: Path(model_id.rsplit("/", 1)[-1]),
    )
    monkeypatch.setattr(onnx_hub, "_inspect_runnable_graph", lambda path: graphs[path.name])

    assert onnx_hub.resolve_hf_onnx_encoder_decoder("org/model") == {
        "image-encoder": Path("first.onnx"),
        "prompt-decoder": Path("second.onnx"),
    }


def test_resolver_prefers_unquantized_graph_family(monkeypatch) -> None:
    graphs = {
        "encoder.onnx": _graph(
            "encoder.onnx",
            inputs=(("pixels", "tensor(float)", 4),),
            outputs=(("embedding", "tensor(float)", 4),),
            precision="fp32",
        ),
        "decoder.onnx": _graph(
            "decoder.onnx",
            inputs=(
                ("points", "tensor(float)", 4),
                ("labels", "tensor(int64)", 3),
                ("embedding", "tensor(float)", 4),
            ),
            outputs=(("scores", "tensor(float)", 3), ("masks", "tensor(float)", 5)),
            precision="fp32",
        ),
        "encoder_quantized.onnx": _graph(
            "encoder_quantized.onnx",
            inputs=(("pixels", "tensor(float)", 4),),
            outputs=(("embedding", "tensor(float)", 4),),
            precision="fp32",
            has_quantized_weights=True,
        ),
        "decoder_quantized.onnx": _graph(
            "decoder_quantized.onnx",
            inputs=(
                ("points", "tensor(float)", 4),
                ("labels", "tensor(int64)", 3),
                ("embedding", "tensor(float)", 4),
            ),
            outputs=(("scores", "tensor(float)", 3), ("masks", "tensor(float)", 5)),
            precision="fp32",
            has_quantized_weights=True,
        ),
    }
    monkeypatch.setattr("huggingface_hub.list_repo_files", lambda *args, **kwargs: list(graphs))
    monkeypatch.setattr(
        onnx_hub,
        "resolve_hf_onnx_path",
        lambda model_id, **kwargs: Path(model_id.rsplit("/", 1)[-1]),
    )
    monkeypatch.setattr(onnx_hub, "_inspect_runnable_graph", lambda path: graphs[path.name])

    result = onnx_hub.resolve_hf_onnx_encoder_decoder("org/model")

    assert result == {
        "image-encoder": Path("encoder.onnx"),
        "prompt-decoder": Path("decoder.onnx"),
    }


def test_resolver_rejects_multiple_valid_graph_pairs_as_ambiguous(monkeypatch) -> None:
    graphs = {
        "encoder_a.onnx": _graph(
            "encoder_a.onnx",
            inputs=(("pixels", "tensor(float)", 4),),
            outputs=(("embedding_a", "tensor(float)", 4),),
            precision="fp32",
        ),
        "decoder_a.onnx": _graph(
            "decoder_a.onnx",
            inputs=(
                ("points", "tensor(float)", 4),
                ("labels", "tensor(int64)", 3),
                ("embedding_a", "tensor(float)", 4),
            ),
            outputs=(("scores", "tensor(float)", 3), ("masks", "tensor(float)", 5)),
            precision="fp32",
        ),
        "encoder_b.onnx": _graph(
            "encoder_b.onnx",
            inputs=(("pixels", "tensor(float)", 4),),
            outputs=(("embedding_b", "tensor(float)", 4),),
            precision="fp32",
        ),
        "decoder_b.onnx": _graph(
            "decoder_b.onnx",
            inputs=(
                ("points", "tensor(float)", 4),
                ("labels", "tensor(int64)", 3),
                ("embedding_b", "tensor(float)", 4),
            ),
            outputs=(("scores", "tensor(float)", 3), ("masks", "tensor(float)", 5)),
            precision="fp32",
        ),
    }
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files",
        lambda *args, **kwargs: list(reversed(graphs)),
    )
    monkeypatch.setattr(
        onnx_hub,
        "resolve_hf_onnx_path",
        lambda model_id, **kwargs: Path(model_id.rsplit("/", 1)[-1]),
    )
    monkeypatch.setattr(onnx_hub, "_inspect_runnable_graph", lambda path: graphs[path.name])

    with pytest.raises(ValueError) as error:
        onnx_hub.resolve_hf_onnx_encoder_decoder("org/ambiguous-model")

    message = str(error.value)
    assert "multiple valid encoder/decoder pairs" in message
    assert "unable to select one unambiguously" in message
    assert message.index("encoder_a.onnx") < message.index("encoder_b.onnx")
    assert "decoder_a.onnx" in message
    assert "decoder_b.onnx" in message


def test_resolver_rejects_decoder_without_score_output(monkeypatch) -> None:
    graphs = {
        "encoder.onnx": _graph(
            "encoder.onnx",
            inputs=(("pixels", "tensor(float)", 4),),
            outputs=(("embedding", "tensor(float)", 4),),
            precision="fp32",
        ),
        "decoder.onnx": _graph(
            "decoder.onnx",
            inputs=(
                ("points", "tensor(float)", 4),
                ("labels", "tensor(int64)", 3),
                ("embedding", "tensor(float)", 4),
            ),
            outputs=(("masks", "tensor(float)", 5),),
            precision="fp32",
        ),
    }
    monkeypatch.setattr("huggingface_hub.list_repo_files", lambda *args, **kwargs: list(graphs))
    monkeypatch.setattr(
        onnx_hub,
        "resolve_hf_onnx_path",
        lambda model_id, **kwargs: Path(model_id.rsplit("/", 1)[-1]),
    )
    monkeypatch.setattr(onnx_hub, "_inspect_runnable_graph", lambda path: graphs[path.name])

    with pytest.raises(ValueError, match="do not contain a runnable promptable"):
        onnx_hub.resolve_hf_onnx_encoder_decoder("org/model")


@pytest.mark.parametrize(
    ("encoder_port", "decoder_port", "encoder_shape", "decoder_shape"),
    [
        (("embedding", "tensor(float)", 4), ("embedding", "tensor(float16)", 4), (), ()),
        (("embedding", "tensor(float)", 4), ("embedding", "tensor(float)", 3), (), ()),
        (
            ("embedding", "tensor(float)", 4),
            ("embedding", "tensor(float)", 4),
            (1, 256, 64, 64),
            (1, 128, 32, 32),
        ),
        (
            ("embedding", "tensor(float)", 4),
            ("embedding", "tensor(float)", 4),
            (1, 256, "spatial", "spatial"),
            (1, 256, 64, 32),
        ),
    ],
)
def test_pairing_rejects_incompatible_shared_port_contracts(
    encoder_port,
    decoder_port,
    encoder_shape,
    decoder_shape,
) -> None:
    encoder = _graph(
        "encoder.onnx",
        inputs=(("pixels", "tensor(float)", 4),),
        outputs=(encoder_port,),
        precision="fp32",
        output_shapes=(encoder_shape,) if encoder_shape else (),
    )
    decoder = _graph(
        "decoder.onnx",
        inputs=(
            ("points", "tensor(float)", 4),
            ("labels", "tensor(int64)", 3),
            decoder_port,
        ),
        outputs=(("scores", "tensor(float)", 3), ("masks", "tensor(float)", 5)),
        precision="fp32",
        input_shapes=((None,) * 4, (None,) * 3, decoder_shape) if decoder_shape else (),
    )

    with pytest.raises(ValueError, match="do not contain a runnable promptable"):
        onnx_hub._select_encoder_decoder_pair([encoder, decoder], precision="fp32")


def test_pairing_accepts_compatible_symbolic_and_static_shared_port_shapes() -> None:
    encoder = _graph(
        "encoder.onnx",
        inputs=(("pixels", "tensor(float)", 4),),
        outputs=(("embedding", "tensor(float)", 4),),
        precision="fp32",
        output_shapes=((1, 256, "height", "width"),),
    )
    decoder = _graph(
        "decoder.onnx",
        inputs=(
            ("points", "tensor(float)", 4),
            ("labels", "tensor(int64)", 3),
            ("embedding", "tensor(float)", 4),
        ),
        outputs=(("scores", "tensor(float)", 3), ("masks", "tensor(float)", 5)),
        precision="fp32",
        input_shapes=((None,) * 4, (None,) * 3, (1, 256, 64, 64)),
    )

    pair = onnx_hub._select_encoder_decoder_pair([encoder, decoder], precision="fp32")

    assert pair.encoder.path == Path("encoder.onnx")
    assert pair.decoder.path == Path("decoder.onnx")


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
