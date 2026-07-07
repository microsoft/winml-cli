# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""End-to-end coverage for the standalone Qwen3 embeddings + lm_head builds.

These complement ``test_qwen3_transformer_only_quant.py``: that test covers the
transformer (w8a16 QDQ); this one covers the two remaining sub-models of the
split Qwen3 graph:

  - **embeddings** (``model_type="qwen3_embeddings_only"``) — the input
    embedding table (``input_ids`` -> ``input_hidden_states``). Built float
    (``precision="fp32"``); it is **not** quantized.
  - **lm_head** (``model_type="qwen3_lm_head_only"``) — the vocab projection
    (``output_hidden_states`` -> ``logits``). Quantized weight-only to int4 via
    MatMulNBits/RTN (``precision="w4a32"``).

Both download Qwen3-0.6B from HuggingFace and run a full CPU export, so they are
gated behind ``slow`` + ``network`` and excluded from the default lane. All
expectations are generated in-code (FP reference), never hardcoded.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from winml.modelkit.models.auto import WinMLAutoModel


pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.network]

MODEL_ID = "Qwen/Qwen3-0.6B"
SEQ_LEN = 8


def _op_counts(onnx_path: str) -> dict[str, int]:
    graph = onnx.load(onnx_path, load_external_data=False).graph
    counts: dict[str, int] = {}
    for node in graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


# =============================================================================
# Embeddings — float export, NOT quantized
# =============================================================================


@pytest.fixture(scope="module")
def embeddings_model(tmp_path_factory):
    """Export the float embeddings sub-model once on CPU (no quantization)."""
    cache_dir = tmp_path_factory.mktemp("qwen3_embeddings")
    return WinMLAutoModel.from_pretrained(
        MODEL_ID,
        task="feature-extraction",
        model_type="qwen3_embeddings_only",
        precision="fp32",
        device="cpu",
        ep="cpu",
        no_compile=True,
        force_rebuild=True,
        shape_config={"seq_len": SEQ_LEN},
        cache_dir=str(cache_dir),
    )


@pytest.mark.timeout(1200)
def test_embeddings_not_quantized(embeddings_model):
    onnx_path = str(embeddings_model.onnx_path)
    counts = _op_counts(onnx_path)

    # Embedding lookup is a Gather; it must stay float (no QDQ, no MatMulNBits).
    assert counts.get("Gather", 0) > 0
    assert counts.get("QuantizeLinear", 0) == 0
    assert counts.get("DequantizeLinear", 0) == 0
    assert counts.get("MatMulNBits", 0) == 0

    # Output is named to chain into the transformer's input_hidden_states.
    out_names = {o.name for o in onnx.load(onnx_path, load_external_data=False).graph.output}
    assert "input_hidden_states" in out_names


@pytest.mark.timeout(1200)
def test_embeddings_parity_against_fp_reference(embeddings_model):
    """The float embeddings ONNX must match HF ``embed_tokens`` exactly."""
    onnx_path = str(embeddings_model.onnx_path)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    hf.eval()
    embed = hf.get_input_embeddings()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # The exported embeddings bake a fixed seq_len (= SEQ_LEN), so feed exactly
    # that many positions. Use a prompt long enough to slice to SEQ_LEN tokens.
    text = "The capital of France is Paris and the capital of Italy is"
    ids = tokenizer([text], return_tensors="pt").input_ids[:, :SEQ_LEN]
    assert ids.shape[1] == SEQ_LEN, f"prompt too short: {ids.shape[1]} < {SEQ_LEN}"
    with torch.no_grad():
        want = embed(ids).to(torch.float32).cpu().numpy()

    # The embedding Gather indexes with int64 (the exported input_ids dtype).
    in_name = session.get_inputs()[0].name
    got = session.run(None, {in_name: ids.to(torch.int64).cpu().numpy()})[0]

    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)


# =============================================================================
# LM head — weight-only int4 (MatMulNBits / RTN)
# =============================================================================


@pytest.fixture(scope="module")
def lm_head_model(tmp_path_factory):
    """Export + int4-RTN-quantize the lm_head sub-model once on CPU."""
    cache_dir = tmp_path_factory.mktemp("qwen3_lm_head")
    return WinMLAutoModel.from_pretrained(
        MODEL_ID,
        task="feature-extraction",
        model_type="qwen3_lm_head_only",
        precision="w4a32",
        device="cpu",
        ep="cpu",
        no_compile=True,
        force_rebuild=True,
        shape_config={"seq_len": SEQ_LEN},
        cache_dir=str(cache_dir),
    )


@pytest.mark.timeout(1800)
def test_lm_head_is_int4_quantized(lm_head_model):
    onnx_path = str(lm_head_model.onnx_path)
    counts = _op_counts(onnx_path)

    # The vocab projection is weight-only int4 -> a MatMulNBits node replaces
    # the float MatMul.
    assert counts.get("MatMulNBits", 0) > 0
    assert counts.get("MatMul", 0) == 0

    out_names = {o.name for o in onnx.load(onnx_path, load_external_data=False).graph.output}
    assert "logits" in out_names


@pytest.mark.timeout(1800)
def test_lm_head_parity_against_fp_reference(lm_head_model):
    """The int4 lm_head must track the FP reference's greedy token closely.

    4-bit weights are lossy, so we don't require bit-exact logits — instead the
    quantized head must pick the same argmax token as the FP head on real
    hidden states (the metric that actually matters for generation).
    """
    onnx_path = str(lm_head_model.onnx_path)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    hf.eval()
    lm_head = hf.lm_head
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # The int4 lm_head bakes a fixed seq_len (= SEQ_LEN), so feed exactly that
    # many positions. Use a prompt long enough to slice to SEQ_LEN tokens.
    text = "The capital of France is Paris and the capital of Italy is"
    ids = tokenizer([text], return_tensors="pt").input_ids[:, :SEQ_LEN]
    assert ids.shape[1] == SEQ_LEN, f"prompt too short: {ids.shape[1]} < {SEQ_LEN}"
    with torch.no_grad():
        hidden = hf.model(input_ids=ids, use_cache=False).last_hidden_state.to(torch.float32)
        want_logits = lm_head(hidden)
    want_tok = int(want_logits[:, -1, :].argmax(-1))

    in_name = session.get_inputs()[0].name
    got_logits = session.run(None, {in_name: hidden.cpu().numpy().astype(np.float32)})[0]
    got_tok = int(got_logits[0, -1].argmax())

    assert got_tok == want_tok, (
        f"int4 lm_head diverged from FP reference: fp={want_tok} quant={got_tok}"
    )
