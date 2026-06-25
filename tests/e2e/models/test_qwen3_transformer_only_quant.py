# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""End-to-end coverage for the transformer-only Qwen3 w8a16 build.

Replaces the former root-level ``test_qwen.py`` / ``qwen3_transformer_only_quantize.py``
scripts. Quantization is now driven entirely through the standard build
pipeline (``WinMLAutoModel.from_pretrained(..., precision="w8a16")``): the
device/precision policy enables the quantize stage, and the
``qwen3_transformer_only`` quant policy registered in
``winml.modelkit.quant.calibration`` (resolved via ``get_quant_finalizer``)
finalizes the reference-matched scheme (int8-symmetric weights, uint16
activations, GroupQueryAttention excluded from QDQ) plus the decode-trajectory
calibration reader.

These tests download Qwen3-0.6B from HuggingFace and run a full CPU export +
quantize, so they are gated behind ``slow`` + ``network`` and excluded from the
default lane. The QNN/NPU build is additionally gated on a real NPU.

All expectations are generated in-code (FP reference greedy decode), never
hardcoded from a prior model run.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from winml.modelkit.config import WinMLBuildConfig
from winml.modelkit.models.auto import WinMLAutoModel
from winml.modelkit.quant import WinMLQuantizationConfig


pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.network]

MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_CACHE = 256
PARITY_TOKENS = 8
DECODE_STEPS = 12
# Keep CPU calibration cheap: the decode reader emits ``samples * 16`` feeds.
CALIB_SAMPLES = 4


def _qnn_available() -> bool:
    """True when a QNN NPU device is reachable via the WinML autoEP path."""
    try:
        from winml.modelkit.winml import get_registered_ep_devices
    except Exception:
        return "QNNExecutionProvider" in ort.get_available_providers()

    try:
        devices = get_registered_ep_devices()
    except Exception:
        return False

    for device in devices:
        ep_name = str(getattr(device, "ep_name", ""))
        device_type = getattr(getattr(device, "device", None), "type", None)
        if ep_name == "QNNExecutionProvider" and str(device_type).endswith("NPU"):
            return True
    return False


def _decoder_onnx_path(model, sub_name: str = "decoder_gen") -> str:
    """Locate the quantized decode ONNX behind the model handle.

    The decode-only build (``seq_len=1``) returns a single
    ``WinMLModelForGenericTask`` whose ``onnx_path`` is the quantized graph; a
    full composite build instead exposes it under ``sub_models[sub_name]``.
    Handle both so the test does not depend on which wrapper the build picks.
    """
    sub_models = getattr(model, "sub_models", None)
    if sub_models and sub_name in sub_models:
        return str(sub_models[sub_name].onnx_path)
    return str(model.onnx_path)


def _qdq_counts(onnx_path: str) -> dict[str, int]:
    graph = onnx.load(onnx_path, load_external_data=False).graph
    counts: dict[str, int] = {}
    for node in graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def _gqa_tensor_set(graph) -> set[str]:
    tensors: set[str] = set()
    for node in graph.node:
        if node.op_type == "GroupQueryAttention":
            tensors.update(node.input)
            tensors.update(node.output)
    return tensors


@pytest.fixture(scope="module")
def decode_quant_model(tmp_path_factory):
    """Build + quantize the decode (seq_len=1) sub-model once on CPU."""
    cache_dir = tmp_path_factory.mktemp("qwen3_w8a16")
    return WinMLAutoModel.from_pretrained(
        MODEL_ID,
        task="text2text-generation",
        model_type="qwen3_transformer_only",
        config=WinMLBuildConfig(quant=WinMLQuantizationConfig(samples=CALIB_SAMPLES)),
        precision="w8a16",
        device="cpu",
        ep="cpu",
        force_rebuild=True,
        shape_config={"max_cache_len": MAX_CACHE, "seq_len": 1},
        cache_dir=str(cache_dir),
    )


@pytest.mark.timeout(2400)
def test_decode_model_is_quantized_with_gqa_excluded(decode_quant_model):
    onnx_path = _decoder_onnx_path(decode_quant_model)
    counts = _qdq_counts(onnx_path)

    # QDQ nodes were inserted via the config-driven pipeline.
    assert counts.get("QuantizeLinear", 0) > 0
    assert counts.get("DequantizeLinear", 0) > 0
    # GroupQueryAttention survives in float (not quantized away).
    assert counts.get("GroupQueryAttention", 0) > 0

    # GQA exclusion contract: no QuantizeLinear/DequantizeLinear touches a GQA
    # input or output tensor (attention stays Cast -> GQA -> Cast).
    graph = onnx.load(onnx_path, load_external_data=False).graph
    gqa_tensors = _gqa_tensor_set(graph)
    touching = [
        node.name
        for node in graph.node
        if node.op_type in ("QuantizeLinear", "DequantizeLinear")
        and (set(node.input) & gqa_tensors or set(node.output) & gqa_tensors)
    ]
    assert touching == []


def _carry_kv(kv: dict[str, np.ndarray], out: dict[str, np.ndarray], num_layers: int) -> None:
    for i in range(num_layers):
        kv[f"past_keys_{i}"] = out[f"present_keys_{i}"]
        kv[f"past_values_{i}"] = out[f"present_values_{i}"]


def _seed_kv_from_fp(past, num_layers, num_kv_heads, head_dim, cur_len):
    """Copy an HF FP prefill cache into the decode model's fixed FP16 buffers."""
    kv: dict[str, np.ndarray] = {}
    for i in range(num_layers):
        layer = past[i] if not hasattr(past, "layers") else None
        if layer is not None:
            k, v = past[i][0], past[i][1]
        else:  # newer per-layer DynamicCache
            k, v = past.layers[i].keys, past.layers[i].values
        kbuf = np.zeros((1, num_kv_heads, MAX_CACHE, head_dim), np.float16)
        vbuf = np.zeros_like(kbuf)
        kbuf[:, :, :cur_len, :] = k[:, :, :cur_len, :].to(torch.float16).cpu().numpy()
        vbuf[:, :, :cur_len, :] = v[:, :, :cur_len, :].to(torch.float16).cpu().numpy()
        kv[f"past_keys_{i}"] = kbuf
        kv[f"past_values_{i}"] = vbuf
    return kv


@pytest.mark.timeout(2400)
def test_decode_parity_against_fp_reference(decode_quant_model):
    """The w8a16 decode model must track the FP reference token-for-token.

    This is the regression guard against the historical "decode collapse":
    a degenerate calibration (single repeated token + zeroed KV) made the
    quantized decode model diverge into garbage after ~1 token. With the
    decode-trajectory reader the quantized greedy trajectory must match the
    FP reference for the first ``PARITY_TOKENS`` tokens.
    """
    onnx_path = _decoder_onnx_path(decode_quant_model)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    want = {i.name for i in session.get_inputs()}

    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    hf.eval()
    cfg = hf.config
    embed = hf.get_input_embeddings()
    lm_head = hf.lm_head
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer([text], return_tensors="pt").input_ids
    cur_len = ids.shape[1]
    assert cur_len < MAX_CACHE

    # --- FP reference greedy decode (generates the expected tokens) ---
    with torch.no_grad():
        out = hf(input_ids=ids, use_cache=True)
    fp_past = out.past_key_values
    first_tok = int(out.logits[:, -1, :].argmax(-1))
    fp_tokens: list[int] = []
    tok, past = first_tok, fp_past
    for _ in range(DECODE_STEPS):
        with torch.no_grad():
            out = hf(input_ids=torch.tensor([[tok]]), past_key_values=past, use_cache=True)
        past = out.past_key_values
        tok = int(out.logits[:, -1, :].argmax(-1))
        fp_tokens.append(tok)

    # --- Quantized decode model greedy decode (own KV, FP embed + lm_head) ---
    with torch.no_grad():
        seed = hf(input_ids=ids, use_cache=True)
    kv = _seed_kv_from_fp(seed.past_key_values, num_layers, num_kv_heads, head_dim, cur_len)
    quant_tokens: list[int] = []
    tok, past_len = first_tok, cur_len
    for _ in range(DECODE_STEPS):
        with torch.no_grad():
            emb = embed(torch.tensor([[tok]])).to(torch.float32).cpu().numpy()
        feeds = {
            "input_hidden_states": emb.astype(np.float32),
            "past_seq_len": np.array([[past_len]], np.int32),
            "total_seq_len": np.array([MAX_CACHE], np.int32),
            **kv,
        }
        feeds = {k: v for k, v in feeds.items() if k in want}
        names = [o.name for o in session.get_outputs()]
        outs = dict(zip(names, session.run(None, feeds), strict=False))
        _carry_kv(kv, outs, num_layers)
        hidden = torch.tensor(outs["output_hidden_states"][:, 0, :])
        with torch.no_grad():
            tok = int(lm_head(hidden).numpy()[0].argmax())
        quant_tokens.append(tok)
        past_len += 1

    assert quant_tokens[:PARITY_TOKENS] == fp_tokens[:PARITY_TOKENS], (
        f"w8a16 decode diverged from FP reference:\n"
        f"  fp   : {fp_tokens[:PARITY_TOKENS]}\n"
        f"  quant: {quant_tokens[:PARITY_TOKENS]}"
    )


@pytest.mark.npu
@pytest.mark.qnn
@pytest.mark.timeout(2400)
@pytest.mark.skipif(not _qnn_available(), reason="requires QNN execution provider (NPU)")
@pytest.mark.parametrize(
    ("task", "seq_len"),
    [("feature-extraction", 64), ("text2text-generation", 1)],
)
def test_npu_build_quantizes(task, seq_len, tmp_path):
    """On real NPU hardware, the w8a16 pipeline produces a quantized graph."""
    model = WinMLAutoModel.from_pretrained(
        MODEL_ID,
        task=task,
        model_type="qwen3_transformer_only",
        precision="w8a16",
        device="npu",
        ep="qnn",
        no_compile=True,
        force_rebuild=True,
        shape_config={"max_cache_len": MAX_CACHE, "seq_len": seq_len},
        cache_dir=str(tmp_path),
    )
    sub_name = "decoder_prefill" if seq_len == 64 else "decoder_gen"
    onnx_path = _decoder_onnx_path(model, sub_name)
    counts = _qdq_counts(onnx_path)
    assert counts.get("QuantizeLinear", 0) > 0
    assert counts.get("GroupQueryAttention", 0) > 0
