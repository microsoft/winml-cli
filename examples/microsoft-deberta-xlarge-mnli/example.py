# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Run one MNLI inference with the WinML-built ONNX and print the prediction.

Mirrors the HuggingFace DeBERTa example
(https://huggingface.co/docs/transformers/main/en/model_doc/deberta) but
loads the quantized ONNX produced by ``winml build`` (step 1 of the
README) via :class:`WinMLAutoModel` instead of the original PyTorch
checkpoint.

Usage::

    uv run python examples/microsoft-deberta-xlarge-mnli/example.py `
      --onnx $HOME/.cache/winml/artifacts/microsoft_deberta-xlarge-mnli/`
            `txtcls_<hash>_quantized.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from winml.modelkit import WinMLAutoModel


HF_MODEL_ID = "microsoft/deberta-xlarge-mnli"

# Default MNLI sample: a textbook entailment pair, mirrors the HF docs style.
DEFAULT_PREMISE = "A soccer game with multiple males playing."
DEFAULT_HYPOTHESIS = "Some men are playing a sport."


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        required=True,
        type=Path,
        help="Path to the quantized ONNX produced by step 1 of the README "
        "(e.g. txtcls_<hash>_quantized.onnx).",
    )
    parser.add_argument(
        "--device",
        default="npu",
        choices=["auto", "npu", "gpu", "cpu"],
        help="Target device (default: npu).",
    )
    parser.add_argument(
        "--ep",
        default="openvino",
        help="Execution provider alias (default: openvino).",
    )
    parser.add_argument(
        "--premise",
        default=DEFAULT_PREMISE,
        help=f'Premise sentence. Default: "{DEFAULT_PREMISE}"',
    )
    parser.add_argument(
        "--hypothesis",
        default=DEFAULT_HYPOTHESIS,
        help=f'Hypothesis sentence. Default: "{DEFAULT_HYPOTHESIS}"',
    )
    return parser.parse_args()


def main() -> None:
    """Load the quantized ONNX, run one MNLI inference, print the prediction."""
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)

    # skip_build=True uses the ONNX as-is; it has already been optimized
    # and quantized by `winml build`. use_cache=False avoids touching the
    # winml artifact cache for this read-only example.
    model = WinMLAutoModel.from_pretrained(
        args.onnx.expanduser(),
        task="text-classification",
        device=args.device,
        ep=args.ep,
        skip_build=True,
        use_cache=False,
    )

    # The ONNX bakes in a fixed sequence length on input_ids; read it and
    # pad/truncate the tokenizer output to match. shape[0] is batch,
    # shape[1] is the static seq_len.
    input_shapes = (model.io_config.get("input_shapes") or [[]])[0]
    seq_len = input_shapes[1] if len(input_shapes) >= 2 else None

    encoding = tokenizer(
        args.premise,
        args.hypothesis,
        padding="max_length" if seq_len else "longest",
        max_length=seq_len,
        truncation=True,
        return_tensors="pt",
    )

    # Only forward the input names the ONNX actually accepts (DeBERTa-v1
    # uses token_type_ids; DeBERTa-v2/v3 don't).
    accepted = set(model.io_config.get("input_names", []))
    forward_kwargs = {}
    for name in ("input_ids", "attention_mask", "token_type_ids"):
        if name in accepted and name in encoding:
            forward_kwargs[name] = encoding[name]

    outputs = model(**forward_kwargs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    pred_id = int(torch.argmax(logits, dim=-1).item())
    score = float(probs[0, pred_id].item())

    # WinML's bare-ONNX path doesn't attach an HF config to the model, so
    # pull id2label from the HF hub for human-readable label names.
    id2label = AutoConfig.from_pretrained(HF_MODEL_ID).id2label
    label = id2label.get(pred_id, str(pred_id))

    print(f"Premise:    {args.premise}")
    print(f"Hypothesis: {args.hypothesis}")
    print(f"Predicted:  {label} (score={score:.4f})")


if __name__ == "__main__":
    main()
