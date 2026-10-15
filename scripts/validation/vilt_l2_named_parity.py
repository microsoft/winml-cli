# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoModelForVisualQuestionAnswering, AutoProcessor


EXPECTED_INPUTS = ["input_ids", "attention_mask", "token_type_ids", "pixel_values"]
DEFAULT_REVISION = "d0a1f6ab88522427a7ae76ceb6e1e1e7b68a1d08"
DEFAULT_QUESTION = "What is the person doing?"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left64 = left.astype(np.float64).ravel()
    right64 = right.astype(np.float64).ravel()
    denominator = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    return float(np.dot(left64, right64) / denominator)


def _top_answer(logits: np.ndarray, id2label: dict[int | str, str]) -> tuple[int, str]:
    index = int(np.argmax(logits[0]))
    answer = id2label.get(index, id2label.get(str(index)))
    if answer is None:
        raise ValueError(f"Missing id2label entry for {index}")
    return index, str(answer)


def _run_onnx(
    label: str,
    path: Path,
    named_inputs: dict[str, np.ndarray],
    reference_logits: np.ndarray,
    id2label: dict[int | str, str],
) -> dict[str, Any]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_specs = [
        {"name": item.name, "shape": item.shape, "type": item.type}
        for item in session.get_inputs()
    ]
    names = [item["name"] for item in input_specs]
    if names != EXPECTED_INPUTS:
        raise ValueError(f"Unexpected ONNX input names/order: {names}")

    ort_inputs = {
        name: named_inputs[name].astype(
            np.float32 if name == "pixel_values" else np.int32,
            copy=False,
        )
        for name in names
    }
    candidate_logits = session.run(["logits"], ort_inputs)[0]

    reference_index, reference_answer = _top_answer(reference_logits, id2label)
    candidate_index, candidate_answer = _top_answer(candidate_logits, id2label)
    reference64 = reference_logits.astype(np.float64)
    candidate64 = candidate_logits.astype(np.float64)

    return {
        "label": label,
        "model": str(path),
        "model_sha256": _sha256(path),
        "providers": session.get_providers(),
        "input_specs": input_specs,
        "named_input_shapes": {name: list(value.shape) for name, value in ort_inputs.items()},
        "named_input_dtypes": {name: str(value.dtype) for name, value in ort_inputs.items()},
        "output_shape": list(candidate_logits.shape),
        "cosine": _cosine(reference_logits, candidate_logits),
        "max_abs": float(np.max(np.abs(reference64 - candidate64))),
        "mean_abs": float(np.mean(np.abs(reference64 - candidate64))),
        "reference_top_index": reference_index,
        "candidate_top_index": candidate_index,
        "reference_top_answer": reference_answer,
        "candidate_top_answer": candidate_answer,
        "top_index_agreement": reference_index == candidate_index,
        "top_answer_agreement": reference_answer == candidate_answer,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ViLT L2 named-input parity against fp32/fp16 ONNX artifacts and "
            "report cosine, max_abs, and top-answer agreement."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF model id or local snapshot path",
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Pinned HF model revision")
    parser.add_argument("--fp32", required=True, help="Path to fp32 ONNX model")
    parser.add_argument("--fp16", required=True, help="Path to fp16 ONNX model")
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Question text for parity input",
    )
    parser.add_argument("--json-out", default="", help="Optional output file path for JSON payload")
    return parser


def main() -> None:
    """Load deterministic ViLT inputs, compare fp32/fp16 ONNX vs PyTorch logits, and emit JSON."""
    args = _build_parser().parse_args()

    fp32_path = Path(args.fp32).resolve()
    fp16_path = Path(args.fp16).resolve()
    if not fp32_path.is_file():
        raise FileNotFoundError(f"Missing fp32 model: {fp32_path}")
    if not fp16_path.is_file():
        raise FileNotFoundError(f"Missing fp16 model: {fp16_path}")

    torch.manual_seed(0)
    np.random.seed(0)

    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForVisualQuestionAnswering.from_pretrained(
        args.model,
        revision=args.revision,
    ).eval()

    image_array = np.arange(384 * 384 * 3, dtype=np.uint32).reshape(384, 384, 3)
    image = Image.fromarray((image_array % 256).astype(np.uint8), mode="RGB")

    encoded = processor(
        images=image,
        text=args.question,
        return_tensors="pt",
        padding="max_length",
        max_length=40,
        truncation=True,
    )
    pt_inputs = {name: encoded[name] for name in EXPECTED_INPUTS}
    named_inputs = {name: value.detach().cpu().numpy() for name, value in pt_inputs.items()}

    with torch.inference_mode():
        reference_logits = model(**pt_inputs).logits.detach().cpu().numpy().astype(np.float32)

    id2label = model.config.id2label
    results = [
        _run_onnx("fp32", fp32_path, named_inputs, reference_logits, id2label),
        _run_onnx("fp16", fp16_path, named_inputs, reference_logits, id2label),
    ]

    payload = {
        "model_source": args.model,
        "model_revision": args.revision,
        "question": args.question,
        "image_generation": "deterministic arange modulo 256 RGB 384x384",
        "reference_shape": list(reference_logits.shape),
        "results": results,
        "pass": all(item["top_answer_agreement"] for item in results),
    }

    text = json.dumps(payload, indent=2)
    print(text)

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")

    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
