# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""One-shot export of the Qwen3 embeddings + lm_head sub-models.

Companion to ``export_qwen3_transformer_only.py``. That script builds the
transformer (prefill + decode). This one builds the two remaining pieces of the
split Qwen3 graph:

  - ``embeddings`` — input embedding table (``input_ids`` -> ``input_hidden_states``).
    Built FLOAT (``precision="fp32"``); NOT quantized. Produces a ``Gather``.
  - ``lm_head``    — vocab projection (``output_hidden_states`` -> ``logits``).
    Quantized weight-only to int4 via MatMulNBits/RTN (``precision="w4a32"``).
    Produces a ``MatMulNBits`` node (no float ``MatMul``).

Both are standalone ``model_type`` builds, invoked separately.

Usage::

    # Build (or reuse cached) both ONNX, print their paths + node summary:
    uv run python scripts/export_qwen3_embeddings_lm_head.py

    # Build only one of them:
    uv run python scripts/export_qwen3_embeddings_lm_head.py --only embeddings
    uv run python scripts/export_qwen3_embeddings_lm_head.py --only lm_head

    # Copy the ONNX (with external data) into a folder:
    uv run python scripts/export_qwen3_embeddings_lm_head.py --output-dir out/qwen3
    #   -> writes embeddings_fp32.onnx and lm_head_w4a32.onnx

    # Different model / device / seq geometry, force a rebuild:
    uv run python scripts/export_qwen3_embeddings_lm_head.py \
        --model-id Qwen/Qwen3-0.6B --device npu --seq-len 64 --force-rebuild
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import onnx

from winml.modelkit.models.auto import WinMLAutoModel
from winml.modelkit.onnx import copy_onnx_model


# Per-component build settings: which model_type to register against and the
# precision that drives its quant policy. Embeddings stay float (``fp32``); the
# lm_head is weight-only int4 with fp32 activations (``w4a32`` — the activations
# are NOT quantized, this is RTN/MatMulNBits).
_COMPONENTS = {
    "embeddings": {"model_type": "qwen3_embeddings_only", "precision": "fp32"},
    "lm_head": {"model_type": "qwen3_lm_head_only", "precision": "w4a32"},
}

# Component -> output file stem (when --output-dir is given). The precision
# suffix is carried in the filename so the two pieces self-document their scheme.
_OUTPUT_STEMS = {
    "embeddings": f"embeddings_{_COMPONENTS['embeddings']['precision']}",
    "lm_head": f"lm_head_{_COMPONENTS['lm_head']['precision']}",
}

# Default EP per device; CPU/NPU/GPU map to their canonical providers.
_DEVICE_TO_EP = {
    "cpu": "CPUExecutionProvider",
    "npu": "QNNExecutionProvider",
    "gpu": "DmlExecutionProvider",
}


def node_summary(path: str | Path) -> str:
    """Return a one-line structural summary of the graph's key ops.

    The interesting markers for these two sub-models are:
      - embeddings: ``Gather`` present, no ``MatMulNBits`` / no QDQ (stays float).
      - lm_head:    ``MatMulNBits`` present, float ``MatMul`` gone (int4 weight-only).
    """
    model = onnx.load(str(path), load_external_data=False)
    counts = collections.Counter(n.op_type for n in model.graph.node)
    return (
        f"Gather={counts['Gather']} MatMul={counts['MatMul']} "
        f"MatMulNBits={counts['MatMulNBits']} "
        f"Q={counts['QuantizeLinear']} DQ={counts['DequantizeLinear']}"
    )


def build_component(name: str, args: argparse.Namespace):
    """Build (or reuse cached) one standalone sub-model and return it."""
    spec = _COMPONENTS[name]
    print(f"\n=== building {name} (model_type={spec['model_type']}, "
          f"precision={spec['precision']}) ===")
    return WinMLAutoModel.from_pretrained(
        args.model_id,
        task="feature-extraction",
        model_type=spec["model_type"],
        precision=spec["precision"],
        device=args.device,
        ep=_DEVICE_TO_EP[args.device],
        no_compile=args.no_compile,
        use_cache=True,
        force_rebuild=args.force_rebuild,
        shape_config={"seq_len": args.seq_len},
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B", help="HF model id or local path.")
    p.add_argument(
        "--device",
        default="cpu",
        choices=sorted(_DEVICE_TO_EP),
        help="Target device (selects the canonical EP). Default: cpu.",
    )
    p.add_argument(
        "--only",
        choices=sorted(_COMPONENTS),
        default=None,
        help="Build only this component. Default: build both.",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Static sequence length baked into both sub-models. Default: 64.",
    )
    p.add_argument(
        "--no-compile",
        dest="no_compile",
        action="store_true",
        default=True,
        help="Skip EPContext compilation (default; these are consumed pre-compile).",
    )
    p.add_argument(
        "--compile",
        dest="no_compile",
        action="store_false",
        help="Enable EPContext compilation (requires the device's compiler/SDK).",
    )
    p.add_argument("--force-rebuild", action="store_true", help="Rebuild even if cached.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, copy the ONNX (with external data) here as "
             "embeddings_fp32.onnx / lm_head_w4a32.onnx.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build (or reuse) the requested sub-models and report/copy them."""
    args = parse_args(argv)

    names = [args.only] if args.only else ["embeddings", "lm_head"]

    output_dir: Path | None = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    for name in names:
        model = build_component(name, args)
        src = Path(model.onnx_path)
        print(f"[{name}] {src}")
        print(f"   {node_summary(src)}")
        if output_dir is not None:
            dst = output_dir / f"{_OUTPUT_STEMS[name]}.onnx"
            copy_onnx_model(src, dst)
            print(f"   -> copied to {dst}")

    print(f"\n=== done in {time.monotonic() - t0:.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
