# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""One-shot export of the Qwen3 transformer-only prefill + decode pair.

Leverages the registered ``WinMLQwen3TransformerOnlyModel`` composite to build
BOTH transformer-only sub-models in a single call:

  - ``decoder_prefill`` — context graph, ``seq_len`` = --prefill-seq-len (64)
  - ``decoder_gen``     — iteration graph, ``seq_len`` = 1

Each sub-model is built through the standard ``build_hf_model`` pipeline, so the
model-type quant finalizer is applied (int8 weight / uint16 activation, GQA
excluded from QDQ). Embeddings and the LM head are NOT part of this graph — they
run separately (e.g. from the bundle).

Usage::

    # Build (or reuse cached) both ONNX, print their paths + node summary:
    uv run python scripts/export_qwen3_transformer_only.py

    # Copy the two ONNX (with external data) into a folder:
    uv run python scripts/export_qwen3_transformer_only.py --output-dir out/qwen3

    # Different model / device / cache geometry, force a rebuild:
    uv run python scripts/export_qwen3_transformer_only.py \
        --model-id Qwen/Qwen3-0.6B --device npu \
        --max-cache-len 256 --prefill-seq-len 64 --force-rebuild
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import onnx

from winml.modelkit.models.hf.qwen3.qwen_transformer_only import (
    WinMLQwen3TransformerOnlyModel,
)
from winml.modelkit.onnx import copy_onnx_model


# Component name -> output file stem used when --output-dir is given.
_OUTPUT_STEMS = {
    "decoder_prefill": "prefill",
    "decoder_gen": "decode",
}

# Default EP per device; CPU/NPU/GPU map to their canonical providers.
_DEVICE_TO_EP = {
    "cpu": "CPUExecutionProvider",
    "npu": "QNNExecutionProvider",
    "gpu": "DmlExecutionProvider",
}


def node_summary(path: str | Path) -> str:
    """Return a one-line QDQ/GQA structural summary of an ONNX graph."""
    model = onnx.load(str(path), load_external_data=False)
    counts = collections.Counter(n.op_type for n in model.graph.node)
    gqa_io: set[str] = set()
    for node in model.graph.node:
        if node.op_type == "GroupQueryAttention":
            gqa_io.update(node.input)
            gqa_io.update(node.output)
    qdq_touching_gqa = sum(
        1
        for n in model.graph.node
        if n.op_type in ("QuantizeLinear", "DequantizeLinear")
        and (set(n.input) & gqa_io or set(n.output) & gqa_io)
    )
    return (
        f"Q={counts['QuantizeLinear']} DQ={counts['DequantizeLinear']} "
        f"GQA={counts['GroupQueryAttention']} QDQ-touching-GQA={qdq_touching_gqa}"
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
    p.add_argument("--precision", default="w8a16", help="Build precision. Default: w8a16.")
    p.add_argument("--max-cache-len", type=int, default=256, help="Static KV cache length.")
    p.add_argument(
        "--prefill-seq-len",
        type=int,
        default=64,
        help="Prefill/context sequence length.",
    )
    p.add_argument(
        "--no-compile",
        dest="no_compile",
        action="store_true",
        default=True,
        help="Skip EPContext compilation (default; transformer-only is consumed pre-compile).",
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
        help="If set, copy the two ONNX (with external data) here as prefill.onnx / decode.onnx.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build (or reuse) both transformer-only ONNX and report/copy them."""
    args = parse_args(argv)

    t0 = time.monotonic()
    model = WinMLQwen3TransformerOnlyModel.from_pretrained(
        args.model_id,
        device=args.device,
        precision=args.precision,
        ep=_DEVICE_TO_EP[args.device],
        no_compile=args.no_compile,
        use_cache=True,
        force_rebuild=args.force_rebuild,
        sub_model_kwargs={
            "decoder_prefill": {
                "shape_config": {
                    "max_cache_len": args.max_cache_len,
                    "seq_len": args.prefill_seq_len,
                }
            },
            "decoder_gen": {"shape_config": {"max_cache_len": args.max_cache_len, "seq_len": 1}},
        },
    )
    elapsed = time.monotonic() - t0

    print(f"\n=== transformer-only build done in {elapsed:.1f}s ===")

    output_dir: Path | None = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for name, sub in model.sub_models.items():
        src = Path(sub.onnx_path)
        print(f"\n[{name}] {src}")
        print(f"   {node_summary(src)}")
        if output_dir is not None:
            dst = output_dir / f"{_OUTPUT_STEMS.get(name, name)}.onnx"
            copy_onnx_model(src, dst)
            print(f"   -> copied to {dst}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
