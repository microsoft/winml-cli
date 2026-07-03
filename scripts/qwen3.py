# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
r"""Qwen3 genai bundle export.

Builds (or reuses) all four components of the Qwen3 genai bundle and assembles
them into an onnxruntime-genai directory:

  - ``ctx.onnx``        — transformer prefill graph (QDQ-quantized)
  - ``iter.onnx``       — transformer decode graph  (QDQ-quantized)
  - ``embeddings.onnx`` — token embedding table     (fp32)
  - ``lm_head.onnx``    — vocab projection          (w4a32 MatMulNBits)
  - ``genai_config.json`` + HF tokenizer files

Inference over the assembled bundle is provided separately (see the genai
inference session), so this script only covers bundle generation.

Usage::

    # Full export to out/bundle (default context_length=2048):
    uv run python scripts/qwen3.py export --device npu --output out/bundle

    # Force rebuild from scratch:
    uv run python scripts/qwen3.py export --device npu --output out/bundle --force-rebuild
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import onnx

from winml.modelkit.models.auto import WinMLAutoModel
from winml.modelkit.models.hf.qwen3.qwen_transformer_only import (
    WinMLQwen3TransformerOnlyModel,
)
from winml.modelkit.onnx import strip_node_attrs


_DEVICE_TO_EP = {
    "cpu": "CPUExecutionProvider",
    "npu": "QNNExecutionProvider",
    "gpu": "DmlExecutionProvider",
}

# Build specs for the two CPU-side companion models.
_COMPANION_COMPONENTS: dict[str, dict[str, str]] = {
    "embeddings": {"model_type": "qwen3_embeddings_only", "precision": "fp32"},
    "lm_head": {"model_type": "qwen3_lm_head_only", "precision": "w4a32"},
}

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BUNDLE = _REPO_ROOT / "out" / "bundle"

# Attributes that com.microsoft::GroupQueryAttention requires for Qwen3.
# Any other attributes (e.g. k_quant_type, local_window_size, qk_output,
# smooth_softmax, v_quant_type) are default-valued extras injected by the
# TorchScript exporter from the ORT op schema; strip them so the bundle
# matches the expected minimal attribute set.
_GQA_KEEP_ATTRS = frozenset({"do_rotary", "kv_num_heads", "num_heads"})


def _strip_gqa_default_attrs(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove exporter-injected default attributes from GQA nodes."""
    return strip_node_attrs(model, "GroupQueryAttention", _GQA_KEEP_ATTRS, domain="com.microsoft")


# ---------------------------------------------------------------------------
# Helpers shared between sub-commands
# ---------------------------------------------------------------------------


def _node_summary(path: str | Path) -> str:
    """One-line op-type summary of an ONNX graph (loads shape metadata only)."""
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
        f"Gather={counts['Gather']} MatMulNBits={counts['MatMulNBits']} "
        f"Q={counts['QuantizeLinear']} DQ={counts['DequantizeLinear']} "
        f"GQA={counts['GroupQueryAttention']} QDQ@GQA={qdq_touching_gqa}"
    )


# ---------------------------------------------------------------------------
# export sub-command
# ---------------------------------------------------------------------------


def _add_export_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "export",
        help="Build the full Qwen3 genai bundle (transformer + embeddings + lm_head).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Exports all four Qwen3 genai components and assembles them into an "
            "onnxruntime-genai bundle directory.  Transformer stages (ctx/iter) are "
            "built for the target device; embeddings and lm_head always run on CPU."
        ),
    )
    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B", help="HF model id or local path.")
    p.add_argument(
        "--device",
        default="npu",
        choices=sorted(_DEVICE_TO_EP),
        help="Target device for transformer stages. Default: npu.",
    )
    p.add_argument("--precision", default="w8a16", help="Transformer precision. Default: w8a16.")
    p.add_argument(
        "--max-cache-len",
        type=int,
        default=2048,
        help="Static KV cache length (context_length). Default: 2048.",
    )
    p.add_argument(
        "--prefill-seq-len",
        type=int,
        default=64,
        help="Prefill/context sequence length baked into ctx.onnx. Default: 64.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_BUNDLE,
        metavar="DIR",
        help=f"Bundle output directory. Default: {_DEFAULT_BUNDLE}.",
    )
    p.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        metavar="ONNX",
        help="Override path to a pre-built embeddings ONNX (skips auto-build).",
    )
    p.add_argument(
        "--lm-head",
        type=Path,
        default=None,
        metavar="ONNX",
        help="Override path to a pre-built lm_head ONNX (skips auto-build).",
    )
    p.add_argument("--force-rebuild", action="store_true", help="Rebuild even if cached.")


def _cmd_export(args: argparse.Namespace) -> int:
    """Build all components and write the genai bundle."""
    from winml.modelkit.models.hf.qwen3.genai import write_genai_bundle

    t0 = time.monotonic()

    # --- Transformer (ctx + iter) ---
    print(f"\n=== building transformer stages (device={args.device}) ===")
    transformer = WinMLQwen3TransformerOnlyModel.from_pretrained(
        args.model_id,
        device=args.device,
        precision=args.precision,
        ep=_DEVICE_TO_EP[args.device],
        no_compile=True,
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
    prefill_path = Path(transformer.sub_models["decoder_prefill"].onnx_path)
    decode_path = Path(transformer.sub_models["decoder_gen"].onnx_path)
    for name, path in (("ctx", prefill_path), ("iter", decode_path)):
        print(f"  [{name}] {path}")
        print(f"        {_node_summary(path)}")

    # --- Companion models (embeddings + lm_head) ---
    embeddings_src = args.embeddings
    lm_head_src = args.lm_head

    for key, override in (("embeddings", embeddings_src), ("lm_head", lm_head_src)):
        if override is not None:
            print(f"\n=== using provided {key}: {override} ===")
            continue
        spec = _COMPANION_COMPONENTS[key]
        print(
            f"\n=== building {key} "
            f"(model_type={spec['model_type']}, precision={spec['precision']}) ==="
        )
        companion = WinMLAutoModel.from_pretrained(
            args.model_id,
            task="feature-extraction",
            model_type=spec["model_type"],
            precision=spec["precision"],
            device="cpu",
            ep=_DEVICE_TO_EP["cpu"],
            no_compile=True,
            use_cache=True,
            force_rebuild=args.force_rebuild,
        )
        companion_path = Path(companion.onnx_path)
        print(f"  [{key}] {companion_path}")
        print(f"        {_node_summary(companion_path)}")
        if key == "embeddings":
            embeddings_src = companion_path
        else:
            lm_head_src = companion_path

    # --- Assemble bundle ---
    print(f"\n=== assembling bundle -> {args.output} ===")
    config_path = write_genai_bundle(
        args.output,
        context_onnx=prefill_path,
        iterator_onnx=decode_path,
        model_id=args.model_id,
        max_cache_len=args.max_cache_len,
        prefill_seq_len=args.prefill_seq_len,
        embeddings_src=embeddings_src,
        lm_head_src=lm_head_src,
        ep="qnn" if args.device == "npu" else args.device,
        transformer_onnx_passes=[_strip_gqa_default_attrs],
    )
    print(f"  genai_config.json -> {config_path}")

    elapsed = time.monotonic() - t0
    print(f"\n=== export complete in {elapsed:.1f}s ===")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse sub-command and dispatch to the appropriate handler."""
    p = argparse.ArgumentParser(
        prog="qwen3",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    _add_export_parser(sub)

    args = p.parse_args(argv)
    return _cmd_export(args)


if __name__ == "__main__":
    sys.exit(main())
