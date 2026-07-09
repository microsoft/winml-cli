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
import sys
import time
from pathlib import Path

from winml.modelkit.models.hf.qwen3.genai import QWEN3_GENAI_BUNDLE_RECIPE
from winml.modelkit.models.winml import build_genai_bundle


_DEVICES = ("cpu", "gpu", "npu")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BUNDLE = _REPO_ROOT / "out" / "bundle"


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
        choices=_DEVICES,
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
    p.set_defaults(func=_cmd_export)


def _cmd_export(args: argparse.Namespace) -> int:
    """Build all components and write the genai bundle.

    Thin wrapper over the shared, architecture-agnostic
    :func:`~winml.modelkit.models.winml.build_genai_bundle` orchestrator driven
    by the registered Qwen3 recipe. The same bundle is produced by
    ``winml build -m <qwen3> -o <dir> --device npu --ep qnn``.
    """
    companion_overrides: dict[str, Path] = {}
    if args.embeddings is not None:
        companion_overrides["embeddings"] = args.embeddings
    if args.lm_head is not None:
        companion_overrides["lm_head"] = args.lm_head

    t0 = time.monotonic()
    config_path = build_genai_bundle(
        args.model_id,
        args.output,
        QWEN3_GENAI_BUNDLE_RECIPE,
        ep="qnn" if args.device == "npu" else args.device,
        device=args.device,
        precision=args.precision,
        max_cache_len=args.max_cache_len,
        prefill_seq_len=args.prefill_seq_len,
        companion_overrides=companion_overrides or None,
        force_rebuild=args.force_rebuild,
        emit=print,
    )
    elapsed = time.monotonic() - t0
    print(f"\n=== export complete in {elapsed:.1f}s: {config_path} ===")
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
    # Each subparser registers its handler via set_defaults(func=...); dispatch
    # generically so new subcommands route to their own handler (not export).
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
