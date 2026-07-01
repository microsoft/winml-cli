# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
r"""onnxruntime-genai inference for a genai bundle (decoder-pipeline).

Loads the genai bundle produced by ``export_qwen3_transformer_only.py
--genai-bundle <DIR>`` and runs greedy text generation using
:class:`~winml.modelkit.session.GenaiSession`.

The bundle directory must contain ``genai_config.json`` and the four ONNX
graphs it references (``embeddings.onnx``, ``ctx.onnx``, ``iter.onnx``,
``lm_head.onnx``) plus HF tokenizer files.

Usage::

    # CPU sanity check (works anywhere onnxruntime-genai is installed)
    uv run python scripts/infer_genai.py --prompt "Hello, who are you?" --chat

    # Qualcomm NPU (registers the QNN EP via the Windows ML EP catalog)
    uv run python scripts/infer_genai.py \\
        --prompt "Explain what a transformer is." \\
        --ep qnn --chat

    # Point at a non-default bundle
    uv run python scripts/infer_genai.py \\
        --model-dir out/my_bundle --prompt "Hi" --ep cpu

    # Pre-compile QNN stages to EPContext on first run; reuse cache on subsequent runs.
    # Eliminates per-run JIT overhead (~60-90 s saved on Snapdragon X Elite).
    uv run python scripts/infer_genai.py \\
        --prompt "Hello" --ep mixed --compile

Dependencies (install in a fresh venv)::

    pip install onnxruntime-genai-winml
    pip install "windowsml[with-ort]"   # registers QNN EP; also provides onnxruntime
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from winml.modelkit.session import GenaiSession, GenerationConfig


# Default bundle directory: <repo-root>/out/qwen3_bundle
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = _REPO_ROOT / "out" / "qwen3_bundle"

_SUPPORTED_EPS = ["cpu", "mixed", "qnn", "dml"]


def _wrap_chat_template(prompt: str) -> str:
    """Wrap *prompt* in the ChatML chat template."""
    return GenaiSession.apply_chatml_template(prompt)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--prompt",
        default="Give me a short introduction to large language models.",
        help="Input prompt (default: %(default)s).",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        metavar="DIR",
        help=(
            "Path to the genai bundle directory containing genai_config.json "
            "and the ONNX / tokenizer files (default: %(default)s)."
        ),
    )
    p.add_argument(
        "--ep",
        choices=_SUPPORTED_EPS,
        default="mixed",
        help="Execution provider: 'mixed' uses genai_config.json as-is (default); "
        "'cpu' forces all stages to CPU; 'qnn'/'dml' for full NPU/GPU.",
    )
    p.add_argument(
        "--max-new",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate (default: %(default)s).",
    )
    p.add_argument(
        "--chat",
        action="store_true",
        help="Wrap --prompt in the ChatML template (<|im_start|>user/assistant).",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Pre-compile QNN pipeline stages to EPContext ONNX before loading. "
            "On first use this triggers ort.ModelCompiler per stage (~60-90 s for iter). "
            "Compiled artifacts are cached in bundle_dir/_compiled/; "
            "subsequent runs reuse the cache and skip JIT. "
            "Has no effect when --ep cpu."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable onnxruntime-genai native model I/O logging.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load the genai bundle and run generation."""
    args = parse_args(argv)

    text = _wrap_chat_template(args.prompt) if args.chat else args.prompt
    gen_cfg = GenerationConfig(max_new_tokens=args.max_new, do_sample=False)

    try:
        session = GenaiSession(
            args.model_dir, ep=args.ep, verbose=args.verbose, compile=args.compile
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[load] ep={args.ep}  bundle={args.model_dir}")
    with session:
        print(f"[ctx]  context_length={session.context_length}")
        print("[gen] ", end="", flush=True)
        t0 = time.monotonic()
        n = 0
        for token_str in session.generate_streaming(text, gen_cfg):
            print(token_str, end="", flush=True)
            n += 1

    dt = time.monotonic() - t0
    print(f"\n\n[done] {n} tokens in {dt:.1f}s  ({n / max(dt, 1e-9):.1f} tok/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
