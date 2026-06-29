# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
r"""onnxruntime-genai inference for the Qwen3 transformer-only pipeline.

Loads the genai bundle produced by ``export_qwen3_transformer_only.py
--genai-bundle <DIR>`` and runs greedy text generation.

The bundle directory must contain ``genai_config.json`` and the four ONNX
graphs it references:

  embeddings.onnx       — embedding lookup (input_ids -> input_hidden_states)
  ctx.onnx              — prefill/context graph (seq_len = prefill_seq_len)
  iter.onnx             — iteration/decode graph (seq_len = 1)
  lm_head.onnx          — lm_head (output_hidden_states -> logits)

It also needs the HF tokenizer files (``tokenizer.json``,
``tokenizer_config.json``, ``vocab.json``, ``merges.txt``,
``generation_config.json``) which ``write_genai_bundle`` downloads
automatically.

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

Dependencies (install in a fresh venv)::

    pip install onnxruntime-genai-winml
    pip install "windowsml[with-ort]"   # registers QNN EP; also provides onnxruntime
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import onnxruntime_genai as og


# Default bundle directory: <repo-root>/out/qwen3_bundle
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = _REPO_ROOT / "out" / "qwen3_bundle"

# The static KV cache length.  Must equal ``context_length`` in genai_config.json
# (and the ``--max-cache-len`` used during the winml build).  Do not lower this
# value — the KV buffer size is baked into the ONNX graphs.
CONTEXT_LENGTH = 256

# Maps the friendly --ep name to the ORT EP canonical name.
_EP_NAME = {
    "cpu": "cpu",
    "qnn": "QNNExecutionProvider",
}


def _register_winml_eps() -> list[str]:
    """Discover and register Windows ML execution providers.

    Walks the WinML EP catalog, calls ``ensure_ready()`` on each provider
    (downloads via Windows Update if needed), then registers the shared
    library with ORT GenAI.  Mirrors ``examples/python/winml.py`` from the
    onnxruntime-genai repo.
    """
    import traceback

    from windowsml import EpCatalog

    registered: list[str] = []
    with EpCatalog() as catalog:
        for provider in catalog.find_all_providers():
            provider.ensure_ready()
            if not provider.library_path:
                continue
            try:
                og.register_execution_provider_library(provider.name, provider.library_path)
                registered.append(provider.name)
            except Exception as exc:
                print(f"[winml] failed to register {provider.name}: {exc}")
                traceback.print_exc()
    return registered


def _build_og_config(model_dir: Path, ep: str) -> og.Config:
    """Create an ``og.Config``, registering WinML EPs when not on CPU."""
    if ep != "cpu":
        registered = _register_winml_eps()
        print(f"[winml] registered EPs: {registered}")

    config = og.Config(str(model_dir))
    config.clear_providers()
    if ep != "cpu":
        config.append_provider(_EP_NAME[ep])
    return config


def _wrap_chat_template(prompt: str) -> str:
    """Wrap *prompt* in the Qwen3 chat template (no thinking mode)."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


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
        choices=sorted(_EP_NAME),
        default="cpu",
        help="Execution provider (default: cpu).",
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
        help="Wrap --prompt in the Qwen3 chat template.",
    )
    p.add_argument(
        "--context-length",
        type=int,
        default=CONTEXT_LENGTH,
        help=(
            "Static KV cache length.  Must match the --max-cache-len used "
            "during the winml build and the genai_config.json context_length "
            "(default: %(default)s).  Do NOT lower this value."
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

    model_dir: Path = args.model_dir
    if not model_dir.exists():
        print(
            f"ERROR: model directory not found: {model_dir}\n"
            "Run export_qwen3_transformer_only.py --genai-bundle <DIR> first.",
            file=sys.stderr,
        )
        return 1

    config_file = model_dir / "genai_config.json"
    if not config_file.exists():
        print(
            f"ERROR: genai_config.json not found in {model_dir}\nThe bundle may be incomplete.",
            file=sys.stderr,
        )
        return 1

    if args.verbose:
        og.set_log_options(enabled=True, model_input_values=True, model_output_shapes=True)

    print(f"[load] ep={args.ep}  bundle={model_dir}")
    config = _build_og_config(model_dir, args.ep)
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()

    text = _wrap_chat_template(args.prompt) if args.chat else args.prompt
    input_tokens = tokenizer.encode(text)
    print(f"[tokens] prompt has {len(input_tokens)} tokens")

    params = og.GeneratorParams(model)
    # max_length must equal the static KV cache size so genai sizes the
    # total_sequence_length input and KV buffers correctly.
    params.set_search_options(
        max_length=args.context_length,
        do_sample=False,
    )

    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)

    print("[gen] ", end="", flush=True)
    t0 = time.monotonic()
    n = 0
    while not generator.is_done():
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        print(tokenizer_stream.decode(new_token), end="", flush=True)
        n += 1
        if n >= args.max_new:
            break

    dt = time.monotonic() - t0
    print(f"\n\n[done] {n} tokens in {dt:.1f}s  ({n / dt:.1f} tok/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
