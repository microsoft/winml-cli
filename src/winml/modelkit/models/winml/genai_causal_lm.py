# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Causal-LM inference over an onnxruntime-genai bundle."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...session import GenaiSession, GenaiSessionError


if TYPE_CHECKING:
    from pathlib import Path

    from ...session import GenerationConfig
    from ...utils.constants import EPNameOrAlias

logger = logging.getLogger(__name__)

# One-hot magnitudes used to steer the next input to the given token, applied
# only after the position's logits have been read.
_FORCE_HIGH = 1e9
_FORCE_LOW = -1e9


@dataclass
class CausalLMOutput:
    """Output of :meth:`WinMLGenaiCausalLM.forward`, mirroring HF's ``.logits``."""

    logits: np.ndarray


class WinMLGenaiCausalLM:
    """Causal-LM inference over a genai bundle via :class:`GenaiSession`.

    Constructed with an already-resolved ``ep`` / ``device`` that pass straight
    to the session, so inference runs on the same runtime and EP as
    ``winml perf --runtime winml-genai``.

    Args:
        bundle_dir: Path to the genai bundle directory.
        ep: EP override, or ``None`` to respect the bundle's ``genai_config.json``.
        device: Concrete device (``npu`` / ``gpu`` / ``cpu``) the forced EP
            targets; only meaningful when *ep* is set.
        context_length: Override for the static KV-cache length; ``None`` uses
            the bundle's configured length.
        compile: Compile to EPContext ONNX before loading, reusing a cached
            ``_compiled/`` when present. Use ``False`` to load an already-compiled
            bundle as-is.
        compile_timeout: Seconds allowed for compilation.
        verbose: Forward verbose logging to the session.
    """

    def __init__(
        self,
        bundle_dir: str | Path,
        ep: EPNameOrAlias | None = None,
        *,
        device: str | None = None,
        context_length: int | None = None,
        compile: bool = True,
        compile_timeout: int = 300,
        verbose: bool = False,
    ) -> None:
        self._session = GenaiSession(
            bundle_dir,
            ep,
            device=device,
            context_length=context_length,
            compile=compile,
            compile_timeout=compile_timeout,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode *text* to token IDs using the bundle tokenizer."""
        return self._session.encode(text)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(self, input_ids: list[int]) -> CausalLMOutput:
        """Run a forward pass over *input_ids*, matching HF's ``forward``.

        Returns the next-token logits at each input position, so a caller can
        score the sequence exactly as with an HF causal LM.

        The genai runtime only exposes last-position logits, so the sequence is
        replayed token by token through the decode loop, pinning each next input
        to the given token (reusing the KV cache, cost O(len)). The final
        position, whose prediction lies beyond the input, is not computed.

        Args:
            input_ids: Token IDs; at least 2 tokens and at most the bundle's
                :attr:`context_length`.

        Returns:
            :class:`CausalLMOutput` whose ``logits`` has shape
            ``(1, seq_len - 1, vocab)``; row ``i`` predicts ``input_ids[i + 1]``.

        Raises:
            ValueError: *input_ids* is too short or exceeds the context length.
        """
        self._session._ensure_loaded()

        n = len(input_ids)
        if n < 2:
            raise ValueError(f"forward needs at least 2 tokens; got {n}.")
        max_length = self._session.context_length
        if max_length is not None and n > max_length:
            raise ValueError(
                f"Sequence of {n} tokens exceeds the bundle context length {max_length}."
            )

        import onnxruntime_genai as og

        # set_logits is only on the low-level generator, not the session's
        # generate path, so reach the session's loaded model directly.
        model = self._session._model

        params = og.GeneratorParams(model)
        # Fixed max_length is required by the compiled pipeline; do_sample=False
        # keeps the decode deterministic.
        params.set_search_options(max_length=max_length, do_sample=False)
        gen = og.Generator(model, params)
        gen.append_tokens([input_ids[0]])

        rows: list[np.ndarray] = []
        for i in range(1, n):
            step = np.asarray(gen.get_logits())[0, -1, :].astype(np.float32)
            rows.append(step)
            # The last read predicts input_ids[n-1]; nothing after it is scored,
            # so skip the otherwise-wasted decode step.
            if i < n - 1:
                target = input_ids[i]
                # Pin the next input to the given token so the pass follows
                # input_ids regardless of the model's own argmax.
                if int(step.argmax()) != target:
                    forced = np.full((1, 1, step.shape[0]), _FORCE_LOW, dtype=np.float32)
                    forced[0, 0, target] = _FORCE_HIGH
                    gen.set_logits(forced)
                gen.generate_next_token()

        return CausalLMOutput(logits=np.stack(rows)[None, ...])

    __call__ = forward

    def generate(
        self,
        prompt: str | list[int],
        config: GenerationConfig | None = None,
        *,
        apply_chat_template: bool = True,
    ) -> str:
        """Generate text from *prompt*, delegating to the underlying session.

        A string *prompt* is wrapped with the bundle's chat template by default
        (matching ``winml perf``); many chat models degenerate without it. Set
        *apply_chat_template* to ``False``, or pass pre-encoded token IDs, to
        skip the wrapping.
        """
        if apply_chat_template and isinstance(prompt, str):
            try:
                prompt = self._session.apply_chat_template(prompt)
            except GenaiSessionError:
                pass  # Bundle ships no chat template; use the raw prompt.
        return self._session.generate(prompt, config)


class HFCausalLM:
    """Adapt a HuggingFace causal LM to the :class:`WinMLGenaiCausalLM` contract.

    Exposes the identical ``encode`` / ``forward`` pair, so a scorer (e.g. the
    text-generation perplexity evaluator) can run an fp32 PyTorch baseline
    through the exact same model-agnostic path as a genai bundle, with no
    branching on the concrete model type.

    ``transformers`` and ``torch`` are imported lazily so importing this module
    never requires them; only constructing / calling the adapter does.

    Args:
        model_id: HuggingFace model ID or local path.
        device: A ``torch.device`` (or device string) to place the model on.
    """

    def __init__(self, model_id: str, device) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
        self._device = device

    def encode(self, text: str) -> list[int]:
        """Encode *text* to token IDs.

        ``add_special_tokens=False`` keeps the raw stream, matching the genai
        bundle tokenizer used by :class:`WinMLGenaiCausalLM`.
        """
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    def forward(self, input_ids: list[int]) -> CausalLMOutput:
        """Run a forward pass over *input_ids*, matching HF's ``forward``.

        Returns:
            :class:`CausalLMOutput` whose ``logits`` has shape
            ``(1, len(input_ids) - 1, vocab)``; row ``i`` predicts
            ``input_ids[i + 1]``. The trailing row (predicting past the input)
            is dropped to match the genai contract.
        """
        import torch

        with torch.no_grad():
            logits = self._model(input_ids=torch.tensor([input_ids], device=self._device)).logits
        trimmed = logits[0, :-1, :].to(torch.float32).cpu().numpy()
        return CausalLMOutput(logits=trimmed[None, ...])

    __call__ = forward


__all__ = ["CausalLMOutput", "HFCausalLM", "WinMLGenaiCausalLM"]
