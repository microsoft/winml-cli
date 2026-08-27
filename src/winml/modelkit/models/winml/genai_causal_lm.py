# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Causal-LM inference over an onnxruntime-genai bundle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...session import GenaiSession, GenaiSessionError


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ...session import GenerationConfig
    from ...utils.constants import EPNameOrAlias

# One-hot magnitudes used to steer the next input to the given token, applied
# only after the position's logits have been read.
_FORCE_HIGH = 1e9
_FORCE_LOW = -1e9


class WinMLGenaiCausalLM:
    """Causal-LM inference over a genai bundle via :class:`GenaiSession`.

    Constructed with an already-resolved ``ep`` / ``device`` that pass straight
    to the session, so inference runs on the same runtime and EP as
    ``winml perf --runtime ort-genai``.

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

    def forward(self, input_ids: list[int]) -> Iterator[np.ndarray]:
        """Yield next-token logits at each input position, one position at a time.

        The genai runtime only exposes last-position logits, so the sequence is
        replayed token by token through the decode loop, pinning each next input
        to the given token (reusing the KV cache, cost O(len)). The final
        position, whose prediction lies beyond the input, is not computed.

        Args:
            input_ids: Token IDs; at least 2 tokens and at most the bundle's
                :attr:`context_length`.

        Yields:
            A ``(vocab,)`` float32 array for each position ``i`` in
            ``range(seq_len - 1)``; the vector predicts ``input_ids[i + 1]``.

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
        # keeps the decode deterministic. The remaining options neutralize any
        # search constraints inherited from the bundle's genai_config.json:
        # ORT GenAI applies them inside generate_next_token() *after* set_logits,
        # so a live min_length / no_repeat_ngram_size / repetition_penalty could
        # re-mask the forced target (e.g. a target EOS below min_length), break
        # teacher forcing, and condition every later logit on the wrong prefix.
        params.set_search_options(
            max_length=max_length,
            do_sample=False,
            min_length=0,
            no_repeat_ngram_size=0,
            repetition_penalty=1.0,
            num_beams=1,
        )
        gen = og.Generator(model, params)
        gen.append_tokens([input_ids[0]])

        for i in range(1, n):
            step = np.asarray(gen.get_logits())[0, -1, :].astype(np.float32)
            yield step
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

    def __init__(
        self,
        model_id: str,
        device: Any,
        *,
        trust_remote_code: bool = False,
        torch_dtype: Any = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs: dict[str, Any] = {}
        if trust_remote_code:
            load_kwargs["trust_remote_code"] = True
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self._model = model.to(device).eval()
        self._device = device

    @classmethod
    def from_model(
        cls,
        model_id: str,
        model: Any,
        device: Any,
        *,
        trust_remote_code: bool = False,
    ) -> HFCausalLM:
        """Adapt an existing caller-owned causal LM without moving or reloading it."""
        from transformers import AutoTokenizer

        load_kwargs = {"trust_remote_code": True} if trust_remote_code else {}
        adapter = cls.__new__(cls)
        adapter._tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
        adapter._model = model.eval()
        adapter._device = device
        return adapter

    def encode(self, text: str) -> list[int]:
        """Encode *text* to token IDs.

        ``add_special_tokens=False`` keeps the raw stream, matching the genai
        bundle tokenizer used by :class:`WinMLGenaiCausalLM`.
        """
        ids: list[int] = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids

    def forward(self, input_ids: list[int]) -> Iterator[np.ndarray]:
        """Yield next-token logits at each input position, one position at a time.

        Yields:
            A ``(vocab,)`` float32 array for each position ``i`` in
            ``range(len(input_ids) - 1)``; the vector predicts
            ``input_ids[i + 1]``. The trailing position (predicting past the
            input) is dropped.
        """
        import torch

        with torch.no_grad():
            logits = self._model(input_ids=torch.tensor([input_ids], device=self._device)).logits
        trimmed = logits[0, :-1, :].to(torch.float32).cpu().numpy()
        yield from trimmed

    __call__ = forward


__all__ = ["HFCausalLM", "WinMLGenaiCausalLM"]
