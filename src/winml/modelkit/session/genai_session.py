# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""GenaiSession — onnxruntime-genai session for multi-model decoder pipelines.

Manages ``og.Model`` + ``og.Generator`` lifecycle for autoregressive text
generation.  Reuses :class:`WinMLEPRegistry` for EP discovery and registration
so EPs are downloaded / registered at most once per process.

Unlike :class:`WinMLSession` (which wraps ``ort.InferenceSession`` for
single-shot inference), ``GenaiSession`` drives a streaming token-by-token
generation loop.  The two classes are peers — neither inherits from the other.

Bundle directory layout expected by ``onnxruntime-genai``::

    <bundle_dir>/
        genai_config.json          ← required; controls pipeline & search
        ctx.onnx                   ← prefill transformer graph
        iter.onnx                  ← decode transformer graph
        embeddings.onnx            ← embedding lookup
        lm_head.onnx               ← logit projection
        tokenizer.json             ← HF tokenizer files
        tokenizer_config.json
        ...

Usage::

    # Context manager (recommended — auto-loads and unloads)
    with GenaiSession("out/qwen3_bundle", ep="qnn") as session:
        for token_str in session.generate_streaming("Hello, who are you?"):
            print(token_str, end="", flush=True)

    # Manual lifecycle
    session = GenaiSession("out/qwen3_bundle", ep="cpu")
    session.load()
    result = session.generate("What is a transformer?")
    session.unload()

Dependencies::

    pip install onnxruntime-genai-winml
    pip install "windowsml[with-ort]"   # registers QNN EP; also provides ORT
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .ep_registry import WinMLEPRegistry


if TYPE_CHECKING:
    from collections.abc import Iterator


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid EP short names.
# "mixed" = use genai_config.json as-is (embeddings/lm_head on CPU,
#           ctx/iter on the target accelerator).
# EP routing is driven entirely by per-stage session_options in the bundle's
# genai_config.json — GenaiSession never calls clear_providers/append_provider.
# ---------------------------------------------------------------------------
_VALID_EPS: frozenset[str] = frozenset({"cpu", "mixed", "qnn", "dml"})
# EPs that require WinML EP discovery + registration before og.Model() init.
_NEEDS_WINML_EPS: frozenset[str] = frozenset({"mixed", "qnn", "dml"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GenerationConfig:
    """Search / sampling parameters for a single generation call.

    All parameters are forwarded to ``og.GeneratorParams.set_search_options``.
    ``max_length`` is **not** configurable here — it is set to the bundle's
    ``context_length`` (read from ``genai_config.json``) because the static KV
    cache size is baked into the ONNX graphs at export time.

    Attributes:
        max_new_tokens: Soft cap on the number of new tokens to generate.
            Generation stops when the model signals EOS, when the KV buffer is
            exhausted (``context_length``), or when this limit is reached,
            whichever comes first.
        do_sample: Enable sampling (``True``) vs greedy (``False``).
        temperature: Sampling temperature.  Ignored when ``do_sample=False``.
        top_p: Nucleus sampling probability mass.  Ignored when
            ``do_sample=False``.
        top_k: Top-K sampling.  ``0`` disables the filter.  Ignored when
            ``do_sample=False``.
        repetition_penalty: Multiplicative penalty for repeated tokens
            (``1.0`` = no penalty).
    """

    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GenaiSessionError(Exception):
    """Base exception for GenaiSession."""


class GenaiNotInstalledError(GenaiSessionError):
    """``onnxruntime-genai`` (or ``onnxruntime-genai-winml``) is not installed."""


class GenaiLoadError(GenaiSessionError):
    """The bundle could not be loaded (bad config, EP unavailable, etc.)."""


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class GenaiSession:
    """ORT GenAI session for multi-model decoder-pipeline inference.

    Wraps ``og.Model`` + ``og.Generator`` to provide a clean generation API.

    The session is **stateless across calls**: each :meth:`generate_streaming`
    call creates a fresh ``og.Generator`` so KV state does not persist between
    prompts.  Thread-safety within a single session is not guaranteed.

    Args:
        bundle_dir: Path to the genai bundle directory.  Must contain
            ``genai_config.json`` and the ONNX files it references.
        ep: Execution provider short name (``"cpu"``, ``"qnn"``, ``"dml"``).
            Non-CPU EPs trigger WinML EP discovery and registration.
        context_length: Override for the static KV cache length.  When
            ``None`` (default), read from ``genai_config.json``.
            Must match the ``--max-cache-len`` used during the winml-cli build.
        verbose: Enable ``onnxruntime-genai`` native model I/O logging.
    """

    def __init__(
        self,
        bundle_dir: str | Path,
        ep: str = "cpu",
        *,
        context_length: int | None = None,
        verbose: bool = False,
    ) -> None:
        self._bundle_dir = Path(bundle_dir)
        self._ep = ep.lower()
        self._context_length_override = context_length
        self._verbose = verbose

        # Resolved at load() time.
        self._context_length: int | None = None

        # og.* handles — None until load() is called.
        self._model: object | None = None
        self._tokenizer: object | None = None

        if not self._bundle_dir.exists():
            raise FileNotFoundError(f"Bundle directory not found: {self._bundle_dir}")
        config_path = self._bundle_dir / "genai_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"genai_config.json not found in {self._bundle_dir}. "
                "Run export_qwen3_transformer_only.py --genai-bundle <DIR> first."
            )
        if self._ep not in _VALID_EPS:
            raise ValueError(f"Unknown EP {ep!r}. Supported: {sorted(_VALID_EPS)}")

        logger.info("GenaiSession initialized: bundle=%s ep=%s", self._bundle_dir, self._ep)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load ``og.Model`` and tokenizer from the bundle directory.

        Idempotent: calling ``load()`` on an already-loaded session is a no-op.

        Raises:
            GenaiNotInstalledError: ``onnxruntime_genai`` is not installed.
            GenaiLoadError: The model could not be loaded (EP error, bad config,
                missing ONNX files, …).
        """
        if self._model is not None:
            return

        og = self._import_og()

        # Register WinML EPs to ORT GenAI when the bundle may use a hardware EP.
        if self._ep in _NEEDS_WINML_EPS:
            self._register_eps(og)

        if self._verbose:
            og.set_log_options(enabled=True, model_input_values=True, model_output_shapes=True)

        try:
            config = og.Config(str(self._bundle_dir))
            # EP routing is driven entirely by genai_config.json (per-stage
            # session_options).  Do NOT call clear_providers/append_provider —
            # those only touch the top-level provider and cannot override
            # per-stage session_options already embedded in the pipeline config.
            self._model = og.Model(config)
            self._tokenizer = og.Tokenizer(self._model)
        except Exception as exc:
            self._model = None
            self._tokenizer = None
            raise GenaiLoadError(
                f"Failed to load genai bundle from {self._bundle_dir}: {exc}"
            ) from exc

        self._context_length = self._context_length_override or self._read_context_length()
        logger.info(
            "GenaiSession loaded: ep=%s context_length=%d",
            self._ep,
            self._context_length,
        )

    def unload(self) -> None:
        """Release ``og.Model`` and tokenizer handles.

        Safe to call on an unloaded session.
        """
        self._model = None
        self._tokenizer = None
        self._context_length = None
        logger.info("GenaiSession unloaded: bundle=%s", self._bundle_dir)

    def __enter__(self) -> GenaiSession:
        self.load()
        return self

    def __exit__(self, *_: object) -> None:
        self.unload()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str | list[int],
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate text and return the full response as a single string.

        This is a convenience wrapper around :meth:`generate_streaming`.

        Args:
            prompt: Input text (auto-encoded) or a pre-encoded token-ID list.
            config: Generation parameters.  Uses :class:`GenerationConfig`
                defaults when ``None``.

        Returns:
            The generated text (not including the prompt).
        """
        return "".join(self.generate_streaming(prompt, config))

    def generate_streaming(
        self,
        prompt: str | list[int],
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """Generate text token-by-token, yielding decoded token strings.

        The method auto-loads the session on the first call (lazy-load
        equivalent of :meth:`load`).

        Each yield is the decoded string for a single new token.  Callers
        typically ``print(token, end="", flush=True)`` to stream output.

        Args:
            prompt: Input text (auto-encoded via the bundle tokenizer) or a
                pre-encoded token-ID list.  Pass a pre-formatted string when
                chat templates or special tokens are needed — the session is
                not aware of any particular model's template format.
            config: Generation parameters.  Uses :class:`GenerationConfig`
                defaults when ``None``.

        Yields:
            Decoded string for each newly generated token.
        """
        self._ensure_loaded()
        og = self._import_og()
        cfg = config or GenerationConfig()

        tokens = (
            self._tokenizer.encode(prompt)  # type: ignore[union-attr]
            if isinstance(prompt, str)
            else prompt
        )

        params = og.GeneratorParams(self._model)
        params.set_search_options(
            max_length=self._context_length,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=cfg.repetition_penalty,
        )

        generator = og.Generator(self._model, params)
        generator.append_tokens(tokens)

        stream = self._tokenizer.create_stream()  # type: ignore[union-attr]
        n = 0
        try:
            while not generator.is_done():
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]
                yield stream.decode(new_token)
                n += 1
                if n >= cfg.max_new_tokens:
                    break
        finally:
            # Explicit deletion releases the KV cache buffer held by the generator.
            del generator

    # ------------------------------------------------------------------
    # Chat-template helpers
    # ------------------------------------------------------------------

    @staticmethod
    def apply_chatml_template(
        prompt: str,
        system: str | None = None,
    ) -> str:
        r"""Wrap *prompt* in the ChatML format used by Qwen2/3, Yi, Mistral, etc.

        Produces::

            <|im_start|>system
            {system}<|im_end|>
            <|im_start|>user
            {prompt}<|im_end|>
            <|im_start|>assistant

        The trailing ``<|im_start|>assistant\\n`` primes the model to respond
        as the assistant role with no leading newline in its output.

        Args:
            prompt: The user turn text.
            system: Optional system prompt.  When ``None`` no system turn is
                prepended.

        Returns:
            Formatted string ready to pass to :meth:`generate` /
            :meth:`generate_streaming`.
        """
        parts: list[str] = []
        if system is not None:
            parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
        return "".join(parts)

    # ------------------------------------------------------------------
    # Tokenizer helpers
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode *text* to a list of token IDs using the bundle tokenizer."""
        self._ensure_loaded()
        return self._tokenizer.encode(text).tolist()  # type: ignore[union-attr]

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token IDs to a string."""
        self._ensure_loaded()
        return self._tokenizer.decode(tokens)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """``True`` if the model is loaded and ready for generation."""
        return self._model is not None

    @property
    def bundle_dir(self) -> Path:
        """Path to the genai bundle directory."""
        return self._bundle_dir

    @property
    def ep(self) -> str:
        """Execution provider short name (as passed to ``__init__``)."""
        return self._ep

    @property
    def context_length(self) -> int | None:
        """Static KV cache length, populated after :meth:`load`."""
        return self._context_length

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    @staticmethod
    def _import_og() -> object:
        """Import and return the ``onnxruntime_genai`` module.

        Raises:
            GenaiNotInstalledError: Package not found.
        """
        try:
            import onnxruntime_genai as og

            return og
        except ImportError as exc:
            raise GenaiNotInstalledError(
                "onnxruntime_genai is not installed. "
                "Install it with: pip install onnxruntime-genai-winml"
            ) from exc

    def _register_eps(self, og: object) -> None:
        """Register WinML EPs with ORT GenAI (idempotent, best-effort)."""
        try:
            registry = WinMLEPRegistry.get_instance()
            if registry.winml_available:
                result = registry.register_execution_providers(ort_genai=True)
                registered = result.get("onnxruntime_genai", [])
                logger.info("WinML EPs registered for ORT GenAI: %s", registered)
        except Exception as exc:
            logger.warning("WinML EP registration skipped: %s", exc)

    def _read_context_length(self) -> int:
        """Read ``model.context_length`` from ``genai_config.json``."""
        cfg = json.loads((self._bundle_dir / "genai_config.json").read_text(encoding="utf-8"))
        return int(cfg["model"]["context_length"])


__all__ = [
    "GenaiLoadError",
    "GenaiNotInstalledError",
    "GenaiSession",
    "GenaiSessionError",
    "GenerationConfig",
]
