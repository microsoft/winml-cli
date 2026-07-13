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
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .ep_registry import WinMLEPRegistry


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level compilation worker.
#
# Must be at module scope so ``multiprocessing`` (spawn start method on Windows)
# can pickle it as the subprocess target.  The body delegates to the shared
# compiler component (:func:`compile_onnx`) instead of re-implementing the
# ONNX Runtime ``ModelCompiler`` wiring — the compiler owns all EP-specific
# EPContext logic (SessionOptions, provider registration, ``.bin`` handling).
# ---------------------------------------------------------------------------


def _compile_stage_worker(src: str, dst: str, ep_alias: str, provider_options: dict) -> None:
    """Compile a single pipeline stage ONNX to EPContext via the shared compiler.

    Executed in a subprocess by :meth:`GenaiSession._compile_stage` so a hang
    or crash can be bounded by a timeout in the parent process.

    The target execution provider is resolved generically from *ep_alias* via
    :meth:`WinMLCompileConfig.for_provider`, so any EPContext-capable
    accelerator (QNN, OpenVINO, VitisAI, …) is compiled with its own EP config.
    There is no provider-specific branching here.

    Args:
        src: Absolute path to the source ONNX file.
        dst: Absolute path where the compiled EPContext ONNX should be written.
        ep_alias: EP short name for this stage (e.g. ``"qnn"``, ``"openvino"``),
            taken from the stage's ``provider_options`` key in
            ``genai_config.json``.
        provider_options: EP provider options taken verbatim from the bundle's
            ``genai_config.json`` (e.g. ``backend_path``, ``htp_performance_mode``,
            ``soc_model`` for QNN).  Merged onto the resolved EP config.

    Raises:
        RuntimeError: If *ep_alias* is not an EPContext-capable EP, or if the
            compiler reports the compilation was unsuccessful.
    """
    from ..compiler import WinMLCompileConfig, compile_onnx

    # for_provider normalizes arbitrary EP aliases and returns None for ones it
    # does not recognize; ep_alias is a runtime str from genai_config.json.
    config = WinMLCompileConfig.for_provider(ep_alias)  # type: ignore[arg-type]
    if config is None:
        raise RuntimeError(f"EP {ep_alias!r} does not support EPContext pre-compilation")
    config.ep_config.provider_options.update(provider_options)
    result = compile_onnx(src, dst, config)
    if not result.success:
        raise RuntimeError(f"Compilation failed: {result.errors}")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GenerationConfig:
    """Search / sampling parameters for a single generation call.

    All parameters are forwarded to ``og.GeneratorParams.set_search_options``.
    ``max_length`` is computed as ``len(prompt) + max_new_tokens``, capped at
    the bundle's ``context_length``, so only the needed KV cache is allocated.

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


@dataclass
class GenerationTiming:
    """Per-generation timing captured at onnxruntime-genai call boundaries.

    onnxruntime-genai exposes no native performance-metrics API (unlike
    OpenVINO GenAI's ``perf_metrics``), so these spans are wall-clock
    measurements taken immediately around the library calls.  The segmentation
    mirrors onnxruntime-genai's official ``benchmark_e2e.py``:

    * ``prefill_s`` — cost of ``Generator.append_tokens`` (the prompt forward
      pass, a.k.a. prompt processing).
    * ``first_token_s`` — cost of the first ``Generator.generate_next_token``.
    * ``decode_s`` — cost of each subsequent ``generate_next_token`` (one entry
      per generated token after the first).

    Detokenization is intentionally excluded — only model-compute boundaries
    are timed, so the numbers reflect the pipeline rather than tokenizer/string
    overhead.

    Attributes:
        input_tokens: Number of prompt tokens fed to ``append_tokens``.
        generated_tokens: Number of tokens produced (including the first).
        prefill_s: Prompt-processing time in seconds.
        first_token_s: Time to produce the first token, in seconds.
        decode_s: Per-token times for the steady-state decode phase (tokens
            after the first), in seconds.
        response_text: Decoded model output text (empty string when not
            captured).
    """

    input_tokens: int = 0
    generated_tokens: int = 0
    prefill_s: float = 0.0
    first_token_s: float = 0.0
    decode_s: list[float] = field(default_factory=list)
    response_text: str = ""

    @property
    def ttft_s(self) -> float:
        """Time to first token = prefill + first decode step, in seconds."""
        return self.prefill_s + self.first_token_s

    @property
    def tpot_s(self) -> float:
        """Mean time per output token over the steady-state decode phase.

        Averages :attr:`decode_s` (tokens after the first).  Returns ``0.0``
        when fewer than two tokens were generated.
        """
        return sum(self.decode_s) / len(self.decode_s) if self.decode_s else 0.0

    @property
    def decode_tokens_per_sec(self) -> float:
        """Steady-state decode throughput (tokens/sec), excluding the first token.

        Returns ``0.0`` when fewer than two tokens were generated.
        """
        total = sum(self.decode_s)
        return len(self.decode_s) / total if total > 0 else 0.0

    @property
    def total_s(self) -> float:
        """Total model-compute time = prefill + first token + all decode steps."""
        return self.prefill_s + self.first_token_s + sum(self.decode_s)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GenaiSessionError(Exception):
    """Base exception for GenaiSession."""


class GenaiNotInstalledError(GenaiSessionError):
    """``onnxruntime_genai`` could not be imported.

    Raised when the package is not installed, or when it is installed but its
    native extension fails to load (e.g. missing runtime dependencies).
    """


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
        ep: Execution provider short name (e.g. ``"cpu"``, ``"qnn"``, ``"dml"``,
            ``"mixed"``), recorded for reporting.  Whether WinML EPs are
            registered and whether stages are pre-compiled is decided from the
            bundle's ``genai_config.json`` (per-stage ``session_options``), not
            from this argument.  Overriding the config via ``ep``/``device`` is
            tracked in #1025.
        context_length: Override for the static KV cache length.  When
            ``None`` (default), read from ``genai_config.json``.
            Must match the ``--max-cache-len`` used during the winml-cli build.
        verbose: Enable ``onnxruntime-genai`` native model I/O logging.
        compile: Pre-compile EPContext-capable pipeline stages (e.g. QNN) to
            EPContext ONNX on first run (inside ``bundle_dir/_compiled/``).
            Subsequent calls reuse the cached EPContext files, eliminating
            per-run JIT overhead.  Only stages that ``genai_config.json`` routes
            to an EPContext-capable EP and that can be compiled without hanging
            are attempted; stages that fail compilation fall back to the
            original ONNX.  A CPU-only bundle is a no-op.
        compile_timeout: Maximum seconds to wait for each stage to compile
            before killing the subprocess and falling back to the original
            ONNX.  Defaults to 300 seconds (5 minutes).
    """

    # Sub-directory within the bundle that holds pre-compiled EPContext ONNX files.
    _COMPILED_SUBDIR: str = "_compiled"

    # Standalone chat-template sidecar written next to genai_config.json (the
    # conventional onnxruntime-genai / Hugging Face filename).  Read to format
    # prompts with the model's own template; absent for bundles that ship none.
    _CHAT_TEMPLATE_FILE: str = "chat_template.jinja"

    def __init__(
        self,
        bundle_dir: str | Path,
        ep: str = "cpu",
        *,
        context_length: int | None = None,
        verbose: bool = False,
        compile: bool = False,
        compile_timeout: int = 300,
    ) -> None:
        self._bundle_dir = Path(bundle_dir)
        self._ep = ep.lower()
        self._context_length_override = context_length
        self._verbose = verbose
        self._compile = compile
        self._compile_timeout = compile_timeout

        # Resolved at load() time.
        self._context_length: int | None = None

        # og.* handles — None until load() is called.
        self._model: Any = None
        self._tokenizer: Any = None

        if not self._bundle_dir.exists():
            raise FileNotFoundError(f"Bundle directory not found: {self._bundle_dir}")
        config_path = self._bundle_dir / "genai_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"genai_config.json not found in {self._bundle_dir}. "
                "Ensure the bundle was created with a winml-cli export command."
            )

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

        cfg = self._read_genai_config()

        # Register WinML EPs only when the bundle routes at least one stage to a
        # hardware EP.  This decision comes from genai_config.json, not the
        # ``ep`` argument (see #1025 for the planned override path).
        hw_ep = self._bundle_uses_hardware_ep(cfg)
        logger.info("Hardware EP detected in genai_config.json: %s", hw_ep)
        if hw_ep is not None:
            self._register_eps()

        if self._verbose:
            og.set_log_options(enabled=True, model_input_values=True, model_output_shapes=True)

        # Pre-compile EPContext-capable stages when requested.  The bundle config
        # decides which stages are compilable; a bundle with none is a no-op.
        load_dir = self._bundle_dir
        if self._compile:
            load_dir = self._prepare_compiled_bundle()

        try:
            config = og.Config(str(load_dir))
            # EP routing is driven entirely by genai_config.json (per-stage
            # session_options).  Do NOT call clear_providers/append_provider —
            # those only touch the top-level provider and cannot override
            # per-stage session_options already embedded in the pipeline config.
            self._model = og.Model(config)
            self._tokenizer = og.Tokenizer(self._model)
        except Exception as exc:
            self._model = None
            self._tokenizer = None
            raise GenaiLoadError(f"Failed to load genai bundle from {load_dir}: {exc}") from exc

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
        cfg = config or GenerationConfig()
        tokens = self._encode_prompt(prompt)
        generator = self._new_generator(cfg, len(tokens))
        generator.append_tokens(tokens)

        stream = self._tokenizer.create_stream()
        n = 0
        while not generator.is_done():
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            yield stream.decode(new_token)
            n += 1
            if n >= cfg.max_new_tokens:
                break

    def generate_timed(
        self,
        prompt: str | list[int],
        config: GenerationConfig | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> GenerationTiming:
        """Run one generation, timing each onnxruntime-genai operation boundary.

        Unlike :meth:`generate_streaming` (which yields decoded text), this
        drives the same ``og.Generator`` loop but records wall-clock spans
        around the library calls and returns a :class:`GenerationTiming`.  It
        does **not** decode tokens — only model-compute boundaries are timed,
        so tokenizer / string overhead is excluded.

        The segmentation matches onnxruntime-genai's official
        ``benchmark_e2e.py``: ``append_tokens`` is the prefill (prompt
        processing) and each ``generate_next_token`` is one decode step.
        onnxruntime-genai has no native perf-metrics API, so the timing is
        taken externally with *clock* (injectable for deterministic tests).

        Args:
            prompt: Input text (auto-encoded via the bundle tokenizer) or a
                pre-encoded token-ID list.
            config: Generation parameters.  Uses :class:`GenerationConfig`
                defaults when ``None``.
            clock: Monotonic clock returning seconds.  Defaults to
                :func:`time.perf_counter`.

        Returns:
            A :class:`GenerationTiming` with per-boundary spans and token counts.

        Raises:
            GenaiSessionError: The bundle produced no tokens (empty output).
        """
        self._ensure_loaded()
        cfg = config or GenerationConfig()
        tokens = self._encode_prompt(prompt)
        generator = self._new_generator(cfg, len(tokens))

        # marks[0]  = before prefill
        # marks[1]  = after prefill (append_tokens)
        # marks[2+k]= after the (k+1)-th generated token
        marks: list[float] = []
        generated = 0
        marks.append(clock())
        generator.append_tokens(tokens)
        marks.append(clock())
        while not generator.is_done():
            generator.generate_next_token()
            marks.append(clock())
            generated += 1
            if generated >= cfg.max_new_tokens:
                break

        if generated == 0:
            raise GenaiSessionError("genai: generation produced no tokens (empty bundle output?)")

        # Fetch all output tokens *after* the timing loop so that
        # get_sequence() (which may trigger host/device copies on hardware EPs)
        # does not pollute per-token decode measurements.
        full_sequence = generator.get_sequence(0)
        output_token_ids = list(full_sequence[len(tokens) :])

        timing = GenerationTiming(
            input_tokens=len(tokens),
            generated_tokens=generated,
            prefill_s=marks[1] - marks[0],
            first_token_s=marks[2] - marks[1],
            decode_s=[marks[i + 1] - marks[i] for i in range(2, 1 + generated)],
            response_text=str(self._tokenizer.decode(output_token_ids)),
        )
        logger.info(
            "generate_timed: input_tokens=%d generated_tokens=%d "
            "prefill=%.3fs ttft=%.3fs tpot=%.3fs decode=%.1f tok/s total=%.3fs",
            timing.input_tokens,
            timing.generated_tokens,
            timing.prefill_s,
            timing.ttft_s,
            timing.tpot_s,
            timing.decode_tokens_per_sec,
            timing.total_s,
        )
        return timing

    # ------------------------------------------------------------------
    # Chat-template helpers
    # ------------------------------------------------------------------

    def apply_chat_template(
        self,
        prompt: str,
        *,
        system: str | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """Format *prompt* with the bundle's own chat template.

        Model-driven: the wrapping comes from the chat template shipped inside
        the bundle (rendered via ``og.Tokenizer.apply_chat_template``), so it
        matches whatever format the model was trained with — ChatML, Llama
        ``[INST]``, Phi ``<|user|>`` etc.  No chat format is hardcoded here.

        The template text is read from the standalone ``chat_template.jinja``
        sidecar when present; otherwise onnxruntime-genai falls back to any
        template embedded in the tokenizer config.

        Args:
            prompt: The user-turn text.
            system: Optional system-turn text prepended before the user turn.
            add_generation_prompt: Append the model's assistant generation
                prefix so it continues in the assistant role (matches the
                Hugging Face ``add_generation_prompt`` semantics).

        Returns:
            The templated prompt string, ready to pass to :meth:`generate` /
            :meth:`generate_streaming` / :meth:`generate_timed`.

        Raises:
            GenaiSessionError: The bundle ships no usable chat template, or the
                template could not be rendered.
        """
        self._ensure_loaded()
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {"add_generation_prompt": add_generation_prompt}
        template_str = self._read_bundle_chat_template()
        if template_str is not None:
            kwargs["template_str"] = template_str

        try:
            rendered = self._tokenizer.apply_chat_template(json.dumps(messages), **kwargs)
        except Exception as exc:
            raise GenaiSessionError(
                f"Could not apply a chat template for bundle {self._bundle_dir}: {exc}. "
                "The bundle may ship no chat template; pass a pre-formatted prompt instead."
            ) from exc
        return str(rendered)

    def _read_bundle_chat_template(self) -> str | None:
        """Return the bundle's standalone chat-template text, or ``None``.

        Reads the conventional ``chat_template.jinja`` sidecar next to
        ``genai_config.json``.  Returns ``None`` when absent so callers can let
        onnxruntime-genai fall back to a template embedded in the tokenizer.
        """
        path = self._bundle_dir / self._CHAT_TEMPLATE_FILE
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Tokenizer helpers
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode *text* to a list of token IDs using the bundle tokenizer."""
        self._ensure_loaded()
        return list(self._tokenizer.encode(text).tolist())

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token IDs to a string."""
        self._ensure_loaded()
        return str(self._tokenizer.decode(tokens))

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

    def _encode_prompt(self, prompt: str | list[int]) -> list[int]:
        """Return prompt token IDs, encoding via the bundle tokenizer if needed."""
        if isinstance(prompt, str):
            return list(self._tokenizer.encode(prompt).tolist())
        return prompt

    def _new_generator(self, cfg: GenerationConfig, prompt_len: int) -> Any:
        """Build an ``og.Generator`` with search options from *cfg*.

        ``max_length`` is set to ``prompt_len + cfg.max_new_tokens``, capped at
        the bundle's ``context_length``.  This avoids pre-allocating KV cache
        for the full context window (which can be 128K+ for DML bundles) when
        only a small generation is requested.

        The prompt is **not** appended — callers decide whether to time
        ``append_tokens`` separately (see :meth:`generate_timed`).
        """
        og = self._import_og()
        assert self._context_length is not None, "_new_generator called before load()"
        max_length = min(prompt_len + cfg.max_new_tokens, self._context_length)
        params = og.GeneratorParams(self._model)
        params.set_search_options(
            max_length=max_length,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=cfg.repetition_penalty,
        )
        return og.Generator(self._model, params)

    def _prepare_compiled_bundle(self) -> Path:
        """Create (or reuse) a *compiled* bundle directory.

        Reads ``genai_config.json`` and finds pipeline stages whose execution
        provider supports EPContext pre-compilation (resolved generically via
        :meth:`WinMLCompileConfig.for_provider` — QNN, OpenVINO, VitisAI, …),
        then tries to compile their ONNX to EPContext format through the shared
        compiler.  Stages on EPs that do not emit EPContext (CPU, DML, …) are
        left untouched and load via JIT.

        The compiled bundle is stored under ``bundle_dir/_compiled/``.  A cached
        EPContext file is reused only when it is newer than the source graph and
        its external-weights sidecar *and* was built with the same provider
        options (see :meth:`_epcontext_is_fresh`); otherwise it is recompiled.

        Returns:
            Path to the compiled bundle directory (may equal ``bundle_dir``
            if no compilable stages were found, or if all compilations failed).
        """
        from ..onnx import is_compiled_onnx

        compiled_dir = self._bundle_dir / self._COMPILED_SUBDIR
        cfg = self._read_genai_config()

        # Collect pipeline stages whose EP supports EPContext pre-compilation.
        # genai_config pipeline entries: [{"context": {...}}, {"iterator": {...}}, ...]
        # provider_options format: [{"<ep_alias>": {...}}]
        pipeline_list: list = cfg.get("model", {}).get("decoder", {}).get("pipeline", [])
        # [(stage_key, onnx_filename, ep_alias, ep_opts), ...]
        compilable_stages: list[tuple[str, str, str, dict]] = []
        for stage_entry in pipeline_list:
            if not isinstance(stage_entry, dict):
                continue
            for stage_key, stage_cfg in stage_entry.items():
                if not isinstance(stage_cfg, dict):
                    continue
                so = stage_cfg.get("session_options", {})
                ep_alias, ep_opts = self._resolve_stage_ep(so.get("provider_options", []))
                if ep_alias is not None:
                    onnx_filename = stage_cfg.get("filename", f"{stage_key}.onnx")
                    compilable_stages.append((stage_key, onnx_filename, ep_alias, ep_opts))

        if not compilable_stages:
            logger.info(
                "No EPContext-capable stages found in genai_config.json; skipping pre-compilation"
            )
            return self._bundle_dir

        compiled_dir.mkdir(exist_ok=True)
        modified_cfg = self._read_genai_config()
        any_compiled = False
        # Original stage ONNX filenames that were replaced by a compiled
        # EPContext artifact (``{stage_key}_ctx.onnx``) and so must NOT be
        # mirrored into compiled_dir.  Stages that fall back to (or already are)
        # their original ONNX are deliberately left out so that
        # :meth:`_mirror_non_onnx_files` links them (and their weights sidecars)
        # in — ort-genai resolves stage filenames relative to compiled_dir, so
        # the file must physically live there.
        compiled_stage_filenames: set[str] = set()

        for stage_key, onnx_filename, ep_alias, ep_opts in compilable_stages:
            src_onnx = self._bundle_dir / onnx_filename
            ctx_onnx = compiled_dir / f"{stage_key}_ctx.onnx"

            # Skip recompilation only when the cache is genuinely up-to-date.
            if self._epcontext_is_fresh(src_onnx, ctx_onnx, ep_opts):
                logger.info("Stage %r: reusing cached EPContext %s", stage_key, ctx_onnx.name)
                # Use just the filename — genai_config.json lives in compiled_dir,
                # so ort-genai resolves filenames relative to compiled_dir.
                self._patch_stage_filename(modified_cfg, stage_key, ctx_onnx.name)
                compiled_stage_filenames.add(onnx_filename)
                any_compiled = True
                continue

            # A stage whose source ONNX is already an EPContext (pre-compiled)
            # model needs no recompilation; reference the original file by its
            # bundle-relative filename and let :meth:`_mirror_non_onnx_files`
            # link it (plus any weights sidecar) into compiled_dir.  A parse
            # failure is treated as "not compiled" so a malformed source falls
            # through to the normal compile path, which has its own error
            # handling.
            already_compiled = False
            if src_onnx.exists():
                try:
                    already_compiled = is_compiled_onnx(src_onnx)
                except (ValueError, OSError):
                    already_compiled = False
            if already_compiled:
                logger.info(
                    "Stage %r: source is already an EPContext model; using as-is",
                    stage_key,
                )
                self._patch_stage_filename(modified_cfg, stage_key, onnx_filename)
                continue

            # Attempt compilation.
            success = self._compile_stage(src_onnx, ctx_onnx, stage_key, ep_alias, ep_opts)
            if success:
                self._write_compile_marker(ctx_onnx, ep_alias, ep_opts)
                self._patch_stage_filename(modified_cfg, stage_key, ctx_onnx.name)
                compiled_stage_filenames.add(onnx_filename)
                any_compiled = True
            else:
                logger.warning(
                    "Stage %r: compilation failed; using original ONNX (JIT fallback)", stage_key
                )
                # Fall back to the original ONNX by its bundle-relative filename.
                # ort-genai resolves stage filenames relative to compiled_dir, so
                # the source ONNX (+ its weights sidecar) is mirrored in by
                # :meth:`_mirror_non_onnx_files` below; patching an absolute path
                # would be wrongly joined onto compiled_dir into a broken path.
                self._patch_stage_filename(modified_cfg, stage_key, onnx_filename)

        if not any_compiled:
            return self._bundle_dir

        # Write the modified genai_config into the compiled sub-directory.
        # ONNX filenames are relative to compiled_dir; ort-genai resolves them
        # from the directory it loads og.Config from.
        compiled_config = compiled_dir / "genai_config.json"
        compiled_config.write_text(
            json.dumps(modified_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # Skip only the stages replaced by a compiled EPContext artifact (which
        # are referenced by their new ``{stage}_ctx.onnx`` path).  Every other
        # ONNX file (embeddings, lm_head, stages on non-EPContext EPs, and any
        # fallback / already-compiled stage that kept its original ONNX) must be
        # symlinked into compiled_dir so ort-genai can find it by filename.
        self._mirror_non_onnx_files(compiled_dir, skip_filenames=compiled_stage_filenames)

        logger.info("Compiled bundle prepared at %s", compiled_dir)
        return compiled_dir

    @staticmethod
    def _resolve_stage_ep(provider_options: list) -> tuple[str | None, dict]:
        """Resolve a stage's EPContext-capable EP from its ``provider_options``.

        A genai_config ``provider_options`` is a list of single-key mappings
        ``[{ep_alias: {opts...}}]``.  Returns ``(ep_alias, opts)`` for the first
        alias that :meth:`WinMLCompileConfig.for_provider` recognizes as
        EPContext-capable; otherwise ``(None, {})`` so the stage stays on JIT.
        This is the single point of EP dispatch — no provider name is hardcoded.
        """
        from ..compiler import WinMLCompileConfig

        for entry in provider_options:
            if not isinstance(entry, dict):
                continue
            for ep_alias, opts in entry.items():
                if WinMLCompileConfig.for_provider(ep_alias) is not None:
                    return ep_alias, dict(opts) if isinstance(opts, dict) else {}
        return None, {}

    @staticmethod
    def _compile_marker_path(ctx_onnx: Path) -> Path:
        """Path of the sidecar recording how *ctx_onnx* was compiled (cache key)."""
        return ctx_onnx.parent / f"{ctx_onnx.name}.meta.json"

    @classmethod
    def _write_compile_marker(cls, ctx_onnx: Path, ep_alias: str, ep_opts: dict) -> None:
        """Record the EP + options a compiled stage was built with (cache key)."""
        cls._compile_marker_path(ctx_onnx).write_text(
            json.dumps({"ep": ep_alias, "provider_options": ep_opts}, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def _epcontext_is_fresh(cls, src_onnx: Path, ctx_onnx: Path, ep_opts: dict) -> bool:
        """Return ``True`` when a cached EPContext file may be reused as-is.

        The cache is stale (forcing a recompile) when any of these hold:

        * the compiled artifact or its marker is missing;
        * the source stage ONNX graph is newer than the compiled artifact;
        * the source external-weights ``.data`` sidecar is newer (decoder graphs
          often keep all weights there, so the ``.onnx`` mtime alone is not a
          sufficient cache key);
        * the provider options recorded at compile time differ from *ep_opts*
          (so changing a knob like ``soc_model`` re-compiles even when the
          ``.onnx`` mtime is unchanged).
        """
        marker = cls._compile_marker_path(ctx_onnx)
        if not ctx_onnx.exists() or not marker.exists():
            return False

        ctx_mtime = ctx_onnx.stat().st_mtime
        # Source graph + its external-weights sidecar must both be no newer.
        sources = [src_onnx, src_onnx.parent / f"{src_onnx.name}.data"]
        if any(s.exists() and s.stat().st_mtime > ctx_mtime for s in sources):
            return False

        try:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return bool(recorded.get("provider_options") == ep_opts)

    @staticmethod
    def _patch_stage_filename(cfg: dict, stage_key: str, filename: str) -> None:
        """Rewrite a pipeline stage's ``filename`` to *filename*.

        *filename* is normally a bundle-relative name (resolved by ort-genai
        against the directory it loads ``genai_config.json`` from), but an
        absolute path is also accepted.
        """
        pipeline_list: list = cfg.get("model", {}).get("decoder", {}).get("pipeline", [])
        for stage_entry in pipeline_list:
            if isinstance(stage_entry, dict) and stage_key in stage_entry:
                stage_cfg = stage_entry[stage_key]
                if isinstance(stage_cfg, dict):
                    stage_cfg["filename"] = filename
                    return

    def _compile_stage(
        self,
        src_onnx: Path,
        ctx_out: Path,
        stage_key: str,
        ep_alias: str,
        ep_opts: dict | None = None,
    ) -> bool:
        """Compile *src_onnx* to EPContext format via the shared compiler.

        Runs :func:`compile_onnx` in a subprocess so that a compilation hang or
        crash does not block the caller.  The EP is resolved generically from
        *ep_alias* and the stage's provider options from ``genai_config.json``
        are forwarded unchanged, so each stage is compiled for its own
        accelerator at exactly the optimization level configured in the bundle.

        Args:
            src_onnx: Source ONNX file path.
            ctx_out: Destination EPContext ONNX path.
            stage_key: Human-readable label for logging.
            ep_alias: EP short name for the stage (e.g. ``"qnn"``, ``"openvino"``).
            ep_opts: EP provider options from genai_config (e.g. backend_path,
                htp_performance_mode, htp_graph_finalization_optimization_mode,
                soc_model for QNN).

        Returns:
            ``True`` if compilation succeeded; ``False`` on timeout or error.
        """
        import multiprocessing

        compile_opts = dict(ep_opts or {})

        logger.info(
            "Compiling stage %r for EP %r: %s → %s (options=%s)",
            stage_key,
            ep_alias,
            src_onnx.name,
            ctx_out.name,
            compile_opts,
        )

        # Snapshot the mtimes of any EPContext artifacts that already exist
        # before spawning the compiler. A teardown-crash salvage then accepts
        # only files this subprocess actually created or rewrote — never a stale
        # leftover from an earlier compile (e.g. one built with different
        # provider options), which merely *looks* like a valid EPContext.
        pre_compile_mtimes = {
            str(p): p.stat().st_mtime_ns
            for p in self._epcontext_candidate_paths(src_onnx, ctx_out)
            if p.exists()
        }

        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=_compile_stage_worker,
            args=(str(src_onnx), str(ctx_out), ep_alias, compile_opts),
        )
        proc.start()
        proc.join(timeout=self._compile_timeout)

        if proc.is_alive():
            logger.error(
                "Stage %r compilation timed out after %ds — killing subprocess.",
                stage_key,
                self._compile_timeout,
            )
            proc.kill()
            proc.join()
            self._discard_compiled_stage(ctx_out)
            return False

        if proc.exitcode != 0:
            if self._salvage_epcontext(src_onnx, ctx_out, pre_compile_mtimes):
                logger.warning(
                    "Stage %r compile subprocess exited %d, but a valid EPContext "
                    "was already written to disk (crash during teardown); salvaging %s",
                    stage_key,
                    proc.exitcode,
                    ctx_out.name,
                )
                return True
            logger.warning("Stage %r compilation failed (exit %d)", stage_key, proc.exitcode)
            self._discard_compiled_stage(ctx_out)
            return False

        logger.info("Stage %r compiled successfully → %s", stage_key, ctx_out)
        return True

    def _salvage_epcontext(
        self, src_onnx: Path, ctx_out: Path, pre_compile_mtimes: dict[str, int]
    ) -> bool:
        """Recover a valid EPContext written by a crashed compile subprocess.

        Some accelerator runtimes (notably QNN on the Hexagon NPU) fault during
        interpreter / driver teardown — i.e. *after* the EPContext ``.onnx`` and
        its weights ``.bin`` have already been flushed to disk. Judging success
        by exit code alone would then discard fully valid compiled work and fall
        back to a JIT compile of the original graph, which just repeats the same
        crashing native path at model-load time.

        The shared compiler first writes the EPContext under Onnx Runtime's auto
        name ``{src_stem}_<device>_ctx.onnx`` *next to the source model*, then
        copies it to *ctx_out*. A teardown crash can leave the artifact in either
        directory, so both are searched. On the first *fresh* and *valid* match,
        the graph (and any external-weights ``.bin`` sidecar) is moved to
        *ctx_out* so the caller can treat it exactly like a cleanly compiled
        stage. The ``.bin`` keeps its name, so the graph's ``ep_cache_context``
        reference still resolves once both files sit in the compiled directory.

        A candidate is *fresh* only if this compile actually produced it: a path
        already present before the subprocess started (``pre_compile_mtimes``)
        must have advanced its modification time, otherwise it is a stale
        leftover from an earlier compile and is ignored. Validity additionally
        requires an ``embed_mode=0`` context's external weights file to exist —
        see :meth:`_epcontext_is_valid`.

        The provider / device token in the auto name is matched with a wildcard,
        so no execution provider is hardcoded.

        Returns ``True`` when a valid EPContext is now present at *ctx_out*.
        """
        for candidate in self._epcontext_candidate_paths(src_onnx, ctx_out):
            if not candidate.exists():
                continue
            prior = pre_compile_mtimes.get(str(candidate))
            if prior is not None and candidate.stat().st_mtime_ns <= prior:
                # Pre-existing and untouched by this compile — a stale artifact
                # (e.g. built with different provider options). Never salvage it.
                continue
            if not self._epcontext_is_valid(candidate):
                continue
            if candidate != ctx_out:
                self._promote_epcontext(candidate, ctx_out)
            return True
        return False

    @staticmethod
    def _epcontext_candidate_paths(src_onnx: Path, ctx_out: Path) -> list[Path]:
        """Ordered EPContext locations a crashed compile may have left behind.

        The canonical output *ctx_out* first, then Onnx Runtime's auto-named
        ``{src_stem}_<device>_ctx.onnx`` beside the canonical output and beside
        the source model (the device token is a wildcard, so no EP is
        hardcoded). ``dict.fromkeys`` dedups the directories, preserving order,
        in case they coincide.
        """
        candidates: list[Path] = [ctx_out]
        for directory in dict.fromkeys([ctx_out.parent, src_onnx.parent]):
            candidates.extend(sorted(directory.glob(f"{src_onnx.stem}_*_ctx.onnx")))
        return candidates

    @staticmethod
    def _epcontext_is_valid(candidate: Path) -> bool:
        """True if *candidate* is an EPContext graph with its weights present.

        Beyond the structural check (an ``EPContext`` node exists), the *main
        context* node (``main_context=1``, the default) with ``embed_mode=0``
        stores its compiled blob in an external file named by
        ``ep_cache_context``; that file must exist and be non-empty next to the
        graph, otherwise the salvaged stage would reference a missing binary and
        crash at load time. ``embed_mode=1`` (or an absent attribute) embeds the
        blob inline, so only the structural check applies — the
        ``ep_cache_context`` bytes are the raw blob, not a path, and must not be
        treated as a filename.

        Secondary partition nodes (``main_context=0``) legitimately omit
        ``ep_cache_context`` per the EPContext schema — a single QNN context can
        hold all partitions, with only the ``main_context=1`` node carrying the
        external reference — so they are skipped here, mirroring the compiler's
        own ``main_context`` handling.
        """
        from ..onnx import is_compiled_onnx, load_onnx

        # Structural gate. ``is_compiled_onnx`` normalises a corrupt / unreadable
        # file to ValueError (missing → OSError), so a garbage leftover from a
        # crashed compile is rejected here rather than propagating.
        try:
            if not is_compiled_onnx(candidate):
                return False
        except (ValueError, OSError):
            return False

        # Parseable and structurally an EPContext; verify the main context's
        # external weights file is present (the load below cannot raise a parse
        # error now that the structural gate has passed).
        model = load_onnx(str(candidate), load_weights=False, validate=False)
        for node in model.graph.node:
            if node.op_type != "EPContext":
                continue
            embed_mode = 1
            main_context = 1
            cache_ref = ""
            for attr in node.attribute:
                if attr.name == "embed_mode":
                    embed_mode = attr.i
                elif attr.name == "main_context":
                    main_context = attr.i
                elif attr.name == "ep_cache_context":
                    cache_ref = attr.s.decode("utf-8", "ignore")
            # Secondary partitions share the main context's blob and carry no
            # external reference of their own — nothing to validate.
            if main_context == 0:
                continue
            if embed_mode == 0:
                if not cache_ref:
                    return False
                ref_path = candidate.parent / cache_ref
                try:
                    if not ref_path.is_file() or ref_path.stat().st_size == 0:
                        return False
                except OSError:
                    return False
        return True

    @staticmethod
    def _promote_epcontext(src_ctx: Path, ctx_out: Path) -> None:
        """Move a salvaged EPContext graph and its ``.bin`` sidecars to *ctx_out*.

        External-weights sidecars are moved first, keeping their original names
        so the graph's ``ep_cache_context`` reference resolves once everything
        lives in *ctx_out*'s directory.
        """
        compiled_dir = ctx_out.parent
        for sidecar in sorted(src_ctx.parent.glob(f"{src_ctx.stem}*.bin")):
            dst_bin = compiled_dir / sidecar.name
            if sidecar != dst_bin:
                sidecar.replace(dst_bin)
        src_ctx.replace(ctx_out)

    @classmethod
    def _discard_compiled_stage(cls, ctx_out: Path) -> None:
        """Remove a partial/failed EPContext artifact and its cache marker."""
        ctx_out.unlink(missing_ok=True)
        cls._compile_marker_path(ctx_out).unlink(missing_ok=True)

    def _mirror_non_onnx_files(
        self, compiled_dir: Path, skip_filenames: set[str] | None = None
    ) -> None:
        """Create symlinks (or copies on Windows) for files not being compiled.

        Links files into *compiled_dir* so ``og.Config`` finds tokenizer files,
        non-compiled ONNX models (embeddings, lm_head, stages on non-EPContext
        EPs), etc.  Only ONNX files listed in *skip_filenames* (the compiled
        stages) and their external ``.data`` siblings are skipped — everything
        else is linked.  Existing files are left untouched.
        """
        skip = set(skip_filenames or [])
        # Also skip the external-weights ``.data`` sidecars of the compiled stages.
        skip_data = {f"{name}.data" for name in skip}
        for src in self._bundle_dir.iterdir():
            if src.name == self._COMPILED_SUBDIR:
                continue
            if src.name in skip or src.name in skip_data:
                continue
            dst = compiled_dir / src.name
            if dst.exists():
                continue
            if src.is_file():
                try:
                    dst.symlink_to(src.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dst)

    @staticmethod
    def _import_og() -> Any:
        """Import and return the ``onnxruntime_genai`` module.

        Raises:
            GenaiNotInstalledError: the module could not be imported — either
                it is not installed, or it is installed but its native
                extension failed to load (e.g. missing runtime dependencies or
                a platform-incompatible build).
        """
        try:
            import onnxruntime_genai as og

            return og
        except ImportError as exc:
            raise GenaiNotInstalledError(
                f"Could not import onnxruntime_genai: {exc}. It may not be "
                "installed, or it is installed but its native module failed "
                "to load (e.g. missing runtime dependencies or a "
                "platform-incompatible build)."
            ) from exc

    def _register_eps(self) -> None:
        """Register WinML EPs with ORT GenAI (idempotent, best-effort)."""
        try:
            registry = WinMLEPRegistry.get_instance()
            if registry.winml_available:
                result = registry.register_execution_providers(ort_genai=True)
                registered = result.get("onnxruntime_genai", [])
                logger.info("WinML EPs registered for ORT GenAI: %s", registered)
        except Exception as exc:
            logger.warning("WinML EP registration skipped: %s", exc)

    def _read_genai_config(self) -> dict[str, Any]:
        """Parse and return the bundle's ``genai_config.json``."""
        config_src = self._bundle_dir / "genai_config.json"
        cfg: dict[str, Any] = json.loads(config_src.read_text(encoding="utf-8"))
        return cfg

    @staticmethod
    def _bundle_uses_hardware_ep(cfg: dict[str, Any]) -> str | None:
        """Return the first non-CPU/DML EP name found, or ``None``.

        WinML EP discovery/registration is only required when the bundle's
        ``genai_config.json`` assigns at least one pipeline stage to a hardware
        execution provider that needs WinML registration (QNN, OpenVINO, ...);
        CPU and DML bundles need none.  The decision is read from the bundle
        config itself.

        Two config layouts are supported:

        1. **Pipeline list** - ``model.decoder.pipeline[*].<stage>.session_options``
        2. **Flat decoder** - ``model.decoder.session_options`` (no ``pipeline``
           wrapper, used by e.g. OpenVINO exports).
        """
        skip_eps = frozenset({"cpu", "dml"})

        def _first_hw_ep(so: object) -> str | None:
            if not isinstance(so, dict):
                return None
            for entry in so.get("provider_options", []):
                if not isinstance(entry, dict):
                    continue
                for name in entry:
                    if str(name).lower() not in skip_eps:
                        return str(name)
            return None

        decoder = cfg.get("model", {}).get("decoder", {})
        if not isinstance(decoder, dict):
            return None

        # Layout 2: flat session_options directly on the decoder.
        ep = _first_hw_ep(decoder.get("session_options"))
        if ep is not None:
            return ep

        # Layout 1: pipeline list with per-stage session_options.
        pipeline_list = decoder.get("pipeline", [])
        if not isinstance(pipeline_list, list):
            return None
        for stage_entry in pipeline_list:
            if not isinstance(stage_entry, dict):
                continue
            for stage_cfg in stage_entry.values():
                if not isinstance(stage_cfg, dict):
                    continue
                ep = _first_hw_ep(stage_cfg.get("session_options"))
                if ep is not None:
                    return ep
        return None

    def _read_context_length(self) -> int:
        """Read ``model.context_length`` from ``genai_config.json``."""
        return int(self._read_genai_config()["model"]["context_length"])


__all__ = [
    "GenaiLoadError",
    "GenaiNotInstalledError",
    "GenaiSession",
    "GenaiSessionError",
    "GenerationConfig",
    "GenerationTiming",
]
