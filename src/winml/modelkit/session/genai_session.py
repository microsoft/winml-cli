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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .._ep_compile_worker import qnn_compile_to_ep_context as _qnn_compile_worker
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
        compile: Pre-compile QNN pipeline stages to EPContext ONNX on first
            run (inside ``bundle_dir/_compiled/``).  Subsequent calls reuse
            the cached EPContext files, eliminating per-run JIT overhead.
            Only stages that can be compiled without hanging are attempted;
            stages that fail compilation fall back to the original ONNX.
            Has no effect when ``ep="cpu"``.
        compile_timeout: Maximum seconds to wait for each stage to compile
            before killing the subprocess and falling back to the original
            ONNX.  Defaults to 300 seconds (5 minutes).
    """

    # Sub-directory within the bundle that holds pre-compiled EPContext ONNX files.
    _COMPILED_SUBDIR: str = "_compiled"

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
        self._model: object | None = None
        self._tokenizer: object | None = None

        if not self._bundle_dir.exists():
            raise FileNotFoundError(f"Bundle directory not found: {self._bundle_dir}")
        config_path = self._bundle_dir / "genai_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"genai_config.json not found in {self._bundle_dir}. "
                "Ensure the bundle was created with a winml-cli export command."
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

        # Determine which bundle directory og.Config should load from.
        load_dir = self._bundle_dir
        if self._compile and self._ep in _NEEDS_WINML_EPS:
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
            # This runs whether the caller exhausts the iterator normally, breaks
            # out early, or the generator is garbage-collected — preventing the NPU
            # memory from being held until a later GC cycle.
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

    def _prepare_compiled_bundle(self) -> Path:
        """Create (or reuse) a *compiled* bundle directory.

        Reads ``genai_config.json``, finds QNN-accelerated stages (those with
        ``QNNExecutionProvider`` in their ``session_options``), and tries to
        compile their ONNX to EPContext format using ``ort.ModelCompiler``.

        The compiled bundle is stored under ``bundle_dir/_compiled/``.  On
        every call the helper checks whether the cached EPContext file is
        newer than the source ONNX; if so, it skips recompilation.

        Returns:
            Path to the compiled bundle directory (may equal ``bundle_dir``
            if no compilable stages were found, or if all compilations failed).
        """
        compiled_dir = self._bundle_dir / self._COMPILED_SUBDIR
        config_src = self._bundle_dir / "genai_config.json"
        cfg = json.loads(config_src.read_text(encoding="utf-8"))

        # Collect pipeline stages that use a QNN EP ("qnn" key in provider_options).
        # genai_config pipeline entries: [{"context": {...}}, {"iterator": {...}}, ...]
        # provider_options format: [{"qnn": {...}}]
        pipeline_list: list = cfg.get("model", {}).get("decoder", {}).get("pipeline", [])
        # [(stage_key, onnx_filename, qnn_opts), ...]
        qnn_stages: list[tuple[str, str, dict]] = []
        for stage_entry in pipeline_list:
            if not isinstance(stage_entry, dict):
                continue
            for stage_key, stage_cfg in stage_entry.items():
                if not isinstance(stage_cfg, dict):
                    continue
                so = stage_cfg.get("session_options", {})
                providers = so.get("provider_options", [])
                for p in providers:
                    if isinstance(p, dict) and "qnn" in p:
                        onnx_filename = stage_cfg.get("filename", f"{stage_key}.onnx")
                        qnn_stages.append((stage_key, onnx_filename, dict(p["qnn"])))
                        break

        if not qnn_stages:
            logger.info("No QNN stages found in genai_config.json; skipping compilation")
            return self._bundle_dir

        compiled_dir.mkdir(exist_ok=True)
        modified_cfg = json.loads(config_src.read_text(encoding="utf-8"))
        any_compiled = False

        for stage_key, onnx_filename, qnn_opts in qnn_stages:
            src_onnx = self._bundle_dir / onnx_filename
            ctx_onnx = compiled_dir / f"{stage_key}_ctx.onnx"

            # Skip recompilation if cache is up-to-date.
            if ctx_onnx.exists() and ctx_onnx.stat().st_mtime >= src_onnx.stat().st_mtime:
                logger.info("Stage %r: reusing cached EPContext %s", stage_key, ctx_onnx.name)
                # Use just the filename — genai_config.json lives in compiled_dir,
                # so ort-genai resolves filenames relative to compiled_dir.
                self._patch_stage_filename(modified_cfg, stage_key, ctx_onnx.name)
                any_compiled = True
                continue

            # Attempt compilation.
            success = self._compile_stage(src_onnx, ctx_onnx, stage_key, qnn_opts)
            if success:
                self._patch_stage_filename(modified_cfg, stage_key, ctx_onnx.name)
                any_compiled = True
            else:
                logger.warning(
                    "Stage %r: compilation failed; using original ONNX (JIT fallback)", stage_key
                )
                # Patch to the absolute source path so ort-genai can find the
                # file when loading from compiled_dir (where it doesn't exist).
                self._patch_stage_filename(modified_cfg, stage_key, str(src_onnx.resolve()))

        if not any_compiled:
            return self._bundle_dir

        # Write the modified genai_config into the compiled sub-directory.
        # ONNX filenames are relative to compiled_dir; ort-genai resolves them
        # from the directory it loads og.Config from.
        compiled_config = compiled_dir / "genai_config.json"
        compiled_config.write_text(
            json.dumps(modified_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # Only skip the QNN-stage ONNX files (compiled → new path, or failed →
        # absolute path patch).  Non-QNN ONNX files (embeddings, lm_head) must
        # be symlinked into compiled_dir so ort-genai can find them by filename.
        compiled_onnx_names = {onnx_filename for _, onnx_filename, _ in qnn_stages}
        self._mirror_non_onnx_files(compiled_dir, skip_filenames=compiled_onnx_names)

        logger.info("Compiled bundle prepared at %s", compiled_dir)
        return compiled_dir

    @staticmethod
    def _patch_stage_filename(cfg: dict, stage_key: str, abs_path: str) -> None:
        """Rewrite a pipeline stage's ``filename`` to an absolute path."""
        pipeline_list: list = cfg.get("model", {}).get("decoder", {}).get("pipeline", [])
        for stage_entry in pipeline_list:
            if isinstance(stage_entry, dict) and stage_key in stage_entry:
                stage_cfg = stage_entry[stage_key]
                if isinstance(stage_cfg, dict):
                    stage_cfg["filename"] = abs_path
                    return

    def _compile_stage(
        self,
        src_onnx: Path,
        ctx_out: Path,
        stage_key: str,
        qnn_opts: dict | None = None,
    ) -> bool:
        """Compile *src_onnx* to EPContext format via ``ort.ModelCompiler``.

        Runs in a subprocess so that a ModelCompiler failure does not block
        the caller.  The QNN options from ``genai_config.json`` are forwarded
        unchanged to the compilation session, so each stage is compiled at
        exactly the optimization level configured in the bundle.

        Args:
            src_onnx: Source ONNX file path.
            ctx_out: Destination EPContext ONNX path.
            stage_key: Human-readable label for logging.
            qnn_opts: QNN provider options from genai_config (e.g. backend_path,
                htp_performance_mode, htp_graph_finalization_optimization_mode,
                soc_model).

        Returns:
            ``True`` if compilation succeeded; ``False`` on timeout or error.
        """
        import multiprocessing

        compile_qnn_opts = dict(qnn_opts or {})

        logger.info(
            "Compiling stage %r: %s → %s (qnn_opts=%s)",
            stage_key,
            src_onnx.name,
            ctx_out.name,
            compile_qnn_opts,
        )

        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=_qnn_compile_worker, args=(str(src_onnx), str(ctx_out), compile_qnn_opts)
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
            ctx_out.unlink(missing_ok=True)
            return False

        if proc.exitcode != 0:
            logger.warning("Stage %r compilation failed (exit %d)", stage_key, proc.exitcode)
            ctx_out.unlink(missing_ok=True)
            return False

        logger.info("Stage %r compiled successfully → %s", stage_key, ctx_out)
        return True

    def _mirror_non_onnx_files(
        self, compiled_dir: Path, skip_filenames: set[str] | None = None
    ) -> None:
        """Create symlinks (or copies on Windows) for files not being compiled.

        Links files into *compiled_dir* so ``og.Config`` finds tokenizer files,
        non-QNN ONNX models (embeddings, lm_head), etc.  Only ONNX files listed
        in *skip_filenames* (the QNN-compiled stages) and their external ``.data``
        siblings are skipped — everything else, including CPU-side ONNX files, is
        linked.  Existing files are left untouched.
        """
        skip = set(skip_filenames or [])
        # Also skip .data sidecars of the compiled-stage ONNX files.
        skip_data = {f"{name}.data" for name in skip} | {f"{name}.data" for name in skip}
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
