# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compiler orchestration class."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .context import CompileContext
from .result import CompileResult


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Sequence

    import onnxruntime as ort

    from ..utils.constants import EPName
    from .configs import WinMLCompileConfig
    from .stages.base import BaseStage


# EP → available compilers. Keys are canonical EPName (or None for the default).
EP_COMPILER_MAPPING: dict[EPName | None, list[str]] = {
    "QNNExecutionProvider": ["ort", "ort_jit", "qairt"],
    None: ["ort", "ort_jit"],
}


def list_compilers(ep: EPName | None) -> str:
    """Return available compilers for an EP as a comma-separated string."""
    compilers = EP_COMPILER_MAPPING.get(ep, EP_COMPILER_MAPPING[None])
    return ", ".join(compilers)


class Compiler:
    """Orchestrates the compilation pipeline.

    The compiler executes stages in order:
    1. OptimizeStage - EP-specific graph transforms (skipped if none registered)
    2. QFormatConvertStage - QLinear-to-QDQ conversion (skipped if not needed)
    3. CompileStage - Generate EPContext model

    Quantization (calibration + QDQ insertion) is handled externally by the
    quantization module before the model reaches this pipeline.

    Example:
        compiler = Compiler()
        result = compiler.compile("model.onnx", WinMLCompileConfig.for_qnn())
    """

    # Registered stages (in execution order)
    _stages: list[type[BaseStage]] | None = None

    def __init__(self, n_total_models: int = 1) -> None:
        """Create a compiler.

        Args:
            n_total_models: Total number of models compiled by this instance. When
                >1, the models share a single EP context (weight sharing) and the
                same shared ``SessionOptions`` is reused across every ``compile``.

        The compile backend (ort.ModelCompiler vs ort.InferenceSession) is taken from
        the config's ``compiler`` setting ("ort_jit" selects the
        InferenceSession backend), surfaced via ``CompileContext.use_inference_session``.
        """
        self.n_total_models = n_total_models
        # The shared SessionOptions: created by CompileStage on the first model and
        # reused for the rest (kept here so it survives between compile() calls).
        self.shared_session_options: ort.SessionOptions | None = None
        self.n_compiled_models = 0

    @classmethod
    def _get_stages(cls) -> list[type[BaseStage]]:
        """Lazy initialization of stages."""
        if cls._stages is None:
            from .stages import (
                CompileStage,
                OptimizeStage,
                QFormatConvertStage,
            )

            cls._stages = [
                OptimizeStage,
                QFormatConvertStage,
                CompileStage,
            ]
        return cls._stages

    def compile(
        self,
        model_path: str | Path,
        output_path: str | Path | None = None,
        config: WinMLCompileConfig | None = None,
    ) -> CompileResult:
        """Execute the compilation pipeline.

        Args:
            model_path: Path to input ONNX model
            output_path: Path for output compiled model (defaults to {model_stem}_ctx.onnx)
            config: Compilation configuration (defaults to QNN with quantization)

        Returns:
            CompileResult with paths and metrics
        """
        start_time = time.time()
        model_path = Path(model_path)

        # If no config provided, skip compilation entirely (passthrough)
        if config is None:
            return CompileResult(
                success=True,
                output_path=model_path,
                errors=[],
                warnings=["No compile config provided, skipping compilation (passthrough)"],
            )

        # Set up working directory (always use temp dir now)
        temp_dir = tempfile.TemporaryDirectory()
        work_dir = Path(temp_dir.name)

        try:
            # Create context from config. Multi-model / weight-sharing state is
            # threaded through so CompileStage can pick the backend, reuse the shared
            # SessionOptions, and detect the last (stop_share) model.
            context = CompileContext(
                model_path=model_path,
                config=config.to_dict(),
                work_dir=work_dir,
                verbose=config.verbose,
                n_compiled_models=self.n_compiled_models,
                n_total_models=self.n_total_models,
                shared_session_options=self.shared_session_options,
            )

            if output_path is not None:
                context.config["output_path"] = str(output_path)

            context.log(f"Starting compilation of {model_path}")
            context.log(f"Execution provider: {context.execution_provider}")

            # Execute stages
            for stage_cls in self._get_stages():
                if context.has_error:
                    break

                if stage_cls.should_run(context):
                    context.log(f"Running stage: {stage_cls.name}")
                    stage = stage_cls()
                    context = stage.process(context)
                else:
                    context.log(f"Skipping stage: {stage_cls.name}")

            # Carry the shared SessionOptions (created/reused by CompileStage) forward
            # so the next model in a shared-context run reuses the same EP + group.
            self.shared_session_options = context.shared_session_options
            self.n_compiled_models += 1

            # Build result
            total_time = time.time() - start_time
            result = self._build_result(context, total_time)

            if result.success:
                context.log(f"Compilation successful in {total_time:.2f}s")
            else:
                context.log(f"Compilation failed: {result.errors}")

            return result

        finally:
            # Cleanup temp directory
            if temp_dir:
                temp_dir.cleanup()

    def _build_result(self, context: CompileContext, total_time: float) -> CompileResult:
        """Build CompileResult from context."""
        return CompileResult(
            success=not context.has_error,
            output_path=context.output_path,
            context_binary_path=context.context_binary_path,
            compile_time=context.metrics.get("compile_time"),
            total_time=total_time,
            input_shapes=context.metrics.get("input_shapes", {}),
            output_shapes=context.metrics.get("output_shapes", {}),
            validation_passed=context.metrics.get("validation_passed", False),
            performance_metrics=context.metrics.get("performance", {}),
            errors=context.errors,
            warnings=context.warnings,
        )


def compile_onnx(
    model_path: str | Path,
    output_path: str | Path | None = None,
    config: WinMLCompileConfig | None = None,
) -> CompileResult:
    """Compile ONNX model to EP-specific format.

    This is the primary API for compiling ONNX models.

    Args:
        model_path: Path to input ONNX model
        output_path: Path for output compiled model (defaults to {model_stem}_ctx.onnx)
        config: Compilation configuration. If None, compilation is skipped (passthrough).

    Returns:
        CompileResult with paths and metrics

    Examples:
        # Skip compilation (passthrough)
        result = compile_onnx("model.onnx")  # config=None skips compilation

        # QNN with quantization using random calibration data
        result = compile_onnx("model.onnx", config=WinMLCompileConfig.for_qnn())

        # Compile with explicit output path
        result = compile_onnx("model.onnx", "model_compiled.onnx", WinMLCompileConfig.for_qnn())

        # CPU compilation (no EPContext)
        config = WinMLCompileConfig.for_cpu()
        result = compile_onnx("model.onnx", config=config)

        # Note: Quantization is handled by WinMLQuantizationConfig
        # in the quant module, not by the compiler. Use the build pipeline
        # (build_hf_model or build_onnx_model) for quantize+compile workflows.
    """
    compiler = Compiler()
    return compiler.compile(model_path=model_path, output_path=output_path, config=config)


def compile_multiple_onnx(
    model_paths: Sequence[str | Path],
    output_path: str | Path | None = None,
    config: WinMLCompileConfig | None = None,
) -> list[CompileResult]:
    """Compile one or more ONNX models, sharing a single EP context when >1.

    A single :class:`Compiler` (``n_total_models=len(model_paths)``) compiles every
    model in sequence, reusing one shared ``SessionOptions`` so the weights are shared
    across the compiled EPContext models. The backend is taken from
    ``config.ep_config.compiler``: ``ort.ModelCompiler`` (default) or
    ``ort.InferenceSession`` when it is ``"ort_jit"``.

    Args:
        model_paths: Input ONNX model paths.
        output_path: Where to write the compiled model(s).

            * With a **single** model it may be a **file** path (the exact
              ``*_ctx.onnx``) or a **directory** (``<stem>_ctx.onnx`` is written into
              it); ``None`` writes next to the input.
            * With **multiple** models it **must be a directory** — each model is
              written as ``<stem>_ctx.onnx`` there, with same-named inputs disambiguated
              by an integer suffix on the later one(s) (with a warning), e.g.
              ``model_ctx.onnx`` then ``model_1_ctx.onnx``.
        config: Compilation configuration. ``None`` skips compilation (passthrough).

    Returns:
        One :class:`CompileResult` per input model, in order.
    """
    paths = [Path(mp) for mp in model_paths]
    out = Path(output_path) if output_path is not None else None
    # A path with a suffix (e.g. ".onnx") is a file; otherwise it's a directory.
    out_is_file = out is not None and bool(out.suffix)

    if len(paths) > 1:
        out_is_dir = out is not None and not out_is_file
        assert out_is_dir, (
            "output_path must be a directory when compiling multiple models "
            f"(shared EP context), got {output_path!r}"
        )

    # Backend is taken from config.ep_config.compiler ("ort_jit" selects
    # the InferenceSession backend), surfaced via CompileContext.use_inference_session.
    compiler = Compiler(n_total_models=len(paths))
    # Compiled in order so the shared context accumulates and the last model flushes it.
    # When writing into a directory, outputs are keyed by filename stem, so disambiguate
    # same-named inputs by suffixing the later one(s) instead of overwriting.
    results: list[CompileResult] = []
    seen_stems: dict[str, int] = {}
    for p in paths:
        count = seen_stems.get(p.stem, 0)
        seen_stems[p.stem] = count + 1
        out_stem = p.stem if count == 0 else f"{p.stem}_{count}"
        if count > 0:
            logger.warning(
                "Input model name %r repeats; writing its compiled output as "
                "'%s_ctx.onnx' to avoid overwriting the earlier one.",
                p.name,
                out_stem,
            )
        if out is None:
            resolved = None
        elif out_is_file:
            # Single-model file path: write exactly there.
            resolved = out
        else:
            resolved = out / f"{out_stem}_ctx.onnx"
        results.append(compiler.compile(model_path=p, output_path=resolved, config=config))
    return results
