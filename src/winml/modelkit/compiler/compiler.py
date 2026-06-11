# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compiler orchestration class."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .context import CompileContext
from .result import CompileResult


if TYPE_CHECKING:
    from ..utils.constants import EPName
    from .configs import WinMLCompileConfig
    from .stages.base import BaseStage


# EP → available compilers. Keys are canonical EPName (or None for the default).
EP_COMPILER_MAPPING: dict[EPName | None, list[str]] = {
    "QNNExecutionProvider": ["ort", "qairt"],
    None: ["ort"],
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

    def __init__(
        self,
        n_total_models: int = 1,
        use_inference_session: bool = False,
    ) -> None:
        """Create a compiler.

        Args:
            n_total_models: Total number of models compiled by this instance. When
                >1, the models share a single EP context (weight sharing) and the
                same shared ``SessionOptions`` is reused across every ``compile``.
            use_inference_session: Select the ``ort.InferenceSession``
                (``ep.context_enable``) backend instead of the default
                ``ort.ModelCompiler``.
        """
        self.n_total_models = n_total_models
        self.use_inference_session = use_inference_session
        # The shared SessionOptions: created by CompileStage on the first model and
        # reused for the rest (kept here so it survives between compile() calls).
        self.inference_session: object | None = None
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
                use_inference_session=self.use_inference_session,
                inference_session=self.inference_session,
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
            self.inference_session = context.inference_session
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
    model_paths: list[str | Path],
    output_path: str | Path | None = None,
    config: WinMLCompileConfig | None = None,
    use_inference_session: bool = False,
) -> list[CompileResult]:
    """Compile one or more ONNX models, sharing a single EP context when >1.

    A single :class:`Compiler` (``n_total_models=len(model_paths)``) compiles every
    model in sequence, reusing one shared ``SessionOptions`` so the weights are shared
    across the compiled EPContext models. The backend is ``ort.ModelCompiler`` by
    default, or ``ort.InferenceSession`` when ``use_inference_session`` is set.

    Args:
        model_paths: Input ONNX model paths.
        output_path: Output directory (or file) for the compiled models.
        config: Compilation configuration. ``None`` skips compilation (passthrough).
        use_inference_session: Use the InferenceSession backend.

    Returns:
        One :class:`CompileResult` per input model, in order.
    """
    compiler = Compiler(
        n_total_models=len(model_paths),
        use_inference_session=use_inference_session,
    )
    # Compiled in order (the comprehension evaluates left-to-right) so the shared
    # context accumulates across models and the last one flushes it.
    return [
        compiler.compile(model_path=mp, output_path=output_path, config=config)
        for mp in model_paths
    ]
