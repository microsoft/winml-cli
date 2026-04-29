# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compile stage - generate EPContext model."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ...onnx import load_onnx, save_onnx
from ...session import WinMLQairtSession, WinMLSession
from ..configs import WinMLCompileConfig
from .base import BaseStage


if TYPE_CHECKING:
    import onnxruntime as ort

    from ..context import CompileContext


# Maps compiler name to session class
COMPILER_SESSION_MAPPING: dict[str, type[WinMLSession]] = {
    "ort": WinMLSession,
    "qairt": WinMLQairtSession,
}


class CompileStage(BaseStage):
    """Compile model."""

    name: ClassVar[str] = "compile"

    @classmethod
    def should_run(cls, context: CompileContext) -> bool:
        """Always run as final stage."""
        return True

    def process(self, context: CompileContext) -> CompileContext:
        """Execute compilation."""
        context.log("Starting compile stage")
        start_time = time.time()

        try:
            # Resolve session class from compiler config
            compiler = context.config.get("compiler", "ort")
            session_cls = COMPILER_SESSION_MAPPING[compiler]

            # Determine final output directory (default: same as input model)
            output_dir = self._get_output_dir(context)
            context.log(f"Output directory: {output_dir}")

            # Ensure model is saved to disk (may be in work_dir if modified)
            model_path = self._ensure_model_file(context)
            context.log(f"Model path: {model_path}")

            ep_config = WinMLCompileConfig.from_dict(context.config).ep_config
            # Create WinMLSession
            context.log(f"Creating {session_cls.__name__} for device: {context.execution_provider}")
            winml_session = session_cls(
                onnx_path=model_path,
                device=context.execution_provider,
                ep_config=ep_config,
            )
            winml_session.compile()

            # Get the underlying session for validation and info collection
            session = winml_session._session
            context.session = session

            # Log actual providers used
            if session is not None:
                actual_providers = session.get_providers()
                context.log(f"Actual providers: {actual_providers}")

                # Validate if requested
                if context.validate:
                    self._validate_model(session, context)

                # Collect model info
                self._collect_model_info(session, context)

            # Find and relocate EPContext files to output directory
            if ep_config.enable_ep_context:
                self._finalize_output(context, model_path, output_dir)

        except Exception as e:
            context.add_error(f"Compilation failed: {e}")
            raise

        finally:
            elapsed = time.time() - start_time
            context.add_metric("compile_time", elapsed)
            context.log(f"Compilation completed in {elapsed:.2f}s")

        return context

    def _get_output_dir(self, context: CompileContext) -> Path:
        """Determine the output directory for compiled model.

        Priority:
        1. config.output_path (if specified)
        2. Original input model's directory (default)
        """
        # Check if output_path is specified in config
        output_path = context.config.get("output_path")
        if output_path:
            output_path = Path(output_path)
            # If it's a file path, use its parent directory
            if output_path.suffix:
                return output_path.parent
            return output_path

        # Default: same directory as original input model
        return context.model_path.parent

    def _ensure_model_file(self, context: CompileContext) -> Path:
        """Ensure model is saved to a file."""
        # If we have a modified model in context, save it
        if context.model is not None:
            if context.work_dir:
                model_path = context.work_dir / "model_to_compile.onnx"
            else:
                # Use temp file
                fd, path = tempfile.mkstemp(suffix=".onnx")
                import os

                os.close(fd)
                model_path = Path(path)

            save_onnx(context.model, model_path)
            return model_path

        # Otherwise use original path
        return context.model_path

    def _build_provider_options(self, context: CompileContext) -> dict[str, str]:
        """Build provider options from context config."""
        return dict(context.config.get("provider_options", {}))

    def _validate_model(self, session: ort.InferenceSession, context: CompileContext) -> None:
        """Validate compiled model produces outputs."""
        context.log("Validating compiled model...")

        # Generate dummy inputs
        inputs = self._generate_dummy_inputs(session)

        # Run warmup
        warmup_runs = context.config.get("warmup_runs", 1)
        try:
            for _ in range(warmup_runs):
                outputs = session.run(None, inputs)
        except Exception as e:
            context.add_error(f"Validation inference failed: {e}")
            return

        # Check outputs
        for i, output in enumerate(outputs):
            if np.isnan(output).any():
                context.add_warning(f"Output {i} contains NaN values")
            if np.isinf(output).any():
                context.add_warning(f"Output {i} contains Inf values")

        context.log("Validation passed")
        context.add_metric("validation_passed", True)

    def _generate_dummy_inputs(self, session: ort.InferenceSession) -> dict[str, np.ndarray]:
        """Generate dummy inputs for validation using all-ones data."""
        inputs = {}

        for input_meta in session.get_inputs():
            name = input_meta.name
            shape = input_meta.shape
            dtype = input_meta.type

            # Handle dynamic dimensions
            shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in shape]

            # Map ORT type to numpy
            type_map = {
                "tensor(float)": np.float32,
                "tensor(float16)": np.float16,
                "tensor(int64)": np.int64,
                "tensor(int32)": np.int32,
                "tensor(uint8)": np.uint8,
                "tensor(int8)": np.int8,
                "tensor(bool)": np.bool_,
            }
            np_dtype = type_map.get(dtype, np.float32)

            inputs[name] = np.ones(shape, dtype=np_dtype)

        return inputs

    def _finalize_output(self, context: CompileContext, model_path: Path, output_dir: Path) -> None:
        """Find EPContext files and copy to output directory.

        WinMLSession saves to work_dir. This method copies the output
        to the final output directory (default: same as input model).
        """
        device = context.execution_provider.lower()

        # Find EPContext in work_dir (where WinMLSession saved it)
        ctx_patterns = [
            model_path.parent / f"{model_path.stem}_{device}_ctx.onnx",
            model_path.parent / f"{model_path.stem}_ctx.onnx",
        ]

        src_ctx_path = None
        for pattern in ctx_patterns:
            if pattern.exists():
                src_ctx_path = pattern
                break

        if src_ctx_path is None:
            context.add_warning("EPContext model not found in work directory")
            return

        # Determine final output filename
        # If user specified a file path (has .onnx suffix), use it as-is
        user_output = context.config.get("output_path")
        if user_output and Path(user_output).suffix:
            final_ctx_path = Path(user_output)
        else:
            original_stem = context.model_path.stem
            final_ctx_name = f"{original_stem}_{device}_ctx.onnx"
            final_ctx_path = output_dir / final_ctx_name

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy EPContext to final location (if different)
        if src_ctx_path != final_ctx_path:
            shutil.copy2(src_ctx_path, final_ctx_path)
            context.log(f"Copied EPContext to: {final_ctx_path}")
        else:
            context.log(f"EPContext already at: {final_ctx_path}")

        context.output_path = final_ctx_path

        # Find and copy binary files (.bin, .onnx.bin, etc.)
        bin_renamed = False
        final_bin_name = None
        for f in src_ctx_path.parent.iterdir():
            if f.name.startswith(src_ctx_path.stem) and f.suffix == ".bin":
                bin_suffix = f.name[len(src_ctx_path.stem) :]
                final_bin_name = f"{final_ctx_path.stem}{bin_suffix}"
                final_bin_path = output_dir / final_bin_name
                if f != final_bin_path:
                    shutil.copy2(f, final_bin_path)
                    context.log(f"Copied binary to: {final_bin_path}")
                    bin_renamed = True
                context.context_binary_path = final_bin_path
                break

        # Update ep_cache_context in ONNX if bin was renamed (external mode)
        if bin_renamed and final_bin_name:
            model = load_onnx(final_ctx_path, validate=False)
            for node in model.graph.node:
                if node.op_type != "EPContext":
                    continue
                attrs = {a.name: a for a in node.attribute}
                if "embed_mode" not in attrs or attrs["embed_mode"].i != 0:
                    continue
                if attrs.get("main_context") and attrs["main_context"].i == 0:
                    continue
                ep_cache_context = attrs.get("ep_cache_context")
                if ep_cache_context:
                    ep_cache_context.s = final_bin_name.encode("utf-8")
                    save_onnx(model, final_ctx_path)
                    context.log(f"Updated ep_cache_context: {final_bin_name}")
                break

    def _collect_model_info(self, session: ort.InferenceSession, context: CompileContext) -> None:
        """Collect model input/output information."""
        input_shapes = {}
        for inp in session.get_inputs():
            input_shapes[inp.name] = list(inp.shape)

        output_shapes = {}
        for out in session.get_outputs():
            output_shapes[out.name] = list(out.shape)

        context.add_metric("input_shapes", input_shapes)
        context.add_metric("output_shapes", output_shapes)
