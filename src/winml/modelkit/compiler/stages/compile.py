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
from ...utils.constants import normalize_ep_name
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
        """Execute compilation.

        Two compile paths, selected from the multi-model state on the context:

        * ``_compile_single_model_compiler`` (single model, default): the existing
          ``WinMLSession`` / ``ort.ModelCompiler`` path — unchanged.
        * ``_compile_multiple`` (``use_inference_session`` and/or
          ``n_total_models > 1``): reuses one shared ``SessionOptions`` so multiple
          models share a single EP context (weight sharing); the backend is
          ``ort.InferenceSession`` when requested, else ``ort.ModelCompiler``.
        """
        context.log("Starting compile stage")
        start_time = time.time()

        try:
            if context.use_inference_session or context.n_total_models > 1:
                self._compile_multiple(context)
            else:
                self._compile_single_model_compiler(context)

        except Exception as e:
            context.add_error(f"Compilation failed: {e}")
            raise

        finally:
            elapsed = time.time() - start_time
            context.add_metric("compile_time", elapsed)
            context.log(f"Compilation completed in {elapsed:.2f}s")

        return context

    def _compile_single_model_compiler(self, context: CompileContext) -> None:
        """Single-model compile via ``WinMLSession`` (``ort.ModelCompiler``)."""
        # Resolve session class from compiler config. "ort_inference_session" must not
        # reach here — it routes to _compile_multiple via context.use_inference_session.
        compiler = context.config.get("compiler", "ort")
        if compiler == "ort_inference_session":
            raise ValueError(
                "'ort_inference_session' is handled by the inference-session path, "
                "not the single-model ModelCompiler path."
            )
        session_cls = COMPILER_SESSION_MAPPING[compiler]

        # Determine final output directory (default: same as input model)
        output_dir = self._get_output_dir(context)
        context.log(f"Output directory: {output_dir}")

        # Ensure model is saved to disk (may be in work_dir if modified)
        model_path = self._ensure_model_file(context)
        context.log(f"Model path: {model_path}")

        ep_config = WinMLCompileConfig.from_dict(context.config).ep_config
        # Derive the target device from the runtime session so the compile
        # stage stays aligned with the actual EPContext filename produced by
        # WinMLSession instead of carrying device metadata in provider_options.
        device = context.config.get("device", "auto")
        explicit_ep = normalize_ep_name(ep_config.provider)
        session_cls_name = getattr(session_cls, "__name__", session_cls.__class__.__name__)
        context.log(f"Creating {session_cls_name} for device: {device}")
        winml_session = session_cls(
            onnx_path=model_path,
            device=device,
            ep_config=ep_config,
            ep=explicit_ep,
        )
        winml_session.compile()

        # Get the underlying session for validation and info collection
        session = winml_session._session
        context.session = session

        resolved_device = getattr(winml_session, "_device", device)
        if isinstance(resolved_device, str) and resolved_device:
            device = resolved_device.lower()

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
            self._finalize_output(context, model_path, output_dir, device=device)

    def _compile_multiple(self, context: CompileContext) -> None:
        """Multi-model / inference-session compile with a shared EP context.

        The shared ``SessionOptions`` (``context.shared_session_options``) is created on
        the first model — the EP is added once and, for a multi-model run, the
        ``ep.share_ep_contexts`` group is opened on it — then reused for every model.
        ``ep.stop_share_ep_contexts`` is added before the final model so the shared
        weights binary is flushed.
        """
        import onnxruntime as ort

        from ...sysinfo.device import resolve_device, resolve_eps
        from ...utils.constants import DEVICE_TO_DEVICE_TYPE
        from ...winml import add_ep_for_device, register_execution_providers

        ep_config = WinMLCompileConfig.from_dict(context.config).ep_config
        multi = context.n_total_models > 1
        is_last = context.n_compiled_models >= context.n_total_models - 1
        use_is = context.use_inference_session

        output_dir = self._get_output_dir(context)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = self._ensure_model_file(context)
        # Honor an explicit output filename (e.g. the de-duplicated <stem>_ctx.onnx
        # that compile_multiple_onnx assigns); otherwise derive it from the model stem.
        user_output = context.config.get("output_path")
        if user_output and Path(user_output).suffix == ".onnx":
            ctx_path = Path(user_output)
        else:
            ctx_path = output_dir / f"{context.model_path.stem}_ctx.onnx"
        backend = "inference_session" if use_is else "model_compiler"
        context.log(
            f"[{backend}] compiling {model_path.name} "
            f"({context.n_compiled_models + 1}/{context.n_total_models}) -> {ctx_path.name}"
        )

        # Build the shared SessionOptions once; reuse it for subsequent models.
        sess_options = context.shared_session_options
        if sess_options is None:
            register_execution_providers(ort=True)
            resolved_device, _ = resolve_device(context.config.get("device", "auto"))
            ep = normalize_ep_name(ep_config.provider) or resolve_eps(resolved_device)[0]
            device_type = DEVICE_TO_DEVICE_TYPE.get(resolved_device.upper())

            sess_options = ort.SessionOptions()
            if use_is:
                sess_options.add_session_config_entry("ep.context_enable", "1")
                sess_options.add_session_config_entry(
                    "ep.context_embed_mode", "1" if ep_config.embed_context else "0"
                )
            if multi:
                sess_options.add_session_config_entry("ep.share_ep_contexts", "1")
            if not add_ep_for_device(
                sess_options, ep, device_type, dict(ep_config.provider_options)
            ):
                raise RuntimeError(f"Could not add {ep} for device type {device_type}")
            context.shared_session_options = sess_options  # captured by Compiler for reuse

        # Last model in a shared run flushes the shared context.
        if multi and is_last:
            sess_options.add_session_config_entry("ep.stop_share_ep_contexts", "1")

        if use_is:
            # InferenceSession backend: ep.context_file_path writes the EPContext
            # wrapper; constructing the session performs the compile.
            sess_options.add_session_config_entry("ep.context_file_path", str(ctx_path))
            session = ort.InferenceSession(str(model_path), sess_options=sess_options)
            context.session = session
            if session.get_providers():
                context.log(f"Actual providers: {session.get_providers()}")
            # Models compiled this way are loadable; validate (run) when requested.
            if context.validate:
                self._validate_model(session, context)
            # Collect I/O info regardless of validation.
            self._collect_model_info(session, context)
        else:
            # ModelCompiler backend: compile straight to the EPContext file. No
            # session is created here (smoke path — outputs are checked, not loaded).
            ort.ModelCompiler(
                sess_options,
                str(model_path),
                embed_compiled_data_into_model=ep_config.embed_context,
            ).compile_to_file(str(ctx_path))

        if ctx_path.exists():
            context.output_path = ctx_path
            bins = [
                f
                for f in output_dir.glob(f"{ctx_path.stem}*.bin")
                if not f.name.endswith("_schematic.bin")
            ]
            if bins:
                context.context_binary_path = bins[0]
        else:
            context.add_warning(f"No EPContext produced for {model_path.name}")

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
        inputs: dict[str, np.ndarray] = {}

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

    def _finalize_output(
        self,
        context: CompileContext,
        model_path: Path,
        output_dir: Path,
        *,
        device: str | None = None,
    ) -> None:
        """Find EPContext files and copy to output directory.

        WinMLSession saves to work_dir. This method copies the output
        to the final output directory (default: same as input model).
        """
        # Find EPContext in work_dir (where WinMLSession saved it).
        # Prefer the runtime-resolved device, then fall back to generic and globbed matches.
        ctx_patterns = []
        if device:
            ctx_patterns.append(model_path.parent / f"{model_path.stem}_{device}_ctx.onnx")
        ctx_patterns.append(model_path.parent / f"{model_path.stem}_ctx.onnx")
        ctx_patterns.extend(sorted(model_path.parent.glob(f"{model_path.stem}_*_ctx.onnx")))

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
            final_ctx_path = output_dir / src_ctx_path.name

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
