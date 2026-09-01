# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compile stage - generate EPContext model."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
from filelock import FileLock
from onnx import AttributeProto

from ...onnx import load_onnx, save_onnx
from ...onnx.epcontext import select_main_epcontext_partition_name
from ...session import (
    EPDeviceTarget,
    WinMLEPRegistry,
    WinMLSession,
    resolve_device,
)
from ...utils.constants import ORT_SESSION_COMPILER
from ..configs import WinMLCompileConfig
from .base import BaseStage


if TYPE_CHECKING:
    import onnxruntime as ort
    from onnx import ModelProto

    from ...utils.constants import EPAlias
    from ..context import CompileContext


COMPILER_SESSION_MAPPING: dict[str, type[WinMLSession] | tuple[str, str]] = {
    "ort": WinMLSession,
    "qairt": ("...session.qairt.qairt_session", "WinMLQairtSession"),
}


def _session_class_for_compiler(compiler: str) -> type[WinMLSession]:
    """Return the selected session backend without importing alternatives."""
    target = COMPILER_SESSION_MAPPING[compiler]
    if not isinstance(target, tuple):
        return target

    import importlib

    module_path, attr_name = target
    module = importlib.import_module(module_path, __package__)
    return cast("type[WinMLSession]", getattr(module, attr_name))


_FINALIZE_THREAD_LOCKS_GUARD = threading.Lock()
_FINALIZE_THREAD_LOCKS: dict[Path, threading.Lock] = {}


def _finalize_thread_lock(lock_path: Path) -> threading.Lock:
    """Return the process-local lock paired with one public output path."""
    resolved = lock_path.resolve(strict=False)
    with _FINALIZE_THREAD_LOCKS_GUARD:
        return _FINALIZE_THREAD_LOCKS.setdefault(resolved, threading.Lock())


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
            if context.use_inference_session or (
                context.enable_ep_context and context.n_total_models > 1
            ):
                self._compile_shared_context(context)
            else:
                self._compile_single_model(context)

        except Exception as e:
            context.add_error(f"Compilation failed: {e}")
            raise

        finally:
            elapsed = time.time() - start_time
            context.add_metric("compile_time", elapsed)
            context.log(f"Compilation completed in {elapsed:.2f}s")

        return context

    def _compile_single_model(self, context: CompileContext) -> None:
        """Compile one model through the configured ModelCompiler or QAIRT backend."""
        compiler = context.config.get("compiler", "ort")
        if compiler == ORT_SESSION_COMPILER:
            raise ValueError(f"{ORT_SESSION_COMPILER!r} requires the inference-session path")
        session_cls = _session_class_for_compiler(compiler)

        output_dir = self._get_output_dir(context)
        context.log(f"Output directory: {output_dir}")
        model_path = self._ensure_model_file(context)
        context.log(f"Model path: {model_path}")

        compile_cfg = WinMLCompileConfig.from_dict(context.config)
        ep_config = compile_cfg.ep_config
        ep_device_dict = context.config.get("ep_device")
        if ep_device_dict:
            target: EPDeviceTarget = EPDeviceTarget.from_dict(ep_device_dict)
        elif compile_cfg.ep_device is not None:
            target = compile_cfg.ep_device
        else:
            ep_str = context.execution_provider
            target = resolve_device(EPDeviceTarget(ep=ep_str or "auto", device=ep_config.device))
        ep_device = WinMLEPRegistry.instance().auto_device(target)
        session_cls_name = getattr(session_cls, "__name__", session_cls.__class__.__name__)
        context.log(f"Creating {session_cls_name} for {target.ep}/{target.device}")
        winml_session = session_cls(
            onnx_path=model_path,
            ep_device=ep_device,
            ep_config=ep_config,
        )
        try:
            winml_session.compile()
            running_model_path = winml_session.running_model_path

            session = winml_session._session
            context.session = session
            if session is not None:
                context.log(f"Actual providers: {session.get_providers()}")
                if context.validate:
                    self._validate_model(session, context)
                self._collect_model_info(session, context)

            if ep_config.enable_ep_context:
                if running_model_path == model_path:
                    context.add_warning(f"No EPContext produced for {model_path.name}")
                    return
                self._finalize_output(
                    context,
                    model_path,
                    output_dir,
                    device=ep_device.device.device_type.lower(),
                    src_ctx_path=running_model_path,
                )
        finally:
            context.session = None
            winml_session.reset()

    def _compile_shared_context(self, context: CompileContext) -> None:
        """Compile through shared SessionOptions for multi-model and ORT-session flows."""
        import onnxruntime as ort

        from ...session.session import _build_session_options

        compile_cfg = WinMLCompileConfig.from_dict(context.config)
        ep_config = compile_cfg.ep_config
        ep_device_dict = context.config.get("ep_device")
        if ep_device_dict:
            target = EPDeviceTarget.from_dict(ep_device_dict)
        elif compile_cfg.ep_device is not None:
            target = compile_cfg.ep_device
        else:
            target = resolve_device(
                EPDeviceTarget(ep=context.execution_provider or "auto", device=ep_config.device)
            )
        ep_device = WinMLEPRegistry.instance().auto_device(target)
        model_path = self._ensure_model_file(context)
        output_dir = self._get_output_dir(context)
        output_dir.mkdir(parents=True, exist_ok=True)
        configured_output = context.config.get("output_path")
        ctx_path = (
            Path(configured_output)
            if configured_output and Path(configured_output).suffix
            else output_dir / f"{context.model_path.stem}_ctx.onnx"
        )

        session_options = context.shared_session_options
        if session_options is None:
            session_options = _build_session_options(ep_device, ep_config)
            session_options.add_session_config_entry("ep.context_enable", "1")
            session_options.add_session_config_entry(
                "ep.context_embed_mode", "1" if ep_config.embed_context else "0"
            )
            if context.n_total_models > 1:
                session_options.add_session_config_entry("ep.share_ep_contexts", "1")
            context.shared_session_options = session_options

        if context.n_total_models > 1 and context.n_compiled_models == context.n_total_models - 1:
            session_options.add_session_config_entry("ep.stop_share_ep_contexts", "1")

        if context.use_inference_session:
            session_options.add_session_config_entry("ep.context_file_path", str(ctx_path))
            session = ort.InferenceSession(str(model_path), sess_options=session_options)
            try:
                context.session = session
                context.log(f"Actual providers: {session.get_providers()}")
                if context.validate:
                    self._validate_model(session, context)
                self._collect_model_info(session, context)
            finally:
                context.session = None
                del session
        else:
            ort.ModelCompiler(
                session_options,
                str(model_path),
                embed_compiled_data_into_model=ep_config.embed_context,
            ).compile_to_file(str(ctx_path))

        if ctx_path.exists():
            context.output_path = ctx_path
            binaries = [
                path
                for path in ctx_path.parent.glob(f"{ctx_path.stem}*.bin")
                if not path.name.endswith("_schematic.bin")
            ]
            if binaries:
                context.context_binary_path = binaries[0]
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
        src_ctx_path: Path | None = None,
    ) -> None:
        """Find EPContext files and copy to output directory.

        WinMLSession saves to work_dir. This method copies the output
        to the final output directory (default: same as input model).
        """
        # _finalize_output only runs after a successful compile, where the
        # target EP is always set; narrow away the Optional for the type checker
        # (a None here would already have failed the compile upstream).
        execution_provider = cast("EPAlias", context.execution_provider)
        output_suffix = execution_provider.lower()

        # WinMLSession.compile() saves ctx as {stem}_{ep_device.device}_ctx.onnx
        # (e.g. _npu_ctx.onnx), while context.execution_provider is the full
        # provider name (e.g. QNNExecutionProvider).  We search both patterns.
        from ...session import ep_to_device

        try:
            ep_device_str = ep_to_device(execution_provider)
        except ValueError:
            ep_device_str = None

        # Find EPContext in work_dir (where WinMLSession saved it)
        ctx_patterns = []
        if device:
            ctx_patterns.append(model_path.parent / f"{model_path.stem}_{device.lower()}_ctx.onnx")
        if ep_device_str:
            ctx_patterns.append(
                model_path.parent / f"{model_path.stem}_{ep_device_str.lower()}_ctx.onnx"
            )
        ctx_patterns.extend(
            [
                model_path.parent / f"{model_path.stem}_{output_suffix}_ctx.onnx",
                model_path.parent / f"{model_path.stem}_ctx.onnx",
            ]
        )

        if src_ctx_path is None:
            for pattern in ctx_patterns:
                if pattern.exists():
                    src_ctx_path = pattern
                    break

        if src_ctx_path is None:
            context.add_warning("EPContext model not found in work directory")
            return

        # An explicit output file takes precedence; directory output preserves
        # the established derived filename behavior.
        original_stem = context.model_path.stem
        configured_output = context.config.get("output_path")
        if configured_output and Path(configured_output).suffix:
            final_ctx_path = Path(configured_output)
        else:
            final_ctx_path = output_dir / f"{original_stem}_{output_suffix}_ctx.onnx"

        publish_lock = final_ctx_path.with_name(f"{final_ctx_path.name}.publish.lock")
        with _finalize_thread_lock(publish_lock), FileLock(publish_lock):
            self._publish_finalized_output(
                context,
                src_ctx_path,
                final_ctx_path,
                output_dir,
            )

    def _publish_finalized_output(
        self,
        context: CompileContext,
        src_ctx_path: Path,
        final_ctx_path: Path,
        output_dir: Path,
    ) -> None:
        """Publish a self-consistent EPContext bundle while holding its output lock."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Validate every external context reference before publishing the final
        # ONNX. The compiler work directory is temporary, so a skipped reference
        # would leave a success-shaped model pointing at a deleted sidecar.
        model = load_onnx(src_ctx_path, validate=False)
        external_refs: dict[bytes, list[AttributeProto]] = {}
        for node in model.graph.node:
            if node.op_type != "EPContext":
                continue
            attrs = {a.name: a for a in node.attribute}
            embed_mode = attrs.get("embed_mode")
            main_context = attrs.get("main_context")
            ep_cache_context = attrs.get("ep_cache_context")
            if embed_mode is None:
                continue
            if embed_mode.type != AttributeProto.INT:
                raise ValueError("EPContext embed_mode must be an integer")
            if embed_mode.i != 0:
                continue
            is_secondary = (
                main_context is not None
                and main_context.type == AttributeProto.INT
                and main_context.i == 0
            )
            if ep_cache_context is None and is_secondary:
                continue
            if ep_cache_context is None or ep_cache_context.type != AttributeProto.STRING:
                raise ValueError("External EPContext node must have a string ep_cache_context")
            external_refs.setdefault(ep_cache_context.s, []).append(ep_cache_context)

        source_root = src_ctx_path.parent.resolve()
        output_root = output_dir.resolve()
        binary_exports: list[tuple[Path, Path, Path, bytes, list[AttributeProto]]] = []
        sources_by_final_binary: dict[Path, Path] = {}
        for raw_ref, cache_attrs in external_refs.items():
            try:
                cache_ref = raw_ref.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("EPContext binary reference must be valid UTF-8") from exc

            relative_ref = Path(cache_ref)
            if not cache_ref or relative_ref.is_absolute() or relative_ref.drive:
                raise ValueError(f"unsafe EPContext binary reference: {cache_ref!r}")

            try:
                source_binary = (source_root / relative_ref).resolve()
                source_binary.relative_to(source_root)
            except (OSError, ValueError) as exc:
                raise ValueError(f"unsafe EPContext binary reference: {cache_ref!r}") from exc

            try:
                source_exists = source_binary.is_file()
            except OSError as exc:
                raise FileNotFoundError(
                    f"EPContext binary is unavailable: {source_binary}"
                ) from exc
            if not source_exists:
                raise FileNotFoundError(f"EPContext binary not found: {source_binary}")

            final_relative_ref = relative_ref
            if relative_ref.name.startswith(src_ctx_path.stem):
                suffix = relative_ref.name[len(src_ctx_path.stem) :]
                final_relative_ref = relative_ref.with_name(f"{final_ctx_path.stem}{suffix}")

            stable_binary = (output_root / final_relative_ref).resolve()
            try:
                stable_binary.relative_to(output_root)
            except ValueError as exc:
                raise ValueError(f"unsafe EPContext binary reference: {cache_ref!r}") from exc

            existing_source = sources_by_final_binary.get(stable_binary)
            if existing_source is not None and existing_source != source_binary:
                raise ValueError(
                    "Distinct EPContext binaries map to the same output path: "
                    f"{existing_source}, {source_binary} -> {stable_binary}"
                )
            sources_by_final_binary[stable_binary] = source_binary
            content_token = self._file_sha256(source_binary)[:16]
            unique_relative_ref = final_relative_ref.with_name(
                f"{final_relative_ref.stem}.{content_token}{final_relative_ref.suffix}"
            )
            unique_binary = (output_root / unique_relative_ref).resolve()
            try:
                unique_binary.relative_to(output_root)
            except ValueError as exc:
                raise ValueError(f"unsafe EPContext binary reference: {cache_ref!r}") from exc
            binary_exports.append(
                (
                    source_binary,
                    unique_binary,
                    stable_binary,
                    unique_relative_ref.as_posix().encode("utf-8"),
                    cache_attrs,
                )
            )

        first_final_binary: Path | None = None
        for (
            source_binary,
            unique_binary,
            stable_binary,
            final_ref_bytes,
            cache_attrs,
        ) in binary_exports:
            self._atomic_copy(source_binary, unique_binary)
            self._atomic_copy(source_binary, stable_binary)
            context.log(f"Published binary generation: {unique_binary}")
            if first_final_binary is None:
                first_final_binary = unique_binary

            for cache_attr in cache_attrs:
                cache_attr.s = final_ref_bytes

        self._atomic_save_onnx(model, final_ctx_path)
        context.log(f"Published EPContext: {final_ctx_path}")

        context.output_path = final_ctx_path
        context.context_binary_path = first_final_binary

        partition_name = select_main_epcontext_partition_name(final_ctx_path)
        if partition_name is not None:
            schematic_name = f"{partition_name}_schematic.bin"
            src_schematic = src_ctx_path.parent / schematic_name
            final_schematic = output_dir / schematic_name
            if src_schematic.is_file() and src_schematic != final_schematic:
                self._atomic_copy(src_schematic, final_schematic)
                context.log(f"Copied schematic to: {final_schematic}")

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        """Copy one file through a same-directory temporary and atomic replace."""
        if source.resolve() == destination.resolve(strict=False):
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            shutil.copy2(source, temporary_path)
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _atomic_save_onnx(model: ModelProto, destination: Path) -> None:
        """Save an ONNX model beside its destination and atomically replace it."""
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.", suffix=destination.suffix, dir=destination.parent
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        temporary_path.unlink(missing_ok=True)
        try:
            save_onnx(model, temporary_path)
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)

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
