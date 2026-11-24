# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compile-only session for an already-bound off-device compiler host."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, NoReturn

import onnxruntime as ort

from ..compiler.offdevice import _compile_bound_model, _validate_compilation_output
from ..onnx import is_compiled_onnx
from ..utils.constants import normalize_ep_name
from .session import SessionState, WinMLSession


if TYPE_CHECKING:
    from ..compiler import EPConfig
    from .ep_registry import WinMLEPDevice


class WinMLOffDeviceSession(WinMLSession):
    """Produce an embedded EPContext without creating an inference session."""

    def __init__(
        self,
        onnx_path: str | Path,
        ep_device: WinMLEPDevice,
        ep_config: EPConfig,
    ) -> None:
        if not ep_config.enable_ep_context or not ep_config.embed_context:
            raise ValueError("Off-device compilation requires embedded EPContext output.")

        provider = ep_device.ep.source.ep_name
        native_device = ep_device.device.ort_handle
        if provider not in {
            "OpenVINOExecutionProvider",
            "QNNExecutionProvider",
            "VitisAIExecutionProvider",
        }:
            raise ValueError(f"Unsupported off-device provider: {provider}")
        if native_device.ep_name != provider or (
            ep_config.provider is not None
            and normalize_ep_name(ep_config.provider) != provider
        ):
            raise ValueError("The compilation configuration and bound provider must match.")

        target = ep_config.target_platform
        if provider == "QNNExecutionProvider":
            if native_device.device.type != ort.OrtHardwareDeviceType.CPU:
                raise ValueError("QNN off-device compilation requires a CPU compilation host.")
            if (
                not isinstance(target, str)
                or not target.isascii()
                or not target.isdecimal()
                or int(target) <= 0
            ):
                raise ValueError("QNN target_platform must be a positive numeric soc_model ID.")
        else:
            if (
                native_device.device.type != ort.OrtHardwareDeviceType.NPU
                or native_device.device.metadata.get("is_virtual") != "1"
            ):
                raise ValueError("Off-device compilation requires a virtual NPU device.")
            if provider == "VitisAIExecutionProvider":
                if target is not None:
                    raise ValueError(
                        "VitisAI uses provider_options['target'], not target_platform."
                    )
                target = ep_config.provider_options.get("target")
                if target != "X2":
                    raise ValueError("VitisAI off-device compilation requires target=X2.")
            elif not target or native_device.ep_metadata.get("npu_platform") != target:
                raise ValueError("OpenVINO target_platform must match the bound virtual NPU.")

        library_path = native_device.ep_metadata.get("library_path")
        if not library_path or Path(library_path).resolve() != ep_device.dll_path.resolve():
            raise ValueError("The compilation device must belong to the selected payload.")

        allowed_options = {"target"} if provider == "VitisAIExecutionProvider" else set()
        if (
            ep_config.provider_option_file_keys
            or set(ep_config.provider_options) - allowed_options
        ):
            raise ValueError("Unsupported off-device provider options.")

        super().__init__(onnx_path, ep_device, ep_config=ep_config)
        self._target = target
        self._compile_dir: TemporaryDirectory[str] | None = None

    def compile(self) -> None:
        """Compile and check the artifact, without running target-device inference."""
        if self.is_compiled:
            return
        try:
            if not self._onnx_path.is_file():
                raise FileNotFoundError(f"ONNX model not found: {self._onnx_path}")
            if is_compiled_onnx(self._onnx_path):
                raise ValueError("The input model already contains EPContext nodes.")

            self._compile_dir = TemporaryDirectory(prefix="winml-offdevice-")
            output_path = Path(self._compile_dir.name) / f"{self._onnx_path.stem}_ctx.onnx"
            _compile_bound_model(
                self._onnx_path,
                output_path,
                self._target,
                entry=self._ep_device.ep.source,
                device=self._ep_device.device.ort_handle,
            )
            _validate_compilation_output(output_path, self._ep)
            self._running_model_path = output_path
            self._state = SessionState.COMPILED
            self._last_error = None
        except Exception as error:
            self._state = SessionState.ERROR
            self._last_error = error
            if self._compile_dir is not None:
                self._compile_dir.cleanup()
                self._compile_dir = None
            raise

    @property
    def is_compiled(self) -> bool:
        """Report artifact compilation independently of an inference session."""
        return self._state == SessionState.COMPILED

    def run(self, inputs: dict[str, Any]) -> NoReturn:
        """Reject inference on this compile-only session."""
        raise RuntimeError("Off-device sessions do not run inference.")

    def reset(self) -> None:
        """Reset compilation state and remove the private staged artifact."""
        super().reset()
        if self._compile_dir is not None:
            self._compile_dir.cleanup()
            self._compile_dir = None
