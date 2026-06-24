# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""OpenVINOSession - inference session backed by OpenVINO Runtime.

Mirrors the subset of :class:`WinMLSession`'s surface that the perf
benchmark engine relies on (``compile`` / ``run`` / ``perf`` plus the
``io_config`` / ``device`` / ``ep_name`` / ``running_model_path``
properties), so a ``winml perf`` run can swap ONNX Runtime for OpenVINO on
the same ONNX file for an apples-to-apples ORT-vs-OV comparison.

ONNX input only: OpenVINO reads the provided ``.onnx`` directly (no
quantize / optimize / compile build).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ...core.onnx_utils import get_io_config
from ..stats import PerfStats


if TYPE_CHECKING:
    from collections.abc import Generator

    from ...utils.constants import EPName


logger = logging.getLogger(__name__)


# ModelKit device policy -> OpenVINO device name. "auto" maps to OpenVINO's
# AUTO plugin, which picks the best available device at compile time.
_OV_DEVICE_MAP = {
    "cpu": "CPU",
    "gpu": "GPU",
    "npu": "NPU",
    "auto": "AUTO",
}

# Canonical EP name surfaced to perf reporting/JSON so OpenVINO runs are
# labeled consistently with the ORT path's provider names.
_OV_EP_NAME = "OpenVINOExecutionProvider"


class OpenVINOSession:
    """ONNX inference session backed by OpenVINO Runtime.

    One session loads and runs a single ONNX file on one OpenVINO device.
    The runtime is imported lazily in :meth:`compile` so importing this
    module never pulls in the ``openvino`` package.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        device: str = "auto",
        provider_options: dict[str, str] | None = None,
    ) -> None:
        """Initialize the session.

        Args:
            onnx_path: Path to the ONNX model file.
            device: Target device policy ("auto", "cpu", "gpu", "npu"),
                mapped to an OpenVINO device name.
            provider_options: OpenVINO config properties forwarded to
                ``Core.compile_model`` (e.g. ``{"PERFORMANCE_HINT":
                "LATENCY"}``). Empty/None means OpenVINO defaults.
        """
        self._onnx_path = Path(onnx_path)
        if not self._onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self._device = str(device).lower()
        self._provider_options = dict(provider_options) if provider_options else {}

        # Populated by compile()
        self._compiled: Any = None
        self._ov_device: str | None = None

        # Cached I/O metadata (lazy-loaded)
        self._io_config: dict | None = None

        # Performance tracking (enabled via perf() context manager)
        self._perf_stats: PerfStats | None = None

        # OpenVINO reads the original ONNX directly, so the running model is
        # always the input path (no EPContext / compiled artifact on disk).
        self._running_model_path = self._onnx_path

        logger.info("OpenVINOSession initialized: %s", onnx_path)

    def compile(self) -> None:
        """Read and compile the ONNX model with OpenVINO Runtime.

        Idempotent: compiles once per session.
        """
        if self._compiled is not None:
            logger.debug("Already compiled for %s", self._device)
            return

        import openvino as ov

        requested = _OV_DEVICE_MAP.get(self._device, self._device.upper())
        core = ov.Core()

        # Friendly fail-fast when the requested device isn't present, instead of
        # a raw backend stack trace from compile_model. AUTO is a virtual plugin
        # (always selectable); concrete devices may be listed plain ("GPU") or
        # indexed ("GPU.0"), so match on the base name.
        available = core.available_devices
        if requested != "AUTO" and not any(
            dev == requested or dev.startswith(f"{requested}.") for dev in available
        ):
            raise RuntimeError(
                f"OpenVINO device '{requested}' (from --device {self._device}) is "
                f"not available. OpenVINO sees: {available}"
            )

        model = core.read_model(str(self._onnx_path))
        self._compiled = core.compile_model(model, requested, self._provider_options)

        # AUTO resolves to a concrete device at compile time; record what was
        # actually selected for display, falling back to the requested name.
        try:
            devices = self._compiled.get_property("EXECUTION_DEVICES")
            self._ov_device = devices[0] if devices else requested
        except Exception:
            self._ov_device = requested

        logger.info(
            "OpenVINO compiled model on %s (requested %s), provider_options=%s",
            self._ov_device,
            requested,
            self._provider_options,
        )

    def run(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Run inference.

        Auto-compiles on first call. Validates and dtype-coerces inputs.

        Args:
            inputs: Input tensors (numpy arrays or torch tensors) keyed by
                input name.

        Returns:
            Dict of output name -> numpy array.
        """
        if not inputs:
            raise ValueError("inputs cannot be empty")

        if self._compiled is None:
            self.compile()
        compiled = self._compiled

        ov_inputs = self._prepare_inputs(inputs)

        if self._perf_stats is not None:
            result = self._perf_stats.record(lambda: compiled(ov_inputs))
        else:
            result = compiled(ov_inputs)

        # Map outputs back to graph order. Index keying avoids OpenVINO
        # output-name normalization mismatches (the order of model.outputs
        # matches the ONNX graph output order get_io_config reads).
        out_names = self.io_config["output_names"]
        return {name: np.asarray(result[i]) for i, name in enumerate(out_names)}

    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Convert inputs to numpy arrays and enforce model input dtypes."""
        io_cfg = self.io_config
        name_to_type = dict(zip(io_cfg["input_names"], io_cfg["input_types"], strict=True))

        ov_inputs: dict[str, np.ndarray] = {}
        for name, value in inputs.items():
            if hasattr(value, "numpy"):  # torch.Tensor
                arr = value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                arr = value
            else:
                arr = np.asarray(value)

            expected_type = name_to_type.get(name)
            if expected_type is not None and arr.dtype != expected_type:
                arr = arr.astype(expected_type)

            ov_inputs[name] = arr

        return ov_inputs

    @contextmanager
    def perf(self, warmup: int = 0) -> Generator[PerfStats, None, None]:
        """Context manager for scoped performance tracking.

        Mirrors :meth:`WinMLSession.perf` so the shared perf engine drives
        either backend identically.

        Args:
            warmup: Number of initial samples to exclude from statistics.

        Yields:
            PerfStats collecting timing data within the context.
        """
        self._perf_stats = PerfStats(warmup=warmup)
        try:
            yield self._perf_stats
        finally:
            self._perf_stats = None

    @property
    def io_config(self) -> dict:
        """ONNX I/O metadata (lazy-loaded, cached).

        Reuses the same extraction path as the ORT session so input/output
        names, shapes and dtypes are identical across runtimes.
        """
        if self._io_config is None:
            from ...onnx import load_onnx
            from ..session import WinMLSession

            model = load_onnx(self._onnx_path, load_weights=False, validate=False)
            self._io_config = get_io_config(model)
            # Reuse the operator-schema-based precision estimate (no
            # architecture assumptions) so reports match the ORT path.
            self._io_config["precision"] = WinMLSession._get_precision(model)
        return self._io_config

    @property
    def device(self) -> str:
        """Target device label for this session."""
        return self._device

    @property
    def ep_name(self) -> EPName | None:
        """Canonical EP name, or None before compile.

        Returns ``"OpenVINOExecutionProvider"`` once compiled so perf
        reporting labels OpenVINO runs consistently with ORT provider names.
        """
        if self._compiled is None:
            return None
        return cast("EPName", _OV_EP_NAME)

    @property
    def running_model_path(self) -> Path:
        """Path to the ONNX model OpenVINO loads (always the input path)."""
        return self._running_model_path

    @property
    def is_compiled(self) -> bool:
        """Whether the model has been compiled."""
        return self._compiled is not None
