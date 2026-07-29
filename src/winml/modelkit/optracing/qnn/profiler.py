# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Deprecated QNN op-tracing runner built on the current session monitor stack."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from .._compat import warn_deprecated
from ..base import _OpTracer


if TYPE_CHECKING:
    from ..result import OpTraceResult


logger = logging.getLogger(__name__)


def _ort_type_to_numpy(ort_type: str) -> np.dtype:
    """Map an ORT tensor type string to a NumPy dtype."""
    mapping: dict[str, np.dtype] = {
        "tensor(float)": np.dtype("float32"),
        "tensor(float16)": np.dtype("float16"),
        "tensor(double)": np.dtype("float64"),
        "tensor(int32)": np.dtype("int32"),
        "tensor(int64)": np.dtype("int64"),
        "tensor(int8)": np.dtype("int8"),
        "tensor(uint8)": np.dtype("uint8"),
        "tensor(bool)": np.dtype("bool"),
    }
    return mapping.get(ort_type, np.dtype("float32"))


def _resolve_shape(shape: list[Any], default_dim: int = 1) -> list[int]:
    """Replace symbolic or ``None`` dimensions with concrete values."""
    return [
        default_dim if not isinstance(dimension, int) or dimension <= 0 else dimension
        for dimension in shape
    ]


class QNNProfiler(_OpTracer):
    """Legacy OpTracer-compatible wrapper around ``WinMLSession.perf()``."""

    def is_available(self) -> bool:
        """Return whether the current session stack can drive QNN op tracing."""
        from ...session.monitor.qnn_monitor import QNNMonitor

        return QNNMonitor.is_available()

    def run(self, iterations: int = 5, warmup: int = 2) -> OpTraceResult:
        """Trace a QNN-backed session and return the structured op-tracing result."""
        from ...compiler import EPConfig
        from ...session import WinMLSession
        from ...session.monitor.qnn_monitor import QNNMonitor

        level = self.level.lower()
        if level not in ("basic", "detail"):
            raise ValueError(f"level must be 'basic' or 'detail', got {self.level!r}")

        monitor = QNNMonitor(
            level=cast("Literal['basic', 'detail']", level),
            output_dir=self.output_dir,
        )
        ep_config = EPConfig(provider="qnn") if level == "detail" else None
        session = WinMLSession(
            self.onnx_path,
            device="npu",
            ep="qnn",
            ep_config=ep_config,
        )
        if level == "detail":
            from ...onnx import is_compiled_onnx

            session.compile()
            if not is_compiled_onnx(session.running_model_path):
                raise RuntimeError(
                    "QNN detail profiling requires an EPContext model, but compilation "
                    f"fell back to the original ONNX model: {session.running_model_path}"
                )

        with session.perf(warmup=warmup, monitor=monitor) as ctx:
            inputs = self._resolve_inputs(session)
            for _ in range(iterations + warmup):
                session.run(inputs)

        result = ctx.monitor.result
        if result is None:
            raise RuntimeError("QNN op-tracing completed without producing a result.")
        return result

    def _resolve_inputs(self, session: Any) -> dict[str, np.ndarray]:
        runtime_session = getattr(session, "_session", None)
        if runtime_session is None:
            raise RuntimeError(
                "WinMLSession.perf() did not build an inference session for tracing."
            )

        if not self.input_data:
            return self._generate_inputs(runtime_session)

        expected = {inp.name: _ort_type_to_numpy(inp.type) for inp in runtime_session.get_inputs()}
        if set(self.input_data) != set(expected):
            logger.warning(
                "--input-data inputs %s do not match the traced model's inputs %s; "
                "falling back to random inputs for op-tracing.",
                sorted(self.input_data),
                sorted(expected),
            )
            return self._generate_inputs(runtime_session)

        return {
            name: np.asarray(array).astype(expected[name])
            for name, array in self.input_data.items()
        }

    @staticmethod
    def _generate_inputs(runtime_session: Any) -> dict[str, np.ndarray]:
        inputs: dict[str, np.ndarray] = {}
        for inp in runtime_session.get_inputs():
            shape = _resolve_shape(inp.shape)
            dtype = _ort_type_to_numpy(inp.type)
            inputs[inp.name] = np.random.rand(*shape).astype(dtype)
        return inputs


def __getattr__(name: str) -> Any:
    if name != "QNNProfiler":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    warn_deprecated(
        "qnn.profiler.QNNProfiler",
        "winml.modelkit.session.monitor.QNNMonitor",
        stacklevel=2,
    )
    globals()[name] = _QNNProfiler
    return _QNNProfiler


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = ["QNNProfiler"]


_QNNProfiler = QNNProfiler
del QNNProfiler

if TYPE_CHECKING:
    QNNProfiler = _QNNProfiler  # type: ignore[misc]  # Static-only compatibility export.
