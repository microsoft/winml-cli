# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLEPMonitor - Abstract base class for EP-specific hardware monitoring.

Defines the common interface that all EP hardware monitors implement.
Each subclass provides data collection for a specific execution provider
(VitisAI, MIGraphX, NvTensorRTRTX, etc.).

v2.4 additions (purely additive — see design spec
``docs/design/perf/2026-05-03-op-trace-parser-interface-spec.md`` §3.1 and
``docs/design/session/monitor/2_coreloop.md`` §4.1):

* :meth:`WinMLEPMonitor.set_onnx_op_types` — concrete no-op default; op-tracing
  monitors override to receive an injected ``node.name -> node.op_type`` map
  built once by :class:`WinMLSession`.
* :meth:`WinMLEPMonitor.set_perf_window` — concrete no-op default; monitors
  that emit one artifact sample per run can align their parsing with
  ``PerfStats`` warmup exclusion.
* :attr:`WinMLEPMonitor.result` — concrete property returning ``self._result`` if
  set, else ``None``. Op-tracing monitors populate ``self._result`` during
  ``__exit__`` parsing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar


if TYPE_CHECKING:
    from pathlib import Path
    from typing import Self

    from .op_metrics import OpTraceResult


class WinMLEPMonitor(ABC):
    """Base class for EP-specific hardware performance monitoring.

    Used as a context manager alongside ``PerfStats`` to collect
    hardware utilization metrics during inference.

    Example::

        with session.perf(warmup=10, monitor=SomeEPMonitor()) as ctx:
            for _ in range(110):
                session.run(inputs)

        print(ctx.stats.mean_ms)
        # Op-tracing monitors expose their data via the typed `result`
        # property (an :class:`OpTraceResult`); proof-of-execution monitors
        # (e.g. VitisAI, OpenVINO) currently expose theirs via ``to_dict()``
        # transitionally — to be replaced by a typed ``proof`` accessor in
        # a follow-up.
        if ctx.monitor.result is not None:
            print(ctx.monitor.result.to_dict())
    """

    # ---- Optional hooks: defaults provided; subclasses override as needed ----

    #: ORT-specific hint: does this monitor's data flush require
    #: ``ort.InferenceSession`` destruction? Example: QNN flushes CSV only
    #: on session destroy. Default: False (no teardown needed).
    requires_session_teardown: ClassVar[bool] = False

    #: Target EP short name (e.g. ``"qnn"``). When set, ``WinMLSession.perf()``
    #: pins the session to this EP so provider options contributed via
    #: :meth:`get_provider_options` actually flow through
    #: ``add_provider_for_devices``. Without this, sessions without an explicit
    #: ``ep`` fall back to ORT's policy-based selection which silently drops
    #: provider options. ``None`` (default) means the monitor doesn't require
    #: a specific EP — e.g. :class:`NullEPMonitor`, ``VitisAIMonitor`` whose
    #: hooks return empty dicts.
    ep_name: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject subclasses that try to shadow load-bearing class vars.

        The ``requires_session_teardown`` flag governs the C-2 teardown
        ordering invariant in ``WinMLSession.perf()``. Catching a non-bool
        shadow at class-definition time keeps the invariant *visible*;
        runtime instance shadowing in ``__init__`` is not catchable here.
        """
        super().__init_subclass__(**kwargs)
        cls_dict_value = cls.__dict__.get("requires_session_teardown")
        if cls_dict_value is not None and not isinstance(cls_dict_value, bool):
            raise TypeError(
                f"{cls.__name__}.requires_session_teardown must be a class-level bool, "
                f"got {type(cls_dict_value).__name__}"
            )

    def get_session_options(self) -> dict[str, str]:
        """Entries to pass to ``SessionOptions.add_session_config_entry()``.

        Default: empty dict. Override in subclasses that need e.g.
        ``"session.disable_cpu_ep_fallback": "1"``.
        """
        return {}

    def get_provider_options(self) -> dict[str, str]:
        """Options to merge into ``add_provider_for_devices([ep], opts)``.

        Default: empty dict. Override in subclasses that need e.g.
        ``"profiling_level": "detailed"``.
        """
        return {}

    def set_onnx_op_types(self, onnx_op_types: dict[str, str]) -> None:  # noqa: B027 - intentional no-op default; op-tracing monitors override
        """Inject the ONNX ``node.name -> node.op_type`` map.

        Default: no-op. Op-tracing monitors override this to store the map
        for use during their ``__exit__`` parsing pass. Non-op-tracing
        monitors (:class:`NullEPMonitor`, ``VitisAIMonitor``)
        inherit the default and ignore the call.

        Called unconditionally by :meth:`WinMLSession.perf` immediately
        before ``mon.__enter__()`` so the monitor has the map available
        for the entire lifetime of the perf window.
        """

    def set_onnx_model_path(self, onnx_model_path: Path) -> None:  # noqa: B027 - intentional no-op default; metadata monitors override
        """Inject the ONNX model path before monitor entry.

        Default: no-op. Monitors that optionally inspect the full graph
        override this hook; all other monitors remain unaffected.
        """

    def set_perf_window(self, warmup: int, measured_iterations: int) -> None:  # noqa: B027 - intentional no-op default; sampling monitors override
        """Provide counts from the completed :meth:`WinMLSession.perf` window.

        Default: no-op. Sampling monitors override this to exclude the
        completed warmup runs and validate the measured artifact count.
        """

    @property
    def result(self) -> OpTraceResult | None:
        """Wrapped op-trace result. ``None`` for monitors that don't produce one.

        Op-tracing monitors set ``self._result`` during ``__exit__`` after
        parsing. The default :func:`getattr` returns ``None`` for monitors
        that never set it (the no-op subclasses :class:`NullEPMonitor`,
        ``VitisAIMonitor``).
        """
        return getattr(self, "_result", None)

    # ---- Mandatory contract ----

    @abstractmethod
    def __enter__(self) -> Self:
        """Start hardware monitoring."""

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Stop hardware monitoring and finalize metrics."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Whether this monitor can work on the current system."""


class NullEPMonitor(WinMLEPMonitor):
    """No-op EP monitor (Null Object Pattern).

    Used when no vendor-specific EP monitor is available.
    Eliminates null checks in the benchmark loop.
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        pass

    @classmethod
    def is_available(cls) -> bool:
        """Always available (it does nothing)."""
        return True
