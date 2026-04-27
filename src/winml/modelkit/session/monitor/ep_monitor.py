# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""EPMonitor - Abstract base class for EP-specific hardware monitoring.

Defines the common interface that all EP hardware monitors implement.
Each subclass provides data collection for a specific execution provider
(VitisAI, MIGraphX, TensorRT, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar


if TYPE_CHECKING:
    from typing_extensions import Self


class EPMonitor(ABC):
    """Base class for EP-specific hardware performance monitoring.

    Used as a context manager alongside ``PerfStats`` to collect
    hardware utilization metrics during inference.

    Example::

        with session.perf(warmup=10, monitor=SomeEPMonitor()) as ctx:
            for _ in range(110):
                session.run(inputs)

        print(ctx.stats.mean_ms)
        print(ctx.monitor.to_dict())
    """

    # ---- Optional hooks: defaults provided; subclasses override as needed ----

    #: ORT-specific hint: does this monitor's data flush require
    #: ``ort.InferenceSession`` destruction? Example: QNN flushes CSV only
    #: on session destroy. Default: False (no teardown needed).
    requires_session_teardown: ClassVar[bool] = False

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

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary of all collected metrics."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Whether this monitor can work on the current system."""


class NullEPMonitor(EPMonitor):
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

    def to_dict(self) -> dict[str, Any]:
        """No-op: returns empty dict."""
        return {}
