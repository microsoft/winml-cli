# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""OpenVINOMonitor — Intel OpenVINO EP per-op profiler surface (unsupported).

The shipping OpenVINO EP wheels (Intel ``openvino-plugin-ep``) do not
implement the CSV-dump mechanism this monitor was originally written
against. Rather than return a working-looking monitor that silently
produces no-data JSON, the CLI layer refuses ``--op-tracing --ep openvino``
unconditionally (see ``commands.perf._select_ep_monitor``).

This class is therefore reduced to a stub: :meth:`is_available` returns
``False`` and :meth:`get_provider_options` is retained so callers that
happen to instantiate the monitor still get the owner-enforced
``PERF_COUNT: YES`` load-config for parity with the historical contract.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from .ep_monitor import WinMLEPMonitor


if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self


logger = logging.getLogger(__name__)

_VALID_DEVICES: frozenset[str] = frozenset({"CPU", "GPU", "NPU", "AUTO"})


class OpenVINOMonitor(WinMLEPMonitor):
    """OpenVINO EP per-op profiler surface — currently unsupported.

    :meth:`is_available` returns ``False`` unconditionally because the
    shipping OpenVINO EP wheels do not produce the profiling CSVs the
    original implementation consumed. See module docstring for context.
    """

    requires_session_teardown: ClassVar[bool] = False
    ep_name: ClassVar[str | None] = "openvino"

    def __init__(
        self,
        level: str = "basic",
        device: str = "AUTO",
        extra_provider_options: Mapping[str, str] | None = None,
    ) -> None:
        """Validate ``level`` / ``device`` and stash provider option extras.

        Args:
            level: Only ``"basic"`` is accepted; anything else raises.
            device: One of ``"CPU"``, ``"GPU"``, ``"NPU"``, ``"AUTO"``.
            extra_provider_options: Merged into provider options.
                ``PERF_COUNT`` inside ``load_config`` is owner-enforced.
        """
        if level != "basic":
            raise ValueError(f"OpenVINOMonitor only supports level='basic', got {level!r}")
        if device not in _VALID_DEVICES:
            raise ValueError(f"device must be one of {sorted(_VALID_DEVICES)}, got {device!r}")
        self._level: str = level
        self._device: str = device
        self._extra: dict[str, str] = dict(extra_provider_options or {})

    @classmethod
    def is_available(cls) -> bool:
        """Always ``False``: this monitor is not a working per-op tracer.

        Kept as a ``@classmethod`` returning ``False`` so callers that
        probe availability get a coherent answer without having to know
        about the CLI-layer refusal.
        """
        return False

    def get_provider_options(self) -> dict[str, str]:
        """Provider options for OpenVINO EP with owner-enforced PERF_COUNT.

        Owner-enforces ``PERF_COUNT: YES`` inside the ``load_config`` JSON
        for the target device; callers cannot disable or weaken it.
        """
        opts: dict[str, str] = dict(self._extra)
        existing: dict[str, Any] = {}
        raw_lc = opts.get("load_config")
        if raw_lc:
            try:
                existing = json.loads(raw_lc)
            except (json.JSONDecodeError, TypeError, ValueError):
                existing = {}
        device_cfg: dict[str, Any] = dict(existing.get(self._device, {}))
        device_cfg["PERF_COUNT"] = "YES"
        existing[self._device] = device_cfg
        opts["load_config"] = json.dumps(existing)
        return opts

    # ------------------------------------------------------------------ #
    # Context-manager surface — satisfies WinMLEPMonitor's abstract      #
    # contract even though this monitor is not a working per-op tracer.  #
    # Kept as no-ops so callers that instantiate the class don't blow up #
    # on ``with`` before hitting the CLI-layer refusal.                  #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        return None
