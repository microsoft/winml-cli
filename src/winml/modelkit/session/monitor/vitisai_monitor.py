"""VitisAIMonitor - AMD NPU proof-of-execution monitor via xrt-smi.

Captures xrt-smi snapshots at start/end of monitoring for command
submission deltas — definitive proof that the AMD NPU executed work.

For real-time utilization/memory/CPU/RAM metrics, use HWMonitor
(which always runs alongside EPMonitors).
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

from .ep_monitor import EPMonitor


if TYPE_CHECKING:
    from typing_extensions import Self


logger = logging.getLogger(__name__)


class VitisAIMonitor(EPMonitor):
    """AMD NPU proof-of-execution monitor via xrt-smi.

    Captures command submission/completion deltas via xrt-smi snapshots
    to definitively prove that the AMD NPU executed work.

    For real-time utilization and memory metrics, use ``HWMonitor``
    (which always runs alongside via ``--monitor``).

    Example::

        with VitisAIMonitor() as ep:
            # ... run inference on AMD NPU ...
            pass

        print(ep.command_submissions)
        print(ep.npu_proven)
    """

    def __init__(self) -> None:
        """Initialize the monitor."""
        # xrt-smi snapshots
        self._xrt_client: Any | None = None
        self._submissions_before: int = 0
        self._submissions_after: int = 0
        self._completions_before: int = 0
        self._completions_after: int = 0
        self._last_hw_status: str = "Unknown"

    def __enter__(self) -> Self:
        """Start monitoring: xrt-smi snapshot (before)."""
        self._xrt_start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Stop monitoring: xrt-smi snapshot (after)."""
        self._xrt_stop()

    # ------------------------------------------------------------------
    # Proof-of-execution metrics (from xrt-smi)
    # ------------------------------------------------------------------

    @property
    def command_submissions(self) -> int:
        """Delta of command_submissions between start and end."""
        return max(0, self._submissions_after - self._submissions_before)

    @property
    def command_completions(self) -> int:
        """Delta of command_completions between start and end."""
        return max(0, self._completions_after - self._completions_before)

    @property
    def hw_context_status(self) -> str:
        """Last observed hardware context status (Idle/Running)."""
        return self._last_hw_status

    @property
    def npu_proven(self) -> bool:
        """True if NPU definitively did work during monitoring.

        Proven if xrt-smi shows command submissions increased.
        """
        return self.command_submissions > 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Whether this monitor can work on the current system.

        Requires Windows + xrt-smi.exe present (AMD NPU driver).
        """
        if sys.platform != "win32":
            return False
        try:
            from ._xrt_smi import XrtSmiClient

            return XrtSmiClient().is_available
        except (ImportError, RuntimeError):
            return False

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary of xrt-smi proof metrics."""
        return {
            "ep": "VitisAI",
            "npu_proven": self.npu_proven,
            "xrt_smi": {
                "command_submissions": self.command_submissions,
                "command_completions": self.command_completions,
                "hw_context_status": self.hw_context_status,
            },
        }

    # ------------------------------------------------------------------
    # Internal: xrt-smi lifecycle
    # ------------------------------------------------------------------

    def _xrt_start(self) -> None:
        """Take xrt-smi snapshot before monitoring."""
        try:
            from ._xrt_smi import XrtSmiClient

            self._xrt_client = XrtSmiClient()
            if not self._xrt_client.is_available:
                logger.debug("xrt-smi not available; skipping snapshot")
                return

            import os

            pid = os.getpid()
            self._submissions_before = self._xrt_client.get_command_submissions(pid)
            self._completions_before = self._xrt_client.get_command_completions(pid)
            logger.debug(
                "xrt-smi before: submissions=%d, completions=%d",
                self._submissions_before,
                self._completions_before,
            )
        except (ImportError, OSError) as exc:
            logger.debug("xrt-smi start failed: %s", exc)

    def _xrt_stop(self) -> None:
        """Take xrt-smi snapshot after monitoring."""
        if self._xrt_client is None or not self._xrt_client.is_available:
            return

        try:
            import os

            pid = os.getpid()
            self._submissions_after = self._xrt_client.get_command_submissions(pid)
            self._completions_after = self._xrt_client.get_command_completions(pid)

            # Get last status
            contexts = self._xrt_client.get_hw_contexts(pid)
            if contexts:
                self._last_hw_status = contexts[-1].status

            logger.debug(
                "xrt-smi after: submissions=%d, completions=%d, status=%s",
                self._submissions_after,
                self._completions_after,
                self._last_hw_status,
            )
        except (ImportError, OSError) as exc:
            logger.debug("xrt-smi stop failed: %s", exc)
