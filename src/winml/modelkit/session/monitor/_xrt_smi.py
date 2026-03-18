# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""xrt-smi subprocess wrapper for AMD NPU hardware context monitoring.

Provides a clean Python interface to ``xrt-smi.exe``, the AMD NPU system
management tool installed with the NPU driver. Used to capture hardware
context snapshots (command submissions, completions, status) for
proof-of-execution verification.

Internal module — not part of the public API.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# Default xrt-smi location (installed with AMD NPU driver).
_XRT_SMI_PATH = Path(r"C:\Windows\System32\AMD\xrt-smi.exe")


@dataclass(frozen=True)
class HwContext:
    """A single hardware context from xrt-smi aie-partitions report."""

    pid: int
    context_id: int
    status: str
    command_submissions: int
    command_completions: int
    gops: str
    fps: str
    latency: str
    priority: str
    errors: int


class XrtSmiClient:
    """Subprocess wrapper for xrt-smi NPU management tool.

    Usage::

        client = XrtSmiClient()
        if client.is_available:
            contexts = client.get_hw_contexts(pid=os.getpid())
            for ctx in contexts:
                print(ctx.command_submissions)
    """

    def __init__(self, exe_path: Path | None = None) -> None:
        self._exe = exe_path or _XRT_SMI_PATH

    @property
    def is_available(self) -> bool:
        """Whether xrt-smi.exe exists on this system."""
        return self._exe.is_file()

    def snapshot(self) -> dict[str, Any]:
        """Take a full JSON snapshot of AIE partitions.

        Returns:
            Parsed JSON dict from ``xrt-smi examine -r aie-partitions``.
            Empty dict if xrt-smi is unavailable or fails.
        """
        if not self.is_available:
            return {}

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".json", delete=False, mode="w"
            ) as tmp:
                tmp_path = tmp.name

            # Remove the temp file before xrt-smi writes to it;
            # xrt-smi refuses to overwrite existing files without --force.
            Path(tmp_path).unlink(missing_ok=True)

            result = subprocess.run(  # noqa: S603
                [
                    str(self._exe),
                    "examine",
                    "-f",
                    "JSON",
                    "-o",
                    tmp_path,
                    "-r",
                    "aie-partitions",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.debug(
                    "xrt-smi failed (rc=%d): %s", result.returncode, result.stderr
                )
                return {}

            with Path(tmp_path).open(encoding="utf-8") as f:
                return json.load(f)

        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
            logger.debug("xrt-smi snapshot failed: %s", exc)
            return {}
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

    def get_hw_contexts(self, pid: int | None = None) -> list[HwContext]:
        """Get hardware contexts, optionally filtered by PID.

        Args:
            pid: If provided, only return contexts matching this PID.

        Returns:
            List of HwContext dataclasses.
        """
        data = self.snapshot()
        contexts: list[HwContext] = []

        for device in data.get("devices", []):
            partitions = (
                device.get("aie_partitions", {}).get("partitions", [])
            )
            for partition in partitions:
                for raw in partition.get("hw_contexts", []):
                    ctx = HwContext(
                        pid=int(raw.get("pid", 0)),
                        context_id=int(raw.get("context_id", 0)),
                        status=raw.get("status", "Unknown"),
                        command_submissions=int(
                            raw.get("command_submissions", 0)
                        ),
                        command_completions=int(
                            raw.get("command_completions", 0)
                        ),
                        gops=raw.get("gops", "N/A"),
                        fps=raw.get("fps", "N/A"),
                        latency=raw.get("latency", "N/A"),
                        priority=raw.get("priority", "Normal"),
                        errors=int(raw.get("errors", 0)),
                    )
                    if pid is None or ctx.pid == pid:
                        contexts.append(ctx)

        return contexts

    def get_command_submissions(self, pid: int | None = None) -> int:
        """Get total command_submissions for a PID (sum across all contexts).

        Args:
            pid: Process ID. Defaults to current process.

        Returns:
            Total command submissions, or 0 if not found.
        """
        if pid is None:
            pid = os.getpid()
        return sum(ctx.command_submissions for ctx in self.get_hw_contexts(pid))

    def get_command_completions(self, pid: int | None = None) -> int:
        """Get total command_completions for a PID (sum across all contexts).

        Args:
            pid: Process ID. Defaults to current process.

        Returns:
            Total command completions, or 0 if not found.
        """
        if pid is None:
            pid = os.getpid()
        return sum(ctx.command_completions for ctx in self.get_hw_contexts(pid))
