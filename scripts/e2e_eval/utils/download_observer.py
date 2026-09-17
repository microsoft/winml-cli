# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Nonblocking client and pure timeout accounting for the disposable observer."""

from __future__ import annotations

import json
import logging
import math
import os
import secrets
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from .process_tree import ManagedProcess


logger = logging.getLogger(__name__)
FRESHNESS_SECONDS = 5.0
OBSERVER_SCRIPT = Path(__file__).resolve().parents[1] / "hf_download_observer.py"


@dataclass(frozen=True)
class Observation:
    """Only positive, fresh observation can pause execution accounting."""

    state: str
    observed_at: float
    epoch: int = 0
    completed_epoch: int = 0
    last_progress: float = 0.0


class ExecutionBudget:
    """Monotonic execution/stall deadlines, independent of native observation."""

    def __init__(self, timeout, stall_timeout, now):
        self.timeout = timeout
        self.stall_timeout = stall_timeout
        self.execution_elapsed = 0.0
        self.last_tick = now
        self.previous = Observation("UNKNOWN", now)
        self.active_epoch = None
        self._uncertain_episode = False
        self.completed_epoch = 0
        self.timed_out = False
        self.hf_download_stalled = False

    def update(self, now, observation):
        """Charge unobserved time and apply each confirmed completion once."""
        paused = 0.0
        if self.previous.state == "ACTIVE":
            paused = max(0.0, min(now, self.previous.observed_at + FRESHNESS_SECONDS) - self.last_tick)
        self.execution_elapsed += max(0.0, now - self.last_tick - paused)
        fresh = now - observation.observed_at <= FRESHNESS_SECONDS
        state = observation.state if fresh else "UNKNOWN"
        if state == "UNKNOWN":
            if self.active_epoch is not None:
                self._uncertain_episode = True
            self.active_epoch = None
        else:
            if observation.completed_epoch > self.completed_epoch:
                if self.active_epoch == observation.completed_epoch and not self._uncertain_episode:
                    self.execution_elapsed = 0.0
                self.completed_epoch = observation.completed_epoch
            if state == "ACTIVE":
                self.active_epoch = observation.epoch
            else:
                self.active_epoch = None
                self._uncertain_episode = False
        self.hf_download_stalled = (
            state == "ACTIVE" and now - observation.last_progress >= self.stall_timeout
        )
        self.timed_out = state != "ACTIVE" and self.execution_elapsed >= self.timeout
        self.previous = observation if fresh else Observation("UNKNOWN", now)
        self.last_tick = now


class DownloadObserver:
    """One isolated observer per CLI; failure degrades to fixed execution time.

    Datagrams bound both memory and receive time: no pipe reads, background
    readers or partial-message waits. A failed observer is not restarted during
    this CLI invocation, preventing stale epochs or restart loops from granting
    extra execution time. The next CLI gets a new observer and session token.
    """

    def __init__(self, pid, env):
        self._socket = None
        self._owner = None
        self._token = secrets.token_hex(16)
        self._seq = 0
        self._started = time.monotonic()
        self._last = Observation("UNKNOWN", self._started)
        self.failure = None
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.bind(("127.0.0.1", 0))
            self._socket.setblocking(False)
            created = psutil.Process(pid).create_time()
            self._owner = ManagedProcess(
                [sys.executable, str(OBSERVER_SCRIPT), "--pid", str(pid),
                 "--created", str(created), "--parent", str(os.getpid()),
                 "--port", str(self._socket.getsockname()[1]), "--token", self._token],
                env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (OSError, psutil.Error, ValueError) as exc:
            self._fail(f"could not start ({type(exc).__name__})")

    def _fail(self, reason):
        if self.failure is None:
            self.failure = reason
            logger.warning("HF download observer unavailable: %s; using execution timeout", reason)
        self.close()

    def poll(self, now):
        """Consume at most eight complete datagrams, without waiting for one."""
        if self._socket is None or self._owner is None:
            return Observation("UNKNOWN", now)
        # A dead helper must not leave a stale ACTIVE lease or completion behind.
        if self._owner.process.poll() is not None:
            self._fail("observer process exited")
            return Observation("UNKNOWN", now)
        for _ in range(8):
            try:
                payload, address = self._socket.recvfrom(4097)
            except BlockingIOError:
                break
            except OSError:
                self._fail("observation channel failed")
                return Observation("UNKNOWN", now)
            if len(payload) > 4096 or address[0] != "127.0.0.1":
                continue
            try:
                value = json.loads(payload)
                if not isinstance(value, dict) or value.get("token") != self._token:
                    continue
                seq = value["seq"]
                if type(seq) is not int or seq <= self._seq:
                    continue
                observation = Observation(**{key: value[key] for key in (
                    "state", "observed_at", "epoch", "completed_epoch", "last_progress"
                )})
                if observation.state not in {"ACTIVE", "IDLE", "UNKNOWN"}:
                    continue
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in (
                    observation.observed_at, observation.last_progress
                )):
                    continue
                if not self._started <= observation.observed_at <= now:
                    continue
                if observation.observed_at < self._last.observed_at:
                    continue
                if not 0 <= observation.last_progress <= observation.observed_at:
                    continue
                if any(type(v) is not int or v < 0 for v in (
                    observation.epoch, observation.completed_epoch
                )) or observation.completed_epoch > observation.epoch:
                    continue
                if observation.epoch < self._last.epoch or observation.completed_epoch < self._last.completed_epoch:
                    continue
                self._seq, self._last = seq, observation
            except (ValueError, TypeError, KeyError, RecursionError):
                continue
        if now - self._last.observed_at > FRESHNESS_SECONDS:
            self._fail("no completed observation within 5 seconds")
            return Observation("UNKNOWN", now)
        return self._last

    def close(self):
        """Kill and reap the disposable observer using bounded process waits."""
        if self._owner is not None:
            owner, self._owner = self._owner, None
            try:
                owner.close()
            except (OSError, subprocess.TimeoutExpired) as exc:
                logger.warning("HF observer cleanup failed: %s", exc)
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
