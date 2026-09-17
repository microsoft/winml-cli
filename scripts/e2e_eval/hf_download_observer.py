# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Disposable HF download observer: never import WinML or the E2E runner.

All filesystem traversal and unsafe native handle inspection stay in this
process. The supervisor consumes bounded datagrams and can kill this process
without relying on its Python interpreter, native threads, or filesystem.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import psutil


def _expand_path(value):
    return Path(os.path.expandvars(os.fspath(value))).expanduser()


def _hf_cache_roots(env):
    home = _expand_path(env.get("HF_HOME") or (
        _expand_path(env.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface"
    ))
    return (
        _expand_path(env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE") or home / "hub"),
        _expand_path(env.get("HF_DATASETS_CACHE") or home / "datasets"),
        _expand_path(env.get("HF_XET_CACHE") or home / "xet"),
    )


def _snapshot_hf_downloads(env):
    hub, datasets, xet = _hf_cache_roots(env)
    result = {}
    for root, patterns in (
        (hub, ("*/blobs/*.incomplete", "*.incomplete")),
        (datasets, ("downloads/*.incomplete",)),
        (xet, ("**/*.incomplete",)),
    ):
        for pattern in patterns:
            for path in root.glob(pattern):
                try:
                    stat = path.stat()
                    result[path] = (stat.st_size, stat.st_mtime_ns)
                except FileNotFoundError:
                    continue  # Atomic download completion raced with the scan.
    return result


def _normalized_path(path):
    return os.path.normcase(os.path.realpath(path))


def _process_tree_open_paths(pid):
    root = psutil.Process(pid)
    result = set()
    for process in [root, *root.children(recursive=True)]:
        try:
            result.update(_normalized_path(item.path) for item in process.open_files())
        except psutil.NoSuchProcess:
            continue
    return result


class DownloadTracker:
    """Track only changed partial files positively associated with this tree.

    Ownership is cached for a download episode, avoiding repeated native scans
    while an already-known file grows. A successful cache scan with no remaining
    owned partials marks the end of a download episode. The CLI, not this timing
    observer, determines whether the downloaded model/data is valid.
    """

    def __init__(self, env, pid):
        self.env = env
        self.pid = pid
        self.previous = {}
        self.owned = {}
        self.epoch = 0
        self.completed_epoch = 0
        self.last_progress = 0.0

    def poll(self, now=None):
        """Return a completed scan; explicit time is for deterministic tests."""
        current = _snapshot_hf_downloads(self.env)
        candidates = {
            path for path, value in current.items()
            if path not in self.owned and self.previous.get(path) != value
        }
        # Do not perform native handle inspection on ordinary no-download perf.
        discovered = set()
        if candidates:
            opened = _process_tree_open_paths(self.pid)
            discovered = {path for path in candidates if _normalized_path(path) in opened}
        # Native inspection may be slow. Timestamp the observed progress after
        # it returns, not when the scan started (which could falsely imply stall).
        if now is None:
            now = time.monotonic()
        was_active = bool(self.owned)
        if discovered and not was_active:
            self.epoch += 1
        for path in discovered:
            self.owned[path] = current[path]
            self.last_progress = now

        for path, previous in list(self.owned.items()):
            if path not in current:
                del self.owned[path]
            elif previous != current[path]:
                self.last_progress = now
                self.owned[path] = current[path]
        if was_active and not self.owned:
            self.completed_epoch = self.epoch
        self.previous = current
        return {
            "state": "ACTIVE" if self.owned else "IDLE",
            "epoch": self.epoch,
            "completed_epoch": self.completed_epoch,
            "last_progress": self.last_progress,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--created", type=float, required=True)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    try:
        root = psutil.Process(args.pid)
        parent = psutil.Process(args.parent)
        if root.create_time() != args.created:
            return  # Do not inspect a reused PID.
    except psutil.NoSuchProcess:
        return
    tracker = DownloadTracker(os.environ, args.pid)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as channel:
        channel.connect(("127.0.0.1", args.port))
        channel.setblocking(False)
        sequence = 0
        while parent.is_running() and root.is_running():
            # A native hang prevents this heartbeat: do not mask it with a thread.
            try:
                state = tracker.poll()
            except (OSError, psutil.Error):
                # Failure is terminal for this optional observer. The client
                # reports UNKNOWN and falls back to normal execution timing.
                return
            sequence += 1
            message = {**state, "token": args.token, "seq": sequence,
                       "observed_at": time.monotonic()}
            try:
                channel.send(json.dumps(message, allow_nan=False).encode())
            except BlockingIOError:
                pass  # The next heartbeat repeats durable completion metadata.
            except OSError:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
