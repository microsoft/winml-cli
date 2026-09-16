# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Opt-in, external stack sampling for an existing E2E process tree.

Run alongside the unmodified CLI, under the agent's own identity. Never inspect
environment variables, frame locals, or open file handles. The sampler neither
suspends nor terminates the target processes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


def _process_snapshot(process: psutil.Process) -> dict:
    memory = process.memory_info()
    cpu = process.cpu_times()
    return {
        "pid": process.pid,
        "parent_pid": process.ppid(),
        "name": process.name(),
        "created_at": process.create_time(),
        "command": process.cmdline(),
        "rss_bytes": memory.rss,
        "private_bytes": getattr(memory, "private", memory.vms),
        "cpu_seconds": cpu.user + cpu.system,
        "threads": process.num_threads(),
    }


def capture_snapshot(
    root: psutil.Process, output_dir: Path, index: int, stop_file: Path
) -> None:
    """Record resource facts and bounded, nonblocking Python stack samples."""
    processes = [root, *root.children(recursive=True)]
    samples = []
    for process in processes:
        if process.pid == os.getpid():
            continue
        with contextlib.suppress(psutil.Error):
            samples.append(_process_snapshot(process))
    memory = psutil.virtual_memory()
    snapshot = {
        "timestamp": time.time(),
        "available_ram_bytes": memory.available,
        "processes": samples,
    }
    (output_dir / f"processes-{index:04}.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    executable = Path(sys.executable).with_name("py-spy.exe" if os.name == "nt" else "py-spy")
    for sample in samples:
        if stop_file.exists():
            break
        if Path(sample["name"]).stem.lower() not in {"python", "pythonw"}:
            continue
        # Windows venv launchers are small forwarding processes, not interpreters.
        if sample["private_bytes"] < 64 * 1024**2:
            continue
        path = output_dir / f"stack-{index:04}-{sample['pid']}.txt"
        with path.open("wb") as stream:
            try:
                result = subprocess.run(  # noqa: S603
                    [str(executable), "dump", "--pid", str(sample["pid"]),
                     "--full-filenames", "--nonblocking"],
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=10,
                    check=False,
                )
                stream.write(f"\nSampler exit code: {result.returncode}\n".encode())
            except (OSError, subprocess.TimeoutExpired) as exc:
                stream.write(f"\nSampler failed: {type(exc).__name__}: {exc}\n".encode())


def collect(root_pid: int, output_dir: Path, stop_file: Path, interval: float = 30) -> None:
    """Observe until the parent exits or the calling task creates its stop file."""
    if interval <= 0:
        raise ValueError("interval must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        root = psutil.Process(root_pid)
    except psutil.NoSuchProcess:
        return
    index = 0
    next_sample = time.monotonic()
    # Bound the sampler lifetime even if the calling shell cannot clean up.
    deadline = next_sample + 2 * 60 * 60
    while root.is_running() and not stop_file.exists() and time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_sample:
            try:
                capture_snapshot(root, output_dir, index, stop_file)
            except (psutil.Error, OSError) as exc:
                print(f"Snapshot failed: {type(exc).__name__}: {exc}", flush=True)
            index += 1
            next_sample = time.monotonic() + interval
        with contextlib.suppress(psutil.TimeoutExpired, psutil.NoSuchProcess):
            root.wait(timeout=min(1.0, max(0.01, next_sample - time.monotonic())))


def main() -> None:
    """Parse the agent-provided process identity and diagnostic destinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-pid", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    args = parser.parse_args()
    collect(args.root_pid, args.output_dir, args.stop_file)


if __name__ == "__main__":
    main()
