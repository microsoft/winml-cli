# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Paired fixed-shape Raw/Offline benchmark using caller-provided NPZ inputs.

Run with the CLI venv. Manifest is a JSON list of absolute NPZ paths, in
native reflection/Runtime schema input order. Generation is outside timing.
"""

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    """Run serial paired measurements with a shared input sequence."""
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--host", type=Path, required=True)
    p.add_argument("--dll-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--luid", required=True)
    p.add_argument("--adapter", default="0")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--offline-worker", action="store_true")
    a = p.parse_args()
    paths = json.loads(a.manifest.read_text())
    if len(paths) < a.iterations + a.warmup:
        raise ValueError("Provide one input NPZ for every warmup and measured iteration")
    batches = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            batches.append([np.ascontiguousarray(archive[k]) for k in archive.files])
    if any(
        len(b) != len(batches[0])
        or any(x.shape != y.shape or x.dtype != y.dtype for x, y in zip(b, batches[0], strict=True))
        for b in batches
    ):
        raise ValueError("Sequence shapes, dtypes and input order must be fixed")
    a.output.mkdir(parents=True, exist_ok=True)
    if a.offline_worker:
        import faulthandler

        faulthandler.dump_traceback_later(60)
        # Only benchmark orchestration changes; the production Runtime path is unchanged.
        import pandas as pd  # noqa: F401
        import scipy.interpolate  # noqa: F401
        import sklearn  # noqa: F401

        from winml.modelkit.commands.perf import perf
        from winml.modelkit.models import WinMLAutoModel  # noqa: F401
        from winml.modelkit.session.runtime_session import WinMLRuntimeSession

        original = WinMLRuntimeSession._prepare_inputs
        read = WinMLRuntimeSession._read_outputs
        index = 0
        last = {}

        def prepare(self, inputs):
            nonlocal index
            prepared = original(self, inputs)
            batch = batches[index % len(batches)]
            index += 1
            return dict(zip(prepared, batch, strict=True))

        def outputs(self, named):
            nonlocal last
            last = read(self, named)
            return last

        WinMLRuntimeSession._prepare_inputs = prepare
        WinMLRuntimeSession._read_outputs = outputs
        perf(
            args=[
                "-m",
                str(a.model),
                "--input-data",
                paths[0],
                "--runtime",
                "winml-runtime",
                "--backend",
                "cgc",
                "--device",
                "gpu",
                "--device-luid",
                a.luid,
                "--iterations",
                str(a.iterations),
                "--warmup",
                str(a.warmup),
                "--no-memory",
                "--no-monitor",
                "--overwrite",
                "-o",
                str(a.output / "perf.json"),
            ],
            standalone_mode=False,
        )
        if index != a.iterations + a.warmup:
            raise ValueError(f"Unexpected Runtime input calls: {index}")
        np.savez(a.output / "outputs.npz", **last)
        return
    root = Path(__file__).resolve().parents[2]
    summary = []
    for repeat, order in enumerate((("raw", "offline"), ("offline", "raw"), ("raw", "offline"))):
        for mode in order:
            work = a.output / f"{repeat}-{mode}"
            work.mkdir(exist_ok=False)
            if mode == "raw":
                for i, array in enumerate(batches[0]):
                    array.tofile(work / f"input-{i}.bin")
                    dtype = {np.dtype("float32"): 15, np.dtype("float16"): 16}[array.dtype]
                    (work / f"input-{i}.txt").write_text(
                        " ".join(map(str, [dtype, array.ndim, *array.shape]))
                    )
                (work / "input-count.txt").write_text(str(len(batches[0])))
                (work / "input-sequence.json").write_text(json.dumps(paths))
                command = [
                    sys.executable,
                    str(root / "src/winml/modelkit/commands/_dxcgc_worker.py"),
                    str(a.host),
                    str(a.model),
                    str(a.dll_dir),
                    str(a.dll_dir),
                    "721",
                    a.adapter,
                    str(a.iterations),
                    str(a.warmup),
                    str(work),
                ]
            else:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--offline-worker",
                    "--manifest",
                    str(a.manifest),
                    "--model",
                    str(a.model),
                    "--host",
                    str(a.host),
                    "--dll-dir",
                    str(a.dll_dir),
                    "--output",
                    str(work),
                    "--luid",
                    a.luid,
                    "--iterations",
                    str(a.iterations),
                    "--warmup",
                    str(a.warmup),
                ]
            with (work / "run.log").open("w") as log:
                subprocess.run(  # noqa: S603 - explicit local argv, no shell
                    command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300
                )
            if mode == "raw":
                with (work / "samples.csv").open() as stream:
                    samples = [float(r["host_ms"]) for r in csv.DictReader(stream)]
            else:
                samples = json.loads((work / "perf.json").read_text())["raw_samples_ms"]
            if len(samples) != a.iterations:
                raise ValueError("Incomplete measurements")
            summary.append({"round": repeat, "mode": mode, "p50_ms": statistics.median(samples)})
            (a.output / "summary.json").write_text(json.dumps(summary, indent=2))
    validation = []
    for repeat in range(3):
        work = a.output / f"{repeat}-raw"
        specs = [
            line.split(chr(9))
            for line in (work / "io.tsv").read_text().splitlines()
            if line.startswith("output")
        ]
        with np.load(a.output / f"{repeat}-offline" / "outputs.npz") as reference:
            if len(specs) != len(reference.files):
                raise ValueError("Output count mismatch")
            actuals, expected = [], []
            for spec, key in zip(specs, reference.files, strict=True):
                value = np.fromfile(
                    work / f"output-{spec[1]}.bin",
                    dtype={15: np.float32, 16: np.float16}[int(spec[2])],
                )
                if value.size != reference[key].size:
                    raise ValueError("Output size mismatch")
                actuals.append(value.astype(float).ravel())
                expected.append(reference[key].astype(float).ravel())
            x, y = np.concatenate(actuals), np.concatenate(expected)
            cosine = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
            if not np.isfinite(cosine) or cosine <= 0.99:
                raise ValueError(f"Paired output mismatch: {cosine}")
            validation.append({"round": repeat, "last_input_cosine": cosine})
    (a.output / "validation.json").write_text(json.dumps(validation, indent=2))
    evidence = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            a.model,
            a.host,
            a.host.with_name("winml-dxcgc-python.dll"),
            a.dll_dir / "dxcgc.dll",
            *map(Path, paths),
        ]
    }
    (a.output / "hashes.json").write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
