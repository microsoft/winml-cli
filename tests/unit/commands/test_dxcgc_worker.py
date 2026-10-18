# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Exercise the ctypes driver without a GPU or native library."""

import csv
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from winml.modelkit.commands import _dxcgc_worker as worker


class Function:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("legacy_locale", [False, True])
def test_python_driver_excludes_warmups_and_propagates_native_error(
    tmp_path, monkeypatch, failure, legacy_locale
):
    (tmp_path / "io.tsv").write_text(
        "input\t0\t15\t2,\t依\noutput\t0\t15\t2,\t結果\n", encoding="utf8"
    )
    if legacy_locale:
        original_read = Path.read_text

        def locale_read(self, encoding=None, errors=None):
            return original_read(self, encoding=encoding or "cp1252", errors=errors)

        monkeypatch.setattr(Path, "read_text", locale_read)
    sequence = []
    for i in range(5):
        path = tmp_path / f"input-{i}.npz"
        np.savez(path, x=np.array([i, 2], dtype=np.float32))
        sequence.append(str(path))
    (tmp_path / "input-sequence.json").write_text(json.dumps(sequence))
    monkeypatch.setattr(
        worker.sys,
        "argv",
        [
            "worker",
            str(tmp_path / "host.exe"),
            "model",
            "dll",
            "sdk",
            "721",
            "0",
            "3",
            "2",
            str(tmp_path),
        ],
    )
    calls = []

    def entry(argc, argv, driver):
        step_type = driver._argtypes_[0]

        @step_type
        def step(context, gpu, run_ms, pointers, sizes, count, inputs, input_sizes, input_count):
            assert count == 1 and sizes[0] == 8
            assert input_count == 1 and input_sizes[0] == 8
            assert (ctypes.c_float * 2).from_address(inputs[0])[0] == len(calls)
            calls.append(count)
            if failure:
                return 1
            gpu[0] = 0.25
            run_ms[0] = 1.25
            output = (ctypes.c_float * 2).from_address(pointers[0])
            output[:] = [len(calls), 2]
            return 0

        return driver(step, None, 3, 2)

    library = SimpleNamespace(
        DxcgcPythonHostAbiVersion=Function(lambda: 4), RunPythonDriven=Function(entry)
    )
    monkeypatch.setattr(worker.ctypes, "CDLL", lambda _: library)
    if failure:
        with pytest.raises(RuntimeError, match="Native inference failed"):
            worker.main()
        assert not (tmp_path / "driver-policy.txt").exists()
    else:
        worker.main()
        assert len(calls) == 5
        with (tmp_path / "samples.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 3
        assert all(float(row["host_ms"]) == 1.25 for row in rows)
        assert all(float(row["gpu_ms"]) == 0.25 for row in rows)
        np.testing.assert_array_equal(np.fromfile(tmp_path / "output-0.bin", np.float32), [5, 2])


def test_worker_rejects_wrong_abi_before_callback(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.sys, "argv", ["worker", "host.exe", *(["x"] * 7), str(tmp_path)])
    monkeypatch.setattr(
        worker.ctypes,
        "CDLL",
        lambda _: SimpleNamespace(DxcgcPythonHostAbiVersion=Function(lambda: 1)),
    )
    with pytest.raises(RuntimeError, match="ABI mismatch"):
        worker.main()
