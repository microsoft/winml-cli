# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the opt-in external CI stack sampler."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import psutil
import pytest
import yaml


@pytest.fixture
def sampler():
    path = Path(__file__).resolve().parents[3] / "scripts/e2e_eval/collect_stacks.py"
    spec = importlib.util.spec_from_file_location("_stack_sampler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_uses_bounded_external_sampling_without_locals(sampler, monkeypatch, tmp_path):
    root = MagicMock()
    child = MagicMock(pid=123)
    root.pid = 456
    root.children.return_value = [child]
    memory = psutil.Process().memory_info()
    samples = {
        root.pid: {"pid": root.pid, "name": "powershell.exe", "private_bytes": memory.vms},
        child.pid: {"pid": child.pid, "name": "python.exe", "private_bytes": 128 * 1024**2},
    }
    monkeypatch.setattr(sampler, "_process_snapshot", lambda process: samples[process.pid])
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        kwargs["stdout"].write(b"external stack captured\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sampler.subprocess, "run", run)
    sampler.capture_snapshot(root, tmp_path, 0, tmp_path / "stop")

    assert len(calls) == 1
    args, options = calls[0]
    assert "--nonblocking" in args
    assert "--locals" not in args
    assert options["timeout"] == 10
    assert options["stdin"] == subprocess.DEVNULL
    stored = json.loads((tmp_path / "processes-0000.json").read_text())
    assert stored["processes"] == list(samples.values())
    assert "external stack captured" in (tmp_path / "stack-0000-123.txt").read_text()


def test_existing_stop_file_does_not_sample(sampler, monkeypatch, tmp_path):
    stop = tmp_path / "stop"
    stop.touch()
    capture = MagicMock()
    monkeypatch.setattr(sampler, "capture_snapshot", capture)
    sampler.collect(psutil.Process().pid, tmp_path, stop)
    capture.assert_not_called()


def test_nonpositive_interval_is_rejected(sampler, tmp_path):
    with pytest.raises(ValueError, match="interval must be positive"):
        sampler.collect(psutil.Process().pid, tmp_path, tmp_path / "stop", interval=0)


def test_pipeline_diagnostics_default_to_off_and_preserve_model_order():
    root = Path(__file__).resolve().parents[3]
    pipeline = yaml.safe_load((root / ".pipelines/Modelkit E2E Test.yml").read_text())
    template = yaml.safe_load((root / ".pipelines/templates/e2e-test-jobs.yml").read_text())
    for document in (pipeline, template):
        parameters = {parameter["name"]: parameter for parameter in document["parameters"]}
        assert parameters["e2eDiagnostics"]["default"] == "off"
    qnn = pipeline["stages"][0]["jobs"][0]
    assert qnn["parameters"]["models"] == "${{ parameters.models }}"
    assert qnn["parameters"]["e2eDiagnostics"] == "${{ parameters.e2eDiagnostics }}"
