# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for safe symbolic ONNX shape inference."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from queue import Queue
from shutil import rmtree

import onnx

from winml.modelkit.onnx import infer_shapes, shape
from winml.modelkit.onnx.shape import _infer_symbolic_shapes_worker


def _make_model() -> onnx.ModelProto:
    input_ = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])
    node = onnx.helper.make_node("Identity", ["input"], ["output"])
    graph = onnx.helper.make_graph([node], "test", [input_], [output])
    return onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])


def _make_large_model() -> onnx.ModelProto:
    input_ = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])
    nodes = []
    previous = "input"
    for index in range(1_000):
        current = f"value_{index}"
        nodes.append(onnx.helper.make_node("Identity", [previous], [current]))
        previous = current
    nodes.append(onnx.helper.make_node("Identity", [previous], ["output"]))
    graph = onnx.helper.make_graph(nodes, "large_test", [input_], [output])
    return onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])


def test_symbolic_inference_keeps_parent_cwd_usable_while_worker_runs() -> None:
    """The parent CWD remains usable while symbolic inference owns its scratch CWD."""
    working_dir = (Path("temp") / "shape-inference-test").resolve()
    rmtree(working_dir, ignore_errors=True)
    working_dir.mkdir(parents=True)

    original_cwd = Path.cwd()
    os.chdir(working_dir)
    try:
        results: list[onnx.ModelProto] = []
        caller = threading.Thread(target=lambda: results.append(infer_shapes(_make_large_model())))
        caller.start()

        assert caller.is_alive()
        assert Path.cwd() == working_dir
        parent_sidecar = working_dir / "parent-sidecar"
        parent_sidecar.write_bytes(b"usable")
        assert parent_sidecar.read_bytes() == b"usable"

        caller.join(timeout=60)
        assert not caller.is_alive()
        assert results
        assert not list(working_dir.glob("winmlcli_shape_*"))
    finally:
        os.chdir(original_cwd)
        rmtree(working_dir)


def test_symbolic_inference_uses_system_scratch_when_cwd_is_unwritable(monkeypatch) -> None:
    """Worker scratch must not be created in the caller's current directory."""
    original_temporary_directory = shape.tempfile.TemporaryDirectory
    created_dirs: list[Path] = []
    working_dir = (Path("temp") / "shape-inference-unwritable-cwd-test").resolve()
    rmtree(working_dir, ignore_errors=True)
    working_dir.mkdir(parents=True)

    def temporary_directory(*args, **kwargs):
        if kwargs.get("dir") == working_dir:
            raise PermissionError("caller CWD is unwritable")
        temporary_dir = original_temporary_directory(*args, **kwargs)
        created_dirs.append(Path(temporary_dir.name))
        return temporary_dir

    monkeypatch.setattr(shape.tempfile, "TemporaryDirectory", temporary_directory)
    original_cwd = Path.cwd()
    os.chdir(working_dir)
    try:
        result = shape.infer_symbolic_shapes(_make_model())

        assert result.graph.name == "test"
        assert created_dirs
        assert all(not scratch.is_relative_to(working_dir) for scratch in created_dirs)
        assert not list(working_dir.glob("winmlcli_shape_*"))
    finally:
        os.chdir(original_cwd)
        rmtree(working_dir)


def test_symbolic_worker_reports_serialization_failure() -> None:
    """A failed worker must return its error to the parent explicitly."""
    working_dir = (Path("temp") / "shape-inference-failure-test").resolve()
    rmtree(working_dir, ignore_errors=True)
    working_dir.mkdir(parents=True)

    original_cwd = Path.cwd()
    os.chdir(working_dir)
    try:
        results: Queue[tuple[bool, str]] = Queue()
        _infer_symbolic_shapes_worker(b"not an onnx model", str(working_dir), results)
        success, error = results.get_nowait()

        assert success is False
        assert "DecodeError" in error
    finally:
        os.chdir(original_cwd)
        rmtree(working_dir)
