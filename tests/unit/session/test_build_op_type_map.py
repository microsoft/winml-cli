# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""v2.4 ``WinMLSession._build_op_type_map``: ONNX ``node.name -> node.op_type``.

The map is consumed by op-tracing monitors via
:meth:`EPMonitor.set_onnx_op_types`. It must be defensive: any failure to
load the ONNX (None path, missing file, corrupt protobuf, missing
``onnx`` package) returns an empty dict so the perf path keeps working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx
from onnx import TensorProto, helper

from winml.modelkit.session.session import WinMLSession


if TYPE_CHECKING:
    from pathlib import Path


def test_build_op_type_map_none_path_returns_empty() -> None:
    """A None ``onnx_path`` returns an empty dict (defensive default)."""
    assert WinMLSession._build_op_type_map(None) == {}


def test_build_op_type_map_missing_file_returns_empty(tmp_path: Path) -> None:
    """A non-existent path returns an empty dict (defensive default)."""
    nonexistent = tmp_path / "does_not_exist.onnx"
    assert WinMLSession._build_op_type_map(nonexistent) == {}


def test_build_op_type_map_corrupt_file_returns_empty(tmp_path: Path) -> None:
    """A corrupt ONNX file returns an empty dict (defensive default)."""
    corrupt = tmp_path / "corrupt.onnx"
    corrupt.write_bytes(b"not a real onnx file")
    assert WinMLSession._build_op_type_map(corrupt) == {}


def test_build_op_type_map_named_nodes_returns_populated_dict(tmp_path: Path) -> None:
    """A well-formed ONNX with named nodes returns a populated map."""
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    mid = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])
    relu_node = helper.make_node("Relu", ["x"], ["y"], name="/layer1/Relu")
    identity_node = helper.make_node("Identity", ["y"], ["z"], name="/layer2/Identity")
    graph = helper.make_graph([relu_node, identity_node], "g", [inp], [out], value_info=[mid])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx_path = tmp_path / "named.onnx"
    onnx.save(model, str(onnx_path))

    op_map = WinMLSession._build_op_type_map(onnx_path)
    assert op_map == {"/layer1/Relu": "Relu", "/layer2/Identity": "Identity"}


def test_build_op_type_map_skips_unnamed_nodes(tmp_path: Path) -> None:
    """Nodes with empty ``name`` are skipped (keys must be non-empty)."""
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])
    # Unnamed Identity node — common in helper-generated minimal models.
    unnamed = helper.make_node("Identity", ["x"], ["z"])
    graph = helper.make_graph([unnamed], "g", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx_path = tmp_path / "unnamed.onnx"
    onnx.save(model, str(onnx_path))

    assert WinMLSession._build_op_type_map(onnx_path) == {}


def test_build_op_type_map_filters_empty_op_type(tmp_path: Path) -> None:
    """CRIT-2: nodes with empty ``op_type`` are filtered out (defensive).

    A malformed ONNX file with an empty ``op_type`` node should NOT
    propagate the empty string as a map value.  ``_build_op_type_map``
    uses ``onnx.load`` without ``check_model``, so ``onnx.checker``
    rejection of empty ``op_type`` does not save us — the filter on the
    dict comprehension does.

    This is double-defense paired with ``_resolve_op_type``'s truthy
    check on the L1 lookup result.
    """
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    mid = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])
    good_node = helper.make_node("Conv", ["x"], ["y"], name="/conv/Conv")
    # Pathological empty op_type with a valid name — helper accepts it,
    # save/load round-trips it, but the filter must drop it.
    empty_node = helper.make_node("", ["y"], ["z"], name="/empty/Op")
    graph = helper.make_graph([good_node, empty_node], "g", [inp], [out], value_info=[mid])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx_path = tmp_path / "empty_op_type.onnx"
    onnx.save(model, str(onnx_path))

    op_map = WinMLSession._build_op_type_map(onnx_path)

    # Conv node mapped; empty-op_type node filtered out.
    assert op_map == {"/conv/Conv": "Conv"}
    assert "/empty/Op" not in op_map
    # All values must be non-empty (the invariant the filter enforces).
    assert all(v for v in op_map.values()), (
        f"all values must be non-empty op_type strings; got {op_map!r}"
    )
