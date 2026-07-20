# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Generated-graph tests for static Split-to-Slice rewriting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
import onnxruntime as ort
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from winml.modelkit.commands.optimize import optimize
from winml.modelkit.optim import get_all_capabilities, optimize_onnx
from winml.modelkit.optim.pipes import (
    PIPES,
    AlgebraicRewritePipe,
    AlgebraicRewritePipeConfig,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


def _tensor(name: str, values: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values), name)


def _model(
    nodes: Sequence[onnx.NodeProto],
    inputs: Sequence[onnx.ValueInfoProto],
    outputs: Sequence[onnx.ValueInfoProto],
    initializers: Sequence[onnx.TensorProto],
    value_info: Sequence[onnx.ValueInfoProto] = (),
) -> onnx.ModelProto:
    graph = helper.make_graph(
        list(nodes),
        "generated_algebraic_graph",
        list(inputs),
        list(outputs),
        initializer=list(initializers),
        value_info=list(value_info),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _info(name: str, shape: Sequence[int | None]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _run(model: onnx.ModelProto, values: dict[str, np.ndarray]) -> list[np.ndarray]:
    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    return session.run(None, values)


def _assert_valid_with_inferred_shapes(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model)
    assert len(inferred.graph.output) == len(model.graph.output)


class TestAlgebraicRegistration:
    """Verify capability registration, flags, and pipe ordering."""

    def test_capability_is_opt_in(self) -> None:
        capability = get_all_capabilities()["static-split-to-slice"]
        assert capability.default is False
        assert capability.cli_flags() == (
            "--enable-static-split-to-slice",
            "--disable-static-split-to-slice",
        )

        config = AlgebraicRewritePipe.build_config(static_split_to_slice=True)
        assert config.static_split_to_slice is True

    def test_cli_lists_algebraic_flag(self) -> None:
        result = CliRunner().invoke(optimize, ["--list-capabilities"])
        assert result.exit_code == 0
        assert "--enable-static-split-to-slice" in result.output

    def test_pipe_is_after_ort_graph_and_before_cleanup(self) -> None:
        names = [pipe.name for pipe in PIPES]
        assert names.index("ort_graph") < names.index("algebraic_rewrite")
        assert names.index("algebraic_rewrite") < names.index("surgery")
        assert PIPES[names.index("algebraic_rewrite")] is AlgebraicRewritePipe
        assert not AlgebraicRewritePipe.should_process(AlgebraicRewritePipeConfig())


class TestStaticSplitToSlice:
    """Test static Split replacement using generated data."""

    def test_positive_equivalence_and_preserved_outputs(self) -> None:
        rng = np.random.default_rng(10)
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 6, 2])
        outputs = [_info("left", [1, 2, 2]), _info("right", [1, 4, 2])]
        split = helper.make_node(
            "Split",
            ["x", "split_sizes"],
            ["left", "right"],
            name="",
            axis=1,
        )
        model = _model(
            [split],
            [x],
            outputs,
            [_tensor("split_sizes", np.asarray([2, 4], dtype=np.int64))],
        )
        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(static_split_to_slice=True),
        )

        assert [node.op_type for node in transformed.graph.node] == ["Slice", "Slice"]
        assert [node.output[0] for node in transformed.graph.node] == ["left", "right"]
        assert [output.name for output in transformed.graph.output] == ["left", "right"]
        assert "split_sizes" not in {
            initializer.name for initializer in transformed.graph.initializer
        }
        _assert_valid_with_inferred_shapes(transformed)
        values = {"x": rng.normal(size=(1, 6, 2)).astype(np.float32)}
        for original, rewritten in zip(_run(model, values), _run(transformed, values), strict=True):
            np.testing.assert_allclose(original, rewritten, rtol=0, atol=0)

    def test_equal_split_and_name_collisions_are_safe(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 2])
        keep = _info("keep", [1, 2, 2])
        split = helper.make_node(
            "Split",
            ["x"],
            ["part_a", "part_b"],
            name="",
            axis=1,
        )
        identity = helper.make_node(
            "Identity",
            ["part_b"],
            ["keep"],
            name="algebraic_split_slice",
        )
        model = _model(
            [split, identity],
            [x],
            [keep, _info("part_a", [1, 2, 2])],
            [],
            value_info=[_info("part_a", [1, 2, 2]), _info("part_b", [1, 2, 2])],
        )
        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(static_split_to_slice=True),
        )
        generated = [node for node in transformed.graph.node if node.op_type == "Slice"]
        assert len(generated) == 2
        assert len({node.name for node in transformed.graph.node}) == len(transformed.graph.node)
        assert all(node.name for node in generated)
        assert {node.output[0] for node in generated} == {"part_a", "part_b"}
        _assert_valid_with_inferred_shapes(transformed)

    def test_dynamic_equal_split_and_malformed_split_are_unchanged(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, None, 2])
        dynamic_equal = helper.make_node("Split", ["x"], ["a", "b"], axis=1)
        malformed = helper.make_node(
            "Split",
            ["x", "bad_sizes"],
            ["c", "d"],
            axis=1,
        )
        model = _model(
            [dynamic_equal, malformed],
            [x],
            [_info("a", [1, None, 2]), _info("b", [1, None, 2])],
            [_tensor("bad_sizes", np.asarray([1, 1], dtype=np.int64))],
        )
        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(static_split_to_slice=True),
        )
        assert [node.op_type for node in transformed.graph.node] == ["Split", "Split"]

    def test_dead_generated_slice_and_constants_are_pruned(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 2])
        model = _model(
            [helper.make_node("Split", ["x"], ["left", "unused"], axis=1)],
            [x],
            [_info("left", [1, 2, 2])],
            [],
        )
        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(static_split_to_slice=True),
        )
        assert [node.op_type for node in transformed.graph.node] == ["Slice"]
        assert transformed.graph.node[0].output[0] == "left"
        assert len(transformed.graph.initializer) == 4

    def test_nested_subgraph_captures_keep_generated_slices_live(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 2])
        then_branch = helper.make_graph(
            [helper.make_node("Identity", ["left"], ["then_output"])],
            "then_branch",
            [],
            [_info("then_output", [1, 2, 2])],
        )
        else_branch = helper.make_graph(
            [helper.make_node("Identity", ["right"], ["else_output"])],
            "else_branch",
            [],
            [_info("else_output", [1, 2, 2])],
        )
        model = _model(
            [
                helper.make_node("Split", ["x"], ["left", "right"], axis=1),
                helper.make_node(
                    "If",
                    ["condition"],
                    ["y"],
                    then_branch=then_branch,
                    else_branch=else_branch,
                ),
            ],
            [x],
            [_info("y", [1, 2, 2])],
            [_tensor("condition", np.asarray(True, dtype=np.bool_))],
            value_info=[_info("left", [1, 2, 2]), _info("right", [1, 2, 2])],
        )
        values = {"x": np.arange(8, dtype=np.float32).reshape(1, 4, 2)}
        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(static_split_to_slice=True),
        )
        assert [node.op_type for node in transformed.graph.node] == [
            "Slice",
            "Slice",
            "If",
        ]
        _assert_valid_with_inferred_shapes(transformed)
        np.testing.assert_array_equal(_run(model, values), _run(transformed, values))

    def test_public_optimize_path_is_idempotent(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 2])
        model = _model(
            [helper.make_node("Split", ["x"], ["a", "b"], name="", axis=1)],
            [x],
            [_info("a", [1, 2, 2]), _info("b", [1, 2, 2])],
            [],
        )
        transformed = optimize_onnx(model, static_split_to_slice=True)
        second = optimize_onnx(transformed, static_split_to_slice=True)
        assert all(node.op_type != "Split" for node in transformed.graph.node)
        assert [
            (node.op_type, tuple(node.input), tuple(node.output), node.name)
            for node in transformed.graph.node
        ] == [
            (node.op_type, tuple(node.input), tuple(node.output), node.name)
            for node in second.graph.node
        ]
        _assert_valid_with_inferred_shapes(second)
