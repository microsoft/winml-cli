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
import pytest
from click.testing import CliRunner

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
    return onnx.numpy_helper.from_array(np.asarray(values), name)


def _model(
    nodes: Sequence[onnx.NodeProto],
    inputs: Sequence[onnx.ValueInfoProto],
    outputs: Sequence[onnx.ValueInfoProto],
    initializers: Sequence[onnx.TensorProto],
    value_info: Sequence[onnx.ValueInfoProto] = (),
) -> onnx.ModelProto:
    graph = onnx.helper.make_graph(
        list(nodes),
        "generated_algebraic_graph",
        list(inputs),
        list(outputs),
        initializer=list(initializers),
        value_info=list(value_info),
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _info(name: str, shape: Sequence[int | None]) -> onnx.ValueInfoProto:
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, list(shape))


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

    def test_capabilities_are_opt_in_and_independent(self) -> None:
        capabilities = get_all_capabilities()
        names = {"static-split-to-slice", "conv-channel-affine-folding"}
        assert names <= capabilities.keys()
        assert all(capabilities[name].default is False for name in names)
        assert all(
            capabilities[name].cli_flags() == (f"--enable-{name}", f"--disable-{name}")
            for name in names
        )

        config = AlgebraicRewritePipe.build_config(
            static_split_to_slice=True,
            conv_channel_affine_folding=False,
        )
        assert config.static_split_to_slice is True
        assert config.conv_channel_affine_folding is False

    def test_cli_lists_algebraic_flag(self) -> None:
        result = CliRunner().invoke(optimize, ["--list-capabilities"])
        assert result.exit_code == 0
        assert "--enable-static-split-to-slice" in result.output
        assert "--enable-conv-channel-affine-folding" in result.output

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
        x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 6, 2])
        outputs = [_info("left", [1, 2, 2]), _info("right", [1, 4, 2])]
        split = onnx.helper.make_node(
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
        x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4, 2])
        keep = _info("keep", [1, 2, 2])
        split = onnx.helper.make_node(
            "Split",
            ["x"],
            ["part_a", "part_b"],
            name="",
            axis=1,
        )
        identity = onnx.helper.make_node(
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
        x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, None, 2])
        dynamic_equal = onnx.helper.make_node("Split", ["x"], ["a", "b"], axis=1)
        malformed = onnx.helper.make_node(
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
        x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4, 2])
        model = _model(
            [onnx.helper.make_node("Split", ["x"], ["left", "unused"], axis=1)],
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
        x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4, 2])
        then_branch = onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["left"], ["then_output"])],
            "then_branch",
            [],
            [_info("then_output", [1, 2, 2])],
        )
        else_branch = onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["right"], ["else_output"])],
            "else_branch",
            [],
            [_info("else_output", [1, 2, 2])],
        )
        model = _model(
            [
                onnx.helper.make_node("Split", ["x"], ["left", "right"], axis=1),
                onnx.helper.make_node(
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
        x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4, 2])
        model = _model(
            [onnx.helper.make_node("Split", ["x"], ["a", "b"], name="", axis=1)],
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


class TestConvChannelAffineFolding:
    """Test conservative direct and channel-routed affine folding."""

    @pytest.fixture
    def affine_model(self) -> tuple[onnx.ModelProto, dict[str, np.ndarray]]:
        rng = np.random.default_rng(11)
        x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2, 3, 3])
        nodes = [
            onnx.helper.make_node("Conv", ["x", "weights"], ["conv_out"], name=""),
            onnx.helper.make_node("Mul", ["conv_out", "scale"], ["mul_out"], name=""),
            onnx.helper.make_node("Add", ["offset", "mul_out"], ["y"], name=""),
        ]
        weights = rng.normal(size=(3, 2, 1, 1)).astype(np.float32)
        scale = rng.uniform(0.5, 1.5, size=(1, 3, 1, 1)).astype(np.float32)
        offset = rng.normal(size=(1, 3, 1, 1)).astype(np.float32)
        model = _model(
            nodes,
            [x],
            [_info("y", [1, 3, 3, 3])],
            [_tensor("weights", weights), _tensor("scale", scale), _tensor("offset", offset)],
            value_info=[
                _info("conv_out", [1, 3, 3, 3]),
                _info("mul_out", [1, 3, 3, 3]),
            ],
        )
        return model, {"x": rng.normal(size=(1, 2, 3, 3)).astype(np.float32)}

    def test_direct_affine_folding_is_exact_and_adds_optional_bias(
        self,
        affine_model: tuple[onnx.ModelProto, dict[str, np.ndarray]],
    ) -> None:
        model, values = affine_model
        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )
        assert [node.op_type for node in transformed.graph.node] == ["Conv"]
        assert transformed.graph.node[0].output[0] == "y"
        assert len(transformed.graph.node[0].input) == 3
        assert {initializer.name for initializer in transformed.graph.initializer} == set(
            transformed.graph.node[0].input[1:]
        )
        _assert_valid_with_inferred_shapes(transformed)
        np.testing.assert_allclose(
            _run(model, values),
            _run(transformed, values),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_shared_conv_output_is_ineligible(
        self,
        affine_model: tuple[onnx.ModelProto, dict[str, np.ndarray]],
    ) -> None:
        model, _ = affine_model
        model.graph.node.append(onnx.helper.make_node("Identity", ["conv_out"], ["other"]))
        model.graph.output.append(_info("other", [1, 3, 3, 3]))
        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )
        assert [node.op_type for node in transformed.graph.node] == [
            "Conv",
            "Mul",
            "Add",
            "Identity",
        ]

    def test_routed_view_graph_output_is_ineligible(self) -> None:
        rng = np.random.default_rng(19)
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weight"], ["conv_out"]),
                onnx.helper.make_node("Split", ["conv_out"], ["left", "right"], axis=1),
                onnx.helper.make_node("Reshape", ["left", "view_shape"], ["left_view"]),
                onnx.helper.make_node("Mul", ["left_view", "scale"], ["left_scaled"]),
                onnx.helper.make_node("Identity", ["right"], ["right_out"]),
            ],
            [_info("x", [1, 1, 2, 2])],
            [
                _info("left_view", [1, 1, 1, 2, 2]),
                _info("left_scaled", [1, 1, 1, 2, 2]),
                _info("right_out", [1, 1, 2, 2]),
            ],
            [
                _tensor("weight", rng.normal(size=(2, 1, 1, 1)).astype(np.float32)),
                _tensor("view_shape", np.asarray([1, 1, 1, 2, 2], dtype=np.int64)),
                _tensor("scale", np.asarray(1.5, dtype=np.float32)),
            ],
            value_info=[
                _info("conv_out", [1, 2, 2, 2]),
                _info("left", [1, 1, 2, 2]),
                _info("right", [1, 1, 2, 2]),
            ],
        )
        values = {"x": rng.normal(size=(1, 1, 2, 2)).astype(np.float32)}
        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )
        assert any(node.op_type == "Mul" for node in transformed.graph.node)
        _assert_valid_with_inferred_shapes(transformed)
        for original, rewritten in zip(
            _run(model, values),
            _run(transformed, values),
            strict=True,
        ):
            np.testing.assert_array_equal(original, rewritten)

    def test_static_split_branches_fold_without_overlapping_ranges(self) -> None:
        rng = np.random.default_rng(12)
        x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1, 2, 2])
        nodes = [
            onnx.helper.make_node("Conv", ["x", "weights"], ["conv_out"]),
            onnx.helper.make_node(
                "Split",
                ["conv_out", "sizes"],
                ["first", "second"],
                axis=1,
            ),
            onnx.helper.make_node("Mul", ["first", "scale_first"], ["first_scaled"]),
            onnx.helper.make_node("Add", ["first_scaled", "offset_first"], ["first_out"]),
            onnx.helper.make_node("Add", ["second", "offset_second"], ["second_out"]),
        ]
        model = _model(
            nodes,
            [x],
            [_info("first_out", [1, 2, 2, 2]), _info("second_out", [1, 2, 2, 2])],
            [
                _tensor("weights", rng.normal(size=(4, 1, 1, 1)).astype(np.float32)),
                _tensor("sizes", np.asarray([2, 2], dtype=np.int64)),
                _tensor("scale_first", rng.uniform(size=(1, 2, 1, 1)).astype(np.float32)),
                _tensor("offset_first", rng.normal(size=(1, 2, 1, 1)).astype(np.float32)),
                _tensor("offset_second", rng.normal(size=(1, 2, 1, 1)).astype(np.float32)),
            ],
            value_info=[
                _info("conv_out", [1, 4, 2, 2]),
                _info("first", [1, 2, 2, 2]),
                _info("second", [1, 2, 2, 2]),
                _info("first_scaled", [1, 2, 2, 2]),
            ],
        )
        values = {"x": rng.normal(size=(1, 1, 2, 2)).astype(np.float32)}
        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )
        assert [node.op_type for node in transformed.graph.node] == ["Conv", "Split"]
        _assert_valid_with_inferred_shapes(transformed)
        np.testing.assert_allclose(
            _run(model, values),
            _run(transformed, values),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_channel_preserving_views_and_nested_slices_fold(self) -> None:
        rng = np.random.default_rng(15)
        x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1, 2, 2])
        nodes = [
            onnx.helper.make_node("Conv", ["x", "weights"], ["conv_out"]),
            onnx.helper.make_node("Reshape", ["conv_out", "view_shape"], ["viewed"]),
            onnx.helper.make_node("Split", ["viewed", "split_sizes"], ["first", "second"], axis=1),
            onnx.helper.make_node("Squeeze", ["first", "squeeze_axes"], ["first_view"]),
            onnx.helper.make_node("Mul", ["first_view", "first_scale"], ["first_scaled"]),
            onnx.helper.make_node("Relu", ["first_scaled"], ["first_out"]),
            onnx.helper.make_node(
                "Slice",
                ["second", "slice_a_starts", "slice_a_ends"],
                ["second_a"],
            ),
            onnx.helper.make_node(
                "Slice",
                [
                    "second",
                    "slice_b_starts",
                    "slice_b_ends",
                    "slice_b_axes",
                    "slice_b_steps",
                ],
                ["second_b"],
            ),
            onnx.helper.make_node("Mul", ["second_a", "second_a_scale"], ["second_a_out"]),
            onnx.helper.make_node("Add", ["second_b", "second_b_offset"], ["second_b_out"]),
        ]
        model = _model(
            nodes,
            [x],
            [
                _info("first_out", [1, 2, 2, 2]),
                _info("second_a_out", [1, 1, 1, 2, 2]),
                _info("second_b_out", [1, 1, 1, 2, 2]),
            ],
            [
                _tensor("weights", rng.normal(size=(4, 1, 1, 1)).astype(np.float32)),
                _tensor("view_shape", np.asarray([1, 4, 1, 2, 2], dtype=np.int64)),
                _tensor("split_sizes", np.asarray([2, 2], dtype=np.int64)),
                _tensor("squeeze_axes", np.asarray([2], dtype=np.int64)),
                _tensor("slice_a_starts", np.asarray([0, 0, 0, 0, 0], dtype=np.int64)),
                _tensor("slice_a_ends", np.asarray([1, 1, 1, 2, 2], dtype=np.int64)),
                _tensor("slice_b_starts", np.asarray([0, 1, 0, 0, 0], dtype=np.int64)),
                _tensor(
                    "slice_b_ends",
                    np.full(5, np.iinfo(np.int64).max, dtype=np.int64),
                ),
                _tensor("slice_b_axes", np.asarray([0, 1, 2, 3, 4], dtype=np.int64)),
                _tensor("slice_b_steps", np.ones(5, dtype=np.int64)),
                _tensor("first_scale", np.asarray(1.25, dtype=np.float32)),
                _tensor("second_a_scale", np.asarray(0.75, dtype=np.float32)),
                _tensor("second_b_offset", np.asarray(-0.5, dtype=np.float32)),
            ],
            value_info=[
                _info("conv_out", [1, 4, 2, 2]),
                _info("viewed", [1, 4, 1, 2, 2]),
                _info("first", [1, 2, 1, 2, 2]),
                _info("second", [1, 2, 1, 2, 2]),
                _info("first_view", [1, 2, 2, 2]),
                _info("first_scaled", [1, 2, 2, 2]),
                _info("second_a", [1, 1, 1, 2, 2]),
                _info("second_b", [1, 1, 1, 2, 2]),
            ],
        )
        values = {"x": rng.normal(size=(1, 1, 2, 2)).astype(np.float32)}
        config = AlgebraicRewritePipeConfig(conv_channel_affine_folding=True)
        transformed = AlgebraicRewritePipe().process(model, config)
        second = AlgebraicRewritePipe().process(transformed, config)

        assert not any(node.op_type in {"Mul", "Add"} for node in transformed.graph.node)
        assert transformed.SerializeToString() == second.SerializeToString()
        _assert_valid_with_inferred_shapes(transformed)
        for original, rewritten in zip(
            _run(model, values),
            _run(transformed, values),
            strict=True,
        ):
            np.testing.assert_allclose(original, rewritten, rtol=2e-5, atol=2e-5)

        public = optimize_onnx(
            model,
            conv_channel_affine_folding=True,
            static_split_to_slice=True,
        )
        second_public = optimize_onnx(
            public,
            conv_channel_affine_folding=True,
            static_split_to_slice=True,
        )
        assert not any(node.op_type in {"Mul", "Add", "Split"} for node in public.graph.node)
        assert [
            (node.op_type, tuple(node.input), tuple(node.output), node.name)
            for node in public.graph.node
        ] == [
            (node.op_type, tuple(node.input), tuple(node.output), node.name)
            for node in second_public.graph.node
        ]
        _assert_valid_with_inferred_shapes(second_public)
        for original, rewritten in zip(
            _run(model, values),
            _run(second_public, values),
            strict=True,
        ):
            np.testing.assert_allclose(original, rewritten, rtol=2e-5, atol=2e-5)

    def test_multiple_independent_affine_matches_fold(self) -> None:
        rng = np.random.default_rng(16)
        nodes = [
            onnx.helper.make_node("Conv", ["x1", "weight1"], ["conv1"]),
            onnx.helper.make_node("Mul", ["conv1", "scale1"], ["y1"]),
            onnx.helper.make_node("Conv", ["x2", "weight2"], ["conv2"]),
            onnx.helper.make_node("Add", ["conv2", "offset2"], ["y2"]),
        ]
        model = _model(
            nodes,
            [_info("x1", [1, 1, 2, 2]), _info("x2", [1, 1, 2, 2])],
            [_info("y1", [1, 1, 2, 2]), _info("y2", [1, 1, 2, 2])],
            [
                _tensor("weight1", rng.normal(size=(1, 1, 1, 1)).astype(np.float32)),
                _tensor("scale1", np.asarray(1.5, dtype=np.float32)),
                _tensor("weight2", rng.normal(size=(1, 1, 1, 1)).astype(np.float32)),
                _tensor("offset2", np.asarray(-0.25, dtype=np.float32)),
            ],
            value_info=[_info("conv1", [1, 1, 2, 2]), _info("conv2", [1, 1, 2, 2])],
        )
        values = {
            "x1": rng.normal(size=(1, 1, 2, 2)).astype(np.float32),
            "x2": rng.normal(size=(1, 1, 2, 2)).astype(np.float32),
        }
        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )
        assert [node.op_type for node in transformed.graph.node] == ["Conv", "Conv"]
        for original, rewritten in zip(
            _run(model, values),
            _run(transformed, values),
            strict=True,
        ):
            np.testing.assert_allclose(original, rewritten, rtol=2e-5, atol=2e-5)

    def test_nested_subgraph_captures_make_affine_fold_ineligible(
        self,
        affine_model: tuple[onnx.ModelProto, dict[str, np.ndarray]],
    ) -> None:
        model, values = affine_model
        branch_shape = [1, 3, 3, 3]
        then_branch = onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["mul_out"], ["then_output"])],
            "then_branch",
            [],
            [_info("then_output", branch_shape)],
        )
        else_branch = onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["conv_out"], ["else_output"])],
            "else_branch",
            [],
            [_info("else_output", branch_shape)],
        )
        model.graph.initializer.append(_tensor("condition", np.asarray(True, dtype=np.bool_)))
        model.graph.node.append(
            onnx.helper.make_node(
                "If",
                ["condition"],
                ["captured"],
                then_branch=then_branch,
                else_branch=else_branch,
            )
        )
        model.graph.output.append(_info("captured", branch_shape))

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )
        assert [node.op_type for node in transformed.graph.node] == [
            "Conv",
            "Mul",
            "Add",
            "If",
        ]
        _assert_valid_with_inferred_shapes(transformed)
        for original, rewritten in zip(
            _run(model, values),
            _run(transformed, values),
            strict=True,
        ):
            np.testing.assert_allclose(original, rewritten, rtol=0, atol=0)

    def test_constant_attribute_affine_is_folded_and_pruned(self) -> None:
        rng = np.random.default_rng(18)
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weight"], ["conv_out"]),
                onnx.helper.make_node("Constant", [], ["scale"], value_float=1.5),
                onnx.helper.make_node("Mul", ["conv_out", "scale"], ["y"]),
            ],
            [_info("x", [1, 1, 2, 2])],
            [_info("y", [1, 1, 2, 2])],
            [_tensor("weight", rng.normal(size=(1, 1, 1, 1)).astype(np.float32))],
            value_info=[_info("conv_out", [1, 1, 2, 2])],
        )
        values = {"x": rng.normal(size=(1, 1, 2, 2)).astype(np.float32)}
        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )
        assert [node.op_type for node in transformed.graph.node] == ["Conv"]
        np.testing.assert_allclose(
            _run(model, values),
            _run(transformed, values),
            rtol=2e-5,
            atol=2e-5,
        )
