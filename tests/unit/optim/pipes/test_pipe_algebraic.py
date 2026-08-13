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
from winml.modelkit.optim.pipes import algebraic as algebraic_pipe


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


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


def _node_signatures(
    model: onnx.ModelProto,
) -> list[tuple[str, str, str, tuple[str, ...], tuple[str, ...]]]:
    return [
        (node.name, node.domain, node.op_type, tuple(node.input), tuple(node.output))
        for node in model.graph.node
    ]


def _assert_byte_identical(original: onnx.ModelProto, transformed: onnx.ModelProto) -> None:
    assert transformed.SerializeToString() == original.SerializeToString(), (
        f"graph mutated:\nbefore={_node_signatures(original)}\n"
        f"after={_node_signatures(transformed)}"
    )


class TestAlgebraicRegistration:
    """Verify capability registration, flags, and pipe ordering."""

    def test_capabilities_are_opt_in_and_independent(self) -> None:
        capabilities = get_all_capabilities()
        names = {
            "static-split-to-slice",
            "gather-slice-to-split-fusion",
            "conv-channel-affine-folding",
            "exp-positive-scale-folding",
        }
        assert names <= capabilities.keys()
        assert names <= AlgebraicRewritePipe.capabilities.keys()
        assert all(capabilities[name].default is False for name in names)
        assert all(
            capabilities[name].cli_flags() == (f"--enable-{name}", f"--disable-{name}")
            for name in names
        )

        config = AlgebraicRewritePipe.build_config(
            static_split_to_slice=True,
            gather_slice_to_split_fusion=True,
            conv_channel_affine_folding=False,
            exp_positive_scale_folding=True,
        )
        assert config.static_split_to_slice is True
        assert config.sibling_slice_to_split is True
        assert config.conv_channel_affine_folding is False
        assert config.exp_positive_scale_folding is True
        assert AlgebraicRewritePipe.should_process(config)

    def test_cli_lists_algebraic_flag(self) -> None:
        result = CliRunner().invoke(optimize, ["--list-capabilities"])
        assert result.exit_code == 0
        assert "--enable-static-split-to-slice" in result.output
        assert "--enable-gather-slice-to-split-fusion" in result.output
        assert "--enable-conv-channel-affine-folding" in result.output
        assert "--enable-exp-positive-scale-folding" in result.output

    def test_pipe_is_after_ort_graph_and_before_cleanup(self) -> None:
        names = [pipe.name for pipe in PIPES]
        assert names.index("ort_graph") < names.index("algebraic_rewrite")
        assert names.index("algebraic_rewrite") < names.index("surgery")
        assert PIPES[names.index("algebraic_rewrite")] is AlgebraicRewritePipe
        assert not AlgebraicRewritePipe.should_process(AlgebraicRewritePipeConfig())

    def test_cli_combines_split_affine_and_exp_folding(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(40)
        model = _model(
            [
                onnx.helper.make_node("Conv", ["conv_input", "weight"], ["conv_out"]),
                onnx.helper.make_node(
                    "Slice",
                    ["conv_out", "first_starts", "first_ends", "channel_axis"],
                    ["first"],
                ),
                onnx.helper.make_node(
                    "Slice",
                    ["conv_out", "second_starts", "second_ends", "channel_axis"],
                    ["second"],
                ),
                onnx.helper.make_node(
                    "Mul",
                    ["first", "first_scale"],
                    ["first_out"],
                    name="target_conv_mul",
                ),
                onnx.helper.make_node(
                    "Add",
                    ["second", "second_offset"],
                    ["second_out"],
                    name="target_conv_add",
                ),
                onnx.helper.make_node(
                    "Add",
                    ["exp_input", "exp_bias"],
                    ["biased"],
                    name="retained_exp_add",
                ),
                onnx.helper.make_node("Exp", ["biased"], ["exponential"]),
                onnx.helper.make_node(
                    "Mul",
                    ["exponential", "exp_scale"],
                    ["exp_out"],
                    name="target_exp_mul",
                ),
            ],
            [_info("conv_input", [1, 1, 2, 2]), _info("exp_input", [1, 2])],
            [
                _info("first_out", [1, 2, 2, 2]),
                _info("second_out", [1, 2, 2, 2]),
                _info("exp_out", [1, 2]),
            ],
            [
                _tensor("weight", rng.normal(size=(4, 1, 1, 1)).astype(np.float32)),
                _tensor("first_starts", np.asarray([0], dtype=np.int64)),
                _tensor("first_ends", np.asarray([2], dtype=np.int64)),
                _tensor("second_starts", np.asarray([2], dtype=np.int64)),
                _tensor("second_ends", np.asarray([4], dtype=np.int64)),
                _tensor("channel_axis", np.asarray([1], dtype=np.int64)),
                _tensor(
                    "first_scale", np.asarray([1.25, 0.75], dtype=np.float32).reshape(1, 2, 1, 1)
                ),
                _tensor(
                    "second_offset", np.asarray([0.5, -0.5], dtype=np.float32).reshape(1, 2, 1, 1)
                ),
                _tensor("exp_bias", np.asarray(-1.0, dtype=np.float32)),
                _tensor("exp_scale", np.asarray([1.5, 2.0], dtype=np.float32)),
            ],
            value_info=[
                _info("conv_out", [1, 4, 2, 2]),
                _info("first", [1, 2, 2, 2]),
                _info("second", [1, 2, 2, 2]),
                _info("biased", [1, 2]),
                _info("exponential", [1, 2]),
            ],
        )
        input_path = tmp_path / "input.onnx"
        output_path = tmp_path / "output.onnx"
        onnx.save_model(model, input_path)

        result = CliRunner().invoke(
            optimize,
            [
                "-m",
                str(input_path),
                "-o",
                str(output_path),
                "--enable-gather-slice-to-split-fusion",
                "--enable-conv-channel-affine-folding",
                "--enable-exp-positive-scale-folding",
                "--no-color",
            ],
        )

        assert result.exit_code == 0, result.output
        transformed = onnx.load_model(output_path)
        names = {node.name for node in transformed.graph.node}
        assert not {"target_conv_mul", "target_conv_add", "target_exp_mul"} & names
        assert any(node.op_type == "Split" for node in transformed.graph.node)
        assert [output.SerializeToString() for output in transformed.graph.output] == [
            output.SerializeToString() for output in model.graph.output
        ]
        _assert_valid_with_inferred_shapes(transformed)
        values = {
            "conv_input": rng.normal(size=(1, 1, 2, 2)).astype(np.float32),
            "exp_input": rng.normal(size=(1, 2)).astype(np.float32),
        }
        for original, rewritten in zip(
            _run(model, values),
            _run(transformed, values),
            strict=True,
        ):
            np.testing.assert_allclose(original, rewritten, rtol=2e-5, atol=2e-5)


class TestNameAllocator:
    """Test generated-name allocation behavior used by batched rewrites."""

    def test_repeated_prefix_allocation_does_not_restart_suffix_probe(self) -> None:
        class CountingNames(set[str]):
            def __init__(self) -> None:
                super().__init__()
                self.contains_count = 0

            def __contains__(self, value: object) -> bool:
                self.contains_count += 1
                return super().__contains__(value)

        allocator = algebraic_pipe._NameAllocator(_model([], [], [], []))
        used = CountingNames()
        allocator._used = used

        names = [allocator.new("algebraic_exp_log_scale") for _ in range(32)]

        assert names == ["algebraic_exp_log_scale"] + [
            f"algebraic_exp_log_scale_{suffix}" for suffix in range(1, 32)
        ]
        assert used.contains_count <= 40


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

    def test_overridable_split_sizes_are_unchanged(self) -> None:
        model = _model(
            [
                onnx.helper.make_node(
                    "Split",
                    ["x", "split_sizes"],
                    ["left", "right"],
                    axis=1,
                )
            ],
            [
                _info("x", [1, 4, 2]),
                onnx.helper.make_tensor_value_info(
                    "split_sizes",
                    onnx.TensorProto.INT64,
                    [2],
                ),
            ],
            [_info("left", [1, 2, 2]), _info("right", [1, 2, 2])],
            [_tensor("split_sizes", np.asarray([2, 2], dtype=np.int64))],
        )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(static_split_to_slice=True),
        )

        assert transformed.SerializeToString() == original

    @pytest.mark.parametrize(
        ("domain", "should_rewrite"),
        [("", True), ("ai.onnx", False), ("com.example", False)],
    )
    def test_only_standard_domain_split_is_rewritten(
        self,
        domain: str,
        should_rewrite: bool,
    ) -> None:
        model = _model(
            [
                onnx.helper.make_node(
                    "Split",
                    ["x", "split_sizes"],
                    ["left", "right"],
                    axis=1,
                    domain=domain,
                )
            ],
            [_info("x", [1, 4, 2])],
            [_info("left", [1, 2, 2]), _info("right", [1, 2, 2])],
            [_tensor("split_sizes", np.asarray([2, 2], dtype=np.int64))],
        )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(static_split_to_slice=True),
        )

        if should_rewrite:
            assert [node.op_type for node in transformed.graph.node] == ["Slice", "Slice"]
        else:
            assert transformed.SerializeToString() == original

    def test_legacy_ir_generated_initializer_rewrite_is_unchanged(self) -> None:
        model = _model(
            [onnx.helper.make_node("Split", ["x", "split_sizes"], ["left", "right"], axis=1)],
            [_info("x", [1, 4, 2])],
            [_info("left", [1, 2, 2]), _info("right", [1, 2, 2])],
            [_tensor("split_sizes", np.asarray([2, 2], dtype=np.int64))],
        )
        model.ir_version = 3
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(static_split_to_slice=True),
        )

        assert transformed.SerializeToString() == original

    @pytest.mark.parametrize(
        "outputs",
        [("", "right"), ("part", "part"), ("x", "right")],
    )
    def test_malformed_split_outputs_are_unchanged(self, outputs: tuple[str, str]) -> None:
        model = _model(
            [onnx.helper.make_node("Split", ["x"], list(outputs), axis=1)],
            [_info("x", [1, 4, 2])],
            [_info("y", [1, 2, 2])],
            [],
        )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(static_split_to_slice=True),
        )

        assert transformed.SerializeToString() == original

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

    def test_sibling_static_slices_fold_to_split(self) -> None:
        values = {"x": np.arange(12, dtype=np.float32).reshape(1, 6, 2)}
        model = _model(
            [
                onnx.helper.make_node(
                    "Slice",
                    ["x", "left_starts", "left_ends", "axis", "steps"],
                    ["left"],
                    name="left_slice",
                ),
                onnx.helper.make_node(
                    "Slice",
                    ["x", "right_starts", "right_ends", "axis", "steps"],
                    ["right"],
                    name="right_slice",
                ),
                onnx.helper.make_node("Relu", ["left"], ["left_out"]),
                onnx.helper.make_node("Relu", ["right"], ["right_out"]),
            ],
            [_info("x", [1, 6, 2])],
            [_info("left_out", [1, 2, 2]), _info("right_out", [1, 4, 2])],
            [
                _tensor("left_starts", np.asarray([0], dtype=np.int64)),
                _tensor("left_ends", np.asarray([2], dtype=np.int64)),
                _tensor("right_starts", np.asarray([2], dtype=np.int64)),
                _tensor("right_ends", np.asarray([6], dtype=np.int64)),
                _tensor("axis", np.asarray([1], dtype=np.int64)),
                _tensor("steps", np.asarray([1], dtype=np.int64)),
            ],
            value_info=[_info("left", [1, 2, 2]), _info("right", [1, 4, 2])],
        )

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipe.build_config(gather_slice_to_split_fusion=True),
        )

        assert [node.op_type for node in transformed.graph.node] == ["Split", "Relu", "Relu"]
        split = transformed.graph.node[0]
        assert list(split.input[:1]) == ["x"]
        assert list(split.output) == ["left", "right"]
        split_sizes_name = split.input[1]
        split_sizes = onnx.numpy_helper.to_array(
            next(value for value in transformed.graph.initializer if value.name == split_sizes_name)
        )
        np.testing.assert_array_equal(split_sizes, np.asarray([2, 4], dtype=np.int64))
        _assert_valid_with_inferred_shapes(transformed)
        for original, rewritten in zip(_run(model, values), _run(transformed, values), strict=True):
            np.testing.assert_array_equal(original, rewritten)

    def test_sibling_static_slices_are_unchanged_before_split_input_opset(self) -> None:
        model = _model(
            [
                onnx.helper.make_node(
                    "Slice",
                    ["x", "left_starts", "left_ends", "axis", "steps"],
                    ["left"],
                ),
                onnx.helper.make_node(
                    "Slice",
                    ["x", "right_starts", "right_ends", "axis", "steps"],
                    ["right"],
                ),
            ],
            [_info("x", [1, 6, 2])],
            [_info("left", [1, 2, 2]), _info("right", [1, 4, 2])],
            [
                _tensor("left_starts", np.asarray([0], dtype=np.int64)),
                _tensor("left_ends", np.asarray([2], dtype=np.int64)),
                _tensor("right_starts", np.asarray([2], dtype=np.int64)),
                _tensor("right_ends", np.asarray([6], dtype=np.int64)),
                _tensor("axis", np.asarray([1], dtype=np.int64)),
                _tensor("steps", np.asarray([1], dtype=np.int64)),
            ],
        )
        model.opset_import[0].version = 12
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipe.build_config(gather_slice_to_split_fusion=True),
        )

        assert transformed.SerializeToString() == original

    @pytest.mark.parametrize(
        "opset_imports",
        [
            [onnx.helper.make_opsetid("ai.onnx", 17), onnx.helper.make_opsetid("", 12)],
            [onnx.helper.make_opsetid("", 17), onnx.helper.make_opsetid("", 12)],
        ],
    )
    def test_sibling_static_slices_are_unchanged_for_ambiguous_standard_opset(
        self,
        opset_imports: list[onnx.OperatorSetIdProto],
    ) -> None:
        model = _model(
            [
                onnx.helper.make_node(
                    "Slice",
                    ["x", "left_starts", "left_ends", "axis", "steps"],
                    ["left"],
                ),
                onnx.helper.make_node(
                    "Slice",
                    ["x", "right_starts", "right_ends", "axis", "steps"],
                    ["right"],
                ),
            ],
            [_info("x", [1, 6, 2])],
            [_info("left", [1, 2, 2]), _info("right", [1, 4, 2])],
            [
                _tensor("left_starts", np.asarray([0], dtype=np.int64)),
                _tensor("left_ends", np.asarray([2], dtype=np.int64)),
                _tensor("right_starts", np.asarray([2], dtype=np.int64)),
                _tensor("right_ends", np.asarray([6], dtype=np.int64)),
                _tensor("axis", np.asarray([1], dtype=np.int64)),
                _tensor("steps", np.asarray([1], dtype=np.int64)),
            ],
        )
        del model.opset_import[:]
        model.opset_import.extend(opset_imports)
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipe.build_config(gather_slice_to_split_fusion=True),
        )

        assert transformed.SerializeToString() == original

    @pytest.mark.parametrize(
        ("left_starts", "left_ends", "right_starts", "right_ends", "axes", "steps"),
        [
            ([0], [2], [3], [6], [1], [1]),
            ([0], [3], [2], [6], [1], [1]),
            ([1], [2], [2], [6], [1], [1]),
            ([0], [2], [2], [6], [1], [2]),
            ([0, 0], [1, 2], [1, 0], [6, 2], [0, 1], [1, 1]),
        ],
    )
    def test_ineligible_sibling_static_slices_are_unchanged(
        self,
        left_starts: list[int],
        left_ends: list[int],
        right_starts: list[int],
        right_ends: list[int],
        axes: list[int],
        steps: list[int],
    ) -> None:
        model = _model(
            [
                onnx.helper.make_node(
                    "Slice",
                    ["x", "left_starts", "left_ends", "axes", "steps"],
                    ["left"],
                ),
                onnx.helper.make_node(
                    "Slice",
                    ["x", "right_starts", "right_ends", "axes", "steps"],
                    ["right"],
                ),
            ],
            [_info("x", [1, 6, 2])],
            [_info("left", [1, 2, 2]), _info("right", [1, 4, 2])],
            [
                _tensor("left_starts", np.asarray(left_starts, dtype=np.int64)),
                _tensor("left_ends", np.asarray(left_ends, dtype=np.int64)),
                _tensor("right_starts", np.asarray(right_starts, dtype=np.int64)),
                _tensor("right_ends", np.asarray(right_ends, dtype=np.int64)),
                _tensor("axes", np.asarray(axes, dtype=np.int64)),
                _tensor("steps", np.asarray(steps, dtype=np.int64)),
            ],
        )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipe.build_config(gather_slice_to_split_fusion=True),
        )

        assert transformed.SerializeToString() == original

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
        assert not transformed.graph.value_info
        _assert_valid_with_inferred_shapes(transformed)
        np.testing.assert_allclose(
            _run(model, values),
            _run(transformed, values),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_legacy_opset_conv_affine_broadcast_is_unchanged(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weights"], ["conv_out"]),
                onnx.helper.make_node(
                    "Mul",
                    ["conv_out", "scale"],
                    ["y"],
                    broadcast=1,
                    axis=0,
                ),
            ],
            [_info("x", [1, 2, 2, 2])],
            [_info("y", [1, 3, 2, 2])],
            [
                _tensor("weights", np.ones((3, 2, 1, 1), dtype=np.float32)),
                _tensor("scale", np.ones((3, 1, 1), dtype=np.float32)),
            ],
            value_info=[_info("conv_out", [1, 3, 2, 2])],
        )
        model.opset_import[0].version = 6
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        assert transformed.SerializeToString() == original

    def test_nonfinite_synthesized_conv_parameters_are_unchanged(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weight"], ["conv_out"]),
                onnx.helper.make_node("Split", ["conv_out"], ["branch"], axis=1),
                onnx.helper.make_node("Split", ["branch"], ["leaf"], axis=1),
                onnx.helper.make_node("Mul", ["leaf", "scale_a"], ["scaled_a"]),
                onnx.helper.make_node("Mul", ["scaled_a", "scale_b"], ["y"]),
            ],
            [_info("x", [1, 1, 1, 1])],
            [_info("y", [1, 1, 1, 1])],
            [
                _tensor("weight", np.asarray([[[[1.0e-38]]]], dtype=np.float32)),
                _tensor("scale_a", np.asarray(1.0e20, dtype=np.float32)),
                _tensor("scale_b", np.asarray(1.0e20, dtype=np.float32)),
            ],
            value_info=[
                _info("conv_out", [1, 1, 1, 1]),
                _info("branch", [1, 1, 1, 1]),
                _info("leaf", [1, 1, 1, 1]),
                _info("scaled_a", [1, 1, 1, 1]),
            ],
        )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        assert transformed.SerializeToString() == original

    def test_nonfinite_synthesized_conv_bias_does_not_partially_mutate(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weight", "bias"], ["conv_out"]),
                onnx.helper.make_node("Mul", ["conv_out", "scale"], ["y"]),
            ],
            [_info("x", [1, 1, 1, 1])],
            [_info("y", [1, 1, 1, 1])],
            [
                _tensor("weight", np.ones((1, 1, 1, 1), dtype=np.float32)),
                _tensor("bias", np.asarray([1.0e38], dtype=np.float32)),
                _tensor("scale", np.asarray(10.0, dtype=np.float32)),
            ],
            value_info=[_info("conv_out", [1, 1, 1, 1])],
        )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        assert transformed.SerializeToString() == original

    def test_float64_affine_values_preserve_weight_precision(self) -> None:
        shape = [1, 1, 1, 1]

        def tensor_info(name: str) -> onnx.ValueInfoProto:
            return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.DOUBLE, shape)

        weight = np.ones(shape, dtype=np.float64)
        scale = np.asarray(1.0 + 2**-30, dtype=np.float64)
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weight"], ["conv_out"]),
                onnx.helper.make_node("Mul", ["conv_out", "scale"], ["y"]),
            ],
            [tensor_info("x")],
            [tensor_info("y")],
            [_tensor("weight", weight), _tensor("scale", scale)],
            value_info=[tensor_info("conv_out")],
        )

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )
        folded_weight = next(
            initializer
            for initializer in transformed.graph.initializer
            if initializer.name == transformed.graph.node[0].input[1]
        )
        folded_values = onnx.numpy_helper.to_array(folded_weight)
        assert folded_values.dtype == np.float64
        np.testing.assert_array_equal(folded_values, weight * scale)

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

    @pytest.mark.parametrize("operand_name", ["scale", "offset"])
    def test_overridable_affine_operands_are_unchanged(
        self,
        affine_model: tuple[onnx.ModelProto, dict[str, np.ndarray]],
        operand_name: str,
    ) -> None:
        model, _ = affine_model
        initializer = next(value for value in model.graph.initializer if value.name == operand_name)
        model.graph.input.append(
            onnx.helper.make_tensor_value_info(
                operand_name,
                initializer.data_type,
                list(initializer.dims),
            )
        )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        assert transformed.SerializeToString() == original

    @pytest.mark.parametrize("parameter_name", ["weight", "bias"])
    def test_overridable_conv_parameters_are_unchanged(self, parameter_name: str) -> None:
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weight", "bias"], ["conv_out"]),
                onnx.helper.make_node("Mul", ["conv_out", "scale"], ["y"]),
            ],
            [_info("x", [1, 1, 2, 2])],
            [_info("y", [1, 1, 2, 2])],
            [
                _tensor("weight", np.ones((1, 1, 1, 1), dtype=np.float32)),
                _tensor("bias", np.ones(1, dtype=np.float32)),
                _tensor("scale", np.asarray(1.5, dtype=np.float32)),
            ],
            value_info=[_info("conv_out", [1, 1, 2, 2])],
        )
        parameter = next(value for value in model.graph.initializer if value.name == parameter_name)
        model.graph.input.append(
            onnx.helper.make_tensor_value_info(
                parameter_name,
                parameter.data_type,
                list(parameter.dims),
            )
        )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        assert transformed.SerializeToString() == original

    @pytest.mark.parametrize("parameter_name", ["weight", "bias"])
    def test_unloaded_external_conv_parameter_is_unchanged(
        self,
        parameter_name: str,
    ) -> None:
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weight", "bias"], ["conv_out"]),
                onnx.helper.make_node("Mul", ["conv_out", "scale"], ["y"]),
            ],
            [_info("x", [1, 1, 2, 2])],
            [_info("y", [1, 1, 2, 2])],
            [
                _tensor("weight", np.ones((1, 1, 1, 1), dtype=np.float32)),
                _tensor("bias", np.ones(1, dtype=np.float32)),
                _tensor("scale", np.asarray(1.5, dtype=np.float32)),
            ],
            value_info=[_info("conv_out", [1, 1, 2, 2])],
        )
        parameter = next(value for value in model.graph.initializer if value.name == parameter_name)
        parameter.ClearField("raw_data")
        parameter.data_location = onnx.TensorProto.EXTERNAL
        location = parameter.external_data.add()
        location.key = "location"
        location.value = f"missing-{parameter_name}.bin"
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        assert transformed.SerializeToString() == original

    @pytest.mark.parametrize("affine_op", ["Mul", "Add"])
    @pytest.mark.parametrize("affine_value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_affine_operand_is_unchanged(
        self,
        affine_op: str,
        affine_value: float,
    ) -> None:
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weight"], ["conv_out"]),
                onnx.helper.make_node("Constant", [], ["affine"], value_float=affine_value),
                onnx.helper.make_node(affine_op, ["conv_out", "affine"], ["y"]),
            ],
            [_info("x", [1, 1, 2, 2])],
            [_info("y", [1, 1, 2, 2])],
            [_tensor("weight", np.ones((1, 1, 1, 1), dtype=np.float32))],
            value_info=[_info("conv_out", [1, 1, 2, 2])],
        )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        assert transformed.SerializeToString() == original

    @pytest.mark.parametrize(
        "constant_name",
        [
            "view_shape",
            "split_sizes",
            "slice_starts",
            "slice_ends",
            "slice_axes",
            "slice_steps",
        ],
    )
    def test_overridable_route_constants_are_unchanged(self, constant_name: str) -> None:
        shape = [1, 2, 1, 2, 2]
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weight"], ["conv_out"]),
                onnx.helper.make_node("Reshape", ["conv_out", "view_shape"], ["viewed"]),
                onnx.helper.make_node(
                    "Split",
                    ["viewed", "split_sizes"],
                    ["split_out"],
                    axis=1,
                ),
                onnx.helper.make_node(
                    "Slice",
                    [
                        "split_out",
                        "slice_starts",
                        "slice_ends",
                        "slice_axes",
                        "slice_steps",
                    ],
                    ["sliced"],
                ),
                onnx.helper.make_node("Mul", ["sliced", "scale"], ["y"]),
            ],
            [_info("x", [1, 1, 2, 2])],
            [_info("y", shape)],
            [
                _tensor("weight", np.ones((2, 1, 1, 1), dtype=np.float32)),
                _tensor("view_shape", np.asarray(shape, dtype=np.int64)),
                _tensor("split_sizes", np.asarray([2], dtype=np.int64)),
                _tensor("slice_starts", np.asarray([0], dtype=np.int64)),
                _tensor("slice_ends", np.asarray([2], dtype=np.int64)),
                _tensor("slice_axes", np.asarray([1], dtype=np.int64)),
                _tensor("slice_steps", np.asarray([1], dtype=np.int64)),
                _tensor("scale", np.asarray(1.5, dtype=np.float32)),
            ],
            value_info=[
                _info("conv_out", [1, 2, 2, 2]),
                _info("viewed", shape),
                _info("split_out", shape),
                _info("sliced", shape),
            ],
        )
        initializer = next(
            value for value in model.graph.initializer if value.name == constant_name
        )
        model.graph.input.append(
            onnx.helper.make_tensor_value_info(
                constant_name,
                initializer.data_type,
                list(initializer.dims),
            )
        )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        assert transformed.SerializeToString() == original

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

    def test_reshape_allowzero_without_literal_zero_folds_affine_branches(self) -> None:
        rng = np.random.default_rng(20)
        branch_shape = [1, 1, 1, 2, 2]
        model = _model(
            [
                onnx.helper.make_node(
                    "Conv",
                    ["x", "weights"],
                    ["conv_out"],
                    name="conv",
                ),
                onnx.helper.make_node(
                    "Reshape",
                    ["conv_out", "view_shape"],
                    ["viewed"],
                    name="allowzero_view",
                    allowzero=1,
                ),
                onnx.helper.make_node(
                    "Split",
                    ["viewed", "split_sizes"],
                    ["scalar_branch", "affine_branch", "nonlinear_branch"],
                    name="channel_split",
                    axis=1,
                ),
                onnx.helper.make_node(
                    "Mul",
                    ["scalar_branch", "scalar_scale"],
                    ["scalar_out"],
                    name="scalar_mul",
                ),
                onnx.helper.make_node(
                    "Mul",
                    ["affine_branch", "affine_scale"],
                    ["affine_scaled"],
                    name="affine_mul",
                ),
                onnx.helper.make_node(
                    "Add",
                    ["affine_scaled", "affine_offset"],
                    ["affine_out"],
                    name="affine_add",
                ),
                onnx.helper.make_node(
                    "Relu",
                    ["nonlinear_branch"],
                    ["nonlinear_out"],
                    name="nonlinear_relu",
                ),
            ],
            [_info("x", [1, 1, 2, 2])],
            [
                _info("scalar_out", branch_shape),
                _info("affine_out", branch_shape),
                _info("nonlinear_out", branch_shape),
            ],
            [
                _tensor("weights", rng.normal(size=(3, 1, 1, 1)).astype(np.float32)),
                _tensor("view_shape", np.asarray([1, -1, 1, 2, 2], dtype=np.int64)),
                _tensor("split_sizes", np.asarray([1, 1, 1], dtype=np.int64)),
                _tensor("scalar_scale", np.asarray(1.25, dtype=np.float32)),
                _tensor("affine_scale", np.asarray([[[[[0.75]]]]], dtype=np.float32)),
                _tensor("affine_offset", np.asarray([[[[[-0.5]]]]], dtype=np.float32)),
            ],
            value_info=[
                _info("conv_out", [1, 3, 2, 2]),
                _info("viewed", [1, 3, 1, 2, 2]),
                _info("scalar_branch", branch_shape),
                _info("affine_branch", branch_shape),
                _info("affine_scaled", branch_shape),
                _info("nonlinear_branch", branch_shape),
            ],
        )
        values = {"x": rng.normal(size=(1, 1, 2, 2)).astype(np.float32)}
        config = AlgebraicRewritePipeConfig(conv_channel_affine_folding=True)
        transformed = AlgebraicRewritePipe().process(model, config)
        second = AlgebraicRewritePipe().process(transformed, config)

        remaining_names = {node.name for node in transformed.graph.node}
        assert not {"scalar_mul", "affine_mul", "affine_add"} & remaining_names
        assert "nonlinear_relu" in remaining_names
        assert transformed.SerializeToString() == second.SerializeToString()
        _assert_valid_with_inferred_shapes(transformed)
        for original, rewritten in zip(
            _run(model, values),
            _run(transformed, values),
            strict=True,
        ):
            np.testing.assert_allclose(original, rewritten, rtol=2e-5, atol=2e-5)

    def test_reshape_allowzero_with_literal_zero_keeps_affine_nodes(self) -> None:
        rng = np.random.default_rng(21)
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weights"], ["conv_out"]),
                onnx.helper.make_node(
                    "Reshape",
                    ["conv_out", "view_shape"],
                    ["viewed"],
                    allowzero=1,
                ),
                onnx.helper.make_node(
                    "Mul",
                    ["viewed", "scale"],
                    ["scaled"],
                    name="zero_shape_mul",
                ),
                onnx.helper.make_node(
                    "Add",
                    ["scaled", "offset"],
                    ["y"],
                    name="zero_shape_add",
                ),
            ],
            [_info("x", [0, 1, 2, 2])],
            [_info("y", [0, 2, 2, 2])],
            [
                _tensor("weights", rng.normal(size=(2, 1, 1, 1)).astype(np.float32)),
                _tensor("view_shape", np.asarray([0, 2, 2, 2], dtype=np.int64)),
                _tensor("scale", np.asarray(1.25, dtype=np.float32)),
                _tensor("offset", np.asarray(-0.5, dtype=np.float32)),
            ],
            value_info=[
                _info("conv_out", [0, 2, 2, 2]),
                _info("viewed", [0, 2, 2, 2]),
                _info("scaled", [0, 2, 2, 2]),
            ],
        )

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        remaining_names = {node.name for node in transformed.graph.node}
        assert {"zero_shape_mul", "zero_shape_add"} <= remaining_names
        _assert_valid_with_inferred_shapes(transformed)

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

    def test_nested_static_channel_splits_fold_affine_leaves(self) -> None:
        rng = np.random.default_rng(22)
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weights"], ["conv_out"]),
                onnx.helper.make_node(
                    "Split",
                    ["conv_out", "outer_sizes"],
                    ["depth", "colors", "keep"],
                    name="outer_split",
                    axis=1,
                ),
                onnx.helper.make_node("Mul", ["depth", "depth_scale"], ["depth_out"]),
                onnx.helper.make_node(
                    "Split",
                    ["colors", "inner_sizes"],
                    ["rgb", "sh"],
                    name="inner_split",
                    axis=1,
                ),
                onnx.helper.make_node("Mul", ["rgb", "rgb_scale"], ["rgb_scaled"]),
                onnx.helper.make_node("Add", ["rgb_scaled", "rgb_offset"], ["rgb_affine"]),
                onnx.helper.make_node("Sigmoid", ["rgb_affine"], ["rgb_out"]),
                onnx.helper.make_node("Mul", ["sh", "sh_scale"], ["sh_scaled"]),
                onnx.helper.make_node("Add", ["sh_scaled", "sh_offset"], ["sh_out"]),
                onnx.helper.make_node("Relu", ["keep"], ["keep_out"]),
            ],
            [_info("x", [1, 1, 2, 2])],
            [
                _info("depth_out", [1, 1, 2, 2]),
                _info("rgb_out", [1, 1, 2, 2]),
                _info("sh_out", [1, 3, 2, 2]),
                _info("keep_out", [1, 1, 2, 2]),
            ],
            [
                _tensor("weights", rng.normal(size=(6, 1, 1, 1)).astype(np.float32)),
                _tensor("outer_sizes", np.asarray([1, 4, 1], dtype=np.int64)),
                _tensor("inner_sizes", np.asarray([1, 3], dtype=np.int64)),
                _tensor("depth_scale", np.asarray(0.25, dtype=np.float32)),
                _tensor("rgb_scale", np.asarray(0.75, dtype=np.float32)),
                _tensor("rgb_offset", np.asarray(-0.125, dtype=np.float32)),
                _tensor("sh_scale", np.asarray(1.5, dtype=np.float32)),
                _tensor("sh_offset", np.asarray(0.25, dtype=np.float32)),
            ],
            value_info=[
                _info("conv_out", [1, 6, 2, 2]),
                _info("depth", [1, 1, 2, 2]),
                _info("colors", [1, 4, 2, 2]),
                _info("keep", [1, 1, 2, 2]),
                _info("rgb", [1, 1, 2, 2]),
                _info("sh", [1, 3, 2, 2]),
                _info("rgb_scaled", [1, 1, 2, 2]),
                _info("rgb_affine", [1, 1, 2, 2]),
                _info("sh_scaled", [1, 3, 2, 2]),
            ],
        )
        values = {"x": rng.normal(size=(1, 1, 2, 2)).astype(np.float32)}
        config = AlgebraicRewritePipeConfig(conv_channel_affine_folding=True)

        transformed = AlgebraicRewritePipe().process(model, config)
        second = AlgebraicRewritePipe().process(transformed, config)

        assert not any(node.op_type in {"Mul", "Add"} for node in transformed.graph.node)
        assert {node.name for node in transformed.graph.node if node.op_type == "Split"} == {
            "outer_split",
            "inner_split",
        }
        assert transformed.SerializeToString() == second.SerializeToString()
        _assert_valid_with_inferred_shapes(transformed)
        for original, rewritten in zip(
            _run(model, values),
            _run(transformed, values),
            strict=True,
        ):
            np.testing.assert_allclose(original, rewritten, rtol=2e-5, atol=2e-5)

    @pytest.mark.parametrize("case", ["custom_domain", "non_channel", "dynamic", "malformed"])
    def test_invalid_nested_split_is_unchanged(self, case: str) -> None:
        nested_inputs = ["branch"]
        nested_domain = ""
        nested_axis = 1
        inputs = [_info("x", [1, 1, 2, 2])]
        initializers = [
            _tensor("weight", np.ones((1, 1, 1, 1), dtype=np.float32)),
            _tensor("scale", np.asarray(1.5, dtype=np.float32)),
        ]
        if case == "custom_domain":
            nested_domain = "com.example"
        elif case == "non_channel":
            nested_axis = 2
        elif case == "dynamic":
            nested_inputs.append("nested_sizes")
            inputs.append(
                onnx.helper.make_tensor_value_info(
                    "nested_sizes",
                    onnx.TensorProto.INT64,
                    [1],
                )
            )
        else:
            nested_inputs.append("nested_sizes")
            initializers.append(_tensor("nested_sizes", np.asarray([2], dtype=np.int64)))
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weight"], ["conv_out"]),
                onnx.helper.make_node("Split", ["conv_out"], ["branch"], axis=1),
                onnx.helper.make_node(
                    "Split",
                    nested_inputs,
                    ["leaf"],
                    axis=nested_axis,
                    domain=nested_domain,
                ),
                onnx.helper.make_node("Mul", ["leaf", "scale"], ["y"]),
            ],
            inputs,
            [_info("y", [1, 1, 2, 2])],
            initializers,
            value_info=[
                _info("conv_out", [1, 1, 2, 2]),
                _info("branch", [1, 1, 2, 2]),
                _info("leaf", [1, 1, 2, 2]),
            ],
        )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        assert transformed.SerializeToString() == original

    def test_duplicate_nested_split_outputs_are_unchanged(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weight"], ["conv_out"]),
                onnx.helper.make_node("Split", ["conv_out"], ["branch"], axis=1),
                onnx.helper.make_node(
                    "Split",
                    ["branch"],
                    ["leaf", "leaf"],
                    axis=1,
                ),
                onnx.helper.make_node("Mul", ["leaf", "scale"], ["y"]),
            ],
            [_info("x", [1, 1, 2, 2])],
            [_info("y", [1, 1, 2, 2])],
            [
                _tensor("weight", np.ones((2, 1, 1, 1), dtype=np.float32)),
                _tensor("scale", np.asarray(1.5, dtype=np.float32)),
            ],
            value_info=[
                _info("conv_out", [1, 2, 2, 2]),
                _info("branch", [1, 2, 2, 2]),
                _info("leaf", [1, 1, 2, 2]),
            ],
        )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        assert transformed.SerializeToString() == original

    def test_duplicate_routed_tensor_and_affine_node_are_unchanged(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weight"], ["conv_out"]),
                onnx.helper.make_node("Split", ["conv_out"], ["branch"], axis=1),
                onnx.helper.make_node(
                    "Slice",
                    ["branch", "left_starts", "left_ends", "axes"],
                    ["leaf"],
                ),
                onnx.helper.make_node(
                    "Slice",
                    ["branch", "right_starts", "right_ends", "axes"],
                    ["leaf"],
                ),
                onnx.helper.make_node("Mul", ["leaf", "scale"], ["y"]),
            ],
            [_info("x", [1, 1, 2, 2])],
            [_info("y", [1, 1, 2, 2])],
            [
                _tensor("weight", np.ones((2, 1, 1, 1), dtype=np.float32)),
                _tensor("left_starts", np.asarray([0], dtype=np.int64)),
                _tensor("left_ends", np.asarray([1], dtype=np.int64)),
                _tensor("right_starts", np.asarray([1], dtype=np.int64)),
                _tensor("right_ends", np.asarray([2], dtype=np.int64)),
                _tensor("axes", np.asarray([1], dtype=np.int64)),
                _tensor("scale", np.asarray(1.5, dtype=np.float32)),
            ],
            value_info=[
                _info("conv_out", [1, 2, 2, 2]),
                _info("branch", [1, 2, 2, 2]),
                _info("leaf", [1, 1, 2, 2]),
            ],
        )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        assert transformed.SerializeToString() == original

    def test_repeated_tensor_across_sibling_routes_is_unchanged(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weight"], ["conv_out"]),
                onnx.helper.make_node(
                    "Split",
                    ["conv_out", "outer_sizes"],
                    ["left", "right"],
                    axis=1,
                ),
                onnx.helper.make_node("Split", ["left"], ["shared"], axis=1),
                onnx.helper.make_node("Split", ["right"], ["shared"], axis=1),
                onnx.helper.make_node("Mul", ["shared", "scale"], ["y"]),
            ],
            [_info("x", [1, 1, 2, 2])],
            [_info("y", [1, 1, 2, 2])],
            [
                _tensor("weight", np.ones((3, 1, 1, 1), dtype=np.float32)),
                _tensor("outer_sizes", np.asarray([1, 2], dtype=np.int64)),
                _tensor("scale", np.asarray(1.5, dtype=np.float32)),
            ],
            value_info=[
                _info("conv_out", [1, 3, 2, 2]),
                _info("left", [1, 1, 2, 2]),
                _info("right", [1, 2, 2, 2]),
                _info("shared", [1, 1, 2, 2]),
            ],
        )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        assert transformed.SerializeToString() == original

    @pytest.mark.parametrize("protection", ["graph_output", "shared", "captured"])
    def test_protected_nested_route_is_unchanged(self, protection: str) -> None:
        nodes = [
            onnx.helper.make_node("Conv", ["x", "weight"], ["conv_out"]),
            onnx.helper.make_node("Split", ["conv_out"], ["branch"], axis=1),
            onnx.helper.make_node("Split", ["branch"], ["leaf"], axis=1),
            onnx.helper.make_node("Mul", ["leaf", "scale"], ["y"]),
        ]
        outputs = [_info("y", [1, 1, 2, 2])]
        initializers = [
            _tensor("weight", np.ones((1, 1, 1, 1), dtype=np.float32)),
            _tensor("scale", np.asarray(1.5, dtype=np.float32)),
        ]
        if protection == "graph_output":
            outputs.append(_info("leaf", [1, 1, 2, 2]))
        elif protection == "shared":
            nodes.append(onnx.helper.make_node("Identity", ["leaf"], ["protected"]))
            outputs.append(_info("protected", [1, 1, 2, 2]))
        else:
            branch = onnx.helper.make_graph(
                [onnx.helper.make_node("Identity", ["leaf"], ["branch_output"])],
                "capturing_branch",
                [],
                [_info("branch_output", [1, 1, 2, 2])],
            )
            initializers.append(_tensor("condition", np.asarray(True, dtype=np.bool_)))
            nodes.append(
                onnx.helper.make_node(
                    "If",
                    ["condition"],
                    ["protected"],
                    then_branch=branch,
                    else_branch=branch,
                )
            )
            outputs.append(_info("protected", [1, 1, 2, 2]))
        model = _model(
            nodes,
            [_info("x", [1, 1, 2, 2])],
            outputs,
            initializers,
            value_info=[
                _info("conv_out", [1, 1, 2, 2]),
                _info("branch", [1, 1, 2, 2]),
                _info("leaf", [1, 1, 2, 2]),
            ],
        )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        assert transformed.SerializeToString() == original

    def test_nested_split_cycle_is_unchanged(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weight"], ["conv_out"]),
                onnx.helper.make_node("Split", ["conv_out"], ["route_a"], axis=1),
                onnx.helper.make_node("Split", ["route_a"], ["route_b"], axis=1),
                onnx.helper.make_node("Split", ["route_b"], ["route_a"], axis=1),
                onnx.helper.make_node("Identity", ["x"], ["y"]),
            ],
            [_info("x", [1, 1, 2, 2])],
            [_info("y", [1, 1, 2, 2])],
            [_tensor("weight", np.ones((1, 1, 1, 1), dtype=np.float32))],
            value_info=[
                _info("conv_out", [1, 1, 2, 2]),
                _info("route_a", [1, 1, 2, 2]),
                _info("route_b", [1, 1, 2, 2]),
            ],
        )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        assert transformed.SerializeToString() == original

    def test_deep_nested_split_route_is_unchanged(self) -> None:
        route_depth = 65
        nodes = [
            onnx.helper.make_node("Conv", ["x", "weight"], ["conv_out"]),
            onnx.helper.make_node("Split", ["conv_out"], ["route_0"], axis=1),
        ]
        value_info = [
            _info("conv_out", [1, 1, 2, 2]),
            _info("route_0", [1, 1, 2, 2]),
        ]
        for route_index in range(route_depth):
            nodes.append(
                onnx.helper.make_node(
                    "Split",
                    [f"route_{route_index}"],
                    [f"route_{route_index + 1}"],
                    axis=1,
                )
            )
            value_info.append(_info(f"route_{route_index + 1}", [1, 1, 2, 2]))
        nodes.append(onnx.helper.make_node("Mul", [f"route_{route_depth}", "scale"], ["y"]))
        model = _model(
            nodes,
            [_info("x", [1, 1, 2, 2])],
            [_info("y", [1, 1, 2, 2])],
            [
                _tensor("weight", np.ones((1, 1, 1, 1), dtype=np.float32)),
                _tensor("scale", np.asarray(1.5, dtype=np.float32)),
            ],
            value_info=value_info,
        )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        assert transformed.SerializeToString() == original

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

    @pytest.mark.parametrize(
        ("domain", "should_fold"),
        [("", True), ("ai.onnx", False), ("com.example", False)],
    )
    def test_only_standard_domain_constant_is_interpreted(
        self,
        domain: str,
        should_fold: bool,
    ) -> None:
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weight"], ["conv_out"]),
                onnx.helper.make_node(
                    "Constant",
                    [],
                    ["scale"],
                    value_float=1.5,
                    domain=domain,
                ),
                onnx.helper.make_node("Mul", ["conv_out", "scale"], ["y"]),
            ],
            [_info("x", [1, 1, 2, 2])],
            [_info("y", [1, 1, 2, 2])],
            [_tensor("weight", np.ones((1, 1, 1, 1), dtype=np.float32))],
            value_info=[_info("conv_out", [1, 1, 2, 2])],
        )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        if should_fold:
            assert [node.op_type for node in transformed.graph.node] == ["Conv"]
        else:
            assert transformed.SerializeToString() == original

    def test_custom_domain_conv_is_unchanged(self) -> None:
        model = _model(
            [
                onnx.helper.make_node(
                    "Conv",
                    ["x", "weight"],
                    ["conv_out"],
                    name="custom_conv",
                    domain="com.example",
                ),
                onnx.helper.make_node(
                    "Mul",
                    ["conv_out", "scale"],
                    ["y"],
                    name="affine_mul",
                ),
            ],
            [_info("x", [1, 1, 2, 2])],
            [_info("y", [1, 1, 2, 2])],
            [
                _tensor("weight", np.ones((1, 1, 1, 1), dtype=np.float32)),
                _tensor("scale", np.asarray(1.5, dtype=np.float32)),
            ],
            value_info=[_info("conv_out", [1, 1, 2, 2])],
        )

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        _assert_byte_identical(model, transformed)

    @pytest.mark.parametrize("affine_op", ["Mul", "Add"])
    @pytest.mark.parametrize("route", ["direct", "nested_split"])
    def test_custom_domain_affine_node_is_unchanged(
        self,
        affine_op: str,
        route: str,
    ) -> None:
        nodes = [onnx.helper.make_node("Conv", ["x", "weight"], ["conv_out"])]
        value_info = [_info("conv_out", [1, 1, 2, 2])]
        affine_input = "conv_out"
        if route == "nested_split":
            nodes.extend(
                [
                    onnx.helper.make_node(
                        "Split",
                        ["conv_out"],
                        ["outer_branch"],
                        axis=1,
                    ),
                    onnx.helper.make_node(
                        "Split",
                        ["outer_branch"],
                        ["affine_input"],
                        axis=1,
                    ),
                ]
            )
            value_info.extend(
                [
                    _info("outer_branch", [1, 1, 2, 2]),
                    _info("affine_input", [1, 1, 2, 2]),
                ]
            )
            affine_input = "affine_input"
        nodes.append(
            onnx.helper.make_node(
                affine_op,
                [affine_input, "affine_value"],
                ["y"],
                name="custom_affine",
                domain="com.example",
            )
        )
        model = _model(
            nodes,
            [_info("x", [1, 1, 2, 2])],
            [_info("y", [1, 1, 2, 2])],
            [
                _tensor("weight", np.ones((1, 1, 1, 1), dtype=np.float32)),
                _tensor("affine_value", np.asarray(1.5, dtype=np.float32)),
            ],
            value_info=value_info,
        )

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        _assert_byte_identical(model, transformed)

    def test_custom_domain_slice_below_nested_split_is_unchanged(self) -> None:
        shape = [1, 2, 2, 2]
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weight"], ["conv_out"]),
                onnx.helper.make_node(
                    "Split",
                    ["conv_out"],
                    ["outer_branch"],
                    axis=1,
                ),
                onnx.helper.make_node(
                    "Split",
                    ["outer_branch"],
                    ["slice_input"],
                    axis=1,
                ),
                onnx.helper.make_node(
                    "Slice",
                    ["slice_input", "starts", "ends", "axes"],
                    ["sliced"],
                    name="custom_slice",
                    domain="com.example",
                ),
                onnx.helper.make_node("Mul", ["sliced", "scale"], ["y"]),
            ],
            [_info("x", [1, 1, 2, 2])],
            [_info("y", [1, 1, 2, 2])],
            [
                _tensor("weight", np.ones((2, 1, 1, 1), dtype=np.float32)),
                _tensor("starts", np.asarray([0], dtype=np.int64)),
                _tensor("ends", np.asarray([1], dtype=np.int64)),
                _tensor("axes", np.asarray([1], dtype=np.int64)),
                _tensor("scale", np.asarray(1.5, dtype=np.float32)),
            ],
            value_info=[
                _info("conv_out", shape),
                _info("outer_branch", shape),
                _info("slice_input", shape),
                _info("sliced", [1, 1, 2, 2]),
            ],
        )

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        _assert_byte_identical(model, transformed)

    @pytest.mark.parametrize("view_op", ["Reshape", "Squeeze", "Unsqueeze"])
    def test_custom_domain_shape_view_below_nested_split_is_unchanged(
        self,
        view_op: str,
    ) -> None:
        source_shape = [1, 1, 1, 2, 2] if view_op == "Squeeze" else [1, 1, 2, 2]
        output_shape = [1, 1, 2, 2] if view_op == "Squeeze" else [1, 1, 1, 2, 2]
        weight_shape = (1, 1, 1, 1, 1) if view_op == "Squeeze" else (1, 1, 1, 1)
        view_parameter = "view_shape" if view_op == "Reshape" else "view_axes"
        view_parameter_values = (
            np.asarray(output_shape, dtype=np.int64)
            if view_op == "Reshape"
            else np.asarray([2], dtype=np.int64)
        )
        model = _model(
            [
                onnx.helper.make_node("Conv", ["x", "weight"], ["conv_out"]),
                onnx.helper.make_node(
                    "Split",
                    ["conv_out"],
                    ["outer_branch"],
                    axis=1,
                ),
                onnx.helper.make_node(
                    "Split",
                    ["outer_branch"],
                    ["view_input"],
                    axis=1,
                ),
                onnx.helper.make_node(
                    view_op,
                    ["view_input", view_parameter],
                    ["viewed"],
                    name="custom_view",
                    domain="com.example",
                ),
                onnx.helper.make_node("Mul", ["viewed", "scale"], ["y"]),
            ],
            [_info("x", source_shape)],
            [_info("y", output_shape)],
            [
                _tensor("weight", np.ones(weight_shape, dtype=np.float32)),
                _tensor(view_parameter, view_parameter_values),
                _tensor("scale", np.asarray(1.5, dtype=np.float32)),
            ],
            value_info=[
                _info("conv_out", source_shape),
                _info("outer_branch", source_shape),
                _info("view_input", source_shape),
                _info("viewed", output_shape),
            ],
        )

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(conv_channel_affine_folding=True),
        )

        _assert_byte_identical(model, transformed)


class TestExpPositiveScaleFolding:
    """Test opt-in positive scale folding into the Exp input."""

    def test_multiple_exp_chains_are_folded_from_live_graph_nodes(self) -> None:
        rng = np.random.default_rng(35)
        model = _model(
            [
                onnx.helper.make_node("Add", ["x0", "bias0"], ["biased0"]),
                onnx.helper.make_node("Exp", ["biased0"], ["exp0"]),
                onnx.helper.make_node("Mul", ["exp0", "scale0"], ["y0"]),
                onnx.helper.make_node("Add", ["x1", "bias1"], ["biased1"]),
                onnx.helper.make_node("Exp", ["biased1"], ["exp1"]),
                onnx.helper.make_node("Mul", ["exp1", "scale1"], ["y1"]),
            ],
            [_info("x0", [1, 2]), _info("x1", [1, 2])],
            [_info("y0", [1, 2]), _info("y1", [1, 2])],
            [
                _tensor("bias0", np.asarray([[0.25, -0.5]], dtype=np.float32)),
                _tensor("scale0", np.asarray([[1.25, 0.75]], dtype=np.float32)),
                _tensor("bias1", np.asarray([[1.0, -1.25]], dtype=np.float32)),
                _tensor("scale1", np.asarray([[2.0, 1.5]], dtype=np.float32)),
            ],
            value_info=[
                _info("biased0", [1, 2]),
                _info("exp0", [1, 2]),
                _info("biased1", [1, 2]),
                _info("exp1", [1, 2]),
            ],
        )
        values = {
            "x0": rng.normal(size=(1, 2)).astype(np.float32),
            "x1": rng.normal(size=(1, 2)).astype(np.float32),
        }

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        assert [node.op_type for node in transformed.graph.node] == ["Add", "Exp", "Add", "Exp"]
        assert [output.name for output in transformed.graph.output] == ["y0", "y1"]
        _assert_valid_with_inferred_shapes(transformed)
        for original, rewritten in zip(
            _run(model, values),
            _run(transformed, values),
            strict=True,
        ):
            np.testing.assert_allclose(original, rewritten, rtol=2e-6, atol=2e-6)

    def test_independent_exp_scale_chains_are_batched(self, monkeypatch) -> None:
        chain_count = 6
        nodes = []
        inputs = []
        outputs = []
        initializers = []
        value_info = []
        for index in range(chain_count):
            inputs.append(_info(f"x{index}", [1, 2]))
            outputs.append(_info(f"y{index}", [1, 2]))
            initializers.append(
                _tensor(f"scale{index}", np.asarray([1.25, 0.75], dtype=np.float32))
            )
            value_info.append(_info(f"exp{index}", [1, 2]))
            nodes.extend(
                [
                    onnx.helper.make_node("Exp", [f"x{index}"], [f"exp{index}"]),
                    onnx.helper.make_node("Mul", [f"exp{index}", f"scale{index}"], [f"y{index}"]),
                ]
            )
        model = _model(nodes, inputs, outputs, initializers, value_info=value_info)
        build_count = 0
        original_build = algebraic_pipe._GraphIndex.build

        def counted_build(model: onnx.ModelProto) -> algebraic_pipe._GraphIndex:
            nonlocal build_count
            build_count += 1
            return original_build(model)

        monkeypatch.setattr(algebraic_pipe._GraphIndex, "build", counted_build)

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        assert build_count <= 6
        assert [node.op_type for node in transformed.graph.node] == ["Add", "Exp"] * chain_count

    def test_serial_exp_mul_keeps_later_scales_to_preserve_float_boundaries(
        self,
    ) -> None:
        model = _model(
            [
                onnx.helper.make_node("Exp", ["x"], ["exp"]),
                onnx.helper.make_node("Mul", ["exp", "scale_a"], ["scaled"]),
                onnx.helper.make_node("Mul", ["scaled", "scale_b"], ["y"]),
            ],
            [_info("x", [1])],
            [_info("y", [1])],
            [
                _tensor("scale_a", np.asarray(1.0e-30, dtype=np.float32)),
                _tensor("scale_b", np.asarray(1.0e30, dtype=np.float32)),
            ],
            value_info=[_info("exp", [1]), _info("scaled", [1])],
        )
        values = {"x": np.asarray([np.log(np.float32(1.0e-20))], dtype=np.float32)}
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        assert transformed.SerializeToString() == original
        np.testing.assert_array_equal(_run(model, values), _run(transformed, values))

    def test_public_optimizer_preserves_serial_exp_mul_float_boundaries(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Exp", ["x"], ["exp"]),
                onnx.helper.make_node("Mul", ["exp", "scale_a"], ["scaled"]),
                onnx.helper.make_node("Mul", ["scaled", "scale_b"], ["y"]),
            ],
            [_info("x", [1])],
            [_info("y", [1])],
            [
                _tensor("scale_a", np.asarray(1.0e-30, dtype=np.float32)),
                _tensor("scale_b", np.asarray(1.0e30, dtype=np.float32)),
            ],
            value_info=[_info("exp", [1]), _info("scaled", [1])],
        )
        values = {"x": np.asarray([np.log(np.float32(1.0e-20))], dtype=np.float32)}

        transformed = optimize_onnx(model, exp_positive_scale_folding=True)

        assert [node.op_type for node in transformed.graph.node] == ["Exp", "Mul", "Mul"]
        _assert_valid_with_inferred_shapes(transformed)
        np.testing.assert_array_equal(_run(model, values), _run(transformed, values))

    def test_user_prefix_names_do_not_block_single_exp_scale_folding(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Add", ["x", "algebraic_exp_adjusted_user"], ["biased"]),
                onnx.helper.make_node("Exp", ["biased"], ["exp"]),
                onnx.helper.make_node("Mul", ["exp", "scale"], ["y"]),
            ],
            [_info("x", [1])],
            [_info("y", [1])],
            [
                _tensor("algebraic_exp_adjusted_user", np.asarray(0.5, dtype=np.float32)),
                _tensor("scale", np.asarray(1.5, dtype=np.float32)),
            ],
            value_info=[_info("biased", [1]), _info("exp", [1])],
        )

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        assert [node.op_type for node in transformed.graph.node] == ["Add", "Exp"]

    def test_serial_exp_mul_boundary_survives_user_renaming_between_runs(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Exp", ["x"], ["exp"]),
                onnx.helper.make_node("Mul", ["exp", "scale_a"], ["scaled"]),
                onnx.helper.make_node("Mul", ["scaled", "scale_b"], ["y"]),
            ],
            [_info("x", [1])],
            [_info("y", [1])],
            [
                _tensor("scale_a", np.asarray(1.0e-30, dtype=np.float32)),
                _tensor("scale_b", np.asarray(1.0e30, dtype=np.float32)),
            ],
            value_info=[_info("exp", [1]), _info("scaled", [1])],
        )
        values = {"x": np.asarray([np.log(np.float32(1.0e-20))], dtype=np.float32)}
        transformed = optimize_onnx(model, exp_positive_scale_folding=True)
        renamed_tensors: dict[str, str] = {}
        for index, node in enumerate(transformed.graph.node):
            node.name = f"user_node_{index}"
            for output_index, output in enumerate(node.output):
                if output.startswith("algebraic_"):
                    renamed_tensors[output] = f"user_tensor_{index}_{output_index}"
                    node.output[output_index] = renamed_tensors[output]
            for input_index, input_name in enumerate(node.input):
                if input_name.startswith("algebraic_"):
                    node.input[input_index] = renamed_tensors.setdefault(
                        input_name,
                        f"user_tensor_input_{index}_{input_index}",
                    )
        for initializer in transformed.graph.initializer:
            if initializer.name.startswith("algebraic_"):
                initializer.name = renamed_tensors.setdefault(
                    initializer.name,
                    f"user_initializer_{initializer.name}",
                )
        for value in (*transformed.graph.value_info, *transformed.graph.output):
            if value.name.startswith("algebraic_"):
                value.name = renamed_tensors.setdefault(value.name, f"user_value_{value.name}")
        renamed = optimize_onnx(transformed, exp_positive_scale_folding=True)

        assert [node.op_type for node in renamed.graph.node] == ["Exp", "Mul", "Mul"]
        np.testing.assert_array_equal(_run(model, values), _run(renamed, values))

    def test_public_optimizer_preserves_serial_exp_mul_boundary_through_view(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Exp", ["x"], ["exp"]),
                onnx.helper.make_node("Mul", ["exp", "scale_a"], ["scaled"]),
                onnx.helper.make_node("Reshape", ["scaled", "shape"], ["viewed"]),
                onnx.helper.make_node("Mul", ["viewed", "scale_b"], ["y"]),
            ],
            [_info("x", [1])],
            [_info("y", [1])],
            [
                _tensor("shape", np.asarray([1], dtype=np.int64)),
                _tensor("scale_a", np.asarray(1.0e-30, dtype=np.float32)),
                _tensor("scale_b", np.asarray(1.0e30, dtype=np.float32)),
            ],
            value_info=[_info("exp", [1]), _info("scaled", [1]), _info("viewed", [1])],
        )
        values = {"x": np.asarray([np.log(np.float32(1.0e-20))], dtype=np.float32)}

        transformed = optimize_onnx(model, exp_positive_scale_folding=True)

        assert sum(node.op_type == "Mul" for node in transformed.graph.node) == 2
        _assert_valid_with_inferred_shapes(transformed)
        np.testing.assert_array_equal(_run(model, values), _run(transformed, values))

    def test_public_optimizer_preserves_serial_exp_mul_boundary_through_flatten(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Exp", ["x"], ["exp"]),
                onnx.helper.make_node("Mul", ["exp", "scale_a"], ["scaled"]),
                onnx.helper.make_node("Flatten", ["scaled"], ["flattened"]),
                onnx.helper.make_node("Mul", ["flattened", "scale_b"], ["y"]),
            ],
            [_info("x", [1, 1])],
            [_info("flattened", [1, 1]), _info("y", [1, 1])],
            [
                _tensor("scale_a", np.asarray(0.005727309, dtype=np.float32)),
                _tensor("scale_b", np.asarray(3.3510647e28, dtype=np.float32)),
            ],
            value_info=[_info("exp", [1, 1]), _info("scaled", [1, 1])],
        )
        values = {"x": np.asarray([[-96.411636]], dtype=np.float32)}

        transformed = optimize_onnx(model, exp_positive_scale_folding=True)

        assert sum(node.op_type == "Mul" for node in transformed.graph.node) == 2
        _assert_valid_with_inferred_shapes(transformed)
        for expected, actual in zip(_run(model, values), _run(transformed, values), strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_public_optimizer_preserves_serial_exp_mul_boundary_through_identity(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Exp", ["x"], ["exp"]),
                onnx.helper.make_node("Mul", ["exp", "scale_a"], ["scaled"]),
                onnx.helper.make_node("Identity", ["scaled"], ["passed"]),
                onnx.helper.make_node("Mul", ["passed", "scale_b"], ["y"]),
            ],
            [_info("x", [1])],
            [_info("passed", [1]), _info("y", [1])],
            [
                _tensor("scale_a", np.asarray(0.005727309, dtype=np.float32)),
                _tensor("scale_b", np.asarray(3.3510647e28, dtype=np.float32)),
            ],
            value_info=[_info("exp", [1]), _info("scaled", [1])],
        )
        values = {"x": np.asarray([-96.411636], dtype=np.float32)}

        transformed = optimize_onnx(model, exp_positive_scale_folding=True)

        assert sum(node.op_type == "Mul" for node in transformed.graph.node) == 2
        _assert_valid_with_inferred_shapes(transformed)
        for expected, actual in zip(_run(model, values), _run(transformed, values), strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_public_optimizer_preserves_serial_exp_mul_boundary_when_intermediate_observed(
        self,
    ) -> None:
        model = _model(
            [
                onnx.helper.make_node("Exp", ["x"], ["exp"]),
                onnx.helper.make_node("Mul", ["exp", "scale_a"], ["scaled"]),
                onnx.helper.make_node("Mul", ["scaled", "scale_b"], ["y"]),
            ],
            [_info("x", [1])],
            [_info("scaled", [1]), _info("y", [1])],
            [
                _tensor("scale_a", np.asarray(0.005727309, dtype=np.float32)),
                _tensor("scale_b", np.asarray(3.3510647e28, dtype=np.float32)),
            ],
            value_info=[_info("exp", [1])],
        )
        values = {"x": np.asarray([-96.411636], dtype=np.float32)}

        transformed = optimize_onnx(model, exp_positive_scale_folding=True)

        assert sum(node.op_type == "Mul" for node in transformed.graph.node) == 2
        _assert_valid_with_inferred_shapes(transformed)
        for expected, actual in zip(_run(model, values), _run(transformed, values), strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_public_optimizer_preserves_serial_exp_mul_boundary_on_branch(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Exp", ["x"], ["exp"]),
                onnx.helper.make_node("Mul", ["exp", "scale_a"], ["scaled"]),
                onnx.helper.make_node("Mul", ["scaled", "scale_b"], ["y"]),
                onnx.helper.make_node("Identity", ["scaled"], ["tap"]),
            ],
            [_info("x", [1])],
            [_info("y", [1]), _info("tap", [1])],
            [
                _tensor("scale_a", np.asarray(0.005727309, dtype=np.float32)),
                _tensor("scale_b", np.asarray(3.3510647e28, dtype=np.float32)),
            ],
            value_info=[_info("exp", [1]), _info("scaled", [1])],
        )
        values = {"x": np.asarray([-96.411636], dtype=np.float32)}

        transformed = optimize_onnx(model, exp_positive_scale_folding=True)

        assert sum(node.op_type == "Mul" for node in transformed.graph.node) == 2
        _assert_valid_with_inferred_shapes(transformed)
        for expected, actual in zip(_run(model, values), _run(transformed, values), strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_public_optimizer_preserves_serial_exp_mul_after_marked_bias_view(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Add", ["x", "bias"], ["biased"]),
                onnx.helper.make_node("Reshape", ["biased", "shape"], ["viewed"]),
                onnx.helper.make_node("Exp", ["viewed"], ["exp"]),
                onnx.helper.make_node("Mul", ["exp", "scale_a"], ["scaled"]),
                onnx.helper.make_node("Mul", ["scaled", "scale_b"], ["y"]),
            ],
            [_info("x", [1])],
            [_info("y", [1])],
            [
                _tensor("bias", np.asarray(0.0, dtype=np.float32)),
                _tensor("shape", np.asarray([1], dtype=np.int64)),
                _tensor("scale_a", np.asarray(1.0e-30, dtype=np.float32)),
                _tensor("scale_b", np.asarray(1.0e30, dtype=np.float32)),
            ],
            value_info=[
                _info("biased", [1]),
                _info("viewed", [1]),
                _info("exp", [1]),
                _info("scaled", [1]),
            ],
        )
        values = {"x": np.asarray([np.log(np.float32(1.0e-20))], dtype=np.float32)}

        transformed = optimize_onnx(model, exp_positive_scale_folding=True)

        assert sum(node.op_type == "Mul" for node in transformed.graph.node) == 2
        _assert_valid_with_inferred_shapes(transformed)
        np.testing.assert_array_equal(_run(model, values), _run(transformed, values))

    def test_serial_exp_mul_stops_after_first_noncompact_scale_without_rebuilds(
        self,
        monkeypatch,
    ) -> None:
        scale_count = 6
        nodes = [
            onnx.helper.make_node("Add", ["x", "bias"], ["biased"]),
            onnx.helper.make_node("Exp", ["biased"], ["exp"]),
        ]
        initializers = [_tensor("bias", np.asarray(-0.25, dtype=np.float32))]
        value_info = [_info("biased", [1, 2]), _info("exp", [1, 2])]
        current = "exp"
        for index in range(scale_count):
            output = "y" if index == scale_count - 1 else f"scaled{index}"
            nodes.append(onnx.helper.make_node("Mul", [current, f"scale{index}"], [output]))
            initializers.append(_tensor(f"scale{index}", np.asarray(1.0e30, dtype=np.float32)))
            if output != "y":
                value_info.append(_info(output, [1, 2]))
            current = output
        model = _model(
            nodes,
            [_info("x", [1, 2])],
            [_info("y", [1, 2])],
            initializers,
            value_info=value_info,
        )
        values = {"x": np.asarray([[0.5, -1.0]], dtype=np.float32)}
        build_count = 0
        original_build = algebraic_pipe._GraphIndex.build

        def counted_build(model: onnx.ModelProto) -> algebraic_pipe._GraphIndex:
            nonlocal build_count
            build_count += 1
            return original_build(model)

        monkeypatch.setattr(algebraic_pipe._GraphIndex, "build", counted_build)

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        assert build_count <= 6
        assert [node.op_type for node in transformed.graph.node] == ["Add", "Exp", *["Mul"] * 6]
        _assert_valid_with_inferred_shapes(transformed)
        np.testing.assert_allclose(
            _run(model, values),
            _run(transformed, values),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_scalar_scale_without_bias_stays_compact(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Exp", ["x"], ["exponential"]),
                onnx.helper.make_node("Mul", ["exponential", "scale"], ["y"]),
            ],
            [_info("x", [1, 2, 3])],
            [_info("y", [1, 2, 3])],
            [_tensor("scale", np.asarray(1.5, dtype=np.float32))],
            value_info=[_info("exponential", [1, 2, 3])],
        )

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        assert [node.op_type for node in transformed.graph.node] == ["Add", "Exp"]
        log_scale_name = transformed.graph.node[0].input[1]
        log_scale = onnx.numpy_helper.to_array(
            next(value for value in transformed.graph.initializer if value.name == log_scale_name)
        )
        assert log_scale.shape == ()
        np.testing.assert_array_equal(log_scale, np.log(np.asarray(1.5, dtype=np.float32)))

    def test_post_view_scale_with_different_flattened_pattern_is_unchanged(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Exp", ["x"], ["exponential"]),
                onnx.helper.make_node("Reshape", ["exponential", "output_shape"], ["reshaped"]),
                onnx.helper.make_node("Mul", ["reshaped", "scale"], ["y"]),
            ],
            [_info("x", [2, 2, 3])],
            [_info("y", [3, 2, 2])],
            [
                _tensor("output_shape", np.asarray([3, 2, 2], dtype=np.int64)),
                _tensor("scale", np.asarray([[[1.0], [2.0]]], dtype=np.float32)),
            ],
            value_info=[
                _info("exponential", [2, 2, 3]),
                _info("reshaped", [3, 2, 2]),
            ],
        )

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        _assert_byte_identical(model, transformed)

    def test_orthogonal_bias_and_scale_broadcast_uses_compact_log_scale_add(self) -> None:
        rng = np.random.default_rng(36)
        model = _model(
            [
                onnx.helper.make_node("Add", ["x", "bias"], ["biased"], name="bias_add"),
                onnx.helper.make_node("Exp", ["biased"], ["exponential"], name="exp"),
                onnx.helper.make_node("Mul", ["exponential", "scale"], ["y"], name="scale_mul"),
            ],
            [_info("x", [2, 2, 3])],
            [_info("y", [2, 2, 3])],
            [
                _tensor(
                    "bias",
                    rng.normal(size=(2, 1, 3)).astype(np.float32),
                ),
                _tensor("scale", np.asarray([[[1.25], [0.75]]], dtype=np.float32)),
            ],
            value_info=[
                _info("biased", [2, 2, 3]),
                _info("exponential", [2, 2, 3]),
            ],
        )
        values = {"x": rng.normal(size=(2, 2, 3)).astype(np.float32)}

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        assert [node.op_type for node in transformed.graph.node] == ["Add", "Add", "Exp"]
        assert transformed.graph.node[0].name == "bias_add"
        assert transformed.graph.node[0].input[1] == "bias"
        assert transformed.graph.node[-1].output[0] == "y"
        log_scale_name = transformed.graph.node[1].input[1]
        log_scale = onnx.numpy_helper.to_array(
            next(value for value in transformed.graph.initializer if value.name == log_scale_name)
        )
        assert log_scale.shape == (1, 2, 1)
        assert not any(tuple(value.dims) == (2, 2, 3) for value in transformed.graph.initializer)
        _assert_valid_with_inferred_shapes(transformed)
        np.testing.assert_allclose(
            _run(model, values),
            _run(transformed, values),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_exp_scale_flag_discloses_relaxed_float_boundary_semantics(self) -> None:
        description = get_all_capabilities()["exp-positive-scale-folding"].description
        assert "relaxed floating-point overflow semantics" in description

    def test_float32_boundary_behavior_is_relaxed_when_enabled(self) -> None:
        x = np.asarray([89.0], dtype=np.float32)
        model = _model(
            [
                onnx.helper.make_node("Exp", ["x"], ["exponential"]),
                onnx.helper.make_node("Mul", ["exponential", "scale"], ["y"]),
            ],
            [_info("x", [1])],
            [_info("y", [1])],
            [_tensor("scale", np.asarray(1.0e-10, dtype=np.float32))],
            value_info=[_info("exponential", [1])],
        )

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        assert [node.op_type for node in transformed.graph.node] == ["Add", "Exp"]
        assert np.isinf(_run(model, {"x": x})[0][0])
        assert np.isfinite(_run(transformed, {"x": x})[0][0])

    @pytest.mark.parametrize("opset_version", [None, 6])
    def test_legacy_or_missing_standard_opset_is_unchanged(
        self,
        opset_version: int | None,
    ) -> None:
        model = _model(
            [
                onnx.helper.make_node("Exp", ["x"], ["exponential"]),
                onnx.helper.make_node(
                    "Mul",
                    ["exponential", "scale"],
                    ["y"],
                    broadcast=1,
                ),
            ],
            [_info("x", [1, 2])],
            [_info("y", [1, 2])],
            [_tensor("scale", np.asarray([1.25, 0.75], dtype=np.float32))],
            value_info=[_info("exponential", [1, 2])],
        )
        del model.opset_import[:]
        if opset_version is not None:
            model.opset_import.append(onnx.helper.make_opsetid("", opset_version))
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        assert transformed.SerializeToString() == original

    @pytest.mark.parametrize(
        "opset_imports",
        [
            [onnx.helper.make_opsetid("ai.onnx", 17), onnx.helper.make_opsetid("", 6)],
            [onnx.helper.make_opsetid("", 17), onnx.helper.make_opsetid("", 6)],
        ],
    )
    def test_ambiguous_standard_opset_is_unchanged(
        self,
        opset_imports: list[onnx.OperatorSetIdProto],
    ) -> None:
        model = _model(
            [
                onnx.helper.make_node("Exp", ["x"], ["exponential"]),
                onnx.helper.make_node(
                    "Mul",
                    ["exponential", "scale"],
                    ["y"],
                    broadcast=1,
                ),
            ],
            [_info("x", [1, 2])],
            [_info("y", [1, 2])],
            [_tensor("scale", np.asarray([1.25, 0.75], dtype=np.float32))],
            value_info=[_info("exponential", [1, 2])],
        )
        del model.opset_import[:]
        model.opset_import.extend(opset_imports)
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        assert transformed.SerializeToString() == original

    def test_huge_shape_product_overflow_is_unchanged(self) -> None:
        huge_dimension = 5_270_498_306_774_157_605
        model = _model(
            [
                onnx.helper.make_node("Exp", ["x"], ["exponential"]),
                onnx.helper.make_node("Reshape", ["exponential", "output_shape"], ["reshaped"]),
                onnx.helper.make_node(
                    "Mul",
                    ["reshaped", "scale"],
                    ["y"],
                ),
            ],
            [_info("x", [7, huge_dimension])],
            [_info("y", [3])],
            [
                _tensor("output_shape", np.asarray([3], dtype=np.int64)),
                _tensor("scale", np.asarray([1.25, 0.75, 1.5], dtype=np.float32)),
            ],
            value_info=[
                _info("exponential", [7, huge_dimension]),
                _info("reshaped", [3]),
            ],
        )

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        _assert_byte_identical(model, transformed)

    def test_simple_numeric_example_matches_log_domain_identity(self) -> None:
        x = np.asarray([[0.0, 2.0]], dtype=np.float32)
        bias = np.asarray([[1.0, -1.0]], dtype=np.float32)
        log_scale = np.asarray([[2.0, 0.5]], dtype=np.float32)
        scale = np.exp(log_scale).astype(np.float32)
        expected_folded_bias = np.asarray([[3.0, -0.5]], dtype=np.float32)
        expected_output = np.exp(x + expected_folded_bias).astype(np.float32)
        model = _model(
            [
                onnx.helper.make_node("Add", ["x", "bias"], ["biased"]),
                onnx.helper.make_node("Exp", ["biased"], ["exponential"]),
                onnx.helper.make_node("Mul", ["exponential", "scale"], ["y"]),
            ],
            [_info("x", [1, 2])],
            [_info("y", [1, 2])],
            [_tensor("bias", bias), _tensor("scale", scale)],
            value_info=[_info("biased", [1, 2]), _info("exponential", [1, 2])],
        )

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        assert [node.op_type for node in transformed.graph.node] == ["Add", "Exp"]
        combined_name = transformed.graph.node[0].input[1]
        combined = onnx.numpy_helper.to_array(
            next(value for value in transformed.graph.initializer if value.name == combined_name)
        )
        np.testing.assert_allclose(combined, expected_folded_bias, rtol=0, atol=2e-7)
        _assert_valid_with_inferred_shapes(transformed)
        np.testing.assert_allclose(_run(model, {"x": x}), [expected_output], rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(
            _run(transformed, {"x": x}),
            [expected_output],
            rtol=2e-6,
            atol=2e-6,
        )

    @pytest.fixture
    def exp_scale_model(self) -> onnx.ModelProto:
        tensor_shape = [1, 2, 1, 2, 2]
        return _model(
            [
                onnx.helper.make_node("Add", ["x", "bias"], ["biased"]),
                onnx.helper.make_node("Reshape", ["biased", "flat_shape"], ["flat"]),
                onnx.helper.make_node("Exp", ["flat"], ["exponential"]),
                onnx.helper.make_node(
                    "Reshape",
                    ["exponential", "tensor_shape"],
                    ["restored"],
                ),
                onnx.helper.make_node("Mul", ["restored", "scale"], ["y"]),
            ],
            [_info("x", tensor_shape)],
            [_info("y", tensor_shape)],
            [
                _tensor("bias", np.asarray(-2.0, dtype=np.float32)),
                _tensor("flat_shape", np.asarray([1, 8], dtype=np.int64)),
                _tensor("tensor_shape", np.asarray(tensor_shape, dtype=np.int64)),
                _tensor(
                    "scale",
                    np.asarray([[[[[1.25, 1.5], [2.0, 0.75]]]]], dtype=np.float32),
                ),
            ],
            value_info=[
                _info("biased", tensor_shape),
                _info("flat", [1, 8]),
                _info("exponential", [1, 8]),
                _info("restored", tensor_shape),
            ],
        )

    def test_broadcast_scale_folds_through_round_trip_reshapes(
        self,
        exp_scale_model: onnx.ModelProto,
    ) -> None:
        rng = np.random.default_rng(30)
        model = exp_scale_model
        tensor_shape = [1, 2, 1, 2, 2]
        values = {"x": rng.normal(size=tensor_shape).astype(np.float32)}

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        assert [node.op_type for node in transformed.graph.node] == [
            "Add",
            "Reshape",
            "Exp",
            "Reshape",
        ]
        assert transformed.graph.node[-1].output[0] == "y"
        combined_name = transformed.graph.node[0].input[1]
        combined = onnx.numpy_helper.to_array(
            next(value for value in transformed.graph.initializer if value.name == combined_name)
        )
        expected = np.asarray(-2.0 + np.log(onnx.numpy_helper.to_array(model.graph.initializer[3])))
        assert combined.shape == (1, 1, 1, 2, 2)
        np.testing.assert_array_equal(combined, expected)
        _assert_valid_with_inferred_shapes(transformed)
        np.testing.assert_allclose(
            _run(model, values),
            _run(transformed, values),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_squeeze_and_unsqueeze_views_fold(self) -> None:
        rng = np.random.default_rng(31)
        model = _model(
            [
                onnx.helper.make_node("Add", ["x", "bias"], ["biased"]),
                onnx.helper.make_node("Unsqueeze", ["biased", "axes"], ["expanded"]),
                onnx.helper.make_node("Exp", ["expanded"], ["exponential"]),
                onnx.helper.make_node("Squeeze", ["exponential", "axes"], ["restored"]),
                onnx.helper.make_node("Mul", ["restored", "scale"], ["y"]),
            ],
            [_info("x", [1, 2, 2])],
            [_info("y", [1, 2, 2])],
            [
                _tensor("bias", np.asarray(-1.0, dtype=np.float32)),
                _tensor("axes", np.asarray([1], dtype=np.int64)),
                _tensor("scale", np.asarray([[[1.0, 1.25], [1.5, 2.0]]], dtype=np.float32)),
            ],
            value_info=[
                _info("biased", [1, 2, 2]),
                _info("expanded", [1, 1, 2, 2]),
                _info("exponential", [1, 1, 2, 2]),
                _info("restored", [1, 2, 2]),
            ],
        )
        values = {"x": rng.normal(size=(1, 2, 2)).astype(np.float32)}

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        assert not any(node.op_type == "Mul" for node in transformed.graph.node)
        _assert_valid_with_inferred_shapes(transformed)
        np.testing.assert_allclose(
            _run(model, values),
            _run(transformed, values),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_scale_without_existing_bias_becomes_pre_exp_add(self) -> None:
        rng = np.random.default_rng(32)
        scale = np.asarray([[[[1.0, 1.25], [1.5, 2.0]]]], dtype=np.float32)
        model = _model(
            [
                onnx.helper.make_node("Exp", ["x"], ["exponential"]),
                onnx.helper.make_node(
                    "Reshape",
                    ["exponential", "output_shape"],
                    ["restored"],
                ),
                onnx.helper.make_node("Mul", ["restored", "scale"], ["y"]),
            ],
            [_info("x", [1, 4])],
            [_info("y", [1, 1, 2, 2])],
            [
                _tensor("output_shape", np.asarray([1, 1, 2, 2], dtype=np.int64)),
                _tensor("scale", scale),
            ],
            value_info=[
                _info("exponential", [1, 4]),
                _info("restored", [1, 1, 2, 2]),
            ],
        )
        values = {"x": rng.normal(size=(1, 4)).astype(np.float32)}

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        assert [node.op_type for node in transformed.graph.node] == ["Add", "Exp", "Reshape"]
        assert transformed.graph.node[-1].output[0] == "y"
        log_scale_name = transformed.graph.node[0].input[1]
        log_scale = onnx.numpy_helper.to_array(
            next(value for value in transformed.graph.initializer if value.name == log_scale_name)
        )
        np.testing.assert_array_equal(log_scale, np.log(scale).reshape(1, 4))
        _assert_valid_with_inferred_shapes(transformed)
        np.testing.assert_allclose(
            _run(model, values),
            _run(transformed, values),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_runtime_bias_keeps_original_add_and_replaces_mul(self) -> None:
        rng = np.random.default_rng(34)
        model = _model(
            [
                onnx.helper.make_node(
                    "Add",
                    ["x", "runtime_bias"],
                    ["biased"],
                    name="runtime_bias_add",
                ),
                onnx.helper.make_node("Exp", ["biased"], ["exponential"]),
                onnx.helper.make_node("Mul", ["exponential", "scale"], ["y"]),
            ],
            [_info("x", [1, 4]), _info("runtime_bias", [1, 4])],
            [_info("y", [1, 4])],
            [_tensor("scale", np.asarray([1.0, 1.25, 1.5, 2.0], dtype=np.float32))],
            value_info=[_info("biased", [1, 4]), _info("exponential", [1, 4])],
        )
        values = {
            "x": rng.normal(size=(1, 4)).astype(np.float32),
            "runtime_bias": rng.normal(size=(1, 4)).astype(np.float32),
        }

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        assert [node.op_type for node in transformed.graph.node] == ["Add", "Add", "Exp"]
        assert transformed.graph.node[0].name == "runtime_bias_add"
        assert list(transformed.graph.node[0].input) == ["x", "runtime_bias"]
        _assert_valid_with_inferred_shapes(transformed)
        np.testing.assert_allclose(
            _run(model, values),
            _run(transformed, values),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_constant_attribute_operands_preserve_float32_dtype(self) -> None:
        rng = np.random.default_rng(33)
        model = _model(
            [
                onnx.helper.make_node("Constant", [], ["bias"], value_float=-1.0),
                onnx.helper.make_node("Constant", [], ["scale"], value_float=1.5),
                onnx.helper.make_node("Add", ["x", "bias"], ["biased"]),
                onnx.helper.make_node("Exp", ["biased"], ["exponential"]),
                onnx.helper.make_node("Mul", ["exponential", "scale"], ["y"]),
            ],
            [_info("x", [1, 4])],
            [_info("y", [1, 4])],
            [],
            value_info=[_info("biased", [1, 4]), _info("exponential", [1, 4])],
        )
        values = {"x": rng.normal(size=(1, 4)).astype(np.float32)}

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        assert [node.op_type for node in transformed.graph.node] == ["Add", "Exp"]
        combined_name = transformed.graph.node[0].input[1]
        combined = onnx.numpy_helper.to_array(
            next(value for value in transformed.graph.initializer if value.name == combined_name)
        )
        assert combined.dtype == np.float32
        _assert_valid_with_inferred_shapes(transformed)
        np.testing.assert_allclose(
            _run(model, values),
            _run(transformed, values),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_direct_float64_chain_preserves_initializer_precision(self) -> None:
        shape = [1, 2]

        def double_info(name: str) -> onnx.ValueInfoProto:
            return onnx.helper.make_tensor_value_info(
                name,
                onnx.TensorProto.DOUBLE,
                shape,
            )

        bias = np.asarray(1.0 + 2**-30, dtype=np.float64)
        scale = np.asarray([[1.0 + 2**-29, 1.0 + 2**-28]], dtype=np.float64)
        model = _model(
            [
                onnx.helper.make_node("Add", ["x", "bias"], ["biased"]),
                onnx.helper.make_node("Exp", ["biased"], ["exponential"]),
                onnx.helper.make_node("Mul", ["exponential", "scale"], ["y"]),
            ],
            [double_info("x")],
            [double_info("y")],
            [_tensor("bias", bias), _tensor("scale", scale)],
            value_info=[double_info("biased"), double_info("exponential")],
        )

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        combined_name = transformed.graph.node[0].input[1]
        combined = onnx.numpy_helper.to_array(
            next(value for value in transformed.graph.initializer if value.name == combined_name)
        )
        assert combined.dtype == np.float64
        np.testing.assert_array_equal(combined, bias + np.log(scale))
        _assert_valid_with_inferred_shapes(transformed)

    def test_invalid_scale_broadcast_is_unchanged(
        self,
        exp_scale_model: onnx.ModelProto,
    ) -> None:
        scale = next(value for value in exp_scale_model.graph.initializer if value.name == "scale")
        scale.CopyFrom(_tensor("scale", np.ones((1, 3, 1, 2, 2), dtype=np.float32)))

        transformed = AlgebraicRewritePipe().process(
            exp_scale_model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        _assert_byte_identical(exp_scale_model, transformed)

    @pytest.mark.parametrize("scale_value", [0.0, -1.0, np.nan, np.inf])
    def test_nonpositive_or_nonfinite_scale_is_unchanged(
        self,
        exp_scale_model: onnx.ModelProto,
        scale_value: float,
    ) -> None:
        scale = next(value for value in exp_scale_model.graph.initializer if value.name == "scale")
        scale.CopyFrom(_tensor("scale", np.full((1, 1, 1, 2, 2), scale_value, np.float32)))

        transformed = AlgebraicRewritePipe().process(
            exp_scale_model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        _assert_byte_identical(exp_scale_model, transformed)

    @pytest.mark.parametrize(
        "constant_name",
        ["flat_shape", "tensor_shape", "scale"],
    )
    def test_overridable_constants_are_unchanged(
        self,
        exp_scale_model: onnx.ModelProto,
        constant_name: str,
    ) -> None:
        initializer = next(
            value for value in exp_scale_model.graph.initializer if value.name == constant_name
        )
        exp_scale_model.graph.input.append(
            onnx.helper.make_tensor_value_info(
                constant_name,
                initializer.data_type,
                list(initializer.dims),
            )
        )

        transformed = AlgebraicRewritePipe().process(
            exp_scale_model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        _assert_byte_identical(exp_scale_model, transformed)

    def test_unloaded_external_scale_is_unchanged(
        self,
        exp_scale_model: onnx.ModelProto,
    ) -> None:
        scale = next(value for value in exp_scale_model.graph.initializer if value.name == "scale")
        scale.ClearField("raw_data")
        scale.data_location = onnx.TensorProto.EXTERNAL
        location = scale.external_data.add()
        location.key = "location"
        location.value = "missing-scale.bin"

        transformed = AlgebraicRewritePipe().process(
            exp_scale_model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        _assert_byte_identical(exp_scale_model, transformed)

    def test_unloaded_external_constant_value_is_unchanged(self) -> None:
        external_scale = _tensor("external_scale_payload", np.asarray(1.5, dtype=np.float32))
        external_scale.ClearField("raw_data")
        external_scale.data_location = onnx.TensorProto.EXTERNAL
        location = external_scale.external_data.add()
        location.key = "location"
        location.value = "missing-constant-scale.bin"
        model = _model(
            [
                onnx.helper.make_node("Exp", ["x"], ["exponential"]),
                onnx.helper.make_node("Constant", [], ["scale"], value=external_scale),
                onnx.helper.make_node("Mul", ["exponential", "scale"], ["y"]),
            ],
            [_info("x", [1])],
            [_info("y", [1])],
            [],
            value_info=[_info("exponential", [1])],
        )

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        _assert_byte_identical(model, transformed)

    @pytest.mark.parametrize("duplicate_name", ["biased", "exponential", "y"])
    def test_duplicate_tensor_definition_is_unchanged(
        self,
        exp_scale_model: onnx.ModelProto,
        duplicate_name: str,
    ) -> None:
        exp_scale_model.graph.node.append(
            onnx.helper.make_node("Identity", ["x"], [duplicate_name])
        )

        transformed = AlgebraicRewritePipe().process(
            exp_scale_model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        _assert_byte_identical(exp_scale_model, transformed)

    @pytest.mark.parametrize("collision", ["initializer", "graph_input", "initializer_copy"])
    def test_cross_kind_definition_collision_is_unchanged(
        self,
        exp_scale_model: onnx.ModelProto,
        collision: str,
    ) -> None:
        if collision == "initializer":
            exp_scale_model.graph.node.append(onnx.helper.make_node("Identity", ["x"], ["scale"]))
        elif collision == "graph_input":
            exp_scale_model.graph.node.append(onnx.helper.make_node("Identity", ["x"], ["x"]))
        else:
            duplicate = onnx.TensorProto()
            duplicate.CopyFrom(
                next(value for value in exp_scale_model.graph.initializer if value.name == "scale")
            )
            exp_scale_model.graph.initializer.append(duplicate)

        transformed = AlgebraicRewritePipe().process(
            exp_scale_model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        _assert_byte_identical(exp_scale_model, transformed)

    def test_malformed_post_exp_cycle_is_unchanged(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Add", ["x", "bias"], ["biased"]),
                onnx.helper.make_node("Exp", ["biased"], ["route_0"]),
                onnx.helper.make_node("Reshape", ["route_0", "shape"], ["route_1"]),
                onnx.helper.make_node("Reshape", ["route_1", "shape"], ["route_0"]),
                onnx.helper.make_node("Mul", ["route_1", "scale"], ["y"]),
            ],
            [_info("x", [1, 4])],
            [_info("y", [1, 4])],
            [
                _tensor("bias", np.asarray(-1.0, dtype=np.float32)),
                _tensor("shape", np.asarray([1, 4], dtype=np.int64)),
                _tensor("scale", np.asarray(1.5, dtype=np.float32)),
            ],
            value_info=[
                _info("biased", [1, 4]),
                _info("route_0", [1, 4]),
                _info("route_1", [1, 4]),
            ],
        )

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        _assert_byte_identical(model, transformed)

    def test_malformed_exp_mul_cycle_is_unchanged(self) -> None:
        model = _model(
            [
                onnx.helper.make_node("Exp", ["y"], ["exponential"]),
                onnx.helper.make_node("Mul", ["exponential", "scale"], ["y"]),
            ],
            [],
            [_info("y", [1, 4])],
            [_tensor("scale", np.asarray(1.5, dtype=np.float32))],
            value_info=[_info("exponential", [1, 4])],
        )

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        _assert_byte_identical(model, transformed)

    @pytest.mark.parametrize("domain", ["ai.onnx", "com.example"])
    @pytest.mark.parametrize("node_index", [2, 3, 4])
    def test_non_empty_domain_interpreted_nodes_are_unchanged(
        self,
        exp_scale_model: onnx.ModelProto,
        node_index: int,
        domain: str,
    ) -> None:
        exp_scale_model.graph.node[node_index].domain = domain

        transformed = AlgebraicRewritePipe().process(
            exp_scale_model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        _assert_byte_identical(exp_scale_model, transformed)

    @pytest.mark.parametrize("protection", ["graph_output", "shared", "captured"])
    def test_observed_intermediate_is_unchanged(
        self,
        exp_scale_model: onnx.ModelProto,
        protection: str,
    ) -> None:
        if protection == "graph_output":
            exp_scale_model.graph.output.append(_info("restored", [1, 2, 1, 2, 2]))
        elif protection == "shared":
            exp_scale_model.graph.node.append(
                onnx.helper.make_node("Identity", ["restored"], ["observed"])
            )
            exp_scale_model.graph.output.append(_info("observed", [1, 2, 1, 2, 2]))
        else:
            branch = onnx.helper.make_graph(
                [onnx.helper.make_node("Identity", ["restored"], ["branch_output"])],
                "capturing_branch",
                [],
                [_info("branch_output", [1, 2, 1, 2, 2])],
            )
            exp_scale_model.graph.initializer.append(
                _tensor("condition", np.asarray(True, dtype=np.bool_))
            )
            exp_scale_model.graph.node.append(
                onnx.helper.make_node(
                    "If",
                    ["condition"],
                    ["observed"],
                    then_branch=branch,
                    else_branch=branch,
                )
            )
            exp_scale_model.graph.output.append(_info("observed", [1, 2, 1, 2, 2]))

        transformed = AlgebraicRewritePipe().process(
            exp_scale_model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        _assert_byte_identical(exp_scale_model, transformed)

    def test_shape_domain_mismatch_is_unchanged(
        self,
        exp_scale_model: onnx.ModelProto,
    ) -> None:
        restored = next(
            value for value in exp_scale_model.graph.value_info if value.name == "restored"
        )
        restored.type.tensor_type.shape.dim[1].ClearField("dim_value")
        restored.type.tensor_type.shape.dim[1].dim_param = "channels"

        transformed = AlgebraicRewritePipe().process(
            exp_scale_model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        _assert_byte_identical(exp_scale_model, transformed)

    def test_deep_view_route_is_unchanged(self) -> None:
        route_depth = 65
        nodes = [
            onnx.helper.make_node("Add", ["x", "bias"], ["biased"]),
            onnx.helper.make_node("Exp", ["biased"], ["route_0"]),
        ]
        value_info = [_info("biased", [1, 4]), _info("route_0", [1, 4])]
        for route_index in range(route_depth):
            nodes.append(
                onnx.helper.make_node(
                    "Reshape",
                    [f"route_{route_index}", "shape"],
                    [f"route_{route_index + 1}"],
                )
            )
            value_info.append(_info(f"route_{route_index + 1}", [1, 4]))
        nodes.append(onnx.helper.make_node("Mul", [f"route_{route_depth}", "scale"], ["y"]))
        model = _model(
            nodes,
            [_info("x", [1, 4])],
            [_info("y", [1, 4])],
            [
                _tensor("bias", np.asarray(-1.0, dtype=np.float32)),
                _tensor("shape", np.asarray([1, 4], dtype=np.int64)),
                _tensor("scale", np.asarray(1.5, dtype=np.float32)),
            ],
            value_info=value_info,
        )

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(exp_positive_scale_folding=True),
        )

        _assert_byte_identical(model, transformed)

    def test_public_optimize_path_is_idempotent(
        self,
        exp_scale_model: onnx.ModelProto,
    ) -> None:
        transformed = optimize_onnx(exp_scale_model, exp_positive_scale_folding=True)
        second = optimize_onnx(transformed, exp_positive_scale_folding=True)

        assert transformed.SerializeToString() == second.SerializeToString()
        assert not any(node.op_type == "Mul" for node in transformed.graph.node)
        _assert_valid_with_inferred_shapes(second)
