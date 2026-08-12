# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Generated-model tests for exact and CUDA contrib graph surgeries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from winml.modelkit.commands.optimize import optimize
from winml.modelkit.optim import Optimizer
from winml.modelkit.optim.pipes import (
    SURGERY_CAPABILITIES,
    SurgeryPipe,
    SurgeryPipeConfig,
)


if TYPE_CHECKING:
    from collections.abc import Callable


_NEW_CAPABILITIES = (
    "simplify-l2-normalization",
    "gathernd-to-resize",
    "silu-to-quick-gelu",
    "scaled-matmul-to-fused-matmul",
)


def _run_ort(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    return ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(None, feeds)


def _attributes(node: onnx.NodeProto) -> dict[str, object]:
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _make_l2_normalization_model(
    *,
    clip_min: float = 0.0,
    axis: int = 1,
    expand_batch: int = 1,
) -> onnx.ModelProto:
    input_shape = [1, 3, 2, 2]
    output_shape = [expand_batch, 3, 2, 2]
    initializers = [
        numpy_helper.from_array(np.asarray([axis], dtype=np.int64), "axes"),
        numpy_helper.from_array(np.asarray(clip_min, dtype=np.float32), "clip_min"),
        numpy_helper.from_array(np.asarray(output_shape, dtype=np.int64), "expand_shape"),
    ]
    nodes = [
        helper.make_node(
            "ReduceL2",
            ["x", "axes"],
            ["norm"],
            name="reduce_l2",
            keepdims=1,
        ),
        helper.make_node("Clip", ["norm", "clip_min"], ["clipped"], name="clip"),
        helper.make_node(
            "Expand",
            ["clipped", "expand_shape"],
            ["expanded"],
            name="expand",
        ),
        helper.make_node("Div", ["x", "expanded"], ["y"], name="div"),
    ]
    graph = helper.make_graph(
        nodes,
        "l2_normalization",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    return onnx.shape_inference.infer_shapes(model)


def _make_silu_model(*, scaled_sigmoid: bool = False) -> onnx.ModelProto:
    nodes = []
    sigmoid_input = "x"
    initializers = []
    if scaled_sigmoid:
        initializers.append(numpy_helper.from_array(np.asarray(2.0, dtype=np.float32), "two"))
        nodes.append(helper.make_node("Mul", ["x", "two"], ["scaled"], name="scale"))
        sigmoid_input = "scaled"
    nodes.extend(
        [
            helper.make_node("Sigmoid", [sigmoid_input], ["sigmoid"], name="sigmoid"),
            helper.make_node("Mul", ["x", "sigmoid"], ["y"], name="silu"),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "silu",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _make_scaled_matmul_model(
    *,
    scalar_scale: bool = True,
    element_type: int = TensorProto.FLOAT,
    scale_element_type: int | None = None,
) -> onnx.ModelProto:
    scale_values = (
        np.asarray(0.125, dtype=np.float32)
        if scalar_scale
        else np.asarray([0.125, 0.25], dtype=np.float32)
    )
    scale_element_type = scale_element_type or element_type
    if scale_element_type == TensorProto.BFLOAT16:
        scale = onnx.TensorProto(name="scale", data_type=TensorProto.BFLOAT16)
        scale.dims.extend(scale_values.shape)
        bits = np.right_shift(scale_values.view(np.uint32), 16).astype(np.uint16)
        scale.raw_data = bits.tobytes()
    else:
        dtype = {
            TensorProto.FLOAT16: np.float16,
            TensorProto.FLOAT: np.float32,
            TensorProto.DOUBLE: np.float64,
        }[scale_element_type]
        scale = numpy_helper.from_array(scale_values.astype(dtype), "scale")
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["a", "b"], ["product"], name="matmul"),
            helper.make_node("Mul", ["product", "scale"], ["y"], name="scale_product"),
        ],
        "scaled_matmul",
        [
            helper.make_tensor_value_info("a", element_type, [2, 4]),
            helper.make_tensor_value_info("b", element_type, [4, 2]),
        ],
        [helper.make_tensor_value_info("y", element_type, [2, 2])],
        initializer=[scale],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _append_index_vector(
    nodes: list[onnx.NodeProto],
    dimension: str,
    prefix: str,
    *,
    half_name: str,
) -> str:
    doubled = f"{prefix}_doubled"
    dimension_float = f"{prefix}_dimension_float"
    doubled_float = f"{prefix}_doubled_float"
    denominator = f"{prefix}_denominator"
    ratio = f"{prefix}_ratio"
    positions = f"{prefix}_positions"
    offset_positions = f"{prefix}_offset_positions"
    scaled_positions = f"{prefix}_scaled_positions"
    indices = f"{prefix}_indices"
    nodes.extend(
        [
            helper.make_node("Mul", [dimension, "int_two"], [doubled], name=doubled),
            helper.make_node(
                "Cast",
                [dimension],
                [dimension_float],
                name=dimension_float,
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Mul",
                [dimension_float, "float_two"],
                [denominator],
                name=denominator,
            ),
            helper.make_node("Div", [dimension_float, denominator], [ratio], name=ratio),
            helper.make_node(
                "Cast",
                [doubled],
                [doubled_float],
                name=doubled_float,
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Range",
                ["float_zero", doubled_float, "float_one"],
                [positions],
                name=positions,
            ),
            helper.make_node(
                "Add",
                [positions, half_name],
                [offset_positions],
                name=offset_positions,
            ),
            helper.make_node(
                "Mul",
                [offset_positions, ratio],
                [scaled_positions],
                name=scaled_positions,
            ),
            helper.make_node(
                "Cast",
                [scaled_positions],
                [indices],
                name=indices,
                to=TensorProto.INT64,
            ),
        ]
    )
    return indices


def _make_gathernd_upsampling_model(*, half: float = 0.5) -> onnx.ModelProto:
    input_shape = [1, 2, 2, 3]
    output_shape = [1, 2, 4, 6]
    initializers = [
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "batch_dim"),
        numpy_helper.from_array(np.asarray([2], dtype=np.int64), "channel_dim"),
        numpy_helper.from_array(np.asarray([-1], dtype=np.int64), "negative_one"),
        numpy_helper.from_array(np.asarray(2, dtype=np.int64), "int_two"),
        numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), "float_zero"),
        numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), "float_one"),
        numpy_helper.from_array(np.asarray(2.0, dtype=np.float32), "float_two"),
        numpy_helper.from_array(np.asarray(half, dtype=np.float32), "half"),
    ]
    nodes = [
        helper.make_node("Shape", ["x"], ["height_vector"], name="height_shape", start=2, end=3),
        helper.make_node("Squeeze", ["height_vector"], ["height"], name="height"),
        helper.make_node("Shape", ["x"], ["width_vector"], name="width_shape", start=3, end=4),
        helper.make_node("Squeeze", ["width_vector"], ["width"], name="width"),
        helper.make_node(
            "Concat",
            ["batch_dim", "channel_dim", "height_vector", "width_vector"],
            ["source_shape"],
            name="source_shape",
            axis=0,
        ),
        helper.make_node(
            "Reshape",
            ["x", "source_shape"],
            ["source"],
            name="source",
            allowzero=1,
        ),
    ]
    height_indices = _append_index_vector(nodes, "height", "height", half_name="half")
    width_indices = _append_index_vector(nodes, "width", "width", half_name="half")
    nodes.extend(
        [
            helper.make_node(
                "Unsqueeze",
                [height_indices, "negative_one"],
                ["height_column"],
                name="height_column",
            ),
            helper.make_node(
                "Max",
                ["height_column", width_indices],
                ["grid_extent"],
                name="grid_extent",
            ),
            helper.make_node(
                "Shape",
                ["grid_extent"],
                ["grid_shape"],
                name="grid_shape",
                start=0,
            ),
            helper.make_node(
                "Expand",
                ["height_column", "grid_shape"],
                ["height_grid"],
                name="height_grid",
            ),
            helper.make_node(
                "Unsqueeze",
                ["height_grid", "negative_one"],
                ["height_component"],
                name="height_component",
            ),
            helper.make_node(
                "Expand",
                [width_indices, "grid_shape"],
                ["width_grid"],
                name="width_grid",
            ),
            helper.make_node(
                "Unsqueeze",
                ["width_grid", "negative_one"],
                ["width_component"],
                name="width_component",
            ),
            helper.make_node(
                "Concat",
                ["height_component", "width_component"],
                ["indices"],
                name="indices",
                axis=-1,
            ),
            helper.make_node(
                "Cast",
                ["source"],
                ["source_float"],
                name="source_float",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Transpose",
                ["source_float"],
                ["spatial_first"],
                name="spatial_first",
                perm=[2, 3, 0, 1],
            ),
            helper.make_node(
                "GatherND",
                ["spatial_first", "indices"],
                ["gathered"],
                name="gathered",
                batch_dims=0,
            ),
            helper.make_node(
                "Transpose",
                ["gathered"],
                ["channels_first"],
                name="channels_first",
                perm=[2, 3, 0, 1],
            ),
            helper.make_node(
                "Cast",
                ["channels_first"],
                ["y"],
                name="output_cast",
                to=TensorProto.FLOAT16,
            ),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "gathernd_upsampling",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, input_shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, output_shape)],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    return onnx.shape_inference.infer_shapes(model)


class TestGraphReductionCapabilities:
    @pytest.mark.parametrize("name", _NEW_CAPABILITIES)
    def test_capability_is_registered_and_disabled(self, name: str) -> None:
        capability = SURGERY_CAPABILITIES[name]
        assert capability.default is False
        assert capability.ort_name is None

    @pytest.mark.parametrize("name", _NEW_CAPABILITIES)
    def test_cli_lists_capability(self, name: str) -> None:
        result = CliRunner().invoke(optimize, ["--list-capabilities"])
        assert result.exit_code == 0
        assert f"--enable-{name}" in result.output

    @pytest.mark.parametrize(
        "kwarg",
        [
            "simplify_l2_normalization",
            "gathernd_to_resize",
            "silu_to_quick_gelu",
            "scaled_matmul_to_fused_matmul",
        ],
    )
    def test_each_capability_enables_surgery_pipe(self, kwarg: str) -> None:
        config = SurgeryPipe.build_config(**{kwarg: True})
        assert getattr(config, kwarg) is True
        assert SurgeryPipe.should_process(config) is True


class TestSimplifyL2Normalization:
    def test_removes_only_clip_and_expand(self) -> None:
        model = _make_l2_normalization_model()
        result = SurgeryPipe().process(
            model,
            SurgeryPipeConfig(simplify_l2_normalization=True),
        )

        assert [node.op_type for node in result.graph.node] == ["ReduceL2", "Div"]
        assert result.graph.node[-1].input == ["x", "norm"]
        onnx.checker.check_model(result)

    @pytest.mark.parametrize(
        "feeds",
        [
            {"x": np.random.RandomState(1).randn(1, 3, 2, 2).astype(np.float32)},
            {"x": np.zeros((1, 3, 2, 2), dtype=np.float32)},
        ],
    )
    def test_preserves_values_including_zero_norm(self, feeds: dict[str, np.ndarray]) -> None:
        model = _make_l2_normalization_model()
        result = SurgeryPipe().process(
            model,
            SurgeryPipeConfig(simplify_l2_normalization=True),
        )
        expected = _run_ort(model, feeds)[0]
        actual = _run_ort(result, feeds)[0]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)

    @pytest.mark.parametrize(
        "builder",
        [
            lambda: _make_l2_normalization_model(clip_min=1.0),
            lambda: _make_l2_normalization_model(axis=2),
            lambda: _make_l2_normalization_model(expand_batch=2),
        ],
    )
    def test_rejects_semantic_near_misses(self, builder: Callable[[], onnx.ModelProto]) -> None:
        model = builder()
        result = SurgeryPipe().process(
            model,
            SurgeryPipeConfig(simplify_l2_normalization=True),
        )
        assert [node.op_type for node in result.graph.node] == [
            "ReduceL2",
            "Clip",
            "Expand",
            "Div",
        ]


class TestGatherNDToResize:
    def test_replaces_exact_grid_and_prunes_grid_builders(self) -> None:
        model = _make_gathernd_upsampling_model()
        result = SurgeryPipe().process(model, SurgeryPipeConfig(gathernd_to_resize=True))

        resize = next(node for node in result.graph.node if node.op_type == "Resize")
        assert not any(node.op_type == "GatherND" for node in result.graph.node)
        assert resize.input[0] == "source"
        assert resize.output[0] == "y"
        assert _attributes(resize) == {
            "coordinate_transformation_mode": b"asymmetric",
            "mode": b"nearest",
            "nearest_mode": b"floor",
        }
        scales = next(init for init in result.graph.initializer if init.name == resize.input[2])
        np.testing.assert_array_equal(
            numpy_helper.to_array(scales),
            np.asarray([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
        )
        onnx.checker.check_model(result)

    def test_preserves_generated_fp16_values(self) -> None:
        model = _make_gathernd_upsampling_model()
        result = SurgeryPipe().process(model, SurgeryPipeConfig(gathernd_to_resize=True))
        feeds = {"x": np.random.RandomState(2).randn(1, 2, 2, 3).astype(np.float16)}

        expected = ReferenceEvaluator(model).run(None, feeds)[0]
        actual = ReferenceEvaluator(result).run(None, feeds)[0]
        np.testing.assert_array_equal(actual, expected)

    def test_rejects_different_coordinate_formula(self) -> None:
        model = _make_gathernd_upsampling_model(half=0.25)
        result = SurgeryPipe().process(model, SurgeryPipeConfig(gathernd_to_resize=True))
        assert any(node.op_type == "GatherND" for node in result.graph.node)
        assert not any(node.op_type == "Resize" for node in result.graph.node)


class TestSiLUToQuickGelu:
    def test_emits_quick_gelu_with_alpha_one(self) -> None:
        model = _make_silu_model()
        result = SurgeryPipe().process(model, SurgeryPipeConfig(silu_to_quick_gelu=True))

        assert len(result.graph.node) == 1
        quick_gelu = result.graph.node[0]
        assert (quick_gelu.domain, quick_gelu.op_type) == ("com.microsoft", "QuickGelu")
        assert _attributes(quick_gelu)["alpha"] == pytest.approx(1.0)
        assert [(opset.domain, opset.version) for opset in result.opset_import].count(
            ("com.microsoft", 1)
        ) == 1
        onnx.checker.check_model(result)

    def test_preserves_values(self) -> None:
        model = _make_silu_model()
        result = SurgeryPipe().process(model, SurgeryPipeConfig(silu_to_quick_gelu=True))
        feeds = {"x": np.random.RandomState(3).randn(2, 3).astype(np.float32)}
        np.testing.assert_allclose(
            _run_ort(result, feeds)[0],
            _run_ort(model, feeds)[0],
            rtol=1e-6,
            atol=1e-7,
        )

    def test_rejects_scaled_sigmoid(self) -> None:
        model = _make_silu_model(scaled_sigmoid=True)
        result = SurgeryPipe().process(model, SurgeryPipeConfig(silu_to_quick_gelu=True))
        assert not any(node.op_type == "QuickGelu" for node in result.graph.node)


class TestScaledMatMulToFusedMatMul:
    def test_emits_fused_matmul_and_removes_scale(self) -> None:
        model = _make_scaled_matmul_model()
        result = SurgeryPipe().process(
            model,
            SurgeryPipeConfig(scaled_matmul_to_fused_matmul=True),
        )

        assert len(result.graph.node) == 1
        fused = result.graph.node[0]
        assert (fused.domain, fused.op_type) == ("com.microsoft", "FusedMatMul")
        assert _attributes(fused)["alpha"] == pytest.approx(0.125)
        assert all(initializer.name != "scale" for initializer in result.graph.initializer)
        onnx.checker.check_model(result)

    def test_preserves_values(self) -> None:
        model = _make_scaled_matmul_model()
        result = SurgeryPipe().process(
            model,
            SurgeryPipeConfig(scaled_matmul_to_fused_matmul=True),
        )
        feeds = {
            "a": np.random.RandomState(4).randn(2, 4).astype(np.float32),
            "b": np.random.RandomState(5).randn(4, 2).astype(np.float32),
        }
        np.testing.assert_array_equal(_run_ort(result, feeds)[0], _run_ort(model, feeds)[0])

    def test_rejects_non_scalar_scale(self) -> None:
        model = _make_scaled_matmul_model(scalar_scale=False)
        result = SurgeryPipe().process(
            model,
            SurgeryPipeConfig(scaled_matmul_to_fused_matmul=True),
        )
        assert [node.op_type for node in result.graph.node] == ["MatMul", "Mul"]

    def test_accepts_bfloat16_scalar(self) -> None:
        model = _make_scaled_matmul_model(element_type=TensorProto.BFLOAT16)
        result = SurgeryPipe().process(
            model,
            SurgeryPipeConfig(scaled_matmul_to_fused_matmul=True),
        )
        assert [node.op_type for node in result.graph.node] == ["FusedMatMul"]
        assert _attributes(result.graph.node[0])["alpha"] == pytest.approx(0.125)

    def test_rejects_scale_with_different_element_type(self) -> None:
        model = _make_scaled_matmul_model(
            element_type=TensorProto.FLOAT16,
            scale_element_type=TensorProto.FLOAT,
        )
        result = SurgeryPipe().process(
            model,
            SurgeryPipeConfig(scaled_matmul_to_fused_matmul=True),
        )
        assert [node.op_type for node in result.graph.node] == ["MatMul", "Mul"]


def test_all_new_surgeries_are_idempotent() -> None:
    models_and_configs = [
        (_make_l2_normalization_model(), SurgeryPipeConfig(simplify_l2_normalization=True)),
        (_make_gathernd_upsampling_model(), SurgeryPipeConfig(gathernd_to_resize=True)),
        (_make_silu_model(), SurgeryPipeConfig(silu_to_quick_gelu=True)),
        (
            _make_scaled_matmul_model(),
            SurgeryPipeConfig(scaled_matmul_to_fused_matmul=True),
        ),
    ]
    for model, config in models_and_configs:
        once = SurgeryPipe().process(model, config)
        twice = SurgeryPipe().process(once, config)
        assert once.SerializeToString() == twice.SerializeToString()


@pytest.mark.parametrize(
    ("model_factory", "option", "removed_op", "added_op", "added_domain"),
    [
        (_make_l2_normalization_model, "simplify_l2_normalization", "Clip", None, ""),
        (
            _make_gathernd_upsampling_model,
            "gathernd_to_resize",
            "GatherND",
            "Resize",
            "",
        ),
        (
            _make_silu_model,
            "silu_to_quick_gelu",
            "Sigmoid",
            "QuickGelu",
            "com.microsoft",
        ),
        (
            _make_scaled_matmul_model,
            "scaled_matmul_to_fused_matmul",
            "MatMul",
            "FusedMatMul",
            "com.microsoft",
        ),
    ],
)
def test_surgery_survives_full_optimizer_pipeline(
    model_factory: Callable[[], onnx.ModelProto],
    option: str,
    removed_op: str,
    added_op: str | None,
    added_domain: str,
) -> None:
    result = Optimizer().optimize(
        model_factory(),
        constant_folding=False,
        **{option: True},
    )

    assert all(node.op_type != removed_op for node in result.graph.node)
    if added_op is not None:
        assert any(
            node.op_type == added_op and node.domain == added_domain for node in result.graph.node
        )
