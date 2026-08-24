# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Surgery Pipe Tests.

Tests for SurgeryPipe which performs pre-optimization model surgery.

Following Cardinal Rules:
- CARDINAL RULE #1: No hardcoded model architectures
- CARDINAL RULE #2: All tests use pytest with code-generated results
- CARDINAL RULE #3: Tests must run and pass

Fixtures used from conftest.py:
- causal_mask_model: Model with extreme float constants (-3.4e38)
- model_with_normal_constants: Model with normal float constants
"""

from __future__ import annotations

import json

import numpy as np
import onnx
import pytest
from onnx import numpy_helper

from winml.modelkit.optim.pipes import (
    SURGERY_CAPABILITIES,
    SurgeryPipe,
    SurgeryPipeConfig,
)
from winml.modelkit.quant.hints import QuantizationHintError


# =============================================================================
# SURGERY CAPABILITIES TESTS
# =============================================================================


class TestSurgeryCapabilities:
    """Test surgery capability definitions."""

    def test_surgery_capabilities_exists(self) -> None:
        """Verify SURGERY_CAPABILITIES dict exists and is not empty."""
        assert SURGERY_CAPABILITIES is not None
        assert len(SURGERY_CAPABILITIES) > 0

    def test_clamp_constant_values_capability_exists(self) -> None:
        """Verify clamp-constant-values capability is defined."""
        assert "clamp-constant-values" in SURGERY_CAPABILITIES

    def test_clamp_constant_values_has_none_ort_name(self) -> None:
        """Verify clamp-constant-values has None ort_name (custom implementation)."""
        cap = SURGERY_CAPABILITIES["clamp-constant-values"]
        assert cap.ort_name is None

    def test_clamp_constant_values_default_is_false(self) -> None:
        """Verify clamp-constant-values defaults to False."""
        cap = SURGERY_CAPABILITIES["clamp-constant-values"]
        assert cap.default is False


# =============================================================================
# SURGERY PIPE CONFIG TESTS
# =============================================================================


class TestSurgeryPipeConfig:
    """Test SurgeryPipeConfig initialization and attributes."""

    def test_default_config(self) -> None:
        """Verify default config values."""
        config = SurgeryPipeConfig()
        assert config.clamp_constant_values is False
        assert config.clamp_min == -1e3
        assert config.clamp_max == 1e3
        assert config.verbose is False

    def test_custom_clamp_range(self) -> None:
        """Verify custom clamp range can be set."""
        config = SurgeryPipeConfig(
            clamp_constant_values=True,
            clamp_min=-1e3,
            clamp_max=1e3,
        )
        assert config.clamp_min == -1e3
        assert config.clamp_max == 1e3

    def test_verbose_flag(self) -> None:
        """Verify verbose flag can be set."""
        config = SurgeryPipeConfig(verbose=True)
        assert config.verbose is True


# =============================================================================
# SURGERY PIPE BUILD CONFIG TESTS
# =============================================================================


class TestSurgeryPipeBuildConfig:
    """Test SurgeryPipe.build_config() method."""

    def test_build_config_returns_surgery_config(self) -> None:
        """Verify build_config returns SurgeryPipeConfig instance."""
        config = SurgeryPipe.build_config()
        assert isinstance(config, SurgeryPipeConfig)

    def test_build_config_default_disabled(self) -> None:
        """Verify build_config defaults to disabled clamp_constant_values."""
        config = SurgeryPipe.build_config()
        assert config.clamp_constant_values is False

    def test_build_config_enable_via_kwarg(self) -> None:
        """Verify clamp_constant_values can be enabled via kwarg."""
        config = SurgeryPipe.build_config(clamp_constant_values=True)
        assert config.clamp_constant_values is True

    def test_build_config_custom_clamp_range(self) -> None:
        """Verify custom clamp range can be set via kwargs."""
        config = SurgeryPipe.build_config(
            clamp_constant_values=True,
            clamp_min=-500,
            clamp_max=500,
        )
        assert config.clamp_min == -500
        assert config.clamp_max == 500

    def test_build_config_verbose(self) -> None:
        """Verify verbose flag can be set via kwarg."""
        config = SurgeryPipe.build_config(verbose=True)
        assert config.verbose is True


# =============================================================================
# SURGERY PIPE SHOULD_PROCESS TESTS
# =============================================================================


class TestSurgeryPipeShouldProcess:
    """Test SurgeryPipe.should_process() method."""

    def test_should_process_false_when_disabled(self) -> None:
        """Verify should_process returns False when clamp_constant_values is False."""
        config = SurgeryPipeConfig(clamp_constant_values=False)
        assert SurgeryPipe.should_process(config) is False

    def test_should_process_true_when_enabled(self) -> None:
        """Verify should_process returns True when clamp_constant_values is True."""
        config = SurgeryPipeConfig(clamp_constant_values=True)
        assert SurgeryPipe.should_process(config) is True


# =============================================================================
# SURGERY PIPE PROCESS TESTS
# =============================================================================


class TestSurgeryPipeProcess:
    """Test SurgeryPipe.process() method."""

    def test_process_returns_model_unchanged_when_disabled(
        self, causal_mask_model: onnx.ModelProto
    ) -> None:
        """Verify process returns model unchanged when clamp_constant_values is False."""
        pipe = SurgeryPipe()
        config = SurgeryPipeConfig(clamp_constant_values=False)

        result = pipe.process(causal_mask_model, config)

        # Should return the same model object (no processing)
        assert result is causal_mask_model

    def test_process_clamps_causal_mask_extreme_values(
        self, causal_mask_model: onnx.ModelProto
    ) -> None:
        """Verify process clamps extreme float constants in causal mask."""
        pipe = SurgeryPipe()
        config = SurgeryPipeConfig(clamp_constant_values=True, clamp_min=-1e4, clamp_max=1e4)

        result = pipe.process(causal_mask_model, config)

        # Check causal_mask.1 is clamped
        for init in result.graph.initializer:
            if init.name == "causal_mask.1":
                tensor = numpy_helper.to_array(init)
                assert tensor.min() >= -1e4, f"Min value {tensor.min()} below clamp_min"
                assert tensor.max() <= 1e4, f"Max value {tensor.max()} above clamp_max"
                break
        else:
            pytest.fail("causal_mask.1 not found in result model")

    def test_process_clamps_mask_value_scalar(self, causal_mask_model: onnx.ModelProto) -> None:
        """Verify process clamps scalar mask_value constant."""
        pipe = SurgeryPipe()
        config = SurgeryPipeConfig(clamp_constant_values=True, clamp_min=-1e4, clamp_max=1e4)

        result = pipe.process(causal_mask_model, config)

        # Check mask_value is clamped
        for init in result.graph.initializer:
            if init.name == "mask_value":
                tensor = numpy_helper.to_array(init)
                assert tensor >= -1e4, f"mask_value {tensor} below clamp_min"
                break
        else:
            pytest.fail("mask_value not found in result model")

    def test_process_preserves_zero_values(self, causal_mask_model: onnx.ModelProto) -> None:
        """Verify process preserves zero values in causal mask (only clamps extremes)."""
        pipe = SurgeryPipe()
        config = SurgeryPipeConfig(clamp_constant_values=True)

        result = pipe.process(causal_mask_model, config)

        for init in result.graph.initializer:
            if init.name == "causal_mask.1":
                tensor = numpy_helper.to_array(init)
                # Causal mask has zeros on lower triangle, they should still be there
                assert 0.0 in tensor, "Zero values should be preserved"
                break

    def test_process_does_not_modify_normal_constants(
        self, model_with_normal_constants: onnx.ModelProto
    ) -> None:
        """Verify process does not modify constants within clamp range."""
        pipe = SurgeryPipe()
        config = SurgeryPipeConfig(clamp_constant_values=True)

        # Get original values
        original_values = None
        for init in model_with_normal_constants.graph.initializer:
            if init.name == "normal_const":
                original_values = numpy_helper.to_array(init).copy()
                break

        result = pipe.process(model_with_normal_constants, config)

        # Values should be unchanged
        for init in result.graph.initializer:
            if init.name == "normal_const":
                result_values = numpy_helper.to_array(init)
                np.testing.assert_array_equal(
                    result_values,
                    original_values,
                    err_msg="Normal constants should not be modified",
                )
                break

    def test_process_custom_clamp_range(self, causal_mask_model: onnx.ModelProto) -> None:
        """Verify process uses custom clamp range."""
        pipe = SurgeryPipe()
        config = SurgeryPipeConfig(
            clamp_constant_values=True,
            clamp_min=-100,
            clamp_max=100,
        )

        result = pipe.process(causal_mask_model, config)

        for init in result.graph.initializer:
            if init.name == "causal_mask.1":
                tensor = numpy_helper.to_array(init)
                assert tensor.min() >= -100, f"Min value {tensor.min()} below custom clamp_min"
                assert tensor.max() <= 100, f"Max value {tensor.max()} above custom clamp_max"
                break

    def test_process_returns_copy_not_original(self, causal_mask_model: onnx.ModelProto) -> None:
        """Verify process returns a copy, not the original model."""
        pipe = SurgeryPipe()
        config = SurgeryPipeConfig(clamp_constant_values=True)

        result = pipe.process(causal_mask_model, config)

        # Result should be a different object
        assert result is not causal_mask_model

    def test_process_model_remains_valid(self, causal_mask_model: onnx.ModelProto) -> None:
        """Verify processed model is still valid ONNX."""
        pipe = SurgeryPipe()
        config = SurgeryPipeConfig(clamp_constant_values=True)

        result = pipe.process(causal_mask_model, config)

        # Should not raise
        onnx.checker.check_model(result)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestSurgeryPipeIntegration:
    """Integration tests for SurgeryPipe with quantization."""

    def test_clamped_causal_mask_quantizes_without_inf_scales(
        self, causal_mask_model: onnx.ModelProto
    ) -> None:
        """Verify clamped causal mask can be quantized without producing inf scales.

        This is the main use case: extreme values like -3.4e38 in attention masks
        cause quantization to produce inf scales, which break QNN compilation.
        """
        import tempfile
        from pathlib import Path

        from onnxruntime.quantization import QuantType, quantize_dynamic

        pipe = SurgeryPipe()
        config = SurgeryPipeConfig(clamp_constant_values=True)

        # Apply surgery
        clamped_model = pipe.process(causal_mask_model, config)

        # Quantize the clamped model
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "clamped.onnx"
            output_path = Path(tmpdir) / "quantized.onnx"

            onnx.save(clamped_model, str(input_path))
            quantize_dynamic(str(input_path), str(output_path), weight_type=QuantType.QUInt8)

            # Load and check for inf scales
            quant_model = onnx.load(str(output_path))
            for init in quant_model.graph.initializer:
                if "scale" in init.name.lower():
                    arr = numpy_helper.to_array(init)
                    assert not np.isinf(arr).any(), f"Found inf in scale tensor: {init.name}"

    def test_surgery_pipe_in_pipes_list(self) -> None:
        """Verify SurgeryPipe is included in PIPES list."""
        from winml.modelkit.optim.pipes import PIPES

        pipe_names = [p.name for p in PIPES]
        assert "surgery" in pipe_names

    def test_surgery_pipe_runs_last(self) -> None:
        """Verify SurgeryPipe runs after other pipes (post-optimization surgery).

        SurgeryPipe runs LAST to clamp constant values AFTER ORT constant folding
        has moved Constant nodes into initializers.
        """
        from winml.modelkit.optim.pipes import PIPES

        # SurgeryPipe should be last in the list
        assert PIPES[-1].name == "surgery"


# =============================================================================
# UNTIE-CONSTANT-BATCHED-MATMUL TESTS
# =============================================================================


def _make_batched_const_matmul_model(
    *,
    const_rank: int = 3,
    const_on_rhs: bool = True,
) -> onnx.ModelProto:
    """Build a model with a batched MatMul that has one constant operand.

    data [2,3,4] @ W(const) [2,4,5] -> out [2,3,5] (const on rhs), or the
    transposed arrangement when ``const_on_rhs`` is False.
    """
    from onnx import TensorProto, helper

    rng = np.random.RandomState(0)
    if const_on_rhs:
        data_shape, w_shape, out_shape = [2, 3, 4], [2, 4, 5], [2, 3, 5]
        mm_inputs = ["data", "W"]
    else:
        data_shape, w_shape, out_shape = [2, 4, 5], [2, 3, 4], [2, 3, 5]
        mm_inputs = ["W", "data"]

    if const_rank == 2:
        w_shape = w_shape[1:]

    w = numpy_helper.from_array(rng.randn(*w_shape).astype(np.float32), "W")
    matmul = helper.make_node("MatMul", mm_inputs, ["out"], name="batched_matmul")
    graph = helper.make_graph(
        [matmul],
        "test_batched_const_matmul",
        [helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape)],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)],
        initializer=[w],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class TestUntieConstantBatchedMatmulCapability:
    """Capability/config plumbing for untie-constant-batched-matmul."""

    def test_capability_exists(self) -> None:
        """Capability is registered with a None ort_name (custom impl)."""
        assert "untie-constant-batched-matmul" in SURGERY_CAPABILITIES
        assert SURGERY_CAPABILITIES["untie-constant-batched-matmul"].ort_name is None

    def test_build_config_enable_via_kwarg(self) -> None:
        """Flag can be toggled through build_config."""
        config = SurgeryPipe.build_config(untie_constant_batched_matmul=True)
        assert config.untie_constant_batched_matmul is True

    def test_should_process_true_when_enabled(self) -> None:
        """should_process is True when only this surgery is enabled."""
        config = SurgeryPipeConfig(untie_constant_batched_matmul=True)
        assert SurgeryPipe.should_process(config) is True


class TestUntieConstantBatchedMatmulProcess:
    """Graph transform behavior."""

    def test_constant_operand_becomes_runtime_valued(self) -> None:
        """The MatMul no longer consumes the initializer directly."""
        model = _make_batched_const_matmul_model()
        result = SurgeryPipe().process(model, SurgeryPipeConfig(untie_constant_batched_matmul=True))

        matmul = next(n for n in result.graph.node if n.op_type == "MatMul")
        initializer_names = {init.name for init in result.graph.initializer}
        # No MatMul input is a direct initializer anymore.
        assert not (set(matmul.input) & initializer_names)
        # An Add node now produces the (formerly constant) operand.
        add_nodes = [n for n in result.graph.node if n.op_type == "Add"]
        assert len(add_nodes) == 1
        assert add_nodes[0].output[0] in matmul.input
        # Graph remains structurally valid.
        onnx.checker.check_model(result)

    def test_numerics_unchanged(self) -> None:
        """+0 tie leaves outputs bit-for-bit identical on ORT CPU."""
        import onnxruntime as ort

        model = _make_batched_const_matmul_model()
        transformed = SurgeryPipe().process(
            model, SurgeryPipeConfig(untie_constant_batched_matmul=True)
        )

        rng = np.random.RandomState(7)
        feed = {"data": rng.randn(2, 3, 4).astype(np.float32)}

        ref = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, feed)[0]
        got = ort.InferenceSession(
            transformed.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, feed)[0]
        np.testing.assert_array_equal(ref, got)

    def test_two_dim_constant_is_left_untouched(self) -> None:
        """Rank-2 constant gemm compiles on OV GPU, so it must not be rewritten."""
        model = _make_batched_const_matmul_model(const_rank=2)
        result = SurgeryPipe().process(model, SurgeryPipeConfig(untie_constant_batched_matmul=True))
        assert not any(n.op_type == "Add" for n in result.graph.node)

    def test_constant_on_lhs_is_handled(self) -> None:
        """A constant rank-3 operand on the LHS is untied too."""
        model = _make_batched_const_matmul_model(const_on_rhs=False)
        result = SurgeryPipe().process(model, SurgeryPipeConfig(untie_constant_batched_matmul=True))
        assert any(n.op_type == "Add" for n in result.graph.node)

    def test_duplicate_node_names_do_not_collide(self) -> None:
        """Two target MatMuls with empty names produce a valid graph.

        Node names are optional in ONNX; exporters routinely leave them blank.
        The generated dynamic-operand names must be unique regardless, or the
        transformed graph would have colliding tensor names and fail validation.
        """
        from onnx import TensorProto, helper

        rng = np.random.RandomState(0)
        w1 = numpy_helper.from_array(rng.randn(2, 4, 5).astype(np.float32), "W1")
        w2 = numpy_helper.from_array(rng.randn(2, 5, 6).astype(np.float32), "W2")
        # Both MatMuls deliberately left unnamed (name="").
        mm1 = helper.make_node("MatMul", ["data", "W1"], ["mid"], name="")
        mm2 = helper.make_node("MatMul", ["mid", "W2"], ["out"], name="")
        graph = helper.make_graph(
            [mm1, mm2],
            "test_dup_names",
            [helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 3, 4])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3, 6])],
            initializer=[w1, w2],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        result = SurgeryPipe().process(model, SurgeryPipeConfig(untie_constant_batched_matmul=True))

        # Both constants are untied and the graph stays structurally valid.
        add_nodes = [n for n in result.graph.node if n.op_type == "Add"]
        assert len(add_nodes) == 2
        assert len({n.output[0] for n in add_nodes}) == 2
        onnx.checker.check_model(result)

        # Numerics are unchanged versus the original model.
        import onnxruntime as ort

        feed = {"data": np.random.RandomState(7).randn(2, 3, 4).astype(np.float32)}
        ref = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, feed)[0]
        got = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, feed)[0]
        np.testing.assert_array_equal(ref, got)


# =============================================================================
# TRIM-SPLIT-GROUPED-CONV TESTS
# =============================================================================


def _make_grouped_conv_tail_slice_model(
    *,
    input_length: int = 3,
    pads: tuple[int, int] = (4, 4),
    slice_end: int = -1,
) -> onnx.ModelProto:
    """Build a grouped 1D Conv whose final output is removed by Slice."""
    from onnx import TensorProto, helper

    rng = np.random.RandomState(0)
    weights = numpy_helper.from_array(rng.randn(32, 2, 8).astype(np.float32), "weights")
    bias = numpy_helper.from_array(rng.randn(32).astype(np.float32), "bias")
    starts = numpy_helper.from_array(np.array([0], dtype=np.int64), "starts")
    ends = numpy_helper.from_array(np.array([slice_end], dtype=np.int64), "ends")
    axes = numpy_helper.from_array(np.array([2], dtype=np.int64), "axes")
    steps = numpy_helper.from_array(np.array([1], dtype=np.int64), "steps")
    conv = helper.make_node(
        "Conv",
        ["input", "weights", "bias"],
        ["conv_output"],
        name="grouped_conv",
        group=16,
        kernel_shape=[8],
        pads=list(pads),
        strides=[1],
        dilations=[1],
    )
    tail_slice = helper.make_node(
        "Slice",
        ["conv_output", "starts", "ends", "axes", "steps"],
        ["output"],
        name="tail_slice",
    )
    graph = helper.make_graph(
        [conv, tail_slice],
        "grouped_conv_tail_slice",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 32, input_length])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32, 3])],
        initializer=[weights, bias, starts, ends, axes, steps],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class TestTrimSplitGroupedConvCapability:
    """Capability/config plumbing for trim-split-grouped-conv."""

    def test_capability_exists_and_defaults_false(self) -> None:
        capability = SURGERY_CAPABILITIES["trim-split-grouped-conv"]
        assert capability.ort_name is None
        assert capability.default is False

    def test_build_config_enable_via_kwarg(self) -> None:
        config = SurgeryPipe.build_config(trim_split_grouped_conv=True)
        assert config.trim_split_grouped_conv is True
        assert SurgeryPipe.should_process(config) is True


class TestTrimSplitGroupedConvProcess:
    """Static reachability rewrite behavior."""

    @staticmethod
    def _assert_no_rewrite(model: onnx.ModelProto) -> None:
        original = model.SerializeToString()
        result = SurgeryPipe().process(
            model,
            SurgeryPipeConfig(trim_split_grouped_conv=True),
        )
        assert model.SerializeToString() == original
        assert result.SerializeToString() == original

    def test_rewrites_supported_graph_and_emits_region_hint(self) -> None:
        model = _make_grouped_conv_tail_slice_model()
        original = model.SerializeToString()

        result = SurgeryPipe().process(model, SurgeryPipeConfig(trim_split_grouped_conv=True))

        assert model.SerializeToString() == original
        onnx.checker.check_model(result, full_check=True)
        assert [node.op_type for node in result.graph.node] == [
            "Split",
            "Conv",
            "Conv",
            "Concat",
        ]
        convs = [node for node in result.graph.node if node.op_type == "Conv"]
        assert len(convs) == 2
        for conv in convs:
            attributes = {
                attribute.name: onnx.helper.get_attribute_value(attribute)
                for attribute in conv.attribute
            }
            assert attributes["group"] == 8
            assert attributes["kernel_shape"] == [5]
            assert attributes["pads"] == [2, 2]
        concat = next(node for node in result.graph.node if node.op_type == "Concat")
        assert concat.output == ["output"]
        initializer_names = {initializer.name for initializer in result.graph.initializer}
        assert not {"weights", "bias", "starts", "ends", "axes", "steps"}.intersection(
            initializer_names
        )
        assert {
            "weights__winml_part0",
            "weights__winml_part1",
            "bias__winml_part0",
            "bias__winml_part1",
        }.issubset(initializer_names)

        metadata = {item.key: item.value for item in result.metadata_props}
        hint = json.loads(metadata["winml.quantization.region_hints"])
        assert hint == {
            "version": 1,
            "regions": [
                {
                    "kind": "conv_concat",
                    "branches": [convs[0].name, convs[1].name],
                    "concat": concat.name,
                }
            ],
        }

    def test_preserves_cpu_outputs_and_is_idempotent(self) -> None:
        import onnxruntime as ort

        model = _make_grouped_conv_tail_slice_model()
        pipe = SurgeryPipe()
        config = SurgeryPipeConfig(trim_split_grouped_conv=True)
        transformed = pipe.process(model, config)
        second_pass = pipe.process(transformed, config)
        feed = {"input": np.random.RandomState(7).randn(1, 32, 3).astype(np.float32)}

        expected = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, feed)[0]
        actual = ort.InferenceSession(
            transformed.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, feed)[0]

        np.testing.assert_array_equal(actual, expected)
        assert second_pass.SerializeToString() == transformed.SerializeToString()

    @pytest.mark.parametrize(
        ("model", "expected_kernel", "expected_pads"),
        [
            (
                _make_grouped_conv_tail_slice_model(
                    input_length=5,
                    pads=(4, 2),
                    slice_end=3,
                ),
                [6],
                [2, 1],
            ),
            (
                _make_grouped_conv_tail_slice_model(
                    input_length=3,
                    pads=(1, 7),
                    slice_end=-1,
                ),
                [4],
                [1, 2],
            ),
        ],
    )
    def test_trims_one_kernel_edge_with_exact_cpu_outputs(
        self,
        model: onnx.ModelProto,
        expected_kernel: list[int],
        expected_pads: list[int],
    ) -> None:
        import onnxruntime as ort

        transformed = SurgeryPipe().process(
            model,
            SurgeryPipeConfig(trim_split_grouped_conv=True),
        )
        convs = [node for node in transformed.graph.node if node.op_type == "Conv"]
        for conv in convs:
            attributes = {
                attribute.name: onnx.helper.get_attribute_value(attribute)
                for attribute in conv.attribute
            }
            assert attributes["kernel_shape"] == expected_kernel
            assert attributes["pads"] == expected_pads

        input_length = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
        feed = {"input": np.random.RandomState(7).randn(1, 32, input_length).astype(np.float32)}
        expected = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, feed)[0]
        actual = ort.InferenceSession(
            transformed.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, feed)[0]
        np.testing.assert_array_equal(actual, expected)

    def test_dynamic_spatial_length_is_unchanged(self) -> None:
        model = _make_grouped_conv_tail_slice_model()
        spatial_dimension = model.graph.input[0].type.tensor_type.shape.dim[2]
        spatial_dimension.ClearField("dim_value")
        spatial_dimension.dim_param = "sequence"

        self._assert_no_rewrite(model)

    def test_conv_fanout_is_unchanged(self) -> None:
        model = _make_grouped_conv_tail_slice_model()
        model.graph.node.append(
            onnx.helper.make_node(
                "Identity",
                ["conv_output"],
                ["other_output"],
                name="other_consumer",
            )
        )

        self._assert_no_rewrite(model)

    def test_conv_output_captured_by_subgraph_is_unchanged(self) -> None:
        from onnx import TensorProto

        model = _make_grouped_conv_tail_slice_model()
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(True, dtype=np.bool_), "condition")
        )

        def branch_graph(name: str) -> onnx.GraphProto:
            output_name = f"{name}_output"
            return onnx.helper.make_graph(
                [
                    onnx.helper.make_node(
                        "Identity",
                        ["conv_output"],
                        [output_name],
                        name=f"{name}_identity",
                    )
                ],
                name,
                [],
                [
                    onnx.helper.make_tensor_value_info(
                        output_name,
                        TensorProto.FLOAT,
                        [1, 32, 4],
                    )
                ],
            )

        model.graph.node.append(
            onnx.helper.make_node(
                "If",
                ["condition"],
                ["captured_conv_output"],
                name="capture_conv_output",
                then_branch=branch_graph("then_branch"),
                else_branch=branch_graph("else_branch"),
            )
        )
        model.graph.output.append(
            onnx.helper.make_tensor_value_info(
                "captured_conv_output",
                TensorProto.FLOAT,
                [1, 32, 4],
            )
        )
        onnx.checker.check_model(model, full_check=True)

        self._assert_no_rewrite(model)

    def test_nonunit_stride_is_unchanged(self) -> None:
        model = _make_grouped_conv_tail_slice_model()
        conv = next(node for node in model.graph.node if node.op_type == "Conv")
        strides = next(attribute for attribute in conv.attribute if attribute.name == "strides")
        strides.CopyFrom(onnx.helper.make_attribute("strides", [2]))

        self._assert_no_rewrite(model)

    def test_slice_that_keeps_full_output_is_unchanged(self) -> None:
        model = _make_grouped_conv_tail_slice_model()
        ends = next(
            initializer for initializer in model.graph.initializer if initializer.name == "ends"
        )
        ends.CopyFrom(numpy_helper.from_array(np.array([4], dtype=np.int64), "ends"))

        self._assert_no_rewrite(model)

    def test_non_grouped_conv_is_unchanged(self) -> None:
        model = _make_grouped_conv_tail_slice_model()
        conv = next(node for node in model.graph.node if node.op_type == "Conv")
        group = next(attribute for attribute in conv.attribute if attribute.name == "group")
        group.CopyFrom(onnx.helper.make_attribute("group", 1))
        weights = next(
            initializer for initializer in model.graph.initializer if initializer.name == "weights"
        )
        weights.CopyFrom(
            numpy_helper.from_array(
                np.random.RandomState(1).randn(32, 32, 8).astype(np.float32),
                "weights",
            )
        )

        self._assert_no_rewrite(model)

    def test_generated_names_do_not_collide(self) -> None:
        model = _make_grouped_conv_tail_slice_model()
        model.graph.node.append(
            onnx.helper.make_node(
                "Identity",
                ["input"],
                ["collision_output"],
                name="grouped_conv__winml_part0",
            )
        )

        result = SurgeryPipe().process(
            model,
            SurgeryPipeConfig(trim_split_grouped_conv=True),
        )

        onnx.checker.check_model(result, full_check=True)
        node_names = [node.name for node in result.graph.node]
        assert len(node_names) == len(set(node_names))
        hint = json.loads(
            next(
                item.value
                for item in result.metadata_props
                if item.key == "winml.quantization.region_hints"
            )
        )
        assert hint["regions"][0]["branches"][0] == "grouped_conv__winml_part0_1"

    def test_shared_source_initializer_is_preserved(self) -> None:
        from onnx import TensorProto

        model = _make_grouped_conv_tail_slice_model()
        model.graph.node.append(
            onnx.helper.make_node(
                "Identity",
                ["weights"],
                ["shared_weights_output"],
                name="shared_weights",
            )
        )
        model.graph.output.append(
            onnx.helper.make_tensor_value_info(
                "shared_weights_output",
                TensorProto.FLOAT,
                [32, 2, 8],
            )
        )

        result = SurgeryPipe().process(
            model,
            SurgeryPipeConfig(trim_split_grouped_conv=True),
        )

        onnx.checker.check_model(result, full_check=True)
        initializer_names = {initializer.name for initializer in result.graph.initializer}
        assert "weights" in initializer_names
        assert not {"bias", "starts", "ends", "axes", "steps"}.intersection(initializer_names)

    def test_source_initializer_captured_by_subgraph_is_preserved(self) -> None:
        from onnx import TensorProto

        model = _make_grouped_conv_tail_slice_model()
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(True, dtype=np.bool_), "condition")
        )

        def branch_graph(name: str) -> onnx.GraphProto:
            identity = onnx.helper.make_node(
                "Identity",
                ["weights"],
                [f"{name}_output"],
                name=f"{name}_identity",
            )
            return onnx.helper.make_graph(
                [identity],
                name,
                [],
                [
                    onnx.helper.make_tensor_value_info(
                        f"{name}_output",
                        TensorProto.FLOAT,
                        [32, 2, 8],
                    )
                ],
            )

        model.graph.node.append(
            onnx.helper.make_node(
                "If",
                ["condition"],
                ["captured_weights_output"],
                name="capture_weights",
                then_branch=branch_graph("then_branch"),
                else_branch=branch_graph("else_branch"),
            )
        )
        model.graph.output.append(
            onnx.helper.make_tensor_value_info(
                "captured_weights_output",
                TensorProto.FLOAT,
                [32, 2, 8],
            )
        )

        result = SurgeryPipe().process(
            model,
            SurgeryPipeConfig(trim_split_grouped_conv=True),
        )

        onnx.checker.check_model(result, full_check=True)
        assert "weights" in {initializer.name for initializer in result.graph.initializer}

    def test_existing_hint_with_nested_capture_fails_without_mutating_input(self) -> None:
        from onnx import TensorProto

        config = SurgeryPipeConfig(trim_split_grouped_conv=True)
        model = SurgeryPipe().process(_make_grouped_conv_tail_slice_model(), config)
        hint = json.loads(
            next(
                item.value
                for item in model.metadata_props
                if item.key == "winml.quantization.region_hints"
            )
        )
        branch_name = hint["regions"][0]["branches"][0]
        branch_output = next(
            node.output[0] for node in model.graph.node if node.name == branch_name
        )
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(True, dtype=np.bool_), "condition")
        )

        def branch_graph(name: str) -> onnx.GraphProto:
            output_name = f"{name}_output"
            return onnx.helper.make_graph(
                [
                    onnx.helper.make_node(
                        "Identity",
                        [branch_output],
                        [output_name],
                        name=f"{name}_identity",
                    )
                ],
                name,
                [],
                [
                    onnx.helper.make_tensor_value_info(
                        output_name,
                        TensorProto.FLOAT,
                        [1, 16, 3],
                    )
                ],
            )

        model.graph.node.append(
            onnx.helper.make_node(
                "If",
                ["condition"],
                ["captured_branch_output"],
                name="capture_branch_output",
                then_branch=branch_graph("then_branch"),
                else_branch=branch_graph("else_branch"),
            )
        )
        model.graph.output.append(
            onnx.helper.make_tensor_value_info(
                "captured_branch_output",
                TensorProto.FLOAT,
                [1, 16, 3],
            )
        )
        onnx.checker.check_model(model, full_check=True)
        original = model.SerializeToString()

        with pytest.raises(QuantizationHintError, match="exclusively feed"):
            SurgeryPipe().process(model, config)

        assert model.SerializeToString() == original

    @pytest.mark.parametrize(
        "metadata_values",
        [
            ["not-json"],
            [
                json.dumps(
                    {
                        "version": 1,
                        "regions": [
                            {
                                "kind": "unknown",
                                "branches": ["old_branch0", "old_branch1"],
                                "concat": "old_concat",
                            }
                        ],
                    }
                )
            ],
            [
                json.dumps(
                    {
                        "version": 1,
                        "regions": [
                            {
                                "kind": "conv_concat",
                                "branches": ["missing_branch0", "missing_branch1"],
                                "concat": "missing_concat",
                            }
                        ],
                    }
                )
            ],
            [
                json.dumps(
                    {
                        "version": 1,
                        "regions": [
                            {
                                "kind": "conv_concat",
                                "branches": ["old_branch0", "old_branch1"],
                                "concat": "old_concat",
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "version": 1,
                        "regions": [
                            {
                                "kind": "conv_concat",
                                "branches": ["other_branch0", "other_branch1"],
                                "concat": "other_concat",
                            }
                        ],
                    }
                ),
            ],
        ],
    )
    def test_invalid_existing_hint_fails_without_mutating_input(
        self,
        metadata_values: list[str],
    ) -> None:
        model = _make_grouped_conv_tail_slice_model()
        for value in metadata_values:
            model.metadata_props.add(key="winml.quantization.region_hints", value=value)
        original = model.SerializeToString()

        with pytest.raises(QuantizationHintError):
            SurgeryPipe().process(
                model,
                SurgeryPipeConfig(trim_split_grouped_conv=True),
            )

        assert model.SerializeToString() == original
