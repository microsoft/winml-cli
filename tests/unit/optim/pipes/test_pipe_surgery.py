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

import numpy as np
import onnx
import pytest
from onnx import numpy_helper

from winml.modelkit.optim.pipes import (
    SURGERY_CAPABILITIES,
    SurgeryPipe,
    SurgeryPipeConfig,
)


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
        result = SurgeryPipe().process(
            model, SurgeryPipeConfig(untie_constant_batched_matmul=True)
        )

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
        result = SurgeryPipe().process(
            model, SurgeryPipeConfig(untie_constant_batched_matmul=True)
        )
        assert not any(n.op_type == "Add" for n in result.graph.node)

    def test_constant_on_lhs_is_handled(self) -> None:
        """A constant rank-3 operand on the LHS is untied too."""
        model = _make_batched_const_matmul_model(const_on_rhs=False)
        result = SurgeryPipe().process(
            model, SurgeryPipeConfig(untie_constant_batched_matmul=True)
        )
        assert any(n.op_type == "Add" for n in result.graph.node)
