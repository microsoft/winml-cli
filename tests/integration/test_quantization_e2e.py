"""End-to-end integration tests for quantization functionality."""

import subprocess

import onnx
import pytest


class TestQuantizationE2E:
    """End-to-end tests for model quantization workflow."""

    @pytest.fixture(scope="class")
    def test_model_path(self, tmp_path_factory):
        """Export ResNet-50 model for testing."""
        temp_dir = tmp_path_factory.mktemp("quantization_e2e")
        model_path = temp_dir / "resnet-50.onnx"

        # Export ResNet-50 model using wmk export
        cmd = [
            "wmk",
            "export",
            "-m",
            "microsoft/resnet-50",
            "-o",
            str(model_path),
            "--no-hierarchy",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(temp_dir))
        if result.returncode != 0:
            pytest.skip(f"Failed to export model: {result.stderr}")

        return model_path

    def _run_quantization_and_validate(
        self, test_model_path, tmp_path, test_name, precision=None
    ):
        """Helper method to run quantization and perform comprehensive validation."""
        pytest.importorskip("onnxruntime", reason="ONNXRuntime required for inference test")
        pytest.importorskip("numpy", reason="NumPy required for inference test")

        import numpy as np
        import onnxruntime as ort

        output_path = tmp_path / f"resnet50_{test_name}_quantized.onnx"

        # Build quantization command using actual wmk quantize CLI flags
        cmd = ["wmk", "quantize", "--model", str(test_model_path), "--output", str(output_path)]

        if precision:
            cmd.extend(["--precision", precision])

        # Run quantization
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Quantization failed: {result.stderr}"

        # Verify output file exists
        assert output_path.exists(), "Quantized model file not created"
        assert output_path.stat().st_size > 0, "Quantized model file is empty"

        # Get model sizes for comparison
        original_size = test_model_path.stat().st_size
        quantized_size = output_path.stat().st_size
        size_reduction = (original_size - quantized_size) / original_size * 100

        print(f"Original: {original_size/1024/1024:.2f} MB")
        print(f"Quantized: {quantized_size/1024/1024:.2f} MB")
        print(f"Size change: {size_reduction:.1f}%")

        # Validate Q/DQ nodes in quantized model
        model = onnx.load(str(output_path))
        node_types = {node.op_type for node in model.graph.node}

        assert "QuantizeLinear" in node_types, "No QuantizeLinear nodes found in quantized model"
        assert (
            "DequantizeLinear" in node_types
        ), "No DequantizeLinear nodes found in quantized model"

        # Count Q/DQ nodes for verification
        q_nodes = sum(1 for node in model.graph.node if node.op_type == "QuantizeLinear")
        dq_nodes = sum(1 for node in model.graph.node if node.op_type == "DequantizeLinear")
        print(f"✓ Model contains {q_nodes} QuantizeLinear and {dq_nodes} DequantizeLinear nodes")

        # Validate the quantized model is not empty
        # Note: QDQ quantization can make models LARGER due to added Q/DQ nodes,
        # especially for small models. Size reduction is not guaranteed.
        assert quantized_size > 0, "Quantized model is empty"

        # Run inference test with random input
        original_session = ort.InferenceSession(str(test_model_path))
        quantized_session = ort.InferenceSession(str(output_path))

        # Get input shape from original model
        input_info = original_session.get_inputs()[0]
        input_shape = input_info.shape

        # Create synthetic input data
        # Handle dynamic dimensions (replace None with 1)
        concrete_shape = [1 if dim is None else dim for dim in input_shape]
        input_data = np.random.randn(*concrete_shape).astype(np.float32)
        input_dict = {input_info.name: input_data}

        # Run inference on both models
        original_output = original_session.run(None, input_dict)
        quantized_output = quantized_session.run(None, input_dict)

        # Verify outputs have same structure
        assert len(original_output) == len(quantized_output), "Output count mismatch"

        # Verify output shapes match
        for orig, quant in zip(original_output, quantized_output, strict=False):
            assert (
                orig.shape == quant.shape
            ), f"Output shape mismatch: {orig.shape} vs {quant.shape}"

        print(
            f"✓ Inference successful - Original: {original_output[0].shape}, Quantized: {quantized_output[0].shape}"
        )

        return output_path

    def test_quantization_default_config_default_ep(self, test_model_path, tmp_path):
        """Test quantization with default configuration and default execution provider."""
        self._run_quantization_and_validate(test_model_path, tmp_path, "default_config_default_ep")

    def test_quantization_int8_precision(self, test_model_path, tmp_path):
        """Test quantization with int8 precision shorthand."""
        self._run_quantization_and_validate(
            test_model_path, tmp_path, "int8_precision", precision="int8"
        )

    def test_quantization_int16_precision(self, test_model_path, tmp_path):
        """Test quantization with int16 precision shorthand."""
        self._run_quantization_and_validate(
            test_model_path, tmp_path, "int16_precision", precision="int16"
        )
