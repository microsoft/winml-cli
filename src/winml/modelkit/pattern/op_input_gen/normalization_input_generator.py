# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Input generators for normalization ONNX operators.

Normalization operators share common patterns:
- Input shapes typically follow (N, C, ...) format for spatial data
- Common epsilon attribute for numerical stability
- Scale and bias parameters are often present

This module provides a base class with common shapes and specialized
subclasses for each normalization operator's specific requirements.
"""

import numpy as np

from ...onnx import SupportedONNXType
from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    InputValueConstraint,
    OpInputGenerator,
    QDQParameterConfig,
    register_runtime_checker_op,
)


class NormalizationInputGenerator(OpInputGenerator):
    """Base class for normalization operator input generators.

    Provides common input shapes for normalization operations:
    - 2D: (N, C) - Simple batch normalization
    - 3D: (N, C, L) - 1D spatial data (e.g., time series)
    - 4D: (N, C, H, W) - 2D spatial data (e.g., images)
    - 5D: (N, C, D, H, W) - 3D spatial data (e.g., videos)

    Common attributes:
    - epsilon: Small constant for numerical stability
    """

    def get_common_data_shapes(self) -> list[tuple[int, ...]]:
        """Return common input shapes for normalization operators.

        Returns shapes covering 2D through 5D tensors with varying
        batch sizes and channel counts.
        """
        return [
            # 1D (N,) - C = 1
            (4,),
            # 2D: (N, C) - Batch, Channels
            (3, 6),
            # 3D: (N, C, L) - Batch, Channels, Length
            (2, 4, 6),
            # 4D: (N, C, H, W) - Batch, Channels, Height, Width
            (2, 4, 5, 5),
            # 5D: (N, C, D, H, W) - Batch, Channels, Depth, Height, Width
            (2, 4, 3, 4, 4),
        ]

    def get_common_epsilon_values(self) -> list[float]:
        """Return common epsilon values for numerical stability.

        Standard values used in deep learning frameworks.
        """
        return [1e-4]

    def get_common_stash_types(self) -> list[int]:
        """Return common stash_type values for normalization operators.

        Standard values used in deep learning frameworks.
        """
        if self.qdq_generator is not None:
            # TODO currently we don't handle optional outputs, so this is unused
            return [1]  # FLOAT
        return [1, 10, 11]  # FLOAT, FLOAT16, DOUBLE

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for normalization operator testing.

        Args:
            properties: Base properties containing X_shape

        Returns:
            Updated properties with normalization-specific derived values (X_dim)
        """
        item = properties.copy()
        input_name = self.op_input_names[0]
        item[f"{input_name}_dim"] = len(item[f"{input_name}_shape"])
        if "attr_num_groups" in item:
            item["num_groups_gt_one"] = item["attr_num_groups"] > 1
        if "attr_axis" in item:
            item["axis_eq_one"] = item["attr_axis"] == 1
            item["axis_eq_zero"] = item["attr_axis"] == 0
            item["axis_is_last"] = item["attr_axis"] in (-1, item[f"{input_name}_dim"] - 1)
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values.

        Returns:
            List of property names that represent shapes with infinite possibilities
        """
        return (
            [f"{input_name}_value" for input_name in self.op_input_names]
            + [f"{input_name}_shape" for input_name in self.op_input_names]
            + ["attr_num_groups", "attr_axes", "attr_axis"]
        )


# ============================================================================
# BatchNormalization - Normalizes each channel across batch dimension
# ============================================================================


@register_runtime_checker_op
class BatchNormalizationInputGenerator(NormalizationInputGenerator):
    """Input generator for BatchNormalization operator.

    Signature: BatchNormalization(X, scale, bias, input_mean, input_var,
                                   *, epsilon=1e-5, momentum=0.9, training_mode=0)

    BatchNorm normalizes each channel independently across the batch dimension.
    Requires pre-computed mean and variance (or calculates them during training).

    Inputs:
    - X: Input tensor (N, C, ...)
    - scale: Scale parameter per channel (C,)
    - bias: Bias parameter per channel (C,)
    - input_mean: Running mean per channel (C,)
    - input_var: Running variance per channel (C,)

    Attributes:
    - epsilon: Small constant for numerical stability
    - momentum: Momentum for running mean/variance
    - training_mode: 0 for inference, 1 for training
    """

    op_name = "BatchNormalization"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return finite attribute values for BatchNormalization.

        training_mode is the primary varying attribute.
        epsilon is also tested with common values.
        """
        attrs = {
            "epsilon": self.get_common_epsilon_values(),
            "momentum": [0.9],
        }
        # opset<14 does not have training_mode
        if "training_mode" in self.op_attribute_names:
            attrs["training_mode"] = [0]  # 0: inference, 1: training
        return attrs

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for BatchNormalization.

        For each data shape:
        - X has the full shape (N, C, ...)
        - scale, bias, mean, var all have shape (C,)
        """
        combinations = []

        # BatchNormalization uses mean/var in older opsets and input_mean/input_var in newer opsets.
        x_name, scale_name, bias_name, mean_name, var_name = self.op_input_names[:5]

        for shape in self.get_common_data_shapes():
            num_channels = shape[1] if len(shape) > 1 else 1  # C dimension

            # Create scale, bias, mean, var tensors with shape (C,)
            scale = InputValueConstraint(np.ones((num_channels,), dtype=np.float32))
            bias = InputValueConstraint(np.zeros((num_channels,), dtype=np.float32))
            mean = InputValueConstraint(np.zeros((num_channels,), dtype=np.float32))
            var = InputValueConstraint(np.ones((num_channels,), dtype=np.float32))

            combinations.append(
                {
                    x_name: InputShapeConstraint(shape),
                    scale_name: scale,
                    bias_name: bias,
                    mean_name: mean,
                    var_name: var,
                }
            )

        return combinations


# ============================================================================
# GroupNormalization - Divides channels into groups and normalizes within groups
# ============================================================================


@register_runtime_checker_op
class GroupNormalizationInputGenerator(NormalizationInputGenerator):
    """Input generator for GroupNormalization operator.

    Signature: GroupNormalization(X, scale, bias, *, num_groups, epsilon=1e-5)

    GroupNorm divides channels into groups and normalizes within each group.
    Useful alternative to BatchNorm that doesn't depend on batch size.

    Inputs:
    - X: Input tensor (N, C, ...)
    - scale: Scale parameter per channel (C,)
    - bias: Bias parameter per channel (C,)

    Attributes:
    - num_groups: Number of groups to divide channels into (C must be divisible)
    - epsilon: Small constant for numerical stability
    """

    op_name = "GroupNormalization"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return finite attribute values for GroupNormalization.

        epsilon is tested with common values.
        num_groups is handled per-shape in input combinations.
        """
        return {
            "epsilon": self.get_common_epsilon_values(),
            "stash_type": self.get_common_stash_types(),
        }

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for GroupNormalization.

        For each data shape, test multiple num_groups values where
        C is divisible by num_groups (e.g., 1, 2, C).
        """
        combinations = []

        for shape in self.get_common_data_shapes():
            if len(shape) <= 1:  # group norm not working for 1D input
                continue
            num_channels = shape[1]  # C dimension

            # Create scale and bias tensors with shape (C,)
            scale = InputValueConstraint(np.ones((num_channels,), dtype=np.float32))
            bias = InputValueConstraint(np.zeros((num_channels,), dtype=np.float32))

            # Test different num_groups values (must divide C evenly)
            # Common choices: 1 (LayerNorm-like), num_channels (InstanceNorm-like),
            # or intermediate values
            valid_num_groups = [1, num_channels]
            # Add middle divisors if C > 2
            if num_channels > 2 and num_channels % 2 == 0:
                valid_num_groups.append(2)

            combinations.extend(
                [
                    {
                        "X": InputShapeConstraint(shape),
                        "scale": scale,
                        "bias": bias,
                        "num_groups": num_groups,
                    }
                    for num_groups in valid_num_groups
                ]
            )

        return combinations


# ============================================================================
# InstanceNormalization - Normalizes each channel independently per instance
# ============================================================================


@register_runtime_checker_op
class InstanceNormalizationInputGenerator(NormalizationInputGenerator):
    """Input generator for InstanceNormalization operator.

    Signature: InstanceNormalization(X, scale, bias, *, epsilon=1e-5)

    InstanceNorm normalizes each channel independently for each instance (sample).
    Commonly used in style transfer and GANs.

    Inputs:
    - X: Input tensor (N, C, ...)
    - scale: Scale parameter per channel (C,)
    - bias: Bias parameter per channel (C,)

    Attributes:
    - epsilon: Small constant for numerical stability
    """

    op_name = "InstanceNormalization"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return finite attribute values for InstanceNormalization."""
        return {
            "epsilon": self.get_common_epsilon_values(),
        }

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for InstanceNormalization.

        InstanceNorm requires at least 3D tensors (N, C, spatial_dims).
        """
        combinations = []

        for shape in self.get_common_data_shapes():
            # Skip 2D shapes - InstanceNorm requires spatial dimensions
            if len(shape) < 3:
                continue

            num_channels = shape[1]  # C dimension

            # Create scale and bias tensors with shape (C,)
            scale = InputValueConstraint(np.ones((num_channels,), dtype=np.float32))
            bias = InputValueConstraint(np.zeros((num_channels,), dtype=np.float32))

            combinations.append(
                {
                    "input": InputShapeConstraint(shape),
                    "scale": scale,
                    "B": bias,  # Parameter name is 'B' in ONNX spec
                }
            )

        return combinations

    def get_qdq_config(self) -> dict[str, QDQParameterConfig] | None:
        """Return QDQ configuration for BatchNormalization operator inputs."""
        return {
            self.op_input_names[0]: QDQParameterConfig(support_activation=True),
            self.op_input_names[1]: QDQParameterConfig(support_weight=True),
            self.op_input_names[2]: QDQParameterConfig(qdq_types=[SupportedONNXType.INT32]),
        }


# ============================================================================
# LayerNormalization - Normalizes across specified axes
# ============================================================================


@register_runtime_checker_op
class LayerNormalizationInputGenerator(NormalizationInputGenerator):
    """Input generator for LayerNormalization operator.

    Signature: LayerNormalization(X, Scale, B, *, axis=-1, epsilon=1e-5)

    LayerNorm normalizes across specified axes, typically the last dimension(s).
    Widely used in transformers and NLP models.

    Inputs:
    - X: Input tensor (any shape)
    - Scale: Scale parameter (shape matches normalized dimensions)
    - B: Bias parameter (optional, shape matches normalized dimensions)

    Attributes:
    - axis: Axis to start normalization from (default: -1)
    - epsilon: Small constant for numerical stability
    """

    op_name = "LayerNormalization"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return finite attribute values for LayerNormalization.

        axis is handled per-shape in input combinations.
        """
        return {
            "epsilon": self.get_common_epsilon_values(),
            "stash_type": self.get_common_stash_types(),
        }

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for LayerNormalization.

        Test different axis values: -1 (last dim), -2 (last 2 dims), etc.
        Scale and B shapes must match the normalized dimensions (from axis to end).
        """
        combinations = []
        axis_options = [-1, 0]

        for shape in self.get_common_data_shapes():
            for axis in axis_options:
                # Compute normalized_shape based on axis
                # axis defines the first dimension to normalize over
                # Scale and B shapes match dimensions from axis to the end
                rank = len(shape)
                normalized_axis = axis if axis >= 0 else rank + axis
                normalized_shape = shape[normalized_axis:]

                scale = InputValueConstraint(np.ones(normalized_shape, dtype=np.float32))
                bias = InputValueConstraint(np.zeros(normalized_shape, dtype=np.float32))

                combinations.append(
                    {
                        "X": InputShapeConstraint(shape),
                        "Scale": scale,  # Note: uppercase S
                        "B": bias,
                        "axis": axis,
                    }
                )

        return combinations

    def get_qdq_config(self) -> dict[str, QDQParameterConfig] | None:
        """Return QDQ configuration for LayerNormalization operator inputs."""
        # Follow https://github.com/microsoft/onnxruntime/blob/727db0d3dc9f7dc5958891d80c1073ef7190f316/onnxruntime/python/tools/quantization/operators/norm.py
        return {
            "X": QDQParameterConfig(support_activation=True),
            "Scale": QDQParameterConfig(support_activation=True, support_weight=True),
            "B": QDQParameterConfig(qdq_types=[SupportedONNXType.INT32]),
        }


# ============================================================================
# LpNormalization - NOT IMPLEMENTED in ONNXRuntime
# ============================================================================


@register_runtime_checker_op
class LpNormalizationInputGenerator(NormalizationInputGenerator):
    """Input generator for LpNormalization operator."""

    op_name = "LpNormalization"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return finite attribute values for LpNormalization."""
        return {"p": [1, 2]}

    def get_input_and_infinite_attribute_combinations(self) -> list[dict[str, InputConstraint]]:
        """Return input combinations for LpNormalization."""
        combinations = []
        for shape in self.get_common_data_shapes():
            if len(shape) < 3:
                continue
            combinations.extend(
                {"input": InputShapeConstraint(shape), "axis": axis} for axis in [0, 1, -1, 2]
            )
        return combinations

    def get_qdq_config(self) -> dict[str, QDQParameterConfig] | None:
        """Return QDQ configuration for LpNormalization operator inputs."""
        return {
            self.op_input_names[0]: QDQParameterConfig(support_activation=True),
        }


# ============================================================================
# MeanVarianceNormalization - Normalizes using mean and variance
# ============================================================================


@register_runtime_checker_op
class MeanVarianceNormalizationInputGenerator(NormalizationInputGenerator):
    """Input generator for MeanVarianceNormalization operator.

    Signature: MeanVarianceNormalization(X, *, axes=[0, 2, 3])

    MVN performs mean-variance normalization across specified axes.
    Commonly used for data preprocessing.

    Inputs:
    - X: Input tensor (any shape)

    Attributes:
    - axes: Axes over which to compute mean and variance
    """

    op_name = "MeanVarianceNormalization"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return finite attribute values for MeanVarianceNormalization.

        axes is handled per-shape in input combinations to ensure validity.
        """
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for MeanVarianceNormalization.

        Test different axes combinations based on tensor rank.
        """
        combinations = [
            {
                "X": InputShapeConstraint(shape),
                "axes": [axis],
            }
            for shape in self.get_common_data_shapes()
            for axis in range(-1, min(len(shape), 2))
        ]

        return combinations  # noqa: RET504


# ============================================================================
# RMSNormalization - Root Mean Square normalization (opset 23+)
# ============================================================================


@register_runtime_checker_op
class RMSNormalizationInputGenerator(NormalizationInputGenerator):
    """Input generator for RMSNormalization operator.

    Signature: RMSNormalization(X, scale, *, axis=-1, epsilon=1e-5, stash_type=1)

    RMSNorm normalizes using root mean square across specified axes,
    without centering (no mean subtraction). Commonly used in LLM architectures.

    Inputs:
    - X: Input tensor (any shape)
    - scale: Scale parameter (shape broadcastable to normalized dimensions)

    Attributes:
    - axis: First normalization dimension (default: -1)
    - epsilon: Small constant for numerical stability
    - stash_type: Floating-point precision for internal computation
    """

    op_name = "RMSNormalization"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return finite attribute values for RMSNormalization.

        axis is handled per-shape in input combinations.
        """
        return {
            "epsilon": self.get_common_epsilon_values(),
            "stash_type": self.get_common_stash_types(),
        }

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for RMSNormalization.

        Test different axis values: -1 (last dim), 0 (all dims).
        Scale shape must match the normalized dimensions (from axis to end).
        """
        combinations = []
        axis_options = [-1, 0]

        for shape in self.get_common_data_shapes():
            for axis in axis_options:
                # Compute normalized_shape based on axis
                rank = len(shape)
                normalized_axis = axis if axis >= 0 else rank + axis
                normalized_shape = shape[normalized_axis:]

                scale = InputValueConstraint(np.ones(normalized_shape, dtype=np.float32))

                combinations.append(
                    {
                        "X": InputShapeConstraint(shape),
                        "scale": scale,
                        "axis": axis,
                    }
                )

        return combinations
