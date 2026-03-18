"""Input generators for Resize ONNX operator.

Resize resizes the input tensor. It supports different interpolation modes
(nearest, linear, cubic) and coordinate transformation modes.
"""

import numpy as np

from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    InputValueConstraint,
    OpInputGenerator,
    register_runtime_checker_op,
)


@register_runtime_checker_op
class ResizeInputGenerator(OpInputGenerator):
    """Input generator for Resize operator.

    Resize signature:
    - Inputs: X, roi (optional), scales (optional), sizes (optional)
    - CRITICAL: One of 'scales' and 'sizes' MUST be specified
    - Attributes: antialias, axes, coordinate_transformation_mode, cubic_coeff_a,
                  exclude_outside, extrapolation_value, keep_aspect_ratio_policy,
                  mode, nearest_mode
    """

    op_name = "Resize"
    expand_optionals = False
    # float values in scales is used
    replace_float_with_dummy_in_query: bool = False

    def get_common_input_shapes(self) -> list[tuple[int, ...]]:
        """Return common input shapes for resize testing.

        ONNX Resize supports N-D tensors. Testing from 1D to 5D.
        """
        return [
            # (10,),  # 1D invalid combination for Resize in ONNX
            # TODO comment: test larger input size for scale_up/scale_down)
            (4, 8),  # 2D
            (2, 4, 8),  # 3D: (N, C, L)
            (2, 3, 8, 8),  # 4D: (N, C, H, W) - typical image
            (2, 3, 4, 8, 8),  # 5D: (N, C, D, H, W) - typical video/3D
        ]

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return finite attribute combinations for Resize."""
        return {"antialias": [0, 1]}

    def get_input_and_infinite_attribute_combinations(self) -> list[dict[str, InputConstraint]]:
        """Return input combinations for Resize operator.

        CRITICAL: Always provide explicit values for all inputs.
        - roi: Empty tensor when not using tf_crop_and_resize mode
        - scales/sizes: One must be specified, other set to empty
        """
        combinations = []

        for x_shape in self.get_common_input_shapes():
            ndim = len(x_shape)

            # Test with scales (upsample by 2x on last dimension)
            scales_up = np.ones(ndim, dtype=np.float32)
            scales_up[-1] = 2.0
            if ndim >= 4:
                scales_up[-2] = 2.0  # Also scale H dimension for 4D+

            # Test with scales (downsample by 0.5x on last dimension)
            scales_down = np.ones(ndim, dtype=np.float32)
            scales_down[-1] = 0.5
            if ndim >= 4:
                scales_down[-2] = 0.5  # Also scale H dimension for 4D+

            # Combination using scales (upsample) - empty sizes
            combinations.append(
                {
                    "coordinate_transformation_mode": "asymmetric",
                    "cubic_coeff_a": -0.75,
                    "mode": "nearest",
                    "nearest_mode": "floor",
                    "X": InputShapeConstraint(x_shape),
                    # Explicit roi to avoid empty-optional rejection on older schemas
                    "roi": InputValueConstraint(np.zeros(2 * ndim, dtype=np.float32)),
                    "scales": InputValueConstraint(scales_up),
                    "extrapolation_value": 0.0,
                    "axes": list(range(ndim)),  # All axes
                }
            )

            # Combination using scales (downsample) - empty sizes
            combinations.append(
                {
                    "coordinate_transformation_mode": "asymmetric",
                    "cubic_coeff_a": -0.75,
                    "mode": "nearest",
                    "nearest_mode": "floor",
                    "X": InputShapeConstraint(x_shape),
                    "roi": InputValueConstraint(np.zeros(2 * ndim, dtype=np.float32)),
                    "scales": InputValueConstraint(scales_down),
                    "extrapolation_value": 0.0,
                    "axes": list(range(ndim)),
                }
            )

            # Combination using sizes (explicit target size) - empty scales
            # Double the spatial dimensions
            target_sizes = list(x_shape)
            if ndim >= 3:
                target_sizes[-1] = x_shape[-1] * 2
                if ndim >= 4:
                    target_sizes[-2] = x_shape[-2] * 2
            else:
                target_sizes[-1] = x_shape[-1] * 2

            combinations.append(
                {
                    "coordinate_transformation_mode": "asymmetric",
                    "cubic_coeff_a": -0.75,
                    "mode": "nearest",
                    "nearest_mode": "floor",
                    "X": InputShapeConstraint(x_shape),
                    "roi": InputValueConstraint(np.zeros(2 * ndim, dtype=np.float32)),
                    "sizes": InputValueConstraint(np.array(target_sizes, dtype=np.int64)),
                    "extrapolation_value": 0.0,
                    "axes": list(range(ndim)),
                }
            )

            # Combination using sizes (halve spatial dimensions) - empty scales
            target_sizes_half = list(x_shape)
            if ndim >= 3:
                target_sizes_half[-1] = max(1, x_shape[-1] // 2)
                if ndim >= 4:
                    target_sizes_half[-2] = max(1, x_shape[-2] // 2)
            else:
                target_sizes_half[-1] = max(1, x_shape[-1] // 2)

            combinations.append(
                {
                    "coordinate_transformation_mode": "half_pixel",
                    "cubic_coeff_a": -0.75,
                    "mode": "linear",
                    "nearest_mode": "floor",
                    "X": InputShapeConstraint(x_shape),
                    "roi": InputValueConstraint(np.zeros(2 * ndim, dtype=np.float32)),
                    "sizes": InputValueConstraint(np.array(target_sizes_half, dtype=np.int64)),
                    "extrapolation_value": 0.0,
                    "axes": list(range(ndim)),
                }
            )

        return combinations

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for Resize operator testing.

        Args:
            properties: Base properties containing X_shape, scales, sizes, etc.

        Returns:
            Updated properties with resize-specific derived values
        """
        item = properties.copy()
        input_name = self.op_input_names[0]  # X

        # Derive X dimension
        item[f"{input_name}_dim"] = len(item[f"{input_name}_shape"])

        # Derive whether using scales or sizes
        scales_value = item.get("scales_value")
        sizes_value = item.get("sizes_value")

        if scales_value is not None:
            item["uses_scales"] = len(scales_value) > 0
            item["scales_up"] = all(s >= 1.0 for s in scales_value)
            item["scales_down"] = all(s <= 1.0 for s in scales_value)
        else:
            item["uses_scales"] = False
            item["scales_up"] = False
            item["scales_down"] = False

        if sizes_value is not None:
            item["uses_sizes"] = len(sizes_value) > 0
        else:
            item["uses_sizes"] = False

        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values.

        Returns:
            List of property names that represent shapes/values with infinite possibilities
        """
        return (
            [f"{input_name}_value" for input_name in self.op_input_names]
            + [f"{input_name}_shape" for input_name in self.op_input_names]
            + ["attr_cubic_coeff_a", "attr_extrapolation_value", "attr_axes"]
        )
