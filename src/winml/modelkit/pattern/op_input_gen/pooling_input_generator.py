"""Input generators for pooling ONNX operators.

This includes MaxPool, AveragePool, and LpPool.
"""

import numpy as np

from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    OpInputGenerator,
    QDQParameterConfig,
    register_runtime_checker_op,
)


class PoolingInputGenerator(OpInputGenerator):
    """Base class for pooling operator input generators."""

    def get_common_shapes_and_kernels(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Return common (input_shape, kernel_shape) pairs.

        Input shape: (N, C, D1, D2, ...)
        Kernel shape: (k1, k2, ...) matching spatial dims.
        """
        return [
            # 1D: (N, C, L), kernel (3,)
            ((2, 2, 10), (3,)),
            # 2D: (N, C, H, W), kernel (3, 3)
            ((2, 2, 10, 10), (3, 3)),
            # 3D: (N, C, D, H, W), kernel (2, 2, 2)
            ((2, 2, 8, 8, 8), (2, 2, 2)),
        ]

    def derive_properties(self, properties: dict) -> dict:
        item = properties.copy()
        input_name = self.op_input_names[0]
        item[f"{input_name}_dim"] = len(item[f"{input_name}_shape"])
        return item

    def get_infinite_property_names(self) -> list[str]:
        return (
            [f"{input_name}_value" for input_name in self.op_input_names]
            + [f"{input_name}_shape" for input_name in self.op_input_names]
            + ["attr_kernel_shape", "attr_strides", "attr_pads", "attr_dilations"]
        )


@register_runtime_checker_op
class MaxPoolInputGenerator(PoolingInputGenerator):
    """Input generator for MaxPool operator."""

    op_name = "MaxPool"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        return {
            "auto_pad": ["NOTSET", "SAME_UPPER"],
            "ceil_mode": [0, 1],
            "storage_order": [0, 1],
        }

    def get_input_and_infinite_attribute_combinations(self) -> list[dict[str, InputConstraint]]:
        combinations = []
        for x_shape, kernel_shape in self.get_common_shapes_and_kernels():
            spatial_dims = len(x_shape) - 2

            # Basic combo with default strides/pads/dilations
            combinations.append(
                {
                    "X": InputShapeConstraint(x_shape),
                    "kernel_shape": kernel_shape,
                    "strides": [1] * spatial_dims,
                    "dilations": [1] * spatial_dims,
                    "pads": [0] * (2 * spatial_dims),
                }
            )

            if self.qdq_generator:
                continue

            # Strides > 1
            combinations.append(
                {
                    "X": InputShapeConstraint(x_shape),
                    "kernel_shape": kernel_shape,
                    "strides": [2] * spatial_dims,
                    "dilations": [1] * spatial_dims,
                    "pads": [0] * (2 * spatial_dims),
                }
            )

            combinations.append(
                {
                    "X": InputShapeConstraint(x_shape),
                    "kernel_shape": kernel_shape,
                    "strides": [2] * spatial_dims,
                    "dilations": [1] * spatial_dims,
                    "pads": [1] * (2 * spatial_dims),
                }
            )

            # Dilations > 1
            combinations.append(
                {
                    "X": InputShapeConstraint(x_shape),
                    "kernel_shape": kernel_shape,
                    "strides": [1] * spatial_dims,
                    "dilations": [2] * spatial_dims,
                    "pads": [0] * (2 * spatial_dims),
                }
            )

            # Pads (valid explicit pads)
            # Note: pads are ignored if auto_pad is not NOTSET, but we provide them anyway
            combinations.append(
                {
                    "X": InputShapeConstraint(x_shape),
                    "kernel_shape": kernel_shape,
                    "strides": [1] * spatial_dims,
                    "dilations": [1] * spatial_dims,
                    "pads": [1] * (2 * spatial_dims),
                }
            )

        return combinations

    def derive_properties(self, properties: dict) -> dict:
        item = super().derive_properties(properties)
        if self.qdq_generator:
            return item

        if "attr_dilations" in item:
            dilations_array = np.array(item["attr_dilations"])
            item["dilations_all_ones"] = bool(np.all(dilations_array == 1))
        else:
            item["dilations_all_ones"] = True  # Default is all ones

        if "attr_strides" in item:
            strides_array = np.array(item["attr_strides"])
            item["strides_all_ones"] = bool(np.all(strides_array == 1))
        else:
            item["strides_all_ones"] = True  # Default is all ones

        if "attr_pads" in item:
            pads_array = np.array(item["attr_pads"])
            item["attr_pads_all_zeros"] = bool(np.all(pads_array == 0))
        else:
            item["attr_pads_all_zeros"] = True  # Default is all zeros

        return item

    def get_qdq_config(self):
        return {
            "X": QDQParameterConfig(support_activation=True)
        }


@register_runtime_checker_op
class AveragePoolInputGenerator(PoolingInputGenerator):
    """Input generator for AveragePool operator."""

    op_name = "AveragePool"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        return {
            "auto_pad": ["NOTSET", "SAME_UPPER"],
            "ceil_mode": [0, 1],
            "count_include_pad": [0, 1],
        }

    def derive_properties(self, properties: dict) -> dict:
        item = super().derive_properties(properties)

        strides = item.get("attr_strides")
        if strides is not None:
            item["has_stride_gt1"] = any(s > 1 for s in strides)

        pads = item.get("attr_pads")
        if pads is not None:
            item["has_padding"] = any(p != 0 for p in pads)

        return item

    def get_input_and_infinite_attribute_combinations(self) -> list[dict[str, InputConstraint]]:
        combinations = []
        for x_shape, kernel_shape in self.get_common_shapes_and_kernels():
            spatial_dims = len(x_shape) - 2

            # Basic combo (strides=1, pads=0)
            combinations.append(
                {
                    "X": InputShapeConstraint(x_shape),
                    "kernel_shape": kernel_shape,
                    "strides": [1] * spatial_dims,
                    "pads": [0] * (2 * spatial_dims),
                }
            )

            # Strides > 1
            combinations.append(
                {
                    "X": InputShapeConstraint(x_shape),
                    "kernel_shape": kernel_shape,
                    "strides": [2] * spatial_dims,
                    "pads": [0] * (2 * spatial_dims),
                }
            )

            # Pads > 0
            combinations.append(
                {
                    "X": InputShapeConstraint(x_shape),
                    "kernel_shape": kernel_shape,
                    "strides": [1] * spatial_dims,
                    "pads": [1] * (2 * spatial_dims),
                }
            )

        return combinations


@register_runtime_checker_op
class LpPoolInputGenerator(PoolingInputGenerator):
    """Input generator for LpPool operator."""

    op_name = "LpPool"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        return {
            "auto_pad": ["NOTSET", "SAME_UPPER"],
            "p": [1, 2],  # L1 and L2 norm
        }

    def get_input_and_infinite_attribute_combinations(self) -> list[dict[str, InputConstraint]]:
        combinations = []
        for x_shape, kernel_shape in self.get_common_shapes_and_kernels():
            spatial_dims = len(x_shape) - 2

            # Basic combo
            combinations.append(
                {
                    "X": InputShapeConstraint(x_shape),
                    "kernel_shape": kernel_shape,
                    "strides": [1] * spatial_dims,
                    "pads": [0] * (2 * spatial_dims),
                }
            )

            # Strides > 1
            combinations.append(
                {
                    "X": InputShapeConstraint(x_shape),
                    "kernel_shape": kernel_shape,
                    "strides": [2] * spatial_dims,
                    "pads": [0] * (2 * spatial_dims),
                }
            )

            # Pads > 0
            combinations.append(
                {
                    "X": InputShapeConstraint(x_shape),
                    "kernel_shape": kernel_shape,
                    "strides": [1] * spatial_dims,
                    "pads": [1] * (2 * spatial_dims),
                }
            )

        return combinations
