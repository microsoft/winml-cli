# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Input generators for convolution ONNX operators.

This module provides input generators for convolution operators:
- Conv: Standard convolution operator
- ConvTranspose: Transposed convolution (deconvolution) operator

Convolution operators perform spatial convolution operations on tensors,
commonly used in computer vision and signal processing applications.
"""

import itertools

import numpy as np

import winml.modelkit.onnx.dtypes as dtypes

from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    OpInputGenerator,
    QDQParameterConfig,
    register_runtime_checker_op,
)


class ConvInputGenerator(OpInputGenerator):
    """Base class for convolution operator input generators.

    Provides common shapes and patterns for convolution operations.
    """

    def derive_properties(self, properties: dict) -> dict:
        """Derive common properties for Conv/ConvTranspose operator testing.

        Args:
            properties: Base properties containing X_shape, W_shape, attrs, etc.

        Returns:
            Updated properties with derived values (X_dim, W_dim, dilations/strides flags)
        """
        item = properties.copy()
        item["X_dim"] = len(item["X_shape"])
        item["W_dim"] = len(item["W_shape"])

        # Check if dilations are uniform (all values are 1)
        if "attr_dilations" in item and self.qdq_generator is None:
            dilations_array = np.array(item["attr_dilations"])
            item["dilations_all_ones"] = bool(np.all(dilations_array == 1))
            item["is_dilations_uniform"] = bool(np.all(dilations_array == dilations_array[0]))

        # Check if strides are uniform (all values are 1)
        if "attr_strides" in item and self.qdq_generator is None:
            strides_array = np.array(item["attr_strides"])
            item["strides_all_ones"] = bool(np.all(strides_array == 1))
            item["is_strides_uniform"] = bool(np.all(strides_array == strides_array[0]))

        # Check if pads are all zeros
        if "attr_pads" in item and self.qdq_generator is None:
            pads_array = np.array(item["attr_pads"])
            item["pads_all_zeros"] = bool(np.all(pads_array == 0))

        item["attr_group_is_one"] = item.get("attr_group", 1) == 1

        return item

    def get_base_conv_shapes(self) -> list[tuple[tuple[int, ...], int, tuple[int, ...]]]:
        """Return base shapes (X_shape, Out_Channels, Kernel_Shape).

        X_shape: (N, C, D1, ...) - C must be divisible by 3 for group=3 testing.
        """
        return [
            # 1D: (N, C, L)
            ((2, 6, 10), 6, (3,)),
            # 2D: (N, C, H, W)
            ((2, 6, 10, 10), 6, (3, 3)),
            # 3D: (N, C, D, H, W)
            ((2, 6, 8, 8, 8), 6, (3, 3, 3)),
            # 4D: (N, C, D1, D2, D3, D4) invalid input
            # ((2, 6, 6, 6, 6, 6), 6, (3, 3, 3, 3), True),
        ]

    def get_attr_options(self, spatial_dims: int) -> dict[str, list]:
        """Return options for attributes based on spatial dimensions."""
        # dilations: {all 1s, range(1, n+1)}
        dilations_opts = [
            [1] * spatial_dims,
            # [2] * spatial_dims # Rare case, will enable later if needed
            list(range(1, spatial_dims + 1)),
        ]

        # strides: {all 1s, 2s, range(1, n+1)}
        strides_opts = [
            [1] * spatial_dims,
            [2] * spatial_dims,
            list(range(1, spatial_dims + 1)),
        ]

        # pads: {all 0s, range(n)}
        # pads length is 2 * spatial_dims. Using range(1, n+1) repeated.
        pads_opts = [
            [0] * (2 * spatial_dims),
            list(range(1, spatial_dims + 1)) * 2,
        ]

        # group: {1, 3}
        group_opts = [1, 3]

        if self.qdq_generator:
            # TODO do not expand these options when QDQ is enabled to reduce combinations
            dilations_opts = dilations_opts[:1]  # only all 1s
            strides_opts = strides_opts[:1]  # only all 1s
            pads_opts = pads_opts[:1]  # only all 0s

        # auto_pad
        auto_pad_opts = ["NOTSET", "SAME_LOWER", "SAME_UPPER", "VALID"]

        return {
            "dilations": dilations_opts,
            "strides": strides_opts,
            "pads": pads_opts,
            "group": group_opts,
            "auto_pad": auto_pad_opts,
        }

    def get_qdq_config(self):
        """Return QDQ configuration for Conv operator inputs."""
        # https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/operators/conv.py
        return {
            "X": QDQParameterConfig(support_activation=True),
            "W": QDQParameterConfig(support_weight=True),
            "B": QDQParameterConfig(qdq_types=[dtypes.SupportedONNXType.INT32]),
        }


@register_runtime_checker_op
class ConvOpInputGenerator(ConvInputGenerator):
    """Input generator for Conv operator."""

    op_name = "Conv"
    expand_optionals = False

    def _get_optional_combinations(self, kernel_shape: tuple) -> list[dict]:
        """Generate optional attribute combinations for Conv.

        Per ONNX spec, only `kernel_shape` has no default value
        (but can be inferred from W).

        Test scenarios:
        1. Without kernel_shape: runtime infers from W
        2. With kernel_shape: explicitly specify

        Args:
            kernel_shape: Kernel spatial shape

        Returns:
            List of dicts with different optional attribute combinations
        """
        return [
            {},  # Case 1: No kernel_shape - runtime infers from W
            {"kernel_shape": kernel_shape},  # Case 2: Explicit kernel_shape
        ]

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return finite attribute sets for Conv operator."""
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input and infinite attribute combinations for Conv operator."""
        combinations = []
        for x_shape, m, k_shape in self.get_base_conv_shapes():
            spatial_dims = len(x_shape) - 2
            opts = self.get_attr_options(spatial_dims)

            for dilations, strides, pads, group, auto_pad in itertools.product(
                opts["dilations"],
                opts["strides"],
                opts["pads"],
                opts["group"],
                opts["auto_pad"],
            ):
                # Skip invalid combination: SAME_LOWER/SAME_UPPER auto_pad with
                # non-uniform dilations (ONNX Runtime doesn't support this)
                # TODO: refine the condition to skip a case when
                # auto_pad is in ("SAME_LOWER", "SAME_UPPER")
                dilations_all_ones = all(d == 1 for d in dilations)
                if auto_pad in ("SAME_LOWER", "SAME_UPPER") and not dilations_all_ones:
                    continue

                # W shape: (M, C/group, k...)
                c = x_shape[1]
                w_shape = (m, c // group, *k_shape)

                # Bias options: None (no bias) first, then with bias
                # None comes first as it's more common in modern networks
                bias_options = [None, InputShapeConstraint((m,))]

                for bias_opt in bias_options:
                    # Base combination - inputs and attributes with default values
                    base_comb = {
                        "X": InputShapeConstraint(x_shape),
                        "W": InputShapeConstraint(w_shape),
                        "dilations": dilations,
                        "group": group,
                        "strides": strides,
                        "auto_pad": auto_pad,
                    }

                    # Add bias if provided (None means no bias)
                    if bias_opt is not None:
                        base_comb["B"] = bias_opt

                    if auto_pad == "NOTSET":
                        base_comb["pads"] = pads

                    # Generate combinations with different optional attribute subsets
                    # Only kernel_shape has no default value (can be inferred from W)
                    for optional_subset in self._get_optional_combinations(k_shape):
                        comb = base_comb.copy()
                        comb.update(optional_subset)
                        combinations.append(comb)

        return combinations

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for Conv operator testing.

        Args:
            properties: Base properties containing X_shape and W_shape

        Returns:
            Updated properties with Conv-specific derived values
        """
        item = super().derive_properties(properties)

        # Conv-specific: kernel_shape can be omitted (inferred from W)
        item["kernel_shape_is_none"] = item.get("attr_kernel_shape") is None

        return item

    def get_infinite_property_names(self) -> list[str]:
        """Get list of attribute names and input names that have infinite value sets.

        Returns:
            List of attribute and input names with infinite value sets.
        """
        return [
            "X_shape",
            "W_shape",
            "attr_dilations",
            "attr_kernel_shape",
            "attr_strides",
            "attr_pads",
            "attr_group",
            "B_shape",
        ]


@register_runtime_checker_op
class ConvTransposeInputGenerator(ConvInputGenerator):
    """Input generator for ConvTranspose operator."""

    op_name = "ConvTranspose"
    expand_optionals = False

    def _calc_output_shape(
        self,
        x_shape: tuple,
        k_shape: tuple,
        strides: list,
        dilations: list,
        pads: list,
        output_padding: list,
        auto_pad: str,
    ) -> list[int]:
        """Calculate ConvTranspose spatial output shape per ONNX spec.

        ONNX formula (for each spatial dimension i):
        out[i] = stride[i] * (in[i] - 1) + output_padding[i]
                 + ((k[i] - 1) * dilation[i] + 1) - pads_begin[i] - pads_end[i]

        Special cases for auto_pad:
        - SAME_UPPER/SAME_LOWER: out[i] = in[i] * stride[i] (pads auto-computed)
        - VALID: equivalent to pads=0
        - NOTSET: uses explicit pads
        """
        spatial_dims = len(k_shape)
        spatial_output_shape = []

        for i in range(spatial_dims):
            d_in = x_shape[2 + i]
            stride = strides[i]
            dilation = dilations[i]
            k = k_shape[i]
            op = output_padding[i]

            # Effective kernel size: k_eff = (k - 1) * dilation + 1
            k_eff = (k - 1) * dilation + 1

            if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
                d_out = d_in * stride + op
            elif auto_pad == "VALID":
                d_out = (d_in - 1) * stride + k_eff + op
            else:  # NOTSET - use explicit pads
                pad_head = pads[i]
                pad_tail = pads[i + spatial_dims]
                d_out = (d_in - 1) * stride - pad_head - pad_tail + k_eff + op

            spatial_output_shape.append(d_out)

        return spatial_output_shape

    def _get_optional_combinations(
        self,
        x_shape: tuple,
        k_shape: tuple,
        strides: list,
        dilations: list,
        pads: list,
        output_padding: list,
        auto_pad: str,
    ) -> list[dict]:
        """Generate optional attribute combinations for ConvTranspose.

        Test scenarios (ordered by priority - simplest first):
        1. No output_padding, no output_shape: most common/simple case
        2. With output_padding, no output_shape
        3. No output_padding, with output_shape
        4. With output_padding, with output_shape
        """
        output_shape = self._calc_output_shape(
            x_shape, k_shape, strides, dilations, pads, output_padding, auto_pad
        )

        return [
            {},  # Case 1: No output_padding, no output_shape
            {"output_padding": output_padding},  # Case 2: With output_padding only
            {"output_shape": tuple(output_shape)},  # Case 3: With output_shape only
            {"output_padding": output_padding, "output_shape": tuple(output_shape)},  # Case 4: Both
        ]

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return finite attribute sets for ConvTranspose operator."""
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input and infinite attribute combinations for ConvTranspose operator."""
        combinations = []
        for x_shape, m, k_shape in self.get_base_conv_shapes():
            spatial_dims = len(x_shape) - 2
            opts = self.get_attr_options(spatial_dims)

            # output_padding options for _get_conv_transpose_optional_combinations
            # Using [0]*n as the test value when output_padding is provided
            output_padding = [0] * spatial_dims

            for (
                dilations,
                strides,
                pads,
                group,
                auto_pad,
            ) in itertools.product(
                opts["dilations"],
                opts["strides"],
                opts["pads"],
                opts["group"],
                opts["auto_pad"],
            ):
                # Skip invalid combination: SAME_LOWER/SAME_UPPER auto_pad with
                # non-uniform dilations (ONNX Runtime doesn't support this)
                # TODO: refine the condition to skip a case when
                # auto_pad is in ("SAME_LOWER", "SAME_UPPER")
                dilations_all_ones = all(d == 1 for d in dilations)
                if auto_pad in ("SAME_LOWER", "SAME_UPPER") and not dilations_all_ones:
                    continue

                # Handle auto_pad compatibility
                current_pads = pads
                if auto_pad != "NOTSET":
                    current_pads = [0] * (2 * spatial_dims)

                # W shape: (C, M/group, k...)
                c = x_shape[1]
                w_shape = (c, m // group, *k_shape)

                # Base combination without optional attributes
                base_comb = {
                    "X": InputShapeConstraint(x_shape),
                    "W": InputShapeConstraint(w_shape),
                    "dilations": dilations,
                    "group": group,
                    "kernel_shape": k_shape,
                    "strides": strides,
                    "auto_pad": auto_pad,
                }

                if auto_pad == "NOTSET":
                    base_comb["pads"] = pads

                # Bias options: None (no bias) first, then with bias
                # None comes first as it's more common in modern networks
                bias_options = [None, InputShapeConstraint((m,))]

                for bias_opt in bias_options:
                    comb_with_bias = base_comb.copy()

                    # Add bias if provided (None means no bias)
                    if bias_opt is not None:
                        comb_with_bias["B"] = bias_opt

                    # Generate combinations with different optional attribute subsets
                    # Handles the dependency between output_padding and output_shape
                    for optional_subset in self._get_optional_combinations(
                        x_shape, k_shape, strides, dilations, current_pads, output_padding, auto_pad
                    ):
                        comb = comb_with_bias.copy()
                        comb.update(optional_subset)
                        combinations.append(comb)
        return combinations

    def get_infinite_property_names(self) -> list[str]:
        """Get list of attribute names and input names that have infinite value sets.

        Returns:
            List of attribute and input names with infinite value sets.
        """
        # Note: attr_pads IS included for ConvTranspose because pads interact
        # with output_shape/output_padding in complex ways, making exact matching less useful.
        # attr_output_shape and attr_output_padding are also infinite because their values
        # depend on input shapes. Use output_shape_is_none and output_padding_is_none
        # (derived in derive_properties) for rule matching instead.
        return [
            "X_shape",
            "W_shape",
            "attr_dilations",
            "attr_kernel_shape",
            "attr_strides",
            "attr_pads",
            "attr_group",
            "attr_output_shape",
            "attr_output_padding",
            "B_shape",
        ]
