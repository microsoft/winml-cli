"""Lower 2D linear, zero-padded GridSample to indexed interpolation."""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper
from onnx.defs import get_schema

from ...onnx import ONNXDomain
from .. import InputInfo, PatternMatchResult, make_single_op_pattern


_SCHEMA, _GridSamplePattern = make_single_op_pattern(get_schema("GridSample", 16))


class LinearGridSamplePattern(_GridSamplePattern):
    """Match floating-point 2D sampling with known spatial and channel dimensions."""

    def check_skeleton_result(self, skeleton_match_result):
        """Reject unsupported modes, types and spatial dimensions."""
        node = skeleton_match_result.matched_nodes[0]
        matcher = skeleton_match_result.matcher
        version = matcher.domain_versions.get(ONNXDomain.AI_ONNX, 0)
        if node.domain not in {"", ONNXDomain.AI_ONNX.value} or version < 16:
            return None
        attributes = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if (
            set(attributes) - {"mode", "padding_mode", "align_corners"}
            or attributes.get("mode", b"linear" if version >= 20 else b"bilinear")
            != (b"linear" if version >= 20 else b"bilinear")
            or attributes.get("padding_mode", b"zeros") != b"zeros"
            or attributes.get("align_corners", 0) not in {0, 1}
            or len(node.input) != 2 or len(node.output) != 1
        ):
            return None
        shapes = [matcher.get_tensor_shape(name) for name in node.input]
        types = [matcher.get_tensor_type_str(name) for name in node.input]
        if types[0] not in {"tensor(float)", "tensor(float16)"} or types[1] != types[0]:
            return None
        if any(shape is None or len(shape) != 4 for shape in shapes):
            return None
        if shapes[1][-1] != 2 or any(
            not isinstance(dim, int) or dim <= 0
            for dim in [*shapes[0][1:], *shapes[1][1:3]]
        ):
            return None
        if (
            isinstance(shapes[0][0], int) and isinstance(shapes[1][0], int)
            and shapes[0][0] != shapes[1][0]
        ):
            return None
        attributes.update({"_shapes": shapes, "_ir_version": matcher.model.ir_version})
        return PatternMatchResult(
            skeleton_match_result=skeleton_match_result,
            schema_input_to_value={"X": node.input[0], "grid": node.input[1]},
            schema_output_to_value={"Y": node.output[0]},
            type_param_to_type={"T1": types[0], "T2": types[1]},
            attributes=attributes,
            input_infos={"X": InputInfo(name="X"), "grid": InputInfo(name="grid")},
        )


class GatherLinearGridSamplePattern(_GridSamplePattern):
    """Interpolate four samples with explicit runtime batch coordinates.

    GatherND uses batch_dims=0 to avoid the symbolic shape inference defect in
    https://github.com/microsoft/onnxruntime/pull/24206.
    """

    def get_onnx_model(
        self, inputs, attributes, is_constant_map, output_dtypes, domain_versions,
        prefix="", input_names=None, output_names=None,
    ):
        """Build batched indexed interpolation without specializing batch."""
        del inputs, is_constant_map
        data_shape, grid_shape = attributes["_shapes"]
        _, channels, height, width = data_shape
        align = attributes.get("align_corners", 0)
        fp16 = output_dtypes[0] == "tensor(float16)"
        element_type = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT
        nodes = []

        def constant(label, value):
            name = f"{prefix}{label}"
            nodes.append(helper.make_node(
                "Constant", [], [name], value=numpy_helper.from_array(np.asarray(value)),
            ))
            return name

        def operation(kind, *arguments, **attrs):
            name = f"{prefix}value_{len(nodes)}"
            nodes.append(helper.make_node(kind, list(arguments), [name], **attrs))
            return name

        data, grid = input_names
        if fp16:
            data = operation("Cast", data, to=TensorProto.FLOAT)
            grid = operation("Cast", grid, to=TensorProto.FLOAT)
        zero = constant("zero", np.float32(0))
        one = constant("one", np.float32(1))
        half = constant("half", np.float32(0.5)) if not align else None
        axes = constant("axes", np.asarray([-1], np.int64))
        coordinates = []
        for axis, size in enumerate((width, height)):
            component = operation(
                "Gather", grid, constant(f"axis_{axis}", np.int64(axis)), axis=3,
            )
            scale = constant(f"scale_{axis}", np.float32((size - 1 if align else size) / 2))
            pixel = operation("Mul", operation("Add", component, one), scale)
            if not align:
                pixel = operation("Sub", pixel, half)
            lower = operation("Floor", pixel)
            fraction = operation("Sub", pixel, lower)
            coordinates.append((lower, operation("Add", lower, one),
                                operation("Sub", one, fraction), fraction))
        source = operation(
            "Reshape", operation("Transpose", data, perm=[0, 2, 3, 1]),
            constant("source_shape", np.asarray([0, -1, channels], np.int64)),
        )
        stride = constant("stride", np.int64(width))
        index_shape = constant("index_shape", np.asarray([0, -1, 1], np.int64))
        sampled_shape = constant(
            "sampled_shape", np.asarray([0, grid_shape[1], grid_shape[2], channels], np.int64),
        )
        limits = [constant(f"limit_{axis}", np.float32(size - 1))
                  for axis, size in enumerate((width, height))]
        batch_indices = None
        terms = []
        for row in range(2):
            for column in range(2):
                valid, indices = [], []
                for coord, limit in zip(
                    [coordinates[0][column], coordinates[1][row]], limits, strict=True,
                ):
                    valid.append(operation(
                        "And", operation("GreaterOrEqual", coord, zero),
                        operation("LessOrEqual", coord, limit),
                    ))
                    indices.append(operation(
                        "Cast", operation("Clip", coord, zero, limit), to=TensorProto.INT64,
                    ))
                linear = operation("Add", indices[0], operation("Mul", indices[1], stride))
                spatial_indices = operation("Reshape", linear, index_shape)
                if batch_indices is None:
                    indices_shape = operation("Shape", spatial_indices)
                    batch_zero = constant("batch_zero", np.int64(0))
                    batch_one = constant("batch_one", np.int64(1))
                    batch_size = operation("Gather", indices_shape, batch_zero, axis=0)
                    batch_range = operation("Range", batch_zero, batch_size, batch_one)
                    batch_indices = operation(
                        "Reshape", batch_range,
                        constant("batch_shape", np.asarray([-1, 1, 1], np.int64)),
                    )
                    batch_indices = operation("Expand", batch_indices, indices_shape)
                sampled = operation(
                    "GatherND", source,
                    operation("Concat", batch_indices, spatial_indices, axis=2), batch_dims=0,
                )
                sampled = operation("Reshape", sampled, sampled_shape)
                sampled = operation(
                    "Where", operation("Unsqueeze", operation("And", *valid), axes), sampled, zero,
                )
                weight = operation("Mul", coordinates[0][column + 2], coordinates[1][row + 2])
                terms.append(operation("Mul", sampled, operation("Unsqueeze", weight, axes)))
        result = operation("Add", operation("Add", terms[0], terms[1]),
                           operation("Add", terms[2], terms[3]))
        result = operation("Transpose", result, perm=[0, 3, 1, 2])
        if fp16:
            result = operation("Cast", result, to=TensorProto.FLOAT16)
        nodes[-1].output[0] = output_names[0]
        output_shape = [data_shape[0], channels, grid_shape[1], grid_shape[2]]
        return helper.make_model(
            helper.make_graph(nodes, f"{prefix}GridSampleToGather",
                [helper.make_tensor_value_info(name, element_type, shape)
                 for name, shape in zip(input_names, [data_shape, grid_shape], strict=True)],
                [helper.make_tensor_value_info(output_names[0], element_type, output_shape)]),
            opset_imports=[helper.make_opsetid(domain.schema_domain, version)
                           for domain, version in domain_versions.items()],
            ir_version=attributes["_ir_version"],
        )


__all__ = ["GatherLinearGridSamplePattern", "LinearGridSamplePattern"]
