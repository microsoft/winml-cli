"""Normalize constant INT32 DequantizeLinear parameters for CGC."""

import numpy as np
from onnx import AttributeProto, GraphProto, ModelProto, TensorProto, numpy_helper


def normalize_int32_dq(model: ModelProto) -> ModelProto:
    """Omit immutable zero points and scalarize singleton INT32 DQ scales.

    Only local initializer-backed data and parameters are considered. Shared
    initializers remain untouched; nested graphs are handled independently.
    """
    versions = {opset.domain: opset.version for opset in model.opset_import}
    result = ModelProto()
    result.CopyFrom(model)
    used_names: set[str] = set()

    def collect_names(graph: GraphProto) -> None:
        used_names.update(value.name for value in graph.initializer)
        used_names.update(value.name for value in [*graph.input, *graph.output, *graph.value_info])
        for node in graph.node:
            used_names.update(node.input)
            used_names.update(node.output)
            for attribute in node.attribute:
                if attribute.type == AttributeProto.GRAPH:
                    collect_names(attribute.g)
                elif attribute.type == AttributeProto.GRAPHS:
                    for subgraph in attribute.graphs:
                        collect_names(subgraph)

    def rewrite(graph: GraphProto) -> bool:
        changed = False
        initializers = {value.name: value for value in graph.initializer}
        inputs = {value.name for value in graph.input}
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == AttributeProto.GRAPH:
                    changed = rewrite(attribute.g) or changed
                elif attribute.type == AttributeProto.GRAPHS:
                    for subgraph in attribute.graphs:
                        changed = rewrite(subgraph) or changed
            supported = (
                node.domain == "" and versions.get("", 0) >= 10
            ) or (node.domain == "com.microsoft" and versions.get("com.microsoft") == 1)
            if not supported or node.op_type != "DequantizeLinear" or len(node.input) not in (2, 3):
                continue
            if any(attribute.name != "axis" for attribute in node.attribute):
                continue
            data = initializers.get(node.input[0])
            scale = initializers.get(node.input[1])
            if data is None or data.data_type != TensorProto.INT32:
                continue
            if scale is None or scale.name in inputs or list(scale.dims) not in ([], [1]):
                continue
            zero_name = node.input[2] if len(node.input) == 3 else ""
            if zero_name:
                zero = initializers.get(zero_name)
                if zero is None or zero.name in inputs or zero.data_type != TensorProto.INT32:
                    continue
                if list(zero.dims) not in ([], [1]) or not np.all(numpy_helper.to_array(zero) == 0):
                    continue
            if list(scale.dims) == [1]:
                name = scale.name + "_cgc_scalar"
                while name in used_names:
                    name += "_"
                used_names.add(name)
                scalar = numpy_helper.from_array(numpy_helper.to_array(scale).reshape(()), name)
                graph.initializer.append(scalar)
                node.input[1] = name
                changed = True
            if len(node.input) == 3:
                del node.input[2]
                changed = True
        return changed

    collect_names(result.graph)
    return result if rewrite(result.graph) else model
