# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Bounded constant folding for CGC static shape subgraphs.

FoundryToolbox currently lacks some constant folding needed by ONNX lowering.
These rewrites fill that gap without an ORT Session or execution-provider graph
transformations. Runtime floating-point computations are left unchanged.
"""

from __future__ import annotations

import logging
import math
from typing import cast

import numpy as np
from onnx import (
    ModelProto,
    TensorProto,
    ValueInfoProto,
    helper,
    numpy_helper,
    shape_inference,
)
from onnx.reference import ReferenceEvaluator


logger = logging.getLogger(__name__)
_MAX_ELEMENTS = 65536
_MAX_NODES = 128
_MAX_CACHED_ELEMENTS = 1048576
_OPERATORS = {"ConstantOfShape", "Concat", "Reshape", "Slice", "Transpose", "Cast"}
_SHAPE_OPERATORS = _OPERATORS | {
    "Mod", "Add", "Sub", "Mul", "Div", "Squeeze", "Unsqueeze", "Gather", "Equal", "Where",
}
_BROADCAST_OPERATORS = {"Mod", "Add", "Sub", "Mul", "Div", "Equal", "Where"}
_INTEGER_TYPES = {
    TensorProto.INT8, TensorProto.INT16, TensorProto.INT32,
    TensorProto.INT64, TensorProto.UINT8, TensorProto.UINT16,
    TensorProto.UINT32, TensorProto.UINT64, TensorProto.BOOL,
}


class _ConstantParameters:
    def __init__(self, model: ModelProto, *, static_shapes: bool = False) -> None:
        self.model = model
        self.static_shapes = static_shapes
        self.shapes = {
            value.name: value.type.tensor_type.shape
            for value in [*model.graph.input, *model.graph.value_info]
            if value.type.tensor_type.HasField("shape")
        }
        self.producers = {
            name: node for node in model.graph.node for name in node.output if name
        }
        self.initializers = {value.name: value for value in model.graph.initializer}
        self.inputs = {value.name for value in model.graph.input}
        self.values: dict[str, np.ndarray] = {}
        self.cached_elements = 0

    def tensor(self, value: TensorProto) -> np.ndarray:
        if value.data_type not in _INTEGER_TYPES or math.prod(value.dims) > _MAX_ELEMENTS:
            raise ValueError("Not a bounded integer tensor")
        return numpy_helper.to_array(value)

    def evaluate(self, name: str, visited: set[str], active: set[str]) -> np.ndarray:
        if name in self.inputs or name in active:
            raise ValueError("Runtime input or cyclic dependency")
        visited.add(name)
        if len(visited) > _MAX_NODES or len(active) >= _MAX_NODES:
            raise ValueError("Constant dependency budget exceeded")
        if name in self.values:
            return self.values[name]
        active.add(name)
        try:
            if name in self.initializers:
                result = self.tensor(self.initializers[name])
            else:
                node = self.producers[name]
                if node.domain not in {"", "ai.onnx"} or len(node.output) != 1:
                    raise ValueError("Unsupported constant producer")
                if node.op_type == "Constant":
                    if len(node.attribute) != 1 or node.attribute[0].name != "value":
                        raise ValueError("Only tensor-valued Constants are supported")
                    result = self.tensor(node.attribute[0].t)
                elif self.static_shapes and node.op_type == "Shape":
                    if len(node.input) != 1:
                        raise ValueError("Invalid Shape inputs")
                    shape = self.shapes.get(node.input[0])
                    if shape is None:
                        tensor = self.initializers.get(node.input[0])
                        if tensor is None:
                            raise ValueError("Unknown tensor rank")
                        shape = helper.make_tensor_type_proto(
                            tensor.data_type, list(tensor.dims)
                        ).tensor_type.shape
                    attributes = {attr.name: helper.get_attribute_value(attr)
                                  for attr in node.attribute}
                    if set(attributes) - {"start", "end"}:
                        raise ValueError("Unsupported Shape attributes")
                    dimensions = list(shape.dim)[attributes.get("start", 0):attributes.get("end")]
                    if not all(dim.HasField("dim_value") and dim.dim_value >= 0
                               for dim in dimensions):
                        raise ValueError("Shape contains dynamic dimensions")
                    result = np.asarray([dim.dim_value for dim in dimensions], dtype=np.int64)
                else:
                    operators = _SHAPE_OPERATORS if self.static_shapes else _OPERATORS
                    if node.op_type not in operators:
                        raise ValueError("Unsupported integer constant operation")
                    inputs = {
                        item: self.evaluate(item, visited, active) for item in node.input if item
                    }
                    if sum(inputs[item].size for item in node.input if item) > _MAX_ELEMENTS:
                        raise ValueError("Constant operation input budget exceeded")
                    if node.op_type in _BROADCAST_OPERATORS:
                        output_shape = np.broadcast_shapes(
                            *(value.shape for value in inputs.values()),
                        )
                        if math.prod(output_shape) > _MAX_ELEMENTS:
                            raise ValueError("Broadcast allocation budget exceeded")
                    if node.op_type == "Gather":
                        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
                        data, indices = (inputs[item] for item in node.input)
                        axis %= data.ndim
                        output_shape = (*data.shape[:axis], *indices.shape, *data.shape[axis + 1:])
                        if math.prod(output_shape) > _MAX_ELEMENTS:
                            raise ValueError("Gather allocation budget exceeded")
                    if node.op_type == "ConstantOfShape":
                        target_shape = inputs[node.input[0]]
                        if (
                            target_shape.ndim != 1 or target_shape.dtype != np.int64
                            or np.any(target_shape < 0)
                            or math.prod(int(dim) for dim in target_shape) > _MAX_ELEMENTS
                            or len(target_shape) > 32
                        ):
                            raise ValueError("ConstantOfShape allocation budget exceeded")
                        if not node.attribute:
                            raise ValueError("Default ConstantOfShape output is floating point")
                        for attribute in node.attribute:
                            if attribute.name != "value":
                                raise ValueError("Unsupported ConstantOfShape attribute")
                            self.tensor(attribute.t)
                    if node.op_type == "Cast":
                        target = next(attr.i for attr in node.attribute if attr.name == "to")
                        if target not in _INTEGER_TYPES:
                            raise ValueError("Only integer Cast targets are supported")
                    fragment = helper.make_model(
                        helper.make_graph(
                            [node], "constant_parameter", [],
                            [ValueInfoProto(name=name)],
                            [numpy_helper.from_array(value, key) for key, value in inputs.items()],
                        ),
                        opset_imports=list(self.model.opset_import),
                        ir_version=self.model.ir_version,
                    )
                    result = cast("list[np.ndarray]", ReferenceEvaluator(fragment).run(None, {}))[0]
            if result.dtype.kind not in "iub" or result.size > _MAX_ELEMENTS:
                raise ValueError("Constant result exceeds supported type or size")
            if self.cached_elements + result.size > _MAX_CACHED_ELEMENTS:
                raise ValueError("Constant cache budget exceeded")
            self.cached_elements += result.size
            self.values[name] = result
            return result
        finally:
            active.remove(name)


def cgc_constant_folding(model: ModelProto) -> ModelProto:
    """Fill FoundryToolbox constant-folding gaps for static Shape chains.

    Only the main graph is changed. Graphs containing Shape fold bounded integer/boolean chains
    to a fixed point with shape inference. This never freezes symbolic input
    dimensions: callers must specialize inputs explicitly before this rule when
    required. Casts of runtime data, including floating-point outputs, remain.
    """
    if not any(node.op_type == "Shape" and node.domain in {"", "ai.onnx"}
               for node in model.graph.node):
        return model
    versions = {item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}}
    if len(versions) != 1 or next(iter(versions)) < 11:
        return model
    rewritten = ModelProto()
    rewritten.CopyFrom(model)
    changed = False
    for _iteration in range(32):
        for value_info in rewritten.graph.value_info:
            value_info.type.tensor_type.ClearField("shape")
        rewritten = shape_inference.infer_shapes(rewritten, strict_mode=True, data_prop=True)
        evaluator = _ConstantParameters(rewritten, static_shapes=True)
        folded = 0
        for node in rewritten.graph.node:
            if node.op_type == "Constant" or len(node.output) != 1:
                continue
            try:
                value = evaluator.evaluate(node.output[0], set(), set())
            except (
                ValueError, KeyError, TypeError, IndexError, StopIteration,
                NotImplementedError, OverflowError, ZeroDivisionError,
            ):
                continue
            node.CopyFrom(helper.make_node(
                "Constant", [], list(node.output), value=numpy_helper.from_array(value),
            ))
            folded += 1
        if not folded:
            return rewritten if changed else model
        changed = True
        logger.info("CGC constant folding: folded %d shape/integer node(s)", folded)
    logger.warning("CGC constant folding reached the 32-round limit; retaining partial folding")
    return rewritten
