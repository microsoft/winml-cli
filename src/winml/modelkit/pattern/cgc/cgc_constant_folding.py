# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Bounded constant folding for CGC Pad parameters and static shape subgraphs.

FoundryToolbox currently lacks some constant folding needed by ONNX lowering.
These rewrites fill that gap without an ORT Session or execution-provider graph
transformations. Runtime floating-point computations are left unchanged.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, deque
from typing import cast

import numpy as np
from onnx import (
    AttributeProto,
    GraphProto,
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
        self.evaluated: set[str] = set()
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
                            [node], "constant_pad_parameter", [],
                            [ValueInfoProto(name=name)],
                            [numpy_helper.from_array(value, key) for key, value in inputs.items()],
                        ),
                        opset_imports=list(self.model.opset_import),
                        ir_version=self.model.ir_version,
                    )
                    result = cast("list[np.ndarray]", ReferenceEvaluator(fragment).run(None, {}))[0]
                self.evaluated.add(name)
            if result.dtype.kind not in "iub" or result.size > _MAX_ELEMENTS:
                raise ValueError("Constant result exceeds supported type or size")
            if self.cached_elements + result.size > _MAX_CACHED_ELEMENTS:
                raise ValueError("Constant cache budget exceeded")
            self.cached_elements += result.size
            self.values[name] = result
            return result
        finally:
            active.remove(name)


def _referenced_names(graph: GraphProto) -> list[str]:
    names = [value.name for value in graph.output]
    for annotation in graph.quantization_annotation:
        names.append(annotation.tensor_name)
        names.extend(item.value for item in annotation.quant_parameter_tensor_names)
    for node in graph.node:
        names.extend(name for name in node.input if name)
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                names.extend(_referenced_names(attribute.g))
            elif attribute.type == AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    names.extend(_referenced_names(child))
    return names


def fold_constant_pad_pads(model: ModelProto) -> ModelProto:
    """Fold constant integer Pad widths without specializing runtime input shapes.

    Only the main graph is rewritten. Nested graph captures and quantization
    annotations conservatively protect shared producers from removal.
    """
    versions = [item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}]
    if not versions or len(set(versions)) != 1 or versions[0] < 11:
        return model
    candidates = [
        (index, node) for index, node in enumerate(model.graph.node)
        if node.domain in {"", "ai.onnx"} and node.op_type == "Pad"
        and len(node.input) >= 2 and node.input[1]
    ]
    if not candidates:
        return model
    evaluator = _ConstantParameters(model)
    types = {
        value.name: value.type for value in
        [*model.graph.input, *model.graph.value_info, *model.graph.output]
    }
    replacements: dict[int, np.ndarray] = {}
    selected_dependencies: set[str] = set()
    for index, node in candidates:
        producer = evaluator.producers.get(node.input[1])
        if producer is None or producer.op_type == "Constant":
            continue
        try:
            visited: set[str] = set()
            pads = evaluator.evaluate(node.input[1], visited, set())
            if pads.dtype != np.int64 or pads.ndim != 1 or pads.size % 2:
                continue
            tensor_type = types.get(node.input[0])
            rank = (
                len(tensor_type.tensor_type.shape.dim)
                if tensor_type is not None and tensor_type.tensor_type.HasField("shape") else None
            )
            if len(node.input) > 3 and node.input[3]:
                if versions[0] < 18:
                    continue
                axes = evaluator.evaluate(node.input[3], visited, set())
                if axes.ndim != 1 or axes.dtype not in {np.dtype("int32"), np.dtype("int64")}:
                    continue
                if pads.size != 2 * axes.size:
                    continue
                if rank is not None:
                    if np.any(axes < -rank) or np.any(axes >= rank):
                        continue
                    if len({int(axis) % rank for axis in axes}) != axes.size:
                        continue
            elif rank is not None and pads.size != 2 * rank:
                continue
            replacements[index] = pads
            selected_dependencies.update(visited)
        except (
            ValueError, KeyError, TypeError, IndexError, StopIteration,
            NotImplementedError, OverflowError,
        ):
            logger.debug("Pad constant parameter is not foldable: %s", node.name, exc_info=True)
    if not replacements:
        return model

    rewritten = ModelProto()
    rewritten.CopyFrom(model)
    used_names = set(evaluator.producers) | set(evaluator.initializers) | evaluator.inputs
    used_names.update(_referenced_names(model.graph))
    used_names.update(value.name for value in model.graph.value_info)
    constants = []
    folded_names: dict[str, str] = {}
    for index, value in replacements.items():
        node = rewritten.graph.node[index]
        source = node.input[1]
        if source not in folded_names:
            name = source + "_folded_pads"
            while name in used_names:
                name += "_"
            used_names.add(name)
            folded_names[source] = name
            constants.append(helper.make_node(
                "Constant", [], [name], value=numpy_helper.from_array(value),
            ))
        node.input[1] = folded_names[source]

    references = Counter(_referenced_names(rewritten.graph))
    removable = {
        node.output[0]: node for node in rewritten.graph.node
        if len(node.output) == 1
        and node.output[0] in evaluator.evaluated & selected_dependencies
    }
    pending = deque(name for name in removable if not references[name])
    removed: set[str] = set()
    while pending:
        name = pending.popleft()
        if name in removed:
            continue
        removed.add(name)
        for source in removable[name].input:
            references[source] -= 1
            if source in removable and not references[source]:
                pending.append(source)
    nodes = [node for node in rewritten.graph.node if not any(n in removed for n in node.output)]
    del rewritten.graph.node[:]
    rewritten.graph.node.extend([*constants, *nodes])
    infos = [value for value in rewritten.graph.value_info if value.name not in removed]
    del rewritten.graph.value_info[:]
    rewritten.graph.value_info.extend(infos)
    logger.info("Folded constant pads for %d Pad node(s)", len(replacements))
    return rewritten


def cgc_constant_folding(model: ModelProto) -> ModelProto:
    """Fill FoundryToolbox constant-folding gaps for Pad and static Shape chains.

    Only the main graph is changed. Pad widths use the existing bounded folder;
    graphs containing Shape also fold bounded integer/boolean constant chains
    to a fixed point with shape inference. This never freezes symbolic input
    dimensions: callers must specialize inputs explicitly before this rule when
    required. Casts of runtime data, including floating-point outputs, remain.
    """
    prepared = fold_constant_pad_pads(model)
    if not any(node.op_type == "Shape" and node.domain in {"", "ai.onnx"}
               for node in prepared.graph.node):
        return prepared
    versions = {item.version for item in prepared.opset_import if item.domain in {"", "ai.onnx"}}
    if len(versions) != 1 or next(iter(versions)) < 11:
        return prepared
    rewritten = ModelProto()
    rewritten.CopyFrom(prepared)
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
            return rewritten if changed else prepared
        changed = True
        logger.info("CGC constant folding: folded %d shape/integer node(s)", folded)
    logger.warning("CGC constant folding reached the 32-round limit; retaining partial folding")
    return rewritten
