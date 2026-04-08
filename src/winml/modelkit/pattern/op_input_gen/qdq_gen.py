# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from onnx.defs import SchemaError

from ...onnx import SupportedONNXType


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from ...onnx import ONNXDomain


class QDQGenerator:
    """Generator for QuantizeLinear and DequantizeLinear op configurations.

    Manages supported types for QDQ operations:
    - DequantizeLinear: quantized input (weight_types) -> float output (dq_output_types)
    - QuantizeLinear: float input (q_input_types) -> quantized output (activation_types)
    """

    # Supported quantized types for weights (DQ input)
    SUPPORTED_WEIGHT_TYPES: ClassVar[set[str]] = {
        SupportedONNXType.INT8.onnx_type,
        SupportedONNXType.UINT8.onnx_type,
        SupportedONNXType.INT16.onnx_type,
        SupportedONNXType.UINT16.onnx_type,
    }

    SUPPORT_DQ_OUTPUT_TYPES: ClassVar[set[str]] = {
        SupportedONNXType.FLOAT.onnx_type,
    }

    # Supported quantized types for activations (Q output)
    SUPPORTED_ACTIVATION_TYPES: ClassVar[set[str]] = {
        SupportedONNXType.INT8.onnx_type,
        SupportedONNXType.UINT8.onnx_type,
        SupportedONNXType.INT16.onnx_type,
        SupportedONNXType.UINT16.onnx_type,
    }

    SUPPORTED_Q_INPUT_TYPES: ClassVar[set[str]] = {
        SupportedONNXType.FLOAT.onnx_type,
    }

    def __init__(self, opset_version: int, domain: ONNXDomain) -> None:
        self.domain = domain

        try:
            self.dequantize_linear_schema = domain.get_op_schema("DequantizeLinear", opset_version)
            logger.debug(
                "DequantizeLinear schema since_version: %s",
                self.dequantize_linear_schema.since_version,
            )
            self.opset_version = self.dequantize_linear_schema.since_version
        except SchemaError as e:
            logger.debug("Failed DequantizeLinear: %s", e)
            raise

        try:
            self.quantize_linear_schema = domain.get_op_schema("QuantizeLinear", opset_version)
            logger.debug(
                "QuantizeLinear schema since_version: %s",
                self.quantize_linear_schema.since_version,
            )
            ql_ver = self.quantize_linear_schema.since_version
            self.opset_version = max(self.opset_version, ql_ver)
        except SchemaError as e:
            logger.debug("Failed QuantizeLinear: %s", e)
            raise

        supported_onnx_types = {x.onnx_type: x for x in SupportedONNXType}
        self._build_dq_type_vars(supported_onnx_types)
        self._build_q_type_vars(supported_onnx_types)

    def _build_dq_type_vars(self, supported_onnx_types: dict[str, SupportedONNXType]) -> None:
        """Create the following mappings for DequantizeLinear.

        self.weight_onnx_types:
            List of supported weight types, both supported by schema and in SUPPORTED_WEIGHT_TYPES
        self.dq_output_onnx_types:
            List of supported output types, both supported by schema and in SUPPORT_DQ_OUTPUT_TYPES
        self.weight_all_onnx_types:
            List of supported weight types, as per schema only
        """
        # Get type constraints from schema
        type_constraints = {
            tc.type_param_str: set(tc.allowed_type_strs)
            for tc in self.dequantize_linear_schema.type_constraints
        }

        # T1 is input type (weights), T2 is output type
        # Find input type constraint (T1)
        input_type_str = self.dequantize_linear_schema.inputs[0].type_str
        schema_weight_types = type_constraints.get(input_type_str, set())

        # Find output type constraint (T2)
        # Note: output may be a concrete type (e.g., "tensor(float)") not a type param
        output_type_str = self.dequantize_linear_schema.outputs[0].type_str
        schema_output_types = type_constraints.get(output_type_str, {output_type_str})

        # Intersect with supported types
        self.weight_onnx_types: list[str] = [
            t
            for t in schema_weight_types
            if t in self.SUPPORTED_WEIGHT_TYPES and t in supported_onnx_types
        ]
        self.dq_output_onnx_types: list[str] = [
            t
            for t in schema_output_types
            if t in self.SUPPORT_DQ_OUTPUT_TYPES and t in supported_onnx_types
        ]
        self.weight_all_onnx_types: list[str] = [
            t for t in schema_weight_types if t in supported_onnx_types
        ]
        logger.debug("DequantizeLinear weight types: %s", self.weight_onnx_types)
        logger.debug("DequantizeLinear output types: %s", self.dq_output_onnx_types)
        logger.debug("DequantizeLinear all weight types: %s", self.weight_all_onnx_types)

    def _build_q_type_vars(self, supported_onnx_types: dict[str, SupportedONNXType]) -> None:
        """Create the following mappings for QuantizeLinear.

        self.activation_onnx_types:
            List of supported activation types, both supported by schema
            and in SUPPORTED_ACTIVATION_TYPES
        self.q_input_onnx_types:
            List of supported input types, both supported by schema
            and in SUPPORTED_Q_INPUT_TYPES
        self.activation_all_onnx_types:
            List of supported activation types, as per schema only
        """
        # Get type constraints from schema
        type_constraints = {
            tc.type_param_str: set(tc.allowed_type_strs)
            for tc in self.quantize_linear_schema.type_constraints
        }

        # T1 is input type, T2 is output type (activations)
        # Find input type constraint (T1)
        input_type_str = self.quantize_linear_schema.inputs[0].type_str
        schema_input_types = type_constraints.get(input_type_str, set())

        # Find output type constraint (T2)
        output_type_str = self.quantize_linear_schema.outputs[0].type_str
        schema_activation_types = type_constraints.get(output_type_str, set())

        # Intersect with supported types
        self.activation_onnx_types: list[str] = [
            t
            for t in schema_activation_types
            if t in self.SUPPORTED_ACTIVATION_TYPES and t in supported_onnx_types
        ]
        self.q_input_onnx_types: list[str] = [
            t
            for t in schema_input_types
            if t in self.SUPPORTED_Q_INPUT_TYPES and t in supported_onnx_types
        ]
        self.activation_all_onnx_types: list[str] = [
            t for t in schema_activation_types if t in supported_onnx_types
        ]
        logger.debug("QuantizeLinear activation types: %s", self.activation_onnx_types)
        logger.debug("QuantizeLinear input types: %s", self.q_input_onnx_types)
        logger.debug("QuantizeLinear all activation types: %s", self.activation_all_onnx_types)
