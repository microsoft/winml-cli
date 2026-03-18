# types currently not supported in onnxscript.onnx_types
# BFLOAT16,
# COMPLEX64,
# COMPLEX128,
# FLOAT4E2M1,
# FLOAT8E4M3FN,
# FLOAT8E4M3FNUZ,
# FLOAT8E5M2,
# FLOAT8E5M2FNUZ,
# FLOAT8E8M0,
# INT4,
# STRING,
# UINT4,

"""ONNX type conversion utilities shared across ModelKit modules.

Canonical home for ONNX ↔ numpy ↔ TensorProto type mappings.
"""

import re
from enum import Enum
from typing import Any

import numpy as np
import onnx


# Utility function
def remove_optional_from_type_annotation(type_annotation: str) -> str:
    """Remove 'Optional' wrapper from type annotation string, if present.

    Args:
        type_annotation: Type annotation string (e.g., "Optional[FLOAT]")

    Returns:
        Type annotation without Optional wrapper (e.g., "FLOAT")
    """
    optional_match = re.match(r"Optional\[(\w+)\]", type_annotation)
    if optional_match:
        return optional_match.group(1)
    return type_annotation


class SupportedONNXType(Enum):
    """Enum for supported ONNX types with conversion methods."""

    FLOAT = ("FLOAT", np.dtype("float32"), "tensor(float)", onnx.TensorProto.FLOAT)
    BOOL = ("BOOL", np.dtype("bool"), "tensor(bool)", onnx.TensorProto.BOOL)
    DOUBLE = ("DOUBLE", np.dtype("float64"), "tensor(double)", onnx.TensorProto.DOUBLE)
    FLOAT16 = ("FLOAT16", np.dtype("float16"), "tensor(float16)", onnx.TensorProto.FLOAT16)
    INT8 = ("INT8", np.dtype("int8"), "tensor(int8)", onnx.TensorProto.INT8)
    INT16 = ("INT16", np.dtype("int16"), "tensor(int16)", onnx.TensorProto.INT16)
    INT32 = ("INT32", np.dtype("int32"), "tensor(int32)", onnx.TensorProto.INT32)
    INT64 = ("INT64", np.dtype("int64"), "tensor(int64)", onnx.TensorProto.INT64)
    UINT8 = ("UINT8", np.dtype("uint8"), "tensor(uint8)", onnx.TensorProto.UINT8)
    UINT16 = ("UINT16", np.dtype("uint16"), "tensor(uint16)", onnx.TensorProto.UINT16)
    UINT32 = ("UINT32", np.dtype("uint32"), "tensor(uint32)", onnx.TensorProto.UINT32)
    UINT64 = ("UINT64", np.dtype("uint64"), "tensor(uint64)", onnx.TensorProto.UINT64)

    @classmethod
    def post_class_definition(cls) -> None:
        """Build lookup dictionaries after class definition.

        This is called automatically after the enum class is fully defined.
        """
        cls._onnx_type_map = {}
        cls._np_type_map = {}
        cls._tensor_proto_type_map = {}
        for member in cls:
            cls._onnx_type_map[member.onnx_type] = member
            cls._np_type_map[member.np_type] = member
            cls._tensor_proto_type_map[member.tensor_proto_type] = member

    @classmethod
    def from_annotation(cls, annotation: str) -> "SupportedONNXType":
        """Create from annotation string like 'BOOL' or 'Optional[BOOL]'.

        Args:
            annotation: Type annotation string (e.g., "BOOL", "Optional[FLOAT]")

        Returns:
            SupportedONNXType enum member

        Raises:
            ValueError: If annotation is not supported
        """
        inner_annotation = remove_optional_from_type_annotation(annotation)
        # ONNX attribute schemas sometimes use plural forms (INTS/FLOATS) to mean a list of scalars.
        # We map them to the underlying scalar dtype so tensor/value casting still works.
        plural_map = {
            "INTS": "INT64",   # axes, splits, etc.
            "FLOATS": "FLOAT",
        }
        if inner_annotation in plural_map:
            inner_annotation = plural_map[inner_annotation]
        try:
            return cls[inner_annotation]
        except KeyError as e:
            raise ValueError(f"Unsupported dtype annotation: {annotation}") from e

    @classmethod
    def from_onnx_type(cls, onnx_type: str) -> "SupportedONNXType":
        """Create from ONNX type string like 'tensor(bool)'.

        Args:
            onnx_type: ONNX type string (e.g., "tensor(bool)")

        Returns:
            SupportedONNXType enum member

        Raises:
            ValueError: If ONNX type is not supported
        """
        if onnx_type not in cls._onnx_type_map:
            raise ValueError(f"Unsupported ONNX type: {onnx_type}")
        return cls._onnx_type_map[onnx_type]

    @classmethod
    def from_np_type(cls, np_type: "np.dtype[Any]") -> "SupportedONNXType":
        """Create from numpy dtype.

        Args:
            np_type: Numpy dtype

        Returns:
            SupportedONNXType enum member

        Raises:
            ValueError: If numpy type is not supported
        """
        if np_type not in cls._np_type_map:
            raise ValueError(f"Unsupported numpy type: {np_type}")
        return cls._np_type_map[np_type]

    @classmethod
    def from_tensor_proto_type(cls, tensor_proto_type: int) -> "SupportedONNXType":
        """Create from ONNX TensorProto data type enum value.

        Args:
            tensor_proto_type: ONNX TensorProto data type (e.g., 1 for FLOAT, 9 for BOOL)

        Returns:
            SupportedONNXType enum member

        Raises:
            ValueError: If TensorProto type is not supported
        """
        if tensor_proto_type not in cls._tensor_proto_type_map:
            raise ValueError(f"Unsupported TensorProto type: {tensor_proto_type}")
        return cls._tensor_proto_type_map[tensor_proto_type]

    @property
    def annotation(self) -> str:
        """Convert to annotation string like 'BOOL'.

        Returns:
            Type annotation string
        """
        return self.value[0]

    @property
    def onnx_type(self) -> str:
        """Convert to ONNX type string like 'tensor(bool)'.

        Returns:
            ONNX type string
        """
        return self.value[2]

    @property
    def np_type(self) -> "np.dtype[Any]":
        """Convert to numpy dtype.

        Returns:
            Numpy dtype
        """
        return self.value[1]

    @property
    def tensor_proto_type(self) -> int:
        """Convert to ONNX TensorProto data type enum value.

        Returns:
            ONNX TensorProto data type (e.g., 1 for FLOAT, 9 for BOOL)
        """
        return self.value[3]

    @classmethod
    def normalize_annotation(cls, annotation: str) -> str:
        """Normalize type annotation by removing 'Optional' wrapper if present.

        Args:
            annotation: Type annotation string (e.g., "Optional[FLOAT]")

        Returns:
            Normalized type annotation (e.g., "FLOAT")
        """
        return cls.from_annotation(annotation).annotation


# Initialize lookup dictionaries after class definition
SupportedONNXType.post_class_definition()
