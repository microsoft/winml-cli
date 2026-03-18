# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pydantic import BaseModel, Field


class OnnxOP(BaseModel):
    """Represents an ONNX operator node for output.

    Attributes:
        op_type: Operator type (e.g., Conv, Relu, Add)
        display_name: Human-readable display name
        node_name: Original ONNX node name
        namespace: ONNX operator namespace (e.g., ai.onnx, com.microsoft)
        node_id: Node identifier (optional)
        attributes: Operator attributes (optional)
    """

    op_type: str = Field(..., description="Operator type")
    display_name: str | None = Field(default=None, description="Human-readable display name")
    node_name: str = Field(..., description="Original ONNX node name")
    namespace: str = Field(default="ai.onnx", description="ONNX operator namespace")
    node_id: str | None = Field(default=None, description="Node identifier")
    attributes: dict[str, str] | None = Field(
        default=None, description="Operator attributes (optional)"
    )
