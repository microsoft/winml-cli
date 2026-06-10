# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Validator for batched MatMul with a constant operand on OpenVINO GPU.

OpenVINO GPU's oneDNN gemm cannot select an implementation for a batched
(rank >= 3) MatMul where an operand is a compile-time constant. The identical
gemm with a dynamic operand, and 2D constant gemm, both compile fine. Models
whose batched MatMul weights fold to constants (e.g. transformer disentangled
attention position terms) therefore fail to compile on OpenVINO GPU with:

    [GPU] Failed to select implementation for ... type: gemm

This validator detects that structural pattern and recommends the
``untie-constant-batched-matmul`` surgery, which makes the constant operand
runtime-valued so gemm implementation selection succeeds.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ...models.information import Action, ActionItem, ActionLevel, Information
from ...utils import infer_ihv_from_ep_name
from .base import ModelValidator


if TYPE_CHECKING:
    from ....utils.constants import EPName
    from ...models.onnx_model import ONNXModel
    from ...models.runtime_checks import PatternRuntime

logger = logging.getLogger(__name__)

# Surgery capability enabled when the pattern is detected (kebab-case to match
# the capability registry / autoconf normalization).
_SURGERY_FLAG = "untie-constant-batched-matmul"


class BatchedConstMatMulValidator(ModelValidator):
    """Detect batched MatMul with a constant operand (OpenVINO GPU only)."""

    def __init__(
        self,
        model: ONNXModel,
        op_runtime_results: list[PatternRuntime] | None = None,
        ep: EPName | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(model, op_runtime_results=op_runtime_results)
        self.ep = ep
        self.device = device

    @property
    def validator_name(self) -> str:
        """Name of this validator for logging/reporting."""
        return "BatchedConstMatMulValidator"

    @property
    def pattern_id(self) -> str:
        """Pattern ID for Information objects."""
        return "MODEL/BatchedConstantMatMul"

    def _is_enabled(self) -> bool:
        """Only relevant for OpenVINO (Intel IHV) on GPU."""
        if (self.device or "").upper() != "GPU":
            return False
        if not self.ep:
            return False
        try:
            from ...models.ihv_type import IHVType

            return infer_ihv_from_ep_name(self.ep) == IHVType.INTEL
        except Exception:  # pragma: no cover - defensive
            return False

    def validate(self) -> Information | None:
        """Detect batched MatMul with a single constant rank>=3 operand."""
        if not self._is_enabled():
            return None

        # Known gap: constants expressed as `Constant` op nodes (rather than
        # graph initializers) are not detected here. The `untie-constant-batched
        # -matmul` surgery in surgery.py has the same limitation, so detection
        # and surgery stay consistent. Most exporters and ORT preprocessing emit
        # weights as initializers, so this covers the disentangled-attention case
        # in practice; `Constant`-node weights would need handling on both sides.
        initializers = {init.name for init in self.graph.initializer}
        rank_by_init = {init.name: len(init.dims) for init in self.graph.initializer}

        offenders: list[str] = []
        for node in self.graph.node:
            if node.op_type != "MatMul" or len(node.input) != 2:
                continue
            const_inputs = [name for name in node.input if name in initializers]
            # Exactly one constant operand (two-constant MatMuls fold away and
            # never reach gemm impl selection).
            if len(const_inputs) != 1:
                continue
            if rank_by_init.get(const_inputs[0], 0) >= 3:
                offenders.append(node.name or const_inputs[0])

        if not offenders:
            return None

        examples = ", ".join(offenders[:3])
        action = Action(
            pattern_from_id="",
            pattern_to_id="",
            level=ActionLevel.REQUIRED,
            status=None,
            action_items=[
                ActionItem(type="GraphOptimization", optimization_options={_SURGERY_FLAG: True})
            ],
            details=(
                "Enable untie-constant-batched-matmul surgery so the constant "
                "operand becomes runtime-valued and OpenVINO GPU can select a "
                "gemm implementation."
            ),
        )
        # https://github.com/openvinotoolkit/openvino/issues/36272
        explanation = (
            f"Model contains {len(offenders)} batched MatMul(s) with a constant "
            f"operand (examples: {examples}). OpenVINO GPU's oneDNN gemm cannot "
            f"select an implementation for a batched MatMul with a constant "
            f"operand, causing a '[GPU] Failed to select implementation ... gemm' "
            f"compile failure. The untie-constant-batched-matmul surgery makes "
            f"the operand runtime-valued without changing numerics. "
            f"It is fixed in openvino==2026.2.0, so no need to apply the surgery "
            f"if using that version or later."
        )
        return Information(
            explanation=explanation,
            actions=[action],
            pattern_id=self.pattern_id,
            status=None,
        )
