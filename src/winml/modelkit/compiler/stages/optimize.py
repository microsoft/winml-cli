# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""EP-specific graph optimization stage."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ...onnx import load_onnx, save_onnx
from ..transforms import get_transforms_for_ep
from .base import BaseStage


if TYPE_CHECKING:
    from ..context import CompileContext


class OptimizeStage(BaseStage):
    """EP-specific ONNX graph transformations.

    Applies registered graph transforms for the target EP.
    No built-in transforms shipped -- stage skeleton + registry only.
    """

    name: ClassVar[str] = "ep-optimize"

    @classmethod
    def should_run(cls, context: CompileContext) -> bool:
        """Run when transforms are registered for the target EP."""
        return len(get_transforms_for_ep(context.execution_provider)) > 0

    def process(self, context: CompileContext) -> CompileContext:
        """Apply registered transforms sequentially."""
        transforms = get_transforms_for_ep(context.execution_provider)
        model = load_onnx(context.model_path, validate=False)
        for t in transforms:
            context.log(f"Applying transform: {type(t).__name__}")
            model = t.transform(model)
        # Save transformed model alongside original
        output_path = context.model_path.with_name(
            context.model_path.stem + "_ep_opt" + context.model_path.suffix
        )
        save_onnx(model, output_path)
        context.model_path = output_path
        context.log(f"EP optimization saved: {output_path}")
        return context
