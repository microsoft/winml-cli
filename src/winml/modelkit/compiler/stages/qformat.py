"""QLinear to QDQ format conversion stage."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ..utils import needs_format_conversion
from .base import BaseStage


if TYPE_CHECKING:
    from ..context import CompileContext


class QFormatConvertStage(BaseStage):
    """Convert between QLinear and QDQ quantization formats.

    QNN requires QDQ format. Only runs when format doesn't match target EP.
    FIXME: Actual conversion not yet implemented.
    """

    name: ClassVar[str] = "qformat-convert"

    @classmethod
    def should_run(cls, context: CompileContext) -> bool:
        """Run when model's quant format is incompatible with target EP."""
        return needs_format_conversion(context.model_path, context.execution_provider)

    def process(self, context: CompileContext) -> CompileContext:
        """Convert quant format. FIXME: Not yet implemented."""
        context.log("QFormat conversion needed but not yet implemented")
        context.add_warning(
            "QFormat conversion (QLinear->QDQ) not yet implemented. "
            "Model may fail at runtime if EP requires QDQ format."
        )
        return context
