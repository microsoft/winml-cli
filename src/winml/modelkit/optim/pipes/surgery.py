# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Surgery pipe for precise model modifications.

This pipe performs targeted graph transformations that are not part of
ONNX Runtime's standard optimization passes. Surgery operations run before
ORT optimizations to prepare models for quantization or specific execution providers.

Use cases:
- Clamp extreme constant values to prevent quantization issues
- Prepare models for specific execution providers (QNN, etc.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ..capabilities import surgery
from .base import BasePipe, PipeConfig, caps_dict


if TYPE_CHECKING:
    import onnx

logger = logging.getLogger(__name__)


# =============================================================================
# MODULE-LEVEL CAPABILITIES
# =============================================================================

SURGERY_CAPABILITIES: dict[str, Any] = caps_dict(
    surgery.CLAMP_CONSTANT_VALUES,
    surgery.REMOVE_ISNAN_IN_ATTENTION_MASK,
)


# =============================================================================
# SURGERYPIPECONFIG
# =============================================================================


@dataclass
class SurgeryPipeConfig(PipeConfig):
    """Configuration for surgery optimization pipe.

    Attributes:
        clamp_constant_values: Whether to clamp extreme float constants
        clamp_min: Minimum value for constant clamping (default: -1e3)
        clamp_max: Maximum value for constant clamping (default: 1e3)
        fix_nan_attention_mask: Replace -inf attention mask with finite value
            and remove Softmax->IsNaN->Where NaN guard patterns
        mask_value: Replacement value for -inf (default: -1e3)
        verbose: Enable verbose logging
    """

    clamp_constant_values: bool = False
    clamp_min: float = -1e3
    clamp_max: float = 1e3
    remove_isnan_in_attention_mask: bool = False
    verbose: bool = False


# =============================================================================
# SURGERYPIPE
# =============================================================================


class SurgeryPipe(BasePipe[SurgeryPipeConfig]):
    """Surgery pipe for precise model modifications.

    This pipe performs targeted graph transformations to prepare models
    for quantization or specific execution providers. It runs before
    ORT optimizations.

    Currently supported operations:
    - clamp-constant-values: Clamp extreme float constants (e.g., -inf → -1e3)
    """

    name: ClassVar[str] = "surgery"
    capabilities: ClassVar[dict[str, Any]] = SURGERY_CAPABILITIES

    @classmethod
    def build_config(cls, **kwargs: Any) -> SurgeryPipeConfig:
        """Build surgery pipe config from kwargs.

        Args:
            **kwargs: User-provided configuration
                - clamp_constant_values: Enable/disable constant clamping
                - clamp_min: Minimum value for clamping (default: -1e3)
                - clamp_max: Maximum value for clamping (default: 1e3)
                - remove_isnan_in_attention_mask: Remove IsNaN guard patterns
                - verbose: Enable verbose logging

        Returns:
            Configured SurgeryPipeConfig
        """
        return SurgeryPipeConfig(
            clamp_constant_values=kwargs.get("clamp_constant_values", False),
            clamp_min=kwargs.get("clamp_min", -1e3),
            clamp_max=kwargs.get("clamp_max", 1e3),
            remove_isnan_in_attention_mask=kwargs.get("remove_isnan_in_attention_mask", False),
            verbose=kwargs.get("verbose", False),
        )

    @classmethod
    def should_process(cls, config: SurgeryPipeConfig) -> bool:
        """Check if surgery pipe should process the model.

        Args:
            config: Surgery pipe configuration

        Returns:
            True if any surgery operation is enabled
        """
        return config.clamp_constant_values or config.remove_isnan_in_attention_mask

    def process(self, model: onnx.ModelProto, config: SurgeryPipeConfig) -> onnx.ModelProto:
        """Apply surgery operations to the model.

        Args:
            model: Input ONNX model (will not be modified)
            config: Surgery pipe configuration

        Returns:
            New model with surgery operations applied
        """
        if not self.should_process(config):
            return model

        # Import onnx inside method to avoid import errors
        import onnx

        # Create a copy of the model to avoid modifying the original
        model_copy = onnx.ModelProto()
        model_copy.CopyFrom(model)

        if config.clamp_constant_values:
            model_copy = self._clamp_constant_values(
                model_copy, config.clamp_min, config.clamp_max, config.verbose
            )

        if config.remove_isnan_in_attention_mask:
            model_copy = self._remove_isnan_in_attention_mask(model_copy, config.verbose)

        return model_copy

    def _clamp_constant_values(
        self,
        model: onnx.ModelProto,
        clamp_min: float,
        clamp_max: float,
        verbose: bool = False,
    ) -> onnx.ModelProto:
        """Clamp extreme float constant values in the model.

        This operation modifies initializers (weights/constants) to clamp
        extreme values like -inf or very large floats to a reasonable range.
        This prevents quantization issues where inf values produce inf scales.

        Args:
            model: ONNX model (modified in place)
            clamp_min: Minimum allowed value
            clamp_max: Maximum allowed value
            verbose: Log details about clamped tensors

        Returns:
            Model with clamped constants
        """
        from onnx import TensorProto, numpy_helper

        clamped_count = 0
        clamped_tensors: list[str] = []

        for initializer in model.graph.initializer:
            # Only process float types
            if initializer.data_type not in (
                TensorProto.FLOAT,
                TensorProto.FLOAT16,
                TensorProto.DOUBLE,
            ):
                continue

            # Convert to numpy array
            tensor = numpy_helper.to_array(initializer)
            original_min = float(tensor.min())
            original_max = float(tensor.max())

            # Check if clamping is needed
            needs_clamp = original_min < clamp_min or original_max > clamp_max

            if needs_clamp:
                # Clamp the values (np.clip is equivalent to torch.clamp)
                clamped = np.clip(tensor, clamp_min, clamp_max)

                # Create new tensor proto with clamped values
                new_tensor = numpy_helper.from_array(clamped, initializer.name)

                # Copy over the initializer
                initializer.CopyFrom(new_tensor)

                clamped_count += 1
                clamped_tensors.append(initializer.name)

                if verbose:
                    logger.info(
                        "Clamped tensor '%s': [%.2e, %.2e] -> [%.2e, %.2e]",
                        initializer.name,
                        original_min,
                        original_max,
                        clamp_min,
                        clamp_max,
                    )

        if clamped_count > 0:
            logger.info(
                "SurgeryPipe: Clamped %d tensor(s) to range [%.2e, %.2e]",
                clamped_count,
                clamp_min,
                clamp_max,
            )
            if verbose:
                logger.debug("Clamped tensors: %s", clamped_tensors)

        return model

    # -----------------------------------------------------------------
    # remove-isnan-in-attention-mask
    # -----------------------------------------------------------------

    def _remove_isnan_in_attention_mask(
        self,
        model: onnx.ModelProto,
        verbose: bool = False,
    ) -> onnx.ModelProto:
        """Remove Softmax → IsNaN → Where NaN guard patterns in attention.

        Pattern: Softmax → IsNaN → Where(isnan, 0, softmax_out)
        Remove IsNaN + guard Where, use Softmax output directly.

        These guards are dead code when clamp_constant_values has already
        replaced -inf with a finite value (Softmax never produces NaN).

        Args:
            model: ONNX model (modified in place).
            verbose: Log details about each removal.

        Returns:
            Model with IsNaN guard patterns removed.
        """
        guard_count = 0

        # Build output→node map
        output_to_node: dict[str, onnx.NodeProto] = {}
        for node in model.graph.node:
            for out in node.output:
                output_to_node[out] = node

        nodes_to_remove: list[onnx.NodeProto] = []
        rewire_map: dict[str, str] = {}

        for node in list(model.graph.node):
            if node.op_type != "IsNaN":
                continue
            producer = output_to_node.get(node.input[0])
            if producer is None or producer.op_type != "Softmax":
                continue
            softmax_out = producer.output[0]
            isnan_out = node.output[0]

            # Find guard Where consuming IsNaN output
            guard_wheres = [
                n for n in model.graph.node if n.op_type == "Where" and isnan_out in n.input
            ]
            if len(guard_wheres) != 1:
                continue
            guard_where = guard_wheres[0]
            if softmax_out not in guard_where.input:
                continue

            guard_out = guard_where.output[0]
            nodes_to_remove.extend([node, guard_where])
            rewire_map[guard_out] = softmax_out
            guard_count += 1
            if verbose:
                logger.info(
                    "  remove-isnan: remove %s + %s, rewire %s -> %s",
                    node.name,
                    guard_where.name,
                    guard_out,
                    softmax_out,
                )

        # Apply rewiring
        for node in model.graph.node:
            for i, inp in enumerate(node.input):
                if inp in rewire_map:
                    node.input[i] = rewire_map[inp]
        for graph_out in model.graph.output:
            if graph_out.name in rewire_map:
                graph_out.name = rewire_map[graph_out.name]

        # Remove dead nodes
        remove_ids = {id(n) for n in nodes_to_remove}
        remaining = [n for n in model.graph.node if id(n) not in remove_ids]
        del model.graph.node[:]
        model.graph.node.extend(remaining)

        if guard_count:
            logger.info(
                "SurgeryPipe: remove-isnan-in-attention-mask: %d IsNaN+Where guards removed",
                guard_count,
            )

        return model
