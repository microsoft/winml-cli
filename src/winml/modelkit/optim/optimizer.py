"""Main Optimizer class for ONNX model optimization.

This module provides the core Optimizer class that orchestrates optimization pipes.
Each pipe represents an optimization technology/API (e.g., ORT graph optimization,
fusion optimization). The optimizer runs all registered pipes sequentially.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar

import onnx

from .registry import auto_enable_dependencies


if TYPE_CHECKING:
    from .pipes.base import BasePipe

# Configure module logger
logger = logging.getLogger(__name__)


class Optimizer:
    """Main optimizer that orchestrates optimization pipes.

    The Optimizer class runs a sequence of optimization pipes on an ONNX model.
    Each pipe is responsible for applying a specific category of optimizations
    (e.g., graph-level optimizations, fusion optimizations).

    Attributes:
        pipes: Class-level list of pipe classes to run. Each pipe is instantiated
            and executed in order. Currently includes ORTGraphPipe and ORTFusionPipe.

    Example:
        >>> optimizer = Optimizer()
        >>> optimized_model = optimizer.optimize(
        ...     model,
        ...     gelu_fusion=True,
        ...     attention_fusion=True
        ... )
    """

    # Static pipe registration (lazy loaded)
    pipes: ClassVar[list[type[BasePipe]]] = []

    @classmethod
    def _initialize_pipes(cls) -> None:
        """Lazy initialization of pipes to avoid import errors."""
        if not cls.pipes:
            from .pipes import PIPES

            cls.pipes = PIPES

    def optimize(self, model: onnx.ModelProto, **kwargs: Any) -> onnx.ModelProto:
        """Run optimization pipeline on the ONNX model.

        Execution order:
        1. Validate input model
        2. Symbolic shape inference (mandatory pre-stage)
        3. For each optimization pipe (capability-driven)
        4. ONNX shape inference (mandatory post-stage)
        5. Validate output model

        Args:
            model: Input ONNX ModelProto to optimize
            **kwargs: Configuration parameters for all pipes. Each pipe extracts
                its relevant parameters during build_config().

        Returns:
            Optimized ONNX ModelProto. The returned model is a new object; the
            input model is not modified.

        Raises:
            onnx.checker.ValidationError: If input model is invalid
            Exception: If any pipe fails during processing

        Example:
            >>> optimizer = Optimizer()
            >>> optimized = optimizer.optimize(
            ...     model,
            ...     gelu_fusion=True,
            ...     attention_fusion=True
            ... )
        """
        # Initialize pipes on first use
        self._initialize_pipes()

        # Auto-enable dependencies (e.g., bias-gelu-fusion requires gelu-fusion)
        kwargs = self._resolve_dependencies(kwargs)

        # PRE-STAGE: Shape inference (required for optimizers)
        # Many optimizers check .Shape() and skip if nullptr (BiasGeluFusion, etc.)
        from ..onnx import infer_shapes

        logger.info("Running shape inference (pre-stage)...")
        start_time = time.time()
        model = infer_shapes(model)
        logger.info("✓ Shape inference (pre-stage) completed in %.2fs", time.time() - start_time)

        # CAPABILITY-DRIVEN: Optimization pipes
        logger.info("Starting optimization pipeline (%d pipes)...", len(self.pipes))
        for pipe_class in self.pipes:
            pipe_name = pipe_class.name if hasattr(pipe_class, "name") else pipe_class.__name__

            # Create pipe instance
            pipe = pipe_class()

            # Build configuration for this pipe
            config = pipe.build_config(**kwargs)

            # Check if pipe should process (using getattr with callable check)
            should_process = getattr(pipe, "should_process", None)
            if callable(should_process) and not should_process(config):
                logger.info("⊘ Skipping %s (no capabilities enabled)", pipe_name)
                continue

            # Process model with timing
            start_time = time.time()
            try:
                logger.info("⚙ Executing %s...", pipe_name)
                model = pipe.process(model, config)
                logger.info("✓ %s completed in %.2fs", pipe_name, time.time() - start_time)
            except Exception as e:
                logger.error("✗ %s failed: %s", pipe_name, e)
                raise

        # MANDATORY POST-STAGE: Shape inference
        # Symbolic handles com.microsoft ops, ONNX fallback for edge cases
        logger.info("Running shape inference...")
        start_time = time.time()
        model = infer_shapes(model)
        logger.info("✓ Shape inference completed in %.2fs", time.time() - start_time)

        # Note: Post-optimization validation removed. Validation happens at:
        # - load_onnx(validate=True) on input (path-based, safe for any size)
        # - load_onnx(validate=True) when consumer loads the saved output
        # In-memory check_model fails on >2GiB models and models with custom
        # domains (com.microsoft from ORT), making it unreliable here.

        logger.info("✓ Optimization pipeline completed successfully")
        return model

    def _resolve_dependencies(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Resolve capability dependencies.

        When a capability is enabled that depends on another capability,
        automatically enable the dependency. For example, bias-gelu-fusion
        depends on gelu-fusion.

        Args:
            kwargs: Original configuration parameters

        Returns:
            Updated kwargs with dependencies resolved
        """
        from .pipes import get_all_capabilities

        # Get all capabilities from all pipes
        all_caps = get_all_capabilities()

        # Convert kwargs (snake_case) to kebab-case config for dependency resolution
        config = {}
        for cap_name, cap_def in all_caps.items():
            python_name = cap_def.python_name
            config[cap_name] = kwargs.get(python_name, cap_def.default)

        # Apply dependency resolution
        resolved_config = auto_enable_dependencies(config, all_caps)

        # Convert back to kwargs format (snake_case)
        result = dict(kwargs)  # Start with original kwargs
        for cap_name, value in resolved_config.items():
            if cap_name in all_caps:
                python_name = all_caps[cap_name].python_name
                result[python_name] = value

        return result

