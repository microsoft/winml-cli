# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Graph optimization pipe using ONNX Runtime SessionOptions.

This pipe manages advanced graph optimizations (fusions, transformers) that we
set as default=False in our capability registry. Basic optimizations like
ConstantFolding and IdentityElimination are enabled by ORT at Level 2.

Design principle: GRAPH_CAPABILITIES only contains our default=False items.
Users explicitly enable the advanced optimizations they want.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from winml.modelkit.onnx import load_onnx, save_onnx

from .base import BasePipe, OptimizationError, PipeConfig, caps_dict


if TYPE_CHECKING:
    import onnx

# Import all capability modules to build capabilities dict
from ..capabilities import (
    activation,
    conv,
    elimination,
    gelu,
    gemm,
    layernorm,
    layout,
    matmul,
    misc,
)
from ..capabilities import graph as graph_caps
from ..registry import BoolCapability


logger = logging.getLogger(__name__)


# =============================================================================
# MODULE-LEVEL CAPABILITIES
# =============================================================================
# Defined at module level so ORTGraphPipeConfig can access in __init__

# Only our default=False capabilities (advanced fusions/transformers).
# We exclude default=True items (ConstantFolding, IdentityElimination, etc.)
# because ORT already enables those at Level 2 - no need to configure them.
GRAPH_CAPABILITIES: dict[str, Any] = caps_dict(
    # GELU fusions (all default=False)
    gelu.GELU_FUSION,
    gelu.FAST_GELU_FUSION,
    gelu.BIAS_GELU_FUSION,
    gelu.QUICK_GELU_FUSION,
    gelu.GELU_APPROXIMATION,
    # Activation fusions (all default=False)
    activation.BIAS_SOFTMAX_FUSION,
    activation.BIAS_DROPOUT_FUSION,
    # Convolution fusions (all default=False)
    conv.CONV_ADD_FUSION,
    conv.CONV_BN_FUSION,
    conv.CONV_MUL_FUSION,
    conv.CONV_ACTIVATION_FUSION,
    # Advanced eliminations (default=False) - basic ones handled by ORT
    elimination.SLICE_ELIMINATION,
    elimination.EXPAND_ELIMINATION,
    elimination.UNSQUEEZE_ELIMINATION,
    # GEMM fusions (all default=False)
    gemm.GEMM_ACTIVATION_FUSION,
    gemm.GEMM_SUM_FUSION,
    gemm.GEMM_TRANSPOSE_FUSION,
    # Graph optimizations
    graph_caps.CONCAT_SLICE_ELIMINATION,
    graph_caps.DOUBLE_QDQ_PAIRS_REMOVER,
    graph_caps.CONSTANT_FOLDING,  # default=True, can be disabled for size-sensitive models
    # LayerNorm fusions (all default=False)
    # Note: BIAS_SKIP_LAYER_NORM_FUSION is FusionOptions only, not a graph optimizer
    layernorm.LAYER_NORM_FUSION,
    layernorm.SKIP_LAYER_NORM_FUSION,
    layernorm.SIMPLIFIED_LAYER_NORM_FUSION,
    # NOTE: EMBED_LAYER_NORM_FUSION removed - uses FusionPipe instead
    # Layout transformers (all default=False)
    layout.TRANSPOSE_OPTIMIZER,
    layout.NHWC_TRANSFORMER,
    layout.NCHWC_TRANSFORMER,
    layout.CONV_ADD_ACTIVATION_FUSION,
    # MatMul fusions (all default=False)
    matmul.MATMUL_ADD_FUSION,
    matmul.MATMUL_ACTIVATION_FUSION,
    matmul.MATMUL_TRANSPOSE_FUSION,
    matmul.MATMUL_SCALE_FUSION,
    matmul.MATMUL_BN_FUSION,
    matmul.DYNAMIC_QUANTIZE_MATMUL_FUSION,
    # Misc fusions (all default=False)
    misc.GATHER_SLICE_TO_SPLIT_FUSION,
    misc.GATHER_TO_SLICE_FUSION,
    misc.PAD_FUSION,
    misc.NOT_WHERE_FUSION,
)


# =============================================================================
# GRAPHPIPECONFIG
# =============================================================================


class ORTGraphPipeConfig(PipeConfig):
    """Configuration for enabling advanced graph optimizations.

    This config only manages default=False capabilities (fusions, transformers).
    Basic optimizations (ConstantFolding, IdentityElimination, etc.) are always
    enabled by ORT at Level 2 - no configuration needed.

    Design: Start with all advanced optimizers disabled, enable specific ones.

    Attributes:
        optimization_level: ORT optimization level (finalized at 2)
        disabled_optimizers: List of ORT optimizer names to disable
        enable_gelu_approximation: Special flag for GeluApproximation (off by default)
        verbose: Enable verbose logging

    Note:
        GeluApproximation is handled specially because ORT requires a separate
        session config entry (optimization.enable_gelu_approximation) rather
        than using the disable_specified_optimizers mechanism.

    Example:
        # No advanced optimizations (only ORT basics run)
        baseline = ORTGraphPipeConfig()

        # Enable specific fusions
        config = ORTGraphPipeConfig(enabled=["gelu_fusion", "layernorm_fusion"])

        # Method chaining
        config = ORTGraphPipeConfig().enable("gelu_fusion").enable("layer_norm_fusion")
    """

    def __init__(
        self,
        enabled: list[str] | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize with all advanced optimizers disabled, enable specified ones.

        Args:
            enabled: List of python_names to enable (e.g., ["gelu_fusion"])
            verbose: Enable verbose logging

        Note:
            Only default=False capabilities are managed here. Basic ORT
            optimizations (ConstantFolding, etc.) run automatically at Level 2.
        """
        self.optimization_level = 2  # Finalized at Level 2
        self.verbose = verbose

        # Special flag for GeluApproximation (requires separate session config)
        # ORT docs: "GeluApproximation has side effects which may change results.
        # It needs to be manually enabled."
        self.enable_gelu_approximation = False

        # Start with default=False optimizers disabled (conservative for advanced fusions)
        # Capabilities with default=True (like ConstantFolding) stay enabled
        self.disabled_optimizers: list[str] = [
            cap.ort_name
            for cap in GRAPH_CAPABILITIES.values()
            if hasattr(cap, "ort_name") and cap.ort_name and not cap.default
        ]

        # IMPORTANT: Also disable L1 variants of optimizers that have both L1 and L2 versions
        # At Level 2, ORT runs both L1 and L2 optimizers. Our capabilities use L2 names
        # (e.g., "LayerNormFusionL2"), but we must also disable L1 names to truly isolate.
        # Without this, L1 variants would still run even when L2 is disabled.
        l1_variants = [
            "LayerNormFusion",  # L1 variant of LayerNormFusionL2
            "GeluFusion",  # L1 variant of GeluFusionL2
        ]
        for l1_name in l1_variants:
            if l1_name not in self.disabled_optimizers:
                self.disabled_optimizers.append(l1_name)

        # IMPORTANT: Always disable these optimizers - they are NOT handled by GraphPipe
        # AttentionFusion and EmbedLayerNormFusion require optimize_model() API with
        # transformer-specific analysis, not available through SessionOptions.
        always_disabled = [
            "AttentionFusion",  # Requires optimize_model() - TBD which pipe handles this
            "EmbedLayerNormFusion",  # Requires optimize_model() - TBD which pipe handles this
        ]
        for name in always_disabled:
            if name not in self.disabled_optimizers:
                self.disabled_optimizers.append(name)

        # Enable specified capabilities
        if enabled:
            for python_name in enabled:
                self.enable(python_name)

    def enable(self, python_name: str) -> ORTGraphPipeConfig:
        """Enable a capability by python_name, including its dependencies.

        Args:
            python_name: Capability name in snake_case (e.g., "gelu_fusion")

        Returns:
            Self for method chaining

        Note:
            - GeluApproximation (gelu_approximation) is handled specially.
              ORT requires a separate session config entry to enable it.
            - Dependencies are auto-enabled. See docs/design/optimization/4_graph_pipe.md
              for the full dependency table.
        """
        # Enable dependencies first (hardcoded for now, see dependency table in docs)
        # TODO: Future redesign should consolidate all dependency logic
        if python_name == "bias_gelu_fusion" or python_name == "gelu_approximation":
            self._enable_ort_name("GeluFusionL2")
            self._enable_ort_name("GeluFusion")  # Also enable L1 variant
        elif python_name == "skip_layer_norm_fusion":
            self._enable_ort_name("LayerNormFusionL2")
            self._enable_ort_name("LayerNormFusion")  # Also enable L1 variant
        elif python_name == "bias_skip_layer_norm_fusion":
            self._enable_ort_name("LayerNormFusionL2")
            self._enable_ort_name("LayerNormFusion")  # Also enable L1 variant
            self._enable_ort_name("SkipLayerNormFusion")
        # NOTE: AttentionFusion and EmbedLayerNormFusion are handled by FusionPipe, not GraphPipe
        elif python_name == "matmul_activation_fusion":
            self._enable_ort_name("MatmulTransposeFusion")
        elif python_name == "conv_add_activation_fusion":
            # ConvAddActivationFusion depends on ConvAddFusion and ConvActivationFusion
            # to fold Add into Conv bias and fuse activation
            self._enable_ort_name("ConvAddFusion")
            self._enable_ort_name("ConvActivationFusion")

        # When enabling L2 variants, also enable corresponding L1 variants
        l2_to_l1_map = {
            "layer_norm_fusion": "LayerNormFusion",
            "gelu_fusion": "GeluFusion",
        }
        if python_name in l2_to_l1_map:
            self._enable_ort_name(l2_to_l1_map[python_name])

        # Find and enable the capability itself
        for cap in GRAPH_CAPABILITIES.values():
            if hasattr(cap, "python_name") and cap.python_name == python_name:
                # Special handling for GeluApproximation
                if hasattr(cap, "ort_name") and cap.ort_name == "GeluApproximation":
                    self.enable_gelu_approximation = True
                    # Also remove from disabled list to avoid conflict
                    if "GeluApproximation" in self.disabled_optimizers:
                        self.disabled_optimizers.remove("GeluApproximation")
                elif (
                    hasattr(cap, "ort_name")
                    and cap.ort_name
                    and cap.ort_name in self.disabled_optimizers
                ):
                    self.disabled_optimizers.remove(cap.ort_name)
                break
        return self

    def _enable_ort_name(self, ort_name: str) -> None:
        """Enable an optimizer by its ORT name (internal helper)."""
        if ort_name in self.disabled_optimizers:
            self.disabled_optimizers.remove(ort_name)

    def disable(self, python_name: str) -> ORTGraphPipeConfig:
        """Disable a capability by python_name.

        Args:
            python_name: Capability name in snake_case (e.g., "constant_folding")

        Returns:
            Self for method chaining
        """
        for cap in GRAPH_CAPABILITIES.values():
            if hasattr(cap, "python_name") and cap.python_name == python_name:
                if hasattr(cap, "ort_name") and cap.ort_name:  # noqa: SIM102
                    if cap.ort_name not in self.disabled_optimizers:
                        self.disabled_optimizers.append(cap.ort_name)
                break
        return self


# =============================================================================
# ORTGRAPHPIPE
# =============================================================================


class ORTGraphPipe(BasePipe):
    """Graph optimization pipe using ORT SessionOptions.

    This pipe applies graph-level optimizations by:
    1. Building a disable list from GRAPH capabilities that are disabled
    2. Configuring ORT SessionOptions with level and disable list
    3. Creating an InferenceSession to trigger optimization
    4. Loading and returning the optimized model
    """

    name: ClassVar[str] = "ort_graph"

    # NOTE: Empirical testing (2024-12) verified ORT has NO item limit on
    # disable_specified_optimizers. Testing with 50+ items and 2500+ characters
    # showed no failures. Previous claim of 32-item limit was incorrect.

    # Reference module-level capabilities
    capabilities: ClassVar[dict[str, Any]] = GRAPH_CAPABILITIES

    @classmethod
    def build_config(cls, **kwargs: Any) -> ORTGraphPipeConfig:
        """Build graph pipe config from kwargs.

        All capabilities in GRAPH_CAPABILITIES are default=False (advanced
        optimizations). Users must explicitly enable what they want.

        Behavior:
        - User explicitly enables (True): Capability is enabled
        - User explicitly disables (False): Capability stays disabled
        - No kwargs: Nothing enabled (only ORT basic optimizations run)

        Args:
            **kwargs: User configuration (snake_case keys like gelu_fusion=True)

        Returns:
            ORTGraphPipeConfig with enabled capabilities
        """
        verbose = kwargs.get("verbose", False)

        # Collect explicitly enabled and disabled capabilities
        explicitly_enabled: list[str] = []
        explicitly_disabled: list[str] = []

        for cap in cls.capabilities.values():
            if isinstance(cap, BoolCapability):
                user_value = kwargs.get(cap.python_name)
                if user_value is True:
                    explicitly_enabled.append(cap.python_name)
                elif user_value is False:
                    explicitly_disabled.append(cap.python_name)

        # Determine final enabled list
        if explicitly_enabled:
            # ISOLATION MODE: only explicitly enabled caps run
            enabled = explicitly_enabled
        else:
            # No explicit enables - use defaults, but respect explicit disables
            enabled = [
                cap.python_name
                for cap in cls.capabilities.values()
                if isinstance(cap, BoolCapability)
                and cap.default
                and cap.python_name not in explicitly_disabled
            ]

        config = ORTGraphPipeConfig(enabled=enabled, verbose=verbose)

        # Explicitly disable capabilities that user set to False
        # This handles default=True caps like constant_folding
        for python_name in explicitly_disabled:
            config.disable(python_name)

        # Verbose output for build_config
        if verbose:
            cls._log_build_config_verbose(enabled, config.disabled_optimizers)

        return config

    @classmethod
    def _log_build_config_verbose(
        cls,
        enabled: list[str],
        disabled: list[str],
    ) -> None:
        """Log verbose build_config output."""
        logger.debug("=" * 70)
        logger.debug("ORTGraphPipe BUILD_CONFIG VERBOSE OUTPUT")
        logger.debug("=" * 70)

        # Optimization level
        logger.debug("[Optimization Level]")
        logger.debug("  Level: 2 (ORT_ENABLE_EXTENDED) - Finalized")

        # Enabled capabilities
        logger.debug("[Enabled Capabilities] (%d)", len(enabled))
        if enabled:
            for name in enabled[:10]:
                logger.debug("  [enabled] %s", name)
            if len(enabled) > 10:
                logger.debug("  ... and %d more", len(enabled) - 10)
        else:
            logger.debug("  (none)")

        # Disabled optimizers
        total_caps = len([c for c in cls.capabilities.values() if hasattr(c, "ort_name")])
        logger.debug("[Disabled Optimizers] (%d/%d capabilities)", len(disabled), total_caps)
        if disabled:
            # Group by priority
            fusions = [n for n in disabled if "fusion" in n.lower()]
            transformers = [n for n in disabled if "transformer" in n.lower()]
            others = [n for n in disabled if n not in fusions and n not in transformers]

            if fusions:
                logger.debug("  Fusions (%d):", len(fusions))
                for name in fusions[:10]:
                    logger.debug("    [disabled] %s", name)
                if len(fusions) > 10:
                    logger.debug("    ... and %d more", len(fusions) - 10)

            if transformers:
                logger.debug("  Transformers (%d):", len(transformers))
                for name in transformers[:5]:
                    logger.debug("    [disabled] %s", name)
                if len(transformers) > 5:
                    logger.debug("    ... and %d more", len(transformers) - 5)

            if others:
                logger.debug("  Others (%d):", len(others))
                for name in others[:5]:
                    logger.debug("    [disabled] %s", name)
                if len(others) > 5:
                    logger.debug("    ... and %d more", len(others) - 5)
        else:
            logger.debug("  (none - all optimizers enabled)")

        logger.debug("=" * 70)

    def _log_process_verbose(
        self,
        config: ORTGraphPipeConfig,
        model: Any,
        input_file: Path,
        output_file: Path,
        disable_list: str,
    ) -> None:
        """Log verbose process output showing ORT SessionOptions."""
        import onnxruntime as ort

        logger.debug("=" * 70)
        logger.debug("ORTGraphPipe PROCESS VERBOSE OUTPUT - ORT SESSION OPTIONS")
        logger.debug("=" * 70)

        # Model info
        logger.debug("[Input Model]")
        logger.debug("  Nodes: %d", len(model.graph.node))
        logger.debug("  Inputs: %d", len(model.graph.input))
        logger.debug("  Outputs: %d", len(model.graph.output))
        logger.debug("  Temp file: %s", input_file)

        # Session options
        logger.debug("[ORT SessionOptions]")
        logger.debug(
            "  graph_optimization_level: %d (ORT_ENABLE_EXTENDED)", config.optimization_level
        )
        logger.debug("  optimized_model_filepath: %s", output_file)
        logger.debug("  providers: ['CPUExecutionProvider']")

        # Session config entries
        logger.debug("[Session Config Entries]")
        if disable_list:
            logger.debug("  optimization.disable_specified_optimizers:")
            items = disable_list.split(";")
            for item in items[:8]:
                logger.debug("    - %s", item)
            if len(items) > 8:
                logger.debug("    ... (%d total)", len(items))
        else:
            logger.debug("  (no disabled optimizers)")

        # ORT version
        logger.debug("[ORT Runtime]")
        logger.debug("  Version: %s", ort.__version__)

        logger.debug("=" * 70)

    @classmethod
    def should_process(cls, config: ORTGraphPipeConfig) -> bool:
        """Check if graph optimization should run.

        Args:
            config: Graph pipe configuration

        Returns:
            True if optimization_level > 0, False otherwise
        """
        return config.optimization_level > 0

    def process(self, model: onnx.ModelProto, config: ORTGraphPipeConfig) -> onnx.ModelProto:
        """Apply graph optimizations using ORT SessionOptions.

        This method:
        1. Saves model to temporary file (ORT requires file-based optimization)
        2. Configures SessionOptions with level and disabled optimizers
        3. Creates InferenceSession to trigger optimization
        4. Loads and returns optimized model
        5. Cleans up temporary files

        Args:
            model: Input ONNX model
            config: Graph pipe configuration

        Returns:
            Optimized ONNX model

        Raises:
            OptimizationError: If ORT optimization fails
        """
        # Import onnxruntime here to avoid import errors if not installed
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise OptimizationError(
                "onnxruntime not installed - cannot apply graph optimizations",
                pipe_name=self.name,
                cause=e,
            ) from e

        # Skip processing if optimization level is 0
        if not self.should_process(config):
            return model

        # Create temporary files for optimization
        input_file = None
        output_file = None

        try:
            # Create input temporary file
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                input_file = Path(f.name)
                save_onnx(model, input_file)

            # Create output temporary file
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                output_file = Path(f.name)

            # Configure session options
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel(
                config.optimization_level
            )
            sess_opts.optimized_model_filepath = str(output_file)

            # Configure disabled optimizers if any
            disable_list = ""
            if config.disabled_optimizers:
                disable_list = ";".join(config.disabled_optimizers)
                # CRITICAL: Use correct ORT API key (NOT "session.disable_...")
                sess_opts.add_session_config_entry(
                    "optimization.disable_specified_optimizers",
                    disable_list,
                )

            # Special handling for GeluApproximation
            # ORT requires a separate session config entry to enable this optimizer
            # (it's disabled by default due to potential accuracy impact)
            if config.enable_gelu_approximation:
                sess_opts.add_session_config_entry(
                    "optimization.enable_gelu_approximation",
                    "1",
                )

            # Verbose output for process
            if config.verbose:
                self._log_process_verbose(config, model, input_file, output_file, disable_list)

            # Create session to trigger optimization
            try:
                _ = ort.InferenceSession(
                    str(input_file), sess_opts, providers=["CPUExecutionProvider"]
                )
            except Exception as e:
                raise OptimizationError(
                    f"ONNX Runtime optimization failed: {e}",
                    pipe_name=self.name,
                    model_info={
                        "optimization_level": config.optimization_level,
                        "disabled_count": len(config.disabled_optimizers),
                    },
                    cause=e,
                ) from e

            # Load and return optimized model
            try:
                return load_onnx(output_file, validate=False)
            except Exception as e:
                raise OptimizationError(
                    f"Failed to load optimized model: {e}",
                    pipe_name=self.name,
                    cause=e,
                ) from e

        finally:
            # Clean up temporary files (always execute)
            if input_file and input_file.exists():
                input_file.unlink(missing_ok=True)
            if output_file and output_file.exists():
                output_file.unlink(missing_ok=True)
