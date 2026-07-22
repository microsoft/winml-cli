# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# S110: try-except-pass for optional model info extraction

"""HTP (Hierarchy-preserving Tags Protocol) Exporter.

This exporter preserves the hierarchical structure of HuggingFace models
when converting to ONNX format by tracing module execution and tagging
ONNX nodes with their source module information.

Key Features:
- Direct module context capture during execution
- Precise hierarchy tag generation
- Comprehensive metadata export
- Optional detailed reporting
"""

from __future__ import annotations

import contextlib
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn as nn
from rich.console import Console

from ...core.onnx_node_tagger import (
    DynamoMetadataTagger,
    ONNXNodeTagger,
    create_node_tagger_from_hierarchy,
)
from ...core.onnx_utils import infer_output_names
from .base_writer import ExportStep
from .hierarchy import TracingHierarchyBuilder
from .monitor import HTPExportMonitor


if TYPE_CHECKING:
    import onnx

    from ..config import WinMLExportConfig


logger = logging.getLogger(__name__)


class HTPConfig:
    """Configuration constants for HTP Exporter."""

    # Strategy and file naming
    STRATEGY_NAME = "htp"
    ONNX_EXTENSION = ".onnx"
    REPORT_SUFFIX = "_htp_export_report.txt"
    METADATA_SUFFIX = "_htp_metadata.json"

    # Console and tree formatting
    CONSOLE_WIDTH = 80
    SEPARATOR_LENGTH = 80
    MODULE_TREE_MAX_LINES = 100
    NODE_TREE_MAX_LINES = 30
    TOP_NODES_COUNT = 20

    # Export defaults
    DEFAULT_TASK = "feature-extraction"

    # Default ONNX export configuration
    # QNN-SAFE DEFAULTS: Static batch/shape by default to prevent BiasGelu operator creation
    DEFAULT_EXPORT_CONFIG: ClassVar[dict[str, Any]] = {
        "opset_version": 17,
        "do_constant_folding": True,
        "verbose": False,  # ONNX internal verbose
        # PyTorch dynamo export disabled by default. Use --dynamo flag to enable
        # for rich node metadata (namespace, class_hierarchy, etc.)
        # dynamic_axes: Not set (defaults to None = static dimensions)
        # This prevents dynamic batch which causes MatMulAddFusion failure
    }

    # Default torch.nn modules to include when torch_module=True
    DEFAULT_TORCH_MODULES: ClassVar[list[str]] = [
        "LayerNorm",
        "Embedding",
    ]

    # Default export statistics structure.
    # Initialised before each export run and returned as a copy at the end.
    DEFAULT_EXPORT_STATS: ClassVar[dict[str, Any]] = {
        # Seconds elapsed from export() entry to final stat collection.
        "export_time": 0.0,
        # Number of named hierarchy modules discovered during tracing.
        "hierarchy_modules": 0,
        # Total ONNX graph nodes in the exported model.
        "onnx_nodes": 0,
        # Nodes that received a hierarchy_tag attribute (0 when embed_hierarchy_attributes=False).
        "tagged_nodes": 0,
        # CARDINAL RULE: tags with an empty/whitespace value must never exist.
        # Sentinel sys.maxsize ensures any non-zero value is immediately visible as a violation.
        "empty_tags": sys.maxsize,
        # Percentage of onnx_nodes that were tagged (0.0 when embed_hierarchy_attributes=False).
        "coverage_percentage": 0.0,
        # Exporter strategy identifier, written into report metadata.
        "strategy": STRATEGY_NAME,
    }


class HTPExporter:
    """HTP Exporter with proper verbose console output.

    This implementation properly separates:
    - verbose: Controls console output (8-step format)
    - enable_reporting: Controls report file generation
    """

    def __init__(
        self,
        verbose: bool = False,
        enable_reporting: bool = False,
        embed_hierarchy_attributes: bool = True,
        torch_module: bool | list[str] = False,
    ) -> None:
        """Initialize HTP exporter.

        Args:
            verbose: Enable verbose console output (8-step format)
            enable_reporting: Enable report file generation
            embed_hierarchy_attributes: Whether to embed hierarchy_tag attributes in ONNX
                                       (disabled by --clean-onnx or --no-hierarchy-attrs)
            torch_module: Include torch.nn modules in hierarchy for proper operation
                         attribution (e.g., ResNet).
                         Can be:
                         - False: Don't include any torch.nn modules (default)
                         - True: Include default modules (LayerNorm, Embedding)
                         - List[str]: Include specific torch.nn module types
        """
        self.verbose = verbose
        self.enable_reporting = enable_reporting
        self.embed_hierarchy_attributes = embed_hierarchy_attributes
        self.torch_module = torch_module
        self.strategy = HTPConfig.STRATEGY_NAME

        # Core components
        self._hierarchy_builder: TracingHierarchyBuilder | None = None
        self._node_tagger: ONNXNodeTagger | DynamoMetadataTagger | None = None
        self._hierarchy_data: dict[str, Any] = {}
        self._tagged_nodes: dict[str, str] = {}
        self._tagging_stats: dict[str, Any] = {}
        # Whether to source hierarchy from dynamo node metadata (set in export()).
        self._use_dynamo_hierarchy: bool = False

        # Export statistics
        self._export_stats = HTPConfig.DEFAULT_EXPORT_STATS.copy()

        # Export monitor will be initialized in export()
        self._monitor: HTPExportMonitor | None = None

        # Rich console for tree rendering
        self.console = Console(width=HTPConfig.CONSOLE_WIDTH)

    def export(
        self,
        model: nn.Module | None = None,
        output_path: str = "",
        *,
        export_config: WinMLExportConfig,
        model_name_or_path: str | None = None,
        task: str | None = None,
        enable_operation_fallback: bool = False,
        metadata_filename: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Export model to ONNX with hierarchy-preserving tags.

        Args:
            model: PyTorch model to export. If None, auto-loads from model_name_or_path.
            output_path: Path for the output ONNX file.
            export_config: Export configuration with I/O specs (input_tensors required).
            model_name_or_path: HF model ID for auto-loading and preprocessor lookup.
            task: Task for auto-generating inputs when input_tensors not provided.
            enable_operation_fallback: Enable fallback node tagging.
            metadata_filename: Custom metadata filename.

        Returns:
            Export statistics dict.
        """
        start_time = time.time()

        # Dynamo records the module hierarchy natively on each ONNX node's
        # metadata_props, so select the hierarchy source (and whether to run the
        # TorchScript trace) from the exporter choice.
        self._use_dynamo_hierarchy = bool(export_config.dynamo)

        # Initialize export monitor
        self._monitor = HTPExportMonitor(
            output_path=output_path,
            model_name=model_name_or_path or "unknown",
            verbose=self.verbose,
            enable_report=self.enable_reporting,
            embed_hierarchy=self.embed_hierarchy_attributes,
        )

        # Use monitor as context manager
        with self._monitor as monitor:
            # Auto-load model if needed
            if model is None:
                if model_name_or_path is None:
                    raise ValueError("Either 'model' or 'model_name_or_path' must be provided.")
                from ...loader import load_hf_model

                model, _, _ = load_hf_model(model_name_or_path)

            # Step 1: Model Preparation
            model.eval()

            monitor.update(
                ExportStep.MODEL_PREP,
                model_class=type(model).__name__,
                total_modules=len(list(model.modules())),
                total_parameters=sum(p.numel() for p in model.parameters()),
            )

            # Step 2: Input Generation from export_config
            inputs = export_config.generate_dummy_inputs()

            # Monitor input generation data
            model_config = getattr(model, "config", None)
            input_gen_data: dict[str, Any] = {
                "method": "from_config",
                "model_type": getattr(model_config, "model_type", "pytorch"),
                "task": task or HTPConfig.DEFAULT_TASK,
                "inputs": {
                    name: {"shape": list(t.shape), "dtype": str(t.dtype)}
                    for name, t in inputs.items()
                },
            }
            monitor.update(ExportStep.INPUT_GEN, **input_gen_data)

            # Step 3: Hierarchy Building
            #
            # Under dynamo the exporter records the module hierarchy natively on
            # each ONNX node's metadata_props, so the TorchScript trace is
            # redundant (and can fail for models that only export via dynamo).
            # The hierarchy is recovered post-export from that metadata in Step 5.
            if self._use_dynamo_hierarchy:
                self._hierarchy_data = {}
            else:
                # Trace under the Optimum patcher so models that inject constant
                # forward arguments at export time (e.g. ViTPose MoE's dataset_index)
                # are traced with the same inputs they are exported with. The export
                # in Step 4 re-enters the patcher; the contexts are sequential, not
                # nested.
                with self._get_optimum_patcher(model, task):
                    self._trace_model_hierarchy(model, inputs)

            execution_steps = (
                self._hierarchy_builder.get_execution_summary().get("execution_steps", 0)
                if self._hierarchy_builder
                else 0
            )
            monitor.update(
                ExportStep.HIERARCHY,
                hierarchy=self._hierarchy_data,
                execution_steps=execution_steps,
            )

            # Step 4: ONNX Export
            self._convert_model_to_onnx(model, output_path, inputs, export_config, task=task)

            # The TorchScript ONNX exporter writes one external-data sidecar per
            # initializer (named by tensor). Record them now: the Step 7 tag-injection
            # re-save consolidates all weights into a single ``<model>.onnx.data``, so
            # these per-tensor sidecars must be pruned afterwards or they linger as
            # orphans that roughly double on-disk size.
            from ...onnx import get_external_data_files

            try:
                pre_consolidation_external = get_external_data_files(output_path)
            except Exception:
                pre_consolidation_external = []

            # Verify ONNX export
            self._verify_onnx_export(output_path, export_config)

            # Update monitor with ONNX export info
            onnx_size_mb = (
                round(Path(output_path).stat().st_size / (1024 * 1024), 2)
                if Path(output_path).exists()
                else 0
            )
            traced_outputs = (
                self._hierarchy_builder.get_outputs() if self._hierarchy_builder else None
            )
            output_names = infer_output_names(traced_outputs) if traced_outputs is not None else []
            # Report the opset the exporter actually produced, not the requested
            # one. torch's dynamo exporter targets a minimum opset of 18 and does
            # not always down-convert to a lower requested value (e.g. ResNet stays
            # at 18), so echoing export_config.opset_version would misreport the
            # real graph. Fall back to the request only if the value can't be read.
            actual_opset = self._read_default_opset(output_path)
            if actual_opset is not None and actual_opset != export_config.opset_version:
                logger.warning(
                    "Requested opset %d but the exporter produced opset %d. "
                    "torch's dynamo exporter targets a minimum opset of 18 and "
                    "cannot always lower to a smaller requested version; pass "
                    "--no-dynamo for a natively opset-%d graph.",
                    export_config.opset_version,
                    actual_opset,
                    export_config.opset_version,
                )
            monitor.update(
                ExportStep.ONNX_EXPORT,
                opset_version=(
                    actual_opset if actual_opset is not None else export_config.opset_version
                ),
                do_constant_folding=export_config.do_constant_folding,
                onnx_size_mb=onnx_size_mb,
                output_names=output_names,
            )

            # Step 5: Node Tagger Creation
            from ...onnx import load_onnx

            onnx_model = load_onnx(output_path, validate=False)

            self._initialize_node_tagger(enable_operation_fallback)

            # Tagger creation is part of node tagging process
            # No separate step needed

            # Step 6: Node Tagging
            self._apply_hierarchy_tags(onnx_model)

            # Update monitor with tagging results
            total_nodes = len(onnx_model.graph.node)
            tagged_nodes_count = len(self._tagged_nodes)
            coverage = (tagged_nodes_count / total_nodes * 100.0) if total_nodes > 0 else 0.0

            monitor.update(
                ExportStep.NODE_TAGGING,
                total_nodes=total_nodes,
                tagged_nodes=self._tagged_nodes,
                tagging_stats=self._tagging_stats,
                coverage=coverage,
            )

            # Step 7: Tag Injection + Graph Metadata
            self._embed_graph_metadata(onnx_model, export_config)
            self._embed_tags_in_onnx(output_path, onnx_model, **kwargs)

            # Prune the per-tensor sidecars orphaned by the consolidated re-save
            # so the export leaves only ``model.onnx`` + ``model.onnx.data``.
            self._cleanup_stale_external_data(output_path, pre_consolidation_external)

            # Update monitor
            monitor.update(ExportStep.TAG_INJECTION)

            # Calculate final statistics before metadata generation
            export_time = time.time() - start_time
            self._export_stats["export_time"] = export_time
            self._export_stats["hierarchy_modules"] = len(self._hierarchy_data)
            total_nodes = len(onnx_model.graph.node)
            self._export_stats["onnx_nodes"] = total_nodes

            self._update_tag_stats(total_nodes)

            # Update monitor with actual export time
            monitor.data.export_time = export_time
            # Also update the start_time to ensure elapsed_time is correct
            monitor.data.start_time = start_time

            # Step 8: Metadata Generation
            # Metadata generation is handled by MetadataWriter in monitor
            # No need to call _generate_metadata_file here

        # The monitor's context manager will handle finalization
        return self._export_stats.copy()

    # Internal implementation methods
    def _trace_model_hierarchy(self, model: nn.Module, inputs: dict) -> None:
        """Build hierarchy internally."""
        # Determine if we need torch.nn exceptions for this model
        exceptions = None
        if self.torch_module is True:
            # Use default torch.nn modules from config
            exceptions = HTPConfig.DEFAULT_TORCH_MODULES
        elif isinstance(self.torch_module, list):
            # Use user-provided list of torch.nn modules
            exceptions = self.torch_module
        # If False, exceptions remains None (no torch.nn modules included)

        self._hierarchy_builder = TracingHierarchyBuilder(exceptions=exceptions)

        # Pass inputs to tracer
        input_args = inputs

        self._hierarchy_builder.trace_model_execution(model, input_args)

        summary = self._hierarchy_builder.get_execution_summary()
        self._hierarchy_data = summary["module_hierarchy"]
        self._export_stats["hierarchy_modules"] = len(self._hierarchy_data)

    @staticmethod
    def _read_default_opset(output_path: str) -> int | None:
        """Return the default-domain (ai.onnx) opset of the exported model.

        Reads only the model proto (external weight files are not loaded), so it
        stays cheap for large models. Returns ``None`` if the model cannot be
        read or declares no default-domain opset import.
        """
        import onnx

        try:
            model = onnx.load(output_path, load_external_data=False)
        except Exception:
            return None
        for opset in model.opset_import:
            if opset.domain in ("", "ai.onnx"):
                return opset.version
        return None

    def _verify_onnx_export(
        self,
        output_path: str,
        export_config: WinMLExportConfig | None = None,
    ) -> None:
        """Verify ONNX export succeeded and check for common issues.

        Post-export validation:
        1. Load ONNX model
        2. Run ONNX checker
        3. Verify batch shape and warn only about unexpected dynamic batch

        Args:
            output_path: Path to exported ONNX file.
            export_config: Export configuration used for the export.
        """
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Load and verify ONNX model
            from ...onnx import load_onnx

            model = load_onnx(output_path)
            logger.info("✅ ONNX verification passed")

            # Check for dynamic batch dimension. Static remains the QNN-safe default,
            # but dynamic dimensions are valid when explicitly requested.
            graph_inputs = model.graph.input
            has_dynamic_batch = False
            requested_dynamic_axes: dict[str, dict[int, str]] = (
                export_config.dynamic_axes
                if export_config and export_config.dynamic_axes is not None
                else {}
            )
            for input_tensor in graph_inputs:
                if input_tensor.type.tensor_type.shape.dim:
                    batch_dim = input_tensor.type.tensor_type.shape.dim[0]
                    if batch_dim.dim_param:  # Has symbolic name (dynamic)
                        has_dynamic_batch = True
                        requested_dynamic_batch = 0 in requested_dynamic_axes.get(
                            input_tensor.name, {}
                        )
                        if requested_dynamic_batch:
                            logger.info(
                                "Dynamic batch confirmed for '%s' as requested.",
                                input_tensor.name,
                            )
                        else:
                            logger.warning(
                                "⚠️  Dynamic batch detected in ONNX for '%s'.\n"
                                "   This may cause QNN compatibility issues.\n"
                                "   Use static batch for QNN-targeted optimization paths.",
                                input_tensor.name,
                            )

            if not has_dynamic_batch:
                logger.info("✅ Static batch confirmed (QNN-compatible)")

        except Exception as e:
            logger.warning("⚠️  ONNX verification failed: %s", e)
            logger.warning("   Export completed but validation encountered issues.")
            # Don't raise - export succeeded, just warn about validation

    def _convert_model_to_onnx(
        self,
        model: nn.Module,
        output_path: str,
        inputs: dict,
        export_config: WinMLExportConfig,
        task: str | None = None,
    ) -> None:
        """Export to ONNX using WinMLExportConfig."""
        # Resolve to absolute path so torch.onnx.export writes external data
        # files (.data) next to the model, not in the current working directory.
        output_path = str(Path(output_path).resolve())
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Input names from config, fallback to inputs dict keys
        input_names = export_config.get_input_names() or list(inputs.keys())

        # Output names: infer from traced hierarchy, validate against config
        traced_outputs = self._hierarchy_builder.get_outputs() if self._hierarchy_builder else None
        inferred_names = infer_output_names(traced_outputs) if traced_outputs is not None else []
        output_names = export_config.get_output_names()

        if output_names and inferred_names and len(output_names) != len(inferred_names):
            logger.warning(
                "Output names count mismatch: config has %d %s, "
                "model trace inferred %d %s. "
                "Keeping config names (from Optimum OnnxConfig).",
                len(output_names),
                output_names,
                len(inferred_names),
                inferred_names,
            )
            # Trust config output names — they come from Optimum's OnnxConfig
            # which knows the actual ONNX graph outputs. infer_output_names
            # may return extra fields from the model's dataclass output.
        elif not output_names and inferred_names:
            output_names = inferred_names

        # Build kwargs for torch.onnx.export
        onnx_kwargs: dict[str, Any] = {
            "opset_version": export_config.opset_version,
            "do_constant_folding": export_config.do_constant_folding,
            "export_params": export_config.export_params,
        }
        # Always explicitly set dynamo — PyTorch 2.10+ defaults to True
        onnx_kwargs["dynamo"] = bool(export_config.dynamo)
        if onnx_kwargs["dynamo"]:
            # torch's dynamo exporter prints capture progress with emoji glyphs
            # (e.g. ✅) that raise UnicodeEncodeError on non-UTF-8 consoles
            # (Windows cp1252). winml drives its own progress UI, so silence
            # torch's verbose printing to keep exports robust across terminals.
            onnx_kwargs["verbose"] = False
        if input_names:
            onnx_kwargs["input_names"] = input_names
        if output_names:
            onnx_kwargs["output_names"] = output_names
        if export_config.dynamic_axes:
            onnx_kwargs["dynamic_axes"] = export_config.dynamic_axes

        with self._get_optimum_patcher(model, task):
            # Models can override input binding by implementing
            # get_export_args(inputs) → tuple of positional args.
            # Default: pass inputs dict as kwargs.
            if hasattr(model, "get_export_args"):
                # hasattr-gated optional protocol; not in nn.Module's static type.
                export_args = model.get_export_args(inputs)  # type: ignore[operator]
                torch.onnx.export(model, export_args, output_path, **onnx_kwargs)
            else:
                torch.onnx.export(model, (), output_path, kwargs=inputs, **onnx_kwargs)

    @staticmethod
    def _get_optimum_patcher(model: nn.Module, task: str | None) -> Any:
        """Get Optimum's model patcher for TorchScript tracing compatibility.

        Optimum patches models to fix tracing issues introduced in
        Transformers 4.53+ (vmap masking, DynamicCache, packed sequences).
        Falls back to no-op context if unavailable.
        """
        try:
            import optimum.exporters.onnx.model_configs  # noqa: F401
            from optimum.exporters.tasks import TasksManager
        except ImportError:
            logger.debug("Optimum not available; skipping model patcher.")
            return contextlib.nullcontext()

        model_config = getattr(model, "config", None)
        model_type = getattr(model_config, "model_type", None) if model_config else None
        if not model_type:
            logger.debug("Model has no config.model_type; skipping Optimum patcher.")
            return contextlib.nullcontext()
        if task is None:
            logger.debug("No task provided; skipping Optimum patcher.")
            return contextlib.nullcontext()

        # TasksManager expects Optimum-canonical task names
        from ...loader import to_optimum_task

        try:
            cfg_cls = TasksManager.get_exporter_config_constructor(
                "onnx",
                model_type=model_type,
                task=to_optimum_task(task),
                library_name="transformers",
            )
            # Pass an explicit empty model_kwargs so patchers that inject extra
            # forward arguments can populate it. Some patchers (e.g. ViTPose MoE,
            # which sets a constant dataset_index) assume a mutable dict and crash
            # on the None default from patch_model_for_export.
            return cfg_cls(model_config).patch_model_for_export(model, model_kwargs={})
        except KeyError:
            logger.debug(
                "Model type '%s' (task='%s') not in Optimum registry; "
                "exporting without Optimum patcher.",
                model_type,
                task,
            )
            return contextlib.nullcontext()
        except Exception:
            logger.warning(
                "Optimum model patcher failed for model_type='%s', task='%s'. "
                "Export may produce incorrect results for models requiring "
                "Transformers 4.53+ tracing patches.",
                model_type,
                task,
                exc_info=True,
            )
            return contextlib.nullcontext()

    def _update_tag_stats(self, total_nodes: int) -> None:
        """Update tagged_nodes, empty_tags, and coverage_percentage in export stats.

        Centralises the embed-aware calculation so _apply_hierarchy_tags and
        the final stats block in export() always stay in sync.
        All three stats are gated on embed_hierarchy_attributes: when hierarchy
        embedding is disabled none of the tags are written to the model, so
        all counts are reported as 0.
        """
        if self.embed_hierarchy_attributes:
            embedded_count = len(self._tagged_nodes)
            empty_tags = sum(1 for tag in self._tagged_nodes.values() if not tag or not tag.strip())
        else:
            embedded_count = 0
            empty_tags = 0
        self._export_stats["tagged_nodes"] = embedded_count
        self._export_stats["empty_tags"] = empty_tags
        self._export_stats["coverage_percentage"] = (
            embedded_count / total_nodes * 100.0 if total_nodes > 0 else 0.0
        )

    def _initialize_node_tagger(self, enable_operation_fallback: bool) -> None:
        """Create node tagger internally.

        Under dynamo the hierarchy comes from each node's dynamo metadata; the
        legacy path builds a tagger from the TorchScript trace hierarchy.
        """
        if self._use_dynamo_hierarchy:
            self._node_tagger = DynamoMetadataTagger()
        else:
            self._node_tagger = create_node_tagger_from_hierarchy(
                self._hierarchy_data, enable_operation_fallback=enable_operation_fallback
            )

    def _apply_hierarchy_tags(self, onnx_model: onnx.ModelProto) -> None:
        """Tag nodes internally."""
        assert self._node_tagger is not None, (
            "_apply_hierarchy_tags called before _initialize_node_tagger"
        )
        # Store ONNX model for later use in displaying operations
        self._onnx_model = onnx_model
        self._tagged_nodes = self._node_tagger.tag_all_nodes(onnx_model)

        # Get statistics
        stats = self._node_tagger.get_tagging_statistics(onnx_model)
        self._tagging_stats = stats

        # Update export stats
        total_nodes = len(onnx_model.graph.node)
        self._export_stats["onnx_nodes"] = total_nodes
        self._update_tag_stats(total_nodes)

    def _embed_graph_metadata(
        self, onnx_model: onnx.ModelProto, export_config: WinMLExportConfig
    ) -> None:
        """Embed winml.io.inputs/outputs in ONNX model-level metadata_props.

        Writes InputTensorSpec/OutputTensorSpec as JSON arrays following the
        WinML Graph Metadata Spec (section 5.4). Includes value_range when
        available for calibration data generation.
        """
        import json

        if export_config.input_tensors:
            io_inputs = [spec.to_dict() for spec in export_config.input_tensors]
            onnx_model.metadata_props.add(
                key="winml.io.inputs",
                value=json.dumps(io_inputs),
            )
            logger.debug("Embedded winml.io.inputs for %d tensors", len(io_inputs))

        if export_config.output_tensors:
            io_outputs = [spec.to_dict() for spec in export_config.output_tensors]
            onnx_model.metadata_props.add(
                key="winml.io.outputs",
                value=json.dumps(io_outputs),
            )

    def _embed_tags_in_onnx(
        self,
        output_path: str,
        onnx_model: onnx.ModelProto,
        **kwargs: Any,
    ) -> None:
        """Inject hierarchy tags into ONNX node metadata_props.

        Tags are stored in node.metadata_props (not node.attribute) to align with
        PyTorch 2.9+ dynamo export which adds rich metadata like namespace,
        class_hierarchy, fx_node, etc. to the same location.

        Metadata keys (winml namespace):
        - winml.hierarchy.tag: Full hierarchy path (e.g., "/Model/Encoder/Layer.0")
        - winml.hierarchy.depth: Depth in hierarchy (e.g., "3" for 3 path segments)
        """
        if self.embed_hierarchy_attributes:
            # Add hierarchy tags as node metadata_props (not attributes)
            for node in onnx_model.graph.node:
                node_name = node.name or f"{node.op_type}_{id(node)}"
                if node_name in self._tagged_nodes:
                    tag = self._tagged_nodes[node_name]
                    # Calculate depth from tag path segments
                    # e.g., "/Model/Encoder/Layer" -> depth = 3
                    depth = len([p for p in tag.split("/") if p])
                    # Use metadata_props with winml namespace
                    node.metadata_props.add(key="winml.hierarchy.tag", value=tag)
                    node.metadata_props.add(key="winml.hierarchy.depth", value=str(depth))

        # Always save — graph metadata (winml.io.*) and/or hierarchy tags may have been added
        from ...onnx import save_onnx

        save_onnx(onnx_model, output_path)

    @staticmethod
    def _cleanup_stale_external_data(output_path: str, previous_external_files: list[str]) -> None:
        """Delete per-tensor external-data sidecars orphaned by consolidation.

        ``previous_external_files`` are the sidecars written by the raw exporter
        (relative filenames). After the consolidated re-save, any of them no longer
        referenced by the model on disk are pure orphans (their weights now live in
        the single ``<model>.onnx.data``) and are removed. Files still referenced —
        e.g. the consolidated sidecar itself — are left untouched.
        """
        from ...onnx import get_external_data_files

        out = Path(output_path)
        try:
            still_referenced = set(get_external_data_files(output_path))
        except Exception:
            # If the final model can't be inspected, keep every sidecar rather than
            # risk deleting data the model still points at.
            return

        for name in previous_external_files:
            if name in still_referenced:
                continue
            stale = out.parent / name
            try:
                if stale.is_file():
                    stale.unlink()
                    logger.debug("Removed orphaned external-data sidecar: %s", stale)
            except OSError:
                logger.debug(
                    "Could not remove orphaned external-data sidecar: %s", stale, exc_info=True
                )
