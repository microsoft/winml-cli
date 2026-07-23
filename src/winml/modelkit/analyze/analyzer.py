# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNXStaticAnalyzer - Main API for ONNX model runtime support analysis.

Public API:
    analyze_onnx() — Flat functional API returning lint + autoconf results.
    ONNXStaticAnalyzer — Class-based API for advanced use cases.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..optim.config import WinMLOptimizationConfig
from ..utils.constants import EP_SUPPORTED_DEVICES, EPName, EPNameOrAlias, normalize_ep_name
from .models.information import Information
from .models.output import RuntimeDebugSummaryEntry
from .models.support_level import SupportLevel
from .utils.timing_utils import make_timing_logger


if TYPE_CHECKING:
    from collections.abc import Callable

    import onnx

    from .models.information import Action
    from .models.output import AnalysisOutput
    from .models.runtime_checks import PatternRuntime, RuntimeTestResult


@dataclass
class LintResult:
    """Lint-style result summarizing errors, warnings, and informational items.

    Attributes:
        errors: Count of unsupported patterns (blocking errors)
        warnings: Count of partial patterns (warnings/optimization opportunities)
        info: Count of information items
        passed: True if no errors and no warnings exist (errors == 0 and warnings == 0)
        error_patterns: List of unsupported pattern IDs (blocking errors)
        warning_patterns: List of partial pattern IDs (warnings/optimizations)
        information: List of information items
        optimization_config: WinML optimization configuration based on detected patterns
    """

    errors: int
    warnings: int
    info: int
    passed: bool
    error_patterns: list[str]
    warning_patterns: list[str]
    information: list[Information]
    optimization_config: WinMLOptimizationConfig


logger = logging.getLogger(__name__)
_log_timing = make_timing_logger(logger)

_RUNTIME_DEBUG_SUMMARY_LEVELS: tuple[SupportLevel, ...] = (
    SupportLevel.UNSUPPORTED,
    SupportLevel.PARTIAL,
    SupportLevel.SUPPORTED,
)


def _normalize_case_indices_for_summary(case_indices: Any) -> list[Any] | None:
    """Normalize case_indices to JSON-friendly list values."""
    if case_indices is None:
        return None
    if isinstance(case_indices, list):
        return case_indices
    if isinstance(case_indices, tuple):
        return list(case_indices)
    return [case_indices]


def _iter_runtime_test_results(pattern_runtime: PatternRuntime) -> list[RuntimeTestResult]:
    """Iterate all RuntimeTestResult objects reachable from PatternRuntime."""
    results = [pattern_runtime.result]
    results.extend(alternative.result for alternative in pattern_runtime.alternatives)
    return results


def _build_runtime_debug_details_summary(
    runtime_summary: dict[str, list[PatternRuntime]],
) -> dict[str, list[str] | dict[str, RuntimeDebugSummaryEntry]] | None:
    """Build debug_details summary grouped by support level and node stable key.

    The returned dict always starts with the ``unknown`` key (in output order),
    followed by ``unsupported``, ``partial``, and ``supported``. ``unknown``
    nodes have no matched rule case data, so they are recorded as a plain list
    of ``node_stable_key`` values; the other levels map ``node_stable_key`` to a
    :class:`RuntimeDebugSummaryEntry`.
    """
    leveled_summary: dict[str, dict[str, RuntimeDebugSummaryEntry]] = {
        level.value: {} for level in _RUNTIME_DEBUG_SUMMARY_LEVELS
    }
    unknown_nodes: set[str] = set()

    for runtime_key in ("op_runtime_check_result", "subgraph_runtime_check_result"):
        for pattern_runtime in runtime_summary.get(runtime_key, []):
            for test_result in _iter_runtime_test_results(pattern_runtime):
                level = test_result.classification

                debug_details = test_result.debug_details
                if not debug_details:
                    continue

                node_stable_key = debug_details.get("node_stable_key")
                if not node_stable_key:
                    continue

                if level == SupportLevel.UNKNOWN:
                    # Unknown nodes carry no rule case data; record the
                    # de-duplicated node key only.
                    unknown_nodes.add(node_stable_key)
                    continue

                if level not in _RUNTIME_DEBUG_SUMMARY_LEVELS:
                    continue

                candidate_entry = RuntimeDebugSummaryEntry(
                    case_indices=_normalize_case_indices_for_summary(
                        debug_details.get("case_indices")
                    ),
                    table_path=debug_details.get("table_path"),
                    table_file=debug_details.get("table_file"),
                )

                level_bucket = leveled_summary[level.value]
                existing_entry = level_bucket.get(node_stable_key)
                if existing_entry is None:
                    level_bucket[node_stable_key] = candidate_entry
                    continue

                if existing_entry.case_indices is None and candidate_entry.case_indices is not None:
                    existing_entry.case_indices = candidate_entry.case_indices

                if existing_entry.table_path is None and candidate_entry.table_path is not None:
                    existing_entry.table_path = candidate_entry.table_path

                if existing_entry.table_file is None and candidate_entry.table_file is not None:
                    existing_entry.table_file = candidate_entry.table_file

    has_any_entry = bool(unknown_nodes) or any(
        leveled_summary[level.value] for level in _RUNTIME_DEBUG_SUMMARY_LEVELS
    )
    if not has_any_entry:
        return None

    # "unknown" is intentionally the first key in output order.
    summary: dict[str, list[str] | dict[str, RuntimeDebugSummaryEntry]] = {
        "unknown": sorted(unknown_nodes)
    }
    summary.update(leveled_summary)
    return summary


def _append_pattern_debug_log(
    *,
    output_path: Path,
    model_path: str,
    ep: str | None,
    device: str,
    extracted_count: int,
    model_signature: str,
    extraction_total_ms: int,
    source_stats: list[dict[str, Any]],
    pattern_matches_by_source: dict[str, dict[str, list[Any]]],
    merge_prep: list[dict[str, Any]] | None = None,
) -> None:
    """Append pattern-extractor debug details to a dedicated file.

    This is a temporary, explicit sink for pattern debug signals introduced by
    the recent pattern extractor changes. It intentionally bypasses global
    logger level filtering.
    """

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    lines: list[str] = [
        f"[{timestamp}] model={model_path} ep={ep} device={device}",
        f"Extracted {extracted_count} patterns",
        f"Pattern extractor debug: model_signature={model_signature} total_ms={extraction_total_ms}",
    ]

    for source_stat in source_stats:
        source = str(source_stat.get("source", ""))
        cache_hit = source_stat.get("cache_hit", False)
        pattern_class_count = source_stat.get("pattern_class_count", 0)
        match_count = source_stat.get("match_count", 0)
        elapsed_ms = source_stat.get("elapsed_ms", 0)

        lines.append(
            "Pattern extractor source="
            f"{source} cache_hit={cache_hit} pattern_class_count={pattern_class_count} "
            f"match_count={match_count} elapsed_ms={elapsed_ms}"
        )

        source_group = pattern_matches_by_source.get(source, {})
        lines.append(
            "Pattern extractor source="
            f"{source} matched pattern classes={sorted(source_group.keys())}"
        )

    merge_prep_entries = merge_prep or []
    lines.append(f"Pattern merge prep entries={len(merge_prep_entries)}")
    for entry in merge_prep_entries:
        source = str(entry.get("source", ""))
        pattern_class = str(entry.get("pattern_class", ""))
        pattern_id = str(entry.get("pattern_id", ""))
        match_count = int(entry.get("match_count", 0))
        match_index = int(entry.get("match_index", 0))
        match_id = str(entry.get("match_id", ""))
        matched_node_keys = entry.get("matched_node_keys", []) or []
        alternatives = entry.get("alternatives", []) or []
        lines.append(
            "Pattern merge prep "
            f"source={source} pattern_class={pattern_class} pattern_id={pattern_id} "
            f"match_count={match_count} match_index={match_index} match_id={match_id} "
            f"matched_nodes={matched_node_keys} alternatives_count={len(alternatives)}"
        )
        lines.append(f"Pattern merge prep alternatives={alternatives}")

        for candidate in entry.get("candidates", []) or []:
            debug_details = candidate.get("debug_details")
            debug_details_text = (
                json.dumps(debug_details, ensure_ascii=True, separators=(",", ":"), default=str)
                if debug_details is not None
                else "None"
            )
            lines.append(
                "Pattern merge prep candidate "
                f"pattern_class={candidate.get('pattern_class')} "
                f"pattern_id={candidate.get('pattern_id')} "
                f"is_alternative={candidate.get('is_alternative')} "
                f"status={candidate.get('status')} "
                f"mismatch_error={candidate.get('mismatch_error')} "
                f"compile={candidate.get('compile')} "
                f"run={candidate.get('run')} "
                f"row_count={candidate.get('row_count')} "
                f"compile_true_rows={candidate.get('compile_true_rows')} "
                f"run_true_rows={candidate.get('run_true_rows')} "
                f"case_indices={candidate.get('case_indices')} "
                f"query_condition_count={candidate.get('query_condition_count')} "
                f"query_condition_keys={candidate.get('query_condition_keys')} "
                f"table_file={candidate.get('table_file')} "
                f"domain={candidate.get('domain')} "
                f"opset_version={candidate.get('opset_version')} "
                f"debug_details={debug_details_text}"
            )

    lines.append("")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        logger.debug("Failed to append pattern debug log: %s", output_path, exc_info=True)


class AnalysisResult:
    """Analysis result wrapper containing the output and additional metadata.

    Attributes:
        output: The analysis output with model metadata and results
    """

    def __init__(
        self,
        output: AnalysisOutput,
    ) -> None:
        """Initialize analysis result.

        Args:
            output: The analysis output
        """
        self.output = output

    def __repr__(self) -> str:
        """String representation of analysis result."""
        pattern_count = sum(self.output.metadata.detected_pattern_count.values())
        return f"AnalysisResult(patterns={pattern_count})"

    def is_fully_supported(self, ep: EPNameOrAlias | None = None) -> bool:
        """Check if model is fully supported on the target EP and device.

        Args:
            ep: Optional execution provider to filter by (e.g., "QNNExecutionProvider").
                If None, checks if all EPs in results are fully supported.

        Returns:
            bool: True if all operators are supported (fully supported)

        Example:
            >>> result = analyzer.analyze(
            ...     "model.onnx",
            ...     ep="QNNExecutionProvider",
            ...     device="NPU"
            ... )
            >>> if result.is_fully_supported("QNNExecutionProvider"):
            ...     print("Deploy to QNN NPU")

            >>> # Check all EPs
            >>> if result.is_fully_supported():
            ...     print("Model supported on all analyzed EPs")
        """
        # Check if we have any results
        if not self.output.results:
            return False

        # Normalize EP if specified
        ep_normalized = normalize_ep_name(ep) if ep else None

        # Track if we found the target EP when filtering
        found_target = ep_normalized is None  # True if not filtering

        for ep_support in self.output.results:
            if ep_normalized and ep_support.ep_type != ep_normalized:
                continue
            found_target = True
            if not ep_support.runtime_support:
                return False
        return found_target

    def has_errors(self, ep: EPNameOrAlias | None = None) -> bool:
        """Check if there are any unsupported patterns (blocking errors).

        Args:
            ep: Optional execution provider to filter by (e.g., "QNNExecutionProvider").
                If None, checks if any EP in results has errors.

        Returns:
            bool: True if unsupported patterns exist (model has blocking errors)

        Example:
            >>> result = analyzer.analyze(
            ...     "model.onnx",
            ...     ep="QNNExecutionProvider",
            ...     device="NPU"
            ... )
            >>> if result.has_errors("QNNExecutionProvider"):
            ...     print("Model has blocking errors on QNN NPU")
        """
        # Check if we have any results
        if not self.output.results:
            return False

        # Normalize EP if specified
        ep_normalized = normalize_ep_name(ep) if ep else None

        for ep_support in self.output.results:
            if ep_normalized and ep_support.ep_type != ep_normalized:
                continue
            if ep_support.has_errors:
                return True
        return False

    def has_warnings(self, ep: EPNameOrAlias | None = None) -> bool:
        """Check if there are any partial patterns (warnings/optimization opportunities).

        Args:
            ep: Optional execution provider to filter by (e.g., "QNNExecutionProvider").
                If None, checks if any EP in results has warnings.

        Returns:
            bool: True if partial patterns exist (model has warnings)

        Example:
            >>> result = analyzer.analyze(
            ...     "model.onnx",
            ...     ep="QNNExecutionProvider",
            ...     device="NPU"
            ... )
            >>> if result.has_warnings("QNNExecutionProvider"):
            ...     print("Model has optimization opportunities on QNN NPU")
        """
        # Check if we have any results
        if not self.output.results:
            return False

        # Normalize EP if specified
        ep_normalized = normalize_ep_name(ep) if ep else None

        for ep_support in self.output.results:
            if ep_normalized and ep_support.ep_type != ep_normalized:
                continue
            if ep_support.has_warnings:
                return True
        return False

    def get_lint_result(self, ep: EPNameOrAlias | None = None) -> LintResult:
        """Get lint-style result with error/warning/info counts.

        Args:
            ep: Optional execution provider to filter by (e.g., "QNNExecutionProvider").
                If None, aggregates counts from all EPs in results.

        Returns:
            LintResult: Lint result with counts, lists, and pass/fail status

        Example:
            >>> result = analyzer.analyze(
            ...     "model.onnx",
            ...     ep="QNNExecutionProvider",
            ...     device="NPU"
            ... )
            >>> lint = result.get_lint_result("QNNExecutionProvider")
            >>> print(f"Errors: {lint.errors}")
            >>> print(f"Warnings: {lint.warnings}")
            >>> print(f"Info: {lint.info}")
            >>> print(f"Passed: {lint.passed}")
            >>> for pattern_id in lint.error_patterns:
            ...     print(f"Error pattern: {pattern_id}")
            >>> print(f"GELU fusion: {lint.optimization_config.get('gelu_fusion')}")
        """
        # Check if we have any results
        if not self.output.results:
            return LintResult(
                errors=0,
                warnings=0,
                info=0,
                passed=True,
                error_patterns=[],
                warning_patterns=[],
                information=[],
                optimization_config=WinMLOptimizationConfig(),
            )

        # Normalize EP if specified
        ep_normalized = normalize_ep_name(ep) if ep else None

        # Aggregate counts and lists
        error_patterns: list[str] = []
        warning_patterns: list[str] = []
        information_list: list[Information] = []

        for ep_support in self.output.results:
            if ep_normalized and ep_support.ep_type != ep_normalized:
                continue

            # Collect unsupported patterns (errors)
            error_patterns.extend(ep_support.classification.get(SupportLevel.UNSUPPORTED, []))

            # Collect partial patterns (warnings)
            warning_patterns.extend(ep_support.classification.get(SupportLevel.PARTIAL, []))

            # Collect information items
            information_list.extend(ep_support.information)

        # Calculate counts
        errors = len(error_patterns)
        warnings = len(warning_patterns)
        info = len(information_list)

        # Passed if no errors and no warnings
        passed = errors == 0 and warnings == 0

        # Generate optimization config
        optimization_config = self.get_optimization_config(ep=ep)

        return LintResult(
            errors=errors,
            warnings=warnings,
            info=info,
            passed=passed,
            error_patterns=error_patterns,
            warning_patterns=warning_patterns,
            information=information_list,
            optimization_config=optimization_config,
        )

    def get_unsupported_operators(self, ep: EPNameOrAlias | None = None) -> list[str]:
        """Get list of unsupported operators for the target EP and device.

        Args:
            ep: Optional execution provider to filter by (e.g., "QNNExecutionProvider").
                If None, returns unsupported operators for all EPs in results.

        Returns:
            list[str]: List of UNSUPPORTED or PARTIAL classified operator names

        Example:
            >>> result = analyzer.analyze(
            ...     "model.onnx",
            ...     ep="QNNExecutionProvider",
            ...     device="NPU"
            ... )
            >>> unsupported = result.get_unsupported_operators("QNNExecutionProvider")
            >>> for op_name in unsupported:
            ...     print(f"Unsupported: {op_name}")
        """
        # Normalize EP if specified
        ep_normalized = normalize_ep_name(ep) if ep else None

        unsupported = []
        for ep_support in self.output.results:
            # Skip if filtering by EP and this isn't the target EP
            if ep_normalized and ep_support.ep_type != ep_normalized:
                continue

            # Collect from classification
            unsupported.extend(ep_support.classification.get(SupportLevel.PARTIAL, []))
            unsupported.extend(ep_support.classification.get(SupportLevel.UNSUPPORTED, []))

        return unsupported

    def get_optimization_opportunities(self, ep: EPNameOrAlias | None = None) -> list[Action]:
        """Get actions for patterns that could be optimized (UNSUPPORTED or PARTIAL status).

        Args:
            ep: Optional execution provider to filter by (e.g., "QNNExecutionProvider").
                If None, returns actions for all EPs in results (deduplicated).

        Returns:
            list[Action]: List of actions for unsupported or partial classified patterns.
                         When ep=None, actions are deduplicated by pattern_from_id and
                         pattern_to_id.

        Example:
            >>> result = analyzer.analyze(
            ...     "model.onnx",
            ...     ep="QNNExecutionProvider",
            ...     driver="NPU"
            ... )
            >>> actions = result.get_optimization_opportunities("QNNExecutionProvider")
            >>> for action in actions:
            ...     print(f"Optimize: {action.pattern_from_id} -> {action.action}")
        """
        # Normalize EP if specified
        ep_normalized = normalize_ep_name(ep) if ep else None

        actions: list[Action] = []
        seen_patterns: set[tuple[str, str]] = set()

        for ep_support in self.output.results:
            # Skip if filtering by EP and this isn't the target EP
            if ep_normalized and ep_support.ep_type != ep_normalized:
                continue

            for info in ep_support.information:
                if info.actions:
                    for action in info.actions:
                        # Deduplicate when merging multiple EPs
                        pattern_key = (action.pattern_from_id, action.pattern_to_id)
                        if pattern_key not in seen_patterns:
                            actions.append(action)
                            seen_patterns.add(pattern_key)
        return actions

    def get_optimization_config(self, ep: EPNameOrAlias | None = None) -> WinMLOptimizationConfig:
        """Generate WinML optimization configuration based on action items.

        This method extracts optimization settings from action_items in Actions,
        reading the optimization_options dictionary to determine which fusion
        passes should be enabled.

        Args:
            ep: Optional execution provider to filter by (e.g., "QNNExecutionProvider").
                If None, uses actions from all EPs in results.

        Returns:
            WinMLOptimizationConfig: Dict-like optimization configuration with fusion flags.

        Example:
            >>> result = analyzer.analyze(
            ...     "model.onnx",
            ...     ep="QNNExecutionProvider",
            ...     device="NPU"
            ... )
            >>> optim = result.get_optimization_config("QNNExecutionProvider")
            >>> print(f"GELU fusion: {optim.get('gelu_fusion', False)}")
            >>> print(f"LayerNorm fusion: {optim.get('layer_norm_fusion', False)}")
            >>> print(f"MatMul+Add fusion: {optim.get('matmul_add_fusion', False)}")

        Action Item Format:
            ActionItem(
                type="GraphOptimization",
                optimization_options={
                    "gelu_fusion": True,
                    "layer_norm_fusion": True,
                    "matmul_add_fusion": True,
                }
            )
        """
        # Get all actions for the specified EP
        actions = self.get_optimization_opportunities(ep=ep)

        # Collect all optimization options from action items
        optim_options: dict[str, bool] = {}
        for action in actions:
            for action_item in action.action_items:
                # Only process GraphOptimization type
                if action_item.type != "GraphOptimization":
                    continue

                if action_item.optimization_options:
                    # Normalize kebab-case keys to snake_case (python_name)
                    # so they match the capability system's python_name format.
                    for key, value in action_item.optimization_options.items():
                        optim_options[key.replace("-", "_")] = value

        # Create and return config from collected options
        return WinMLOptimizationConfig(**optim_options)

    def to_json(self) -> str:
        """Export result as JSON string.

        Returns:
            str: JSON representation of analysis result

        Example:
            >>> result = analyzer.analyze("model.onnx")
            >>> json_output = result.to_json()
            >>> with open("result.json", "w") as f:
            ...     f.write(json_output)
        """
        return self.output.model_dump_json(indent=2)

    def to_dict(self) -> dict:
        """Export result as dictionary.

        Returns:
            dict: Dictionary representation

        Example:
            >>> result = analyzer.analyze("model.onnx")
            >>> data = result.to_dict()
            >>> print(data["metadata"]["opset_version"])
        """
        return self.output.model_dump()


@dataclass
class AnalyzerConfig:
    """Static analyzer configuration.

    Attributes:
        enable_information: Generate recommendations
        pattern_detection_timeout: Max seconds for pattern detection
        max_memory_mb: Memory limit in MB
        rule_database_path: Custom rule database path
    """

    enable_information: bool = False
    pattern_detection_timeout: int = 300
    max_memory_mb: int = 2048
    rule_database_path: str | None = None


class ONNXStaticAnalyzer:
    """Analyze ONNX models for runtime support.

    Main entry point for ONNX model analysis. Provides static analysis
    capabilities to determine runtime support across NPU execution providers.

    Attributes:
        config: Analyzer configuration
        loader: ONNX model loader
        pattern_extractor: Pattern detection engine
        runtime_checker: Runtime support checker
        information_engine: Recommendation generator
        output_aggregator: Results aggregator
    """

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        """Initialize static analyzer.

        Args:
            config: Optional analyzer configuration
                If None, uses default configuration

        Example:
            >>> analyzer = ONNXStaticAnalyzer()
            >>> # With custom config
            >>> config = AnalyzerConfig(enable_information=True)
            >>> analyzer = ONNXStaticAnalyzer(config=config)
        """
        from .core.information_engine import InformationEngine
        from .core.output_aggregator import OutputAggregator

        self.config = config or AnalyzerConfig()

        # Initialize core components
        self.information_engine_cls = InformationEngine
        self.output_aggregator = OutputAggregator()

        logger.info("Initialized ONNXStaticAnalyzer with config: %s", self.config)

    def analyze(
        self,
        model_path: str,
        ep: EPNameOrAlias | None = None,
        device: str | None = None,
        enable_information: bool = True,
        htp_metadata_path: str | None = None,
        for_debug: bool = False,
        pattern_debug_log_path: str | Path | None = None,
        run_unknown_op: bool = False,
        save_node_types: set[str] | None = None,
        on_node_result: Callable | None = None,
        on_ep_start: Callable | None = None,
    ) -> AnalysisResult:
        """Analyze ONNX model for runtime support.

        Performs complete analysis pipeline:
        1. Load and validate ONNX model
        2. Extract operator and subgraph patterns
        3. Check runtime support against rule database
        4. Generate recommendations (if enabled)

        Args:
            model_path: Path to ONNX model file
            ep: Target execution provider (e.g., "QNNExecutionProvider",
                "OpenVINOExecutionProvider", "VitisAIExecutionProvider").
                Also supports aliases: "qnn", "openvino", "vitisai".
                If None, analyzes all supported EPs.
            device: Device type (e.g., "CPU", "GPU", "NPU").
                If None, uses "NPU" as default.
            enable_information: Whether to generate recommendations
                Default: True
            htp_metadata_path: Optional path to HTP metadata JSON file
                for pattern extraction from hierarchy traces
            for_debug: Whether to include runtime debug payloads in check results.
                Default: False
            pattern_debug_log_path: Optional path for temporary pattern debug
                file sink. When set with ``for_debug=True``, the recent
                pattern-extractor debug signals are appended to this file.
            run_unknown_op: Whether to run unknown operators on the local machine
                if possible. Default: True
            save_node_types: Set of node types to save for further analysis
                (e.g., {"partial", "unsupported"}). Default: None (save nothing)

        Returns:
            AnalysisResult: Analysis result wrapper containing:
            - output: AnalysisOutput with metadata, results, and information

        Raises:
            FileNotFoundError: If model file doesn't exist
            onnx.checker.ValidationError: If model is invalid ONNX
            RuntimeError: If analysis fails

        Example:
            >>> analyzer = ONNXStaticAnalyzer()
            >>> result = analyzer.analyze(
            ...     "resnet50.onnx",
            ...     ep="QNNExecutionProvider",
            ...     device="NPU"
            ... )
            >>> print(f"Opset: {result.output.metadata.opset_version}")
            >>> print(f"Total ops: {result.output.metadata.total_operators}")

            >>> # Using EP alias
            >>> result = analyzer.analyze(
            ...     "model.onnx",
            ...     ep="openvino",  # Short for OpenVINOExecutionProvider
            ...     device="GPU"
            ... )

            >>> # With recommendations and model validation
            >>> result = analyzer.analyze(
            ...     "model.onnx",
            ...     ep="qnn",
            ...     device="NPU",
            ...     enable_information=True
            ... )
            >>> for info in result.output.results[0].information:
            ...     print(f"{info.pattern_id}: {info.explanation}")

        Note:
            Analysis time depends on model size. See Performance section in docs.
        """
        import onnx

        total_start = time.perf_counter()

        # Normalize EP name (convert aliases to full names)
        ep_normalized = normalize_ep_name(ep)
        if ep != ep_normalized:
            logger.debug("EP alias '%s' normalized to '%s'", ep, ep_normalized)

        # Validate model path
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info("Starting analysis for model: %s", model_path)
        logger.info("Target: %s on %s", ep_normalized, device)

        # Load ONNX model
        try:
            load_model_start = time.perf_counter()
            # Load without external data — static analysis only needs graph structure,
            # shapes, and small embedded constants; not multi-GB weight tensors.
            model_proto = onnx.load(str(model_file), load_external_data=False)
            # Skip onnx.checker.check_model() which rejects custom attributes
            load_model_ms = int((time.perf_counter() - load_model_start) * 1000)
        except (OSError, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}") from e

        # Delegate to analyze_from_proto
        delegate_start = time.perf_counter()
        result = self.analyze_from_proto(
            model_proto=model_proto,
            ep=ep_normalized,
            device=device,
            enable_information=enable_information,
            model_path=str(model_file),
            htp_metadata_path=htp_metadata_path,
            for_debug=for_debug,
            pattern_debug_log_path=pattern_debug_log_path,
            run_unknown_op=run_unknown_op,
            save_node_types=save_node_types,
            on_node_result=on_node_result,
            on_ep_start=on_ep_start,
        )
        delegate_ms = int((time.perf_counter() - delegate_start) * 1000)
        _log_timing(
            "analyzer.analyze",
            model=model_file.name,
            ep=ep_normalized,
            device=device,
            load_model_ms=load_model_ms,
            analyze_from_proto_ms=delegate_ms,
            total_ms=int((time.perf_counter() - total_start) * 1000),
        )
        return result

    def analyze_from_proto(
        self,
        model_proto: onnx.ModelProto,
        ep: EPNameOrAlias | None = None,
        device: str | None = None,
        enable_information: bool = True,
        model_path: str | None = None,
        htp_metadata_path: str | None = None,
        for_debug: bool = False,
        pattern_debug_log_path: str | Path | None = None,
        run_unknown_op: bool = False,
        save_node_types: set[str] | None = None,
        on_node_result: Callable | None = None,
        on_ep_start: Callable | None = None,
    ) -> AnalysisResult:
        """Analyze ONNX model from ModelProto object.

        Use this method when you already have a loaded ONNX model
        in memory (e.g., after model transformation or optimization).

        Args:
            model_proto: ONNX ModelProto object
            ep: Target execution provider (e.g., "QNNExecutionProvider",
                "OpenVINOExecutionProvider", "DmlExecutionProvider").
                Also supports aliases: "qnn", "openvino", "vitisai".
                If None, analyzes all supported EPs.
            device: Target device type (e.g., "CPU", "GPU", "NPU").
                If None, uses "NPU" as default.
            enable_information: Whether to generate recommendations
            model_path: Optional path to model file (for metadata)
            htp_metadata_path: Optional path to HTP metadata JSON file
                for pattern extraction from hierarchy traces
            for_debug: Whether to include runtime debug payloads in check results.
                Default: False
            pattern_debug_log_path: Optional path for temporary pattern debug
                file sink. When set with ``for_debug=True``, the recent
                pattern-extractor debug signals are appended to this file.
            run_unknown_op: Whether to run unknown operators on local machine
                if possible. Default: True
            save_node_types: Set of node types to save for further analysis
                (e.g., {"partial", "unsupported"}). Default: None (save nothing)

        Returns:
            AnalysisResult: Analysis result wrapper with output

        Example:
            >>> import onnx
            >>> model = onnx.load("model.onnx")
            >>> # Apply transformations
            >>> model = optimize_model(model)
            >>> # Analyze optimized model
            >>> analyzer = ONNXStaticAnalyzer()
            >>> result = analyzer.analyze_from_proto(
            ...     model,
            ...     ep="QNNExecutionProvider",
            ...     device="NPU"
            ... )
        """
        from .core.onnx_loader import ONNXLoader
        from .core.pattern_extractor import PatternExtractor
        from .core.runtime_checker import RuntimeChecker

        # Normalize EP name (convert aliases to full names)
        total_start = time.perf_counter()
        ep_normalized = normalize_ep_name(ep)
        if ep != ep_normalized:
            logger.debug("EP alias '%s' normalized to '%s'", ep, ep_normalized)

        logger.info("Analyzing model from ModelProto")

        # Resolve device — rule files are device-specific (CPU/GPU/NPU).
        if device is not None and device.lower() == "auto":
            from ..sysinfo import resolve_device

            resolved, _ = resolve_device("auto", ep=ep_normalized)
            device_to_use = resolved.upper()
            logger.info("Device 'auto' resolved to: %s", device_to_use)
        else:
            device_to_use = device if device is not None else "NPU"
            logger.info("Using device: %s", device_to_use)

        # Determine which EPs to analyze
        eps_to_analyze: list[EPName] = []
        if ep_normalized is None:
            # Analyze all EPs that support the target device
            eps_to_analyze = [
                ep_name
                for ep_name, supported_devices in EP_SUPPORTED_DEVICES.items()
                if device_to_use.lower() in supported_devices
            ]
            logger.info("No EP specified, analyzing all supported EPs: %s", eps_to_analyze)
        else:
            eps_to_analyze = [ep_normalized]

        # Step 1: Create ONNXModel and extract patterns (once)
        extraction_start = time.perf_counter()
        logger.info("Loading model and extracting patterns...")
        onnx_loader = ONNXLoader(model_proto=model_proto)
        onnx_model = onnx_loader.load()

        # Override model_path if provided (for models loaded from file)
        if model_path:
            object.__setattr__(onnx_model, "model_path", model_path)

        pattern_extractor = PatternExtractor(onnx_model, htp_metadata_path=htp_metadata_path)
        extraction_result = pattern_extractor.summary(
            ep=ep_normalized,
            device=device_to_use,
            for_debug=for_debug,
        )

        metadata = extraction_result["summary"]
        pattern_matches = extraction_result["subgraph_patterns"]
        pattern_matches_by_source = extraction_result["subgraph_patterns_by_source"]
        source_stats = extraction_result["source_stats"]
        merge_prep = extraction_result.get("merge_prep", [])
        extraction_total_ms = extraction_result["total_extract_ms"]
        model_signature = extraction_result["model_signature"]
        logger.info("Extracted %d patterns", len(pattern_matches))
        extraction_ms = int((time.perf_counter() - extraction_start) * 1000)

        if for_debug and pattern_debug_log_path is not None:
            _append_pattern_debug_log(
                output_path=Path(pattern_debug_log_path),
                model_path=str(onnx_model.model_path),
                ep=ep_normalized,
                device=device_to_use,
                extracted_count=len(pattern_matches),
                model_signature=model_signature,
                extraction_total_ms=extraction_total_ms,
                source_stats=source_stats,
                pattern_matches_by_source=pattern_matches_by_source,
                merge_prep=merge_prep,
            )

        if for_debug:
            logger.info(
                "Pattern extractor debug: model_signature=%s total_ms=%d",
                model_signature,
                extraction_total_ms,
            )
            for source_stat in source_stats:
                logger.info(
                    "Pattern extractor source=%s cache_hit=%s pattern_class_count=%d match_count=%d elapsed_ms=%d",
                    source_stat["source"],
                    source_stat["cache_hit"],
                    source_stat["pattern_class_count"],
                    source_stat["match_count"],
                    source_stat["elapsed_ms"],
                )
                source_group = pattern_matches_by_source.get(source_stat["source"], {})
                logger.info(
                    "Pattern extractor source=%s matched pattern classes=%s",
                    source_stat["source"],
                    sorted(source_group.keys()),
                )

        # Pattern extraction results are currently for debug/observability only.
        # Runtime checker continues with operator-level checks while the new
        # pattern flow is being built out.
        pattern_matches_for_runtime: list[Any] = []

        # Step 2: Check runtime support for each EP
        check_op_results: dict[EPName, list[PatternRuntime]] = {}
        information_list: dict[EPName, list[Information]] = {}
        runtime_debug_details_summary: dict[
            str, dict[str, list[str] | dict[str, RuntimeDebugSummaryEntry]]
        ] = {}
        ep_runtime_timing: dict[str, int] = {}
        ep_info_timing: dict[str, int] = {}
        for current_ep in eps_to_analyze:
            logger.info("Checking runtime support for %s...", current_ep)
            if on_ep_start:
                try:
                    on_ep_start(current_ep, metadata.operator_counts)
                except Exception:
                    logger.debug("on_ep_start callback failed", exc_info=True)
            runtime_summary_start = time.perf_counter()
            runtime_checker = RuntimeChecker(
                ep=current_ep,
                device=device_to_use,
                model=onnx_model,
                patterns=pattern_matches_for_runtime,
            )
            # TODO: add VitisAIExecutionProvider back once non-QDQ
            # data is ready, and run_unknown_op is supported for QDQ ops
            run_unknown_op_for_ep = run_unknown_op
            if current_ep == "VitisAIExecutionProvider":
                run_unknown_op_for_ep = False

            runtime_summary = runtime_checker.summary(
                patterns=pattern_matches_for_runtime,
                for_debug=for_debug,
                run_unknown_op=run_unknown_op_for_ep,
                save_node_types=save_node_types,
                on_node_result=on_node_result,
            )
            runtime_summary_ms = int((time.perf_counter() - runtime_summary_start) * 1000)
            ep_runtime_timing[current_ep] = runtime_summary_ms

            if for_debug:
                ep_debug_summary = _build_runtime_debug_details_summary(runtime_summary)
                if ep_debug_summary is not None:
                    runtime_debug_details_summary[current_ep] = ep_debug_summary

            # Convert runtime summary to expected format
            op_results_list = runtime_summary.get("op_runtime_check_result", [])
            subgraph_results_list = runtime_summary.get("subgraph_runtime_check_result", [])

            check_op_results[current_ep] = op_results_list  # Use EP name as key

            # Step 3: Generate information (if enabled)
            if enable_information or self.config.enable_information:
                logger.info("Generating recommendations for %s...", current_ep)
                # Always create InformationEngine to run model-level validators
                # even if there are no runtime check results
                information_start = time.perf_counter()
                engine = self.information_engine_cls(
                    op_runtime_results=op_results_list,
                    subgraph_runtime_results=subgraph_results_list,
                    ep=current_ep,
                    model=onnx_model,
                    device=device_to_use,
                )
                information_list[current_ep] = engine.summary()  # Use EP name as key
                ep_info_timing[current_ep] = int((time.perf_counter() - information_start) * 1000)

        # Step 4: Aggregate results
        logger.info("Aggregating results...")
        aggregate_start = time.perf_counter()
        output = self.output_aggregator.aggregate(
            metadata=metadata,
            check_results=check_op_results,
            information_list=information_list,
            device=device_to_use,
        )

        if runtime_debug_details_summary:
            for ep_support in output.results:
                ep_debug_summary = runtime_debug_details_summary.get(ep_support.ep_type)
                if ep_debug_summary is not None:
                    ep_support.runtime_debug_details_summary = ep_debug_summary

        aggregate_ms = int((time.perf_counter() - aggregate_start) * 1000)

        _log_timing(
            "analyzer.analyze_from_proto",
            ep=ep_normalized,
            device=device_to_use,
            eps=len(eps_to_analyze),
            patterns=len(pattern_matches),
            extraction_ms=extraction_ms,
            aggregate_ms=aggregate_ms,
            runtime_ms_by_ep=ep_runtime_timing,
            information_ms_by_ep=ep_info_timing,
            total_ms=int((time.perf_counter() - total_start) * 1000),
        )

        logger.info("Analysis complete")
        return AnalysisResult(output=output)


# =============================================================================
# FLAT FUNCTIONAL API
# =============================================================================


@dataclass
class AnalyzeResult:
    """Result of ONNX model analysis with lint and optional autoconf.

    This is the return type of :func:`analyze_onnx` — a flat convenience wrapper.
    For the class-based API with full output access, use :class:`ONNXStaticAnalyzer`
    which returns :class:`AnalysisResult`.

    Attributes:
        lint: Lint-style result with error/warning/info counts and pattern lists.
        optimization_config: Auto-discovered optimization config (fusion flags).
            ``None`` when ``autoconf=False`` was passed to :func:`analyze_onnx`.
    """

    lint: LintResult
    optimization_config: WinMLOptimizationConfig | None

    @property
    def has_errors(self) -> bool:
        """True if blocking errors (unsupported patterns) exist."""
        return self.lint.errors > 0


def analyze_onnx(
    model: str | Path,
    *,
    ep: EPNameOrAlias | None = None,
    device: str | None = None,
    autoconf: bool = True,
    run_unknown_op: bool = False,
    on_ep_start: Callable | None = None,
    on_node_result: Callable | None = None,
    output_path: Path | None = None,
) -> AnalyzeResult:
    """Analyze an ONNX model and return lint + autoconf results.

    Convenience wrapper around :class:`ONNXStaticAnalyzer` that provides a flat
    functional API returning both lint diagnostics and auto-discovered
    optimization configuration in a single call.

    Args:
        model: Path to ONNX model file.
        ep: Target execution provider (e.g., ``"qnn"``, ``"QNNExecutionProvider"``).
            Aliases are normalized automatically.
            When ``None``, results aggregate across ALL EPs — use this only for
            exploratory analysis. For the build loop, always pass an explicit EP.
        device: Target device (e.g., ``"NPU"``, ``"GPU"``, ``"CPU"``).
            Defaults to ``"NPU"`` if ``None``.
        autoconf: Whether to generate optimization configuration from
            detected patterns. Default ``True``. When ``False``, skips the
            information engine entirely for faster lint-only analysis
            (``optimization_config`` will be ``None``).
        output_path: Optional file path to write the full :class:`AnalysisResult`
            as JSON. The file is written (or overwritten) after each call, so
            repeated calls with the same path keep the most recent result.

    Returns:
        AnalyzeResult with lint diagnostics and optional optimization config.

    Raises:
        FileNotFoundError: If model file doesn't exist.
        RuntimeError: If analysis fails.

    Example:
        >>> from winml.modelkit.analyze import analyze_onnx
        >>> result = analyze_onnx("optimized.onnx", ep="qnn", device="NPU")
        >>> if result.has_errors:
        ...     print(f"Errors: {result.lint.error_patterns}")
        >>> if result.optimization_config:
        ...     print(f"Autoconf: {result.optimization_config.to_dict()}")

        >>> # Save full analysis JSON alongside the model
        >>> result = analyze_onnx(
        ...     "model.onnx", ep="qnn", output_path=Path("analyze_result.json")
        ... )

        >>> # Lint-only (skip autoconf — faster, no information engine)
        >>> result = analyze_onnx("model.onnx", ep="qnn", autoconf=False)
        >>> assert result.optimization_config is None
    """
    model_path = str(model)

    if ep is None:
        logger.warning(
            "analyze_onnx called with ep=None — results will aggregate all EPs. "
            "For the build pipeline, always pass an explicit ep."
        )

    # Information engine is only needed when autoconf=True.
    # When autoconf=False, skip it for faster lint-only analysis.
    analyzer = ONNXStaticAnalyzer()
    analysis = analyzer.analyze(
        model_path=model_path,
        ep=ep,
        device=device,
        enable_information=autoconf,
        run_unknown_op=run_unknown_op,
        on_ep_start=on_ep_start,
        on_node_result=on_node_result,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(analysis.to_json(), encoding="utf-8")
        logger.debug("Analysis result written: %s", output_path)

    # Extract lint result (always computed — uses RuntimeChecker classification)
    lint = analysis.get_lint_result(ep=ep)

    # When autoconf=True, lint.optimization_config is already populated by
    # get_lint_result() which internally calls get_optimization_config().
    # When autoconf=False, information engine was skipped so
    # lint.optimization_config is empty — we set top-level to None.
    optimization_config = lint.optimization_config if autoconf else None

    return AnalyzeResult(
        lint=lint,
        optimization_config=optimization_config,
    )
