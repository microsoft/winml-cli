"""OutputAggregator - Assemble final JSON output from analysis results.

Implements FR-026-031 (Output assembly and structure).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..models.information import Information
    from ..models.runtime_checks import PatternRuntime

from ..models.output import AnalysisOutput, EPSupport, ModelStats
from ..models.support_level import SupportLevel
from ..utils import infer_ihv_from_ep_name


logger = logging.getLogger(__name__)


class OutputAggregator:
    """Aggregate analysis results into output format.

    Responsibilities:
    - Accept pre-built ModelStats
    - Build EPSupport objects for each Execution Provider
    - Assemble AnalysisOutput for JSON serialization
    - Include runtime check results and information

    FR-026-031: Complete output assembly with metadata, results, and information

    Attributes:
        analyzer_version: Version string for output
    """

    def __init__(self, analyzer_version: str = "0.1.0") -> None:
        """Initialize aggregator.

        Args:
            analyzer_version: Version string for output (default: "0.1.0")
        """
        self.analyzer_version = analyzer_version
        logger.info("Initialized OutputAggregator with version %s", analyzer_version)

    def aggregate(
        self,
        metadata: ModelStats,
        check_results: dict[str, list[PatternRuntime]],  # EP name -> check results
        information_list: dict[str, list[Information]],  # EP name -> information
        device: str | None = None,  # Device type
    ) -> AnalysisOutput:
        """Aggregate all analysis results.

        Args:
            metadata: Pre-built model metadata
            check_results: Runtime check results per EP name (list of PatternRuntime)
            information_list: Generated information per EP name
            device: Device type (e.g., CPU, GPU, NPU)

        Returns:
            AnalysisOutput: Complete analysis output ready for JSON serialization

        Output Structure:
            - analyzer_version: Version string
            - analysis_timestamp: Current datetime
            - metadata: Model metadata (path, opset, operator stats)
            - results: List of EPSupport objects

        Example:
            >>> aggregator = OutputAggregator("0.1.0")
            >>> metadata = ModelStats(
            ...     model_path="model.onnx",
            ...     opset_version=13,
            ...     producer_name="pytorch",
            ...     producer_version="1.9",
            ...     total_operators=176,
            ...     operator_counts={"Conv": 53, "Relu": 53},
            ...     unique_operator_types=2,
            ...     detected_pattern_count=10,
            ...     detected_patterns=patterns
            ... )
            >>> output = aggregator.aggregate(
            ...     metadata=metadata,
            ...     check_results=check_results,
            ...     information_list=information_list
            ... )
            >>> json_output = output.model_dump_json()
        """
        logger.info("Aggregating analysis results for model: %s", metadata.model_path)

        # Input validation
        if not check_results and not information_list:
            logger.warning("Both check_results and information_list are empty")

        # Build IHV support sections for all EP names from both sources
        all_ep_names = set(check_results.keys()) | set(information_list.keys())
        results: list[EPSupport] = []

        for ep_name in all_ep_names:
            ep_check_results = check_results.get(ep_name, [])
            ep_information = information_list.get(ep_name, [])

            logger.debug(f"Building EP support for {ep_name} with device: {device}")
            ep_support = self.build_ep_support(
                check_results=ep_check_results,
                information_list=ep_information,
                ep_type=ep_name,
                device_type=device,
            )
            results.append(ep_support)

        # Create final output
        output = AnalysisOutput(
            analyzer_version=self.analyzer_version,
            metadata=metadata,
            results=results,
        )

        logger.info(
            "Aggregation complete: %d IHV results, %d patterns",
            len(results),
            sum(metadata.detected_pattern_count.values()),
        )

        return output

    def build_ep_support(
        self,
        check_results: list[PatternRuntime],
        information_list: list[Information],
        ep_type: str,
        device_type: str | None = None,
        ep_version: str | None = None,
        driver_version: str | None = None,
    ) -> EPSupport:
        """Build Execution Provider support section.

        Args:
            check_results: Runtime check results (list of PatternRuntime)
            information_list: Generated information
            ep_type: Execution Provider name (e.g., QNNExecutionProvider)
            device_type: Device type (e.g., CPU, GPU, NPU)
            ep_version: Optional EP version
            driver_version: Optional driver version

        Returns:
            EPSupport: Execution Provider support object with classification and information

        Process:
            1. Classify patterns by support level from check_results
            2. Determine overall runtime_support status (False if any BLACK)
            3. Assemble EPSupport with classification and information
        """
        # Infer IHVType from EP name using utility function
        ihv = infer_ihv_from_ep_name(ep_type)

        logger.debug("Building IHV support for %s", ihv.value)

        # Classify patterns by support level
        classification: dict[SupportLevel, list[str]] = {
            SupportLevel.WHITE: [],
            SupportLevel.GRAY: [],
            SupportLevel.BLACK: [],
            SupportLevel.UNKNOWN: [],
        }

        # Only process patterns that have check results
        for pattern_runtime in check_results:
            # Classify based on result
            support_level = pattern_runtime.result.classification
            # Deduplicate: only append if not already in the list
            if pattern_runtime.pattern_id not in classification[support_level]:
                classification[support_level].append(pattern_runtime.pattern_id)

        # Determine overall runtime support
        # Support is False if any patterns are BLACK, True otherwise
        has_black_or_gray = (
            len(classification[SupportLevel.BLACK])
            + len(classification[SupportLevel.UNKNOWN])
            + len(classification[SupportLevel.GRAY])
            > 0
        )
        runtime_support = not has_black_or_gray

        # Check if BLACK patterns exist (blocking errors)
        has_black = len(classification[SupportLevel.BLACK]) > 0

        # Check if GRAY patterns exist (warnings/optimizations)
        has_gray = len(classification[SupportLevel.GRAY]) > 0

        if not check_results:
            # No check results available
            logger.warning("No check results for EP %s", ep_type)
            runtime_support = False
            has_black = False
            has_gray = False

        logger.debug(
            "EP %s classification: WHITE=%d, GRAY=%d, BLACK=%d, UNKNOWN=%d",
            ep_type,
            len(classification[SupportLevel.WHITE]),
            len(classification[SupportLevel.GRAY]),
            len(classification[SupportLevel.BLACK]),
            len(classification[SupportLevel.UNKNOWN]),
        )

        logger.debug(f"Creating EPSupport with device_type: {device_type}")
        return EPSupport(
            ihv_type=ihv,
            ep_type=ep_type,
            device_type=device_type,
            ep_version=ep_version,
            driver_version=driver_version,
            runtime_support=runtime_support,
            has_errors=has_black,
            has_warnings=has_gray,
            classification=classification,
            information=information_list,
        )
