# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""RuntimeChecker - Check pattern support against runtime rules.

Implements FR-005 (Runtime support checking), FR-006 (Pattern matching),
FR-016-020 (Support classification).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import tqdm

from ...pattern.config import UnifiedPatternConfig
from ..models.runtime_checks import (
    AlternativeType,
    PatternAlternative,
    PatternRuntime,
    RuntimeTestResult,
)
from ..utils.timing_utils import make_timing_logger
from .runtime_checker_query import RuntimeCheckerQuery


if TYPE_CHECKING:
    from collections.abc import Callable

    import onnx

    from winml.modelkit.pattern.match import PatternMatchResult

    from ..models.onnx_model import ONNXModel

logger = logging.getLogger(__name__)
_log_timing = make_timing_logger(logger)

# Runtime check result status constants
RESULT_SUCCESS = "success"
RESULT_FAIL = "fail"
RESULT_NO_DATA = "no_data"


class RuntimeChecker:
    """Check operator and subgraph pattern support against runtime rules.

    High-level interface for checking operator-level and subgraph-level
    support for a target Execution Provider (EP).

    Responsibilities:
    - Query runtime support via RuntimeCheckerQuery
    - Convert ONNX nodes to pattern matches
    - Classify support level (supported/partial/unsupported)
    - Aggregate runtime check results

    FR-005: Runtime support checking
    FR-006: Pattern matching against rule database
    FR-016-020: Support classification logic

    Attributes:
        model: ONNX model to analyze (optional)
        patterns: List of PatternMatch for subgraph detection (optional)
        ep: Target execution provider (e.g., "QNNExecutionProvider")
        device: Device string (e.g., "CPU" | "GPU" | "NPU")
    """

    def __init__(
        self,
        ep: str,
        device: str,
        model: ONNXModel | None = None,
        patterns: list[PatternMatchResult] | None = None,
        pattern_config: UnifiedPatternConfig | None = None,
        dynamic_axis_strict_mode: bool = False,
    ) -> None:
        """Initialize runtime checker.

        Args:
            ep: Target execution provider name
            device: Device string (e.g., "CPU" | "GPU" | "NPU")
            model: ONNX model to analyze (optional)
            patterns: List of PatternMatchResult for subgraph detection (optional)
            pattern_config: Pattern configuration for reading alternatives.
                           If None, a default UnifiedPatternConfig is created.
            dynamic_axis_strict_mode: If False (default), maps any dynamic axes to (0,)
                for matching against first_axis test data. If True, preserves exact
                dynamic axis indices.

        Raises:
            ValueError: If neither model nor patterns is provided
        """
        if model is None and patterns is None:
            raise ValueError("At least one of 'model' or 'patterns' must be provided")

        if not ep or not ep.strip():
            raise ValueError("ep parameter cannot be empty")

        if not device or not device.strip():
            raise ValueError("device parameter cannot be empty")

        self._model = model
        self._patterns = patterns

        self._ep = ep
        self._device = device

        # Pattern configuration for reading alternatives from JSON
        self._pattern_config = pattern_config or UnifiedPatternConfig()
        self._dynamic_axis_strict_mode = dynamic_axis_strict_mode

        # Lazy-initialized RuntimeCheckerQuery (cached for reuse)
        self._query: RuntimeCheckerQuery | None = None

        logger.info(
            "Initialized RuntimeChecker for EP=%s, driver=%s",
            ep,
            device,
        )

    def _get_query(self) -> RuntimeCheckerQuery:
        """Get or create cached RuntimeCheckerQuery.

        Returns:
            RuntimeCheckerQuery instance (cached or newly created)

        Raises:
            ValueError: If model is not available
        """
        if self._model is None:
            raise ValueError(
                "Cannot create RuntimeCheckerQuery without ONNX model. "
                "RuntimeChecker was initialized without model."
            )

        if self._query is None:
            model_proto = self._model.get_model()
            self._query = RuntimeCheckerQuery(
                model_proto=model_proto,
                ep_name=self._ep,
                device_type=self._device,
                model_path=self._model.model_path,
                dynamic_axis_strict_mode=self._dynamic_axis_strict_mode,
                node_key_by_node_id=self._model.get_node_key_map(),
            )

        return self._query

    def op_support(
        self,
        run_unknown_op: bool = True,
        save_node_types: set[str] | None = None,
        on_node_result: Callable | None = None,
    ) -> list[PatternRuntime]:
        """Check operator-level runtime support.

        Returns operator-level runtime check results for each operator.

        Args:
            on_node_result: Optional per-node progress callback.
                When provided, tqdm progress bar is suppressed (caller
                handles progress display via Rich Live).

                Signature::

                    (result: PatternRuntime) -> None

                The ``PatternRuntime`` passed to the callback has:

                - ``pattern_id`` (str): Full pattern ID, e.g.
                  ``"OP/ai.onnx/Conv"``. Use ``split("/")[-1]`` to get
                  the display name (``"Conv"``).
                - ``result.classification`` (SupportLevel): The support
                  level enum. Call ``.value`` to get the string, e.g.
                  ``"supported"``, ``"partial"``, ``"unsupported"``,
                  ``"unknown"``.

        Returns:
            List[PatternRuntime]: Runtime results for each operator pattern

        Raises:
            ValueError: If initialized without ONNXModel
        """
        if self._model is None:
            raise ValueError(
                "op_support() requires ONNXModel. "
                "RuntimeChecker was initialized with list[PatternMatchResult]."
            )

        logger.info("Checking operator-level runtime support")

        total_start = time.perf_counter()
        results: list[PatternRuntime] = []
        run_for_node_total_ms = 0
        callback_total_ms = 0

        # Get all nodes from model
        model_proto = self._model.get_model()
        # Get cached RuntimeCheckerQuery
        query = self._get_query()
        # Use tqdm for progress unless caller provides a callback
        nodes = model_proto.graph.node
        iterator = nodes if on_node_result else tqdm.tqdm(nodes)
        for node in iterator:
            node_start = time.perf_counter()
            result = query.run_for_node(
                node,
                run_unknown_op=run_unknown_op,
                save_node_types=save_node_types,
            )
            run_for_node_total_ms += int((time.perf_counter() - node_start) * 1000)
            results.append(result)
            if on_node_result:
                callback_start = time.perf_counter()
                try:
                    on_node_result(result)
                except Exception:
                    logger.debug("on_node_result callback failed", exc_info=True)
                callback_total_ms += int((time.perf_counter() - callback_start) * 1000)

        logger.info("Checked %d operators", len(results))
        total_ms = int((time.perf_counter() - total_start) * 1000)
        _log_timing(
            "runtime_checker.op_support",
            ep=self._ep,
            device=self._device,
            nodes=len(results),
            total_ms=total_ms,
            run_for_node_ms=run_for_node_total_ms,
            callback_ms=callback_total_ms,
            overhead_ms=total_ms - run_for_node_total_ms - callback_total_ms,
            avg_run_for_node_ms=(run_for_node_total_ms // len(results) if results else 0),
        )

        return results

    def subgraph_support(
        self,
        patterns: list[PatternMatchResult] | None = None,
        run_unknown_op: bool = True,
    ) -> list[PatternRuntime]:
        """Check subgraph-level runtime support.

        Given detected patterns, check runtime support.
        Each pattern returns result + optional replacement Information.

        Args:
            patterns: List of PatternMatchResult objects to check.
                      If None, uses patterns from initialization.

        Returns:
            List[PatternRuntime]: Runtime results for each pattern with alternatives

        Raises:
            ValueError: If patterns is None and RuntimeChecker was not initialized with patterns
        """
        # Determine which patterns to use
        if patterns is None:
            if self._patterns is None:
                raise ValueError(
                    "patterns parameter is required when RuntimeChecker "
                    "is not initialized with patterns"
                )
            patterns = self._patterns

        logger.info(
            "Checking subgraph pattern support via per-node operator aggregation for %d patterns",
            len(patterns),
        )

        total_start = time.perf_counter()
        query_pattern_total_ms = 0
        results: list[PatternRuntime] = []
        for pattern in patterns:
            pattern_start = time.perf_counter()
            pattern_runtime = self.query_pattern_support(pattern, run_unknown_op=run_unknown_op)
            query_pattern_total_ms += int((time.perf_counter() - pattern_start) * 1000)
            results.append(pattern_runtime)

        total_ms = int((time.perf_counter() - total_start) * 1000)
        _log_timing(
            "runtime_checker.subgraph_support",
            ep=self._ep,
            device=self._device,
            patterns=len(results),
            total_ms=total_ms,
            query_pattern_ms=query_pattern_total_ms,
            overhead_ms=total_ms - query_pattern_total_ms,
            avg_query_pattern_ms=(query_pattern_total_ms // len(results) if results else 0),
        )

        return results

    def query_pattern_support(
        self,
        pattern: PatternMatchResult,
        run_unknown_op: bool = True,
    ) -> PatternRuntime:
        """Evaluate a single pattern's runtime support + replacements.

        Args:
            pattern: PatternMatchResult object to check

        Returns:
            PatternRuntime: Runtime result with pattern_id, result, and alternatives

        Process:
            1. Check original pattern support via RuntimeCheckerQuery.run_for_subgraph
            2. Check possible replacement patterns (alternatives)
            3. For each alternative, evaluate its support status
            4. Return PatternRuntime with results and alternatives
        """
        if self._model is None:
            raise ValueError(
                f"Cannot lookup pattern support for '{pattern.pattern.pattern_id}' "
                f"without ONNX model. RuntimeChecker was initialized without model."
            )

        pattern_id = pattern.pattern.pattern_id

        # Get cached RuntimeCheckerQuery and check pattern support
        query = self._get_query()
        pattern_runtime = query.run_for_subgraph(pattern, run_unknown_op=run_unknown_op)
        result = pattern_runtime.result

        logger.debug(
            "Pattern %s: %s (compile=%s, run=%s)",
            pattern_id,
            result.classification.value,
            result.compile,
            result.run,
        )

        # Build alternatives from pattern config (JSON)
        # TODO: Replace mock RuntimeTestResult with actual runtime checks
        alternatives: list[PatternAlternative] = []
        config_alternatives = self._pattern_config.get_alternatives(pattern.pattern)
        for config_alt in config_alternatives:
            alternative = PatternAlternative(
                pattern_id=config_alt.pattern_to_id,
                result=RuntimeTestResult(
                    compile=True,
                    run=True,
                    reason=config_alt.reason or f"Alternative for {pattern_id}",
                ),
                alternative_type=AlternativeType.EQUIVALENT,
            )
            alternatives.append(alternative)
            logger.debug(
                "Added alternative %s for pattern %s",
                config_alt.pattern_to_id,
                pattern_id,
            )

        return PatternRuntime(
            pattern_id=pattern_id,
            result=result,
            alternatives=alternatives,
            pattern_match=pattern,
        )

    def summary(
        self,
        patterns: list[PatternMatchResult] | None = None,
        run_unknown_op: bool = True,
        save_node_types: set[str] | None = None,
        on_node_result: Callable | None = None,
    ) -> dict[str, list[PatternRuntime]]:
        """Combine operator-level & pattern-level runtime results.

        Args:
            patterns: List of PatternMatchResult objects to check.
                      If None, uses patterns from initialization.

        Returns:
            Dict containing both op_support and subgraph_support results:
                - op_runtime_check_result: Operator-level runtime check
                  results (only if model provided)
                - subgraph_runtime_check_result: Subgraph pattern check
                  results
        """
        logger.info("Generating runtime support summary")

        total_start = time.perf_counter()
        summary_dict: dict[str, list[PatternRuntime]] = {}
        op_support_ms = 0
        subgraph_support_ms = 0
        merge_ms = 0

        # Get operator-level support (only if model is available)
        if self._model is not None:
            op_start = time.perf_counter()
            op_results = self.op_support(
                run_unknown_op=run_unknown_op,
                save_node_types=save_node_types,
                on_node_result=on_node_result,
            )
            op_support_ms = int((time.perf_counter() - op_start) * 1000)
            summary_dict["op_runtime_check_result"] = op_results

        # Get subgraph-level support
        subgraph_start = time.perf_counter()
        pattern_results = self.subgraph_support(patterns, run_unknown_op=run_unknown_op)
        subgraph_support_ms = int((time.perf_counter() - subgraph_start) * 1000)
        summary_dict["subgraph_runtime_check_result"] = pattern_results

        merge_start = time.perf_counter()
        # Build stable node key -> PatternRuntime from pattern_results
        node_to_pattern_runtime: dict[str, PatternRuntime] = {}
        for pr in pattern_results:
            if (
                (not pr.result.no_data)
                and pr.pattern_match
                and hasattr(pr.pattern_match, "skeleton_match_result")
            ):
                smr = pr.pattern_match.skeleton_match_result
                if smr and smr.matched_node_names:
                    for node_key in smr.matched_node_names:
                        node_to_pattern_runtime[node_key] = pr

        # Override matching op_results
        merged = []
        for op_r in summary_dict["op_runtime_check_result"]:
            node_key = self._get_node_key(op_r)
            if node_key in node_to_pattern_runtime:
                # Replace with pattern-level result, keeping original pattern_match for traceability
                pr = node_to_pattern_runtime[node_key]
                merged.append(
                    PatternRuntime(
                        pattern_id=op_r.pattern_id,  # keep original op pattern_id
                        result=pr.result,  # use subgraph-level result
                        alternatives=[],  # subgraph alternatives belong to the subgraph, not the op
                        pattern_match=op_r.pattern_match,
                    )
                )
            else:
                merged.append(op_r)

        summary_dict["op_runtime_check_result"] = merged
        merge_ms = int((time.perf_counter() - merge_start) * 1000)

        total_ms = int((time.perf_counter() - total_start) * 1000)
        _log_timing(
            "runtime_checker.summary",
            ep=self._ep,
            device=self._device,
            op_results=len(summary_dict.get("op_runtime_check_result", [])),
            subgraph_results=len(summary_dict.get("subgraph_runtime_check_result", [])),
            total_ms=total_ms,
            op_support_ms=op_support_ms,
            subgraph_support_ms=subgraph_support_ms,
            merge_ms=merge_ms,
            overhead_ms=total_ms - op_support_ms - subgraph_support_ms - merge_ms,
        )

        return summary_dict

    def _get_node_key(self, op_runtime: PatternRuntime) -> str:
        """Extract stable node key from an op-level PatternRuntime."""
        pm = op_runtime.pattern_match
        if pm and hasattr(pm, "skeleton_match_result"):
            node_keys = pm.skeleton_match_result.matched_node_names
            if node_keys:
                return node_keys[0]
        return ""

    def _make_op_key(self, node: onnx.NodeProto) -> str:
        """Generate operator key from node.

        Internal method to create unique key for operator.

        Args:
            node: ONNX node

        Returns:
            Operator key string (e.g., "OP/ai.onnx/Conv")

        Note:
            This is an internal method.
        """
        # Detect namespace
        namespace = "ai.onnx"  # Default namespace
        if node.domain:
            if node.domain == "com.microsoft":
                namespace = "com.microsoft"
            elif node.domain != "":
                namespace = node.domain

        return f"OP/{namespace}/{node.op_type}"
