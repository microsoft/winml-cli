# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""RuntimeChecker - Check operator support against runtime rules.

Implements FR-005 (Runtime support checking) and FR-016-020
(Support classification).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import tqdm

from ..utils.timing_utils import make_timing_logger
from .runtime_checker_query import RuntimeCheckerQuery


if TYPE_CHECKING:
    from collections.abc import Callable

    from ...pattern.match import PatternMatchResult
    from ...utils.constants import EPName
    from ..models.onnx_model import ONNXModel
    from ..models.runtime_checks import PatternRuntime, RuntimeTestResult

logger = logging.getLogger(__name__)
_log_timing = make_timing_logger(logger)

# Runtime check result status constants
RESULT_SUCCESS = "success"
RESULT_FAIL = "fail"
RESULT_NO_DATA = "no_data"


class RuntimeChecker:
    """Check operator support against runtime rules.

    High-level interface for checking operator-level support for a target
    Execution Provider (EP).

    Responsibilities:
    - Query runtime support via RuntimeCheckerQuery
    - Convert ONNX nodes to pattern matches
    - Classify support level (supported/partial/unsupported)
    - Aggregate runtime check results

    FR-005: Runtime support checking
    FR-016-020: Support classification logic

    Attributes:
        model: ONNX model to analyze
        ep: Target execution provider (e.g., "QNNExecutionProvider")
        device: Device string (e.g., "CPU" | "GPU" | "NPU")
    """

    def __init__(
        self,
        ep: EPName,
        device: str,
        model: ONNXModel,
        dynamic_axis_strict_mode: bool = False,
        pattern_matched_node_status_by_key: dict[str, str] | None = None,
    ) -> None:
        """Initialize runtime checker.

        Args:
            ep: Target execution provider name
            device: Device string (e.g., "CPU" | "GPU" | "NPU")
            model: ONNX model to analyze
            dynamic_axis_strict_mode: If False (default), maps any dynamic axes to (0,)
                for matching against first_axis test data. If True, preserves exact
                dynamic axis indices.
            pattern_matched_node_status_by_key: Optional stable node-key ->
                pattern status map (supported/partial/unsupported/unknown)
                used when matched nodes bypass parquet lookup.

        Raises:
            ValueError: If model is not provided
        """
        if model is None:
            raise ValueError("'model' is required")

        if not device or not device.strip():
            raise ValueError("device parameter cannot be empty")

        self._model = model

        self._ep: EPName = ep
        self._device = device

        self._dynamic_axis_strict_mode = dynamic_axis_strict_mode
        self._pattern_matched_node_status_by_key: dict[str, str] = dict(
            pattern_matched_node_status_by_key or {}
        )

        # Lazy-initialized RuntimeCheckerQuery (cached for reuse)
        self._query: RuntimeCheckerQuery | None = None

        # Pre-compute rule-data availability once at construction time so that
        # op_support() can read the cached result without repeated filesystem probes.
        from ..utils.ep_utils import has_any_rule_data, has_rule_data_for_ep

        self._has_rule_data: bool = has_rule_data_for_ep(ep, device)
        self._has_any_rule_data: bool = has_any_rule_data() if not self._has_rule_data else True

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
                pattern_matched_node_status_by_key=self._pattern_matched_node_status_by_key,
            )

        return self._query

    def op_support(
        self,
        for_debug: bool = False,
        run_unknown_op: bool = False,
        save_node_types: set[str] | None = None,
        on_node_result: Callable | None = None,
        node_output_filter: set[str] | None = None,
    ) -> list[PatternRuntime]:
        """Check operator-level runtime support.

        Returns operator-level runtime check results for each operator.

        Args:
            for_debug: Whether to include runtime debug details for each node.
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
            node_output_filter: Optional set of output tensor names. When
                provided, only nodes that produce at least one of these tensors
                are checked; all other nodes are skipped. This lets callers
                restrict the check to a specific subset of the graph (e.g. the
                operators an optimization introduced) without paying to check
                every node.

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

        # Emit a diagnostic once if rule data is absent for this EP+device.
        # Uses the pre-computed flags from __init__ (no repeated filesystem probes).
        if not self._has_rule_data:
            if not self._has_any_rule_data:
                logger.warning(
                    "No runtime check data found. Follow "
                    "https://github.com/microsoft/winml-cli/blob/main/CONTRIBUTING.md "
                    "to set up runtime check files."
                )
            else:
                logger.info(
                    "No runtime check data for %s on %s — op analysis will return no_data.",
                    self._ep,
                    self._device,
                )

        total_start = time.perf_counter()
        results: list[PatternRuntime] = []
        run_for_node_total_ms = 0
        callback_total_ms = 0

        # Get cached RuntimeCheckerQuery
        query = self._get_query()
        # Use the same graph snapshot as RuntimeCheckerQuery (post shape inference).
        nodes = query.model_proto.graph.node
        # Use tqdm for progress unless caller provides a callback
        iterator = nodes if on_node_result else tqdm.tqdm(nodes)
        for node in iterator:
            if node_output_filter is not None and not (
                node_output_filter.intersection(node.output)
            ):
                # Node is outside the requested subset — skip it.
                continue
            node_start = time.perf_counter()
            result = query.run_for_node(
                node,
                for_debug=for_debug,
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

    def summary(
        self,
        for_debug: bool = False,
        run_unknown_op: bool = False,
        save_node_types: set[str] | None = None,
        on_node_result: Callable | None = None,
    ) -> dict[str, list[PatternRuntime]]:
        """Return operator-level runtime results.

        Returns:
            Dict containing operator-level runtime check results.
        """
        logger.info("Generating runtime support summary")

        total_start = time.perf_counter()
        summary_dict: dict[str, list[PatternRuntime]] = {}
        op_support_ms = 0

        # Get operator-level support (only if model is available)
        if self._model is not None:
            op_start = time.perf_counter()
            op_results = self.op_support(
                for_debug=for_debug,
                run_unknown_op=run_unknown_op,
                save_node_types=save_node_types,
                on_node_result=on_node_result,
            )
            op_support_ms = int((time.perf_counter() - op_start) * 1000)
            summary_dict["op_runtime_check_result"] = op_results

        total_ms = int((time.perf_counter() - total_start) * 1000)
        _log_timing(
            "runtime_checker.summary",
            ep=self._ep,
            device=self._device,
            op_results=len(summary_dict.get("op_runtime_check_result", [])),
            total_ms=total_ms,
            op_support_ms=op_support_ms,
            overhead_ms=total_ms - op_support_ms,
        )

        return summary_dict

    def check_pattern_locally(
        self,
        pattern_match: PatternMatchResult,
        *,
        fallback_reason: str,
        for_debug: bool = False,
    ) -> RuntimeTestResult | None:
        """Compile and run one complete matched pattern on the local EP."""
        return self._get_query().try_local_pattern_check(
            pattern_match,
            fallback_reason=fallback_reason,
            for_debug=for_debug,
        )

    def close_local_checks(self) -> None:
        """Release resources used by local compile/run fallback checks."""
        if self._query is not None:
            self._query.close_local_checks()
