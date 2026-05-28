# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""InformationEngine - Generate actionable information from runtime check results.

Implements FR-021-028 (Information generation), processes runtime results to create
actionable guidance for pattern compatibility issues. Also integrates model-level
validation checks.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path

    import onnx

    from ...utils.constants import EPName
    from ..models.information import Action, Information
    from ..models.onnx_model import ONNXModel
    from ..models.runtime_checks import PatternAlternative, PatternRuntime

from ..models.information import ActionLevel
from ..models.support_level import SupportLevel
from ..utils.timing_utils import make_timing_logger


logger = logging.getLogger(__name__)
_log_timing = make_timing_logger(logger)

# Constants
DOC_CHECKER_PREFIX = "Information from sdk:"


class InformationEngine:
    """Generate information from runtime check results.

    Processes operator and subgraph runtime results to create actionable
    information with explanations and suggested actions. Also integrates
    model-level validation checks.

    Responsibilities:
    - Process operator-level patterns without alternatives
    - Process patterns with alternatives
    - Run model-level validation checks via ModelValidatorManager
    - Generate Information objects with explanations and actions
    - Classify actions as required/optional/warning

    FR-021-025: Generate operator-level and pattern-level information
    FR-026: Classify actions by priority
    FR-027: Provide detailed explanations
    FR-028: Support multi-IHV information generation

    Attributes:
        op_runtime_results: List of operator-level runtime results
        subgraph_runtime_results: List of subgraph-level runtime results
        ep: Target execution provider
        model: Optional ONNX model for model-level validation
    """

    def __init__(
        self,
        op_runtime_results: list[PatternRuntime],
        subgraph_runtime_results: list[PatternRuntime],
        ep: EPName,
        model: ONNXModel,
        device: str,
        rules_dir: Path | None = None,
        shape_inferred_model_proto: onnx.ModelProto | None = None,
    ) -> None:
        """Initialize information engine.

        Args:
            op_runtime_results: List of PatternRuntime for operator-level patterns
            subgraph_runtime_results: List of PatternRuntime for subgraph-level patterns
            ep: Target execution provider (e.g., "QNNExecutionProvider")
            rules_dir: Path to rules directory (optional, defaults to package rules/)
            model: Optional ONNX model for model-level validation.
                        If provided, model-level validation checks will be run.
            device: Device type (e.g., "NPU", "GPU", "CPU") for device-specific validation.
            shape_inferred_model_proto: Pre-inferred model proto to avoid redundant
                shape inference. If provided, DocConstraintChecker will reuse it.

        Implementation:
            - Stores operator and subgraph runtime results separately
            - Each PatternRuntime contains pattern info via pattern_id
            - Stores EP for context in information generation
            - Loads predefined information rules from JSON files
            - If model provided, will run model-level validators

        Raises:
            ValueError: If both op_runtime_results and subgraph_runtime_results are empty
                       AND model is not provided (at least one source of information needed)
        """
        total_start = time.perf_counter()
        if not op_runtime_results and not subgraph_runtime_results and model is None:
            raise ValueError(
                "At least one of op_runtime_results, "
                "subgraph_runtime_results, or model "
                "must be provided"
            )

        self._op_runtime_results = op_runtime_results
        self._subgraph_runtime_results = subgraph_runtime_results
        self._ep: EPName = ep
        self._model = model
        self._device = device

        # Load predefined information rules
        from ..utils import infer_ihv_from_ep_name
        from ..utils.rule_loader import RuleLoader

        self._rule_loader = RuleLoader(rules_dir=rules_dir)

        # Infer IHV from EP name for per-IHV rule loading
        infer_ihv_start = time.perf_counter()
        try:
            ihv_type = infer_ihv_from_ep_name(self._ep)
            logger.info("Inferred IHV type %s from EP %s", ihv_type.value, self._ep)
        except ValueError as e:
            logger.warning("Could not infer IHV from EP %s: %s. Loading all rules.", self._ep, e)
            ihv_type = None
        infer_ihv_ms = int((time.perf_counter() - infer_ihv_start) * 1000)

        load_predefined_start = time.perf_counter()
        self._predefined_info = self._rule_loader.load_information_rules(ihv_type=ihv_type)
        load_predefined_ms = int((time.perf_counter() - load_predefined_start) * 1000)

        # Build pattern_id -> Information lookup for quick matching
        self._info_by_pattern: dict[str, Information] = {
            info.pattern_id: info for info in self._predefined_info if info.pattern_id
        }

        # Initialize Doc Constraint Checker
        self._doc_checker = None
        init_doc_checker_ms = 0
        doc_checker_initialized = False
        try:
            from .doc_constraint_checker import DocConstraintChecker

            # Prefer the pre-inferred model proto to avoid redundant shape inference
            if shape_inferred_model_proto is not None:
                model_proto = shape_inferred_model_proto
                skip_inference = True
            else:
                model_proto = self._model.get_model()
                skip_inference = False

            init_doc_checker_start = time.perf_counter()
            self._doc_checker = DocConstraintChecker(
                model_proto,
                self._ep,
                self._device,
                skip_shape_inference=skip_inference,
                node_key_by_node_id=self._model.get_node_key_map(),
            )
            init_doc_checker_ms = int((time.perf_counter() - init_doc_checker_start) * 1000)
            doc_checker_initialized = True
            logger.info(
                "Initialized Doc Constraint Checker with %d operators",
                len(self._doc_checker.get_operators_with_constraints()),
            )
        except Exception as e:
            logger.warning("Failed to initialize Doc Constraint Checker: %s", e)

        logger.info(
            "Initialized InformationEngine with %d operator results and %d subgraph results",
            len(op_runtime_results),
            len(subgraph_runtime_results),
        )
        logger.info("Loaded %d predefined information rules", len(self._predefined_info))
        _log_timing(
            "information_engine.init",
            ep=self._ep,
            device=self._device,
            op_runtime_results=len(op_runtime_results),
            subgraph_runtime_results=len(subgraph_runtime_results),
            predefined_rules=len(self._predefined_info),
            doc_checker_initialized=doc_checker_initialized,
            infer_ihv_ms=infer_ihv_ms,
            load_predefined_rules_ms=load_predefined_ms,
            init_doc_checker_ms=init_doc_checker_ms,
            total_ms=int((time.perf_counter() - total_start) * 1000),
        )

    @property
    def op_runtime_results(self) -> list[PatternRuntime]:
        """List of operator-level runtime results."""
        return self._op_runtime_results

    @property
    def subgraph_runtime_results(self) -> list[PatternRuntime]:
        """List of subgraph-level runtime results."""
        return self._subgraph_runtime_results

    def summary(self) -> list[Information]:
        """Generate information list from runtime results.

        Returns:
            List[Information]: Information objects with explanations and actions

        Process:
            1. Run model-level validation checks via _check_model()
            2. Process operator-level patterns via _check_single_ops()
            3. Process operator and subgraph-level patterns via _check_patterns()
               - _check_patterns() prioritizes predefined information rules
            4. Combine results into Information objects
            5. Return complete list of Information
        """
        logger.info("Generating information summary")
        total_start = time.perf_counter()

        # Run model-level validation checks
        check_model_start = time.perf_counter()
        model_info = self._check_model()
        check_model_ms = int((time.perf_counter() - check_model_start) * 1000)

        # Process operator-level patterns without alternatives
        check_single_ops_start = time.perf_counter()
        ops_info = self._check_single_ops()
        check_single_ops_ms = int((time.perf_counter() - check_single_ops_start) * 1000)

        # Process patterns with alternatives (includes predefined information matching)
        check_patterns_start = time.perf_counter()
        pattern_info = self._check_patterns()
        check_patterns_ms = int((time.perf_counter() - check_patterns_start) * 1000)

        # Combine results
        combine_start = time.perf_counter()
        all_info = ops_info + pattern_info + model_info
        combine_ms = int((time.perf_counter() - combine_start) * 1000)

        logger.info("Generated %d information items", len(all_info))
        _log_timing(
            "information_engine.summary",
            ep=self._ep,
            device=self._device,
            op_runtime_results=len(self._op_runtime_results),
            subgraph_runtime_results=len(self._subgraph_runtime_results),
            model_info=len(model_info),
            single_ops_info=len(ops_info),
            pattern_info=len(pattern_info),
            total_info=len(all_info),
            check_model_ms=check_model_ms,
            check_single_ops_ms=check_single_ops_ms,
            check_patterns_ms=check_patterns_ms,
            combine_ms=combine_ms,
            total_ms=int((time.perf_counter() - total_start) * 1000),
        )

        return all_info

    def _check_model(self) -> list[Information]:
        """Run model-level validation checks.

        Returns:
            List[Information]: Information objects from model validators

        Process:
            1. Create ModelValidatorManager with model and op_runtime_results
            2. Run all validators to collect Information
            3. Return collected Information objects

        Raises:
            Exception: Validation errors are caught and logged, not raised
        """
        logger.debug("Running model-level validation checks")
        total_start = time.perf_counter()

        # Skip if model is not provided
        if self._model is None:
            logger.debug("Skipping model validation: model is None")
            _log_timing(
                "information_engine.check_model",
                ep=self._ep,
                device=self._device,
                skipped=True,
                total_ms=int((time.perf_counter() - total_start) * 1000),
            )
            return []

        try:
            from .model_validators import ModelValidatorManager

            manager_init_start = time.perf_counter()
            validator_manager = ModelValidatorManager(
                self._model,
                op_runtime_results=self._op_runtime_results,
                device=self._device,
            )
            manager_init_ms = int((time.perf_counter() - manager_init_start) * 1000)

            run_validators_start = time.perf_counter()
            model_info = validator_manager.run_all_validators()
            run_validators_ms = int((time.perf_counter() - run_validators_start) * 1000)

            logger.info(
                "Model validation complete: %d issue(s) detected",
                len(model_info),
            )

            _log_timing(
                "information_engine.check_model",
                ep=self._ep,
                device=self._device,
                validators=len(getattr(validator_manager, "validators", [])),
                issues=len(model_info),
                manager_init_ms=manager_init_ms,
                run_validators_ms=run_validators_ms,
                total_ms=int((time.perf_counter() - total_start) * 1000),
            )

            return model_info

        except Exception as e:
            logger.exception(
                "Model validation failed (%s). Continuing with pattern-level analysis...",
                type(e).__name__,
            )
            _log_timing(
                "information_engine.check_model",
                ep=self._ep,
                device=self._device,
                failed=True,
                error_type=type(e).__name__,
                total_ms=int((time.perf_counter() - total_start) * 1000),
            )
            return []

    def _check_single_ops(self) -> list[Information]:
        """Generate information for operator-level patterns without alternatives.

        Returns:
            List[Information]: Information objects for operator patterns

        Process:
            1. Iterate through self.op_runtime_results
            2. Aggregate by (op_type, classification, reason)
            3. For each group, check result.classification:
               - UNSUPPORTED: Required action to replace unsupported operator
               - PARTIAL: Optional action to optimize performance
               - SUPPORTED: No action needed
            4. Does NOT include alternative patterns from runtime_result.alternatives
            5. Generate explanation and actions based on classification with aggregated count
        """
        from collections import defaultdict

        from ..models.information import Action, Information

        logger.debug("Checking single operator patterns")
        total_start = time.perf_counter()

        # Group runtime results by (pattern_id, classification, reason)
        # Different reasons should create separate issues
        grouped_results: dict[tuple[str, SupportLevel, str | None], list[PatternRuntime]] = (
            defaultdict(list)
        )

        group_loop_start = time.perf_counter()
        doc_query_count = 0
        doc_query_total_ms = 0
        doc_query_slow_count = 0
        max_doc_query_ms = 0
        max_doc_query_pattern: str | None = None

        for runtime_result in self._op_runtime_results:
            if not self._validate_runtime_result(runtime_result):
                continue

            pattern_id = runtime_result.pattern_id
            classification = runtime_result.result.classification
            reason = runtime_result.result.reason

            # Skip SUPPORTED patterns (no action needed) and patterns with alternatives
            if classification == SupportLevel.SUPPORTED or runtime_result.alternatives:
                continue

            # For failed/unknown operators, query doc checker for detailed constraint info
            doc_check_reason = None
            if self._doc_checker and classification in [
                SupportLevel.UNSUPPORTED,
                SupportLevel.PARTIAL,
                SupportLevel.UNKNOWN,
            ]:
                doc_query_start = time.perf_counter()
                doc_check_reason = self._query_doc_constraints(runtime_result, pattern_id)
                doc_query_ms = int((time.perf_counter() - doc_query_start) * 1000)
                doc_query_count += 1
                doc_query_total_ms += doc_query_ms
                if doc_query_ms >= 20:
                    doc_query_slow_count += 1
                if doc_query_ms > max_doc_query_ms:
                    max_doc_query_ms = doc_query_ms
                    max_doc_query_pattern = pattern_id
                # Append doc checker reason to original reason
                if doc_check_reason:
                    reason = f"{reason}; {doc_check_reason}" if reason else doc_check_reason
                else:
                    logger.debug("Doc checker returned None for %s", pattern_id)

            # Group by (pattern_id, classification, reason) to separate different error types
            key = (pattern_id, classification, reason)
            grouped_results[key].append(runtime_result)

        group_loop_ms = int((time.perf_counter() - group_loop_start) * 1000)

        # Generate Information for each group
        info_list: list[Information] = []
        build_info_start = time.perf_counter()

        for (pattern_id, classification, reason), runtime_results in grouped_results.items():
            # Use the reason from the group key
            count = len(runtime_results)
            actions: list[Action] = []

            if classification == SupportLevel.PARTIAL:
                # Partial support - optional optimization
                if count == 1:
                    if reason:
                        explanation = (
                            f"Operator '{pattern_id}' has partial support "
                            f"(compiles fail, fallback to CPU): {reason}"
                        )
                    else:
                        explanation = (
                            f"Operator '{pattern_id}' has partial support "
                            f"(compiles fail, fallback to CPU)"
                        )
                else:
                    if reason:
                        explanation = (
                            f"{count} instances of operator '{pattern_id}' have partial support "
                            f"(compiles fail, fallback to CPU): {reason}"
                        )
                    else:
                        explanation = (
                            f"{count} instances of operator '{pattern_id}' have partial support "
                            f"(compiles fail, fallback to CPU)"
                        )

                # action = Action(
                #     pattern_from_id=pattern_id,
                #     pattern_to_id="",
                #     level=ActionLevel.OPTIONAL,
                #     status=SupportLevel.PARTIAL,
                #     details=(
                #         f"Pattern '{pattern_id}' has partial support. "
                #         f"Consider optimizing the input for better runtime performance. "
                #         f"{reason or 'No additional details.'}"
                #     ),
                # )
                # actions.append(action)
                logger.debug(
                    "Operator %s: PARTIAL (partial support) - %d instances", pattern_id, count
                )

            elif classification == SupportLevel.UNSUPPORTED:
                # Not supported - required action
                if count == 1:
                    if reason:
                        explanation = f"Operator '{pattern_id}' is not supported: {reason}"
                    else:
                        explanation = f"Operator '{pattern_id}' is not supported"
                else:
                    if reason:
                        explanation = (
                            f"{count} instances of operator "
                            f"'{pattern_id}' are not "
                            f"supported: {reason}"
                        )
                    else:
                        explanation = (
                            f"{count} instances of operator '{pattern_id}' are not supported"
                        )

                action = Action(
                    pattern_from_id=pattern_id,
                    pattern_to_id="",
                    level=ActionLevel.REQUIRED,
                    status=None,
                    details=(
                        f"Pattern '{pattern_id}' is not supported. "
                        f"Replace or remove unsupported operator '{pattern_id}'. "
                        f"{reason or 'No alternatives available.'}"
                    ),
                )
                actions.append(action)
                logger.debug(
                    "Operator %s: UNSUPPORTED - %d instances",
                    pattern_id,
                    count,
                )

            elif classification == SupportLevel.UNKNOWN:
                # Unknown support - generate information only if doc
                # checker found constraint violations. Check if
                # reason contains DOC_CHECKER_PREFIX to identify
                # doc checker findings.
                if reason and DOC_CHECKER_PREFIX in reason:
                    if count == 1:
                        explanation = (
                            f"Operator '{pattern_id}' has "
                            f"unknown support status with "
                            f"constraint violations: {reason}"
                        )
                    else:
                        explanation = (
                            f"{count} instances of operator "
                            f"'{pattern_id}' have unknown "
                            f"support status with constraint "
                            f"violations: {reason}"
                        )

                    logger.debug(
                        "Operator %s: UNKNOWN with constraint violations - %d instances",
                        pattern_id,
                        count,
                    )
                else:
                    # Skip UNKNOWN operators without constraint
                    # violations
                    logger.debug(
                        "Operator %s: UNKNOWN (status unclear)"
                        " - skipping (no constraint violations)",
                        pattern_id,
                    )
                    continue

            else:  # Should not reach here due to filtering
                logger.debug(
                    "Operator %s: Unexpected classification %s", pattern_id, classification
                )
                continue

            info = Information(
                explanation=explanation,
                actions=actions if actions else None,
                pattern_id=pattern_id,
                pattern_list=runtime_results,  # All runtime results for this group
            )
            info_list.append(info)

        build_info_ms = int((time.perf_counter() - build_info_start) * 1000)

        logger.debug(
            "Generated %d single operator information items from %d operators",
            len(info_list),
            sum(len(results) for results in grouped_results.values()),
        )

        _log_timing(
            "information_engine.check_single_ops",
            ep=self._ep,
            device=self._device,
            runtime_results=len(self._op_runtime_results),
            grouped_items=len(grouped_results),
            generated_info=len(info_list),
            doc_queries=doc_query_count,
            doc_query_total_ms=doc_query_total_ms,
            doc_query_slow_count=doc_query_slow_count,
            max_doc_query_ms=max_doc_query_ms,
            max_doc_query_pattern=max_doc_query_pattern,
            group_loop_ms=group_loop_ms,
            build_info_ms=build_info_ms,
            total_ms=int((time.perf_counter() - total_start) * 1000),
        )

        return info_list

    def _query_doc_constraints(self, runtime_result: PatternRuntime, pattern_id: str) -> str | None:
        """Query doc checker for detailed constraint information.

        Args:
            runtime_result: PatternRuntime object with pattern match info
            pattern_id: Pattern identifier (e.g., "OP/ai.onnx/Conv")

        Returns:
            str: Detailed constraint check reason, or None if not available

        Process:
            1. Extract operator node from pattern_match
            2. Query doc checker for this node
            3. Extract constraint violation details
            4. Format as human-readable reason
        """
        try:
            total_start = time.perf_counter()
            logger.debug("Querying doc constraints for pattern: %s", pattern_id)

            # Extract ONNX node from pattern match
            if not hasattr(runtime_result, "pattern_match") or not runtime_result.pattern_match:
                logger.debug("No pattern_match found for %s", pattern_id)
                return None

            pattern_match = runtime_result.pattern_match

            # PatternMatch has matched_node_names (list[ONNXOp]), not matched_nodes
            if (
                not hasattr(pattern_match, "matched_node_names")
                or not pattern_match.matched_node_names
            ):
                logger.debug("No matched_node_names found in pattern_match for %s", pattern_id)
                return None

            # Get the first matched stable node key (for single-op patterns)
            onnx_op = pattern_match.matched_node_names[0]
            node_key = onnx_op.node_name
            logger.debug(
                "Extracted node key for %s: %s (op_type=%s)",
                pattern_id,
                node_key,
                onnx_op.op_type,
            )

            # Resolve ONNX NodeProto from stable sidecar key
            node_lookup_start = time.perf_counter()
            node = self._model.get_node_by_key(node_key)
            if node is None:
                node = self._model.get_node_by_name(node_key)
            node_lookup_ms = int((time.perf_counter() - node_lookup_start) * 1000)

            if node is None:
                logger.debug("Could not find node %s in model graph", node_key)
                _log_timing(
                    "information_engine.doc_constraints",
                    ep=self._ep,
                    pattern_id=pattern_id,
                    node=node_key,
                    found_node=False,
                    node_lookup_ms=node_lookup_ms,
                    total_ms=int((time.perf_counter() - total_start) * 1000),
                )
                return None

            # Query doc checker
            if self._doc_checker is None:
                logger.debug("Doc checker not initialized, skipping doc constraints query")
                return None
            logger.debug("Running doc checker for node: %s", node.name)
            checker_start = time.perf_counter()
            doc_result = self._doc_checker.run_for_node(node)
            checker_ms = int((time.perf_counter() - checker_start) * 1000)

            logger.debug(
                "Doc checker result for %s: compile=%s, run=%s, no_data=%s, reason=%s",
                pattern_id,
                doc_result.result.compile,
                doc_result.result.run,
                doc_result.result.no_data,
                doc_result.result.reason,
            )

            # Extract detailed reason from doc checker
            if not doc_result.result.compile or not doc_result.result.run:
                doc_reason = doc_result.result.reason
                if doc_reason and doc_reason != "OK":
                    result_msg = f"{DOC_CHECKER_PREFIX} {doc_reason}"
                    logger.debug(
                        "Returning doc constraint reason for %s: %s", pattern_id, result_msg
                    )
                    total_ms = int((time.perf_counter() - total_start) * 1000)
                    if total_ms >= 50:
                        _log_timing(
                            "information_engine.doc_constraints.slow_query",
                            ep=self._ep,
                            pattern_id=pattern_id,
                            node=node_key,
                            node_lookup_ms=node_lookup_ms,
                            checker_ms=checker_ms,
                            total_ms=total_ms,
                            hit_reason=True,
                        )
                    return result_msg
                logger.debug("Doc checker failed but no detailed reason for %s", pattern_id)
            else:
                logger.debug("Doc checker passed for %s, no constraint violations", pattern_id)

            total_ms = int((time.perf_counter() - total_start) * 1000)
            if total_ms >= 50:
                _log_timing(
                    "information_engine.doc_constraints.slow_query",
                    ep=self._ep,
                    pattern_id=pattern_id,
                    node=node_key,
                    node_lookup_ms=node_lookup_ms,
                    checker_ms=checker_ms,
                    total_ms=total_ms,
                    hit_reason=False,
                )

            return None

        except Exception as e:
            logger.warning(
                "Failed to query doc constraints for %s: %s", pattern_id, e, exc_info=True
            )
            return None

    def _validate_runtime_result(self, runtime_result: PatternRuntime) -> bool:
        """Validate pattern runtime data.

        Args:
            runtime_result: PatternRuntime to validate

        Returns:
            bool: True if valid, False if should be skipped
        """
        if not runtime_result.pattern_id:
            logger.warning("Skipping pattern with empty pattern_id")
            return False
        if runtime_result.result.classification is None:
            logger.warning(
                "Skipping pattern %s with None classification",
                runtime_result.pattern_id,
            )
            return False
        return True

    def _determine_action_level_and_status(
        self,
        current_classification: SupportLevel,
        alternative_classification: SupportLevel,
    ) -> tuple[ActionLevel | None, SupportLevel | None]:
        """Determine action level and status based on classification transition.

        Args:
            current_classification: Current pattern classification
            alternative_classification: Alternative pattern classification

        Returns:
            tuple: (ActionLevel, SupportLevel or None)

        Logic:
            - UNSUPPORTED → SUPPORTED/PARTIAL: REQUIRED
            - UNSUPPORTED → UNKNOWN: WARNING
            - PARTIAL → SUPPORTED: REQUIRED
            - UNKNOWN → SUPPORTED/PARTIAL: OPTIONAL
            - Otherwise: No action recommended
        """
        if current_classification == SupportLevel.UNSUPPORTED:
            if alternative_classification == SupportLevel.SUPPORTED:
                return ActionLevel.REQUIRED, SupportLevel.SUPPORTED
            if alternative_classification == SupportLevel.PARTIAL:
                return ActionLevel.REQUIRED, SupportLevel.PARTIAL
            if alternative_classification == SupportLevel.UNKNOWN:
                return ActionLevel.WARNING, None

        if (
            current_classification == SupportLevel.PARTIAL
            and alternative_classification == SupportLevel.SUPPORTED
        ):
            return ActionLevel.REQUIRED, SupportLevel.SUPPORTED

        if current_classification == SupportLevel.UNKNOWN and alternative_classification in (
            SupportLevel.SUPPORTED,
            SupportLevel.PARTIAL,
        ):
            return ActionLevel.OPTIONAL, alternative_classification

        # No improvement or already optimal
        return None, alternative_classification

    def _create_action(
        self,
        pattern_from_id: str,
        pattern_to_id: str,
        level: ActionLevel,
        status: SupportLevel | None,
        alt_type: str | None = None,
        action_id: str | None = None,
        action_items: list | None = None,
        enabled: bool = True,
    ) -> Action:
        """Create an Action object with appropriate details.

        Args:
            pattern_from_id: Original pattern identifier
            pattern_to_id: Target pattern identifier
            level: Action priority level
            status: Expected support level after action
            alt_type: Alternative type description
            action_id: Optional action ID (for predefined actions)
            action_items: Optional action items (for predefined actions)
            enabled: Whether action is enabled

        Returns:
            Action: Configured action object
        """
        from ..models.information import Action

        # Generate details based on level and status
        if level == ActionLevel.REQUIRED:
            if status == SupportLevel.SUPPORTED:
                details = (
                    f"Pattern '{pattern_from_id}' is not supported. "
                    f"Replace '{pattern_from_id}' with '{pattern_to_id}'"
                )
                if alt_type:
                    details += (
                        f" ({alt_type} alternative). "
                        f"Alternative '{pattern_to_id}' ({alt_type}) is fully supported."
                    )
            elif status == SupportLevel.PARTIAL:
                details = (
                    f"Pattern '{pattern_from_id}' is not supported. "
                    f"Replace '{pattern_from_id}' with '{pattern_to_id}'"
                )
                if alt_type:
                    details += (
                        f" ({alt_type} alternative, partial support). "
                        f"Alternative '{pattern_to_id}' ({alt_type}) has partial support."
                    )
            else:
                details = (
                    f"Pattern '{pattern_from_id}' is not supported. "
                    f"Replace or remove unsupported operator '{pattern_from_id}'."
                )

        elif level == ActionLevel.OPTIONAL:
            status_text = (
                "fully supported" if status == SupportLevel.SUPPORTED else "partially supported"
            )
            current_status = "unknown support status" if status else "partial support"
            details = (
                f"Pattern '{pattern_from_id}' has {current_status}. "
                f"Consider using '{pattern_to_id}' instead of '{pattern_from_id}'"
            )
            if alt_type:
                details += (
                    f" ({alt_type} alternative, {status_text}). "
                    f"Alternative '{pattern_to_id}' ({alt_type}) is {status_text}"
                )
                if status == SupportLevel.SUPPORTED:
                    details += " and may improve performance."
                details += "."

        elif level == ActionLevel.WARNING:
            if pattern_to_id:
                details = (
                    f"Pattern '{pattern_from_id}' is not supported. "
                    f"Alternative '{pattern_to_id}' for '{pattern_from_id}' "
                    f"has unknown status. Verify before replacing."
                )
                if alt_type:
                    details += (
                        f" Alternative '{pattern_to_id}' ({alt_type}) has unknown support status."
                    )
            else:
                details = (
                    f"Pattern '{pattern_from_id}' is not supported "
                    f"and no alternatives are available. "
                    f"Manual replacement or removal required."
                )
        else:
            details = f"Pattern '{pattern_from_id}' status requires review."

        action_kwargs: dict[str, Any] = {
            "pattern_from_id": pattern_from_id,
            "pattern_to_id": pattern_to_id,
            "level": level,
            "status": status,
            "details": details,
            "enabled": enabled,
        }

        if action_id:
            action_kwargs["action_id"] = action_id
        if action_items:
            action_kwargs["action_items"] = action_items

        return Action(**action_kwargs)

    def _check_patterns(self) -> list[Information]:
        """Generate information for patterns with alternatives.

        Returns:
            List[Information]: Information objects for patterns with alternatives

        Process:
            1. First pass: Match predefined information rules by pattern_id
            2. Second pass: Generate information dynamically for unmatched patterns:
               - Iterate through op_runtime_results and subgraph_runtime_results
               - For each runtime result, check result.classification:
                 * UNSUPPORTED: Required action for unsupported pattern
                 * PARTIAL: Optional action to enable fusion/optimization
                 * SUPPORTED: No action needed
               - Include alternative patterns from runtime_result.alternatives
            3. Return combined results (predefined first, then generated)
            4. Deduplicate information by Information_id
        """
        logger.debug("Checking patterns with alternatives")
        total_start = time.perf_counter()

        info_list: list[Information] = []
        matched_pattern_ids: set[str] = set()
        seen_info_ids: set[str] = set()

        # Collect all runtime results by pattern_id for aggregation
        from collections import defaultdict

        pattern_to_runtime_results: dict[str, list[PatternRuntime]] = defaultdict(list)

        # Collect all runtime results for potential predefined information matching
        collect_runtime_start = time.perf_counter()
        for runtime_result in self._op_runtime_results + self._subgraph_runtime_results:
            pattern_id = runtime_result.pattern_id
            pattern_to_runtime_results[pattern_id].append(runtime_result)
        collect_runtime_ms = int((time.perf_counter() - collect_runtime_start) * 1000)

        # First pass: Match predefined information rules with aggregated runtime results
        predefined_start = time.perf_counter()
        for pattern_id, runtime_results in pattern_to_runtime_results.items():
            if pattern_id in self._info_by_pattern:
                predefined_info = self._info_by_pattern[pattern_id]
                matched_pattern_ids.add(pattern_id)

                # Deduplicate by Information_id
                if predefined_info.Information_id not in seen_info_ids:
                    # Process predefined information with all runtime results for this pattern
                    processed_info = self._process_predefined_information(
                        predefined_info, runtime_results
                    )
                    info_list.append(processed_info)
                    seen_info_ids.add(predefined_info.Information_id)
                    logger.debug(
                        "Matched predefined information for pattern: %s (%d instances)",
                        pattern_id,
                        len(runtime_results),
                    )
                else:
                    logger.debug(
                        "Skipped duplicate predefined information for pattern: %s (ID: %s)",
                        pattern_id,
                        predefined_info.Information_id,
                    )
        predefined_ms = int((time.perf_counter() - predefined_start) * 1000)

        # Second pass: Generate information dynamically for unmatched patterns
        # Process operator patterns with alternatives
        op_alternatives_start = time.perf_counter()
        generated_op_alt_info = 0
        for runtime_result in self._op_runtime_results:
            if runtime_result.pattern_id in matched_pattern_ids:
                continue
            if not runtime_result.alternatives:
                continue

            info = self._process_pattern_with_alternatives(runtime_result)
            if info:
                info_list.append(info)
                generated_op_alt_info += 1
        op_alternatives_ms = int((time.perf_counter() - op_alternatives_start) * 1000)

        # Process subgraph patterns with aggregation
        from collections import defaultdict

        from ..models.information import Information

        grouped_subgraph_results: dict[
            tuple[str, SupportLevel, str | None], list[PatternRuntime]
        ] = defaultdict(list)

        group_subgraph_start = time.perf_counter()
        for runtime_result in self._subgraph_runtime_results:
            if runtime_result.pattern_id in matched_pattern_ids:
                continue

            pattern_id = runtime_result.pattern_id
            classification = runtime_result.result.classification
            reason = runtime_result.result.reason

            # Group by (pattern_id, classification, reason)
            key = (pattern_id, classification, reason)
            grouped_subgraph_results[key].append(runtime_result)
        group_subgraph_ms = int((time.perf_counter() - group_subgraph_start) * 1000)

        # Generate Information for each aggregated subgraph group
        process_subgraph_start = time.perf_counter()
        generated_subgraph_info = 0
        for (
            pattern_id,
            classification,
            _reason,
        ), runtime_results in grouped_subgraph_results.items():
            count = len(runtime_results)

            # Use the first runtime_result as representative for alternatives
            # (all instances of the same pattern should have same alternatives)
            representative = runtime_results[0]

            # SUPPORTED patterns with no alternatives don't need information
            if classification == SupportLevel.SUPPORTED and not representative.alternatives:
                continue

            # Generate explanation with count
            if count == 1:
                if classification == SupportLevel.SUPPORTED:
                    explanation = f"Pattern '{pattern_id}' is fully supported"
                elif classification == SupportLevel.PARTIAL:
                    explanation = (
                        f"Pattern '{pattern_id}' has partial support (compiles but not optimized)"
                    )
                elif classification == SupportLevel.UNSUPPORTED:
                    explanation = f"Pattern '{pattern_id}' is not supported"
                else:  # UNKNOWN
                    explanation = f"Pattern '{pattern_id}' has unknown support status"
            else:
                if classification == SupportLevel.SUPPORTED:
                    explanation = f"{count} instances of pattern '{pattern_id}' are fully supported"
                elif classification == SupportLevel.PARTIAL:
                    explanation = (
                        f"{count} instances of pattern '{pattern_id}' have partial support "
                        f"(compile but not optimized)"
                    )
                elif classification == SupportLevel.UNSUPPORTED:
                    explanation = f"{count} instances of pattern '{pattern_id}' are not supported"
                else:  # UNKNOWN
                    explanation = (
                        f"{count} instances of pattern '{pattern_id}' have unknown support status"
                    )

            # Extract actions from representative (same for all instances)
            actions = self._extract_actions(representative)

            logger.debug(
                "Pattern %s: %s with %d actions - %d instances",
                pattern_id,
                classification.value if classification else "UNKNOWN",
                len(actions),
                count,
            )

            info = Information(
                explanation=explanation,
                actions=actions if actions else None,
                pattern_id=pattern_id,
                pattern_list=runtime_results,  # All aggregated runtime results
            )
            info_list.append(info)
            generated_subgraph_info += 1
        process_subgraph_ms = int((time.perf_counter() - process_subgraph_start) * 1000)

        logger.debug(
            "Generated %d pattern information items "
            "(%d predefined, %d generated, "
            "%d aggregated subgraphs)",
            len(info_list),
            len(matched_pattern_ids),
            len(info_list) - len(seen_info_ids) - len(grouped_subgraph_results),
            len(grouped_subgraph_results),
        )

        _log_timing(
            "information_engine.check_patterns",
            ep=self._ep,
            device=self._device,
            unique_patterns=len(pattern_to_runtime_results),
            matched_predefined_patterns=len(matched_pattern_ids),
            predefined_info_ids=len(seen_info_ids),
            generated_op_alt_info=generated_op_alt_info,
            grouped_subgraph_keys=len(grouped_subgraph_results),
            generated_subgraph_info=generated_subgraph_info,
            collect_runtime_ms=collect_runtime_ms,
            predefined_ms=predefined_ms,
            op_alternatives_ms=op_alternatives_ms,
            group_subgraph_ms=group_subgraph_ms,
            process_subgraph_ms=process_subgraph_ms,
            total_ms=int((time.perf_counter() - total_start) * 1000),
        )

        return info_list

    def _process_predefined_information(
        self, predefined_info: Information, runtime_results: list[PatternRuntime]
    ) -> Information:
        """Process predefined information by filling in action level and status.

        Args:
            predefined_info: Predefined Information object from JSON rules
            runtime_results: List of PatternRuntime objects for all instances of this pattern

        Returns:
            Information: Processed information with updated action level and status

        Process:
            1. Use first runtime_result as representative for classification and alternatives
            2. Check runtime_result.result.classification
            3. Check runtime_result.alternatives for each action
            4. Fill in action.level and action.status based on classifications
            5. Include all runtime_results in pattern_list
        """
        from copy import deepcopy

        from ..models.information import Action, Information

        # Deep copy to avoid modifying the original predefined info
        processed_info = deepcopy(predefined_info)

        if not processed_info.actions:
            return Information(
                Information_id=processed_info.Information_id,
                explanation=processed_info.explanation,
                actions=None,
                pattern_id=processed_info.pattern_id,
                enabled=processed_info.enabled,
                status=runtime_results[0].result.classification if runtime_results else None,
                pattern_list=runtime_results,
            )

        # Use first runtime_result as representative
        # (all should have same classification/alternatives)
        runtime_result = runtime_results[0]
        pattern_id = runtime_result.pattern_id
        classification = runtime_result.result.classification

        # Build a lookup map for alternatives by pattern_id
        alternatives_map: dict[str, PatternAlternative] = {
            alt.pattern_id: alt for alt in runtime_result.alternatives
        }

        # Process each action in predefined information
        updated_actions: list[Action] = []
        for action in processed_info.actions:
            # Create a new action with updated level and status
            pattern_to_id = action.pattern_to_id

            # If action has no target pattern (general recommendation), keep as-is
            if not pattern_to_id:
                updated_actions.append(action)
                continue

            # Check if this action's target pattern exists in alternatives
            if pattern_to_id not in alternatives_map:
                logger.warning(
                    "Action target pattern '%s' not found in alternatives for '%s'",
                    pattern_to_id,
                    pattern_id,
                )
                updated_actions.append(action)
                continue

            # Get alternative runtime result
            alternative = alternatives_map[pattern_to_id]
            alt_classification = alternative.result.classification

            # Validate alternative data
            if alt_classification is None:
                logger.warning(
                    "Alternative %s has None classification, keeping original action",
                    pattern_to_id,
                )
                updated_actions.append(action)
                continue

            # Determine level and status based on classifications
            level, status = self._determine_action_level_and_status(
                classification, alt_classification
            )

            # If no improvement, keep original action
            if level is None:
                updated_actions.append(action)
                continue

            # Create updated action with filled level and status
            updated_action = Action(
                action_id=action.action_id,
                pattern_from_id=action.pattern_from_id,
                pattern_to_id=action.pattern_to_id,
                level=level,
                status=status,
                action_items=action.action_items,
                enabled=action.enabled,
                details=action.details,
            )
            updated_actions.append(updated_action)

        # Create new Information with updated actions and status
        return Information(
            Information_id=processed_info.Information_id,
            explanation=processed_info.explanation,
            actions=updated_actions if updated_actions else None,
            pattern_id=processed_info.pattern_id,
            enabled=processed_info.enabled,
            status=classification,  # Set information status from runtime result
            pattern_list=runtime_results,  # All aggregated runtime results
        )

    def _process_pattern_with_alternatives(
        self, pattern_runtime: PatternRuntime
    ) -> Information | None:
        """Process a single pattern with alternatives.

        Args:
            pattern_runtime: PatternRuntime object with result and alternatives

        Returns:
            Information object with actions, or None if pattern is supported with no alternatives
            (indicating no action needed)
        """
        from ..models.information import Information

        if not self._validate_runtime_result(pattern_runtime):
            return None

        pattern_id = pattern_runtime.pattern_id
        classification = pattern_runtime.result.classification

        # SUPPORTED patterns with no alternatives don't need information
        if classification == SupportLevel.SUPPORTED and not pattern_runtime.alternatives:
            return None

        # Generate explanation based on classification
        if classification == SupportLevel.SUPPORTED:
            explanation = f"Pattern '{pattern_id}' is fully supported"
        elif classification == SupportLevel.PARTIAL:
            explanation = f"Pattern '{pattern_id}' has partial support (compiles but not optimized)"
        elif classification == SupportLevel.UNSUPPORTED:
            explanation = f"Pattern '{pattern_id}' is not supported"
        else:  # UNKNOWN
            explanation = f"Pattern '{pattern_id}' has unknown support status"

        # Extract actions from pattern and alternatives
        actions = self._extract_actions(pattern_runtime)

        logger.debug(
            "Pattern %s: %s with %d actions",
            pattern_id,
            classification.value,
            len(actions),
        )

        return Information(
            explanation=explanation,
            actions=actions if actions else None,
            pattern_id=pattern_id,
            pattern_list=[pattern_runtime],
        )

    def _extract_actions(self, pattern_runtime: PatternRuntime) -> list[Action]:
        """Extract and format action items from runtime result.

        Args:
            pattern_runtime: PatternRuntime object with result and alternatives

        Returns:
            List[Action]: Formatted action objects

        Process:
            1. Check pattern_runtime.result.classification
            2. For each alternative, determine if it's better
            3. Create actions using helper method
            4. Handle case with no alternatives for unsupported patterns
        """
        actions: list[Action] = []
        pattern_id = pattern_runtime.pattern_id
        classification = pattern_runtime.result.classification

        # Process alternatives
        for alternative in pattern_runtime.alternatives:
            alt_pattern_id = alternative.pattern_id
            alt_classification = alternative.result.classification

            # Validate alternative data
            if not alt_pattern_id or alt_classification is None:
                logger.warning("Skipping invalid alternative for pattern %s", pattern_id)
                continue
            if alternative.alternative_type is None:
                logger.warning("Skipping alternative %s with None type", alt_pattern_id)
                continue

            alt_type = alternative.alternative_type.value

            # Determine action level and status
            level, status = self._determine_action_level_and_status(
                classification, alt_classification
            )

            # Skip if no improvement (None level)
            if level is None:
                continue

            # Create action using helper
            action = self._create_action(
                pattern_from_id=pattern_id,
                pattern_to_id=alt_pattern_id,
                level=level,
                status=status,
                alt_type=alt_type,
            )
            actions.append(action)

        # If no alternatives found but pattern is UNSUPPORTED, add warning
        if classification == SupportLevel.UNSUPPORTED and not actions:
            reason = pattern_runtime.result.reason
            action = self._create_action(
                pattern_from_id=pattern_id,
                pattern_to_id="",
                level=ActionLevel.WARNING,
                status=None,
            )
            # Override details with reason
            if reason:
                action.details = (
                    f"Pattern '{pattern_id}' is not supported "
                    f"and no alternatives are available. {reason}"
                )
            actions.append(action)

        return actions
