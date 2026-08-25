# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Pattern-based subgraph rewriting pipe.

Uses PatternMatcher + PatternRewriter to replace matched subgraphs with
alternative patterns. Active rewrites are selected via --enable-{cap} CLI
flags, where each capability corresponds to a RewriteGroup from the JSON
rule files in modelkit/pattern/rules/.

Design notes
------------
- Capabilities are auto-generated from REWRITE_CAPABILITIES (JSON-derived).
- No pattern class names are hardcoded here (CARDINAL RULE #1).
- PatternMatcher + PatternRewriter live in modelkit.pattern.
- Conflict detection: matches sharing nodes trigger a WARNING and are all
  skipped — no partial rewrites.
- The pipe returns the model unchanged when no rules are configured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from .base import BasePipe, OptimizationError, PipeConfig
from .rewrite_rules import REWRITE_CAPABILITIES, RewriteRule


if TYPE_CHECKING:
    import onnx

    from winml.modelkit.pattern.match import PatternMatchResult

logger = logging.getLogger(__name__)


@dataclass
class RewritePipeConfig(PipeConfig):
    """Configuration for the pattern-rewriting pipe.

    Attributes:
        rules: Ordered list of rewrite rules to apply.
    """

    rules: list[RewriteRule] = field(default_factory=list)


def _detect_conflicts(
    matches: list[PatternMatchResult],
) -> tuple[list[PatternMatchResult], list[list[PatternMatchResult]]]:
    """Separate clean matches from conflicting groups.

    Two matches conflict when their matched node sets share at least one node.
    All matches in a conflicting group are returned together so the caller can
    skip all of them (no partial rewrites).

    Args:
        matches: All candidate PatternMatchResult objects.

    Returns:
        (clean, conflicts) where:
          clean:     matches with no node overlap — safe to rewrite.
          conflicts: list of conflict groups (each group has ≥2 matches).
    """
    node_to_match_indices: dict[str, list[int]] = {}
    for i, m in enumerate(matches):
        for node_name in m.skeleton_match_result.matched_node_keys:
            node_to_match_indices.setdefault(node_name, []).append(i)

    conflicting_indices: set[int] = set()
    # Use dict keyed by frozenset to deduplicate groups (same conflict reported
    # once per shared node but we want one group per set of conflicting matches).
    seen_groups: dict[frozenset[int], list[PatternMatchResult]] = {}
    for idxs in node_to_match_indices.values():
        if len(idxs) > 1:
            key = frozenset(idxs)
            if key not in seen_groups:
                seen_groups[key] = [matches[i] for i in idxs]
            conflicting_indices.update(idxs)

    clean = [m for i, m in enumerate(matches) if i not in conflicting_indices]
    return clean, list(seen_groups.values())


class RewritePipe(BasePipe[RewritePipeConfig]):
    """Pattern-based subgraph rewriting pipe.

    A pure graph transformation tool. Accepts rewrite rules via bool
    capabilities (--enable-{source}-{target} flags, all lowercase).
    Does not load configs, does not know about IHVs.

    Capabilities are auto-generated from JSON rule files in
    modelkit/pattern/rules/ — every RewriteGroup becomes one BoolCapability.
    """

    name: ClassVar[str] = "rewrite"
    capabilities: ClassVar[dict[str, Any]] = REWRITE_CAPABILITIES

    def __init__(self) -> None:
        self._analysis_matches: list[PatternMatchResult] | None = None

    @classmethod
    def build_config(cls, **kwargs: Any) -> RewritePipeConfig:
        """Build config from kwargs (populated by capability_options decorator).

        Scans all rewrite capabilities. For each enabled capability, expands
        the RewriteGroup into one RewriteRule per source variant.

        Examples:
            --enable-subgraphgelupattern-singlegelupattern
            → kwargs["subgraphgelupattern_singlegelupattern"] = True
            → 4 RewriteRules (one per Gelu variant → SingleGeluPattern)

            --enable-matmuladdpattern-reshapegemmreshapepattern
            → kwargs["matmuladdpattern_reshapegemmreshapepattern"] = True
            → 1 RewriteRule (MatMulAddPattern → ReshapeGemmReshapePattern)
        """
        from .rewrite_rules import get_rewrite_rules_for_capability

        rules: list[RewriteRule] = []
        for cap_name, cap in REWRITE_CAPABILITIES.items():
            if kwargs.get(cap.python_name) is True:
                rules.extend(get_rewrite_rules_for_capability(cap_name))

        return RewritePipeConfig(rules=rules)

    @classmethod
    def should_process(cls, config: RewritePipeConfig) -> bool:
        """Return True only when at least one rewrite rule is configured."""
        return len(config.rules) > 0

    def process(
        self,
        model: onnx.ModelProto,
        config: RewritePipeConfig,
    ) -> onnx.ModelProto:
        """Apply pattern rewrites to the model.

        Args:
            model: Input ONNX model (not modified in place).
            config: Pipe configuration from build_config().

        Returns:
            New ONNX model with rewritten subgraphs, or the original model
            object when no rules are configured or no matches are found.
        """
        return self._process(model, config, skip_shape_inference=False)

    def prepare_analysis_model(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Infer shapes and match all source patterns once for analysis."""
        from ...onnx import infer_onnx_shapes
        from ...pattern.base import PatternMatcher
        from .rewrite_rules import get_rewrite_rules_for_capability

        prepared = model.__class__()
        prepared.CopyFrom(model)
        prepared = infer_onnx_shapes(prepared)

        matcher = PatternMatcher(prepared, skip_shape_inference=True)
        for cap_name in REWRITE_CAPABILITIES:
            for rule in get_rewrite_rules_for_capability(cap_name):
                matcher.register_pattern(rule.source)
        self._analysis_matches = matcher.match()
        return prepared

    def process_analysis(
        self,
        model: onnx.ModelProto,
        config: RewritePipeConfig,
    ) -> onnx.ModelProto:
        """Probe a rewrite using shape information and matches prepared once."""
        if self._analysis_matches is None:
            return self._process(model, config, skip_shape_inference=True)
        source_names = {rule.source.__class__.__name__ for rule in config.rules}
        matches = [
            match
            for match in self._analysis_matches
            if match.skeleton_match_result.pattern.__class__.__name__ in source_names
        ]
        return self._rewrite_matches(model, config, matches)

    @classmethod
    def requires_analysis_clone(cls) -> bool:
        """Pattern matching is read-only and rewriting creates its own copy."""
        return False

    def _process(
        self,
        model: onnx.ModelProto,
        config: RewritePipeConfig,
        *,
        skip_shape_inference: bool,
    ) -> onnx.ModelProto:
        """Apply configured rewrites, optionally reusing inferred shapes."""
        if not config.rules:
            return model

        try:
            from ...pattern.base import PatternMatcher

            # Register source patterns for this normal optimization run.
            matcher = PatternMatcher(model, skip_shape_inference=skip_shape_inference)
            for rule in config.rules:
                matcher.register_pattern(rule.source)

            all_matches = matcher.match()
            return self._rewrite_matches(model, config, all_matches)

        except OptimizationError:
            raise
        except Exception as e:
            raise OptimizationError(
                f"Pattern matching/rewriting failed: {e}",
                pipe_name="rewrite",
                cause=e,
            ) from e

    def _rewrite_matches(
        self,
        model: onnx.ModelProto,
        config: RewritePipeConfig,
        all_matches: list[PatternMatchResult],
    ) -> onnx.ModelProto:
        """Apply configured rewrites to an existing set of pattern matches."""
        if not config.rules:
            return model
        if not all_matches:
            logger.debug("RewritePipe: no pattern matches found; model unchanged.")
            return model

        try:
            from ...pattern.base import PatternRewriter

            source_to_target: dict[str, type] = {}
            target_registry: dict[str, type] = {}
            for rule in config.rules:
                target_cls = type(rule.target)
                source_to_target[rule.source.__class__.__name__] = target_cls
                target_registry[target_cls.__name__] = target_cls

            # Map each match to its target class
            candidate_pairs: list[tuple[PatternMatchResult, type]] = []
            for m in all_matches:
                src_name = m.skeleton_match_result.pattern.__class__.__name__
                mapped_cls = source_to_target.get(src_name)
                if mapped_cls is not None:
                    candidate_pairs.append((m, mapped_cls))

            if not candidate_pairs:
                matched_class_names = sorted(
                    {m.skeleton_match_result.pattern.__class__.__name__ for m in all_matches}
                )
                registered_source_names = sorted(source_to_target.keys())
                logger.warning(
                    "RewritePipe: %d match(es) found but none mapped to a rewrite target. "
                    "Matched pattern classes: %s; registered source classes: %s",
                    len(all_matches),
                    matched_class_names,
                    registered_source_names,
                )
                return model

            # Conflict detection: warn and drop all matches that share nodes
            candidates = [m for m, _ in candidate_pairs]
            clean_matches, conflicts = _detect_conflicts(candidates)

            for group in conflicts:
                names = [m.skeleton_match_result.pattern.__class__.__name__ for m in group]
                shared_node_sets = [
                    set(m.skeleton_match_result.matched_node_keys) for m in group
                ]
                shared_nodes = sorted(set.intersection(*shared_node_sets))
                logger.warning(
                    "RewritePipe: conflict detected — %d matches share nodes %s; "
                    "skipping all conflicting matches: %s",
                    len(group),
                    shared_nodes,
                    names,
                )

            clean_ids = {id(m) for m in clean_matches}
            rewrite_pairs = [(m, t) for m, t in candidate_pairs if id(m) in clean_ids]

            if not rewrite_pairs:
                logger.debug("RewritePipe: all matches were conflicted; model unchanged.")
                return model

            # Group by target class for PatternRewriter
            grouped: dict[str, list[PatternMatchResult]] = {}
            for m, target_cls in rewrite_pairs:
                grouped.setdefault(target_cls.__name__, []).append(m)

            rewrite_args = [
                (match_list, target_registry[target_name])
                for target_name, match_list in grouped.items()
                if target_name in target_registry
            ]

            total = sum(len(ml) for ml, _ in rewrite_args)
            logger.info(
                "RewritePipe: rewriting %d match(es) across %d target class(es).",
                total,
                len(rewrite_args),
            )

            rewriter = PatternRewriter(model)
            return rewriter.rewrite(rewrite_args)

        except OptimizationError:
            raise
        except Exception as e:
            raise OptimizationError(
                f"Pattern matching/rewriting failed: {e}",
                pipe_name="rewrite",
                cause=e,
            ) from e
