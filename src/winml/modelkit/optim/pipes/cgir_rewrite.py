# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CGIR-specific model rewrites applied before target EP compilation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from onnx import AttributeProto, version_converter

from ...onnx import ONNXDomain, check_onnx_model, get_captured_tensor_names
from ...pattern import PatternMatcher, PatternRewriter
from .base import BasePipe, OptimizationError, PipeConfig
from .cgir_rewrite_rules import (
    CGIR_REWRITE_CAPABILITIES,
    CGIR_REWRITE_RULES,
    CGIRModelRewriteRule,
    CGIRRewriteRule,
)


if TYPE_CHECKING:
    import onnx


logger = logging.getLogger(__name__)


class _CGIRPatternRewriter(PatternRewriter):
    """Keep subgraph captures alive during CGIR-specific constant cleanup."""

    def _remove_unused_constants(self, model: onnx.ModelProto) -> None:
        graph = model.graph
        consumed = {name for node in graph.node for name in node.input if name}
        consumed.update(output.name for output in graph.output)
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == AttributeProto.GRAPH:
                    consumed.update(get_captured_tensor_names(attribute.g))
                elif attribute.type == AttributeProto.GRAPHS:
                    for subgraph in attribute.graphs:
                        consumed.update(get_captured_tensor_names(subgraph))

        for node in list(graph.node):
            if node.op_type == "Constant" and not consumed.intersection(node.output):
                graph.node.remove(node)
        removed_initializers = {
            initializer.name for initializer in graph.initializer
            if initializer.name not in consumed
        }
        for initializer in list(graph.initializer):
            if initializer.name in removed_initializers:
                graph.initializer.remove(initializer)
        for graph_input in list(graph.input):
            if graph_input.name in removed_initializers:
                graph.input.remove(graph_input)


@dataclass
class CGIRRewritePipeConfig(PipeConfig):
    """Configuration for target-specific CGIR rewrites."""

    rules: list[CGIRRewriteRule | CGIRModelRewriteRule] = field(default_factory=list)


class CGIRRewritePipe(BasePipe[CGIRRewritePipeConfig]):
    """Apply CGIR compatibility rewrites that may prepare the full model."""

    name: ClassVar[str] = "cgir_rewrite"
    capabilities: ClassVar[dict[str, Any]] = CGIR_REWRITE_CAPABILITIES

    @classmethod
    def get_compatibility_options(cls) -> dict[str, bool]:
        """Return options enabling every compatibility rule, without redundant aliases."""
        return {rule.capability.python_name: True for rule in CGIR_REWRITE_RULES}

    @classmethod
    def build_config(cls, **kwargs: Any) -> CGIRRewritePipeConfig:
        """Build the enabled CGIR rewrite configuration."""
        rules = [
            rule
            for rule in CGIR_REWRITE_RULES
            if any(
                kwargs.get(capability.python_name) is True
                for capability in (rule.capability, *rule.aliases)
            )
        ]
        return CGIRRewritePipeConfig(rules=rules)

    @classmethod
    def should_process(cls, config: CGIRRewritePipeConfig) -> bool:
        """Return whether at least one CGIR rewrite is enabled."""
        return bool(config.rules)

    def process(
        self,
        model: onnx.ModelProto,
        config: CGIRRewritePipeConfig,
    ) -> onnx.ModelProto:
        """Apply each enabled rule once, matching against the preceding rule's result."""
        if not config.rules:
            return model

        try:
            rewritten_model = model
            matcher: PatternMatcher | None = None
            for rule in config.rules:
                if isinstance(rule, CGIRModelRewriteRule):
                    if rule.minimum_opset and not any(
                        entry.domain == "" and entry.version >= rule.minimum_opset
                        for entry in rewritten_model.opset_import
                    ):
                        continue
                    prepared_model = rule.transform(rewritten_model)
                    if prepared_model is not rewritten_model:
                        logger.info(
                            "CGIR compatibility: %s applied model rewrite",
                            rule.capability.name,
                        )
                    rewritten_model = prepared_model
                    matcher = None
                    continue
                if matcher is None:
                    matcher = PatternMatcher(rewritten_model)
                matcher.patterns.clear()
                matcher.register_pattern(rule.source())
                matches = matcher.match()
                if not matches:
                    continue

                current_opset = matcher.domain_versions.get(ONNXDomain.AI_ONNX)
                if current_opset is None:
                    raise ValueError("Model does not declare the default ONNX opset")
                if current_opset < rule.minimum_opset:
                    rewritten_model = version_converter.convert_version(
                        rewritten_model,
                        rule.minimum_opset,
                    )
                    matcher = PatternMatcher(rewritten_model)
                    matcher.register_pattern(rule.source())
                    matches = matcher.match()
                    if not matches:
                        continue

                original_outputs = {match.skeleton_match_result.output for match in matches}
                rewritten_model = _CGIRPatternRewriter(rewritten_model).rewrite(
                    [(matches, rule.target)],
                )
                residual_matcher = PatternMatcher(rewritten_model)
                residual_matcher.register_pattern(rule.source())
                # A folded Cast can expose a downstream match; do not fold to a fixed point.
                if any(
                    match.skeleton_match_result.output in original_outputs
                    for match in residual_matcher.match()
                ):
                    raise ValueError(
                        f"CGIR rule {rule.capability.name} left selected source patterns "
                        "in the model",
                    )
                matcher = residual_matcher
                if rule.warning:
                    logger.warning("%s (%d match(es))", rule.warning, len(matches))
                logger.info(
                    "CGIR compatibility: %s rewrote %d node(s)", rule.capability.name, len(matches)
                )
            if rewritten_model is not model:
                check_onnx_model(rewritten_model)
            return rewritten_model
        except OptimizationError:
            raise
        except Exception as error:
            raise OptimizationError(
                f"CGIR rewrite failed: {error}",
                pipe_name=self.name,
                cause=error,
            ) from error


__all__ = ["CGIRRewritePipe", "CGIRRewritePipeConfig"]
