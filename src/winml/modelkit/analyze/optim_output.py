# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""EP/device support of the operators an optimization would produce.

This bridges the optimizer's dry-run applicability analysis with the analyze
runtime-support rule data. For each applicable optimization it takes the model
that optimization would produce, then checks whether the operators the
optimization *introduces* (its added and modified nodes) are supported on a
target execution provider and device — answering "if I apply this optimization,
will its output run on my target?".

The check is universal and architecture-agnostic (CARDINAL RULE #1): the set of
produced operators is derived entirely from the concrete graph diff carried on
each :class:`~winml.modelkit.optim.CapabilityFinding`; nothing about specific
operators or model architectures is hardcoded.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from ..optim.analysis import NodeRef
from .models.onnx_model import ONNXModel
from .models.support_level import SupportLevel


if TYPE_CHECKING:
    from collections.abc import Iterable

    from onnx import ModelProto, NodeProto

    from ..optim import CapabilityFinding
    from ..utils.constants import EPName


logger = logging.getLogger(__name__)

# Ordering used to summarize a mix of per-operator support levels into a single
# "worst" level. Higher rank = more concerning for the target.
_SEVERITY: dict[SupportLevel, int] = {
    SupportLevel.SUPPORTED: 0,
    SupportLevel.UNKNOWN: 1,
    SupportLevel.PARTIAL: 2,
    SupportLevel.UNSUPPORTED: 3,
}


@dataclass(frozen=True)
class ProducedOperatorSupport:
    """Support of a single operator an optimization would introduce.

    Attributes:
        op_type: The produced operator's op type (e.g. ``"Gemm"``).
        label: Human-readable node label (op type + name/output).
        change: Whether the node is ``"added"`` or ``"modified"`` by the
            optimization.
        support: Support level of the operator on the target EP/device.
        reason: Optional reason string from the runtime check.
    """

    op_type: str
    label: str
    change: str
    support: SupportLevel
    reason: str | None = None

    def to_dict(self) -> dict[str, str]:
        """Return the stable JSON representation of this produced operator."""
        data = {
            "op_type": self.op_type,
            "label": self.label,
            "change": self.change,
            "support": self.support.value,
        }
        if self.reason is not None:
            data["reason"] = self.reason
        return data


@dataclass
class OptimizationOutputSupport:
    """Per-optimization view of its produced operators' target support.

    Attributes:
        name: Capability name (kebab-case, e.g. ``"matmul-add-fusion"``).
        enable_flag: CLI flag that turns the optimization on.
        category: Capability category (e.g. ``"matmul"``).
        description: Human-readable capability description.
        pipe_name: Name of the pipe that owns the capability.
        operators: Per-operator support for every produced (added/modified) node.
        error: Non-empty when the support check could not be completed.
    """

    name: str
    enable_flag: str
    category: str
    description: str
    pipe_name: str
    removed_nodes: list[NodeRef] = field(default_factory=list)
    added_nodes: list[NodeRef] = field(default_factory=list)
    modified_nodes: list[NodeRef] = field(default_factory=list)
    removed_initializers: list[str] = field(default_factory=list)
    added_initializers: list[str] = field(default_factory=list)
    modified_initializers: list[str] = field(default_factory=list)
    operators: list[ProducedOperatorSupport] = field(default_factory=list)
    error: str | None = None

    @property
    def worst_support(self) -> SupportLevel:
        """Most concerning support level across produced operators.

        Returns ``SUPPORTED`` when there are no produced operators to check.
        """
        if not self.operators:
            return SupportLevel.SUPPORTED
        return max((op.support for op in self.operators), key=lambda level: _SEVERITY[level])

    def support_counts(self) -> dict[SupportLevel, int]:
        """Return a count of produced operators per support level."""
        return dict(Counter(op.support for op in self.operators))

    @staticmethod
    def _node_ref_dict(ref: NodeRef) -> dict[str, object]:
        """Return the stable JSON representation of one graph-delta node."""
        data: dict[str, object] = {
            "op_type": ref.op_type,
            "name": ref.name,
            "outputs": list(ref.outputs),
        }
        if ref.domain and ref.domain != "ai.onnx":
            data["domain"] = ref.domain
        return data

    def to_dict(self) -> dict[str, object]:
        """Return actionable graph-delta and target-support evidence."""
        data: dict[str, object] = {
            "name": self.name,
            "enable_flag": self.enable_flag,
            "category": self.category,
            "description": self.description,
            "pipe_name": self.pipe_name,
            "worst_support": self.worst_support.value,
            "support_counts": {
                level.value: count for level, count in self.support_counts().items()
            },
            "graph_delta": {
                "removed_nodes": [self._node_ref_dict(ref) for ref in self.removed_nodes],
                "added_nodes": [self._node_ref_dict(ref) for ref in self.added_nodes],
                "modified_nodes": [self._node_ref_dict(ref) for ref in self.modified_nodes],
                "removed_initializers": self.removed_initializers,
                "added_initializers": self.added_initializers,
                "modified_initializers": self.modified_initializers,
            },
            "operators": [operator.to_dict() for operator in self.operators],
        }
        if self.error is not None:
            data["error"] = self.error
        return data


def _produced_node_refs(
    finding: CapabilityFinding,
) -> list[tuple[str, NodeRef]]:
    """Return ``(change_kind, NodeRef)`` pairs for a finding's produced nodes.

    Produced nodes are those an optimization introduces: newly added nodes and
    nodes modified in place. Removed nodes are excluded — they no longer exist
    after the optimization, so their target support is irrelevant.
    """
    pairs: list[tuple[str, NodeRef]] = []
    pairs.extend(("added", ref) for ref in finding.added_nodes)
    pairs.extend(("modified", ref) for ref in finding.modified_nodes)
    return pairs


def _check_one(
    finding: CapabilityFinding,
    produced_model: ModelProto,
    ep: EPName,
    device: str,
    model_path: str,
) -> OptimizationOutputSupport:
    """Check target support for the operators one optimization would produce."""
    from .core.runtime_checker import RuntimeChecker

    result = OptimizationOutputSupport(
        name=finding.name,
        enable_flag=finding.enable_flag,
        category=finding.category,
        description=finding.description,
        pipe_name=finding.pipe_name,
        removed_nodes=list(finding.removed_nodes),
        added_nodes=list(finding.added_nodes),
        modified_nodes=list(finding.modified_nodes),
        removed_initializers=list(finding.removed_initializers),
        added_initializers=list(finding.added_initializers),
        modified_initializers=list(finding.modified_initializers),
    )

    produced = _produced_node_refs(finding)
    if not produced:
        return result

    # Output tensor names uniquely identify produced nodes within the graph.
    produced_outputs = {out for _, ref in produced for out in ref.outputs}

    try:
        onnx_model = ONNXModel.from_onnx_model(produced_model, model_path)
        checker = RuntimeChecker(ep=ep, device=device, model=onnx_model)
        # Only evaluate the produced nodes; skip the rest of the graph.
        runtimes = checker.op_support(
            node_output_filter=produced_outputs,
            on_node_result=lambda _r: None,
        )
    except Exception as exc:  # pragma: no cover - defensive, keeps analyze alive
        logger.warning(
            "Could not check produced-operator support for '%s' on %s/%s: %s",
            finding.name,
            ep,
            device,
            exc,
        )
        result.error = str(exc)
        return result

    # Map each produced output tensor to its support result.
    support_by_output: dict[str, tuple[SupportLevel, str | None]] = {}
    for runtime in runtimes:
        node = _matched_node(runtime)
        outputs = tuple(node.output) if node is not None else ()
        classification = runtime.result.classification
        reason = runtime.result.reason
        for out in outputs:
            support_by_output[out] = (classification, reason)

    for change, ref in produced:
        support, reason = _lookup_support(ref, support_by_output)
        result.operators.append(
            ProducedOperatorSupport(
                op_type=ref.qualified_op_type(),
                label=ref.label(),
                change=change,
                support=support,
                reason=reason,
            )
        )

    return result


def _matched_node(runtime: object) -> NodeProto | None:
    """Best-effort extraction of the ONNX node behind a ``PatternRuntime``."""
    pattern_match = getattr(runtime, "pattern_match", None)
    skeleton = getattr(pattern_match, "skeleton_match_result", None)
    matched_nodes = getattr(skeleton, "matched_nodes", None)
    if matched_nodes:
        return cast("NodeProto", matched_nodes[0])
    logger.debug(
        "Could not extract matched node from runtime (type=%s); produced operators "
        "correlated via this runtime will report UNKNOWN support",
        type(runtime).__name__,
    )
    return None


def _lookup_support(
    ref: NodeRef,
    support_by_output: dict[str, tuple[SupportLevel, str | None]],
) -> tuple[SupportLevel, str | None]:
    """Resolve a produced node's support via any of its output tensors."""
    for out in ref.outputs:
        if out in support_by_output:
            return support_by_output[out]
    return SupportLevel.UNKNOWN, "not evaluated"


def check_optimization_output_support(
    optimization_outputs: Iterable[tuple[CapabilityFinding, ModelProto]],
    ep: EPName,
    device: str,
    model_path: str,
) -> list[OptimizationOutputSupport]:
    """Check produced-operator support for applicable optimizations on a target.

    Args:
        optimization_outputs: ``(finding, produced_model)`` pairs from
            :func:`winml.modelkit.optim.iter_optimization_outputs`. Materialize
            the iterator once and reuse it across targets — the dry-run that
            produces these pairs is target-independent, so only the support
            lookup needs to run per EP/device.
        ep: Target execution provider (canonical name).
        device: Target device string (e.g. ``"NPU"``).
        model_path: Path to the source model, used for external-data resolution.

    Returns:
        One :class:`OptimizationOutputSupport` per applicable optimization, in
        the same order as ``optimization_outputs``.
    """
    return [
        _check_one(finding, produced_model, ep, device, model_path)
        for finding, produced_model in optimization_outputs
    ]
