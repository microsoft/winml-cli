# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Applicability analysis for the capability-driven optimizer (dry-run).

This module answers the question "which optimizations *could* be applied to
this model, and on which nodes?" without producing an optimized output.

Approach (universal, architecture-agnostic — CARDINAL RULE #1):
    The pipeline is walked pipe-by-pipe exactly as the real optimizer runs it.
    For each pipe a baseline output is produced with the pipe's default config.
    Every boolean capability owned by the pipe (that is off by default) is then
    probed independently: the pipe is re-run on the *same upstream model* with
    only that capability enabled (plus auto-enabled dependencies), and the
    resulting graph is diffed against the baseline. A non-empty diff means the
    capability is applicable; the diff itself names the affected nodes and
    constants.

No operator names, tensor names, or architectures are hardcoded — every result
is derived from the concrete graph diff.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from onnx import AttributeProto, GraphProto, ModelProto, NodeProto


if TYPE_CHECKING:
    from collections.abc import Iterator

    from .registry import CapabilityDef


logger = logging.getLogger(__name__)


# =============================================================================
# RESULT TYPES
# =============================================================================


@dataclass(frozen=True)
class NodeRef:
    """A lightweight reference to a graph node for reporting.

    Attributes:
        op_type: The node's operator type (e.g. ``"MatMul"``).
        name: The node's name (may be empty — ONNX names are optional).
        outputs: The node's output tensor names.
    """

    op_type: str
    name: str
    outputs: tuple[str, ...]

    def label(self) -> str:
        """Return a human-readable identifier for this node.

        Falls back to the first output tensor name when the node is unnamed,
        since output names are unique within a graph.
        """
        ident = self.name or (self.outputs[0] if self.outputs else "?")
        return f"{self.op_type} '{ident}'"


@dataclass
class CapabilityFinding:
    """The result of probing a single optimization capability.

    Attributes:
        name: Capability name (kebab-case, e.g. ``"matmul-add-fusion"``).
        python_name: snake_case identifier used in optimizer kwargs.
        enable_flag: CLI flag that turns the capability on.
        category: Capability category value (e.g. ``"matmul"``).
        description: Human-readable capability description.
        pipe_name: Name of the pipe that owns the capability.
        removed_nodes: Nodes present in the baseline but gone after the probe
            (consumed / eliminated by the optimization).
        added_nodes: Nodes introduced by the probe (e.g. a fused op).
        modified_nodes: Nodes whose definition changed in place.
        removed_initializers: Initializer (constant) names removed by the probe.
        added_initializers: Initializer names added by the probe.
        modified_initializers: Initializer names whose data changed.
    """

    name: str
    python_name: str
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

    @property
    def applicable(self) -> bool:
        """True when the capability changes the model in any observable way."""
        return bool(
            self.removed_nodes
            or self.added_nodes
            or self.modified_nodes
            or self.removed_initializers
            or self.added_initializers
            or self.modified_initializers
        )

    @property
    def affected_node_count(self) -> int:
        """Total number of nodes touched (removed + added + modified)."""
        return len(self.removed_nodes) + len(self.added_nodes) + len(self.modified_nodes)

    def op_histogram(self, kind: str) -> list[tuple[str, int]]:
        """Return an op-type frequency list for one node bucket.

        Args:
            kind: One of ``"removed"``, ``"added"`` or ``"modified"``.

        Returns:
            ``(op_type, count)`` pairs ordered from most to least frequent.
        """
        nodes = {
            "removed": self.removed_nodes,
            "added": self.added_nodes,
            "modified": self.modified_nodes,
        }[kind]
        return Counter(n.op_type for n in nodes).most_common()


# =============================================================================
# GRAPH DIFF HELPERS
# =============================================================================


def _node_identity(node: NodeProto) -> tuple[Any, ...]:
    """Return a key that identifies a node stably across transformations.

    Output tensor names are unique within a valid ONNX graph, so they make a
    node identifiable even when its inputs are rewired or its attributes change
    (that manifests as a "modified" node rather than a remove+add pair).

    Nodes without outputs (rare) fall back to a structural signature.
    """
    if len(node.output) > 0:
        return tuple(node.output)
    return ("\0no-output", node.op_type, node.name, tuple(node.input))


def _collect_nodes(
    graph: GraphProto,
    scope: tuple[Any, ...],
    table: dict[tuple[Any, ...], tuple[bytes, NodeRef]],
) -> None:
    """Populate ``table`` with ``{key: (serialized, NodeRef)}`` for every node.

    Recurses into subgraphs (If/Loop/Scan bodies) so nested rewrites are not
    missed. Keys are scoped by the containing node to keep subgraph nodes
    distinct from top-level nodes.
    """
    for node in graph.node:
        key = (scope, _node_identity(node))
        table[key] = (
            node.SerializeToString(),
            NodeRef(node.op_type, node.name, tuple(node.output)),
        )
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH:
                _collect_nodes(attr.g, (*scope, _node_identity(node), attr.name), table)
            elif attr.type == AttributeProto.GRAPHS:
                for i, sub in enumerate(attr.graphs):
                    _collect_nodes(sub, (*scope, _node_identity(node), attr.name, i), table)


def _diff_nodes(
    base: dict[tuple[Any, ...], tuple[bytes, NodeRef]],
    probe: dict[tuple[Any, ...], tuple[bytes, NodeRef]],
) -> tuple[list[NodeRef], list[NodeRef], list[NodeRef]]:
    """Diff two node tables into (removed, added, modified) node lists."""
    base_keys = set(base)
    probe_keys = set(probe)
    removed = [base[k][1] for k in base_keys - probe_keys]
    added = [probe[k][1] for k in probe_keys - base_keys]
    modified = [probe[k][1] for k in (base_keys & probe_keys) if base[k][0] != probe[k][0]]
    return removed, added, modified


def _collect_initializers(model: ModelProto) -> dict[str, bytes]:
    """Return ``{initializer_name: serialized_bytes}`` for the top-level graph."""
    return {init.name: init.SerializeToString() for init in model.graph.initializer}


def _diff_initializers(
    base: dict[str, bytes],
    probe: dict[str, bytes],
) -> tuple[list[str], list[str], list[str]]:
    """Diff two initializer tables into (removed, added, modified) name lists."""
    base_names = set(base)
    probe_names = set(probe)
    removed = sorted(base_names - probe_names)
    added = sorted(probe_names - base_names)
    modified = sorted(n for n in (base_names & probe_names) if base[n] != probe[n])
    return removed, added, modified


# =============================================================================
# ANALYSIS DRIVER
# =============================================================================


def _clone(model: ModelProto) -> ModelProto:
    """Deep-copy a model.

    ``CopyFrom`` is used rather than ``SerializeToString`` round-tripping so
    that models larger than the 2 GiB protobuf serialization limit can still be
    cloned in memory.
    """
    copy = ModelProto()
    copy.CopyFrom(model)
    return copy


def _run_pipe(pipe: Any, model: ModelProto, config: Any) -> ModelProto:
    """Run a pipe on a *clone* of ``model``, respecting ``should_process``.

    Cloning is mandatory: some pipes serialize the model via ``save_onnx`` which
    can rewrite tensors to external-data references in place. Returning the
    input unchanged when the pipe opts out keeps the baseline faithful to the
    real pipeline.
    """
    should_process = getattr(pipe, "should_process", None)
    if callable(should_process) and not should_process(config):
        return model
    result: ModelProto = pipe.process(_clone(model), config)
    return result


def _iter_findings(
    model: ModelProto,
    capabilities: dict[str, CapabilityDef],
) -> Iterator[tuple[CapabilityFinding, ModelProto]]:
    """Yield ``(finding, produced_model)`` for every applicable optimization.

    This is the shared core behind :func:`analyze_model` and
    :func:`iter_optimization_outputs`. It walks the pipeline pipe-by-pipe
    exactly as the real optimizer does, probing each default-off boolean
    capability in isolation and diffing the result against the pipe baseline.

    ``produced_model`` is the concrete ONNX model that results from enabling the
    single capability (plus auto-enabled dependencies). It contains the added
    and modified nodes named by the finding, so downstream consumers can inspect
    the produced operators directly (e.g. to check their EP/device support).

    Findings are yielded lazily in pipeline order; only applicable capabilities
    (those that actually change the graph or its constants) are emitted.
    """
    from ..onnx import infer_shapes
    from .pipes import PIPES
    from .registry import BoolCapability, auto_enable_dependencies

    # Baseline kwargs = every capability at its default value.
    default_kwargs = {cap.python_name: cap.default for cap in capabilities.values()}
    kebab_defaults = {name: cap.default for name, cap in capabilities.items()}

    # Mandatory pre-stage — mirrors Optimizer.optimize().
    current = infer_shapes(_clone(model))

    for pipe_class in PIPES:
        pipe = pipe_class()

        base_config = pipe.build_config(**default_kwargs)
        base_out = _run_pipe(pipe, current, base_config)
        base_nodes: dict[tuple[Any, ...], tuple[bytes, NodeRef]] = {}
        _collect_nodes(base_out.graph, (), base_nodes)
        base_inits = _collect_initializers(base_out)

        probe_caps = [
            (name, cap)
            for name, cap in pipe.capabilities.items()
            if isinstance(cap, BoolCapability) and not cap.default
        ]

        for cap_name, cap in probe_caps:
            # Enable only this capability (plus its dependencies) on top of the
            # all-defaults configuration.
            kebab = dict(kebab_defaults)
            kebab[cap_name] = True
            kebab = auto_enable_dependencies(kebab, capabilities)
            probe_kwargs = {
                capabilities[name].python_name: value
                for name, value in kebab.items()
                if name in capabilities
            }

            probe_config = pipe.build_config(**probe_kwargs)
            should_process = getattr(pipe, "should_process", None)
            if callable(should_process) and not should_process(probe_config):
                # Pipe would not run for this capability — nothing to apply.
                continue

            try:
                probe_out = pipe.process(_clone(current), probe_config)
            except Exception as exc:
                logger.warning(
                    "Could not evaluate capability '%s' on pipe '%s': %s",
                    cap_name,
                    pipe.name,
                    exc,
                )
                continue

            probe_nodes: dict[tuple[Any, ...], tuple[bytes, NodeRef]] = {}
            _collect_nodes(probe_out.graph, (), probe_nodes)
            removed, added, modified = _diff_nodes(base_nodes, probe_nodes)

            probe_inits = _collect_initializers(probe_out)
            rem_init, add_init, mod_init = _diff_initializers(base_inits, probe_inits)

            finding = CapabilityFinding(
                name=cap.name,
                python_name=cap.python_name,
                enable_flag=f"--enable-{cap.name}",
                category=cap.category.value,
                description=cap.description,
                pipe_name=pipe.name,
                removed_nodes=removed,
                added_nodes=added,
                modified_nodes=modified,
                removed_initializers=rem_init,
                added_initializers=add_init,
                modified_initializers=mod_init,
            )

            if finding.applicable:
                yield finding, probe_out

        # Advance the pipeline exactly as the real optimizer would.
        current = base_out


def analyze_model(
    model: ModelProto,
    capabilities: dict[str, CapabilityDef],
) -> list[CapabilityFinding]:
    """Probe every applicable optimization capability against ``model``.

    Each boolean capability that is off by default is enabled in isolation and
    its effect on the graph is measured by diffing against the pipe's baseline
    output. Integer and choice capabilities are parameters rather than on/off
    optimizations and are not probed.

    Args:
        model: The input ONNX model (never modified).
        capabilities: The full capability registry (kebab-case keyed), e.g.
            from ``optim.pipes.get_all_capabilities()``.

    Returns:
        Applicable findings in pipeline order, each naming the affected nodes
        and constants.
    """
    return [finding for finding, _ in _iter_findings(model, capabilities)]


def iter_optimization_outputs(
    model: ModelProto,
    capabilities: dict[str, CapabilityDef],
) -> Iterator[tuple[CapabilityFinding, ModelProto]]:
    """Yield each applicable optimization together with the model it produces.

    Like :func:`analyze_model`, but also exposes the concrete ONNX model that
    results from applying each optimization in isolation. The produced model
    contains the finding's added and modified nodes, enabling callers to inspect
    or further analyze the operators an optimization would introduce — for
    example, checking whether those operators are supported on a target
    execution provider and device.

    Args:
        model: The input ONNX model (never modified).
        capabilities: The full capability registry (kebab-case keyed).

    Yields:
        ``(finding, produced_model)`` pairs in pipeline order, one per applicable
        optimization. The pairs are produced lazily; materialize the iterator if
        the produced models must outlive iteration.
    """
    yield from _iter_findings(model, capabilities)
