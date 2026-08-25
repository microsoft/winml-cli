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
    constants. When one capability is intentionally owned by multiple pipes, it
    is probed once across the full pipeline so the report matches the single
    public ``--enable-*`` flag.

No operator names, tensor names, or architectures are hardcoded — every result
is derived from the concrete graph diff.
"""

from __future__ import annotations

import logging
from array import array
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from google.protobuf.internal import api_implementation
from onnx import AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from .registry import CapabilityDef


logger = logging.getLogger(__name__)
_PROTOBUF_IMPLEMENTATION = api_implementation.Type()


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
        domain: The node's operator domain. Empty means the default ONNX domain.
    """

    op_type: str
    name: str
    outputs: tuple[str, ...]
    domain: str = ""

    def qualified_op_type(self) -> str:
        """Return the operator type, qualified when it uses a custom domain."""
        if self.domain and self.domain != "ai.onnx":
            return f"{self.domain}::{self.op_type}"
        return self.op_type

    def label(self) -> str:
        """Return a human-readable identifier for this node.

        Falls back to the first output tensor name when the node is unnamed,
        since output names are unique within a graph.
        """
        ident = self.name or (self.outputs[0] if self.outputs else "?")
        return f"{self.qualified_op_type()} '{ident}'"


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
        return Counter(n.qualified_op_type() for n in nodes).most_common()


# =============================================================================
# GRAPH DIFF HELPERS
# =============================================================================


def _node_identity(node: NodeProto) -> tuple[Any, ...]:
    """Return a key that identifies a node stably across transformations.

    Output tensor names are unique within a valid ONNX graph, so they make a
    node identifiable even when its inputs are rewired or its attributes change
    (that manifests as a "modified" node rather than a remove+add pair).

    Conversely, an optimization that *renames* a node's outputs (e.g. a MatMul
    output ``mm`` -> ``mm/MatMulAddFusion``) changes this key, so the same
    logical node is reported as a remove + add pair rather than a single
    modification. This is an intentional semantics choice: the diff describes
    the concrete graph delta (which tensors/nodes appear and disappear), not a
    logical node-to-node correspondence, so output renames can inflate the
    removed/added counts even though the net effect is a single rewritten node.

    Nodes without outputs (rare) fall back to a structural signature.
    """
    if len(node.output) > 0:
        return tuple(node.output)
    return ("\0no-output", node.domain, node.op_type, node.name, tuple(node.input))


def _collect_nodes(
    graph: GraphProto,
    scope: tuple[Any, ...],
    table: dict[tuple[Any, ...], tuple[bytes, NodeRef]],
) -> None:
    """Populate ``table`` with ``{key: (serialized, NodeRef)}`` for every node.

    Walks subgraphs (If/Loop/Scan bodies) so nested rewrites are not missed.
    Keys are scoped by the containing node to keep subgraph nodes distinct from
    top-level nodes. Traversal is iterative (explicit stack) rather than
    recursive so that pathologically deep subgraph nesting cannot exhaust
    Python's recursion limit.
    """
    stack: list[tuple[GraphProto, tuple[Any, ...]]] = [(graph, scope)]
    while stack:
        cur_graph, cur_scope = stack.pop()
        for node in cur_graph.node:
            key = (cur_scope, _node_identity(node))
            table[key] = (
                node.SerializeToString(),
                NodeRef(node.op_type, node.name, tuple(node.output), node.domain),
            )
            for attr in node.attribute:
                if attr.type == AttributeProto.GRAPH:
                    stack.append((attr.g, (*cur_scope, _node_identity(node), attr.name)))
                elif attr.type == AttributeProto.GRAPHS:
                    for i, sub in enumerate(attr.graphs):
                        stack.append((sub, (*cur_scope, _node_identity(node), attr.name, i)))


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


def _initializers_equal(base: TensorProto, probe: TensorProto) -> bool:
    """Compare initializer content while ignoring storage-location metadata.

    An external-data ``TensorProto`` keeps its file path/offset/length in
    ``external_data`` (with ``data_location = EXTERNAL``) instead of inline
    ``raw_data``. Those location fields change whenever a pipe re-saves the
    model — relocated offsets, a different sidecar path — even when the tensor
    itself is unchanged, which would otherwise surface as a spurious "modified"
    initializer. Some ONNX processing also explicitly serializes
    ``data_location = DEFAULT`` on inline tensors, which is semantically
    identical to leaving the optional field unset. Stripping both location
    fields keeps the diff keyed on tensor identity/content rather than storage
    metadata.

    The common equality fast path is important for large models: protobuf
    comparison reads payloads in place, while serializing every initializer
    would allocate and copy all model weights for every capability probe.
    """
    protobuf_equal = base == probe
    if protobuf_equal and _PROTOBUF_IMPLEMENTATION == "upb":
        return True

    float_field_types = {"float_data": "f", "double_data": "d"}
    for field_name, typecode in float_field_types.items():
        if (
            array(typecode, getattr(base, field_name)).tobytes()
            != array(typecode, getattr(probe, field_name)).tobytes()
        ):
            return False

    if protobuf_equal:
        return True

    ignored_fields = {"data_location", "external_data", *float_field_types}
    for descriptor in TensorProto.DESCRIPTOR.fields:
        if descriptor.name in ignored_fields:
            continue

        base_value = getattr(base, descriptor.name)
        probe_value = getattr(probe, descriptor.name)
        if descriptor.message_type is not None:
            if descriptor.has_presence and (
                base.HasField(descriptor.name) != probe.HasField(descriptor.name)
            ):
                return False
            if base_value != probe_value:
                return False
        elif base_value != probe_value:
            return False

    return True


def _collect_initializers(model: ModelProto) -> dict[str, TensorProto]:
    """Return initializer references keyed by name for the top-level graph."""
    return {init.name: init for init in model.graph.initializer}


def _diff_initializers(
    base: dict[str, TensorProto],
    probe: dict[str, TensorProto],
) -> tuple[list[str], list[str], list[str]]:
    """Diff two initializer tables into (removed, added, modified) name lists."""
    base_names = set(base)
    probe_names = set(probe)
    removed = sorted(base_names - probe_names)
    added = sorted(probe_names - base_names)
    modified = sorted(
        n for n in (base_names & probe_names) if not _initializers_equal(base[n], probe[n])
    )
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

    Because every pipe run operates on a fresh clone, the shared model threaded
    through the pipeline is never mutated in place, so returning the input
    object unchanged when the pipe opts out cannot alias into a later probe.
    """
    should_process = getattr(pipe, "should_process", None)
    if callable(should_process) and not should_process(config):
        return model
    result: ModelProto = pipe.process(_clone(model), config)
    return result


def _run_pipeline(
    pipe_classes: list[type[Any]],
    model: ModelProto,
    kwargs: dict[str, Any],
) -> ModelProto:
    """Run the capability-driven pipe sequence from ``model`` with ``kwargs``."""
    current = model
    for pipe_class in pipe_classes:
        pipe = pipe_class()
        current = _run_pipe(pipe, current, pipe.build_config(**kwargs))
    return current


def _iter_findings(
    model: ModelProto,
    capabilities: dict[str, CapabilityDef],
    *,
    on_probe_start: Callable[[str], None] | None = None,
    on_probe_complete: Callable[[str], None] | None = None,
    **optimizer_kwargs: Any,
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

    pipe_probe_counts = Counter(
        name
        for pipe_class in PIPES
        for name, cap in pipe_class.capabilities.items()
        if isinstance(cap, BoolCapability) and not cap.default
    )
    remaining_probes = Counter(dict.fromkeys(pipe_probe_counts, 1))
    shared_cap_names = {
        name for name, count in pipe_probe_counts.items() if count > 1 and name in capabilities
    }
    shared_cap_owners = {
        name: [
            pipe_class.name
            for pipe_class in PIPES
            if name in pipe_class.capabilities
            and isinstance(pipe_class.capabilities[name], BoolCapability)
            and not pipe_class.capabilities[name].default
        ]
        for name in shared_cap_names
    }

    def complete_probe(cap_name: str) -> None:
        if remaining_probes[cap_name] <= 0:
            return
        remaining_probes[cap_name] -= 1
        if remaining_probes[cap_name] == 0 and on_probe_complete is not None:
            on_probe_complete(cap_name)

    # Baseline kwargs = every capability at its default value.
    default_kwargs = {cap.python_name: cap.default for cap in capabilities.values()}
    default_kwargs.update(optimizer_kwargs)
    kebab_defaults = {name: cap.default for name, cap in capabilities.items()}

    # Mandatory pre-stage — mirrors Optimizer.optimize(). The clone is required:
    # infer_shapes may mutate large models in place (for models over the protobuf
    # limit it round-trips through save_onnx with external data), and this
    # function must never modify the caller's input model.
    current = infer_shapes(_clone(model))
    pipeline_input = current
    full_base_out: ModelProto | None = None

    def full_pipeline_baseline() -> ModelProto:
        nonlocal full_base_out
        if full_base_out is None:
            full_base_out = _run_pipeline(PIPES, pipeline_input, default_kwargs)
        return full_base_out

    for pipe_class in PIPES:
        pipe = pipe_class()
        probe_caps = [
            (name, cap)
            for name, cap in pipe.capabilities.items()
            if isinstance(cap, BoolCapability) and not cap.default
        ]

        base_config = pipe.build_config(**default_kwargs)
        try:
            base_out = _run_pipe(pipe, current, base_config)
        except Exception as exc:
            # A pipe failing on its own default config would otherwise abort the
            # whole scan. Skip this pipe's probes and leave ``current`` untouched
            # so downstream pipes still see the last good model.
            logger.warning(
                "Skipping pipe '%s' — baseline run failed on default config: %s",
                getattr(pipe, "name", pipe_class.__name__),
                exc,
            )
            for cap_name, _ in probe_caps:
                complete_probe(cap_name)
            continue
        base_nodes: dict[tuple[Any, ...], tuple[bytes, NodeRef]] = {}
        _collect_nodes(base_out.graph, (), base_nodes)
        base_inits = _collect_initializers(base_out)
        prepared_probe_model: ModelProto
        analysis_prepared = False
        try:
            try:
                prepared_probe_model = pipe.prepare_analysis_model(current)
                analysis_prepared = True
            except Exception as exc:
                logger.warning(
                    "Could not prepare accelerated analysis for pipe '%s': %s. "
                    "Falling back to isolated capability probes.",
                    pipe.name,
                    exc,
                )

            for cap_name, cap in probe_caps:
                if cap_name in shared_cap_names:
                    owners = shared_cap_owners[cap_name]
                    if owners and owners[0] == pipe.name:
                        if on_probe_start is not None:
                            on_probe_start(cap_name)
                        try:
                            ep_device = optimizer_kwargs.get("ep_device")
                            if ep_device is not None and cap.ep_constraint is not None:
                                from ..utils.constants import normalize_ep_name

                                target_ep = normalize_ep_name(ep_device.device.ep_name)
                                if not any(
                                    normalize_ep_name(name) == target_ep
                                    for name in cap.ep_constraint
                                ):
                                    logger.debug(
                                        "Skipping capability '%s': target EP %s is not in %s",
                                        cap.name,
                                        target_ep,
                                        cap.ep_constraint,
                                    )
                                    continue

                            kebab = dict(kebab_defaults)
                            kebab[cap_name] = True
                            kebab = auto_enable_dependencies(kebab, capabilities)
                            probe_kwargs = {
                                capabilities[name].python_name: value
                                for name, value in kebab.items()
                                if name in capabilities
                            }
                            probe_kwargs.update(optimizer_kwargs)

                            try:
                                shared_base_out = full_pipeline_baseline()
                                probe_out = _run_pipeline(PIPES, pipeline_input, probe_kwargs)
                            except Exception as exc:
                                logger.warning(
                                    "Could not evaluate shared capability '%s': %s",
                                    cap_name,
                                    exc,
                                )
                                continue

                            shared_base_nodes: dict[tuple[Any, ...], tuple[bytes, NodeRef]] = {}
                            shared_probe_nodes: dict[tuple[Any, ...], tuple[bytes, NodeRef]] = {}
                            _collect_nodes(shared_base_out.graph, (), shared_base_nodes)
                            _collect_nodes(probe_out.graph, (), shared_probe_nodes)
                            removed, added, modified = _diff_nodes(
                                shared_base_nodes,
                                shared_probe_nodes,
                            )

                            shared_base_inits = _collect_initializers(shared_base_out)
                            probe_inits = _collect_initializers(probe_out)
                            rem_init, add_init, mod_init = _diff_initializers(
                                shared_base_inits,
                                probe_inits,
                            )

                            finding = CapabilityFinding(
                                name=cap.name,
                                python_name=cap.python_name,
                                enable_flag=f"--enable-{cap.name}",
                                category=cap.category.value,
                                description=cap.description,
                                pipe_name="+".join(owners),
                                removed_nodes=removed,
                                added_nodes=added,
                                modified_nodes=modified,
                                removed_initializers=rem_init,
                                added_initializers=add_init,
                                modified_initializers=mod_init,
                            )

                            if finding.applicable:
                                yield finding, probe_out
                        finally:
                            complete_probe(cap_name)
                    continue

                if on_probe_start is not None:
                    on_probe_start(cap_name)
                try:
                    ep_device = optimizer_kwargs.get("ep_device")
                    if ep_device is not None and cap.ep_constraint is not None:
                        from ..utils.constants import normalize_ep_name

                        target_ep = normalize_ep_name(ep_device.device.ep_name)
                        if not any(
                            normalize_ep_name(name) == target_ep for name in cap.ep_constraint
                        ):
                            logger.debug(
                                "Skipping capability '%s': target EP %s is not in %s",
                                cap.name,
                                target_ep,
                                cap.ep_constraint,
                            )
                            continue

                    # Enable only this capability (plus its dependencies) on top of
                    # the all-defaults configuration.
                    kebab = dict(kebab_defaults)
                    kebab[cap_name] = True
                    kebab = auto_enable_dependencies(kebab, capabilities)
                    probe_kwargs = {
                        capabilities[name].python_name: value
                        for name, value in kebab.items()
                        if name in capabilities
                    }
                    probe_kwargs.update(optimizer_kwargs)

                    probe_config = pipe.build_config(**probe_kwargs)
                    should_process = getattr(pipe, "should_process", None)
                    if callable(should_process) and not should_process(probe_config):
                        # Pipe would not run for this capability — nothing to apply.
                        continue

                    try:
                        if analysis_prepared:
                            probe_input = (
                                _clone(prepared_probe_model)
                                if pipe.requires_analysis_clone()
                                else prepared_probe_model
                            )
                            probe_out = pipe.process_analysis(probe_input, probe_config)
                        else:
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
                finally:
                    complete_probe(cap_name)
        finally:
            try:
                pipe.finish_analysis()
            except Exception as exc:
                logger.warning(
                    "Could not finish analysis cleanup for pipe '%s': %s",
                    pipe.name,
                    exc,
                )

        # Advance the pipeline exactly as the real optimizer would.
        current = base_out


def analyze_model(
    model: ModelProto,
    capabilities: dict[str, CapabilityDef],
    *,
    on_probe_start: Callable[[str], None] | None = None,
    on_probe_complete: Callable[[str], None] | None = None,
    **optimizer_kwargs: Any,
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
        **optimizer_kwargs: Pipeline context forwarded to every pipe configuration,
            such as a resolved ``ep_device``.
        on_probe_start: Optional callback invoked with the capability name
            before each probe begins.
        on_probe_complete: Optional callback invoked with the capability name
            after all of its probes finish or are skipped.

    Returns:
        Applicable findings in pipeline order, each naming the affected nodes
        and constants.
    """
    return [
        finding
        for finding, _ in _iter_findings(
            model,
            capabilities,
            on_probe_start=on_probe_start,
            on_probe_complete=on_probe_complete,
            **optimizer_kwargs,
        )
    ]


def iter_optimization_outputs(
    model: ModelProto,
    capabilities: dict[str, CapabilityDef],
    **optimizer_kwargs: Any,
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
        **optimizer_kwargs: Pipeline context forwarded to every pipe configuration.

    Yields:
        ``(finding, produced_model)`` pairs in pipeline order, one per applicable
        optimization. The pairs are produced lazily; materialize the iterator if
        the produced models must outlive iteration.
    """
    yield from _iter_findings(model, capabilities, **optimizer_kwargs)
