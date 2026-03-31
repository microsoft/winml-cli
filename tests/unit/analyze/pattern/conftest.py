# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Shared test infrastructure for pattern tests.

Provides constants, helpers, and fixtures used across all pattern test files.
The core function `generate_self_matching_model` leverages the PatternInputGenerator
registry to create ONNX models that self-match any registered pattern, eliminating
manual input construction boilerplate.
"""

from __future__ import annotations

import functools
from typing import Any

import onnx
from onnx import helper

from winml.modelkit.onnx import ONNXDomain
from winml.modelkit.pattern.base import (
    InvalidPatternMatcherModelError,
    PatternMatcher,
    get_pattern_input_generator,
)


# ---------------------------------------------------------------------------
# Shared constants (previously duplicated across 4+ test files)
# ---------------------------------------------------------------------------

TEST_DOMAIN_VERSIONS: dict[ONNXDomain, int] = {ONNXDomain.AI_ONNX: 17}

PATTERNS_REQUIRING_NEWER_OPSET: dict[str, dict[ONNXDomain, int]] = {
    "TransposeAttention": {ONNXDomain.AI_ONNX: 23},
    "TransposedSingleRMSNormalization": {ONNXDomain.AI_ONNX: 23},
}

# Patterns to skip for validation/self-matching (runtime limitations)
#
# ReshapeTransposeReshapeLowDim is a one-directional transform: it consumes
# an UNMERGED >=6D Reshape-Transpose-Reshape and emits a MERGED <=5D variant.
# The generated (merged) model is intentionally NOT re-matchable by this pattern,
# so the standard "generate model → self-match" assumption does not hold.
# See TestReshapeTransposeReshapeLowDimPattern.test_already_merged_4d_rtr_is_not_matched
# for targeted coverage of the no-self-match behaviour.
SKIP_VALIDATION_PATTERNS: set[str] = {
    "TransposeAttention",
    "ReshapeTransposeReshapeLowDim",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_domain_versions(pattern_name: str) -> dict[ONNXDomain, int]:
    """Return the correct domain versions for a given pattern name."""
    return PATTERNS_REQUIRING_NEWER_OPSET.get(pattern_name, TEST_DOMAIN_VERSIONS)


def create_pattern_generator(pattern_name: str) -> Any:
    """Instantiate a PatternInputGenerator from the registry.

    Args:
        pattern_name: Registered name of the pattern (e.g. "Gelu1").

    Returns:
        An initialized PatternInputGenerator instance.
    """
    gen_class = get_pattern_input_generator(pattern_name)
    domain_versions = get_domain_versions(pattern_name)
    return gen_class(domain_versions=domain_versions)


@functools.cache
def generate_self_matching_model(
    pattern_name: str,
    *,
    max_iter: int = 50,
) -> tuple[onnx.ModelProto, Any, Any]:
    """Generate an ONNX model that self-matches the given pattern.

    Uses the PatternInputGenerator registry to produce valid inputs, then
    iterates over input/type/constant-map combinations until a self-match
    succeeds.  Results are cached per *pattern_name* so that multiple tests
    reuse the same generated model without redundant work.

    Args:
        pattern_name: Registered name of the pattern.
        max_iter: Maximum number of ``iter()`` yielded combinations to try.

    Returns:
        Tuple of (model, pattern_instance, match_result).

    Raises:
        ValueError: If no input/constant combination produces a self-match.
    """
    gen = create_pattern_generator(pattern_name)
    pattern = gen.pattern

    for i, (kwargs, tags) in enumerate(gen.iter()):
        if i >= max_iter:
            break

        output_dtypes = gen.infer_output_types(kwargs, tags)
        if not output_dtypes:
            continue

        for is_constant_map in gen._iter_constant_combinations(kwargs):
            try:
                model = gen._create_model(kwargs, is_constant_map, output_dtypes)
                matcher = PatternMatcher(model)
                matcher.register_pattern(pattern)
                results = matcher.match()
            except (
                onnx.checker.ValidationError,
                InvalidPatternMatcherModelError,
                ValueError,
                AssertionError,
            ):
                continue

            if len(results) == 1:
                return model, pattern, results[0]

    raise ValueError(
        f"Could not create a self-matching model for pattern '{pattern_name}'. "
        "This likely indicates a bug in the pattern or its input generator."
    )


def add_first_op_output_to_graph_output(
    model: onnx.ModelProto,
) -> onnx.ModelProto:
    """Add the first op's output as an additional graph output.

    This makes patterns non-removable because an intermediate tensor is
    consumed outside the matched subgraph.

    Args:
        model: The ONNX model to modify.

    Returns:
        A new ONNX model with the first op's output added as a graph output.
    """
    first_node = model.graph.node[0]
    first_output_name = first_node.output[0]

    output_type = None
    for vi in model.graph.value_info:
        if vi.name == first_output_name:
            output_type = vi.type
            break

    assert output_type is not None, "Expected to find type info for first op output"

    new_output = helper.make_tensor_value_info(
        first_output_name,
        output_type.tensor_type.elem_type,
        None,
    )

    new_outputs = [*list(model.graph.output), new_output]
    new_graph = helper.make_graph(
        nodes=list(model.graph.node),
        name=model.graph.name,
        inputs=list(model.graph.input),
        outputs=new_outputs,
        initializer=list(model.graph.initializer),
        value_info=list(model.graph.value_info),
    )

    return helper.make_model(
        new_graph,
        opset_imports=list(model.opset_import),
    )
