# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Universal tests auto-parametrized across all registered patterns.

Each test runs for every pattern registered in the PatternInputGenerator registry.
This ensures that adding a new pattern automatically gains baseline test coverage
for ONNX generation, self-matching, removability, schema, and match structure.
"""

import onnx
import pytest

from winml.modelkit.pattern.base import (
    PatternMatcher,
    get_registered_pattern_input_generators,
)

from .conftest import (
    SKIP_VALIDATION_PATTERNS,
    add_first_op_output_to_graph_output,
    create_pattern_generator,
    generate_self_matching_model,
)


_ALL_PATTERN_NAMES = sorted(get_registered_pattern_input_generators())


def _skip_if_unsupported(pattern_name: str) -> None:
    if pattern_name in SKIP_VALIDATION_PATTERNS:
        pytest.skip(f"Skipping {pattern_name} (known runtime limitation)")


@pytest.mark.parametrize("pattern_name", _ALL_PATTERN_NAMES)
class TestUniversalPatternONNXGeneration:
    """Verify every pattern produces a valid, self-matching ONNX model."""

    def test_generates_valid_onnx_model(self, pattern_name: str) -> None:
        """The generated ONNX model passes onnx.checker validation."""
        _skip_if_unsupported(pattern_name)
        model, _pattern, _result = generate_self_matching_model(pattern_name)
        onnx.checker.check_model(model)

    def test_self_match_returns_one_result(self, pattern_name: str) -> None:
        """Pattern matches its own generated model exactly once."""
        _skip_if_unsupported(pattern_name)
        _model, _pattern, result = generate_self_matching_model(pattern_name)
        assert result is not None, f"Pattern {pattern_name} did not self-match"

    def test_self_match_is_removable(self, pattern_name: str) -> None:
        """Self-matched pattern is removable (no intermediate outputs exposed)."""
        _skip_if_unsupported(pattern_name)
        _model, _pattern, result = generate_self_matching_model(pattern_name)
        assert result.skeleton_match_result.removable is True

    def test_self_match_not_removable_with_intermediate_output(
        self, pattern_name: str
    ) -> None:
        """Pattern becomes non-removable when an intermediate output is a graph output."""
        _skip_if_unsupported(pattern_name)
        model, pattern, _result = generate_self_matching_model(pattern_name)

        modified_model = add_first_op_output_to_graph_output(model)

        matcher = PatternMatcher(modified_model)
        matcher.register_pattern(pattern)
        results = matcher.match()

        assert len(results) == 1, f"Expected 1 match, got {len(results)}"
        assert results[0].skeleton_match_result.removable is False, (
            "Expected removable=False when intermediate output is a graph output"
        )

    def test_node_count_matches_skeleton(self, pattern_name: str) -> None:
        """Generated model node count equals the skeleton's node_op_types length."""
        _skip_if_unsupported(pattern_name)
        model, pattern, _result = generate_self_matching_model(pattern_name)
        skeleton = pattern.get_skeleton()
        assert len(model.graph.node) == len(skeleton.node_op_types)

    def test_op_types_match_skeleton(self, pattern_name: str) -> None:
        """Generated model op types match the skeleton's node_op_types."""
        _skip_if_unsupported(pattern_name)
        model, pattern, _result = generate_self_matching_model(pattern_name)
        skeleton = pattern.get_skeleton()
        actual_op_types = [node.op_type for node in model.graph.node]
        assert actual_op_types == skeleton.node_op_types


@pytest.mark.parametrize("pattern_name", _ALL_PATTERN_NAMES)
class TestUniversalPatternSchema:
    """Verify every pattern has a well-formed schema."""

    def test_schema_has_name_inputs_outputs(self, pattern_name: str) -> None:
        """Schema has a non-empty name, at least one input, output, and type constraint."""
        gen = create_pattern_generator(pattern_name)
        schema = gen.pattern.get_schema()

        assert schema.name, "Schema name must be non-empty"
        assert len(schema.inputs) >= 1, "Schema must have at least 1 input"
        assert len(schema.outputs) >= 1, "Schema must have at least 1 output"
        assert len(schema.type_constraints) >= 1, "Schema must have at least 1 type constraint"


@pytest.mark.parametrize("pattern_name", _ALL_PATTERN_NAMES)
class TestUniversalPatternMatchStructure:
    """Verify the structure of match results for every pattern."""

    def test_match_has_valid_structure(self, pattern_name: str) -> None:
        """Match result has populated schema mappings, type params, and input infos."""
        _skip_if_unsupported(pattern_name)
        _model, pattern, result = generate_self_matching_model(pattern_name)
        schema = pattern.get_schema()

        # Schema input mappings present
        for inp in schema.inputs:
            assert inp.name in result.schema_input_to_value, (
                f"Missing input mapping for '{inp.name}'"
            )

        # Schema output mappings present
        for out in schema.outputs:
            assert out.name in result.schema_output_to_value, (
                f"Missing output mapping for '{out.name}'"
            )

        # Type parameter mappings present
        for tc in schema.type_constraints:
            assert tc.type_param_str in result.type_param_to_type, (
                f"Missing type param mapping for '{tc.type_param_str}'"
            )

        # Input infos present for each schema input
        for inp in schema.inputs:
            assert inp.name in result.input_infos, (
                f"Missing input info for '{inp.name}'"
            )
            info = result.input_infos[inp.name]
            assert isinstance(info.is_constant, bool)
