# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for PatternMatcher robustness against invalid/incomplete models."""

from __future__ import annotations

import numpy as np
import pytest
from onnx import ModelProto, TensorProto, helper, load, numpy_helper, save
from onnx.defs import get_schema

from winml.modelkit.onnx import ONNXDomain
from winml.modelkit.pattern import (
    Pattern,
    PatternMatcher,
    PatternSchema,
    Skeleton,
    SkeletonMatchResult,
    make_single_op_pattern,
)


def _make_simple_model() -> ModelProto:
    """Create a minimal valid ONNX model."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", ["X"], ["Y"], name="id0")
    graph = helper.make_graph([node], "test", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_two_identity_model() -> ModelProto:
    """Create a two-node chain with one internal edge."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    id0 = helper.make_node("Identity", ["X"], ["mid"], name="id0")
    id1 = helper.make_node("Identity", ["mid"], ["Y"], name="id1")
    graph = helper.make_graph([id0, id1], "two_identity", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class _TwoIdentityPattern(Pattern):
    """Minimal two-node pattern used to verify matcher invariants."""

    def get_skeleton(self) -> Skeleton:
        return Skeleton(
            node_op_types=["Identity", "Identity"],
            node_domains=[ONNXDomain.AI_ONNX, ONNXDomain.AI_ONNX],
            edges=[(-1, 0, 0, 0), (0, 0, 1, 0)],
            exit_nodes=[1],
            n_inputs=1,
        )

    def get_schema(self) -> PatternSchema:
        return PatternSchema(name="TwoIdentityPattern", doc="", inputs=[], outputs=[])

    def get_internal_constants_and_attributes(
        self, inputs, attributes, is_constant_map, domain_versions
    ):
        return [], {}


class TestPatternMatcherOnnxValidationFailure:
    """PatternMatcher should not abort when onnx.checker fails."""

    def test_invalid_model_does_not_raise(self):
        """A model that fails onnx.checker should still be matchable.

        Before the fix, this raised InvalidPatternMatcherModelError.
        """
        # Build a model with an intentionally invalid node (unknown op in
        # default domain, which onnx.checker rejects)
        x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
        node = helper.make_node("NotARealOp", ["X"], ["Y"], name="bad_node")
        graph = helper.make_graph([node], "bad_graph", [x], [y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        # Should NOT raise — validation failure is logged, not raised
        matcher = PatternMatcher(model, raise_on_invalid_model=True)
        assert "bad_node" in matcher.node_lookup


class TestPatternMatcherExternalData:
    """PatternMatcher should handle models with missing external data."""

    def test_missing_external_data_does_not_raise(self, tmp_path):
        """Initializer referencing a non-existent external file should not crash.

        Before the fix, numpy_helper.to_array raised because the external
        data file was missing.
        """
        # Create a model with an initializer that claims external data
        x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
        node = helper.make_node("Add", ["X", "W"], ["Y"], name="add0")
        graph = helper.make_graph([node], "ext_data_graph", [x], [y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        # Add an initializer with real data first, then save with external data.
        # Use a large-enough tensor so ONNX actually externalizes it
        # (small tensors may be kept inline).
        w_array = np.ones([256, 256], dtype=np.float32)
        w_tensor = numpy_helper.from_array(w_array, name="W")
        model.graph.initializer.append(w_tensor)

        model_path = tmp_path / "model.onnx"
        save(
            model,
            str(model_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.onnx.data",
        )

        # Reload without external data (simulates how analyzer loads)
        model_no_ext = load(str(model_path), load_external_data=False)

        # Delete the external data file to simulate it being inaccessible
        (tmp_path / "model.onnx.data").unlink()

        # Should NOT raise — missing external data is skipped gracefully
        matcher = PatternMatcher(model_no_ext, raise_on_invalid_model=True)
        assert "add0" in matcher.node_lookup
        # The tensor value should not be populated (data is unavailable)
        assert "W" not in matcher.tensor_values


class TestPatternMatcherUnnamedNodeKeys:
    """PatternMatcher should produce stable keys for unnamed nodes."""

    def test_unnamed_node_match_has_stable_key(self):
        """Unnamed ONNX nodes should map to deterministic internal keys."""
        x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
        unnamed_node = helper.make_node("Identity", ["X"], ["Y"])
        graph = helper.make_graph([unnamed_node], "unnamed_node_graph", [x], [y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        matcher = PatternMatcher(model, raise_on_invalid_model=True)

        _, identity_pattern_cls = make_single_op_pattern(get_schema("Identity", 17))
        matcher.register_pattern(identity_pattern_cls())

        results = matcher.match_skeleton()
        assert len(results) == 1
        assert results[0].matched_node_keys == ["node_0"]

    def test_skeleton_result_requires_aligned_matched_node_keys(self):
        """SkeletonMatchResult must be created with aligned stable node keys."""
        node = helper.make_node("Identity", ["X"], ["Y"], name="id0")

        class _MockPattern:
            pass

        with pytest.raises(ValueError, match="matched_node_keys"):
            SkeletonMatchResult(
                pattern=_MockPattern(),  # type: ignore[arg-type]
                matched_nodes=[node],
                matched_node_keys=[],
                matcher=None,  # type: ignore[arg-type]
            )


class TestPatternMatcherLookupInvariants:
    """PatternMatcher should fail loudly when internal edge registration is broken."""

    def test_missing_registered_edge_asserts(self):
        """Missing edge_info entries for non-virtual inputs should raise immediately."""
        matcher = PatternMatcher(_make_two_identity_model())
        matcher.register_pattern(_TwoIdentityPattern())

        matcher.edge_info_by_name["mid"].pop("id1")

        with pytest.raises(AssertionError, match="Missing edge registration"):
            matcher.match_skeleton()
