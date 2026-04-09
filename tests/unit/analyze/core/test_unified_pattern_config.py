# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for UnifiedPatternConfig."""

from winml.modelkit.pattern.config import PatternAlternative, UnifiedPatternConfig


class TestUnifiedPatternConfig:
    """Test UnifiedPatternConfig class."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = UnifiedPatternConfig()

        skeleton_patterns = config.get_skeleton_patterns()
        htp_patterns = config.get_htp_patterns()

        # Should load all patterns from default.json
        # 8 patterns: Gelu1-4, MatMulAdd, LayerNormPow, LayerNormMul, ReshapeTransposeReshape
        assert len(skeleton_patterns) == 8, (
            f"Expected 8 skeleton patterns, got {len(skeleton_patterns)}"
        )
        assert len(htp_patterns) == 1, f"Expected 1 HTP pattern, got {len(htp_patterns)}"

        # Check pattern IDs (note: multiple Pattern classes can share same pattern_id)
        skeleton_pattern_ids = {p.pattern_id for p in skeleton_patterns}
        expected_ids = {
            "SUBGRAPH/GeluPattern",  # Shared by Gelu1-4
            "SUBGRAPH/GemmPattern",  # MatMulAdd
            "SUBGRAPH/LayerNormalizationPattern",  # Shared by Pow and Mul variants
            "SUBGRAPH/ReshapeTransposeReshapePattern",
        }
        assert skeleton_pattern_ids == expected_ids, f"Pattern IDs mismatch: {skeleton_pattern_ids}"

    def test_load_qnn_config_with_inheritance(self):
        """Test loading QNN configuration with inheritance from default."""
        config = UnifiedPatternConfig(ihv_type="QNN")

        skeleton_patterns = config.get_skeleton_patterns()
        htp_patterns = config.get_htp_patterns()

        # Should load all patterns from default + QNN overrides
        # 9 patterns: 8 from default + TransposeAttentionPattern from QNN
        assert len(skeleton_patterns) == 9, (
            f"Expected 9 skeleton patterns, got {len(skeleton_patterns)}"
        )
        # HTP patterns should be inherited from default
        assert len(htp_patterns) == 1, f"Expected 1 HTP pattern, got {len(htp_patterns)}"

    def test_get_alternatives_for_qnn_gelu(self):
        """Test getting alternatives for QNN Gelu pattern."""
        config = UnifiedPatternConfig(ihv_type="QNN")
        patterns = config.get_skeleton_patterns()

        # Find Gelu1Pattern by class name
        gelu1_pattern = next((p for p in patterns if p.__class__.__name__ == "Gelu1Pattern"), None)
        assert gelu1_pattern is not None, "Gelu1Pattern not found"

        # Get alternatives
        alternatives = config.get_alternatives(gelu1_pattern)
        assert len(alternatives) == 1, f"Expected 1 alternative, got {len(alternatives)}"

        alt = alternatives[0]
        assert isinstance(alt, PatternAlternative)
        assert alt.pattern_to_id == "SUBGRAPH/GeluPattern"
        assert alt.pattern_class == "SingleGeluPattern"
        assert alt.module == "winml.modelkit.pattern.gelu_patterns"
        assert alt.priority == 1
        assert "Gelu" in alt.reason

    def test_get_alternatives_for_default_gelu(self):
        """Test getting alternatives for default Gelu pattern."""
        config = UnifiedPatternConfig()  # Default config
        patterns = config.get_skeleton_patterns()

        # Find Gelu1Pattern by class name
        gelu1_pattern = next((p for p in patterns if p.__class__.__name__ == "Gelu1Pattern"), None)
        assert gelu1_pattern is not None, "Gelu1Pattern not found"

        # Get alternatives (default now has alternatives too)
        alternatives = config.get_alternatives(gelu1_pattern)
        assert len(alternatives) == 1, (
            f"Expected 1 alternative for default, got {len(alternatives)}"
        )

        # Verify the alternative
        alt = alternatives[0]
        assert alt.pattern_to_id == "SUBGRAPH/GeluPattern"
        assert alt.pattern_class == "SingleGeluPattern"
        assert alt.module == "winml.modelkit.pattern.gelu_patterns"

    def test_alternatives_sorted_by_priority(self):
        """Test that alternatives are sorted by priority (highest first)."""
        config = UnifiedPatternConfig(ihv_type="QNN")
        patterns = config.get_skeleton_patterns()

        # Find MatMulAddPattern by class name
        matmuladd_pattern = next(
            (p for p in patterns if p.__class__.__name__ == "MatMulAddPattern"), None
        )
        assert matmuladd_pattern is not None, "MatMulAddPattern not found"

        alternatives = config.get_alternatives(matmuladd_pattern)
        assert len(alternatives) == 4, f"Expected 4 alternatives, got {len(alternatives)}"

        # Verify sorted by priority (highest=1 first)
        assert alternatives[0].priority == 1
        for alt in alternatives[1:]:
            assert alt.priority >= alternatives[0].priority

    def test_clear_resets_state(self):
        """Test that clear() resets all internal state."""
        config = UnifiedPatternConfig()

        # Load patterns
        _ = config.get_skeleton_patterns()
        _ = config.get_htp_patterns()

        # Clear
        config.clear()

        # After clear, should reload patterns on next access
        skeleton_patterns = config.get_skeleton_patterns()
        assert len(skeleton_patterns) > 0, "Patterns should be reloaded after clear"

    def test_missing_ihv_config_falls_back_to_default(self):
        """Test that missing IHV config falls back to default."""
        # Use a non-existent IHV type
        config = UnifiedPatternConfig(ihv_type="NonExistentIHV")

        # Should load default patterns with a warning
        skeleton_patterns = config.get_skeleton_patterns()
        assert len(skeleton_patterns) == 8, "Should fall back to default patterns"

    def test_alternatives_with_pattern_class(self):
        """Test that alternatives with pattern_class field are loaded correctly."""
        config = UnifiedPatternConfig(ihv_type="QNN")
        patterns = config.get_skeleton_patterns()

        # Find Gelu1Pattern by class name
        gelu1_pattern = next((p for p in patterns if p.__class__.__name__ == "Gelu1Pattern"), None)
        assert gelu1_pattern is not None, "Gelu1Pattern not found"

        # Get alternatives
        alternatives = config.get_alternatives(gelu1_pattern)
        assert len(alternatives) == 1, f"Expected 1 alternative, got {len(alternatives)}"

        alt = alternatives[0]
        assert isinstance(alt, PatternAlternative)

        # Verify the new fields are present
        assert alt.pattern_to_id == "SUBGRAPH/GeluPattern"
        assert alt.pattern_class == "SingleGeluPattern"
        assert alt.module == "winml.modelkit.pattern.gelu_patterns"
        assert alt.priority == 1
        assert "Gelu" in alt.reason
