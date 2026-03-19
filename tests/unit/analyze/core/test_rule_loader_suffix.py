# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for RuleLoader suffix-based filtering.

Tests the new feature where Information rules are loaded
based on file suffix (_information.json).
"""

import json
import tempfile
from pathlib import Path

import pytest

from winml.modelkit.analyze.models.information import Information
from winml.modelkit.analyze.utils.rule_loader import RuleLoader


@pytest.fixture
def temp_rules_dir():
    """Create a temporary rules directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rules_path = Path(tmpdir) / "rules"
        rules_path.mkdir()

        information_rules_path = rules_path / "information_rules"
        information_rules_path.mkdir()

        yield information_rules_path


class TestRuleLoaderSuffixFiltering:
    """Test suffix-based filtering for Information rules."""

    def test_loads_only_information_suffix_files(self, temp_rules_dir):
        """Test that only *_information.json files are loaded."""
        # Create valid information file
        valid_info = [
            {
                "explanation": "Test information",
                "actions": [],
            }
        ]
        (temp_rules_dir / "qc_information.json").write_text(
            json.dumps(valid_info), encoding="utf-8"
        )

        # Create non-information files
        mapping_data = [{"OP/ai.onnx/Conv": "QNN_OP_CONV_2D"}]
        (temp_rules_dir / "onnx_to_qnn_mapping.json").write_text(
            json.dumps(mapping_data), encoding="utf-8"
        )

        runtime_data = [{"op_type": "Conv", "compile": True}]
        (temp_rules_dir / "QNNExecutionProvider_NPU_from_doc.json").write_text(
            json.dumps(runtime_data), encoding="utf-8"
        )

        # Load rules
        loader = RuleLoader(rules_dir=temp_rules_dir.parent)
        rules = loader.load_information_rules()

        # Should only load from qc_information.json
        assert len(rules) == 1
        assert isinstance(rules[0], Information)
        assert rules[0].explanation == "Test information"

    def test_loads_multiple_information_files(self, temp_rules_dir):
        """Test that multiple *_information.json files are loaded."""
        # Create multiple information files
        qc_info = [{"explanation": "QC information", "actions": []}]
        (temp_rules_dir / "qc_information.json").write_text(json.dumps(qc_info), encoding="utf-8")

        intel_info = [{"explanation": "Intel information", "actions": []}]
        (temp_rules_dir / "intel_information.json").write_text(
            json.dumps(intel_info), encoding="utf-8"
        )

        # Load rules
        loader = RuleLoader(rules_dir=temp_rules_dir.parent)
        rules = loader.load_information_rules()

        # Should load both files
        assert len(rules) == 2
        explanations = {rule.explanation for rule in rules}
        assert "QC information" in explanations
        assert "Intel information" in explanations

    def test_ignores_files_without_information_suffix(self, temp_rules_dir):
        """Test that files without _information.json suffix are ignored."""
        # Create file with wrong suffix
        wrong_suffix = [{"explanation": "Should be ignored", "actions": []}]
        (temp_rules_dir / "qc_rules.json").write_text(json.dumps(wrong_suffix), encoding="utf-8")

        (temp_rules_dir / "information.json").write_text(json.dumps(wrong_suffix), encoding="utf-8")

        # Load rules
        loader = RuleLoader(rules_dir=temp_rules_dir.parent)
        rules = loader.load_information_rules()

        # Should not load any files
        assert len(rules) == 0

    def test_handles_empty_information_file(self, temp_rules_dir):
        """Test that empty _information.json files are handled correctly."""
        # Create empty information file
        (temp_rules_dir / "empty_information.json").write_text("[]", encoding="utf-8")

        # Load rules
        loader = RuleLoader(rules_dir=temp_rules_dir.parent)
        rules = loader.load_information_rules()

        # Should return empty list without errors
        assert len(rules) == 0

    def test_filters_disabled_actions(self, temp_rules_dir):
        """Test that disabled actions are filtered out."""
        # Create information with mix of enabled/disabled actions
        info_with_actions = [
            {
                "explanation": "Test with actions",
                "actions": [
                    {
                        "pattern_from_id": "OP/ai.onnx/Conv",
                        "pattern_to_id": "OP/ai.onnx/Relu",
                        "level": "required",
                        "action": "Replace",
                        "status": "supported",
                        "details": "Replace with Relu",
                        "enabled": True,
                    },
                    {
                        "pattern_from_id": "OP/ai.onnx/Add",
                        "pattern_to_id": "OP/ai.onnx/Mul",
                        "level": "optional",
                        "action": "Consider",
                        "status": "partial",
                        "details": "Consider replacement",
                        "enabled": False,  # This should be filtered
                    },
                ],
            }
        ]
        (temp_rules_dir / "actions_information.json").write_text(
            json.dumps(info_with_actions), encoding="utf-8"
        )

        # Load rules
        loader = RuleLoader(rules_dir=temp_rules_dir.parent)
        rules = loader.load_information_rules()

        # Should have 1 rule with 1 action (disabled filtered out)
        assert len(rules) == 1
        assert len(rules[0].actions) == 1
        assert rules[0].actions[0].pattern_from_id == "OP/ai.onnx/Conv"

    def test_handles_missing_information_rules_directory(self):
        """Test that missing directory is handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_path = Path(tmpdir) / "rules"
            rules_path.mkdir()
            # Don't create information_rules subdirectory

            loader = RuleLoader(rules_dir=rules_path)
            rules = loader.load_information_rules()

            # Should return empty list without errors
            assert len(rules) == 0


class TestRuleLoaderBackwardCompatibility:
    """Test backward compatibility with existing rule files."""

    def test_existing_qc_information_file_loads(self, temp_rules_dir):
        """Test that existing qc_information.json file loads correctly."""
        # Simulate existing file structure
        qc_info = [
            {
                "explanation": "Conv not supported on NPU",
                "actions": [
                    {
                        "pattern_from_id": "OP/ai.onnx/Conv",
                        "pattern_to_id": "OP/ai.onnx/QLinearConv",
                        "level": "required",
                        "action": "Replace with quantized version",
                        "status": "unsupported",
                        "details": "Conv requires quantization",
                    }
                ],
            }
        ]
        (temp_rules_dir / "qc_information.json").write_text(json.dumps(qc_info), encoding="utf-8")

        loader = RuleLoader(rules_dir=temp_rules_dir.parent)
        rules = loader.load_information_rules()

        assert len(rules) == 1
        assert rules[0].explanation == "Conv not supported on NPU"
        assert len(rules[0].actions) == 1
        assert rules[0].actions[0].level.value == "required"
