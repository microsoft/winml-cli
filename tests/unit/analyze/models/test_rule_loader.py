# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Unit tests for RuleLoader with IHV filtering and graceful degradation.

Tests verify:
- load_runtime_rules with IHV filtering
- load_pattern_rules
- load_information_rules
- get_rules_for_pattern method
- Graceful degradation when rule files are missing
- Rule file parsing and validation
"""

import json
import tempfile
from pathlib import Path

import pytest

from winml.modelkit.analyze.models.ihv_type import IHVType
from winml.modelkit.analyze.utils.rule_loader import RuleLoader


class TestRuleLoaderBasicLoading:
    """Test basic rule loading functionality."""

    @pytest.fixture
    def temp_rules_dir(self):
        """Create a temporary rules directory with test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_dir = Path(tmpdir) / "rules"
            rules_dir.mkdir()

            # Create subdirectories
            runtime_dir = rules_dir / "runtime_check_rules"
            pattern_dir = rules_dir / "pattern_rules"
            information_dir = rules_dir / "information_rules"

            runtime_dir.mkdir()
            pattern_dir.mkdir()
            information_dir.mkdir()

            # Create QC rules
            qc_rules = [
                {
                    "pattern_id": "OP/ai.onnx/Conv",
                    "ihv_type": "QC",
                    "ep_version": "*",
                    "driver_version": "*",
                    "test_result": {
                        "compile": True,
                        "run": True,
                    },
                }
            ]
            (runtime_dir / "qc_rules.json").write_text(json.dumps(qc_rules), encoding="utf-8")

            # Create Intel rules
            intel_rules = [
                {
                    "pattern_id": "OP/ai.onnx/Conv",
                    "ihv_type": "Intel",
                    "ep_version": "2023.3",
                    "driver_version": "*",
                    "test_result": {
                        "compile": True,
                        "run": True,
                    },
                }
            ]
            (runtime_dir / "intel_rules.json").write_text(json.dumps(intel_rules), encoding="utf-8")

            # Create AMD rules
            amd_rules = [
                {
                    "pattern_id": "OP/ai.onnx/Relu",
                    "ihv_type": "AMD",
                    "test_result": {
                        "compile": True,
                        "run": True,
                    },
                }
            ]
            (runtime_dir / "amd_rules.json").write_text(json.dumps(amd_rules), encoding="utf-8")

            # Create pattern rules
            pattern_rules = [
                {
                    "pattern_id": "SUBGRAPH/GELU",
                    "pattern_name": "GELU",
                    "description": "Gaussian Error Linear Unit",
                    "node_topology": {
                        "n1": "Div",
                        "n2": "Erf",
                        "n3": "Add",
                        "n4": "Mul",
                    },
                    "edge_topology": [["n1", "n2"], ["n2", "n3"], ["n3", "n4"]],
                }
            ]
            (pattern_dir / "gelu_patterns.json").write_text(
                json.dumps(pattern_rules), encoding="utf-8"
            )

            # Create information rules
            information_rules = [
                {
                    "Information_id": "info-001",
                    "pattern_id": "SUBGRAPH/GELU",
                    "explanation": "Replace with optimized GELU",
                    "actions": [
                        {
                            "pattern_from_id": "SUBGRAPH/GELU_Erf",
                            "pattern_to_id": "OP/ai.onnx/Gelu",
                            "type": "required",
                            "action": "Use native Gelu operator",
                            "status": "white",
                            "details": "Native Gelu is fully supported",
                        }
                    ],
                }
            ]
            (information_dir / "gelu_information.json").write_text(
                json.dumps(information_rules), encoding="utf-8"
            )

            yield rules_dir

    def test_load_runtime_rules_all_ihvs(self, temp_rules_dir):
        """Test loading runtime rules for all IHVs."""
        loader = RuleLoader(temp_rules_dir)
        rules_by_ihv = loader.load_runtime_rules()

        assert "QC" in rules_by_ihv
        assert "Intel" in rules_by_ihv
        assert "AMD" in rules_by_ihv
        assert len(rules_by_ihv["QC"]) == 1
        assert len(rules_by_ihv["Intel"]) == 1
        assert len(rules_by_ihv["AMD"]) == 1
        assert rules_by_ihv["QC"][0].pattern_id == "OP/ai.onnx/Conv"

    def test_load_runtime_rules_specific_ihv(self, temp_rules_dir):
        """Test loading runtime rules for a specific IHV."""
        loader = RuleLoader(temp_rules_dir)
        rules_by_ihv = loader.load_runtime_rules(ihv_type=IHVType.QC)

        assert "QC" in rules_by_ihv
        assert "Intel" not in rules_by_ihv
        assert "AMD" not in rules_by_ihv
        assert len(rules_by_ihv["QC"]) == 1

    def test_load_runtime_rules_ihv_filtering(self, temp_rules_dir):
        """Test that IHV filtering works correctly."""
        loader = RuleLoader(temp_rules_dir)

        # Load only Intel rules
        intel_rules = loader.load_runtime_rules(ihv_type=IHVType.INTEL)
        assert "Intel" in intel_rules
        assert len(intel_rules) == 1
        assert intel_rules["Intel"][0].ihv_type == IHVType.INTEL

        # Load only AMD rules
        amd_rules = loader.load_runtime_rules(ihv_type=IHVType.AMD)
        assert "AMD" in amd_rules
        assert len(amd_rules) == 1
        assert amd_rules["AMD"][0].pattern_id == "OP/ai.onnx/Relu"

    def test_load_information_rules(self, temp_rules_dir):
        """Test loading information rules."""
        loader = RuleLoader(temp_rules_dir)
        informations = loader.load_information_rules()

        assert len(informations) == 1
        assert informations[0].pattern_id == "SUBGRAPH/GELU"
        assert informations[0].explanation == "Replace with optimized GELU"

    def test_get_rules_for_pattern(self, temp_rules_dir):
        """Test getting rules for a specific pattern and IHV."""
        loader = RuleLoader(temp_rules_dir)

        # Get QC rules for Conv pattern
        qc_conv_rules = loader.get_rules_for_pattern("OP/ai.onnx/Conv", IHVType.QC)
        assert len(qc_conv_rules) == 1
        assert qc_conv_rules[0].ihv_type == IHVType.QC

        # Get Intel rules for Conv pattern
        intel_conv_rules = loader.get_rules_for_pattern("OP/ai.onnx/Conv", IHVType.INTEL)
        assert len(intel_conv_rules) == 1
        assert intel_conv_rules[0].ep_version == "2023.3"

        # Get rules for non-existent pattern
        non_existent_rules = loader.get_rules_for_pattern("OP/ai.onnx/NonExistent", IHVType.QC)
        assert len(non_existent_rules) == 0

    def test_rule_caching(self, temp_rules_dir):
        """Test that loaded rules are cached."""
        loader = RuleLoader(temp_rules_dir)

        # First call loads rules
        rules1 = loader.get_rules_for_pattern("OP/ai.onnx/Conv", IHVType.QC)

        # Second call should use cache
        rules2 = loader.get_rules_for_pattern("OP/ai.onnx/Conv", IHVType.QC)

        assert rules1 == rules2
        assert len(loader.runtime_rules) > 0


class TestRuleLoaderGracefulDegradation:
    """Test graceful degradation when rule files are missing."""

    def test_missing_runtime_rules_file(self):
        """Test that missing runtime rules file is handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_dir = Path(tmpdir) / "rules"
            rules_dir.mkdir()
            (rules_dir / "runtime_check_rules").mkdir()

            # No files created - should return empty dict with keys for each IHV
            loader = RuleLoader(rules_dir)
            rules_by_ihv = loader.load_runtime_rules(ihv_type=IHVType.QC)

            assert "QC" in rules_by_ihv
            assert rules_by_ihv["QC"] == []

    def test_missing_information_rules_directory(self):
        """Test that missing information_rules directory is handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_dir = Path(tmpdir) / "rules"
            rules_dir.mkdir()

            # No information_rules directory created
            loader = RuleLoader(rules_dir)
            informations = loader.load_information_rules()

            assert informations == []

    def test_invalid_json_in_rule_file(self):
        """Test that invalid JSON is handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_dir = Path(tmpdir) / "rules"
            rules_dir.mkdir()
            runtime_dir = rules_dir / "runtime_check_rules"
            runtime_dir.mkdir()

            # Create invalid JSON file
            (runtime_dir / "qc_rules.json").write_text("{ invalid json content", encoding="utf-8")

            loader = RuleLoader(rules_dir)
            rules_by_ihv = loader.load_runtime_rules(ihv_type=IHVType.QC)

            # Should return empty list due to JSON error
            assert rules_by_ihv == {"QC": []}

    def test_empty_rule_file(self):
        """Test that empty rule files are handled correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_dir = Path(tmpdir) / "rules"
            rules_dir.mkdir()
            runtime_dir = rules_dir / "runtime_check_rules"
            runtime_dir.mkdir()

            # Create empty JSON array
            (runtime_dir / "qc_rules.json").write_text(json.dumps([]), encoding="utf-8")

            loader = RuleLoader(rules_dir)
            rules_by_ihv = loader.load_runtime_rules(ihv_type=IHVType.QC)

            assert rules_by_ihv == {"QC": []}

    def test_malformed_rule_in_file(self):
        """Test that malformed rules are skipped but valid rules are loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_dir = Path(tmpdir) / "rules"
            rules_dir.mkdir()
            runtime_dir = rules_dir / "runtime_check_rules"
            runtime_dir.mkdir()

            # Create file with one valid and one invalid rule
            rules = [
                {
                    "pattern_id": "OP/ai.onnx/Conv",
                    "ihv_type": "QC",
                    "test_result": {
                        "compile": True,
                        "run": True,
                    },
                },
                {
                    # Missing required fields
                    "pattern_id": "OP/ai.onnx/Relu",
                },
            ]
            (runtime_dir / "qc_rules.json").write_text(json.dumps(rules), encoding="utf-8")

            loader = RuleLoader(rules_dir)
            rules_by_ihv = loader.load_runtime_rules(ihv_type=IHVType.QC)

            # Should load only the valid rule
            assert len(rules_by_ihv["QC"]) == 1
            assert rules_by_ihv["QC"][0].pattern_id == "OP/ai.onnx/Conv"


class TestRuleLoaderWildcardMatching:
    """Test wildcard matching in loaded rules."""

    @pytest.fixture
    def temp_wildcard_rules(self):
        """Create rules with wildcard patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_dir = Path(tmpdir) / "rules"
            rules_dir.mkdir()
            runtime_dir = rules_dir / "runtime_check_rules"
            runtime_dir.mkdir()

            # Create rules with wildcards
            qc_rules = [
                {
                    "pattern_id": "OP/ai.onnx/Conv",
                    "ihv_type": "QC",
                    "ep_version": "*",
                    "driver_version": "*",
                    "type_vars": {"T": "*"},
                    "attributes": {"kernel_shape": "*", "pads": "*"},
                    "test_result": {
                        "compile": True,
                        "run": True,
                    },
                },
                {
                    "pattern_id": "OP/ai.onnx/Conv",
                    "ihv_type": "QC",
                    "ep_version": "2.0",
                    "driver_version": "1.0.0",
                    "type_vars": {"T": "float32"},
                    "attributes": {"kernel_shape": "[3, 3]"},
                    "test_result": {
                        "compile": True,
                        "run": True,
                    },
                },
            ]
            (runtime_dir / "qc_rules.json").write_text(json.dumps(qc_rules), encoding="utf-8")

            yield rules_dir

    def test_load_wildcard_rules(self, temp_wildcard_rules):
        """Test that wildcard rules are loaded correctly."""
        loader = RuleLoader(temp_wildcard_rules)
        rules_by_ihv = loader.load_runtime_rules(ihv_type=IHVType.QC)

        assert len(rules_by_ihv["QC"]) == 2

        # First rule has wildcards
        wildcard_rule = rules_by_ihv["QC"][0]
        assert wildcard_rule.ep_version == "*"
        assert wildcard_rule.driver_version == "*"
        assert wildcard_rule.type_vars == {"T": "*"}
        assert wildcard_rule.attributes == {"kernel_shape": "*", "pads": "*"}

        # Second rule has specific values
        specific_rule = rules_by_ihv["QC"][1]
        assert specific_rule.ep_version == "2.0"
        assert specific_rule.type_vars == {"T": "float32"}

    def test_get_all_matching_rules_for_pattern(self, temp_wildcard_rules):
        """Test getting all rules (wildcard and specific) for a pattern."""
        loader = RuleLoader(temp_wildcard_rules)

        conv_rules = loader.get_rules_for_pattern("OP/ai.onnx/Conv", IHVType.QC)

        assert len(conv_rules) == 2
        assert any(rule.ep_version == "*" for rule in conv_rules)
        assert any(rule.ep_version == "2.0" for rule in conv_rules)


class TestRuleLoaderDefaultPath:
    """Test default path behavior."""

    def test_default_rules_dir_path(self):
        """Test that default rules directory path is set correctly."""
        loader = RuleLoader()

        # Should default to src/analyze/rules/
        assert loader.rules_dir.name == "rules"
        assert "analyze" in str(loader.rules_dir)


class TestRuleLoaderWithRealMockData:
    """Test rule loading with real mock data files."""

    @pytest.fixture
    def mock_data_rules_dir(self):
        """Return path to actual mock data rules directory."""
        # Get the path to tests/mock_data/analyze/rules
        test_file_path = Path(__file__)
        tests_dir = test_file_path.parent.parent.parent.parent
        return tests_dir / "mock_data" / "analyze" / "rules"

    def test_load_real_qc_runtime_rules(self, mock_data_rules_dir):
        """Test loading real QC runtime rules from mock data."""
        loader = RuleLoader(mock_data_rules_dir)
        rules_by_ihv = loader.load_runtime_rules(ihv_type=IHVType.QC)

        assert "QC" in rules_by_ihv
        assert len(rules_by_ihv["QC"]) >= 1

        # Verify the Conv rule structure
        conv_rule = rules_by_ihv["QC"][0]
        assert conv_rule.pattern_id == "OP/ai.onnx/Conv"
        assert conv_rule.ihv_type == IHVType.QC
        assert conv_rule.ep_version == "2.18.0"
        assert conv_rule.driver_version == "1.5.3"
        assert conv_rule.namespace == "ai.onnx"
        assert conv_rule.op_version == 11

        # Verify type variables
        assert conv_rule.type_vars["T"] == "float32"
        assert conv_rule.type_vars["W"] == "float32|float16"

        # Verify attributes
        assert conv_rule.attributes["kernel_shape"] == "[3, 3]"
        assert conv_rule.attributes["strides"] == "[1, 1]"
        assert conv_rule.attributes["pads"] == "*"

        # Verify test result
        assert conv_rule.test_result.compile is True
        assert conv_rule.test_result.run is True
        assert conv_rule.test_result.reason is None

    def test_load_real_qc_information_rules(self, mock_data_rules_dir):
        """Test loading real QC information rules from mock data."""
        loader = RuleLoader(mock_data_rules_dir)
        informations = loader.load_information_rules()

        assert len(informations) >= 1

        # Find GELU_Erf information
        gelu_info = next((i for i in informations if i.pattern_id == "SUBGRAPH/GELU_Erf"), None)
        assert gelu_info is not None

        # Verify information structure
        assert gelu_info.Information_id == "550e8400-e29b-41d4-a716-446655440000"
        assert "Erf-based GELU" in gelu_info.explanation
        assert "30%" in gelu_info.explanation

        # Verify actions
        actions = gelu_info.actions
        assert len(actions) >= 1

        action = actions[0]
        assert action.pattern_from_id == "SUBGRAPH/GELU_Erf"
        assert action.pattern_to_id == "OP/ai.onnx/Gelu"
        assert "Replace" in action.details or "native Gelu operator" in action.details
        assert "opset 20+" in action.details or "native Gelu operator" in action.details

    def test_real_data_cross_reference(self, mock_data_rules_dir):
        """
        Test that runtime and information rules cross-reference.
        """
        loader = RuleLoader(mock_data_rules_dir)

        # Load all rule types
        informations = loader.load_information_rules()
        runtime_rules = loader.load_runtime_rules()

        # Verify information references patterns
        gelu_info_pattern_ids = [i.pattern_id for i in informations]
        assert "SUBGRAPH/GELU_Erf" in gelu_info_pattern_ids

        # Verify Conv operator has runtime rules
        qc_rules = runtime_rules.get("QC", [])
        conv_rule_exists = any(rule.pattern_id == "OP/ai.onnx/Conv" for rule in qc_rules)
        assert conv_rule_exists

    def test_real_data_alternatives_in_runtime_rules(self, mock_data_rules_dir):
        """Test that runtime rules contain alternative implementations."""
        loader = RuleLoader(mock_data_rules_dir)
        rules_by_ihv = loader.load_runtime_rules(ihv_type=IHVType.QC)

        conv_rule = rules_by_ihv["QC"][0]

        # Verify alternatives exist
        assert hasattr(conv_rule, "alternatives")
        assert conv_rule.alternatives is not None
        assert len(conv_rule.alternatives) >= 3

        # Verify alternative structure (alternatives are list of dicts)
        alternatives = conv_rule.alternatives

        # First alternative: {"OP/ai.onnx/QLinearConv": "QDQ"}
        assert "OP/ai.onnx/QLinearConv" in alternatives[0]
        assert alternatives[0]["OP/ai.onnx/QLinearConv"] == "QDQ"

        # Second alternative: {"OP/ai.onnx/ConvInteger": "equivalent"}
        assert "OP/ai.onnx/ConvInteger" in alternatives[1]
        assert alternatives[1]["OP/ai.onnx/ConvInteger"] == "equivalent"

        # Third alternative: {"SUBGRAPH/DepthwiseConv2d": "approximation"}
        assert "SUBGRAPH/DepthwiseConv2d" in alternatives[2]
        assert alternatives[2]["SUBGRAPH/DepthwiseConv2d"] == "approximation"

    def test_real_data_input_shapes_and_constants(self, mock_data_rules_dir):
        """Test that runtime rules contain input shape and constant information."""
        loader = RuleLoader(mock_data_rules_dir)
        rules_by_ihv = loader.load_runtime_rules(ihv_type=IHVType.QC)

        conv_rule = rules_by_ihv["QC"][0]

        # Verify input shapes
        assert hasattr(conv_rule, "input_shapes")
        assert conv_rule.input_shapes is not None
        assert "input_0" in conv_rule.input_shapes
        assert "input_1" in conv_rule.input_shapes

        # Input 0 should be [1, 3, 224, 224] - standard image input
        assert conv_rule.input_shapes["input_0"] == [1, 3, 224, 224]

        # Input 1 should be [64, 3, 3, 3] - Conv kernel weights
        assert conv_rule.input_shapes["input_1"] == [64, 3, 3, 3]

        # Verify constant indicators
        assert hasattr(conv_rule, "input_is_constant")
        assert conv_rule.input_is_constant is not None
        assert conv_rule.input_is_constant["input_0"] is False  # Image is not constant
        assert conv_rule.input_is_constant["input_1"] is True  # Weights are constant

    def test_get_rules_for_conv_pattern_real_data(self, mock_data_rules_dir):
        """Test getting Conv rules using real mock data."""
        loader = RuleLoader(mock_data_rules_dir)

        # Get Conv rules for QC
        conv_rules = loader.get_rules_for_pattern("OP/ai.onnx/Conv", IHVType.QC)

        assert len(conv_rules) >= 1
        assert all(rule.pattern_id == "OP/ai.onnx/Conv" for rule in conv_rules)
        assert all(rule.ihv_type == IHVType.QC for rule in conv_rules)

    def test_real_data_wildcard_in_attributes(self, mock_data_rules_dir):
        """Test that wildcard values in attributes are preserved."""
        loader = RuleLoader(mock_data_rules_dir)
        rules_by_ihv = loader.load_runtime_rules(ihv_type=IHVType.QC)

        conv_rule = rules_by_ihv["QC"][0]

        # Verify that 'pads' attribute has wildcard value
        assert conv_rule.attributes["pads"] == "*"

        # Verify specific values for other attributes
        assert conv_rule.attributes["kernel_shape"] == "[3, 3]"
        assert conv_rule.attributes["strides"] == "[1, 1]"
