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
import os
import tempfile
from pathlib import Path

import pytest

from winml.modelkit.analyze import IHVType, RuleLoader
from winml.modelkit.analyze.utils import get_runtime_rules_search_dirs, resolve_rule_zip_path
from winml.modelkit.analyze.utils import rule_expander as rule_expander_module
from winml.modelkit.analyze.utils import rule_loader as rule_loader_module
from winml.modelkit.analyze.utils.rule_loader import _DEFAULT_RUNTIME_RULES_DIR, _RULE_LOADER_DIR


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
                            "status": "supported",
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


class TestResolveRuleZipPath:
    """Test resolve_rule_zip_path and get_runtime_rules_search_dirs."""

    def test_default_search_dir_included(self, monkeypatch):
        """Default embedded dir is always in the search list."""
        monkeypatch.delenv("MODELKIT_RULES_DIR", raising=False)
        dirs = get_runtime_rules_search_dirs()
        assert len(dirs) >= 1
        assert dirs[0].name == "runtime_check_rules"

    def test_env_var_adds_dirs(self, monkeypatch):
        """MODELKIT_RULES_DIR adds extra search directories."""
        monkeypatch.setenv("MODELKIT_RULES_DIR", f"/extra/path1{os.pathsep}/extra/path2")
        dirs = get_runtime_rules_search_dirs()
        assert len(dirs) == 3
        assert dirs[0] == Path("/extra/path1").resolve()
        assert dirs[1] == Path("/extra/path2").resolve()
        assert dirs[2].name == "runtime_check_rules"

    def test_env_var_relative_path_resolved_from_module_dir(self, monkeypatch):
        """Relative MODELKIT_RULES_DIR entries are resolved from rule_loader.py dir."""
        relative_entry = "custom/rules"
        monkeypatch.setenv("MODELKIT_RULES_DIR", relative_entry)

        dirs = get_runtime_rules_search_dirs()

        assert len(dirs) == 2
        assert dirs[0] == (_RULE_LOADER_DIR / relative_entry).resolve()
        assert dirs[1] == _DEFAULT_RUNTIME_RULES_DIR

    def test_env_var_empty_ignored(self, monkeypatch):
        """Empty MODELKIT_RULES_DIR is treated as unset."""
        monkeypatch.setenv("MODELKIT_RULES_DIR", "  ")
        dirs = get_runtime_rules_search_dirs()
        assert len(dirs) == 1

    def test_resolve_finds_file_in_env_dir(self, monkeypatch):
        """resolve_rule_zip_path finds a zip in an env var directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_name = "QNN_NPU_ai_onnx_opset13.zip"
            (Path(tmpdir) / zip_name).write_bytes(b"PK")
            monkeypatch.setenv("MODELKIT_RULES_DIR", tmpdir)

            result = resolve_rule_zip_path(zip_name)
            assert result == Path(tmpdir).resolve() / zip_name
            assert result.exists()

    def test_resolve_fallback_to_default(self, monkeypatch):
        """When no directory has the file, returns the default path."""
        monkeypatch.delenv("MODELKIT_RULES_DIR", raising=False)
        result = resolve_rule_zip_path("nonexistent_file.zip")
        assert result == _DEFAULT_RUNTIME_RULES_DIR / "nonexistent_file.zip"

    def test_resolve_prefers_env_over_default(self, monkeypatch):
        """Env var dirs are searched first (before default dir)."""
        zip_name = "test_priority.zip"

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / zip_name).write_bytes(b"PK")
            monkeypatch.setenv("MODELKIT_RULES_DIR", tmpdir)

            result = resolve_rule_zip_path(zip_name)
            assert result == Path(tmpdir).resolve() / zip_name

    def test_resolve_auto_expands_when_marker_missing(self, monkeypatch):
        """Auto-expand is triggered when zip exists and expanded marker is missing."""
        zip_name = "QNN_NPU_ai_onnx_opset13.zip"

        with tempfile.TemporaryDirectory() as tmpdir:
            rules_dir = Path(tmpdir)
            (rules_dir / zip_name).write_bytes(b"PK")
            monkeypatch.setenv("MODELKIT_RULES_DIR", tmpdir)
            monkeypatch.setattr(rule_loader_module, "_EXPAND_CHECKED_DIRS", set())

            calls: list[Path] = []

            def _fake_expand_rules_zip_dir(
                rules_dir: Path,
                *,
                output_dir: Path | None = None,
                glob_pattern: str = "*.zip",
                marker_filename: str = "expanded",
            ):
                del output_dir, glob_pattern, marker_filename
                calls.append(rules_dir.resolve())

            monkeypatch.setattr(
                rule_expander_module,
                "expand_rules_zip_dir",
                _fake_expand_rules_zip_dir,
            )

            result = resolve_rule_zip_path(zip_name)

            assert result == rules_dir.resolve() / zip_name
            assert calls == [rules_dir.resolve()]

    def test_resolve_skips_auto_expand_when_marker_exists(self, monkeypatch):
        """Auto-expand is skipped when expanded marker already exists."""
        zip_name = "QNN_NPU_ai_onnx_opset13.zip"

        with tempfile.TemporaryDirectory() as tmpdir:
            rules_dir = Path(tmpdir)
            (rules_dir / zip_name).write_bytes(b"PK")
            (rules_dir / rule_expander_module.EXPANDED_MARKER_FILE).touch()
            monkeypatch.setenv("MODELKIT_RULES_DIR", tmpdir)
            monkeypatch.setattr(rule_loader_module, "_EXPAND_CHECKED_DIRS", set())

            called = False

            def _fake_expand_rules_zip_dir(
                rules_dir: Path,
                *,
                output_dir: Path | None = None,
                glob_pattern: str = "*.zip",
                marker_filename: str = "expanded",
            ):
                del rules_dir, output_dir, glob_pattern, marker_filename
                nonlocal called
                called = True

            monkeypatch.setattr(
                rule_expander_module,
                "expand_rules_zip_dir",
                _fake_expand_rules_zip_dir,
            )

            result = resolve_rule_zip_path(zip_name)

            assert result == rules_dir.resolve() / zip_name
            assert called is False
