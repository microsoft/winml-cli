"""Tests for registry CLI flag generation methods and validation.

Tests the cli_flag() and cli_flags() methods of capability classes
that generate CLI argument strings for the optimize command.
Also tests validation functions for error handling coverage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from pathlib import Path

from winml.modelkit.optim.registry import (
    BoolCapability,
    CapabilityCategory,
    ChoiceCapability,
    IntCapability,
    auto_enable_dependencies,
    defaults,
    validate,
    validate_dependencies,
)


class TestBoolCapabilityCliFlags:
    """Tests for BoolCapability.cli_flags() method."""

    def test_cli_flags_returns_tuple(self) -> None:
        """cli_flags() returns a tuple of two strings."""
        cap = BoolCapability(
            name="gelu-fusion",
            ort_name="GeluFusionL2",
            description="GELU fusion",
            category=CapabilityCategory.GELU,
            default=False,
        )
        result = cap.cli_flags()

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_cli_flags_enable_disable_format(self) -> None:
        """cli_flags() generates --enable-X and --disable-X flags."""
        cap = BoolCapability(
            name="gelu-fusion",
            ort_name="GeluFusionL2",
            description="GELU fusion",
            category=CapabilityCategory.GELU,
            default=False,
        )
        enable_flag, disable_flag = cap.cli_flags()

        assert enable_flag == "--enable-gelu-fusion"
        assert disable_flag == "--disable-gelu-fusion"

    def test_cli_flags_preserves_kebab_case(self) -> None:
        """cli_flags() preserves kebab-case in capability name."""
        cap = BoolCapability(
            name="skip-layer-norm-fusion",
            ort_name="SkipLayerNormFusion",
            description="Skip LayerNorm fusion",
            category=CapabilityCategory.LAYER_NORM,
            default=False,
        )
        enable_flag, disable_flag = cap.cli_flags()

        assert enable_flag == "--enable-skip-layer-norm-fusion"
        assert disable_flag == "--disable-skip-layer-norm-fusion"

    def test_cli_flags_simple_name(self) -> None:
        """cli_flags() works with simple single-word names."""
        cap = BoolCapability(
            name="attention",
            ort_name="AttentionFusion",
            description="Attention fusion",
            category=CapabilityCategory.ATTENTION,
            default=False,
        )
        enable_flag, disable_flag = cap.cli_flags()

        assert enable_flag == "--enable-attention"
        assert disable_flag == "--disable-attention"


class TestIntCapabilityCliFlag:
    """Tests for IntCapability.cli_flag() method."""

    def test_cli_flag_returns_string(self) -> None:
        """cli_flag() returns a string."""
        cap = IntCapability(
            name="opt-level",
            ort_name="OptLevel",
            description="Optimization level",
            category=CapabilityCategory.CONTROL,
            default=2,
            min_value=0,
            max_value=3,
        )
        result = cap.cli_flag()

        assert isinstance(result, str)

    def test_cli_flag_value_syntax(self) -> None:
        """cli_flag() generates --X=<value> syntax."""
        cap = IntCapability(
            name="opt-level",
            ort_name="OptLevel",
            description="Optimization level",
            category=CapabilityCategory.CONTROL,
            default=2,
            min_value=0,
            max_value=3,
        )
        flag = cap.cli_flag()

        assert flag == "--opt-level=<value>"

    def test_cli_flag_kebab_case_name(self) -> None:
        """cli_flag() preserves kebab-case in capability name."""
        cap = IntCapability(
            name="max-batch-size",
            ort_name="MaxBatchSize",
            description="Maximum batch size",
            category=CapabilityCategory.CONTROL,
            default=16,
            min_value=1,
            max_value=128,
        )
        flag = cap.cli_flag()

        assert flag == "--max-batch-size=<value>"


class TestChoiceCapabilityCliFlag:
    """Tests for ChoiceCapability.cli_flag() method."""

    def test_cli_flag_returns_string(self) -> None:
        """cli_flag() returns a string."""
        cap = ChoiceCapability(
            name="layout",
            ort_name="Layout",
            description="Data layout",
            category=CapabilityCategory.LAYOUT,
            default="NCHW",
            choices=("NCHW", "NHWC"),
        )
        result = cap.cli_flag()

        assert isinstance(result, str)

    def test_cli_flag_choice_syntax(self) -> None:
        """cli_flag() generates --X={A,B,C} syntax."""
        cap = ChoiceCapability(
            name="layout",
            ort_name="Layout",
            description="Data layout",
            category=CapabilityCategory.LAYOUT,
            default="NCHW",
            choices=("NCHW", "NHWC"),
        )
        flag = cap.cli_flag()

        assert flag == "--layout={NCHW,NHWC}"

    def test_cli_flag_multiple_choices(self) -> None:
        """cli_flag() handles multiple choices correctly."""
        cap = ChoiceCapability(
            name="precision",
            ort_name="Precision",
            description="Model precision",
            category=CapabilityCategory.CONTROL,
            default="fp32",
            choices=("fp32", "fp16", "int8", "int4"),
        )
        flag = cap.cli_flag()

        assert flag == "--precision={fp32,fp16,int8,int4}"

    def test_cli_flag_single_choice(self) -> None:
        """cli_flag() handles single choice."""
        cap = ChoiceCapability(
            name="backend",
            ort_name="Backend",
            description="Backend",
            category=CapabilityCategory.CONTROL,
            default="cpu",
            choices=("cpu",),
        )
        flag = cap.cli_flag()

        assert flag == "--backend={cpu}"

    def test_cli_flag_kebab_case_name(self) -> None:
        """cli_flag() preserves kebab-case in capability name."""
        cap = ChoiceCapability(
            name="model-type",
            ort_name="ModelType",
            description="Model type",
            category=CapabilityCategory.CONTROL,
            default="bert",
            choices=("bert", "gpt", "clip"),
        )
        flag = cap.cli_flag()

        assert flag == "--model-type={bert,gpt,clip}"


class TestCapabilityCliFlagsIntegration:
    """Integration tests for CLI flags with actual capabilities."""

    def test_graph_capability_flags(self) -> None:
        """Test CLI flags for actual graph capabilities."""
        from winml.modelkit.optim.pipes.graph import GRAPH_CAPABILITIES

        # Check a few actual capabilities
        for _cap_name, cap in list(GRAPH_CAPABILITIES.items())[:5]:
            if isinstance(cap, BoolCapability):
                enable, disable = cap.cli_flags()
                assert enable.startswith("--enable-")
                assert disable.startswith("--disable-")
                assert cap.name in enable
                assert cap.name in disable

    def test_fusion_capability_flags(self) -> None:
        """Test CLI flags for actual fusion capabilities."""
        from winml.modelkit.optim.pipes.fusion import ORTFusionPipe

        for _cap_name, cap in list(ORTFusionPipe.capabilities.items())[:5]:
            if isinstance(cap, BoolCapability):
                enable, disable = cap.cli_flags()
                assert enable.startswith("--enable-")
                assert disable.startswith("--disable-")


class TestListCapabilitiesCommand:
    """Tests for --list-capabilities CLI option."""

    def test_list_capabilities_exits_successfully(self) -> None:
        """--list-capabilities exits with success code 0."""
        from click.testing import CliRunner

        from winml.modelkit.commands.optimize import optimize

        runner = CliRunner()
        result = runner.invoke(optimize, ["--list-capabilities"])

        assert result.exit_code == 0

    def test_list_capabilities_compact_shows_categories(self) -> None:
        """--list-capabilities (compact mode) shows category groups."""
        from click.testing import CliRunner

        from winml.modelkit.commands.optimize import optimize

        runner = CliRunner()
        result = runner.invoke(optimize, ["--list-capabilities"])

        # Compact mode shows lowercase category names
        assert "gelu:" in result.output
        assert "attention:" in result.output

    def test_list_capabilities_compact_shows_flags(self) -> None:
        """--list-capabilities (compact mode) shows --enable flags for disabled defaults."""
        from click.testing import CliRunner

        from winml.modelkit.commands.optimize import optimize

        runner = CliRunner()
        result = runner.invoke(optimize, ["--list-capabilities"])

        # Should show --enable-xxx for capabilities that default to False
        assert "--enable-gelu-fusion" in result.output
        assert "Available optimization flags" in result.output

    def test_list_capabilities_compact_shows_verbose_hint(self) -> None:
        """--list-capabilities (compact mode) shows hint to use --verbose."""
        from click.testing import CliRunner

        from winml.modelkit.commands.optimize import optimize

        runner = CliRunner()
        result = runner.invoke(optimize, ["--list-capabilities"])

        assert "--list-capabilities --verbose" in result.output

    def test_list_capabilities_verbose_shows_categories(self) -> None:
        """--list-capabilities --verbose shows category headers."""
        from click.testing import CliRunner

        from winml.modelkit.commands.optimize import optimize

        runner = CliRunner()
        result = runner.invoke(optimize, ["--list-capabilities", "--verbose"])

        # Verbose mode shows uppercase category headers
        assert "GELU" in result.output
        assert "ATTENTION" in result.output

    def test_list_capabilities_verbose_shows_defaults(self) -> None:
        """--list-capabilities --verbose shows default values."""
        from click.testing import CliRunner

        from winml.modelkit.commands.optimize import optimize

        runner = CliRunner()
        result = runner.invoke(optimize, ["--list-capabilities", "--verbose"])

        # Should show "Default:" labels
        assert "Default:" in result.output
        # BoolCapability defaults show as "enabled" or "disabled"
        assert "disabled" in result.output or "enabled" in result.output

    def test_list_capabilities_verbose_shows_ort_names(self) -> None:
        """--list-capabilities --verbose shows ORT optimizer names."""
        from click.testing import CliRunner

        from winml.modelkit.commands.optimize import optimize

        runner = CliRunner()
        result = runner.invoke(optimize, ["--list-capabilities", "--verbose"])

        # Should show ORT names in brackets
        assert "[ORT:" in result.output

    def test_list_capabilities_short_flag(self) -> None:
        """-l is short flag for --list-capabilities."""
        from click.testing import CliRunner

        from winml.modelkit.commands.optimize import optimize

        runner = CliRunner()
        result = runner.invoke(optimize, ["-l"])

        assert result.exit_code == 0
        # Compact mode shows flags
        assert "--enable-gelu-fusion" in result.output

    def test_list_capabilities_does_not_require_model(self) -> None:
        """--list-capabilities works without --model argument."""
        from click.testing import CliRunner

        from winml.modelkit.commands.optimize import optimize

        runner = CliRunner()
        # Should NOT require model when listing capabilities
        result = runner.invoke(optimize, ["--list-capabilities"])

        assert result.exit_code == 0
        # Should NOT show "Missing option '--model'" error
        assert "Missing option" not in result.output

    def test_list_capabilities_verbose_shows_descriptions(self) -> None:
        """--list-capabilities --verbose shows capability descriptions."""
        from click.testing import CliRunner

        from winml.modelkit.commands.optimize import optimize
        from winml.modelkit.optim.pipes import get_all_capabilities

        runner = CliRunner()
        result = runner.invoke(optimize, ["--list-capabilities", "--verbose"])

        # Get one capability description to verify it appears
        all_caps = get_all_capabilities()
        first_cap = next(iter(all_caps.values()))
        # At least part of the description should appear
        assert first_cap.description[:20] in result.output


# =============================================================================
# CAPABILITY VALIDATION TESTS
# =============================================================================


class TestCapabilityProperties:
    """Tests for capability property methods."""

    def test_config_name_equals_name(self) -> None:
        """config_name property returns same as name."""
        cap = BoolCapability(
            name="test-cap",
            ort_name="TestCap",
            description="Test",
            category=CapabilityCategory.GELU,
            default=False,
        )
        assert cap.config_name == "test-cap"
        assert cap.config_name == cap.name

    def test_python_name_converts_kebab_to_snake(self) -> None:
        """python_name converts kebab-case to snake_case."""
        cap = BoolCapability(
            name="multi-word-name",
            ort_name="MultiWordName",
            description="Test",
            category=CapabilityCategory.GELU,
            default=False,
        )
        assert cap.python_name == "multi_word_name"


class TestBoolCapabilityValidation:
    """Tests for BoolCapability validation errors."""

    def test_bool_capability_rejects_non_bool_default(self) -> None:
        """BoolCapability raises TypeError for non-bool default."""
        with pytest.raises(TypeError, match="must have bool default"):
            BoolCapability(
                name="test",
                ort_name="Test",
                description="Test",
                category=CapabilityCategory.GELU,
                default="not a bool",  # type: ignore[arg-type]
            )

    def test_bool_capability_rejects_int_default(self) -> None:
        """BoolCapability raises TypeError for int default (even 0 or 1)."""
        with pytest.raises(TypeError, match="must have bool default"):
            BoolCapability(
                name="test",
                ort_name="Test",
                description="Test",
                category=CapabilityCategory.GELU,
                default=1,  # type: ignore[arg-type]
            )


class TestIntCapabilityValidation:
    """Tests for IntCapability validation errors."""

    def test_int_capability_rejects_non_int_default(self) -> None:
        """IntCapability raises TypeError for non-int default."""
        with pytest.raises(TypeError, match="must have int default"):
            IntCapability(
                name="test",
                ort_name="Test",
                description="Test",
                category=CapabilityCategory.CONTROL,
                default="not an int",  # type: ignore[arg-type]
                min_value=0,
                max_value=10,
            )

    def test_int_capability_rejects_default_below_min(self) -> None:
        """IntCapability raises ValueError for default below min_value."""
        with pytest.raises(ValueError, match="outside range"):
            IntCapability(
                name="test",
                ort_name="Test",
                description="Test",
                category=CapabilityCategory.CONTROL,
                default=-1,
                min_value=0,
                max_value=10,
            )

    def test_int_capability_rejects_default_above_max(self) -> None:
        """IntCapability raises ValueError for default above max_value."""
        with pytest.raises(ValueError, match="outside range"):
            IntCapability(
                name="test",
                ort_name="Test",
                description="Test",
                category=CapabilityCategory.CONTROL,
                default=100,
                min_value=0,
                max_value=10,
            )


class TestChoiceCapabilityValidation:
    """Tests for ChoiceCapability validation errors."""

    def test_choice_capability_rejects_non_str_default(self) -> None:
        """ChoiceCapability raises TypeError for non-str default."""
        with pytest.raises(TypeError, match="must have str default"):
            ChoiceCapability(
                name="test",
                ort_name="Test",
                description="Test",
                category=CapabilityCategory.CONTROL,
                default=123,  # type: ignore[arg-type]
                choices=("a", "b"),
            )

    def test_choice_capability_rejects_empty_choices(self) -> None:
        """ChoiceCapability raises ValueError for empty choices."""
        with pytest.raises(ValueError, match="must have at least one choice"):
            ChoiceCapability(
                name="test",
                ort_name="Test",
                description="Test",
                category=CapabilityCategory.CONTROL,
                default="a",
                choices=(),
            )

    def test_choice_capability_rejects_invalid_default(self) -> None:
        """ChoiceCapability raises ValueError for default not in choices."""
        with pytest.raises(ValueError, match="not in choices"):
            ChoiceCapability(
                name="test",
                ort_name="Test",
                description="Test",
                category=CapabilityCategory.CONTROL,
                default="c",
                choices=("a", "b"),
            )


class TestDefaultsFunction:
    """Tests for defaults() standalone function."""

    def test_defaults_returns_dict(self) -> None:
        """defaults() returns dictionary of default values."""
        caps = {
            "bool-cap": BoolCapability(
                name="bool-cap",
                ort_name="BoolCap",
                description="Test",
                category=CapabilityCategory.GELU,
                default=False,
            ),
            "int-cap": IntCapability(
                name="int-cap",
                ort_name="IntCap",
                description="Test",
                category=CapabilityCategory.CONTROL,
                default=5,
                min_value=0,
                max_value=10,
            ),
        }
        result = defaults(caps)

        assert result == {"bool-cap": False, "int-cap": 5}

    def test_defaults_empty_dict(self) -> None:
        """defaults() returns empty dict for empty capabilities."""
        result = defaults({})
        assert result == {}


class TestValidateFunction:
    """Tests for validate() standalone function."""

    def test_validate_unknown_capability(self) -> None:
        """validate() reports unknown capability keys."""
        caps: dict = {}
        config = {"unknown-cap": True}

        errors = validate(config, caps)

        assert len(errors) == 1
        assert "Unknown capability 'unknown-cap'" in errors[0]

    def test_validate_bool_type_error(self) -> None:
        """validate() reports type error for bool capability."""
        caps = {
            "test": BoolCapability(
                name="test",
                ort_name="Test",
                description="Test",
                category=CapabilityCategory.GELU,
                default=False,
            )
        }
        config = {"test": "not a bool"}

        errors = validate(config, caps)

        assert len(errors) == 1
        assert "expects bool" in errors[0]

    def test_validate_int_type_error(self) -> None:
        """validate() reports type error for int capability."""
        caps = {
            "test": IntCapability(
                name="test",
                ort_name="Test",
                description="Test",
                category=CapabilityCategory.CONTROL,
                default=5,
                min_value=0,
                max_value=10,
            )
        }
        config = {"test": "not an int"}

        errors = validate(config, caps)

        assert len(errors) == 1
        assert "expects int" in errors[0]

    def test_validate_int_range_error(self) -> None:
        """validate() reports range error for int capability."""
        caps = {
            "test": IntCapability(
                name="test",
                ort_name="Test",
                description="Test",
                category=CapabilityCategory.CONTROL,
                default=5,
                min_value=0,
                max_value=10,
            )
        }
        config = {"test": 100}

        errors = validate(config, caps)

        assert len(errors) == 1
        assert "outside range" in errors[0]

    def test_validate_choice_type_error(self) -> None:
        """validate() reports type error for choice capability."""
        caps = {
            "test": ChoiceCapability(
                name="test",
                ort_name="Test",
                description="Test",
                category=CapabilityCategory.CONTROL,
                default="a",
                choices=("a", "b"),
            )
        }
        config = {"test": 123}

        errors = validate(config, caps)

        assert len(errors) == 1
        assert "expects str" in errors[0]

    def test_validate_choice_invalid_value(self) -> None:
        """validate() reports invalid choice value."""
        caps = {
            "test": ChoiceCapability(
                name="test",
                ort_name="Test",
                description="Test",
                category=CapabilityCategory.CONTROL,
                default="a",
                choices=("a", "b"),
            )
        }
        config = {"test": "c"}

        errors = validate(config, caps)

        assert len(errors) == 1
        assert "not in choices" in errors[0]


class TestValidateDependenciesFunction:
    """Tests for validate_dependencies() standalone function."""

    def test_validate_dependencies_missing_dependency(self) -> None:
        """validate_dependencies() reports missing dependency."""
        caps = {
            "child": BoolCapability(
                name="child",
                ort_name="Child",
                description="Child",
                category=CapabilityCategory.GELU,
                default=False,
                depends_on=("parent",),
            ),
            "parent": BoolCapability(
                name="parent",
                ort_name="Parent",
                description="Parent",
                category=CapabilityCategory.GELU,
                default=False,
            ),
        }
        config = {"child": True, "parent": False}

        errors = validate_dependencies(config, caps)

        assert len(errors) == 1
        assert "'child' requires 'parent'" in errors[0]

    def test_validate_dependencies_unknown_dependency(self) -> None:
        """validate_dependencies() reports unknown dependency."""
        caps = {
            "child": BoolCapability(
                name="child",
                ort_name="Child",
                description="Child",
                category=CapabilityCategory.GELU,
                default=False,
                depends_on=("nonexistent",),
            ),
        }
        config = {"child": True}

        errors = validate_dependencies(config, caps)

        assert len(errors) == 1
        assert "depends on unknown capability" in errors[0]

    def test_validate_dependencies_conflict(self) -> None:
        """validate_dependencies() reports conflicting capabilities."""
        caps = {
            "cap1": BoolCapability(
                name="cap1",
                ort_name="Cap1",
                description="Cap1",
                category=CapabilityCategory.GELU,
                default=False,
                conflicts_with=("cap2",),
            ),
            "cap2": BoolCapability(
                name="cap2",
                ort_name="Cap2",
                description="Cap2",
                category=CapabilityCategory.GELU,
                default=False,
            ),
        }
        config = {"cap1": True, "cap2": True}

        errors = validate_dependencies(config, caps)

        assert len(errors) == 1
        assert "'cap1' conflicts with 'cap2'" in errors[0]

    def test_validate_dependencies_unknown_conflict(self) -> None:
        """validate_dependencies() reports unknown conflict capability."""
        caps = {
            "cap1": BoolCapability(
                name="cap1",
                ort_name="Cap1",
                description="Cap1",
                category=CapabilityCategory.GELU,
                default=False,
                conflicts_with=("nonexistent",),
            ),
        }
        config = {"cap1": True}

        errors = validate_dependencies(config, caps)

        assert len(errors) == 1
        assert "conflicts with unknown capability" in errors[0]

    def test_validate_dependencies_int_capability_enabled(self) -> None:
        """validate_dependencies() handles non-bool capability enabled check."""
        caps = {
            "int-cap": IntCapability(
                name="int-cap",
                ort_name="IntCap",
                description="Int Cap",
                category=CapabilityCategory.CONTROL,
                default=5,
                min_value=0,
                max_value=10,
                depends_on=("parent",),
            ),
            "parent": BoolCapability(
                name="parent",
                ort_name="Parent",
                description="Parent",
                category=CapabilityCategory.GELU,
                default=False,
            ),
        }
        # Int capability is "enabled" when present in config
        config = {"int-cap": 5, "parent": False}

        errors = validate_dependencies(config, caps)

        assert len(errors) == 1
        assert "'int-cap' requires 'parent'" in errors[0]

    def test_validate_dependencies_int_depends_on_int(self) -> None:
        """validate_dependencies() handles non-bool depending on non-bool."""
        caps = {
            "child-int": IntCapability(
                name="child-int",
                ort_name="ChildInt",
                description="Child",
                category=CapabilityCategory.CONTROL,
                default=5,
                min_value=0,
                max_value=10,
                depends_on=("parent-int",),
            ),
            "parent-int": IntCapability(
                name="parent-int",
                ort_name="ParentInt",
                description="Parent",
                category=CapabilityCategory.CONTROL,
                default=5,
                min_value=0,
                max_value=10,
            ),
        }
        # Child is in config but parent is not
        config = {"child-int": 5}

        errors = validate_dependencies(config, caps)

        assert len(errors) == 1
        assert "'child-int' requires 'parent-int'" in errors[0]

    def test_validate_dependencies_int_conflicts_with_int(self) -> None:
        """validate_dependencies() handles non-bool conflicting with non-bool."""
        caps = {
            "cap1": IntCapability(
                name="cap1",
                ort_name="Cap1",
                description="Cap1",
                category=CapabilityCategory.CONTROL,
                default=5,
                min_value=0,
                max_value=10,
                conflicts_with=("cap2",),
            ),
            "cap2": IntCapability(
                name="cap2",
                ort_name="Cap2",
                description="Cap2",
                category=CapabilityCategory.CONTROL,
                default=5,
                min_value=0,
                max_value=10,
            ),
        }
        # Both in config = conflict
        config = {"cap1": 5, "cap2": 5}

        errors = validate_dependencies(config, caps)

        assert len(errors) == 1
        assert "'cap1' conflicts with 'cap2'" in errors[0]


class TestAutoEnableDependenciesFunction:
    """Tests for auto_enable_dependencies() standalone function."""

    def test_auto_enable_basic_dependency(self) -> None:
        """auto_enable_dependencies() enables missing dependency."""
        caps = {
            "child": BoolCapability(
                name="child",
                ort_name="Child",
                description="Child",
                category=CapabilityCategory.GELU,
                default=False,
                depends_on=("parent",),
            ),
            "parent": BoolCapability(
                name="parent",
                ort_name="Parent",
                description="Parent",
                category=CapabilityCategory.GELU,
                default=False,
            ),
        }
        config = {"child": True}

        result = auto_enable_dependencies(config, caps)

        assert result["child"] is True
        assert result["parent"] is True

    def test_auto_enable_int_capability_dependency(self) -> None:
        """auto_enable_dependencies() handles non-bool capability with dependency."""
        caps = {
            "int-cap": IntCapability(
                name="int-cap",
                ort_name="IntCap",
                description="Int Cap",
                category=CapabilityCategory.CONTROL,
                default=5,
                min_value=0,
                max_value=10,
                depends_on=("parent",),
            ),
            "parent": BoolCapability(
                name="parent",
                ort_name="Parent",
                description="Parent",
                category=CapabilityCategory.GELU,
                default=False,
            ),
        }
        config = {"int-cap": 5}

        result = auto_enable_dependencies(config, caps)

        assert result["int-cap"] == 5
        assert result["parent"] is True

    def test_auto_enable_preserves_original(self) -> None:
        """auto_enable_dependencies() does not modify original config."""
        caps = {
            "child": BoolCapability(
                name="child",
                ort_name="Child",
                description="Child",
                category=CapabilityCategory.GELU,
                default=False,
                depends_on=("parent",),
            ),
            "parent": BoolCapability(
                name="parent",
                ort_name="Parent",
                description="Parent",
                category=CapabilityCategory.GELU,
                default=False,
            ),
        }
        config = {"child": True}

        auto_enable_dependencies(config, caps)

        assert "parent" not in config  # Original unchanged


# =============================================================================
# CONFIG FILE LOADING TESTS
# =============================================================================


class TestConfigFileLoading:
    """Tests for CLI config file loading functions."""

    def test_load_json_valid(self, tmp_path: Path) -> None:
        """load_config() loads valid JSON file."""
        from winml.modelkit.commands.optimize import load_config

        config_file = tmp_path / "config.json"
        config_file.write_text('{"gelu-fusion": true, "layer-norm-fusion": true}')

        result = load_config(config_file)

        assert result["gelu-fusion"] is True
        assert result["layer-norm-fusion"] is True

    def test_load_json_invalid_syntax(self, tmp_path: Path) -> None:
        """load_config() raises ClickException for invalid JSON."""
        import click

        from winml.modelkit.commands.optimize import load_config

        config_file = tmp_path / "config.json"
        config_file.write_text("{invalid json}")

        with pytest.raises(click.ClickException, match="Invalid JSON"):
            load_config(config_file)

    def test_load_json_non_dict(self, tmp_path: Path) -> None:
        """load_config() raises ClickException for non-dict JSON."""
        import click

        from winml.modelkit.commands.optimize import load_config

        config_file = tmp_path / "config.json"
        config_file.write_text('["array", "not", "dict"]')

        with pytest.raises(click.ClickException, match="must contain a JSON object"):
            load_config(config_file)

    def test_load_yaml_valid(self, tmp_path: Path) -> None:
        """load_config() loads valid YAML file."""
        from winml.modelkit.commands.optimize import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text("gelu-fusion: true\nlayer-norm-fusion: true")

        result = load_config(config_file)

        assert result["gelu-fusion"] is True
        assert result["layer-norm-fusion"] is True

    def test_load_yaml_yml_extension(self, tmp_path: Path) -> None:
        """load_config() accepts .yml extension."""
        from winml.modelkit.commands.optimize import load_config

        config_file = tmp_path / "config.yml"
        config_file.write_text("gelu-fusion: true")

        result = load_config(config_file)

        assert result["gelu-fusion"] is True

    def test_load_yaml_invalid_syntax(self, tmp_path: Path) -> None:
        """load_config() raises ClickException for invalid YAML."""
        import click

        from winml.modelkit.commands.optimize import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid:\n  - [unclosed")

        with pytest.raises(click.ClickException, match="Invalid YAML"):
            load_config(config_file)

    def test_load_yaml_non_dict(self, tmp_path: Path) -> None:
        """load_config() raises ClickException for non-dict YAML."""
        import click

        from winml.modelkit.commands.optimize import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text("- item1\n- item2")

        with pytest.raises(click.ClickException, match="must contain a YAML mapping"):
            load_config(config_file)

    def test_load_unsupported_format(self, tmp_path: Path) -> None:
        """load_config() raises ClickException for unsupported format."""
        import click

        from winml.modelkit.commands.optimize import load_config

        config_file = tmp_path / "config.txt"
        config_file.write_text("some content")

        with pytest.raises(click.ClickException, match="Unsupported config format"):
            load_config(config_file)

    def test_load_file_not_found(self, tmp_path: Path) -> None:
        """load_config() raises ClickException for missing file."""
        import click

        from winml.modelkit.commands.optimize import load_config

        with pytest.raises(click.ClickException, match="Config file not found"):
            load_config(tmp_path / "nonexistent.json")
