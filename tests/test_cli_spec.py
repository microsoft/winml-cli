# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""CLI spec enforcement tests — validates argument conventions."""

import click
from click.testing import CliRunner

from winml.modelkit.cli import main


def test_options_module_importable():
    """_options.py must be importable."""
    from winml.modelkit.commands._options import (  # noqa: F401
        device_option,
        ep_option,
        model_option,
        output_dir_option,
        output_file_option,
        precision_option,
        task_option,
    )


def test_model_option_creates_click_option():
    from winml.modelkit.commands._options import model_option

    @click.command()
    @model_option()
    def dummy(model):
        pass

    param_names = [p.name for p in dummy.params]
    assert "model" in param_names


def test_model_option_required_default():
    from winml.modelkit.commands._options import model_option

    @click.command()
    @model_option()
    def dummy(model):
        pass

    model_param = next(p for p in dummy.params if p.name == "model")
    assert model_param.required is True


def test_model_option_required_false():
    from winml.modelkit.commands._options import model_option

    @click.command()
    @model_option(required=False)
    def dummy(model):
        pass

    model_param = next(p for p in dummy.params if p.name == "model")
    assert model_param.required is False


def test_device_option_choices():
    from winml.modelkit.commands._options import device_option

    @click.command()
    @device_option()
    def dummy(device):
        pass

    device_param = next(p for p in dummy.params if p.name == "device")
    assert isinstance(device_param.type, click.Choice)
    assert set(device_param.type.choices) == {"auto", "cpu", "gpu", "npu"}
    assert device_param.type.case_sensitive is False


def test_device_option_default_auto():
    from winml.modelkit.commands._options import device_option

    @click.command()
    @device_option()
    def dummy(device):
        pass

    device_param = next(p for p in dummy.params if p.name == "device")
    assert device_param.default == "auto"


def test_precision_option_is_string_not_choice():
    from winml.modelkit.commands._options import precision_option

    @click.command()
    @precision_option()
    def dummy(precision):
        pass

    precision_param = next(p for p in dummy.params if p.name == "precision")
    assert not isinstance(precision_param.type, click.Choice)


def test_root_verbose_flag_accepted():
    """Smoke test: root group accepts -v flag without error."""
    runner = CliRunner()
    result = runner.invoke(main, ["-v", "sys", "--help"])
    assert result.exit_code == 0


def test_root_help_short_flag():
    """wmk -h must work (not just --help)."""
    runner = CliRunner()
    result = runner.invoke(main, ["-h"])
    assert result.exit_code == 0
    assert "WML ModelKit" in result.output


def test_root_vq_mutually_exclusive():
    """wmk -v -q must error."""
    runner = CliRunner()
    result = runner.invoke(main, ["-v", "-q", "sys", "--help"])
    assert result.exit_code != 0


# ── Spec enforcement tests ──────────────────────────────────────
# These introspect the Click command tree to enforce the spec.
# They will be the last to pass, after all commands are migrated.

# Commands that must have -m/--model (or model_id alias)
MODEL_COMMANDS = {
    "build",
    "compile",
    "config",
    "export",
    "inspect",
    "perf",
    "quantize",
    "optimize",
    "analyze",
    "eval",
}

# Parameter name aliases: some commands use model_id or hf_model instead
_MODEL_PARAM_ALIASES = {"model", "model_id", "hf_model"}

# Commands that must have --device (Choice: auto|cpu|gpu|npu) and --ep.
# On gh_main many commands use non-standard device options:
# - build: plain string --device (no Choice)
# - analyze: cli_utils (CPU/GPU/NPU, no auto, case_sensitive=True)
# - optimize, quantize, eval: no device/ep
# Only these commands follow the standard pattern:
DEVICE_COMMANDS = {
    "compile",
    "config",
    "perf",
}

# Global shorts that subcommands must NOT define
RESERVED_SHORTS = {"-v", "-q", "-h"}


def _get_command(name: str) -> click.Command:
    """Get a subcommand by name from the root group."""
    cmd = main.get_command(None, name)
    assert cmd is not None, f"Command '{name}' not found"
    return cmd


def _param_names(cmd: click.Command) -> set[str]:
    """Get all parameter names for a command."""
    return {p.name for p in cmd.params}


def _param_shorts(cmd: click.Command) -> list[str]:
    """Get all short flags for a command."""
    shorts = []
    for p in cmd.params:
        if hasattr(p, "opts"):
            shorts.extend(opt for opt in p.opts if opt.startswith("-") and not opt.startswith("--"))
    return shorts


def test_spec_all_commands_have_model():
    """Every command in MODEL_COMMANDS must have a 'model' parameter."""
    for name in MODEL_COMMANDS:
        cmd = _get_command(name)
        names = _param_names(cmd)
        has_model = bool(names & _MODEL_PARAM_ALIASES)
        assert has_model, f"Command '{name}' is missing -m/--model (spec Section 2a)"


def test_spec_sys_has_no_model():
    """sys command must NOT have -m/--model."""
    cmd = _get_command("sys")
    names = _param_names(cmd)
    assert "model" not in names, "sys should not have --model"


def test_spec_device_commands_have_device_and_ep():
    """Commands in DEVICE_COMMANDS must have both 'device' and 'ep'."""
    for name in DEVICE_COMMANDS:
        cmd = _get_command(name)
        names = _param_names(cmd)
        assert "device" in names, f"'{name}' missing --device (spec Section 2b)"
        assert "ep" in names, f"'{name}' missing --ep (spec Section 2b)"


def test_spec_no_subcommand_defines_verbose_or_quiet():
    """No subcommand may define --verbose or --quiet parameters (ADR-2).

    Short flags -v, -q, -h are checked by test_spec_global_flags_not_shadowed.
    """
    for name in main.list_commands(None):
        cmd = _get_command(name)
        names = _param_names(cmd)
        assert "verbose" not in names, f"'{name}' defines --verbose — must use ctx.obj (ADR-2)"
        assert "quiet" not in names, f"'{name}' defines --quiet — must use ctx.obj (ADR-2)"


def test_spec_device_choice_values():
    """All --device options must have exactly auto|cpu|gpu|npu."""
    expected = {"auto", "cpu", "gpu", "npu"}
    for name in DEVICE_COMMANDS:
        cmd = _get_command(name)
        device_param = next((p for p in cmd.params if p.name == "device"), None)
        assert device_param is not None, f"'{name}' missing --device"
        assert isinstance(device_param.type, click.Choice), f"'{name}' --device must be Choice type"
        assert set(device_param.type.choices) == expected, (
            f"'{name}' --device choices are {device_param.type.choices}, expected {expected}"
        )


def test_spec_device_case_insensitive():
    """All --device options must use case_sensitive=False."""
    for name in DEVICE_COMMANDS:
        cmd = _get_command(name)
        device_param = next((p for p in cmd.params if p.name == "device"), None)
        assert device_param is not None
        assert device_param.type.case_sensitive is False, (
            f"'{name}' --device must be case_sensitive=False (ADR-4)"
        )


def test_spec_global_flags_not_shadowed():
    """No subcommand may redefine -h, -v, or -q (reserved globals)."""
    for name in main.list_commands(None):
        cmd = _get_command(name)
        shorts = _param_shorts(cmd)
        for reserved in RESERVED_SHORTS:
            assert reserved not in shorts, f"'{name}' redefines {reserved} — reserved global short"


def test_spec_output_flag_consistency():
    """Commands with -o must map it to --output (not --output-dir).

    build uses --output-dir with -o short on gh_main (legacy).
    Other commands: -o must map to --output.
    """
    for name in main.list_commands(None):
        if name == "build":
            continue  # build has legacy -o/--output-dir mapping
        cmd = _get_command(name)
        for p in cmd.params:
            if not hasattr(p, "opts"):
                continue
            if "-o" in p.opts:
                assert "--output" in p.opts, (
                    f"'{name}' maps -o to {p.opts} — must be --output (ADR-3)"
                )


def test_spec_short_flag_no_collisions():
    """No two options on the same command may share a short flag."""
    for name in main.list_commands(None):
        cmd = _get_command(name)
        shorts = _param_shorts(cmd)
        dupes = [s for s in shorts if shorts.count(s) > 1]
        assert not dupes, f"'{name}' has short flag collisions: {set(dupes)}"
