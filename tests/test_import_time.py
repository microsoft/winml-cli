# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Regression tests for lazy loading and import-time tracking.

These tests ensure that importing ModelKit modules and running CLI commands
do not pull in heavy ML dependencies (torch, transformers, optimum, etc.)
unless the functionality actually requires them.

Every test runs in a fresh subprocess so sys.modules starts clean.

Test Categories:
    (A) Per-module isolation: verify each modelkit.* package's import budget
    (B) Per-command: verify each CLI command's import budget (--help and --model)
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from winml.modelkit.cli import LazyGroup


# ---------------------------------------------------------------------------
# Discovery — dynamic lists from the actual codebase
# ---------------------------------------------------------------------------

# Discover commands the same way LazyGroup does (filesystem scan, no imports)
_CLI_COMMANDS = LazyGroup().list_commands(ctx=None)  # type: ignore[arg-type]

HEAVY_PREFIXES = ("torch", "transformers", "optimum", "diffusers", "sklearn")


def _run_in_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    """Run Python code in a fresh subprocess via a temp script approach."""
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def assert_no_heavy_imports(
    setup_code: str,
    *,
    forbidden: tuple[str, ...] = HEAVY_PREFIXES,
    allowed: tuple[str, ...] = (),
) -> None:
    """Assert that running setup_code loads no forbidden modules.

    Args:
        setup_code: Python code to execute (will be dedented).
        forbidden: Module prefixes that must NOT appear in sys.modules.
        allowed: Module prefixes to exclude from the forbidden check.
    """
    script = textwrap.dedent(f"""\
        import sys
        {setup_code}
        loaded = sorted(set(
            m.split('.')[0] for m in sys.modules
            if m.startswith({forbidden!r})
        ))
        allowed = set({allowed!r})
        bad = [m for m in loaded if m not in allowed]
        if bad:
            print(f"FAIL: unexpected heavy modules: {{bad}}", file=sys.stderr)
            print(f"  allowed: {{allowed}}", file=sys.stderr)
            sys.exit(1)
    """)
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Import budget violated.\nstderr: {result.stderr.strip()}"


def assert_cli_no_heavy_imports(
    cli_args: list[str],
    *,
    allowed: tuple[str, ...] = (),
) -> None:
    """Assert that invoking ``main(cli_args)`` loads no forbidden modules.

    Uses try/except to catch SystemExit and Click errors gracefully.
    """
    args_str = repr(cli_args)
    script = textwrap.dedent(f"""\
        import sys
        from winml.modelkit.cli import main
        import click
        try:
            main({args_str}, standalone_mode=False)
        except (SystemExit, click.exceptions.UsageError, Exception):
            pass
        loaded = sorted(set(
            m.split('.')[0] for m in sys.modules
            if m.startswith({HEAVY_PREFIXES!r})
        ))
        allowed = set({allowed!r})
        bad = [m for m in loaded if m not in allowed]
        if bad:
            print(f"FAIL: unexpected heavy modules: {{bad}}", file=sys.stderr)
            print(f"  allowed: {{allowed}}", file=sys.stderr)
            sys.exit(1)
    """)
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Import budget violated for args {cli_args}.\nstderr: {result.stderr.strip()}"
    )


# ===========================================================================
# (A) Per-Module Isolation Tests
# ===========================================================================


class TestModuleIsolation:
    """Verify each modelkit.* module's import budget."""

    @pytest.mark.parametrize(
        "module",
        [
            "winml.modelkit",
            "winml.modelkit.cli",
            "winml.modelkit.cache",
            "winml.modelkit.compiler",
            "winml.modelkit.config",
            "winml.modelkit.core",
            "winml.modelkit.export",
            "winml.modelkit.loader",
            "winml.modelkit.onnx",
            "winml.modelkit.optim",
            "winml.modelkit.optracing",
            "winml.modelkit.quant",
            "winml.modelkit.session",
            "winml.modelkit.sysinfo",
            "winml.modelkit.utils",
        ],
    )
    def test_module_no_heavy_deps(self, module: str) -> None:
        """Importing this module must not load torch/transformers/optimum."""
        assert_no_heavy_imports(f"import {module}")

    @pytest.mark.parametrize(
        ("module", "allowed"),
        [
            ("winml.modelkit.build", ("torch", "torchgen")),
            ("winml.modelkit.data", ("torch", "torchgen", "torchvision")),
            (
                "winml.modelkit.datasets",
                ("torch", "torchgen", "torchvision", "transformers", "sklearn"),
            ),
            ("winml.modelkit.inspect", (*HEAVY_PREFIXES, "torchgen", "torchvision")),
            ("winml.modelkit.models", (*HEAVY_PREFIXES, "torchgen", "torchvision")),
            ("winml.modelkit.quant", ("torch", "torchgen", "torchvision")),
        ],
    )
    def test_module_with_expected_deps(self, module: str, allowed: tuple[str, ...]) -> None:
        """Modules that legitimately need heavy deps — verify nothing extra."""
        assert_no_heavy_imports(f"import {module}", allowed=allowed)

    def test_lazy_access_triggers_import(self) -> None:
        """Accessing WinMLAutoModel must trigger the full import chain."""
        script = textwrap.dedent("""\
            import sys
            from winml.modelkit import WinMLAutoModel
            assert 'torch' in sys.modules, (
                'torch should be loaded after accessing WinMLAutoModel'
            )
        """)
        result = _run_in_subprocess(script)
        assert result.returncode == 0, (
            f"Lazy access did not trigger torch.\nstderr: {result.stderr}"
        )


# ===========================================================================
# (B) Per-Command Tests — --help (no module imports via LazyGroup)
# ===========================================================================


class TestCommandHelp:
    """Verify ``wmk`` and ``wmk <cmd> --help`` do not load heavy deps."""

    def test_wmk_bare(self) -> None:
        """Bare ``wmk`` (no args) must not load heavy deps."""
        assert_cli_no_heavy_imports([])

    def test_wmk_help(self) -> None:
        """``wmk --help`` must not load heavy deps."""
        assert_cli_no_heavy_imports(["--help"])

    @pytest.mark.parametrize("cmd", _CLI_COMMANDS)
    def test_command_help_no_heavy_deps(self, cmd: str) -> None:
        """``wmk <cmd> --help`` must not load heavy deps."""
        assert_cli_no_heavy_imports([cmd, "--help"])


# ===========================================================================
# (B) Per-Command Tests — with --model (actual command execution)
# ===========================================================================

_FAKE_ONNX = "nonexistent_test_model.onnx"
_HF_MODEL = "microsoft/resnet-50"


class TestCommandWithModel:
    """Verify import budgets when commands are invoked with --model.

    Commands that operate on ONNX files should NOT need torch/transformers.
    Commands that operate on HF models legitimately need them.

    We use a fake model path so commands fail at file I/O, but the import
    chain is already established by that point.
    """

    @pytest.mark.parametrize(
        ("cmd_args", "allowed"),
        [
            # ONNX-path commands — should NOT need torch/transformers
            (
                ["compile", "--model", _FAKE_ONNX, "-o", "o.onnx", "--ep", "qnn"],
                (),
            ),
            (
                ["quantize", "--model", _FAKE_ONNX, "-o", "o.onnx", "--ep", "qnn"],
                (),
            ),
            (
                ["optimize", "--model", _FAKE_ONNX, "-o", "o.onnx"],
                ("torch", "torchgen"),  # ORT tools.__init__ pulls torch
            ),
            (
                ["perf", "--model", _FAKE_ONNX],
                (),
            ),
            (
                ["static-analyzer", "check", "--model", _FAKE_ONNX, "--ep", "qnn"],
                ("torch", "torchgen"),  # ORT tools.__init__ pulls torch
            ),
            # HF model commands — legitimately need heavy deps
            (
                ["inspect", "-m", _HF_MODEL],
                (*HEAVY_PREFIXES, "torchgen", "torchvision"),
            ),
            (
                ["config", "-m", _HF_MODEL, "--device", "npu", "--precision", "int8"],
                (*HEAVY_PREFIXES, "torchgen", "torchvision"),
            ),
        ],
        ids=[
            "compile-onnx",
            "quantize-onnx",
            "optimize-onnx",
            "perf-onnx",
            "static-analyzer-onnx",
            "inspect-hf",
            "config-hf",
        ],
    )
    def test_command_import_budget(self, cmd_args: list[str], allowed: tuple[str, ...]) -> None:
        """Verify each command's import budget with --model."""
        assert_cli_no_heavy_imports(cmd_args, allowed=allowed)
