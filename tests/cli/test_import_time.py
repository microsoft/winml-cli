# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Regression tests for lazy loading and import-time tracking.

These tests ensure that importing WinML CLI modules and running CLI commands
do not pull in heavy ML dependencies (torch, transformers, optimum, etc.)
unless the functionality actually requires them.

Every test runs in a fresh subprocess so sys.modules starts clean.

Test Categories:
    (A) Per-module isolation: verify each winml.modelkit.* package's import budget
    (B) Per-command: verify each CLI command's import budget (--help and --model)
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


# ---------------------------------------------------------------------------
# Discovery — dynamic lists from the actual codebase
# ---------------------------------------------------------------------------


# Discover commands by scanning the commands/ directory (same logic as cli.py)
def _discover_command_names() -> list[str]:
    from pathlib import Path

    # Walk up until we find the repo root (marked by pyproject.toml).
    # Resilient to this file's depth within tests/.
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    commands_dir = root / "src" / "winml" / "modelkit" / "commands"
    return sorted(f.stem for f in commands_dir.glob("*.py") if not f.name.startswith("_"))


_CLI_COMMANDS = _discover_command_names()

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
    """Verify each winml.modelkit.* module's import budget."""

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
            "winml.modelkit.analyze",
            "winml.modelkit.pattern",
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
            (
                "winml.modelkit.eval",
                ("torch", "torchgen", "torchvision", "transformers", "sklearn"),
            ),
            ("winml.modelkit.inspect", (*HEAVY_PREFIXES, "torchgen", "torchvision")),
            ("winml.modelkit.models", (*HEAVY_PREFIXES, "torchgen", "torchvision")),
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

    # -- Gap 2: lazy-trigger tests for subpackage __getattr__ implementations --

    def test_lazy_core_get_io_config(self) -> None:
        """core.get_io_config must be lazily accessible and callable."""
        script = textwrap.dedent("""\
            import winml.modelkit.core
            obj = winml.modelkit.core.get_io_config
            assert obj is not None
            assert callable(obj)
        """)
        result = _run_in_subprocess(script)
        assert result.returncode == 0, (
            f"core.get_io_config not lazily accessible.\nstderr: {result.stderr}"
        )

    def test_lazy_export_resolve_io_specs(self) -> None:
        """export.resolve_io_specs must be lazily accessible and callable."""
        script = textwrap.dedent("""\
            import winml.modelkit.export
            obj = winml.modelkit.export.resolve_io_specs
            assert obj is not None
            assert callable(obj)
        """)
        result = _run_in_subprocess(script)
        assert result.returncode == 0, (
            f"export.resolve_io_specs not lazily accessible.\nstderr: {result.stderr}"
        )

    def test_lazy_loader_load_hf_model(self) -> None:
        """loader.load_hf_model must be lazily accessible and callable."""
        script = textwrap.dedent("""\
            import winml.modelkit.loader
            obj = winml.modelkit.loader.load_hf_model
            assert obj is not None
            assert callable(obj)
        """)
        result = _run_in_subprocess(script)
        assert result.returncode == 0, (
            f"loader.load_hf_model not lazily accessible.\nstderr: {result.stderr}"
        )

    def test_lazy_quant_quantize_onnx(self) -> None:
        """quant.quantize_onnx must be lazily accessible and callable."""
        script = textwrap.dedent("""\
            import winml.modelkit.quant
            obj = winml.modelkit.quant.quantize_onnx
            assert obj is not None
            assert callable(obj)
        """)
        result = _run_in_subprocess(script)
        assert result.returncode == 0, (
            f"quant.quantize_onnx not lazily accessible.\nstderr: {result.stderr}"
        )

    # -- Gap 3: AttributeError negative test --

    def test_nonexistent_attr_raises(self) -> None:
        """Importing a nonexistent attribute must raise ImportError."""
        script = textwrap.dedent("""\
            try:
                from winml.modelkit import nonexistent_xyz_12345
            except ImportError:
                pass  # expected
            else:
                raise AssertionError(
                    "Expected ImportError for nonexistent attribute"
                )
        """)
        result = _run_in_subprocess(script)
        assert result.returncode == 0, (
            f"Nonexistent attr did not raise ImportError.\nstderr: {result.stderr}"
        )

    # -- Gap 4: __dir__ correctness test --

    def test_dir_includes_lazy_attrs(self) -> None:
        """dir(winml.modelkit) must include lazy attrs without loading torch."""
        script = textwrap.dedent("""\
            import sys
            import winml.modelkit
            assert "WinMLAutoModel" in dir(winml.modelkit), (
                "WinMLAutoModel missing from dir()"
            )
            loaded = sorted(set(
                m.split('.')[0] for m in sys.modules
                if m.startswith(('torch', 'transformers', 'optimum', 'diffusers', 'sklearn'))
            ))
            if loaded:
                print(f"FAIL: dir() triggered heavy imports: {loaded}", file=sys.stderr)
                sys.exit(1)
        """)
        result = _run_in_subprocess(script)
        assert result.returncode == 0, f"dir() test failed.\nstderr: {result.stderr}"


# ===========================================================================
# (C) _LAZY_IMPORTS Dict Consistency Tests
# ===========================================================================

_LAZY_MODULES = [
    "winml.modelkit",
    "winml.modelkit.core",
    "winml.modelkit.export",
    "winml.modelkit.loader",
    "winml.modelkit.quant",
    "winml.modelkit.models",
    "winml.modelkit.onnx",
]


class TestLazyImportsDict:
    """Verify the standardized _LAZY_IMPORTS pattern across all modules."""

    @pytest.mark.parametrize("module", _LAZY_MODULES)
    def test_lazy_imports_dict_exists(self, module: str) -> None:
        """Each module must define a non-empty _LAZY_IMPORTS dict."""
        script = textwrap.dedent(f"""\
            import {module} as mod
            lazy = getattr(mod, '_LAZY_IMPORTS', None)
            assert lazy is not None, '_LAZY_IMPORTS not found on {module}'
            assert isinstance(lazy, dict), (
                f'_LAZY_IMPORTS is {{type(lazy).__name__}}, expected dict'
            )
            assert len(lazy) > 0, '_LAZY_IMPORTS is empty'
        """)
        result = _run_in_subprocess(script)
        assert result.returncode == 0, (
            f"_LAZY_IMPORTS check failed for {module}.\nstderr: {result.stderr.strip()}"
        )

    @pytest.mark.parametrize("module", _LAZY_MODULES)
    def test_lazy_imports_all_consistent(self, module: str) -> None:
        """Every key in _LAZY_IMPORTS must also appear in __all__."""
        script = textwrap.dedent(f"""\
            import {module} as mod
            lazy = set(mod._LAZY_IMPORTS.keys())
            all_ = set(mod.__all__)
            missing = lazy - all_
            assert not missing, f'In _LAZY_IMPORTS but not __all__: {{missing}}'
        """)
        result = _run_in_subprocess(script)
        assert result.returncode == 0, (
            f"_LAZY_IMPORTS/__all__ drift in {module}.\nstderr: {result.stderr.strip()}"
        )

    @pytest.mark.parametrize("module", _LAZY_MODULES)
    def test_lazy_imports_all_resolvable(self, module: str) -> None:
        """Every _LAZY_IMPORTS entry must resolve to a real attribute.

        Convention: ``_LAZY_IMPORTS`` maps a lazy attribute name to a
        ``(submodule_path, real_attr_name)`` tuple, where ``submodule_path``
        is relative (e.g. ``".config"``) resolved against the host package.
        """
        script = textwrap.dedent(f"""\
            import importlib
            import {module} as mod
            errors = []
            for lazy_name, (submodule_path, real_attr) in mod._LAZY_IMPORTS.items():
                try:
                    sub = importlib.import_module(submodule_path, package={module!r})
                    if not hasattr(sub, real_attr):
                        errors.append(
                            f'{{lazy_name}}: {{submodule_path}}.{{real_attr}} not found'
                        )
                except ImportError as exc:
                    errors.append(f'{{lazy_name}}: cannot import {{submodule_path}} ({{exc}})')
            if errors:
                raise AssertionError(
                    f'Unresolvable _LAZY_IMPORTS in {module}:\\n' + '\\n'.join(errors)
                )
        """)
        result = _run_in_subprocess(script)
        assert result.returncode == 0, (
            f"Unresolvable _LAZY_IMPORTS in {module}.\nstderr: {result.stderr.strip()}"
        )


# ===========================================================================
# (B) Per-Command Tests -- --help (no heavy imports at command load time)
# ===========================================================================


class TestCommandHelp:
    """Verify ``winml`` and ``winml <cmd> --help`` do not load heavy deps."""

    def test_winml_bare(self) -> None:
        """Bare ``winml`` (no args) must not load heavy deps."""
        assert_cli_no_heavy_imports([])

    def test_winml_help(self) -> None:
        """``winml --help`` must not load heavy deps."""
        assert_cli_no_heavy_imports(["--help"])

    @pytest.mark.parametrize("cmd", _CLI_COMMANDS)
    def test_command_help_no_heavy_deps(self, cmd: str) -> None:
        """``winml <cmd> --help`` must not load heavy deps."""
        assert_cli_no_heavy_imports([cmd, "--help"])

    def test_inspect_list_tasks_no_heavy_deps(self) -> None:
        """``winml inspect --list-tasks`` must not load transformers/optimum.

        Regression guard for the latency fix where ``--list-tasks`` used to
        import ``optimum.exporters.tasks.TasksManager`` (and transitively
        ``transformers``), turning a static dict lookup into a ~12 s wait.
        """
        assert_cli_no_heavy_imports(["inspect", "--list-tasks"])


# Note: this file deliberately does NOT cover per-command runtime import
# budgets (e.g., "winml compile --model X.onnx" not pulling torch). Those
# tests would invoke handler bodies and cross from CLI-surface territory
# into feature-pipeline territory. The init-time guarantees here cover:
#   - importing winml.modelkit.* subpackages (TestModuleIsolation)
#   - winml --help and winml <cmd> --help (TestCommandHelp)
#   - lazy-import dict structure (TestLazyImportsDict)
# If per-command runtime budgets become a concern, they belong in a
# feature-test file with mocks at the dispatch boundary — not here.
