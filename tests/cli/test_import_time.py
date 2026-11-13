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

# Fixtures for TestCommandWithModel: paths that fail I/O without triggering imports.
_FAKE_ONNX = "does-not-exist.onnx"
_HF_MODEL = "microsoft/resnet-50"


def _run_in_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    """Run Python code in a fresh subprocess via a temp script approach."""
    return subprocess.run(  # noqa: S603
        [sys.executable, "-X", "utf8", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


@pytest.mark.parametrize(
    "module",
    [
        "decoder_only",
        "encoder_decoder",
        "composite_model",
        "genai_bundle",
        "kv_cache",
        "zero_shot_image_classification",
    ],
)
def test_model_modules_import_directly_in_fresh_process(module: str) -> None:
    result = _run_in_subprocess(f"""
        import importlib
        importlib.import_module("winml.modelkit.models.winml.{module}")
    """)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "symbol",
    ["COMPOSITE_MODEL_REGISTRY", "GENAI_BUNDLE_REGISTRY", "WINML_MODEL_CLASS_MAPPING"],
)
def test_model_registry_is_populated_on_first_access(symbol: str) -> None:
    code = f"""
        import json
        import winml.modelkit.models as models
        import winml.modelkit.models.winml as wrappers
        from winml.modelkit.models.winml import {symbol} as registry
        assert registry
        assert getattr(wrappers, {symbol!r}) is registry
        if {symbol!r} in models.__all__:
            assert getattr(models, {symbol!r}) is registry
        print(json.dumps(sorted(map(repr, registry))))
    """
    lazy = _run_in_subprocess(code)
    eager = _run_in_subprocess("import winml.modelkit.models.hf\n" + textwrap.dedent(code))
    assert lazy.returncode == eager.returncode == 0, lazy.stderr + eager.stderr
    assert lazy.stdout == eager.stdout


def test_composite_lookup_initializes_registrations() -> None:
    result = _run_in_subprocess("""
        from types import SimpleNamespace
        from unittest.mock import patch
        from winml.modelkit.models.winml import WinMLCompositeModel
        from winml.modelkit.models.winml import composite_model
        try:
            WinMLCompositeModel.from_onnx({})
        except ValueError:
            pass
        populated = dict(composite_model.COMPOSITE_MODEL_REGISTRY)
        import winml.modelkit.models.hf
        assert populated
        assert populated == composite_model.COMPOSITE_MODEL_REGISTRY
        for (model_type, task), cls in populated.items():
            with patch.object(cls, "__init__", return_value=None):
                model = WinMLCompositeModel.from_onnx(
                    {}, task=task, hf_config=SimpleNamespace(model_type=model_type)
                )
                assert type(model) is cls
    """)
    assert result.returncode == 0, result.stdout + result.stderr


def test_specialization_override_survives_registration_imports() -> None:
    reference = _run_in_subprocess("""
        import json
        from winml.modelkit.models.winml import WINML_MODEL_CLASS_MAPPING
        print(json.dumps(next(iter(WINML_MODEL_CLASS_MAPPING))))
    """)
    assert reference.returncode == 0, reference.stderr
    result = _run_in_subprocess(f"""
        import json
        from winml.modelkit.models.winml import register_specialization, get_winml_class
        model_type, task = json.loads({reference.stdout!r})
        register_specialization(model_type, task, "WinMLModelForGenericTask")
        import winml.modelkit.models.hf
        from winml.modelkit.models.winml import WinMLModelForGenericTask
        assert get_winml_class(model_type, task) is WinMLModelForGenericTask
    """)
    assert result.returncode == 0, result.stdout + result.stderr


def test_successful_onnx_perf_does_not_import_hf(tmp_path) -> None:
    result = _run_in_subprocess(f"""
        import json
        import sys
        from pathlib import Path
        import onnx
        from winml.modelkit.cli import main

        model_path = Path({str(tmp_path)!r}) / "generated.onnx"
        report_path = model_path.with_suffix(".json")
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["input"], ["output"])],
            "generated",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4])],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 4])],
        )
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=8
        )
        onnx.checker.check_model(model)
        onnx.save(model, model_path)
        main([
            "perf", "-m", str(model_path), "--device", "cpu", "--ep", "cpu",
            "--skip-build", "--no-memory", "--warmup", "1", "--iterations", "2",
            "-o", str(report_path),
        ], standalone_mode=False)
        report = json.loads(report_path.read_text())
        assert len(report["raw_samples_ms"]) == 2
        assert not set({(*HEAVY_PREFIXES, "datasets", "PIL")!r}).intersection(sys.modules)
    """)
    assert result.returncode == 0, result.stdout + result.stderr


def test_random_dataset_does_not_import_transformers() -> None:
    result = _run_in_subprocess("""
        import sys
        import numpy as np
        import torch
        from winml.modelkit.datasets import RandomDataset

        dataset = RandomDataset(
            model_path=None,
            io_config={
                "input_names": ["input"],
                "input_shapes": [[1, 4]],
                "input_types": [np.dtype("float32")],
            },
            max_samples=2,
        )
        assert len(dataset) == 2
        assert isinstance(dataset[0]["input"], torch.Tensor)
        torch.testing.assert_close(dataset[0]["input"], dataset[0]["input"])
        assert not {"transformers", "torchvision", "optimum"}.intersection(sys.modules)
    """)
    assert result.returncode == 0, result.stdout + result.stderr


def test_successful_onnx_eval_compare_does_not_import_transformers(tmp_path) -> None:
    result = _run_in_subprocess(f"""
        import json
        import sys
        from pathlib import Path
        import onnx
        from winml.modelkit.cli import main

        model_path = Path({str(tmp_path)!r}) / "generated.onnx"
        report_path = model_path.with_suffix(".json")
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["input"], ["output"])],
            "generated",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4])],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 4])],
        )
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=8
        )
        onnx.checker.check_model(model)
        onnx.save(model, model_path)
        main([
            "eval", "--mode", "compare", "-m", str(model_path),
            "--reference", str(model_path), "--device", "cpu", "--ep", "cpu",
            "--reference-device", "cpu", "--reference-ep", "cpu", "--skip-build",
            "--samples", "2", "-o", str(report_path),
        ], standalone_mode=False)
        report = json.loads(report_path.read_text())
        assert report["metrics"]
        assert not {{"transformers", "torchvision", "optimum"}}.intersection(sys.modules)
    """)
    assert result.returncode == 0, result.stdout + result.stderr


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
    result = _run_in_subprocess(script)
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
    result = _run_in_subprocess(script)
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
            "winml.modelkit.quant",
            "winml.modelkit.session",
            "winml.modelkit.analyze",
            "winml.modelkit.pattern",
            "winml.modelkit.sysinfo",
            "winml.modelkit.utils",
            "winml.modelkit.models",
            "winml.modelkit.models.winml",
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
        ],
    )
    def test_module_with_expected_deps(self, module: str, allowed: tuple[str, ...]) -> None:
        """Modules that legitimately need heavy deps — verify nothing extra."""
        assert_no_heavy_imports(f"import {module}", allowed=allowed)

    def test_auto_model_and_generic_wrapper_stay_lightweight(self) -> None:
        """Factory and artifact wrappers must not initialize HF export support."""
        assert_no_heavy_imports(
            "from winml.modelkit import WinMLAutoModel; "
            "from winml.modelkit.models import WinMLModelForGenericTask",
            forbidden=(*HEAVY_PREFIXES, "PIL", "datasets"),
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
    "winml.modelkit.session",
    "winml.modelkit.session.monitor",
    "winml.modelkit.sysinfo",
    "winml.modelkit.quant",
    "winml.modelkit.models",
    "winml.modelkit.models.winml",
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
                    elif {module!r}.startswith("winml.modelkit.models"):
                        assert getattr(mod, lazy_name) is getattr(sub, real_attr)
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

    def test_inspect_model_type_hides_onnxruntime_startup_warning(self) -> None:
        """``winml inspect --model-type`` must not leak ORT native warning noise."""
        script = textwrap.dedent("""\
            from winml.modelkit.cli import main

            main(["inspect", "--model-type", "bert", "--format", "json"], standalone_mode=False)
        """)
        result = _run_in_subprocess(script)
        assert result.returncode == 0, result.stderr.strip()
        assert "Init provider bridge failed" not in result.stderr
        assert "onnxruntime" not in result.stderr


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


# ===========================================================================
# (D) Wall-clock guardrail — catches cumulative slowdown sys.modules misses
# ===========================================================================


class TestWallClock:
    """A large batch of lightweight imports can degrade startup without any
    single heavy module triggering a budget failure. This guardrail catches
    that."""

    def test_winml_help_under_8s(self) -> None:
        """``winml --help`` in a fresh subprocess must complete in < 8s.

        The ceiling includes Python cold-start (300-800 ms on Windows +
        AV scanning on GitHub Actions shared runners), so 3s would be
        flaky. 8s still catches the ~10s transformers regression this
        test was written to guard against, with ~2s headroom.
        """
        import time

        script = (
            "from winml.modelkit.cli import main\n"
            "try:\n"
            "    main(['--help'], standalone_mode=False)\n"
            "except SystemExit:\n"
            "    pass\n"
        )
        t0 = time.perf_counter()
        result = _run_in_subprocess(script)
        elapsed = time.perf_counter() - t0
        assert result.returncode == 0, f"exit {result.returncode}: {result.stderr}"
        assert elapsed < 8.0, f"winml --help took {elapsed:.2f}s (limit 8.0s)"
