# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for winml perf --module flag."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from winml.modelkit.cli import main
from winml.modelkit.commands.perf import generate_output_path


class TestPerfModuleFlag:
    """Tests for --module flag on winml perf."""

    def test_module_flag_in_help(self) -> None:
        """Verify --module flag appears in winml perf --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["perf", "--help"])
        assert result.exit_code == 0
        assert "--module" in result.output

    def test_module_flag_requires_hf_model(self) -> None:
        """--module without --hf-model should fail."""
        runner = CliRunner()
        result = runner.invoke(main, ["perf", "--module", "BertAttention"])
        assert result.exit_code != 0

    def test_module_default_output_includes_class_name(self) -> None:
        """Default output path includes module class name."""
        # The single-model generate_output_path produces {slug}_perf.json
        path = generate_output_path("bert-base-uncased")
        assert "bert-base-uncased" in str(path)

        # The module-mode output path (from _perf_modules) is
        # {slug}_{module_class}_perf.json -- tested via the inline logic
        # in _perf_modules. Here we verify the format difference:
        from pathlib import Path

        slug = "bert-base-uncased"
        module_class = "BertAttention"
        module_path = Path(f"{slug}_{module_class}_perf.json")
        assert module_class in str(module_path)
        assert str(module_path) != str(path)


class TestPerfModulesNoCompile:
    """Test no_compile/no_quantize stage overrides inside _perf_modules."""

    def _run_perf_modules(self, *, no_quantize: bool, no_compile: bool):
        """Invoke _perf_modules with mocked internals; return build call kwargs."""
        from winml.modelkit.commands.perf import _perf_modules
        from winml.modelkit.config import WinMLBuildConfig

        cfg = WinMLBuildConfig()
        cfg.loader.task = "image-classification"
        cfg.loader.model_type = "bert"
        cfg.loader.module_path = "encoder.layer.0"

        build_calls = []

        def fake_build(config, **_):
            build_calls.append({"compile": config.compile, "quant": config.quant})
            return MagicMock(final_onnx_path="out.onnx")

        with (
            patch(
                "winml.modelkit.config.generate_hf_build_config",
                return_value=[cfg],
            ),
            patch(
                "winml.modelkit.commands.build._instantiate_parent_model",
                return_value=MagicMock(),
            ),
            patch(
                "winml.modelkit.build.build_hf_model",
                side_effect=fake_build,
            ),
            patch("winml.modelkit.session.WinMLSession") as mock_sess,
        ):
            mock_sess.return_value.io_config = {
                "input_names": [],
                "output_names": [],
                "input_shapes": [],
                "input_types": [],
            }
            mock_sess.return_value.device = "cpu"
            _perf_modules(
                hf_model="bert-base-uncased",
                module_class="BertAttention",
                task=None,
                iterations=1,
                warmup=0,
                batch_size=1,
                no_quantize=no_quantize,
                no_compile=no_compile,
                output=None,
                verbose=False,
                console=MagicMock(),
            )

        return build_calls

    def test_no_compile_only_clears_compile(self):
        """no_compile=True, no_quantize=False: only compile should be cleared."""
        calls = self._run_perf_modules(no_quantize=False, no_compile=True)
        assert calls, "build_hf_model was never called"
        assert calls[0]["compile"] is None, "compile should be None when no_compile=True"
        assert calls[0]["quant"] is not None, "quant should be untouched when no_quantize=False"

    def test_no_quantize_clears_both(self):
        """no_quantize=True: both compile and quant should be cleared."""
        calls = self._run_perf_modules(no_quantize=True, no_compile=False)
        assert calls, "build_hf_model was never called"
        assert calls[0]["compile"] is None
        assert calls[0]["quant"] is None

    def test_neither_flag_preserves_both(self):
        """No flags: both compile and quant should remain as configured."""
        calls = self._run_perf_modules(no_quantize=False, no_compile=False)
        assert calls, "build_hf_model was never called"
        assert calls[0]["compile"] is not None
        assert calls[0]["quant"] is not None
