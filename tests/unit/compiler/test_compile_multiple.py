# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for ``compile_multiple_onnx`` output-name handling.

``Compiler`` is mocked so these exercise the per-model output naming / de-dup
logic only — no real compilation or EP runtime is needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.compiler import compile_multiple_onnx


def _output_names(mock_compiler_cls: MagicMock) -> list[str]:
    """Filenames passed as ``output_path`` to each ``Compiler.compile`` call, in order."""
    calls = mock_compiler_cls.return_value.compile.call_args_list
    return [Path(c.kwargs["output_path"]).name for c in calls]


class TestCompileMultipleNaming:
    @patch("winml.modelkit.compiler.compiler.Compiler")
    def test_duplicate_names_suffixed_with_warning(
        self, mock_compiler_cls: MagicMock, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two inputs with the same filename: the later one gets an ``_1`` suffix and warns."""
        m1 = tmp_path / "a" / "model.onnx"
        m2 = tmp_path / "b" / "model.onnx"
        out_dir = tmp_path / "out"
        mock_compiler_cls.return_value.compile.return_value = MagicMock(success=True)

        with caplog.at_level(logging.WARNING):
            results = compile_multiple_onnx([m1, m2], out_dir)

        assert len(results) == 2
        names = _output_names(mock_compiler_cls)
        assert names == ["model_ctx.onnx", "model_1_ctx.onnx"]
        # Both land in the requested output directory.
        for c in mock_compiler_cls.return_value.compile.call_args_list:
            assert Path(c.kwargs["output_path"]).parent == out_dir
        assert "repeats" in caplog.text

    @patch("winml.modelkit.compiler.compiler.Compiler")
    def test_triple_duplicate_names_increment(
        self, mock_compiler_cls: MagicMock, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Three same-named inputs increment the suffix: _ , _1, _2."""
        models = [tmp_path / d / "m.onnx" for d in ("a", "b", "c")]
        mock_compiler_cls.return_value.compile.return_value = MagicMock(success=True)

        with caplog.at_level(logging.WARNING):
            compile_multiple_onnx(models, tmp_path / "out")

        assert _output_names(mock_compiler_cls) == [
            "m_ctx.onnx",
            "m_1_ctx.onnx",
            "m_2_ctx.onnx",
        ]
        assert caplog.text.count("repeats") == 2

    @patch("winml.modelkit.compiler.compiler.Compiler")
    def test_unique_names_no_suffix_no_warning(
        self, mock_compiler_cls: MagicMock, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Distinct filenames keep their stems and emit no warning."""
        m1 = tmp_path / "a.onnx"
        m2 = tmp_path / "b.onnx"
        mock_compiler_cls.return_value.compile.return_value = MagicMock(success=True)

        with caplog.at_level(logging.WARNING):
            compile_multiple_onnx([m1, m2], tmp_path / "out")

        assert _output_names(mock_compiler_cls) == ["a_ctx.onnx", "b_ctx.onnx"]
        assert "repeats" not in caplog.text


def _single_output(mock_compiler_cls: MagicMock) -> Path | None:
    """The ``output_path`` passed to the single ``Compiler.compile`` call."""
    out = mock_compiler_cls.return_value.compile.call_args.kwargs["output_path"]
    return Path(out) if out is not None else None


class TestCompileMultipleOutputPath:
    @patch("winml.modelkit.compiler.compiler.Compiler")
    def test_multiple_models_require_directory(
        self, mock_compiler_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Multiple models with a file output_path (has a suffix) is rejected."""
        m1 = tmp_path / "a" / "m.onnx"
        m2 = tmp_path / "b" / "m.onnx"
        with pytest.raises(AssertionError, match="must be a directory"):
            compile_multiple_onnx([m1, m2], tmp_path / "out.onnx")

    @patch("winml.modelkit.compiler.compiler.Compiler")
    def test_multiple_models_reject_none_output(
        self, mock_compiler_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Multiple models with no output_path is rejected (would break shared context)."""
        m1 = tmp_path / "a" / "m.onnx"
        m2 = tmp_path / "b" / "m.onnx"
        with pytest.raises(AssertionError, match="must be a directory"):
            compile_multiple_onnx([m1, m2], None)

    @patch("winml.modelkit.compiler.compiler.Compiler")
    def test_single_model_file_output_path(
        self, mock_compiler_cls: MagicMock, tmp_path: Path
    ) -> None:
        """A single model accepts a file output_path and writes exactly there."""
        mock_compiler_cls.return_value.compile.return_value = MagicMock(success=True)
        out_file = tmp_path / "custom_name.onnx"
        compile_multiple_onnx([tmp_path / "model.onnx"], out_file)
        assert _single_output(mock_compiler_cls) == out_file

    @patch("winml.modelkit.compiler.compiler.Compiler")
    def test_single_model_dir_output_path(
        self, mock_compiler_cls: MagicMock, tmp_path: Path
    ) -> None:
        """A single model with a directory output_path writes <stem>_ctx.onnx into it."""
        mock_compiler_cls.return_value.compile.return_value = MagicMock(success=True)
        out_dir = tmp_path / "out"
        compile_multiple_onnx([tmp_path / "model.onnx"], out_dir)
        assert _single_output(mock_compiler_cls) == out_dir / "model_ctx.onnx"
