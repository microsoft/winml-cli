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
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from winml.modelkit.compiler import compile_multiple_onnx


if TYPE_CHECKING:
    import pytest


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
