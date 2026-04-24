# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for winml.modelkit.commands.eval._resolve_model_path."""

from __future__ import annotations

import click
import pytest

from winml.modelkit.commands.eval import _resolve_model_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def onnx_file(tmp_path):
    """Create a placeholder .onnx file on disk."""
    f = tmp_path / "model.onnx"
    f.write_bytes(b"")
    return f


@pytest.fixture
def onnx_vision(tmp_path):
    f = tmp_path / "vision.onnx"
    f.write_bytes(b"")
    return f


@pytest.fixture
def onnx_text(tmp_path):
    f = tmp_path / "text.onnx"
    f.write_bytes(b"")
    return f


# ---------------------------------------------------------------------------
# Empty -m
# ---------------------------------------------------------------------------


class TestEmptyModel:
    def test_no_model_no_id_raises(self):
        with pytest.raises(click.UsageError, match="model is required"):
            _resolve_model_path(model=(), model_id=None)

    def test_model_id_only(self):
        path, mid = _resolve_model_path(model=(), model_id="openai/clip-vit-base-patch32")
        assert path is None
        assert mid == "openai/clip-vit-base-patch32"


# ---------------------------------------------------------------------------
# Single plain -m (HF ID or .onnx file)
# ---------------------------------------------------------------------------


class TestSinglePlain:
    def test_plain_hf_id_no_model_id(self):
        """-m <hf_id> populates model_id when --model-id omitted."""
        path, mid = _resolve_model_path(model=("microsoft/resnet-50",), model_id=None)
        assert path is None
        assert mid == "microsoft/resnet-50"

    def test_plain_hf_id_explicit_model_id_wins(self):
        """Explicit --model-id takes precedence over an HF-ID-shaped -m."""
        path, mid = _resolve_model_path(
            model=("microsoft/resnet-50",), model_id="Intel/bert-base-uncased-mrpc",
        )
        assert path is None
        assert mid == "Intel/bert-base-uncased-mrpc"

    def test_plain_onnx_with_model_id(self, onnx_file):
        path, mid = _resolve_model_path(
            model=(str(onnx_file),), model_id="microsoft/resnet-50",
        )
        assert path == str(onnx_file)
        assert mid == "microsoft/resnet-50"

    def test_plain_onnx_without_model_id_raises(self, onnx_file):
        with pytest.raises(click.UsageError, match="--model-id is required"):
            _resolve_model_path(model=(str(onnx_file),), model_id=None)

    def test_plain_onnx_missing_file_raises(self, tmp_path):
        missing = tmp_path / "does-not-exist.onnx"
        with pytest.raises(click.BadParameter, match="ONNX file not found"):
            _resolve_model_path(model=(str(missing),), model_id="some/id")

    def test_multiple_plain_raises(self, onnx_file):
        """Multiple plain -m values without role=path are ambiguous."""
        with pytest.raises(click.UsageError, match="role=path"):
            _resolve_model_path(
                model=(str(onnx_file), str(onnx_file)), model_id="some/id",
            )


# ---------------------------------------------------------------------------
# Composite -m role=path
# ---------------------------------------------------------------------------


class TestComposite:
    def test_two_roles(self, onnx_vision, onnx_text):
        path, mid = _resolve_model_path(
            model=(
                f"image-encoder={onnx_vision}",
                f"text-encoder={onnx_text}",
            ),
            model_id="openai/clip-vit-base-patch32",
        )
        assert path == {
            "image-encoder": str(onnx_vision),
            "text-encoder": str(onnx_text),
        }
        assert mid == "openai/clip-vit-base-patch32"

    def test_composite_requires_model_id(self, onnx_vision, onnx_text):
        with pytest.raises(click.UsageError, match="--model-id is required"):
            _resolve_model_path(
                model=(
                    f"image-encoder={onnx_vision}",
                    f"text-encoder={onnx_text}",
                ),
                model_id=None,
            )

    def test_duplicate_roles_raise(self, onnx_vision, onnx_text):
        with pytest.raises(click.BadParameter, match="Duplicate role"):
            _resolve_model_path(
                model=(
                    f"image-encoder={onnx_vision}",
                    f"image-encoder={onnx_text}",
                ),
                model_id="some/id",
            )

    def test_missing_path_raises(self, onnx_vision, tmp_path):
        missing = tmp_path / "no.onnx"
        with pytest.raises(click.BadParameter, match="ONNX file not found"):
            _resolve_model_path(
                model=(
                    f"image-encoder={onnx_vision}",
                    f"text-encoder={missing}",
                ),
                model_id="some/id",
            )

    def test_empty_role_raises(self, onnx_vision):
        with pytest.raises(click.BadParameter, match="role and path"):
            _resolve_model_path(
                model=(f"={onnx_vision}",), model_id="some/id",
            )

    def test_empty_path_raises(self):
        with pytest.raises(click.BadParameter, match="role and path"):
            _resolve_model_path(
                model=("image-encoder=",), model_id="some/id",
            )

    def test_whitespace_stripped(self, onnx_vision):
        """Role and path are trimmed of surrounding whitespace."""
        path, _mid = _resolve_model_path(
            model=(f"  image-encoder  =  {onnx_vision}  ",),
            model_id="some/id",
        )
        assert path == {"image-encoder": str(onnx_vision)}


# ---------------------------------------------------------------------------
# Mixing forms
# ---------------------------------------------------------------------------


class TestMixedForms:
    def test_plain_and_role_path_mixed_raises(self, onnx_file, onnx_vision):
        with pytest.raises(click.UsageError, match="Cannot mix"):
            _resolve_model_path(
                model=(str(onnx_file), f"text-encoder={onnx_vision}"),
                model_id="some/id",
            )
