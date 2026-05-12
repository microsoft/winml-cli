# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for winml.modelkit.loader.onnx_hub.

Covers Hub-style ONNX reference detection and download. Uses mock
``hf_hub_download`` callables so no network access is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from winml.modelkit.loader.onnx_hub import (
    _split_hf_onnx_path,
    is_hf_onnx_path,
    maybe_resolve_hf_onnx_path,
    resolve_hf_onnx_path,
)


if TYPE_CHECKING:
    from pathlib import Path


class TestIsHfOnnxPath:
    """Hub ONNX reference detection."""

    def test_three_segment_onnx_recognized(self) -> None:
        """Repo-id + nested file path is a valid Hub ONNX reference."""
        assert is_hf_onnx_path("onnx-community/sam3-tracker-ONNX/onnx/vision_encoder_int8.onnx")

    def test_three_segments_minimum(self) -> None:
        """Two segments are treated as a plain HF model ID, not a file ref."""
        assert is_hf_onnx_path("org/repo/file.onnx")
        assert not is_hf_onnx_path("org/file.onnx")

    def test_plain_hf_model_id_rejected(self) -> None:
        """org/name HF IDs are not Hub ONNX references."""
        assert not is_hf_onnx_path("microsoft/resnet-50")
        assert not is_hf_onnx_path("facebook/sam2.1-hiera-small")

    def test_non_onnx_extension_rejected(self) -> None:
        """Only .onnx file references match."""
        assert not is_hf_onnx_path("org/repo/path/file.bin")
        assert not is_hf_onnx_path("org/repo/path/file")

    def test_existing_local_path_takes_precedence(self, tmp_path: Path) -> None:
        """A real on-disk path that looks like a Hub ref is left alone."""
        local = tmp_path / "org" / "repo" / "file.onnx"
        local.parent.mkdir(parents=True)
        local.write_bytes(b"")
        assert not is_hf_onnx_path(str(local))

    def test_none_and_empty_inputs(self) -> None:
        """None and empty string are not Hub references."""
        assert not is_hf_onnx_path(None)
        assert not is_hf_onnx_path("")


class TestSplitHfOnnxPath:
    """Internal _split_hf_onnx_path helper."""

    def test_three_segments(self) -> None:
        """First two segments form repo_id; third is filename."""
        repo_id, filename = _split_hf_onnx_path("org/repo/file.onnx")
        assert repo_id == "org/repo"
        assert filename == "file.onnx"

    def test_nested_filename_preserved(self) -> None:
        """Multi-segment filenames inside the repo are kept intact."""
        repo_id, filename = _split_hf_onnx_path(
            "onnx-community/sam3-tracker-ONNX/onnx/vision_encoder_int8.onnx"
        )
        assert repo_id == "onnx-community/sam3-tracker-ONNX"
        assert filename == "onnx/vision_encoder_int8.onnx"

    def test_too_few_segments_raises(self) -> None:
        """Inputs with fewer than three segments raise ValueError."""
        with pytest.raises(ValueError, match=r"org/repo/path/to/file\.onnx"):
            _split_hf_onnx_path("org/file.onnx")


class TestResolveHfOnnxPath:
    """Download path: hf_hub_download is called once per file."""

    def test_downloads_onnx_and_attempts_sidecar(self, tmp_path: Path) -> None:
        """Resolver requests both the .onnx file and a .onnx_data sidecar."""
        from huggingface_hub.utils import EntryNotFoundError

        downloaded = tmp_path / "vision_encoder_int8.onnx"
        downloaded.write_bytes(b"")

        calls: list[dict[str, object]] = []

        def _fake_download(*, repo_id, filename, revision, cache_dir, token):
            calls.append(
                {
                    "repo_id": repo_id,
                    "filename": filename,
                    "revision": revision,
                    "cache_dir": cache_dir,
                    "token": token,
                }
            )
            if filename.endswith(".onnx_data"):
                # Most small inline-weight models have no sidecar; the
                # resolver must tolerate the missing file.
                raise EntryNotFoundError(filename)
            return str(downloaded)

        with patch("huggingface_hub.hf_hub_download", side_effect=_fake_download):
            result = resolve_hf_onnx_path(
                "onnx-community/sam3-tracker-ONNX/onnx/vision_encoder_int8.onnx",
                revision="main",
                cache_dir=str(tmp_path / "cache"),
                token=None,
            )

        assert result == downloaded
        assert [c["filename"] for c in calls] == [
            "onnx/vision_encoder_int8.onnx",
            "onnx/vision_encoder_int8.onnx_data",
        ]
        assert calls[0]["repo_id"] == "onnx-community/sam3-tracker-ONNX"

    def test_sidecar_present(self, tmp_path: Path) -> None:
        """When the sidecar exists, both files download successfully."""
        downloaded = tmp_path / "vision_encoder.onnx"
        sidecar = tmp_path / "vision_encoder.onnx_data"
        downloaded.write_bytes(b"")
        sidecar.write_bytes(b"")

        def _fake_download(*, repo_id, filename, revision, cache_dir, token):
            return str(downloaded if filename.endswith(".onnx") else sidecar)

        with patch("huggingface_hub.hf_hub_download", side_effect=_fake_download):
            result = resolve_hf_onnx_path("org/repo/onnx/vision_encoder.onnx")

        assert result == downloaded

    def test_sidecar_oserror_warns_loudly(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Sidecar download OSError is logged at WARNING (not silently ignored).

        Regression: a previous implementation swallowed any
        ``OSError`` (disk full, permission denied, network blip) at
        ``logger.debug`` level. That hid real environmental problems and
        led to confusing failures later when the model loader tried to
        resolve missing external initializers. This test verifies the
        warning is emitted so the user sees something is wrong.
        """
        import logging

        downloaded = tmp_path / "vision_encoder.onnx"
        downloaded.write_bytes(b"")

        def _fake_download(*, repo_id, filename, revision, cache_dir, token):
            if filename.endswith(".onnx_data"):
                raise OSError("disk full")
            return str(downloaded)

        with (
            patch("huggingface_hub.hf_hub_download", side_effect=_fake_download),
            caplog.at_level(logging.WARNING, logger="winml.modelkit.loader.onnx_hub"),
        ):
            result = resolve_hf_onnx_path("org/repo/onnx/vision_encoder.onnx")

        # Main download still succeeds even when the sidecar fails.
        assert result == downloaded
        # Critically: the OSError must surface as a WARNING.
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        warning_messages = [r.getMessage() for r in warning_records]
        assert any("disk full" in m for m in warning_messages), (
            f"Expected a WARNING containing 'disk full'; got {warning_messages}"
        )


class TestMaybeResolveHfOnnxPath:
    """Convenience wrapper that combines is_hf_onnx_path + resolve_hf_onnx_path."""

    def test_none_passes_through(self) -> None:
        """``None`` returns ``None`` without touching the network."""
        with patch("huggingface_hub.hf_hub_download") as mock:
            assert maybe_resolve_hf_onnx_path(None) is None
            mock.assert_not_called()

    def test_plain_hf_model_id_passes_through(self) -> None:
        """An HF model id (e.g. ``microsoft/resnet-50``) is returned unchanged."""
        with patch("huggingface_hub.hf_hub_download") as mock:
            assert maybe_resolve_hf_onnx_path("microsoft/resnet-50") == "microsoft/resnet-50"
            mock.assert_not_called()

    def test_local_path_passes_through(self, tmp_path: Path) -> None:
        """Existing local ``.onnx`` paths take precedence over Hub interpretation."""
        local = tmp_path / "model.onnx"
        local.write_bytes(b"")
        with patch("huggingface_hub.hf_hub_download") as mock:
            assert maybe_resolve_hf_onnx_path(str(local)) == str(local)
            mock.assert_not_called()

    def test_hub_ref_is_resolved(self, tmp_path: Path) -> None:
        """A Hub-style ONNX ref triggers ``resolve_hf_onnx_path``."""
        from huggingface_hub.utils import EntryNotFoundError

        downloaded = tmp_path / "vision_encoder_int8.onnx"
        downloaded.write_bytes(b"")

        def _fake_download(*, repo_id, filename, revision, cache_dir, token):
            if filename.endswith(".onnx_data"):
                raise EntryNotFoundError(filename)
            return str(downloaded)

        with patch("huggingface_hub.hf_hub_download", side_effect=_fake_download):
            result = maybe_resolve_hf_onnx_path(
                "onnx-community/sam3-tracker-ONNX/onnx/vision_encoder_int8.onnx"
            )

        assert result == str(downloaded)
