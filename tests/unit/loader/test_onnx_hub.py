# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for winml.modelkit.loader.onnx_hub.

Covers the Hub-style ONNX **download** path. Classification and
the combined classify+download wrapper live in
``winml.modelkit.utils.model_input`` and are covered by
``tests/unit/utils/test_model_input.py``.

Uses mock ``hf_hub_download`` callables so no network access is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from winml.modelkit.loader.onnx_hub import (
    _split_hf_onnx_path,
    resolve_hf_onnx_path,
)


if TYPE_CHECKING:
    from pathlib import Path


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


class TestResolveHfOnnxPathDiscovery:
    """``EntryNotFoundError`` on the main file is enriched with a file listing.

    The user typically gets here by guessing the wrong path inside a valid
    Hub repo (e.g. ``onnx/vision_encoder.onnx`` when only ``int8`` and
    ``fp16`` variants exist). The error message must list the ``.onnx``
    files that *are* available so the user can correct the path without
    having to open the Hub web UI.
    """

    def test_missing_file_lists_available_onnx(self) -> None:
        """Wrong filename: error names available .onnx files in the repo."""
        from huggingface_hub.utils import EntryNotFoundError

        def _fake_download(*, repo_id, filename, revision, cache_dir, token):
            # Main file is missing; sidecar should never be reached
            # because the main download raises first.
            raise EntryNotFoundError(filename)

        repo_files = [
            "README.md",
            "config.json",
            "onnx/vision_encoder_int8.onnx",
            "onnx/vision_encoder_fp16.onnx",
            "onnx/prompt_encoder_mask_decoder_int8.onnx",
        ]

        with (
            patch("huggingface_hub.hf_hub_download", side_effect=_fake_download),
            patch(
                "huggingface_hub.list_repo_files",
                return_value=repo_files,
            ) as mock_list,
            pytest.raises(FileNotFoundError) as exc_info,
        ):
            resolve_hf_onnx_path(
                "onnx-community/sam3-tracker-ONNX/onnx/vision_encoder.onnx"
            )

        msg = str(exc_info.value)
        # Names the bad path and the repo
        assert "onnx/vision_encoder.onnx" in msg
        assert "onnx-community/sam3-tracker-ONNX" in msg
        # Lists every .onnx file that *is* present
        assert "onnx/vision_encoder_int8.onnx" in msg
        assert "onnx/vision_encoder_fp16.onnx" in msg
        assert "onnx/prompt_encoder_mask_decoder_int8.onnx" in msg
        # Does not include non-ONNX files
        assert "README.md" not in msg
        assert "config.json" not in msg
        # list_repo_files was called with the repo derived from the bad path
        mock_list.assert_called_once()
        assert (
            mock_list.call_args.args[0] == "onnx-community/sam3-tracker-ONNX"
            or mock_list.call_args.kwargs.get("repo_id")
            == "onnx-community/sam3-tracker-ONNX"
        )

    def test_missing_file_listing_failure_falls_back_gracefully(self) -> None:
        """If list_repo_files itself fails, the error still surfaces cleanly.

        The hint is best-effort -- we must not mask the original
        ``EntryNotFoundError`` because the listing step also failed
        (gated repo, network blip, auth issue).
        """
        from huggingface_hub.utils import EntryNotFoundError

        def _fake_download(*, repo_id, filename, revision, cache_dir, token):
            raise EntryNotFoundError(filename)

        with (
            patch("huggingface_hub.hf_hub_download", side_effect=_fake_download),
            patch(
                "huggingface_hub.list_repo_files",
                side_effect=ConnectionError("network down"),
            ),
            pytest.raises(FileNotFoundError) as exc_info,
        ):
            resolve_hf_onnx_path("org/repo/onnx/missing.onnx")

        msg = str(exc_info.value)
        assert "onnx/missing.onnx" in msg
        assert "org/repo" in msg
        # Generic fallback hint is included.
        assert "Could not list available .onnx files" in msg

    def test_missing_file_no_onnx_in_repo(self) -> None:
        """Repo exists but has no .onnx files at all -- hint says so."""
        from huggingface_hub.utils import EntryNotFoundError

        def _fake_download(*, repo_id, filename, revision, cache_dir, token):
            raise EntryNotFoundError(filename)

        with (
            patch("huggingface_hub.hf_hub_download", side_effect=_fake_download),
            patch(
                "huggingface_hub.list_repo_files",
                return_value=["README.md", "config.json", "pytorch_model.bin"],
            ),
            pytest.raises(FileNotFoundError) as exc_info,
        ):
            resolve_hf_onnx_path("org/pytorch-only/onnx/model.onnx")

        msg = str(exc_info.value)
        assert "No .onnx files were found" in msg
        assert "org/pytorch-only" in msg
