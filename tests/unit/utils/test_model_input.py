# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for winml.modelkit.utils.model_input.

Covers the single ``-m/--model`` value classifier (:func:`classify_model_input`)
and the classify+download resolver (:func:`resolve_model_input`) that
together replace the previous trio of detectors (``is_hub_model``,
``is_hf_onnx_path``, ``is_onnx_file_path``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from winml.modelkit.utils.model_input import (
    ModelInputKind,
    classify_model_input,
    resolve_model_input,
)


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# classify_model_input: hub_onnx
# ---------------------------------------------------------------------------


class TestClassifyHubOnnx:
    """``org/repo/.../file.onnx`` -> ``hub_onnx``."""

    def test_three_segment_onnx_recognized(self) -> None:
        """Repo-id + nested file path is a valid Hub ONNX reference."""
        mi = classify_model_input("onnx-community/sam3-tracker-ONNX/onnx/vision_encoder_int8.onnx")
        assert mi.kind == "hub_onnx"
        assert mi.hf_id == "onnx-community/sam3-tracker-ONNX"
        assert mi.local_path is None  # not downloaded by classify

    def test_three_segments_minimum(self) -> None:
        """Two segments are too few for a Hub ONNX reference."""
        assert classify_model_input("org/repo/file.onnx").kind == "hub_onnx"
        # Two segments ending in .onnx is invalid (not a Hub ref, not a
        # plausible HF id either).
        assert classify_model_input("org/file.onnx").kind == "invalid"

    def test_uppercase_onnx_extension_accepted(self) -> None:
        """Case-insensitive ``.onnx`` matches the rest of the CLI."""
        assert classify_model_input("org/repo/path/file.ONNX").kind == "hub_onnx"
        assert classify_model_input("org/repo/path/file.OnNx").kind == "hub_onnx"


# ---------------------------------------------------------------------------
# classify_model_input: hf_id
# ---------------------------------------------------------------------------


class TestClassifyHfId:
    """``org/name`` (no .onnx suffix) -> ``hf_id``."""

    def test_plain_hf_model_id(self) -> None:
        mi = classify_model_input("microsoft/resnet-50")
        assert mi.kind == "hf_id"
        assert mi.hf_id == "microsoft/resnet-50"
        assert mi.local_path is None

    def test_single_segment_hf_id(self) -> None:
        """Single-segment IDs (e.g. ``bert-base-uncased``) are still hf_id."""
        mi = classify_model_input("bert-base-uncased")
        assert mi.kind == "hf_id"
        assert mi.hf_id == "bert-base-uncased"

    def test_three_segment_non_onnx_is_invalid(self) -> None:
        """A non-.onnx three-plus-segment string is a path that doesn't exist.

        Two-plus ``/`` components look like a filesystem path rather than a
        HuggingFace id (which is at most ``org/name``), so the unified
        classifier reports it as invalid with a friendly message instead of
        forwarding a doomed ``org/repo/path`` id to the Hub.
        """
        mi = classify_model_input("org/repo/path/file.bin")
        assert mi.kind == "invalid"
        assert mi.error is not None
        mi2 = classify_model_input("org/repo/path/file")
        assert mi2.kind == "invalid"
        assert mi2.error is not None


# ---------------------------------------------------------------------------
# classify_model_input: local_onnx / build_dir
# ---------------------------------------------------------------------------


class TestClassifyLocal:
    """Local-path branch: existing files/dirs + ``./``/``../``/``~/``/abs prefixes."""

    def test_existing_local_onnx(self, tmp_path: Path) -> None:
        """An existing on-disk .onnx file is classified as local_onnx."""
        local = tmp_path / "org" / "repo" / "file.onnx"
        local.parent.mkdir(parents=True)
        local.write_bytes(b"")
        mi = classify_model_input(str(local))
        assert mi.kind == "local_onnx"
        assert mi.local_path == str(local)
        assert mi.hf_id is None

    def test_existing_build_dir(self, tmp_path: Path) -> None:
        """An existing directory is classified as build_dir."""
        d = tmp_path / "build_out"
        d.mkdir()
        mi = classify_model_input(str(d))
        assert mi.kind == "build_dir"
        assert mi.local_path == str(d)

    def test_relative_path_prefixes_rejected_as_hub(self) -> None:
        """``./``, ``../``, ``~/`` prefixes block Hub interpretation."""
        # These strings all have three slash-separated segments and end in
        # .onnx, so without local-path rejection they would be misclassified
        # as Hub references.
        for value in (
            "./org/repo/file.onnx",
            "../org/repo/file.onnx",
            "~/org/repo/file.onnx",
        ):
            mi = classify_model_input(value)
            # .onnx suffix + local-path prefix => local_onnx (download
            # attempt would fail later, but classification is correct)
            assert mi.kind == "local_onnx", f"expected local_onnx for {value!r}, got {mi}"

    def test_unix_absolute_path_rejected_as_hub(self) -> None:
        """Unix-style absolute paths are treated as local even without an existing file."""
        mi = classify_model_input("/tmp/org/repo/file.onnx")  # noqa: S108 - fake path is not a real tempfile
        assert mi.kind == "local_onnx"

    def test_windows_absolute_path_rejected_as_hub(self) -> None:
        """Windows drive-letter absolute paths are treated as local."""
        # Both backslash and forward-slash separators after the drive
        # letter are common on Windows; both must be rejected.
        for value in (
            r"C:\models\org\repo\file.onnx",
            "C:/models/org/repo/file.onnx",
            r"D:\org\repo\file.onnx",
        ):
            mi = classify_model_input(value)
            assert mi.kind == "local_onnx", f"expected local_onnx for {value!r}, got {mi}"


# ---------------------------------------------------------------------------
# classify_model_input: invalid / edge
# ---------------------------------------------------------------------------


class TestClassifyEdge:
    """Empty / unparsable inputs."""

    def test_empty_string(self) -> None:
        mi = classify_model_input("")
        assert mi.kind == "invalid"
        assert mi.raw == ""

    def test_raw_preserved(self) -> None:
        """`raw` always echoes the original input regardless of kind."""
        for value in (
            "microsoft/resnet-50",
            "org/repo/path/file.onnx",
            "./model.onnx",
        ):
            assert classify_model_input(value).raw == value


# ---------------------------------------------------------------------------
# classify_model_input: folder metadata
# ---------------------------------------------------------------------------


class TestClassifyFolderMetadata:
    """``build_dir`` results carry folder-shape flags for CLI call sites."""

    def test_hf_source_folder(self, tmp_path: Path) -> None:
        """config.json but no onnx -> HF source folder."""
        (tmp_path / "config.json").write_text("{}")
        mi = classify_model_input(str(tmp_path))
        assert mi.kind == "build_dir"
        assert mi.is_hf_folder
        assert not mi.folder_has_onnx
        assert not mi.is_winml_cli_folder

    def test_loose_onnx_folder(self, tmp_path: Path) -> None:
        """onnx but no manifest -> loadable dir, not a winml build output."""
        (tmp_path / "model.onnx").write_bytes(b"\x00")
        mi = classify_model_input(str(tmp_path))
        assert mi.kind == "build_dir"
        assert mi.folder_has_onnx
        assert not mi.is_winml_cli_folder
        assert not mi.is_hf_folder

    def test_winml_build_folder(self, tmp_path: Path) -> None:
        """onnx + build_manifest.json -> winml cli build output."""
        (tmp_path / "model.onnx").write_bytes(b"\x00")
        (tmp_path / "build_manifest.json").write_text("{}")
        mi = classify_model_input(str(tmp_path))
        assert mi.folder_has_onnx
        assert mi.is_winml_cli_folder

    def test_winml_build_folder_cache_prefixed(self, tmp_path: Path) -> None:
        """Cache-key-prefixed artifact names are recognized too."""
        (tmp_path / "abc123_model.onnx").write_bytes(b"\x00")
        (tmp_path / "abc123_build_manifest.json").write_text("{}")
        mi = classify_model_input(str(tmp_path))
        assert mi.folder_has_onnx
        assert mi.is_winml_cli_folder

    def test_manifest_without_onnx_is_not_winml_folder(self, tmp_path: Path) -> None:
        """A manifest with no onnx (partial build) is not a winml build dir."""
        (tmp_path / "build_manifest.json").write_text("{}")
        mi = classify_model_input(str(tmp_path))
        assert not mi.folder_has_onnx
        assert not mi.is_winml_cli_folder


# ---------------------------------------------------------------------------
# classify_model_input: error messages (pure, never raises)
# ---------------------------------------------------------------------------


class TestClassifyErrors:
    """Invalid inputs return kind=invalid with an actionable ``error`` message."""

    def test_empty_is_invalid_with_message(self) -> None:
        mi = classify_model_input("")
        assert mi.kind == ModelInputKind.INVALID
        assert mi.error == "Model input cannot be empty."

    def test_missing_onnx_file(self) -> None:
        mi = classify_model_input("does_not_exist.onnx")
        assert mi.kind == ModelInputKind.INVALID
        assert mi.error is not None
        assert "ONNX file not found" in mi.error

    def test_unsupported_local_file(self, tmp_path: Path) -> None:
        other = tmp_path / "weights.safetensors"
        other.write_bytes(b"\x00")
        mi = classify_model_input(str(other))
        assert mi.kind == ModelInputKind.INVALID
        assert mi.error is not None
        assert "Unsupported model file" in mi.error

    def test_missing_path_shaped_value(self) -> None:
        mi = classify_model_input("./nope/model_dir")
        assert mi.kind == ModelInputKind.INVALID
        assert mi.error is not None
        assert "does not exist" in mi.error

    def test_invalid_hf_id(self) -> None:
        mi = classify_model_input("has spaces")
        assert mi.kind == ModelInputKind.INVALID
        assert mi.error is not None
        assert "not a valid HuggingFace" in mi.error

    def test_valid_kinds_have_no_error(self, tmp_path: Path) -> None:
        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"\x00")
        for value in ("microsoft/resnet-50", "org/repo/path/x.onnx", str(onnx)):
            assert classify_model_input(value).error is None


# ---------------------------------------------------------------------------
# ModelInputKind: dual string / member semantics
# ---------------------------------------------------------------------------


class TestModelInputKind:
    """The enum members double as their string values for both call styles."""

    def test_member_equals_string_value(self) -> None:
        assert ModelInputKind.HUB_ONNX == "hub_onnx"
        assert ModelInputKind.ONNX_FILE == "local_onnx"

    def test_identity_check_against_member(self) -> None:
        mi = classify_model_input("bert-base-uncased")
        assert mi.kind is ModelInputKind.HF_ID

    def test_str_renders_bare_value(self) -> None:
        assert str(ModelInputKind.HF_ID) == "hf_id"


# ---------------------------------------------------------------------------
# resolve_model_input: pass-through + download
# ---------------------------------------------------------------------------


class TestResolveModelInput:
    """``resolve_model_input`` == classify + download for hub_onnx only."""

    def test_hf_id_pass_through_no_network(self) -> None:
        """``microsoft/resnet-50`` returns unchanged; no Hub download attempted."""
        with patch("huggingface_hub.hf_hub_download") as mock:
            mi = resolve_model_input("microsoft/resnet-50")
            assert mi.kind == "hf_id"
            assert mi.local_path is None
            mock.assert_not_called()

    def test_local_onnx_pass_through_no_network(self, tmp_path: Path) -> None:
        """Existing local ``.onnx`` paths take precedence over Hub interpretation."""
        local = tmp_path / "model.onnx"
        local.write_bytes(b"")
        with patch("huggingface_hub.hf_hub_download") as mock:
            mi = resolve_model_input(str(local))
            assert mi.kind == "local_onnx"
            assert mi.local_path == str(local)
            mock.assert_not_called()

    def test_hub_ref_is_downloaded(self, tmp_path: Path) -> None:
        """A Hub-style ONNX ref triggers ``resolve_hf_onnx_path`` and populates local_path."""
        from huggingface_hub.utils import EntryNotFoundError

        downloaded = tmp_path / "vision_encoder_int8.onnx"
        downloaded.write_bytes(b"")

        def _fake_download(*, repo_id, filename, revision, cache_dir, token):
            if filename.endswith(".onnx_data"):
                raise EntryNotFoundError(filename)
            return str(downloaded)

        with patch("huggingface_hub.hf_hub_download", side_effect=_fake_download):
            mi = resolve_model_input(
                "onnx-community/sam3-tracker-ONNX/onnx/vision_encoder_int8.onnx"
            )

        assert mi.kind == "hub_onnx"
        assert mi.local_path == str(downloaded)
        assert mi.hf_id == "onnx-community/sam3-tracker-ONNX"
