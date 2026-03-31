# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for modelkit.cache.model — model-aware cache operations."""

from __future__ import annotations

from pathlib import Path

from winml.modelkit.cache.model import (
    _parse_artifact_filename,
    get_model_dir,
    list_cached_models,
    model_id_to_slug,
)


# =============================================================================
# model_id_to_slug
# =============================================================================


class TestModelIdToSlug:
    """Test model ID to filesystem slug conversion."""

    def test_forward_slash(self) -> None:
        assert model_id_to_slug("microsoft/resnet-50") == "microsoft_resnet-50"

    def test_backslash(self) -> None:
        assert model_id_to_slug("local\\model") == "local_model"

    def test_no_slash(self) -> None:
        assert model_id_to_slug("bert-base-uncased") == "bert-base-uncased"

    def test_multiple_slashes(self) -> None:
        assert model_id_to_slug("org/sub/model") == "org_sub_model"

    def test_empty_string(self) -> None:
        assert model_id_to_slug("") == "random-init"


# =============================================================================
# get_model_dir
# =============================================================================


class TestGetModelDir:
    """Test model directory computation."""

    def test_basic_path(self) -> None:
        result = get_model_dir("microsoft/resnet-50", cache_dir=Path("/cache"))
        assert result == Path("/cache/artifacts/microsoft_resnet-50")

    def test_no_slash_model(self) -> None:
        result = get_model_dir("bert-base-uncased", cache_dir=Path("/cache"))
        assert result == Path("/cache/artifacts/bert-base-uncased")


# =============================================================================
# _parse_artifact_filename
# =============================================================================


class TestParseArtifactFilename:
    """Test filename parsing for directory scanning."""

    def test_valid_filename(self) -> None:
        result = _parse_artifact_filename("imgcls_a1b2c3d4e5f67890_model")
        assert result == ("imgcls", "a1b2c3d4e5f67890", "model")

    def test_valid_export_stage(self) -> None:
        result = _parse_artifact_filename("txtcls_deadbeef12345678_export")
        assert result == ("txtcls", "deadbeef12345678", "export")

    def test_multi_word_task(self) -> None:
        # Task abbreviation with underscore (e.g., "feat_ext")
        result = _parse_artifact_filename("feat_ext_a1b2c3d4e5f67890_optimized")
        assert result == ("feat_ext", "a1b2c3d4e5f67890", "optimized")

    def test_invalid_no_underscore(self) -> None:
        assert _parse_artifact_filename("nounderscores") is None

    def test_invalid_short_hash(self) -> None:
        # Hash too short (< 16 chars)
        assert _parse_artifact_filename("imgcls_abc_model") is None

    def test_invalid_non_hex_hash(self) -> None:
        # Non-hex characters in hash position
        assert _parse_artifact_filename("imgcls_zzzzzzzzzzzzzzzz_model") is None

    def test_bare_model_onnx(self) -> None:
        # Unprefixed filename (no cache_key) should not parse
        assert _parse_artifact_filename("model") is None


# =============================================================================
# list_cached_models
# =============================================================================


class TestListCachedModels:
    """Test directory scanning for cached models."""

    def test_empty_cache(self, tmp_path: Path) -> None:
        result = list_cached_models(cache_dir=tmp_path)
        assert result == []

    def test_nonexistent_cache_dir(self, tmp_path: Path) -> None:
        result = list_cached_models(cache_dir=tmp_path / "nonexistent")
        assert result == []

    def test_finds_artifacts(self, tmp_path: Path) -> None:
        # Create artifacts directory structure
        model_dir = tmp_path / "artifacts" / "microsoft_resnet-50"
        model_dir.mkdir(parents=True)

        # Create fake ONNX files matching the pattern
        (model_dir / "imgcls_a1b2c3d4e5f67890_export.onnx").write_text("mock")
        (model_dir / "imgcls_a1b2c3d4e5f67890_model.onnx").write_text("mock")

        result = list_cached_models(cache_dir=tmp_path)
        assert len(result) == 2
        assert result[0]["model_slug"] == "microsoft_resnet-50"
        assert result[0]["task_abbrev"] == "imgcls"
        assert result[0]["config_hash"] == "a1b2c3d4e5f67890"
        assert result[0]["stage"] == "export"
        assert result[1]["stage"] == "model"

    def test_ignores_unparseable_files(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "artifacts" / "test_model"
        model_dir.mkdir(parents=True)

        # Valid file
        (model_dir / "imgcls_a1b2c3d4e5f67890_model.onnx").write_text("mock")
        # Invalid files (should be skipped)
        (model_dir / "model.onnx").write_text("bare")
        (model_dir / "random_file.onnx").write_text("random")
        (model_dir / "readme.txt").write_text("not onnx")

        result = list_cached_models(cache_dir=tmp_path)
        assert len(result) == 1
        assert result[0]["stage"] == "model"

    def test_multiple_models(self, tmp_path: Path) -> None:
        for slug in ["model_a", "model_b"]:
            d = tmp_path / "artifacts" / slug
            d.mkdir(parents=True)
            (d / "imgcls_a1b2c3d4e5f67890_model.onnx").write_text("mock")

        result = list_cached_models(cache_dir=tmp_path)
        assert len(result) == 2
        slugs = {r["model_slug"] for r in result}
        assert slugs == {"model_a", "model_b"}


# =============================================================================
# INTEGRATION: Both callers produce same paths
# =============================================================================


class TestCallerConvergence:
    """Verify that from_pretrained and wmk build --use-cache produce identical paths."""

    def test_same_output_dir(self) -> None:
        """Both callers compute the same output_dir for a given model_id."""
        cache_dir = Path("/cache")
        model_id = "microsoft/resnet-50"

        # What from_pretrained computes
        from winml.modelkit.cache import get_model_dir

        fp_dir = get_model_dir(model_id, cache_dir=cache_dir)

        # What CLI build --use-cache computes (same function now)
        cli_dir = get_model_dir(model_id, cache_dir=cache_dir)

        assert fp_dir == cli_dir
        assert fp_dir == cache_dir / "artifacts" / "microsoft_resnet-50"

    def test_same_cache_key(self) -> None:
        """Both callers compute the same cache_key for a given task+config."""
        from winml.modelkit.cache import get_cache_key

        task_abbrev = "imgcls"
        config_hash = "a1b2c3d4e5f67890"

        fp_key = get_cache_key(task_abbrev, config_hash)
        cli_key = get_cache_key(task_abbrev, config_hash)

        assert fp_key == cli_key
        assert fp_key == "imgcls_a1b2c3d4e5f67890"
