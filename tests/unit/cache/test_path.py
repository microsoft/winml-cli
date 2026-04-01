# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for modelkit.cache.path — pure path primitives."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import pytest

from winml.modelkit.cache.path import (
    get_artifact_path,
    get_artifacts_dir,
    get_cache_dir,
    get_cache_key,
)


# =============================================================================
# get_cache_dir
# =============================================================================


class TestGetCacheDir:
    """Test cache directory resolution."""

    def test_default_is_home_cache_winml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WINML_CACHE_DIR", raising=False)
        result = get_cache_dir()
        assert result == Path.home() / ".cache" / "winml"

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WINML_CACHE_DIR", "/custom/cache")
        result = get_cache_dir()
        assert result == Path("/custom/cache")

    def test_explicit_override_takes_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WINML_CACHE_DIR", "/env/cache")
        result = get_cache_dir(override="/explicit/cache")
        assert result == Path("/explicit/cache")

    def test_explicit_override_as_path(self) -> None:
        result = get_cache_dir(override=Path("/some/path"))
        assert result == Path("/some/path")

    def test_none_override_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WINML_CACHE_DIR", raising=False)
        result = get_cache_dir(override=None)
        assert result == Path.home() / ".cache" / "winml"


# =============================================================================
# get_artifacts_dir
# =============================================================================


class TestGetArtifactsDir:
    """Test artifacts directory computation."""

    def test_appends_artifacts(self) -> None:
        result = get_artifacts_dir(Path("/cache/root"))
        assert result == Path("/cache/root/artifacts")

    def test_none_resolves_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WINML_CACHE_DIR", raising=False)
        result = get_artifacts_dir()
        assert result == Path.home() / ".cache" / "winml" / "artifacts"


# =============================================================================
# get_cache_key
# =============================================================================


class TestGetCacheKey:
    """Test cache key assembly."""

    def test_basic_format(self) -> None:
        result = get_cache_key("imgcls", "a1b2c3d4e5f67890")
        assert result == "imgcls_a1b2c3d4e5f67890"

    def test_with_different_task(self) -> None:
        result = get_cache_key("txtcls", "deadbeef12345678")
        assert result == "txtcls_deadbeef12345678"


# =============================================================================
# get_artifact_path
# =============================================================================


class TestGetArtifactPath:
    """Test artifact path computation."""

    def test_basic_path(self) -> None:
        model_dir = Path("/cache/artifacts/microsoft_resnet-50")
        result = get_artifact_path(model_dir, "imgcls_abc123", "model")
        assert result == model_dir / "imgcls_abc123_model.onnx"

    def test_custom_extension(self) -> None:
        model_dir = Path("/cache/artifacts/bert")
        result = get_artifact_path(model_dir, "nsp_def456", "config", ext=".json")
        assert result == model_dir / "nsp_def456_config.json"

    def test_different_stages(self) -> None:
        model_dir = Path("/out")
        key = "imgcls_hash"
        assert get_artifact_path(model_dir, key, "export").name == "imgcls_hash_export.onnx"
        assert get_artifact_path(model_dir, key, "optimized").name == "imgcls_hash_optimized.onnx"
        assert get_artifact_path(model_dir, key, "model").name == "imgcls_hash_model.onnx"
