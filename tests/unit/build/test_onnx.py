# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for build_onnx_model() — mock-based, no network, no actual ONNX models.

Tests the ONNX build pipeline (optimize -> quantize -> compile) independently
of HuggingFace loading and export.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.build.hf import BuildResult
from winml.modelkit.build.onnx import build_onnx_model


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_onnx_config():
    """Create a minimal WinMLBuildConfig for ONNX builds (export=None)."""
    from winml.modelkit.config import WinMLBuildConfig

    return WinMLBuildConfig.from_dict(
        {
            "loader": {"task": "image-classification"},
            "export": None,
            "optim": {},
            "quant": {
                "mode": "qdq",
                "samples": 10,
                "task": "image-classification",
                "model_id": "test-model",
            },
            "compile": {
                "execution_provider": "qnn",
            },
        }
    )


@pytest.fixture
def sample_onnx_config_minimal():
    """Config with quant=None, compile=None for minimal pipeline."""
    from winml.modelkit.config import WinMLBuildConfig

    return WinMLBuildConfig.from_dict(
        {
            "loader": {},
            "export": None,
            "optim": {},
            "quant": None,
            "compile": None,
        }
    )


@pytest.fixture
def fake_onnx(tmp_path: Path) -> Path:
    """Create a fake ONNX file in tmp_path."""
    onnx_file = tmp_path / "input_model.onnx"
    onnx_file.write_text("fake-onnx")
    return onnx_file


def _create_file_side_effect(output_kwarg_name: str, return_value: object = None):
    """Create a side_effect that writes a mock file at the output path.

    build_onnx_model() uses copy_onnx_model() at the end, so every stage
    must actually create a file at its output path.
    """

    def side_effect(*args: object, **kwargs: object) -> object:
        path = kwargs.get(output_kwarg_name)
        if path is not None:
            Path(path).write_text("mock")
        return return_value

    return side_effect


def _default_analyze_result():
    """Build a default AnalyzeResult with no opportunities (analyzer converges)."""
    from winml.modelkit.analyze import AnalyzeResult, LintResult
    from winml.modelkit.optim import WinMLOptimizationConfig

    config = WinMLOptimizationConfig()
    lint = LintResult(
        errors=0,
        warnings=0,
        info=0,
        passed=True,
        error_patterns=[],
        warning_patterns=[],
        information=[],
        optimization_config=config,
    )
    return AnalyzeResult(lint=lint, optimization_config=config)


@pytest.fixture
def mock_onnx_pipeline():
    """Mock all pipeline stage functions for ONNX builds.

    Mocks optimize_onnx, analyze_onnx (via common.py), quantize_onnx,
    compile_onnx, copy_onnx_model, and is_quantized_onnx.
    """
    quant_result = MagicMock()
    quant_result.success = True
    quant_result.errors = []
    quant_result.nodes_quantized = 42
    quant_result.nodes_skipped = 3
    quant_result.calibration_time_seconds = 1.5
    quant_result.qdq_insertion_time_seconds = 0.8

    compile_result = MagicMock()
    compile_result.output_path = None
    compile_result.success = True

    with (
        patch(
            "winml.modelkit.build.common.optimize_onnx",
            side_effect=_create_file_side_effect("output"),
        ) as m_optimize,
        patch(
            "winml.modelkit.build.common.analyze_onnx",
            return_value=_default_analyze_result(),
        ) as m_analyze,
        patch(
            "winml.modelkit.build.onnx.quantize_onnx",
            side_effect=_create_file_side_effect("output_path", quant_result),
        ) as m_quantize,
        patch(
            "winml.modelkit.build.onnx.compile_onnx",
            side_effect=_create_file_side_effect("output_path", compile_result),
        ) as m_compile,
        patch(
            "winml.modelkit.build.onnx.is_quantized_onnx",
            return_value=False,
        ) as m_has_qdq,
        patch(
            "winml.modelkit.build.onnx.copy_onnx_model",
            side_effect=lambda src, dst: Path(dst).write_text("mock"),
        ) as m_copy,
        patch(
            "winml.modelkit.build.common.copy_onnx_model",
            side_effect=lambda src, dst: Path(dst).write_text("mock"),
        ) as m_copy_common,
    ):
        yield {
            "optimize": m_optimize,
            "analyze": m_analyze,
            "quantize": m_quantize,
            "compile": m_compile,
            "is_quantized_onnx": m_has_qdq,
            "copy": m_copy,
            "copy_common": m_copy_common,
            "quant_result": quant_result,
            "compile_result": compile_result,
        }


# =============================================================================
# BASIC PIPELINE TESTS
# =============================================================================


class TestBuildOnnxModelBasic:
    """Test build_onnx_model() pipeline basics."""

    def test_build_onnx_model_basic(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config, mock_onnx_pipeline
    ) -> None:
        """Full pipeline runs and produces a BuildResult."""
        output_dir = tmp_path / "output"
        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config,
            output_dir=output_dir,
        )
        assert isinstance(result, BuildResult)
        assert result.output_dir == output_dir
        assert result.final_onnx_path == output_dir / "model.onnx"
        assert result.reused is False
        assert result.elapsed >= 0

    def test_creates_output_dir(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config, mock_onnx_pipeline
    ) -> None:
        """Output directory is created if it doesn't exist."""
        output_dir = tmp_path / "new_dir"
        assert not output_dir.exists()

        build_onnx_model(fake_onnx, config=sample_onnx_config, output_dir=output_dir)
        assert output_dir.is_dir()

    def test_all_stages_completed(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config, mock_onnx_pipeline
    ) -> None:
        """All stages appear in stages_completed."""
        output_dir = tmp_path / "output"
        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config,
            output_dir=output_dir,
        )
        assert "optimize" in result.stages_completed
        assert "quantize" in result.stages_completed
        assert "compile" in result.stages_completed
        assert result.stages_skipped == []

    def test_stage_timings_populated(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config, mock_onnx_pipeline
    ) -> None:
        """Stage timings are recorded."""
        output_dir = tmp_path / "output"
        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config,
            output_dir=output_dir,
        )
        assert "optimize" in result.stage_timings
        assert all(t >= 0 for t in result.stage_timings.values())

    def test_persists_config(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config, mock_onnx_pipeline
    ) -> None:
        """Config JSON is written to output dir."""
        output_dir = tmp_path / "output"
        build_onnx_model(fake_onnx, config=sample_onnx_config, output_dir=output_dir)

        config_path = output_dir / "winml_build_config.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert "optim" in data


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================


class TestBuildOnnxValidation:
    """Test input validation for build_onnx_model()."""

    def test_build_onnx_file_not_found(self, tmp_path: Path, sample_onnx_config) -> None:
        """Nonexistent ONNX path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="ONNX file not found"):
            build_onnx_model(
                tmp_path / "nonexistent.onnx",
                config=sample_onnx_config,
                output_dir=tmp_path / "output",
            )

    def test_onnx_path_is_directory_rejected(self, tmp_path: Path, sample_onnx_config) -> None:
        """Directory path instead of file raises ValueError."""
        dir_path = tmp_path / "a_dir"
        dir_path.mkdir()
        with pytest.raises(ValueError, match="not a file"):
            build_onnx_model(
                dir_path,
                config=sample_onnx_config,
                output_dir=tmp_path / "output",
            )

    def test_output_dir_is_file_rejected(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config
    ) -> None:
        """Output path that is a file raises ValueError."""
        file_path = tmp_path / "a_file.txt"
        file_path.write_text("not a dir")
        with pytest.raises(ValueError, match="not a directory"):
            build_onnx_model(
                fake_onnx,
                config=sample_onnx_config,
                output_dir=file_path,
            )


# =============================================================================
# STAGE SKIP TESTS
# =============================================================================


class TestBuildOnnxStageSkip:
    """Test stage skipping behavior."""

    def test_build_onnx_skip_quant_when_none(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config_minimal, mock_onnx_pipeline
    ) -> None:
        """config.quant=None skips quantize stage."""
        output_dir = tmp_path / "output"
        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config_minimal,
            output_dir=output_dir,
        )
        assert "quantize" in result.stages_skipped
        assert "quantize" not in result.stages_completed
        mock_onnx_pipeline["quantize"].assert_not_called()

    def test_build_onnx_skip_compile_when_none(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config_minimal, mock_onnx_pipeline
    ) -> None:
        """config.compile=None skips compile stage."""
        output_dir = tmp_path / "output"
        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config_minimal,
            output_dir=output_dir,
        )
        assert "compile" in result.stages_skipped
        assert "compile" not in result.stages_completed
        mock_onnx_pipeline["compile"].assert_not_called()


# =============================================================================
# PRE-QUANTIZED DETECTION TESTS
# =============================================================================


class TestBuildOnnxPreQuantized:
    """Test pre-quantized model auto-detection."""

    def test_build_onnx_pre_quantized_detection(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config, mock_onnx_pipeline
    ) -> None:
        """Model with QDQ nodes skips quant even if config.quant is set."""
        # Make is_quantized_onnx return True
        mock_onnx_pipeline["is_quantized_onnx"].return_value = True

        output_dir = tmp_path / "output"
        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config,
            output_dir=output_dir,
        )
        assert "quantize" in result.stages_skipped
        assert "quantize" not in result.stages_completed
        mock_onnx_pipeline["quantize"].assert_not_called()

    def test_build_onnx_non_quantized_proceeds(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config, mock_onnx_pipeline
    ) -> None:
        """Model without QDQ nodes proceeds with quantization."""
        mock_onnx_pipeline["is_quantized_onnx"].return_value = False

        output_dir = tmp_path / "output"
        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config,
            output_dir=output_dir,
        )
        assert "quantize" in result.stages_completed
        mock_onnx_pipeline["quantize"].assert_called_once()

    def test_pre_quantized_skips_optimize_and_quantize(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config, mock_onnx_pipeline
    ) -> None:
        """QDQ model skips both optimize AND quantize stages."""
        mock_onnx_pipeline["is_quantized_onnx"].return_value = True

        output_dir = tmp_path / "output"
        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config,
            output_dir=output_dir,
        )
        assert "optimize" in result.stages_skipped
        assert "quantize" in result.stages_skipped
        assert "optimize" not in result.stages_completed
        assert "quantize" not in result.stages_completed
        mock_onnx_pipeline["optimize"].assert_called_once()
        mock_onnx_pipeline["quantize"].assert_not_called()

    def test_pre_quantized_still_compiles(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config, mock_onnx_pipeline
    ) -> None:
        """QDQ model still runs compile stage."""
        mock_onnx_pipeline["is_quantized_onnx"].return_value = True

        output_dir = tmp_path / "output"
        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config,
            output_dir=output_dir,
        )
        assert "compile" in result.stages_completed
        mock_onnx_pipeline["compile"].assert_called_once()

    def test_pre_quantized_runs_analyze_only(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config, mock_onnx_pipeline
    ) -> None:
        """Pre-quantized path runs optimize but skips autoconf (no analyze)."""
        mock_onnx_pipeline["is_quantized_onnx"].return_value = True

        output_dir = tmp_path / "output"
        build_onnx_model(
            fake_onnx,
            config=sample_onnx_config,
            output_dir=output_dir,
        )
        # max_optim_iterations=0 means no analyze loop runs
        mock_onnx_pipeline["analyze"].assert_not_called()
        mock_onnx_pipeline["optimize"].assert_called_once()

    def test_skip_optimize_kwarg(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config, mock_onnx_pipeline
    ) -> None:
        """skip_optimize=True forces optimize+quantize skip even without QDQ."""
        mock_onnx_pipeline["is_quantized_onnx"].return_value = False

        output_dir = tmp_path / "output"
        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config,
            output_dir=output_dir,
            skip_optimize=True,
        )
        assert "optimize" in result.stages_skipped
        assert "quantize" in result.stages_skipped
        mock_onnx_pipeline["optimize"].assert_called_once()
        mock_onnx_pipeline["quantize"].assert_not_called()


# =============================================================================
# REUSE TESTS
# =============================================================================


class TestBuildOnnxReuse:
    """Test artifact reuse and rebuild."""

    def test_reuses_existing_artifact(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config, mock_onnx_pipeline
    ) -> None:
        """Existing model.onnx is reused when rebuild=False."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "model.onnx").write_text("existing")

        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config,
            output_dir=output_dir,
        )
        assert result.reused is True
        assert result.stages_completed == []
        mock_onnx_pipeline["optimize"].assert_not_called()

    def test_rebuild_overwrites_existing(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config, mock_onnx_pipeline
    ) -> None:
        """rebuild=True re-runs pipeline even if model.onnx exists."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "model.onnx").write_text("old")

        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config,
            output_dir=output_dir,
            rebuild=True,
        )
        assert result.reused is False
        assert "optimize" in result.stages_completed


# =============================================================================
# BUILD MANIFEST TESTS
# =============================================================================


class TestBuildOnnxManifest:
    """Test winml_manifest.json for ONNX builds."""

    def test_manifest_written(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config, mock_onnx_pipeline
    ) -> None:
        """Manifest file is created after a successful build."""
        output_dir = tmp_path / "output"
        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config,
            output_dir=output_dir,
        )
        assert result.manifest_path is not None
        assert result.manifest_path.exists()

    def test_manifest_content(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config, mock_onnx_pipeline
    ) -> None:
        """Manifest contains correct source, stages, etc."""
        output_dir = tmp_path / "output"
        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config,
            output_dir=output_dir,
        )
        data = json.loads(result.manifest_path.read_text())

        assert data["schema_version"] == 1
        assert data["source"] == "onnx"
        assert data["input_onnx"] == str(fake_onnx)
        assert data["final_artifact"] == "model.onnx"
        assert isinstance(data["elapsed_seconds"], float)
        assert data["timestamp"]  # non-empty ISO timestamp

        # Three stages for ONNX builds (no export)
        stage_names = [s["name"] for s in data["stages"]]
        assert stage_names == ["optimize", "quantize", "compile"]

    def test_manifest_quant_metrics(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config, mock_onnx_pipeline
    ) -> None:
        """Manifest includes QuantizeResult metrics when quantize runs."""
        output_dir = tmp_path / "output"
        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config,
            output_dir=output_dir,
        )
        data = json.loads(result.manifest_path.read_text())

        quant_stage = next(s for s in data["stages"] if s["name"] == "quantize")
        assert quant_stage["status"] == "completed"
        assert quant_stage["nodes_quantized"] == 42
        assert quant_stage["nodes_skipped"] == 3
        assert quant_stage["calibration_time_seconds"] == 1.5
        assert quant_stage["qdq_insertion_time_seconds"] == 0.8

    def test_manifest_not_written_on_reuse(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config, mock_onnx_pipeline
    ) -> None:
        """Manifest is NOT written when reusing a cached artifact."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "model.onnx").write_text("existing")

        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config,
            output_dir=output_dir,
        )
        assert result.reused is True
        assert result.manifest_path is None
        assert not (output_dir / "winml_manifest.json").exists()


# =============================================================================
# ANALYZE JSON OUTPUT TESTS
# =============================================================================


class TestOnnxAnalyzeJsonOutput:
    """Test that analyze_result.json is written to the build folder."""

    def test_analyze_onnx_called_with_output_path(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config_minimal, mock_onnx_pipeline
    ) -> None:
        """analyze_onnx is called with output_path = output_dir/analyze_result.json."""
        output_dir = tmp_path / "output"
        build_onnx_model(fake_onnx, config=sample_onnx_config_minimal, output_dir=output_dir)

        for call in mock_onnx_pipeline["analyze"].call_args_list:
            assert call.kwargs["output_path"] == output_dir / "analyze_result.json"

    def test_no_output_path_for_prequantized(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config, mock_onnx_pipeline
    ) -> None:
        """Pre-quantized path never calls analyze_onnx (no JSON written)."""
        mock_onnx_pipeline["is_quantized_onnx"].return_value = True
        build_onnx_model(fake_onnx, config=sample_onnx_config, output_dir=tmp_path / "output")
        mock_onnx_pipeline["analyze"].assert_not_called()


# =============================================================================
# CACHE KEY TESTS
# =============================================================================


class TestBuildOnnxCacheKey:
    """Test cache_key parameter for artifact naming."""

    def test_no_cache_key_produces_model_onnx(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config_minimal, mock_onnx_pipeline
    ) -> None:
        """cache_key=None (default) produces model.onnx as the final artifact."""
        output_dir = tmp_path / "output"
        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config_minimal,
            output_dir=output_dir,
        )
        assert result.final_onnx_path == output_dir / "model.onnx"

    def test_cache_key_prefixes_final_artifact(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config_minimal, mock_onnx_pipeline
    ) -> None:
        """cache_key prefixes the final artifact filename."""
        output_dir = tmp_path / "output"
        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config_minimal,
            output_dir=output_dir,
            cache_key="imgcls_abc1234567890123",
        )
        assert result.final_onnx_path == output_dir / "imgcls_abc1234567890123_model.onnx"

    def test_cache_key_prefixes_config_path(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config_minimal, mock_onnx_pipeline
    ) -> None:
        """cache_key prefixes the config JSON filename."""
        output_dir = tmp_path / "output"
        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config_minimal,
            output_dir=output_dir,
            cache_key="imgcls_abc1234567890123",
        )
        assert result.config_path == output_dir / "imgcls_abc1234567890123_winml_build_config.json"
        assert result.config_path.exists()

    def test_cache_key_reuse_checks_prefixed_path(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config_minimal, mock_onnx_pipeline
    ) -> None:
        """Existing prefixed model.onnx is reused when rebuild=False."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "imgcls_abc1234567890123_model.onnx").write_text("existing")

        result = build_onnx_model(
            fake_onnx,
            config=sample_onnx_config_minimal,
            output_dir=output_dir,
            cache_key="imgcls_abc1234567890123",
        )
        assert result.reused is True
        mock_onnx_pipeline["optimize"].assert_not_called()

    def test_cache_key_rebuild_does_not_remove_unrelated_artifacts(
        self, tmp_path: Path, fake_onnx: Path, sample_onnx_config_minimal, mock_onnx_pipeline
    ) -> None:
        """rebuild=True with cache_key removes only matching prefixed files, not unrelated ones."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        other = output_dir / "other_model.onnx"
        other.write_text("other-model")

        build_onnx_model(
            fake_onnx,
            config=sample_onnx_config_minimal,
            output_dir=output_dir,
            cache_key="imgcls_abc1234567890123",
            rebuild=True,
        )
        assert other.exists(), "unrelated artifacts should not be removed"
