# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for build API — mock-based, no network, no actual builds.

Tests build_hf_model() and BuildResult independently of CLI and WinMLAutoModel.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.build.hf import BuildResult, build_hf_model


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_config():
    """Create a minimal WinMLBuildConfig for testing."""
    from winml.modelkit.config import WinMLBuildConfig

    return WinMLBuildConfig.from_dict(
        {
            "loader": {
                "task": "image-classification",
                "model_class": "AutoModelForImageClassification",
            },
            "export": {"opset_version": 17, "batch_size": 1},
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
def sample_config_no_quant_compile():
    """Config with quant=None, compile=None."""
    from winml.modelkit.config import WinMLBuildConfig

    return WinMLBuildConfig.from_dict(
        {
            "loader": {"task": "image-classification"},
            "export": {"opset_version": 17},
            "optim": {},
            "quant": None,
            "compile": None,
        }
    )


def _create_file_side_effect(output_kwarg_name: str, return_value: object = None):
    """Create a side_effect that writes a mock file at the output path.

    build_hf_model() does ``shutil.copy2(current_path, final_path)`` at the
    end, so every stage must actually create a file at its output path.

    Args:
        output_kwarg_name: The keyword argument name that carries the output path.
        return_value: Optional value to return from the mock call.
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
def mock_pipeline():
    """Mock all pipeline stage functions.

    Each mock creates an empty file at its output path so that
    ``shutil.copy2`` in the finalize step does not fail.

    The analyzer is mocked to return "no opportunities" by default,
    so existing tests that don't care about the analyzer still pass.
    """
    mock_model = MagicMock()

    quant_result = MagicMock()
    quant_result.success = True
    quant_result.errors = []
    quant_result.nodes_quantized = 42
    quant_result.nodes_skipped = 3
    quant_result.calibration_time_seconds = 1.5
    quant_result.qdq_insertion_time_seconds = 0.8

    compile_result = MagicMock()
    compile_result.output_path = None

    with (
        patch("winml.modelkit.build.hf._load_model", return_value=mock_model) as m_load,
        patch(
            "winml.modelkit.build.hf.export_onnx",
            side_effect=_create_file_side_effect("output_path"),
        ) as m_export,
        patch(
            "winml.modelkit.build.common.optimize_onnx",
            side_effect=_create_file_side_effect("output"),
        ) as m_optimize,
        patch(
            "winml.modelkit.build.common.quantize_onnx",
            side_effect=_create_file_side_effect("output_path", quant_result),
        ) as m_quantize,
        patch(
            "winml.modelkit.build.common.compile_onnx",
            side_effect=_create_file_side_effect("output_path", compile_result),
        ) as m_compile,
        patch(
            "winml.modelkit.build.common.analyze_onnx",
            return_value=_default_analyze_result(),
        ) as m_analyze,
        patch(
            "winml.modelkit.build.common.is_quantized_onnx",
            return_value=False,
        ) as m_has_qdq,
        patch(
            "winml.modelkit.build.common.copy_onnx_model",
            side_effect=lambda src, dst: Path(dst).write_text("mock"),
        ),
    ):
        yield {
            "load": m_load,
            "export": m_export,
            "optimize": m_optimize,
            "quantize": m_quantize,
            "compile": m_compile,
            "analyze": m_analyze,
            "is_quantized_onnx": m_has_qdq,
            "model": mock_model,
        }


# =============================================================================
# BUILD RESULT TESTS
# =============================================================================


class TestBuildResult:
    """Test BuildResult dataclass."""

    def test_default_values(self) -> None:
        result = BuildResult(
            output_dir=Path("/out"),
            final_onnx_path=Path("/out/model.onnx"),
            config_path=Path("/out/winml_build_config.json"),
        )
        assert result.stages_completed == []
        assert result.stages_skipped == []
        assert result.stage_timings == {}
        assert result.elapsed == 0.0
        assert result.reused is False

    def test_reused_result(self) -> None:
        result = BuildResult(
            output_dir=Path("/out"),
            final_onnx_path=Path("/out/model.onnx"),
            config_path=Path("/out/winml_build_config.json"),
            reused=True,
        )
        assert result.reused is True


# =============================================================================
# BUILD API TESTS
# =============================================================================


class TestBuildHfModel:
    """Test build_hf_model() function."""

    def test_creates_output_dir(self, tmp_path: Path, sample_config, mock_pipeline) -> None:
        output_dir = tmp_path / "new_dir"
        assert not output_dir.exists()

        build_hf_model(config=sample_config, output_dir=output_dir, model_id="test")
        assert output_dir.is_dir()

    def test_produces_build_result(self, tmp_path: Path, sample_config, mock_pipeline) -> None:
        result = build_hf_model(config=sample_config, output_dir=tmp_path, model_id="test")
        assert isinstance(result, BuildResult)
        assert result.output_dir == tmp_path
        assert result.final_onnx_path == tmp_path / "model.onnx"
        assert result.config_path == tmp_path / "winml_build_config.json"
        assert result.reused is False
        assert result.elapsed >= 0

    def test_persists_config(self, tmp_path: Path, sample_config, mock_pipeline) -> None:
        build_hf_model(config=sample_config, output_dir=tmp_path, model_id="test")
        config_path = tmp_path / "winml_build_config.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert "export" in data
        assert "optim" in data

    def test_all_stages_completed(self, tmp_path: Path, sample_config, mock_pipeline) -> None:
        result = build_hf_model(config=sample_config, output_dir=tmp_path, model_id="test")
        assert "export" in result.stages_completed
        assert "optimize" in result.stages_completed
        assert "quantize" in result.stages_completed
        assert "compile" in result.stages_completed
        assert result.stages_skipped == []

    def test_stage_timings_populated(self, tmp_path: Path, sample_config, mock_pipeline) -> None:
        result = build_hf_model(config=sample_config, output_dir=tmp_path, model_id="test")
        assert "export" in result.stage_timings
        assert "optimize" in result.stage_timings
        assert all(t >= 0 for t in result.stage_timings.values())

    def test_no_quant_skips_quantize(
        self, tmp_path: Path, sample_config_no_quant_compile, mock_pipeline
    ) -> None:
        result = build_hf_model(
            config=sample_config_no_quant_compile, output_dir=tmp_path, model_id="test"
        )
        assert "quantize" in result.stages_skipped
        assert "compile" in result.stages_skipped
        assert "quantize" not in result.stages_completed
        mock_pipeline["quantize"].assert_not_called()
        mock_pipeline["compile"].assert_not_called()

    def test_reuses_existing_artifact(self, tmp_path: Path, sample_config, mock_pipeline) -> None:
        # Create fake existing artifact
        (tmp_path / "model.onnx").write_text("existing")

        result = build_hf_model(config=sample_config, output_dir=tmp_path, model_id="test")
        assert result.reused is True
        assert result.stages_completed == []
        mock_pipeline["export"].assert_not_called()

    def test_rebuild_overwrites_existing(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        # Create fake existing artifacts
        (tmp_path / "model.onnx").write_text("old")
        (tmp_path / "export.onnx").write_text("old_export")

        result = build_hf_model(
            config=sample_config,
            output_dir=tmp_path,
            model_id="test",
            rebuild=True,
        )
        assert result.reused is False
        assert "export" in result.stages_completed
        mock_pipeline["export"].assert_called_once()

    def test_rebuild_cleans_old_artifacts(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        # Create fake stale artifacts
        (tmp_path / "model.onnx").write_text("old")
        (tmp_path / "quantized.onnx").write_text("stale_quant")

        build_hf_model(
            config=sample_config,
            output_dir=tmp_path,
            model_id="test",
            rebuild=True,
        )
        # Old artifacts should have been removed before rebuild
        # (new ones created by mocked pipeline won't actually write files)
        # The key is that the old stale files were unlinked

    def test_config_not_persisted_on_reuse(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        (tmp_path / "model.onnx").write_text("existing")

        build_hf_model(config=sample_config, output_dir=tmp_path, model_id="test")
        # Config should NOT be written when reusing
        assert not (tmp_path / "winml_build_config.json").exists()

    def test_pretrained_weights_calls_load_model(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        build_hf_model(config=sample_config, output_dir=tmp_path, model_id="bert-base")
        mock_pipeline["load"].assert_called_once()
        call_args = mock_pipeline["load"].call_args
        assert call_args[0][1] == "bert-base"  # model_id

    def test_pretrained_load_threads_model_class(self, sample_config) -> None:
        """Pretrained load path forwards ``loader.model_class`` to load_hf_model.

        Regression (#836): the pretrained branch dropped the explicit
        ``model_class``, so e.g. CLIP feature-extraction resolved to the full
        CLIPModel (which requires ``pixel_values``) instead of the configured
        CLIPTextModelWithProjection, failing export with text-only inputs.
        """
        from winml.modelkit.build.hf import _load_model

        with patch("winml.modelkit.loader.load_hf_model") as m_load:
            m_load.return_value = (MagicMock(), MagicMock(), "image-classification")
            _load_model(sample_config, "test-model", trust_remote_code=False)

        m_load.assert_called_once()
        assert m_load.call_args.kwargs["model_class"] == "AutoModelForImageClassification"

    def test_pre_loaded_model_skips_load(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        pre_loaded = MagicMock()
        build_hf_model(
            config=sample_config,
            output_dir=tmp_path,
            model_id="test",
            pytorch_model=pre_loaded,
        )
        mock_pipeline["load"].assert_not_called()
        # export should receive the pre-loaded model
        export_call = mock_pipeline["export"].call_args
        assert export_call.kwargs["model"] is pre_loaded

    def test_export_failure_raises_runtime_error(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        mock_pipeline["export"].side_effect = RuntimeError("ONNX export failed")
        with pytest.raises(RuntimeError, match="ONNX export failed"):
            build_hf_model(config=sample_config, output_dir=tmp_path, model_id="test")

    def test_quantize_failure_raises_runtime_error(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        failed_result = MagicMock()
        failed_result.success = False
        failed_result.errors = ["calibration failed"]
        # Override side_effect (not return_value) because side_effect takes priority
        mock_pipeline["quantize"].side_effect = _create_file_side_effect(
            "output_path", failed_result
        )

        with pytest.raises(RuntimeError, match="Quantization failed"):
            build_hf_model(config=sample_config, output_dir=tmp_path, model_id="test")


class TestBuildValidation:
    """Test input validation."""

    def test_empty_model_id_rejected(self, tmp_path: Path, sample_config) -> None:
        with pytest.raises(ValueError, match="empty string"):
            build_hf_model(config=sample_config, output_dir=tmp_path, model_id="")

    def test_whitespace_model_id_rejected(self, tmp_path: Path, sample_config) -> None:
        with pytest.raises(ValueError, match="empty string"):
            build_hf_model(config=sample_config, output_dir=tmp_path, model_id="   ")

    def test_output_dir_is_file_rejected(self, tmp_path: Path, sample_config) -> None:
        file_path = tmp_path / "a_file.txt"
        file_path.write_text("not a dir")
        with pytest.raises(ValueError, match="not a directory"):
            build_hf_model(config=sample_config, output_dir=file_path, model_id="test")

    def test_invalid_config_rejected(self, tmp_path: Path) -> None:
        """build_hf_model() rejects invalid config via validate()."""
        from winml.modelkit.config import WinMLBuildConfig

        bad_config = WinMLBuildConfig.from_dict(
            {
                "loader": {"task": None},  # Missing required task
                "export": {"opset_version": 17},
                "optim": {},
            }
        )
        with pytest.raises(ValueError, match="Config validation failed"):
            build_hf_model(config=bad_config, output_dir=tmp_path, model_id="test")


# =============================================================================
# CACHE_KEY TESTS
# =============================================================================


class TestCacheKey:
    """Test cache_key parameter prefixes artifact filenames."""

    def test_cache_key_prefixes_final_path(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        result = build_hf_model(
            config=sample_config,
            output_dir=tmp_path,
            model_id="test",
            cache_key="imgcls_abc123",
        )
        assert result.final_onnx_path == tmp_path / "imgcls_abc123_model.onnx"

    def test_cache_key_prefixes_config_path(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        result = build_hf_model(
            config=sample_config,
            output_dir=tmp_path,
            model_id="test",
            cache_key="imgcls_abc123",
        )
        assert result.config_path == tmp_path / "imgcls_abc123_winml_build_config.json"
        assert result.config_path.exists()

    def test_cache_key_prefixes_export_path(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        build_hf_model(
            config=sample_config,
            output_dir=tmp_path,
            model_id="test",
            cache_key="imgcls_abc123",
        )
        # Verify export was called with prefixed path
        export_call = mock_pipeline["export"].call_args
        assert export_call.kwargs["output_path"] == tmp_path / "imgcls_abc123_export.onnx"

    def test_cache_key_prefixes_optimize_path(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        build_hf_model(
            config=sample_config,
            output_dir=tmp_path,
            model_id="test",
            cache_key="imgcls_abc123",
        )
        opt_call = mock_pipeline["optimize"].call_args
        assert opt_call.kwargs["output"] == tmp_path / "imgcls_abc123_optimized.onnx"

    def test_cache_key_none_uses_unprefixed_names(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        """cache_key=None (default) preserves original unprefixed behavior."""
        result = build_hf_model(
            config=sample_config,
            output_dir=tmp_path,
            model_id="test",
            cache_key=None,
        )
        assert result.final_onnx_path == tmp_path / "model.onnx"
        assert result.config_path == tmp_path / "winml_build_config.json"

    def test_cache_key_reuse_checks_prefixed_path(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        """Reuse detection uses the prefixed final path."""
        # Create prefixed artifact
        (tmp_path / "imgcls_abc123_model.onnx").write_text("existing")

        result = build_hf_model(
            config=sample_config,
            output_dir=tmp_path,
            model_id="test",
            cache_key="imgcls_abc123",
        )
        assert result.reused is True
        mock_pipeline["export"].assert_not_called()

    def test_cache_key_rebuild_cleans_only_onnx(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        """Rebuild removes *.onnx but non-onnx files are untouched."""
        (tmp_path / "imgcls_abc123_model.onnx").write_text("old")
        (tmp_path / "keep_me.txt").write_text("important")

        build_hf_model(
            config=sample_config,
            output_dir=tmp_path,
            model_id="test",
            cache_key="imgcls_abc123",
            rebuild=True,
        )
        assert (tmp_path / "keep_me.txt").exists()

    def test_rebuild_scoped_to_cache_key(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        """Rebuild only removes files matching the current cache_key prefix."""
        # Create artifacts for TWO different cache keys
        (tmp_path / "imgcls_abc123_model.onnx").write_text("variant_a")
        (tmp_path / "txtcls_def456_model.onnx").write_text("variant_b")

        build_hf_model(
            config=sample_config,
            output_dir=tmp_path,
            model_id="test",
            cache_key="imgcls_abc123",
            rebuild=True,
        )
        # variant_b should survive
        assert (tmp_path / "txtcls_def456_model.onnx").exists()


# =============================================================================
# BUILD MANIFEST TESTS
# =============================================================================


class TestBuildManifest:
    """Test build_manifest.json writing."""

    def test_build_manifest_written(self, tmp_path: Path, sample_config, mock_pipeline) -> None:
        """Manifest file is created after a successful build."""
        result = build_hf_model(config=sample_config, output_dir=tmp_path, model_id="test")
        manifest_path = tmp_path / "build_manifest.json"
        assert manifest_path.exists()
        assert result.manifest_path == manifest_path

    def test_build_manifest_not_written_on_reuse(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        """Manifest is NOT written when reusing a cached artifact."""
        (tmp_path / "model.onnx").write_text("existing")

        result = build_hf_model(config=sample_config, output_dir=tmp_path, model_id="test")
        assert result.reused is True
        assert result.manifest_path is None
        assert not (tmp_path / "build_manifest.json").exists()

    def test_build_manifest_content(self, tmp_path: Path, sample_config, mock_pipeline) -> None:
        """Manifest contains correct stages, model_id, task, etc."""
        result = build_hf_model(
            config=sample_config, output_dir=tmp_path, model_id="microsoft/resnet-50"
        )
        data = json.loads(result.manifest_path.read_text())

        assert data["schema_version"] == 1
        assert data["model_id"] == "microsoft/resnet-50"
        assert data["task"] == "image-classification"
        assert data["final_artifact"] == "model.onnx"
        assert isinstance(data["elapsed_seconds"], float)
        assert data["timestamp"]  # non-empty ISO timestamp

        # Four stages (analyze is part of optimize, not separate)
        stage_names = [s["name"] for s in data["stages"]]
        assert stage_names == ["export", "optimize", "quantize", "compile"]

        # export and optimize are always completed
        completed = [s for s in data["stages"] if s["status"] == "completed"]
        completed_names = [s["name"] for s in completed]
        assert "export" in completed_names
        assert "optimize" in completed_names

        # Each completed stage has a filename and elapsed_seconds
        for s in completed:
            assert s["filename"] is not None
            assert isinstance(s["elapsed_seconds"], float)

    def test_build_manifest_with_cache_key(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        """Manifest filename is prefixed when cache_key is set."""
        result = build_hf_model(
            config=sample_config,
            output_dir=tmp_path,
            model_id="test",
            cache_key="imgcls_abc123",
        )
        expected = tmp_path / "imgcls_abc123_build_manifest.json"
        assert expected.exists()
        assert result.manifest_path == expected

        data = json.loads(expected.read_text())
        assert data["cache_key"] == "imgcls_abc123"
        assert data["config_hash"] == "abc123"


# =============================================================================
# ANALYZER AUTOCONF LOOP TESTS
# =============================================================================


class TestBuildAnalyzerLoop:
    """Tests for the analyzer autoconf loop (stage 3.5)."""

    def _make_analyze_result(
        self,
        *,
        has_opportunities: bool = False,
        has_errors: bool = False,
        error_patterns: list[str] | None = None,
        optimization_config: dict | None = None,
    ):
        """Build a mock AnalyzeResult."""
        from winml.modelkit.analyze import AnalyzeResult, LintResult
        from winml.modelkit.optim import WinMLOptimizationConfig

        config = WinMLOptimizationConfig(**(optimization_config or {}))
        lint = LintResult(
            errors=len(error_patterns) if error_patterns else (1 if has_errors else 0),
            warnings=1 if has_opportunities else 0,
            info=0,
            passed=not has_errors and not has_opportunities,
            error_patterns=error_patterns or (["BlackNode"] if has_errors else []),
            warning_patterns=["SUBGRAPH/GeluPattern"] if has_opportunities else [],
            information=[],
            optimization_config=config,
        )
        return AnalyzeResult(lint=lint, optimization_config=config)

    def test_autoconf_converges_in_one_iteration(
        self, tmp_path: Path, sample_config_no_quant_compile, mock_pipeline
    ) -> None:
        """Autoconf finds no opportunities -> single iteration."""
        result_no_opps = self._make_analyze_result(has_opportunities=False)

        with patch("winml.modelkit.build.common.analyze_onnx", return_value=result_no_opps) as m:
            result = build_hf_model(
                config=sample_config_no_quant_compile,
                output_dir=tmp_path,
                model_id="test",
                ep="qnn",
                device="NPU",
            )

        # Autoconf is part of optimize, not a separate stage
        assert "optimize" in result.stages_completed
        # Two analyze calls: one in loop (no autoconf), one final validation
        assert m.call_count == 2

    def test_autoconf_discovers_and_reoptimizes(
        self, tmp_path: Path, sample_config_no_quant_compile, mock_pipeline
    ) -> None:
        """Autoconf finds gelu_fusion -> re-optimizes -> converges on 2nd iteration."""
        result_with_gelu = self._make_analyze_result(
            has_opportunities=True,
            optimization_config={"gelu_fusion": True},
        )
        result_converged = self._make_analyze_result(has_opportunities=False)

        with patch(
            "winml.modelkit.build.common.analyze_onnx",
            side_effect=[result_with_gelu, result_converged, result_converged],
        ) as m_analyze:
            result = build_hf_model(
                config=sample_config_no_quant_compile,
                output_dir=tmp_path,
                model_id="test",
                ep="qnn",
                device="NPU",
            )

        assert "optimize" in result.stages_completed
        # 3 analyze calls: initial (found gelu) + after re-optimize (converged) + final validation
        assert m_analyze.call_count == 3
        # optimize_onnx called: once initial + once re-optimize in autoconf
        assert mock_pipeline["optimize"].call_count == 2

    def test_autoconf_runs_without_ep(
        self, tmp_path: Path, sample_config_no_quant_compile, mock_pipeline
    ) -> None:
        """Autoconf runs even without EP (portable-first: all-EP aggregation)."""
        result_no_opps = self._make_analyze_result(has_opportunities=False)

        with patch("winml.modelkit.build.common.analyze_onnx", return_value=result_no_opps) as m:
            build_hf_model(
                config=sample_config_no_quant_compile,
                output_dir=tmp_path,
                model_id="test",
            )

        call_kwargs = m.call_args.kwargs
        assert call_kwargs["ep"] is None

    def test_autoconf_max_iterations_stops_loop(
        self, tmp_path: Path, sample_config_no_quant_compile, mock_pipeline
    ) -> None:
        """Loop stops at hack_max_optim_iterations even if not converged."""
        result_always_has_opps = self._make_analyze_result(
            has_opportunities=True,
            optimization_config={"gelu_fusion": True},
        )

        with patch(
            "winml.modelkit.build.common.analyze_onnx",
            return_value=result_always_has_opps,
        ) as m:
            build_hf_model(
                config=sample_config_no_quant_compile,
                output_dir=tmp_path,
                model_id="test",
                ep="qnn",
                hack_max_optim_iterations=3,
            )

        # 4 analyze calls: 1 initial + 3 re-analyze after autoconf re-optimizations
        assert m.call_count == 4

    def test_autoconf_unsupported_nodes_raise(
        self, tmp_path: Path, sample_config_no_quant_compile, mock_pipeline
    ) -> None:
        """Unsupported nodes after convergence raise RuntimeError."""
        result_with_errors = self._make_analyze_result(
            has_opportunities=False,
            has_errors=True,
            error_patterns=["UnsupportedOp"],
        )

        with (
            patch("winml.modelkit.build.common.analyze_onnx", return_value=result_with_errors),
            pytest.raises(RuntimeError, match="Unsupported nodes persist"),
        ):
            build_hf_model(
                config=sample_config_no_quant_compile,
                output_dir=tmp_path,
                model_id="test",
                ep="qnn",
            )

    def test_manifest_records_analyze_details(
        self, tmp_path: Path, sample_config_no_quant_compile, mock_pipeline
    ) -> None:
        """Manifest has analyze_details with lint + autoconf (no separate stage)."""
        result_with_gelu = self._make_analyze_result(
            has_opportunities=True,
            optimization_config={"gelu_fusion": True},
        )
        result_converged = self._make_analyze_result(has_opportunities=False)

        with patch(
            "winml.modelkit.build.common.analyze_onnx",
            side_effect=[result_with_gelu, result_converged, result_converged],
        ):
            result = build_hf_model(
                config=sample_config_no_quant_compile,
                output_dir=tmp_path,
                model_id="test",
                ep="qnn",
            )

        data = json.loads(result.manifest_path.read_text())
        assert data["analyze_iterations"] == 2
        assert data["analyze_unsupported_node_count"] == 0

        # No separate "analyze" stage in stages list
        stage_names = [s["name"] for s in data["stages"]]
        assert "analyze" not in stage_names

        # But analyze_details has lint + autoconf
        details = data["analyze_details"]
        assert details["lint"]["errors"] == 0
        assert details["lint"]["passed"] is True
        assert details["autoconf"] == {"gelu_fusion": True}

    def test_autoconf_merges_config_for_downstream(
        self, tmp_path: Path, sample_config_no_quant_compile, mock_pipeline
    ) -> None:
        """Autoconf flags are merged into config.optim for downstream stages."""
        result_with_flags = self._make_analyze_result(
            has_opportunities=True,
            optimization_config={"gelu_fusion": True, "layer_norm_fusion": True},
        )
        result_converged = self._make_analyze_result(has_opportunities=False)

        with patch(
            "winml.modelkit.build.common.analyze_onnx",
            side_effect=[result_with_flags, result_converged, result_converged],
        ):
            build_hf_model(
                config=sample_config_no_quant_compile,
                output_dir=tmp_path,
                model_id="test",
                ep="qnn",
            )

        assert sample_config_no_quant_compile.optim.get("gelu_fusion") is True
        assert sample_config_no_quant_compile.optim.get("layer_norm_fusion") is True


# =============================================================================
# PRE-QUANTIZED (QDQ) DETECTION TESTS
# =============================================================================


class TestBuildHfPreQuantized:
    """Test pre-quantized detection in HF build pipeline."""

    def test_post_export_qdq_skips_optimize_and_quantize(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        """Exported QDQ/QOperator ONNX truly skips both optimize AND quantize.

        Regression: previously the pre-quantized branch logged "skipping
        optimize" but still invoked ``optimize_onnx``. That hidden call
        crashed for QOperator models with ``ConvInteger`` (no CPU kernel).
        """
        mock_pipeline["is_quantized_onnx"].return_value = True

        output_dir = tmp_path / "output"
        result = build_hf_model(
            config=sample_config,
            output_dir=output_dir,
            pytorch_model=mock_pipeline["model"],
        )
        assert "optimize" in result.stages_skipped
        assert "quantize" in result.stages_skipped
        assert "optimize" not in result.stages_completed
        assert "quantize" not in result.stages_completed
        mock_pipeline["optimize"].assert_not_called()
        mock_pipeline["quantize"].assert_not_called()

    def test_post_export_qdq_still_exports(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        """Export stage always runs regardless of QDQ detection."""
        mock_pipeline["is_quantized_onnx"].return_value = True

        output_dir = tmp_path / "output"
        result = build_hf_model(
            config=sample_config,
            output_dir=output_dir,
            pytorch_model=mock_pipeline["model"],
        )
        assert "export" in result.stages_completed
        mock_pipeline["export"].assert_called_once()

    def test_post_export_qdq_still_compiles(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        """Compile still runs for pre-quantized models."""
        mock_pipeline["is_quantized_onnx"].return_value = True

        output_dir = tmp_path / "output"
        result = build_hf_model(
            config=sample_config,
            output_dir=output_dir,
            pytorch_model=mock_pipeline["model"],
        )
        assert "compile" in result.stages_completed
        mock_pipeline["compile"].assert_called_once()

    def test_post_export_qdq_runs_analyze_only(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        """Pre-quantized path skips both optimize AND analyze (max_iters=0)."""
        mock_pipeline["is_quantized_onnx"].return_value = True

        output_dir = tmp_path / "output"
        build_hf_model(
            config=sample_config,
            output_dir=output_dir,
            pytorch_model=mock_pipeline["model"],
        )
        # max_optim_iterations=0 means no analyze loop runs.
        # Optimize is also skipped via skip_optimize=True.
        mock_pipeline["analyze"].assert_not_called()
        mock_pipeline["optimize"].assert_not_called()

    def test_skip_optimize_kwarg(self, tmp_path: Path, sample_config, mock_pipeline) -> None:
        """skip_optimize=True forces optimize+quantize skip."""
        mock_pipeline["is_quantized_onnx"].return_value = False

        output_dir = tmp_path / "output"
        result = build_hf_model(
            config=sample_config,
            output_dir=output_dir,
            pytorch_model=mock_pipeline["model"],
            skip_optimize=True,
        )
        assert "optimize" in result.stages_skipped
        assert "quantize" in result.stages_skipped
        mock_pipeline["optimize"].assert_not_called()
        mock_pipeline["quantize"].assert_not_called()


# =============================================================================
# ANALYZE JSON OUTPUT TESTS
# =============================================================================


class TestAnalyzeJsonOutput:
    """Test that analyze_result.json is written to the build folder."""

    def test_analyze_onnx_called_with_output_path(
        self, tmp_path: Path, sample_config_no_quant_compile, mock_pipeline
    ) -> None:
        """analyze_onnx is called with output_path pointing to analyze_result.json."""
        build_hf_model(
            config=sample_config_no_quant_compile,
            output_dir=tmp_path,
            model_id="test",
            ep="qnn",
        )
        # Both loop call and final call pass output_path
        for call in mock_pipeline["analyze"].call_args_list:
            assert call.kwargs["output_path"] == tmp_path / "analyze_result.json"

    def test_analyze_output_path_respects_cache_key(
        self, tmp_path: Path, sample_config_no_quant_compile, mock_pipeline
    ) -> None:
        """analyze_result.json is prefixed when cache_key is set."""
        build_hf_model(
            config=sample_config_no_quant_compile,
            output_dir=tmp_path,
            model_id="test",
            cache_key="imgcls_abc123",
        )
        expected = tmp_path / "imgcls_abc123_analyze_result.json"
        for call in mock_pipeline["analyze"].call_args_list:
            assert call.kwargs["output_path"] == expected

    def test_no_output_path_for_prequantized(
        self, tmp_path: Path, sample_config, mock_pipeline
    ) -> None:
        """Pre-quantized path never calls analyze_onnx (no JSON written)."""
        mock_pipeline["is_quantized_onnx"].return_value = True
        build_hf_model(
            config=sample_config,
            output_dir=tmp_path / "output",
            pytorch_model=mock_pipeline["model"],
        )
        mock_pipeline["analyze"].assert_not_called()

    def test_analyze_onnx_writes_json_to_disk(self, tmp_path: Path) -> None:
        """analyze_onnx with output_path writes a valid JSON file."""
        import onnx

        from winml.modelkit.analyze import analyze_onnx

        # Build a trivial ONNX model (single Relu node)
        node = onnx.helper.make_node("Relu", inputs=["x"], outputs=["y"])
        graph = onnx.helper.make_graph(
            [node],
            "test",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
        )
        model_proto = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
        )
        model_path = tmp_path / "tiny.onnx"
        onnx.save(model_proto, str(model_path))

        output_path = tmp_path / "analyze_result.json"
        analyze_onnx(model_path, ep="qnn", output_path=output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "metadata" in data
        assert "results" in data

    def test_analyze_onnx_no_output_path_writes_nothing(self, tmp_path: Path) -> None:
        """analyze_onnx without output_path writes no file (backward compat)."""
        import onnx

        from winml.modelkit.analyze import analyze_onnx

        node = onnx.helper.make_node("Relu", inputs=["x"], outputs=["y"])
        graph = onnx.helper.make_graph(
            [node],
            "test",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
        )
        model_proto = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
        )
        model_path = tmp_path / "tiny.onnx"
        onnx.save(model_proto, str(model_path))

        analyze_onnx(model_path, ep="qnn")  # no output_path

        assert not any(tmp_path.glob("*.json"))
