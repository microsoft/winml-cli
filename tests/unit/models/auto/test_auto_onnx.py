# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for WinMLAutoModel.from_onnx() classmethod.

Verifies:
- from_onnx() auto-generates config via generate_build_config(onnx_path=...)
- from_onnx() uses explicit config when provided
- from_pretrained() delegates ONNX files to from_onnx()
- from_onnx passes ep and device through to build_onnx_model()
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, ClassVar
from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.models.auto import WinMLAutoModel


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def fake_onnx(tmp_path: Path) -> Path:
    """Create a fake ONNX file for testing."""
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"fake-onnx")
    return onnx_file


def _make_build_result(tmp_path: Path) -> MagicMock:
    """Create a mock BuildResult with the expected attributes."""
    result = MagicMock()
    result.final_onnx_path = tmp_path / "model.onnx"
    result.output_dir = tmp_path
    return result


class TestFromOnnx:
    """Test WinMLAutoModel.from_onnx()."""

    def test_auto_generates_config_when_none(self, fake_onnx: Path, tmp_path: Path):
        """from_onnx() without config auto-generates via generate_build_config."""
        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_check_device_ep",
                return_value=("npu", ["npu", "cpu"], ["QNNExecutionProvider"]),
            ),
            patch(
                "winml.modelkit.config.precision.resolve_eps",
                return_value=["QNNExecutionProvider"],
            ),
            patch("winml.modelkit.build.build_onnx_model") as mock_build,
            patch("winml.modelkit.models.auto.get_winml_class") as mock_get_class,
        ):
            mock_build.return_value = _make_build_result(tmp_path)
            mock_instance = MagicMock()
            mock_get_class.return_value = lambda **kw: mock_instance

            WinMLAutoModel.from_onnx(
                str(fake_onnx),
                task="image-classification",
                device="npu",
            )

        mock_build.assert_called_once()
        call_kwargs = mock_build.call_args.kwargs
        config = call_kwargs["config"]
        # ONNX builds have export=None (no HF export needed)
        assert config.export is None

    def test_uses_explicit_config_as_override(self, fake_onnx: Path, tmp_path: Path):
        """from_onnx() with explicit config merges it as override on generated config."""
        from winml.modelkit.config import WinMLBuildConfig
        from winml.modelkit.optim.config import WinMLOptimizationConfig

        # Override with specific optim flags (export=None inherited from base)
        explicit_config = WinMLBuildConfig(
            export=None,  # preserve ONNX sentinel
            optim=WinMLOptimizationConfig(gelu_fusion=True),
            quant=None,
        )

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
            patch(
                "winml.modelkit.config.precision.resolve_eps",
                return_value=["CPUExecutionProvider"],
            ),
            patch("winml.modelkit.build.build_onnx_model") as mock_build,
            patch("winml.modelkit.models.auto.get_winml_class") as mock_get_class,
        ):
            mock_build.return_value = _make_build_result(tmp_path)
            mock_instance = MagicMock()
            mock_get_class.return_value = lambda **kw: mock_instance

            WinMLAutoModel.from_onnx(
                fake_onnx,
                task="image-classification",
                config=explicit_config,
            )

        call_kwargs = mock_build.call_args.kwargs
        # Config is generated with override applied
        assert call_kwargs["config"].export is None  # ONNX sentinel preserved
        assert call_kwargs["config"].quant is None  # from override
        assert call_kwargs["config"].optim.get("gelu_fusion") is True  # from override

    def test_passes_ep_and_device_to_build(self, fake_onnx: Path, tmp_path: Path):
        """from_onnx() forwards ep and device through to build_onnx_model."""
        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("npu", ["npu", "cpu"]),
            ),
            patch("winml.modelkit.build.build_onnx_model") as mock_build,
            patch("winml.modelkit.models.auto.get_winml_class") as mock_get_class,
        ):
            mock_build.return_value = _make_build_result(tmp_path)
            mock_instance = MagicMock()
            mock_get_class.return_value = lambda **kw: mock_instance

            WinMLAutoModel.from_onnx(
                fake_onnx,
                task="image-classification",
                device="npu",
                ep="qnn",
            )

        call_kwargs = mock_build.call_args.kwargs
        assert call_kwargs["ep"] == "qnn"
        assert call_kwargs["device"] == "npu"

    def test_passes_allow_unsupported_nodes_to_build(self, fake_onnx: Path, tmp_path: Path):
        """from_onnx() forwards allow_unsupported_nodes through to build_onnx_model."""
        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
            patch("winml.modelkit.build.build_onnx_model") as mock_build,
            patch("winml.modelkit.models.auto.get_winml_class") as mock_get_class,
        ):
            mock_build.return_value = _make_build_result(tmp_path)
            mock_instance = MagicMock()
            mock_get_class.return_value = lambda **kw: mock_instance

            WinMLAutoModel.from_onnx(
                fake_onnx,
                task="image-classification",
                device="cpu",
                allow_unsupported_nodes=True,
            )

        assert mock_build.call_args.kwargs["allow_unsupported_nodes"] is True

    def test_returns_winml_pretrained_model(self, fake_onnx: Path, tmp_path: Path):
        """from_onnx() returns the inference wrapper from get_winml_class."""
        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
            patch(
                "winml.modelkit.config.precision.resolve_eps",
                return_value=["CPUExecutionProvider"],
            ),
            patch("winml.modelkit.build.build_onnx_model") as mock_build,
            patch("winml.modelkit.models.auto.get_winml_class") as mock_get_class,
        ):
            mock_build.return_value = _make_build_result(tmp_path)
            mock_instance = MagicMock()
            mock_get_class.return_value = lambda **kw: mock_instance

            result = WinMLAutoModel.from_onnx(
                fake_onnx,
                task="image-classification",
            )

        assert result is mock_instance


class TestFromPretrainedDelegatesToFromOnnx:
    """Test that from_pretrained delegates .onnx files to from_onnx."""

    def test_delegates_onnx_to_from_onnx(self, fake_onnx: Path, tmp_path: Path):
        """from_pretrained with .onnx file delegates to from_onnx."""
        with patch.object(WinMLAutoModel, "from_onnx") as mock_from_onnx:
            mock_from_onnx.return_value = MagicMock()

            WinMLAutoModel.from_pretrained(
                str(fake_onnx),
                task="image-classification",
                device="cpu",
                precision="fp32",
            )

        mock_from_onnx.assert_called_once()
        call_kwargs = mock_from_onnx.call_args.kwargs
        assert call_kwargs["task"] == "image-classification"
        assert call_kwargs["device"] == "cpu"
        assert call_kwargs["precision"] == "fp32"

    def test_passes_ep_from_kwargs(self, fake_onnx: Path, tmp_path: Path):
        """from_pretrained extracts ep from kwargs and passes to from_onnx."""
        with patch.object(WinMLAutoModel, "from_onnx") as mock_from_onnx:
            mock_from_onnx.return_value = MagicMock()

            WinMLAutoModel.from_pretrained(
                str(fake_onnx),
                task="image-classification",
                ep="qnn",
            )

        call_kwargs = mock_from_onnx.call_args.kwargs
        assert call_kwargs["ep"] == "qnn"


# =============================================================================
# from_onnx cache dir and cache_key tests
# =============================================================================


class TestFromOnnxCacheDirAndKey:
    """Verify from_onnx uses content-addressed model dirs and passes cache_key."""

    def test_uses_content_hash_for_model_dir(self, fake_onnx: Path, tmp_path: Path):
        """from_onnx uses the ONNX content hash as model_id for get_model_dir."""
        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
            patch(
                "winml.modelkit.config.precision.resolve_eps",
                return_value=["CPUExecutionProvider"],
            ),
            patch("winml.modelkit.build.build_onnx_model") as mock_build,
            patch("winml.modelkit.models.auto.get_winml_class") as mock_get_class,
            patch("winml.modelkit.models.auto.get_model_dir") as mock_get_model_dir,
        ):
            mock_build.return_value = _make_build_result(tmp_path)
            mock_get_class.return_value = lambda **kw: MagicMock()
            mock_get_model_dir.return_value = tmp_path / "model_dir"

            WinMLAutoModel.from_onnx(
                fake_onnx,
                task="image-classification",
                device="cpu",
            )

        mock_get_model_dir.assert_called_once()
        model_id_arg = mock_get_model_dir.call_args.args[0]
        expected_hash = hashlib.sha256(fake_onnx.read_bytes()).hexdigest()[:16]
        assert model_id_arg == f"onnx-{expected_hash}"
        assert model_id_arg != str(fake_onnx.resolve())

    def test_replacing_same_path_content_gets_different_model_dir(self, tmp_path: Path):
        """Replacing an ONNX file at the same path changes its cache model dir."""
        from winml.modelkit.cache import get_cache_dir, get_model_dir
        from winml.modelkit.onnx import get_onnx_model_hash

        onnx_path = tmp_path / "model.onnx"
        cache = get_cache_dir()

        onnx_path.write_bytes(b"first-content")
        model_dir_a = get_model_dir(f"onnx-{get_onnx_model_hash(onnx_path)}", cache_dir=cache)

        onnx_path.write_bytes(b"second-content")
        model_dir_b = get_model_dir(f"onnx-{get_onnx_model_hash(onnx_path)}", cache_dir=cache)

        assert model_dir_a != model_dir_b

    def test_onnx_model_hash_includes_external_data(self, tmp_path: Path):
        """Changing external data changes the ONNX model content hash."""
        import numpy as np
        import onnx
        from onnx import TensorProto, helper

        from winml.modelkit.onnx import get_onnx_model_hash

        onnx_path = tmp_path / "external.onnx"
        data_path = tmp_path / "external.onnx.data"
        tensor = helper.make_tensor(
            "weight",
            TensorProto.FLOAT,
            [4],
            np.arange(4, dtype=np.float32).tobytes(),
            raw=True,
        )
        graph = helper.make_graph([], "external-data-test", [], [], [tensor])
        model = helper.make_model(graph)
        onnx.save_model(
            model,
            str(onnx_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_path.name,
            size_threshold=0,
        )

        original_hash = get_onnx_model_hash(onnx_path)
        data_path.write_bytes(data_path.read_bytes() + b"changed")

        assert get_onnx_model_hash(onnx_path) != original_hash

    def test_passes_cache_key_to_build_onnx_model(self, fake_onnx: Path, tmp_path: Path):
        """from_onnx computes and passes a cache_key to build_onnx_model."""
        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
            patch(
                "winml.modelkit.config.precision.resolve_eps",
                return_value=["CPUExecutionProvider"],
            ),
            patch("winml.modelkit.build.build_onnx_model") as mock_build,
            patch("winml.modelkit.models.auto.get_winml_class") as mock_get_class,
        ):
            mock_build.return_value = _make_build_result(tmp_path)
            mock_get_class.return_value = lambda **kw: MagicMock()

            WinMLAutoModel.from_onnx(
                fake_onnx,
                task="image-classification",
                device="cpu",
            )

        call_kwargs = mock_build.call_args.kwargs
        assert "cache_key" in call_kwargs
        # cache_key must be non-empty and contain the task abbreviation
        assert call_kwargs["cache_key"]
        assert "imgcls" in call_kwargs["cache_key"]


class TestFromOnnxDictDispatch:
    """from_onnx with dict onnx_path delegates to WinMLCompositeModel.from_onnx."""

    def test_dict_dispatches_to_composite(self, tmp_path: Path):
        """Dict onnx_path calls WinMLCompositeModel.from_onnx."""
        with patch(
            "winml.modelkit.models.winml.composite_model.WinMLCompositeModel.from_onnx"
        ) as mock_from_onnx:
            mock_from_onnx.return_value = MagicMock()

            WinMLAutoModel.from_onnx(
                {"encoder": str(tmp_path / "enc.onnx"), "decoder": str(tmp_path / "dec.onnx")},
                task="translation",
                skip_build=True,
            )

            mock_from_onnx.assert_called_once()
            call_kwargs = mock_from_onnx.call_args.kwargs
            assert call_kwargs["task"] == "translation"
            assert call_kwargs["skip_build"] is True

    def test_hf_config_dispatches_composite_via_registry(self, tmp_path: Path):
        """hf_config kwarg threads through so model_type registry lookup works.

        Exercises the real WinMLCompositeModel.from_onnx body via a fake
        subclass in a temporary registry slot. hf_config must be a dedicated
        parameter on WinMLAutoModel.from_onnx (distinct from ``config``, which
        is a WinMLBuildConfig and has no ``model_type`` attribute).
        """
        from winml.modelkit.models.winml.composite_model import (
            COMPOSITE_MODEL_REGISTRY,
            WinMLCompositeModel,
        )

        # Minimal HF-config stand-in: only attribute access (.model_type) is
        # required; no isinstance check happens on hf_config in the dispatch.
        class _FakeHFConfig:
            model_type = "_test_dispatch_model_"

        enc_path = tmp_path / "enc.onnx"
        dec_path = tmp_path / "dec.onnx"
        enc_path.write_bytes(b"fake")
        dec_path.write_bytes(b"fake")

        test_key = ("_test_dispatch_model_", "_test_task_")

        class _FakeComposite(WinMLCompositeModel):
            _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {
                "encoder": "feature-extraction",
                "decoder": "translation",
            }

            def forward(self, **kwargs):  # type: ignore[override]
                pass

        assert test_key not in COMPOSITE_MODEL_REGISTRY
        COMPOSITE_MODEL_REGISTRY[test_key] = _FakeComposite
        try:
            # Patch WinMLAutoModel.from_onnx: outer dict call falls through to
            # the real implementation, inner per-component Path calls mocked.
            _real_from_onnx = WinMLAutoModel.from_onnx
            sub_mock = MagicMock()
            sub_calls: list = []

            def _side_effect(onnx_path, **kw):  # type: ignore[no-untyped-def]
                if isinstance(onnx_path, dict):
                    return _real_from_onnx(onnx_path, **kw)
                sub_calls.append((onnx_path, kw))
                return sub_mock

            with patch.object(WinMLAutoModel, "from_onnx", side_effect=_side_effect):
                result = WinMLAutoModel.from_onnx(
                    {"encoder": str(enc_path), "decoder": str(dec_path)},
                    task="_test_task_",
                    hf_config=_FakeHFConfig(),
                    skip_build=True,
                )

            assert isinstance(result, _FakeComposite)
            assert len(sub_calls) == 2
            tasks_called = {kw["task"] for _, kw in sub_calls}
            assert tasks_called == {"feature-extraction", "translation"}
        finally:
            COMPOSITE_MODEL_REGISTRY.pop(test_key, None)

    def test_from_onnx_dict_without_hf_config_raises(self, tmp_path: Path):
        """Dict dispatch without hf_config surfaces a clear registry-miss error.

        Guards against silent fallback: unregistered ``(model_type, task)`` must
        raise ValueError immediately, not accept a wrong-typed kwarg and mis-dispatch.
        """
        enc_path = tmp_path / "enc.onnx"
        dec_path = tmp_path / "dec.onnx"
        enc_path.write_bytes(b"fake")
        dec_path.write_bytes(b"fake")

        with pytest.raises(ValueError, match="No composite model"):
            WinMLAutoModel.from_onnx(
                {"encoder": str(enc_path), "decoder": str(dec_path)},
                task="_unregistered_task_",
                skip_build=True,
            )
