# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for WinMLAutoModel.from_pretrained kwarg handling.

Covers two fixes:
1. EP fix: inference wrapper must receive the user's explicit --ep, not the
   compile-time resolved_ep derived from --device.
2. no_compile: build_config.compile must be set to None after policy resolution
   when no_compile=True is passed, because generate_hf_build_config's device
   policy (STEP 4.5) always overwrites compile after the override merge.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Patch targets for auto.py
# - Module-level imports: patch via winml.modelkit.models.auto.<name>
# - Local (in-function) imports: patch via the source module
_AUTO = "winml.modelkit.models.auto"
_GENERATE_HF_BUILD_CONFIG = "winml.modelkit.config.generate_hf_build_config"
_BUILD_HF_MODEL = "winml.modelkit.build.build_hf_model"


def _make_build_config(compile_provider: str | None = "qnn"):
    """Return a minimal WinMLBuildConfig with controllable compile field."""
    from winml.modelkit.compiler.configs import EPConfig, WinMLCompileConfig
    from winml.modelkit.config import WinMLBuildConfig

    cfg = WinMLBuildConfig()
    cfg.compile = (
        WinMLCompileConfig(ep_config=EPConfig(provider=compile_provider))
        if compile_provider
        else None
    )
    cfg.loader.task = "image-classification"
    return cfg


def _mock_winml_class():
    """MagicMock that works as a class: has __name__ set."""
    m = MagicMock()
    m.__name__ = "MockWinMLClass"
    return m


def _base_patches(build_cfg, build_side_effect=None, winml_class=None):
    """Return a list of context managers that stub out the heavy internals."""
    if winml_class is None:
        winml_class = _mock_winml_class()
    return [
        patch(_GENERATE_HF_BUILD_CONFIG, return_value=build_cfg),
        patch(
            f"{_AUTO}.load_hf_model",
            return_value=(MagicMock(), MagicMock(model_type="resnet"), MagicMock()),
        ),
        patch(f"{_AUTO}.get_cache_dir", return_value=MagicMock()),
        patch(f"{_AUTO}.get_model_dir", return_value=MagicMock()),
        patch(f"{_AUTO}.get_cache_key", return_value="key"),
        patch(
            _BUILD_HF_MODEL,
            side_effect=build_side_effect,
            return_value=MagicMock(final_onnx_path="model.onnx"),
        ),
        patch(f"{_AUTO}.get_winml_class", return_value=winml_class),
    ]


# ---------------------------------------------------------------------------
# EP fix tests
# ---------------------------------------------------------------------------


class TestFromPretrainedEpFix:
    """Inference wrapper must NOT receive the compile-derived EP.

    Before the fix, from_pretrained passed ep=resolved_ep (e.g. "qnn") to
    the inference wrapper even when the user only specified --device npu.
    This caused WinMLSession to enter explicit-EP mode and log a warning on
    machines without QNN hardware.

    After the fix, ep=kwargs.get("ep") is passed, so:
    - No explicit --ep  → wrapper gets ep=None  → device policy applies
    - Explicit --ep qnn → wrapper gets ep="qnn" → explicit mode (intended)
    """

    def _call_and_capture_wrapper_ep(self, extra_kwargs: dict):
        from winml.modelkit.models.auto import WinMLAutoModel

        build_cfg = _make_build_config(compile_provider="qnn")
        mock_winml_class = _mock_winml_class()
        patches = _base_patches(build_cfg, winml_class=mock_winml_class)

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            WinMLAutoModel.from_pretrained("some/model", **extra_kwargs)

        return mock_winml_class.call_args.kwargs.get("ep")

    def test_no_explicit_ep_wrapper_gets_none(self):
        """Without --ep, inference wrapper should receive ep=None (device policy)."""
        ep = self._call_and_capture_wrapper_ep({"device": "npu"})
        assert ep is None, (
            f"Expected ep=None (device policy), got ep={ep!r}. "
            "The wrapper should not receive the compile-derived EP."
        )

    def test_explicit_ep_forwarded_to_wrapper(self):
        """With --ep qnn, inference wrapper should receive ep='qnn'."""
        ep = self._call_and_capture_wrapper_ep({"device": "npu", "ep": "qnn"})
        assert ep == "qnn", f"Expected ep='qnn' (user's explicit EP), got ep={ep!r}."

    def test_no_ep_no_device_wrapper_gets_none(self):
        """Without --ep or --device, wrapper should receive ep=None."""
        ep = self._call_and_capture_wrapper_ep({})
        assert ep is None


# ---------------------------------------------------------------------------
# no_compile kwarg tests
# ---------------------------------------------------------------------------


class TestFromPretrainedNoCompile:
    """build_config.compile must be None when no_compile=True.

    generate_hf_build_config's STEP 4.5 always re-sets compile from device
    policy, so the override dict approach doesn't work. The fix applies
    no_compile AFTER generate_hf_build_config returns.
    """

    def _call_and_capture_compile(self, extra_kwargs: dict):
        from winml.modelkit.models.auto import WinMLAutoModel

        build_cfg = _make_build_config(compile_provider="qnn")
        captured = {}

        def fake_build(**kwargs):
            captured["compile"] = kwargs["config"].compile
            return MagicMock(final_onnx_path="model.onnx")

        patches = _base_patches(build_cfg, build_side_effect=fake_build)

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            WinMLAutoModel.from_pretrained("some/model", **extra_kwargs)

        return captured.get("compile")

    def test_no_compile_true_clears_compile(self):
        """no_compile=True must set build_config.compile=None before build."""
        compile_val = self._call_and_capture_compile({"no_compile": True})
        assert compile_val is None, (
            f"build_config.compile should be None when no_compile=True, got {compile_val!r}"
        )

    def test_no_compile_false_preserves_compile(self):
        """no_compile=False must leave build_config.compile intact."""
        compile_val = self._call_and_capture_compile({"no_compile": False})
        assert compile_val is not None, (
            "build_config.compile should not be cleared when no_compile=False"
        )

    def test_no_compile_absent_preserves_compile(self):
        """Omitting no_compile kwarg must leave build_config.compile intact."""
        compile_val = self._call_and_capture_compile({})
        assert compile_val is not None, (
            "build_config.compile should not be cleared when no_compile is not passed"
        )
