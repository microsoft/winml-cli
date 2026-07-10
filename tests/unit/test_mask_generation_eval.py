# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "mask_generation_eval.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("mask_generation_eval", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestProfileEpNotice:
    def test_sam3_dml_returns_detailed_notice(self):
        module = _load_script_module()

        notice = module._profile_ep_notice(module.SAM3_PROFILE, "dml")

        assert notice is not None
        assert "not validated on DML" in notice
        assert "int8" in notice
        assert "fp16" in notice
        assert "use --ep cpu" in notice

    def test_non_risky_combo_returns_none(self):
        module = _load_script_module()

        assert module._profile_ep_notice(module.SAM3_PROFILE, "cpu") is None


class TestPreprocessProfiles:
    def test_sam3_uses_direct_resize_and_independent_axis_scales(self):
        module = _load_script_module()
        image = Image.new("RGB", (800, 400), color=(128, 128, 128))

        pixel_values, scale_x, scale_y, new_h, new_w = module.preprocess_image(
            image,
            module.SAM3_PROFILE,
        )

        assert pixel_values.shape == (1, 3, 1008, 1008)
        assert scale_x == pytest.approx(1008 / 800)
        assert scale_y == pytest.approx(1008 / 400)
        assert (new_h, new_w) == (1008, 1008)
