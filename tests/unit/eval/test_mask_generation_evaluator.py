# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for the mask-generation evaluator (pure helpers + validation).

End-to-end tests that exercise a real ORT session require the SAM 3 ONNX
files cached and are run only in the integration suite -- see
``scripts/sam3_smoke_eval.py`` for the script form.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from winml.modelkit.eval.config import DatasetConfig, WinMLEvaluationConfig
from winml.modelkit.eval.mask_generation_evaluator import (
    _TARGET_SIZE,
    WinMLMaskGenerationEvaluator,
    _build_decoder_inputs,
    _build_providers,
    _postprocess_mask,
    _preprocess_image,
)


# ----------------------------------------------------------------------
# _preprocess_image
# ----------------------------------------------------------------------


class TestPreprocessImage:
    def test_shape_and_dtype(self) -> None:
        img = Image.new("RGB", (640, 480), color=(127, 127, 127))
        pv, _scale_x, _scale_y = _preprocess_image(img)
        assert pv.shape == (1, 3, _TARGET_SIZE, _TARGET_SIZE)
        assert pv.dtype == np.float32

    def test_scale_for_landscape(self) -> None:
        # SAM 3 image processor does a direct resize to 1008x1008; the per-
        # axis scale factors are TARGET / orig_dim independently of aspect.
        img = Image.new("RGB", (800, 400), color=(128, 128, 128))
        _, scale_x, scale_y = _preprocess_image(img)
        assert scale_x == pytest.approx(_TARGET_SIZE / 800)
        assert scale_y == pytest.approx(_TARGET_SIZE / 400)

    def test_no_padding_full_target_filled(self) -> None:
        # Direct resize means every pixel in the output corresponds to a
        # real input pixel -- there is no zero-padded border region.
        img = Image.new("RGB", (200, 100), color=(255, 0, 0))
        pv, _, _ = _preprocess_image(img)
        # Red channel after rescale + (x - 0.5)/0.5 normalization == 1.0
        # everywhere; no zero border.
        assert np.allclose(pv[0, 0], 1.0)

    def test_normalization_applied(self) -> None:
        # SAM 3 normalization: (pixel/255 - 0.5) / 0.5
        # Black input -> (0 - 0.5) / 0.5 == -1.0
        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        pv, _, _ = _preprocess_image(img)
        center_r = pv[0, 0, _TARGET_SIZE // 2, _TARGET_SIZE // 2]
        assert center_r == pytest.approx(-1.0, abs=1e-4)


# ----------------------------------------------------------------------
# _postprocess_mask
# ----------------------------------------------------------------------


class TestPostprocessMask:
    def test_recovers_original_shape(self) -> None:
        # Low-res mask (256x256) -> original 480x640. With direct-resize
        # preprocessing the low-res mask maps 1:1 to the full original
        # image regardless of aspect ratio.
        low = np.random.RandomState(0).rand(256, 256).astype(np.float32) - 0.5
        out = _postprocess_mask(low, orig_h=480, orig_w=640)
        assert out.shape == (480, 640)
        assert out.dtype == bool

    def test_thresholding_at_zero(self) -> None:
        # All-positive logits -> all-True mask
        low = np.ones((256, 256), dtype=np.float32) * 5.0
        out = _postprocess_mask(low, orig_h=100, orig_w=100)
        assert out.all()

        low_neg = -low
        out_neg = _postprocess_mask(low_neg, orig_h=100, orig_w=100)
        assert not out_neg.any()


# ----------------------------------------------------------------------
# _build_decoder_inputs
# ----------------------------------------------------------------------


def _fake_emb() -> dict[str, np.ndarray]:
    return {
        "image_embeddings.0": np.zeros((1, 32, 288, 288), dtype=np.float32),
        "image_embeddings.1": np.zeros((1, 64, 144, 144), dtype=np.float32),
        "image_embeddings.2": np.zeros((1, 256, 72, 72), dtype=np.float32),
    }


class TestBuildDecoderInputsBbox:
    def test_bbox_shape_and_scale(self) -> None:
        prompt = {"bbox": [10, 20, 30, 40]}
        feed = _build_decoder_inputs(
            prompt=prompt,
            prompt_mode="bbox",
            scale_x=2.0,
            scale_y=3.0,
            emb=_fake_emb(),
        )
        # boxes: (1, 1, 4) -- x scaled by 2, y scaled by 3
        assert feed["input_boxes"].shape == (1, 1, 4)
        np.testing.assert_array_almost_equal(
            feed["input_boxes"][0, 0],
            [20.0, 60.0, 60.0, 120.0],
        )
        # points / labels are empty for bbox mode
        assert feed["input_points"].shape == (1, 1, 0, 2)
        assert feed["input_labels"].shape == (1, 1, 0)
        assert feed["input_labels"].dtype == np.int64

    def test_includes_all_three_embeddings(self) -> None:
        prompt = {"bbox": [0, 0, 10, 10]}
        feed = _build_decoder_inputs(
            prompt=prompt,
            prompt_mode="bbox",
            scale_x=1.0,
            scale_y=1.0,
            emb=_fake_emb(),
        )
        for k in ("image_embeddings.0", "image_embeddings.1", "image_embeddings.2"):
            assert k in feed

    def test_uses_embedding_names_requested_by_decoder(self) -> None:
        prompt = {"bbox": [0, 0, 10, 10]}
        embedding = np.zeros((1, 8, 4, 4), dtype=np.float32)
        feed = _build_decoder_inputs(
            prompt=prompt,
            prompt_mode="bbox",
            scale_x=1.0,
            scale_y=1.0,
            emb={"encoder_feature": embedding},
            required_embed_names=("encoder_feature",),
        )

        assert feed["encoder_feature"] is embedding
        assert "image_embeddings.0" not in feed


class TestBuildDecoderInputsPoint:
    def test_point_shape_and_scale(self) -> None:
        prompt = {"point": [15, 25], "label": 1}
        feed = _build_decoder_inputs(
            prompt=prompt,
            prompt_mode="point",
            scale_x=2.0,
            scale_y=3.0,
            emb=_fake_emb(),
        )
        assert feed["input_points"].shape == (1, 1, 1, 2)
        np.testing.assert_array_almost_equal(feed["input_points"][0, 0, 0], [30.0, 75.0])
        # labels = foreground (1)
        assert feed["input_labels"].shape == (1, 1, 1)
        assert feed["input_labels"][0, 0, 0] == 1
        # boxes empty
        assert feed["input_boxes"].shape == (1, 0, 4)

    def test_point_only_decoder_omits_unsupported_box_tensor(self) -> None:
        feed = _build_decoder_inputs(
            prompt={"point": [15, 25], "label": 1},
            prompt_mode="point",
            scale_x=2.0,
            scale_y=3.0,
            emb={
                "image_embeddings": np.zeros((1, 256, 64, 64), dtype=np.float32),
                "image_positional_embeddings": np.zeros((1, 256, 64, 64), dtype=np.float32),
            },
            required_embed_names=("image_embeddings", "image_positional_embeddings"),
            required_input_names=(
                "input_points",
                "input_labels",
                "image_embeddings",
                "image_positional_embeddings",
            ),
        )

        assert "input_boxes" not in feed
        assert set(feed) == {
            "input_points",
            "input_labels",
            "image_embeddings",
            "image_positional_embeddings",
        }


class TestBuildDecoderInputsInvalidMode:
    def test_text_mode_rejected_with_helpful_message(self) -> None:
        with pytest.raises(ValueError, match="Text-prompt"):
            _build_decoder_inputs(
                prompt={"text": "person"},
                prompt_mode="text",
                scale_x=1.0,
                scale_y=1.0,
                emb=_fake_emb(),
            )


# ----------------------------------------------------------------------
# _build_providers
# ----------------------------------------------------------------------


class TestBuildProviders:
    def test_cpu_always_works(self) -> None:
        providers, opts = _build_providers("cpu")
        assert providers == ["CPUExecutionProvider"]
        assert opts == [{}]

    def test_full_provider_name_is_honored(self, monkeypatch) -> None:
        import onnxruntime as ort

        monkeypatch.setattr(
            ort,
            "get_available_providers",
            lambda: ["QNNExecutionProvider", "CPUExecutionProvider"],
        )

        providers, opts = _build_providers("QNNExecutionProvider")

        assert providers == ["QNNExecutionProvider", "CPUExecutionProvider"]
        assert opts == [{}, {}]

    def test_unavailable_requested_ep_raises(self, monkeypatch) -> None:
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])

        with pytest.raises(ValueError, match=r"QNNExecutionProvider.*not available"):
            _build_providers("qnn")

    def test_device_without_ep_uses_resolved_device_ep(self, monkeypatch) -> None:
        import onnxruntime as ort

        monkeypatch.setattr(
            "winml.modelkit.sysinfo.resolve_eps",
            lambda device: ["QNNExecutionProvider"] if device == "npu" else [],
        )
        monkeypatch.setattr(
            ort,
            "get_available_providers",
            lambda: ["QNNExecutionProvider", "CPUExecutionProvider"],
        )

        providers, _opts = _build_providers(None, device="npu")

        assert providers[0] == "QNNExecutionProvider"


# ----------------------------------------------------------------------
# Evaluator constructor validation
# ----------------------------------------------------------------------


def _make_config(model_path) -> WinMLEvaluationConfig:
    ds = DatasetConfig(path="mattmdjaga/human_parsing_dataset", split="train", samples=2)
    return WinMLEvaluationConfig(
        model_id="onnx-community/sam3-tracker-ONNX",
        task="mask-generation",
        model_path=model_path,
        dataset=ds,
        device="cpu",
        ep="cpu",
    )


class TestEvaluatorValidation:
    def test_rejects_single_model_path(self) -> None:
        cfg = _make_config("some/path.onnx")
        with pytest.raises(TypeError, match="role=path"):
            WinMLMaskGenerationEvaluator(cfg, model=None)

    def test_rejects_missing_decoder_role(self) -> None:
        cfg = _make_config({"image-encoder": "enc.onnx"})
        with pytest.raises(ValueError, match="prompt-decoder"):
            WinMLMaskGenerationEvaluator(cfg, model=None)

    def test_rejects_missing_encoder_role(self) -> None:
        cfg = _make_config({"prompt-decoder": "dec.onnx"})
        with pytest.raises(ValueError, match="image-encoder"):
            WinMLMaskGenerationEvaluator(cfg, model=None)


class _NoValidSamples:
    def __len__(self) -> int:
        return 0

    def iter_valid(self, max_samples: int | None = None):
        return iter(())


class TestComputeFailures:
    def test_raises_when_no_samples_are_evaluated(self) -> None:
        evaluator = object.__new__(WinMLMaskGenerationEvaluator)
        evaluator.config = _make_config({"image-encoder": "enc.onnx", "prompt-decoder": "dec.onnx"})
        evaluator.data = _NoValidSamples()

        with pytest.raises(RuntimeError, match="processed 0 valid samples"):
            evaluator.compute()


# ----------------------------------------------------------------------
# Registry wiring
# ----------------------------------------------------------------------


class TestEvaluatorRegistered:
    def test_task_resolves_to_evaluator(self) -> None:
        from winml.modelkit.eval import WinMLEvaluationConfig
        from winml.modelkit.eval.evaluate import get_evaluator_class

        cls = get_evaluator_class(WinMLEvaluationConfig(task="mask-generation"))
        assert cls is WinMLMaskGenerationEvaluator

    def test_keypoint_detection_remains_registered(self) -> None:
        from winml.modelkit.eval import WinMLEvaluationConfig
        from winml.modelkit.eval.evaluate import _DEFAULT_DATASETS, get_evaluator_class
        from winml.modelkit.eval.keypoint_detection_evaluator import (
            WinMLKeypointDetectionEvaluator,
        )
        from winml.modelkit.utils.eval_utils import TASK_SCHEMAS

        cls = get_evaluator_class(WinMLEvaluationConfig(task="keypoint-detection"))

        assert cls is WinMLKeypointDetectionEvaluator
        assert "keypoint-detection" in TASK_SCHEMAS
        assert (
            _DEFAULT_DATASETS["keypoint-detection"]["columns_mapping"]["keypoints_key"]
            == "keypoints"
        )

    def test_default_dataset_registered(self) -> None:
        from winml.modelkit.eval.evaluate import _DEFAULT_DATASETS

        default = _DEFAULT_DATASETS["mask-generation"]
        assert default["path"] == "mattmdjaga/human_parsing_dataset"
        assert default["split"] == "train"


# ----------------------------------------------------------------------
# Profile-aware preprocessing / postprocessing (SAM 2 family)
# ----------------------------------------------------------------------


from winml.modelkit.eval.mask_generation_evaluator import (  # noqa: E402
    SAM2_PROFILE,
    SAM3_PROFILE,
    _postprocess_for_profile,
    _preprocess_for_profile,
    _resolve_profile,
)


class TestPreprocessForProfileSam2:
    def test_shape_and_dtype(self) -> None:
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        pv, sx, sy, new_h, new_w = _preprocess_for_profile(SAM2_PROFILE, img)
        # SAM 2 target is 1024x1024
        assert pv.shape == (1, 3, 1024, 1024)
        assert pv.dtype == np.float32
        # uniform longest-side scale (single factor used for both axes)
        assert sx == pytest.approx(sy)
        # 640 is the longer side -> scale = 1024 / 640 = 1.6
        assert sx == pytest.approx(1024 / 640)
        # post-resize content dims, pre-pad: w fills target, h shorter
        assert new_w == 1024
        assert new_h == round(480 * 1024 / 640)

    def test_zero_padding_present_in_letterbox_region(self) -> None:
        # ImageNet mean ~0.485; with letterbox padding using zero in the
        # raw pixel domain, post-normalization the pad region equals
        # (0 - mean) / std ~ -2.1 for R channel; the content region is
        # bounded.  We just check that the bottom rows are equal across
        # x (a constant value indicating a uniform pad) and != to the
        # content rows.
        img = Image.new("RGB", (640, 320), color=(255, 255, 255))
        pv, _sx, _sy, new_h, _new_w = _preprocess_for_profile(SAM2_PROFILE, img)
        assert new_h < 1024  # there IS a pad region
        pad_row = pv[0, 0, -1, :]
        content_row = pv[0, 0, new_h // 2, :]
        # pad row is uniform (single value across x); content row is uniform
        # too because input is white, so distinguish by VALUE not constancy.
        assert not np.allclose(pad_row, content_row)

    def test_imagenet_normalization(self) -> None:
        # White input: (1.0 - 0.485) / 0.229 ~ 2.248 for R channel.
        img = Image.new("RGB", (1024, 1024), color=(255, 255, 255))
        pv, _, _, _, _ = _preprocess_for_profile(SAM2_PROFILE, img)
        center_r = pv[0, 0, 512, 512]
        assert center_r == pytest.approx((1.0 - 0.485) / 0.229, abs=1e-3)


class TestPostprocessForProfileSam2:
    def test_recovers_original_shape_with_crop_then_resize(self) -> None:
        # Letterbox-padded encoder input was 1024x1024; content region
        # was 768x1024 (i.e. shorter side padded).  Postprocess should
        # crop low-res back to that content aspect, then resize to the
        # original 600x800 image.
        low = np.random.RandomState(0).rand(256, 256).astype(np.float32) - 0.5
        out = _postprocess_for_profile(
            SAM2_PROFILE,
            low,
            orig_h=600,
            orig_w=800,
            new_h=768,
            new_w=1024,
        )
        assert out.shape == (600, 800)
        assert out.dtype == bool

    def test_thresholding_at_zero(self) -> None:
        low = np.ones((256, 256), dtype=np.float32) * 5.0
        out = _postprocess_for_profile(
            SAM2_PROFILE,
            low,
            orig_h=100,
            orig_w=100,
            new_h=1024,
            new_w=1024,
        )
        assert out.all()
        out_neg = _postprocess_for_profile(
            SAM2_PROFILE,
            -low,
            orig_h=100,
            orig_w=100,
            new_h=1024,
            new_w=1024,
        )
        assert not out_neg.any()


# ----------------------------------------------------------------------
# _resolve_profile -- dispatch logic
# ----------------------------------------------------------------------


class _StubInput:
    def __init__(self, shape):
        self.shape = shape


class _StubSession:
    def __init__(self, shape):
        self._shape = shape

    def get_inputs(self):
        return [_StubInput(self._shape)]


def _cfg(model_id: str) -> WinMLEvaluationConfig:
    ds = DatasetConfig(path="mattmdjaga/human_parsing_dataset", split="train", samples=2)
    return WinMLEvaluationConfig(
        model_id=model_id,
        task="mask-generation",
        model_path={"image-encoder": "enc.onnx", "prompt-decoder": "dec.onnx"},
        dataset=ds,
        device="cpu",
        ep="cpu",
    )


class TestResolveProfile:
    def test_shape_signal_picks_sam2(self) -> None:
        sess = _StubSession([1, 3, 1024, 1024])
        prof = _resolve_profile(_cfg("ambiguous/name"), sess)
        assert prof is SAM2_PROFILE

    def test_shape_signal_picks_sam3(self) -> None:
        sess = _StubSession([1, 3, 1008, 1008])
        prof = _resolve_profile(_cfg("ambiguous/name"), sess)
        assert prof is SAM3_PROFILE

    def test_falls_back_to_model_id_sam2(self) -> None:
        # Dynamic shape (strings) -> fall through to model_id matching
        sess = _StubSession([1, 3, "H", "W"])
        prof = _resolve_profile(
            _cfg("onnx-community/sam2.1-hiera-small-ONNX"),
            sess,
        )
        assert prof is SAM2_PROFILE

    def test_falls_back_to_model_id_sam3(self) -> None:
        sess = _StubSession([1, 3, "H", "W"])
        prof = _resolve_profile(
            _cfg("onnx-community/sam3-tracker-ONNX"),
            sess,
        )
        assert prof is SAM3_PROFILE

    def test_default_is_sam3(self) -> None:
        sess = _StubSession([1, 3, "H", "W"])
        prof = _resolve_profile(_cfg("unknown/family"), sess)
        assert prof is SAM3_PROFILE
