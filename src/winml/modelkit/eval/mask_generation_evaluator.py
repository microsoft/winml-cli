# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Promptable mask-generation evaluator for SAM-family ONNX models.

Does *not* go through HF's ``pipeline`` / ``evaluate`` libraries because:

1. HF's ``mask-generation`` task is a high-level wrapper around the
   *full* PyTorch SAM model -- it isn't compatible with raw ORT sessions.
2. Mask-generation here is *composite*: encoder + decoder must be
   orchestrated manually (the same as :file:`scripts/sam3_smoke_eval.py`
   does informally).  The base :class:`WinMLEvaluator`'s single-model
   pipeline assumption doesn't fit.

The evaluator instead drives two ORT sessions directly:

* **image-encoder** -- consumes ``pixel_values``, emits 3 multi-scale
  image embeddings (``image_embeddings.0/1/2`` for SAM 3).
* **prompt-decoder** -- consumes a prompt (bbox or point) plus the
  embeddings, emits up to 3 candidate masks plus their predicted IoU.

For each sample we derive the prompt from the GT mask (so we're measuring
the model's ability to *trace boundaries* given a known prompt -- the
standard SAM eval setup), pick the highest predicted-IoU mask, map it
back to the original image resolution, and accumulate mIoU + Dice via
:class:`~winml.modelkit.eval.metrics.BinarySegmentationMetric`.

Text-prompt mode is intentionally not implemented yet -- the publicly
cached SAM 3 ONNX decoder does not accept a text input port (see
``input_points``/``input_labels``/``input_boxes`` only).  Text-concept
prompting requires a separate text-encoder ONNX that is not yet on the
Hub; tracked as a follow-up.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from PIL import Image
    from transformers.pipelines.base import Pipeline

    from ..models.winml.base import WinMLPreTrainedModel
    from ..utils.constants import EPNameOrAlias
    from .config import WinMLEvaluationConfig


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Per-family preprocessing profiles.
#
# The ONNX-community SAM 2.1 and SAM 3 exports share the *decoder* I/O
# schema (``input_points``/``input_labels``/``input_boxes`` ->
# ``iou_scores``/``pred_masks``/``object_score_logits``) and the encoder
# output names (``image_embeddings.{0,1,2}``); only the *image* side
# differs:
#
# * SAM 3 Tracker: direct bilinear resize to 1008x1008 (no padding) with
#   mean/std = 0.5/0.5 (preprocessor_config.json on
#   onnx-community/sam3-tracker-ONNX).
# * SAM 2.1: longest-side bilinear resize to 1024 with zero-pad to a
#   1024x1024 square; ImageNet mean/std.  Matches the SAM-paper
#   convention and ``onnx-community/sam2.1-hiera-small-ONNX``.
#
# A profile bundles those constants together.  The active profile is
# resolved per-evaluator from the encoder's static ``pixel_values`` shape
# (falling back to a ``model_id`` substring heuristic, then SAM 3).
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _MaskGenProfile:
    """Per-family preprocessing constants for a SAM-style ONNX export."""

    name: str
    target_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    # "direct"           -> resize per-axis to target_size x target_size
    # "longest_side_pad" -> longest-side resize, zero-pad bottom/right to square
    resize_mode: str


SAM3_PROFILE = _MaskGenProfile(
    name="sam3",
    target_size=1008,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    resize_mode="direct",
)


SAM2_PROFILE = _MaskGenProfile(
    name="sam2",
    target_size=1024,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    resize_mode="longest_side_pad",
)


PROMPT_ENCODER_PROFILE = _MaskGenProfile(
    name="prompt-encoder",
    target_size=1024,
    mean=(0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0),
    resize_mode="direct",
)

_RELEASE_METADATA_NAME = "winml_release_metadata.json"


# Back-compat module-level SAM 3 constant (preserved so existing imports
# from tests/scripts keep working unchanged).
_TARGET_SIZE = SAM3_PROFILE.target_size

__all__ = ["_TARGET_SIZE", "WinMLMaskGenerationEvaluator"]


class WinMLMaskGenerationEvaluator(WinMLEvaluator):
    """Evaluator for SAM-style promptable mask generation.

    Constructor accepts the standard ``(config, model)`` signature so the
    registry dispatch in :mod:`~winml.modelkit.eval.evaluate` works
    unmodified.  The ``model`` argument may be ``None`` -- this evaluator
    reads ``config.model_path`` (a ``dict[str, str]`` mapping
    ``image-encoder`` / ``prompt-decoder`` to ONNX file paths) and
    constructs its own ORT sessions, bypassing the
    ``WinMLAutoModel`` composite-registry path.
    """

    # Required sub-model role names (must appear as keys in
    # ``config.model_path`` when it is a dict).
    _ENCODER_ROLE = "image-encoder"
    _DECODER_ROLE = "prompt-decoder"

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel | None,
    ) -> None:
        if not isinstance(config.model_path, dict):
            raise TypeError(
                "Mask-generation evaluation requires composite `-m role=path` "
                "model arguments.  Pass --model image-encoder=<enc.onnx> and "
                f"--model {self._DECODER_ROLE}=<dec.onnx>.",
            )
        for role in (self._ENCODER_ROLE, self._DECODER_ROLE):
            if role not in config.model_path:
                raise ValueError(
                    f"Missing required `-m {role}=<path>` argument.  "
                    f"Got roles: {sorted(config.model_path)}.",
                )

        # Pre-seed the attributes that ``prepare_data`` (invoked from the
        # base ``WinMLEvaluator.__init__``) depends on.  ``self.config``
        # is needed by ``_load_sessions`` (it reads ``config.model_path``
        # and ``config.ep``) so we set it before calling super.
        self.config = config
        mapping = config.dataset.columns_mapping or {}
        self._prompt_mode: str = mapping.get("prompt_mode", "bbox")
        if self._prompt_mode not in {"bbox", "point"}:
            raise ValueError(
                f"Unsupported prompt_mode={self._prompt_mode!r} for mask-generation evaluation. "
                "Use prompt_mode='bbox' or 'point'."
            )
        self._enc_sess, self._dec_sess = self._load_sessions()
        self._encoder_prompt_inputs = _resolve_point_prompt_inputs(self._enc_sess)
        self._decoder_input_names = _node_names(self._dec_sess.get_inputs())
        self._decoder_prompt_inputs = _resolve_point_prompt_inputs(self._dec_sess)
        if self._encoder_prompt_inputs and self._decoder_prompt_inputs:
            raise ValueError(
                "Prompt inputs are present on both composite components; routing is ambiguous."
            )
        if not self._encoder_prompt_inputs and not self._decoder_prompt_inputs:
            raise ValueError("Neither composite component exposes a point-prompt contract.")
        self._prompt_component = (
            self._ENCODER_ROLE if self._encoder_prompt_inputs else self._DECODER_ROLE
        )
        if self._prompt_component == self._ENCODER_ROLE:
            if "prompt_mode" in mapping and self._prompt_mode == "bbox":
                raise ValueError(
                    "The encoder prompt contract accepts points only; use prompt_mode='point'."
                )
            self._prompt_mode = "point"
        elif "input_boxes" not in self._decoder_input_names:
            if "prompt_mode" in mapping and self._prompt_mode == "bbox":
                raise ValueError(
                    "The decoder does not accept box prompts; use prompt_mode='point'."
                )
            self._prompt_mode = "point"
        self._encoder_input_name = _resolve_encoder_input_name(
            self._enc_sess,
            excluded=set(self._encoder_prompt_inputs or ()),
        )
        self._encoder_output_names = _node_names(self._enc_sess.get_outputs())
        self._embedding_input_names = _resolve_embedding_input_names(
            self._dec_sess,
            self._encoder_output_names,
        )
        self._mask_output_name, self._score_output_name = _resolve_decoder_output_names(
            self._dec_sess
        )
        # Pick the per-family preprocessing profile from the encoder's
        # static input shape (falling back to a model_id heuristic, then
        # SAM 3).  Threaded through preprocess + postprocess in _predict.
        self._profile = _resolve_profile(self.config, self._enc_sess)
        logger.info(
            "Mask-generation profile: %s (target=%d, mean=%s, std=%s, resize=%s)",
            self._profile.name,
            self._profile.target_size,
            self._profile.mean,
            self._profile.std,
            self._profile.resize_mode,
        )

        # Defer the rest of attribute setup (``self.model``, ``self.data``,
        # ``self.pipe``) to the base class so we satisfy the evaluator
        # contract and CodeQL's ``py/missing-call-to-init`` rule.  The
        # base ``prepare_pipeline`` is overridden here to return ``None``,
        # so it is safe to call from ``WinMLEvaluator.__init__``.
        super().__init__(config, model)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # WinMLEvaluator overrides
    # ------------------------------------------------------------------

    def prepare_data(self) -> Any:
        """Build a :class:`MaskGenerationDataset` from ``config.dataset``."""
        from ..datasets.mask_generation import MaskGenerationDataset

        ds = self.config.dataset
        mapping = ds.columns_mapping or {}

        # ``model_name`` is required by ``BaseTaskDataset`` for API parity
        # with other datasets; mask-generation does not actually consult
        # the model's image processor.  Fall back to a safe sentinel when
        # ``model_id`` is unset (composite mask-gen sometimes runs without
        # a single canonical model_id).
        return MaskGenerationDataset(
            model_name=self.config.model_id or "sam-mask-generation",
            dataset_name=ds.path or MaskGenerationDataset.DEFAULT_DATASET,
            dataset_config_name=ds.name,
            data_split=ds.split or MaskGenerationDataset.DEFAULT_SPLIT,
            max_samples=ds.samples,
            prompt_mode=self._prompt_mode,  # type: ignore[arg-type]
            image_col=mapping.get("input_column"),
            mask_col=mapping.get("mask_column"),
            text_col=mapping.get("text_column"),
            revision=ds.revision,
            streaming=ds.streaming,
            shuffle=ds.shuffle,
            seed=ds.seed,
        )

    def prepare_pipeline(self) -> Pipeline | None:  # type: ignore[override]
        """No HF pipeline -- ORT sessions are driven directly in ``compute``."""
        return None

    def compute(self) -> dict[str, Any]:
        """Run mask-generation eval and return mIoU / Dice."""
        from .metrics import BinarySegmentationMetric

        metric = BinarySegmentationMetric()
        # ``self.data`` length is the *over-fetch* candidate window the
        # dataset built to absorb coverage-filter drops; the user's actual
        # requested count lives in ``config.dataset.samples``.  Iterating
        # past that would silently inflate cost (we saw 23 evaluations for
        # ``--samples 3`` before this cap).
        requested = self.config.dataset.samples
        logger.info(
            "Mask-generation eval: requesting %d samples (candidate window=%d)",
            requested,
            len(self.data),
        )

        processed = 0
        for sample in self.data.iter_valid(max_samples=requested):
            try:
                pred_mask = self._predict(sample)
            except Exception as e:
                logger.warning(
                    "Skipping sample %s: prediction failed (%s)",
                    sample.get("sample_id", "?"),
                    e,
                )
                continue
            metric.update(pred_mask, sample["gt_mask"])
            processed += 1
            if processed % 10 == 0:
                logger.info("  processed %d samples", processed)

        if processed == 0:
            raise RuntimeError(
                "Mask-generation eval processed 0 valid samples. Check dataset "
                "columns, mask coverage filters, model I/O, and EP availability."
            )

        return metric.compute()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_sessions(self) -> tuple[Any, Any]:
        """Construct ORT sessions for the encoder + decoder."""
        import onnxruntime as ort

        paths = self.config.model_path
        assert isinstance(paths, dict)  # already validated in __init__

        providers, provider_options = _build_providers(
            self.config.ep,
            device=self.config.device,
        )
        logger.info(
            "Creating ORT sessions for mask-generation (providers=%s)",
            providers,
        )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        enc = ort.InferenceSession(
            paths[self._ENCODER_ROLE],
            sess_options=sess_opts,
            providers=providers,
            provider_options=provider_options,
        )
        dec = ort.InferenceSession(
            paths[self._DECODER_ROLE],
            sess_options=sess_opts,
            providers=providers,
            provider_options=provider_options,
        )
        logger.info("  encoder providers: %s", enc.get_providers())
        logger.info("  decoder providers: %s", dec.get_providers())
        return enc, dec

    def _predict(self, sample: dict[str, Any]) -> np.ndarray:
        """Run encoder + decoder for one sample, return binary mask."""
        image = sample["image"]
        gt = sample["gt_mask"]
        prompt = sample["prompt"]

        pixel_values, scale_x, scale_y, new_h, new_w = _preprocess_for_profile(
            self._profile,
            image,
        )
        encoder_feed = {self._encoder_input_name: pixel_values}
        if self._encoder_prompt_inputs is not None:
            encoder_feed.update(
                _build_encoder_prompt_inputs(
                    prompt=prompt,
                    prompt_mode=self._prompt_mode,
                    session=self._enc_sess,
                    coordinate_name=self._encoder_prompt_inputs[0],
                    label_name=self._encoder_prompt_inputs[1],
                    original_width=image.size[0],
                    original_height=image.size[1],
                )
            )
        enc_out = self._enc_sess.run(None, encoder_feed)
        emb = dict(zip(self._encoder_output_names, enc_out, strict=True))

        if self._encoder_prompt_inputs is not None:
            dec_inputs = {name: emb[name] for name in self._embedding_input_names}
        else:
            dec_inputs = _build_decoder_inputs(
                prompt=prompt,
                prompt_mode=self._prompt_mode,
                scale_x=scale_x,
                scale_y=scale_y,
                emb=emb,
                required_embed_names=self._embedding_input_names,
                required_input_names=self._decoder_input_names,
            )
        dec_out = self._dec_sess.run(
            [self._score_output_name, self._mask_output_name],
            dec_inputs,
        )
        dec_by_name = dict(
            zip((self._score_output_name, self._mask_output_name), dec_out, strict=True)
        )
        best_low_res = _select_best_mask(
            dec_by_name[self._mask_output_name],
            dec_by_name[self._score_output_name],
        )

        return _postprocess_for_profile(
            self._profile,
            best_low_res,
            orig_h=gt.shape[0],
            orig_w=gt.shape[1],
            new_h=new_h,
            new_w=new_w,
        )


# ----------------------------------------------------------------------
# Pure helpers (kept at module scope so they're easy to test in isolation)
# ----------------------------------------------------------------------


def _resolve_profile(
    config: WinMLEvaluationConfig,
    enc_sess: Any,
) -> _MaskGenProfile:
    """Pick the per-family preprocessing profile for the active model.

    Resolution priority:

    1. **Encoder static input shape**.  If the encoder's ``pixel_values``
       has a static last dim that matches a registered profile's
       ``target_size`` (e.g. 1024 -> SAM 2, 1008 -> SAM 3), use that.
       This is the most reliable signal because it comes from the actual
       ONNX export.
    2. **``config.model_id`` substring**.  Falls back to matching common
       family identifiers (``sam2`` / ``sam-2`` -> SAM 2;
       ``sam3`` / ``sam-3`` -> SAM 3) when the encoder shape is dynamic.
    3. **Default SAM 3** -- preserves the original evaluator behaviour.
    """
    if _resolve_point_prompt_inputs(enc_sess) is not None:
        return _resolve_release_profile(config, enc_sess)

    known = (SAM3_PROFILE, SAM2_PROFILE)

    try:
        shape = enc_sess.get_inputs()[0].shape
    except Exception:
        shape = []
    if len(shape) >= 4 and isinstance(shape[-1], int):
        for prof in known:
            if shape[-1] == prof.target_size:
                return prof

    mid = (config.model_id or "").lower()
    if "sam2" in mid or "sam-2" in mid:
        return SAM2_PROFILE
    if "sam3" in mid or "sam-3" in mid:
        return SAM3_PROFILE

    return SAM3_PROFILE


def _resolve_release_profile(config: WinMLEvaluationConfig, enc_sess: Any) -> _MaskGenProfile:
    """Resolve image preprocessing from a release runtime contract, never an ID heuristic."""
    if not isinstance(config.model_path, dict):
        raise TypeError("Release-backed mask generation requires composite model paths.")
    encoder_path = Path(config.model_path[WinMLMaskGenerationEvaluator._ENCODER_ROLE])
    candidates = (
        encoder_path.parent / _RELEASE_METADATA_NAME,
        encoder_path.parent / "metadata.json",
    )
    metadata_path = next((path for path in candidates if path.is_file()), None)
    if metadata_path is None:
        raise ValueError(
            "A prompt-bearing encoder requires persisted release runtime metadata; "
            f"expected {_RELEASE_METADATA_NAME!r} beside {encoder_path.name!r}."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        model_files = metadata["model_files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid release runtime metadata {metadata_path}: {error}") from error
    if not isinstance(model_files, dict):
        raise TypeError("Release runtime metadata 'model_files' must be an object.")

    session_inputs = {node.name: node for node in enc_sess.get_inputs()}
    matching = [
        contract
        for contract in model_files.values()
        if isinstance(contract, dict)
        and isinstance(contract.get("inputs"), dict)
        and set(contract["inputs"]) == set(session_inputs)
    ]
    if len(matching) != 1:
        raise ValueError(
            "Release runtime metadata must contain exactly one graph contract matching "
            f"encoder inputs {sorted(session_inputs)}; found {len(matching)}."
        )
    inputs = matching[0]["inputs"]
    image_names = [
        name
        for name, spec in inputs.items()
        if isinstance(spec, dict) and spec.get("io_type") == "image"
    ]
    if len(image_names) != 1:
        raise ValueError(
            "Release runtime metadata must identify exactly one encoder input as io_type=image."
        )
    image_name = image_names[0]
    image_spec = inputs[image_name]
    shape = image_spec.get("shape")
    value_range = image_spec.get("value_range")
    session_shape = list(getattr(session_inputs[image_name], "shape", ()))
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or shape != session_shape
        or shape[1] != 3
        or not isinstance(shape[-1], int)
        or shape[-2] != shape[-1]
        or image_spec.get("dtype") != "float32"
        or value_range != [0.0, 1.0]
    ):
        raise ValueError(
            "Unsupported or inconsistent release image contract; expected a matching "
            "square NCHW RGB float32 input with value_range [0.0, 1.0]."
        )
    return _MaskGenProfile(
        name=PROMPT_ENCODER_PROFILE.name,
        target_size=shape[-1],
        mean=PROMPT_ENCODER_PROFILE.mean,
        std=PROMPT_ENCODER_PROFILE.std,
        resize_mode=PROMPT_ENCODER_PROFILE.resize_mode,
    )


def _node_names(nodes: Any) -> tuple[str, ...]:
    """Return ORT input/output names, rejecting unnamed nodes."""
    names = tuple(getattr(node, "name", "") for node in nodes)
    if any(not name for name in names):
        raise ValueError(f"ORT session contains unnamed I/O nodes: {names}")
    return names


def _resolve_encoder_input_name(enc_sess: Any, *, excluded: set[str] | None = None) -> str:
    """Pick the encoder image input name from the actual ONNX session."""
    excluded = excluded or set()
    inputs = [node for node in enc_sess.get_inputs() if node.name not in excluded]
    names = _node_names(inputs)
    if not names:
        raise ValueError("Encoder ONNX session has no inputs.")
    if "pixel_values" in names:
        return "pixel_values"
    rank4 = [str(node.name) for node in inputs if len(getattr(node, "shape", ())) == 4]
    if len(rank4) == 1:
        return rank4[0]
    if len(names) == 1:
        return names[0]
    raise ValueError(
        "Could not identify encoder image input. Expected a single input or "
        f"'pixel_values'; got {list(names)}."
    )


def _resolve_embedding_input_names(
    dec_sess: Any,
    encoder_output_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Match decoder embedding inputs to encoder output names."""
    decoder_inputs = _node_names(dec_sess.get_inputs())
    embedding_inputs = tuple(name for name in decoder_inputs if name in encoder_output_names)
    if not embedding_inputs:
        raise ValueError(
            "Decoder ONNX session has no embedding inputs matching encoder outputs. "
            f"Decoder inputs: {list(decoder_inputs)}; encoder outputs: "
            f"{list(encoder_output_names)}."
        )
    return embedding_inputs


def _resolve_decoder_output_names(dec_sess: Any) -> tuple[str, str]:
    """Validate and return decoder outputs needed by the metric path."""
    nodes = dec_sess.get_outputs()
    outputs = _node_names(nodes)
    mask_candidates = [node.name for node in nodes if len(getattr(node, "shape", ())) >= 4]
    score_candidates = [node.name for node in nodes if 1 <= len(getattr(node, "shape", ())) <= 3]
    if "pred_masks" in outputs:
        mask_candidates = ["pred_masks"]
    elif "masks" in outputs:
        mask_candidates = ["masks"]
    if "iou_scores" in outputs:
        score_candidates = ["iou_scores"]
    elif "scores" in outputs:
        score_candidates = ["scores"]
    if len(mask_candidates) != 1 or len(score_candidates) != 1:
        raise ValueError(
            "Could not identify one mask output and one score output from decoder "
            f"contract. Got outputs: {list(outputs)}."
        )
    return mask_candidates[0], score_candidates[0]


def _resolve_point_prompt_inputs(session: Any) -> tuple[str, str] | None:
    """Resolve one coordinate/label input pair by rank/shape, failing on ambiguity."""
    nodes = session.get_inputs()
    coordinates = []
    for node in nodes:
        shape = getattr(node, "shape", ())
        if isinstance(shape, (list, tuple)) and len(shape) in {3, 4} and shape[-1] == 2:
            coordinates.append(node)
    labels = [
        node
        for node in nodes
        if len(getattr(node, "shape", ())) in {2, 3}
        and node not in coordinates
        and (
            "int" in getattr(node, "type", "").lower()
            or "label" in getattr(node, "name", "").lower()
        )
    ]
    if not coordinates and not labels:
        return None
    if len(coordinates) != 1 or len(labels) != 1:
        raise ValueError(
            "Point-prompt inputs are incomplete or ambiguous: "
            f"coordinates={[node.name for node in coordinates]}, "
            f"labels={[node.name for node in labels]}."
        )
    return coordinates[0].name, labels[0].name


def _numpy_dtype(node_type: str) -> Any:
    lowered = node_type.lower()
    if "int64" in lowered:
        return np.int64
    if "int32" in lowered:
        return np.int32
    return np.float32


def _build_encoder_prompt_inputs(
    *,
    prompt: dict[str, Any],
    prompt_mode: str,
    session: Any,
    coordinate_name: str,
    label_name: str,
    original_width: int,
    original_height: int,
) -> dict[str, np.ndarray]:
    """Build normalized point tensors for a prompt-bearing encoder contract."""
    if prompt_mode != "point":
        raise ValueError("Encoder-side prompt routing currently requires prompt_mode='point'.")
    nodes = {node.name: node for node in session.get_inputs()}
    coordinate = nodes[coordinate_name]
    label = nodes[label_name]
    coordinate_shape = tuple(coordinate.shape)
    label_shape = tuple(label.shape)
    if any(not isinstance(dim, int) or dim <= 0 for dim in coordinate_shape + label_shape):
        raise ValueError(
            "Encoder-side point prompts require positive static coordinate and label shapes."
        )
    coords = np.zeros(coordinate_shape, dtype=_numpy_dtype(coordinate.type))
    labels = np.full(label_shape, -1, dtype=_numpy_dtype(label.type))
    px, py = prompt["point"]
    normalized = (px / original_width, py / original_height)
    if len(coordinate_shape) == 3:
        coords[0, 0] = normalized
        labels[0, 0] = 1
    else:
        coords[0, 0, 0] = normalized
        labels[0, 0, 0] = 1
    return {coordinate_name: coords, label_name: labels}


def _select_best_mask(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Select the highest-scoring low-resolution mask across common SAM layouts."""
    if masks.ndim == 4:
        candidates = masks[0]
        score_values = scores.reshape(-1)
    elif masks.ndim == 5:
        candidates = masks[0, 0]
        score_values = scores[0, 0].reshape(-1)
    else:
        raise ValueError(f"Unsupported mask output rank {masks.ndim}; expected 4 or 5.")
    if candidates.shape[0] != score_values.size:
        raise ValueError(
            f"Mask/score candidate count mismatch: {candidates.shape[0]} vs {score_values.size}."
        )
    return np.asarray(candidates[int(score_values.argmax())])


def _preprocess_for_profile(
    profile: _MaskGenProfile,
    img: Image.Image,
) -> tuple[np.ndarray, float, float, int, int]:
    """Profile-driven preprocessing.

    Returns ``(pixel_values, scale_x, scale_y, new_h, new_w)``:

    * ``pixel_values`` -- ``(1, 3, T, T)`` fp32 NCHW.
    * ``scale_x`` / ``scale_y`` -- multiply original pixel coords to map
      into encoder-input space (so prompts can be transformed for the
      decoder).  For ``longest_side_pad`` they are equal (single uniform
      scale); for ``direct`` they differ per axis.
    * ``new_h`` / ``new_w`` -- post-resize, pre-pad dimensions; needed by
      the postprocess step to undo the padding before resizing to the
      original image.  For ``direct`` they equal ``T``.
    """
    from PIL import Image as PILImage

    img = img.convert("RGB")
    orig_w, orig_h = img.size
    target = profile.target_size
    mean = np.asarray(profile.mean, dtype=np.float32)
    std = np.asarray(profile.std, dtype=np.float32)

    if profile.resize_mode == "direct":
        scale_x = target / orig_w
        scale_y = target / orig_h
        resized = img.resize((target, target), PILImage.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        new_h = target
        new_w = target
    elif profile.resize_mode == "longest_side_pad":
        # SAM 2.1 convention: longest-side resize preserving aspect ratio,
        # then zero-pad bottom/right to a square.  Prompts use a single
        # uniform scale (``scale_x == scale_y``).
        scale = target / max(orig_h, orig_w)
        new_h = round(orig_h * scale)
        new_w = round(orig_w * scale)
        resized = img.resize((new_w, new_h), PILImage.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        pad_h = target - new_h
        pad_w = target - new_w
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        scale_x = scale
        scale_y = scale
    else:
        raise ValueError(
            f"Unsupported resize_mode={profile.resize_mode!r} for profile "
            f"{profile.name!r}; expected 'direct' or 'longest_side_pad'.",
        )

    pixel_values = arr.transpose(2, 0, 1)[None, ...]
    return pixel_values.astype(np.float32), scale_x, scale_y, new_h, new_w


def _postprocess_for_profile(
    profile: _MaskGenProfile,
    pred_mask: np.ndarray,
    orig_h: int,
    orig_w: int,
    new_h: int,
    new_w: int,
) -> np.ndarray:
    """Profile-driven postprocessing.

    * ``direct`` -- low-res mask maps 1:1 to the full original image; a
      single resize is enough.
    * ``longest_side_pad`` -- up-sample the low-res mask to the encoder
      input size, crop off the zero-pad region (back to ``new_h x new_w``),
      then resize to the original image dimensions.
    """
    from PIL import Image as PILImage

    if profile.resize_mode == "direct":
        pil = PILImage.fromarray(pred_mask.astype(np.float32))
        final = pil.resize((orig_w, orig_h), PILImage.Resampling.BILINEAR)
        return np.asarray(final, dtype=np.float32) > 0

    if profile.resize_mode == "longest_side_pad":
        target = profile.target_size
        pil = PILImage.fromarray(pred_mask.astype(np.float32))
        up = pil.resize((target, target), PILImage.Resampling.BILINEAR)
        up_arr = np.asarray(up, dtype=np.float32)
        cropped = up_arr[:new_h, :new_w]
        pil2 = PILImage.fromarray(cropped)
        final = pil2.resize((orig_w, orig_h), PILImage.Resampling.BILINEAR)
        return np.asarray(final, dtype=np.float32) > 0

    raise ValueError(
        f"Unsupported resize_mode={profile.resize_mode!r} for profile "
        f"{profile.name!r}; expected 'direct' or 'longest_side_pad'.",
    )


# ----------------------------------------------------------------------
# Back-compat SAM 3 wrappers.  Preserved so existing imports / tests that
# call ``_preprocess_image(img)`` -> 3-tuple keep working unchanged.
# ----------------------------------------------------------------------


def _preprocess_image(
    img: Image.Image,
) -> tuple[np.ndarray, float, float]:
    """SAM 3 preprocessing wrapper -- direct resize to 1008x1008, mean=std=0.5.

    Returns the original 3-tuple ``(pixel_values, scale_x, scale_y)``;
    profile-aware callers should use :func:`_preprocess_for_profile`.
    """
    pv, sx, sy, _new_h, _new_w = _preprocess_for_profile(SAM3_PROFILE, img)
    return pv, sx, sy


def _postprocess_mask(
    pred_mask: np.ndarray,
    orig_h: int,
    orig_w: int,
) -> np.ndarray:
    """SAM 3 postprocessing wrapper -- direct resize back to original."""
    return _postprocess_for_profile(
        SAM3_PROFILE,
        pred_mask,
        orig_h=orig_h,
        orig_w=orig_w,
        new_h=SAM3_PROFILE.target_size,
        new_w=SAM3_PROFILE.target_size,
    )


def _build_decoder_inputs(
    prompt: dict[str, Any],
    prompt_mode: str,
    scale_x: float,
    scale_y: float,
    emb: dict[str, np.ndarray],
    required_embed_names: tuple[str, ...] = (
        "image_embeddings.0",
        "image_embeddings.1",
        "image_embeddings.2",
    ),
    required_input_names: tuple[str, ...] = (
        "input_points",
        "input_labels",
        "input_boxes",
    ),
) -> dict[str, np.ndarray]:
    """Assemble the decoder feed dict for bbox or point prompts.

    See decoder signature:

    * ``input_points``: ``(batch=1, 1, num_points, 2)`` fp32 in resized
      (1008) coordinates.
    * ``input_labels``: ``(batch=1, 1, num_points)`` int64 (1=foreground,
      0=background, -1=padding/null).
    * ``input_boxes``: ``(batch=1, num_boxes, 4)`` fp32 in resized coords,
      ``[x0, y0, x1, y1]`` order.

    For *point* prompts we still must satisfy ``input_boxes``; ORT does
    not accept a zero-size box dim across all builds, so we pass a sentinel
    full-image box (rejected by SAM's prompt encoder via the all-foreground
    point) plus a single fg point.  This matches SAM 1/2 reference impls.
    """
    if prompt_mode == "bbox":
        x0, y0, x1, y1 = prompt["bbox"]
        box = np.array(
            [[[x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]]],
            dtype=np.float32,
        )  # (1, 1, 4)
        points: np.ndarray = np.zeros((1, 1, 0, 2), dtype=np.float32)
        labels = np.zeros((1, 1, 0), dtype=np.int64)
    elif prompt_mode == "point":
        px, py = prompt["point"]
        points = np.array(
            [[[[px * scale_x, py * scale_y]]]],
            dtype=np.float32,
        )  # (1, 1, 1, 2)
        labels = np.ones((1, 1, 1), dtype=np.int64)  # 1 = foreground
        # Empty box (0 num_boxes).  If a future runtime build rejects
        # zero-size dims here, switch to a [0, 0, _TARGET_SIZE, _TARGET_SIZE]
        # sentinel and rely on the point to override.
        box = np.zeros((1, 0, 4), dtype=np.float32)
    else:
        raise ValueError(
            f"Unsupported prompt_mode={prompt_mode!r} (expected 'bbox' or 'point'). "
            "Text-prompt mode is not yet supported for SAM 3 ONNX -- the cached "
            "decoder export has no text input port; tracked as a follow-up.",
        )

    missing_embeds = [key for key in required_embed_names if key not in emb]
    if missing_embeds:
        raise ValueError(
            f"Encoder output missing required keys {missing_embeds}. Got: {list(emb.keys())}"
        )

    prompt_feed = {
        "input_points": points,
        "input_labels": labels,
        "input_boxes": box,
    }
    feed = {name: value for name, value in prompt_feed.items() if name in required_input_names}
    feed.update({name: emb[name] for name in required_embed_names})
    return feed


def _build_providers(
    ep: str | None,
    *,
    device: str = "cpu",
) -> tuple[list[str], list[dict[str, Any]]]:
    """Map ``--ep`` to ORT provider list + per-provider options.

    Accepts both shorthand aliases (``qnn``/``dml``/``cpu``) and canonical
    ORT provider names.  If no EP is provided, derives the first compatible
    EP from the resolved device.  Requested accelerators fail clearly when
    unavailable instead of silently running the evaluator on CPU.
    """
    import onnxruntime as ort

    from ..sysinfo import resolve_eps
    from ..utils.constants import EP_SUPPORTED_DEVICES, normalize_ep_name

    if ep is None:
        compatible = resolve_eps(device)
        if not compatible:
            raise ValueError(
                f"No execution provider is available for device {device!r}. "
                "Pass --ep explicitly or choose a different --device."
            )
        primary = compatible[0]
    else:
        normalized = normalize_ep_name(cast("EPNameOrAlias", ep))
        if normalized is None or normalized not in EP_SUPPORTED_DEVICES:
            raise ValueError(f"Unknown EP {ep!r}. Expected one of: {sorted(EP_SUPPORTED_DEVICES)}")
        primary = normalized

    avail = set(ort.get_available_providers())
    if primary not in avail:
        raise ValueError(
            f"Requested EP {primary!r} is not available. Available providers: {sorted(avail)}"
        )

    providers: list[str] = [primary]
    if primary != "CPUExecutionProvider" and "CPUExecutionProvider" in avail:
        providers.append("CPUExecutionProvider")

    provider_options: list[dict[str, Any]] = [{} for _ in providers]
    if primary == "VitisAIExecutionProvider":
        install_dir = os.environ.get("RYZEN_AI_INSTALLATION_PATH", "")
        xclbin = Path(install_dir) / "voe-4.0-win_amd64" / "xclbins" / "phoenix" / "4x4.xclbin"
        if install_dir and xclbin.exists():
            provider_options[0] = {
                "target": "X1",
                "xlnx_enable_py3_round": 0,
                "xclbin": str(xclbin),
            }
        else:
            logger.warning(
                "RYZEN_AI_INSTALLATION_PATH unset or xclbin missing; VitisAI may "
                "fall back to CPU. Activate the Ryzen AI conda env first.",
            )

    return providers, provider_options
