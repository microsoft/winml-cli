# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
r"""Generic mIoU/Dice smoke test for promptable mask-generation ONNX models.

Works with any SAM-family encoder + prompt-decoder ONNX pair that follows
the standard naming convention:

* **Encoder** input ``pixel_values`` ``(B, 3, T, T)`` -> outputs
  ``image_embeddings.0``, ``image_embeddings.1``, ``image_embeddings.2``.
* **Decoder** inputs ``input_points``, ``input_labels``, ``input_boxes``,
  ``image_embeddings.{0,1,2}`` -> outputs ``iou_scores``, ``pred_masks``,
  ``object_score_logits``.

Supports two datasets:

* ``--dataset human_parsing`` (default) -- ``mattmdjaga/human_parsing_dataset``,
  one binary foreground mask per sample.  Fast, but humans-only.
* ``--dataset coco`` -- COCO val2017 instances via ``merve/coco`` annotations
  (instance masks decoded with ``pycocotools``) and images fetched from
  the COCO CDN.  This is the standard cross-domain SAM benchmark and
  the only one whose numbers are directly comparable to the published
  SAM / SAM 2 / SAM 3 papers.

Supports three prompt strategies (``--prompt-type``):

* ``point`` -- single positive click at the GT mask's centroid (snapped
  to a foreground pixel for non-convex masks).  This is the standard
  "1-click" SAM eval protocol.
* ``bbox`` -- tight GT bounding box as box prompt.  Standard "box-prompt"
  protocol; typically scores 5-10 mIoU higher than 1-click.
* ``point+box`` -- both; used to be the default behaviour of this script.
* ``all`` -- run all three and report a per-prompt-type breakdown.

Encoder output is cached per unique image_id, so running multiple prompt
types or multiple annotations per COCO image only re-runs the cheap
decoder (~25 ms per call vs ~12 s for the SAM 3 encoder).

Run:

    # SAM 3 baseline on the humans slice (back-compat, fast)
    python scripts/mask_generation_eval.py --preset sam3 --num-samples 10

    # SAM 3 on COCO val2017, full 3-way prompt comparison
    python scripts/mask_generation_eval.py --preset sam3 --dataset coco \\
        --num-samples 50 --prompt-type all

    # Custom encoder/decoder
    python scripts/mask_generation_eval.py \\
        --encoder onnx-community/<repo>/onnx/vision_encoder_int8.onnx \\
        --decoder onnx-community/<repo>/onnx/prompt_encoder_mask_decoder_int8.onnx \\
        --target-size 1024 --mean 0.485,0.456,0.406 --std 0.229,0.224,0.225 \\
        --dataset coco --num-samples 20 --prompt-type bbox

Notes:
* The bundled ``sam3`` preset is validated on ``--ep cpu``.
* ``--ep dml`` is experimental for SAM 3 and can silently produce incorrect
    masks with the bundled int8 ONNX weights.
* If you want to experiment with DML, pass your own fp32 encoder/decoder pair
    instead of the bundled preset.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


# --------------------------------------------------------------------------- #
# Built-in profiles (preprocessing + Hub paths).
# Add a new one to the dict at the bottom of this section to register it.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MaskGenProfile:
    """Per-model preprocessing + I/O config for the generic harness."""

    name: str
    encoder_ref: str  # Hub ONNX ref: <org>/<repo>/<path>.onnx
    decoder_ref: str
    target_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    resize_mode: str


# SAM 3 Tracker -- published preprocessing constants from the model card +
# our own ``WinMLMaskGenerationEvaluator`` (which uses 0.5/0.5/0.5).
SAM3_PROFILE = MaskGenProfile(
    name="sam3",
    encoder_ref="onnx-community/sam3-tracker-ONNX/onnx/vision_encoder_int8.onnx",
    decoder_ref="onnx-community/sam3-tracker-ONNX/onnx/prompt_encoder_mask_decoder_int8.onnx",
    target_size=1008,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    resize_mode="direct",
)


# SAM 2.1 -- ImageNet stats and 1024x1024 input (SAM-paper convention).
SAM2_1_PROFILE = MaskGenProfile(
    name="sam2.1",
    encoder_ref="onnx-community/sam2.1-hiera-small-ONNX/onnx/vision_encoder_int8.onnx",
    decoder_ref="onnx-community/sam2.1-hiera-small-ONNX/onnx/prompt_encoder_mask_decoder_int8.onnx",
    target_size=1024,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    resize_mode="longest_side_pad",
)


PROFILES: dict[str, MaskGenProfile] = {
    "sam3": SAM3_PROFILE,
    "sam2.1": SAM2_1_PROFILE,
}


PROMPT_TYPES = ("point", "bbox", "point+box")


def _profile_ep_notice(profile: MaskGenProfile, ep: str) -> str | None:
    """Return a compatibility notice for risky profile/EP combinations."""
    if profile.name != "sam3" or ep != "dml":
        return None

    encoder_kind = "fp16" if "fp16" in profile.encoder_ref else "int8"
    decoder_kind = "fp16" if "fp16" in profile.decoder_ref else "int8"
    return (
        "WARNING: the bundled SAM 3 preset is not validated on DML and can "
        "silently produce incorrect masks.\n"
        f"  encoder: {profile.encoder_ref} ({encoder_kind})\n"
        f"  decoder: {profile.decoder_ref} ({decoder_kind})\n"
        "  DML currently shows two failure modes on this model family: int8 "
        "weights can produce wrong results, and fp16 weights can still decode "
        "to empty masks after sigmoid/thresholding.\n"
        "  Recommended: use --ep cpu for the published preset, or pass custom "
        "fp32 ONNX refs if you want to experiment with DML."
    )


def _emit_profile_ep_notice(profile: MaskGenProfile, ep: str) -> None:
    notice = _profile_ep_notice(profile, ep)
    if notice:
        print(notice)


# --------------------------------------------------------------------------- #
# Eval-sample container.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalSample:
    """A single (image, instance-mask) example.

    ``image_id`` groups annotations belonging to the same source image so
    the harness only runs the encoder once per unique image.  Samples
    must be sorted by ``image_id`` for the cache to be effective.
    """

    image: Image.Image
    gt_mask: np.ndarray  # H x W uint8 (0/1)
    name: str  # display name in the per-sample log
    image_id: str  # cache key for encoder embeddings


# --------------------------------------------------------------------------- #
# Preprocessing / postprocessing.
# --------------------------------------------------------------------------- #


def preprocess_image(
    img: Image.Image,
    profile: MaskGenProfile,
) -> tuple[np.ndarray, float, float, int, int]:
    """Resize according to the profile and normalize.

    Returns:
        pixel_values: (1, 3, T, T) fp32, NCHW
        scale_x, scale_y: original pixel coords -> encoder-input coords
        new_h, new_w: dimensions after resize (before padding)
    """
    img = img.convert("RGB")
    orig_w, orig_h = img.size
    target = profile.target_size

    if profile.resize_mode == "direct":
        scale_x = target / orig_w
        scale_y = target / orig_h
        new_h = target
        new_w = target
        resized = img.resize((target, target), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
    elif profile.resize_mode == "longest_side_pad":
        scale = target / max(orig_h, orig_w)
        scale_x = scale
        scale_y = scale
        new_h = round(orig_h * scale)
        new_w = round(orig_w * scale)
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
    else:
        raise ValueError(
            f"Unsupported resize_mode={profile.resize_mode!r}; expected "
            "'direct' or 'longest_side_pad'."
        )

    arr = (arr - np.array(profile.mean, dtype=np.float32)) / np.array(
        profile.std,
        dtype=np.float32,
    )

    if profile.resize_mode == "longest_side_pad":
        pad_h = target - new_h
        pad_w = target - new_w
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

    pixel_values = arr.transpose(2, 0, 1)[None, ...]
    return pixel_values.astype(np.float32), scale_x, scale_y, new_h, new_w


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight xyxy bbox of nonzero pixels, or ``None`` if mask is empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def sample_point_in_mask(mask: np.ndarray) -> tuple[int, int] | None:
    """Pick one foreground point near the mask centroid.

    Returns ``(x, y)`` in original-image pixel coordinates, or ``None``
    if the mask is empty.  For non-convex masks (centroid outside the
    mask) we snap to the nearest foreground pixel; this is the same
    fallback used by the HF SAM image processor's ``point_grid``.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    cy = round(float(ys.mean()))
    cx = round(float(xs.mean()))
    if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx]:
        return cx, cy
    d2 = (ys - cy) ** 2 + (xs - cx) ** 2
    i = int(d2.argmin())
    return int(xs[i]), int(ys[i])


def postprocess_mask(
    pred_mask: np.ndarray,
    profile: MaskGenProfile,
    orig_h: int,
    orig_w: int,
    new_h: int,
    new_w: int,
) -> np.ndarray:
    """Un-pad and resize a low-res mask back to original image coords."""
    pil = Image.fromarray(pred_mask.astype(np.float32))
    up = pil.resize((profile.target_size, profile.target_size), Image.BILINEAR)
    up_arr = np.asarray(up, dtype=np.float32)

    cropped = up_arr if profile.resize_mode == "direct" else up_arr[:new_h, :new_w]

    pil2 = Image.fromarray(cropped)
    final = pil2.resize((orig_w, orig_h), Image.BILINEAR)
    return np.asarray(final, dtype=np.float32) > 0


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary IoU."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary Dice coefficient."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pp = pred.sum()
    pg = gt.sum()
    if pp + pg == 0:
        return 0.0
    return 2.0 * float(np.logical_and(pred, gt).sum()) / float(pp + pg)


# --------------------------------------------------------------------------- #
# Inference driver -- split into encode (per image) + decode (per prompt) so
# embeddings can be cached across prompt types and across multiple
# annotations on the same COCO image.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EncoderState:
    """Cached encoder outputs and resize metadata for one source image."""

    embeddings: dict[str, np.ndarray]
    scale_x: float
    scale_y: float
    new_h: int
    new_w: int
    encode_time: float


def encode_image(
    enc_sess: ort.InferenceSession,
    profile: MaskGenProfile,
    img: Image.Image,
) -> EncoderState:
    """Encode one image and cache the resulting embeddings plus timing."""
    pixel_values, scale_x, scale_y, new_h, new_w = preprocess_image(img, profile)
    t0 = time.monotonic()
    enc_out = enc_sess.run(None, {"pixel_values": pixel_values})
    elapsed = time.monotonic() - t0
    enc_names = [o.name for o in enc_sess.get_outputs()]
    emb = dict(zip(enc_names, enc_out, strict=True))
    return EncoderState(emb, scale_x, scale_y, new_h, new_w, elapsed)


def make_prompts(
    prompt_type: str,
    gt_mask: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> dict[str, np.ndarray] | None:
    """Build the (points, labels, boxes) inputs for the given prompt strategy.

    Returns ``None`` if the GT mask is degenerate and no usable prompt can
    be derived (caller should skip the sample).  All coordinates are in
    encoder-input space (i.e. ``x * scale_x`` and ``y * scale_y``).
    """
    if prompt_type not in PROMPT_TYPES:
        raise ValueError(f"Unknown prompt_type {prompt_type!r}")

    bbox = bbox_from_mask(gt_mask)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox

    if prompt_type in ("point", "point+box"):
        pt = sample_point_in_mask(gt_mask)
        if pt is None:
            return None
        px, py = pt
        points = np.array(
            [[px * scale_x, py * scale_y]],
            dtype=np.float32,
        ).reshape(1, 1, 1, 2)
        labels = np.array([[1]], dtype=np.int64).reshape(1, 1, 1)
    else:
        points = np.zeros((1, 1, 0, 2), dtype=np.float32)
        labels = np.zeros((1, 1, 0), dtype=np.int64)

    if prompt_type in ("bbox", "point+box"):
        boxes = np.array(
            [[x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]],
            dtype=np.float32,
        )[None, ...]
    else:
        boxes = np.zeros((1, 0, 4), dtype=np.float32)

    return {
        "input_points": points,
        "input_labels": labels,
        "input_boxes": boxes,
    }


def decode_with_prompt(
    dec_sess: ort.InferenceSession,
    profile: MaskGenProfile,
    state: EncoderState,
    prompts: dict[str, np.ndarray],
    orig_h: int,
    orig_w: int,
) -> tuple[np.ndarray, float, float]:
    """Run the decoder once with the supplied prompts.

    Returns ``(binary_mask, pred_iou, decode_seconds)``.
    """
    dec_inputs = {
        **prompts,
        "image_embeddings.0": state.embeddings["image_embeddings.0"],
        "image_embeddings.1": state.embeddings["image_embeddings.1"],
        "image_embeddings.2": state.embeddings["image_embeddings.2"],
    }
    t0 = time.monotonic()
    iou_scores, pred_masks, _obj_logits = dec_sess.run(
        ["iou_scores", "pred_masks", "object_score_logits"],
        dec_inputs,
    )
    elapsed = time.monotonic() - t0
    iou_preds = iou_scores[0, 0]
    best_idx = int(iou_preds.argmax())
    best_low_res = pred_masks[0, 0, best_idx]
    best_iou_pred = float(iou_preds[best_idx])
    binary = postprocess_mask(
        best_low_res,
        profile,
        orig_h,
        orig_w,
        state.new_h,
        state.new_w,
    )
    return binary, best_iou_pred, elapsed


# --------------------------------------------------------------------------- #
# Datasets.
# --------------------------------------------------------------------------- #


def _load_human_parsing(n: int) -> list[EvalSample]:
    """Sample n binary-mask examples from ``mattmdjaga/human_parsing_dataset``.

    Same dataset the production ``MaskGenerationDataset`` defaults to.
    Multi-class body-part labels are collapsed to a single binary
    foreground mask.  Degenerate samples (coverage <5% or >95%) are
    skipped so the prompt is meaningful.
    """
    from datasets import load_dataset

    ds_name = "mattmdjaga/human_parsing_dataset"
    print(f"Loading {ds_name} (streaming, taking {n} samples)...")
    ds = load_dataset(ds_name, split="train", streaming=True)
    samples: list[EvalSample] = []
    for i, ex in enumerate(ds):
        if len(samples) >= n:
            break
        img = ex["image"]
        mask_arr = np.asarray(ex["mask"])
        binary = (mask_arr > 0).astype(np.uint8)
        coverage = binary.sum() / binary.size
        if coverage < 0.05 or coverage > 0.95:
            continue
        samples.append(
            EvalSample(
                image=img,
                gt_mask=binary,
                name=f"hp_{i:04d}",
                image_id=f"hp_{i:04d}",  # unique per sample (no cross-sample reuse)
            ),
        )
    if not samples:
        sys.exit(f"No usable samples from {ds_name}")
    print(f"  loaded {len(samples)}")
    return samples


_COCO_ANNOTATIONS_REPO = "merve/coco"
_COCO_ANNOTATIONS_FILE = "annotations/instances_val2017.json"
_COCO_IMAGE_BASE = "http://images.cocodataset.org/val2017"


def _download_coco_image(file_name: str, cache_dir: Path) -> Image.Image:
    """Fetch a COCO val2017 image from the public CDN, with a local cache."""
    import urllib.request

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / file_name
    if not cached.exists():
        url = f"{_COCO_IMAGE_BASE}/{file_name}"
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            data = resp.read()
        cached.write_bytes(data)
    with cached.open("rb") as f:
        return Image.open(io.BytesIO(f.read())).convert("RGB")


def _load_coco(n: int, min_area: float = 1024.0, seed: int = 0) -> list[EvalSample]:
    """Sample n COCO val2017 instance annotations.

    Annotations are filtered to ``iscrowd == 0`` and ``area >= min_area``
    so prompts are meaningful (tiny objects aren't a useful SAM
    benchmark; crowd RLEs combine multiple instances).  Samples are
    sorted by ``image_id`` so the encoder cache is effective.
    """
    try:
        from huggingface_hub import hf_hub_download
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        sys.exit(
            f"COCO eval requires huggingface_hub + pycocotools: {exc}",
        )

    print(f"Loading COCO val2017 annotations ({_COCO_ANNOTATIONS_REPO})...")
    ann_path = hf_hub_download(
        repo_id=_COCO_ANNOTATIONS_REPO,
        filename=_COCO_ANNOTATIONS_FILE,
        repo_type="dataset",
    )
    with Path(ann_path).open(encoding="utf-8") as f:
        coco = json.load(f)
    images_by_id = {im["id"]: im for im in coco["images"]}
    cats_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    annotations = [
        a
        for a in coco["annotations"]
        if a.get("iscrowd", 0) == 0 and a.get("area", 0.0) >= min_area
    ]
    annotations.sort(key=lambda a: a["image_id"])
    rng = np.random.default_rng(seed)
    if len(annotations) > n:
        # Sample n annotations, then re-sort by image_id for cache locality.
        idx = rng.choice(len(annotations), size=n, replace=False)
        chosen = sorted((annotations[int(i)] for i in idx), key=lambda a: a["image_id"])
    else:
        chosen = annotations

    image_cache_dir = Path.home() / ".cache" / "winml" / "coco_val2017_images"
    samples: list[EvalSample] = []
    fetched_images: dict[int, Image.Image] = {}
    print(f"  decoding {len(chosen)} instance masks (downloading images on demand)...")
    for ann in chosen:
        image_id = ann["image_id"]
        img_meta = images_by_id[image_id]
        h, w = int(img_meta["height"]), int(img_meta["width"])
        seg = ann["segmentation"]
        if isinstance(seg, list):
            rles = mask_utils.frPyObjects(seg, h, w)
            rle = mask_utils.merge(rles)
        elif isinstance(seg, dict):
            rle = (
                seg
                if isinstance(seg.get("counts"), bytes)
                else mask_utils.frPyObjects(
                    seg,
                    h,
                    w,
                )
            )
        else:
            continue
        gt = mask_utils.decode(rle).astype(np.uint8)
        if gt.sum() == 0:
            continue
        if image_id not in fetched_images:
            try:
                fetched_images[image_id] = _download_coco_image(
                    img_meta["file_name"],
                    image_cache_dir,
                )
            except Exception as exc:
                print(f"    SKIP image {image_id}: download failed ({exc})")
                continue
        img = fetched_images[image_id]
        cat_name = cats_by_id.get(ann["category_id"], "?")
        samples.append(
            EvalSample(
                image=img,
                gt_mask=gt,
                name=f"coco_{image_id}_{ann['id']}_{cat_name}",
                image_id=f"coco_{image_id}",
            ),
        )
    if not samples:
        sys.exit("No usable COCO samples (download failures or all filtered).")
    print(f"  loaded {len(samples)} annotations across {len({s.image_id for s in samples})} images")
    return samples


def load_eval_samples(dataset: str, n: int) -> list[EvalSample]:
    """Load eval samples from the configured dataset."""
    if dataset == "human_parsing":
        return _load_human_parsing(n)
    if dataset == "coco":
        return _load_coco(n)
    raise ValueError(f"Unknown dataset {dataset!r}")


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #


def _build_providers(ep: str) -> tuple[list[str], list[dict]]:
    """Return (providers, provider_options) for the requested EP.

    Falls back to CPU with a warning if the requested EP isn't installed.
    VitisAI (AMD NPU) requires extra config on Phoenix/Hawk Point; if
    ``RYZEN_AI_INSTALLATION_PATH`` is set we wire the xclbin
    automatically, otherwise we warn and let ORT fall back.
    """
    providers_map = {
        "cpu": ["CPUExecutionProvider"],
        "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
        "vitisai": ["VitisAIExecutionProvider", "CPUExecutionProvider"],
    }
    providers = list(providers_map[ep])
    avail = set(ort.get_available_providers())
    if providers[0] not in avail:
        print(
            f"WARNING: EP '{providers[0]}' not available "
            f"(available: {sorted(avail)}). Falling back to CPU only.",
        )
        return ["CPUExecutionProvider"], [{}]

    provider_options: list[dict] = [{} for _ in providers]
    if providers[0] == "VitisAIExecutionProvider":
        install_dir = Path(os.environ.get("RYZEN_AI_INSTALLATION_PATH", ""))
        xclbin = install_dir / "voe-4.0-win_amd64" / "xclbins" / "phoenix" / "4x4.xclbin"
        if install_dir and xclbin.exists():
            provider_options[0] = {
                "target": "X1",
                "xlnx_enable_py3_round": 0,
                "xclbin": str(xclbin),
            }
            print(f"  VitisAI PHX config: target=X1, xclbin={xclbin}")
        else:
            print(
                "  WARNING: RYZEN_AI_INSTALLATION_PATH not set or xclbin "
                f"not found at '{xclbin}'. VitisAI will likely fall back "
                "to CPU. Run inside the Ryzen AI conda env so the env var "
                "is populated.",
            )
    return providers, provider_options


def _resolve_local(ref: str) -> Path:
    """Resolve a Hub-ONNX ref or local path to a local file.

    Lazily imports our own resolver so the script also runs standalone
    (e.g. from a checkout where ``src/`` is on PYTHONPATH but the package
    isn't installed in editable mode).
    """
    p = Path(ref)
    if p.exists():
        return p
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from winml.modelkit.loader.onnx_hub import resolve_hf_onnx_path

    return resolve_hf_onnx_path(ref)


def main() -> int:
    """Run the mask-generation benchmark from the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PROFILES),
        default=None,
        help=(
            "Built-in profile. Overrides --encoder/--decoder/--target-size/"
            "--mean/--std/--resize-mode."
        ),
    )
    parser.add_argument("--encoder", help="Hub ONNX ref or local path to encoder.")
    parser.add_argument("--decoder", help="Hub ONNX ref or local path to decoder.")
    parser.add_argument("--target-size", type=int, default=1024)
    parser.add_argument(
        "--mean",
        default="0.485,0.456,0.406",
        help="Comma-separated per-channel mean for normalization.",
    )
    parser.add_argument(
        "--std",
        default="0.229,0.224,0.225",
        help="Comma-separated per-channel std for normalization.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=("direct", "longest_side_pad"),
        default="longest_side_pad",
        help="Resize policy for custom profiles.",
    )
    parser.add_argument("--name", default="custom", help="Label used in output table.")
    parser.add_argument(
        "--dataset",
        choices=("human_parsing", "coco"),
        default="human_parsing",
        help="Evaluation dataset.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="For COCO this is the number of annotations (instances), not images.",
    )
    parser.add_argument(
        "--prompt-type",
        choices=(*PROMPT_TYPES, "all"),
        default="point+box",
        help="Prompt strategy. 'all' runs every type per sample and reports a breakdown.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("out/mask_gen_eval"))
    parser.add_argument(
        "--ep",
        choices=["cpu", "dml", "vitisai"],
        default="cpu",
        help="Execution provider.",
    )
    args = parser.parse_args()

    if args.preset:
        profile = PROFILES[args.preset]
    else:
        if not (args.encoder and args.decoder):
            parser.error("--encoder and --decoder are required when --preset is not used.")
        profile = MaskGenProfile(
            name=args.name,
            encoder_ref=args.encoder,
            decoder_ref=args.decoder,
            target_size=args.target_size,
            mean=tuple(float(x) for x in args.mean.split(",")),  # type: ignore[arg-type]
            std=tuple(float(x) for x in args.std.split(",")),  # type: ignore[arg-type]
            resize_mode=args.resize_mode,
        )

    _emit_profile_ep_notice(profile, args.ep)

    prompt_types = list(PROMPT_TYPES) if args.prompt_type == "all" else [args.prompt_type]

    out_dir = args.out_dir / profile.name / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Profile: {profile.name}")
    print(f"  encoder: {profile.encoder_ref}")
    print(f"  decoder: {profile.decoder_ref}")
    print(f"  target_size={profile.target_size}  mean={profile.mean}  std={profile.std}")
    print(f"  dataset: {args.dataset}  prompt_types: {prompt_types}")

    enc_path = _resolve_local(profile.encoder_ref)
    dec_path = _resolve_local(profile.decoder_ref)
    print(f"  encoder local: {enc_path}")
    print(f"  decoder local: {dec_path}")

    providers, provider_options = _build_providers(args.ep)
    print(f"Creating ORT sessions (providers={providers})...")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    enc_sess = ort.InferenceSession(
        str(enc_path),
        sess_options=sess_opts,
        providers=providers,
        provider_options=provider_options,
    )
    dec_sess = ort.InferenceSession(
        str(dec_path),
        sess_options=sess_opts,
        providers=providers,
        provider_options=provider_options,
    )
    print(f"  encoder providers: {enc_sess.get_providers()}")
    print(f"  decoder providers: {dec_sess.get_providers()}")

    samples = load_eval_samples(args.dataset, args.num_samples)
    print(f"Got {len(samples)} samples. Running {profile.name}...")

    # Per-prompt-type rows: name, iou_gt, dice, iou_pred, dec_sec, enc_sec
    rows: dict[str, list[tuple[str, float, float, float, float, float]]] = {
        pt: [] for pt in prompt_types
    }
    cache: dict[str, EncoderState] = {}
    encode_count = 0

    for i, sample in enumerate(samples):
        if sample.image_id not in cache:
            try:
                cache[sample.image_id] = encode_image(enc_sess, profile, sample.image)
                encode_count += 1
            except Exception as exc:
                print(f"  [{i + 1:3d}/{len(samples)}] {sample.name} ENCODE FAILED: {exc}")
                continue
        state = cache[sample.image_id]
        orig_h, orig_w = sample.gt_mask.shape
        for pt in prompt_types:
            prompts = make_prompts(pt, sample.gt_mask, state.scale_x, state.scale_y)
            if prompts is None:
                continue
            try:
                pred, iou_pred, dec_sec = decode_with_prompt(
                    dec_sess,
                    profile,
                    state,
                    prompts,
                    orig_h,
                    orig_w,
                )
            except Exception as exc:
                print(f"  [{i + 1:3d}/{len(samples)}] {sample.name} {pt} FAILED: {exc}")
                continue
            score_iou = iou(pred, sample.gt_mask)
            score_dice = dice(pred, sample.gt_mask)
            rows[pt].append(
                (sample.name, score_iou, score_dice, iou_pred, dec_sec, state.encode_time),
            )
            print(
                f"  [{i + 1:3d}/{len(samples)}] {sample.name[:50]:50s} "
                f"{pt:11s} IoU={score_iou:.3f} Dice={score_dice:.3f} "
                f"pIoU={iou_pred:.3f} dec={dec_sec * 1000:.1f}ms",
            )
            vis = np.stack(
                [
                    (sample.gt_mask * 255).astype(np.uint8),
                    (pred * 255).astype(np.uint8),
                    np.zeros_like(sample.gt_mask, dtype=np.uint8),
                ],
                axis=-1,
            )
            vis_path = out_dir / f"{i:03d}_{Path(sample.name).stem}_{pt.replace('+', '_')}.png"
            Image.fromarray(vis).save(vis_path)

    # Aggregate.
    if not any(rows.values()):
        print("No successful runs.")
        return 1

    print("\n" + "=" * 80)
    print(
        f"{profile.name}  --  mask-generation eval  (dataset={args.dataset}, "
        f"unique_images={encode_count}, EP={args.ep})",
    )
    print("=" * 80)
    header = (
        f"{'prompt':<11} {'n':>4} {'mIoU':>7} {'Dice':>7} {'pIoU':>7} "
        f"{'mIoU>=0.5':>9} {'mIoU>=0.75':>10} {'enc_s/img':>10} {'dec_ms':>7}"
    )
    print(header)
    print("-" * len(header))
    enc_times = []
    for pt in prompt_types:
        pt_rows = rows[pt]
        if not pt_rows:
            print(f"{pt:<11} (no successful runs)")
            continue
        ious = np.array([r[1] for r in pt_rows])
        dices = np.array([r[2] for r in pt_rows])
        pious = np.array([r[3] for r in pt_rows])
        dec_ms = np.array([r[4] for r in pt_rows]) * 1000
        enc_s = np.array([r[5] for r in pt_rows])
        enc_times.extend(enc_s.tolist())
        miou = float(ious.mean())
        rate50 = float((ious >= 0.5).mean())
        rate75 = float((ious >= 0.75).mean())
        print(
            f"{pt:<11} {len(pt_rows):>4d} {miou:>7.4f} {dices.mean():>7.4f} "
            f"{pious.mean():>7.4f} {rate50:>9.2%} {rate75:>10.2%} "
            f"{enc_s.mean():>10.2f} {dec_ms.mean():>7.1f}",
        )
    if enc_times:
        unique_enc = list(
            {
                s.image_id: cache[s.image_id].encode_time for s in samples if s.image_id in cache
            }.values(),
        )
        if unique_enc:
            print(
                f"\nEncoder: {len(unique_enc)} unique runs, mean {np.mean(unique_enc):.2f}s/image",
            )
    print(f"Visualizations: {out_dir}/  (red=GT, green=prediction)")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
