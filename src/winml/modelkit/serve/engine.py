# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""InferenceEngine — core inference component for winml serve.

Uses HF ``transformers.pipeline`` for preprocessing and postprocessing,
sharing the same code path as ``winml eval``.  The WinMLPreTrainedModel
(ONNX Runtime backend) is passed directly to the pipeline.

Loading strategies (auto-detected from model_path):
  1. HF model ID  (e.g. "microsoft/resnet-50")
       → WinMLAutoModel.from_pretrained(model_id)
  2. Build output directory  (contains model.onnx + build_manifest.json)
       → read manifest → instantiate WinMLPreTrainedModel directly + HF config
  3. Raw .onnx file  (requires task=)
       → WinMLAutoModel.from_onnx(onnx_path, task=task)

EP switching (Phase 1):
  switch_ep() reloads the session with a new EP.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .schema import Prediction, PredictionResult


if TYPE_CHECKING:
    from ..models.winml.base import WinMLPreTrainedModel

logger = logging.getLogger(__name__)

# Rolling window size for latency tracking (bounds memory for long-running servers)
_LATENCY_WINDOW = 200

# Tasks where pipeline input is an image (single file)
_IMAGE_TASKS = {
    "image-classification",
    "image-segmentation",
    "object-detection",
    "semantic-segmentation",
    "depth-estimation",
    "image-to-text",
    "zero-shot-image-classification",
    "zero-shot-object-detection",
}

# Tasks where pipeline input is text
_TEXT_TASKS = {
    "text-classification",
    "sentiment-analysis",
    "token-classification",
    "text-generation",
    "text2text-generation",
    "fill-mask",
    "question-answering",
    "zero-shot-classification",
}

# Tasks where pipeline input is audio (single file)
_AUDIO_TASKS = {
    "audio-classification",
    "automatic-speech-recognition",
}

# Tasks that accept file(s) + text together
_MULTIMODAL_TASKS = {
    "visual-question-answering",
    "document-question-answering",
    "image-text-to-text",
}


class InferenceEngine:
    """Stateful inference engine backed by HF Pipeline.

    Not thread-safe on its own — callers (SingleModelManager, winml run) must
    ensure exclusive access before calling predict() or switch_ep().
    """

    def __init__(self) -> None:
        self._model: WinMLPreTrainedModel | None = None
        self._pipeline: Any | None = None  # transformers.Pipeline
        self._model_id: str | None = None
        self._task: str | None = None
        self._device: str = "auto"
        self._ep: str | None = None
        self._model_path: str | None = None  # original arg for reload()
        self._request_count: int = 0
        self._last_request_at: datetime | None = None
        self._load_start: float = time.time()
        self._latency_samples: deque[float] = deque(maxlen=_LATENCY_WINDOW)

    # ------------------------------------------------------------------
    # Public loading API
    # ------------------------------------------------------------------

    def load(
        self,
        model_path: str | Path,
        *,
        task: str | None = None,
        device: str = "auto",
        ep: str | None = None,
    ) -> None:
        """Load model from model_path.

        Args:
            model_path: HF model ID, build output dir, or .onnx file path.
            task: Required when model_path is a raw .onnx file.
            device: "auto" | "cpu" | "gpu" | "npu".
            ep: Explicit EP short name (e.g. "dml", "qnn").  Overrides device.
        """
        self._model_path = str(model_path)
        self._device = device
        self._ep = ep
        self._load_start = time.time()

        path = Path(model_path)

        if path.is_dir():
            self._load_from_build_dir(path, task=task, device=device, ep=ep)
        elif path.suffix == ".onnx" and path.exists():
            self._load_from_onnx(path, task=task, device=device, ep=ep)
        else:
            self._load_from_hf(str(model_path), task=task, device=device, ep=ep)

        # Create HF pipeline for preprocess + postprocess
        self._pipeline = self._create_pipeline()

    def unload(self) -> None:
        """Release ORT session and free memory."""
        self._model = None
        self._pipeline = None
        logger.info("InferenceEngine: model unloaded")

    def reload(self) -> None:
        """Reload with the same parameters (used after unload or EP switch)."""
        if self._model_path is None:
            raise RuntimeError("reload() called before load()")
        self.load(
            self._model_path,
            task=self._task,
            device=self._device,
            ep=self._ep,
        )

    def switch_ep(self, ep: str) -> None:
        """Switch to a different execution provider."""
        logger.info("Switching EP: %s → %s", self._ep, ep)
        self._ep = ep
        self._latency_samples.clear()
        self.unload()
        self.reload()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        *,
        files: list[bytes] | None = None,
        text: str | None = None,
        tensor_inputs: dict[str, list] | None = None,
        top_k: int = 5,
        **pipeline_kwargs: Any,
    ) -> PredictionResult:
        """Run inference via HF Pipeline and return structured result.

        Args:
            files: Raw media file bytes (image, audio, video).  Task
                determines interpretation.  Most tasks use ``files[0]``.
            text: Input text string (for text / multimodal tasks).
            tensor_inputs: Raw tensor dict — bypasses pipeline, runs
                model directly.
            top_k: Max predictions for classification tasks.
            **pipeline_kwargs: Extra keyword arguments forwarded to the
                HF pipeline (e.g. ``threshold``, ``max_new_tokens``).

        Returns:
            PredictionResult with predictions and latency.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        pipeline_kwargs.setdefault("top_k", top_k)

        t0 = time.perf_counter()

        if tensor_inputs is not None:
            predictions = self._predict_raw_tensors(tensor_inputs)
        elif self._pipeline is not None:
            predictions = self._predict_pipeline(
                files=files,
                text=text,
                **pipeline_kwargs,
            )
        else:
            raise RuntimeError("No pipeline available. Model may not be fully loaded.")

        latency_ms = (time.perf_counter() - t0) * 1000
        self._latency_samples.append(latency_ms)
        self._request_count += 1
        self._last_request_at = datetime.now(tz=timezone.utc)

        session = getattr(self._model, "_session", None)
        ep_name = getattr(session, "_ep", self._ep)

        return PredictionResult(
            task=self._task or "unknown",
            model_id=self._model_id,
            device=self._device,
            ep=ep_name,
            predictions=predictions,
            latency_ms=round(latency_ms, 2),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_path(self) -> str | None:
        """Original model path passed to load()."""
        return self._model_path

    @property
    def is_loaded(self) -> bool:
        """True if the ORT session is active."""
        return self._model is not None

    @property
    def model_id(self) -> str | None:
        """HF model ID, or None for ONNX-only loads."""
        return self._model_id

    @property
    def task(self) -> str | None:
        """Canonical task name (e.g. 'image-classification')."""
        return self._task

    @property
    def device(self) -> str:
        """Device string passed at load time."""
        return self._device

    @property
    def ep(self) -> str | None:
        """Active EP short name (from session or stored value)."""
        session = getattr(self._model, "_session", None)
        return getattr(session, "_ep", self._ep)

    @property
    def request_count(self) -> int:
        """Number of successful predict() calls since load."""
        return self._request_count

    @property
    def last_request_at(self) -> datetime | None:
        """UTC datetime of the last predict() call."""
        return self._last_request_at

    @property
    def latency_stats(self) -> dict[str, float]:
        """Live latency stats over last 200 requests (milliseconds)."""
        samples = sorted(self._latency_samples)
        n = len(samples)
        if n == 0:
            return {
                "mean_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "p50_ms": 0.0,
                "p90_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "sample_count": 0,
            }

        def _pct(p: float) -> float:
            k = (p / 100) * (n - 1)
            f = int(k)
            c = min(f + 1, n - 1)
            return round(samples[f] + (k - f) * (samples[c] - samples[f]), 2)

        return {
            "mean_ms": round(sum(samples) / n, 2),
            "min_ms": round(samples[0], 2),
            "max_ms": round(samples[-1], 2),
            "p50_ms": _pct(50),
            "p90_ms": _pct(90),
            "p95_ms": _pct(95),
            "p99_ms": _pct(99),
            "sample_count": n,
        }

    @property
    def memory_mb(self) -> float:
        """Approximate resident memory used by this process (MB)."""
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Private — pipeline creation
    # ------------------------------------------------------------------

    def _create_pipeline(self) -> Any:
        """Create HF pipeline via the shared factory."""
        if self._task is None or self._model is None:
            return None

        from ..pipeline import create_pipeline

        return create_pipeline(self._task, self._model, self._model_id)

    # ------------------------------------------------------------------
    # Private — inference paths
    # ------------------------------------------------------------------

    def _predict_pipeline(
        self,
        *,
        files: list[bytes] | None,
        text: str | None,
        **pipe_kwargs: Any,
    ) -> list[Prediction] | dict[str, Any]:
        """Run inference through HF Pipeline (handles preprocess + postprocess).

        Known tasks get input-type validation; unknown tasks accept
        whichever input is provided.  All ``pipe_kwargs`` are forwarded
        to the HF pipeline unchanged.
        """
        from PIL import Image

        first_file = files[0] if files else None

        if self._task in _IMAGE_TASKS:
            if first_file is None:
                raise ValueError("Image file required for image tasks")
            pipe_input: Any = Image.open(BytesIO(first_file)).convert("RGB")
        elif self._task in _AUDIO_TASKS:
            if first_file is None:
                raise ValueError("Audio file required for audio tasks")
            pipe_input = self._decode_audio(first_file)
        elif self._task in _MULTIMODAL_TASKS:
            if first_file is None:
                raise ValueError("File required for multimodal tasks")
            pipe_input = {
                "image": Image.open(BytesIO(first_file)).convert("RGB"),
                "question": text or "",
            }
        elif self._task in _TEXT_TASKS:
            if text is None:
                raise ValueError("Text required for text tasks")
            pipe_input = text
        elif first_file is not None:
            pipe_input = Image.open(BytesIO(first_file)).convert("RGB")
        elif text is not None:
            pipe_input = text
        else:
            raise ValueError("Provide file(s) or text for pipeline inference")

        raw_result = self._pipeline(pipe_input, **pipe_kwargs)
        return self._normalize_pipeline_output(raw_result)

    @staticmethod
    def _decode_audio(data: bytes) -> dict[str, Any]:
        """Decode audio bytes into a dict accepted by HF audio pipelines.

        Returns ``{"raw": np.ndarray, "sampling_rate": int}``.
        """
        import numpy as np
        import soundfile as sf

        audio_array, sampling_rate = sf.read(BytesIO(data))
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        return {"raw": audio_array.astype(np.float32), "sampling_rate": sampling_rate}

    def _normalize_pipeline_output(self, raw: Any) -> list[Prediction] | dict[str, Any]:
        """Convert HF pipeline output to our standard format."""
        # Classification tasks return list of {"label": ..., "score": ...}
        is_classification = (
            isinstance(raw, list)
            and raw
            and isinstance(raw[0], dict)
            and "label" in raw[0]
            and "score" in raw[0]
        )
        if is_classification:
            return [
                Prediction(
                    label=str(item["label"]),
                    score=round(float(item["score"]), 6),
                )
                for item in raw
            ]
        # Other tasks: return as-is dict
        if isinstance(raw, dict):
            return raw
        # Fallback
        return {"raw": str(raw)}

    def _predict_raw_tensors(self, tensor_inputs: dict[str, list]) -> dict[str, Any]:
        """Bypass pipeline: run model directly with pre-processed tensors."""
        import numpy as np
        import torch

        inputs_torch = {
            k: torch.from_numpy(np.array(v)) if not isinstance(v, torch.Tensor) else v
            for k, v in tensor_inputs.items()
        }
        output = self._model(**inputs_torch)

        # Convert output to serializable dict
        result: dict[str, Any] = {}
        for attr in ("logits", "pred_boxes", "pred_masks", "pred_labels"):
            val = getattr(output, attr, None)
            if val is not None:
                try:
                    result[attr] = val.detach().cpu().numpy().tolist()
                except Exception:
                    result[attr] = str(val)
        return result or {"raw": str(output)}

    # ------------------------------------------------------------------
    # Private — loading strategies
    # ------------------------------------------------------------------

    def _load_from_build_dir(
        self,
        build_dir: Path,
        *,
        task: str | None,
        device: str,
        ep: str | None,
    ) -> None:
        manifest_path = build_dir / "build_manifest.json"
        onnx_path = build_dir / "model.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(f"model.onnx not found in {build_dir}")

        model_id: str | None = None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            model_id = manifest.get("model_id")
            task = task or manifest.get("task")

        self._model_id = model_id

        from ..models.winml import get_winml_class

        winml_class = get_winml_class(None, task or "")
        self._model = winml_class(onnx_path=onnx_path, config=None, device=device)
        self._task = task or getattr(self._model, "task", None)

        if model_id:
            self._attach_hf_config(model_id)

        logger.info("Loaded from build dir: task=%s model_id=%s", task, model_id)

    def _load_from_onnx(
        self,
        onnx_path: Path,
        *,
        task: str | None,
        device: str,
        ep: str | None,
    ) -> None:
        from ..models.auto import WinMLAutoModel

        self._task = task
        self._model_id = None
        self._model = WinMLAutoModel.from_onnx(
            onnx_path, task=task, device=device, ep=ep, skip_build=True
        )
        logger.info("Loaded from ONNX: %s task=%s", onnx_path, task)

    def _load_from_hf(
        self,
        model_id: str,
        *,
        task: str | None,
        device: str,
        ep: str | None,
    ) -> None:
        from ..models.auto import WinMLAutoModel

        self._model_id = model_id
        self._model = WinMLAutoModel.from_pretrained(model_id, task=task, device=device, ep=ep)
        self._task = (
            task
            or getattr(self._model, "task", None)
            or getattr(getattr(self._model, "config", None), "task", None)
        )
        logger.info("Loaded from HF: %s task=%s", model_id, self._task)

    # ------------------------------------------------------------------
    # Private — HF config
    # ------------------------------------------------------------------

    def _attach_hf_config(self, model_id: str) -> None:
        try:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(model_id)
            if self._model is not None:
                self._model.config = hf_config
            logger.debug("Attached HF config from %s", model_id)
        except Exception as exc:
            logger.warning("Could not load HF config for %s: %s", model_id, exc)
