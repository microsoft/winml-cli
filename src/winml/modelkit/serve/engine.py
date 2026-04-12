# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""InferenceEngine — core inference component for wmk serve.

Uses HF ``transformers.pipeline`` for preprocessing and postprocessing,
sharing the same code path as ``wmk eval``.  The WinMLPreTrainedModel
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
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..session.stats import PerfStats
from .schema import Prediction, PredictionResult


if TYPE_CHECKING:
    from ..models.winml.base import WinMLPreTrainedModel

logger = logging.getLogger(__name__)

# Infer task from WinML model class name (fallback when task is not passed)
_CLASS_TO_TASK: dict[str, str] = {
    "WinMLModelForImageClassification": "image-classification",
    "WinMLModelForSequenceClassification": "text-classification",
    "WinMLModelForImageSegmentation": "image-segmentation",
    "WinMLModelForObjectDetection": "object-detection",
    "WinMLModelForSemanticSegmentation": "image-segmentation",
}

# Tasks where pipeline input is an image
_IMAGE_TASKS = {
    "image-classification",
    "image-segmentation",
    "object-detection",
    "semantic-segmentation",
}

# Tasks where pipeline input is text
_TEXT_TASKS = {
    "text-classification",
    "sentiment-analysis",
    "token-classification",
}


class InferenceEngine:
    """Stateful inference engine backed by HF Pipeline.

    Not thread-safe on its own — callers (SingleModelManager, wmk run) must
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
        self._perf: PerfStats = PerfStats()

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
        self.unload()
        self.reload()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        *,
        image_bytes: bytes | None = None,
        text: str | None = None,
        tensor_inputs: dict[str, list] | None = None,
        top_k: int = 5,
    ) -> PredictionResult:
        """Run inference via HF Pipeline and return structured result.

        Args:
            image_bytes: Raw image file bytes (for image tasks).
            text: Input text string (for text tasks).
            tensor_inputs: Raw tensor dict — bypasses pipeline, runs
                model directly.
            top_k: Max predictions for classification tasks.

        Returns:
            PredictionResult with predictions and latency.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        t0 = time.perf_counter()

        if tensor_inputs is not None:
            predictions = self._predict_raw_tensors(tensor_inputs)
        elif self._pipeline is not None:
            predictions = self._predict_pipeline(image_bytes=image_bytes, text=text, top_k=top_k)
        else:
            raise RuntimeError("No pipeline available. Model may not be fully loaded.")

        latency_ms = (time.perf_counter() - t0) * 1000
        self._perf._samples.append(latency_ms)
        if len(self._perf._samples) > 200:
            self._perf._samples = self._perf._samples[-200:]
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
        p = self._perf
        return {
            "mean_ms": round(p.mean_ms, 2),
            "min_ms": round(p.min_ms, 2),
            "max_ms": round(p.max_ms, 2),
            "p50_ms": round(p.p50_ms, 2),
            "p90_ms": round(p.p90_ms, 2),
            "p95_ms": round(p.p95_ms, 2),
            "p99_ms": round(p.p99_ms, 2),
            "sample_count": p.count,
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
    # Private — pipeline creation (same approach as eval)
    # ------------------------------------------------------------------

    def _create_pipeline(self) -> Any:
        """Create HF pipeline for the loaded model.

        Same pattern as eval's WinMLEvaluator.prepare_pipeline():
        pass the WinMLPreTrainedModel directly to transformers.pipeline().
        """
        from transformers import pipeline

        if self._task is None or self._model is None:
            return None

        # Build pipeline kwargs — use model_id for processor loading
        kwargs: dict[str, Any] = {
            "framework": "pt",
            "device": "cpu",  # tensor placement; ORT EP handles actual device
        }
        if self._model_id:
            kwargs["tokenizer"] = self._model_id
            kwargs["feature_extractor"] = self._model_id
            kwargs["image_processor"] = self._model_id

        pipe = pipeline(self._task, model=self._model, **kwargs)

        # Match eval: set tokenizer padding for fixed-shape ONNX text models
        if self._task in _TEXT_TASKS and pipe.tokenizer is not None:
            io_config = getattr(self._model, "io_config", None) or {}
            shapes = io_config.get("input_shapes", [[]])
            if shapes and len(shapes[0]) > 1 and isinstance(shapes[0][1], int):
                pipe._preprocess_params.setdefault("padding", "max_length")
                pipe._preprocess_params.setdefault("max_length", shapes[0][1])
                pipe._preprocess_params.setdefault("truncation", True)

        logger.info("Created HF pipeline: task=%s model=%s", self._task, self._model_id)
        return pipe

    # ------------------------------------------------------------------
    # Private — inference paths
    # ------------------------------------------------------------------

    def _predict_pipeline(
        self,
        *,
        image_bytes: bytes | None,
        text: str | None,
        top_k: int,
    ) -> list[Prediction] | dict[str, Any]:
        """Run inference through HF Pipeline (handles preprocess + postprocess)."""
        from PIL import Image

        pipe_input: Any = None
        pipe_kwargs: dict[str, Any] = {}

        if self._task in _IMAGE_TASKS:
            if image_bytes is None:
                raise ValueError("image_bytes required for image tasks")
            pipe_input = Image.open(BytesIO(image_bytes)).convert("RGB")
            pipe_kwargs["top_k"] = top_k
        elif self._task in _TEXT_TASKS:
            if text is None:
                raise ValueError("text required for text tasks")
            pipe_input = text
            pipe_kwargs["top_k"] = top_k
        else:
            raise ValueError(f"Unsupported task '{self._task}' for pipeline inference")

        raw_result = self._pipeline(pipe_input, **pipe_kwargs)
        return self._normalize_pipeline_output(raw_result)

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
        self._task = task or _CLASS_TO_TASK.get(type(self._model).__name__)

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
            or _CLASS_TO_TASK.get(type(self._model).__name__)
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
