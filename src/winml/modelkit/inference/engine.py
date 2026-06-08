# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""InferenceEngine — core inference component for WinML CLI.

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

Input dispatch:
  Registry-driven — TASK_REGISTRY[task] provides user_inputs schema and
  PipelineMapping.  Binary decoding, validation, and pipeline routing are
  all data-driven (zero per-task if branches).
"""

from __future__ import annotations

import inspect
import json
import logging
import tempfile
import time
from collections import deque
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .tasks import BINARY_TYPES, TASK_REGISTRY, InputField, PipelineMapping
from .types import Prediction, PredictionResult


if TYPE_CHECKING:
    from ..models.winml.base import WinMLPreTrainedModel
    from ..utils.constants import EPNameOrAlias

logger = logging.getLogger(__name__)

# Rolling window size for latency tracking (bounds memory for long-running servers)
_LATENCY_WINDOW = 200

# Sentinel for "parameter not provided" (distinct from None)
_UNSET: Any = object()

# ---------------------------------------------------------------------------
# Binary decoders — keyed by InputField.type
# ---------------------------------------------------------------------------

_PY_TYPE_TO_SCHEMA = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
}


def _decode_audio(data: bytes) -> dict[str, Any]:
    """Decode audio bytes → {"raw": mono float32 ndarray, "sampling_rate": int}."""
    import numpy as np
    import soundfile as sf

    audio_array, sampling_rate = sf.read(BytesIO(data))
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)  # stereo → mono
    return {"raw": audio_array.astype(np.float32), "sampling_rate": sampling_rate}


def _decode_video(data: bytes) -> str:
    """Write video bytes to a temp file and return the path.

    HF VideoClassificationPipeline accepts a file path string.
    The temp file is NOT auto-deleted — callers should clean up
    after inference (e.g. in a finally block).
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(data)
        return tmp.name


def _decode_image(data: bytes) -> Any:
    """Decode image bytes → PIL.Image.Image (RGB)."""
    from PIL import Image

    return Image.open(BytesIO(data)).convert("RGB")


_DECODERS: dict[str, Any] = {
    "image": _decode_image,
    "audio": _decode_audio,
    "video": _decode_video,
}


def _sanitize_numpy(obj: Any) -> Any:
    """Recursively convert numpy scalars to Python types for JSON serialization.

    HF pipelines (e.g. NER/TokenClassification) return dicts containing
    ``numpy.float32`` scores and ``numpy.int64`` offsets that pydantic
    cannot serialize.  This function converts them to native Python types.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: _sanitize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_numpy(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Build-directory artifact discovery
# ---------------------------------------------------------------------------


def _find_build_artifacts(build_dir: Path, *, task: str | None = None) -> tuple[Path, dict | None]:
    """Locate model.onnx and build_manifest.json inside a build/cache directory.

    Supports both plain layout (``model.onnx``) and cache-key-prefixed layout
    (``{cache_key}_model.onnx``) so that ``InferenceEngine.load()`` can load
    directly from ``~/.cache/winml/artifacts/{slug}/`` without a full rebuild.

    When *task* is provided, only returns artifacts whose manifest ``task``
    matches.  A cache directory may contain multiple task variants (e.g.
    ``feat_*`` for feature-extraction and ``txtcls_*`` for text-classification);
    using the wrong ONNX (different output head) would produce garbage results.

    Returns:
        ``(onnx_path, manifest_dict)`` — manifest is ``None`` when no
        manifest file is found.

    Raises:
        FileNotFoundError: if no matching ``*model.onnx`` is found.
    """
    # Try plain layout first (bare build output)
    plain_onnx = build_dir / "model.onnx"
    plain_manifest = build_dir / "build_manifest.json"
    if plain_onnx.exists():
        manifest = json.loads(plain_manifest.read_text()) if plain_manifest.exists() else None
        if task is None or manifest is None or manifest.get("task") == task:
            return plain_onnx, manifest

    # Cache-key-prefixed layout: {cache_key}_model.onnx
    # Scan all candidates and match by task when specified.
    candidates: list[tuple[Path, dict | None]] = []
    for onnx_path in sorted(build_dir.glob("*_model.onnx")):
        prefix = onnx_path.name.rsplit("_model.onnx", 1)[0]
        manifest_path = build_dir / f"{prefix}_build_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if task is not None:
            if manifest is None or manifest.get("task") == task:
                return onnx_path, manifest
        else:
            candidates.append((onnx_path, manifest))

    if candidates:
        # task=None: if all variants share the same task, pick the first.
        # If multiple different tasks exist, raise to force the caller to specify.
        tasks = {m.get("task") for _, m in candidates if m is not None}
        if len(tasks) > 1:
            raise FileNotFoundError(
                f"Multiple task variants found in {build_dir}: {tasks}. "
                "Pass task= to select the correct ONNX model."
            )
        return candidates[0]

    raise FileNotFoundError(f"No model.onnx matching task={task!r} found in {build_dir}")


# ---------------------------------------------------------------------------
# Lightweight schema helpers (no model load required)
# ---------------------------------------------------------------------------


def _resolve_hf_task(model_id: str, task: str | None) -> str | None:
    """Resolve task from an HF model ID using AutoConfig only (no model weights).

    When *task* is provided, normalises it and returns immediately.
    Otherwise downloads the small ``config.json`` from the Hub and infers
    the task via ``resolve_task_and_model_class``.
    """
    if task is not None:
        # If the task has its own entry in TASK_REGISTRY (e.g.
        # sentence-similarity), preserve it as-is.  Only fall back to
        # Optimum's synonym mapping for unknown tasks — otherwise
        # sentence-similarity gets collapsed to feature-extraction and
        # loses its distinct input schema.
        if task in TASK_REGISTRY:
            return task
        from ..loader import normalize_task

        return normalize_task(task)

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id)

        from ..loader import resolve_task_and_model_class

        resolved_task, _ = resolve_task_and_model_class(config)
        return resolved_task
    except Exception as exc:
        logger.warning("Could not detect task for %s: %s", model_id, exc)
        return None


def _discover_pipeline_params_from_task(task: str | None) -> list[dict]:
    """Discover pipeline parameters from the pipeline *class* (no instance needed).

    Looks up the ``transformers`` pipeline class for *task* and inspects
    ``_sanitize_parameters`` — the same introspection that
    ``_discover_pipeline_params`` performs on an instance, but without
    loading any model weights or creating an ORT session.
    """
    if task is None:
        return []

    try:
        from transformers.pipelines import SUPPORTED_TASKS

        task_info = SUPPORTED_TASKS.get(task)
        if not task_info:
            return []
        pipeline_class = task_info.get("impl")
        if pipeline_class is None:
            return []

        sig = inspect.signature(pipeline_class._sanitize_parameters)
    except (ImportError, ValueError, TypeError, KeyError, AttributeError):
        return []

    params: list[dict] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        entry = _build_param_entry(name, param)
        if entry is not None:
            params.append(entry)
    return params


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------


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
        self._ep: EPNameOrAlias | None = None
        self._model_path: str | None = None  # original arg for reload()
        self._request_count: int = 0
        self._last_request_at: datetime | None = None
        self._load_start: float = time.time()
        self._latency_samples: deque[float] = deque(maxlen=_LATENCY_WINDOW)

        # Registry-driven schema (resolved at load time)
        self._user_input_schema: list[InputField] | None = None
        self._pipeline_mapping: PipelineMapping | None = None
        self._pipeline_params: list[dict] | None = None

    # ------------------------------------------------------------------
    # Public loading API
    # ------------------------------------------------------------------

    def load(
        self,
        model_path: str | Path,
        *,
        task: str | None = None,
        device: str = "auto",
        ep: EPNameOrAlias | None = None,
        skip_build: bool = True,
        allow_unsupported_nodes: bool = False,
    ) -> None:
        """Load model from model_path.

        Args:
            model_path: HF model ID, build output dir, or .onnx file path.
            task: Required when model_path is a raw .onnx file.
            device: "auto" | "cpu" | "gpu" | "npu".
            ep: Explicit EP short name (e.g. "dml", "qnn").  Overrides device.
            skip_build: When True (default), use a raw .onnx file as-is. When
                False, run the build pipeline (optimize/quantize/compile)
                build output directories.
            allow_unsupported_nodes: If True, warn instead of raising when the
                analyzer reports unsupported nodes during an HF build. Note: has
                build directory (no build/analyze step runs in those paths).
        """
        self._model_path = str(model_path)
        self._ep = ep
        self._load_start = time.time()

        path = Path(model_path)

        if path.is_dir():
            try:
                self._load_from_build_dir(path, task=task, device=device, ep=ep)
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                # No cached ONNX for this task (or corrupt manifest) — check
                # if the manifest has a model_id we can rebuild from (e.g.
                # cache was built for a different task like feature-extraction
                # but caller wants text-classification).
                model_id = self._resolve_model_id_from_dir(path)
                if model_id:
                    logger.info(
                        "No cached ONNX for task=%s in %s — rebuilding from %s",
                        task,
                        path,
                        model_id,
                    )
                    self._load_from_hf(
                        model_id,
                        task=task,
                        device=device,
                        ep=ep,
                        allow_unsupported_nodes=allow_unsupported_nodes,
                    )
                else:
                    raise
        elif path.suffix == ".onnx" and path.exists():
            self._load_from_onnx(path, task=task, device=device, ep=ep, skip_build=skip_build)
        else:
            self._load_from_hf(
                str(model_path),
                task=task,
                device=device,
                ep=ep,
                allow_unsupported_nodes=allow_unsupported_nodes,
            )

        # Create HF pipeline for preprocess + postprocess
        self._pipeline = self._create_pipeline()

        # Resolve schema from registry
        self._resolve_schema()

        # Discover pipeline parameters
        if self._pipeline is not None:
            self._pipeline_params = _discover_pipeline_params(self._pipeline)

    def load_schema_only(
        self,
        model_path: str | Path,
        *,
        task: str | None = None,
        device: str = "auto",
        ep: EPNameOrAlias | None = None,
    ) -> None:
        """Lightweight load for schema display — no model build or ORT session.

        Resolves task and populates ``user_input_schema`` and
        ``pipeline_params`` without downloading model weights, running the
        build pipeline (export/optimize/analyze), or creating an ORT
        session.  For HF model IDs this is orders of magnitude faster
        than ``load()`` because only the Hub ``config.json`` is fetched.

        For build directories, reads the manifest file directly.
        For ``.onnx`` files, uses the caller-provided *task*.

        Falls back to ``load()`` only when the task cannot be determined
        without a full model load.
        """
        self._model_path = str(model_path)
        self._device = device
        self._ep = ep

        path = Path(model_path)

        if path.is_dir():
            # Build dir: read manifest for task + model_id (no ORT session)
            try:
                _, manifest = _find_build_artifacts(path, task=task)
            except FileNotFoundError:
                manifest = None
            if manifest is not None:
                self._model_id = manifest.get("model_id")
                self._task = task or manifest.get("task")
            else:
                self._task = task
        elif path.suffix == ".onnx" and path.exists():
            self._model_id = str(model_path)
            self._task = task
        else:
            # HF model ID: resolve task from config.json only
            self._model_id = str(model_path)
            self._task = _resolve_hf_task(str(model_path), task)

        # Resolve user_input_schema from registry (only needs task name)
        if self._task:
            spec = TASK_REGISTRY.get(self._task)
            if spec is not None:
                self._user_input_schema = spec.user_inputs
                self._pipeline_mapping = spec.mapping

        # Discover pipeline params from class (no model instance needed)
        self._pipeline_params = _discover_pipeline_params_from_task(self._task)

    def unload(self) -> None:
        """Release ORT session and free memory."""
        self._model = None
        self._pipeline = None
        self._user_input_schema = None
        self._pipeline_mapping = None
        self._pipeline_params = None
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

    def switch_ep(self, ep: EPNameOrAlias) -> None:
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
        inputs: dict[str, Any],
        task: str | None = None,
        **pipeline_kwargs: Any,
    ) -> PredictionResult:
        """Run inference via HF Pipeline and return structured result.

        Args:
            inputs: Named input dict matching the task's user_inputs schema.
                For pipeline tasks, values are decoded and routed
                automatically based on TASK_REGISTRY.
                For raw tensor mode (no pipeline), values are numpy-
                serialisable lists passed directly to the model.
            task: Optional task override.  When provided, input validation
                and pipeline dispatch use the schema for *task* instead of
                the model's loaded task.  This allows e.g. a
                ``feature-extraction`` model to serve ``sentence-similarity``
                requests with different input fields.
            **pipeline_kwargs: Extra keyword arguments forwarded to the
                HF pipeline (e.g. ``top_k``, ``max_new_tokens``).

        Returns:
            PredictionResult with predictions and latency.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Resolve effective task/schema/mapping as local variables so that
        # concurrent threads (via run_in_executor) don't clobber each other.
        effective_task = self._task
        effective_schema = self._user_input_schema
        effective_mapping = self._pipeline_mapping
        if task and task != self._task:
            spec = TASK_REGISTRY.get(task)
            if spec:
                effective_task = task
                effective_schema = spec.user_inputs
                effective_mapping = spec.mapping

        t0 = time.perf_counter()

        # Validate inputs against schema (skipped when schema is None)
        validated = self._validate_inputs(inputs, schema=effective_schema)

        if self._pipeline is not None:
            temp_paths: list[str] = []
            pipe_input = self._prepare_pipeline_input(
                validated,
                pipeline_kwargs,
                temp_paths,
                schema=effective_schema,
                mapping=effective_mapping,
            )
            # Filter kwargs to only those accepted by the pipeline so that
            # the UI can send a generic param set (e.g. top_k) without
            # breaking pipelines that don't support it.
            accepted = self._accepted_pipeline_kwargs()
            filtered_kwargs = (
                {k: v for k, v in pipeline_kwargs.items() if k in accepted}
                if accepted is not None
                else pipeline_kwargs
            )
            try:
                raw_result = self._pipeline(pipe_input, **filtered_kwargs)
            finally:
                for p in temp_paths:
                    Path(p).unlink(missing_ok=True)
            predictions = self._normalize_pipeline_output(
                raw_result,
                inputs=validated,
                task=effective_task,
            )
        else:
            predictions = self._predict_raw_tensors(validated)

        latency_ms = (time.perf_counter() - t0) * 1000
        self._latency_samples.append(latency_ms)
        self._request_count += 1
        self._last_request_at = datetime.now(tz=timezone.utc)

        session = getattr(self._model, "_session", None)
        ep_name = getattr(session, "_ep", self._ep)

        return PredictionResult(
            task=effective_task or "unknown",
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
        except (ImportError, OSError):
            return 0.0

    @property
    def user_input_schema(self) -> list[InputField] | None:
        """Resolved user_inputs schema from TASK_REGISTRY, or None."""
        return self._user_input_schema

    @property
    def pipeline_params(self) -> list[dict] | None:
        """Discovered pipeline parameters from _sanitize_parameters, or None."""
        return self._pipeline_params

    @property
    def pipeline_mapping(self) -> PipelineMapping | None:
        """Resolved PipelineMapping from TASK_REGISTRY, or None."""
        return self._pipeline_mapping

    # ------------------------------------------------------------------
    # Private — schema resolution
    # ------------------------------------------------------------------

    def _resolve_schema(self) -> None:
        """Resolve user_inputs and pipeline_mapping from TASK_REGISTRY."""
        if not self._task:
            return

        spec = TASK_REGISTRY.get(self._task)
        if spec is None:
            # Unregistered task — pass inputs as-is, no validation
            return

        if self._pipeline is not None:
            self._user_input_schema = spec.user_inputs
            self._pipeline_mapping = spec.mapping
        else:
            logger.warning(
                "Task '%s' is in the registry but no pipeline is available "
                "— falling through to raw tensor mode without schema validation.",
                self._task,
            )

    # ------------------------------------------------------------------
    # Private — input validation
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        inputs: dict[str, Any],
        *,
        schema: list[InputField] | None | Any = _UNSET,
    ) -> dict[str, Any]:
        """Validate inputs against schema and inject defaults.

        Returns a new dict with defaults applied for missing optional fields.
        When schema is None (raw tensor / unregistered task), returns inputs as-is.
        """
        if schema is _UNSET:
            schema = self._user_input_schema
        if schema is None:
            return inputs

        # Lazy import for PIL type check
        try:
            from PIL import Image

            image_types: tuple[type, ...] = (bytes, Image.Image)
        except ImportError:
            image_types = (bytes,)

        type_checks: dict[str, tuple[type, ...]] = {
            "image": image_types,
            "audio": (bytes, dict),
            "video": (bytes, str),
            "text": (str,),
            "json": (dict, list),
            "number": (int, float),
            "boolean": (bool,),
        }

        schema_names = {f.name for f in schema}

        # Reject unknown inputs (catches typos early)
        unknown = set(inputs.keys()) - schema_names
        if unknown:
            raise ValueError(
                f"Unknown input(s): {sorted(unknown)}. This model accepts: {sorted(schema_names)}"
            )

        result = dict(inputs)
        for field in schema:
            # Check required fields
            if field.required and field.name not in result:
                raise ValueError(
                    f"Missing required input '{field.name}'. "
                    f"This model expects: {[f.name for f in schema]}"
                )

            # Inject defaults for missing optional fields
            if not field.required and field.name not in result and field.default is not None:
                result[field.name] = field.default

            # Type check
            if field.name in result:
                value = result[field.name]
                # bool must be checked before number (isinstance(True, int) is True)
                if field.type != "boolean" and isinstance(value, bool):
                    raise TypeError(f"Input '{field.name}' expects type '{field.type}', got bool")
                expected = type_checks.get(field.type)
                if expected and not isinstance(value, expected):
                    raise TypeError(
                        f"Input '{field.name}' expects type '{field.type}' "
                        f"({expected}), got {type(value).__name__}"
                    )

        return result

    # ------------------------------------------------------------------
    # Private — pipeline dispatch (registry-driven, zero if-branches)
    # ------------------------------------------------------------------

    def _prepare_pipeline_input(
        self,
        inputs: dict[str, Any],
        pipeline_kwargs: dict[str, Any],
        temp_paths: list[str] | None = None,
        *,
        schema: list[InputField] | None | Any = _UNSET,
        mapping: PipelineMapping | None | Any = _UNSET,
    ) -> Any:
        """Convert validated inputs → pipeline positional argument.

        Also routes pipe_kwargs inputs into pipeline_kwargs (mutates in-place).
        Binary decoding (image→PIL, audio→dict, video→path) is inferred from
        the schema type — no per-task branching.

        Args:
            temp_paths: If provided, video temp-file paths are appended so
                callers can clean them up after inference.
            schema: Override for self._user_input_schema (thread-safe dispatch).
            mapping: Override for self._pipeline_mapping (thread-safe dispatch).
        """
        if mapping is _UNSET:
            mapping = self._pipeline_mapping
        if mapping is None:
            # Unregistered task: pass inputs as-is
            return inputs

        if schema is _UNSET:
            schema = self._user_input_schema
        schema = schema or []

        # 1. Decode binary inputs (skip if already decoded, e.g. PIL from Gradio)
        decoded = dict(inputs)
        for field in schema:
            if field.type in BINARY_TYPES and field.name in decoded:
                val = decoded[field.name]
                if isinstance(val, bytes):
                    decoded[field.name] = _DECODERS[field.type](val)
                    if field.type == "video" and temp_paths is not None:
                        temp_paths.append(decoded[field.name])

        # 2. Route kwargs-bound inputs into pipeline_kwargs
        for name in mapping.pipe_kwargs:
            if name in decoded:
                pipeline_kwargs[name] = decoded[name]

        # 3. Build the positional argument
        if isinstance(mapping.pipe_input, str):
            return decoded[mapping.pipe_input]

        result = {k: decoded[k] for k in mapping.pipe_input if k in decoded}
        if mapping.pipe_input_as_list:
            return list(result.values())
        return result

    # ------------------------------------------------------------------
    # Private — pipeline creation
    # ------------------------------------------------------------------

    def _create_pipeline(self) -> Any:
        """Create HF pipeline via the shared factory."""
        if self._task is None or self._model is None:
            return None

        from .pipeline import create_pipeline

        return create_pipeline(self._task, self._model, self._model_id)

    # ------------------------------------------------------------------
    # Private — pipeline kwarg filtering
    # ------------------------------------------------------------------

    def _accepted_pipeline_kwargs(self) -> frozenset[str] | None:
        """Return the set of kwarg names accepted by this pipeline.

        Built from the already-discovered ``pipeline_params``.  Returns
        ``None`` when discovery hasn't run (caller should pass kwargs
        through unfiltered in that case).
        """
        if self._pipeline_params is None:
            return None
        return frozenset(p["name"] for p in self._pipeline_params)

    # ------------------------------------------------------------------
    # Private — output normalization
    # ------------------------------------------------------------------

    def _normalize_pipeline_output(
        self,
        raw: Any,
        inputs: dict[str, Any] | None = None,
        *,
        task: str | None | Any = _UNSET,
    ) -> list[Prediction] | dict[str, Any]:
        """Convert HF pipeline output to our standard format.

        If the task has a registered ``postprocess`` callback in
        TASK_REGISTRY, it is called first.  Otherwise the default
        heuristic-based normalisation runs.
        """
        if task is _UNSET:
            task = self._task
        # Registry-driven postprocess — lets tasks opt-in to custom
        # output transformation without any if/else branching here.
        spec = TASK_REGISTRY.get(task or "")
        if spec and spec.postprocess is not None:
            return spec.postprocess(raw, pipeline=self._pipeline, inputs=inputs)

        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            # Classification / detection: list of {"label": ..., "score": ...}
            if "label" in raw[0] and "score" in raw[0]:
                return [
                    Prediction(
                        label=str(item["label"]),
                        score=round(float(item["score"]), 6) if item["score"] is not None else None,
                    )
                    for item in raw
                ]
            # Non-classification list of dicts (e.g. text-generation, NER).
            # Sanitize numpy scalars so pydantic/JSON serialization works
            # (NER pipelines return np.float32 scores).
            result = raw[0] if len(raw) == 1 else {"results": raw}
            return _sanitize_numpy(result)
        # Other tasks: return as-is dict
        if isinstance(raw, dict):
            return _sanitize_numpy(raw)
        # Fallback
        return {"raw": str(raw)}

    # ------------------------------------------------------------------
    # Private — raw tensor inference (no pipeline)
    # ------------------------------------------------------------------

    def _predict_raw_tensors(self, tensor_inputs: dict[str, Any]) -> dict[str, Any]:
        """Bypass pipeline: run model directly with pre-processed tensors."""
        import numpy as np
        import torch

        inputs_torch = {
            k: torch.from_numpy(np.array(v)) if not isinstance(v, torch.Tensor) else v
            for k, v in tensor_inputs.items()
        }
        output = self._model(**inputs_torch)

        # Convert output to serializable dict — iterate dynamically over
        # all output keys (ModelOutput is dict-like) to avoid hardcoding
        # architecture-specific attribute names.
        result: dict[str, Any] = {}
        items = output.items() if hasattr(output, "items") else ()
        for key, val in items:
            if val is None:
                continue
            try:
                if isinstance(val, torch.Tensor):
                    result[key] = val.detach().cpu().numpy().tolist()
                else:
                    result[key] = val
            except Exception:
                result[key] = str(val)
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
        ep: EPNameOrAlias | None,
    ) -> None:
        onnx_path, manifest = _find_build_artifacts(build_dir, task=task)

        model_id: str | None = None
        if manifest is not None:
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

    @staticmethod
    def _resolve_model_id_from_dir(build_dir: Path) -> str | None:
        """Extract model_id from any manifest in the directory (task-agnostic)."""
        for manifest_path in build_dir.glob("*build_manifest.json"):
            manifest = json.loads(manifest_path.read_text())
            model_id = manifest.get("model_id")
            if model_id:
                return model_id
        return None

    def _load_from_onnx(
        self,
        onnx_path: Path,
        *,
        task: str | None,
        device: str,
        ep: EPNameOrAlias | None,
        skip_build: bool = True,
    ) -> None:
        from ..models.auto import WinMLAutoModel

        self._task = task
        self._model_id = None
        self._model = WinMLAutoModel.from_onnx(
            onnx_path, task=task, device=device, ep=ep, skip_build=skip_build
        )
        logger.info("Loaded from ONNX: %s task=%s skip_build=%s", onnx_path, task, skip_build)

    def _load_from_hf(
        self,
        model_id: str,
        *,
        task: str | None,
        device: str,
        ep: EPNameOrAlias | None,
        allow_unsupported_nodes: bool = False,
    ) -> None:
        from ..models.auto import WinMLAutoModel

        self._model_id = model_id
        self._model = WinMLAutoModel.from_pretrained(
            model_id,
            task=task,
            device=device,
            ep=ep,
            allow_unsupported_nodes=allow_unsupported_nodes,
        )
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


# ---------------------------------------------------------------------------
# Pipeline parameter discovery (§8 of design doc)
# ---------------------------------------------------------------------------


# Sample values for -P parameters, keyed by discovered type
_PARAM_TYPE_SAMPLES: dict[str, str] = {
    "integer": "5",
    "number": "0.7",
}

# Well-known HF pipeline params whose defaults are hidden behind sentinels
# (e.g. top_k="" in TextClassificationPipeline).  Stores (type, sample_value)
# so we can fix the discovered type when the signature default is a sentinel.
_WELL_KNOWN_PARAMS: dict[str, tuple[str, Any]] = {
    # Classification / detection
    "top_k": ("integer", 5),
    "threshold": ("number", 0.5),
    # Token classification (NER)
    "aggregation_strategy": ("string", "simple"),
    "ignore_labels": ("string", '["O"]'),
    "stride": ("integer", 0),
    # Generation
    "top_p": ("number", 0.9),
    "temperature": ("number", 0.7),
    "max_new_tokens": ("integer", 100),
    "max_length": ("integer", 512),
    "num_beams": ("integer", 4),
    # QA
    "doc_stride": ("integer", 128),
    "max_answer_len": ("integer", 15),
    # General
    "batch_size": ("integer", 1),
}


def _pick_sample_value(name: str, ptype: str, default: Any) -> str | None:
    """Pick a useful sample value for a pipeline parameter.

    Priority: actual default → well-known table → type-based placeholder.
    """
    if default is not None and str(default).strip() != "":
        return str(default)
    if name in _WELL_KNOWN_PARAMS:
        return str(_WELL_KNOWN_PARAMS[name][1])
    return _PARAM_TYPE_SAMPLES.get(ptype)


def _build_param_entry(name: str, param: inspect.Parameter) -> dict[str, Any] | None:
    """Build a single param dict from an inspect.Parameter, or None to skip."""
    entry: dict[str, Any] = {"name": name}
    default = param.default
    if default is not inspect.Parameter.empty and default is not None:
        entry["type"] = _PY_TYPE_TO_SCHEMA.get(type(default), "any")
        entry["default"] = default
    else:
        entry["type"] = "any"
    # When the signature uses a sentinel default (e.g. top_k=""), the
    # discovered type may be wrong.  Override with the well-known type.
    if name in _WELL_KNOWN_PARAMS and entry["type"] in ("any", "string"):
        entry["type"] = _WELL_KNOWN_PARAMS[name][0]
    sample = _pick_sample_value(name, entry["type"], entry.get("default"))
    if sample is not None:
        entry["sample_value"] = sample
    # Only expose params that have a known default or sample value;
    # params with neither (e.g. offset_mapping) are internal noise.
    if "default" not in entry and "sample_value" not in entry:
        return None
    return entry


def _discover_pipeline_params(pipeline: Any) -> list[dict]:
    """Extract parameters from pipeline._sanitize_parameters signature.

    Returns a list of {"name": ..., "type": ..., "default": ..., "sample_value": ...} dicts.
    Best-effort: _sanitize_parameters is a private API, signatures may
    change across transformers versions.
    """
    try:
        sig = inspect.signature(pipeline._sanitize_parameters)
    except (ValueError, TypeError):
        return []

    params: list[dict] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        entry = _build_param_entry(name, param)
        if entry is not None:
            params.append(entry)
    return params
