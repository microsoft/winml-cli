# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Main quantizer implementation."""

from __future__ import annotations

import logging
import tempfile
import traceback
from pathlib import Path
from typing import Any

from .config import QuantizeResult, WinMLQuantizationConfig
from .passes import BaseQuantPass, FP16Pass, RTNPass, StaticPass


logger = logging.getLogger(__name__)

# Precision strings that expand to multiple sequential passes.
_COMPOSITE_PRECISIONS: dict[str, list[str]] = {}


def expand_precision(
    precision: str | None = None,
    config: WinMLQuantizationConfig | None = None,
) -> list[BaseQuantPass]:
    """Expand a precision string into an ordered list of quantization passes.

    All passes share the same ``config`` so every pass can read the fields
    relevant to it.  When *precision* is omitted, ``config.mode`` is used so
    that ``expand_precision(config=cfg)`` works as a single-precision
    convenience.

    Supported values:

    ============= =======================
    precision     passes
    ============= =======================
    ``fp16``      ``[FP16Pass(config)]``
    ``rtn``       ``[RTNPass(config)]``
    ``static``    ``[StaticPass(config)]``
    ``dynamic``   ``[StaticPass(config)]``  (placeholder until DynamicPass is implemented)
    ============= =======================

    Args:
        precision: Precision string (e.g. ``"fp16"``).  When *None*, falls back
            to ``config.mode`` (or ``"static"`` if *config* is also *None*).
        config: Shared quantization configuration.  If *None*, a default
            :class:`WinMLQuantizationConfig` is used.

    Returns:
        Ordered list of :class:`~winml.modelkit.quant.passes.BaseQuantPass`
        instances ready to be executed by :class:`Quantizer`.

    Raises:
        ValueError: If *precision* is not recognised.
    """
    config = config or WinMLQuantizationConfig()
    effective_precision = precision if precision is not None else config.mode

    _pass_types: dict[str, type[BaseQuantPass]] = {
        "fp16": FP16Pass,
        "rtn": RTNPass,
        "static": StaticPass,
        "dynamic": StaticPass,
    }

    if effective_precision in _pass_types:
        return [_pass_types[effective_precision](config)]

    if effective_precision in _COMPOSITE_PRECISIONS:
        return [_pass_types[step](config) for step in _COMPOSITE_PRECISIONS[effective_precision]]

    raise ValueError(
        f"Unknown precision {effective_precision!r}. "
        f"Valid values: {sorted(_pass_types) + sorted(_COMPOSITE_PRECISIONS)}"
    )


class Quantizer:
    """Orchestrate a sequential pipeline of quantization passes.

    Each pass receives the output of the previous pass as its input.  For a
    single-pass pipeline no temporary files are created.  For multi-pass
    pipelines, intermediate models are written to a ``tempfile.TemporaryDirectory``
    that is cleaned up automatically on success *or* failure.

    :class:`QuantizeResult` fields are merged across passes:

    - ``success`` — logical AND of all pass results
    - ``output_path`` — path written by the final pass
    - Timing fields — summed across passes
    - ``nodes_quantized`` — summed across passes
    - ``errors`` / ``warnings`` — concatenated across passes

    Example::

        from winml.modelkit.quant import Quantizer, expand_precision, WinMLQuantizationConfig

        config = WinMLQuantizationConfig(mode="rtn", rtn_bits=4)
        quantizer = Quantizer(expand_precision("rtn", config))
        result = quantizer.run("model.onnx", "model_rtn.onnx")
    """

    def __init__(self, passes: list[BaseQuantPass]) -> None:
        if not passes:
            raise ValueError("Quantizer requires at least one pass.")
        self._passes = passes

    @property
    def passes(self) -> list[BaseQuantPass]:
        """Return a copy of the pass list."""
        return list(self._passes)

    def run(
        self,
        model_path: str | Path,
        output_path: str | Path,
        *,
        use_external_data: bool = True,
    ) -> QuantizeResult:
        """Run the quantization pipeline.

        Args:
            model_path: Path to the input ONNX model.
            output_path: Path for the final output model.
            use_external_data: Whether to write large tensors as external data.

        Returns:
            Merged :class:`~winml.modelkit.quant.config.QuantizeResult`.
        """
        model_path = Path(model_path)
        output_path = Path(output_path)

        if not model_path.exists():
            return QuantizeResult(
                success=False,
                output_path=None,
                errors=[f"Model not found: {model_path}"],
            )

        if len(self._passes) == 1:
            return self._run_pass(self._passes[0], model_path, output_path, use_external_data)

        return self._run_multi_pass(model_path, output_path, use_external_data)

    def _run_pass(
        self,
        pass_: BaseQuantPass,
        model_path: Path,
        output_path: Path,
        use_external_data: bool,
    ) -> QuantizeResult:
        try:
            return pass_.run(model_path, output_path, use_external_data=use_external_data)
        except Exception:
            logger.exception("Pass %s failed", type(pass_).__name__)
            return QuantizeResult(
                success=False,
                output_path=None,
                errors=[traceback.format_exc()],
            )

    def _run_multi_pass(
        self,
        model_path: Path,
        output_path: Path,
        use_external_data: bool,
    ) -> QuantizeResult:
        accumulated = QuantizeResult(success=True, output_path=None)

        with tempfile.TemporaryDirectory(prefix="winml_quant_") as tmp_dir:
            current_input = model_path

            for i, pass_ in enumerate(self._passes):
                is_last = i == len(self._passes) - 1
                if is_last:
                    current_output = output_path
                else:
                    current_output = Path(tmp_dir) / f"pass_{i}_{type(pass_).__name__}.onnx"

                logger.info(
                    "Pass %d/%d: %s  %s -> %s",
                    i + 1,
                    len(self._passes),
                    type(pass_).__name__,
                    current_input.name,
                    current_output.name,
                )

                result = self._run_pass(pass_, current_input, current_output, use_external_data)
                accumulated = _merge_results(accumulated, result)

                if not result.success:
                    logger.error("Pass %s failed — aborting pipeline.", type(pass_).__name__)
                    break

                current_input = current_output

        return accumulated


def _merge_results(base: QuantizeResult, new: QuantizeResult) -> QuantizeResult:
    """Merge two QuantizeResult objects, accumulating stats."""
    return QuantizeResult(
        success=base.success and new.success,
        output_path=new.output_path if new.output_path is not None else base.output_path,
        calibration_path=new.calibration_path or base.calibration_path,
        calibration_time_seconds=base.calibration_time_seconds + new.calibration_time_seconds,
        qdq_insertion_time_seconds=base.qdq_insertion_time_seconds + new.qdq_insertion_time_seconds,
        postproc_time_seconds=base.postproc_time_seconds + new.postproc_time_seconds,
        total_time_seconds=base.total_time_seconds + new.total_time_seconds,
        nodes_quantized=base.nodes_quantized + new.nodes_quantized,
        nodes_skipped=base.nodes_skipped + new.nodes_skipped,
        errors=base.errors + new.errors,
        warnings=base.warnings + new.warnings,
    )


def _check_input_model_opset(model_path: Path) -> str | None:
    """Return a clear error message if *model_path* is empty/corrupt, else None.

    Mirrors ORT's ``get_opset_version`` requirement: a usable model must declare
    a default (``""`` / ``ai.onnx``) opset import. A zero-byte or truncated file
    parses into an (almost) empty ModelProto with no such opset import — the
    signature of a previous stage that failed to finish writing (most commonly
    because it ran out of disk space). Detecting it here lets us surface the
    real cause instead of ORT's opaque "Failed to find proper ai.onnx domain".

    A zero-byte file (the most common disk-full artefact) is caught up front
    with a cheap ``stat`` so the healthy success path never pays for a full
    proto parse. The full parse via ``onnx.load_model`` (graph only — no
    external weights, so it never trips over a missing ``.data`` sidecar) is the
    fallback for the rarer truncated-but-nonzero case.
    """
    from onnx import load_model

    # Fast path: a zero-byte output is the most common disk-full artefact.
    try:
        if model_path.stat().st_size == 0:
            return (
                f"Input ONNX model is empty (zero bytes): {model_path}. "
                "A previous build stage may have run out of disk space. "
                "Free up disk space and rebuild."
            )
    except OSError:
        # stat() failing is unexpected (existence was already checked); fall
        # through to the full parse, which surfaces a clear error either way.
        pass

    try:
        model = load_model(str(model_path), load_external_data=False)
    except Exception as e:
        return (
            f"Input ONNX model could not be parsed: {model_path} ({e}). "
            "The file may be truncated or corrupt — for example, a previous "
            "build stage may have run out of disk space. Free up disk space "
            "and rebuild."
        )

    has_default_opset = any(opset.domain in ("", "ai.onnx") for opset in model.opset_import)
    if not has_default_opset:
        return (
            f"Input ONNX model is empty or corrupt (no ai.onnx opset import): "
            f"{model_path}. It may have been truncated by a previous failed "
            "write (e.g. insufficient disk space). Free up disk space and rebuild."
        )
    return None


def quantize_onnx(
    model_path: str | Path,
    output_path: str | Path | None = None,
    config: WinMLQuantizationConfig | None = None,
    **kwargs: Any,
) -> QuantizeResult:
    """Quantize an ONNX model.

    Backward-compatible entry point.  Internally builds a :class:`Quantizer`
    pipeline from ``config.mode`` via :func:`expand_precision`.

    The quantization mode is driven by ``config.mode``:

    - ``"fp16"`` — FP16 conversion (no quantization)
    - ``"rtn"`` — RTN weight-only quantization
    - ``"static"`` / ``"dynamic"`` — QDQ calibrated quantization

    Args:
        model_path: Path to input ONNX model.
        output_path: Path for output model (defaults to ``{model_stem}_quantized.onnx``).
        config: Quantization configuration (uses defaults if *None*).

    Returns:
        :class:`QuantizeResult` with path to final output model and metrics.

    Examples:
        >>> result = quantize_onnx("model.onnx", config=WinMLQuantizationConfig(mode="rtn"))
        >>> result = quantize_onnx("model.onnx", config=WinMLQuantizationConfig(mode="fp16"))
    """
    model_path = Path(model_path)
    config = config or WinMLQuantizationConfig()

    if output_path is not None:
        output_path = Path(output_path)
    else:
        output_path = model_path.parent / f"{model_path.stem}_quantized.onnx"

    use_external_data: bool = kwargs.pop("use_external_data", True)
    if kwargs:
        raise TypeError(f"quantize_onnx() got unexpected keyword arguments: {sorted(kwargs)}")

    # Guard against an empty/corrupt input model before building the pipeline.
    # A previous stage that ran out of disk space can leave a truncated/zero-byte
    # .onnx behind; without this check a pass fails deep inside ORT with the
    # opaque "Failed to find proper ai.onnx domain". Surface the real cause
    # instead, and catch it before the model-type finalizer reads the model. A
    # missing file is left to Quantizer.run(), which reports a clear
    # "Model not found".
    if model_path.exists():
        opset_error = _check_input_model_opset(model_path)
        if opset_error is not None:
            return QuantizeResult(
                success=False,
                output_path=None,
                errors=[opset_error],
            )

    # Apply model-type-specific quant finalizer if registered. Some model types
    # finalize calibration reader / nodes-to-exclude / dtypes only once the
    # exported ONNX exists.
    if config.model_type and config.calibration_data is None:
        from .calibration import get_quant_finalizer

        finalizer = get_quant_finalizer(config.model_type)
        if finalizer is not None:
            config = finalizer.finalize(config, onnx_path=model_path, model_id=config.model_id)

    passes = expand_precision(config=config)
    return Quantizer(passes).run(model_path, output_path, use_external_data=use_external_data)
