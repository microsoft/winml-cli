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
from .passes import BaseQuantPass, FP16Pass, QDQPass, RTNPass


logger = logging.getLogger(__name__)

# Precision strings that expand to multiple sequential passes.
_COMPOSITE_PRECISIONS: dict[str, list[str]] = {
    "w4a16": ["rtn", "fp16"],
}


def expand_precision(
    mode: str,
    config: WinMLQuantizationConfig | None = None,
) -> list[BaseQuantPass]:
    """Expand a precision string into an ordered list of quantization passes.

    All passes share the same ``config`` so every pass can read the fields
    relevant to it.

    Supported values:

    ========= =======================
    mode      passes
    ========= =======================
    ``fp16``  ``[FP16Pass(config)]``
    ``rtn``   ``[RTNPass(config)]``
    ``static``  ``[QDQPass(config)]``
    ``dynamic`` ``[QDQPass(config)]``
    ``w4a16`` ``[RTNPass(config), FP16Pass(config)]``
    ========= =======================

    Args:
        mode: Precision string (e.g. ``"w4a16"``).
        config: Shared quantization configuration.  If *None*, a default
            :class:`WinMLQuantizationConfig` is used.

    Returns:
        Ordered list of :class:`~winml.modelkit.quant.passes.BaseQuantPass`
        instances ready to be executed by :class:`Quantizer`.

    Raises:
        ValueError: If *mode* is not recognised.
    """
    config = config or WinMLQuantizationConfig()

    _pass_factories: dict[str, BaseQuantPass] = {
        "fp16": FP16Pass(config),
        "rtn": RTNPass(config),
        "static": QDQPass(config),
        "dynamic": QDQPass(config),
    }

    if mode in _pass_factories:
        return [_pass_factories[mode]]

    if mode in _COMPOSITE_PRECISIONS:
        return [_pass_factories[step] for step in _COMPOSITE_PRECISIONS[mode]]

    raise ValueError(
        f"Unknown precision mode {mode!r}. "
        f"Valid values: {sorted(_pass_factories) + sorted(_COMPOSITE_PRECISIONS)}"
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

        config = WinMLQuantizationConfig(mode="w4a16", rtn_bits=4)
        quantizer = Quantizer(expand_precision("w4a16", config))
        result = quantizer.run("model.onnx", "model_w4a16.onnx")
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
    - ``"w4a16"`` — RTN int4 followed by FP16 conversion

    Args:
        model_path: Path to input ONNX model.
        output_path: Path for output model (defaults to ``{model_stem}_qdq.onnx``).
        config: Quantization configuration (uses defaults if *None*).

    Returns:
        :class:`QuantizeResult` with path to final output model and metrics.

    Examples:
        >>> result = quantize_onnx("model.onnx", config=WinMLQuantizationConfig(mode="rtn"))
        >>> result = quantize_onnx("model.onnx", config=WinMLQuantizationConfig(mode="fp16"))
        >>> result = quantize_onnx("model.onnx", config=WinMLQuantizationConfig(mode="w4a16"))
    """
    model_path = Path(model_path)
    config = config or WinMLQuantizationConfig()

    if output_path is not None:
        output_path = Path(output_path)
    else:
        output_path = model_path.parent / f"{model_path.stem}_qdq.onnx"

    # Apply model-type-specific quant finalizer if registered. Some model types
    # finalize calibration reader / nodes-to-exclude / dtypes only once the
    # exported ONNX exists.
    if config.model_type and config.calibration_data is None:
        from .calibration import get_quant_finalizer

        finalizer = get_quant_finalizer(config.model_type)
        if finalizer is not None:
            config = finalizer.finalize(config, onnx_path=model_path, model_id=config.model_id)

    if kwargs:
        raise TypeError(f"quantize_onnx() got unexpected keyword arguments: {sorted(kwargs)}")
    passes = expand_precision(config.mode, config)
    return Quantizer(passes).run(model_path, output_path, use_external_data=use_external_data)
