# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import onnx

from ...onnx import ONNXDomain
from ...pattern.op_input_gen import get_runtime_checker_op
from ...pattern.op_input_gen.op_input_gen import (
    InputShapeConstraint,
    OpInputGenerator,
    model_from_b64,
)
from .check_ops import get_ep_checker
from .runner import ResilientRunner


_FAILED_TO_FREE_LIBRARY_TOKEN = "failed to free library"  # noqa: S105
_DURATION_US_RE = re.compile(r"\(\s*\d+\s*us\)")
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?")


class NeedRestartError(RuntimeError):
    """Raised when a fatal provider error requires a machine restart."""


def _strip_us_durations(text: str) -> str:
    return _DURATION_US_RE.sub("", text)


def _strip_timestamps(text: str) -> str:
    return _TIMESTAMP_RE.sub("<ts>", text)


def _clean_result_payload(result: dict[str, Any]) -> None:
    if not isinstance(result, dict):
        return

    def _clean_string(val: str) -> str:
        return _strip_timestamps(_strip_us_durations(val))

    for key in ("stdout", "stderr"):
        val = result.get(key)
        if isinstance(val, str):
            result[key] = _clean_string(val)

    res_payload = result.get("result")
    if isinstance(res_payload, dict):
        reason = res_payload.get("reason")
        if isinstance(reason, str):
            res_payload["reason"] = _clean_string(reason)


def _contains_failed_to_free(record: dict[str, Any]) -> bool:
    token = _FAILED_TO_FREE_LIBRARY_TOKEN

    def _has_token(value: object) -> bool:
        return isinstance(value, str) and token in value.lower()

    if not isinstance(record, dict):
        return False

    result_payload = record.get("result")
    if isinstance(result_payload, dict) and _has_token(result_payload.get("reason")):
        return True

    return _has_token(record.get("stdout")) or _has_token(record.get("stderr"))


def _raise_if_fatal(record: dict[str, Any], stage: str) -> None:
    if _contains_failed_to_free(record):
        raise NeedRestartError(f"Fatal ep error during {stage}; restart recommended")


def _constraint_to_value(constraint: Any, type_annotation: str) -> Any:
    if constraint is None:
        return None

    if not isinstance(constraint, dict):
        return constraint

    constraint_type = constraint.get("type")
    if constraint_type == "shape":
        shape = constraint.get("shape", [])
        min_max = constraint.get("min_max")
        return InputShapeConstraint(shape, min_max=min_max).get_value(type_annotation)

    if constraint_type == "value":
        value = constraint.get("value")
        dtype = constraint.get("dtype")
        if dtype is not None and isinstance(value, list):
            return np.asarray(value, dtype=dtype)
        if isinstance(value, list):
            return np.asarray(value)
        return value

    if constraint_type == "variadic":
        return [
            _constraint_to_value(element, type_annotation)
            for element in constraint.get("elements", [])
        ]

    return constraint


def _build_kwargs_from_case(generator: OpInputGenerator, case: dict[str, Any]) -> dict[str, Any]:
    type_var_comb = case.get("type_vars", {})
    applied_type_annotations = {
        name: generator._apply_type_var_combination(type_annotation, type_var_comb)
        for name, type_annotation in generator.type_annotations.items()
    }

    attrs = case.get("attrs", {})
    input_constraints = case.get("input_constraints", {})

    applied_input_comb: dict[str, Any] = {}
    for input_name, constraint in input_constraints.items():
        if input_name in attrs:
            continue
        type_annotation = applied_type_annotations.get(input_name, "")
        applied_input_comb[input_name] = _constraint_to_value(constraint, type_annotation)

    kwargs = {**attrs, **applied_input_comb}

    if generator.op_variadic_input_name is not None and generator.op_variadic_input_name in kwargs:
        variadic_input = kwargs.pop(generator.op_variadic_input_name)
        for idx, tensor in enumerate(variadic_input):
            kwargs[f"{generator.op_variadic_input_name}__{idx}"] = tensor
        variadic_keys = [
            f"{generator.op_variadic_input_name}__{i}" for i in range(len(variadic_input))
        ]
        normalized_key_order = (
            generator.op_input_names + variadic_keys + generator.op_attribute_names
        )
    else:
        normalized_key_order = generator.op_input_names + generator.op_attribute_names

    kwargs = {k: kwargs[k] for k in normalized_key_order if k in kwargs}
    return generator.filter_kwargs_by_opset(kwargs)


def _normalize_qdq_types(case: dict[str, Any]) -> dict[str, Any] | None:
    qdq_types = case.get("qdq_types")
    if not isinstance(qdq_types, dict):
        return None
    return qdq_types


class RunCaseRunner:
    """Execute runtime checker cases; op configuration is provided per call."""

    def __init__(self) -> None:
        # Reuse a single child-process runner per instance to avoid per-case spawn cost.
        self._runner: ResilientRunner | None = ResilientRunner(capture_output=True, timeout_sec=60)

    def close(self) -> None:
        """Shut down the underlying child-process runner and release resources."""
        try:
            if self._runner is not None:
                self._runner.shutdown()
        except Exception:
            pass

    def _ensure_runner(self) -> ResilientRunner:
        if self._runner is None:
            self._runner = ResilientRunner(capture_output=True, timeout_sec=60)
        return self._runner

    def run_case_check_result(
        self,
        case: dict[str, Any],
        op_name: str,
        op_domain: str,
        opset_version: int,
        ep_name: str,
        device: str,
        timing_hook: Callable[[dict[str, float]], None] | None = None,
    ) -> dict[str, Any]:
        """Run a single case and return compile/run check results."""
        domain = ONNXDomain.from_str(op_domain)
        schema = domain.get_op_schema(op_name, opset_version)
        generator_cls = get_runtime_checker_op(op_name, domain=op_domain)
        ep_checker = get_ep_checker(ep_name, device)

        generator: OpInputGenerator = generator_cls(schema)
        runner = self._ensure_runner()

        t0 = time.perf_counter()

        kwargs = _build_kwargs_from_case(generator, case)
        qdq_types = _normalize_qdq_types(case)
        model_bytes_b64 = case.get("model_bytes_b64")
        if not isinstance(model_bytes_b64, str):
            raise TypeError("Case is missing model_bytes_b64 payload")
        model_bytes = model_from_b64(model_bytes_b64)

        # Validate that the decoded bytes form a valid ONNX model before dispatching.
        try:
            onnx.load_from_string(model_bytes)
            # some known issue with onnx.checker throwing like
            # "Field 'shape' of 'type' is required but missing"
            # onnx.checker.check_model(onnx_model)
        except Exception as exc:
            raise ValueError("Decoded model_bytes_b64 is not a valid ONNX model") from exc

        input_kwargs = {k: v for k, v in kwargs.items() if generator._is_input_key(k)}
        ep_checker_inputs = generator.create_input_dict(input_kwargs, qdq_types=qdq_types)
        t_build_done = time.perf_counter()

        compile_result = runner.run(ep_checker.check_compile, model_bytes, ep_checker_inputs)
        _clean_result_payload(compile_result)
        _raise_if_fatal(compile_result, "compile")
        t_compile_done = time.perf_counter()

        run_result = runner.run(ep_checker.check_run, model_bytes, ep_checker_inputs)
        _clean_result_payload(run_result)
        _raise_if_fatal(run_result, "run")
        t_run_done = time.perf_counter()

        if timing_hook is not None:
            timing_hook(
                {
                    "build_ms": (t_build_done - t0) * 1000,
                    "compile_ms": (t_compile_done - t_build_done) * 1000,
                    "run_ms": (t_run_done - t_compile_done) * 1000,
                    "total_ms": (t_run_done - t0) * 1000,
                }
            )

        return {
            "compile": compile_result,
            "run": run_result,
        }
