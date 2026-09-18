# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML Runtime inference backend (``windowsml.runtime`` pipeline API).

:class:`WinMLRuntimeSession` runs prebuilt ONNX and MLIR models through the
Windows ML Runtime pipeline API, exposing the same small
surface (``io_config`` / ``run`` / ``perf`` / ``device`` / ``ep_name``) as the
ORT-backed :class:`~winml.modelkit.session.session.WinMLSession`. Both satisfy
the :class:`~winml.modelkit.session.backend.InferenceBackend` protocol, so
``perf`` and ``eval`` can pick a backend uniformly through
:func:`~winml.modelkit.session.backend.create_session`.

Unlike the ORT path (which may run the full export/optimize/quantize/compile
pipeline), this backend loads a prebuilt ``.onnx`` or ``.mlir`` directly via
``Runtime.load_model`` and drives inference through a single model stage. Output
retrieval is by tensor name for ORT-backed stages (``OrtNamedBindings.output``)
and by ordinal for non-ORT stages (``Stage.output``); both return caller-owned
NumPy copies, so ``run`` yields real named outputs for parity/eval.

Runtime availability is optional: importing ``windowsml.runtime`` loads the
preview ``WinMLRuntimeCore.dll`` and, for ONNX models, a matching
``onnxruntime-windowsml`` ORT distribution. When either is absent the import is
guarded and surfaced as a clear, actionable :class:`click.ClickException`.
"""

from __future__ import annotations

import ctypes
import json
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast

import click

from ..export.cgc.artifacts import cgc_metadata_path
from ._runtime_import import import_runtime


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    import numpy as np

    from ..utils.constants import RuntimeBackend
    from .ep_registry import WinMLEPDevice
    from .session import PerfContext
    from .stats import PerfStats

logger = logging.getLogger(__name__)


# =============================================================================
# Pure helpers (no native dependency -- unit-testable in isolation)
# =============================================================================

# WinML Runtime ``TensorDataType`` enum name -> NumPy dtype string. Keyed by the
# enum *name* so these helpers never import the native ``windowsml.runtime``
# module. Only types ``Runtime.tensor_from_numpy`` round-trips are listed;
# BFLOAT16 / INT4 / UINT4 / UNDEFINED have no NumPy equivalent here.
_TENSOR_DTYPE_TO_NUMPY: dict[str, str] = {
    "FLOAT32": "float32",
    "FLOAT16": "float16",
    "FLOAT64": "float64",
    "INT8": "int8",
    "UINT8": "uint8",
    "INT16": "int16",
    "UINT16": "uint16",
    "INT32": "int32",
    "UINT32": "uint32",
    "INT64": "int64",
    "UINT64": "uint64",
    "BOOL": "bool",
}

# ModelKit device class -> WinML Runtime ``ExecutionTargetKind`` member name.
_DEVICE_TO_KIND_NAME: dict[str, str] = {"cpu": "CPU", "gpu": "GPU", "npu": "NPU"}
_DYNAMIC_DIM_SENTINELS = {(1 << 64) - 1}


def _numpy_dtype_for(data_type: Any) -> str:
    """Map a Runtime ``TensorDataType`` (enum or name) to a NumPy dtype string.

    Raises a clear :class:`click.ClickException` for tensor element types the
    Runtime backend cannot materialize as NumPy arrays (e.g. bfloat16, int4).
    """
    name = getattr(data_type, "name", str(data_type))
    dtype = _TENSOR_DTYPE_TO_NUMPY.get(name)
    if dtype is None:
        raise click.ClickException(
            f"--runtime winml-runtime cannot benchmark a model whose I/O uses tensor "
            f"data type {name!r}: it has no NumPy mapping "
            "(bfloat16 / int4 / uint4 / undefined are unsupported)."
        )
    return dtype


def _shape_with_dynamic_dims(shape: Any) -> list[int | None]:
    """Normalize a Runtime schema shape to the dynamic-dim convention.

    Model schema reports a free dimension as ``0`` and post-build stage schema
    reports it as ``UINT64_MAX``. Perf treats ``None`` as dynamic, so normalize
    both Runtime representations.
    """
    resolved: list[int | None] = []
    for dim in shape:
        d = int(dim)
        resolved.append(None if d <= 0 or d in _DYNAMIC_DIM_SENTINELS else d)
    return resolved


def synth_io_config(model_schema: Any, ort_schema: Any | None) -> dict[str, Any]:
    """Build an ``io_config`` dict from Runtime schema objects.

    ``model_schema`` provides ordinal ``(dtype, shape)`` descriptors
    (:meth:`ModelSchema.input_desc`); ``ort_schema`` provides tensor names
    (:meth:`OrtModelSchema.input_name`). When ``ort_schema`` is ``None`` (a
    model that does not expose ONNX name metadata), positional
    ``input_{i}`` / ``output_{i}`` names are synthesized so ordinal binding
    still works.

    The returned dict matches the keys ``generate_random_inputs`` consumes
    (``input_names`` / ``input_shapes`` / ``input_types``) plus output metadata
    for reporting and output retrieval.
    """
    input_names: list[str] = []
    input_shapes: list[list[int | None]] = []
    input_types: list[str] = []
    for i in range(model_schema.input_count):
        dtype, shape = model_schema.input_desc(i)
        name = ort_schema.input_name(i) if ort_schema is not None else f"input_{i}"
        input_names.append(name or f"input_{i}")
        input_shapes.append(_shape_with_dynamic_dims(shape))
        input_types.append(_numpy_dtype_for(dtype))

    output_names: list[str] = []
    output_shapes: list[list[int | None]] = []
    output_types: list[str] = []
    for i in range(model_schema.output_count):
        dtype, shape = model_schema.output_desc(i)
        oname = ort_schema.output_name(i) if ort_schema is not None else f"output_{i}"
        output_names.append(oname or f"output_{i}")
        output_shapes.append(_shape_with_dynamic_dims(shape))
        output_types.append(_numpy_dtype_for(dtype))

    return {
        "input_names": input_names,
        "input_shapes": input_shapes,
        "input_types": input_types,
        "output_names": output_names,
        "output_shapes": output_shapes,
        "output_types": output_types,
    }


def _apply_io_metadata(io_config: dict[str, Any], model_path: Path) -> None:
    """Preserve ONNX input ranges and restore names for CGC MLIR stages."""
    if model_path.suffix.lower() == ".onnx":
        from ..onnx import get_io_config

        source_io = get_io_config(model_path)
        io_config["value_ranges"] = {
            name: value_range
            for name, value_range in source_io["value_ranges"].items()
            if name in io_config["input_names"]
        }
        return

    metadata_path = cgc_metadata_path(model_path)
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    input_names = [item["name"] for item in metadata.get("inputs", [])]
    output_names = [item["name"] for item in metadata.get("outputs", [])]
    if len(input_names) != len(io_config["input_names"]):
        raise click.ClickException(f"Runtime input count does not match {metadata_path.name}.")
    if len(output_names) != len(io_config["output_names"]):
        raise click.ClickException(f"Runtime output count does not match {metadata_path.name}.")
    io_config["input_names"] = input_names
    io_config["output_names"] = output_names


def resolve_provider_kind(
    device: str | None,
    ep: str | None,
    ep_source: str | None,
    *,
    resolve_fn: Callable[[str, str, str | None], tuple[str, str]] | None = None,
) -> tuple[str, str]:
    """Resolve ``--device`` / ``--ep`` to a concrete ``(provider_name, device_class)``.

    By default this reuses ModelKit's :func:`resolve_device` -- the exact
    resolution the ORT ``--runtime winml-ort`` path uses -- so device/EP semantics
    stay consistent across backends. ``resolve_fn`` is an injection seam for
    tests: it receives ``(ep, device, source)`` and returns
    ``(full_ep_name, device_class)``.
    """
    if resolve_fn is not None:
        return resolve_fn(ep or "auto", device or "auto", ep_source)

    from .ep_device import EPDeviceTarget, resolve_device

    resolved = resolve_device(
        EPDeviceTarget(ep=ep or "auto", device=device or "auto", source=ep_source)
    )
    return resolved.ep, resolved.device


def kind_name_for_device(device_class: str) -> str:
    """Map a resolved device class to an ``ExecutionTargetKind`` member name."""
    try:
        return _DEVICE_TO_KIND_NAME[device_class]
    except KeyError:
        raise click.ClickException(
            f"--runtime winml-runtime cannot target device {device_class!r}; "
            f"expected one of {sorted(_DEVICE_TO_KIND_NAME)}."
        ) from None


def _stage_schema(wr: Any, stage: Any) -> Any:
    """Return the authoritative post-build stage schema.

    The Runtime projection does not yet expose ``Stage.schema()`` in all preview
    wheels. Its ``IWinMLStageSchema`` ABI matches the descriptor surface wrapped
    by ``ModelSchema``, so use that wrapper until the public convenience method
    is available.
    """
    schema_factory = getattr(stage, "schema", None)
    if callable(schema_factory):
        return schema_factory()

    bindings = getattr(wr, "_b", None)
    schema_type = getattr(wr, "ModelSchema", None)
    if bindings is None or schema_type is None:
        raise click.ClickException(
            "The installed windowsml Runtime projection does not expose stage schema access."
        )

    interface = stage.interface.QueryInterface(bindings.IWinMLStageSchema)
    return schema_type(interface, stage)


def _to_numpy(value: Any) -> np.ndarray:
    """Coerce a run input value (numpy array or torch tensor) to a NumPy array."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return value
    # Duck-type torch tensors without importing torch: detach + cpu + numpy.
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return cast("np.ndarray", value.detach().cpu().numpy())
    if hasattr(value, "numpy"):
        return cast("np.ndarray", value.numpy())
    return np.asarray(value)


# =============================================================================
# Native glue (windowsml.runtime)
# =============================================================================
class _DXCoreAdapter:
    """Owned ``IDXCoreAdapter`` pointer kept alive with the Runtime target."""

    def __init__(self, pointer: ctypes.c_void_p, module: Any) -> None:
        if not pointer.value:
            raise ValueError("DXCore adapter pointer must not be null.")
        self._pointer = pointer
        self._module = module

    @property
    def pointer(self) -> int:
        assert self._pointer.value is not None
        return self._pointer.value

    @staticmethod
    def _method(
        interface: ctypes.c_void_p,
        index: int,
        restype: Any,
        *argtypes: Any,
    ) -> Any:
        vtable = ctypes.cast(interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtable[index])

    @classmethod
    def from_luid(cls, luid_value: int) -> _DXCoreAdapter:
        """Resolve an owned adapter through ``IDXCoreAdapterFactory``."""
        from comtypes import GUID  # type: ignore[import-not-found, import-untyped, unused-ignore]

        class LUID(ctypes.Structure):
            _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]

        if not 0 <= luid_value <= 0xFFFFFFFFFFFFFFFF:
            raise click.ClickException(f"Adapter LUID is outside uint64 range: {luid_value!r}.")

        try:
            dxcore = ctypes.WinDLL("dxcore.dll")
        except (AttributeError, OSError) as exc:
            raise click.ClickException(f"Could not load dxcore.dll: {exc}") from exc

        factory_iid = GUID("{78EE5945-C36E-4B13-A669-005DD11C0F06}")
        adapter_iid = GUID("{F0DB4C7F-FE5A-42A2-BD62-F2A6CF6FC83E}")
        create_factory = dxcore.DXCoreCreateAdapterFactory
        create_factory.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
        create_factory.restype = ctypes.c_long

        factory = ctypes.c_void_p()
        hr = create_factory(ctypes.byref(factory_iid), ctypes.byref(factory))
        if hr < 0:
            raise click.ClickException(
                f"DXCoreCreateAdapterFactory failed (0x{hr & 0xFFFFFFFF:08X})."
            )

        adapter = ctypes.c_void_p()
        try:
            luid = LUID(
                luid_value & 0xFFFFFFFF,
                ctypes.c_int32(luid_value >> 32).value,
            )
            get_adapter = cls._method(
                factory,
                4,
                ctypes.c_long,
                ctypes.POINTER(LUID),
                ctypes.POINTER(GUID),
                ctypes.POINTER(ctypes.c_void_p),
            )
            hr = get_adapter(
                factory,
                ctypes.byref(luid),
                ctypes.byref(adapter_iid),
                ctypes.byref(adapter),
            )
            if hr < 0:
                raise click.ClickException(
                    f"DXCore could not resolve adapter LUID {luid_value} (0x{hr & 0xFFFFFFFF:08X})."
                )
            return cls(adapter, dxcore)
        finally:
            cls._release(factory)

    @classmethod
    def _release(cls, interface: ctypes.c_void_p) -> None:
        if interface.value:
            cls._method(interface, 2, ctypes.c_ulong)(interface)
            interface.value = None

    def close(self) -> None:
        self._release(self._pointer)
        self._module = None


@dataclass(frozen=True)
class _ResolvedRuntimeTarget:
    execution_target: Any
    device_class: str
    provider_name: str | None = None
    adapter: _DXCoreAdapter | None = None


# WinML native failures surface as ``windowsml.runtime.WinMLError`` carrying an
# ``hresult`` attribute. A couple of them are common enough during setup that a
# raw traceback is unhelpful, so they are translated into actionable guidance.
_HRESULT_REVISION_MISMATCH = 0x8007051A  # ERROR_REVISION_MISMATCH
_HRESULT_NOT_SUPPORTED = 0x80070032  # ERROR_NOT_SUPPORTED


def _winml_hresult(exc: BaseException) -> int | None:
    """Return a WinMLError's normalized 32-bit HRESULT, or ``None`` if not one."""
    hr = getattr(exc, "hresult", None)
    if isinstance(hr, int):
        return hr & 0xFFFFFFFF
    return None


def _winml_click_error(
    exc: BaseException,
    hr: int,
    phase: str,
    *,
    provider_name: str | None = None,
    device_class: str | None = None,
) -> click.ClickException:
    """Map a native ``(hresult, phase)`` to a clear, actionable ClickException."""
    hex_hr = f"0x{hr:08X}"
    if hr == _HRESULT_REVISION_MISMATCH:
        return click.ClickException(
            "--runtime winml-runtime: the WinML Runtime and 'onnxruntime-windowsml' are "
            f"version-incompatible ({hex_hr}). The Runtime core requires a newer ORT API "
            "than the installed 'onnxruntime-windowsml' provides. Install an "
            "'onnxruntime-windowsml' build that matches your 'windowsml' Runtime version."
        )
    if hr == _HRESULT_NOT_SUPPORTED and phase == "build":
        provider = provider_name or "the requested execution provider"
        device = repr(device_class) if device_class else "the requested device"
        return click.ClickException(
            f"--runtime winml-runtime: building the pipeline for {provider} on {device} is "
            f"not supported by this WinML Runtime build ({hex_hr}). The execution provider "
            "may be unavailable in this environment (for example the ORT provider bridge "
            "failed to load -- look for an 'Init provider bridge failed' warning above). "
            "Try '--device cpu', or install an 'onnxruntime-windowsml' build that includes "
            "this provider."
        )
    label = {"load": "loading the model", "build": "building the pipeline", "run": "inference"}.get(
        phase, phase
    )
    return click.ClickException(f"--runtime winml-runtime: {label} failed ({hex_hr}): {exc}")


@contextmanager
def _translate_native_errors(
    phase: str, *, provider_name: str | None = None, device_class: str | None = None
) -> Iterator[None]:
    """Translate native ``WinMLError``s from a phase into actionable ClickExceptions.

    Errors we raise ourselves (``ClickException``) pass through untouched, and any
    non-WinML exception (no ``hresult``) is re-raised as-is so genuine bugs stay
    visible.
    """
    try:
        yield
    except click.ClickException:
        raise
    except Exception as exc:
        hr = _winml_hresult(exc)
        if hr is None:
            raise
        raise _winml_click_error(
            exc, hr, phase, provider_name=provider_name, device_class=device_class
        ) from exc


def _resolve_mlir_target(
    runtime: Any,
    ep_device: WinMLEPDevice,
) -> _ResolvedRuntimeTarget:
    """Create a Runtime hardware target from the device carried by an EP pair."""
    device = ep_device.device
    device_class = device.device_type.lower()
    logger.info(
        "winml-runtime MLIR target resolved to %s / %s",
        device_class,
        device.hardware_name,
    )
    if device_class == "cpu":
        with _translate_native_errors("build", device_class=device_class):
            target = runtime.create_cpu_target()
        return _ResolvedRuntimeTarget(target, device_class)

    adapter_luid = device.adapter_luid
    adapter = None
    adapter_error = None
    if adapter_luid is not None:
        try:
            adapter = _DXCoreAdapter.from_luid(adapter_luid)
        except click.ClickException as exc:
            adapter_error = exc
    if adapter is None:
        raise click.ClickException(
            "Cannot resolve the selected device LUID. Run 'winml sys' and "
            "select another device with --device-luid <LUID>."
        ) from adapter_error
    try:
        with _translate_native_errors("build", device_class=device_class):
            target = runtime.create_target_from_adapter(adapter.pointer)
        return _ResolvedRuntimeTarget(target, device_class, adapter=adapter)
    except Exception:
        adapter.close()
        raise


def _resolve_onnx_target(
    runtime: Any,
    wr: Any,
    device: str,
    ep: str | None,
    ep_source: str | None,
    ep_device: WinMLEPDevice | None = None,
) -> _ResolvedRuntimeTarget:
    """Create an ORT-compatible Runtime target for ONNX input."""
    if ep_device is None:
        provider_name, device_class = resolve_provider_kind(device, ep, ep_source)
    else:
        provider_name = ep_device.device.ep_name
        device_class = ep_device.device.device_type.lower()
    logger.info("winml-runtime target resolved to %s / %s", device_class, provider_name)
    adapter: _DXCoreAdapter | None = None
    with _translate_native_errors(
        "build",
        provider_name=provider_name,
        device_class=device_class,
    ):
        if provider_name == "CPUExecutionProvider" and device_class == "cpu":
            target = runtime.create_cpu_target()
        else:
            kind = getattr(wr.ExecutionTargetKind, kind_name_for_device(device_class))
            if ep_device is None or device_class == "cpu":
                target = runtime.create_ort_execution_target(provider_name, kind)
            else:
                adapter_luid = ep_device.device.adapter_luid
                if adapter_luid is None:
                    raise click.ClickException(
                        f"The selected {device_class.upper()} device "
                        f"{ep_device.device.hardware_name!r} does not expose "
                        "adapter LUID metadata."
                    )
                adapter = _DXCoreAdapter.from_luid(adapter_luid)
                try:
                    hardware_target = runtime.create_target_from_adapter(adapter.pointer)
                    target = runtime.create_ort_execution_target(
                        provider_name,
                        kind,
                        hardware_target,
                    )
                except Exception:
                    adapter.close()
                    raise
    return _ResolvedRuntimeTarget(target, device_class, provider_name, adapter)


def _bind_inputs(
    runtime: Any,
    wr: Any,
    stage: Any,
    input_names: list[str],
    inputs: dict[str, Any],
    *,
    use_named_bindings: bool = True,
) -> Any:
    """Bind inputs, preferring name-based ORT bindings.

    Returns the :class:`OrtNamedBindings` handle when the stage is ORT-backed
    (so outputs can later be fetched by name), else ``None`` after binding by
    ordinal.
    """
    named = None
    if use_named_bindings:
        try:
            named = stage.ort_bindings()
        except wr.NotSupportedError:
            named = None

    if named is not None:
        for name in input_names:
            named.bind_input(name, runtime.tensor_from_numpy(inputs[name]))
        return named

    for index, name in enumerate(input_names):
        stage.bind_input(index, runtime.tensor_from_numpy(inputs[name]))
    return None


def _stage_diagnostics(wr: Any, stage: Any) -> tuple[str | None, bool]:
    """Return ``(selected_provider, is_pinned)`` when the stage exposes ORT diagnostics."""
    try:
        diag = stage.ort_diagnostics()
    except wr.NotSupportedError:
        return None, False
    try:
        return diag.selected_provider, diag.is_provider_pinned
    except Exception:  # diagnostics are advisory; never fail over them
        logger.debug("Failed to read ORT stage diagnostics", exc_info=True)
        return None, False


# =============================================================================
# Session
# =============================================================================
class WinMLRuntimeSession:
    """WinML Runtime pipeline session for a single prebuilt model artifact.

    Mirrors the small :class:`~winml.modelkit.session.session.WinMLSession`
    surface (``io_config`` / ``run`` / ``perf`` / ``device`` / ``ep_name``) but
    is backed by the ``windowsml.runtime`` pipeline API instead of ORT. Native
    resources (model, pipeline, stage) are built lazily on first use so
    constructing a session never imports the preview native library.
    """

    def __init__(
        self,
        model_path: str | Path,
        ep_device: WinMLEPDevice | None = None,
        *,
        device: str | None = "auto",
        ep: str | None = None,
        ep_source: str | None = None,
        provider_options: Mapping[str, str] | None = None,
        session_options: Callable[[], Any] | None = None,
        backend: RuntimeBackend,
    ) -> None:
        """Initialize a Runtime session.

        Args:
            model_path: Path to a prebuilt ONNX or MLIR model.
            device: Device shortcut (``cpu``/``gpu``/``npu``/``auto``).
            ep: Optional EP short or full name to pin (e.g. ``"qnn"``).
            ep_source: Optional EP source tag (from ``--ep name@source``).
            provider_options: Runtime EP options. Ignored for MLIR input and
                rejected for ONNX because the Runtime pipeline API does not
                expose a per-EP option hook.
            backend: Resolved backend used for model execution.
        """
        self._lock = threading.RLock()
        self._model_path = Path(model_path)
        self._is_mlir = self._model_path.suffix.lower() == ".mlir"
        self._backend = backend
        if self._backend == "cgc" and ep_device is None:
            raise ValueError("ep_device is required for CGC Runtime sessions.")
        self._ep_device = ep_device
        if self._backend == "cgc":
            assert ep_device is not None
            device = ep_device.device.device_type.lower()
            ep = None
            ep_source = None
            provider_options = None
        elif ep_device is not None:
            device = ep_device.device.device_type.lower()
            ep = ep_device.ep_short_name
            ep_source = ep_device.source_tag
        self._device_req = device or "auto"
        self._ep_req = ep
        self._ep_source = ep_source
        if session_options is not None:
            raise ValueError("session_options are not supported by the Windows ML Runtime backend.")
        if provider_options:
            raise click.ClickException(
                "--ep-options are not supported with --runtime winml-runtime because "
                "the Windows ML Runtime API cannot apply provider options."
            )

        # Native state, built lazily by _ensure_built().
        self._wr: Any = None
        self._runtime: Any = None
        self._adapter_handle: _DXCoreAdapter | None = None
        self._model: Any = None
        self._pipeline: Any = None
        self._stage: Any = None
        self._io_config: dict[str, Any] | None = None
        self._provider_name: str | None = None
        self._device_class: str | None = None
        self._selected_provider: str | None = None
        self._is_pinned: bool = False
        self._has_named_bindings = False
        self._compiled_artifacts: TemporaryDirectory[str] | None = None
        self._built = False

        # Perf tracking, enabled inside perf().
        self._perf_stats: PerfStats | None = None

    # -- lifecycle ----------------------------------------------------------
    def _ensure_built(self) -> None:
        """Import the runtime, resolve the target, load and build the pipeline.

        Idempotent; native errors are translated into ClickExceptions.
        """
        with self._lock:
            self._ensure_built_locked()

    def _resolve_target(self, runtime: Any, wr: Any) -> _ResolvedRuntimeTarget:
        if self._backend == "cgc":
            assert self._ep_device is not None
            return _resolve_mlir_target(runtime, self._ep_device)
        return _resolve_onnx_target(
            runtime,
            wr,
            self._device_req,
            self._ep_req,
            self._ep_source,
            self._ep_device,
        )

    def _load_onnx_on_ort(self, runtime: Any) -> tuple[Any, Any, bool]:
        """Load ONNX directly for execution by the ORT backend."""
        with _translate_native_errors("load"):
            model = runtime.load_model(str(self._model_path))
        return model, model.ort_schema(), True

    def _load_onnx_on_cgc(
        self,
        runtime: Any,
        resolved_target: _ResolvedRuntimeTarget,
    ) -> tuple[Any, Any, bool]:
        """Compile ONNX to CGIR and reload it for execution by the CGC backend."""
        with _translate_native_errors("load"):
            source_model = runtime.load_model(str(self._model_path))
        try:
            ort_schema = source_model.ort_schema()

            self._compiled_artifacts = TemporaryDirectory(prefix="winml-runtime-cgc-")
            artifact_path = Path(self._compiled_artifacts.name) / "model.mlir"
            with _translate_native_errors("build", device_class="gpu"):
                compiler = resolved_target.execution_target.model_compiler()
                try:
                    compiler.compile_to_file(source_model, str(artifact_path))
                finally:
                    compiler.close()

            with _translate_native_errors("load"):
                model = runtime.load_model(str(artifact_path))
            return model, ort_schema, False
        except Exception:
            source_model.close()
            raise

    def _load_mlir(self, runtime: Any) -> tuple[Any, None, bool]:
        """Load MLIR directly for execution by the CGC backend."""
        with _translate_native_errors("load"):
            return runtime.load_model(str(self._model_path)), None, False

    def _ensure_built_locked(self) -> None:
        if self._built:
            return

        wr = import_runtime()
        runtime = wr.Runtime()
        resolved_target = self._resolve_target(runtime, wr)
        try:
            if self._is_mlir:
                model, ort_schema, has_named_bindings = self._load_mlir(runtime)
            elif self._backend == "cgc":
                model, ort_schema, has_named_bindings = self._load_onnx_on_cgc(
                    runtime, resolved_target
                )
            else:
                model, ort_schema, has_named_bindings = self._load_onnx_on_ort(runtime)
            builder = runtime.create_pipeline_builder()
            with _translate_native_errors(
                "build",
                provider_name=resolved_target.provider_name,
                device_class=resolved_target.device_class,
            ):
                stage = builder.add_model_stage(model, resolved_target.execution_target)
                pipeline = builder.build()
                stage_schema = _stage_schema(wr, stage)
                io_config = synth_io_config(stage_schema, ort_schema)
                _apply_io_metadata(io_config, self._model_path)
            selected_provider, is_pinned = _stage_diagnostics(wr, stage)
        except Exception:
            if resolved_target.adapter is not None:
                resolved_target.adapter.close()
            raise

        self._wr = wr
        self._runtime = runtime
        self._adapter_handle = resolved_target.adapter
        self._model = model
        self._pipeline = pipeline
        self._stage = stage
        self._io_config = io_config
        self._provider_name = resolved_target.provider_name
        self._device_class = resolved_target.device_class
        self._selected_provider = selected_provider
        self._is_pinned = is_pinned
        self._has_named_bindings = has_named_bindings
        self._built = True

    # -- metadata -----------------------------------------------------------
    @property
    def io_config(self) -> dict:
        """I/O metadata derived from the Runtime model schema.

        Keys mirror :attr:`WinMLSession.io_config` (``input_names`` /
        ``input_shapes`` / ``input_types`` / ``output_names`` /
        ``output_shapes``), plus ``output_types`` for output materialization.
        """
        self._ensure_built()
        assert self._io_config is not None
        return self._io_config

    @property
    def running_model_path(self) -> Path:
        """Return the model artifact loaded by the Runtime."""
        return self._model_path

    @property
    def device(self) -> str:
        """Resolved Runtime target device class."""
        self._ensure_built()
        assert self._device_class is not None
        return self._device_class

    @property
    def ep_name(self) -> str | None:
        """Provider the stage resolved to, or ``None`` before the pipeline is built.

        Returns the diagnostics-reported selected provider when available,
        falling back to the requested provider name (non-ORT stages expose no
        diagnostics).
        """
        if not self._built:
            return None
        return self._selected_provider or self._provider_name

    @property
    def requested_provider(self) -> str | None:
        """The provider name resolved from ``--device``/``--ep`` (pre-diagnostics)."""
        self._ensure_built()
        return self._provider_name

    @property
    def is_pinned(self) -> bool:
        """Whether the stage pinned the requested provider (from ORT diagnostics)."""
        self._ensure_built()
        return self._is_pinned

    # -- inference ----------------------------------------------------------
    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Coerce each input to a contiguous NumPy array of the model's dtype.

        Accepts torch tensors or NumPy arrays and down/up-casts to the schema
        input dtype (mirrors ``WinMLSession._prepare_inputs``), so the same feed
        dict works against either backend.
        """
        import numpy as np

        assert self._io_config is not None
        names = self._io_config["input_names"]
        dtypes = self._io_config["input_types"]
        prepared: dict[str, np.ndarray] = {}
        for name, want in zip(names, dtypes, strict=True):
            if name not in inputs:
                raise ValueError(f"Missing input {name!r}; expected inputs {list(names)}")
            arr = _to_numpy(inputs[name])
            if arr.dtype != np.dtype(want):
                arr = arr.astype(want)
            prepared[name] = arr if arr.flags.c_contiguous else np.ascontiguousarray(arr)
        return prepared

    def _read_outputs(self, named: Any) -> dict[str, np.ndarray]:
        """Fetch every declared output as a caller-owned NumPy array.

        ORT-backed stages fetch by tensor name via the bound
        :class:`OrtNamedBindings`; non-ORT stages fetch by ordinal.
        """
        assert self._io_config is not None
        output_names = self._io_config["output_names"]
        outputs: dict[str, np.ndarray] = {}
        if named is not None:
            for name in output_names:
                outputs[name] = named.output(name).to_numpy()
        else:
            for index, name in enumerate(output_names):
                outputs[name] = self._stage.output(index).to_numpy()
        return outputs

    def run(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Run one inference and return ``{output_name: ndarray}``.

        Inputs are prepared (torch->numpy, dtype-cast) outside the timed region;
        binding, ``Pipeline.run`` and output read-back are timed when inside a
        :meth:`perf` window, keeping the measured cost comparable to the ORT
        backend's ``session.run`` (feed + infer + fetch).
        """
        with self._lock:
            return self._run_locked(inputs)

    def _run_locked(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        if not inputs:
            raise ValueError("inputs cannot be empty")
        self._ensure_built_locked()
        prepared = self._prepare_inputs(inputs)
        assert self._io_config is not None
        input_names = self._io_config["input_names"]
        output_count = len(self._io_config["output_names"])

        def _do() -> dict[str, np.ndarray]:
            named = _bind_inputs(
                self._runtime,
                self._wr,
                self._stage,
                input_names,
                prepared,
                use_named_bindings=self._has_named_bindings,
            )
            with _translate_native_errors("run"):
                # Older projections (including 2.7.9) materialize outputs automatically.
                # Newer projections require an explicit request on every execution.
                request_output = getattr(self._stage, "request_output", None)
                if callable(request_output):
                    for index in range(output_count):
                        request_output(index)
                self._pipeline.run()
            return self._read_outputs(named)

        if self._perf_stats is not None:
            return self._perf_stats.record(_do)
        return _do()

    def compile(self) -> None:
        """Build the Runtime pipeline if it has not already been built."""
        self._ensure_built()

    # -- perf ---------------------------------------------------------------
    @contextmanager
    def perf(self, warmup: int = 0, monitor: Any | None = None) -> Iterator[PerfContext]:
        """Scoped perf window; :meth:`run` calls inside accumulate timing.

        The Runtime pipeline exposes no EP monitor / op-tracing hooks, so a
        non-``None`` *monitor* is rejected with a clear error rather than
        silently ignored.
        """
        from .monitor.ep_monitor import NullEPMonitor
        from .session import PerfContext
        from .stats import PerfStats

        if monitor is not None:
            raise click.ClickException(
                "--monitor / op-tracing is not supported with --runtime winml-runtime; "
                "the Windows ML Runtime pipeline exposes no EP monitor hooks. Re-run "
                "without --monitor, or use --runtime winml-ort for op-level tracing."
            )
        if self._perf_stats is not None:
            raise RuntimeError(
                "WinMLRuntimeSession.perf() is already active. Nested perf windows "
                "are not supported."
            )

        self._ensure_built()
        stats = PerfStats(warmup=warmup)
        self._perf_stats = stats
        try:
            yield PerfContext(stats=stats, monitor=NullEPMonitor())
        finally:
            self._perf_stats = None

    @property
    def perf_stats(self) -> Any:
        """Active :class:`PerfStats` inside a :meth:`perf` window, else ``None``."""
        return self._perf_stats

    # -- teardown -----------------------------------------------------------
    def close(self) -> None:
        """Release native handles (best-effort; safe to call more than once)."""
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        for attr in ("_pipeline", "_stage", "_model", "_runtime", "_adapter_handle"):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            closer = getattr(obj, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # teardown is best-effort
                    logger.debug("Failed to close runtime %s", attr, exc_info=True)
            setattr(self, attr, None)
        if self._compiled_artifacts is not None:
            self._compiled_artifacts.cleanup()
            self._compiled_artifacts = None
        self._built = False

    def reset(self) -> None:
        """Release the pipeline so a later call can rebuild it."""
        with self._lock:
            self._close_locked()
            self._wr = None
            self._io_config = None
            self._provider_name = None
            self._device_class = None
            self._selected_provider = None
            self._is_pinned = False
            self._has_named_bindings = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Finalization must tolerate partial construction and interpreter shutdown.
            pass
