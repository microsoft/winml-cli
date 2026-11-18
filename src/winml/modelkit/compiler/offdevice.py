# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Discover and use NPU compiler payloads without matching target hardware."""

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from traceback import clear_frames
from typing import TYPE_CHECKING
from uuid import uuid4

import onnxruntime as ort
from onnx import helper

from ..ep_path import _parse_winmlcli_ep_path, discover_all_eps
from ..onnx import get_external_data_files, is_compiled_onnx, load_onnx
from ..session.ep_registry import _entry_source_tag, _suppress_dll_load_dialogs
from ..utils.constants import normalize_ep_name


if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..ep_path import EPEntry


_OPENVINO_EP = "OpenVINOExecutionProvider"
_QNN_EP = "QNNExecutionProvider"
_VITISAI_EP = "VitisAIExecutionProvider"


def _compilation_entries(
    ep_name: str | None, source: str | None, *, explicit_ep: bool = True
) -> list[EPEntry]:
    provider = normalize_ep_name(ep_name or _OPENVINO_EP)
    if provider not in {_OPENVINO_EP, _QNN_EP, _VITISAI_EP}:
        raise ValueError("Off-device mode currently supports only OpenVINO, QNN and VitisAI NPU.")
    overrides = _parse_winmlcli_ep_path(provider if explicit_ep else None)
    return [
        entry
        for entry in discover_all_eps(extra_sources=overrides)
        if entry.ep_name == provider and (source is None or _entry_source_tag(entry) == source)
    ]


def list_compilation_eps(
    ep_name: str | None = None, *, source: str | None = None
) -> list[dict[str, object]]:
    """List implemented compilation EPs and payloads without loading their DLLs."""
    rows = []
    for provider in [ep_name] if ep_name is not None else [_OPENVINO_EP, _QNN_EP, _VITISAI_EP]:
        entries = _compilation_entries(provider, source, explicit_ep=ep_name is not None)
        canonical = normalize_ep_name(provider)
        row = {
            "provider_name": canonical,
            "device_type": "npu",
            "payload_available": bool(entries),
            "payloads": [
                {
                    "library_path": str(entry.dll_path.resolve()),
                    "source": _entry_source_tag(entry),
                    "version": entry.version,
                }
                for entry in entries
            ],
        }
        if canonical == _QNN_EP:
            row.update(
                compilation_device_type="cpu",
                target_parameter="soc_model",
                targets_enumerable=False,
            )
        elif canonical == _VITISAI_EP:
            row.update(
                compilation_device_type="virtual_npu",
                target_parameter="target",
                targets_enumerable=False,
            )
        rows.append(row)
    return rows


@contextmanager
def _registered_compilation_devices(
    ep_name: str, source: str | None
) -> Iterator[tuple[EPEntry, list[ort.OrtEpDevice]]]:
    """Keep the selected payload and its compilation host handles alive for the caller."""
    provider = normalize_ep_name(ep_name)
    is_qnn = provider == _QNN_EP
    is_vitisai = provider == _VITISAI_EP
    entries = _compilation_entries(ep_name, source)
    if not entries:
        raise ValueError(
            f"No {provider.removesuffix('ExecutionProvider')} compiler payload was found "
            "for the selected source. "
            "Install a compatible payload or set WINMLCLI_EP_PATH."
        )
    entry = entries[0]
    library_path = entry.dll_path.resolve()
    # Canonical names keep provider identity stable in compiled contexts.
    registration_name = (
        provider if is_qnn or is_vitisai else f"{_OPENVINO_EP}_{uuid4().hex}.virtual"
    )
    registrations = ExitStack()
    devices: list[ort.OrtEpDevice] = []
    try:
        with _suppress_dll_load_dialogs():
            if is_vitisai:
                # ORT scopes allow_virtual_devices to active ".virtual" registrations.
                bootstrap = f"{provider}_{uuid4().hex}.virtual"
                ort.register_execution_provider_library(bootstrap, str(library_path))
                registrations.callback(ort.unregister_execution_provider_library, bootstrap)
            ort.register_execution_provider_library(registration_name, str(library_path))
            registrations.callback(ort.unregister_execution_provider_library, registration_name)
        devices = [
            device
            for device in ort.get_ep_devices()
            if device.ep_name == provider
            and Path(device.ep_metadata.get("library_path", "")).resolve() == library_path
            and (
                (provider == _QNN_EP and device.device.type == ort.OrtHardwareDeviceType.CPU)
                or (
                    provider in {_OPENVINO_EP, _VITISAI_EP}
                    and device.device.type == ort.OrtHardwareDeviceType.NPU
                    and device.device.metadata.get("is_virtual") == "1"
                )
            )
        ]
        if not devices:
            raise ValueError(
                f"{provider} payload {library_path} exposed no "
                f"{'CPU compilation hosts' if is_qnn else 'virtual NPU targets'}. "
                "No physical-device fallback is permitted."
            )
        if provider == _OPENVINO_EP and any(
            not device.ep_metadata.get("npu_platform") for device in devices
        ):
            raise ValueError("An OpenVINO virtual NPU target is missing npu_platform.")
        devices.sort(
            key=lambda device: (
                device.ep_metadata.get("npu_platform", ""),
                device.device.device_id,
            )
        )
        yield entry, devices
    finally:
        devices.clear()
        registrations.close()


def list_compilation_targets(ep_name: str, *, source: str | None = None) -> list[dict[str, object]]:
    """Snapshot virtual NPU targets from the highest-priority selected payload."""
    if normalize_ep_name(ep_name) == _QNN_EP:
        raise ValueError(
            "QNN does not enumerate SoC targets. Use --target with the positive numeric "
            "soc_model ID documented by QAIRT for your target device."
        )
    if normalize_ep_name(ep_name) == _VITISAI_EP:
        raise ValueError(
            "VitisAI does not advertise NPU architecture targets. Use --target X2 "
            "to select the supported compiler backend, not a destination SoC."
        )
    with _registered_compilation_devices(ep_name, source) as (entry, devices):
        targets: list[dict[str, object]] = []
        seen: set[str] = set()
        for device in devices:
            target: dict[str, object] = {
                "provider_name": device.ep_name,
                "library_path": str(entry.dll_path.resolve()),
                "source": _entry_source_tag(entry),
                "runtime_version": ort.__version__,
                "npu_platform": device.ep_metadata["npu_platform"],
                "vendor_id": device.device.vendor_id,
                "device_id": device.device.device_id,
                "metadata": dict(device.ep_metadata),
                "provider_options": dict(device.ep_options),
                "hardware_metadata": dict(device.device.metadata),
            }
            identity = json.dumps(target, sort_keys=True)
            if identity not in seen:
                seen.add(identity)
                targets.append(target)
        return targets


def _validate_compilation_output(model_path: Path, ep_name: str = _OPENVINO_EP) -> None:
    """Require a self-contained selected-provider EPContext without running inference."""
    if get_external_data_files(model_path):
        raise ValueError("Off-device output contains unsupported external initializer files.")
    model = load_onnx(model_path, load_weights=False, validate=False)
    contexts = [
        node
        for node in model.graph.node
        if node.op_type == "EPContext" and node.domain == "com.microsoft"
    ]
    if not contexts:
        raise ValueError("Off-device compilation produced no EPContext.")
    main_contexts = 0
    for node in contexts:
        attributes = {
            attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute
        }
        if (
            attributes.get("source") != ep_name.encode("utf-8")
            or attributes.get("embed_mode", 1) != 1
        ):
            raise ValueError(f"Off-device output must contain embedded {ep_name} EPContext nodes.")
        main_context = attributes.get("main_context", 1)
        cache = attributes.get("ep_cache_context", b"")
        if main_context not in (0, 1) or not isinstance(cache, bytes):
            raise ValueError("Off-device output contains malformed EPContext attributes.")
        if main_context == 1:
            if not cache:
                raise ValueError("Off-device output contains an empty main EPContext.")
            main_contexts += 1
    if not main_contexts:
        raise ValueError("Off-device output contains no main EPContext.")


def compile_off_device(
    model_path: str | Path,
    output_path: str | Path,
    target: str,
    *,
    ep_name: str = _OPENVINO_EP,
    source: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Compile one embedded NPU model on Windows without matching target hardware."""
    ep_name = normalize_ep_name(ep_name)
    is_qnn = ep_name == _QNN_EP
    if is_qnn and (not target.isascii() or not target.isdecimal() or int(target) <= 0):
        raise ValueError("QNN --target must be a positive numeric soc_model ID from QAIRT.")
    if ep_name == _VITISAI_EP and target != "X2":
        raise ValueError(
            "VitisAI off-device mode requires --target X2 (compiler backend, not an NPU SoC)."
        )
    model_path = Path(model_path).resolve()
    output_path = Path(output_path).absolute()
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    input_files = [
        model_path,
        *(model_path.parent / name for name in get_external_data_files(model_path)),
    ]
    if any(
        output_path.resolve() == input_file.resolve()
        or (output_path.exists() and input_file.exists() and output_path.samefile(input_file))
        for input_file in input_files
    ):
        raise ValueError(
            "The compiled output must not overwrite the input model or its data files."
        )
    if output_path.is_dir():
        raise IsADirectoryError(f"Expected an output file, not a directory: {output_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    if not target:
        raise ValueError("An explicit NPU compilation target is required.")
    if is_compiled_onnx(model_path):
        raise ValueError("The input model already contains EPContext nodes.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_path.stem}.compile-", dir=output_path.parent) as work:
        compiled_path = Path(work) / output_path.name
        with _registered_compilation_devices(ep_name, source) as (entry, devices):
            if ep_name in {_QNN_EP, _VITISAI_EP}:
                matches = list(devices)
            elif ep_name == _OPENVINO_EP:
                matches = [
                    device for device in devices if device.ep_metadata["npu_platform"] == target
                ]
            else:
                raise ValueError(f"Unsupported off-device provider: {ep_name}")
            if len(matches) != 1:
                if is_qnn:
                    raise ValueError(
                        f"QNN offline compilation requires exactly one CPU host "
                        f"(found {len(matches)})."
                    )
                if ep_name == _VITISAI_EP:
                    raise ValueError(
                        f"VitisAI offline compilation requires exactly one virtual NPU "
                        f"(found {len(matches)})."
                    )
                available = ", ".join(sorted({d.ep_metadata["npu_platform"] for d in devices}))
                raise ValueError(
                    f"Target {target!r} must match exactly one virtual NPU device "
                    f"(found {len(matches)}). Available platforms: {available}."
                )
            options = None
            compiler = None
            try:
                provider_options = {}
                if is_qnn:
                    backend_path = entry.dll_path.resolve().with_name("QnnHtp.dll")
                    if not backend_path.is_file():
                        raise FileNotFoundError(
                            f"QNN offline compilation requires the HTP backend: {backend_path}"
                        )
                    provider_options = {"backend_path": str(backend_path), "soc_model": target}
                elif ep_name == _VITISAI_EP:
                    provider_options = {
                        "target": target,
                        "enable_cache_file_io_in_mem": "1",
                    }
                options = ort.SessionOptions()
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                options.add_provider_for_devices(matches, provider_options)
                compiler = ort.ModelCompiler(
                    options,
                    str(model_path),
                    embed_compiled_data_into_model=True,
                    flags=(
                        ort.OrtCompileApiFlags.ERROR_IF_NO_NODES_COMPILED
                        | ort.OrtCompileApiFlags.ERROR_IF_OUTPUT_FILE_EXISTS
                    ),
                )
                compiler.compile_to_file(str(compiled_path))
            except BaseException as error:
                # ORT's Python wrapper retains the compiler in traceback frames.
                clear_frames(error.__traceback__)
                raise
            finally:
                compiler = None
                options = None
                matches.clear()

        _validate_compilation_output(compiled_path, ep_name)
        if overwrite:
            compiled_path.replace(output_path)
        else:
            # Windows rename fails atomically if the destination already exists.
            compiled_path.rename(output_path)
    return output_path
