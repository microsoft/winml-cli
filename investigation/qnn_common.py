"""Shared helpers for the LoRA/QNN investigation.

Centralizes the WinML-based QNN EP registration and device selection so every
phase script uses the exact same path. This ORT build (1.24.5) ships QNN as a
WinML plugin EP, so we register via ``windowsml.EpCatalog`` +
``ort.register_execution_provider_library`` and select a device with
``SessionOptions.add_provider_for_devices`` — mirroring the repo's
``winml.add_ep_for_device`` ("NEVER modify") helper. The plan's
``providers=[("QNNExecutionProvider", {"backend_path": "QnnCpu.dll"})]`` path
does NOT apply here; see investigation/NOTES.md.
"""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort

QNN = "QNNExecutionProvider"

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models" / "lora_test"
INVEST = ROOT / "investigation"

_REGISTERED = False


def register_qnn() -> None:
    """Register the QNN WinML plugin EP with ORT (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    from windowsml import EpCatalog

    with EpCatalog() as catalog:
        for provider in catalog.find_all_providers():
            if provider.name != QNN:
                continue
            provider.ensure_ready()
            ort.register_execution_provider_library(provider.name, provider.library_path)
            break
        else:
            raise RuntimeError(f"{QNN} not found in WinML EpCatalog")
    _REGISTERED = True


def qnn_devices() -> dict[ort.OrtHardwareDeviceType, ort.OrtEpDevice]:
    """Return QNN EpDevices keyed by hardware device type (NPU/GPU/CPU)."""
    register_qnn()
    out: dict[ort.OrtHardwareDeviceType, ort.OrtEpDevice] = {}
    for d in ort.get_ep_devices():
        if d.ep_name == QNN:
            out.setdefault(d.device.type, d)
    return out


def make_qnn_session(
    model_path: str | Path,
    *,
    device_type: ort.OrtHardwareDeviceType | None = None,
    ep_options: dict | None = None,
    session_config: dict | None = None,
) -> ort.InferenceSession:
    """Build an InferenceSession on the QNN EP for the given hardware device.

    device_type defaults to NPU (the device class this investigation targets).
    session_config entries (e.g. ``ep.context_enable``) are applied to the
    SessionOptions before the provider is added.
    """
    register_qnn()
    if device_type is None:
        device_type = ort.OrtHardwareDeviceType.NPU
    devices = qnn_devices()
    if device_type not in devices:
        raise RuntimeError(
            f"No QNN EpDevice of type {device_type}; available: {list(devices)}"
        )
    so = ort.SessionOptions()
    for k, v in (session_config or {}).items():
        so.add_session_config_entry(k, str(v))
    so.add_provider_for_devices([devices[device_type]], ep_options or {})
    return ort.InferenceSession(str(model_path), so)


def cpu_session(model_path: str | Path) -> ort.InferenceSession:
    """Plain CPU EP reference session."""
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
