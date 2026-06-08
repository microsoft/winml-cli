# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLDevice - vendor-normalized adapter over ort.OrtEpDevice.

Single concrete class; per-EP metadata schemas handled by internal dispatch
tables keyed on self._ort.ep_name. OS-bound: cannot be hand-constructed;
instantiated only via wrap_ort_device(handle) after a successful ORT
registration produced the underlying OrtEpDevice. Tests may instantiate
directly when the OrtEpDevice handle is mocked.

See docs/design/session/4_winml_device.md for the full design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping

    import onnxruntime as ort


class WinMLDevice:
    """Vendor-normalized adapter over ort.OrtEpDevice."""

    def __init__(self, ort_device: ort.OrtEpDevice) -> None:
        self._ort = ort_device

    # ---- common properties (no per-EP dispatch needed) ------------------

    @property
    def ep_name(self) -> str:
        """Canonical EP name reported by ORT (e.g. ``"OpenVINOExecutionProvider"``)."""
        return self._ort.ep_name

    @property
    def device_type(self) -> str:
        """'NPU' | 'GPU' | 'CPU' - uppercased from device.type.name."""
        return self._ort.device.type.name.upper()

    @property
    def hardware_name(self) -> str:
        """Prefers ep_metadata['FULL_DEVICE_NAME']; falls back to device.metadata['Description']."""
        return (
            self._ort.ep_metadata.get("FULL_DEVICE_NAME")
            or self._ort.device.metadata.get("Description")
            or "<unknown>"
        )

    @property
    def vendor(self) -> str:
        """Hardware vendor string (e.g. ``"Intel"``) from the underlying OrtEpDevice."""
        return self._ort.device.vendor

    @property
    def ep_vendor(self) -> str:
        """EP vendor string (e.g. ``"Microsoft"``) from the underlying OrtEpDevice."""
        return self._ort.ep_vendor

    @property
    def library_path(self) -> str | None:
        """Plugin DLL path from ``ep_metadata['library_path']``, or ``None`` if unset."""
        return self._ort.ep_metadata.get("library_path") or None

    # ---- vendor-specific properties - internal dispatch on ep_name ------

    @property
    def memory_bytes(self) -> int | None:
        """Total device memory in bytes, or None when not applicable / unknown."""
        ep = self._ort.ep_name
        device_type = self.device_type
        # OpenVINO uses NPU_DEVICE_TOTAL_MEM_SIZE / GPU_DEVICE_TOTAL_MEM_SIZE
        if "OpenVINO" in ep:
            key = {
                "NPU": "NPU_DEVICE_TOTAL_MEM_SIZE",
                "GPU": "GPU_DEVICE_TOTAL_MEM_SIZE",
            }.get(device_type)
            if key:
                raw = self._ort.ep_metadata.get(key)
                if raw:
                    try:
                        return int(raw)
                    except ValueError:
                        return None
        # DML uses device.metadata['DxgiVideoMemory'] (e.g., '128 MB')
        if ep == "DmlExecutionProvider":
            raw = self._ort.device.metadata.get("DxgiVideoMemory", "")
            # Parse '<N> MB' / '<N> GB' / '<N> B' to bytes (cheap, best-effort)
            parts = raw.split()
            if len(parts) == 2:
                try:
                    n = int(parts[0])
                    unit = parts[1].upper()
                    multiplier = {
                        "B": 1,
                        "KB": 1024,
                        "MB": 1024**2,
                        "GB": 1024**3,
                    }.get(unit, 0)
                    if multiplier:
                        return n * multiplier
                except ValueError:
                    pass
        return None

    @property
    def architecture(self) -> str | None:
        """Short architecture string, or None."""
        ep = self._ort.ep_name
        if "OpenVINO" in ep:
            raw = self._ort.ep_metadata.get("DEVICE_ARCHITECTURE")
            if not raw:
                return None
            # 'GPU: vendor=0x8086 arch=v20.4.4' -> 'v20.4.4'; 'intel64' passes through
            if "arch=" in raw:
                return raw.split("arch=", 1)[1].strip()
            return raw
        return None

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Normalized capability flags. Empty tuple when unknown."""
        ep = self._ort.ep_name
        if "OpenVINO" in ep:
            raw = self._ort.ep_metadata.get("OPTIMIZATION_CAPABILITIES", "")
            tokens = raw.split()
            rewrites = {
                "GPU_HW_MATMUL": "MatMul",
                "GPU_USM_MEMORY": "USM",
                "EXPORT_IMPORT": "",
            }
            return tuple(rewrites.get(t, t) for t in tokens if rewrites.get(t, t))
        return ()

    @property
    def driver_version(self) -> str | None:
        """NPU driver version string, or ``None`` when unknown / not applicable."""
        ep = self._ort.ep_name
        if "OpenVINO" in ep and self.device_type == "NPU":
            return self._ort.ep_metadata.get("NPU_DRIVER_VERSION")
        return None

    @property
    def compiler_version(self) -> str | None:
        """NPU compiler version string, or ``None`` when unknown / not applicable."""
        ep = self._ort.ep_name
        if "OpenVINO" in ep and self.device_type == "NPU":
            return self._ort.ep_metadata.get("NPU_COMPILER_VERSION")
        return None

    # ---- introspection + display ----------------------------------------

    def available_metadata(self) -> Mapping[str, str]:
        """Raw ep_metadata mapping - for --verbose / debug dumps."""
        return dict(self._ort.ep_metadata)

    def facts(self) -> tuple[str, ...]:
        """Render-ready facts for one-line-per-device display.

        Joins memory / architecture / driver / compiler / capabilities into
        a tuple of strings ready for '  |  '.join(...).
        """
        out: list[str] = []
        if (m := self.memory_bytes) is not None:
            out.append(f"Memory: {_format_bytes(m)}")
        if (a := self.architecture) is not None:
            out.append(f"Architecture: {a}")
        if (d := self.driver_version) is not None:
            out.append(f"Driver: {d}")
        if (c := self.compiler_version) is not None:
            out.append(f"Compiler: {c}")
        if caps := self.capabilities:
            out.append(f"Capabilities: {', '.join(caps)}")
        return tuple(out)


def _format_bytes(n: int) -> str:
    """Format bytes as a human-readable string. Best-effort."""
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


def wrap_ort_device(d: ort.OrtEpDevice) -> WinMLDevice:
    """Factory - construct a WinMLDevice from an ort.OrtEpDevice handle."""
    return WinMLDevice(d)
