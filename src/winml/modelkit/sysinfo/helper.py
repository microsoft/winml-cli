# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import json
import subprocess
from typing import Any, TypeVar


_T = TypeVar("_T")


def _get_property_from_json(
    json_obj: dict,
    property_name: str,
    property_type: type[_T] = str,  # type: ignore[assignment]
    default: _T | None = None,
) -> _T:
    """Extract and validate a property from a JSON object."""
    if not isinstance(json_obj, dict):
        raise TypeError("json_obj must be a dictionary.")
    if property_name not in json_obj:
        if default is not None:
            return default
        raise ValueError(f"json_obj must contain a '{property_name}' key.")
    value = json_obj[property_name]
    if value is None and default is not None:
        return default
    if not isinstance(value, property_type):
        raise TypeError(f"{property_name} must be of type {property_type.__name__}.")
    return value


class CimInstance:
    """Represents a WMI CIM instance retrieved via PowerShell."""

    @staticmethod
    def get_by_class_name(class_name: str) -> list[CimInstance]:
        """Get all CIM instances of the specified class name."""
        results = CimInstance.get_many_by_class_name([class_name])
        return results.get(class_name, [])

    @staticmethod
    def get_many_by_class_name(
        class_names: list[str],
    ) -> dict[str, list[CimInstance]]:
        """Get CIM instances for multiple class names in a single PowerShell call.

        Returns:
            Mapping of class_name -> list of CimInstance.
        """
        if not class_names:
            return {}

        # Build a single PowerShell script that queries all classes and wraps
        # each result set with its class name for demuxing.
        parts: list[str] = ["[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "]
        parts.append("$result = @{}; ")
        parts.extend(
            f"$result['{name}'] = @(Get-CimInstance -ClassName {name} "
            f"-ErrorAction SilentlyContinue); "
            for name in class_names
        )
        parts.append("$result | ConvertTo-Json -Depth 99")

        try:
            output = subprocess.check_output(  # noqa: S603 - Input is trusted (class_names from code)
                [  # noqa: S607 - PowerShell path is standard on Windows
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "".join(parts),
                ],
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            return {name: [] for name in class_names}

        raw = output.decode("utf-8").strip()
        if not raw:
            return {name: [] for name in class_names}

        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {name: [] for name in class_names}

        result: dict[str, list[CimInstance]] = {}
        for name in class_names:
            items = parsed.get(name, [])
            if isinstance(items, dict):
                items = [items]
            if not isinstance(items, list):
                items = []
            result[name] = [CimInstance(obj) for obj in items if isinstance(obj, dict)]
        return result

    def __init__(self, json_obj: dict) -> None:
        """Initialize a CIM instance from a JSON object."""
        self._obj = json_obj

    def get_property(self, property_name: str, property_type: type[_T]) -> _T:
        """Get a property value from the CIM instance."""
        return _get_property_from_json(
            json_obj=self._obj, property_name=property_name, property_type=property_type
        )

    def try_get_property(self, property_name: str, property_type: type[_T], default: _T) -> _T:
        """Get a property value from the CIM instance with a default fallback."""
        return _get_property_from_json(
            json_obj=self._obj,
            property_name=property_name,
            property_type=property_type,
            default=default,
        )


class PnpDevice:
    """Represents a Plug and Play device retrieved via PowerShell."""

    @staticmethod
    def get_by_class_name(
        class_name: str,
        extra_property_keys: list[str] | None = None,
    ) -> list[PnpDevice]:
        """Get all PnP devices of the specified class name.

        Args:
            class_name: PnP device class name (e.g. "ComputeAccelerator").
            extra_property_keys: Optional list of DEVPKEY names to fetch via
                Get-PnpDeviceProperty.  When provided, a **single** batched
                PowerShell call retrieves only the requested keys for every
                device, instead of spawning one process per device for all
                properties.  Pass ``None`` (default) to skip extra properties
                entirely — callers that don't need them avoid the cost.
        """
        output = None
        try:
            output = subprocess.check_output(  # noqa: S603 - Input is trusted (class_name from code)
                [  # noqa: S607 - PowerShell path is standard on Windows
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                    + f"Get-PnpDevice -Class {class_name} | ConvertTo-Json -Depth 99",
                ],
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            return []

        json_array = json.loads(output.decode("utf-8"))
        if isinstance(json_array, dict):
            # Powershell will return a single object as a dict.
            json_array = [json_array]
        if not isinstance(json_array, list):
            raise TypeError(f"Expected a list from Get-PnpDevice, got {type(json_array)}")

        # Batch-fetch extra properties for all devices in one PowerShell call.
        all_extra: dict[str, dict[str, object]] = {}  # pnp_id -> {key: value}
        if extra_property_keys and json_array:
            pnp_ids = [_get_property_from_json(obj, "PNPDeviceID", str) for obj in json_array]
            all_extra = PnpDevice._batch_get_extra_properties(pnp_ids, extra_property_keys)

        return [
            PnpDevice(
                json_obj, all_extra.get(_get_property_from_json(json_obj, "PNPDeviceID", str), {})
            )
            for json_obj in json_array
        ]

    @staticmethod
    def _batch_get_extra_properties(
        pnp_ids: list[str],
        property_keys: list[str],
    ) -> dict[str, dict[str, object]]:
        """Fetch specific extra properties for multiple devices in one PowerShell call.

        Returns:
            Mapping of pnp_id -> {property_key: value}.
        """
        # Build a PowerShell script that queries all devices and returns structured JSON.
        # Using -KeyName filters to only the properties we need (much faster than all).
        keys_array = ", ".join(f"'{k}'" for k in property_keys)
        ids_array = ", ".join(f"'{pid}'" for pid in pnp_ids)
        ps_script = (
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            f"$ids = @({ids_array}); "
            f"$keys = @({keys_array}); "
            "$result = @(); "
            "foreach ($id in $ids) { "
            "  try { "
            "    $props = Get-PnpDeviceProperty -InstanceId $id -KeyName $keys "
            "      -ErrorAction SilentlyContinue; "
            "    foreach ($p in $props) { "
            "      $result += @{ InstanceId = $id; KeyName = $p.KeyName; Data = $p.Data } "
            "    } "
            "  } catch { } "
            "} "
            "$result | ConvertTo-Json -Depth 99"
        )

        try:
            output = subprocess.check_output(  # noqa: S603 - Input is trusted
                [  # noqa: S607
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    ps_script,
                ],
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            return {}

        raw = output.decode("utf-8").strip()
        if not raw:
            return {}

        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return {}

        result: dict[str, dict[str, object]] = {}
        for item in parsed:
            inst_id = _get_property_from_json(item, "InstanceId", str)
            key = _get_property_from_json(item, "KeyName", str)
            value = _get_property_from_json(item, "Data", object)
            result.setdefault(inst_id, {})[key] = value
        return result

    def __init__(
        self,
        json_obj: dict,
        extra_properties: dict[str, object] | None = None,
    ) -> None:
        """Initialize a PnP device from a JSON object.

        Args:
            json_obj: Device data from Get-PnpDevice.
            extra_properties: Pre-fetched extra properties (key -> value).
                When ``None``, no extra properties are available.
        """
        self._pnp_id = _get_property_from_json(json_obj, "PNPDeviceID", str)
        self._pnp_device_obj = json_obj
        self._extra_properties: dict[str, object] = extra_properties or {}

    def get_property(self, property_name: str, property_type: type[_T]) -> _T:
        """Get a property value from the PnP device."""
        return _get_property_from_json(self._pnp_device_obj, property_name, property_type)

    def try_get_property(self, property_name: str, property_type: type[_T], default: _T) -> _T:
        """Get a property value from the PnP device with a default fallback."""
        return _get_property_from_json(
            json_obj=self._pnp_device_obj,
            property_name=property_name,
            property_type=property_type,
            default=default,
        )

    def get_extra_property(self, property_name: str, property_type: type[_T]) -> _T:
        """Get an extra device property from Get-PnpDeviceProperty."""
        return _get_property_from_json(self._extra_properties, property_name, property_type)

    def try_get_extra_property(
        self, property_name: str, property_type: type[_T], default: _T
    ) -> _T:
        """Get an extra device property with a default fallback."""
        return _get_property_from_json(
            json_obj=self._extra_properties,
            property_name=property_name,
            property_type=property_type,
            default=default,
        )


class AppxPackage:
    """Represents an AppX package retrieved via PowerShell."""

    @staticmethod
    def get_by_hint(hint: str) -> list[AppxPackage]:
        """Get AppX packages matching the given hint string."""
        output = None
        try:
            output = subprocess.check_output(  # noqa: S603 - Input is trusted (hint from code)
                [  # noqa: S607 - PowerShell path is standard on Windows
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                    + f"Get-AppxPackage {hint} | ConvertTo-Json -Depth 99",
                ],
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            return []
        json_str = str(output, encoding="utf-8")
        if json_str.strip() == "":
            return []
        json_array = json.loads(json_str)
        if isinstance(json_array, dict):
            # Powershell will return a single object as a dict.
            json_array = [json_array]
        if not isinstance(json_array, list):
            raise TypeError(f"Expected a list from Get-AppxPackage, got {type(json_array)}")
        return [AppxPackage(json_obj) for json_obj in json_array]

    def __init__(self, json_obj: dict) -> None:
        """Initialize an AppX package from a JSON object."""
        self._obj = json_obj

    def get_property(self, property_name: str, property_type: type[_T]) -> _T:
        """Get a property value from the AppX package."""
        return _get_property_from_json(self._obj, property_name, property_type)

    def try_get_property(self, property_name: str, property_type: type[_T], default: _T) -> _T:
        """Get a property value from the AppX package with a default fallback."""
        return _get_property_from_json(
            json_obj=self._obj,
            property_name=property_name,
            property_type=property_type,
            default=default,
        )


def query_all_hardware(
    cim_class_names: list[str],
    pnp_class_name: str | None = None,
    pnp_extra_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Query CIM instances and PnP devices in a single PowerShell process.

    This avoids paying the PowerShell cold-start cost multiple times.

    Args:
        cim_class_names: WMI class names to query (e.g. Win32_Processor).
        pnp_class_name: Optional PnP device class (e.g. ComputeAccelerator).
        pnp_extra_keys: DEVPKEY names to fetch for each PnP device.

    Returns:
        ``{"cim": {class_name: [CimInstance, ...]},
           "pnp": [PnpDevice, ...]}``
    """
    # Build a single PowerShell script
    parts: list[str] = ["[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "]
    parts.append("$r = @{}; ")

    # CIM queries
    parts.extend(
        f"$r['{name}'] = @(Get-CimInstance -ClassName {name} -ErrorAction SilentlyContinue); "
        for name in cim_class_names
    )

    # PnP device + extra properties
    if pnp_class_name:
        parts.append(
            f"$pnp = @(Get-PnpDevice -Class {pnp_class_name} -ErrorAction SilentlyContinue); "
        )
        parts.append("$r['_pnp'] = $pnp; ")

        if pnp_extra_keys:
            keys_str = ", ".join(f"'{k}'" for k in pnp_extra_keys)
            parts.append(
                "$pnpProps = @(); "
                "foreach ($d in $pnp) { "
                "  try { "
                f"    $props = Get-PnpDeviceProperty -InstanceId $d.InstanceId "
                f"-KeyName {keys_str} -ErrorAction SilentlyContinue; "
                "    foreach ($p in $props) { "
                "      $pnpProps += @{ InstanceId = $d.InstanceId; "
                "KeyName = $p.KeyName; Data = $p.Data } "
                "    } "
                "  } catch { } "
                "} "
                "$r['_pnpProps'] = $pnpProps; "
            )

    parts.append("$r | ConvertTo-Json -Depth 99")

    try:
        output = subprocess.check_output(  # noqa: S603
            ["powershell", "-NoProfile", "-Command", "".join(parts)],  # noqa: S607
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return {"cim": {n: [] for n in cim_class_names}, "pnp": []}

    raw = output.decode("utf-8").strip()
    if not raw:
        return {"cim": {n: [] for n in cim_class_names}, "pnp": []}

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        return {"cim": {n: [] for n in cim_class_names}, "pnp": []}

    # Demux CIM results
    cim_result: dict[str, list[CimInstance]] = {}
    for name in cim_class_names:
        items = parsed.get(name, [])
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            items = []
        cim_result[name] = [CimInstance(obj) for obj in items if isinstance(obj, dict)]

    # Demux PnP results
    pnp_result: list[PnpDevice] = []
    if pnp_class_name:
        # Build extra-properties lookup
        extra_map: dict[str, dict[str, object]] = {}
        for item in parsed.get("_pnpProps", []) or []:
            if not isinstance(item, dict):
                continue
            inst_id = item.get("InstanceId", "")
            key = item.get("KeyName", "")
            if inst_id and key:
                extra_map.setdefault(inst_id, {})[key] = item.get("Data")

        pnp_raw = parsed.get("_pnp", [])
        if isinstance(pnp_raw, dict):
            pnp_raw = [pnp_raw]
        for obj in pnp_raw or []:
            if not isinstance(obj, dict):
                continue
            # Get-PnpDevice returns "InstanceId" (matches the PS property
            # name), while Win32_PnPEntity / CIM returns "PNPDeviceID".
            # Both contain the same device path string, so we can use
            # PNPDeviceID from the JSON to look up batched properties
            # keyed by InstanceId.
            pnp_id = obj.get("PNPDeviceID", "") or obj.get("InstanceId", "")
            pnp_result.append(PnpDevice(obj, extra_map.get(pnp_id, {})))

    return {"cim": cim_result, "pnp": pnp_result}
