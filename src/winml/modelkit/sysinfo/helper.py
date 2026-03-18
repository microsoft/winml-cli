import json
import subprocess
from typing import TypeVar


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
    def get_by_class_name(class_name: str) -> list["CimInstance"]:
        """Get all CIM instances of the specified class name."""
        output = None
        try:
            output = subprocess.check_output(  # noqa: S603 - Input is trusted (class_name from code)
                [  # noqa: S607 - PowerShell path is standard on Windows
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                    + f"Get-CimInstance -ClassName {class_name} | "
                    + "ConvertTo-Json -Depth 99",
                ]
            )
        except subprocess.CalledProcessError:
            # This will throw if no matching device is found
            return []
        json_array = json.loads(output.decode("utf-8"))
        if isinstance(json_array, dict):
            # Powershell will return a single object as a dict.
            json_array = [json_array]
        if not isinstance(json_array, list):
            raise TypeError(f"Expected a list from Get-CimInstance, got {type(json_array)}")
        return [CimInstance(json_obj) for json_obj in json_array]

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
    def get_by_class_name(class_name: str) -> list["PnpDevice"]:
        """Get all PnP devices of the specified class name."""
        output = None
        try:
            output = subprocess.check_output(  # noqa: S603 - Input is trusted (class_name from code)
                [  # noqa: S607 - PowerShell path is standard on Windows
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                    + f"Get-PnpDevice -Class {class_name} | ConvertTo-Json -Depth 99",
                ]
            )
        except subprocess.CalledProcessError:
            return []

        json_array = json.loads(output.decode("utf-8"))
        if isinstance(json_array, dict):
            # Powershell will return a single object as a dict.
            json_array = [json_array]
        if not isinstance(json_array, list):
            raise TypeError(f"Expected a list from Get-PnpDevice, got {type(json_array)}")
        return [PnpDevice(json_obj) for json_obj in json_array]

    def __init__(self, json_obj: dict) -> None:
        """Initialize a PnP device from a JSON object."""
        self._pnp_id = _get_property_from_json(json_obj, "PNPDeviceID", str)
        self._pnp_device_obj = json_obj
        output = b"[]"
        try:
            output = subprocess.check_output(  # noqa: S603 - Input is trusted (pnp_id from WMI)
                [  # noqa: S607 - PowerShell path is standard on Windows
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                    + f"Get-PnpDeviceProperty -InstanceId '{self._pnp_id}' | "
                    + "ConvertTo-Json -Depth 99",
                ]
            )
        except subprocess.CalledProcessError:
            # This may happen if the device has no extra properties
            pass

        property_list = json.loads(output.decode("utf-8"))
        if not isinstance(property_list, list):
            raise TypeError(
                f"Expected a list from Get-PnpDeviceProperty, got {type(property_list)}"
            )
        self._extra_properties: dict[str, object] = {}
        for prop in property_list:
            key = _get_property_from_json(prop, "KeyName", str)
            value = _get_property_from_json(prop, "Data", object)
            self._extra_properties[key] = value

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
    def get_by_hint(hint: str) -> list["AppxPackage"]:
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
                ]
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
