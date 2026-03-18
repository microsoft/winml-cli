import hashlib
import importlib.metadata
import platform
import re
from enum import Enum
from pathlib import Path

from .helper import AppxPackage, CimInstance


class OS:
    """Represents operating system information from Windows WMI."""

    @staticmethod
    def get() -> "OS":
        """Get the current operating system information."""
        cim_instances = CimInstance.get_by_class_name("Win32_OperatingSystem")
        if not cim_instances:
            raise RuntimeError("No Win32_OperatingSystem instance found.")
        return OS(cim_instances[0])

    def __init__(self, cim_instance: CimInstance) -> None:
        """Initialize an OS instance from a CIM instance."""
        self._caption = cim_instance.try_get_property("Caption", str, "")
        self._version = cim_instance.try_get_property("Version", str, "")
        self._architecture = cim_instance.try_get_property("OSArchitecture", str, "")
        self._sku = cim_instance.try_get_property("OperatingSystemSKU", int, 0)
        self._build_number = cim_instance.try_get_property("BuildNumber", str, "0")

    @property
    def caption(self) -> str:
        """OS caption."""
        return self._caption

    @property
    def version(self) -> str:
        """OS version."""
        return self._version

    @property
    def architecture(self) -> str:
        """OS architecture."""
        return self._architecture

    @property
    def sku(self) -> int:
        """OS SKU (Stock Keeping Unit)."""
        return self._sku

    @property
    def build_number(self) -> str:
        """OS build number."""
        return self._build_number

    def is_windows_11(self) -> bool:
        """Check if the OS is Windows 11 based on build number.
        
        Windows 11 has build number >= 22000.
        This is more reliable than checking Caption, which may report
        Windows 10 for compatibility reasons.
        """
        try:
            return int(self._build_number) >= 22000
        except (ValueError, TypeError):
            return False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "caption": self._caption,
            "version": self._version,
            "architecture": self._architecture,
            "sku": self._sku,
            "buildNumber": self._build_number,
            "isWindows11": self.is_windows_11(),
        }


class PythonRuntime:
    """Represents Python runtime information."""

    @staticmethod
    def get() -> "PythonRuntime":
        """Get the current Python runtime information."""
        return PythonRuntime()

    def __init__(self) -> None:
        """Initialize Python runtime information."""
        self._version = platform.python_version()
        self._implementation = platform.python_implementation()
        self._architecture = platform.machine()
        self._compiler = platform.python_compiler()
        build_info = platform.python_build()
        self._build_number = build_info[1] if len(build_info) > 1 else "Unknown"

    @property
    def version(self) -> str:
        """Python version."""
        return self._version

    @property
    def implementation(self) -> str:
        """Python implementation (e.g., CPython, PyPy)."""
        return self._implementation

    @property
    def architecture(self) -> str:
        """Python architecture."""
        return self._architecture

    @property
    def compiler(self) -> str:
        """Python compiler."""
        return self._compiler

    @property
    def build_number(self) -> str:
        """Python build number."""
        return self._build_number

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "version": self._version,
            "implementation": self._implementation,
            "architecture": self._architecture,
            "compiler": self._compiler,
            "buildNumber": self._build_number,
        }


class PipPackage:
    """Represents a Python package installed via pip."""

    @staticmethod
    def get_all() -> list["PipPackage"]:
        """Get all installed pip packages."""
        return [
            PipPackage(dist.metadata["Name"], dist.version)
            for dist in importlib.metadata.distributions()
        ]

    def __init__(self, name: str, version: str) -> None:
        """Initialize a pip package with name and version."""
        self._name = name
        self._version = version

    @property
    def name(self) -> str:
        """Package name."""
        return self._name

    @property
    def version(self) -> str:
        """Package version."""
        return self._version

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {"name": self._name, "version": self._version}


def calculate_folder_hash(folder_path: str) -> str | None:
    """Calculate a hash of all files in a folder."""
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return None

    hash_obj = hashlib.blake2b()
    file_list = sorted(folder.rglob("*"))

    for file_path in file_list:
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(folder)
        normalized_path = relative_path.as_posix()

        path_bytes = normalized_path.encode("utf-8")
        hash_obj.update(len(path_bytes).to_bytes(4, "little"))
        hash_obj.update(path_bytes)

        try:
            file_content = file_path.read_bytes()
            hash_obj.update(len(file_content).to_bytes(8, "little"))
            hash_obj.update(file_content)
        except (PermissionError, OSError):
            # Skip files that cannot be read due to permissions or other OS errors
            pass

    return hash_obj.hexdigest()


class EpPackage:
    """Represents an Execution Provider (EP) package."""

    class SignatureKind(Enum):
        """Package signature types."""

        NONE = 0
        Developer = 1  # pylint: disable=invalid-name
        Enterprise = 2  # pylint: disable=invalid-name
        Store = 3  # pylint: disable=invalid-name
        System = 4  # pylint: disable=invalid-name

    @staticmethod
    def get_all() -> list["EpPackage"]:
        """Get all installed WinML Execution Provider packages."""
        packages = []
        for package in AppxPackage.get_by_hint("*WinML*EP*"):
            name = package.get_property("Name", str)
            if re.match(r"^Microsoft\.WindowsMLRuntime\.\d+\.\d+$", name):
                # skip the deprecated WinML runtime package
                continue
            packages.append(EpPackage(package))
        return packages

    def __init__(self, appx_package: AppxPackage) -> None:
        """Initialize an EP package from an AppX package."""
        self._name = appx_package.get_property("PackageFullName", str)
        self._version = appx_package.get_property("Version", str)
        self._publisher = appx_package.get_property("Publisher", str)
        self._architecture = appx_package.get_property("Architecture", int)
        signature_kind = appx_package.get_property("SignatureKind", int)
        self._signature_kind = EpPackage.SignatureKind(signature_kind)
        self._install_location = appx_package.get_property("InstallLocation", str)
        self._status = appx_package.get_property("Status", int)
        try:
            self._ep_folder_hash = calculate_folder_hash(self._install_location) or ""
        except Exception:  # pylint: disable=broad-exception-caught
            self._ep_folder_hash = ""

    @property
    def name(self) -> str:
        """Package full name."""
        return self._name

    @property
    def version(self) -> str:
        """Package version."""
        return self._version

    @property
    def publisher(self) -> str:
        """Package publisher."""
        return self._publisher

    @property
    def architecture(self) -> int:
        """Package architecture."""
        return self._architecture

    @property
    def signature_kind(self) -> SignatureKind:
        """Package signature kind."""
        return self._signature_kind

    @property
    def install_location(self) -> str:
        """Package install location."""
        return self._install_location

    @property
    def status(self) -> int:
        """Package status."""
        return self._status

    @property
    def ep_hash(self) -> str:
        """Execution provider folder hash."""
        return self._ep_folder_hash

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self._name,
            "version": self._version,
            "publisher": self._publisher,
            "architecture": self._architecture,
            "signatureKind": self._signature_kind.name,
            "installLocation": self._install_location,
            "epHash": self._ep_folder_hash,
            "status": self._status,
        }
