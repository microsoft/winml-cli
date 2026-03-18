# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import re

from .hardware import CPU, GPU, NPU, RAM
from .software import OS, EpPackage, PipPackage, PythonRuntime


class WindowsAppRuntimeVersion:
    """Represents the Windows App Runtime version from pip packages."""

    _package_name_suffix = "-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap"

    def __init__(self, pip_packages: list[PipPackage]) -> None:
        """Initialize Windows App Runtime version from pip packages."""
        version = None
        for package in pip_packages:
            if package.name.endswith(self._package_name_suffix):
                version = package.version
                break
        if version is None:
            raise ValueError(
                f"Package ending with '{self._package_name_suffix}' not found in pip packages."
            )
        version = re.sub(r"^\d+!", "", version)
        version = re.sub(r"\.dev(\d+)", r"-experimental\1", version)
        # .dev0 are converted from -experimental packages instead of -experimental0
        if version.endswith("-experimental0"):
            version = version[:-1]
        self._version = version

    @property
    def version(self) -> str:
        """Windows App Runtime version."""
        return self._version


class SysInfo:
    """Comprehensive system information collector."""

    def __init__(self) -> None:
        """Initialize system information by collecting hardware and software data."""
        self._cpu_list = CPU.get_all()
        self._gpu_list = GPU.get_all()
        self._npu_list = NPU.get_all()
        self._ram_list = RAM.get_all()
        self._os = OS.get()
        self._python_runtime = PythonRuntime.get()
        self._pip_packages = PipPackage.get_all()
        self._ep_packages = EpPackage.get_all()
        self._windows_app_runtime_version = WindowsAppRuntimeVersion(self._pip_packages)

    @property
    def cpu_list(self) -> list[CPU]:
        """List of CPUs."""
        return self._cpu_list

    @property
    def gpu_list(self) -> list[GPU]:
        """List of GPUs."""
        return self._gpu_list

    @property
    def npu_list(self) -> list[NPU]:
        """List of NPUs."""
        return self._npu_list

    @property
    def ram_list(self) -> list[RAM]:
        """List of RAM modules."""
        return self._ram_list

    @property
    def os(self) -> OS:
        """Operating system information."""
        return self._os

    @property
    def python_runtime(self) -> PythonRuntime:
        """Python runtime information."""
        return self._python_runtime

    @property
    def pip_packages(self) -> list[PipPackage]:
        """List of installed pip packages."""
        return self._pip_packages

    @property
    def ep_packages(self) -> list[EpPackage]:
        """List of execution provider packages."""
        return self._ep_packages

    @property
    def windows_app_runtime_version(self) -> WindowsAppRuntimeVersion:
        """Windows App Runtime version."""
        return self._windows_app_runtime_version

    def to_dict(self) -> dict:
        """Convert all system information to a dictionary."""
        return {
            "cpuList": [cpu.to_dict() for cpu in self._cpu_list],
            "gpuList": [gpu.to_dict() for gpu in self._gpu_list],
            "npuList": [npu.to_dict() for npu in self._npu_list],
            "ramList": [ram.to_dict() for ram in self._ram_list],
            "os": self._os.to_dict(),
            "pythonRuntime": self._python_runtime.to_dict(),
            "pipPackages": [pkg.to_dict() for pkg in self._pip_packages],
            "epPackages": [pkg.to_dict() for pkg in self._ep_packages],
            "windowsAppRuntimeVersion": self._windows_app_runtime_version.version,
        }
