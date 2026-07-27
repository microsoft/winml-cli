# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""System information command for WinML CLI.

Displays detailed information about the system environment, including:
- Python and OS information
- Core ML library versions (torch, transformers, onnx, etc.)
- Hardware capabilities (CPU, GPU, memory)
- Backend SDK availability (QNN, OpenVINO)
- Export readiness assessment
- Available devices and execution providers

Usage:
    winml sys
    winml sys --format json
    winml sys --format compact
    winml sys --verbose
    winml sys --list-device
    winml sys --list-ep
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.wintypes
import json
import logging
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, cast

import click
from rich.markup import escape

from ..ep_path import (
    DirectorySource,
    EPEntry,
    MSIXPackageSource,
    NuGetSource,
    PyPISource,
    WinMLCatalogSource,
)
from ..session import WinMLEPRegistry
from ..sysinfo import OS
from ..utils import cli as cli_utils
from ..utils.logging import configure_logging


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from rich.console import Console


logger = logging.getLogger(__name__)

# Rich is imported lazily — Console/Table/Panel only matter at render time,
# and pulling them at module scope adds ~50 ms to every `winml sys`
# invocation (including --format json/compact, which never render anything
# Rich-styled).
# Public module-level Console handle. Tests replace this via monkeypatch to
# capture output. Prefer :func:`_get_console` at call sites so a monkeypatched
# instance is picked up.
console: Console | None = None


def _get_console() -> Console:
    """Return a process-level Console, importing rich on first use."""
    global console
    if console is None:
        from rich.console import Console as _RichConsole

        console = _RichConsole()
    return console


# `--list-ep` indent levels (single source of truth — avoids the literal
# space-count copies that were drifting across helpers).
#   L1 EP header        →  2 sp  e.g. "  QNNExecutionProvider ..."
#   L2 entry row        →  4 sp  e.g. "    [primary] PyPI ..."
#   L3 meta (Path/...)  → 14 sp  aligned under the L2 source-kind column
#                         (`_INDENT_L2 + len("[status] ") + width 9 + sep`)
#   L4 device facts     → 16 sp  one indent past the L3 device line
_INDENT_L2 = "    "
_INDENT_L3 = " " * 14
_INDENT_L4 = " " * 16


def _get_python_info() -> dict[str, Any]:
    """Gather Python environment information."""
    return {
        "version": platform.python_version(),
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
    }


# Intentionally limited to the host architectures Windows 11 ships on.
# 32-bit ARM (ARMNT 0xC4, ARM 0x1C4) and IA64 are not Windows 11 host
# targets, so IsWow64Process2 will not report them in practice; an
# unmapped value falls through to None and logs at debug level.
_IMAGE_FILE_MACHINE_TO_NAME = {
    0x8664: "AMD64",
    0xAA64: "ARM64",
    0x14C: "x86",
}


if sys.platform == "win32":
    try:
        _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _IS_WOW64_PROCESS_2 = _KERNEL32.IsWow64Process2
        _IS_WOW64_PROCESS_2.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_ushort),
            ctypes.POINTER(ctypes.c_ushort),
        ]
        _IS_WOW64_PROCESS_2.restype = ctypes.wintypes.BOOL
    except (OSError, AttributeError):
        _KERNEL32 = None  # type: ignore[assignment]
        _IS_WOW64_PROCESS_2 = None  # type: ignore[assignment]
else:
    _KERNEL32 = None  # type: ignore[assignment]
    _IS_WOW64_PROCESS_2 = None  # type: ignore[assignment]


def _query_native_machine_via_win32() -> int | None:
    """Call IsWow64Process2 and return the native IMAGE_FILE_MACHINE_* code.

    Returns None on non-Windows or when the API call fails. Logs the
    GetLastError code at debug level on failure so a regression in
    IsWow64Process2 wiring is not silently swallowed.
    """
    if sys.platform != "win32" or _IS_WOW64_PROCESS_2 is None or _KERNEL32 is None:
        return None

    proc = ctypes.c_ushort(0)
    native = ctypes.c_ushort(0)
    if not _IS_WOW64_PROCESS_2(
        _KERNEL32.GetCurrentProcess(), ctypes.byref(proc), ctypes.byref(native)
    ):
        logger.debug("IsWow64Process2 failed: GetLastError=%d", ctypes.get_last_error())
        return None
    return native.value


def _get_windows_native_machine() -> str | None:
    """Return the host architecture name, or None when unavailable.

    platform.machine() returns the *process* arch (PROCESSOR_ARCHITECTURE),
    so an x64 Python running under ARM64 emulation reports "AMD64". This
    consults IsWow64Process2 for the real host machine type, which is what
    the user expects to see in `winml sys`. PROCESSOR_ARCHITEW6432 is
    unreliable on ARM64 (Prism emulation does not set it on Snapdragon X).
    """
    if sys.platform != "win32":
        return None
    raw = _query_native_machine_via_win32()
    if raw is None:
        return None
    name = _IMAGE_FILE_MACHINE_TO_NAME.get(raw)
    if name is None:
        logger.debug("IsWow64Process2 returned unmapped native machine: 0x%x", raw)
    return name


def _get_platform_info() -> dict[str, Any]:
    """Gather OS and platform information."""
    system = platform.system()
    release = platform.release()
    machine = platform.machine()

    # For Windows, use OS class for accurate Windows 11 detection
    # platform.release() may incorrectly report '10' on some Python versions
    if system == "Windows":
        try:
            os_info = OS.get()
            # Only override if it's actually Windows 11
            # Otherwise keep the original platform.release() value
            if os_info.is_windows_11():
                release = "11"
        except Exception:
            # Fallback to platform.release() if OS detection fails
            pass

        native_machine = _get_windows_native_machine()
        if native_machine:
            machine = native_machine

    return {
        "system": system,
        "release": release,
        "machine": machine,
        "processor": platform.processor() or "Unknown",
    }


def _get_library_versions() -> dict[str, str | None]:
    """Gather versions of key ML libraries."""
    libraries: dict[str, str | None] = {}

    # Core libraries
    lib_names = [
        "torch",
        "transformers",
        "onnx",
        "optimum",
        "numpy",
        "click",
        "rich",
    ]

    for lib in lib_names:
        try:
            libraries[lib] = version(lib)
        except PackageNotFoundError:  # noqa: PERF203
            libraries[lib] = None

    # onnxruntime has multiple distribution variants
    ort_variants = [
        "onnxruntime",
        "onnxruntime-windowsml",
        "onnxruntime-gpu",
        "onnxruntime-silicon",
    ]
    libraries["onnxruntime"] = None
    for variant in ort_variants:
        try:
            ver = version(variant)
            # Include variant suffix if not base onnxruntime
            if variant != "onnxruntime":
                suffix = variant.replace("onnxruntime-", "")
                libraries["onnxruntime"] = f"{ver} ({suffix})"
            else:
                libraries["onnxruntime"] = ver
            break
        except PackageNotFoundError:
            continue

    return libraries


def _get_torch_info(verbose: bool = False) -> dict[str, Any]:
    """Gather PyTorch-specific information.

    Reads the installed version via ``importlib.metadata`` so the default
    path does not ``import torch`` — importing torch costs ~1.5 s warm and
    used to dominate ``winml sys`` latency (issue #558). CUDA details
    require the torch module and are only gathered when ``verbose`` is set.
    """
    info: dict[str, Any] = {"available": False}

    try:
        info["version"] = version("torch")
        info["available"] = True
    except PackageNotFoundError:
        logger.debug("PyTorch not available")
        return info

    if not verbose:
        return info

    try:
        import torch
    except ImportError:
        return info

    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = str(torch.backends.cudnn.version())
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    return info


def _check_qnn_sdk() -> dict[str, Any]:
    """Check QNN SDK availability."""
    import os

    info: dict[str, Any] = {"installed": False}

    # Check common environment variables and paths
    qnn_sdk_root = os.environ.get("QNN_SDK_ROOT")
    qairt_sdk_root = os.environ.get("QAIRT_SDK_ROOT")

    if qnn_sdk_root:
        info["installed"] = True
        info["path"] = qnn_sdk_root
        info["source"] = "QNN_SDK_ROOT"
    elif qairt_sdk_root:
        info["installed"] = True
        info["path"] = qairt_sdk_root
        info["source"] = "QAIRT_SDK_ROOT"

    # TODO: Check for qnn-convert executable
    # TODO: Parse version from SDK

    return info


def _check_openvino() -> dict[str, Any]:
    """Check OpenVINO availability."""
    info: dict[str, Any] = {"installed": False}

    try:
        import openvino

        info["installed"] = True
        info["version"] = openvino.__version__
    except ImportError:
        logger.debug("OpenVINO not available")

    return info


def _gather_system_info(verbose: bool = False) -> dict[str, Any]:
    """Gather all system information.

    Args:
        verbose: Include additional diagnostic information

    Returns:
        Dictionary containing all system information
    """
    info = {
        "python": _get_python_info(),
        "platform": _get_platform_info(),
        "libraries": _get_library_versions(),
        "torch": _get_torch_info(verbose=verbose),
        "backends": {
            "qnn": _check_qnn_sdk(),
            "openvino": _check_openvino(),
        },
    }

    # Assess export readiness
    libs = info["libraries"]
    info["export_readiness"] = {
        "onnx_export": all(
            [
                libs.get("torch"),
                libs.get("onnx"),
                libs.get("transformers"),
            ]
        ),
        "qnn_ready": info["backends"]["qnn"]["installed"],
        "openvino_ready": info["backends"]["openvino"]["installed"],
    }

    return info


def _output_text(info: dict[str, Any], verbose: bool = False) -> None:
    """Output system info in human-readable text format."""
    from rich.panel import Panel
    from rich.table import Table

    console = _get_console()

    # Title
    console.print(
        Panel.fit(
            "[bold]WinML CLI System Information[/bold]",
            border_style="blue",
        )
    )

    # Python & Platform
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Python Version", info["python"]["version"])
    table.add_row("Python Executable", info["python"]["executable"])
    table.add_row("OS", f"{info['platform']['system']} {info['platform']['release']}")
    table.add_row("Machine", info["platform"]["machine"])

    console.print("\n[bold blue]Environment[/bold blue]")
    console.print(table)

    # Libraries
    lib_table = Table(show_header=True, box=None, padding=(0, 2))
    lib_table.add_column("Library", style="bold")
    lib_table.add_column("Version")
    lib_table.add_column("Status")

    for lib, lib_version in info["libraries"].items():
        status = "[green]OK[/green]" if lib_version else "[red]Not installed[/red]"
        lib_table.add_row(lib, lib_version or "-", status)

    console.print("\n[bold blue]ML Libraries[/bold blue]")
    console.print(lib_table)

    # PyTorch details (CUDA fields only populated with --verbose)
    torch_info = info["torch"]
    if torch_info["available"] and "cuda_available" in torch_info:
        console.print("\n[bold blue]PyTorch Details[/bold blue]")
        torch_table = Table(show_header=False, box=None, padding=(0, 2))
        torch_table.add_column("Key", style="bold")
        torch_table.add_column("Value")

        torch_table.add_row("CUDA Available", str(torch_info["cuda_available"]))
        if torch_info["cuda_available"]:
            torch_table.add_row("CUDA Version", torch_info.get("cuda_version", "N/A"))
            torch_table.add_row("GPU Count", str(torch_info.get("gpu_count", 0)))
            for i, gpu in enumerate(torch_info.get("gpu_devices", [])):
                torch_table.add_row(f"GPU {i}", gpu)

        console.print(torch_table)

    # Backend SDKs and Export Readiness are diagnostic info — only render
    # under --verbose so the default `winml sys` stays focused on Python,
    # libraries, devices, and EPs.
    if verbose:
        backends = info["backends"]
        qnn = backends["qnn"]
        ov = backends["openvino"]

        console.print("\n[bold blue]Backend SDKs[/bold blue]")
        backend_table = Table(show_header=False, box=None, padding=(0, 2))
        backend_table.add_column("Backend", style="bold")
        backend_table.add_column("Status")
        backend_table.add_column("Details")

        if qnn["installed"]:
            backend_table.add_row("QNN SDK", "[green]Installed[/green]", qnn.get("path", "-")[:50])
        else:
            backend_table.add_row("QNN SDK", "[yellow]Not found[/yellow]", "-")
        if ov["installed"]:
            backend_table.add_row("OpenVINO", "[green]Installed[/green]", ov.get("version", "-"))
        else:
            backend_table.add_row("OpenVINO", "[yellow]Not found[/yellow]", "-")

        console.print(backend_table)

        readiness = info["export_readiness"]
        console.print("\n[bold blue]Export Readiness[/bold blue]")
        ready_table = Table(show_header=False, box=None, padding=(0, 2))
        ready_table.add_column("Capability", style="bold")
        ready_table.add_column("Status")

        onnx_ready = "[green]Ready[/green]" if readiness["onnx_export"] else "[red]Not ready[/red]"
        ready_table.add_row("ONNX Export", onnx_ready)
        qnn_status = (
            "[green]Ready[/green]" if readiness["qnn_ready"] else "[yellow]SDK required[/yellow]"
        )
        ready_table.add_row("QNN Compilation", qnn_status)
        ov_status = (
            "[green]Ready[/green]"
            if readiness["openvino_ready"]
            else "[yellow]Not installed[/yellow]"
        )
        ready_table.add_row("OpenVINO Conversion", ov_status)

        console.print(ready_table)


def _output_json(info: dict[str, Any]) -> None:
    """Output system info as JSON."""
    click.echo(json.dumps(info, indent=2))


def _output_compact(info: dict[str, Any]) -> None:
    """Output system info in compact format."""
    py = info["python"]
    plat = info["platform"]
    libs = info["libraries"]
    torch_info = info["torch"]
    readiness = info["export_readiness"]

    lines = [
        f"Python: {py['version']} ({plat['system']})",
        f"torch: {libs.get('torch', 'N/A')} | "
        f"transformers: {libs.get('transformers', 'N/A')} | "
        f"onnx: {libs.get('onnx', 'N/A')}",
    ]

    if torch_info["available"] and torch_info.get("cuda_available"):
        lines.append(
            f"CUDA: {torch_info.get('cuda_version', 'N/A')} | "
            f"GPU: {torch_info.get('gpu_count', 0)} device(s)"
        )

    qnn = info["backends"]["qnn"]
    ov = info["backends"]["openvino"]
    sdk_parts = []
    if qnn["installed"]:
        sdk_parts.append("QNN: OK")
    if ov["installed"]:
        sdk_parts.append(f"OpenVINO: {ov.get('version', 'OK')}")
    if sdk_parts:
        lines.append(" | ".join(sdk_parts))

    onnx_status = "OK" if readiness["onnx_export"] else "FAIL"
    lines.append(f"Export Ready: ONNX {onnx_status}")

    for line in lines:
        click.echo(line)


# --- Device listing ---


def _gather_device_info(
    ep_info: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Gather available device information in priority order.

    Args:
        ep_info: Optional output of :func:`_gather_ep_info`. When passed,
            each device is enriched with per-source ``device_facts``
            (Architecture + Driver) that survived the subprocess boundary
            through :meth:`WinMLEP.to_dict`. Without this the enrichment
            is a no-op because filesystem-backed EPs are registered only
            in isolated child processes, so the parent's
            ``WinMLEPRegistry._registered`` never holds their live
            ``WinMLDevice`` handles.

    Each ``XPU.get_all()`` call spawns a PowerShell subprocess; the slowest
    of the three (Win32_Processor WMI query, ~1.3 s warm) sets the floor.
    We run them in parallel so the wall time is max() instead of sum()
    (issue #558).

    Returns:
        List of device dicts with type, priority, and details.
    """
    from ..sysinfo import CPU, GPU, NPU

    # NPU > GPU > CPU priority order.
    hw_queries: list[tuple[str, type[NPU] | type[GPU] | type[CPU]]] = [
        ("NPU", NPU),
        ("GPU", GPU),
        ("CPU", CPU),
    ]

    with ThreadPoolExecutor(max_workers=len(hw_queries)) as pool:
        futures = [(label, pool.submit(cls.get_all)) for label, cls in hw_queries]
        # Sequence (not list) because list is invariant in its element type:
        # fut.result() at runtime is list[CPU] | list[GPU] | list[NPU], none
        # of which are list[Any]. Sequence is covariant, so this accepts
        # all three. The cast at .result() is needed because pool.submit
        # collapses the union-typed `cls.get_all` callable to Future[object].
        ordered_results: list[tuple[str, Sequence[Any] | Exception]] = []
        for label, fut in futures:
            try:
                ordered_results.append((label, cast("Sequence[Any]", fut.result())))
            except Exception as e:  # noqa: PERF203 - per-future error capture
                ordered_results.append((label, e))

    result: list[dict[str, Any]] = []
    priority = 1
    for device_label, items in ordered_results:
        if isinstance(items, Exception):
            logger.warning("Failed to get %s details: %s", device_label, items)
            # CPU always exists, NPU/GPU may not — only surface CPU errors.
            if device_label == "CPU":
                result.append(
                    {
                        "priority": priority,
                        "type": device_label,
                        "name": "(detection error)",
                        "details": {"error": str(items)},
                    }
                )
                priority += 1
            continue

        for item in items:
            entry: dict[str, Any] = {
                "priority": priority,
                "type": device_label,
                "name": item.name,
                "details": {},
            }
            if device_label in ("NPU", "GPU"):
                entry["details"] = {
                    "driver": item.driver_version,
                    "manufacturer": item.manufacturer,
                }
            elif device_label == "CPU":
                entry["details"] = {
                    "cores": item.core_count,
                    "threads": item.thread_count,
                    "architecture": item.architecture.name,
                }
            result.append(entry)
            priority += 1

    # Enrich each device entry with device_facts (Architecture + Driver)
    # captured from any registered EP whose .devices saw this hardware.
    # First successful match per (device_type, hardware_name) pair wins;
    # later EPs binding the same device can't override those facts because
    # they're device-intrinsic per docs/design/session/4_winml_device.md
    # §4.1. ``setdefault`` ensures sysinfo-provided values aren't
    # clobbered when both sources have the same key (e.g. ``driver``).
    #
    # Data source: ``ep_info`` (already gathered by :func:`_gather_ep_info`).
    # Reading it here — instead of ``WinMLEPRegistry._registered`` — is
    # what makes this work now that filesystem-backed EPs live only in
    # isolated child processes; ``WinMLEP.to_dict`` serializes
    # ``device_facts`` into the returned dict so the fact survives the
    # subprocess boundary.
    if ep_info:
        try:
            for entry in result:
                match_type = entry["type"]
                sysinfo_name = entry["name"]
                matched_dev = _find_matching_device(
                    ep_info,
                    match_type,
                    sysinfo_name,
                )
                if matched_dev is None:
                    continue
                # ``device_facts`` values are ``"Label: Value"`` strings;
                # split on the first colon so we can merge by lowercased
                # key alongside sysinfo's details.
                for fact in matched_dev.get("device_facts") or []:
                    label, _, value = fact.partition(": ")
                    if label and value:
                        entry["details"].setdefault(label.lower(), value)
        except Exception as e:
            logger.warning("device_facts enrichment failed: %s", e)

    return result


def _find_matching_device(
    ep_info: dict[str, dict[str, Any]],
    match_type: str,
    sysinfo_name: str,
) -> dict[str, Any] | None:
    """Return the first device dict in ``ep_info`` matching type + fuzzy name.

    Fuzzy relation: substring-in-either-direction covers both bias cases
    (OpenVINO appends "(iGPU)" to FULL_DEVICE_NAME that sysinfo's WMI
    query doesn't include; sysinfo may report a wordier form ORT trims).
    """
    for record in ep_info.values():
        for source_desc in record.get("entries", ()):
            for dev in source_desc.get("devices") or ():
                if dev.get("device_type") != match_type:
                    continue
                hw = dev.get("hardware_name", "") or ""
                if hw == sysinfo_name or sysinfo_name in hw or hw in sysinfo_name:
                    return cast("dict[str, Any]", dev)
    return None


def _output_device_text(devices: list[dict[str, Any]]) -> None:
    """Display device list in rich text format.

    Per ``docs/design/session/4_winml_device.md`` §4.1, the *Available
    Devices* section renders device-intrinsic facts (Architecture +
    Driver). When :func:`_gather_device_info` enriched the entry from a
    registered ``WinMLDevice.device_facts()`` call the values land in the
    same ``details`` dict alongside sysinfo's keys (``driver``, ``manufacturer``,
    ``cores``, ``threads``, ``architecture``), so we read them
    uniformly here.
    """
    console = _get_console()
    console.print("\n[bold blue]Available Devices (priority order)[/bold blue]")
    for dev in devices:
        name = escape(dev["name"])
        console.print(
            f"  [bold]#{dev['priority']}[/bold]  [cyan]{dev['type']:5s}[/cyan] {name}"
        )
        details = dev.get("details", {})
        if "error" in details:
            console.print(f"             [red]Error: {escape(details['error'])}[/red]")
        elif dev["type"] in ("NPU", "GPU"):
            parts = [
                f"Driver: {details.get('driver', 'N/A')}",
                f"Manufacturer: {details.get('manufacturer', 'N/A')}",
            ]
            if arch := details.get("architecture"):
                parts.append(f"Architecture: {arch}")
            console.print(f"             {' | '.join(parts)}")
        elif dev["type"] == "CPU":
            console.print(
                f"             Cores: {details.get('cores', 'N/A')} | "
                f"Threads: {details.get('threads', 'N/A')} | "
                f"Architecture: {details.get('architecture', 'N/A')}"
            )


# --- EP listing ---


def _describe_source(entry: EPEntry) -> dict[str, Any]:
    """Build a JSON-friendly per-source descriptor for ``--list-ep``.

    Reads ``entry.version`` as the single source-of-truth for version
    metadata — each EPSource subclass populates it at ``.resolve()`` time
    from its own canonical source (importlib.metadata for PyPI, cache
    subdir name for NuGet, ``Package.Id.Version`` for MSIX). Subclass
    dispatch here only adds source-kind-specific identifying fields
    (distribution, family prefix, catalog name, etc.) — no version
    recovery.

    The canonical short ``source_tag`` (``"pypi"``, ``"bundled"`` …) is
    derived by :func:`session.ep_registry._entry_source_tag` so adding a
    new ``EPSource`` subclass means updating the tag table in exactly
    one place.
    """
    from ..session.ep_registry import _entry_source_tag

    source = entry.source
    desc: dict[str, Any] = {
        "source_kind": type(source).__name__,
        "source_tag": _entry_source_tag(entry),
    }
    if entry.version is not None:
        desc["version"] = entry.version
    if isinstance(source, PyPISource):
        desc["distribution"] = source.distribution
    elif isinstance(source, MSIXPackageSource):
        desc["family_name_prefix"] = source.family_name_prefix
    elif isinstance(source, NuGetSource):
        desc["nuget_id"] = source.distribution
    elif isinstance(source, WinMLCatalogSource):
        desc["catalog_name"] = source.catalog_name
    elif isinstance(source, DirectorySource):
        desc["root"] = str(source.root)
        if source.env_var:
            desc["env_var"] = source.env_var
    return desc


@contextlib.contextmanager
def isolated_ep_register(
    ep_name: str,
    dll_path: Path,
    *,
    timeout: float = 30.0,
) -> Iterator[dict[str, Any]]:
    """Register ``dll_path`` in a fresh subprocess; yield the ``to_dict()``.

    Windows' loaded-modules table is process-wide and base-name keyed, so
    registering multiple installs of the same EP in one process leaks the
    first-loaded ``plugin_impl.dll``'s metadata into every later call. A
    fresh subprocess per call keeps the child's table empty. Raises
    :class:`WinMLEPRegistrationFailed` on any failure so callers handle
    both isolated and in-process paths identically.
    """
    import inspect
    import subprocess
    import textwrap

    from ..session import WinMLEPRegistrationFailed

    def _worker() -> None:
        """Runs in the subprocess. Source shipped via ``inspect.getsource``."""
        import json as _json
        import sys as _sys
        from pathlib import Path as _Path

        from winml.modelkit.session import (
            DirectorySource,
            EPEntry,
            WinMLEPRegistry,
        )

        ep, dll_str = _sys.argv[1], _sys.argv[2]
        dll = _Path(dll_str)
        source = DirectorySource(
            root=dll.parent,
            dll_patterns={ep: dll.name},
        )
        entry = EPEntry(ep_name=ep, dll_path=dll, source=source)
        winml_ep = WinMLEPRegistry.instance().register_ep(entry)
        _sys.stdout.write(_json.dumps(winml_ep.to_dict()))

    # Ship the nested function's source verbatim to a fresh Python via
    # ``-c``. ``dedent`` strips the indent introduced by the def being
    # inside another function; appending ``_worker()`` invokes it in the
    # child's top-level namespace.
    worker_script = textwrap.dedent(inspect.getsource(_worker)) + "\n_worker()\n"

    try:
        # S603: subprocess invocation is controlled — sys.executable is the
        # current interpreter, worker_script is inspected from a local function
        # in this module, and ep_name/dll_path come from the discovered
        # registry (validated earlier). No untrusted shell interpolation.
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", worker_script, ep_name, str(dll_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        # ``subprocess.run(timeout=…)`` drains stdout/stderr into the
        # ``TimeoutExpired`` before killing the child, so ``exc.stderr``
        # carries whatever ORT / the plugin's C++ init printed in the
        # up-to-``timeout`` seconds before the hang was declared.
        # Preserve the tail so a real driver-hang investigation isn't
        # left with just ``"timed out after 30s"`` and nothing else.
        stderr_tail = (
            exc.stderr.strip()[-500:]
            if isinstance(exc.stderr, str)
            else (exc.stderr or b"").decode(errors="replace").strip()[-500:]
        )
        raise WinMLEPRegistrationFailed(
            f"isolated register of {dll_path} timed out after {timeout}s"
            + (f"; stderr={stderr_tail!r}" if stderr_tail else ""),
            dll_path=dll_path,
        ) from exc

    if proc.returncode != 0:
        stderr_tail = proc.stderr.strip()
        # The child's real error is its last non-empty stderr line (the
        # exception message). Pass it as ``raw_error`` so the ``[failed]``
        # row shows a clean reason instead of the wrapper prefix or a
        # mid-traceback fragment left by the ``[-500:]`` slice.
        last_line = next(
            (ln for ln in reversed(stderr_tail.splitlines()) if ln.strip()),
            "",
        )
        raise WinMLEPRegistrationFailed(
            f"isolated register of {dll_path} exited {proc.returncode}: "
            f"{stderr_tail[-500:]}",
            dll_path=dll_path,
            raw_error=last_line,
        )
    try:
        ep_dict = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise WinMLEPRegistrationFailed(
            f"isolated register of {dll_path} produced invalid JSON: {exc}; "
            f"stdout tail={proc.stdout[-200:]!r}",
            dll_path=dll_path,
        ) from exc
    yield ep_dict


def _gather_ep_info() -> dict[str, dict[str, Any]]:
    """Gather comprehensive EP inventory across every source.

    Walks :meth:`WinMLEPRegistry.all_discovered` (PyPI / NuGet / WinMLCatalog /
    DirectorySource / live MSIX enumeration via :func:`list_msix_eps` in
    the default source list) plus any ``WINMLCLI_EP_PATH`` env-var
    override. Each discovered :class:`EPEntry` is fed through
    :meth:`WinMLEPRegistry.register_ep` (Path B — broad enumeration); the
    DLL is loaded so we can surface per-device facts from the resulting
    :class:`WinMLEP`. Registration failures (broken DLL, missing
    runtime, etc.) and L2-check failures (sysinfo/WMI unavailable) are
    both captured as ``status="failed"`` rows carrying the exception.

    Built-in EPs (CPU, Azure, DML) flow through the same main loop —
    they're synthesized into :attr:`WinMLEPRegistry._discovered` at init
    via :class:`BuiltinSource`. This command does not import onnxruntime
    directly. Compatibility is queried straight from :data:`EP_CATALOG`.

    Status derivation follows ``2_coreloop.md`` §7.1.1 in strict
    precedence: L1 ``"failed"`` (register_ep raised) > L2
    ``"incompatible"`` (vendor rule mismatched) > §7.1.2 precedence
    (first compatible row ``"primary"``, later ``"shadowed"``). The
    discovery-time ``EPEntry.status`` field is intentionally ignored —
    it reflects precedence-position in the source list, not whether the
    DLL actually loaded.

    Returns:
        Dict ``ep_name -> {entries: [...]}``. The actual device set lives
        per-entry under ``entries[].devices[]`` and reflects what ORT
        really exposed for THIS source, not a static catalog claim.
    """
    from ..ep_path import EP_CATALOG
    from ..session import WinMLEPRegistrationFailed

    # Failed rows carry ``compatible=None`` because the L2 check was
    # never evaluated; ``False`` would misclassify as L2-incompatible.
    # The build loop short-circuits on ``err is not None`` first.
    registry = WinMLEPRegistry.instance()
    all_entries = list(registry.all_discovered())
    fs_entries = [e for e in all_entries if not e.is_built_in()]

    def _process(
        entry: EPEntry,
    ) -> tuple[EPEntry, dict[str, Any] | None, WinMLEPRegistrationFailed | None, bool | None]:
        try:
            if entry.is_built_in():
                winml_ep = registry.register_ep(entry)
                compatible = EP_CATALOG.is_compatible(entry.ep_name)
                return (entry, winml_ep.to_dict(), None, compatible)
            with isolated_ep_register(entry.ep_name, entry.dll_path) as ep_dict:
                compatible = EP_CATALOG.is_compatible(entry.ep_name)
            return (entry, ep_dict, None, compatible)
        except WinMLEPRegistrationFailed as e:
            return (entry, {"plugin_version": e.fallback_version}, e, None)
        except Exception as e:
            logger.warning("Failed to process EP %s: %s", entry.ep_name, e)
            # Prefix the original type name so --format json's ``error``
            # field still surfaces the root cause instead of masking every
            # unexpected failure as ``WinMLEPRegistrationFailed: ...``.
            wrapped = WinMLEPRegistrationFailed(
                f"{type(e).__name__}: {e}",
                dll_path=entry.dll_path,
            )
            return (entry, {"plugin_version": wrapped.fallback_version}, wrapped, None)

    # Cap at 4 workers: each child is a cold Python + ORT init
    # (~200 MB RAM, ~2 s CPU); wider trades wall-clock for memory.
    if fs_entries:
        with ThreadPoolExecutor(max_workers=min(len(fs_entries), 4)) as pool:
            fs_iter = iter(list(pool.map(_process, fs_entries)))
    else:
        fs_iter = iter(())

    ep_records: dict[
        str,
        list[
            tuple[EPEntry, dict[str, Any] | None, WinMLEPRegistrationFailed | None, bool | None]
        ],
    ] = {}
    for entry in all_entries:
        row = next(fs_iter) if not entry.is_built_in() else _process(entry)
        ep_records.setdefault(entry.ep_name, []).append(row)

    # --- Tag DLL paths that came via WinMLCatalogSource (for the
    # "(catalog default)" annotation in the renderer). Built as a set
    # rather than checked inline with ``isinstance(entry.source, ...)``
    # because a non-Catalog row (e.g. a PyPI install) can resolve to
    # the same DLL the Catalog row points at, and we want every row
    # sharing that path to carry the tag — not just the Catalog one.
    catalog_default_paths: set[Path] = {
        entry.dll_path
        for rows in ep_records.values()
        for entry, _, _, _ in rows
        if isinstance(entry.source, WinMLCatalogSource)
    }

    # --- Build per-EP output dicts.
    record_by_ep: dict[str, dict[str, Any]] = {}
    for ep_name, rows in ep_records.items():
        entries_out: list[dict[str, Any]] = []
        # ``primary_seen`` is per-ep_name (matches §7.1.2 spec literally
        # — first successful, vendor-compatible row wins ``primary``).
        primary_seen = False
        for entry, ep_dict, err, compatible in rows:
            desc = _describe_source(entry)
            if err is not None:
                desc["status"] = "failed"
                # Full ORT message for --format json; compact code+reason
                # (already parsed by err.__init__) for the text render.
                desc["error"] = f"{type(err).__name__}: {err}"
                desc["error_code"] = err.code
                desc["error_reason"] = err.reason
            elif not compatible:
                desc["status"] = "incompatible"
            else:
                desc["status"] = "primary" if not primary_seen else "shadowed"
                primary_seen = True

            # Built-ins carry Path() sentinel; None suppresses the Path: row.
            desc["dll_path"] = None if entry.is_built_in() else str(entry.dll_path)
            if entry.dll_path in catalog_default_paths:
                desc["is_catalog_default"] = True

            if ep_dict is not None and (pv := ep_dict.get("plugin_version")) is not None:
                desc["plugin_version"] = pv
            # Devices only for usable rows: an incompatible row registers
            # with a CPU fallback whose device line would mislead readers.
            if ep_dict is not None and compatible:
                desc["devices"] = ep_dict.get("devices") or []
            entries_out.append(desc)

        # EP-level "compatible" is derived at render time from
        # entry["status"] (any primary/shadowed row -> compatible).
        record_by_ep[ep_name] = {"entries": entries_out}

    return record_by_ep


_SOURCE_KIND_LABEL = {
    "PyPISource": "PyPI",
    "MSIXPackageSource": "MSIX",
    "NuGetSource": "NuGet",
    "WinMLCatalogSource": "Catalog",
    "DirectorySource": "FS",
    "BuiltinSource": "bundled",
}


def _format_devices_from_handles(devices: list[dict[str, Any]]) -> list[str]:
    """Render the per-source ``Devices:`` block per ``console_mockup.py``.

    Layout (matches ``docs/design/session/console_mockup.py::_render_facts_block``)::

        Devices:
          NPU:  Memory: 16.0 GB   |  Capabilities: FP16, INT8
          GPU:  Memory: 16.3 GB   |  Capabilities: FP32, FP16, INT8, BIN
          CPU:                        Capabilities: BF16, FP32, FP16, INT8, BIN

    Hardware names are intentionally NOT repeated here — they're already
    rendered once per physical device in the "Available Devices" section
    above. This keeps the EP inventory compact and focused on
    EP-mediated facts (``Memory`` / ``Capabilities``, per
    ``4_winml_device.md §4.1``).

    The ``vendor`` field is kept on the incoming dict for JSON consumers
    but is not surfaced in text — ORT reports it inconsistently
    ("Intel" vs "Intel Corporation" for the same ``vendor_id 0x8086``).
    """
    if not devices:
        return []
    lines: list[str] = [f"{_INDENT_L3}[dim]Devices:[/dim]"]
    for d in devices:
        dev_type = d.get("device_type", "?")
        facts = d.get("facts") or []
        body = escape("  |  ".join(facts)) if facts else "[dim](no metadata published)[/dim]"
        # Fixed 4-char abbrev column so "NPU:" / "GPU:" / "CPU:" align.
        type_label = f"[bold cyan]{dev_type:3s}[/bold cyan]:"
        lines.append(f"{_INDENT_L4}{type_label} {body}")
    return lines


def _output_ep_text(eps: dict[str, dict[str, Any]]) -> None:
    """Display the comprehensive EP inventory in rich text format."""
    console = _get_console()
    console.print("\n[bold blue]Available Execution Providers[/bold blue]")
    if not eps:
        console.print("  [yellow]No execution providers found.[/yellow]")
        return

    for ep_name, record in eps.items():
        # Rich treats square brackets as markup; escape the literal status
        # tags with backslashes so [primary] / [failed] etc. render as text.
        # See 2_coreloop.md §7.1.1 for the L1 (failed) vs L2 (incompatible)
        # split — the EP-level tag here means "no row would actually be
        # registered as primary/shadowed", which collapses L1-failed and
        # L2-incompatible into one header tag.
        any_usable = any(e["status"] in ("primary", "shadowed") for e in record["entries"])
        compat_tag = "" if any_usable else r"  [bold red]\[incompatible][/bold red]"
        console.print(f"  [bold]{ep_name}[/bold]{compat_tag}")

        for entry in record["entries"]:
            status = entry.get("status", "?")
            kind = entry.get("source_kind", "?")
            # Status colour mirrors §7.1.2:
            #   primary       — green  (this EP's precedence-winner)
            #   shadowed      — yellow (registered cleanly; not Scenario A's pick)
            #   failed        — red    (register_ep raised; carries error field)
            #   incompatible  — red    (vendor rule overrides a successful register)
            status_color = {
                "primary": "green",
                "shadowed": "yellow",
                "failed": "red",
                "incompatible": "red",
            }.get(status, "white")
            tag = f"[{status_color}]\\[{status}][/{status_color}]"

            extras: list[str] = []
            # ``entry["version"]`` is the single source of truth for any
            # version string (populated per-EPSource-subclass at
            # ``.resolve()`` time and copied verbatim by
            # :func:`_describe_source`); render "?" when absent.
            ver = entry.get("version") or "?"
            if "distribution" in entry:
                extras.append(f"{entry['distribution']} {ver}")
            if "nuget_id" in entry:
                extras.append(f"{entry['nuget_id']} {ver}")
            if "family_name_prefix" in entry:
                # Drop the trailing ``_<publisherId>`` (e.g. ``8wekyb3d8bbwe``)
                # for compact CLI display; show the prefix verbatim if no
                # underscore is present.
                family = entry["family_name_prefix"]
                head, sep, _publisher = family.rpartition("_")
                short_family = head if sep else family
                ver = entry.get("version") or "?"
                extras.append(f"{short_family} v{ver}")
                if entry.get("is_catalog_default"):
                    extras.append("[dim](catalog default)[/dim]")
            elif entry.get("is_catalog_default") and kind == "WinMLCatalogSource":
                extras.append("[dim](catalog default)[/dim]")
            if "root" in entry:
                extras.append(f"root={entry['root']}")

            short_kind = _SOURCE_KIND_LABEL.get(kind, kind)
            extras_str = "  ".join(extras) if extras else ""
            console.print(f"{_INDENT_L2}{tag} [bold]{short_kind:9}[/bold] {extras_str}")
            # Runtime plugin version from ORT's ``ep_metadata['version']`` —
            # rendered on its own ``Version:`` row (semantically distinct
            # from the source-package version inside ``extras_str``).
            if plugin_ver := entry.get("plugin_version"):
                console.print(f"{_INDENT_L3}[dim]Version:[/dim] {plugin_ver}")
            if entry.get("dll_path"):
                console.print(f"{_INDENT_L3}[dim]Path:[/dim]    {entry['dll_path']}")
            if entry.get("error"):
                # Prefer the compact ``error_reason`` on the human render;
                # the raw ``error`` text is still emitted through
                # ``--format json`` for callers that want the ORT payload.
                short_err = entry.get("error_reason") or entry["error"]
                console.print(f"{_INDENT_L3}[red]Error:[/red] {escape(short_err)}")
            for line in _format_devices_from_handles(entry.get("devices") or []):
                console.print(line)


def _gather(
    *,
    system: bool = False,
    devices: bool = False,
    eps: bool = False,
    verbose: bool = False,
    tolerant: bool = False,
) -> dict[str, Any]:
    """Build a sysinfo dict containing the requested sections.

    EPs run before devices so :func:`_gather_ep_info` populates the
    per-source device inventory first — :func:`_gather_device_info`
    then reads ``device_facts`` (Architecture + Driver) out of the
    already-serialized inventory to enrich each top-level device row.
    Reading from the inventory rather than
    :attr:`WinMLEPRegistry._registered` is what keeps enrichment
    working now that filesystem-backed EPs are registered only in
    isolated subprocesses (their live handles never exist in the parent).

    When ``tolerant=True``, a per-section failure is logged at WARNING
    and the section is filled with an empty container so downstream
    renderers still see consistent keys; otherwise the failure is
    converted to ``click.ClickException``.
    """
    info: dict[str, Any] = {}
    if system:
        info.update(_gather_system_info(verbose=verbose))
    if eps:
        try:
            info["executionProviders"] = _gather_ep_info()
        except Exception as e:
            if not tolerant:
                logger.exception("Failed to detect execution providers")
                raise click.ClickException(f"Error detecting execution providers: {e}") from e
            logger.warning("EP detection failed (tolerant): %s", e)
            info["executionProviders"] = {}
    if devices:
        try:
            info["devices"] = _gather_device_info(info.get("executionProviders"))
        except Exception as e:
            if not tolerant:
                logger.exception("Failed to detect devices")
                raise click.ClickException(f"Error detecting devices: {e}") from e
            logger.warning("Device detection failed (tolerant): %s", e)
            info["devices"] = []
    return info


def _render_text(info: dict[str, Any], verbose: bool) -> None:
    """Rich-console output. Renders whichever sections are present."""
    console = _get_console()
    if "python" in info:
        _output_text(info, verbose=verbose)
        console.print()
    if "devices" in info:
        _output_device_text(info["devices"])
        console.print()
    if "executionProviders" in info:
        _output_ep_text(info["executionProviders"])


def _render_json(info: dict[str, Any], _verbose: bool) -> None:
    """JSON dump — keys present in ``info`` determine the payload shape."""
    _output_json(info)


def _render_compact(info: dict[str, Any], _verbose: bool) -> None:
    """One-line-per-aspect summary."""
    if "python" in info:
        _output_compact(info)
    if "devices" in info:
        parts = [f"{d['type']}: {d['name'].strip()}" for d in info["devices"]]
        click.echo(" | ".join(parts) if parts else "No devices found")
    if "executionProviders" in info:
        parts = list(info["executionProviders"])
        click.echo("EPs: " + ", ".join(parts) if parts else "EPs: none")


_RENDERERS: dict[str, Callable[[dict[str, Any], bool], None]] = {
    "text": _render_text,
    "json": _render_json,
    "compact": _render_compact,
}


@click.command()
@cli_utils.format_option(choices=["text", "json", "compact"], default="text")
@click.option(
    "--list-device",
    is_flag=True,
    default=False,
    help="List available devices in priority order",
)
@click.option(
    "--list-ep",
    is_flag=True,
    default=False,
    help="List available execution providers",
)
@cli_utils.verbosity_options()
@cli_utils.no_color_option()
@click.pass_context
def sysinfo(
    ctx: click.Context,
    output_format: cli_utils.OutputFormat,
    verbose: int,
    quiet: bool,
    list_device: bool,
    list_ep: bool,
) -> None:
    r"""Display system information for WinML CLI export.

    This command gathers and displays information relevant to ONNX model
    export, including Python version, library versions, hardware
    capabilities, and backend SDK availability.

    Use this to diagnose issues with model export or verify your
    environment is properly configured.

    \b
    Examples:
        # Display system info (human-readable format)
        winml sys

        # Get output as JSON for scripting
        winml sys --format json

        # Show detailed info
        winml sys --verbose

        # Compact format for quick overview
        winml sys --format compact

        # List available devices
        winml sys --list-device

        # List execution providers as JSON
        winml sys --list-ep --format json
    """
    # Merge top-level -v/-q with subcommand-level flags so either position works.
    verbose, quiet = cli_utils.resolve_verbosity(ctx, verbose, quiet)

    # Standard verbosity contract: stderr-only logs in the shared format.
    # `-v` here keeps its `sys`-specific second job of expanding the displayed
    # diagnostics; see the table-DEBUG audit follow-up for fully decoupling
    # them.
    configure_logging(verbosity=verbose, quiet=quiet)

    fmt = output_format.lower()
    if list_device or list_ep:
        # Explicit-section mode: raise on per-section error so the
        # user knows their pin didn't produce a result.
        info = _gather(
            devices=list_device,
            eps=list_ep,
            verbose=bool(verbose),
            tolerant=False,
        )
    else:
        # Default mode: always include system info; sections only
        # for non-compact formats (compact is a sysinfo overview by
        # convention); tolerant so a broken section doesn't blank
        # the whole report.
        include_sections = fmt != "compact"
        info = _gather(
            system=True,
            devices=include_sections,
            eps=include_sections,
            verbose=bool(verbose),
            tolerant=True,
        )
    _RENDERERS[fmt](info, bool(verbose))
