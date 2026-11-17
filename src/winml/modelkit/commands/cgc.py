# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""D3D12 adapters on this machine, and the patterns their drivers declare.

Two questions get asked on every new machine and after every driver update: which
adapters are here, and what does this driver's MLIR-program implementation claim it
can match. Both answers come from public Windows APIs through ``ctypes``, so nothing
has to be built and no extra package has to be installed.

It does need one thing: an Agility SDK ``D3D12Core.dll`` of SDK version 720 or newer.
The MLIR-program feature is a preview feature and the inbox D3D12 runtime does not
serve it. Which redist answered opens every run on the ``redist:`` line, because the
same driver reports different things through the 720 and 721 exchange shapes; if none
can be found the failure names every place that was searched.

Columns:
    IDX         selection index, as accepted by ``-a``
    ADAPTER     DXCore DriverDescription
    DRIVER      DXCore DriverVersion, the four 16-bit parts
    TYPE        hardware, integrated or software
    ATTRIBUTES  the DXCore attributes the adapter advertises: ML, CC, GFX
    MLIR        whether the driver implements the MLIR-program exchange; ``?`` when no
                Agility SDK redist was found, since the question cannot be asked

Counts are reported as ``N patterns / M rules``. A pattern is one
``cgc_pattern.pattern`` declaration; a rule is one flat match alternative after
``any_of`` expansion. The two differ, and comparing declaration counts alone across
drivers can read a loss where the matching surface actually grew.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import json
import logging
import os
import re
import struct
import sys
import sysconfig
import tempfile
import uuid
import webbrowser
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import click

from .. import commands
from ..utils import cli as cli_utils
from ..utils.logging import configure_logging


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)


class GUID(ctypes.Structure):
    """A Windows GUID, laid out for ``ctypes``."""

    _fields_ = (
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    )


def guid(text: str) -> GUID:
    """Build a :class:`GUID` from its registry form, as the Windows SDK headers print it.

    Args:
        text: For example ``78EE5945-C36E-4B13-A669-005DD11C0F06``.

    Returns:
        The populated GUID.
    """
    return GUID.from_buffer_copy(uuid.UUID(text).bytes_le)


def vcall(this: ctypes.c_void_p, index: int, restype: Any, *argtypes: Any) -> Callable[..., Any]:
    """Bind vtable slot *index* of COM pointer *this*.

    Replaces a comtypes dependency with six lines.

    Args:
        this: The COM interface pointer.
        index: Zero-based vtable slot.
        restype: ``ctypes`` return type.
        *argtypes: ``ctypes`` argument types, excluding the implicit ``this``.

    Returns:
        A callable that invokes the method, passing ``this`` automatically.
    """
    vtbl = ctypes.cast(this, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    fn = proto(vtbl[index])

    return lambda *args: fn(this, *args)


def release(p: ctypes.c_void_p | None) -> None:
    """Call ``IUnknown::Release`` on *p* when it is non-NULL.

    Args:
        p: The COM interface pointer, or None.
    """
    if p:
        vcall(p, 2, ctypes.c_ulong)()


E_FAIL = -2147467259  # 0x80004005 as a signed HRESULT

HRESULTS = {
    0x00000000: "S_OK",
    0x80004001: "E_NOTIMPL",
    0x80004005: "E_FAIL",
    0x8000FFFF: "E_UNEXPECTED",
    0x80070057: "E_INVALIDARG",
    0x887A0004: "DXGI_ERROR_UNSUPPORTED",
    0x887E0001: "D3D12_ERROR_ADAPTER_NOT_FOUND",
    0x887E0002: "D3D12_ERROR_DRIVER_VERSION_MISMATCH",
    0x887E0003: "D3D12_ERROR_INVALID_REDIST",
}


def hrs(code: int) -> str:
    """Render an HRESULT as hex, plus its name when one is known.

    Args:
        code: The HRESULT, signed or unsigned.

    Returns:
        For example ``0x80070057 (E_INVALIDARG)``; unknown codes print raw.
    """
    u = code & 0xFFFFFFFF
    return f"0x{u:08X}" + (f" ({HRESULTS[u]})" if u in HRESULTS else "")


def hr(name: str, code: int) -> None:
    """Exit when *code* is a failing HRESULT.

    Args:
        name: The call being checked, for the message.
        code: The returned HRESULT.

    Raises:
        click.ClickException: When *code* indicates failure.
    """
    if code < 0:
        raise click.ClickException(f"{name} failed: {hrs(code)}")


# ----------------------------------------------------------------------- dxcore.dll
# dxcore_interface.h. IDXCoreAdapterFactory : IUnknown -> 3 CreateAdapterList.
# IDXCoreAdapterList -> 3 GetAdapter, 4 GetAdapterCount, 7 Sort.
# IDXCoreAdapter -> 4 IsAttributeSupported, 5 IsPropertySupported, 6 GetProperty,
#                   7 GetPropertySize.
IID_IDXCoreAdapterFactory = guid("78EE5945-C36E-4B13-A669-005DD11C0F06")
IID_IDXCoreAdapterList = guid("526C7776-40E9-459B-B711-F32AD76DFC28")
IID_IDXCoreAdapter = guid("F0DB4C7F-FE5A-42A2-BD62-F2A6CF6FC83E")
ATTR_GENERIC_ML = guid("B71B0D41-1088-422F-A27C-0250B7D3A988")
ATTR_CORE_COMPUTE = guid("248E2800-A793-4724-ABAA-23A6DE1BE090")
ATTR_D3D12_GRAPHICS = guid("0C9ECE4D-2F6E-4F01-8C96-E89E331B47B1")

PROP_DRIVER_VERSION, PROP_DRIVER_DESCRIPTION = 1, 2
PROP_IS_HARDWARE, PROP_IS_INTEGRATED = 11, 12
PREF_HARDWARE, PREF_HIGH_PERFORMANCE = 0, 2

Adapter = dict[str, Any]


def read_adapter_props(ad: ctypes.c_void_p) -> Adapter:
    """Read every property the listing shows from one ``IDXCoreAdapter``.

    ``IsPropertySupported`` returns a C++ bool, one byte.

    Args:
        ad: The adapter interface pointer.

    Returns:
        The adapter record, without ``index``, ``ptr``, ``mlir`` or ``ir_version``.
    """
    is_prop = vcall(ad, 5, ctypes.c_bool, ctypes.c_uint32)
    get_prop = vcall(ad, 6, ctypes.c_long, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p)
    get_size = vcall(ad, 7, ctypes.c_long, ctypes.c_uint32, ctypes.POINTER(ctypes.c_size_t))
    is_attr = vcall(ad, 4, ctypes.c_bool, ctypes.POINTER(GUID))

    def read(pid: int, label: str) -> Any:
        # The size DXCore reports is honoured rather than assumed. The two flags are 1
        # byte, not a 4-byte BOOL; reading them as BOOL marks every adapter "software",
        # which silently changes which adapter the default -a selects.
        n = ctypes.c_size_t()
        hr(f"GetPropertySize({label})", get_size(pid, ctypes.byref(n)))
        buf = ctypes.create_string_buffer(n.value)
        hr(f"GetProperty({label})", get_prop(pid, n.value, buf))
        return buf

    p: Adapter = {
        "description": "",
        "driver_version": "",
        "is_hardware": False,
        "is_integrated": False,
    }
    if is_prop(PROP_DRIVER_DESCRIPTION):
        # The size includes the NUL, which .value stops at.
        description = read(PROP_DRIVER_DESCRIPTION, "DriverDescription").value
        p["description"] = description.decode("utf-8", "replace")
    if is_prop(PROP_DRIVER_VERSION):
        v = ctypes.c_uint64()
        hr("GetProperty(DriverVersion)", get_prop(PROP_DRIVER_VERSION, 8, ctypes.byref(v)))
        p["driver_version"] = format_driver_version(v.value)
    for key, pid in (("is_hardware", PROP_IS_HARDWARE), ("is_integrated", PROP_IS_INTEGRATED)):
        if is_prop(pid):
            p[key] = any(read(pid, key).raw)
    p["generic_ml"] = bool(is_attr(ctypes.byref(ATTR_GENERIC_ML)))
    p["core_compute"] = bool(is_attr(ctypes.byref(ATTR_CORE_COMPUTE)))
    p["d3d12_graphics"] = bool(is_attr(ctypes.byref(ATTR_D3D12_GRAPHICS)))
    return p


def format_driver_version(raw: int) -> str:
    """Render the DXCore u64 driver version as four 16-bit parts, high word first.

    Args:
        raw: The packed 64-bit value.

    Returns:
        For example ``32.0.16.3004``.
    """
    return f"{(raw >> 48) & 0xFFFF}.{(raw >> 32) & 0xFFFF}.{(raw >> 16) & 0xFFFF}.{raw & 0xFFFF}"


def list_adapters() -> list[Adapter]:
    """Enumerate every D3D12 adapter, sorted hardware-then-high-performance.

    Creates no device, so this costs nothing and needs no redist.

    Returns:
        One record per adapter. Each owns a ``ptr`` that :func:`close_adapters`
        must release.
    """
    try:
        dxcore = ctypes.WinDLL("dxcore.dll")
    except OSError as e:
        raise click.ClickException(f"dxcore.dll could not be loaded: {e}") from e
    dxcore.DXCoreCreateAdapterFactory.restype = ctypes.c_long
    dxcore.DXCoreCreateAdapterFactory.argtypes = [
        ctypes.POINTER(GUID),
        ctypes.POINTER(ctypes.c_void_p),
    ]

    factory = ctypes.c_void_p()
    hr(
        "DXCoreCreateAdapterFactory",
        dxcore.DXCoreCreateAdapterFactory(
            ctypes.byref(IID_IDXCoreAdapterFactory), ctypes.byref(factory)
        ),
    )
    create_list = vcall(
        factory,
        3,
        ctypes.c_long,
        ctypes.c_uint32,
        ctypes.POINTER(GUID),
        ctypes.POINTER(GUID),
        ctypes.POINTER(ctypes.c_void_p),
    )

    alist = ctypes.c_void_p()
    hr(
        "CreateAdapterList(GENERIC_ML)",
        create_list(
            1,
            ctypes.byref(ATTR_GENERIC_ML),
            ctypes.byref(IID_IDXCoreAdapterList),
            ctypes.byref(alist),
        ),
    )
    if vcall(alist, 4, ctypes.c_uint32)() == 0:
        release(alist)
        alist = ctypes.c_void_p()
        hr(
            "CreateAdapterList(CORE_COMPUTE)",
            create_list(
                1,
                ctypes.byref(ATTR_CORE_COMPUTE),
                ctypes.byref(IID_IDXCoreAdapterList),
                ctypes.byref(alist),
            ),
        )

    prefs = (ctypes.c_uint32 * 2)(PREF_HARDWARE, PREF_HIGH_PERFORMANCE)
    vcall(alist, 7, ctypes.c_long, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32))(2, prefs)

    count = vcall(alist, 4, ctypes.c_uint32)()
    get_adapter = vcall(
        alist,
        3,
        ctypes.c_long,
        ctypes.c_uint32,
        ctypes.POINTER(GUID),
        ctypes.POINTER(ctypes.c_void_p),
    )
    adapters: list[Adapter] = []
    for i in range(count):
        ad = ctypes.c_void_p()
        hr("GetAdapter", get_adapter(i, ctypes.byref(IID_IDXCoreAdapter), ctypes.byref(ad)))
        props = read_adapter_props(ad)
        props["index"] = i
        props["ptr"] = ad
        props["mlir"], props["ir_version"] = None, 0  # set by probe_mlir_support
        adapters.append(props)
    release(alist)
    release(factory)
    return adapters


def close_adapters(adapters: list[Adapter]) -> None:
    """Release every adapter pointer in *adapters*.

    Args:
        adapters: The records returned by :func:`list_adapters`.
    """
    for a in adapters:
        release(a.pop("ptr", None))


def adapter_slug(description: str) -> str:
    """Turn an adapter description into a directory-safe name.

    ``"NVIDIA GeForce RTX 5090 D"`` becomes ``"nvidia-geforce-rtx-5090-d"``.

    Args:
        description: The DXCore DriverDescription.

    Returns:
        The slug.
    """
    s = re.sub(r"\((?:r|tm|c)\)", "", description, flags=re.I)
    s = re.sub(r"[^0-9a-z]+", "-", s.lower())
    # Never empty: dump_dir joins it, and "" would collapse the per-adapter level.
    return s.strip("-") or "adapter"


def select_adapter(adapters: list[Adapter], wanted: str) -> Adapter:
    """Resolve ``-a`` to one adapter.

    ``-a`` takes a description substring or a bare index. With neither, the first
    hardware adapter wins -- not index 0, which on many machines is a software adapter.

    Args:
        adapters: Every adapter found; never empty.
        wanted: The ``-a`` value, possibly empty.

    Returns:
        The selected adapter.

    Raises:
        click.UsageError: When nothing matches *wanted*.
    """
    if not wanted:
        return next((a for a in adapters if a["is_hardware"]), adapters[0])
    # A number is an index when one exists, else a substring: "5060" names the RTX
    # 5060 on a three-adapter machine rather than failing as an out-of-range index.
    if wanted.isdigit() and int(wanted) < len(adapters):
        return adapters[int(wanted)]
    hit = next((a for a in adapters if wanted.lower() in a["description"].lower()), None)
    if hit is None:
        found = ", ".join(a["description"] for a in adapters)
        raise click.UsageError(f"no adapter matching {wanted!r}; found: {found}")
    return hit


# ------------------------------------------------------- the Agility SDK redist
# The MLIR-program feature is a preview feature: the inbox D3D12 runtime answers
# E_INVALIDARG for it, so a device must be created against an Agility SDK redist. Which
# exchange ABI to speak is a property of that redist, which is why its SDK version is
# read here rather than guessed, and read by parsing the PE export table rather than by
# loading the DLL -- loading pins a core in-process before the choice has been made.

MACHINES = {0x014C: "x86", 0x8664: "x64", 0xAA64: "arm64"}

#: The oldest Agility SDK that serves D3D MLIR programs. Anything below this answers
#: E_INVALIDARG for the feature and would report a capable driver as "no" -- a silently
#: wrong answer, so such a core is named and skipped rather than used.
MIN_SDK_VERSION = 720

#: Environment variables naming a redist directory, in the order they are consulted.
REDIST_ENV_VARS = ("WINML_D3D12_DIR", "D3D12_DIR")

#: Where a redist may sit under each search root. The Agility SDK's own convention is a
#: ``D3D12`` subfolder beside the binaries; ``bin`` holding ``D3D12Core.dll`` directly
#: is accepted too.
REDIST_SUBDIRS = ("bin/D3D12", "bin")


def host_machine() -> str:
    """Return this interpreter's machine name as the PE header spells it.

    Returns:
        ``x64``, ``x86``, ``arm64``, or ``?``.
    """
    # The interpreter's own build, which is what decides whether this process can load
    # the DLL. platform.machine() would answer the same on a native install, but on 3.11
    # it runs `cmd /c ver` first, ~45 ms on every run.
    return {"win-amd64": "x64", "win32": "x86", "win-arm64": "arm64"}.get(
        sysconfig.get_platform(), "?"
    )


def pe_export_u32(path: Path, want: bytes) -> tuple[str | None, int | None]:
    """Read a UINT32 data export out of a PE file without loading it.

    Args:
        path: The PE file.
        want: The export name, as bytes.

    Returns:
        ``(machine, value)``, either of which is None when unavailable.

    Raises:
        struct.error: When the file is truncated, or the export table points outside
            every section.
    """
    data = path.read_bytes()
    if data[:2] != b"MZ":
        return None, None
    (pe,) = struct.unpack_from("<I", data, 0x3C)
    if data[pe : pe + 4] != b"PE\0\0":
        return None, None
    machine, nsec = struct.unpack_from("<HH", data, pe + 4)
    (opt_size,) = struct.unpack_from("<H", data, pe + 20)
    arch = MACHINES.get(machine, f"0x{machine:04X}")

    opt = pe + 24
    (magic,) = struct.unpack_from("<H", data, opt)
    dd = opt + (112 if magic == 0x20B else 96 if magic == 0x10B else 0)
    if dd == opt:
        return arch, None
    (exp_rva,) = struct.unpack_from("<I", data, dd)
    if not exp_rva:
        return arch, None

    # (VirtualSize, VirtualAddress, SizeOfRawData, PointerToRawData) per section.
    sections = [struct.unpack_from("<IIII", data, opt + opt_size + i * 40 + 8) for i in range(nsec)]

    def off(rva: int) -> int:
        for vsize, vaddr, rawsize, rawptr in sections:
            if vaddr <= rva < vaddr + max(vsize, rawsize):
                return int(rawptr + (rva - vaddr))
        # A malformed export table is a reason to reject this candidate and try
        # the next one, so raise something inspect_redist already catches.
        raise struct.error(f"RVA 0x{rva:X} is outside every section")

    n_names, a_funcs, a_names, a_ords = struct.unpack_from("<IIII", data, off(exp_rva) + 24)
    for i in range(n_names):
        (name_rva,) = struct.unpack_from("<I", data, off(a_names) + i * 4)
        name = off(name_rva)
        if data[name : name + len(want) + 1] == want + b"\0":
            (ordinal,) = struct.unpack_from("<H", data, off(a_ords) + i * 2)
            (func_rva,) = struct.unpack_from("<I", data, off(a_funcs) + ordinal * 4)
            return arch, int(struct.unpack_from("<I", data, off(func_rva))[0])
    return arch, None


def inspect_redist(path: Path) -> tuple[int | None, str]:
    """Decide whether *path* is a usable Agility SDK redist.

    Args:
        path: A candidate directory.

    Returns:
        ``(sdk_version, reason)``. ``sdk_version`` is None when the directory cannot
        be used, and ``reason`` then says why.
    """
    dll = path / "D3D12Core.dll"
    try:
        # Every filesystem call is inside the guard: an unreachable share, an
        # ACL-restricted directory or a stale mount must be a rejected candidate, not a
        # traceback, because every implicit root is probed on every run.
        if not path.is_dir():
            return None, "missing"
        if not dll.is_file():
            return None, "empty" if not any(path.iterdir()) else "no D3D12Core.dll"
        arch, version = pe_export_u32(dll, b"D3D12SDKVersion")
    except (OSError, struct.error) as e:
        return None, f"unreadable: {e}"
    if version is None:
        return None, "no D3D12SDKVersion export"
    if arch != host_machine():
        return None, f"arch {arch} != {host_machine()}"
    if version < MIN_SDK_VERSION:
        return None, (
            f"SDK {version} is older than {MIN_SDK_VERSION} and does not serve D3D MLIR programs"
        )
    return version, f"SDK {version}"


def _redist_roots() -> list[tuple[str, Path]]:
    """Return the directories that may hold a ``bin`` folder, in search order.

    Two placements are supported, matching where a redist is actually dropped:
    the root of a development environment, and the root of a wheel installation.

    The development root is the directory holding the virtual environment -- the
    ``bin`` folder sits beside ``.venv``. That is checked before the worktree
    marker because it is the only rule that works once winml-cli is installed
    from a wheel into a project's venv: there is no ``pyproject.toml`` to walk up
    to from ``site-packages``, but ``.venv``'s parent is still the project.

    "Installation root" is deliberately read broadly -- the installed ``winml``
    package, the ``site-packages`` directory holding it, and the environment
    prefix -- because a wheel can deliver files to any of the three depending on
    whether it ships them as package data or as data files.

    Returns:
        ``(label, root)`` pairs. Duplicates are dropped downstream, so the common
        case where the venv sits in the worktree yields one set of candidates.
    """
    package = Path(__file__).resolve().parent.parent.parent  # .../winml
    roots: list[tuple[str, Path]] = []
    # sys.prefix differs from base_prefix only inside a virtual environment; outside
    # one its parent is the interpreter's install root and means nothing here.
    #
    # This is the one root not bounded by the project or install tree -- it walks *up*
    # out of the venv -- so it stops at a filesystem anchor. A venv placed directly at
    # a root would otherwise make that root the highest-priority candidate: `C:\.venv`
    # yields `C:\bin\D3D12` and `\\server\share\venv` yields `\\server\share\bin`.
    # Since the resolved directory is handed to CreateDeviceFactory, which loads
    # D3D12Core.dll as native code into this process, and default Windows ACLs let a
    # standard user create directories at a drive root, such a layout would turn the
    # first candidate into a plantable DLL path. resolve() first so a junctioned
    # prefix does not defeat the de-duplication below either.
    if sys.prefix != sys.base_prefix:
        venv_parent = Path(sys.prefix).resolve().parent
        if venv_parent != venv_parent.parent:
            roots.append(("<venv-parent>", venv_parent))
    # A development checkout: the nearest ancestor holding a pyproject.toml; none in a
    # wheel install. Same anchor guard as <venv-parent>: the walk reaches every
    # ancestor, so a pyproject.toml at a volume root would otherwise become a base.
    here = Path(__file__).resolve()
    repo = next((d for d in here.parents if (d / "pyproject.toml").is_file()), None)
    if repo is not None and repo != repo.parent:
        roots.append(("<worktree>", repo))
    roots.append(("<package>", package))
    roots.append(("<site-packages>", package.parent))
    roots.append(("<prefix>", Path(sys.prefix)))
    return roots


def _named_redist(explicit: str = "") -> tuple[str, Path] | None:
    """Return the redist the caller named by hand, if any.

    The single place that decides precedence -- the flag, then each variable in
    :data:`REDIST_ENV_VARS`, first non-empty wins -- so the candidate list and the
    decision whether an unusable redist is fatal can never disagree.

    Args:
        explicit: The ``--d3d12-dir`` value, possibly empty.

    Returns:
        ``(label, path)``, or None when the caller named nothing.
    """
    if explicit:
        return "--d3d12-dir", Path(explicit)
    for var in REDIST_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return f"${var}", Path(value)
    return None


# TODO(tests): the documented search order (flag, variables, then each bin root)
# and the filesystem-anchor guard on <venv-parent> have no test.
def redist_candidates(explicit: str = "") -> Iterator[tuple[str, Path]]:
    """Yield ``(label, path)`` in resolution order.

    A redist the caller named (see :func:`_named_redist`) is the only candidate, usable
    or not: falling through to a different core would answer the capability question
    about a runtime the caller did not ask for.

    Args:
        explicit: The ``--d3d12-dir`` value, possibly empty.

    Yields:
        A label for the failure report, and the directory to inspect.
    """
    named = _named_redist(explicit)
    if named is not None:
        yield named
        return
    seen: set[Path] = set()
    for label, root in _redist_roots():
        for sub in REDIST_SUBDIRS:
            path = root / sub
            if path not in seen:
                seen.add(path)
                yield f"{label}/{sub}", path


def resolve_redist(
    explicit: str = "",
) -> tuple[Path | None, int | None, list[tuple[str, Path, str]]]:
    """Find the first usable redist.

    Args:
        explicit: The ``--d3d12-dir`` value, possibly empty.

    Returns:
        ``(path, sdk_version, tried)``. *path* is None when nothing usable was found,
        and *tried* records every candidate and why it was rejected.
    """
    tried: list[tuple[str, Path, str]] = []
    for label, path in redist_candidates(explicit):
        version, reason = inspect_redist(path)
        tried.append((label, path, reason))
        if version is not None:
            return path, version, tried
    return None, None, tried


def redist_failure(tried: list[tuple[str, Path, str]]) -> str:
    """Name every place looked and why each was rejected.

    Silence here costs a day.

    Args:
        tried: The candidates from :func:`resolve_redist`.

    Returns:
        The multi-line report, for stderr.
    """
    out = [
        "cgc: no usable D3D12 Agility SDK redist found (need a directory containing",
        f"     D3D12Core.dll of SDK {MIN_SDK_VERSION} or newer, built for {host_machine()}).",
        "",
        "Looked in, in order:",
    ]
    for i, (label, path, reason) in enumerate(tried, 1):
        out.append(f"  {i}. {label:<26} {path}")
        out.append(f"     {reason}")
    out += [
        "",
        "Fix one of:",
        "  - point at one:  winml cgc adapters --d3d12-dir <dir containing D3D12Core.dll>",
        f"  - set the env:   {REDIST_ENV_VARS[0]}=<that directory>",
        "",
        "Note: C:/Windows/System32/D3D12Core.dll is the inbox core and does not",
        "implement D3D MLIR programs. cgc does not fall back to it, because doing so",
        "creates a healthy-looking device with the MLIR path silently absent.",
    ]
    return "\n".join(out)


# ------------------------------------------------------------------- d3d12.dll
CLSID_D3D12SDKConfiguration = guid("7CDA6ACA-A03E-49C8-9458-0334D20E07CE")
IID_ID3D12SDKConfiguration1 = guid("8AAF9303-AD25-48B9-9A57-D9C37E009D9F")
IID_ID3D12DeviceFactory = guid("61F307D3-D34E-4E7C-8374-3BA4DE23CCCB")
IID_ID3D12Device = guid("189819F1-1DB6-4B57-BE54-1821339B85F7")

FEATURE_LEVELS = (0x0100, 0x1000, 0xB000)  # 1_0_GENERIC, 1_0_CORE, 11_0

FEATURE_MLIR_EXCHANGE = 69  # the same id under both ABIs
FEATURE_MLIR_70 = 70  # INTERFACE_SUPPORT on 720, COMPUTE_GRAPH_VERSION on 721
MLIR_EXCHANGE_SUBGRAPH_DECLARATION = 0

# include/utilities/MlirInterfaceGuids.h:51 -- the one DXML-private constant here.
GUID_SUBGRAPH_DECLARATION_REQUEST = guid("EB53032A-1116-4E71-8309-54347C1E5A26")

CGC_MAX_IR_VERSION = (0, 7, 0, 0)  # cmake/version.cmake CGC_VERSION; an upper bound


class FeatureDataMLIRComputeGraphVersion(ctypes.Structure):
    """``D3D12_FEATURE_DATA_MLIR_COMPUTE_GRAPH_VERSION``, the 721 capability query."""

    _fields_ = (("HighestVersion", ctypes.c_uint64),)


class FeatureDataMLIRExchange721(ctypes.Structure):
    """``D3D12_FEATURE_DATA_MLIR_EXCHANGE`` as SDK 721 shapes it.

    48 bytes on x64, with 4 bytes of padding after Type. The driver reads it by size,
    so a drifted layout is silently wrong; ``test_721_exchange_struct_is_48_bytes``
    pins it.
    """

    _fields_ = (
        ("Type", ctypes.c_uint32),
        ("IRVersion", ctypes.c_uint64),  # D3D12_VERSION_NUMBER, a UINT64 union
        ("pInputData", ctypes.c_void_p),
        ("InputDataSizeInBytes", ctypes.c_size_t),
        ("pOutputData", ctypes.c_void_p),
        ("OutputDataSizeInBytes", ctypes.POINTER(ctypes.c_size_t)),
    )


class FeatureDataMLIRExchange720(ctypes.Structure):
    """``D3D12_FEATURE_DATA_MLIR_EXCHANGE`` as SDK 720 shapes it."""

    _fields_ = (
        ("MlirInterface", GUID),
        ("pInputData", ctypes.c_void_p),
        ("InputDataSizeInBytes", ctypes.c_size_t),
        ("pOutputData", ctypes.c_void_p),
        ("OutputDataSizeInBytes", ctypes.POINTER(ctypes.c_size_t)),
    )


class FeatureDataMLIRInterfaceSupport720(ctypes.Structure):
    """``D3D12_FEATURE_DATA_MLIR_INTERFACE_SUPPORT``, the 720 capability query."""

    _fields_ = (
        ("NumMlirInterfaces", ctypes.c_uint32),
        ("pMlirInterfacesRequested", ctypes.POINTER(GUID)),
        ("pMlirInterfacesSupported", ctypes.POINTER(ctypes.wintypes.BOOL)),
    )


def make_version_number(major: int, minor: int, build: int, rev: int) -> int:
    """Pack four 16-bit parts into a ``D3D12_VERSION_NUMBER``.

    Args:
        major: High word.
        minor: Second word.
        build: Third word.
        rev: Low word.

    Returns:
        The packed 64-bit value.
    """
    return (
        ((major & 0xFFFF) << 48)
        | ((minor & 0xFFFF) << 32)
        | ((build & 0xFFFF) << 16)
        | (rev & 0xFFFF)
    )


class RedistUnusable(click.ClickException):
    """D3D12 cannot load, or rejects, the redist itself, so no adapter can be asked."""


def create_device(
    adapter_ptr: ctypes.c_void_p, redist: Path, sdk_version: int
) -> tuple[ctypes.c_void_p | None, int]:
    """Create an ``ID3D12Device`` on *adapter_ptr* through the given redist.

    Args:
        adapter_ptr: The DXCore adapter pointer.
        redist: The directory holding ``D3D12Core.dll``.
        sdk_version: That core's ``D3D12SDKVersion``.

    Returns:
        ``(device, 0)``, or ``(None, hresult)`` when this adapter could not get a
        device at any feature level.

    Raises:
        RedistUnusable: When D3D12 cannot load or rejects the redist itself, since
            then no adapter can be asked through it.
    """
    # The path must be absolute: a relative one resolves against the host exe, which is
    # python.exe. CreateDeviceFactory takes it as a narrow string, but D3D12 decodes it
    # as UTF-8 -- not the active ANSI codepage. Encoding it as ANSI makes any path with
    # a non-ASCII character (an accented profile name, CJK) fail with
    # D3D12_ERROR_INVALID_REDIST.
    # TODO(tests): no test covers a non-ASCII or CJK redist path.
    try:
        redist_utf8 = str(redist.resolve()).encode("utf-8")
    except UnicodeEncodeError as e:  # only an unpaired surrogate in a file name
        raise click.ClickException(f"redist path is not valid Unicode: {redist}") from e
    try:
        d3d12 = ctypes.WinDLL("d3d12.dll")
    except OSError as e:
        # A host without the D3D12 runtime (Server Core without the graphics
        # feature, some container images) must fail the way every other
        # could-not-ask failure does, not with a bare traceback.
        raise RedistUnusable(f"d3d12.dll could not be loaded: {e}") from e
    d3d12.D3D12GetInterface.restype = ctypes.c_long
    d3d12.D3D12GetInterface.argtypes = [
        ctypes.POINTER(GUID),
        ctypes.POINTER(GUID),
        ctypes.POINTER(ctypes.c_void_p),
    ]

    cfg, factory = ctypes.c_void_p(), ctypes.c_void_p()
    try:
        rc = d3d12.D3D12GetInterface(
            ctypes.byref(CLSID_D3D12SDKConfiguration),
            ctypes.byref(IID_ID3D12SDKConfiguration1),
            ctypes.byref(cfg),
        )
        if rc < 0 or not cfg:
            # This system cannot load an Agility core at all, so no adapter can be asked.
            raise RedistUnusable(
                f"D3D12 cannot load an Agility SDK redist on this system "
                f"(D3D12GetInterface: {hrs(rc if rc < 0 else E_FAIL)})"
            )

        # ID3D12SDKConfiguration1 -> 4 CreateDeviceFactory.
        create_factory = vcall(
            cfg,
            4,
            ctypes.c_long,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.POINTER(GUID),
            ctypes.POINTER(ctypes.c_void_p),
        )
        rc = create_factory(
            sdk_version,
            redist_utf8,
            ctypes.byref(IID_ID3D12DeviceFactory),
            ctypes.byref(factory),
        )
        if rc < 0 or not factory:
            # D3D12 rejected the redist itself, so no adapter can be asked through it.
            # Returning per adapter would record every driver as "no" -- a confident
            # wrong answer, and a permanent status=unsupported baseline under --dump.
            raise RedistUnusable(
                f"D3D12 rejected the redist at {redist} "
                f"(CreateDeviceFactory: {hrs(rc if rc < 0 else E_FAIL)}); cannot ask "
                f"whether any driver implements MLIR programs"
            )

        create = vcall(
            factory,
            9,
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(GUID),
            ctypes.POINTER(ctypes.c_void_p),
        )
        last = 0
        for level in FEATURE_LEVELS:
            dev = ctypes.c_void_p()
            last = create(adapter_ptr, level, ctypes.byref(IID_ID3D12Device), ctypes.byref(dev))
            if last >= 0 and dev:
                return dev, 0
        # No feature level gave this adapter a device: a per-adapter "no". A success
        # code with a NULL device is still a failure, so it is never reported as S_OK.
        return None, last if last < 0 else E_FAIL
    finally:
        release(factory)
        release(cfg)


def check_feature_support(device: ctypes.c_void_p) -> Any:
    """Bind ``ID3D12Device::CheckFeatureSupport``, vtable slot 13.

    Args:
        device: The device pointer.

    Returns:
        The bound callable. This is the only call that does real work.
    """
    return vcall(device, 13, ctypes.c_long, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32)


def supports_exchange(device: ctypes.c_void_p, sdk_version: int) -> tuple[bool, int, int]:
    """Ask whether this *driver* implements the exchange.

    Decided by value, never by HRESULT.

    Args:
        device: The device pointer.
        sdk_version: The redist's SDK version, which picks the ABI shape.

    Returns:
        ``(supported, hresult, negotiated_ir_version)``.
    """
    cfs = check_feature_support(device)
    if sdk_version >= 721:
        data = FeatureDataMLIRComputeGraphVersion(make_version_number(*CGC_MAX_IR_VERSION))
        rc = cfs(FEATURE_MLIR_70, ctypes.byref(data), ctypes.sizeof(data))
        # 70 is answered by the runtime, not the driver: WARP and Intel both return
        # S_OK here with HighestVersion == 0. The value is the answer.
        if rc < 0:
            return False, rc, 0
        return data.HighestVersion != 0, rc, data.HighestVersion
    supported = ctypes.wintypes.BOOL(0)
    data_720 = FeatureDataMLIRInterfaceSupport720(
        1, ctypes.pointer(GUID_SUBGRAPH_DECLARATION_REQUEST), ctypes.pointer(supported)
    )
    rc = cfs(FEATURE_MLIR_70, ctypes.byref(data_720), ctypes.sizeof(data_720))
    return (rc >= 0 and bool(supported.value)), rc, 0


# TODO(tests): TestCapabilityOnRealHardware cannot fail -- it only asserts that
# mlir is a bool, which holds even when no device was created at all.
def probe_mlir_support(
    adapter: Adapter, redist: Path, sdk_version: int, *, verbose: bool = False
) -> ctypes.c_void_p | None:
    """Set ``adapter["mlir"]``, and ``adapter["ir_version"]`` when the driver answers.

    Any failing HRESULT means no exchange -- do not read into it.

    Args:
        adapter: The adapter record, mutated in place.
        redist: The directory holding ``D3D12Core.dll``.
        sdk_version: That core's SDK version.
        verbose: Write the raw capability HRESULT to stderr.

    Returns:
        The device when the driver implements the exchange, for the caller to ask and
        then release; else None.

    Raises:
        RedistUnusable: From :func:`create_device`, when the redist itself cannot be
            used; no adapter's answer is recorded then.
    """
    device, rc = create_device(adapter["ptr"], redist, sdk_version)
    if device is None:
        adapter["mlir"] = False
        if verbose:
            click.echo(f"  {adapter['description']}: CreateDevice failed: {hrs(rc)}", err=True)
        return None
    ok, rc, negotiated = supports_exchange(device, sdk_version)
    adapter["mlir"], adapter["ir_version"] = ok, negotiated
    if verbose:
        extra = f", IR version {format_driver_version(negotiated)}" if negotiated else ""
        click.echo(f"  {adapter['description']}: feature 70 -> {hrs(rc)}{extra}", err=True)
    if not ok:
        release(device)
        return None
    return device


def mlir_exchange(
    device: ctypes.c_void_p, sdk_version: int, ir_version: int
) -> tuple[bytes | None, int]:
    """Fetch the subgraph-declaration payload.

    Uses the two-call size-then-fill protocol.

    Args:
        device: The device pointer.
        sdk_version: The redist's SDK version, which picks the ABI shape.
        ir_version: The IR version feature 70 negotiated, for the 721 shape.

    Returns:
        ``(payload, 0)``, or ``(None, hresult)`` when the exchange failed.
    """
    cfs = check_feature_support(device)

    def build(out_ptr: ctypes.c_void_p | None, size_ptr: Any) -> ctypes.Structure:
        if sdk_version >= 721:
            return FeatureDataMLIRExchange721(
                Type=MLIR_EXCHANGE_SUBGRAPH_DECLARATION,
                IRVersion=ir_version,
                pOutputData=out_ptr,
                OutputDataSizeInBytes=size_ptr,
            )
        return FeatureDataMLIRExchange720(
            MlirInterface=GUID_SUBGRAPH_DECLARATION_REQUEST,
            pOutputData=out_ptr,
            OutputDataSizeInBytes=size_ptr,
        )

    size = ctypes.c_size_t(0)
    first = build(None, ctypes.pointer(size))
    rc = cfs(FEATURE_MLIR_EXCHANGE, ctypes.byref(first), ctypes.sizeof(first))
    if rc < 0:
        return None, rc
    if size.value == 0:
        return b"", 0

    buf = ctypes.create_string_buffer(size.value)
    written = ctypes.c_size_t(size.value)
    second = build(ctypes.cast(buf, ctypes.c_void_p), ctypes.pointer(written))
    rc = cfs(FEATURE_MLIR_EXCHANGE, ctypes.byref(second), ctypes.sizeof(second))
    if rc < 0:
        return None, rc
    if written.value > size.value:
        raise click.ClickException(
            f"driver overran the response buffer: {written.value} > {size.value}"
        )
    return buf.raw[: written.value], 0


#: A line break in any form a Windows producer may emit: CRLF, a run of CRs before LF
#: (CRLF written again through a text-mode stream), or a bare CR.
_LINE_BREAK_RE = re.compile(rb"\r+\n|\r")


def normalize_text_payload(data: bytes) -> bytes:
    """Normalise a text payload to LF line endings, without its C string terminator.

    The driver writes Windows line endings and reports its size including the NUL
    that terminates the string. Both are normalised once, before anything else
    reads the payload, so the counts, ``patterns.mlir`` and ``patterns.json`` all
    describe exactly the same bytes. Only a text payload is normalised: rewriting
    line endings inside MLIR bytecode would corrupt it.

    Normalising is idempotent -- the result contains no CR at all -- so applying it
    to already-normalised text is a no-op.

    Args:
        data: The text payload as the exchange returned it.

    Returns:
        The payload with trailing NULs removed and every line break turned into LF.
    """
    # TODO(tests): CRLF, CR-runs before LF, bare CRs and a trailing NUL are not tested
    # directly, and nothing asserts normalize(normalize(x)) == normalize(x).
    return _LINE_BREAK_RE.sub(b"\n", data.rstrip(b"\x00"))


def is_mlir_bytecode(data: bytes) -> bool:
    """Report whether *data* is MLIR bytecode rather than text.

    Args:
        data: The payload.

    Returns:
        True when it carries the bytecode magic (utilities/MlirEncoding.h).
    """
    return data.startswith(b"ML\xefR")


# ------------------------------------------------------------- counting patterns
# A pattern is one cgc_pattern.pattern declaration. A rule is one flat match
# alternative after any_of expansion: an `any_of { all_of {..} all_of {..} }` block
# contributes one rule per all_of branch, and several blocks in one pattern multiply.
# Do not confuse this with apply_native_constraint "cgc_is_any_of", which is an
# op-family whitelist and no multiplier at all -- searching for the substring any_of
# conflates the two.

PATTERN_DECL_RE = re.compile(rb"cgc_pattern\.pattern\s+@([A-Za-z0-9_.]+)")
ANY_OF_RE = re.compile(rb"(?<![A-Za-z_])any_of\s*\{")
#: A brace, or an ``all_of`` keyword together with the brace that opens its block.
BRACE_TOKEN_RE = re.compile(rb"(?<![A-Za-z_])all_of\s*\{|[{}]")
COMMENT_RE = re.compile(rb"//[^\n]*")
#: A double-quoted string, or a comment. The string alternative comes first so that a
#: "//" inside one -- a URL, a path, a loc() reference -- is not read as a comment,
#: which would blank the rest of that line, closing brace included.
STRING_OR_COMMENT_RE = re.compile(rb'"(?:[^"\\\n]|\\.)*"|//[^\n]*')
BENEFIT_RE = re.compile(rb"benefit\((\d+)\)")
ATTRIBUTES_TAIL_RE = re.compile(rb"attributes\s*$")

# The five kinds of claim, in the order they are reported: most claimed first. Only a
# kernel runs work on the driver; the rest arrange the graph so that a kernel can match.
# The kind is read off the body, never the name -- a name ending in _fusion may be a
# rewrite.
KIND_ORDER = ("kernel", "fusion", "hint", "mark", "rewrite")
KIND_BLURB = {
    "kernel": "declares a jitFunction -- the only kind that claims work",
    "fusion": "collapses a marked cluster into one native op",
    "hint": "anchors a cluster and sets the order clusters grow in",
    "mark": "tags an op so a fusion root can absorb it later",
    "rewrite": "graph surgery, so that other patterns can match",
}


def _classify(body: bytes) -> str:
    """Read a pattern's kind off its body, never its name.

    Args:
        body: The pattern body, braces included.

    Returns:
        One of :data:`KIND_ORDER`.
    """
    if b"jitFunction" in body:
        return "kernel"
    if b"subgraph_rewrite_desc" in body:
        return "fusion"
    if b"cgc_add_pattern_cluster_hint" in body:
        return "hint"
    if b"cgc_mark_pattern_cluster_op" in body:
        return "mark"
    return "rewrite"


def _balanced(text: bytes, open_brace: int) -> int:
    """Find the ``}`` closing the ``{`` at *open_brace*.

    Args:
        text: The haystack.
        open_brace: Index of the opening brace.

    Returns:
        Index of the matching close brace, or the last index when unbalanced.
    """
    depth = 0
    for token in BRACE_TOKEN_RE.finditer(text, open_brace):
        depth += -1 if token.group() == b"}" else 1
        if depth == 0:
            return token.start()
    return len(text) - 1


def _any_of_branches(body: bytes, open_brace: int) -> int:
    """Count ``all_of`` branches directly inside an ``any_of`` block.

    Args:
        body: The pattern body.
        open_brace: Index of the ``any_of`` block's opening brace.

    Returns:
        The branch count, at least 1.
    """
    depth = branches = 0
    for token in BRACE_TOKEN_RE.finditer(body, open_brace):
        if token.group() == b"}":
            depth -= 1
            if depth == 0:
                break
        else:
            depth += 1
            if depth == 2 and token.group() != b"{":  # an all_of block
                branches += 1
    return branches or 1


def _blank_comments(data: bytes) -> bytes:
    """Replace every ``//`` comment outside a string with spaces of the same length.

    Deleting comments would shift every offset after them. Blanking keeps offsets
    valid, so a span found here also slices the original bytes -- comments intact,
    for display -- and a brace inside a comment can no longer unbalance the match.

    Args:
        data: The MLIR text payload.

    Returns:
        The same bytes with comment text replaced by spaces.
    """
    return STRING_OR_COMMENT_RE.sub(
        lambda m: b" " * len(m.group()) if m.group().startswith(b"//") else m.group(), data
    )


def _pattern_spans(text: bytes) -> Iterator[tuple[re.Match[bytes], int, int]]:
    """Yield each declaration that has a body, with that body's bounds.

    Some patterns declare an ``attributes { depends = .. }`` block between the name and
    the body. It is skipped: read as the body, it holds no jitFunction, and a kernel
    would be filed under rewrite.

    Args:
        text: Comment-blanked MLIR text.

    Yields:
        ``(declaration match, open brace, close brace)``.
    """
    declarations = list(PATTERN_DECL_RE.finditer(text))
    for index, m in enumerate(declarations):
        # The body has to start before the next declaration does. Without that bound a
        # declaration with no body of its own adopts the next one's, and both are then
        # reported, the first carrying two declarations' text.
        limit = declarations[index + 1].start() if index + 1 < len(declarations) else len(text)
        try:
            i = text.index(b"{", m.end(), limit)
            if ATTRIBUTES_TAIL_RE.search(text[m.end() : i]):
                i = text.index(b"{", _balanced(text, i) + 1, limit)
        except ValueError:
            continue  # a declaration with no body: nothing to read
        yield m, i, _balanced(text, i)


def pattern_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bucket records by kind, in :data:`KIND_ORDER`, with each bucket's totals.

    A kind nothing was declared for is left out rather than reported as zero.

    Args:
        records: The parsed patterns.

    Returns:
        One entry per non-empty kind.
    """
    groups: list[dict[str, Any]] = []
    for kind in KIND_ORDER:
        members = [r for r in records if r["kind"] == kind]
        if not members:
            continue
        members.sort(key=lambda r: (-(r["benefit"] or 0), r["name"]))
        groups.append(
            {
                "kind": kind,
                "members": members,
                "patterns": len(members),
                "rules": sum(r["rules"] for r in members),
            }
        )
    return groups


def format_pattern_listing(records: list[dict[str, Any]]) -> list[str]:
    """Format the ``-v`` listing: every pattern, grouped by what it claims.

    Args:
        records: The records from :func:`split_patterns`.

    Returns:
        The lines to print.
    """
    lines: list[str] = []
    for group in pattern_groups(records):
        lines.append(
            f"{group['kind']:<8} {group['patterns']:2d} patterns, "
            f"{group['rules']:2d} rules   {KIND_BLURB[group['kind']]}"
        )
        for r in group["members"]:
            benefit = "   -" if r["benefit"] is None else f"{r['benefit']:4d}"
            extra = f"  {r['rules']} rules" if r["rules"] > 1 else ""
            lines.append(f"  {benefit}  {r['short_name']}{extra}")
    return lines


#: A source-file boundary in a merged dump. Anchored on a filename with an
#: extension: a bare "// from" also occurs in prose ("// from outside the root's
#: own chain"), and must not open a section.
SECTION_RE = re.compile(rb"^[ \t]*//[ \t]*from[ \t]+(\S+\.(?:pdll|mlir))[ \t]*$", re.MULTILINE)

#: The kernel a pattern declares. Driver dumps write ``#cgc.string<"Name">``; the
#: plain string form is accepted as well.
KERNEL_RE = re.compile(rb'jitFunction\s*=\s*(?:#cgc\.string<)?"([^"]+)"')

#: Identifies the layout of ``patterns.json`` for readers such as the atlas page.
PATTERNS_JSON_FORMAT = "winml-cgc-patterns/1"


# TODO(tests): the tests for the prose "// from" anchor and the kernel regex would
# still pass with that guard removed -- they need inputs that fail without it.
def split_patterns(data: bytes) -> list[dict[str, Any]]:
    """Split a dump into one record per ``cgc_pattern.pattern`` declaration.

    Each record carries the counts (kind, benefit, rules) and what a reader needs
    beyond them: the source file the pattern came from, the kernel it declares, and
    the pattern's text with its comments.

    Args:
        data: The MLIR text payload, already passed through
            :func:`normalize_text_payload`.

    Returns:
        Records in declaration order, each with ``index``, ``name``,
        ``short_name``, ``kind``, ``benefit``, ``rules``, ``source``, ``kernel``,
        ``lines`` and ``mlir``.
    """
    text = _blank_comments(data)
    sections = [
        (m.start(), m.group(1).decode("utf-8", "replace")) for m in SECTION_RE.finditer(data)
    ]
    records: list[dict[str, Any]] = []
    for index, (m, open_brace, close_brace) in enumerate(_pattern_spans(text), 1):
        head, body = text[m.end() : open_brace], text[open_brace:close_brace]
        name = m.group(1).decode("ascii", "replace")
        benefit = BENEFIT_RE.search(head)
        rules = 1
        for a in ANY_OF_RE.finditer(body):
            rules *= _any_of_branches(body, a.end() - 1)
        kernel = KERNEL_RE.search(body)
        raw = data[m.start() : close_brace + 1]
        records.append(
            {
                "index": index,
                "name": name,
                "short_name": name.split(".")[-1],  # the vendor prefix carries nothing
                "kind": _classify(body),
                "benefit": int(benefit.group(1)) if benefit else None,
                "rules": rules,
                "source": next((s for pos, s in reversed(sections) if pos < m.start()), None),
                "kernel": kernel.group(1).decode("utf-8", "replace") if kernel else None,
                "lines": raw.count(b"\n") + 1,
                "mlir": raw.decode("utf-8", "replace"),
            }
        )
    return records


def patterns_document(
    data: bytes, adapter: Adapter, sdk: int | None, redist: Path | None
) -> dict[str, Any]:
    """Build the ``patterns.json`` document for one dump.

    Args:
        data: The MLIR text payload, already passed through
            :func:`normalize_text_payload`.
        adapter: The adapter the dump describes.
        sdk: The redist's SDK version.
        redist: The redist directory.

    Returns:
        The document: format tag, provenance, summary, per-kind and per-source
        grouping, and every pattern.
    """
    # TODO(tests): summary.sources, meta.driver_version/redist/encoding and the
    # "(unattributed)" source group are not asserted by any test.
    patterns = split_patterns(data)
    kernels = [p["kernel"] for p in patterns if p["kernel"]]
    sources: dict[str, list[int]] = {}
    for p in patterns:
        sources.setdefault(p["source"] or "(unattributed)", []).append(p["index"])
    return {
        "format": PATTERNS_JSON_FORMAT,
        "meta": {
            "adapter": adapter["description"],
            "driver_version": adapter["driver_version"],
            "sdk_version": sdk,
            "redist": str(redist) if redist else None,
            "encoding": "text",
            "bytes": len(data),
        },
        "summary": {
            "patterns": len(patterns),
            "rules": sum(p["rules"] for p in patterns),
            "kernel_patterns": len(kernels),
            "distinct_kernels": len(set(kernels)),
            "sources": len(sources),
            "top_benefit": max((p["benefit"] or 0 for p in patterns), default=0),
        },
        "kinds": {g["kind"]: g["patterns"] for g in pattern_groups(patterns)},
        "sources": [{"name": name, "patterns": ids} for name, ids in sources.items()],
        "patterns": patterns,
    }


def write_patterns_json(path: Path, document: dict[str, Any]) -> None:
    """Write ``patterns.json``.

    Args:
        path: The file to write.
        document: The document from :func:`patterns_document`.
    """
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def adapter_type(a: Adapter) -> str:
    """Classify an adapter for the TYPE column.

    Args:
        a: The adapter record.

    Returns:
        ``hardware``, ``integrated`` or ``software``.
    """
    if not a["is_hardware"]:
        return "software"
    return "integrated" if a["is_integrated"] else "hardware"


def adapter_attrs(a: Adapter) -> str:
    """Render the DXCore attributes an adapter advertises.

    Args:
        a: The adapter record.

    Returns:
        A space-separated subset of ``ML CC GFX``, or ``-``.
    """
    return (
        " ".join(
            k
            for k, v in (
                ("ML", a["generic_ml"]),
                ("CC", a["core_compute"]),
                ("GFX", a["d3d12_graphics"]),
            )
            if v
        )
        or "-"
    )


def print_adapters(adapters: list[Adapter], stream: Any = None) -> None:
    """Print the adapter table.

    Args:
        adapters: Every adapter found.
        stream: Destination, defaulting to stdout.
    """
    out = sys.stdout if stream is None else stream
    hdr = f"{'IDX':<3} {'ADAPTER':<30} {'DRIVER':<17} {'TYPE':<10} {'ATTRIBUTES':<11} MLIR"
    out.write(hdr + "\n")
    out.write("-" * len(hdr) + "\n")
    for a in adapters:
        mark = {True: "yes", False: "no", None: "?"}[a["mlir"]]
        out.write(
            f"{a['index']:<3} {a['description'][:30]:<30} {a['driver_version']:<17} "
            f"{adapter_type(a):<10} {adapter_attrs(a):<11} {mark}\n"
        )


def write_metadata(
    path: Path,
    adapter: Adapter,
    status: str,
    encoding: str,
    sdk: int | None,
    redist: Path | None,
    counts: tuple[int, int] | None,
    received_bytes: int | None = None,
) -> None:
    """Write ``metadata.txt`` beside a dump.

    The first six keys are the ones ``dxcgc-dump-driver-patterns`` writes, unchanged so
    existing readers keep working; the rest record which redist and ABI produced the
    dump, and how the payload that was stored differs from the one received -- a text
    dump is stored with LF line endings. Additive only.

    Args:
        path: The file to write.
        adapter: The adapter the dump describes.
        status: ``dumped``, ``unsupported`` or ``empty``.
        encoding: ``text``, ``bytecode`` or ``none``.
        sdk: The redist's SDK version.
        redist: The redist directory.
        counts: ``(patterns, rules)``, when known.
        received_bytes: The size of the payload as the exchange returned it,
            terminator included, when there was one.
    """
    text_file = "patterns.mlir" if status == "dumped" and encoding == "text" else ""
    sdk_version = str(sdk) if sdk else ""
    patterns, rules = (str(counts[0]), str(counts[1])) if counts else ("", "")
    lines: list[tuple[str, str]] = [
        ("adapter", adapter["description"]),
        ("driver_version", adapter["driver_version"]),
        ("status", status),
        ("received_encoding", encoding),
        ("text_file", text_file),
        ("bytecode_file", "patterns.mlirbc" if encoding == "bytecode" else ""),
        ("abi", sdk_version),
        ("sdk_version", sdk_version),
        ("redist", str(redist) if redist else ""),
        ("patterns", patterns),
        ("rules", rules),
        ("received_bytes", "" if received_bytes is None else str(received_bytes)),
        ("line_endings", "lf" if text_file else ""),
    ]
    path.write_text("".join(f"{k}={v}\n" for k, v in lines), encoding="utf-8", newline="\n")


DUMP_FILES = ("patterns.mlir", "patterns.mlirbc", "patterns.json", "metadata.txt")


def clear_dump(dest: Path) -> None:
    """Remove a previous dump's files from *dest*.

    ``--overwrite`` REPLACES a dump. Without this, re-dumping a driver that now answers
    in text leaves the previous run's ``patterns.mlirbc`` beside the new
    ``patterns.mlir``, and the directory describes two different dumps at once.

    Args:
        dest: The dump directory.
    """
    for name in DUMP_FILES:
        (dest / name).unlink(missing_ok=True)


def dump_dir(adapter: Adapter) -> Path:
    """Return the directory a dump for *adapter* is filed under.

    Args:
        adapter: The adapter record.

    Returns:
        ``patterns/<adapter-slug>/<driver-version>``, relative to the working directory.
    """
    # Both parts fall back to a placeholder: an empty one would be joined away, and
    # the dump would lose a level of the layout that keeps captures apart.
    version = str(adapter["driver_version"]) or "unknown-version"
    return Path("patterns") / adapter_slug(adapter["description"]) / version


def redist_line(redist: Path | None, sdk: int | None) -> str:
    """Render the ``redist:`` line reported on every run.

    Which core answered decides what the rest of the output means.

    Args:
        redist: The resolved redist directory, or None.
        sdk: Its SDK version, or None.

    Returns:
        The line.
    """
    if redist is None:
        return "redist: none -- MLIR support unknown (run with -v to see where cgc looked)"
    return f"redist: {redist} (SDK {sdk}, {host_machine()})"


#: The viewer for a patterns.json dump, shipped as package data beside this module.
ATLAS_PAGE = "cgc-pattern-atlas.html"
ATLAS_DIR = "assets"


def atlas_page() -> Path:
    """Locate the pattern atlas page that ships with this package.

    Asking the package rather than walking the filesystem is what makes ``--open``
    work the same way from a wheel, from an editable install and from a checkout.

    Returns:
        The page on disk.

    Raises:
        click.ClickException: When the package was installed without its assets.
    """
    page = Path(str(resources.files(commands).joinpath(ATLAS_DIR, ATLAS_PAGE)))
    if not page.is_file():
        raise click.ClickException(
            f"{ATLAS_DIR}/{ATLAS_PAGE} is missing from this install of winml-cli, so "
            f"there is no page to open; the dump itself is still written"
        )
    return page


def read_patterns_json(document: Path) -> dict[str, Any]:
    """Read a ``patterns.json``, refusing anything that is not one.

    Every reason a named file cannot be shown is decided here, so ``--open`` reports
    a mistyped or unrelated path the way it reports any other bad argument.

    Args:
        document: The file named by ``--open``.

    Returns:
        The document.

    Raises:
        click.UsageError: When it is missing, unreadable, not JSON, or not a
            :data:`PATTERNS_JSON_FORMAT` document.
    """
    if not document.is_file():
        raise click.UsageError(f"--open: {document} is not a file")
    try:
        raw = json.loads(document.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise click.UsageError(f"--open: {document} could not be read as JSON: {e}") from e
    # Anything that is not a mapping fails the format check below, and "meta" has to
    # be one too: open_atlas labels the page through it.
    data: dict[str, Any] = cast("dict[str, Any]", raw) if isinstance(raw, dict) else {}
    if data.get("format") != PATTERNS_JSON_FORMAT or not isinstance(data.get("meta", {}), dict):
        raise click.UsageError(
            f"--open: {document} is not a {PATTERNS_JSON_FORMAT} document; pass a "
            f"patterns.json written by --dump"
        )
    return data


def open_atlas(document: Path | None) -> None:
    """Open the pattern atlas in a browser, showing *document* when there is one.

    The page reads a file the viewer picks, and a page opened from disk may not read
    one beside it, so a dump is embedded into a throwaway copy of the page instead.

    Args:
        document: A ``patterns.json`` to show, or None to open the empty page.

    Raises:
        click.UsageError: When *document* is not a patterns.json.
    """
    page = atlas_page()
    if document is not None:
        data = read_patterns_json(document)
        data.setdefault("meta", {})["label"] = str(document)
        # Every "<" is escaped, so no pattern's own text can close the element early.
        payload = json.dumps(data).replace("<", "\\u003c")
        seeded = page.read_text(encoding="utf-8").replace(
            '<script type="application/json" id="seed"></script>',
            f'<script type="application/json" id="seed">{payload}</script>',
        )
        # Named after the dump, not a fresh temp file per run: the browser reads the
        # page after this process is gone, so it cannot be cleaned up on the way out,
        # and two dumps opened together must not land on the same name.
        stem = adapter_slug(f"{document.parent.parent.name}-{document.parent.name}")
        page = Path(tempfile.gettempdir()) / f"winml-{stem}-{ATLAS_PAGE}"
        page.write_text(seeded, encoding="utf-8", newline="\n")
    click.echo(f"opening {page}")
    webbrowser.open(page.as_uri())


@dataclass(kw_only=True)
class Options:
    """The resolved flags for one ``adapter`` invocation.

    Attributes:
        adapter: The ``-a`` selector.
        verbose: The ``-v`` flag.
        d3d12_dir: The ``--d3d12-dir`` value.
        dump: The ``--dump`` flag.
        overwrite: The ``--overwrite`` flag.
        open_atlas: The ``--open`` value: None when it was not passed, an empty
            string for this run's own dump, else the ``patterns.json`` to show.
    """

    adapter: str = ""
    verbose: bool = False
    d3d12_dir: str = ""
    dump: bool = False
    overwrite: bool = False
    open_atlas: str | None = None


# TODO(tests): nothing drives report_patterns end to end: the dumped, unsupported
# and empty branches, the patterns.json write, --overwrite, and the refusal.
def report_patterns(opts: Options, adapter: Adapter, redist: Path, sdk: int) -> Path | None:
    """Report, and optionally dump, what one adapter's driver declares.

    Args:
        opts: The flags.
        adapter: The selected adapter.
        redist: The resolved redist.
        sdk: Its SDK version.

    Returns:
        The ``patterns.json`` this run wrote, for :func:`open_atlas`, or None.
    """
    dest = dump_dir(adapter)
    if opts.dump:
        # An empty directory, left by a run that died between mkdir and the writes, is
        # reused; guard_output only refuses one that holds files.
        cli_utils.guard_output(dest, opts.overwrite, label="Dump")

    desc = adapter["description"]
    status, encoding, received, data, doc = "unsupported", "none", None, b"", None
    device = probe_mlir_support(adapter, redist, sdk, verbose=opts.verbose)
    if device is None:
        click.echo(f"{desc}: driver does not implement MLIR programs")
    else:
        try:
            payload, rc = mlir_exchange(device, sdk, adapter["ir_version"])
        finally:
            release(device)
        if payload is None:
            # The driver said it implements the exchange and then refused it.
            raise click.ClickException(
                f"MLIR exchange failed after the driver claimed support: {hrs(rc)}"
            )
        received = len(payload)
        bytecode = is_mlir_bytecode(payload)
        # Normalised once, here, so a payload that is only its terminator is empty.
        data = payload if bytecode else normalize_text_payload(payload)
        if not data:
            status = "empty"
            click.echo(f"{desc}: driver returned an empty subgraph declaration")
        elif bytecode:
            status, encoding = "dumped", "bytecode"
            click.echo(
                f"{desc}: {len(data)} bytes of MLIR bytecode; rendering it "
                f"as text needs cgc-opt, so cgc cannot count it"
            )
        else:
            status, encoding = "dumped", "text"
            doc = patterns_document(data, adapter, sdk, redist)
            click.echo(
                f"{desc}: {doc['summary']['patterns']} patterns / {doc['summary']['rules']} rules"
            )
            if opts.verbose:
                click.echo("")
                for line in format_pattern_listing(doc["patterns"]):
                    click.echo(line)

    if not opts.dump:
        return None
    dest.mkdir(parents=True, exist_ok=True)
    clear_dump(dest)
    if status == "dumped":
        name = "patterns.mlir" if encoding == "text" else "patterns.mlirbc"
        (dest / name).write_bytes(data)
        click.echo(f"wrote {dest / name} ({len(data)} bytes)")
    counts = None
    if doc is not None:
        # Bytecode cannot be split without the real MLIR parser, so only a text dump
        # gets the structured copy.
        write_patterns_json(dest / "patterns.json", doc)
        click.echo(f"wrote {dest / 'patterns.json'}")
        counts = (doc["summary"]["patterns"], doc["summary"]["rules"])
    write_metadata(dest / "metadata.txt", adapter, status, encoding, sdk, redist, counts, received)
    click.echo(f"wrote {dest / 'metadata.txt'}")
    return dest / "patterns.json" if doc is not None else None


# TODO(tests): the Click exceptions raised on these paths are untested as a caller
# sees them -- exit 1 for "could not ask", exit 2 for bad arguments.
def resolve_run(opts: Options) -> tuple[Path | None, int | None, tuple[str, Path] | None]:
    """Resolve the redist both subcommands answer through, and report which it is.

    Args:
        opts: The flags.

    Returns:
        ``(redist, sdk_version, named)``; *named* is the source the caller named it
        by, when they named one.

    Raises:
        click.ClickException: When a named redist cannot be used.
    """
    redist, sdk, tried = resolve_redist(opts.d3d12_dir)
    named = _named_redist(opts.d3d12_dir)
    if redist is None and (named or opts.verbose):
        # Asked for a specific redist, or asked to be told: an unusable one is fatal.
        click.echo(redist_failure(tried), err=True)
        if named:
            raise click.ClickException(
                f"{named[0]} named a redist that cannot be used; refusing to answer "
                f"about a different one."
            )
    # Which core answered decides what every other line means, so it is always
    # reported. When there is none the line says so, and the MLIR column reads ?
    # rather than no: the question cannot be asked, so it is left unanswered.
    click.echo(redist_line(redist, sdk))
    return redist, sdk, named


# TODO(tests): the Click exceptions raised on these paths are untested as a caller
# sees them -- exit 1 for "could not ask", exit 2 for bad arguments.
def run_adapters(opts: Options) -> None:
    """Run the ``adapters`` subcommand: the table of every D3D12 adapter.

    Args:
        opts: The flags.
    """
    adapters = list_adapters()
    if not adapters:
        click.echo("no D3D12 adapters found")
        return

    try:
        redist, sdk, named = resolve_run(opts)
        if redist is not None:
            try:
                for a in adapters:
                    release(probe_mlir_support(a, redist, sdk or 0, verbose=opts.verbose))
            except RedistUnusable as exc:
                # D3D12 cannot use this redist at all. One the caller named must be
                # usable, so that fails; otherwise the listing stands, MLIR unanswered.
                if named:
                    raise
                for a in adapters:
                    a["mlir"] = None
                click.echo(f"cgc: {exc.format_message()}", err=True)
        print_adapters(adapters)
    finally:
        close_adapters(adapters)


def run_patterns(opts: Options) -> None:
    """Run the ``patterns`` subcommand: what a driver declares it can match.

    With no ``-a`` every adapter is asked, so a machine's answer is one command.

    Args:
        opts: The flags.

    Raises:
        click.UsageError: When ``--dump`` did not name its adapter, or ``--open``
            named something that is not a patterns.json.
        click.ClickException: When an adapter could not be asked. Sweeping every
            adapter, the others are still reported first.
    """
    if opts.dump and opts.open_atlas:
        # Two dumps to show, this run's and a named one. Silently preferring either
        # throws away what the caller asked for.
        raise click.UsageError(
            "--open takes a patterns.json only without --dump; with --dump it shows "
            "the dump this run writes"
        )
    if opts.open_atlas is not None and not opts.dump:
        # Nothing of this run's own to show: a named dump, or the empty page to drop
        # a file on. Either way no driver is asked.
        open_atlas(Path(opts.open_atlas) if opts.open_atlas else None)
        return
    if opts.dump and not opts.adapter:
        # Required rather than defaulted: a dump is filed under a slug and a driver
        # version, and dumping whichever adapter happened to be first writes a
        # correct-looking directory for the wrong device.
        raise click.UsageError("--dump needs an adapter: name one with -a <substring|index>")
    adapters = list_adapters()
    if not adapters:
        click.echo("no D3D12 adapters found")
        return

    try:
        redist, sdk, _ = resolve_run(opts)
        if redist is None or sdk is None:
            # One condition for the whole run: nothing was asked, so no adapter is
            # named as having failed.
            raise click.ClickException(
                "cgc patterns needs an Agility SDK redist; none was found "
                "(run with -v to see where cgc looked)"
            )
        targets = [select_adapter(adapters, opts.adapter)] if opts.adapter else adapters
        written: Path | None = None
        unanswered: list[str] = []
        for adapter in targets:
            try:
                written = report_patterns(opts, adapter, redist, sdk) or written
            except RedistUnusable:  # noqa: PERF203 - next to a device, try costs nothing
                raise  # no adapter can be asked through this redist
            except click.ClickException as exc:
                # Sweeping every adapter: one driver that cannot answer must not cost
                # the others their line. The run still fails, once, at the end.
                if len(targets) == 1:
                    raise
                click.echo(f"cgc: {adapter['description']}: {exc.format_message()}", err=True)
                unanswered.append(adapter["description"])
    finally:
        close_adapters(adapters)
    if opts.open_atlas is not None:
        if written is None:
            click.echo("cgc: no patterns.json was written, so the atlas opens empty", err=True)
        open_atlas(written)
    if unanswered:
        raise click.ClickException(f"could not ask: {', '.join(unanswered)}")


# A plain Group: the root LazyGroup already instruments `cgc` itself, and an
# ActionGroup here would instrument the subcommand as well, so every run would count
# twice where every other command counts once.
@click.group(
    name="cgc",
    invoke_without_command=True,
)
@click.pass_context
def cgc(ctx: click.Context) -> None:
    """Inspect D3D12 adapters and the MLIR patterns their drivers declare.

    Answers two questions on any Windows machine: which D3D12 adapters are here
    (`adapters`), and what patterns does a driver declare it can match (`patterns`).

    Querying MLIR support or patterns needs an Agility SDK D3D12Core.dll of SDK 720 or
    newer; point at one with --d3d12-dir or $WINML_D3D12_DIR. Listing adapters needs
    neither a redist nor a build. `patterns --open` without --dump only opens the
    viewer or an existing dump and needs neither a GPU nor a redist.
    """
    # Mirror the root `winml` group: a bare invocation prints help and succeeds,
    # rather than Click's default usage error.
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(0)


@cgc.command("adapters")
@click.option(
    "--d3d12-dir",
    default="",
    metavar="PATH",
    help="Directory holding D3D12Core.dll ($WINML_D3D12_DIR, $D3D12_DIR).",
)
@cli_utils.verbosity_options()
@click.pass_context
def adapters_cmd(ctx: click.Context, d3d12_dir: str, verbose: int, quiet: bool) -> None:
    """List DXCore ML or core-compute adapters and their MLIR support.

    Lists generic-ML adapters, or core-compute adapters if the first list is empty.
    Needs nothing beyond Windows. The MLIR column is answered through the Agility
    SDK redist named on the redist: line -- without one it reads ?, since the question
    cannot be asked. `winml cgc patterns` then reports what a driver declares.

    \b
    Examples:
        # Every adapter, with its driver version and MLIR support
        winml cgc adapters

    \b
        # Show the raw capability HRESULT behind each answer
        winml cgc adapters -v

    \b
        # Ask through a particular Agility SDK redist
        winml cgc adapters --d3d12-dir <dir containing D3D12Core.dll>
    """  # noqa: D301 - the \b escapes must become backspace characters
    verbose, quiet = cli_utils.resolve_verbosity(ctx, verbose, quiet)
    configure_logging(verbosity=verbose, quiet=quiet)
    opts = Options(verbose=bool(verbose), d3d12_dir=d3d12_dir)
    logger.debug("cgc adapters: %s", vars(opts))
    run_adapters(opts)


@cgc.command("patterns")
@click.option(
    "-a",
    "--adapter",
    "adapter_sel",
    default="",
    metavar="SUBSTRING|INDEX",
    help="The adapter to ask, by description substring or index; --dump needs it. "
    "Without it, every adapter is asked.",
)
@click.option(
    "--d3d12-dir",
    default="",
    metavar="PATH",
    help="Directory holding D3D12Core.dll ($WINML_D3D12_DIR, $D3D12_DIR).",
)
@click.option(
    "--dump",
    is_flag=True,
    default=False,
    help="Write a dump under ./patterns (needs -a). Non-empty text produces patterns.mlir and "
    "patterns.json; bytecode produces patterns.mlirbc. Successful queries also write metadata.txt.",
)
@cli_utils.overwrite_option()
@click.option(
    "--open",
    "open_atlas",
    is_flag=False,
    flag_value="",
    default=None,
    metavar="[PATTERNS.JSON]",
    help="Open the pattern atlas. Without --dump, show a named patterns.json or an empty "
    "page without querying drivers. With --dump, show this run's dump when available.",
)
@cli_utils.verbosity_options()
@click.pass_context
def patterns_cmd(
    ctx: click.Context,
    adapter_sel: str,
    d3d12_dir: str,
    dump: bool,
    overwrite: bool,
    open_atlas: str | None,
    verbose: int,
    quiet: bool,
) -> None:
    """Report the MLIR patterns a driver declares it can match.

    Querying a driver needs an Agility SDK D3D12Core.dll of SDK 720 or newer: name one
    with --d3d12-dir, or set $WINML_D3D12_DIR. With no -a every enumerated adapter is
    asked. --dump writes under the adapter description and driver version; adapters
    with the same normalized description share a destination.

    --open without --dump opens an existing JSON dump or an empty atlas without
    querying drivers, so it needs neither a GPU nor a redist.

    stdout carries results and stderr diagnostics. A multi-adapter run can print
    partial results before exiting 1. Exit 0 includes unsupported or empty answers;
    exit 1 means the operation could not complete; exit 2 is bad arguments.

    \b
    Examples:
        # What every adapter's driver declares
        winml cgc patterns

    \b
        # One adapter, by substring or index
        winml cgc patterns -a nvidia

    \b
        # Every pattern it declares, grouped by kind
        winml cgc patterns -a nvidia -v

    \b
        # Keep the declaration on disk, and look at it
        winml cgc patterns -a nvidia --dump --open

    \b
        # Look at a dump taken earlier, with no hardware involved
        winml cgc patterns --open patterns/nvidia.../32.0.16.3004/patterns.json

    \b
        # Ask a different redist the same question
        winml cgc patterns -a nvidia --d3d12-dir <dir>
    """  # noqa: D301 - the \b escapes must become backspace characters
    # Merge top-level -v/-q with subcommand-level flags so either position works.
    verbose, quiet = cli_utils.resolve_verbosity(ctx, verbose, quiet)

    # Standard verbosity contract: stderr-only logs in the shared format. `-v`
    # here keeps its cgc-specific second job of adding the raw capability
    # HRESULTs and the full pattern listing.
    configure_logging(verbosity=verbose, quiet=quiet)

    opts = Options(
        adapter=adapter_sel,
        verbose=bool(verbose),
        d3d12_dir=d3d12_dir,
        dump=dump,
        overwrite=overwrite,
        open_atlas=open_atlas,
    )
    logger.debug("cgc patterns: %s", vars(opts))
    run_patterns(opts)
