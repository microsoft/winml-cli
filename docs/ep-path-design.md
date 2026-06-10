# EP_PATH: Unified Execution Provider Discovery

> **2026-05-29:** the env var was renamed from `MODELKIT_EP_PATH` to `WINMLCLI_EP_PATH`. The old name is no longer recognized.

Status: implemented in commits `17b81c9a..eab52093` (and ABC refactor `ef9a5bdd`), as of 2026-05-08. Targets ORT 1.24.x (the version pinned in this repo) and 1.25.x. Companion to [`docs/ep-sideloading-research.md`](ep-sideloading-research.md), which inventories what each IHV actually ships today; this document specifies the discovery mechanism that consumes those origins.

**See also**: [`docs/ep-path-msix-source.md`](ep-path-msix-source.md) extends this design with `MsixPackageSource` (version-pinned MSIX EP discovery via `Windows.Management.Deployment.PackageManager`), `is_compatible()` API on every `EpSource`, the `discover_all_eps()` function for comprehensive shadowing discovery, and the comprehensive `winml sys --list-ep` inventory.

## Executive summary

- The current `EP_PLUGIN_REGISTRY` in [`src/winml/modelkit/winml.py:28`](../src/winml/modelkit/winml.py) is a `dict[str, tuple[pypi_pkg, relative_path]]`. It only resolves PyPI-installed plugin EPs. Real-world EPs ship from at least four origins (PyPI wheels, Microsoft Store / Windows Update MSIX, third-party EXE installers, manually unzipped GitHub release artifacts). A single tuple cannot encode all of them.
- This document proposes replacing the dict with two cooperating structures: an ordered list `EP_PATH: list[EpSource]` (path-list, analogous to `PATH`) plus a small canonical-name table `EP_DLL_NAMES: dict[ep_name, list[dll_filename]]`. Each `EpSource` is a typed tagged-union covering the four origins; resolution is decoupled from path layout.
- Discovery walks `EP_PATH` in order; first match per EP name wins. Origin-specific resolvers handle the irregularities (PyPI distribution lookup via `importlib.metadata`, MSIX lookup via the WinAppSDK `ExecutionProviderCatalog` runtime API, filesystem glob for installer/ZIP roots). The legacy `register_execution_provider_library(name, path)` ORT 1.24 API is the only sink.
- Open work: verifying exact MSIX install paths under `C:\Program Files\WindowsApps\`, the NVIDIA ZIP internal layout, and confirming the Python WinAppSDK ML binding name. These are itemized in **Open questions / TODOs**.

## Background: why a path list, not a registry dict

The current registry encodes a single static fact per EP: "this PyPI distribution name, with this relative path inside it." That works for two of the EPs we ship (`OpenVINOExecutionProvider`, `QNNExecutionProvider`) and only because Intel and Qualcomm both publish plugin-form wheels to PyPI. The wheel structure is verifiable in this repo's venv:

```text
onnxruntime_ep_openvino-1.4.0/onnxruntime_providers_openvino_plugin.dll
onnxruntime_qnn-2.1.0/libs/{amd64|arm64ec}/onnxruntime_providers_qnn.dll
```

(Confirmed via `uv run python -c "import importlib.metadata as md; ..."`, this session.)

The other EPs we want to support — VitisAI, MIGraphX, NvTensorRtRtx — never reach a venv via pip in their typical deployment paths. They arrive through:

1. **MSIX delivery** via Windows Update D-week, mediated by the WinAppSDK `ExecutionProviderCatalog` API. The on-disk path is decided by the OS package manager and is queryable only at runtime via `WinMLEpGetLibraryPath` / Python `provider.library_path` ([learn.microsoft.com/.../register-execution-providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/register-execution-providers)).
2. **Per-machine EXE installers** that drop binaries into `C:\Program Files\<vendor>\<version>\...` and set an environment variable like `RYZEN_AI_INSTALLATION_PATH` ([ryzenai.docs.amd.com/en/latest/inst.html](https://ryzenai.docs.amd.com/en/latest/inst.html)).
3. **Manual ZIP extraction** from GitHub releases — the user picks the directory, no envvar is set, no installer touches the registry. NVIDIA ships `TensorRT-RTX-EP-ABI-v0.1.zip` (108 MB) this way ([github.com/NVIDIA/TensorRT-RTX-EP-ABI/releases/tag/v0.1.0](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI/releases/tag/v0.1.0)).
4. **Power-user custom builds** — someone compiled their own EP from source and wants to point at `D:\src\onnxruntime\build\Release\`.

These four origins have nothing in common except that the end product is `register_execution_provider_library(canonical_name, dll_path)`. That single-line ORT API call is the funnel; the work is everything upstream of it.

A path list — `EP_PATH = [src1, src2, src3, ...]` — collapses the four into one ordered iteration. Each element knows how to resolve itself (PyPI lookup, runtime catalog query, filesystem glob, literal path) and yields zero or more `(canonical_ep_name, absolute_dll_path)` pairs. The registry's job becomes: walk `EP_PATH`, dedupe by EP name (precedence = list order), call ORT.

## Per-EP origin investigation

For each EP, three columns: PyPI status, MSIX status, third-party status. Verified facts have a citation; unverified items are explicitly flagged. The canonical EP name in each section is the `ep_name` string ORT/Windows ML expects in `register_execution_provider_library`.

### Intel OpenVINO

Canonical name: `"OpenVINOExecutionProvider"` ([learn.microsoft.com/.../supported-execution-providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers), `EpName: "OpenVINOExecutionProvider"`).

- **PyPI**: `onnxruntime-ep-openvino` 1.4.0, currently installed in this repo's venv. Top-level package `onnxruntime_ep_openvino/` contains `onnxruntime_providers_openvino_plugin.dll` plus the OpenVINO runtime DLLs (`openvino.dll`, `openvino_intel_*_plugin.dll`, `tbb12.dll`). Verified this session via `importlib.metadata.distribution('onnxruntime-ep-openvino').files`.
- **MSIX**: `1.8.69.0` (OpenVINO 2026.0), released 2026 3D (third week of March 2026, [supported-execution-providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)). Upcoming `1.8.79.0` / OpenVINO 2026.1, GA 2026 5D. The MSIX package family pattern is `Microsoft.WindowsAppRuntime.WinML.OpenVINO_<version>_x64__8wekyb3d8bbwe`. **TODO**: verify the exact `Program Files\WindowsApps\` package directory name and the DLL leaf — likely `onnxruntime_providers_openvino_plugin.dll`, but the MSIX may rename it. The runtime path is queryable via `provider.library_path` so we do not need to hardcode it.
- **Third-party**: standalone OpenVINO Toolkit installs (`C:\Program Files (x86)\Intel\openvino_2026.x\`) ship the OpenVINO runtime but, as of 2026-04-27, **do not ship a registrable ORT plugin DLL** — that DLL is built by Microsoft / the ORT team and packaged into `onnxruntime-ep-openvino` and the MSIX. We do not need to support a third-party row for OpenVINO.

### Qualcomm QNN

Canonical name: `"QNNExecutionProvider"` ([supported-execution-providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers), `EpName: "QNNExecutionProvider"`).

- **PyPI**: `onnxruntime-qnn` 2.1.0, currently installed. Layout `onnxruntime_qnn/libs/{amd64|arm64ec}/onnxruntime_providers_qnn.dll` plus QAIRT runtime DLLs (`QnnHtp.dll`, `QnnCpu.dll`, `Genie.dll`, `libQnnHtpV*Skel.so`). Verified this session.
- **MSIX**: `2.2420.43.0` / QAIRT 2.42, released 2026 4D ([supported-execution-providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)). 1.8.x branch's last QNN MSIX is `1.8.30.0` / QAIRT 2.40 (2026 1D). MSIX path pattern `Microsoft.WindowsAppRuntime.WinML.QNN_*_arm64__8wekyb3d8bbwe`; **TODO**: verify exact directory and DLL leaf.
- **Third-party**: Qualcomm QAIRT SDK ZIP (download from Qualcomm developer site) ships QAIRT runtime DLLs but, like OpenVINO, **does not include the ORT plugin DLL**. The plugin is Microsoft-built. No third-party row.

### AMD VitisAI

Canonical name: `"VitisAIExecutionProvider"` ([supported-execution-providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers), `EpName: "VitisAIExecutionProvider"`).

- **PyPI**: `onnxruntime-vitisai` returns HTTP 404 on PyPI as of 2026-04-27 (verified this session). AMD owns four placeholder package names — `onnxruntime-ep-amdgpu`, `onnxruntime-ep-rocm`, `onnxruntime-ep-migraphx`, `onnxruntime-ep-hipdnn` — at version 0.0.0 with summary "comming soon" (verified `onnxruntime-ep-rocm` and `onnxruntime-ep-migraphx` this session via `pypi.org/pypi/<name>/json`). No PyPI delivery for VitisAI today. Existing `ep-sideloading-research.md:10` records the same.
- **MSIX**: `1.8.59.0`, released 2026 4D ([supported-execution-providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)). Upcoming `1.8.62.0` (EP 2705) GA 2026 5D. Requires AMD Adrenalin in the bounded range `25.6.3 → 25.9.1` with NPU driver `32.00.0203.280 → 32.00.0203.297` (both endpoints inclusive, per Microsoft Learn's min/max columns). **TODO**: verify on-disk MSIX path; runtime-queryable via WinAppSDK.
- **Third-party**: AMD Ryzen AI Software EXE installer (latest 1.7.1, [ryzenai.docs.amd.com/en/latest/inst.html](https://ryzenai.docs.amd.com/en/latest/inst.html)). Default install root `C:\Program Files\RyzenAI\1.7.1\`, also exported as `%RYZEN_AI_INSTALLATION_PATH%`. The ORT plugin DLL is `onnxruntime_providers_vitisai.dll`, located in `%RYZEN_AI_INSTALLATION_PATH%\deployment\` according to AMD's deployment guide (web search result quoting `xcopy /Y "%RYZEN_AI_INSTALLATION_PATH%\deployment\onnxruntime_providers_vitisai.dll" .`). The accompanying `onnxruntime_providers_shared.dll`, `onnxruntime_vitisai_ep.dll`, `onnxruntime_vitis_ai_custom_ops.dll`, plus `voe-4.0-win_amd64\xclbins\` (NPU binaries) live alongside ([app_development.html](https://ryzenai.docs.amd.com/en/latest/app_development.html)). The installer also writes wheels into a conda env at `C:\Program Files\RyzenAI\1.7.1\python\Lib\site-packages\` — but that env is parallel to ours, not a venv we share. **TODO**: confirm whether `C:\Program Files\onnxruntime\bin\` (mentioned in some manual-install paths from older 1.0.x docs) is still used by 1.7.x; current 1.7.x docs point to `<install>\deployment\`.

### AMD MIGraphX

Canonical name: `"MIGraphXExecutionProvider"` ([supported-execution-providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers), `EpName: "MIGraphXExecutionProvider"`).

- **PyPI**:
  - `onnxruntime-migraphx` 1.25.0 exists ([pypi.org/pypi/onnxruntime-migraphx/json](https://pypi.org/pypi/onnxruntime-migraphx/json), this session). Uploaded by Microsoft; summary is the generic "ONNX Runtime is a runtime accelerator..." This is a **Linux-only vendor distro** — wheels are `manylinux_2_34_x86_64` only, no Windows wheels — that overrides `onnxruntime/` and ships `onnxruntime/capi/libonnxruntime_providers_migraphx.so`. Note the platform inversion vs `onnxruntime-trt-rtx` (Windows-only). Not consumable from `EP_PATH` regardless.
  - `onnxruntime-ep-migraphx` 0.0.0, "comming soon" placeholder from AMD ([pypi.org/pypi/onnxruntime-ep-migraphx/json](https://pypi.org/pypi/onnxruntime-ep-migraphx/json), this session). Not yet shipping.
  - `onnxruntime-rocm` 1.22.2.post1 ([pypi.org/pypi/onnxruntime-rocm/json](https://pypi.org/pypi/onnxruntime-rocm/json), this session). Community-built — project URL points to [github.com/Looong01/onnxruntime-rocm-build](https://github.com/Looong01/onnxruntime-rocm-build); the PyPI `author` field reads `Microsoft Corporation; Loong` (single-`o` "Loong" matches the upstream PyPI metadata, while the GitHub handle is `Looong01` with three o's). Vendor distro, not AMD-official, not a plugin.
  - `onnxruntime-ep-rocm` 0.0.0, "comming soon" placeholder from AMD.
  - **Net**: no usable PyPI plugin for MIGraphX today. Skip the PyPI row in defaults; revisit when AMD ships `onnxruntime-ep-migraphx>=1.0`.
- **MSIX**: `1.8.55.0`, released 2026 4D ([supported-execution-providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)). Upcoming `1.8.56.0` / GPU EP Ver49 GA 2026 5D. Requires AMD GPU driver 25.10.13.09 (exact). MSIX is the only practical channel today. **TODO**: verify on-disk path; runtime-queryable via WinAppSDK.
- **Third-party**: AMD ROCm install (`C:\Program Files\AMD\ROCm\...`) ships HIP/MIGraphX runtime libraries but, as of 2026-04-27, the ORT plugin DLL for MIGraphX is delivered via MSIX, not via a standalone ROCm install. **TODO**: verify whether ROCm 6.x or AMD's HIP SDK ships an `onnxruntime_providers_migraphx.dll` (probably not). If a future Ryzen AI / Radeon install ships one, the third-party row mirrors the VitisAI one.

### NVIDIA TensorRT-RTX

Canonical name: **`"NvTensorRtRtxExecutionProvider"`** (camelCase, per Microsoft Learn supported-execution-providers page, `EpName: "NvTensorRtRtxExecutionProvider"`). NVIDIA's GitHub README at [github.com/NVIDIA/TensorRT-RTX-EP-ABI](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI) uses the PascalCase variant `"NvTensorRTRTXExecutionProvider"` — that is NVIDIA's bespoke registration string for the standalone ZIP path, not the name returned by `OrtEpDevice.ep_name` for the MSIX-delivered EP. The registry must use the camelCase form as the canonical key and treat the PascalCase form as an alias to normalize at the registry boundary.

- **PyPI**: `onnxruntime-trt-rtx` 1.23.2 (PyPI `author = Microsoft Corporation`, [pypi.org/pypi/onnxruntime-trt-rtx/json](https://pypi.org/pypi/onnxruntime-trt-rtx/json), this session). This is a **Windows-only vendor distro** that overrides `onnxruntime/`; using it is mutually exclusive with the plugin model (replaces the in-tree `onnxruntime` package). The wheel ships TensorRT-RTX runtime DLLs at version 1.2 (`tensorrt_rtx_1_2.dll`, `tensorrt_onnxparser_rtx_1_2.dll`), whereas the third-party GitHub ZIP (below) ships TensorRT-RTX 1.4 — the two channels target different TRT-RTX runtime versions. `onnxruntime-ep-tensorrt` and `onnxruntime-ep-trt-rtx` both 404 (verified this session). No plugin-form PyPI delivery exists.
- **MSIX**: `0.0.28.0`, released 2026 4D under WinML 2.x ([supported-execution-providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)). Earlier WinML 1.8.x branch ships `1.8.24.0` (2026 2D). Requires NVIDIA driver 32.0.15.5585+ and CUDA 12.5+. **TODO**: verify on-disk MSIX path; runtime-queryable via WinAppSDK.
- **Third-party**: GitHub release `NVIDIA/TensorRT-RTX-EP-ABI` v0.1.0 (2026-04-09). Single asset: `TensorRT-RTX-EP-ABI-v0.1.zip`, `108,298,586` bytes (≈108 MB decimal / ≈103 MiB binary, [download URL](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI/releases/download/v0.1.0/TensorRT-RTX-EP-ABI-v0.1.zip), verified via `gh api releases/tags/v0.1.0`, this session). The release body's **Package Contents** section explicitly enumerates the ZIP's payload: `onnxruntime_providers_nv_tensorrt_rtx.dll` (the registrable plugin), `tensorrt_rtx_1_4.dll`, `tensorrt_onnxparser_rtx_1_4.dll`, `tensorrt_plugins.dll`, `onnxruntime_providers_nv_tensorrt_rtx.pdb`, `LICENSE`, `Privacy.md`, `ThirdPartyNotices.txt`, `TRT_RTX_Acknowledgements.txt`. The contents list carries no subdirectory annotations — strongly suggesting a flat top-level layout (final confirmation requires `unzip -l`, ~30s, but the table is already authoritative). The README also documents the registration call: `register_execution_provider_library("NvTensorRTRTXExecutionProvider", "onnxruntime_providers_nv_tensorrt_rtx.dll")` — note that registration string is NVIDIA's PascalCase form, which the registry should normalize to the camelCase canonical `NvTensorRtRtxExecutionProvider`.

## Design

### EP_PATH shape

**Recommendation: option B (annotated entries) over option A (raw paths).** The justification follows from the per-EP investigation: every origin needs different metadata at resolution time.

- PyPI sources need a **distribution name**, not a path — the path is a function of `importlib.metadata.distribution(name).locate_file(rel)`, which depends on the active venv's `site-packages` location. A raw path list would have to be rebuilt per process.
- MSIX sources have **no static path**; the path comes from a runtime call to `ExecutionProviderCatalog`. A raw path list cannot represent this without sentinel values.
- Third-party installer sources have a **path that's only known via an envvar** (`RYZEN_AI_INSTALLATION_PATH`) which may be unset. A raw path list would force eager evaluation; we want lazy.
- ZIP/custom-build sources are the only origin where a literal path makes sense.

A tagged union accommodates all four. Option C (manifest-driven JSON files) was rejected: nobody else uses such a format, and it pushes complexity onto every consumer to write manifests for paths they already know.

```python
# Proposed shape (illustrative, exact module path TBD in migration plan).
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

@dataclass(frozen=True)
class PyPiSource:
    """A pip-installed plugin EP wheel."""
    distribution: str                         # e.g. "onnxruntime-ep-openvino"
    relative_dll: str                         # path inside the wheel, POSIX
    eps: tuple[str, ...]                      # canonical EP names this source provides
    arch_resolver: callable | None = None     # optional: returns a relative path tweaked per machine arch

@dataclass(frozen=True)
class WinMLCatalogSource:
    """An MSIX EP delivered via the WinAppSDK ExecutionProviderCatalog."""
    catalog_name: str                         # the name passed to WinMLEpCatalogFindProvider, e.g. "QNN"
    eps: tuple[str, ...]                      # canonical EP names this source provides

@dataclass(frozen=True)
class DirectorySource:
    """A directory tree that contains a plugin DLL (installer drop, unzipped archive, custom build)."""
    root: Path                                # absolute, may include glob like .../1.7.1
    dll_patterns: dict[str, str]              # ep_name -> filename or relative glob
    env_var: str | None = None                # if set, root is `Path(os.environ[env_var])`; if envvar absent, source is skipped
    required_marker: str | None = None        # optional file inside root used as a sanity check before scanning

EpSource = PyPiSource | WinMLCatalogSource | DirectorySource

EP_PATH: list[EpSource] = [...]               # see "Default contents"
```

### Discovery algorithm

The registry walks `EP_PATH` in order. For each source, it asks: "what `(ep_name, absolute_dll_path)` pairs do you produce, given the current machine state?" The first source that returns a path for a given EP wins; later sources are skipped for that EP. After the walk, the registry calls `register_execution_provider_library` once per `(ep_name, dll_path)`. Failure to register one EP does not abort the walk.

```text
for source in EP_PATH:
    for ep_name, dll_path in source.resolve():        # may yield 0..N pairs, lazy
        if ep_name in already_registered:
            log.debug(f"{ep_name} already resolved by an earlier source, skipping {dll_path}")
            continue
        if not dll_path.exists():
            log.warning(f"{ep_name}: {source} produced {dll_path} which does not exist")
            continue
        try:
            ort.register_execution_provider_library(ep_name, str(dll_path))
            already_registered[ep_name] = (source, dll_path)
        except RuntimeError as e:
            log.error(f"{ep_name}: ORT rejected {dll_path}: {e}")
```

Per-source `resolve()` semantics:

- `PyPiSource.resolve()` calls `importlib.metadata.distribution(self.distribution)`; on `PackageNotFoundError`, yields nothing. Otherwise, computes `dist.locate_file(self.relative_dll)`; yields `(ep, path)` for each ep in `self.eps` if the file exists. The `arch_resolver` hook covers QNN's `amd64`/`arm64ec` split — see [`src/winml/modelkit/winml.py:22`](../src/winml/modelkit/winml.py).
- `WinMLCatalogSource.resolve()` lazily imports the WinAppSDK ML Python binding (the fully qualified module name is **TODO**, see open questions). Calls `catalog.find_all_providers()`, filters by `provider.name == self.catalog_name`, and for any provider whose `ready_state != NotPresent`, calls `ensure_ready_async().get()` and yields `(ep, Path(provider.library_path))`. If the binding is unavailable (Linux, ORT 1.24 without WinAppSDK ML, or WinAppSDK < 1.8), `resolve()` yields nothing silently.
- `DirectorySource.resolve()`: if `env_var` is set and `os.environ.get(env_var)` is empty, yield nothing. Otherwise, walks `self.root` (or `Path(os.environ[env_var]) / self.root` if root is relative) looking for each `dll_patterns[ep]` filename; supports glob through `Path.glob()`. Skips if `required_marker` is set and missing.

### Default contents

Out of the box, on a fresh Windows install with this repo's venv, `EP_PATH` should resolve every PyPI plugin we declare a dependency on, plus opportunistically pick up MSIX-installed EPs and the two well-known installer roots. Linux and macOS get a shorter list (no MSIX; OpenVINO and QNN PyPI wheels still apply on Linux for matching ABI; nothing else).

```python
# Windows defaults (illustrative; concrete table in "Per-origin × per-EP map" below).
EP_PATH_WINDOWS = [
    # 1. PyPI plugin wheels — primary source for OpenVINO and QNN today.
    PyPiSource(
        distribution="onnxruntime-ep-openvino",
        relative_dll="onnxruntime_ep_openvino/onnxruntime_providers_openvino_plugin.dll",
        eps=("OpenVINOExecutionProvider",),
    ),
    PyPiSource(
        distribution="onnxruntime-qnn",
        relative_dll="onnxruntime_qnn/libs/{arch}/onnxruntime_providers_qnn.dll",
        eps=("QNNExecutionProvider",),
        arch_resolver=_qnn_arch_resolver,   # picks amd64 vs arm64ec
    ),

    # 2. WinAppSDK ExecutionProviderCatalog — opportunistic MSIX pickup for any
    #    EP we don't already have via PyPI. Order matters: PyPI wins if both are present.
    WinMLCatalogSource(catalog_name="OpenVINO",     eps=("OpenVINOExecutionProvider",)),
    WinMLCatalogSource(catalog_name="QNN",          eps=("QNNExecutionProvider",)),
    WinMLCatalogSource(catalog_name="VitisAI",      eps=("VitisAIExecutionProvider",)),
    WinMLCatalogSource(catalog_name="MIGraphX",     eps=("MIGraphXExecutionProvider",)),
    WinMLCatalogSource(catalog_name="NvTensorRtRtx", eps=("NvTensorRTRTXExecutionProvider",)),

    # 3. Well-known third-party installers, gated by envvar so they no-op on machines
    #    without the installer present.
    DirectorySource(
        root=Path("deployment"),
        env_var="RYZEN_AI_INSTALLATION_PATH",
        dll_patterns={"VitisAIExecutionProvider": "onnxruntime_providers_vitisai.dll"},
        required_marker="onnxruntime_providers_shared.dll",
    ),
]

# Linux defaults — only PyPI plugins. No MSIX, no Ryzen AI Windows installer.
EP_PATH_LINUX = [
    PyPiSource(
        distribution="onnxruntime-ep-openvino",
        relative_dll="onnxruntime_ep_openvino/libonnxruntime_providers_openvino_plugin.so",
        eps=("OpenVINOExecutionProvider",),
    ),
    # QNN does not ship an x86_64 Linux PyPI wheel as of 2026-04-27 (TODO verify).
]

# macOS — empty. No plugin EPs ship for darwin today.
EP_PATH_DARWIN = []
```

### Override mechanisms

Two override surfaces, in increasing precedence:

1. **Environment variable `WINML_EP_PATH`** (Windows path-list semantics: `;`-separated on Windows, `:`-separated on POSIX). Each entry is a directory; the registry interprets each as a `DirectorySource(root=Path(entry), dll_patterns=EP_DLL_NAMES)` — meaning every known DLL filename is searched in every supplied directory. This is the analog of `PATH`. Power-user knob, no per-EP filtering. Entries from `WINML_EP_PATH` are **prepended** to the default list (highest precedence).
2. **Function argument `register_execution_providers(extra_sources=[...])`**. Same `EpSource` types as the default list. Useful for tests and embedded apps that want surgical control. Sources passed here are inserted at index 0, before envvar-derived entries.

The `EP_DLL_NAMES` table that the envvar path uses is a small static dict:

```python
EP_DLL_NAMES: dict[str, list[str]] = {
    "OpenVINOExecutionProvider":     ["onnxruntime_providers_openvino_plugin.dll", "libonnxruntime_providers_openvino_plugin.so"],
    "QNNExecutionProvider":          ["onnxruntime_providers_qnn.dll"],
    "VitisAIExecutionProvider":      ["onnxruntime_providers_vitisai.dll"],
    "MIGraphXExecutionProvider":      ["onnxruntime_providers_migraphx.dll"],   # mirrors the VitisAI naming pattern (no "_plugin" suffix); confirmed by inspecting onnxruntime-migraphx wheel which ships libonnxruntime_providers_migraphx.so
    "NvTensorRtRtxExecutionProvider": ["onnxruntime_providers_nv_tensorrt_rtx.dll", "libonnxruntime_providers_nv_tensorrt_rtx.so"],
}
```

We deliberately reject a config-file mechanism (TOML/YAML in user home or project root). It would add a parser, a search precedence rule, and a config schema — all without solving a problem that a list-of-`EpSource`-in-Python doesn't already solve. Tests that need overrides pass `extra_sources=` directly.

### Conflict resolution

**Rule: list order is precedence; first valid hit wins per EP name; later hits are silently skipped (logged at DEBUG level).**

Rationale:

- A user who explicitly sets `WINML_EP_PATH` is signaling they have a custom build they want loaded; we honor that over the default PyPI wheel.
- A user with both a PyPI plugin and an MSIX-installed EP gets the PyPI version, because PyPI sits earlier in the default list. PyPI is more deterministic (locked to `pyproject.toml`) than MSIX (Windows Update can change MSIX out from under us). For repeatability we prefer the venv-pinned source.
- ORT itself rejects a second `register_execution_provider_library` call with the same name (`library is already registered under <name>`, [`docs/ep-sideloading-research.md:34`](ep-sideloading-research.md)). So the conflict is also enforced at the ORT layer; we just avoid the exception.

### Registration semantics on failure

Per-source failure modes and the registry's response:

| Failure | Source class | Response |
|---|---|---|
| Distribution not installed | `PyPiSource` | Yield nothing; no log (this is the common case for optional EPs). |
| File missing inside an installed distribution | `PyPiSource` | Log WARN with full path; yield nothing for that EP. |
| WinAppSDK ML Python binding not importable | `WinMLCatalogSource` | Log DEBUG once per process; yield nothing. |
| `ExecutionProviderCatalog.find_all_providers()` raises | `WinMLCatalogSource` | Log WARN; yield nothing. |
| `ensure_ready_async().get()` returns non-Success status | `WinMLCatalogSource` | Log WARN with `result.status`; yield nothing. |
| `env_var` set but `Path(os.environ[env_var])` does not exist | `DirectorySource` | Log WARN; yield nothing. |
| `required_marker` missing | `DirectorySource` | Log WARN with the expected marker; yield nothing. |
| `register_execution_provider_library` raises | sink | Log ERROR with the EP name + path + exception; **continue** the walk (do not raise). The current `WinML.register_execution_providers` already does this with a `try/except Exception` ([`src/winml/modelkit/winml.py:111`](../src/winml/modelkit/winml.py)). Preserve the behavior. |

The registry never raises for "EP unavailable." It returns the dict of successfully registered EPs. Callers that need a specific EP must check the return value or call `ort.get_ep_devices()` and filter.

### Trade-off table: option A vs B vs C

The brief in the task description offered three shapes. The full analysis:

| Concern | Option A: raw `list[Path]` | Option B: tagged `list[EpSource]` (chosen) | Option C: manifest-driven |
|---|---|---|---|
| PyPI venv-portable resolution | Forces `Path(site_packages) / ...` to be computed eagerly per process; breaks when the venv moves. | Lazy via `importlib.metadata` — works regardless of venv layout. | Manifest must encode the distribution name; same lookup logic ends up living in the manifest parser. No win. |
| MSIX (no static path) | Cannot represent — would need a sentinel. | Tag `WinMLCatalogSource` covers it cleanly. | Manifest can declare "look up via WinML catalog with this name" — but that's just option B in JSON. |
| Per-EP filtering inside one path | Impossible: a path is just a directory. | Native via `dll_patterns`. | Native, but every contributor must know the manifest schema. |
| ABI-version pinning ("only load if plugin version matches host ORT") | Impossible. | Future extension via an `EpSource.compat: VersionRange` field. | Native, but unused today. |
| Windows + Linux symmetry | Manual with `os.name` checks at every site. | One default list per platform, swapped at module init. | Same. |
| Diagnostics ("which source provided this DLL?") | Lost — the path is anonymous. | Trivial: log the source object. | Same. |
| Surface area | 0 new types. | 4 new types (3 sources + the union). | 1 new type (manifest schema) but every consumer touches the parser. |
| Composability with `WINML_EP_PATH` env var | Native — the env var IS a list of paths. | The env var is parsed into `DirectorySource` entries; loses no fidelity. | The env var has to point at manifest files, not directories — much less ergonomic for end users. |

Option B is the only shape that satisfies all four origins without leaking origin-specific logic into the consumer. Option A is a non-starter for MSIX. Option C duplicates option B's structure in JSON without adding anything that callers want.

### Worked example: discovery on three machines

To make the algorithm concrete, three scenarios under the default Windows `EP_PATH`:

**Machine 1: Snapdragon X laptop, fresh `uv sync` of this repo.**

```text
EP_PATH walk:
  PyPiSource(onnxruntime-ep-openvino) -> distribution exists, locate_file resolves to
        .../site-packages/onnxruntime_ep_openvino/onnxruntime_providers_openvino_plugin.dll
        yield ("OpenVINOExecutionProvider", <abs path>)
  PyPiSource(onnxruntime-qnn)         -> distribution exists, arch_resolver picks "arm64ec",
        locate_file resolves to
        .../site-packages/onnxruntime_qnn/libs/arm64ec/onnxruntime_providers_qnn.dll
        yield ("QNNExecutionProvider", <abs path>)
  WinMLCatalogSource("OpenVINO")       -> WinAppSDK ML binding not installed (likely),
        ImportError, yield nothing.  Even if it were installed, OpenVINO already registered;
        skip on dedup.
  WinMLCatalogSource("QNN")            -> same; skip.
  WinMLCatalogSource("VitisAI")        -> WinAppSDK absent, yield nothing.
  WinMLCatalogSource("MIGraphX")       -> same.
  WinMLCatalogSource("NvTensorRtRtx")  -> same.
  DirectorySource(RYZEN_AI_INSTALLATION_PATH) -> envvar unset, yield nothing.

Result: {"OpenVINOExecutionProvider": [...], "QNNExecutionProvider": [...]}
```

Identical to today's behavior. No regression.

**Machine 2: Ryzen AI laptop, Ryzen AI Software 1.7.1 installed, no PyPI EP wheels.**

```text
EP_PATH walk:
  PyPiSource(onnxruntime-ep-openvino) -> PackageNotFoundError, yield nothing.
  PyPiSource(onnxruntime-qnn)         -> PackageNotFoundError, yield nothing.
  WinMLCatalogSource("OpenVINO")       -> assume WinAppSDK ML installed; provider state
        NotPresent (not on AMD hw), find_all_providers returns it but EnsureReadyAsync
        would fail with "incompatible hardware"; yield nothing.
  WinMLCatalogSource("QNN")            -> NotPresent; yield nothing.
  WinMLCatalogSource("VitisAI")        -> Ready (MSIX installed via Windows Update),
        TryRegister/library_path returns
        C:\Program Files\WindowsApps\Microsoft.WindowsAppRuntime.WinML.VitisAI_1.8.59.0_x64__8wekyb3d8bbwe\onnxruntime_providers_vitisai.dll
        yield ("VitisAIExecutionProvider", <that path>).
  ...continues...
  DirectorySource(RYZEN_AI_INSTALLATION_PATH) -> envvar = "C:\Program Files\RyzenAI\1.7.1",
        root resolves to "C:\Program Files\RyzenAI\1.7.1\deployment",
        required_marker "onnxruntime_providers_shared.dll" present,
        dll for VitisAI present, BUT VitisAIExecutionProvider already registered from
        WinMLCatalogSource (earlier in list); skip on dedup.

Result: {"VitisAIExecutionProvider": [...]} from the MSIX path.
```

Note the conflict resolution outcome: MSIX won over installer because MSIX appeared first in `EP_PATH`. If the team prefers the installer-shipped DLL to win (because it tracks Ryzen AI's release cadence rather than Windows Update's), reorder the default list. The mechanism does not encode a policy; the default list does.

**Machine 3: developer machine, custom build of OpenVINO plugin at `D:\src\onnxruntime\build\openvino\Release\`.**

```text
WINML_EP_PATH=D:\src\onnxruntime\build\openvino\Release

EP_PATH walk (envvar entries prepended):
  DirectorySource(D:\src\onnxruntime\build\openvino\Release, dll_patterns=EP_DLL_NAMES)
        -> finds onnxruntime_providers_openvino_plugin.dll,
        yield ("OpenVINOExecutionProvider", D:\src\...\Release\onnxruntime_providers_openvino_plugin.dll)
  PyPiSource(onnxruntime-ep-openvino) -> already registered; skip.
  ... rest as machine 1 ...

Result: {"OpenVINOExecutionProvider": [<dev build path>], "QNNExecutionProvider": [...]}
```

The dev build wins because the envvar source is prepended. This is the explicit "I know what I'm doing" override path.

### Interaction with the WinML 2.x `library_path` property

A subtle point about `WinMLCatalogSource`: the WinAppSDK ML Python binding exposes `provider.library_path` (per [register-execution-providers.md](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/register-execution-providers), the Python `register_execution_provider_library` example). That property is populated only after `ensure_ready_async().get()` returns `Success` — before that, the EP is `NotPresent` or `NotReady` and the library is not on disk in a registrable form.

The resolver MUST gate on the ready state:

```text
for provider in catalog.find_all_providers():
    if provider.name != self.catalog_name:
        continue
    if provider.ready_state == NotPresent:
        # Don't auto-download. Downloads can be hundreds of MB.
        # Caller has to opt in (TODO: add an `auto_download: bool = False` kwarg).
        continue
    if provider.ready_state in (NotReady, Ready):
        result = provider.ensure_ready_async().get()
        if result.status != Success:
            log.warning(...)
            continue
    yield (canonical_name, Path(provider.library_path))
```

The `auto_download=False` default avoids surprising the user with a multi-second to multi-minute network operation on the first call. Apps that want eager download (e.g., a startup wizard) pass `auto_download=True` to the source constructor or the registry function.

Note also that the WinAppSDK doc explicitly warns the Python `EnsureAndRegisterCertifiedAsync()` and `RegisterCertifiedAsync()` calls **do not register EPs to the Python ORT env** ("`# Please DO NOT use this API.`" — both in the C# tab examples for Python). We do registration ourselves via `ort.register_execution_provider_library` after retrieving `library_path`. Not via `provider.TryRegister()`.

## Per-origin × per-EP map

Concrete entries for each (EP, origin) cell under the Option B design. "n/a" means no shipping channel exists for that pair as of 2026-04-27. "TODO" means the shape is known but a key path or DLL leaf needs verification.

| EP | PyPI origin | MSIX origin | Third-party origin | Custom path origin |
|---|---|---|---|---|
| `OpenVINOExecutionProvider` | `PyPiSource(distribution="onnxruntime-ep-openvino", relative_dll="onnxruntime_ep_openvino/onnxruntime_providers_openvino_plugin.dll", eps=("OpenVINOExecutionProvider",))` (verified, 1.4.0) | `WinMLCatalogSource(catalog_name="OpenVINO", eps=("OpenVINOExecutionProvider",))` (MSIX 1.8.69.0; runtime-resolved path) | n/a (Intel OpenVINO Toolkit standalone install does not ship an ORT plugin DLL) | `WINML_EP_PATH=D:\custom\openvino\` + `EP_DLL_NAMES["OpenVINOExecutionProvider"]` |
| `QNNExecutionProvider` | `PyPiSource(distribution="onnxruntime-qnn", relative_dll="onnxruntime_qnn/libs/{arch}/onnxruntime_providers_qnn.dll", eps=("QNNExecutionProvider",), arch_resolver=...)` (verified, 2.1.0) | `WinMLCatalogSource(catalog_name="QNN", eps=("QNNExecutionProvider",))` (MSIX 2.2420.43.0; runtime-resolved path) | n/a (Qualcomm QAIRT SDK ZIP does not ship an ORT plugin DLL) | `WINML_EP_PATH=D:\custom\qnn\amd64\` |
| `VitisAIExecutionProvider` | n/a (404 on PyPI; AMD placeholder names at 0.0.0) | `WinMLCatalogSource(catalog_name="VitisAI", eps=("VitisAIExecutionProvider",))` (MSIX 1.8.59.0; runtime-resolved path) | `DirectorySource(env_var="RYZEN_AI_INSTALLATION_PATH", root=Path("deployment"), dll_patterns={"VitisAIExecutionProvider": "onnxruntime_providers_vitisai.dll"}, required_marker="onnxruntime_providers_shared.dll")` (verified DLL leaf via AMD docs; default install root `C:\Program Files\RyzenAI\1.7.1\`) | `WINML_EP_PATH=C:\Program Files\RyzenAI\1.7.1\deployment\` |
| `MIGraphXExecutionProvider` | n/a (`onnxruntime-ep-migraphx` is 0.0.0 placeholder; `onnxruntime-migraphx` is a vendor distro, not a plugin) | `WinMLCatalogSource(catalog_name="MIGraphX", eps=("MIGraphXExecutionProvider",))` (MSIX 1.8.55.0; runtime-resolved path) | TODO — investigate whether AMD ROCm 6.x or HIP SDK installers ship a registrable MIGraphX plugin DLL outside MSIX. As of 2026-04-27 no evidence of a third-party plugin drop. | `WINML_EP_PATH=D:\custom\migraphx\` (DLL leaf TBD; **TODO verify name**) |
| `NvTensorRtRtxExecutionProvider` | n/a (`onnxruntime-trt-rtx` is a vendor distro; `onnxruntime-ep-tensorrt`/`-ep-trt-rtx` 404) | `WinMLCatalogSource(catalog_name="NvTensorRtRtx", eps=("NvTensorRtRtxExecutionProvider",))` (MSIX 0.0.28.0; runtime-resolved path) | `DirectorySource(env_var="NVIDIA_TRT_RTX_EP", root=Path("."), dll_patterns={"NvTensorRtRtxExecutionProvider": "onnxruntime_providers_nv_tensorrt_rtx.dll"})` — user unzips the GitHub release ZIP somewhere and sets `NVIDIA_TRT_RTX_EP` to that root. The release body documents the package contents (`onnxruntime_providers_nv_tensorrt_rtx.dll`, `tensorrt_rtx_1_4.dll`, `tensorrt_onnxparser_rtx_1_4.dll`, `tensorrt_plugins.dll`, plus license/notice files) without subdirectory annotations, suggesting a flat layout. | `WINML_EP_PATH=D:\unzipped\trt-rtx-ep\` |

## Migration plan

### What changes from the caller's perspective

The 90% case — a downstream caller of `register_execution_providers()` — sees no change. The signature accepts a new optional `extra_sources` kwarg; existing callers (none of which pass it) continue to receive the same `dict[str, list[str]]` return shape from the same function name in the same module.

The 10% case — code that imports `EP_PLUGIN_REGISTRY` or `resolve_plugin_dll` directly — needs migration. We searched: zero such callers exist in this repo (`Grep "EP_PLUGIN_REGISTRY|resolve_plugin_dll" src/ tests/` returns only the definitions themselves). Removal is safe.

The 0.1% case — code that wants to add a new EP origin without modifying `winml.py` — gets a new affordance: build an `EpSource` and pass it via `extra_sources=[my_source]`. This is the test fixture path and the hook for unblocking the WinAppSDK ML integration before it lands in the default list.

### Comparison: before/after on a representative call

Today:

```python
# src/winml/modelkit/winml.py current behavior
EP_PLUGIN_REGISTRY = {
    "OpenVINOExecutionProvider": ("onnxruntime-ep-openvino",
                                  "onnxruntime_ep_openvino/onnxruntime_providers_openvino_plugin.dll"),
    "QNNExecutionProvider":      ("onnxruntime-qnn",
                                  f"onnxruntime_qnn/libs/{_qnn_arch_dir()}/onnxruntime_providers_qnn.dll"),
}

def resolve_plugin_dll(ep_name):
    pkg, rel = EP_PLUGIN_REGISTRY[ep_name]
    return Path(metadata.distribution(pkg).locate_file(rel))   # may not exist

# Callers iterate EP_PLUGIN_REGISTRY keys, call resolve_plugin_dll, register one-by-one.
```

Proposed:

```python
# src/winml/modelkit/winml.py proposed
EP_PATH: list[EpSource] = _default_ep_path_for_platform()

def register_execution_providers(ort=True, ort_genai=False, extra_sources=None):
    sources = (extra_sources or []) + _parse_winml_ep_path() + EP_PATH
    resolved = {}    # ep_name -> (source, abs_path)
    for source in sources:
        for ep_name, dll_path in source.resolve():
            if ep_name in resolved:
                logger.debug("skip %s from %s; already from %s", ep_name, source, resolved[ep_name][0])
                continue
            if not dll_path.is_file():
                logger.warning("missing DLL for %s at %s", ep_name, dll_path)
                continue
            resolved[ep_name] = (source, dll_path)
    return _register_with_ort(resolved, ort=ort, ort_genai=ort_genai)
```

The `_register_with_ort` helper preserves the current behavior of catching exceptions per-EP and returning a `dict[module_name, list[ep_name]]`.

### Arch resolver detail

QNN's arch-split (`amd64` vs `arm64ec`) is the only PyPI plugin currently in scope that needs per-machine-arch resolution. The existing helper in [`src/winml/modelkit/winml.py:22`](../src/winml/modelkit/winml.py):

```python
def _qnn_arch_dir() -> str:
    return "arm64ec" if platform.machine().lower() in ("arm64", "aarch64") else "amd64"
```

Becomes a callable on `PyPiSource`:

```python
def _qnn_arch_resolver(rel_template: str) -> str:
    arch = "arm64ec" if platform.machine().lower() in ("arm64", "aarch64") else "amd64"
    return rel_template.format(arch=arch)

PyPiSource(
    distribution="onnxruntime-qnn",
    relative_dll="onnxruntime_qnn/libs/{arch}/onnxruntime_providers_qnn.dll",
    eps=("QNNExecutionProvider",),
    arch_resolver=_qnn_arch_resolver,
)
```

The default resolver (`arch_resolver=None`) is a passthrough; the QNN entry is the only one in `EP_PATH` today that overrides it. Adding similar logic for future EPs (e.g., a hypothetical OpenVINO arm64 variant) is a one-function addition with no schema change.

### Module placement

The current registry lives in [`src/winml/modelkit/winml.py`](../src/winml/modelkit/winml.py), 168 lines, mixing `EP_PLUGIN_REGISTRY`, `resolve_plugin_dll`, `WinML` singleton, and `add_ep_for_device` (an unrelated session-options helper). The new design adds three dataclass types, a sized default list, an envvar parser, an arch resolver, and a per-source `resolve()` dispatch — roughly 300 lines of new code. Keep it in one module to avoid an import-cycle headache and to match the project's flat layout, but split internally:

```text
src/winml/modelkit/winml.py
    # Public API:
    register_execution_providers(...)        # unchanged signature, plus extra_sources kwarg
    EP_PATH                                   # exported for inspection / tests
    EpSource, PyPiSource, WinMLCatalogSource, DirectorySource

    # Internal:
    _DEFAULT_EP_PATH_WINDOWS / _LINUX / _DARWIN
    _qnn_arch_resolver        # extracted from current _qnn_arch_dir
    _parse_winml_ep_path      # consumes WINML_EP_PATH env var
    _winml_catalog_resolve    # the WinAppSDK ML import + call
```

The existing `add_ep_for_device` stays as is, untouched.

### Backwards-compatible API

The public surface is `register_execution_providers(ort: bool = True, ort_genai: bool = False)`. All existing callers pass no further args. Add one new keyword:

```python
def register_execution_providers(
    ort: bool = True,
    ort_genai: bool = False,
    extra_sources: list[EpSource] | None = None,
) -> dict[str, list[str]]:
```

`extra_sources` defaults to `None`; existing call sites (`WinML().register_execution_providers(ort=ort, ort_genai=ort_genai)` at line 122-133) need no change. The `WinML` singleton's internal `_ep_paths: dict[str, str]` becomes `_resolved: dict[str, tuple[EpSource, Path]]`; the rest of the singleton machinery is unchanged.

`EP_PLUGIN_REGISTRY` (the public dict at line 28) is **deleted** — it has no callers outside this module (verifiable with `Grep "EP_PLUGIN_REGISTRY"`). `resolve_plugin_dll` (line 40) likewise has no external callers and can be removed; if any test uses it, replace with `next((p for s in EP_PATH if isinstance(s, PyPiSource) for ep, p in s.resolve() if ep == name), None)`.

### Step-by-step migration

1. Land the dataclasses + the new `EP_PATH` Windows default with only the two `PyPiSource` entries that match today's `EP_PLUGIN_REGISTRY`. The output is byte-for-byte identical to today's registration on a stock install. No behavior change.
2. Add `WinMLCatalogSource` and the WinAppSDK ML import gate. On any machine where the Python binding is missing (which is currently every machine in this repo's CI — verify with `uv run python -c "import winml"` returning ImportError), the source no-ops. Land it dark.
3. Add `DirectorySource` with the `RYZEN_AI_INSTALLATION_PATH` entry. This activates the AMD third-party path on Ryzen AI machines but does nothing on machines without the envvar.
4. Add `WINML_EP_PATH` parsing and the `extra_sources` kwarg. Both are inert until used.
5. Once 1-4 are stable, delete `EP_PLUGIN_REGISTRY` and `resolve_plugin_dll`.

Each step is a separate commit; each commit can be reverted independently if a regression surfaces.

### Test boundaries

Unit tests (no real DLLs needed):

- `PyPiSource.resolve()` against a fixture distribution name that does and does not exist (`importlib.metadata` is monkeypatchable).
- `DirectorySource.resolve()` with `tmp_path` populated to mimic a Ryzen AI install root.
- `WINML_EP_PATH` parser: empty string, single entry, multi entry with `;` and `:` separators, entries with spaces, nonexistent paths.
- Conflict resolution: a config with two sources providing the same EP and assert the first wins; ORT is mocked so we just check `register_execution_provider_library` arguments.
- `extra_sources` precedence: an `extra_sources` entry overrides a default-list entry for the same EP.

Integration tests (real DLLs, gated by hardware availability):

- On the Snapdragon CI box: assert `QNNExecutionProvider` registers from the `onnxruntime-qnn` PyPI source.
- On any Windows box with `onnxruntime-ep-openvino` installed: assert OpenVINO registers.
- MSIX path: not testable in this repo until a CI image with WinAppSDK ML preinstalled exists. Document the manual smoke test (run `register_execution_providers()` on a machine with a Windows-Update-installed VitisAI MSIX and verify `ort.get_ep_devices()` shows it).
- ZIP path: gated on a fixture ZIP; download once, cache in `tests/_assets/trt-rtx-ep-fixture/`, assert `DirectorySource` finds the DLL.

The current test file is **TODO**: locate via `Grep "register_execution_providers" tests/`. Existing tests targeting `EP_PLUGIN_REGISTRY` (if any) need to be rewritten against the new shape.

### Security considerations

`register_execution_provider_library(name, path)` calls `LoadLibrary(path)` directly ([`docs/ep-sideloading-research.md:27`](ep-sideloading-research.md)). Any DLL loaded this way runs in-process with full access to whatever the host has. Three threat surfaces matter for `EP_PATH`:

1. **Untrusted `WINML_EP_PATH` values.** A malicious envvar pointing at `C:\Temp\` could redirect EP loading to a planted DLL with one of the canonical names. Mitigation: this is no worse than what `PATH` already enables (any executable resolution); we do not introduce new attack surface. We also do not auto-elevate or suppress AppLocker / WDAC; the OS loader policy still applies. Document the `WINML_EP_PATH` precedence in the user-facing README so that admins of locked-down machines know to clear it via Group Policy.
2. **Glob expansion in `DirectorySource.root`.** If an attacker controls a parent directory we glob, they could plant a same-named DLL in a sibling path. Mitigation: use `Path.glob()` only with literal patterns we author (no user-supplied glob), and require `required_marker` for installer roots.
3. **NVIDIA ZIP / Ryzen AI installer integrity.** Out of scope for `EP_PATH`. The user / admin is responsible for the integrity of bytes on disk; the registry only finds and loads. We document this clearly.

We do **not** verify code signatures before `LoadLibrary`. ORT itself doesn't, and adding a signature check at the registry layer would block the legitimate "developer custom build" use case without meaningfully raising the bar (an attacker who can plant a DLL can also strip a signature check). The explicit non-goal is documented below.

### Non-goals

- **No code-signature verification** of the loaded DLL. See above.
- **No automatic MSIX download.** `WinMLCatalogSource` defaults to `auto_download=False`. Callers that want eager install pass `auto_download=True`.
- **No transitive dependency loading.** `register_execution_provider_library` triggers Win32's loader; if a plugin needs `openvino.dll` next to it and it's absent, ORT raises and we log + skip. We do not add the plugin's directory to `os.add_dll_directory()` ourselves. (The PyPI plugins ship their dependency DLLs sibling to the plugin DLL, where the loader finds them by default; the MSIX layout is OS-managed; the installer + ZIP cases are the user's responsibility per [`registration semantics on failure`](#registration-semantics-on-failure).)
- **No version negotiation.** ORT 1.24's `register_execution_provider_library` does not surface a plugin's expected ABI version through the Python API. If a plugin compiled against ORT 1.25 is loaded into a process running ORT 1.24, the `LoadLibrary` may succeed and the failure manifests later as a missing entry point or undefined behavior. Detecting this in `EP_PATH` would require parsing each plugin's PE export table, which is out of scope for this design.
- **No async / threaded discovery.** All of `EP_PATH` resolves synchronously on first call to `register_execution_providers()`. The PyPI lookups are microsecond-scale; the WinML catalog query is millisecond-scale; the filesystem walks are bounded by the size of the installer root. Total cost is dominated by `register_execution_provider_library` itself, which already loads each DLL synchronously. No win from threading.

## Open questions / TODOs

The design above is internally consistent, but several concrete facts could not be verified from documentation alone in this session. None are blockers for landing steps 1-4 of the migration plan, but they should be resolved before declaring step 5 done.

1. **WinAppSDK ML Python binding name**. The Microsoft Learn doc shows `winml.ExecutionProviderCatalog.get_default()` and `provider.library_path` ([initialize-execution-providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/initialize-execution-providers), Python tab) but does not specify the importable module path. Likely candidates: `winui3.microsoft.windows.ai.machinelearning` (the doc comment hints at this), `winrt.microsoft.windows.ai.machinelearning`, or a Python package shipped with the WinAppSDK 1.8 NuGet. **Action**: install WinAppSDK 1.8 on a test Windows box and probe `import` paths.
2. **`Microsoft.WindowsAppRuntime.WinML.<EP>_<ver>_<arch>__8wekyb3d8bbwe` directory layout under `C:\Program Files\WindowsApps\`**. Strongly ACL-restricted; readable only by elevated processes / via the WinAppSDK API. Since `provider.library_path` returns the resolved DLL at runtime, we do **not** strictly need to hardcode this path — but documenting it lets us validate end-to-end without the SDK. **Action**: on a machine with at least one EP MSIX installed, run an elevated PowerShell `Get-AppxPackage Microsoft.WindowsAppRuntime.WinML.*` and `Get-AppxPackage ... | Format-List InstallLocation`.
3. **NVIDIA `TensorRT-RTX-EP-ABI-v0.1.zip` internal layout** — *partially resolved from the release body*. The GitHub release's Package Contents section enumerates the ZIP payload (see `### NVIDIA TensorRT-RTX` above): `onnxruntime_providers_nv_tensorrt_rtx.dll`, `tensorrt_rtx_1_4.dll`, `tensorrt_onnxparser_rtx_1_4.dll`, `tensorrt_plugins.dll`, plus the `.pdb` and license/notice files. No subdirectory annotations appear, suggesting a flat top-level. **Action remaining**: one-off `gh release download v0.1.0 -R NVIDIA/TensorRT-RTX-EP-ABI -p '*.zip' && unzip -l` to confirm flat-vs-`bin/`-vs-`lib/`. ~30s effort; not a blocker for the registry design (we glob anyway).
4. **MIGraphX plugin DLL filename**. The MS Learn doc gives the EP name (`MIGraphXExecutionProvider`) but no DLL leaf. Convention for MSIX-shipped plugin EPs is `onnxruntime_providers_<name>_plugin.dll`; for in-tree contrib EPs it's `onnxruntime_providers_<name>.dll`. `EP_DLL_NAMES["MIGraphXExecutionProvider"]` is currently a guess. **Action**: read the strings from a captured MSIX install or wait for `onnxruntime-ep-migraphx` to ship a non-placeholder version.
5. **TensorRT-RTX EP name capitalization** — *resolved from MS Learn*. Microsoft's supported-execution-providers page is authoritative for the MSIX-delivered EP (`EpName: "NvTensorRtRtxExecutionProvider"`, camelCase), since the catalog API registers the EP under that exact name. NVIDIA's PascalCase `"NvTensorRTRTXExecutionProvider"` in their GitHub README is the registration string they chose for the standalone-ZIP path; treat it as an alias and normalize to the camelCase canonical at the registry boundary. The registry's `EP_DLL_NAMES` keys and `Per-origin × per-EP map` rows have been updated accordingly. **Empirical confirmation** via `ort.get_ep_devices().ep_name` on a machine with the MSIX EP installed is still desirable but no longer blocking.
6. **QNN Linux PyPI status** — *resolved from PyPI metadata*. `onnxruntime-qnn` 2.1.0's PyPI release listing shows `manylinux_2_34_aarch64` wheels for cp311–cp314 (Linux ARM64) alongside the Windows wheels. **No `manylinux_x86_64` wheel exists** — Linux x86_64 is not supported via this package. Net: `EP_PATH` defaults for Linux can include the `onnxruntime-qnn` PyPI source on aarch64 only; on Linux x86_64 there is no PyPI plugin path and Qualcomm-direct downloads remain the only channel.
7. **MIGraphX third-party install path**. The default-list entry for VitisAI works because AMD's Ryzen AI installer sets `RYZEN_AI_INSTALLATION_PATH`. AMD's ROCm installer does not appear to set an analogous variable, and MIGraphX's Windows ORT plugin DLL outside MSIX is unconfirmed. **Action**: install the latest ROCm-for-Windows / Adrenalin Pro stack on a Radeon test box and grep for `onnxruntime_providers_migraphx*.dll`.
8. **Behavior under WinML 1.8.x vs 2.x branches**. `Microsoft.WindowsAppSDK.ML` 1.8.x and 2.x both expose `ExecutionProviderCatalog`, but only 2.x supports newer EP versions per the support-matrix table. The Python binding may differ between major versions. **Action**: verify the `WinMLCatalogSource` resolver works against both.
