# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import email
import json
import shutil
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path
from zipfile import ZipFile

import pytest

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 900,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(  # noqa: S603
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, (
        f"Command failed with exit code {result.returncode}: {command!r}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


def test_built_distributions_include_available_providers(tmp_path: Path) -> None:
    """Keep the provider validity mapping in both source and wheel distributions."""
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to build the distributions"
    config_path = "winml/modelkit/analyze/utils/avalizble_ep_device_ops/avaliable_providers.json"
    expected_config = json.loads((REPO_ROOT / "src" / config_path).read_text(encoding="utf-8"))
    assert expected_config
    dist_dir = tmp_path / "dist"

    _run([uv, "build", "--out-dir", str(dist_dir), str(REPO_ROOT)], cwd=tmp_path)

    wheels = list(dist_dir.glob("winml_cli-*.whl"))
    sdists = list(dist_dir.glob("winml_cli-*.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == 1
    with ZipFile(wheels[0]) as wheel:
        assert config_path in wheel.namelist()
        assert json.loads(wheel.read(config_path)) == expected_config
        metadata_path = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = email.message_from_bytes(wheel.read(metadata_path))
        python_requirement = SpecifierSet(metadata["Requires-Python"])
        assert str(python_requirement) == ">=3.14"
        assert "3.14" in python_requirement
        assert all(version not in python_requirement for version in ("3.11", "3.12", "3.13"))
        assert "Programming Language :: Python :: 3.14" in metadata.get_all("Classifier", [])
        requirements = [Requirement(raw) for raw in metadata.get_all("Requires-Dist", [])]
        for name in ("numpy", "scipy", "onnx", "onnxruntime-windowsml"):
            matching = [requirement for requirement in requirements if requirement.name == name]
            assert len(matching) == 1
            assert "python_version" not in str(matching[0].marker)
    with tarfile.open(sdists[0]) as sdist:
        config_members = [
            member for member in sdist.getmembers() if member.name.endswith(f"/src/{config_path}")
        ]
        assert len(config_members) == 1
        config_file = sdist.extractfile(config_members[0])
        assert config_file is not None
        with config_file:
            assert json.load(config_file) == expected_config


@pytest.mark.parametrize("platform_tag", ["win_arm64", "win_amd64"])
def test_windows_runtime_wheel_profile(tmp_path: Path, platform_tag: str) -> None:
    """Check both the installer's platform tag and the dependencies it will select."""
    uv = shutil.which("uv")
    assert uv is not None
    _run(
        [
            uv,
            "build",
            "--wheel",
            f"-C=--build-option=--plat-name={platform_tag}",
            "--out-dir",
            str(tmp_path),
            str(REPO_ROOT),
        ],
        cwd=tmp_path,
    )
    (archive,) = tmp_path.glob("*.whl")
    assert archive.name.endswith(f"-py3-none-{platform_tag}.whl")
    with ZipFile(archive) as wheel:
        (tag_info,) = (n for n in wheel.namelist() if n.endswith(".dist-info/WHEEL"))
        (metadata,) = (n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))
        assert email.message_from_bytes(wheel.read(tag_info)).get_all("Tag") == [
            f"py3-none-{platform_tag}"
        ]
        # Cross-building these tags is only valid while native code ships separately.
        assert not any(
            n.lower().endswith((".dll", ".pyd", ".exe", ".so")) for n in wheel.namelist()
        )
        requirements = email.message_from_bytes(wheel.read(metadata)).get_all("Requires-Dist") or []
    # CPython 3.14 reports the ARM64 host for both native and emulated interpreters.
    env = {"sys_platform": "win32", "platform_machine": "ARM64", "extra": ""}
    active = {
        r.name: r
        for r in map(Requirement, requirements)
        if r.marker is None or r.marker.evaluate(env)
    }
    assert {"ml-dtypes", "numpy", "onnx", "onnxruntime-windowsml", "windowsml"} <= active.keys()
    full_stack = {"torch", "torchvision", "pyarrow", "pandas", "optimum"}
    if platform_tag == "win_arm64":
        assert full_stack.isdisjoint(active)
        assert "llama" in active["windowsml"].extras
    else:
        assert full_stack <= active.keys()
        assert "windowsml-llama-core" not in active
        assert "llama" not in active["windowsml"].extras


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "win32", reason="Windows wheel dependencies are required")
def test_qnn_extra_installs_compatible_runtime_from_built_wheel(tmp_path: Path) -> None:
    """Install the QNN extra outside the source tree and verify both EP packages."""
    workspace = tmp_path.resolve()
    assert not workspace.is_relative_to(REPO_ROOT)

    uv = shutil.which("uv")
    assert uv is not None, "uv is required to build and install the wheel"

    dist_dir = workspace / "dist"
    consumer_dir = workspace / "consumer"
    environment_dir = consumer_dir / ".venv"
    consumer_dir.mkdir()

    _run(
        [
            uv,
            "build",
            "--wheel",
            "--no-config",
            "--out-dir",
            str(dist_dir),
            str(REPO_ROOT),
        ],
        cwd=consumer_dir,
    )
    wheels = list(dist_dir.glob("winml_cli-*.whl"))
    assert len(wheels) == 1

    _run(
        [
            uv,
            "venv",
            "--no-project",
            "--python",
            sys.executable,
            str(environment_dir),
        ],
        cwd=consumer_dir,
    )
    python = environment_dir / "Scripts" / "python.exe"
    _run(
        [
            uv,
            "pip",
            "install",
            "--no-config",
            "--python",
            str(python),
            f"{wheels[0]}[qnn]",
        ],
        cwd=consumer_dir,
    )

    probe = textwrap.dedent(
        """
        import json
        from importlib.metadata import PackageNotFoundError, metadata, version

        import onnxruntime as ort
        from winml.modelkit.ep_path import WinMLCatalogSource, _default_ep_sources

        def installed_version(name):
            try:
                return version(name)
            except PackageNotFoundError:
                return None

        qnn_catalog_configured = any(
            isinstance(source, WinMLCatalogSource)
            and "QNNExecutionProvider" in source.eps
            for source in _default_ep_sources()
        )
        provided_extras = metadata("winml-cli").get_all("Provides-Extra") or []
        providers = ort.get_available_providers()

        print(json.dumps({
            "dml_available": "DmlExecutionProvider" in providers,
            "onnxruntime": installed_version("onnxruntime"),
            "onnxruntime_qnn": installed_version("onnxruntime-qnn"),
            "onnxruntime_windowsml": installed_version("onnxruntime-windowsml"),
            "qnn_extra_provided": "qnn" in provided_extras,
            "qnn_catalog_configured": qnn_catalog_configured,
            "requirements": metadata("winml-cli").get_all("Requires-Dist") or [],
            "windowsml": installed_version("windowsml"),
        }))
        """
    )
    result = _run([str(python), "-I", "-c", probe], cwd=consumer_dir)
    installed = json.loads(result.stdout.strip())
    runtime_requirement = next(
        requirement
        for raw in installed["requirements"]
        if (requirement := Requirement(raw)).name == "onnxruntime-windowsml"
    )
    windowsml_requirement = next(
        requirement
        for raw in installed["requirements"]
        if (requirement := Requirement(raw)).name == "windowsml"
    )

    assert installed["onnxruntime"] is None
    assert installed["onnxruntime_windowsml"] in runtime_requirement.specifier
    assert installed["onnxruntime_qnn"] is None
    assert installed["windowsml"] in windowsml_requirement.specifier
    assert installed["dml_available"] is True
    assert installed["qnn_extra_provided"] is True
    assert installed["qnn_catalog_configured"] is True
