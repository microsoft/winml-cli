# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


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
            "windowsml": installed_version("windowsml"),
        }))
        """
    )
    result = _run([str(python), "-I", "-c", probe], cwd=consumer_dir)
    installed = json.loads(result.stdout.strip())

    assert installed["onnxruntime"] is None
    assert installed["onnxruntime_windowsml"].startswith("1.24.")
    assert installed["onnxruntime_qnn"] is None
    assert installed["windowsml"] == "2.0.300"
    assert installed["dml_available"] is True
    assert installed["qnn_extra_provided"] is True
    assert installed["qnn_catalog_configured"] is True
