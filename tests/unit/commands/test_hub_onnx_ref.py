# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for Hub-hosted ONNX (``org/repo/path/to/file.onnx``) input on CLI commands.

Validates that ``wmk config`` and ``wmk build`` recognize Hub-style ONNX
references, call ``resolve_hf_onnx_path`` to download the file, and then
dispatch through the existing local-ONNX (Scenario D) code path. No actual
downloads happen -- ``hf_hub_download`` is mocked.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from winml.modelkit.session import EPDeviceTarget


if TYPE_CHECKING:
    from pathlib import Path


HUB_ONNX_REF = "onnx-community/sam3-tracker-ONNX/onnx/prompt_encoder_mask_decoder_int8.onnx"


_DEVICE_TO_EPS = {
    "npu": ["QNNExecutionProvider"],
    "gpu": ["DmlExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
}


def _fake_resolve_device(target):
    """Side effect for session.resolve_device that honours the requested device."""
    device = getattr(target, "device", "auto")
    resolved = device.lower() if device != "auto" else "npu"
    return EPDeviceTarget(ep="auto", device=resolved)


@pytest.fixture(autouse=True)
def mock_resolve_device():
    """Mock hardware detection so config/build tests run on any host.

    Build/config CLIs auto-resolve device + EP at the top of the command,
    so ``resolve_device``, ``resolve_eps``, and ``resolve_check_device_ep``
    must all be patched (mirrors ``tests/unit/commands/test_build.py``).
    """
    mock_registry = MagicMock()
    mock_registry.is_ep_available.return_value = False

    with (
        patch(
            "winml.modelkit.session.resolve_device",
            side_effect=_fake_resolve_device,
        ),
        patch(
            "winml.modelkit.session.available_eps_for_device",
            side_effect=lambda device: list(_DEVICE_TO_EPS.get(device, [])),
        ),
        patch(
            "winml.modelkit.session.ep_registry.WinMLEPRegistry.get_instance",
            return_value=mock_registry,
        ),
    ):
        yield


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def fake_local_onnx(tmp_path: Path) -> Path:
    """Fake local ONNX file the mocked downloader returns."""
    path = tmp_path / "downloaded.onnx"
    path.write_bytes(b"fake-onnx-data")
    return path


@pytest.fixture
def mock_hf_download(fake_local_onnx: Path):
    """Patch ``huggingface_hub.hf_hub_download`` to return ``fake_local_onnx``.

    Sidecar lookups raise ``EntryNotFoundError`` to simulate a model whose
    weights are inlined and has no ``.onnx_data`` companion.
    """
    from huggingface_hub.utils import EntryNotFoundError

    def _fake(*, repo_id, filename, revision, cache_dir, token):
        if filename.endswith(".onnx_data"):
            raise EntryNotFoundError(filename)
        return str(fake_local_onnx)

    with patch("huggingface_hub.hf_hub_download", side_effect=_fake) as mock:
        yield mock


@pytest.fixture
def sample_config_file(tmp_path: Path) -> Path:
    """Create a minimal JSON config file for ``wmk build``."""
    config = {
        "loader": {"task": "mask-generation"},
        "export": None,
        "optim": {},
        "quant": None,
        "compile": None,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path


# =============================================================================
# wmk config -m <hub-onnx-ref>
# =============================================================================


class TestConfigHubOnnxRef:
    """``wmk config`` recognizes Hub-style ONNX references."""

    def test_config_resolves_hub_ref_and_uses_onnx_path(
        self,
        runner: CliRunner,
        mock_hf_download: MagicMock,
        fake_local_onnx: Path,
    ) -> None:
        """Hub ref is downloaded, then config is generated via the ONNX branch."""
        from winml.modelkit.commands.config import config

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=True),
        ):
            result = runner.invoke(config, ["-m", HUB_ONNX_REF])

        assert result.exit_code == 0, f"Failed: {result.output}"
        # Resolver was invoked on the Hub reference.
        assert mock_hf_download.called
        repo_filenames = [c.kwargs["filename"] for c in mock_hf_download.call_args_list]
        assert "onnx/prompt_encoder_mask_decoder_int8.onnx" in repo_filenames
        # Output JSON marks an ONNX (Scenario D) build: export=None.
        start = result.output.index("{")
        end = result.output.rindex("}") + 1
        data = json.loads(result.output[start:end])
        assert data.get("export") is None


# =============================================================================
# wmk build -m <hub-onnx-ref>
# =============================================================================


class TestBuildHubOnnxRef:
    """``wmk build`` recognizes Hub-style ONNX references."""

    def test_build_resolves_hub_ref_and_dispatches_to_onnx_pipeline(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_hf_download: MagicMock,
        fake_local_onnx: Path,
        tmp_path: Path,
    ) -> None:
        """Hub ref is downloaded once, then build dispatches the ONNX pipeline."""
        from winml.modelkit.commands.build import build

        output_dir = tmp_path / "out"
        with patch(
            "winml.modelkit.commands.build._build_onnx_pipeline",
            return_value=[],
        ) as mock_pipeline:
            result = runner.invoke(
                build,
                ["-c", str(sample_config_file), "-m", HUB_ONNX_REF, "-o", str(output_dir)],
                obj={"debug": False},
            )

        assert result.exit_code == 0, f"Build failed: {result.output}"
        assert mock_hf_download.called
        # Pipeline was called with the locally-resolved path, not the Hub ref.
        mock_pipeline.assert_called_once()
        assert mock_pipeline.call_args.kwargs["onnx_path"] == fake_local_onnx
