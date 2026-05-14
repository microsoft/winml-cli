# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E tests for the ``winml sys`` CLI command.

Two things motivate a separate e2e file (vs. the existing CLI tests in
``tests/cli/test_main.py::TestSysCommand``):

1. The CLI tests mock ``_gather_device_info`` / ``_gather_ep_info`` to
   keep them fast. This file deliberately does **not** mock those, so the
   real PowerShell / WMI / PnP probe path is exercised — that catches
   integration bugs (wrong EP→device mappings, broken WMI queries on
   specific hardware, missing PnP properties) that mocked CLI tests
   cannot.
2. PR #595 (issue #558) cut warm latency 2-3x by avoiding ``import torch``
   on the default path and gating CUDA fields behind ``--verbose``. Those
   are observable JSON/text contracts that need a regression guard.

Joint coverage for issues #506 (functional E2E) and #507 (feature-owner
self-check).

Markers:
    e2e: Auto-skipped unless ``-m e2e`` is passed.
"""

from __future__ import annotations

import json
import re
from typing import ClassVar

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.sys import sysinfo


pytestmark = [pytest.mark.e2e]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_sys(*args: str) -> tuple[int, str]:
    """Invoke ``winml sys`` with the given args; return ``(exit_code, output)``."""
    runner = CliRunner()
    result = runner.invoke(sysinfo, list(args), obj={})
    return result.exit_code, result.output


def _run_sys_json(*args: str) -> dict:
    """Invoke ``winml sys ... --format json`` and return the parsed JSON payload."""
    runner = CliRunner()
    result = runner.invoke(sysinfo, [*args, "--format", "json"], obj={}, catch_exceptions=False)
    assert result.exit_code == 0, f"sys failed (exit {result.exit_code}):\n{result.output}"
    return json.loads(result.output)


def _torch_installed() -> bool:
    from importlib.metadata import PackageNotFoundError, version

    try:
        version("torch")
    except PackageNotFoundError:
        return False
    return True


# ---------------------------------------------------------------------------
# Behavioral contracts from PR #595 (issue #558)
# ---------------------------------------------------------------------------


class TestSysTorchContract:
    """Pin the torch-version / verbose-gating contract introduced in #595.

    These behaviors are how PR #595 cut warm latency 2-3x by avoiding
    ``import torch`` on the default path. Regressing them re-introduces
    the original 5-6 s latency bug.
    """

    def test_default_json_torch_version_matches_importlib_metadata(self):
        """Default JSON: ``torch.version`` comes from ``importlib.metadata``.

        Asserting equality with ``metadata.version("torch")`` catches a
        regression to ``torch.__version__`` (which adds ``+cpu``/``+cu118``
        local-version segments in some environments) without making the
        assertion environment-specific.
        """
        if not _torch_installed():
            pytest.skip("torch not installed")
        from importlib.metadata import version

        data = _run_sys_json()
        assert data["torch"]["available"] is True
        assert data["torch"]["version"] == version("torch")

    def test_default_json_omits_cuda_field(self):
        """Default JSON: ``torch.cuda_available`` is gated behind ``--verbose``."""
        data = _run_sys_json()
        assert "cuda_available" not in data["torch"], (
            "torch.cuda_available leaked into the default --format json output; "
            "PR #595 gates CUDA detection behind --verbose to avoid `import torch`."
        )

    def test_verbose_json_includes_cuda_field(self):
        """``--verbose --format json``: ``torch.cuda_available`` is populated."""
        if not _torch_installed():
            pytest.skip("torch not installed")
        data = _run_sys_json("--verbose")
        assert "cuda_available" in data["torch"], (
            "--verbose --format json should populate torch.cuda_available"
        )
        assert isinstance(data["torch"]["cuda_available"], bool)

    def test_default_text_omits_pytorch_details_panel(self):
        """Default text output drops the ``PyTorch Details`` panel."""
        exit_code, output = _run_sys()
        assert exit_code == 0, output
        assert "PyTorch Details" not in output, (
            "Default text output should not render the PyTorch Details panel "
            "(PR #595 removed it to avoid `import torch`)."
        )

    def test_verbose_text_restores_pytorch_details_panel(self):
        """``--verbose`` text output restores the ``PyTorch Details`` panel."""
        if not _torch_installed():
            pytest.skip("torch not installed")
        exit_code, output = _run_sys("--verbose")
        assert exit_code == 0, output
        assert "PyTorch Details" in output, "--verbose should restore the PyTorch Details panel"


# ---------------------------------------------------------------------------
# JSON shape contracts
# ---------------------------------------------------------------------------


class TestSysJsonShape:
    """Validate the JSON output shape that downstream scripts depend on."""

    REQUIRED_TOP_KEYS: ClassVar[set[str]] = {
        "python",
        "platform",
        "libraries",
        "torch",
        "backends",
        "export_readiness",
        "devices",
        "executionProviders",
    }

    def test_default_json_has_required_top_keys(self):
        data = _run_sys_json()
        missing = self.REQUIRED_TOP_KEYS - data.keys()
        assert not missing, f"Missing top-level keys: {missing}"

    def test_default_json_python_version_is_well_formed(self):
        data = _run_sys_json()
        assert re.match(r"^\d+\.\d+\.\d+", data["python"]["version"])

    def test_default_json_devices_have_sequential_priority(self):
        """Devices come back with sequential 1-based priorities."""
        data = _run_sys_json()
        devices = data["devices"]
        assert isinstance(devices, list)
        if devices:
            priorities = [d["priority"] for d in devices]
            assert priorities == list(range(1, len(priorities) + 1)), (
                f"Device priorities are not sequential 1..N: {priorities}"
            )

    def test_default_json_eps_have_name_device_path(self):
        data = _run_sys_json()
        eps = data["executionProviders"]
        assert isinstance(eps, list)
        for ep in eps:
            assert set(ep.keys()) >= {"name", "device", "path"}
            assert isinstance(ep["name"], str) and ep["name"]
            assert isinstance(ep["device"], str)


# ---------------------------------------------------------------------------
# Flag variations
# ---------------------------------------------------------------------------


class TestSysFlagVariations:
    """Exercise each behavior-bearing flag (#506 acceptance criterion)."""

    def test_format_text_default(self):
        exit_code, output = _run_sys()
        assert exit_code == 0
        # Default text contains a rendered "Python Version" row.
        assert "Python" in output

    def test_format_compact(self):
        exit_code, output = _run_sys("--format", "compact")
        assert exit_code == 0
        # Compact output is a fixed line layout starting with "Python: ...".
        assert output.lstrip().startswith("Python:"), (
            f"Unexpected compact output prefix: {output[:80]!r}"
        )

    def test_list_device_only_json_single_key(self):
        data = _run_sys_json("--list-device")
        assert set(data.keys()) == {"devices"}, (
            f"--list-device --format json should yield only 'devices', got {list(data)}"
        )

    def test_list_ep_only_json_single_key(self):
        data = _run_sys_json("--list-ep")
        assert set(data.keys()) == {"executionProviders"}, (
            f"--list-ep --format json should yield only 'executionProviders', got {list(data)}"
        )

    def test_list_device_and_ep_json_is_single_object(self):
        """``--list-device --list-ep --format json`` yields one valid JSON object.

        Regression guard: an earlier shape printed two concatenated JSON
        objects which is not valid JSON. The single-object form is the
        documented contract.
        """
        data = _run_sys_json("--list-device", "--list-ep")
        assert set(data.keys()) == {"devices", "executionProviders"}

    def test_list_device_compact(self):
        exit_code, output = _run_sys("--list-device", "--format", "compact")
        assert exit_code == 0
        assert output.strip(), "expected non-empty compact device output"

    def test_list_ep_compact(self):
        exit_code, output = _run_sys("--list-ep", "--format", "compact")
        assert exit_code == 0
        assert output.lstrip().startswith("EPs:"), f"Expected 'EPs:' prefix, got: {output[:80]!r}"


# ---------------------------------------------------------------------------
# Cross-view consistency (regression guard for #559-class bugs)
# ---------------------------------------------------------------------------


class TestSysCrossViewConsistency:
    """The same EP / device should report the same fields regardless of which
    view surfaced it. Mismatch between ``winml sys`` and ``winml sys
    --list-ep`` is the failure mode that motivated this guard."""

    def test_eps_match_across_default_and_list_ep_views(self):
        default = _run_sys_json()
        list_ep = _run_sys_json("--list-ep")
        assert default["executionProviders"] == list_ep["executionProviders"]

    def test_devices_match_across_default_and_list_device_views(self):
        default = _run_sys_json()
        list_dev = _run_sys_json("--list-device")
        assert default["devices"] == list_dev["devices"]


# ---------------------------------------------------------------------------
# Bad path
# ---------------------------------------------------------------------------


class TestSysBadPath:
    def test_invalid_format_value_rejected_cleanly(self):
        """An unknown ``--format`` value exits non-zero without a traceback."""
        exit_code, output = _run_sys("--format", "bogus")
        assert exit_code != 0
        assert "Traceback (most recent call last)" not in output
        # Click's friendly error for a bad Choice value.
        assert "bogus" in output.lower() or "invalid" in output.lower()
