# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for (EP, device) pair validation and default-device inference.

Regression tests for:
* Issue #513 - ``winml analyze --ep cpu`` silently uses NPU as device default.
* Issue #521 - ``winml compile --device cpu --ep qnn`` silently uses QNN/NPU.

Both bugs are rooted in the absence of a single EP-to-device support table.
The fix introduces ``EP_SUPPORTED_DEVICES`` in ``utils.constants``: an
ordered tuple per EP whose first element is the canonical default device.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def _mock_ep_registry_available():
    """Default: ``WinMLEPRegistry`` reports every EP as available.

    The compile CLI's ``_resolve_compile_provider`` consults the registry to
    reject EPs not registered on the host. Stub it to ``True`` so the policy
    tests in this file don't need to fabricate a registry; the negative-path
    tests in ``TestCompileEpAvailability`` patch the singleton locally to
    override this default.
    """
    mock_registry = MagicMock()
    mock_registry.is_ep_available.return_value = True
    with patch(
        "winml.modelkit.session.ep_registry.WinMLEPRegistry.get_instance",
        return_value=mock_registry,
    ):
        yield


# =============================================================================
# EP_SUPPORTED_DEVICES policy table
# =============================================================================


class TestEpSupportedDevices:
    """The canonical EP -> supported-device-tuple map. First element = default."""

    def test_import_available(self) -> None:
        from winml.modelkit.utils.constants import EP_SUPPORTED_DEVICES

        assert isinstance(EP_SUPPORTED_DEVICES, dict)
        assert len(EP_SUPPORTED_DEVICES) > 0

    def test_keys_match_supported_eps(self) -> None:
        from winml.modelkit.utils.constants import EP_SUPPORTED_DEVICES, SUPPORTED_EPS

        assert set(EP_SUPPORTED_DEVICES) == set(SUPPORTED_EPS)

    def test_values_are_non_empty_tuples_of_lowercase_devices(self) -> None:
        from winml.modelkit.utils.constants import EP_SUPPORTED_DEVICES, SUPPORTED_DEVICES

        valid = {d.lower() for d in SUPPORTED_DEVICES}
        for ep, devs in EP_SUPPORTED_DEVICES.items():
            assert isinstance(devs, tuple) and len(devs) > 0, ep
            assert all(d in valid for d in devs), (ep, devs)

    @pytest.mark.parametrize(
        ("ep", "default"),
        [
            ("CPUExecutionProvider", "cpu"),
            ("QNNExecutionProvider", "npu"),
            ("VitisAIExecutionProvider", "npu"),
            ("DmlExecutionProvider", "gpu"),
            ("NvTensorRTRTXExecutionProvider", "gpu"),
            ("MIGraphXExecutionProvider", "gpu"),
            # OpenVINO defaults to NPU (first entry in the supported tuple).
            ("OpenVINOExecutionProvider", "npu"),
        ],
    )
    def test_default_device_per_ep(self, ep: str, default: str) -> None:
        from winml.modelkit.utils.constants import EP_SUPPORTED_DEVICES

        assert EP_SUPPORTED_DEVICES[ep][0] == default


# =============================================================================
# Compile resolver -- incompatible (device, ep) pair rejection (Issue #521)
# =============================================================================


class TestCompileIncompatiblePair:
    """``_resolve_compile_provider`` must reject (device, ep) combos that the
    policy table marks as unsupported."""

    @pytest.mark.parametrize(
        ("device", "ep"),
        [
            ("cpu", "qnn"),
            ("cpu", "dml"),
            ("cpu", "vitisai"),
            ("cpu", "migraphx"),
            ("npu", "cpu"),
            ("gpu", "cpu"),
            ("npu", "dml"),
            ("npu", "migraphx"),
        ],
    )
    def test_incompatible_pair_rejected(self, device: str, ep: str) -> None:
        from winml.modelkit.commands.compile import _resolve_compile_provider

        with pytest.raises((click.UsageError, click.ClickException)):
            _resolve_compile_provider(device, ep)

    @pytest.mark.parametrize(
        ("device", "ep", "expected"),
        [
            ("cpu", "cpu", "CPUExecutionProvider"),
            ("npu", "qnn", "QNNExecutionProvider"),
            ("gpu", "qnn", "QNNExecutionProvider"),
            ("npu", "vitisai", "VitisAIExecutionProvider"),
            ("gpu", "dml", "DmlExecutionProvider"),
            ("gpu", "migraphx", "MIGraphXExecutionProvider"),
            ("gpu", "openvino", "OpenVINOExecutionProvider"),
        ],
    )
    def test_compatible_pair_returns_canonical(self, device: str, ep: str, expected: str) -> None:
        from winml.modelkit.commands.compile import _resolve_compile_provider

        assert _resolve_compile_provider(device, ep) == expected


# =============================================================================
# Compile resolver -- EP host-availability check
# =============================================================================


@pytest.fixture
def mock_registry_qnn_only():
    """``WinMLEPRegistry`` mock: only QNN is advertised by the host.

    Overrides the module-level ``_mock_ep_registry_available`` autouse fixture
    so we can exercise the negative path of the resolver's availability check.
    """
    mock_registry = MagicMock()
    mock_registry.is_ep_available.side_effect = lambda ep: ep == "QNNExecutionProvider"
    with patch(
        "winml.modelkit.session.ep_registry.WinMLEPRegistry.get_instance",
        return_value=mock_registry,
    ):
        yield mock_registry


class TestCompileEpAvailability:
    """``_resolve_compile_provider`` must reject EPs not registered on the
    current host (regression for the review on PR #641: silent fallback to
    QNN/CPU when the requested EP isn't installed)."""

    def test_unavailable_ep_rejected(self, mock_registry_qnn_only) -> None:
        from winml.modelkit.commands.compile import _resolve_compile_provider

        with pytest.raises(click.UsageError) as exc:
            _resolve_compile_provider("gpu", "openvino")
        msg = str(exc.value)
        assert "is not registered on this host" in msg
        # Lists what IS available so the user sees the recovery path.
        assert "QNNExecutionProvider" in msg

    def test_no_eps_available_lists_none(self) -> None:
        """When the host advertises no compile EP, the error lists ``none``."""
        from winml.modelkit.commands.compile import _resolve_compile_provider

        mock_registry = MagicMock()
        mock_registry.is_ep_available.return_value = False
        with (
            patch(
                "winml.modelkit.session.ep_registry.WinMLEPRegistry.get_instance",
                return_value=mock_registry,
            ),
            pytest.raises(click.UsageError) as exc,
        ):
            _resolve_compile_provider("npu", "qnn")
        assert "is not registered on this host" in str(exc.value)
        assert "none" in str(exc.value)

    def test_available_ep_returns_canonical(self, mock_registry_qnn_only) -> None:
        from winml.modelkit.commands.compile import _resolve_compile_provider

        assert _resolve_compile_provider("npu", "qnn") == "QNNExecutionProvider"

    def test_device_conflict_wins_over_availability(self, mock_registry_qnn_only) -> None:
        """Incompatible ``(device, ep)`` is reported before host availability —
        fixing host availability alone wouldn't make the pair valid."""
        from winml.modelkit.commands.compile import _resolve_compile_provider

        with pytest.raises(click.UsageError) as exc:
            _resolve_compile_provider("cpu", "qnn")
        assert "cannot run on" in str(exc.value)


# =============================================================================
# Analyze CLI -- compatibility check fires only when BOTH flags are explicit
# =============================================================================


@pytest.fixture
def fake_analyze_env(tmp_path, monkeypatch):
    """Stub out parquet discovery + ``ONNXStaticAnalyzer`` so the CLI runs
    end-to-end without real rule data or a real model."""
    fake_model = tmp_path / "model.onnx"
    fake_model.write_bytes(b"fake")
    parquet = tmp_path / "rules.parquet"
    parquet.touch()

    monkeypatch.setattr(
        "winml.modelkit.commands.analyze._discover_runtime_rule_parquet_files",
        lambda: ([tmp_path], [parquet]),
    )
    monkeypatch.setattr(
        "winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep",
        lambda ep, device: True,
    )

    mock_result = MagicMock()
    mock_result.is_fully_supported.return_value = True
    mock_result.output.results = []
    mock_result.to_json.return_value = "{}"

    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = mock_result

    monkeypatch.setattr(
        "winml.modelkit.analyze.ONNXStaticAnalyzer",
        lambda: mock_analyzer,
    )

    return fake_model, mock_analyzer


class TestAnalyzeDefaultDevice:
    """Analyze fills the ``--device`` default based on ``--ep`` when the user
    omits ``--device``:

    * ``(None, None)``      -> NPU (legacy fallback; analyzer scans all EPs)
    * ``(ep, None)``        -> ``EP_SUPPORTED_DEVICES[ep][0]``
    * ``(None, device)``    -> ``device`` stays; ``ep`` remains ``None`` so the
                               analyzer scans all EPs.
    """

    @pytest.mark.parametrize(
        ("ep", "expected_device"),
        [
            ("cpu", "CPU"),
            ("qnn", "NPU"),
            ("vitisai", "NPU"),
            ("dml", "GPU"),
            ("migraphx", "GPU"),
        ],
    )
    def test_ep_alone_infers_default_device(
        self, fake_analyze_env, ep: str, expected_device: str
    ) -> None:
        from winml.modelkit.commands.analyze import analyze

        model, mock_analyzer = fake_analyze_env
        result = CliRunner().invoke(analyze, ["-m", str(model), "--ep", ep, "--quiet"], obj={})

        assert result.exit_code == 0, result.output
        kwargs = mock_analyzer.analyze.call_args.kwargs
        assert kwargs["device"] == expected_device, (
            f"--ep {ep} should infer device {expected_device!r}, got {kwargs['device']!r}"
        )

    def test_no_flags_defaults_to_npu(self, fake_analyze_env) -> None:
        from winml.modelkit.commands.analyze import analyze

        model, mock_analyzer = fake_analyze_env
        result = CliRunner().invoke(analyze, ["-m", str(model), "--quiet"], obj={})

        assert result.exit_code == 0, result.output
        kwargs = mock_analyzer.analyze.call_args.kwargs
        assert kwargs["device"] == "NPU"
        # ``ep`` should remain None so the analyzer scans all EPs.
        assert kwargs["ep"] is None

    def test_device_only_keeps_ep_none(self, fake_analyze_env) -> None:
        from winml.modelkit.commands.analyze import analyze

        model, mock_analyzer = fake_analyze_env
        result = CliRunner().invoke(
            analyze, ["-m", str(model), "--device", "GPU", "--quiet"], obj={}
        )

        assert result.exit_code == 0, result.output
        kwargs = mock_analyzer.analyze.call_args.kwargs
        assert kwargs["device"] == "GPU"
        assert kwargs["ep"] is None


# Note: analyze's explicit-pair rejection is handled by the original
# ``has_rule_data_for_ep`` validation block (rule-data-driven), not by an
# ``EP_SUPPORTED_DEVICES`` policy check. Testing that path requires nuanced
# rule-data fixtures and is out of scope here.
