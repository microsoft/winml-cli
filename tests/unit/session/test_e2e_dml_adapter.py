# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Guard the shared-agent test scope without weakening normal adapter tests."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.e2e.dml_adapter import perf_test_luid, physical_dml_test_luid


def _native(luid="physical"):
    return SimpleNamespace(device_type="GPU", luid=luid, vendor_id=1, device_id=2, name="GPU")


def _selected(*luids, different_hardware=False):
    return SimpleNamespace(
        ep=SimpleNamespace(
            devices=[
                SimpleNamespace(
                    device_type="GPU",
                    ort_handle=SimpleNamespace(
                        luid=luid,
                        device=SimpleNamespace(
                            vendor_id=1,
                            device_id=3 if different_hardware and luid == "stale" else 2,
                            metadata={"Description": "GPU"},
                        ),
                    ),
                )
                for luid in luids
            ]
        )
    )


def test_unset_flag_does_not_enumerate_or_pin(monkeypatch):
    monkeypatch.delenv("WINML_E2E_PIN_SINGLE_DML_GPU", raising=False)
    with patch("winml.modelkit.sysinfo.enumerate_compute_adapters") as enumerate_native:
        assert physical_dml_test_luid() is None
        assert perf_test_luid("dml", "gpu") is None
    enumerate_native.assert_not_called()


@pytest.mark.parametrize(
    "native,selected,error",
    [
        ([], _selected("stale"), "exactly one"),
        ([_native(), _native("second")], _selected("physical"), "exactly one"),
        ([_native()], _selected("stale"), "missing from DML"),
        ([_native()], _selected("physical", None), "publish an adapter LUID"),
        ([_native()], _selected("physical", "stale", different_hardware=True), "Unexpected"),
    ],
)
def test_invalid_inventory_still_fails(monkeypatch, native, selected, error):
    monkeypatch.setenv("WINML_E2E_PIN_SINGLE_DML_GPU", "1")
    with (
        patch("winml.modelkit.sysinfo.enumerate_compute_adapters", return_value=native),
        patch("winml.modelkit.sysinfo.get_ep_device_luid", side_effect=lambda d: d.luid),
        pytest.raises(AssertionError, match=error),
    ):
        physical_dml_test_luid(selected)


def test_only_same_hardware_duplicate_is_outside_scope(monkeypatch, caplog):
    monkeypatch.setenv("WINML_E2E_PIN_SINGLE_DML_GPU", "1")
    with (
        patch("winml.modelkit.sysinfo.enumerate_compute_adapters", return_value=[_native()]),
        patch("winml.modelkit.sysinfo.get_ep_device_luid", side_effect=lambda d: d.luid),
    ):
        assert physical_dml_test_luid(_selected("physical", "stale")) == "physical"
    assert "stale" in caplog.text


@pytest.mark.parametrize(
    "ep,device,pin",
    [
        ("DmlExecutionProvider", "gpu", "physical"),
        ("OpenVINOExecutionProvider", "gpu", "physical"),
        ("CPUExecutionProvider", "cpu", None),
    ],
)
def test_resolved_gpu_is_pinned_without_changing_ep(monkeypatch, ep, device, pin):
    monkeypatch.setenv("WINML_E2E_PIN_SINGLE_DML_GPU", "1")
    with (
        patch(
            "winml.modelkit.session.resolve_device",
            return_value=SimpleNamespace(ep=ep, device=device),
        ),
        patch("tests.e2e.dml_adapter.physical_dml_test_luid", return_value="physical") as select,
    ):
        assert perf_test_luid(None, "gpu") == pin
    assert select.call_count == int(pin is not None)


def test_manual_selectors_do_not_get_an_implicit_luid(monkeypatch, tmp_path):
    from tests.e2e.test_perf_e2e import _build_perf_args

    monkeypatch.setenv("WINML_E2E_PIN_SINGLE_DML_GPU", "1")
    with patch("tests.e2e.test_perf_e2e.perf_test_luid") as select:
        args = _build_perf_args(
            model_arg="model.onnx",
            output_file=tmp_path / "perf.json",
            ep="dml",
            device="gpu",
            use_test_gpu=False,
        )
    assert "--device-luid" not in args
    select.assert_not_called()


@pytest.mark.parametrize(
    "ep,device",
    [(None, "config"), (None, "cpu"), (None, "auto"), ("qnn", "npu")],
)
def test_unrelated_cli_targets_are_not_resolved(monkeypatch, ep, device):
    monkeypatch.setenv("WINML_E2E_PIN_SINGLE_DML_GPU", "1")
    with patch("winml.modelkit.session.resolve_device") as resolve:
        assert perf_test_luid(ep, device) is None
    resolve.assert_not_called()


def test_strict_enumeration_still_fails_on_other_agents(monkeypatch, tmp_path):
    from tests.e2e import test_perf_e2e as perf_tests

    monkeypatch.delenv("WINML_E2E_PIN_SINGLE_DML_GPU", raising=False)
    with (
        patch.object(perf_tests, "require_ep"),
        patch("winml.modelkit.session.WinMLEPRegistry.instance") as registry,
        patch("winml.modelkit.sysinfo.enumerate_compute_adapters", return_value=[_native()]),
        patch("winml.modelkit.sysinfo.get_ep_device_luid", side_effect=lambda d: d.luid),
        patch.object(perf_tests, "_run_winml_cli_subprocess") as run,
        pytest.raises(AssertionError),
    ):
        registry.return_value.auto_device.return_value = _selected("physical", "stale")
        perf_tests.TestPerfONNXDirect().test_dml_device_luid_selection(
            tmp_path, tmp_path / "model.onnx", "luid", lambda *args: None
        )
    run.assert_not_called()


def test_explicit_openvino_gpu_uses_the_same_physical_luid(monkeypatch):
    monkeypatch.setenv("WINML_E2E_PIN_SINGLE_DML_GPU", "1")
    with (
        patch(
            "winml.modelkit.session.resolve_device",
            return_value=SimpleNamespace(ep="OpenVINOExecutionProvider", device="gpu"),
        ) as resolve,
        patch("tests.e2e.dml_adapter.physical_dml_test_luid", return_value="physical"),
    ):
        assert perf_test_luid("openvino", "gpu") == "physical"
    assert resolve.call_args.args[0].ep == "openvino"
