# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Physical adapter selection stays aligned with inference and monitoring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click import ClickException
from click.testing import CliRunner

from winml.modelkit.commands.perf import (
    BenchmarkConfig,
    BenchmarkResult,
    PerfBenchmark,
    _get_monitor_binding,
    _resolve_perf_ep_device,
    perf,
)
from winml.modelkit.ep_path import BuiltinSource, EPEntry
from winml.modelkit.session import EPDeviceTarget, WinMLDevice, WinMLEP, expand_ep_name
from winml.modelkit.sysinfo import get_ep_device_luid


@pytest.fixture
def gpu_ep() -> WinMLEP:
    devices = []
    for index in range(2):
        handle = MagicMock()
        handle.ep_name = "DmlExecutionProvider"
        handle.device.type.name = "GPU"
        handle.device.metadata = {"LUID": str((index + 1) << 32)}
        handle.ep_metadata = {}
        handle.ep_options = {"device_id": str(index)}
        devices.append(WinMLDevice(handle))
    return WinMLEP(
        source=EPEntry(
            ep_name=devices[0].ep_name,
            dll_path=Path(),
            source=BuiltinSource(eps=(devices[0].ep_name,)),
        ),
        devices=tuple(devices),
        arg0=devices[0].ep_name,
    )


@pytest.mark.parametrize("index", range(2))
def test_pinned_binding_and_monitor_use_same_device(gpu_ep, index, caplog):
    selected = gpu_ep.ep_devices()[index]
    luid = get_ep_device_luid(selected.device.ort_handle)
    target = EPDeviceTarget(ep="dml", device="gpu")
    with patch("winml.modelkit.session.WinMLEPRegistry.instance") as instance:
        instance.return_value.auto_device.return_value = selected
        assert _resolve_perf_ep_device(target, luid, {"non_selector": "value"}) is selected
        instance.return_value.auto_device.assert_called_once_with(target, device_luid=luid)
    assert _get_monitor_binding(selected, "gpu", None) == ("gpu", luid, "gpu")
    assert "Multiple devices" not in caplog.text


@pytest.mark.parametrize("selector", ["0", "unknown"])
def test_conflicting_provider_selector_rejected(gpu_ep, selector):
    selected = gpu_ep.ep_devices()[1]
    luid = get_ep_device_luid(selected.device.ort_handle)
    with (
        patch("winml.modelkit.session.WinMLEPRegistry.instance") as instance,
        pytest.raises(ValueError, match="--ep-options conflict with --device-luid"),
    ):
        instance.return_value.auto_device.return_value = selected
        _resolve_perf_ep_device(
            EPDeviceTarget(ep="dml", device="gpu"), luid, {"device_id": selector}
        )


@pytest.mark.parametrize(
    "inventory", ["multiple", "single", "same-luid", "missing-luids", "other-kind"]
)
def test_unpinned_warning_only_for_multiple_adapters(gpu_ep, inventory, caplog):
    if inventory == "single":
        gpu_ep = WinMLEP(source=gpu_ep.source, devices=gpu_ep.devices[:1], arg0=gpu_ep.arg0)
    elif inventory == "same-luid":
        gpu_ep.devices[1].ort_handle.device.metadata = dict(
            gpu_ep.devices[0].ort_handle.device.metadata
        )
    elif inventory == "missing-luids":
        for device in gpu_ep.devices:
            device.ort_handle.device.metadata = {}
    elif inventory == "other-kind":
        gpu_ep.devices[1].ort_handle.device.type.name = "NPU"
    selected = gpu_ep.ep_devices()[0]
    with patch("winml.modelkit.session.WinMLEPRegistry.instance") as instance:
        instance.return_value.auto_device.return_value = selected
        assert (
            _resolve_perf_ep_device(EPDeviceTarget(ep="dml", device="gpu"), None, None) is selected
        )
    should_warn = inventory in ("multiple", "missing-luids")
    assert ("Multiple devices" in caplog.text) == should_warn
    if should_warn:
        assert "--device-luid" in caplog.text
        assert "winml sys" in caplog.text


def test_benchmark_resolves_pin_once_before_loading(gpu_ep):
    selected = gpu_ep.ep_devices()[1]
    luid = get_ep_device_luid(selected.device.ort_handle)
    config = BenchmarkConfig(model_id="fake/model", ep="dml", device="gpu", device_luid=luid)
    benchmark = PerfBenchmark(config)
    with patch("winml.modelkit.session.WinMLEPRegistry.instance") as instance:
        instance.return_value.auto_device.return_value = selected
        benchmark._resolve_device_ep()
        benchmark._resolve_device_ep()
        instance.return_value.auto_device.assert_called_once_with(
            EPDeviceTarget(ep=expand_ep_name("dml"), device="gpu"), device_luid=luid
        )
    assert benchmark._ep_device is selected
    assert BenchmarkResult(config=config).to_dict()["benchmark_info"]["device_luid"] == luid


def test_module_resolves_pin_before_build(gpu_ep):
    from winml.modelkit.commands.perf import _perf_modules

    selected = gpu_ep.ep_devices()[1]
    luid = get_ep_device_luid(selected.device.ort_handle)
    with (
        patch("winml.modelkit.session.WinMLEPRegistry.instance") as instance,
        patch(
            "winml.modelkit.config.generate_hf_build_config",
            side_effect=RuntimeError("stop before build"),
        ),
        pytest.raises(ClickException, match="stop before build"),
    ):
        instance.return_value.auto_device.return_value = selected
        _perf_modules(
            hf_model="fake/model",
            module_class="Linear",
            task=None,
            iterations=1,
            warmup=0,
            batch_size=1,
            no_quantize=True,
            no_optimize=True,
            no_analyze=True,
            max_optim_iterations=None,
            no_compile=True,
            output=None,
            verbose=False,
            console=MagicMock(),
            device="gpu",
            ep="dml",
            device_luid=luid,
        )
    instance.return_value.auto_device.assert_called_once_with(
        EPDeviceTarget(ep=expand_ep_name("dml"), device="gpu"), device_luid=luid
    )


@pytest.mark.parametrize("module_mode", [False, True])
def test_cli_forwards_luid(gpu_ep, tmp_path, module_mode):
    luid = get_ep_device_luid(gpu_ep.devices[1].ort_handle)
    assert luid is not None
    with (
        patch("winml.modelkit.commands.perf.PerfBenchmark") as benchmark,
        patch("winml.modelkit.commands.perf._perf_modules") as modules,
        patch("winml.modelkit.commands.perf.display_console_report"),
        patch("winml.modelkit.commands.perf.write_json_report"),
    ):
        args = [
            "-m",
            "fake/model",
            "--device",
            "gpu",
            "--ep",
            "dml",
            "--device-luid",
            luid,
            "-o",
            str(tmp_path / "result.json"),
        ]
        if module_mode:
            args.extend(["--module", "Linear"])
        result = CliRunner().invoke(perf, args)
    assert result.exit_code == 0, result.output
    if module_mode:
        assert modules.call_args.kwargs["device_luid"] == luid
    else:
        assert benchmark.call_args.args[0].device_luid == luid


@pytest.mark.parametrize("invalid", ["garbage", "0", "0x1_0x2", "0x00000000_0x0000000G"])
def test_cli_rejects_invalid_luid_before_model_loading(invalid):
    with patch("winml.modelkit.commands.perf.PerfBenchmark") as benchmark:
        result = CliRunner().invoke(perf, ["-m", "fake/model", "--device-luid", invalid])
    assert result.exit_code == 2
    assert "winml sys" in result.output
    benchmark.assert_not_called()


def test_genai_rejects_pin(gpu_ep):
    luid = get_ep_device_luid(gpu_ep.devices[0].ort_handle)
    with patch("winml.modelkit.commands.perf._run_genai_runtime") as genai:
        result = CliRunner().invoke(
            perf, ["-m", "fake/model", "--runtime", "ort-genai", "--device-luid", luid]
        )
    assert result.exit_code == 2
    assert "--device-luid is not supported" in result.output
    genai.assert_not_called()
