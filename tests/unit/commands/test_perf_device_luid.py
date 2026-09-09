# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Physical adapter selection stays aligned with inference and monitoring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click import ClickException
from click.testing import CliRunner

from winml.modelkit.commands.perf import (
    BenchmarkConfig,
    BenchmarkResult,
    PerfBenchmark,
    _get_monitor_binding,
    _get_provider_bound_device,
    _pre_bench_kwargs_from_ep_device,
    _resolve_perf_ep_device,
    perf,
)
from winml.modelkit.ep_path import BuiltinSource, EPEntry
from winml.modelkit.session import (
    DeviceNotFound,
    EPDeviceTarget,
    UnknownListingPick,
    WinMLDevice,
    WinMLEP,
    WinMLEPNotDiscovered,
    WinMLEPRegistrationFailed,
    expand_ep_name,
)
from winml.modelkit.sysinfo import get_ep_device_luid


@pytest.fixture
def gpu_ep() -> WinMLEP:
    devices = []
    for index in range(2):
        handle = MagicMock()
        handle.ep_name = "DmlExecutionProvider"
        handle.device.type.name = "GPU"
        handle.device.metadata = {
            "LUID": str((index + 1) << 32),
            "Description": f"GPU {index}",
        }
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


@pytest.mark.parametrize("index", range(2))
@pytest.mark.parametrize("default_index", range(2))
def test_unpinned_provider_option_selects_adapter_without_warning(
    gpu_ep, index, default_index, caplog
):
    selected = gpu_ep.ep_devices()[default_index]
    options = dict(gpu_ep.devices[index].ort_handle.ep_options)
    with patch("winml.modelkit.session.WinMLEPRegistry.instance") as instance:
        instance.return_value.auto_device.return_value = selected
        result = _resolve_perf_ep_device(EPDeviceTarget(ep="dml", device="gpu"), None, options)
    assert result.device is gpu_ep.devices[index]
    assert result.ep is selected.ep
    assert result.device.hardware_name == gpu_ep.devices[index].hardware_name
    assert get_ep_device_luid(result.device.ort_handle) == get_ep_device_luid(
        gpu_ep.devices[index].ort_handle
    )
    identity = _pre_bench_kwargs_from_ep_device(
        result,
        model_id=None,
        task=None,
        opset=None,
        inputs=None,
        outputs=None,
        cached_onnx_path=None,
        onnx_file=None,
    )
    assert identity["hardware_name"] == gpu_ep.devices[index].hardware_name
    assert "Multiple devices" not in caplog.text
    assert _get_monitor_binding(result, "gpu", options) == (
        "gpu",
        get_ep_device_luid(gpu_ep.devices[index].ort_handle),
        "gpu",
    )


@pytest.mark.parametrize("index", range(2))
@pytest.mark.parametrize("default_index", range(2))
def test_provider_selection_preserves_identity_without_luid(gpu_ep, index, default_index, caplog):
    for device in gpu_ep.devices:
        device.ort_handle.device.metadata.pop("LUID")
    expected = gpu_ep.devices[index]
    options = dict(expected.ort_handle.ep_options)
    with patch("winml.modelkit.session.WinMLEPRegistry.instance") as instance:
        instance.return_value.auto_device.return_value = gpu_ep.ep_devices()[default_index]
        result = _resolve_perf_ep_device(EPDeviceTarget(ep="dml", device="gpu"), None, options)
    assert result.device is expected
    assert result.device.hardware_name == expected.hardware_name
    assert _get_provider_bound_device(result, options).selected_by_options
    assert _get_monitor_binding(result, "gpu", options) == ("cpu", None, None)
    assert "Multiple devices" not in caplog.text

    benchmark = PerfBenchmark(
        BenchmarkConfig(model_id="fake/model", device="gpu", ep="dml", ep_options=options)
    )
    benchmark._ep_device = result
    benchmark._model = SimpleNamespace(device="gpu", ep_name=expand_ep_name("dml"))
    heuristic = MagicMock()
    with (
        patch("winml.modelkit.commands.perf.sys.platform", "win32"),
        patch.dict("sys.modules", {"winml.modelkit.sysinfo.pdh_adapters": heuristic}),
    ):
        assert benchmark._resolve_adapter_luid() is None
    heuristic.resolve_adapter_luid.assert_not_called()


def test_default_adapter_without_luid_does_not_guess_monitor(gpu_ep):
    selected = gpu_ep.ep_devices()[0]
    selected.device.ort_handle.device.metadata.pop("LUID")
    assert not _get_provider_bound_device(selected).selected_by_options
    assert _get_monitor_binding(selected, "gpu", None) == ("cpu", None, None)


def test_provider_selection_matches_all_advertised_constraints(gpu_ep):
    for index, device in enumerate(gpu_ep.devices):
        device.ort_handle.ep_options["adapter_index"] = str(index)
    binding = _get_provider_bound_device(
        gpu_ep.ep_devices()[0], {"device_id": "1", "adapter_index": "0"}
    )
    assert binding.device is None
    assert not binding.selected_by_options


@pytest.mark.parametrize("module_mode", [False, True])
@pytest.mark.parametrize("exposes_gpu", [False, True])
def test_cross_kind_provider_selector_rejected_before_build(gpu_ep, module_mode, exposes_gpu):
    for index, device in enumerate(gpu_ep.devices):
        device.ort_handle.ep_name = expand_ep_name("qnn")
        device.ort_handle.ep_options = {}
        device.ort_handle.device.type.name = "NPU" if index == 0 else "GPU"
    devices = gpu_ep.devices if exposes_gpu else gpu_ep.devices[:1]
    qnn_ep = WinMLEP(
        source=EPEntry(
            ep_name=expand_ep_name("qnn"),
            dll_path=Path(),
            source=BuiltinSource(eps=(expand_ep_name("qnn"),)),
        ),
        devices=devices,
        arg0=expand_ep_name("qnn"),
    )
    with (
        patch("winml.modelkit.session.WinMLEPRegistry.instance") as instance,
        patch("winml.modelkit.config.generate_hf_build_config") as build_config,
        patch("winml.modelkit.commands.perf.PerfBenchmark._load_model") as load_model,
    ):
        instance.return_value.auto_device.return_value = qnn_ep.ep_devices()[0]
        if module_mode:
            result = CliRunner().invoke(
                perf,
                [
                    "-m",
                    "fake/model",
                    "--module",
                    "Linear",
                    "--device",
                    "npu",
                    "--ep",
                    "qnn",
                    "--ep-options",
                    "backend_type=gpu",
                ],
                catch_exceptions=False,
            )
            assert result.exit_code == 1
            assert "--ep-options select device 'gpu'" in result.output
            assert "--device gpu" in result.output
        else:
            benchmark = PerfBenchmark(
                BenchmarkConfig(
                    model_id="fake/model",
                    device="npu",
                    ep="qnn",
                    ep_options={"backend_type": "gpu"},
                )
            )
            with pytest.raises(ValueError, match="--ep-options select device 'gpu'"):
                benchmark._resolve_device_ep()
        build_config.assert_not_called()
        load_model.assert_not_called()


@pytest.mark.parametrize("options", [{"non_selector": "value"}, {"device_id": "unknown"}])
def test_unpinned_options_without_unique_adapter_keep_warning(gpu_ep, options, caplog):
    with patch("winml.modelkit.session.WinMLEPRegistry.instance") as instance:
        instance.return_value.auto_device.return_value = gpu_ep.ep_devices()[0]
        _resolve_perf_ep_device(EPDeviceTarget(ep="dml", device="gpu"), None, options)
    assert "Multiple devices" in caplog.text
    assert "using the first" not in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        DeviceNotFound("requested LUID is unavailable"),
        ValueError("--ep-options conflict with --device-luid"),
        WinMLEPNotDiscovered("EP not installed"),
        WinMLEPRegistrationFailed("DLL failed"),
        UnknownListingPick("dml", "pypi"),
    ],
)
def test_module_cli_formats_resolution_errors(gpu_ep, error):
    luid = get_ep_device_luid(gpu_ep.devices[0].ort_handle)
    with (
        patch("winml.modelkit.commands.perf._resolve_perf_ep_device", side_effect=error),
        patch("winml.modelkit.config.generate_hf_build_config") as build_config,
    ):
        result = CliRunner().invoke(
            perf,
            [
                "-m",
                "fake/model",
                "--module",
                "Linear",
                "--device",
                "gpu",
                "--ep",
                "dml",
                "--device-luid",
                luid,
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 1
    assert f"Error: Error resolving benchmark device: {error}" in result.output
    assert "Traceback" not in result.output
    build_config.assert_not_called()


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
@pytest.mark.parametrize("tuning_options", [None, {"enable_metacommands": "1"}])
def test_cli_forwards_luid(gpu_ep, tmp_path, module_mode, tuning_options):
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
        for key, value in (tuning_options or {}).items():
            args.extend(["--ep-options", f"{key}={value}"])
        result = CliRunner().invoke(perf, args)
    assert result.exit_code == 0, result.output
    if module_mode:
        assert modules.call_args.kwargs["device_luid"] == luid
        assert modules.call_args.kwargs["ep_options"] == tuning_options
    else:
        assert benchmark.call_args.args[0].device_luid == luid
        assert benchmark.call_args.args[0].ep_options == tuning_options


@pytest.mark.parametrize("module_mode", [False, True])
@pytest.mark.parametrize("device_id", ["0", "1", ""])
def test_cli_rejects_device_id_with_luid_before_resolution(gpu_ep, module_mode, device_id):
    luid = get_ep_device_luid(gpu_ep.devices[0].ort_handle)
    with (
        patch("winml.modelkit.commands.perf.cli_utils.normalize_model_arg") as normalize,
        patch("winml.modelkit.session.WinMLEPRegistry.instance") as registry,
        patch("winml.modelkit.commands.perf.PerfBenchmark") as benchmark,
        patch("winml.modelkit.commands.perf._perf_modules") as modules,
    ):
        args = [
            "-m",
            "fake/model",
            "--device-luid",
            luid,
            "--ep-options",
            f" device_id = {device_id} ",
            "--ep-options",
            "enable_metacommands=1",
        ]
        if module_mode:
            args.extend(["--module", "Linear"])
        result = CliRunner().invoke(perf, args, catch_exceptions=False)
    assert result.exit_code == 2
    assert "--device-luid cannot be combined with --ep-options device_id=" in result.output
    normalize.assert_not_called()
    registry.assert_not_called()
    benchmark.assert_not_called()
    modules.assert_not_called()


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
