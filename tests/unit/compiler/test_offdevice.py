# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for virtual-target discovery and compile-only artifact transactions."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import ref

import onnx
import onnxruntime as ort
import pytest
from google.protobuf.message import DecodeError

from tests.fixtures.create_test_models import create_simple_conv_model
from tests.unit.compiler.test_compiler_stages import create_epcontext_onnx, create_simple_model
from winml.modelkit.compiler import (
    compile_off_device,
    list_compilation_eps,
    list_compilation_targets,
    offdevice,
)
from winml.modelkit.ep_path import DirectorySource, EPEntry


@pytest.fixture
def payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Supply one discovered payload and mocked native registration operations."""
    provider = "OpenVINOExecutionProvider"
    dll = tmp_path / "onnxruntime_providers_openvino_plugin.dll"
    dll.touch()
    entry = EPEntry(
        ep_name=provider,
        dll_path=dll,
        source=DirectorySource(root=tmp_path, dll_patterns={provider: dll.name}),
    )
    discovery = Mock(return_value=[entry])
    register = Mock()
    unregister = Mock()
    device = SimpleNamespace(
        ep_name=provider,
        ep_metadata={"library_path": str(dll), "npu_platform": "4000"},
        ep_options={},
        device=SimpleNamespace(
            type=ort.OrtHardwareDeviceType.NPU,
            vendor_id=0x8086,
            device_id=25662,
            metadata={"is_virtual": "1"},
        ),
    )
    get_devices = Mock(return_value=[device])
    monkeypatch.setattr(offdevice, "discover_all_eps", discovery)
    monkeypatch.setattr(offdevice, "_parse_winmlcli_ep_path", lambda name: [])
    monkeypatch.setattr(offdevice, "_suppress_dll_load_dialogs", nullcontext)
    monkeypatch.setattr(offdevice.ort, "register_execution_provider_library", register)
    monkeypatch.setattr(offdevice.ort, "unregister_execution_provider_library", unregister)
    monkeypatch.setattr(offdevice.ort, "get_ep_devices", get_devices)
    return SimpleNamespace(
        entry=entry,
        device=device,
        discovery=discovery,
        register=register,
        unregister=unregister,
        get_devices=get_devices,
    )


def test_inventory_does_not_register_or_probe_devices(payload: SimpleNamespace) -> None:
    rows = list_compilation_eps()
    assert [row["provider_name"] for row in rows] == [
        "OpenVINOExecutionProvider",
        "QNNExecutionProvider",
        "VitisAIExecutionProvider",
    ]
    assert rows[0]["payload_available"] is True
    assert rows[1]["payload_available"] is False
    assert rows[0]["payloads"][0]["library_path"] == str(payload.entry.dll_path)
    payload.register.assert_not_called()
    payload.get_devices.assert_not_called()


def test_missing_payload_is_inventory_state_but_target_error(payload: SimpleNamespace) -> None:
    payload.discovery.return_value = []
    rows = list_compilation_eps()
    assert rows[0]["payload_available"] is False
    assert rows[0]["payloads"] == []
    with pytest.raises(ValueError, match="No OpenVINO compiler payload"):
        list_compilation_targets("openvino")
    payload.register.assert_not_called()


def test_targets_filter_deduplicate_and_copy_metadata(payload: SimpleNamespace) -> None:
    physical = deepcopy(payload.device)
    physical.device.metadata["is_virtual"] = "0"
    other_ep = deepcopy(payload.device)
    other_ep.ep_name = "QNNExecutionProvider"
    other_library = deepcopy(payload.device)
    other_library.ep_metadata["library_path"] = str(payload.entry.dll_path.with_name("other.dll"))
    gpu = deepcopy(payload.device)
    gpu.device.type = ort.OrtHardwareDeviceType.GPU
    second_target = deepcopy(payload.device)
    second_target.ep_metadata["npu_platform"] = "5010"
    payload.get_devices.return_value = [
        second_target,
        physical,
        other_ep,
        other_library,
        gpu,
        payload.device,
        payload.device,
    ]
    expected_metadata = dict(payload.device.ep_metadata)
    rows = list_compilation_targets("openvino")
    assert [row["npu_platform"] for row in rows] == sorted(
        [expected_metadata["npu_platform"], second_target.ep_metadata["npu_platform"]]
    )
    assert rows[0]["metadata"] == expected_metadata
    payload.device.ep_metadata["npu_platform"] = "changed"
    assert rows[0]["metadata"] == expected_metadata
    registration, library = payload.register.call_args.args
    assert registration.startswith(payload.entry.ep_name)
    assert registration.endswith(".virtual")
    assert Path(library) == payload.entry.dll_path
    payload.unregister.assert_called_once_with(registration)


def test_no_virtual_target_is_failure_not_physical_fallback(payload: SimpleNamespace) -> None:
    payload.device.device.metadata["is_virtual"] = "0"
    with pytest.raises(ValueError, match="No physical-device fallback"):
        list_compilation_targets("openvino")
    payload.unregister.assert_called_once()


@pytest.fixture
def qnn_payload(compilation: SimpleNamespace, payload: SimpleNamespace) -> SimpleNamespace:
    provider = "QNNExecutionProvider"
    dll = payload.entry.dll_path.with_name("onnxruntime_providers_qnn.dll")
    dll.touch()
    dll.with_name("QnnHtp.dll").touch()
    payload.entry = EPEntry(
        ep_name=provider,
        dll_path=dll,
        source=DirectorySource(root=dll.parent, dll_patterns={provider: dll.name}),
    )
    payload.discovery.return_value = [payload.entry]
    payload.device.ep_name = provider
    payload.device.ep_metadata = {"library_path": str(dll)}
    payload.device.device.type = ort.OrtHardwareDeviceType.CPU
    payload.device.device.metadata = {"Description": "Compilation host"}
    for attribute in compilation.context.graph.node[0].attribute:
        if attribute.name == "source":
            attribute.s = provider.encode("utf-8")
    return payload


def test_qnn_inventory_does_not_invent_targets_or_load_the_payload(
    qnn_payload: SimpleNamespace,
) -> None:
    rows = list_compilation_eps("qnn")
    assert rows[0]["provider_name"] == "QNNExecutionProvider"
    assert rows[0]["payload_available"] is True
    assert rows[0]["compilation_device_type"] == "cpu"
    assert rows[0]["target_parameter"] == "soc_model"
    assert rows[0]["targets_enumerable"] is False
    with pytest.raises(ValueError, match="QNN does not enumerate SoC targets"):
        list_compilation_targets("qnn")
    qnn_payload.register.assert_not_called()
    qnn_payload.get_devices.assert_not_called()


def test_qnn_binds_cpu_host_to_explicit_htp_backend_and_soc(
    compilation: SimpleNamespace, qnn_payload: SimpleNamespace
) -> None:
    physical = deepcopy(qnn_payload.device)
    physical.device.type = ort.OrtHardwareDeviceType.NPU
    other_library = deepcopy(qnn_payload.device)
    other_library.ep_metadata["library_path"] = str(
        qnn_payload.entry.dll_path.with_name("other.dll")
    )
    other_provider = deepcopy(qnn_payload.device)
    other_provider.ep_name = "CPUExecutionProvider"
    qnn_payload.get_devices.return_value = [
        physical,
        other_library,
        other_provider,
        qnn_payload.device,
    ]
    result = compile_off_device(
        compilation.model, compilation.output, "60", ep_name="qnn", source="directory"
    )
    assert result.is_file()
    assert compilation.bindings == [
        (
            (qnn_payload.device,),
            {
                "backend_path": str(qnn_payload.entry.dll_path.with_name("QnnHtp.dll")),
                "soc_model": "60",
            },
        )
    ]
    compilation.options.add_session_config_entry.assert_called_once_with(
        "session.disable_cpu_ep_fallback", "1"
    )
    qnn_payload.register.assert_called_once_with(
        "QNNExecutionProvider", str(qnn_payload.entry.dll_path)
    )
    qnn_payload.unregister.assert_called_once_with("QNNExecutionProvider")
    assert onnx.load(result).graph == compilation.context.graph


@pytest.mark.parametrize("target", ["", "0", "-1", "SM8650", "60,88", "\uff16\uff10"])
def test_qnn_requires_explicit_numeric_soc_before_registration(
    target: str, compilation: SimpleNamespace, qnn_payload: SimpleNamespace
) -> None:
    with pytest.raises(ValueError, match="positive numeric soc_model"):
        compile_off_device(compilation.model, compilation.output, target, ep_name="qnn")
    qnn_payload.register.assert_not_called()
    compilation.factory.assert_not_called()


@pytest.mark.parametrize(
    "case", ["missing-payload", "wrong-source", "physical-only", "ambiguous", "missing-htp"]
)
def test_qnn_unavailable_compiler_never_falls_back(
    case: str, compilation: SimpleNamespace, qnn_payload: SimpleNamespace
) -> None:
    if case == "missing-payload":
        qnn_payload.discovery.return_value = []
    elif case == "physical-only":
        qnn_payload.device.device.type = ort.OrtHardwareDeviceType.NPU
    elif case == "ambiguous":
        qnn_payload.get_devices.return_value *= 2
    elif case == "missing-htp":
        qnn_payload.entry.dll_path.with_name("QnnHtp.dll").unlink()
    source = "pypi" if case == "wrong-source" else None
    with pytest.raises((ValueError, FileNotFoundError)):
        compile_off_device(
            compilation.model, compilation.output, "60", ep_name="qnn", source=source
        )
    compilation.factory.assert_not_called()
    assert not compilation.output.exists()
    assert not list(compilation.output.parent.iterdir())
    if case in {"missing-payload", "wrong-source"}:
        qnn_payload.register.assert_not_called()
    else:
        qnn_payload.unregister.assert_called_once_with("QNNExecutionProvider")


def test_qnn_registration_collision_never_unregisters_another_owner(
    compilation: SimpleNamespace, qnn_payload: SimpleNamespace
) -> None:
    qnn_payload.register.side_effect = RuntimeError("library already registered")
    compilation.output.parent.mkdir()
    compilation.output.write_bytes(b"previous output")
    with pytest.raises(RuntimeError, match="already registered"):
        compile_off_device(
            compilation.model, compilation.output, "60", ep_name="qnn", overwrite=True
        )
    qnn_payload.unregister.assert_not_called()
    compilation.factory.assert_not_called()
    assert compilation.output.read_bytes() == b"previous output"


def test_qnn_rejects_context_with_temporary_provider_identity(
    compilation: SimpleNamespace, qnn_payload: SimpleNamespace
) -> None:
    for attribute in compilation.context.graph.node[0].attribute:
        if attribute.name == "source":
            attribute.s = b"QNNExecutionProvider_temporary.virtual"
    with pytest.raises(ValueError, match="embedded QNNExecutionProvider"):
        compile_off_device(compilation.model, compilation.output, "60", ep_name="qnn")
    qnn_payload.unregister.assert_called_once_with("QNNExecutionProvider")
    assert not compilation.output.exists()


def test_missing_platform_fails_and_unregisters(payload: SimpleNamespace) -> None:
    del payload.device.ep_metadata["npu_platform"]
    with pytest.raises(ValueError, match="missing npu_platform"):
        list_compilation_targets("openvino")
    payload.unregister.assert_called_once()


def test_enumeration_error_still_unregisters(payload: SimpleNamespace) -> None:
    payload.get_devices.side_effect = RuntimeError("device enumeration failed")
    with pytest.raises(RuntimeError, match="device enumeration failed"):
        list_compilation_targets("openvino")
    payload.unregister.assert_called_once()


def test_registration_error_does_not_unregister_unowned_library(payload: SimpleNamespace) -> None:
    payload.register.side_effect = RuntimeError("DLL registration failed")
    with pytest.raises(RuntimeError, match="DLL registration failed"):
        list_compilation_targets("openvino")
    payload.unregister.assert_not_called()


def test_source_pin_does_not_fall_back_to_another_source(payload: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="No OpenVINO compiler payload"):
        list_compilation_targets("openvino", source="pypi")
    payload.register.assert_not_called()


def test_unsupported_provider_is_rejected_before_discovery(payload: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="only OpenVINO, QNN and VitisAI NPU"):
        list_compilation_targets("dml")
    payload.discovery.assert_not_called()


@pytest.fixture
def compilation(
    payload: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    model_path = tmp_path / "model.onnx"
    create_simple_model(model_path)
    template = tmp_path / "template.onnx"
    create_epcontext_onnx(template, "compiled context", embed_mode=1)
    context = onnx.load(template)
    context.graph.node[0].attribute.append(
        onnx.helper.make_attribute("source", payload.entry.ep_name)
    )
    options = Mock()
    bindings = []
    options.add_provider_for_devices.side_effect = lambda devices, values: bindings.append(
        (tuple(devices), dict(values))
    )
    native = Mock()

    def write_output(path: str) -> None:
        payload.register.assert_called()
        payload.unregister.assert_not_called()
        onnx.save(context, path)

    native.compile_to_file.side_effect = write_output
    factory = Mock(return_value=native)
    monkeypatch.setattr(ort, "SessionOptions", Mock(return_value=options))
    monkeypatch.setattr(ort, "ModelCompiler", factory)
    monkeypatch.setattr(
        ort, "InferenceSession", Mock(side_effect=AssertionError("Inference is forbidden"))
    )
    return SimpleNamespace(
        model=model_path,
        output=tmp_path / "output" / "compiled.onnx",
        context=context,
        options=options,
        bindings=bindings,
        native=native,
        factory=factory,
    )


@pytest.fixture
def vitisai_payload(compilation: SimpleNamespace, payload: SimpleNamespace) -> SimpleNamespace:
    provider = "VitisAIExecutionProvider"
    dll = payload.entry.dll_path.with_name("onnxruntime_vitisai_ep.dll")
    dll.touch()
    payload.entry = EPEntry(
        ep_name=provider,
        dll_path=dll,
        source=DirectorySource(root=dll.parent, dll_patterns={provider: dll.name}),
    )
    payload.discovery.return_value = [payload.entry]
    payload.device.ep_name = provider
    payload.device.ep_metadata = {"library_path": str(dll)}
    payload.device.device.vendor_id = 0x1022
    for attribute in compilation.context.graph.node[0].attribute:
        if attribute.name == "source":
            attribute.s = provider.encode("utf-8")
    return payload


def test_vitisai_inventory_does_not_invent_architectures_or_load_dll(
    vitisai_payload: SimpleNamespace,
) -> None:
    row = list_compilation_eps("vitisai")[0]
    assert row["provider_name"] == "VitisAIExecutionProvider"
    assert row["payload_available"] is True
    assert row["compilation_device_type"] == "virtual_npu"
    assert row["target_parameter"] == "target"
    assert row["targets_enumerable"] is False
    with pytest.raises(ValueError, match="not a destination SoC"):
        list_compilation_targets("vitisai")
    vitisai_payload.register.assert_not_called()
    vitisai_payload.get_devices.assert_not_called()


def test_unrecognized_dll_inventory_requires_explicit_provider(
    vitisai_payload: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def overrides(provider: str | None) -> list:
        return [vitisai_payload.entry] if provider == "VitisAIExecutionProvider" else []

    monkeypatch.setattr(offdevice, "_parse_winmlcli_ep_path", overrides)
    vitisai_payload.discovery.side_effect = lambda extra_sources: extra_sources
    assert all(not row["payload_available"] for row in list_compilation_eps())
    assert list_compilation_eps("vitisai")[0]["payload_available"] is True
    vitisai_payload.register.assert_not_called()


@pytest.mark.parametrize("overwrite", [False, True])
def test_vitisai_binds_only_owned_virtual_npu_and_cleans_up_before_publish(
    overwrite: bool,
    compilation: SimpleNamespace,
    vitisai_payload: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary_directory = Mock(wraps=offdevice.TemporaryDirectory)
    monkeypatch.setattr(offdevice, "TemporaryDirectory", temporary_directory)
    physical = deepcopy(vitisai_payload.device)
    physical.device.metadata["is_virtual"] = "0"
    bootstrap_device = deepcopy(vitisai_payload.device)
    bootstrap_device.ep_name += "_temporary.virtual"
    other_library = deepcopy(vitisai_payload.device)
    other_library.ep_metadata["library_path"] = str(
        vitisai_payload.entry.dll_path.with_name("other.dll")
    )
    cpu = deepcopy(vitisai_payload.device)
    cpu.device.type = ort.OrtHardwareDeviceType.CPU
    vitisai_payload.get_devices.return_value = [
        physical,
        bootstrap_device,
        other_library,
        cpu,
        vitisai_payload.device,
    ]
    if overwrite:
        compilation.output.parent.mkdir()
        compilation.output.write_bytes(b"previous output")

    def unregister(_name: str) -> None:
        if overwrite:
            assert compilation.output.read_bytes() == b"previous output"
        else:
            assert not compilation.output.exists()

    vitisai_payload.unregister.side_effect = unregister
    result = compile_off_device(
        compilation.model,
        compilation.output,
        "X2",
        ep_name="vitisai",
        source="directory",
        overwrite=overwrite,
    )
    assert onnx.load(result).graph == compilation.context.graph
    bound_devices, provider_options = compilation.bindings[0]
    assert len(compilation.bindings) == 1
    assert bound_devices == (vitisai_payload.device,)
    staged = Path(compilation.native.compile_to_file.call_args.args[0])
    assert provider_options == {"target": "X2", "enable_cache_file_io_in_mem": "1"}
    temporary_directory.assert_called_once()
    assert staged.parent.parent == compilation.output.parent
    assert not staged.parent.exists()
    compilation.options.add_session_config_entry.assert_called_once_with(
        "session.disable_cpu_ep_fallback", "1"
    )
    names = [call.args[0] for call in vitisai_payload.register.call_args_list]
    assert len(names) == 2
    assert names[0].startswith("VitisAIExecutionProvider_")
    assert names[0].endswith(".virtual")
    assert names[1] == "VitisAIExecutionProvider"
    assert all(
        call.args[1] == str(vitisai_payload.entry.dll_path)
        for call in vitisai_payload.register.call_args_list
    )
    assert [call.args[0] for call in vitisai_payload.unregister.call_args_list] == names[::-1]
    assert list(result.parent.iterdir()) == [result]
    ort.InferenceSession.assert_not_called()


@pytest.mark.parametrize("target", ["", "X1", "STX", "x2"])
def test_vitisai_requires_explicit_supported_backend_before_loading(
    target: str, compilation: SimpleNamespace, vitisai_payload: SimpleNamespace
) -> None:
    with pytest.raises(ValueError, match="requires --target X2"):
        compile_off_device(compilation.model, compilation.output, target, ep_name="vitisai")
    vitisai_payload.register.assert_not_called()
    compilation.factory.assert_not_called()


@pytest.mark.parametrize(
    "phase",
    [
        "bootstrap-registration",
        "canonical-registration",
        "enumeration",
        "physical-only",
        "ambiguous",
        "compile",
        "canonical-unregister",
        "bootstrap-unregister",
    ],
)
def test_vitisai_failure_preserves_output_and_releases_only_owned_registrations(
    phase: str,
    compilation: SimpleNamespace,
    vitisai_payload: SimpleNamespace,
) -> None:
    compilation.output.parent.mkdir()
    compilation.output.write_bytes(b"previous output")
    owned = []

    def register(name: str, _library: str) -> None:
        if (phase == "bootstrap-registration" and name.endswith(".virtual")) or (
            phase == "canonical-registration" and name == "VitisAIExecutionProvider"
        ):
            raise RuntimeError("native failure")
        owned.append(name)

    def unregister(name: str) -> None:
        assert compilation.output.read_bytes() == b"previous output"
        if (phase == "canonical-unregister" and name == "VitisAIExecutionProvider") or (
            phase == "bootstrap-unregister" and name.endswith(".virtual")
        ):
            raise RuntimeError("native failure")

    def fail_compile(path: str) -> None:
        Path(path).write_bytes(b"partial output")
        raise RuntimeError("native failure")

    vitisai_payload.register.side_effect = register
    vitisai_payload.unregister.side_effect = unregister
    if phase == "enumeration":
        vitisai_payload.get_devices.side_effect = RuntimeError("native failure")
    elif phase == "physical-only":
        vitisai_payload.device.device.metadata["is_virtual"] = "0"
    elif phase == "ambiguous":
        vitisai_payload.get_devices.return_value *= 2
    elif phase == "compile":
        compilation.native.compile_to_file.side_effect = fail_compile
    with pytest.raises(
        (RuntimeError, ValueError),
        match=r"native failure|No physical-device fallback|exactly one virtual NPU",
    ):
        compile_off_device(
            compilation.model, compilation.output, "X2", ep_name="vitisai", overwrite=True
        )
    assert [call.args[0] for call in vitisai_payload.unregister.call_args_list] == owned[::-1]
    assert compilation.output.read_bytes() == b"previous output"
    assert list(compilation.output.parent.iterdir()) == [compilation.output]
    if phase not in {"compile", "canonical-unregister", "bootstrap-unregister"}:
        compilation.factory.assert_not_called()


def test_vitisai_rejects_context_with_temporary_provider_identity(
    compilation: SimpleNamespace, vitisai_payload: SimpleNamespace
) -> None:
    for attribute in compilation.context.graph.node[0].attribute:
        if attribute.name == "source":
            attribute.s = b"VitisAIExecutionProvider_temporary.virtual"
    with pytest.raises(ValueError, match="embedded VitisAIExecutionProvider"):
        compile_off_device(compilation.model, compilation.output, "X2", ep_name="vitisai")
    assert vitisai_payload.unregister.call_count == 2
    assert not compilation.output.exists()


@pytest.mark.parametrize("overwrite", [False, True])
def test_compilation_binds_only_selected_virtual_device_and_publishes_after_unregister(
    overwrite: bool, compilation: SimpleNamespace, payload: SimpleNamespace
) -> None:
    if overwrite:
        compilation.output.parent.mkdir()
        compilation.output.write_bytes(b"previous output")

    def unregister(_name: str) -> None:
        if overwrite:
            assert compilation.output.read_bytes() == b"previous output"
        else:
            assert not compilation.output.exists()

    payload.unregister.side_effect = unregister
    result = compile_off_device(
        compilation.model, compilation.output, "4000", source="directory", overwrite=overwrite
    )
    assert result == compilation.output
    assert onnx.load(result, load_external_data=False).graph == compilation.context.graph
    compilation.options.add_session_config_entry.assert_called_once_with(
        "session.disable_cpu_ep_fallback", "1"
    )
    assert compilation.bindings == [((payload.device,), {})]
    compilation.factory.assert_called_once_with(
        compilation.options,
        str(compilation.model),
        embed_compiled_data_into_model=True,
        flags=(
            ort.OrtCompileApiFlags.ERROR_IF_NO_NODES_COMPILED
            | ort.OrtCompileApiFlags.ERROR_IF_OUTPUT_FILE_EXISTS
        ),
    )
    staged = Path(compilation.native.compile_to_file.call_args.args[0])
    assert staged != compilation.output
    assert staged.parent.parent == compilation.output.parent
    assert list(compilation.output.parent.iterdir()) == [compilation.output]
    payload.unregister.assert_called_once_with(payload.register.call_args.args[0])


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("missing", FileNotFoundError),
        ("input", ValueError),
        ("hardlink", ValueError),
        ("directory", IsADirectoryError),
        ("existing", FileExistsError),
        ("target", ValueError),
        ("compiled", ValueError),
    ],
)
def test_invalid_inputs_fail_before_registration(
    case: str,
    error: type[Exception],
    compilation: SimpleNamespace,
    payload: SimpleNamespace,
    tmp_path: Path,
) -> None:
    model, output, target = compilation.model, compilation.output, "4000"
    if case == "missing":
        model = tmp_path / "missing.onnx"
    elif case == "input":
        output = model
    elif case == "hardlink":
        output = tmp_path / "alias.onnx"
        output.hardlink_to(model)
    elif case == "directory":
        output = tmp_path
    elif case == "existing":
        output.parent.mkdir()
        output.write_bytes(b"previous output")
    elif case == "target":
        target = ""
    elif case == "compiled":
        onnx.save(compilation.context, model)
    with pytest.raises(error):
        compile_off_device(model, output, target, overwrite=case != "existing")
    payload.register.assert_not_called()
    compilation.factory.assert_not_called()


def test_external_input_weights_cannot_be_overwritten(
    compilation: SimpleNamespace, payload: SimpleNamespace
) -> None:
    model = create_simple_conv_model()
    for tensor in model.graph.initializer:
        tensor.CopyFrom(
            onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(tensor), name=tensor.name)
        )
    onnx.save_model(
        model,
        compilation.model,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    weights = compilation.model.with_name("weights.bin")
    original = weights.read_bytes()
    with pytest.raises(ValueError, match="input model or its data files"):
        compile_off_device(compilation.model, weights, "4000", overwrite=True)
    assert weights.read_bytes() == original
    payload.register.assert_not_called()


@pytest.mark.parametrize(("target", "count"), [("unavailable", 1), ("4000", 2)])
def test_missing_or_ambiguous_platform_never_compiles(
    target: str, count: int, compilation: SimpleNamespace, payload: SimpleNamespace
) -> None:
    payload.get_devices.return_value = [payload.device] * count
    with pytest.raises(ValueError, match="must match exactly one virtual NPU device"):
        compile_off_device(compilation.model, compilation.output, target)
    compilation.factory.assert_not_called()
    payload.unregister.assert_called_once()
    assert not compilation.output.exists()


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("source", "OtherExecutionProvider", "embedded OpenVINO"),
        ("embed_mode", 0, "embedded OpenVINO"),
        ("main_context", 2, "malformed EPContext"),
        ("ep_cache_context", 7, "malformed EPContext"),
        ("ep_cache_context", b"", "empty main EPContext"),
        ("main_context", 0, "no main EPContext"),
    ],
)
def test_invalid_context_preserves_existing_output(
    attribute: str,
    value: str | int | bytes,
    message: str,
    compilation: SimpleNamespace,
) -> None:
    node = compilation.context.graph.node[0]
    for existing in node.attribute:
        if existing.name == attribute:
            existing.CopyFrom(onnx.helper.make_attribute(attribute, value))
    compilation.output.parent.mkdir()
    compilation.output.write_bytes(b"previous output")
    with pytest.raises(ValueError, match=message):
        compile_off_device(compilation.model, compilation.output, "4000", overwrite=True)
    assert compilation.output.read_bytes() == b"previous output"
    assert list(compilation.output.parent.iterdir()) == [compilation.output]


@pytest.mark.parametrize(
    ("artifact", "error"),
    [
        ("missing", FileNotFoundError),
        ("malformed", DecodeError),
        ("uncompiled", ValueError),
        ("wrong_domain", ValueError),
        ("external", ValueError),
    ],
)
def test_missing_malformed_or_nonportable_artifact_is_not_published(
    artifact: str, error: type[Exception], compilation: SimpleNamespace
) -> None:
    if artifact == "missing":
        compilation.native.compile_to_file.side_effect = None
    elif artifact == "malformed":
        compilation.native.compile_to_file.side_effect = lambda path: Path(path).write_bytes(
            b"not an ONNX protobuf"
        )
    elif artifact == "uncompiled":
        compilation.native.compile_to_file.side_effect = lambda path: create_simple_model(
            Path(path)
        )
    elif artifact == "wrong_domain":
        compilation.context.graph.node[0].domain = "other"
    else:
        tensor = onnx.TensorProto(name="external", data_type=onnx.TensorProto.FLOAT, dims=[1])
        tensor.data_location = onnx.TensorProto.EXTERNAL
        tensor.external_data.add(key="location", value="weights.bin")
        compilation.context.graph.initializer.append(tensor)
    with pytest.raises(error):
        compile_off_device(compilation.model, compilation.output, "4000")
    assert not compilation.output.exists()
    assert not list(compilation.output.parent.iterdir())


@pytest.mark.parametrize("phase", ["compile", "unregister"])
def test_partial_compile_or_cleanup_failure_preserves_existing_output(
    phase: str, compilation: SimpleNamespace, payload: SimpleNamespace
) -> None:
    def fail_compile(path: str) -> None:
        Path(path).write_bytes(b"partial output")
        raise RuntimeError("native failure")

    compilation.output.parent.mkdir()
    compilation.output.write_bytes(b"previous output")
    if phase == "compile":
        compilation.native.compile_to_file.side_effect = fail_compile
    else:
        payload.unregister.side_effect = RuntimeError("native failure")
    with pytest.raises(RuntimeError, match="native failure"):
        compile_off_device(compilation.model, compilation.output, "4000", overwrite=True)
    assert compilation.output.read_bytes() == b"previous output"
    assert list(compilation.output.parent.iterdir()) == [compilation.output]
    payload.unregister.assert_called_once()


def test_concurrent_output_is_not_overwritten_without_permission(
    compilation: SimpleNamespace,
) -> None:
    write_output = compilation.native.compile_to_file.side_effect

    def concurrent_write(path: str) -> None:
        write_output(path)
        compilation.output.write_bytes(b"concurrent output")

    compilation.native.compile_to_file.side_effect = concurrent_write
    with pytest.raises(FileExistsError):
        compile_off_device(compilation.model, compilation.output, "4000")
    assert compilation.output.read_bytes() == b"concurrent output"
    assert list(compilation.output.parent.iterdir()) == [compilation.output]


def test_secondary_context_without_its_own_cache_is_supported(
    compilation: SimpleNamespace,
) -> None:
    secondary = deepcopy(compilation.context.graph.node[0])
    secondary.name = "secondary"
    secondary.output[0] += "_secondary"
    for attribute in secondary.attribute:
        if attribute.name == "main_context":
            attribute.i = 0
        elif attribute.name == "ep_cache_context":
            attribute.s = b""
    compilation.context.graph.node.append(secondary)
    result = compile_off_device(compilation.model, compilation.output, "4000")
    assert len(onnx.load(result, load_external_data=False).graph.node) == 2


def test_context_attribute_defaults_are_supported(compilation: SimpleNamespace) -> None:
    node = compilation.context.graph.node[0]
    for attribute in list(node.attribute):
        if attribute.name in {"main_context", "embed_mode"}:
            node.attribute.remove(attribute)
    result = compile_off_device(compilation.model, compilation.output, "4000")
    assert result.is_file()


@pytest.mark.parametrize(
    ("phase", "error_type"),
    [("compile", RuntimeError), ("construct", RuntimeError), ("compile", KeyboardInterrupt)],
)
def test_compiler_and_options_are_released_before_unregister_on_failure(
    phase: str,
    error_type: type[BaseException],
    payload: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    create_simple_model(model_path)
    runtime_compiler = ort.ModelCompiler
    references = {}

    class Options:
        add_session_config_entry = Mock()
        add_provider_for_devices = Mock()

    class NativeCompiler:
        def __init__(self, options: Options) -> None:
            self.options = options

        def compile_to_file(self, _path: str) -> None:
            raise error_type("native compilation failed")

    def create_compiler(options: Options, _model: str, **_kwargs: object) -> object:
        compiler = object.__new__(runtime_compiler)
        compiler._model_compiler = NativeCompiler(options)
        references["compiler"] = ref(compiler)
        references["options"] = ref(options)
        if phase == "construct":
            raise error_type("native compilation failed")
        return compiler

    def unregister(_name: str) -> None:
        assert references["compiler"]() is None, "Compiler is still alive at unregister"
        assert references["options"]() is None, "Options are still alive at unregister"

    monkeypatch.setattr(ort, "SessionOptions", Options)
    monkeypatch.setattr(ort, "ModelCompiler", create_compiler)
    payload.unregister.side_effect = unregister
    with pytest.raises(error_type, match="native compilation failed"):
        compile_off_device(model_path, tmp_path / "compiled.onnx", "4000")
    payload.unregister.assert_called_once()
