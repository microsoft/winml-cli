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
from click.testing import CliRunner
from google.protobuf.message import DecodeError

from tests.fixtures.create_test_models import create_simple_conv_model
from tests.unit.compiler.test_compiler_stages import create_epcontext_onnx, create_simple_model
from winml.modelkit.compiler import (
    CompileContext,
    CompileStage,
    EPConfig,
    WinMLCompileConfig,
    compile_onnx,
    list_compilation_eps,
    list_compilation_targets,
    offdevice,
)
from winml.modelkit.ep_path import DirectorySource, EPEntry
from winml.modelkit.session import (
    EPDeviceTarget,
    WinMLEPRegistry,
    WinMLOffDeviceSession,
    bind_compilation_device,
)
from winml.modelkit.utils.constants import normalize_ep_name


def _compile_model(
    model_path: Path,
    output_path: Path,
    target: str,
    *,
    ep_name: str = "openvino",
    source: str | None = None,
    overwrite: bool = False,
    validate: bool = False,
) -> Path:
    """Run the shared pipeline with the requested test compilation configuration."""
    is_vitisai = normalize_ep_name(ep_name) == "VitisAIExecutionProvider"
    config = WinMLCompileConfig.from_dict(
        {
            "execution_provider": ep_name,
            "compiler": "offdevice",
            "device": "npu",
            "embed_context": True,
            "validate": validate,
            "target_platform": None if is_vitisai else target,
            "provider_options": {"target": target} if is_vitisai else {},
            "ep_device": EPDeviceTarget(ep=ep_name, device="npu", source=source).to_dict(),
        }
    )
    result = compile_onnx(model_path, output_path, config, overwrite=overwrite)
    assert result.success, result.errors
    assert result.output_path is not None
    return result.output_path


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


@pytest.mark.parametrize(
    ("payload_fixture", "provider", "platform", "options"),
    [
        ("payload", "openvino", "4000", {}),
        ("qnn_payload", "qnn", "60", {}),
        ("vitisai_payload", "vitisai", None, {"target": "X2"}),
    ],
)
def test_compilation_binding_outlives_session_cleanup(
    payload_fixture: str,
    provider: str,
    platform: str | None,
    options: dict[str, str],
    compilation: SimpleNamespace,
    request: pytest.FixtureRequest,
) -> None:
    selected = request.getfixturevalue(payload_fixture)
    config = EPConfig(
        provider=provider,
        compiler="offdevice",
        embed_context=True,
        target_platform=platform,
        provider_options=options,
    )
    target = EPDeviceTarget(ep=provider, device="npu", source="directory")

    with bind_compilation_device(target, ep_config=config) as bound:
        assert bound.device.ort_handle is selected.device
        assert bound.ep.arg0 == selected.register.call_args_list[-1].args[0]
        selected.unregister.assert_not_called()
        session = WinMLOffDeviceSession(compilation.model, bound, config)
        try:
            session.compile()
            artifact = session.running_model_path
            assert artifact.is_file()
        finally:
            session.reset()
        assert not artifact.exists()
        selected.unregister.assert_not_called()

    selected.unregister.assert_called()
    compilation.factory.assert_called_once()


def test_compilation_binding_unregisters_when_the_caller_fails(
    payload: SimpleNamespace,
) -> None:
    config = EPConfig(compiler="offdevice", target_platform="4000")
    target = EPDeviceTarget(ep="openvino", device="npu")
    with (
        pytest.raises(RuntimeError, match="Caller failed"),
        bind_compilation_device(target, ep_config=config),
    ):
        raise RuntimeError("Caller failed")
    payload.unregister.assert_called_once()


@pytest.mark.parametrize(
    ("platform", "duplicate"),
    [("unavailable", False), ("4000", True)],
)
def test_compilation_binding_rejects_missing_or_ambiguous_platforms(
    platform: str,
    duplicate: bool,
    payload: SimpleNamespace,
) -> None:
    if duplicate:
        payload.get_devices.return_value = [payload.device, payload.device]
    config = EPConfig(compiler="offdevice", target_platform=platform)
    target = EPDeviceTarget(ep="openvino", device="npu")
    with (
        pytest.raises(ValueError, match="must match exactly one virtual NPU"),
        bind_compilation_device(target, ep_config=config),
    ):
        pytest.fail("An invalid compilation binding was accepted")
    payload.unregister.assert_called_once()


def test_regular_compilation_binding_uses_the_existing_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Mock()
    monkeypatch.setattr(WinMLEPRegistry, "instance", Mock(return_value=registry))
    target = EPDeviceTarget(ep="cpu", device="cpu")
    with bind_compilation_device(target, ep_config=EPConfig()) as bound:
        assert bound is registry.auto_device.return_value
    registry.auto_device.assert_called_once_with(target)


@pytest.mark.parametrize(
    ("payload_fixture", "provider", "platform", "options"),
    [
        ("payload", "openvino", "4000", {}),
        ("qnn_payload", "qnn", "60", {}),
        ("vitisai_payload", "vitisai", None, {"target": "X2"}),
    ],
)
def test_offdevice_stage_publishes_before_cleanup(
    payload_fixture: str,
    provider: str,
    platform: str | None,
    options: dict[str, str],
    compilation: SimpleNamespace,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = request.getfixturevalue(payload_fixture)
    config = WinMLCompileConfig(
        ep_config=EPConfig(
            provider=provider,
            compiler="offdevice",
            embed_context=True,
            target_platform=platform,
            provider_options=options,
        ),
        validate=False,
    ).to_dict()
    config["output_path"] = str(compilation.output)
    compilation.output.parent.mkdir()
    context = CompileContext(model_path=compilation.model, config=config)
    forbidden = Mock(side_effect=AssertionError("Physical device resolution is forbidden"))
    monkeypatch.setattr(
        "winml.modelkit.compiler.stages.compile.resolve_device", forbidden
    )
    monkeypatch.setattr(WinMLEPRegistry.instance(), "auto_device", forbidden)
    original_reset = WinMLOffDeviceSession.reset
    resets = []

    def reset(session: WinMLOffDeviceSession) -> None:
        selected.unregister.assert_not_called()
        assert context.output_path == compilation.output
        assert context.output_path.is_file()
        staged_path = session.running_model_path
        original_reset(session)
        assert not staged_path.exists()
        resets.append(staged_path)

    monkeypatch.setattr(WinMLOffDeviceSession, "reset", reset)

    CompileStage().process(context)

    assert len(resets) == 1
    assert context.session is None
    assert onnx.load(context.output_path).graph == compilation.context.graph
    compilation.factory.assert_called_once()
    selected.unregister.assert_called()
    forbidden.assert_not_called()


def test_offdevice_stage_unregisters_after_compilation_failure(
    compilation: SimpleNamespace,
    payload: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compilation.factory.side_effect = RuntimeError("Native compile failed")
    context = CompileContext(
        model_path=compilation.model,
        config={
            "execution_provider": "openvino",
            "compiler": "offdevice",
            "embed_context": True,
            "target_platform": "4000",
            "validate": False,
        },
    )
    original_reset = WinMLOffDeviceSession.reset
    resets = []

    def reset(session: WinMLOffDeviceSession) -> None:
        payload.unregister.assert_not_called()
        original_reset(session)
        resets.append(session)

    monkeypatch.setattr(WinMLOffDeviceSession, "reset", reset)

    with pytest.raises(RuntimeError, match="Native compile failed"):
        CompileStage().process(context)

    assert len(resets) == 1
    assert context.session is None
    assert context.output_path is None
    assert context.has_error
    payload.unregister.assert_called_once()


def test_offdevice_stage_rejects_shared_contexts_before_registration(
    compilation: SimpleNamespace,
    payload: SimpleNamespace,
) -> None:
    context = CompileContext(
        model_path=compilation.model,
        config={"compiler": "offdevice"},
        n_total_models=2,
    )

    with pytest.raises(ValueError, match="does not support shared multi-model contexts"):
        CompileStage().process(context)

    payload.register.assert_not_called()
    payload.get_devices.assert_not_called()
    compilation.factory.assert_not_called()


@pytest.mark.parametrize("validate", [False, True])
def test_offdevice_cli_runs_the_shared_pipeline(
    validate: bool,
    compilation: SimpleNamespace,
    payload: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from winml.modelkit.commands.compile import compile as compile_command

    forbidden = Mock(side_effect=AssertionError("Physical device resolution is forbidden"))
    monkeypatch.setattr(
        "winml.modelkit.commands.compile.resolve_device", forbidden
    )

    result = CliRunner().invoke(
        compile_command,
        [
            "--compiler", "offdevice",
            "--ep", "openvino@directory",
            "--target-platform", "4000",
            "--model", str(compilation.model),
            "--output", str(compilation.output),
            "--validate" if validate else "--no-validate",
        ],
    )

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "Artifact validation: passed (embedded-context checks)" in output
    assert (
        "Target-device inference validation: skipped (off-device compilation)" in output
    )
    assert onnx.load(compilation.output).graph == compilation.context.graph
    compilation.factory.assert_called_once()
    payload.unregister.assert_called_once()
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    ("overwrite", "created_during_compile"),
    [(False, False), (True, False), (False, True)],
)
def test_offdevice_pipeline_preserves_output_overwrite_policy(
    overwrite: bool,
    created_during_compile: bool,
    compilation: SimpleNamespace,
    payload: SimpleNamespace,
) -> None:
    config = WinMLCompileConfig(
        ep_config=EPConfig(
            provider="openvino",
            compiler="offdevice",
            embed_context=True,
            target_platform="4000",
        ),
        validate=False,
    )
    previous = compilation.model.read_bytes()
    compilation.output.parent.mkdir()
    if created_during_compile:
        original_write = compilation.native.compile_to_file.side_effect

        def write_output(path: str) -> None:
            original_write(path)
            compilation.output.write_bytes(previous)

        compilation.native.compile_to_file.side_effect = write_output
    else:
        compilation.output.write_bytes(previous)

    if overwrite:
        result = compile_onnx(
            compilation.model,
            compilation.output,
            config,
            overwrite=True,
        )
        assert result.success
        assert onnx.load(result.output_path).graph == compilation.context.graph
    else:
        with pytest.raises(FileExistsError):
            compile_onnx(
                compilation.model,
                compilation.output,
                config,
                overwrite=False,
            )
        assert compilation.output.read_bytes() == previous
    payload.unregister.assert_called_once()
    staged = Path(compilation.native.compile_to_file.call_args.args[0])
    assert not staged.parent.exists()
    assert list(compilation.output.parent.iterdir()) == [compilation.output]


@pytest.mark.parametrize("destination_kind", ["model", "hardlink", "external-data"])
def test_offdevice_pipeline_never_overwrites_input_files(
    destination_kind: str,
    compilation: SimpleNamespace,
    payload: SimpleNamespace,
) -> None:
    if destination_kind == "external-data":
        model = create_simple_conv_model()
        for tensor in model.graph.initializer:
            tensor.CopyFrom(
                onnx.numpy_helper.from_array(
                    onnx.numpy_helper.to_array(tensor), name=tensor.name
                )
            )
        onnx.save_model(
            model,
            compilation.model,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.bin",
            size_threshold=0,
        )
        destination = compilation.model.parent / "weights.bin"
    elif destination_kind == "hardlink":
        destination = compilation.model.with_name("alias.onnx")
        destination.hardlink_to(compilation.model)
    else:
        destination = compilation.model
    previous = destination.read_bytes()
    config = WinMLCompileConfig(
        ep_config=EPConfig(
            provider="openvino",
            compiler="offdevice",
            embed_context=True,
            target_platform="4000",
        ),
        validate=False,
    )

    with pytest.raises(ValueError, match="must not overwrite the input model or its data files"):
        compile_onnx(compilation.model, destination, config, overwrite=True)

    assert destination.read_bytes() == previous
    payload.unregister.assert_called_once()


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
    with pytest.raises(
        ValueError, match=r"QNN does not enumerate SoC targets.*Use --target-platform"
    ):
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
    result = _compile_model(
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
def test_qnn_requires_explicit_numeric_soc_before_native_compilation(
    target: str, compilation: SimpleNamespace, qnn_payload: SimpleNamespace
) -> None:
    with pytest.raises(
        ValueError, match="QNN target_platform must be a positive numeric soc_model"
    ):
        _compile_model(compilation.model, compilation.output, target, ep_name="qnn")
    qnn_payload.unregister.assert_called_once_with("QNNExecutionProvider")
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
        _compile_model(
            compilation.model, compilation.output, "60", ep_name="qnn", source=source
        )
    compilation.factory.assert_not_called()
    assert not compilation.output.exists()
    assert not compilation.output.parent.exists()
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
        _compile_model(
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
        _compile_model(compilation.model, compilation.output, "60", ep_name="qnn")
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
    with pytest.raises(
        ValueError, match=r"Use --ep-options target=X2.*not a destination SoC"
    ):
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
def test_vitisai_pipeline_binds_only_owned_virtual_npu_and_cleans_up(
    overwrite: bool,
    compilation: SimpleNamespace,
    vitisai_payload: SimpleNamespace,
) -> None:
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
        assert onnx.load(compilation.output).graph == compilation.context.graph
        staged = Path(compilation.native.compile_to_file.call_args.args[0])
        assert not staged.parent.exists()

    vitisai_payload.unregister.side_effect = unregister
    result = _compile_model(
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
def test_vitisai_requires_explicit_supported_backend_before_native_compilation(
    target: str, compilation: SimpleNamespace, vitisai_payload: SimpleNamespace
) -> None:
    with pytest.raises(ValueError, match="requires target=X2"):
        _compile_model(compilation.model, compilation.output, target, ep_name="vitisai")
    assert vitisai_payload.unregister.call_count == 2
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
def test_vitisai_pipeline_failure_releases_only_owned_registrations(
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
        if phase in {"canonical-unregister", "bootstrap-unregister"}:
            assert onnx.load(compilation.output).graph == compilation.context.graph
        else:
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
        _compile_model(
            compilation.model, compilation.output, "X2", ep_name="vitisai", overwrite=True
        )
    assert [call.args[0] for call in vitisai_payload.unregister.call_args_list] == owned[::-1]
    if phase in {"canonical-unregister", "bootstrap-unregister"}:
        assert onnx.load(compilation.output).graph == compilation.context.graph
        assert list(compilation.output.parent.iterdir()) == [compilation.output]
    else:
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
        _compile_model(compilation.model, compilation.output, "X2", ep_name="vitisai")
    assert vitisai_payload.unregister.call_count == 2
    assert not compilation.output.exists()


@pytest.mark.parametrize("overwrite", [False, True])
def test_pipeline_publishes_and_resets_before_unregister(
    overwrite: bool, compilation: SimpleNamespace, payload: SimpleNamespace
) -> None:
    if overwrite:
        compilation.output.parent.mkdir()
        compilation.output.write_bytes(b"previous output")

    def unregister(_name: str) -> None:
        assert onnx.load(compilation.output).graph == compilation.context.graph
        staged = Path(compilation.native.compile_to_file.call_args.args[0])
        assert not staged.parent.exists()

    payload.unregister.side_effect = unregister
    result = _compile_model(
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
    assert not staged.parent.exists()
    assert list(compilation.output.parent.iterdir()) == [compilation.output]
    payload.unregister.assert_called_once_with(payload.register.call_args.args[0])


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("missing", FileNotFoundError),
        ("compiled", ValueError),
    ],
)
def test_invalid_model_never_reaches_native_compilation(
    case: str,
    error: type[Exception],
    compilation: SimpleNamespace,
    payload: SimpleNamespace,
    tmp_path: Path,
) -> None:
    model = compilation.model
    if case == "missing":
        model = tmp_path / "missing.onnx"
    elif case == "compiled":
        onnx.save(compilation.context, model)
    with pytest.raises(error):
        _compile_model(model, compilation.output, "4000")
    if case == "missing":
        payload.register.assert_not_called()
    else:
        payload.unregister.assert_called_once()
    compilation.factory.assert_not_called()


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
@pytest.mark.parametrize("validate", [False, True])
def test_invalid_context_preserves_existing_output(
    attribute: str,
    value: str | int | bytes,
    message: str,
    compilation: SimpleNamespace,
    validate: bool,
) -> None:
    node = compilation.context.graph.node[0]
    for existing in node.attribute:
        if existing.name == attribute:
            existing.CopyFrom(onnx.helper.make_attribute(attribute, value))
    compilation.output.parent.mkdir()
    compilation.output.write_bytes(b"previous output")
    with pytest.raises(ValueError, match=message):
        _compile_model(
            compilation.model,
            compilation.output,
            "4000",
            overwrite=True,
            validate=validate,
        )
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
@pytest.mark.parametrize("validate", [False, True])
def test_missing_malformed_or_nonportable_artifact_is_not_published(
    artifact: str,
    error: type[Exception],
    compilation: SimpleNamespace,
    validate: bool,
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
        _compile_model(
            compilation.model, compilation.output, "4000", validate=validate
        )
    assert not compilation.output.exists()
    assert not compilation.output.parent.exists()


@pytest.mark.parametrize("validate", [False, True])
def test_offdevice_reports_skipped_inference(
    validate: bool,
    compilation: SimpleNamespace,
) -> None:
    import json

    config = WinMLCompileConfig.from_dict(
        {
            "execution_provider": "openvino",
            "compiler": "offdevice",
            "target_platform": "4000",
            "embed_context": True,
            "validate": validate,
        }
    )
    result = compile_onnx(compilation.model, compilation.output, config)

    assert result.success, result.errors
    assert result.output_path == compilation.output
    assert result.validation_passed is False
    warning = "Target-device inference validation: skipped (off-device compilation)"
    assert warning in result.warnings
    serialized = json.loads(json.dumps(result.to_dict()))
    assert warning in serialized["warnings"]
    assert serialized["validation_passed"] is False


@pytest.mark.parametrize("phase", ["compile", "unregister"])
def test_pipeline_failure_preserves_output_until_publication(
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
        _compile_model(compilation.model, compilation.output, "4000", overwrite=True)
    if phase == "compile":
        assert compilation.output.read_bytes() == b"previous output"
        assert list(compilation.output.parent.iterdir()) == [compilation.output]
    else:
        assert onnx.load(compilation.output).graph == compilation.context.graph
        assert list(compilation.output.parent.iterdir()) == [compilation.output]
    payload.unregister.assert_called_once()


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
    result = _compile_model(compilation.model, compilation.output, "4000")
    assert len(onnx.load(result, load_external_data=False).graph.node) == 2


def test_context_attribute_defaults_are_supported(compilation: SimpleNamespace) -> None:
    node = compilation.context.graph.node[0]
    for attribute in list(node.attribute):
        if attribute.name in {"main_context", "embed_mode"}:
            node.attribute.remove(attribute)
    result = _compile_model(compilation.model, compilation.output, "4000")
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
        _compile_model(model_path, tmp_path / "compiled.onnx", "4000")
    payload.unregister.assert_called_once()
