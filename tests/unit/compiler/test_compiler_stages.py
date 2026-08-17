# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for compiler stages."""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx
import pytest
from onnx import TensorProto, helper


if TYPE_CHECKING:
    from pathlib import Path

    from winml.modelkit.utils.constants import EPAlias


def create_simple_model(path: Path) -> None:
    """Create a simple ONNX model for testing."""
    # Create a simple model: Y = Identity(X)
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])

    node = helper.make_node("Identity", ["X"], ["Y"])

    graph = helper.make_graph([node], "test_model", [x_info], [y_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    onnx.save(model, str(path))


def create_qlinear_model(path: Path) -> None:
    """Create a model with QLinearConv ops (not QDQ format)."""
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])
    y_info = helper.make_tensor_value_info("output_0", TensorProto.FLOAT, [1, 1])

    node = helper.make_node("QLinearConv", ["X"], ["output_0"])

    graph = helper.make_graph([node], "qlinear_model", [x_info], [y_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))


class TestOptimizeStage:
    """Test EP-specific graph optimization stage."""

    def test_should_not_run_when_no_transforms(self, tmp_path):
        """Stage should skip when no transforms are registered for the EP."""
        from winml.modelkit.compiler import CompileContext, OptimizeStage, clear_transforms

        clear_transforms()

        model_path = tmp_path / "model.onnx"
        create_simple_model(model_path)

        context = CompileContext(
            model_path=model_path,
            config={"execution_provider": "qnn"},
        )

        assert not OptimizeStage.should_run(context)

    def test_should_run_when_transforms_registered(self, tmp_path):
        """Stage should run when transforms are registered for the EP."""
        from winml.modelkit.compiler import (
            CompileContext,
            OptimizeStage,
            clear_transforms,
            register_transform,
        )

        clear_transforms()

        class DummyTransform:
            def applies_to(self, ep: EPAlias) -> bool:
                return ep == "qnn"

            def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
                return model

        register_transform(DummyTransform())

        model_path = tmp_path / "model.onnx"
        create_simple_model(model_path)

        context = CompileContext(
            model_path=model_path,
            config={"execution_provider": "qnn"},
        )

        assert OptimizeStage.should_run(context)
        clear_transforms()

    def test_process_applies_transforms(self, tmp_path):
        """Stage should apply transforms and save output model."""
        from winml.modelkit.compiler import (
            CompileContext,
            OptimizeStage,
            clear_transforms,
            register_transform,
        )

        clear_transforms()

        transform_called = []

        class TrackingTransform:
            def applies_to(self, ep: EPAlias) -> bool:
                return ep == "qnn"

            def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
                transform_called.append(True)
                return model

        register_transform(TrackingTransform())

        model_path = tmp_path / "model.onnx"
        create_simple_model(model_path)

        context = CompileContext(
            model_path=model_path,
            config={"execution_provider": "qnn"},
        )

        stage = OptimizeStage()
        result = stage.process(context)

        assert len(transform_called) == 1
        assert result.model_path.name == "model_ep_opt.onnx"
        assert result.model_path.exists()
        clear_transforms()


class TestQFormatConvertStage:
    """Test QLinear-to-QDQ format conversion stage."""

    def test_should_not_run_for_plain_model(self, tmp_path):
        """Stage should skip for models without QLinear ops."""
        from winml.modelkit.compiler import CompileContext, QFormatConvertStage

        model_path = tmp_path / "model.onnx"
        create_simple_model(model_path)

        context = CompileContext(
            model_path=model_path,
            config={"execution_provider": "qnn"},
        )

        assert not QFormatConvertStage.should_run(context)

    def test_should_run_for_qlinear_model_on_qnn(self, tmp_path):
        """Stage should run when model has QLinear ops targeting QNN."""
        from winml.modelkit.compiler import CompileContext, QFormatConvertStage

        model_path = tmp_path / "model.onnx"
        create_qlinear_model(model_path)

        context = CompileContext(
            model_path=model_path,
            config={"execution_provider": "qnn"},
        )

        assert QFormatConvertStage.should_run(context)

    def test_process_adds_warning(self, tmp_path):
        """Stage should add warning since conversion is not yet implemented."""
        from winml.modelkit.compiler import CompileContext, QFormatConvertStage

        model_path = tmp_path / "model.onnx"
        create_qlinear_model(model_path)

        context = CompileContext(
            model_path=model_path,
            config={"execution_provider": "qnn"},
        )

        stage = QFormatConvertStage()
        result = stage.process(context)

        assert len(result.warnings) == 1
        assert "not yet implemented" in result.warnings[0]


class TestCompileContext:
    """Test compile context."""

    def test_context_properties(self, tmp_path):
        """Test context property accessors."""
        from winml.modelkit.compiler import CompileContext

        model_path = tmp_path / "model.onnx"
        create_simple_model(model_path)

        context = CompileContext(
            model_path=model_path,
            config={
                "execution_provider": "cpu",
                "enable_ep_context": True,
                "validate": False,
            },
        )

        assert context.execution_provider == "cpu"
        assert context.enable_ep_context is True
        assert context.validate is False

    def test_error_handling(self, tmp_path):
        """Test error and warning handling."""
        from winml.modelkit.compiler import CompileContext

        context = CompileContext(
            model_path=tmp_path / "model.onnx",
            config={},
        )

        assert not context.has_error
        assert len(context.errors) == 0

        context.add_error("Test error")
        assert context.has_error
        assert len(context.errors) == 1

        context.add_warning("Test warning")
        assert len(context.warnings) == 1

    def test_logging(self, tmp_path):
        """Test logging."""
        from winml.modelkit.compiler import CompileContext

        context = CompileContext(
            model_path=tmp_path / "model.onnx",
            config={},
            verbose=False,
        )

        context.log("Test message")
        assert len(context.logs) == 1
        assert "Test message" in context.logs[0]

    def test_no_quant_fields(self, tmp_path):
        """Verify quant-related fields have been removed from context."""
        from winml.modelkit.compiler import CompileContext

        context = CompileContext(
            model_path=tmp_path / "model.onnx",
            config={},
        )

        assert not hasattr(context, "skip_calibration")
        assert not hasattr(context, "skip_qdq")
        assert not hasattr(context, "tensors_data")
        assert not hasattr(context, "calibration_path")
        assert not hasattr(context, "quantize")


class TestCompileResult:
    """Test CompileResult."""

    def test_no_quant_fields(self):
        """Verify quant-related fields have been removed from result."""
        from winml.modelkit.compiler import CompileResult

        result = CompileResult(success=True)
        assert not hasattr(result, "calibration_time")
        assert not hasattr(result, "qdq_time")
        assert not hasattr(result, "calibration_path")

    def test_to_dict(self):
        """Test serialization."""
        from winml.modelkit.compiler import CompileResult

        result = CompileResult(
            success=True,
            compile_time=1.5,
            total_time=2.0,
        )
        d = result.to_dict()

        assert d["success"] is True
        assert d["compile_time"] == 1.5
        assert d["total_time"] == 2.0
        assert "calibration_time" not in d
        assert "qdq_time" not in d
        assert "calibration_path" not in d

    def test_str(self):
        """Test string representation."""
        from winml.modelkit.compiler import CompileResult

        result = CompileResult(
            success=True,
            compile_time=1.5,
            total_time=2.0,
        )
        s = str(result)
        assert "success=True" in s
        assert "compile_time" in s
        assert "calibration_time" not in s
        assert "qdq_time" not in s


def create_epcontext_onnx(
    path: Path,
    bin_name: str,
    embed_mode: int = 0,
    *,
    partition_name: str | None = None,
) -> None:
    """Create mock EPContext ONNX model for testing.

    Args:
        path: Output path for the ONNX model
        bin_name: Name of the external binary file (for ep_cache_context attribute)
        embed_mode: 0=external binary, 1=embedded
    """
    # Create EPContext node with attributes
    attrs: dict[str, object] = {}
    if partition_name is not None:
        attrs["partition_name"] = partition_name

    ep_context_node = helper.make_node(
        "EPContext",
        inputs=[],
        outputs=["output"],
        name="ep_context_0",
        domain="com.microsoft",
        embed_mode=embed_mode,
        ep_cache_context=bin_name,
        main_context=1,  # This is the main context
        **attrs,
    )

    # Input/output
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

    graph = helper.make_graph(
        [ep_context_node],
        "epcontext_model",
        [],
        [output_info],
    )

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 9

    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


class TestCompileStageProcess:
    def test_ort_session_uses_inference_session_path(self, tmp_path):
        """The public ort_session compiler selects the dedicated ORT session path."""
        from unittest.mock import MagicMock, patch

        from winml.modelkit.compiler import CompileContext, CompileStage

        model_path = tmp_path / "model.onnx"
        create_simple_model(model_path)
        session = MagicMock()
        session.get_providers.return_value = ["QNNExecutionProvider"]
        session.get_inputs.return_value = []
        session.get_outputs.return_value = []
        session_options = MagicMock()

        context = CompileContext(
            model_path=model_path,
            config={
                "execution_provider": "qnn",
                "compiler": "ort_session",
                "enable_ep_context": True,
                "validate": False,
            },
        )

        with (
            patch(
                "winml.modelkit.session.session._build_session_options",
                return_value=session_options,
            ),
            patch("onnxruntime.InferenceSession", return_value=session) as inference_session,
        ):
            CompileStage().process(context)

        inference_session.assert_called_once()
        assert context.shared_session_options is session_options
        assert context.session is None

    def test_process_preserves_trtrtx_provider_options(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from winml.modelkit.compiler import CompileContext, CompileStage

        model_path = tmp_path / "model.onnx"
        create_simple_model(model_path)

        fake_session = MagicMock()
        fake_session.get_providers.return_value = ["NvTensorRTRTXExecutionProvider"]
        fake_session.get_inputs.return_value = []
        fake_session.get_outputs.return_value = []

        fake_winml_session = MagicMock()
        fake_winml_session._session = fake_session
        fake_winml_session.running_model_path = tmp_path / "model_trtrtx_ctx.onnx"

        context = CompileContext(
            model_path=model_path,
            config={
                "execution_provider": "nvtensorrtrtx",
                "provider_options": {"device_type": "GPU", "precision": "fp16"},
                "enable_ep_context": True,
                "validate": False,
            },
        )

        mock_session_cls = MagicMock(return_value=fake_winml_session)
        with (
            patch.dict(
                "winml.modelkit.compiler.stages.compile.COMPILER_SESSION_MAPPING",
                {"ort": mock_session_cls},
                clear=False,
            ),
            patch.object(CompileStage, "_finalize_output"),
        ):
            stage = CompileStage()
            stage.process(context)

        passed_ep_config = mock_session_cls.call_args.kwargs["ep_config"]
        assert passed_ep_config.provider_options == {"device_type": "GPU", "precision": "fp16"}
        fake_winml_session.reset.assert_called_once()
        assert context.session is None

    def test_process_finalizes_markerless_context_before_reset(self, tmp_path):
        """Session-owned EPContext artifacts remain present until publication."""
        from unittest.mock import MagicMock, patch

        from winml.modelkit.compiler import CompileContext, CompileStage
        from winml.modelkit.session import EPDeviceTarget

        model_path = tmp_path / "model.onnx"
        create_simple_model(model_path)
        markerless_path = tmp_path / "model_npu_ctx_private.onnx"
        markerless_path.write_bytes(b"markerless context")
        events: list[str] = []
        fake_winml_session = MagicMock()
        fake_winml_session._session = None
        fake_winml_session.running_model_path = markerless_path

        def _reset() -> None:
            events.append("reset")
            markerless_path.unlink()

        def _finalize(*_args, src_ctx_path: Path, **_kwargs) -> None:
            assert src_ctx_path.is_file()
            events.append("finalize")

        fake_winml_session.reset.side_effect = _reset
        ep_device = MagicMock()
        ep_device.device.device_type = "NPU"
        context = CompileContext(
            model_path=model_path,
            config={
                "execution_provider": "qnn",
                "enable_ep_context": True,
                "validate": False,
                "ep_device": EPDeviceTarget(ep="qnn", device="npu").to_dict(),
            },
        )

        with (
            patch.dict(
                "winml.modelkit.compiler.stages.compile.COMPILER_SESSION_MAPPING",
                {"ort": MagicMock(return_value=fake_winml_session)},
                clear=False,
            ),
            patch("winml.modelkit.compiler.stages.compile.WinMLEPRegistry.instance") as registry,
            patch.object(CompileStage, "_finalize_output", side_effect=_finalize),
        ):
            registry.return_value.auto_device.return_value = ep_device
            CompileStage().process(context)

        assert events == ["finalize", "reset"]

    def test_process_reconstructs_explicit_serialized_device(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from winml.modelkit.compiler import CompileContext, CompileStage
        from winml.modelkit.session import EPDeviceTarget

        model_path = tmp_path / "model.onnx"
        create_simple_model(model_path)

        fake_session = MagicMock()
        fake_session.get_providers.return_value = ["QNNExecutionProvider"]
        fake_session.get_inputs.return_value = []
        fake_session.get_outputs.return_value = []

        fake_winml_session = MagicMock()
        fake_winml_session._session = fake_session
        identity_ctx_path = tmp_path / "model_1234567890abcdef_ctx.onnx"
        fake_winml_session.running_model_path = identity_ctx_path

        context = CompileContext(
            model_path=model_path,
            config={
                "execution_provider": "qnn",
                "device": "gpu",
                "enable_ep_context": True,
                "validate": False,
            },
        )

        mock_session_cls = MagicMock(return_value=fake_winml_session)
        mock_registry = MagicMock()
        resolved_ep_device = MagicMock()
        resolved_ep_device.device.device_type = "GPU"
        mock_registry.auto_device.return_value = resolved_ep_device
        stage = CompileStage()
        with (
            patch.dict(
                "winml.modelkit.compiler.stages.compile.COMPILER_SESSION_MAPPING",
                {"ort": mock_session_cls},
                clear=False,
            ),
            patch(
                "winml.modelkit.compiler.stages.compile.WinMLEPRegistry.instance",
                return_value=mock_registry,
            ),
            patch(
                "winml.modelkit.compiler.stages.compile.resolve_device",
                side_effect=lambda target: EPDeviceTarget(
                    ep=target.ep,
                    device=("npu" if target.device == "auto" else target.device),
                    source=target.source,
                ),
            ) as mock_resolve_device,
            patch.object(stage, "_finalize_output") as mock_finalize_output,
        ):
            stage.process(context)

        mock_resolve_device.assert_called_once_with(EPDeviceTarget(ep="qnn", device="gpu"))
        assert mock_finalize_output.call_args.kwargs["device"] == "gpu"
        assert mock_finalize_output.call_args.kwargs["src_ctx_path"] == identity_ctx_path

    def test_single_model_compile_fallback_does_not_publish_stale_context(self, tmp_path):
        """A raw running path wins over stale identity artifacts in the same directory."""
        from unittest.mock import MagicMock, patch

        from winml.modelkit.compiler import CompileContext, CompileStage

        model_path = tmp_path / "model.onnx"
        create_simple_model(model_path)
        stale_context = tmp_path / "model_npu_staleidentity_ctx.onnx"
        create_epcontext_onnx(stale_context, "embedded", embed_mode=1)
        fake_session = MagicMock()
        fake_session.get_providers.return_value = ["QNNExecutionProvider"]
        fake_session.get_inputs.return_value = []
        fake_session.get_outputs.return_value = []
        fake_winml_session = MagicMock()
        fake_winml_session._session = fake_session
        fake_winml_session.running_model_path = model_path
        context = CompileContext(
            model_path=model_path,
            config={
                "execution_provider": "qnn",
                "device": "npu",
                "enable_ep_context": True,
                "validate": False,
            },
        )
        stage = CompileStage()
        ep_device = MagicMock()
        ep_device.device.device_type = "NPU"

        with (
            patch.dict(
                "winml.modelkit.compiler.stages.compile.COMPILER_SESSION_MAPPING",
                {"ort": MagicMock(return_value=fake_winml_session)},
                clear=False,
            ),
            patch("winml.modelkit.compiler.stages.compile.WinMLEPRegistry.instance") as registry,
            patch.object(stage, "_finalize_output") as finalize_output,
        ):
            registry.return_value.auto_device.return_value = ep_device
            stage.process(context)

        finalize_output.assert_not_called()
        assert context.output_path is None
        assert context.warnings == [f"No EPContext produced for {model_path.name}"]

    def test_multi_model_sequence_shares_options_and_closes_context(self, tmp_path):
        """First, intermediate, and final models share one EP context in sequence."""
        from unittest.mock import MagicMock, patch

        from winml.modelkit.compiler import CompileContext, CompileStage

        session = MagicMock()
        session.get_providers.return_value = ["QNNExecutionProvider"]
        session.get_inputs.return_value = []
        session.get_outputs.return_value = []
        session_options = MagicMock()
        stage = CompileStage()
        previous_options = None

        with (
            patch(
                "winml.modelkit.session.session._build_session_options",
                return_value=session_options,
            ),
            patch("onnxruntime.InferenceSession", return_value=session) as inference_session,
        ):
            for index in range(3):
                model_path = tmp_path / f"model_{index}.onnx"
                create_simple_model(model_path)
                context = CompileContext(
                    model_path=model_path,
                    config={
                        "execution_provider": "qnn",
                        "compiler": "ort_session",
                        "enable_ep_context": True,
                        "validate": False,
                    },
                    n_compiled_models=index,
                    n_total_models=3,
                    shared_session_options=previous_options,
                )
                stage.process(context)
                previous_options = context.shared_session_options

        assert inference_session.call_count == 3
        assert previous_options is session_options
        assert session_options.add_session_config_entry.call_args_list == [
            (("ep.context_enable", "1"),),
            (("ep.context_embed_mode", "0"),),
            (("ep.share_ep_contexts", "1"),),
            (("ep.context_file_path", str(tmp_path / "model_0_ctx.onnx")),),
            (("ep.context_file_path", str(tmp_path / "model_1_ctx.onnx")),),
            (("ep.stop_share_ep_contexts", "1"),),
            (("ep.context_file_path", str(tmp_path / "model_2_ctx.onnx")),),
        ]

    def test_multi_model_runtime_only_config_skips_shared_epcontext(self, tmp_path):
        """Runtime-only multi-model configs should not force shared EPContext setup."""
        from unittest.mock import MagicMock, patch

        from winml.modelkit.compiler import CompileContext, CompileStage, WinMLCompileConfig

        model_path = tmp_path / "model.onnx"
        create_simple_model(model_path)

        session = MagicMock()
        session.get_providers.return_value = ["CPUExecutionProvider"]
        session.get_inputs.return_value = []
        session.get_outputs.return_value = []
        session_options = MagicMock()
        ep_device = MagicMock()
        ep_device.device.device_type = "CPU"
        ep_device.device.ep_name = "CPUExecutionProvider"
        ep_device.device.hardware_name = "cpu"

        context = CompileContext(
            model_path=model_path,
            config=WinMLCompileConfig.for_cpu().to_dict() | {"validate": False},
            n_total_models=2,
        )

        with (
            patch(
                "winml.modelkit.session.session._build_session_options",
                return_value=session_options,
            ),
            patch("onnxruntime.InferenceSession", return_value=session) as inference_session,
            patch("onnxruntime.ModelCompiler") as model_compiler,
            patch(
                "winml.modelkit.compiler.stages.compile.WinMLEPRegistry.instance",
            ) as mock_registry,
        ):
            mock_registry.return_value.auto_device.return_value = ep_device
            CompileStage().process(context)

        assert session_options.add_session_config_entry.call_args_list == []
        model_compiler.assert_not_called()
        inference_session.assert_called_once()
        assert context.shared_session_options is None
        assert context.output_path is None
        assert context.context_binary_path is None
        assert context.warnings == []


class TestCompileStageFinalizeOutput:
    """Test CompileStage._finalize_output method."""

    def test_uses_resolved_nondefault_device_to_find_epcontext(self, tmp_path):
        """The resolved device selects the EPContext artifact produced by ORT."""
        from winml.modelkit.compiler import CompileContext, CompileStage

        work_dir = tmp_path / "work"
        output_dir = tmp_path / "output"
        work_dir.mkdir()
        output_dir.mkdir()

        original_model_path = tmp_path / "mymodel.onnx"
        create_simple_model(original_model_path)

        ctx_path = work_dir / "model_to_compile_gpu_ctx.onnx"
        create_epcontext_onnx(ctx_path, "embedded_data", embed_mode=1)
        context = CompileContext(
            model_path=original_model_path,
            config={
                "execution_provider": "openvino",
                "output_path": str(output_dir),
            },
            work_dir=work_dir,
        )

        CompileStage()._finalize_output(
            context,
            work_dir / "model_to_compile.onnx",
            output_dir,
            device="gpu",
        )

        expected = output_dir / "mymodel_openvino_ctx.onnx"
        assert context.output_path == expected
        assert expected.exists()
        assert not context.warnings

    def test_updates_ep_cache_context_in_external_mode(self, tmp_path):
        """Test that ep_cache_context attribute is updated when bin is renamed.

        Key branch: embed_mode == 0 (external), update ep_cache_context to new filename
        """
        from winml.modelkit.compiler import CompileContext, CompileStage

        # Setup: work_dir with EPContext pointing to old bin name
        work_dir = tmp_path / "work"
        output_dir = tmp_path / "output"
        work_dir.mkdir()
        output_dir.mkdir()

        # Create source model path
        original_model_path = tmp_path / "mymodel.onnx"
        create_simple_model(original_model_path)

        # Create EPContext in work_dir with old bin name
        old_bin_name = "model_to_compile_qnn_ctx.bin"
        ctx_path = work_dir / "model_to_compile_qnn_ctx.onnx"
        create_epcontext_onnx(ctx_path, old_bin_name, embed_mode=0)

        # Create the old bin file
        old_bin_path = work_dir / old_bin_name
        old_bin_path.write_bytes(b"fake binary content")

        # Create context
        context = CompileContext(
            model_path=original_model_path,
            config={
                "execution_provider": "qnn",
                "output_path": str(output_dir),
            },
            work_dir=work_dir,
        )

        # Run _finalize_output
        stage = CompileStage()
        stage._finalize_output(context, ctx_path.parent / "model_to_compile.onnx", output_dir)

        # Directory output keeps the existing filename derived from the input model.
        final_ctx_path = output_dir / "mymodel_qnn_ctx.onnx"
        assert final_ctx_path.exists(), f"Expected {final_ctx_path} to exist"

        # Load and check the attribute was updated
        model = onnx.load(str(final_ctx_path))
        for node in model.graph.node:
            if node.op_type == "EPContext":
                for attr in node.attribute:
                    if attr.name == "ep_cache_context":
                        # Should be updated to the derived output filename.
                        assert b"mymodel_qnn_ctx" in attr.s, f"Expected updated name, got {attr.s}"
                        break

    def test_finalize_output_preserves_previous_referenced_binary_generation(self, tmp_path):
        """A later publication cannot overwrite a binary used by an older ONNX."""
        from winml.modelkit.compiler import CompileContext, CompileStage

        work_dir = tmp_path / "work"
        output_dir = tmp_path / "output"
        work_dir.mkdir()
        output_dir.mkdir()
        original_model_path = tmp_path / "mymodel.onnx"
        create_simple_model(original_model_path)
        context = CompileContext(
            model_path=original_model_path,
            config={"execution_provider": "qnn", "output_path": str(output_dir)},
            work_dir=work_dir,
        )
        stage = CompileStage()

        first_ctx = work_dir / "first_identity_ctx.onnx"
        create_epcontext_onnx(first_ctx, "first_identity_ctx_qnn.bin", embed_mode=0)
        (work_dir / "first_identity_ctx_qnn.bin").write_bytes(b"first binary")
        stage._finalize_output(
            context,
            work_dir / "model_to_compile.onnx",
            output_dir,
            src_ctx_path=first_ctx,
        )
        first_published_model = onnx.load(str(context.output_path), load_external_data=False)
        first_ref = next(
            attr.s.decode("utf-8")
            for node in first_published_model.graph.node
            for attr in node.attribute
            if node.op_type == "EPContext" and attr.name == "ep_cache_context"
        )
        first_published_binary = output_dir / first_ref
        assert first_published_binary.read_bytes() == b"first binary"

        second_ctx = work_dir / "second_identity_ctx.onnx"
        create_epcontext_onnx(second_ctx, "second_identity_ctx_qnn.bin", embed_mode=0)
        (work_dir / "second_identity_ctx_qnn.bin").write_bytes(b"second binary")
        stage._finalize_output(
            context,
            work_dir / "model_to_compile.onnx",
            output_dir,
            src_ctx_path=second_ctx,
        )
        second_published_model = onnx.load(str(context.output_path), load_external_data=False)
        second_ref = next(
            attr.s.decode("utf-8")
            for node in second_published_model.graph.node
            for attr in node.attribute
            if node.op_type == "EPContext" and attr.name == "ep_cache_context"
        )

        assert second_ref != first_ref
        assert first_published_binary.read_bytes() == b"first binary"
        assert (output_dir / second_ref).read_bytes() == b"second binary"
        assert (output_dir / "mymodel_qnn_ctx_qnn.bin").read_bytes() == b"second binary"

    def test_updates_matching_cache_reference_with_malformed_main_context(self, tmp_path):
        """Binary renames follow the referenced file, not malformed main metadata."""
        from winml.modelkit.compiler import CompileContext, CompileStage

        work_dir = tmp_path / "work"
        output_dir = tmp_path / "output"
        work_dir.mkdir()
        output_dir.mkdir()

        original_model_path = tmp_path / "mymodel.onnx"
        create_simple_model(original_model_path)

        old_bin_name = "model_to_compile_qnn_ctx.bin"
        ctx_path = work_dir / "model_to_compile_qnn_ctx.onnx"
        create_epcontext_onnx(ctx_path, old_bin_name, embed_mode=0)
        model = onnx.load(str(ctx_path))
        node = model.graph.node[0]
        retained_attrs = [attr for attr in node.attribute if attr.name != "main_context"]
        del node.attribute[:]
        node.attribute.extend(retained_attrs)
        node.attribute.append(helper.make_attribute("main_context", "not-an-integer"))
        onnx.save(model, str(ctx_path))
        (work_dir / old_bin_name).write_bytes(b"fake binary content")

        context = CompileContext(
            model_path=original_model_path,
            config={
                "execution_provider": "qnn",
                "output_path": str(output_dir),
            },
            work_dir=work_dir,
        )

        CompileStage()._finalize_output(
            context, ctx_path.parent / "model_to_compile.onnx", output_dir
        )

        final_ctx_path = output_dir / "mymodel_qnn_ctx.onnx"
        final_model = onnx.load(str(final_ctx_path))
        cache_attr = next(
            attr for attr in final_model.graph.node[0].attribute if attr.name == "ep_cache_context"
        )
        final_bin_path = output_dir / cache_attr.s.decode("utf-8")
        assert final_bin_path.read_bytes() == b"fake binary content"

    def test_finalize_output_copies_all_referenced_context_binaries(self, tmp_path):
        """Every external EPContext reference must survive temporary workdir cleanup."""
        from winml.modelkit.compiler import CompileContext, CompileStage

        work_dir = tmp_path / "work"
        output_dir = tmp_path / "output"
        work_dir.mkdir()
        output_dir.mkdir()

        original_model_path = tmp_path / "mymodel.onnx"
        create_simple_model(original_model_path)

        main_bin_name = "model_to_compile_qnn_ctx.bin"
        secondary_bin_name = "model_to_compile_qnn_ctx_partition_1.bin"
        ctx_path = work_dir / "model_to_compile_qnn_ctx.onnx"
        create_epcontext_onnx(ctx_path, main_bin_name, embed_mode=0)
        model = onnx.load(str(ctx_path))
        model.graph.node.append(
            helper.make_node(
                "EPContext",
                inputs=[],
                outputs=["secondary_output"],
                name="ep_context_1",
                domain="com.microsoft",
                embed_mode=0,
                ep_cache_context=secondary_bin_name,
                main_context=0,
            )
        )
        onnx.save(model, str(ctx_path))
        (work_dir / main_bin_name).write_bytes(b"main binary")
        (work_dir / secondary_bin_name).write_bytes(b"secondary binary")

        context = CompileContext(
            model_path=original_model_path,
            config={
                "execution_provider": "qnn",
                "output_path": str(output_dir),
            },
            work_dir=work_dir,
        )

        CompileStage()._finalize_output(
            context, ctx_path.parent / "model_to_compile.onnx", output_dir
        )

        final_ctx_path = output_dir / "mymodel_qnn_ctx.onnx"
        final_model = onnx.load(str(final_ctx_path))
        cache_refs = {
            attr.s.decode("utf-8")
            for node in final_model.graph.node
            for attr in node.attribute
            if node.op_type == "EPContext" and attr.name == "ep_cache_context"
        }
        assert len(cache_refs) == 2
        assert any(
            ref.startswith("mymodel_qnn_ctx.") and ref.endswith(".bin") for ref in cache_refs
        )
        assert any(
            ref.startswith("mymodel_qnn_ctx_partition_1.") and ref.endswith(".bin")
            for ref in cache_refs
        )
        assert {((output_dir / ref).read_bytes()) for ref in cache_refs} == {
            b"main binary",
            b"secondary binary",
        }
        assert (output_dir / "mymodel_qnn_ctx.bin").read_bytes() == b"main binary"
        assert (output_dir / "mymodel_qnn_ctx_partition_1.bin").read_bytes() == b"secondary binary"

    def test_finalize_output_allows_secondary_partition_without_cache_reference(self, tmp_path):
        """Secondary EPContext nodes may share the main node's external binary."""
        from winml.modelkit.compiler import CompileContext, CompileStage

        work_dir = tmp_path / "work"
        output_dir = tmp_path / "output"
        work_dir.mkdir()
        output_dir.mkdir()

        original_model_path = tmp_path / "mymodel.onnx"
        create_simple_model(original_model_path)

        main_bin_name = "model_to_compile_qnn_ctx.bin"
        ctx_path = work_dir / "model_to_compile_qnn_ctx.onnx"
        create_epcontext_onnx(ctx_path, main_bin_name, embed_mode=0)
        model = onnx.load(str(ctx_path))
        model.graph.node.append(
            helper.make_node(
                "EPContext",
                inputs=[],
                outputs=["secondary_output"],
                name="ep_context_1",
                domain="com.microsoft",
                embed_mode=0,
                main_context=0,
            )
        )
        onnx.save(model, str(ctx_path))
        (work_dir / main_bin_name).write_bytes(b"shared binary")

        context = CompileContext(
            model_path=original_model_path,
            config={
                "execution_provider": "qnn",
                "output_path": str(output_dir),
            },
            work_dir=work_dir,
        )

        CompileStage()._finalize_output(
            context, ctx_path.parent / "model_to_compile.onnx", output_dir
        )

        assert (output_dir / "mymodel_qnn_ctx.onnx").exists()
        assert (output_dir / "mymodel_qnn_ctx.bin").read_bytes() == b"shared binary"

    def test_finalize_output_fails_for_missing_external_context_binary(self, tmp_path):
        """A finalized model must not retain a reference into the temporary workdir."""
        from winml.modelkit.compiler import CompileContext, CompileStage

        work_dir = tmp_path / "work"
        output_dir = tmp_path / "output"
        work_dir.mkdir()
        output_dir.mkdir()

        original_model_path = tmp_path / "mymodel.onnx"
        create_simple_model(original_model_path)
        missing_bin_name = "model_to_compile_qnn_ctx.bin"
        ctx_path = work_dir / "model_to_compile_qnn_ctx.onnx"
        create_epcontext_onnx(ctx_path, missing_bin_name, embed_mode=0)

        context = CompileContext(
            model_path=original_model_path,
            config={
                "execution_provider": "qnn",
                "output_path": str(output_dir),
            },
            work_dir=work_dir,
        )

        with pytest.raises(FileNotFoundError, match=missing_bin_name):
            CompileStage()._finalize_output(
                context, ctx_path.parent / "model_to_compile.onnx", output_dir
            )

        assert context.output_path is None
        assert not (output_dir / "mymodel_qnn_ctx.onnx").exists()

    def test_finalize_output_fails_for_escaping_external_context_reference(self, tmp_path):
        """External context references must remain inside the compiler workdir."""
        from winml.modelkit.compiler import CompileContext, CompileStage

        work_dir = tmp_path / "work"
        output_dir = tmp_path / "output"
        work_dir.mkdir()
        output_dir.mkdir()

        original_model_path = tmp_path / "mymodel.onnx"
        create_simple_model(original_model_path)
        ctx_path = work_dir / "model_to_compile_qnn_ctx.onnx"
        create_epcontext_onnx(ctx_path, "../escaped.bin", embed_mode=0)
        (tmp_path / "escaped.bin").write_bytes(b"outside workdir")

        context = CompileContext(
            model_path=original_model_path,
            config={
                "execution_provider": "qnn",
                "output_path": str(output_dir),
            },
            work_dir=work_dir,
        )

        with pytest.raises(ValueError, match="unsafe EPContext binary reference"):
            CompileStage()._finalize_output(
                context, ctx_path.parent / "model_to_compile.onnx", output_dir
            )

        assert context.output_path is None
        assert not (output_dir / "mymodel_qnn_ctx.onnx").exists()

    def test_finalize_output_fails_for_colliding_context_binary_destinations(self, tmp_path):
        """Distinct context binaries must not overwrite one another after renaming."""
        from winml.modelkit.compiler import CompileContext, CompileStage

        work_dir = tmp_path / "work"
        output_dir = tmp_path / "output"
        work_dir.mkdir()
        output_dir.mkdir()

        original_model_path = tmp_path / "mymodel.onnx"
        create_simple_model(original_model_path)

        renamed_source = "model_to_compile_qnn_ctx.bin"
        existing_final_name = "mymodel_qnn_ctx.bin"
        ctx_path = work_dir / "model_to_compile_qnn_ctx.onnx"
        create_epcontext_onnx(ctx_path, renamed_source, embed_mode=0)
        model = onnx.load(str(ctx_path))
        model.graph.node.append(
            helper.make_node(
                "EPContext",
                inputs=[],
                outputs=["secondary_output"],
                name="ep_context_1",
                domain="com.microsoft",
                embed_mode=0,
                ep_cache_context=existing_final_name,
                main_context=0,
            )
        )
        onnx.save(model, str(ctx_path))
        (work_dir / renamed_source).write_bytes(b"first binary")
        (work_dir / existing_final_name).write_bytes(b"second binary")

        context = CompileContext(
            model_path=original_model_path,
            config={
                "execution_provider": "qnn",
                "output_path": str(output_dir),
            },
            work_dir=work_dir,
        )

        with pytest.raises(ValueError, match="same output path"):
            CompileStage()._finalize_output(
                context, ctx_path.parent / "model_to_compile.onnx", output_dir
            )

        assert context.output_path is None
        assert not (output_dir / "mymodel_qnn_ctx.onnx").exists()

    def test_finalize_output_copies_partition_schematic_sidecar(self, tmp_path):
        """Persist the exact QNN schematic named by EPContext partition metadata."""
        from winml.modelkit.compiler import CompileContext, CompileStage

        work_dir = tmp_path / "work"
        output_dir = tmp_path / "output"
        work_dir.mkdir()
        output_dir.mkdir()

        original_model_path = tmp_path / "mymodel.onnx"
        create_simple_model(original_model_path)

        partition_name = "unit_test_partition_123"
        ctx_path = work_dir / "model_to_compile_qnn_ctx.onnx"
        create_epcontext_onnx(
            ctx_path,
            "model_to_compile_qnn_ctx.bin",
            embed_mode=0,
            partition_name=partition_name,
        )
        (work_dir / "model_to_compile_qnn_ctx.bin").write_bytes(b"fake binary")
        (work_dir / f"{partition_name}_schematic.bin").write_bytes(b"schematic")
        (work_dir / "unrelated_schematic.bin").write_bytes(b"unrelated")

        context = CompileContext(
            model_path=original_model_path,
            config={
                "execution_provider": "qnn",
                "output_path": str(output_dir),
            },
            work_dir=work_dir,
        )

        CompileStage()._finalize_output(
            context, ctx_path.parent / "model_to_compile.onnx", output_dir
        )

        assert (output_dir / f"{partition_name}_schematic.bin").read_bytes() == b"schematic"
        assert not (output_dir / "unrelated_schematic.bin").exists()

    def test_finalize_output_copies_only_main_partition_schematic(self, tmp_path):
        """Secondary partition schematics must not be paired with the main context binary."""
        from winml.modelkit.compiler import CompileContext, CompileStage

        work_dir = tmp_path / "work"
        output_dir = tmp_path / "output"
        work_dir.mkdir()
        output_dir.mkdir()

        original_model_path = tmp_path / "mymodel.onnx"
        create_simple_model(original_model_path)

        main_partition = "main_partition"
        secondary_partition = "secondary_partition"
        ctx_path = work_dir / "model_to_compile_qnn_ctx.onnx"
        create_epcontext_onnx(
            ctx_path,
            "model_to_compile_qnn_ctx.bin",
            embed_mode=0,
            partition_name=main_partition,
        )
        model = onnx.load(str(ctx_path))
        model.graph.node.append(
            helper.make_node(
                "EPContext",
                inputs=[],
                outputs=["secondary_output"],
                name="ep_context_1",
                domain="com.microsoft",
                embed_mode=0,
                ep_cache_context="secondary.bin",
                main_context=0,
                partition_name=secondary_partition,
            )
        )
        onnx.save(model, str(ctx_path))

        (work_dir / "model_to_compile_qnn_ctx.bin").write_bytes(b"fake binary")
        (work_dir / "secondary.bin").write_bytes(b"secondary binary")
        (work_dir / f"{main_partition}_schematic.bin").write_bytes(b"main")
        (work_dir / f"{secondary_partition}_schematic.bin").write_bytes(b"secondary")

        context = CompileContext(
            model_path=original_model_path,
            config={
                "execution_provider": "qnn",
                "output_path": str(output_dir),
            },
            work_dir=work_dir,
        )

        CompileStage()._finalize_output(
            context, ctx_path.parent / "model_to_compile.onnx", output_dir
        )

        assert (output_dir / f"{main_partition}_schematic.bin").read_bytes() == b"main"
        assert not (output_dir / f"{secondary_partition}_schematic.bin").exists()

    def test_finalize_output_rejects_path_like_partition_schematic_name(self, tmp_path):
        """EPContext partition metadata must not escape sidecar directories."""
        from winml.modelkit.compiler import CompileContext, CompileStage

        work_dir = tmp_path / "work"
        output_dir = tmp_path / "output"
        escaped_dir = tmp_path / "escaped"
        work_dir.mkdir()
        output_dir.mkdir()
        escaped_dir.mkdir()

        original_model_path = tmp_path / "mymodel.onnx"
        create_simple_model(original_model_path)

        ctx_path = work_dir / "model_to_compile_qnn_ctx.onnx"
        create_epcontext_onnx(
            ctx_path,
            "model_to_compile_qnn_ctx.bin",
            embed_mode=0,
            partition_name="../escaped/not_a_stem",
        )
        (work_dir / "model_to_compile_qnn_ctx.bin").write_bytes(b"fake binary")
        (escaped_dir / "not_a_stem_schematic.bin").write_bytes(b"escaped")

        context = CompileContext(
            model_path=original_model_path,
            config={
                "execution_provider": "qnn",
                "output_path": str(output_dir),
            },
            work_dir=work_dir,
        )

        CompileStage()._finalize_output(
            context, ctx_path.parent / "model_to_compile.onnx", output_dir
        )

        assert not (output_dir / "not_a_stem_schematic.bin").exists()

    def test_finalize_output_rejects_invalid_utf8_partition_name(self, tmp_path):
        """Malformed partition metadata must not abort compiler finalization."""
        from winml.modelkit.compiler import CompileContext, CompileStage

        work_dir = tmp_path / "work"
        output_dir = tmp_path / "output"
        work_dir.mkdir()
        output_dir.mkdir()

        original_model_path = tmp_path / "mymodel.onnx"
        create_simple_model(original_model_path)

        ctx_path = work_dir / "model_to_compile_qnn_ctx.onnx"
        create_epcontext_onnx(
            ctx_path,
            "model_to_compile_qnn_ctx.bin",
            embed_mode=0,
            partition_name="placeholder",
        )
        model = onnx.load(str(ctx_path))
        partition_attr = next(
            attr for attr in model.graph.node[0].attribute if attr.name == "partition_name"
        )
        partition_attr.s = b"\xff"
        onnx.save(model, str(ctx_path))

        (work_dir / "model_to_compile_qnn_ctx.bin").write_bytes(b"fake binary")
        (work_dir / "placeholder_schematic.bin").write_bytes(b"unbound")

        context = CompileContext(
            model_path=original_model_path,
            config={
                "execution_provider": "qnn",
                "output_path": str(output_dir),
            },
            work_dir=work_dir,
        )

        CompileStage()._finalize_output(
            context, ctx_path.parent / "model_to_compile.onnx", output_dir
        )

        assert (output_dir / "mymodel_qnn_ctx.onnx").exists()
        assert not (output_dir / "placeholder_schematic.bin").exists()

    def test_skips_update_when_embedded(self, tmp_path):
        """Test that ep_cache_context is not modified when embed_mode=1.

        Key branch: attrs["embed_mode"].i != 0 -> skip update
        """
        from winml.modelkit.compiler import CompileContext, CompileStage

        # Setup: work_dir with embedded EPContext
        work_dir = tmp_path / "work"
        output_dir = tmp_path / "output"
        work_dir.mkdir()
        output_dir.mkdir()

        original_model_path = tmp_path / "mymodel.onnx"
        create_simple_model(original_model_path)

        # Create embedded EPContext (embed_mode=1)
        ctx_path = work_dir / "model_to_compile_qnn_ctx.onnx"
        create_epcontext_onnx(ctx_path, "embedded_data", embed_mode=1)

        context = CompileContext(
            model_path=original_model_path,
            config={
                "execution_provider": "qnn",
                "output_path": str(output_dir),
            },
            work_dir=work_dir,
        )

        stage = CompileStage()
        stage._finalize_output(context, ctx_path.parent / "model_to_compile.onnx", output_dir)

        # Verify: output should exist but ep_cache_context should be unchanged
        final_ctx_path = output_dir / "mymodel_qnn_ctx.onnx"
        assert final_ctx_path.exists()

        model = onnx.load(str(final_ctx_path))
        for node in model.graph.node:
            if node.op_type == "EPContext":
                for attr in node.attribute:
                    if attr.name == "ep_cache_context":
                        # Should remain as original (embedded doesn't need path update)
                        assert attr.s == b"embedded_data"
                        break

    def test_warns_when_epcontext_not_found(self, tmp_path):
        """Test that warning is added when EPContext file not found.

        Key branch: if src_ctx_path is None: add_warning
        """
        from winml.modelkit.compiler import CompileContext, CompileStage

        work_dir = tmp_path / "work"
        output_dir = tmp_path / "output"
        work_dir.mkdir()
        output_dir.mkdir()

        original_model_path = tmp_path / "mymodel.onnx"
        create_simple_model(original_model_path)

        # Don't create any EPContext file

        context = CompileContext(
            model_path=original_model_path,
            config={
                "execution_provider": "qnn",
                "output_path": str(output_dir),
            },
            work_dir=work_dir,
        )

        stage = CompileStage()
        # Pass a model_path in work_dir that doesn't have corresponding ctx
        stage._finalize_output(context, work_dir / "model_to_compile.onnx", output_dir)

        # Verify warning was added
        assert len(context.warnings) == 1
        assert "EPContext model not found" in context.warnings[0]

    def test_finalize_output_respects_user_file_path(self, tmp_path):
        """Test that -o file path is used as the final output filename.

        Before the fix, _finalize_output always generated
        '{original_stem}_{device}_ctx.onnx', ignoring the user-specified
        output path.
        """
        from winml.modelkit.compiler import CompileContext, CompileStage

        work_dir = tmp_path / "work"
        output_dir = tmp_path / "output"
        work_dir.mkdir()
        output_dir.mkdir()

        original_model_path = tmp_path / "mymodel.onnx"
        create_simple_model(original_model_path)

        # User wants: output/compiled.onnx (not output/mymodel_qnn_ctx.onnx)
        user_output = output_dir / "compiled.onnx"

        # Create EPContext in work_dir
        ctx_path = work_dir / "model_to_compile_qnn_ctx.onnx"
        create_epcontext_onnx(ctx_path, "model_to_compile_qnn_ctx.bin", embed_mode=1)

        context = CompileContext(
            model_path=original_model_path,
            config={
                "execution_provider": "qnn",
                "output_path": str(user_output),
            },
            work_dir=work_dir,
        )

        stage = CompileStage()
        stage._finalize_output(context, ctx_path.parent / "model_to_compile.onnx", output_dir)

        # Should use the user-specified filename, not the auto-generated one
        assert context.output_path == user_output
        assert user_output.exists()
        # The auto-generated name should NOT exist
        auto_name = output_dir / "model_to_compile_qnn_ctx.onnx"
        assert not auto_name.exists()

    def test_finalize_output_bin_uses_user_stem(self, tmp_path):
        """Test that .bin companion file uses the user-specified stem.

        Before the fix, .bin was always named '{original_stem}_{device}_ctx.bin'
        even when the user specified a custom output filename.
        """
        from winml.modelkit.compiler import CompileContext, CompileStage

        work_dir = tmp_path / "work"
        output_dir = tmp_path / "output"
        work_dir.mkdir()
        output_dir.mkdir()

        original_model_path = tmp_path / "mymodel.onnx"
        create_simple_model(original_model_path)

        user_output = output_dir / "compiled.onnx"

        # Create EPContext with external bin (embed_mode=0)
        ctx_path = work_dir / "model_to_compile_qnn_ctx.onnx"
        old_bin_name = "model_to_compile_qnn_ctx.bin"
        create_epcontext_onnx(ctx_path, old_bin_name, embed_mode=0)

        # Create the bin file
        (work_dir / old_bin_name).write_bytes(b"fake binary")

        context = CompileContext(
            model_path=original_model_path,
            config={
                "execution_provider": "qnn",
                "output_path": str(user_output),
            },
            work_dir=work_dir,
        )

        stage = CompileStage()
        stage._finalize_output(context, ctx_path.parent / "model_to_compile.onnx", output_dir)

        # Bin should use the user-specified stem: "compiled.bin"
        expected_bin = output_dir / "compiled.bin"
        assert expected_bin.exists(), (
            f"Expected {expected_bin}, found: {list(output_dir.iterdir())}"
        )
        # The old auto-generated name should NOT exist
        assert not (output_dir / "model_to_compile_qnn_ctx.bin").exists()


class TestCompilerPipeline:
    """Test Compiler class pipeline configuration."""

    def test_new_pipeline_stages(self):
        """Verify the pipeline uses the new stages."""
        from winml.modelkit.compiler import (
            Compiler,
            CompileStage,
            OptimizeStage,
            QFormatConvertStage,
        )

        # Reset cached stages
        Compiler._stages = None
        stages = Compiler._get_stages()

        assert len(stages) == 3
        assert stages[0] is OptimizeStage
        assert stages[1] is QFormatConvertStage
        assert stages[2] is CompileStage

        # Clean up
        Compiler._stages = None

    def test_passthrough_when_no_config(self, tmp_path):
        """Compile with no config returns passthrough result."""
        from winml.modelkit.compiler import Compiler

        model_path = tmp_path / "model.onnx"
        create_simple_model(model_path)

        compiler = Compiler()
        result = compiler.compile(model_path)

        assert result.success is True
        assert "passthrough" in result.warnings[0].lower()

    def test_final_compile_releases_shared_session_options(self, tmp_path):
        """The last compile in a run must not retain native ORT SessionOptions."""
        from unittest.mock import MagicMock

        from winml.modelkit.compiler import Compiler

        model_path = tmp_path / "model.onnx"
        create_simple_model(model_path)
        shared_options = object()

        class _Stage:
            name = "fake"

            @classmethod
            def should_run(cls, _context):
                return True

            def process(self, context):
                context.shared_session_options = shared_options
                return context

        old_stages = Compiler._stages
        Compiler._stages = [_Stage]
        try:
            config = MagicMock()
            config.to_dict.return_value = {"execution_provider": "qnn"}
            config.verbose = False
            compiler = Compiler()
            compiler.compile(model_path, config=config)
        finally:
            Compiler._stages = old_stages

        assert compiler.shared_session_options is None
