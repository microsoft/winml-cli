# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for compiler stages."""

from pathlib import Path

import onnx
from onnx import TensorProto, helper


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
            def applies_to(self, ep: str) -> bool:
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
            def applies_to(self, ep: str) -> bool:
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


def create_epcontext_onnx(path: Path, bin_name: str, embed_mode: int = 0) -> None:
    """Create mock EPContext ONNX model for testing.

    Args:
        path: Output path for the ONNX model
        bin_name: Name of the external binary file (for ep_cache_context attribute)
        embed_mode: 0=external binary, 1=embedded
    """
    # Create EPContext node with attributes
    ep_context_node = helper.make_node(
        "EPContext",
        inputs=[],
        outputs=["output"],
        name="ep_context_0",
        domain="com.microsoft",
        embed_mode=embed_mode,
        ep_cache_context=bin_name,
        main_context=1,  # This is the main context
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

        context = CompileContext(
            model_path=model_path,
            config={
                "execution_provider": "nv_tensorrt_rtx",
                "provider_options": {"device_type": "GPU", "precision": "fp16"},
                "enable_ep_context": True,
                "validate": False,
            },
        )

        mock_session_cls = MagicMock(return_value=fake_winml_session)
        with patch.dict(
            "winml.modelkit.compiler.stages.compile.COMPILER_SESSION_MAPPING",
            {"ort": mock_session_cls},
            clear=False,
        ):
            stage = CompileStage()
            stage.process(context)

        passed_ep_config = mock_session_cls.call_args.kwargs["ep_config"]
        assert passed_ep_config.provider_options == {"device_type": "GPU", "precision": "fp16"}
        assert mock_session_cls.call_args.kwargs["ep"] == "NvTensorRTRTXExecutionProvider"


class TestCompileStageFinalizeOutput:
    """Test CompileStage._finalize_output method."""

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

        # Verify: output EPContext should preserve the source ctx filename
        final_ctx_path = output_dir / "model_to_compile_qnn_ctx.onnx"
        assert final_ctx_path.exists(), f"Expected {final_ctx_path} to exist"

        # Load and check the attribute was updated
        model = onnx.load(str(final_ctx_path))
        for node in model.graph.node:
            if node.op_type == "EPContext":
                for attr in node.attribute:
                    if attr.name == "ep_cache_context":
                        # Should be updated to new name based on the source ctx stem
                        assert b"model_to_compile_qnn_ctx" in attr.s, (
                            f"Expected updated name, got {attr.s}"
                        )
                        break

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
        final_ctx_path = output_dir / "model_to_compile_qnn_ctx.onnx"
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
