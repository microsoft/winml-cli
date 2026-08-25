# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLSession tests with simple ONNX model.

Test Scope:
1. Instantiate WinMLSession with an explicit EPDeviceTarget
2. Verify session state, providers, and inference behavior
3. Test perf() context manager

Key Principle:
- Use EPDeviceTarget-based construction (Task 7 API)
- CPU tests use the real OrtEpDevice with a mocked WinMLEPRegistry
- NPU/QNN tests use fake OrtEpDevice fixtures (mocked ORT)
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from onnx import TensorProto, helper, load, numpy_helper, save, save_model

from winml.modelkit.compiler import EPConfig
from winml.modelkit.session import (
    EPDeviceTarget,
    PerfContext,
    WinMLEPDevice,
    WinMLEPMonitorMismatch,
    WinMLSession,
)
from winml.modelkit.session.session import SessionState


def _stub_registry(monkeypatch: pytest.MonkeyPatch, ep_device: object) -> MagicMock:
    """Provide the public registry contract for ergonomic session construction."""
    from winml.modelkit.session.ep_registry import WinMLEPRegistry

    registry = MagicMock()
    registry.auto_device.return_value = ep_device
    registry.available_eps.return_value = frozenset(
        {getattr(getattr(ep_device, "device", None), "ep_name", "CPUExecutionProvider")}
    )
    monkeypatch.setattr(WinMLEPRegistry, "instance", classmethod(lambda _cls: registry))
    return registry


def _write_fake_epcontext(session: WinMLSession, path: str) -> None:
    """Write a valid EPContext graph and its optional external binary."""
    ctx_path = Path(path)
    if session._embed_context:
        cache_value = b"embedded context"
    else:
        binary_path = ctx_path.with_name(f"{ctx_path.stem}_qnn.bin")
        binary_path.write_bytes(b"external context")
        cache_value = binary_path.name
    node = helper.make_node(
        "EPContext",
        inputs=[],
        outputs=["output"],
        name="ep_context_0",
        domain="com.microsoft",
        embed_mode=1 if session._embed_context else 0,
        ep_cache_context=cache_value,
        main_context=1,
    )
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    graph = helper.make_graph([node], "epcontext_graph", [], [output])
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 9
    save(model, ctx_path)


def _compile_with_fake_ort(session: WinMLSession) -> MagicMock:
    """Compile through mocked ORT while preserving its file-output contract."""
    inference_session = MagicMock()
    inference_session.get_providers.return_value = ["QNNExecutionProvider"]
    model_compiler = MagicMock()

    model_compiler.return_value.compile_to_file.side_effect = lambda path: _write_fake_epcontext(
        session, path
    )
    with (
        patch("winml.modelkit.session.session._build_session_options", return_value=MagicMock()),
        patch("winml.modelkit.session.session.ort.ModelCompiler", model_compiler),
        patch(
            "winml.modelkit.session.session.ort.InferenceSession",
            return_value=inference_session,
        ),
    ):
        session.compile()
    return model_compiler


def _cache_path(session: WinMLSession) -> Path:
    """Return the deterministic EPContext path for this test session."""
    return session._epcontext_cache_path(session._epcontext_cache_identity())


def _compiled_generation(session: WinMLSession, model_compiler: MagicMock) -> Path:
    """Return the single compiled generation and validate its identity namespace."""
    model_compiler.return_value.compile_to_file.assert_called_once()
    generation = Path(model_compiler.return_value.compile_to_file.call_args.args[0])
    cache_path = _cache_path(session)
    assert generation.parent == cache_path.parent
    assert generation.name.startswith(f"{cache_path.stem}_")
    assert generation.suffix == cache_path.suffix
    return generation


class TestWinMLSessionInstantiation:
    """Test WinMLSession instantiation with EPDeviceTarget-based selection."""

    def test_session_init_with_npu_device(
        self, simple_matmul_onnx: Path, qnn_npu_ep_device: EPDeviceTarget, fake_ort_npu: MagicMock
    ):
        """Test that WinMLSession can be initialized with an NPU WinMLEPDevice.

        ORT InferenceSession is also mocked because the fake_ort_npu MagicMock
        cannot be passed to add_provider_for_devices() (requires a real C++ object).
        """
        with (
            patch("winml.modelkit.session.session.ort.InferenceSession"),
            patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()),
        ):
            session = WinMLSession(onnx_path=simple_matmul_onnx, ep_device=qnn_npu_ep_device)

        assert session.device == "npu"
        assert session.state == SessionState.INITIALIZED

    def test_session_init_with_cpu_ep_device(
        self, simple_matmul_onnx: Path, cpu_ep_device: EPDeviceTarget
    ):
        """Test that WinMLSession can be initialized with a CPU WinMLEPDevice."""
        session = WinMLSession(onnx_path=simple_matmul_onnx, ep_device=cpu_ep_device)

        assert session.device == "cpu"
        assert session.state == SessionState.INITIALIZED

    def test_session_init_file_not_found(self, tmp_path: Path, cpu_ep_device: EPDeviceTarget):
        """Test that WinMLSession raises an ORT error for a non-existent ONNX file."""
        from onnxruntime.capi.onnxruntime_pybind11_state import NoSuchFile

        with pytest.raises(NoSuchFile):
            WinMLSession(onnx_path=tmp_path / "nonexistent.onnx", ep_device=cpu_ep_device)

    def test_ep_name_is_none_before_compile(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: EPDeviceTarget,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """ep_name returns None before compile() since no providers are bound yet."""
        _stub_registry(monkeypatch, cpu_ep_device)
        session = WinMLSession(onnx_path=simple_matmul_onnx, device="cpu", ep="cpu")
        assert session.ep_name is None

    def test_ep_name_after_compile(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: EPDeviceTarget,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """ep_name returns the primary provider name once the session is built."""
        _stub_registry(monkeypatch, cpu_ep_device)
        session = WinMLSession(onnx_path=simple_matmul_onnx, device="cpu", ep="cpu")
        session.compile()
        assert isinstance(session.ep_name, str)
        assert session.ep_name.endswith("ExecutionProvider")

    def test_explicit_ep_cpu_binds_cpu_execution_provider(self, simple_matmul_onnx: Path):
        """`--ep cpu` must bind CPUExecutionProvider explicitly.

        Regression: previously the explicit-EP branch carried a
        `self._ep != "cpu"` exception, so `ep="cpu"` fell through to
        PREFER_CPU policy. On systems with OpenVINO (or any other
        CPU-capable EP) registered, ORT then chose OV-on-CPU as the primary
        and silently ignored the user's `--ep cpu` choice. The fix routes
        `ep="cpu"` through `add_provider_for_devices` like any other EP, so
        the resulting session has CPUExecutionProvider as the primary
        provider regardless of what else is registered.
        """
        session = WinMLSession(onnx_path=simple_matmul_onnx, device="cpu", ep="cpu")
        session.compile()
        assert session.ep_name == "CPUExecutionProvider"
        assert session._session.get_providers()[0] == "CPUExecutionProvider"


class TestWinMLSessionCompilation:
    """Test WinMLSession compilation (EPContext creation)."""

    @pytest.mark.skip(reason="Lazy init design is not implemented in source code.")
    def test_compile_creates_epcontext(
        self, simple_matmul_onnx: Path, qnn_npu_ep_device: EPDeviceTarget, fake_ort_npu: MagicMock
    ):
        """
        Test that compile() creates EPContext file.

        With new lazy init design:
        - compile() creates EPContext file only
        - _init_session() (called by run()) creates InferenceSession
        """
        with patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg:
            mock_reg.instance.return_value.register_ep.return_value = [fake_ort_npu]
            session = WinMLSession(onnx_path=simple_matmul_onnx, ep_device=qnn_npu_ep_device)

        # Compile creates EPContext file
        session.compile()

        # EPContext file should exist
        ctx_path = simple_matmul_onnx.parent / f"{simple_matmul_onnx.stem}_ctx.onnx"
        assert ctx_path.exists(), f"EPContext not created: {ctx_path}"

        # Session not created yet (lazy init)
        assert not session.is_compiled

    def test_compile_is_idempotent(self, cpu_winml_session: WinMLSession):
        """Test that calling compile() multiple times is safe (idempotent).

        __init__ already creates _session, so compile() is a no-op and the
        _session object reference is unchanged. State transitions to COMPILED
        only after run() is called.
        """
        session = cpu_winml_session

        # _session is already set by __init__; compile() returns immediately
        first_session = session._session
        assert first_session is not None
        session.compile()
        # State stays INITIALIZED — compile() returned early, no state change
        assert session.state == SessionState.INITIALIZED
        assert session._session is first_session

        # Second compile also a no-op
        session.compile()
        assert session._session is first_session

    def test_compile_rebuilds_legacy_cache_without_identity_marker(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
    ) -> None:
        """A newer sibling context without cache identity is not reusable."""
        ctx_path = simple_matmul_onnx.with_name(f"{simple_matmul_onnx.stem}_npu_ctx.onnx")
        ctx_path.write_bytes(b"legacy context")
        source_mtime = simple_matmul_onnx.stat().st_mtime_ns
        os.utime(ctx_path, ns=(source_mtime + 1_000_000, source_mtime + 1_000_000))

        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
        )
        model_compiler = _compile_with_fake_ort(session)
        compiled_path = _compiled_generation(session, model_compiler)

        assert session.running_model_path == compiled_path
        assert ctx_path.read_bytes() == b"legacy context"

    def test_compile_rebuilds_cache_with_non_object_marker(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
    ) -> None:
        """A valid JSON marker with the wrong shape is a cache miss, not an error."""
        legacy_path = simple_matmul_onnx.with_name(f"{simple_matmul_onnx.stem}_npu_ctx.onnx")
        legacy_path.write_bytes(b"legacy context")
        marker_path = WinMLSession._epcontext_cache_marker_path(legacy_path)
        marker_path.write_text("[]", encoding="utf-8")
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
        )

        model_compiler = _compile_with_fake_ort(session)
        compiled_path = _compiled_generation(session, model_compiler)

        assert session.running_model_path == compiled_path

    def test_compile_reuses_cache_with_matching_identity(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
    ) -> None:
        """An unchanged source and compile identity reuse the sibling context."""
        first_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(
                provider="qnn",
                provider_options={"htp_performance_mode": "burst"},
                enable_ep_context=True,
            ),
        )
        first_compiler = _compile_with_fake_ort(first_session)
        first_compiler.return_value.compile_to_file.assert_called_once()
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(
                provider="qnn",
                provider_options={"htp_performance_mode": "burst"},
                enable_ep_context=True,
            ),
        )

        model_compiler = _compile_with_fake_ort(session)

        model_compiler.assert_not_called()

    def test_compile_rebuilds_cache_when_provider_options_change(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
    ) -> None:
        """Effective provider options are part of the direct-session cache key."""
        old_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(
                provider="qnn",
                provider_options={"htp_performance_mode": "default"},
                enable_ep_context=True,
            ),
        )
        _compile_with_fake_ort(old_session)
        new_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(
                provider="qnn",
                provider_options={"htp_performance_mode": "burst"},
                enable_ep_context=True,
            ),
        )

        model_compiler = _compile_with_fake_ort(new_session)
        ctx_path = _compiled_generation(new_session, model_compiler)

        assert new_session.running_model_path == ctx_path

    def test_compile_rebuilds_cache_when_provider_option_file_content_changes(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
        tmp_path: Path,
    ) -> None:
        """Existing file-valued provider options are fingerprinted by content."""
        option_file = tmp_path / "compiler-input.bin"
        option_file.write_bytes(b"alpha")
        first_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(
                provider="qnn",
                provider_options={"compiler_input": str(option_file)},
                provider_option_file_keys={"compiler_input"},
                enable_ep_context=True,
            ),
        )
        _compile_with_fake_ort(first_session)
        original_stat = option_file.stat()
        option_file.write_bytes(b"bravo")
        os.utime(
            option_file,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        second_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(
                provider="qnn",
                provider_options={"compiler_input": str(option_file)},
                provider_option_file_keys={"compiler_input"},
                enable_ep_context=True,
            ),
        )

        model_compiler = _compile_with_fake_ort(second_session)
        ctx_path = _compiled_generation(second_session, model_compiler)

        assert second_session.running_model_path == ctx_path

    def test_relative_provider_option_file_is_canonicalized_for_identity_and_ort(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """File-valued relative options are resolved once before hashing and ORT binding."""
        option_name = "compiler-input.bin"
        cwd_dir = tmp_path / "cwd"
        cwd_dir.mkdir()
        cwd_option_file = cwd_dir / option_name
        cwd_option_file.write_bytes(b"cwd option")
        model_dir_option_file = simple_matmul_onnx.parent / option_name
        model_dir_option_file.write_bytes(b"model option")
        monkeypatch.chdir(cwd_dir)
        captured_provider_options: list[dict[str, str]] = []

        def _session_options(*_args, provider_options, **_kwargs):
            captured_provider_options.append(dict(provider_options))
            return MagicMock()

        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(
                provider="qnn",
                provider_options={"compiler_input": option_name},
                provider_option_file_keys={"compiler_input"},
                enable_ep_context=True,
            ),
        )
        expected_path = str(cwd_option_file.resolve())
        monkeypatch.setattr(
            "winml.modelkit.session.session._build_session_options",
            _session_options,
        )
        monkeypatch.setattr(
            "winml.modelkit.session.session.ort.ModelCompiler",
            MagicMock(
                return_value=SimpleNamespace(
                    compile_to_file=lambda path: _write_fake_epcontext(session, path)
                )
            ),
        )
        runtime_session = MagicMock()
        runtime_session.get_providers.return_value = ["QNNExecutionProvider"]
        monkeypatch.setattr(
            "winml.modelkit.session.session.ort.InferenceSession",
            lambda *_args, **_kwargs: runtime_session,
        )

        session.compile()

        identity = session._epcontext_cache_identity()
        assert session._provider_options["compiler_input"] == expected_path
        assert identity["provider_option_files"]["compiler_input"]["path"] == expected_path
        assert captured_provider_options
        assert all(
            options["compiler_input"] == expected_path for options in captured_provider_options
        )

    def test_plain_provider_option_matching_cwd_file_is_not_rewritten(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Existing files do not make unrelated provider-option values file-backed."""
        (tmp_path / "default").write_bytes(b"unrelated")
        monkeypatch.chdir(tmp_path)
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(
                provider="qnn",
                provider_options={"htp_performance_mode": "default"},
                enable_ep_context=True,
            ),
        )

        identity = session._epcontext_cache_identity()

        assert session._provider_options["htp_performance_mode"] == "default"
        assert "htp_performance_mode" not in identity["provider_option_files"]

    def test_declared_provider_option_file_must_resolve(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
    ) -> None:
        """An explicitly file-backed option cannot retain an ambiguous relative value."""
        with pytest.raises(ValueError, match="compiler_input"):
            WinMLSession(
                onnx_path=simple_matmul_onnx,
                ep_device=qnn_npu_ep_device,
                ep_config=EPConfig(
                    provider="qnn",
                    provider_options={"compiler_input": "missing.bin"},
                    provider_option_file_keys={"compiler_input"},
                    enable_ep_context=True,
                ),
            )

    def test_different_compile_identities_use_distinct_context_paths(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
    ) -> None:
        """One identity cannot overwrite artifacts loaded by another session."""
        sessions = [
            WinMLSession(
                onnx_path=simple_matmul_onnx,
                ep_device=qnn_npu_ep_device,
                ep_config=EPConfig(
                    provider="qnn",
                    provider_options={"mode": mode},
                    enable_ep_context=True,
                ),
            )
            for mode in ("first", "second")
        ]

        paths = [
            session._epcontext_cache_path(session._epcontext_cache_identity())
            for session in sessions
        ]

        assert paths[0] != paths[1]
        assert all(path.parent == simple_matmul_onnx.parent for path in paths)
        assert all(path.name.startswith(f"{simple_matmul_onnx.stem}_npu_") for path in paths)
        assert all(path.name.endswith("_ctx.onnx") for path in paths)

    def test_compile_rebuilds_cache_when_embed_mode_changes(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
    ) -> None:
        """Embedded and external EPContext artifacts never share a cache entry."""
        old_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True, embed_context=False),
        )
        _compile_with_fake_ort(old_session)
        new_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True, embed_context=True),
        )

        model_compiler = _compile_with_fake_ort(new_session)
        ctx_path = _compiled_generation(new_session, model_compiler)

        assert new_session.running_model_path == ctx_path

    def test_compile_rebuilds_cache_when_external_context_binary_is_missing(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
    ) -> None:
        """A marker cannot make an EPContext with a missing binary reusable."""
        first_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
        )
        _compile_with_fake_ort(first_session)
        ctx_path = first_session.running_model_path
        binary_path = ctx_path.with_name(f"{ctx_path.stem}_qnn.bin")
        binary_path.unlink()
        second_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
        )

        model_compiler = _compile_with_fake_ort(second_session)

        _compiled_generation(second_session, model_compiler)

    def test_compile_rebuilds_cache_when_external_context_binary_changes(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
    ) -> None:
        """A replaced EPContext binary invalidates an otherwise matching marker."""
        first_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
        )
        _compile_with_fake_ort(first_session)
        ctx_path = first_session.running_model_path
        binary_path = ctx_path.with_name(f"{ctx_path.stem}_qnn.bin")
        binary_path.write_bytes(b"replaced external context")
        second_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
        )

        model_compiler = _compile_with_fake_ort(second_session)

        _compiled_generation(second_session, model_compiler)

    def test_compile_rebuilds_cache_when_binary_content_changes_with_same_metadata(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
    ) -> None:
        """Content digests catch replacements that preserve size and mtime."""
        first_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
        )
        _compile_with_fake_ort(first_session)
        ctx_path = first_session.running_model_path
        binary_path = ctx_path.with_name(f"{ctx_path.stem}_qnn.bin")
        original_stat = binary_path.stat()
        binary_path.write_bytes(b"tampered content")
        os.utime(
            binary_path,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        second_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
        )

        model_compiler = _compile_with_fake_ort(second_session)

        _compiled_generation(second_session, model_compiler)

    def test_compile_failure_preserves_other_identity_cache_and_uses_source(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
    ) -> None:
        """A failed identity compile leaves other caches intact and uses source."""
        first_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(
                provider="qnn",
                provider_options={"mode": "old"},
                enable_ep_context=True,
            ),
        )
        _compile_with_fake_ort(first_session)
        ctx_path = _cache_path(first_session)
        marker_path = first_session._epcontext_cache_marker_path(ctx_path)
        assert marker_path.is_file()
        second_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(
                provider="qnn",
                provider_options={"mode": "new"},
                enable_ep_context=True,
            ),
        )
        runtime_session = MagicMock()
        runtime_session.get_providers.return_value = ["QNNExecutionProvider"]
        inference_session = MagicMock(return_value=runtime_session)
        model_compiler = MagicMock()
        model_compiler.return_value.compile_to_file.side_effect = RuntimeError("compile failed")
        with (
            patch(
                "winml.modelkit.session.session._build_session_options",
                return_value=MagicMock(),
            ),
            patch("winml.modelkit.session.session.ort.ModelCompiler", model_compiler),
            patch(
                "winml.modelkit.session.session.ort.InferenceSession",
                inference_session,
            ),
        ):
            second_session.compile()

        assert marker_path.exists()
        assert inference_session.call_count == 1
        assert inference_session.call_args.args[0] == str(simple_matmul_onnx)

    def test_compile_failure_removes_private_generation_and_sidecar(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Failed private generations are removed with sidecars before fallback."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
        )
        written_paths: list[Path] = []

        class _FailingCompiler:
            def __init__(self, *_args, **_kwargs):
                pass

            def compile_to_file(self, path: str) -> None:
                generation_path = Path(path)
                sidecar_path = generation_path.with_name(f"{generation_path.stem}_partial.bin")
                generation_path.write_bytes(b"partial context")
                sidecar_path.write_bytes(b"partial sidecar")
                written_paths.extend([generation_path, sidecar_path])
                raise RuntimeError("compile failed")

        runtime_session = MagicMock()
        runtime_session.get_providers.return_value = ["QNNExecutionProvider"]
        monkeypatch.setattr(
            "winml.modelkit.session.session._build_session_options",
            lambda *_args, **_kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            "winml.modelkit.session.session.ort.ModelCompiler",
            _FailingCompiler,
        )
        monkeypatch.setattr(
            "winml.modelkit.session.session.ort.InferenceSession",
            lambda *_args, **_kwargs: runtime_session,
        )

        session.compile()

        assert session.running_model_path == simple_matmul_onnx
        assert written_paths
        assert all(not path.exists() for path in written_paths)

    def test_compile_rebuilds_cache_when_source_external_data_changes(
        self,
        tmp_path: Path,
        qnn_npu_ep_device: WinMLEPDevice,
    ) -> None:
        """Referenced source weight sidecars participate in cache identity."""
        source_path = tmp_path / "external_model.onnx"
        weight = numpy_helper.from_array(np.ones((4, 4), dtype=np.float32), name="weight")
        input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
        output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
        node = helper.make_node("MatMul", ["input", "weight"], ["output"])
        graph = helper.make_graph([node], "external_graph", [input_info], [output_info], [weight])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        save_model(
            model,
            source_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="external_model.onnx.data",
            size_threshold=0,
        )
        data_path = tmp_path / "external_model.onnx.data"
        first_session = WinMLSession(
            onnx_path=source_path,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
        )
        _compile_with_fake_ort(first_session)
        data_path.write_bytes(data_path.read_bytes() + b"changed")
        second_session = WinMLSession(
            onnx_path=source_path,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
        )

        model_compiler = _compile_with_fake_ort(second_session)

        ctx_path = _compiled_generation(second_session, model_compiler)
        assert second_session.running_model_path == ctx_path

    def test_source_external_data_introspection_failure_disables_cache(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Uncertain source identity recompiles instead of falling back to ONNX-only cache keys."""
        first_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
        )
        _compile_with_fake_ort(first_session)
        ctx_path = _cache_path(first_session)
        marker_path = first_session._epcontext_cache_marker_path(ctx_path)
        assert marker_path.is_file()

        def _fail_external_data(_model_path: Path) -> list[str]:
            raise PermissionError("cannot inspect external data")

        monkeypatch.setattr(
            "winml.modelkit.onnx.external_data.get_external_data_files",
            _fail_external_data,
        )
        second_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
        )

        model_compiler = _compile_with_fake_ort(second_session)

        model_compiler.return_value.compile_to_file.assert_called_once()
        assert marker_path.exists()

    def test_source_change_during_compile_retries_with_new_identity(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A generation is published only under the source identity it compiled."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
        )
        compile_calls = 0

        class _SourceChangingCompiler:
            def __init__(self, *_args, **_kwargs):
                pass

            def compile_to_file(self, path: str) -> None:
                nonlocal compile_calls
                compile_calls += 1
                _write_fake_epcontext(session, path)
                if compile_calls == 1:
                    model = load(simple_matmul_onnx)
                    model.producer_name = "changed-during-compile"
                    save(model, simple_matmul_onnx)

        runtime_session = MagicMock()
        runtime_session.get_providers.return_value = ["QNNExecutionProvider"]
        monkeypatch.setattr(
            "winml.modelkit.session.session._build_session_options",
            lambda *_args, **_kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            "winml.modelkit.session.session.ort.ModelCompiler",
            _SourceChangingCompiler,
        )
        monkeypatch.setattr(
            "winml.modelkit.session.session.ort.InferenceSession",
            lambda *_args, **_kwargs: runtime_session,
        )

        session.compile()

        current_identity = session._epcontext_cache_identity()
        cache_path = session._epcontext_cache_path(current_identity)
        assert compile_calls == 2
        assert session.running_model_path == session._epcontext_cached_generation(
            cache_path,
            current_identity,
        )

    def test_final_identity_failure_discards_unpublished_generation(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A retry cannot abandon a markerless generation after preparation."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
        )
        stable_identity = session._epcontext_cache_identity()
        identity_calls = 0
        compiled_paths: list[Path] = []

        def _identity() -> dict[str, object]:
            nonlocal identity_calls
            identity_calls += 1
            if identity_calls == 4:
                raise OSError("final identity unavailable")
            if identity_calls == 5:
                raise ValueError("cache identity unavailable")
            return stable_identity

        def _compile(path: str) -> None:
            compiled_path = Path(path)
            compiled_paths.append(compiled_path)
            _write_fake_epcontext(session, path)

        def _fail_marker(*_args, **_kwargs) -> None:
            raise PermissionError("marker publication failed")

        model_compiler = MagicMock()
        model_compiler.return_value.compile_to_file.side_effect = _compile
        monkeypatch.setattr(session, "_epcontext_cache_identity", _identity)
        monkeypatch.setattr(session, "_write_epcontext_cache_marker", _fail_marker)
        monkeypatch.setattr(
            "winml.modelkit.session.session._build_session_options",
            lambda *_args, **_kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            "winml.modelkit.session.session.ort.ModelCompiler",
            model_compiler,
        )

        prepared = session._compile_epcontext_with_stable_source(
            simple_matmul_onnx.parent / "compile.log"
        )
        try:
            assert len(compiled_paths) == 2
            assert not compiled_paths[0].exists()
            assert not compiled_paths[0].with_name(f"{compiled_paths[0].stem}_qnn.bin").exists()
            assert prepared.path == compiled_paths[1]
            assert prepared.path.exists()
        finally:
            prepared.release()
            for compiled_path in compiled_paths:
                session._discard_epcontext_generation(compiled_path)

    def test_marker_write_failure_keeps_compiled_context_usable(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cache metadata failure does not discard a successful compilation."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
        )
        first_compiler = _compile_with_fake_ort(session)
        first_compiler.return_value.compile_to_file.assert_called_once()
        cached_path = session.running_model_path
        cached_bytes = cached_path.read_bytes()
        marker_path = session._epcontext_cache_marker_path(_cache_path(session))
        marker_path.unlink()
        session.reset()

        def _fail_marker(*_args, **_kwargs):
            raise PermissionError("marker directory is read-only")

        monkeypatch.setattr(session, "_write_epcontext_cache_marker", _fail_marker)
        model_compiler = _compile_with_fake_ort(session)

        model_compiler.return_value.compile_to_file.assert_called_once()
        assert session.running_model_path != cached_path
        assert cached_path.read_bytes() == cached_bytes
        assert session._session is not None
        generation_path = Path(model_compiler.return_value.compile_to_file.call_args.args[0])
        sidecar_path = generation_path.with_name(f"{generation_path.stem}_qnn.bin")
        assert generation_path.is_file()
        assert sidecar_path.is_file()

        session.reset()

        assert not generation_path.exists()
        assert not sidecar_path.exists()

    def test_custom_session_options_factory_disables_cache_reuse(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
    ) -> None:
        """Opaque SessionOptions factory state is never represented as a reusable cache key."""
        first_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
        )
        _compile_with_fake_ort(first_session)
        second_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
            session_options=MagicMock,
        )

        model_compiler = _compile_with_fake_ort(second_session)

        model_compiler.return_value.compile_to_file.assert_called_once()
        assert second_session.running_model_path != first_session.running_model_path
        generation_path = Path(model_compiler.return_value.compile_to_file.call_args.args[0])
        sidecar_path = generation_path.with_name(f"{generation_path.stem}_qnn.bin")
        assert generation_path.is_file()
        assert sidecar_path.is_file()

        second_session.reset()

        assert not generation_path.exists()
        assert not sidecar_path.exists()

    def test_markerless_generation_survives_perf_rebuild_until_reset(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Temporary perf rebuilds can reopen a session-owned markerless model."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(provider="qnn", enable_ep_context=True),
            session_options=MagicMock,
        )
        opened_paths: list[Path] = []

        def _runtime_session(path: str | Path, *_args, **_kwargs) -> MagicMock:
            model_path = Path(path)
            if model_path != simple_matmul_onnx:
                assert model_path.is_file()
            opened_paths.append(model_path)
            runtime_session = MagicMock()
            runtime_session.get_providers.return_value = ["QNNExecutionProvider"]
            return runtime_session

        model_compiler = MagicMock()
        model_compiler.return_value.compile_to_file.side_effect = lambda path: (
            _write_fake_epcontext(session, path)
        )
        monitor = MagicMock()
        monitor.ep_name = "qnn"
        monitor.requires_session_teardown = False
        monitor.get_provider_options.return_value = {"profiling_level": "detailed"}
        monitor.get_session_options.return_value = {}
        monkeypatch.setattr(
            "winml.modelkit.session.session._build_session_options",
            lambda *_args, **_kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            "winml.modelkit.session.session.ort.ModelCompiler",
            model_compiler,
        )
        monkeypatch.setattr(
            "winml.modelkit.session.session.ort.InferenceSession",
            _runtime_session,
        )

        session.compile()
        generation_path = session.running_model_path
        sidecar_path = generation_path.with_name(f"{generation_path.stem}_qnn.bin")

        assert generation_path.is_file()
        with session.perf(monitor=monitor):
            assert session.running_model_path == generation_path
        assert generation_path.is_file()
        assert opened_paths.count(generation_path) == 3

        session.reset()

        assert not generation_path.exists()
        assert not sidecar_path.exists()

    def test_successful_epcontext_cache_prunes_old_identity_artifacts(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A bounded successful cache removes old generations, markers, locks, and sidecars."""
        monkeypatch.setattr(
            "winml.modelkit.session.session._EPCONTEXT_CACHE_MAX_GENERATIONS",
            1,
            raising=False,
        )
        first_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(
                provider="qnn",
                provider_options={"mode": "old"},
                enable_ep_context=True,
            ),
        )
        first_compiler = _compile_with_fake_ort(first_session)
        first_generation = _compiled_generation(first_session, first_compiler)
        first_sidecar = first_generation.with_name(f"{first_generation.stem}_qnn.bin")
        first_cache_path = _cache_path(first_session)
        first_marker = first_session._epcontext_cache_marker_path(first_cache_path)
        first_lock = first_cache_path.with_name(f"{first_cache_path.name}.lock")
        assert first_generation.is_file()
        assert first_sidecar.is_file()
        assert first_marker.is_file()
        first_lock.write_text("stale lock", encoding="utf-8")
        assert first_lock.is_file()
        first_session.reset()

        second_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(
                provider="qnn",
                provider_options={"mode": "new"},
                enable_ep_context=True,
            ),
        )
        second_compiler = _compile_with_fake_ort(second_session)
        second_generation = _compiled_generation(second_session, second_compiler)
        second_cache_path = _cache_path(second_session)
        second_marker = second_session._epcontext_cache_marker_path(second_cache_path)

        assert not first_generation.exists()
        assert not first_sidecar.exists()
        assert not first_marker.exists()
        assert not first_lock.exists()
        assert second_generation.is_file()
        assert second_marker.is_file()

    def test_cache_prune_skips_marker_replaced_before_lease(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stale scan cannot unlink a newly published generation marker."""
        from winml.modelkit.session.session import _EPContextCacheLease

        old_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(
                provider="qnn",
                provider_options={"mode": "old"},
                enable_ep_context=True,
            ),
        )
        old_compiler = _compile_with_fake_ort(old_session)
        old_generation = _compiled_generation(old_session, old_compiler)
        old_identity = old_session._epcontext_cache_identity()
        old_cache_path = old_session._epcontext_cache_path(old_identity)
        old_lock_path = old_cache_path.with_name(f"{old_cache_path.name}.lock")
        current_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(
                provider="qnn",
                provider_options={"mode": "current"},
                enable_ep_context=True,
            ),
        )
        current_compiler = _compile_with_fake_ort(current_session)
        _compiled_generation(current_session, current_compiler)
        current_cache_path = _cache_path(current_session)
        replacement_generation = old_cache_path.with_name(
            f"{old_cache_path.stem}_replacement{old_cache_path.suffix}"
        )
        real_acquire = _EPContextCacheLease.acquire
        replaced = False

        def _acquire(lock_path: Path, *, blocking: bool = True):
            nonlocal replaced
            if (
                not blocking
                and not replaced
                and lock_path.resolve(strict=False) == old_lock_path.resolve(strict=False)
            ):
                replaced = True
                _write_fake_epcontext(old_session, str(replacement_generation))
                old_session._write_epcontext_cache_marker(
                    old_cache_path,
                    replacement_generation,
                    old_identity,
                )
            return real_acquire(lock_path, blocking=blocking)

        monkeypatch.setattr(
            "winml.modelkit.session.session._EPCONTEXT_CACHE_MAX_GENERATIONS",
            1,
        )
        monkeypatch.setattr(
            _EPContextCacheLease,
            "acquire",
            staticmethod(_acquire),
        )

        current_session._prune_epcontext_cache(current_cache_path)

        assert replaced is True
        assert replacement_generation.is_file()
        assert (
            old_session._epcontext_cached_generation(
                old_cache_path,
                old_identity,
            )
            == replacement_generation
        )
        assert not old_generation.exists()

    def test_cache_hit_generation_is_pinned_until_runtime_session_opens(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pruning a different identity cannot delete a cache hit before ORT opens it."""
        from winml.modelkit.session.session import _epcontext_thread_lock

        checked = False
        old_generation: Path | None = None
        old_lock: Path | None = None

        def _session_options(*_args, provider_options, **_kwargs):
            return SimpleNamespace(mode=provider_options["mode"])

        class _ModeCompiler:
            def __init__(self, session_options, *_args, **_kwargs):
                self.mode = session_options.mode

            def compile_to_file(self, path: str) -> None:
                _write_fake_epcontext(first_session, path)

        def _runtime_session(path: str, *_args, **_kwargs):
            nonlocal checked
            model_path = Path(path)
            if old_generation is not None and old_lock is not None and model_path == old_generation:
                checked = True
                assert _epcontext_thread_lock(old_lock).locked()
            runtime_session = MagicMock()
            runtime_session.get_providers.return_value = ["QNNExecutionProvider"]
            return runtime_session

        monkeypatch.setattr(
            "winml.modelkit.session.session._build_session_options",
            _session_options,
        )
        monkeypatch.setattr(
            "winml.modelkit.session.session.ort.ModelCompiler",
            _ModeCompiler,
        )
        monkeypatch.setattr(
            "winml.modelkit.session.session.ort.InferenceSession",
            _runtime_session,
        )
        first_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(
                provider="qnn",
                provider_options={"mode": "old"},
                enable_ep_context=True,
            ),
        )
        first_session.compile()
        old_generation = first_session.running_model_path
        old_cache_path = _cache_path(first_session)
        old_lock = old_cache_path.with_name(f"{old_cache_path.name}.lock")
        first_session.reset()
        hit_session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=qnn_npu_ep_device,
            ep_config=EPConfig(
                provider="qnn",
                provider_options={"mode": "old"},
                enable_ep_context=True,
            ),
        )

        hit_session.compile()

        assert checked is True
        assert hit_session.running_model_path == old_generation

    def test_concurrent_different_identities_use_distinct_artifacts(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Concurrent different identities never write one shared artifact."""
        sessions = [
            WinMLSession(
                onnx_path=simple_matmul_onnx,
                ep_device=qnn_npu_ep_device,
                ep_config=EPConfig(
                    provider="qnn",
                    provider_options={"mode": mode},
                    enable_ep_context=True,
                ),
            )
            for mode in ("first", "second")
        ]
        state_lock = threading.Lock()
        first_entered = threading.Event()
        second_entered = threading.Event()
        state = {"active": 0, "max_active": 0}
        compiled_paths: dict[str, Path] = {}

        def _session_options(*_args, provider_options, **_kwargs):
            return SimpleNamespace(mode=provider_options["mode"])

        class _ConcurrentCompiler:
            def __init__(self, session_options, *_args, **_kwargs):
                self.mode = session_options.mode

            def compile_to_file(self, path: str) -> None:
                with state_lock:
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                if self.mode == "first":
                    first_entered.set()
                    assert second_entered.wait(timeout=1)
                else:
                    assert first_entered.wait(timeout=1)
                    second_entered.set()
                compiled_paths[self.mode] = Path(path)
                session = sessions[0] if self.mode == "first" else sessions[1]
                _write_fake_epcontext(session, path)
                with state_lock:
                    state["active"] -= 1

        inference_session = MagicMock()
        inference_session.get_providers.return_value = ["QNNExecutionProvider"]
        monkeypatch.setattr(
            "winml.modelkit.session.session._build_session_options",
            _session_options,
        )
        monkeypatch.setattr(
            "winml.modelkit.session.session.ort.ModelCompiler",
            _ConcurrentCompiler,
        )
        monkeypatch.setattr(
            "winml.modelkit.session.session.ort.InferenceSession",
            lambda *_args, **_kwargs: inference_session,
        )
        monkeypatch.setattr(
            "winml.modelkit.session.session._suppress_native_output",
            lambda *_args, **_kwargs: nullcontext(),
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(sessions[0].compile)
            assert first_entered.wait(timeout=1)
            second_future = executor.submit(sessions[1].compile)
            first_future.result(timeout=5)
            second_future.result(timeout=5)

        assert state["max_active"] == 2
        assert compiled_paths["first"] != compiled_paths["second"]
        assert sessions[0].running_model_path == compiled_paths["first"]
        assert sessions[1].running_model_path == compiled_paths["second"]

    def test_concurrent_matching_identity_compiles_once(
        self,
        simple_matmul_onnx: Path,
        qnn_npu_ep_device: WinMLEPDevice,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A waiter rechecks the marker and reuses the first identity artifact."""
        sessions = [
            WinMLSession(
                onnx_path=simple_matmul_onnx,
                ep_device=qnn_npu_ep_device,
                ep_config=EPConfig(
                    provider="qnn",
                    provider_options={"mode": "shared"},
                    enable_ep_context=True,
                ),
            )
            for _ in range(2)
        ]
        first_entered = threading.Event()
        release_first = threading.Event()
        compile_calls = 0

        class _SingleCompiler:
            def __init__(self, *_args, **_kwargs):
                pass

            def compile_to_file(self, path: str) -> None:
                nonlocal compile_calls
                compile_calls += 1
                first_entered.set()
                assert release_first.wait(timeout=1)
                _write_fake_epcontext(sessions[0], path)

        inference_session = MagicMock()
        inference_session.get_providers.return_value = ["QNNExecutionProvider"]
        monkeypatch.setattr(
            "winml.modelkit.session.session._build_session_options",
            lambda *_args, **_kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            "winml.modelkit.session.session.ort.ModelCompiler",
            _SingleCompiler,
        )
        monkeypatch.setattr(
            "winml.modelkit.session.session.ort.InferenceSession",
            lambda *_args, **_kwargs: inference_session,
        )
        monkeypatch.setattr(
            "winml.modelkit.session.session._suppress_native_output",
            lambda *_args, **_kwargs: nullcontext(),
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(sessions[0].compile)
            assert first_entered.wait(timeout=1)
            second_future = executor.submit(sessions[1].compile)
            release_first.set()
            first_future.result(timeout=5)
            second_future.result(timeout=5)

        assert compile_calls == 1
        assert sessions[0].running_model_path == sessions[1].running_model_path

    def test_runtime_compile_bypasses_model_compiler(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: EPDeviceTarget,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Lazy runtime sessions construct ORT directly without AOT artifacts."""
        _stub_registry(monkeypatch, cpu_ep_device)
        with (
            patch("winml.modelkit.session.session.ort.InferenceSession") as inference_session,
            patch("winml.modelkit.session.session.ort.ModelCompiler") as model_compiler,
        ):
            session = WinMLSession(onnx_path=simple_matmul_onnx, device="cpu")
            assert session._session is None

            session.compile()

        inference_session.assert_called_once()
        model_compiler.assert_not_called()
        assert not (simple_matmul_onnx.parent / "compile.log").exists()

    def test_runtime_compile_after_reset_reuses_constructor_monitor_baseline(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: EPDeviceTarget,
    ):
        """reset()+compile() rebuilds with constructor baseline provider/session options."""
        from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor

        baseline_session_entries = {"baseline.session.entry": "enabled"}

        class _BaselineMonitor(WinMLEPMonitor):
            @classmethod
            def is_available(cls):
                return True

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def to_dict(self):
                return {"ep": "baseline"}

            def get_provider_options(self):
                return {"baseline_provider_key": "baseline"}

            def get_session_options(self):
                return dict(baseline_session_entries)

        initial_so = MagicMock()
        rebuilt_so = MagicMock()
        factory = MagicMock(side_effect=[initial_so, rebuilt_so])

        with patch(
            "winml.modelkit.session.session.ort.InferenceSession",
            side_effect=[MagicMock(), MagicMock()],
        ):
            session = WinMLSession(
                onnx_path=simple_matmul_onnx,
                ep_device=cpu_ep_device,
                ep_monitor=_BaselineMonitor(),
                session_options=factory,
            )
            baseline_provider_options = dict(session._provider_options)

            session.reset()
            session.compile()

        rebuilt_so.add_session_config_entry.assert_called_once_with(
            "baseline.session.entry",
            "enabled",
        )
        assert rebuilt_so.add_provider_for_devices.call_args.args[1] == baseline_provider_options
        assert session._active_session_option_entries == baseline_session_entries

    def test_run_uses_epcontext_after_compile(self, cpu_winml_session: WinMLSession):
        """Test that run() works after compile() was called."""
        session = cpu_winml_session

        # compile() is a no-op when _session is already set
        session.compile()

        # Run should succeed
        sample_input = {"A": np.random.randn(1, 4).astype(np.float32)}
        session.run(sample_input)

        # Session should be compiled
        assert session.is_compiled
        assert session.state == SessionState.COMPILED


class TestWinMLSessionProviders:
    """Test that session providers are valid after initialization."""

    def test_providers_are_valid_and_include_fallback(
        self, cpu_winml_session: WinMLSession, sample_input: dict
    ):
        """
        Test that session providers are valid and include CPUExecutionProvider.

        The session is bound to CPUExecutionProvider via an explicit EPDeviceTarget.
        """
        session = cpu_winml_session

        # Run inference to confirm the session is functional
        session.run(sample_input)

        # Get actual providers used by session
        actual_providers = session._session.get_providers()

        # Must have at least one provider
        assert len(actual_providers) > 0, "Session must have at least one provider"

        # CPUExecutionProvider should always be present
        assert "CPUExecutionProvider" in actual_providers, (
            f"CPUExecutionProvider not in providers: {actual_providers}"
        )

        print(f"Active providers: {actual_providers}")

    def test_cpu_provider_always_available(
        self, cpu_winml_session: WinMLSession, sample_input: dict
    ):
        """Test that CPUExecutionProvider is available after CPU EPDeviceTarget init.

        Also pins the `is_compiled` -> run() -> `is_compiled` behavior and the
        output dtype check (assertions ported from deleted device='auto' tests;
        no other test pins these on device='cpu').
        """
        session = cpu_winml_session
        outputs = session.run(sample_input)

        assert session.is_compiled
        providers = session._session.get_providers()
        assert "CPUExecutionProvider" in providers
        assert outputs["C"].dtype == np.float32


class TestWinMLSessionInference:
    """Test WinMLSession inference execution."""

    def test_basic_inference(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test basic inference with MatMul model."""
        session = cpu_winml_session

        # Run inference
        outputs = session.run(sample_input)

        # Check output
        assert "C" in outputs
        assert outputs["C"].shape == (1, 4)
        assert outputs["C"].dtype == np.float32

    def test_inference_already_compiled_on_init(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that WinMLSession is compiled immediately after __init__."""
        session = cpu_winml_session

        # __init__ creates _session eagerly — is_compiled is True immediately
        assert session.is_compiled

        # Run should succeed
        outputs = session.run(sample_input)
        assert "C" in outputs

    def test_inference_with_torch_tensor(
        self,
        cpu_winml_session: WinMLSession,
    ):
        """Test inference with torch.Tensor input (converted to numpy)."""
        pytest.importorskip("torch")
        import torch

        session = cpu_winml_session

        # Create torch tensor input
        torch_input = {"A": torch.randn(1, 4)}

        # Run inference (should convert to numpy internally)
        outputs = session.run(torch_input)

        assert "C" in outputs
        assert outputs["C"].shape == (1, 4)

    def test_inference_ignores_inputs_not_declared_by_model(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        inputs = {
            **sample_input,
            "unexpected": np.zeros((1, 4), dtype=np.int64),
        }

        outputs = cpu_winml_session.run(inputs)

        assert "C" in outputs

    def test_inference_empty_input_raises(self, cpu_winml_session: WinMLSession):
        """Test that empty input raises ValueError."""
        session = cpu_winml_session

        with pytest.raises(ValueError, match="inputs cannot be empty"):
            session.run({})


class TestWinMLSessionStateManagement:
    """Test WinMLSession state machine."""

    def test_state_transitions(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test state transitions: COMPILED -> INFERRING -> COMPILED.

        __init__ creates the session eagerly so the initial state is COMPILED.
        """
        session = cpu_winml_session

        # __init__ creates the ORT session eagerly — state starts at INITIALIZED
        # but _session is already populated.
        assert session.state == SessionState.INITIALIZED

        # After run
        session.run(sample_input)
        assert session.state == SessionState.COMPILED

        # Run again (should return to COMPILED)
        session.run(sample_input)
        assert session.state == SessionState.COMPILED

    def test_reset_returns_to_initialized(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that reset() returns session to INITIALIZED state."""
        session = cpu_winml_session

        # Run to transition to COMPILED
        session.run(sample_input)
        assert session.is_compiled
        assert session._running_model_path is not None

        session.reset()
        assert session.state == SessionState.INITIALIZED
        assert not session.is_compiled
        assert session._running_model_path is None


class TestWinMLSessionMetadata:
    """Test WinMLSession metadata methods."""

    def test_io_config_before_session_init(
        self,
        cpu_winml_session: WinMLSession,
    ):
        """Test that io_config is available and reflects the ONNX model."""
        session = cpu_winml_session

        # io_config reads the ONNX file directly
        io_cfg = session.io_config

        assert io_cfg["input_names"] == ["A"]
        assert io_cfg["output_names"] == ["C"]
        assert io_cfg["input_shapes"] == [[1, 4]]


class TestWinMLSessionPrecisionDetection:
    """Test `_get_precision` estimation across the detection ladder."""

    @staticmethod
    def _save(model, path: Path) -> Path:
        from onnx import save

        save(model, str(path))
        return path

    def test_precision_fp32_from_initializers(
        self, simple_matmul_onnx: Path, cpu_ep_device: WinMLEPDevice
    ):
        """Float initializers (fp32) → 'fp32'."""
        session = WinMLSession(onnx_path=simple_matmul_onnx, ep_device=cpu_ep_device)
        assert session.io_config["precision"] == "fp32"

    def test_precision_fp16_from_initializers(self, tmp_path: Path, cpu_ep_device: WinMLEPDevice):
        """Float initializers (fp16) → 'fp16'."""
        import numpy as np
        from onnx import TensorProto, helper

        a = helper.make_tensor_value_info("A", TensorProto.FLOAT16, [1, 4])
        c = helper.make_tensor_value_info("C", TensorProto.FLOAT16, [1, 4])
        b_vals = np.random.randn(4, 4).astype(np.float16)
        b = helper.make_tensor("B", TensorProto.FLOAT16, [4, 4], b_vals.tobytes(), raw=True)
        node = helper.make_node("MatMul", ["A", "B"], ["C"])
        graph = helper.make_graph([node], "fp16", [a], [c], [b])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        path = self._save(model, tmp_path / "fp16.onnx")

        session = WinMLSession(onnx_path=path, ep_device=cpu_ep_device)
        assert session.io_config["precision"] == "fp16"

    def test_precision_int8_from_qdq(self, tmp_path: Path, cpu_ep_device: WinMLEPDevice):
        """QDQ pair with int8 zero_point on a weight initializer → 'int8'."""
        import numpy as np
        from onnx import TensorProto, helper

        a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
        c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])

        w_q = helper.make_tensor(
            "W_q",
            TensorProto.INT8,
            [4, 4],
            np.zeros((4, 4), dtype=np.int8).tobytes(),
            raw=True,
        )
        w_scale = helper.make_tensor("W_scale", TensorProto.FLOAT, [], [0.1])
        w_zp = helper.make_tensor(
            "W_zp", TensorProto.INT8, [], np.array([0], dtype=np.int8).tobytes(), raw=True
        )

        dq = helper.make_node("DequantizeLinear", ["W_q", "W_scale", "W_zp"], ["W"], name="dq")
        matmul = helper.make_node("MatMul", ["A", "W"], ["C"], name="mm")

        graph = helper.make_graph([dq, matmul], "qdq_int8", [a], [c], [w_q, w_scale, w_zp])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        path = self._save(model, tmp_path / "qdq_int8.onnx")

        session = WinMLSession(onnx_path=path, ep_device=cpu_ep_device)
        assert session.io_config["precision"] == "w8a8"

    def test_precision_w8a16_mixed_qdq(self, tmp_path: Path, cpu_ep_device: WinMLEPDevice):
        """Activation quantized to uint16 + weight to int8 → 'w8a16'."""
        import numpy as np
        from onnx import TensorProto, helper

        a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
        c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])

        # Activation Q→DQ with uint16 zero_point (dynamic input → activation side)
        a_scale = helper.make_tensor("A_scale", TensorProto.FLOAT, [], [0.05])
        a_zp = helper.make_tensor(
            "A_zp",
            TensorProto.UINT16,
            [],
            np.array([0], dtype=np.uint16).tobytes(),
            raw=True,
        )
        q_act = helper.make_node("QuantizeLinear", ["A", "A_scale", "A_zp"], ["A_q"], name="q_act")
        dq_act = helper.make_node(
            "DequantizeLinear", ["A_q", "A_scale", "A_zp"], ["A_d"], name="dq_act"
        )

        # Weight DQ with int8 zero_point (initializer → weight side)
        w_q = helper.make_tensor(
            "W_q",
            TensorProto.INT8,
            [4, 4],
            np.zeros((4, 4), dtype=np.int8).tobytes(),
            raw=True,
        )
        w_scale = helper.make_tensor("W_scale", TensorProto.FLOAT, [], [0.1])
        w_zp = helper.make_tensor(
            "W_zp", TensorProto.INT8, [], np.array([0], dtype=np.int8).tobytes(), raw=True
        )
        dq_w = helper.make_node("DequantizeLinear", ["W_q", "W_scale", "W_zp"], ["W"], name="dq_w")

        matmul = helper.make_node("MatMul", ["A_d", "W"], ["C"], name="mm")

        graph = helper.make_graph(
            [q_act, dq_act, dq_w, matmul],
            "qdq_w8a16",
            [a],
            [c],
            [a_scale, a_zp, w_q, w_scale, w_zp],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        path = self._save(model, tmp_path / "qdq_w8a16.onnx")

        # Precision detection is a static read of the QDQ graph; the uint16
        # activation zero-point is a valid w8a16 signal but ORT's CPU EP rejects
        # it at compile time, so mock the InferenceSession — we only assert on
        # the statically-derived io_config precision, not on a runnable session.
        with patch("winml.modelkit.session.session.ort.InferenceSession"):
            session = WinMLSession(onnx_path=path, ep_device=cpu_ep_device)
        assert session.io_config["precision"] == "w8a16"

    def test_precision_int8_ignores_int32_bias_zp(
        self, tmp_path: Path, cpu_ep_device: WinMLEPDevice
    ):
        """INT32 bias DQ on the weight side must not poison the label.

        Mirrors the NPU-quantized ResNet-50 case: every Conv has an
        INT8-weight DQ alongside an INT32-bias DQ. The bias is a quant
        accumulator, not a weight, so it must be excluded from weight-side
        bit-width counting; otherwise the result becomes 'w32a8'.
        """
        import numpy as np
        from onnx import TensorProto, helper

        a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
        c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])

        # Activation Q→DQ with UINT8 zero_point
        a_scale = helper.make_tensor("A_scale", TensorProto.FLOAT, [], [0.05])
        a_zp = helper.make_tensor(
            "A_zp", TensorProto.UINT8, [], np.array([0], dtype=np.uint8).tobytes(), raw=True
        )
        q_act = helper.make_node("QuantizeLinear", ["A", "A_scale", "A_zp"], ["A_q"], name="q_act")
        dq_act = helper.make_node(
            "DequantizeLinear", ["A_q", "A_scale", "A_zp"], ["A_d"], name="dq_act"
        )

        # Weight DQ with INT8 zero_point (initializer → weight side)
        w_q = helper.make_tensor(
            "W_q",
            TensorProto.INT8,
            [4, 4],
            np.zeros((4, 4), dtype=np.int8).tobytes(),
            raw=True,
        )
        w_scale = helper.make_tensor("W_scale", TensorProto.FLOAT, [], [0.1])
        w_zp = helper.make_tensor(
            "W_zp", TensorProto.INT8, [], np.array([0], dtype=np.int8).tobytes(), raw=True
        )
        dq_w = helper.make_node("DequantizeLinear", ["W_q", "W_scale", "W_zp"], ["W"], name="dq_w")

        # Bias DQ with INT32 zero_point (initializer → would be classified
        # weight-side; this is the node that previously poisoned the label).
        b_q = helper.make_tensor(
            "B_q", TensorProto.INT32, [4], np.zeros(4, dtype=np.int32).tobytes(), raw=True
        )
        b_scale = helper.make_tensor("B_scale", TensorProto.FLOAT, [], [0.005])
        b_zp = helper.make_tensor(
            "B_zp", TensorProto.INT32, [], np.array([0], dtype=np.int32).tobytes(), raw=True
        )
        dq_b = helper.make_node("DequantizeLinear", ["B_q", "B_scale", "B_zp"], ["B"], name="dq_b")

        matmul = helper.make_node("MatMul", ["A_d", "W"], ["MM"], name="mm")
        add = helper.make_node("Add", ["MM", "B"], ["C"], name="add_bias")

        graph = helper.make_graph(
            [q_act, dq_act, dq_w, dq_b, matmul, add],
            "qdq_with_int32_bias",
            [a],
            [c],
            [a_scale, a_zp, w_q, w_scale, w_zp, b_q, b_scale, b_zp],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        path = self._save(model, tmp_path / "qdq_int32_bias.onnx")

        session = WinMLSession(onnx_path=path, ep_device=cpu_ep_device)
        assert session.io_config["precision"] == "w8a8"

    def test_precision_matmulnbits_w4a16(self, tmp_path: Path, cpu_ep_device: WinMLEPDevice):
        """MatMulNBits with bits=4 + fp16 initializers → 'w4a16'."""
        import numpy as np
        from onnx import TensorProto, helper

        a = helper.make_tensor_value_info("A", TensorProto.FLOAT16, [1, 32])
        c = helper.make_tensor_value_info("C", TensorProto.FLOAT16, [1, 16])

        # MatMulNBits packed-weight + scales (dummy shapes — schema doesn't validate)
        w_packed = helper.make_tensor(
            "W",
            TensorProto.UINT8,
            [16, 1, 16],
            np.zeros((16, 1, 16), dtype=np.uint8).tobytes(),
            raw=True,
        )
        scales = helper.make_tensor(
            "scales",
            TensorProto.FLOAT16,
            [16],
            np.ones(16, dtype=np.float16).tobytes(),
            raw=True,
        )

        node = helper.make_node(
            "MatMulNBits",
            ["A", "W", "scales"],
            ["C"],
            domain="com.microsoft",
            K=32,
            N=16,
            bits=4,
            block_size=32,
        )

        graph = helper.make_graph([node], "mmnbits_w4", [a], [c], [w_packed, scales])
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )
        model.ir_version = 7
        path = self._save(model, tmp_path / "mmnbits_w4.onnx")

        session = WinMLSession(onnx_path=path, ep_device=cpu_ep_device)
        assert session.io_config["precision"] == "w4a16"

    def test_precision_no_signal_returns_none(self, tmp_path: Path, cpu_ep_device: WinMLEPDevice):
        """No QDQ ops, no MatMulNBits, no float initializers → None."""
        from onnx import TensorProto, helper

        a = helper.make_tensor_value_info("A", TensorProto.INT64, [1, 4])
        c = helper.make_tensor_value_info("C", TensorProto.INT64, [1, 4])

        identity = helper.make_node("Identity", ["A"], ["C"])
        graph = helper.make_graph([identity], "no_signal", [a], [c])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        path = self._save(model, tmp_path / "no_signal.onnx")

        session = WinMLSession(onnx_path=path, ep_device=cpu_ep_device)
        assert session.io_config["precision"] is None


@pytest.mark.skip(reason="Re-batching not yet implemented")
class TestWinMLSessionReBatching:
    """Test re-batching for static batch size models."""

    def test_rebatch_splits_large_batch(self, static_batch1_onnx: Path):
        """Test that batch > model_batch triggers re-batching (lines 343-363)."""
        session = WinMLSession(
            onnx_path=static_batch1_onnx,
            device="cpu",
        )

        # Model expects batch=1, we send batch=3
        large_input = {"A": np.random.randn(3, 4).astype(np.float32)}
        outputs = session.run(large_input)

        # Output should have batch=3 (concatenated from 3 runs of batch=1)
        assert "C" in outputs
        assert outputs["C"].shape == (3, 4)

    def test_rebatch_with_batch2_model(self, static_batch2_onnx: Path):
        """Test re-batching with batch=2 model and batch=4 input (exact multiple)."""
        session = WinMLSession(
            onnx_path=static_batch2_onnx,
            device="cpu",
        )

        # Model expects batch=2, we send batch=4 (exact multiple)
        # Should split into: [2, 2] -> 2 runs
        large_input = {"A": np.random.randn(4, 4).astype(np.float32)}
        outputs = session.run(large_input)

        # Output should have batch=4 (concatenated)
        assert "C" in outputs
        assert outputs["C"].shape == (4, 4)

    def test_rebatch_preserves_values(self, static_batch1_onnx: Path):
        """Test that re-batched outputs are numerically correct."""
        session = WinMLSession(
            onnx_path=static_batch1_onnx,
            device="cpu",
        )

        # Create known input
        np.random.seed(123)
        input_data = np.random.randn(3, 4).astype(np.float32)

        # Run with re-batching (batch=3 on batch=1 model)
        outputs = session.run({"A": input_data})

        # Run each row individually and compare
        for i in range(3):
            session.reset()
            single_output = session.run({"A": input_data[i : i + 1]})
            np.testing.assert_allclose(
                outputs["C"][i : i + 1],
                single_output["C"],
                rtol=1e-5,
                err_msg=f"Re-batched output[{i}] doesn't match single inference",
            )

    def test_no_rebatch_when_batch_fits(self, static_batch2_onnx: Path):
        """Test that batch <= model_batch runs directly without splitting."""
        session = WinMLSession(
            onnx_path=static_batch2_onnx,
            device="cpu",
        )

        # Model expects batch=2, we send batch=2 (exact fit)
        exact_input = {"A": np.random.randn(2, 4).astype(np.float32)}
        outputs = session.run(exact_input)

        assert "C" in outputs
        assert outputs["C"].shape == (2, 4)

    def test_batch_smaller_than_model_fails(self, static_batch2_onnx: Path):
        """Test that batch < model_batch fails with static batch model.

        ORT with static batch models requires exact batch size match.
        Sending batch=1 to a batch=2 model raises INVALID_ARGUMENT.
        """
        from winml.modelkit.session.session import InferenceError

        session = WinMLSession(
            onnx_path=static_batch2_onnx,
            device="cpu",
        )

        # Model expects batch=2, we send batch=1 (smaller) - ORT rejects this
        small_input = {"A": np.random.randn(1, 4).astype(np.float32)}

        with pytest.raises(InferenceError, match="INVALID_ARGUMENT"):
            session.run(small_input)


class TestWinMLSessionErrorState:
    """Test error state handling."""

    def test_run_in_error_state_raises(self, cpu_winml_session: WinMLSession):
        """Test that run() raises InferenceError when session is in error state."""
        from winml.modelkit.session.session import InferenceError

        session = cpu_winml_session

        # Trigger first run
        sample = {"A": np.random.randn(1, 4).astype(np.float32)}
        session.run(sample)

        # Manually set error state
        session._state = SessionState.ERROR
        session._last_error = RuntimeError("Test error")

        # Run should raise InferenceError
        with pytest.raises(InferenceError, match="Session in error state"):
            session.run(sample)

    def test_reset_clears_error_state(self, cpu_winml_session: WinMLSession):
        """Test that reset() clears error state and allows re-run."""
        session = cpu_winml_session

        # Run, then set error state
        sample = {"A": np.random.randn(1, 4).astype(np.float32)}
        session.run(sample)
        session._state = SessionState.ERROR
        session._last_error = RuntimeError("Test error")

        # Reset should clear error
        session.reset()
        assert session.state == SessionState.INITIALIZED
        assert session._last_error is None

        # Should be able to run again
        outputs = session.run(sample)
        assert "C" in outputs


class TestWinMLSessionExplicitProviders:
    """Test EPConfig provider_options passthrough with EPDeviceTarget-based init."""

    def test_explicit_cpu_provider(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: EPDeviceTarget,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that ep_config is accepted and CPU provider is active."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=cpu_ep_device,
            ep_config=EPConfig(provider="cpu", provider_options={}),
        )

        outputs = session.run(sample_input)

        providers = session._session.get_providers()
        assert "CPUExecutionProvider" in providers
        assert "C" in outputs

    def test_explicit_provider_with_options(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: EPDeviceTarget,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that ep_config.provider_options is accepted without error."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=cpu_ep_device,
            ep_config=EPConfig(provider="cpu", provider_options={}),
        )

        outputs = session.run(sample_input)

        providers = session._session.get_providers()
        assert "CPUExecutionProvider" in providers
        assert "C" in outputs

    def test_ep_config_provider_options_forwarded(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: EPDeviceTarget,
        sample_input: dict[str, np.ndarray],
    ):
        """Verify ep_config.provider_options is forwarded to add_provider_for_devices."""
        import onnxruntime as ort

        options = {"arbitrary_key": "arbitrary_value"}
        captured: list[dict[str, str]] = []
        real_method = ort.SessionOptions.add_provider_for_devices

        def spy(self_sess, ep_devices, provider_opts):
            captured.append(dict(provider_opts))
            return real_method(self_sess, ep_devices, provider_opts)

        with patch.object(ort.SessionOptions, "add_provider_for_devices", spy):
            session = WinMLSession(
                onnx_path=simple_matmul_onnx,
                ep_device=cpu_ep_device,
                ep_config=EPConfig(provider="cpu", provider_options=options),
            )
            outputs = session.run(sample_input)

        assert options in captured, f"provider_options not forwarded; got calls with: {captured}"
        assert "C" in outputs

    def test_runtime_provider_options_forwarded(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: EPDeviceTarget,
        sample_input: dict[str, np.ndarray],
    ):
        """Runtime ``provider_options`` kwarg is forwarded to add_provider_for_devices."""
        import onnxruntime as ort

        options = {"runtime_only_key": "runtime_only_value"}
        captured: list[dict[str, str]] = []
        real_method = ort.SessionOptions.add_provider_for_devices

        def spy(self_sess, ep_devices, provider_opts):
            captured.append(dict(provider_opts))
            return real_method(self_sess, ep_devices, provider_opts)

        with patch.object(ort.SessionOptions, "add_provider_for_devices", spy):
            session = WinMLSession(
                onnx_path=simple_matmul_onnx,
                ep_device=cpu_ep_device,
                provider_options=options,
            )
            outputs = session.run(sample_input)

        assert options in captured, f"provider_options not forwarded; got calls with: {captured}"
        assert "C" in outputs

    def test_runtime_provider_options_override_ep_config(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: EPDeviceTarget,
    ):
        """Runtime ``provider_options`` merge on top of and override ep_config options."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=cpu_ep_device,
            ep_config=EPConfig(
                provider="cpu",
                provider_options={"shared": "from_build", "build_only": "x"},
            ),
            provider_options={"shared": "from_runtime", "runtime_only": "y"},
        )

        # Runtime value wins for the shared key; both source-specific keys survive.
        assert session._provider_options == {
            "shared": "from_runtime",
            "build_only": "x",
            "runtime_only": "y",
        }

    def test_runtime_provider_options_do_not_mutate_ep_config(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: EPDeviceTarget,
    ):
        """Session-local option overrides leave the caller's config reusable."""
        ep_config = EPConfig(
            provider="cpu",
            provider_options={"shared": "from_build", "build_only": "x"},
        )

        WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=cpu_ep_device,
            ep_config=ep_config,
            provider_options={"shared": "from_runtime", "runtime_only": "y"},
        )

        assert ep_config.provider_options == {
            "shared": "from_build",
            "build_only": "x",
        }

    def test_explicit_unavailable_target_propagates_structured_error(
        self,
        simple_matmul_onnx: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """An unavailable explicit EP/device request never falls back silently."""
        from winml.modelkit.session import WinMLEPNotDiscovered

        registry = _stub_registry(monkeypatch, None)
        registry.auto_device.side_effect = WinMLEPNotDiscovered("QNN unavailable")

        with pytest.raises(WinMLEPNotDiscovered, match="QNN unavailable"):
            WinMLSession(onnx_path=simple_matmul_onnx, device="gpu", ep="qnn")


class TestWinMLSessionPerfTracking:
    """Test WinMLSession performance tracking with context manager."""

    def test_perf_disabled_by_default(self, cpu_winml_session: WinMLSession):
        """Test that performance tracking is disabled by default."""
        session = cpu_winml_session
        assert session.perf_stats is None

    def test_perf_context_manager_returns_stats(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Legacy direct stats access delegates through the PerfContext."""
        from winml.modelkit.session import PerfStats

        session = cpu_winml_session

        with session.perf() as stats:
            assert isinstance(stats, PerfContext)
            assert isinstance(stats.stats, PerfStats)
            assert stats.count == 0
            assert stats.samples_ms == []
            assert stats.monitor is not None

    def test_perf_records_samples(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that inference runs are recorded within context."""
        session = cpu_winml_session

        with session.perf() as ctx:
            for _ in range(5):
                session.run(sample_input)

            stats = ctx.stats
            assert stats.count == 5
            assert len(stats.samples_ms) == 5
            assert all(t > 0 for t in stats.samples_ms)

    def test_perf_stats_computed_correctly(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that computed stats are correct."""
        session = cpu_winml_session

        with session.perf() as ctx:
            for _ in range(10):
                session.run(sample_input)

        stats = ctx.stats
        assert stats.count == 10
        assert stats.total_ms > 0
        assert stats.mean_ms > 0
        assert stats.min_ms > 0
        assert stats.max_ms >= stats.min_ms
        assert stats.p50_ms > 0
        assert stats.p90_ms >= stats.p50_ms
        assert stats.p99_ms >= stats.p90_ms

    def test_perf_warmup_excludes_samples(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test warmup parameter excludes first N samples."""
        session = cpu_winml_session

        with session.perf(warmup=3) as ctx:
            for _ in range(10):
                session.run(sample_input)

        stats = ctx.stats
        # 10 total, 3 warmup = 7 effective
        assert stats.total_count == 10
        assert stats.count == 7
        assert len(stats.samples_ms) == 7
        assert len(stats.all_samples_ms) == 10

    def test_perf_disabled_after_context(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that perf tracking is disabled after context exits."""
        session = cpu_winml_session

        with session.perf() as ctx:
            session.run(sample_input)
            stats = ctx.stats
            assert stats.count == 1

        # After context, perf_stats should be None
        assert session.perf_stats is None

        # Running outside context should not record
        session.run(sample_input)
        # stats object still has data from context
        assert stats.count == 1

    def test_perf_output_not_affected(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that perf tracking doesn't affect inference output."""
        session = cpu_winml_session

        # Run without tracking
        output_no_perf = session.run(sample_input)

        # Run with tracking
        with session.perf():
            output_with_perf = session.run(sample_input)

        # Outputs should be identical
        np.testing.assert_array_equal(output_no_perf["C"], output_with_perf["C"])

    def test_perf_stats_accessible_after_context(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that stats object remains accessible after context."""
        session = cpu_winml_session

        with session.perf(warmup=2) as ctx:
            for _ in range(5):
                session.run(sample_input)

        stats = ctx.stats
        # Stats still accessible after context
        assert stats.count == 3  # 5 - 2 warmup
        assert stats.mean_ms > 0
        assert stats.p99_ms > 0


# =============================================================================
# Task 7: EPDeviceTarget-based constructor (hard break)
# =============================================================================


def test_winml_session_accepts_ep_device(tmp_path, qnn_npu_ep_device, fake_ort_npu) -> None:
    """WinMLSession compiles with an explicit WinMLEPDevice."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")  # minimal placeholder; ORT is mocked
    with (
        patch("winml.modelkit.session.session.ort.InferenceSession") as mock_sess,
        patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()),
    ):
        sess = WinMLSession(onnx_path, ep_device=qnn_npu_ep_device)
    mock_sess.assert_called_once()
    assert sess._ep_device is qnn_npu_ep_device


def test_winml_session_rejects_legacy_ep_kwarg(tmp_path, qnn_npu_ep_device) -> None:
    """Legacy ep="qnn" kwarg now raises TypeError."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    with pytest.raises(TypeError):
        WinMLSession(onnx_path, ep="qnn")  # type: ignore[call-arg]


def test_winml_session_accepts_device_kwarg_lazily(tmp_path, cpu_ep_device, monkeypatch) -> None:
    """The public device shortcut resolves a registry device without creating ORT."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    _stub_registry(monkeypatch, cpu_ep_device)

    session = WinMLSession(onnx_path, device="cpu")

    assert session.device == "cpu"
    assert session._session is None


def test_winml_session_path_only_defaults_to_auto_lazily(
    tmp_path, cpu_ep_device, monkeypatch
) -> None:
    """The legacy path-only constructor resolves the automatic device policy."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    registry = _stub_registry(monkeypatch, cpu_ep_device)

    session = WinMLSession(onnx_path)

    assert session._session is None
    target = registry.auto_device.call_args.args[0]
    assert target.device.lower() == "cpu"


def test_winml_session_positional_device_uses_legacy_policy(
    tmp_path, cpu_ep_device, monkeypatch
) -> None:
    """A positional device string retains the legacy constructor form."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    registry = _stub_registry(monkeypatch, cpu_ep_device)

    session = WinMLSession(onnx_path, "cpu")

    assert session._session is None
    target = registry.auto_device.call_args.args[0]
    assert target.device.lower() == "cpu"


def test_winml_session_positional_resolved_target_retains_current_api(
    tmp_path, qnn_npu_ep_device, fake_ort_npu
) -> None:
    """A resolved target in the second position is not treated as a device string."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    with (
        patch("winml.modelkit.session.session.ort.InferenceSession") as inference_session,
        patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()),
    ):
        session = WinMLSession(onnx_path, qnn_npu_ep_device)

    inference_session.assert_called_once()
    assert session._ep_device is qnn_npu_ep_device


def test_winml_session_rejects_conflicting_positional_and_keyword_devices(
    tmp_path, cpu_ep_device, monkeypatch
) -> None:
    """The legacy positional policy cannot conflict with the explicit keyword."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    _stub_registry(monkeypatch, cpu_ep_device)

    with pytest.raises(TypeError, match="device"):
        WinMLSession(onnx_path, "cpu", device="gpu")


# =============================================================================
# Task 8: perf() validation + save/restore
# =============================================================================


def test_perf_validates_monitor_ep_name_match(tmp_path, qnn_npu_ep_device, fake_ort_npu) -> None:
    """Monitor for QNN against an OpenVINO WinMLEPDevice -> WinMLEPMonitorMismatch."""
    from .conftest import make_stub_winml_ep_device

    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    fake_ov = MagicMock()
    fake_ov.ep_name = "OpenVINOExecutionProvider"
    fake_ov.device.type.name = "NPU"
    fake_ov.device.vendor_id = 0x8086
    fake_ov.device.device_id = 0x0BD0
    openvino_ep_device = make_stub_winml_ep_device(fake_ov, "OpenVINOExecutionProvider")
    qnn_monitor = MagicMock()
    qnn_monitor.ep_name = "qnn"
    qnn_monitor.get_provider_options.return_value = {}
    qnn_monitor.get_session_options.return_value = {}
    with (
        patch("winml.modelkit.session.session.ort.InferenceSession"),
        patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()),
    ):
        sess = WinMLSession(onnx_path, ep_device=openvino_ep_device)
        with pytest.raises(WinMLEPMonitorMismatch), sess.perf(monitor=qnn_monitor):
            pass


def test_perf_preserves_save_restore(tmp_path, qnn_npu_ep_device, fake_ort_npu) -> None:
    """Mid-perf raise must restore _provider_options snapshot."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    bad_monitor = MagicMock()
    bad_monitor.ep_name = "qnn"
    bad_monitor.get_provider_options.return_value = {"oops": "x"}
    bad_monitor.get_session_options.return_value = {}
    bad_monitor.__enter__.side_effect = RuntimeError("boom")
    with (
        patch("winml.modelkit.session.session.ort.InferenceSession"),
        patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()),
    ):
        sess = WinMLSession(onnx_path, ep_device=qnn_npu_ep_device)
        snapshot = dict(sess._provider_options)
        with pytest.raises(RuntimeError), sess.perf(monitor=bad_monitor):
            pass
        assert sess._provider_options == snapshot
        assert sess._ep == "QNNExecutionProvider"
        assert sess._active_session_option_entries == {}  # back to empty (pre-perf state)


def test_perf_releases_native_session_before_each_replacement(
    tmp_path, qnn_npu_ep_device, fake_ort_npu
) -> None:
    """A monitored rebuild never overlaps baseline and replacement native sessions."""
    import gc
    import weakref

    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    prior_sessions: list[weakref.ReferenceType[object]] = []

    class _NativeSession:
        def get_providers(self) -> list[str]:
            return ["QNNExecutionProvider"]

    def make_native_session(*_args, **_kwargs):
        gc.collect()
        assert not prior_sessions or prior_sessions[-1]() is None
        native_session = _NativeSession()
        prior_sessions.append(weakref.ref(native_session))
        return native_session

    monitor = MagicMock()
    monitor.ep_name = "qnn"
    monitor.requires_session_teardown = False
    monitor.get_provider_options.return_value = {"profiling_level": "detailed"}
    monitor.get_session_options.return_value = {}

    with (
        patch(
            "winml.modelkit.session.session.ort.InferenceSession",
            side_effect=make_native_session,
        ),
        patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()),
    ):
        session = WinMLSession(onnx_path, ep_device=qnn_npu_ep_device)
        with session.perf(monitor=monitor):
            pass

    assert session._session is not None


def test_perf_keeps_active_model_path_while_monitored(
    tmp_path, qnn_npu_ep_device, fake_ort_npu
) -> None:
    """The monitored session reports the active EPContext path throughout perf()."""
    onnx_path = tmp_path / "source.onnx"
    active_model_path = tmp_path / "active_ctx.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    active_model_path.write_bytes(b"\x08\x01")
    monitor = MagicMock()
    monitor.ep_name = "qnn"
    monitor.requires_session_teardown = False
    monitor.get_provider_options.return_value = {"profiling_level": "detailed"}
    monitor.get_session_options.return_value = {}

    with (
        patch("winml.modelkit.session.session.ort.InferenceSession"),
        patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()),
    ):
        session = WinMLSession(onnx_path, ep_device=qnn_npu_ep_device)
        session._running_model_path = active_model_path
        with session.perf(monitor=monitor):
            assert session._running_model_path == active_model_path
            assert session.running_model_path == active_model_path

    assert session._running_model_path == active_model_path
    assert session.running_model_path == active_model_path
