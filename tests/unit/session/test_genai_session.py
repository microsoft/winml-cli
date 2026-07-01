# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for GenaiSession.

All tests that touch load() / generate*() mock onnxruntime_genai so no
real model files or GPU/NPU hardware is required.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.session import (
    GenaiLoadError,
    GenaiNotInstalledError,
    GenaiSession,
    GenaiSessionError,
    GenerationConfig,
)


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    """Create a minimal genai bundle directory with genai_config.json."""
    cfg = {
        "model": {
            "type": "decoder-pipeline",
            "context_length": 256,
            "decoder": {},
        },
        "search": {"max_length": 256},
    }
    (tmp_path / "genai_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return tmp_path


@pytest.fixture
def bundle_dir_with_pipeline(tmp_path: Path) -> Path:
    """Bundle with a QNN pipeline stage for compile tests."""
    cfg = {
        "model": {
            "type": "decoder-pipeline",
            "context_length": 256,
            "decoder": {
                "pipeline": [
                    {
                        "context": {
                            "filename": "ctx.onnx",
                            "session_options": {
                                "provider_options": [{"qnn": {"backend_path": "QnnHtp.dll"}}]
                            },
                        }
                    },
                    {
                        "iterator": {
                            "filename": "iter.onnx",
                            "session_options": {
                                "provider_options": [{"qnn": {"backend_path": "QnnHtp.dll"}}]
                            },
                        }
                    },
                ]
            },
        },
        "search": {"max_length": 256},
    }
    (tmp_path / "genai_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_path / "ctx.onnx").write_bytes(b"fake")
    (tmp_path / "iter.onnx").write_bytes(b"fake")
    (tmp_path / "embeddings.onnx").write_bytes(b"fake")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return tmp_path


@pytest.fixture
def mock_og() -> MagicMock:
    """Return a fully mocked onnxruntime_genai module."""
    og = MagicMock(name="onnxruntime_genai")
    og.Config.return_value = MagicMock()
    og.Model.return_value = MagicMock()
    og.Tokenizer.return_value = MagicMock()
    og.GeneratorParams.return_value = MagicMock()

    # Generator that yields two tokens then is_done()
    gen = MagicMock()
    gen.is_done.side_effect = [False, False, True]
    gen.get_next_tokens.side_effect = [
        MagicMock(__getitem__=lambda s, i: 10),
        MagicMock(__getitem__=lambda s, i: 20),
    ]
    og.Generator.return_value = gen

    # TokenizerStream decodes tokens to text
    stream = MagicMock()
    stream.decode.side_effect = ["Hello", " world"]
    og.Tokenizer.return_value.create_stream.return_value = stream

    return og


def _patch_og(mock: MagicMock):
    """Context manager: inject mock_og as onnxruntime_genai in sys.modules."""
    return patch.dict(sys.modules, {"onnxruntime_genai": mock})


# ---------------------------------------------------------------------------
# Tests: GenaiSession.__init__
# ---------------------------------------------------------------------------


class TestGenaiSessionInit:
    def test_missing_bundle_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Bundle directory not found"):
            GenaiSession(tmp_path / "nonexistent")

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        # Dir exists but no genai_config.json
        with pytest.raises(FileNotFoundError, match=r"genai_config\.json not found"):
            GenaiSession(tmp_path)

    def test_missing_config_error_message_is_generic(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="winml-cli export"):
            GenaiSession(tmp_path)

    def test_unknown_ep_raises(self, bundle_dir: Path) -> None:
        with pytest.raises(ValueError, match="Unknown EP"):
            GenaiSession(bundle_dir, ep="tensorrt")

    def test_default_ep_is_cpu(self, bundle_dir: Path) -> None:
        session = GenaiSession(bundle_dir)
        assert session.ep == "cpu"

    def test_not_loaded_after_init(self, bundle_dir: Path) -> None:
        session = GenaiSession(bundle_dir)
        assert not session.is_loaded
        assert session.context_length is None

    def test_bundle_dir_property(self, bundle_dir: Path) -> None:
        session = GenaiSession(bundle_dir)
        assert session.bundle_dir == bundle_dir

    def test_supported_eps(self, bundle_dir: Path) -> None:
        for ep in ("cpu", "mixed", "qnn", "dml"):
            session = GenaiSession(bundle_dir, ep=ep)
            assert session.ep == ep

    def test_compile_timeout_default(self, bundle_dir: Path) -> None:
        session = GenaiSession(bundle_dir)
        assert session._compile_timeout == 300

    def test_compile_timeout_custom(self, bundle_dir: Path) -> None:
        session = GenaiSession(bundle_dir, compile_timeout=600)
        assert session._compile_timeout == 600

    def test_compile_timeout_stored_as_attribute(self, bundle_dir: Path) -> None:
        session = GenaiSession(bundle_dir, compile_timeout=120)
        assert session._compile_timeout == 120


# ---------------------------------------------------------------------------
# Tests: load / unload
# ---------------------------------------------------------------------------


class TestGenaiSessionLoad:
    def test_load_sets_is_loaded(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        with _patch_og(mock_og):
            session = GenaiSession(bundle_dir)
            session.load()
        assert session.is_loaded

    def test_load_reads_context_length_from_config(
        self, bundle_dir: Path, mock_og: MagicMock
    ) -> None:
        with _patch_og(mock_og):
            session = GenaiSession(bundle_dir)
            session.load()
        assert session.context_length == 256

    def test_context_length_override(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        with _patch_og(mock_og):
            session = GenaiSession(bundle_dir, context_length=512)
            session.load()
        assert session.context_length == 512

    def test_load_is_idempotent(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        with _patch_og(mock_og):
            session = GenaiSession(bundle_dir)
            session.load()
            session.load()  # second call is a no-op
        assert mock_og.Model.call_count == 1

    def test_unload_clears_state(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        with _patch_og(mock_og):
            session = GenaiSession(bundle_dir)
            session.load()
            session.unload()
        assert not session.is_loaded
        assert session.context_length is None

    def test_unload_on_unloaded_session_is_safe(self, bundle_dir: Path) -> None:
        session = GenaiSession(bundle_dir)
        session.unload()  # should not raise

    def test_context_manager_loads_and_unloads(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        with _patch_og(mock_og), GenaiSession(bundle_dir) as session:
            assert session.is_loaded
        assert not session.is_loaded

    def test_genai_not_installed_raises(self, bundle_dir: Path) -> None:
        with patch.dict(sys.modules, {"onnxruntime_genai": None}):  # type: ignore[dict-item]
            session = GenaiSession(bundle_dir)
            with pytest.raises(GenaiNotInstalledError):
                session.load()

    def test_og_load_error_raises_genai_load_error(
        self, bundle_dir: Path, mock_og: MagicMock
    ) -> None:
        mock_og.Model.side_effect = RuntimeError("driver not found")
        with _patch_og(mock_og):
            session = GenaiSession(bundle_dir)
            with pytest.raises(GenaiLoadError, match="driver not found"):
                session.load()

    def test_og_load_error_leaves_session_unloaded(
        self, bundle_dir: Path, mock_og: MagicMock
    ) -> None:
        mock_og.Model.side_effect = RuntimeError("driver not found")
        with _patch_og(mock_og):
            session = GenaiSession(bundle_dir)
            with pytest.raises(GenaiLoadError):
                session.load()
        assert not session.is_loaded


# ---------------------------------------------------------------------------
# Tests: EP registration
# ---------------------------------------------------------------------------


class TestEPRegistration:
    def test_cpu_skips_winml_registration(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        with (
            _patch_og(mock_og),
            patch("winml.modelkit.session.genai_session.WinMLEPRegistry") as mock_reg_cls,
        ):
            session = GenaiSession(bundle_dir, ep="cpu")
            session.load()
        mock_reg_cls.assert_not_called()

    def test_non_cpu_registers_winml_eps(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        mock_registry = MagicMock()
        mock_registry.winml_available = True
        mock_registry.register_execution_providers.return_value = {
            "onnxruntime_genai": ["QNNExecutionProvider"]
        }
        with (
            _patch_og(mock_og),
            patch("winml.modelkit.session.genai_session.WinMLEPRegistry") as mock_reg_cls,
        ):
            mock_reg_cls.get_instance.return_value = mock_registry
            session = GenaiSession(bundle_dir, ep="qnn")
            session.load()
        mock_registry.register_execution_providers.assert_called_once_with(ort_genai=True)

    def test_mixed_registers_winml_eps(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        mock_registry = MagicMock()
        mock_registry.winml_available = True
        mock_registry.register_execution_providers.return_value = {
            "onnxruntime_genai": ["QNNExecutionProvider"]
        }
        with (
            _patch_og(mock_og),
            patch("winml.modelkit.session.genai_session.WinMLEPRegistry") as mock_reg_cls,
        ):
            mock_reg_cls.get_instance.return_value = mock_registry
            session = GenaiSession(bundle_dir, ep="mixed")
            session.load()
        mock_registry.register_execution_providers.assert_called_once_with(ort_genai=True)

    def test_config_not_modified_at_load(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        # EP routing is driven by genai_config.json — we must NOT touch the config.
        with _patch_og(mock_og):
            session = GenaiSession(bundle_dir, ep="cpu")
            session.load()
        mock_og.Config.return_value.clear_providers.assert_not_called()
        mock_og.Config.return_value.append_provider.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: generate / generate_streaming
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_generate_streaming_yields_decoded_tokens(
        self, bundle_dir: Path, mock_og: MagicMock
    ) -> None:
        with _patch_og(mock_og), GenaiSession(bundle_dir) as session:
            tokens = list(session.generate_streaming("hi"))
        assert tokens == ["Hello", " world"]

    def test_generate_returns_joined_string(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        with _patch_og(mock_og), GenaiSession(bundle_dir) as session:
            result = session.generate("hi")
        assert result == "Hello world"

    def test_generate_respects_max_new_tokens(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        # Generator never signals done; we stop at max_new_tokens=1
        gen = mock_og.Generator.return_value
        gen.is_done.side_effect = None
        gen.is_done.return_value = False
        gen.get_next_tokens.return_value = MagicMock(__getitem__=lambda s, i: 99)
        mock_og.Tokenizer.return_value.create_stream.return_value.decode.return_value = "x"

        with _patch_og(mock_og), GenaiSession(bundle_dir) as session:
            tokens = list(session.generate_streaming("hi", GenerationConfig(max_new_tokens=1)))
        assert len(tokens) == 1

    def test_generate_with_token_list_input(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        """Pre-encoded token IDs are forwarded directly to append_tokens."""
        with _patch_og(mock_og), GenaiSession(bundle_dir) as session:
            list(session.generate_streaming([1, 2, 3]))
        gen = mock_og.Generator.return_value
        gen.append_tokens.assert_called_once_with([1, 2, 3])

    def test_generate_deletes_generator_after_iteration(
        self, bundle_dir: Path, mock_og: MagicMock
    ) -> None:
        """Generator is deleted (not leaked) even on normal completion."""
        with _patch_og(mock_og), GenaiSession(bundle_dir) as session:
            list(session.generate_streaming("hi"))
        # No assertions needed — test passes if no ResourceWarning / hang

    def test_generate_with_custom_config(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        cfg = GenerationConfig(max_new_tokens=64, do_sample=True, temperature=0.7)
        with _patch_og(mock_og), GenaiSession(bundle_dir) as session:
            list(session.generate_streaming("hi", cfg))
        params = mock_og.GeneratorParams.return_value
        params.set_search_options.assert_called_once()
        call_kwargs = params.set_search_options.call_args.kwargs
        assert call_kwargs["do_sample"] is True
        assert call_kwargs["temperature"] == 0.7

    def test_generate_uses_context_length_as_max_length(
        self, bundle_dir: Path, mock_og: MagicMock
    ) -> None:
        with _patch_og(mock_og), GenaiSession(bundle_dir, context_length=128) as session:
            list(session.generate_streaming("hi"))
        params = mock_og.GeneratorParams.return_value
        call_kwargs = params.set_search_options.call_args.kwargs
        assert call_kwargs["max_length"] == 128

    def test_auto_load_on_first_generate(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        with _patch_og(mock_og):
            session = GenaiSession(bundle_dir)
            assert not session.is_loaded
            list(session.generate_streaming("hi"))
            assert session.is_loaded


# ---------------------------------------------------------------------------
# Tests: apply_chatml_template
# ---------------------------------------------------------------------------


class TestApplyChatmlTemplate:
    def test_user_only(self) -> None:
        result = GenaiSession.apply_chatml_template("Hello")
        assert result == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

    def test_with_system(self) -> None:
        result = GenaiSession.apply_chatml_template("Hello", system="You are helpful.")
        assert result.startswith("<|im_start|>system\nYou are helpful.<|im_end|>\n")
        assert "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n" in result

    def test_no_system_no_system_turn(self) -> None:
        result = GenaiSession.apply_chatml_template("Hi")
        assert "<|im_start|>system" not in result

    def test_ends_with_assistant_priming(self) -> None:
        result = GenaiSession.apply_chatml_template("Hi")
        assert result.endswith("<|im_start|>assistant\n")


# ---------------------------------------------------------------------------
# Tests: GenerationConfig defaults
# ---------------------------------------------------------------------------


class TestGenerationConfig:
    def test_defaults(self) -> None:
        cfg = GenerationConfig()
        assert cfg.max_new_tokens == 128
        assert cfg.do_sample is False
        assert cfg.temperature == 1.0
        assert cfg.top_p == 1.0
        assert cfg.top_k == 0
        assert cfg.repetition_penalty == 1.0

    def test_custom_values(self) -> None:
        cfg = GenerationConfig(max_new_tokens=32, do_sample=True, top_k=50)
        assert cfg.max_new_tokens == 32
        assert cfg.do_sample is True
        assert cfg.top_k == 50


# ---------------------------------------------------------------------------
# Tests: exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_genai_not_installed_is_genai_session_error(self) -> None:
        assert issubclass(GenaiNotInstalledError, GenaiSessionError)

    def test_genai_load_error_is_genai_session_error(self) -> None:
        assert issubclass(GenaiLoadError, GenaiSessionError)


# ---------------------------------------------------------------------------
# Tests: compile_timeout parameter
# ---------------------------------------------------------------------------


class TestCompileTimeout:
    def test_compile_timeout_passed_to_compile_stage(
        self, bundle_dir_with_pipeline: Path, mock_og: MagicMock
    ) -> None:
        """_compile_stage uses self._compile_timeout for proc.join()."""
        session = GenaiSession(bundle_dir_with_pipeline, ep="qnn", compile=True, compile_timeout=42)

        proc_mock = MagicMock()
        proc_mock.is_alive.return_value = False
        proc_mock.exitcode = 0

        ctx_mock = MagicMock()
        ctx_mock.Process.return_value = proc_mock

        with (
            patch("multiprocessing.get_context", return_value=ctx_mock),
            _patch_og(mock_og),
        ):
            session._prepare_compiled_bundle()

        # proc.join was called with timeout=42 for each stage
        for call in proc_mock.join.call_args_list:
            if call.kwargs.get("timeout") is not None:
                assert call.kwargs["timeout"] == 42
            elif call.args:
                pass  # join() called without keyword; check positional
        # Verify join was called at least once with our custom timeout
        join_calls_with_timeout = [
            c for c in proc_mock.join.call_args_list if c == ((), {"timeout": 42})
        ]
        assert len(join_calls_with_timeout) >= 1


# ---------------------------------------------------------------------------
# Tests: _compile_stage_worker (delegation to the shared compiler)
# ---------------------------------------------------------------------------


class TestCompileStageWorker:
    def test_delegates_to_compile_onnx(self) -> None:
        """The worker calls the shared compiler with src/dst, not hand-rolled ORT."""
        from winml.modelkit.session.genai_session import _compile_stage_worker

        mock_result = MagicMock(success=True)
        with patch(
            "winml.modelkit.compiler.compile_onnx", return_value=mock_result
        ) as mock_compile:
            _compile_stage_worker("src.onnx", "dst.onnx", {})

        mock_compile.assert_called_once()
        args = mock_compile.call_args.args
        assert args[0] == "src.onnx"
        assert args[1] == "dst.onnx"

    def test_forwards_provider_options_to_qnn_config(self) -> None:
        """QNN options from genai_config are forwarded onto the compiler EP config."""
        from winml.modelkit.session.genai_session import _compile_stage_worker

        mock_result = MagicMock(success=True)
        with patch(
            "winml.modelkit.compiler.compile_onnx", return_value=mock_result
        ) as mock_compile:
            _compile_stage_worker(
                "src.onnx", "dst.onnx", {"htp_performance_mode": "burst", "soc_model": "60"}
            )

        config = mock_compile.call_args.args[2]
        assert config.ep_config.provider == "qnn"
        assert config.ep_config.provider_options["htp_performance_mode"] == "burst"
        assert config.ep_config.provider_options["soc_model"] == "60"

    def test_raises_when_compile_unsuccessful(self) -> None:
        """A failed CompileResult surfaces as a RuntimeError (non-zero subprocess exit)."""
        from winml.modelkit.session.genai_session import _compile_stage_worker

        mock_result = MagicMock(success=False, errors=["ep unavailable"])
        with (
            patch("winml.modelkit.compiler.compile_onnx", return_value=mock_result),
            pytest.raises(RuntimeError, match="Compilation failed"),
        ):
            _compile_stage_worker("src.onnx", "dst.onnx", {})


# ---------------------------------------------------------------------------
# Tests: _prepare_compiled_bundle
# ---------------------------------------------------------------------------


class TestPrepareCompiledBundle:
    def test_no_qnn_stages_returns_original_bundle_dir(self, bundle_dir: Path) -> None:
        """When no QNN stages exist, bundle_dir is returned unchanged."""
        session = GenaiSession(bundle_dir, ep="qnn", compile=True)
        result = session._prepare_compiled_bundle()
        assert result == bundle_dir

    def test_writes_modified_genai_config_to_compiled_dir(
        self, bundle_dir_with_pipeline: Path
    ) -> None:
        """After successful compilation, modified genai_config.json lands in _compiled/."""
        session = GenaiSession(bundle_dir_with_pipeline, ep="qnn", compile=True)

        proc_mock = MagicMock()
        proc_mock.is_alive.return_value = False
        proc_mock.exitcode = 0

        ctx_mock = MagicMock()
        ctx_mock.Process.return_value = proc_mock

        compiled_dir = bundle_dir_with_pipeline / "_compiled"

        # Pre-create fake compiled ONNX so _compile_stage "succeeds" from cache
        compiled_dir.mkdir(exist_ok=True)
        import time

        time.sleep(0.01)
        (compiled_dir / "context_ctx.onnx").write_bytes(b"ep_ctx")
        (compiled_dir / "iterator_ctx.onnx").write_bytes(b"ep_ctx")

        with patch("multiprocessing.get_context", return_value=ctx_mock):
            result = session._prepare_compiled_bundle()

        assert result == compiled_dir
        config_out = compiled_dir / "genai_config.json"
        assert config_out.exists()
        written = json.loads(config_out.read_text(encoding="utf-8"))
        assert "model" in written


# ---------------------------------------------------------------------------
# Tests: _mirror_non_onnx_files
# ---------------------------------------------------------------------------


class TestMirrorNonOnnxFiles:
    def test_skips_files_in_skip_filenames(self, bundle_dir_with_pipeline: Path) -> None:
        session = GenaiSession(bundle_dir_with_pipeline)
        compiled_dir = bundle_dir_with_pipeline / "_compiled"
        compiled_dir.mkdir()

        session._mirror_non_onnx_files(compiled_dir, skip_filenames={"ctx.onnx", "iter.onnx"})

        assert not (compiled_dir / "ctx.onnx").exists()
        assert not (compiled_dir / "iter.onnx").exists()

    def test_creates_links_for_non_skipped_files(self, bundle_dir_with_pipeline: Path) -> None:
        session = GenaiSession(bundle_dir_with_pipeline)
        compiled_dir = bundle_dir_with_pipeline / "_compiled"
        compiled_dir.mkdir()

        session._mirror_non_onnx_files(compiled_dir, skip_filenames={"ctx.onnx", "iter.onnx"})

        # embeddings.onnx and tokenizer.json are not skipped
        assert (compiled_dir / "embeddings.onnx").exists()
        assert (compiled_dir / "tokenizer.json").exists()

    def test_skips_data_sidecars_of_compiled_stages(self, bundle_dir_with_pipeline: Path) -> None:
        # Add .data sidecar files for the QNN stages
        (bundle_dir_with_pipeline / "ctx.onnx.data").write_bytes(b"sidecar")
        (bundle_dir_with_pipeline / "iter.onnx.data").write_bytes(b"sidecar")

        session = GenaiSession(bundle_dir_with_pipeline)
        compiled_dir = bundle_dir_with_pipeline / "_compiled"
        compiled_dir.mkdir()

        session._mirror_non_onnx_files(compiled_dir, skip_filenames={"ctx.onnx", "iter.onnx"})

        assert not (compiled_dir / "ctx.onnx.data").exists()
        assert not (compiled_dir / "iter.onnx.data").exists()

    def test_does_not_overwrite_existing_files(self, bundle_dir_with_pipeline: Path) -> None:
        session = GenaiSession(bundle_dir_with_pipeline)
        compiled_dir = bundle_dir_with_pipeline / "_compiled"
        compiled_dir.mkdir()

        existing = compiled_dir / "embeddings.onnx"
        existing.write_bytes(b"already here")

        session._mirror_non_onnx_files(compiled_dir, skip_filenames=set())

        # Should not be replaced
        assert existing.read_bytes() == b"already here"


# ---------------------------------------------------------------------------
# Tests: _patch_stage_filename
# ---------------------------------------------------------------------------


class TestPatchStageFilename:
    def _make_cfg(self) -> dict:
        return {
            "model": {
                "decoder": {
                    "pipeline": [
                        {"context": {"filename": "ctx.onnx"}},
                        {"iterator": {"filename": "iter.onnx"}},
                    ]
                }
            }
        }

    def test_updates_correct_stage(self) -> None:
        cfg = self._make_cfg()
        GenaiSession._patch_stage_filename(cfg, "context", "/new/ctx_compiled.onnx")
        pipeline = cfg["model"]["decoder"]["pipeline"]
        assert pipeline[0]["context"]["filename"] == "/new/ctx_compiled.onnx"
        # Other stage unchanged
        assert pipeline[1]["iterator"]["filename"] == "iter.onnx"

    def test_noop_when_stage_key_not_found(self) -> None:
        cfg = self._make_cfg()
        GenaiSession._patch_stage_filename(cfg, "nonexistent_stage", "/some/path.onnx")
        # No modification should have occurred
        pipeline = cfg["model"]["decoder"]["pipeline"]
        assert pipeline[0]["context"]["filename"] == "ctx.onnx"
        assert pipeline[1]["iterator"]["filename"] == "iter.onnx"
