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
import os
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
    GenerationTiming,
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
def bundle_dir_cpu_pipeline(tmp_path: Path) -> Path:
    """CPU-only bundle WITH pipeline stages (no hardware provider_options).

    Used to verify that forcing a hardware EP override rewrites the whole
    pipeline and flips the registration/compile gates on, even though the
    bundle itself never asked for a hardware EP.
    """
    cfg = {
        "model": {
            "type": "decoder-pipeline",
            "context_length": 256,
            "decoder": {
                "pipeline": [
                    {"embeddings": {"filename": "embeddings.onnx"}},
                    {
                        "context": {
                            "filename": "ctx.onnx",
                            "session_options": {"provider_options": []},
                        }
                    },
                ]
            },
        },
        "search": {"max_length": 256},
    }
    (tmp_path / "genai_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_path / "ctx.onnx").write_bytes(b"fake")
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
    # get_sequence returns the full sequence (prompt + generated tokens).
    # Default prompt is "hi" which encodes to a single-element list, so the
    # full sequence is [<prompt_token>, 10, 20].
    gen.get_sequence.return_value = [0, 10, 20]
    og.Generator.return_value = gen

    # TokenizerStream decodes tokens to text
    stream = MagicMock()
    stream.decode.side_effect = ["Hello", " world"]
    og.Tokenizer.return_value.create_stream.return_value = stream

    return og


def _patch_og(mock: MagicMock):
    """Context manager: inject mock_og as onnxruntime_genai in sys.modules."""
    return patch.dict(sys.modules, {"onnxruntime_genai": mock})


def _clock_from(values: list[float]):
    """Return a clock callable that yields ``values`` in order (one per call)."""
    it = iter(values)
    return lambda: next(it)


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

    def test_default_ep_is_none_respects_config(self, bundle_dir: Path) -> None:
        # No ep override -> respect the bundle's genai_config.json routing.
        session = GenaiSession(bundle_dir)
        assert session.ep is None

    def test_not_loaded_after_init(self, bundle_dir: Path) -> None:
        session = GenaiSession(bundle_dir)
        assert not session.is_loaded
        assert session.context_length is None

    def test_bundle_dir_property(self, bundle_dir: Path) -> None:
        session = GenaiSession(bundle_dir)
        assert session.bundle_dir == bundle_dir

    def test_supported_eps(self, bundle_dir: Path) -> None:
        # A concrete override is normalized to its canonical short alias,
        # whether passed as an alias or a full *ExecutionProvider name.
        for ep, expected in (
            ("cpu", "cpu"),
            ("qnn", "qnn"),
            ("dml", "dml"),
            ("openvino", "openvino"),
            ("QNNExecutionProvider", "qnn"),
            ("DmlExecutionProvider", "dml"),
        ):
            session = GenaiSession(bundle_dir, ep=ep)
            assert session.ep == expected

    def test_invalid_ep_raises_value_error(self, bundle_dir: Path) -> None:
        # The retired "mixed"/"auto" sentinels (and any unknown provider) are
        # rejected rather than silently becoming a no-op override.
        for bad in ("mixed", "auto", "not-an-ep"):
            with pytest.raises(ValueError, match="Unknown execution provider"):
                GenaiSession(bundle_dir, ep=bad)

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

    def test_hardware_ep_bundle_registers_winml_eps(
        self, bundle_dir_with_pipeline: Path, mock_og: MagicMock
    ) -> None:
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
            # ep defaults to None (respect config); registration is driven by
            # the bundle config, which routes the ctx/iter stages to QNN.
            session = GenaiSession(bundle_dir_with_pipeline)
            session.load()
        mock_registry.register_execution_providers.assert_called_once_with(ort_genai=True)

    def test_force_hardware_ep_on_cpu_bundle_registers(
        self, bundle_dir_cpu_pipeline: Path, mock_og: MagicMock
    ) -> None:
        # Overriding a CPU-only *pipeline* bundle to a hardware EP rewrites every
        # stage, so the effective config now routes to hardware -> WinML EPs are
        # registered even though the bundle itself never asked for them.
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
            session = GenaiSession(bundle_dir_cpu_pipeline, ep="qnn")
            session.load()
        mock_registry.register_execution_providers.assert_called_once_with(ort_genai=True)

    def test_force_cpu_on_hardware_bundle_skips_registration(
        self, bundle_dir_with_pipeline: Path, mock_og: MagicMock
    ) -> None:
        # Overriding a hardware bundle to CPU strips every stage's hardware
        # provider_options, so the effective config is CPU-only -> no WinML EP
        # registration.
        with (
            _patch_og(mock_og),
            patch("winml.modelkit.session.genai_session.WinMLEPRegistry") as mock_reg_cls,
        ):
            session = GenaiSession(bundle_dir_with_pipeline, ep="cpu")
            session.load()
        mock_reg_cls.assert_not_called()

    def test_force_ep_on_empty_pipeline_is_noop(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        # A bundle with no pipeline stages has nothing to route: forcing "qnn"
        # cannot invent hardware stages, so registration stays off.
        with (
            _patch_og(mock_og),
            patch("winml.modelkit.session.genai_session.WinMLEPRegistry") as mock_reg_cls,
        ):
            session = GenaiSession(bundle_dir, ep="qnn")
            session.load()
        mock_reg_cls.assert_not_called()

    def test_config_not_modified_at_load(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        # EP routing is driven by genai_config.json — we must NOT touch the config.
        with _patch_og(mock_og):
            session = GenaiSession(bundle_dir, ep="cpu")
            session.load()
        mock_og.Config.return_value.clear_providers.assert_not_called()
        mock_og.Config.return_value.append_provider.assert_not_called()

    def test_override_loads_model_from_derived_bundle(
        self, bundle_dir_with_pipeline: Path, mock_og: MagicMock
    ) -> None:
        # A concrete override rewrites the config, so og.Model must load from the
        # derived _compiled/ directory (not the original bundle) to pick it up.
        with _patch_og(mock_og):
            session = GenaiSession(bundle_dir_with_pipeline, ep="cpu")
            session.load()
        compiled_dir = bundle_dir_with_pipeline / "_compiled"
        mock_og.Config.assert_called_once_with(str(compiled_dir))
        written = json.loads((compiled_dir / "genai_config.json").read_text(encoding="utf-8"))
        assert GenaiSession._bundle_uses_hardware_ep(written) is None

    def test_no_override_loads_model_from_original_bundle(
        self, bundle_dir_with_pipeline: Path, mock_og: MagicMock
    ) -> None:
        # ep=None + compile=False: no derived bundle, load straight from source.
        with (
            _patch_og(mock_og),
            patch("winml.modelkit.session.genai_session.WinMLEPRegistry"),
        ):
            session = GenaiSession(bundle_dir_with_pipeline)
            session.load()
        mock_og.Config.assert_called_once_with(str(bundle_dir_with_pipeline))
        assert not (bundle_dir_with_pipeline / "_compiled").exists()


# ---------------------------------------------------------------------------
# Tests: _bundle_uses_hardware_ep (config-driven EP detection)
# ---------------------------------------------------------------------------


class TestBundleUsesHardwareEp:
    @staticmethod
    def _pipeline(*stages: dict) -> dict:
        return {"model": {"decoder": {"pipeline": list(stages)}}}

    def test_empty_config_returns_none(self) -> None:
        assert GenaiSession._bundle_uses_hardware_ep({}) is None

    def test_no_pipeline_returns_none(self) -> None:
        assert GenaiSession._bundle_uses_hardware_ep({"model": {"decoder": {}}}) is None

    def test_cpu_only_stages_return_none(self) -> None:
        cfg = self._pipeline(
            {"embeddings": {"filename": "embeddings.onnx", "session_options": {}}},
            {"lm_head": {"filename": "lm_head.onnx"}},
        )
        assert GenaiSession._bundle_uses_hardware_ep(cfg) is None

    def test_explicit_cpu_provider_returns_none(self) -> None:
        cfg = self._pipeline({"context": {"session_options": {"provider_options": [{"cpu": {}}]}}})
        assert GenaiSession._bundle_uses_hardware_ep(cfg) is None

    def test_qnn_stage_returns_ep_name(self) -> None:
        cfg = self._pipeline(
            {
                "context": {
                    "session_options": {
                        "provider_options": [{"qnn": {"backend_path": "QnnHtp.dll"}}]
                    }
                }
            }
        )
        assert GenaiSession._bundle_uses_hardware_ep(cfg) == "qnn"

    def test_dml_only_returns_none(self) -> None:
        cfg = self._pipeline(
            {"embeddings": {"session_options": {}}},
            {"context": {"session_options": {"provider_options": [{"dml": {}}]}}},
        )
        assert GenaiSession._bundle_uses_hardware_ep(cfg) is None

    def test_malformed_entries_return_none(self) -> None:
        cfg = {"model": {"decoder": {"pipeline": ["not-a-dict", {"x": "not-a-dict"}, {}]}}}
        assert GenaiSession._bundle_uses_hardware_ep(cfg) is None

    def test_provider_options_not_a_list_returns_none(self) -> None:
        cfg = self._pipeline({"context": {"session_options": {"provider_options": {}}}})
        assert GenaiSession._bundle_uses_hardware_ep(cfg) is None

    # Flat-decoder layout (no pipeline wrapper) ----------------------------

    def test_flat_decoder_openvino_returns_ep_name(self) -> None:
        cfg = {
            "model": {
                "decoder": {
                    "session_options": {"provider_options": [{"OpenVINO": {"device_type": "NPU"}}]}
                }
            }
        }
        assert GenaiSession._bundle_uses_hardware_ep(cfg) == "OpenVINO"

    def test_flat_decoder_cpu_only_returns_none(self) -> None:
        cfg = {"model": {"decoder": {"session_options": {"provider_options": [{"cpu": {}}]}}}}
        assert GenaiSession._bundle_uses_hardware_ep(cfg) is None

    def test_flat_decoder_no_provider_options_returns_none(self) -> None:
        cfg = {"model": {"decoder": {"session_options": {"log_id": "test"}}}}
        assert GenaiSession._bundle_uses_hardware_ep(cfg) is None


# ---------------------------------------------------------------------------
# Tests: _apply_ep_override (explicit arg > bundle config)
# ---------------------------------------------------------------------------


class TestEPOverride:
    """The ``ep`` override rewrites the whole pipeline generically."""

    @staticmethod
    def _pipeline_cfg(*stages: dict) -> dict:
        return {"model": {"context_length": 256, "decoder": {"pipeline": list(stages)}}}

    def test_none_override_returns_config_verbatim(self, bundle_dir: Path) -> None:
        # ep=None must not copy or touch the config (config-driven bundles
        # behave exactly as before).
        session = GenaiSession(bundle_dir)
        cfg = self._pipeline_cfg(
            {"context": {"session_options": {"provider_options": [{"qnn": {}}]}}}
        )
        effective, changed = session._apply_ep_override(cfg)
        assert changed is False
        assert effective is cfg

    def test_force_cpu_strips_hardware_provider_options(self, bundle_dir: Path) -> None:
        session = GenaiSession(bundle_dir, ep="cpu")
        cfg = self._pipeline_cfg(
            {"context": {"session_options": {"provider_options": [{"qnn": {}}]}}},
            {"iterator": {"session_options": {"provider_options": [{"dml": {}}]}}},
        )
        effective, changed = session._apply_ep_override(cfg)
        assert changed is True
        stages = effective["model"]["decoder"]["pipeline"]
        assert stages[0]["context"]["session_options"]["provider_options"] == []
        assert stages[1]["iterator"]["session_options"]["provider_options"] == []
        assert GenaiSession._bundle_uses_hardware_ep(effective) is None

    def test_force_cpu_leaves_cpu_stage_untouched(self, bundle_dir: Path) -> None:
        # A stage with no session_options is already CPU; forcing CPU is a no-op.
        session = GenaiSession(bundle_dir, ep="cpu")
        cfg = self._pipeline_cfg({"embeddings": {"filename": "embeddings.onnx"}})
        effective, changed = session._apply_ep_override(cfg)
        assert changed is False
        assert "session_options" not in effective["model"]["decoder"]["pipeline"][0]["embeddings"]

    def test_force_hardware_ep_routes_all_stages(self, bundle_dir: Path) -> None:
        session = GenaiSession(bundle_dir, ep="qnn")
        cfg = self._pipeline_cfg(
            {"embeddings": {"filename": "embeddings.onnx"}},
            {"context": {"session_options": {"provider_options": []}}},
        )
        effective, changed = session._apply_ep_override(cfg)
        assert changed is True
        stages = effective["model"]["decoder"]["pipeline"]
        assert stages[0]["embeddings"]["session_options"]["provider_options"] == [{"qnn": {}}]
        assert stages[1]["context"]["session_options"]["provider_options"] == [{"qnn": {}}]
        assert GenaiSession._bundle_uses_hardware_ep(effective) == "qnn"

    def test_force_same_ep_preserves_existing_options(self, bundle_dir: Path) -> None:
        # Re-selecting the bundle's own EP keeps its finely-tuned options.
        session = GenaiSession(bundle_dir, ep="qnn")
        opts = {"backend_path": "QnnHtp.dll", "soc_model": "60"}
        cfg = self._pipeline_cfg(
            {"context": {"session_options": {"provider_options": [{"qnn": dict(opts)}]}}}
        )
        effective, _ = session._apply_ep_override(cfg)
        stage = effective["model"]["decoder"]["pipeline"][0]["context"]
        assert stage["session_options"]["provider_options"] == [{"qnn": opts}]

    def test_force_different_hardware_ep_drops_foreign_options(self, bundle_dir: Path) -> None:
        # Switching QNN -> DML must not carry QNN's backend_path across.
        session = GenaiSession(bundle_dir, ep="dml")
        cfg = self._pipeline_cfg(
            {"context": {"session_options": {"provider_options": [{"qnn": {"backend_path": "x"}}]}}}
        )
        effective, changed = session._apply_ep_override(cfg)
        assert changed is True
        stage = effective["model"]["decoder"]["pipeline"][0]["context"]
        assert stage["session_options"]["provider_options"] == [{"dml": {}}]

    def test_full_name_override_uses_canonical_alias(self, bundle_dir: Path) -> None:
        # A full *ExecutionProvider name overrides identically to its alias.
        session = GenaiSession(bundle_dir, ep="OpenVINOExecutionProvider")
        cfg = self._pipeline_cfg({"context": {"session_options": {"provider_options": []}}})
        effective, _ = session._apply_ep_override(cfg)
        stage = effective["model"]["decoder"]["pipeline"][0]["context"]
        assert stage["session_options"]["provider_options"] == [{"openvino": {}}]

    def test_override_does_not_mutate_input_config(self, bundle_dir: Path) -> None:
        session = GenaiSession(bundle_dir, ep="cpu")
        cfg = self._pipeline_cfg(
            {"context": {"session_options": {"provider_options": [{"qnn": {"backend_path": "x"}}]}}}
        )
        original = json.loads(json.dumps(cfg))
        session._apply_ep_override(cfg)
        assert cfg == original


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

    def test_encode_returns_token_ids(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        """encode() delegates to the bundle tokenizer and returns a list of IDs."""
        mock_og.Tokenizer.return_value.encode.return_value.tolist.return_value = [5, 6, 7]
        with _patch_og(mock_og), GenaiSession(bundle_dir) as session:
            ids = session.encode("hi there")
        assert ids == [5, 6, 7]
        mock_og.Tokenizer.return_value.encode.assert_called_once_with("hi there")

    def test_encode_auto_loads(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        """encode() auto-loads the session on first use."""
        mock_og.Tokenizer.return_value.encode.return_value.tolist.return_value = [1]
        with _patch_og(mock_og):
            session = GenaiSession(bundle_dir)
            assert not session.is_loaded
            session.encode("hi")
            assert session.is_loaded


# ---------------------------------------------------------------------------
# Tests: GenerationTiming dataclass
# ---------------------------------------------------------------------------


class TestGenerationTiming:
    def test_ttft_is_prefill_plus_first_token(self) -> None:
        t = GenerationTiming(prefill_s=0.4, first_token_s=0.1)
        assert t.ttft_s == pytest.approx(0.5)

    def test_tpot_is_mean_of_decode_steps(self) -> None:
        t = GenerationTiming(decode_s=[0.2, 0.4])
        assert t.tpot_s == pytest.approx(0.3)

    def test_tpot_is_zero_without_decode_steps(self) -> None:
        assert GenerationTiming(decode_s=[]).tpot_s == 0.0

    def test_decode_tokens_per_sec(self) -> None:
        # 4 decode steps totalling 1.0s -> 4 tokens/sec
        t = GenerationTiming(decode_s=[0.25, 0.25, 0.25, 0.25])
        assert t.decode_tokens_per_sec == pytest.approx(4.0)

    def test_decode_tokens_per_sec_is_zero_without_decode_steps(self) -> None:
        assert GenerationTiming(decode_s=[]).decode_tokens_per_sec == 0.0

    def test_total_is_prefill_plus_first_plus_decode(self) -> None:
        t = GenerationTiming(prefill_s=1.0, first_token_s=0.5, decode_s=[0.2, 0.3])
        assert t.total_s == pytest.approx(2.0)

    def test_defaults_are_zero(self) -> None:
        t = GenerationTiming()
        assert t.input_tokens == 0
        assert t.generated_tokens == 0
        assert t.ttft_s == 0.0
        assert t.total_s == 0.0
        assert t.decode_s == []


# ---------------------------------------------------------------------------
# Tests: generate_timed (og-boundary timing)
# ---------------------------------------------------------------------------


class TestGenerateTimed:
    def test_segments_prefill_first_token_and_decode(
        self, bundle_dir: Path, mock_og: MagicMock
    ) -> None:
        # mock_og generator yields 2 tokens (is_done: F, F, T).
        # clock calls: before append(0.0), after append(1.0), token1(2.5), token2(3.0).
        clock = _clock_from([0.0, 1.0, 2.5, 3.0])
        with _patch_og(mock_og), GenaiSession(bundle_dir) as session:
            timing = session.generate_timed([1, 2, 3, 4, 5], clock=clock)

        assert timing.input_tokens == 5
        assert timing.generated_tokens == 2
        assert timing.prefill_s == pytest.approx(1.0)
        assert timing.first_token_s == pytest.approx(1.5)
        assert timing.decode_s == pytest.approx([0.5])
        # TTFT = prefill + first token.
        assert timing.ttft_s == pytest.approx(2.5)
        assert timing.total_s == pytest.approx(3.0)

    def test_does_not_decode_tokens(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        """Only model-compute boundaries are timed — no tokenizer detokenization."""
        clock = _clock_from([0.0, 1.0, 2.5, 3.0])
        with _patch_og(mock_og), GenaiSession(bundle_dir) as session:
            session.generate_timed([1, 2, 3], clock=clock)

        stream = mock_og.Tokenizer.return_value.create_stream.return_value
        stream.decode.assert_not_called()

    def test_forwards_token_list_to_append_tokens(
        self, bundle_dir: Path, mock_og: MagicMock
    ) -> None:
        clock = _clock_from([0.0, 1.0, 2.5, 3.0])
        with _patch_og(mock_og), GenaiSession(bundle_dir) as session:
            session.generate_timed([7, 8, 9], clock=clock)

        mock_og.Generator.return_value.append_tokens.assert_called_once_with([7, 8, 9])

    def test_respects_max_new_tokens(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        gen = mock_og.Generator.return_value
        gen.is_done.side_effect = None
        gen.is_done.return_value = False  # never signals done
        # max_new_tokens=1 -> single token: clock before(0.0), after append(1.0), token1(2.0)
        clock = _clock_from([0.0, 1.0, 2.0])
        with _patch_og(mock_og), GenaiSession(bundle_dir) as session:
            timing = session.generate_timed([1, 2], GenerationConfig(max_new_tokens=1), clock=clock)

        assert timing.generated_tokens == 1
        # A single token has no steady-state decode phase.
        assert timing.decode_s == []
        assert timing.tpot_s == 0.0
        assert timing.decode_tokens_per_sec == 0.0
        assert timing.ttft_s == pytest.approx(2.0)  # prefill 1.0 + first token 1.0

    def test_raises_when_no_tokens(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        gen = mock_og.Generator.return_value
        gen.is_done.side_effect = None
        gen.is_done.return_value = True  # done immediately -> 0 tokens
        clock = _clock_from([0.0, 1.0])
        with (
            _patch_og(mock_og),
            GenaiSession(bundle_dir) as session,
            pytest.raises(GenaiSessionError, match="no tokens"),
        ):
            session.generate_timed([1, 2], clock=clock)

    def test_auto_loads_on_first_call(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        clock = _clock_from([0.0, 1.0, 2.5, 3.0])
        with _patch_og(mock_og):
            session = GenaiSession(bundle_dir)
            assert not session.is_loaded
            session.generate_timed([1, 2, 3], clock=clock)
            assert session.is_loaded

    def test_uses_context_length_as_max_length(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        """max_length = min(prompt_len + max_new_tokens, context_length)."""
        clock = _clock_from([0.0, 1.0, 2.5, 3.0])
        with _patch_og(mock_og), GenaiSession(bundle_dir, context_length=128) as session:
            session.generate_timed([1, 2, 3], clock=clock)

        params = mock_og.GeneratorParams.return_value
        # prompt_len=3 + max_new_tokens=128 = 131, capped at context_length=128
        assert params.set_search_options.call_args.kwargs["max_length"] == 128

    def test_max_length_is_prompt_plus_max_new_tokens(
        self, bundle_dir: Path, mock_og: MagicMock
    ) -> None:
        """When context_length is large, max_length = prompt_len + max_new_tokens."""
        clock = _clock_from([0.0, 1.0, 2.5, 3.0])
        cfg = GenerationConfig(max_new_tokens=64)
        with _patch_og(mock_og), GenaiSession(bundle_dir, context_length=131072) as session:
            session.generate_timed([1, 2, 3, 4, 5], cfg, clock=clock)

        params = mock_og.GeneratorParams.return_value
        # prompt_len=5 + max_new_tokens=64 = 69, well under context_length
        assert params.set_search_options.call_args.kwargs["max_length"] == 69


# ---------------------------------------------------------------------------
# Tests: apply_chat_template
# ---------------------------------------------------------------------------


class TestApplyChatTemplate:
    def test_delegates_to_tokenizer(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        mock_og.Tokenizer.return_value.apply_chat_template.return_value = "TEMPLATED"
        with _patch_og(mock_og), GenaiSession(bundle_dir) as session:
            result = session.apply_chat_template("Hello")
        assert result == "TEMPLATED"

    def test_builds_user_message(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        tok = mock_og.Tokenizer.return_value
        tok.apply_chat_template.return_value = "x"
        with _patch_og(mock_og), GenaiSession(bundle_dir) as session:
            session.apply_chat_template("Hello")
        messages = json.loads(tok.apply_chat_template.call_args.args[0])
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_prepends_system_message(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        tok = mock_og.Tokenizer.return_value
        tok.apply_chat_template.return_value = "x"
        with _patch_og(mock_og), GenaiSession(bundle_dir) as session:
            session.apply_chat_template("Hi", system="You are helpful.")
        messages = json.loads(tok.apply_chat_template.call_args.args[0])
        assert messages == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]

    def test_forwards_add_generation_prompt(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        tok = mock_og.Tokenizer.return_value
        tok.apply_chat_template.return_value = "x"
        with _patch_og(mock_og), GenaiSession(bundle_dir) as session:
            session.apply_chat_template("Hi", add_generation_prompt=False)
        assert tok.apply_chat_template.call_args.kwargs["add_generation_prompt"] is False

    def test_passes_sidecar_template_str(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        (bundle_dir / "chat_template.jinja").write_text("TMPL-BODY", encoding="utf-8")
        tok = mock_og.Tokenizer.return_value
        tok.apply_chat_template.return_value = "x"
        with _patch_og(mock_og), GenaiSession(bundle_dir) as session:
            session.apply_chat_template("Hi")
        assert tok.apply_chat_template.call_args.kwargs["template_str"] == "TMPL-BODY"

    def test_no_sidecar_omits_template_str(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        tok = mock_og.Tokenizer.return_value
        tok.apply_chat_template.return_value = "x"
        with _patch_og(mock_og), GenaiSession(bundle_dir) as session:
            session.apply_chat_template("Hi")
        assert "template_str" not in tok.apply_chat_template.call_args.kwargs

    def test_raises_when_template_unavailable(self, bundle_dir: Path, mock_og: MagicMock) -> None:
        mock_og.Tokenizer.return_value.apply_chat_template.side_effect = RuntimeError(
            "no chat template"
        )
        with (
            _patch_og(mock_og),
            GenaiSession(bundle_dir) as session,
            pytest.raises(GenaiSessionError, match="chat template"),
        ):
            session.apply_chat_template("Hi")


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
            _compile_stage_worker("src.onnx", "dst.onnx", "qnn", {})

        mock_compile.assert_called_once()
        args = mock_compile.call_args.args
        assert args[0] == "src.onnx"
        assert args[1] == "dst.onnx"

    def test_forwards_provider_options_to_ep_config(self) -> None:
        """Stage options from genai_config are forwarded onto the resolved EP config."""
        from winml.modelkit.session.genai_session import _compile_stage_worker

        mock_result = MagicMock(success=True)
        with patch(
            "winml.modelkit.compiler.compile_onnx", return_value=mock_result
        ) as mock_compile:
            _compile_stage_worker(
                "src.onnx",
                "dst.onnx",
                "qnn",
                {"htp_performance_mode": "burst", "soc_model": "60"},
            )

        config = mock_compile.call_args.args[2]
        assert config.ep_config.provider == "qnn"
        assert config.ep_config.provider_options["htp_performance_mode"] == "burst"
        assert config.ep_config.provider_options["soc_model"] == "60"

    def test_dispatches_generically_per_ep_alias(self) -> None:
        """A non-QNN EPContext EP (e.g. OpenVINO) is resolved from its alias, not hardcoded."""
        from winml.modelkit.session.genai_session import _compile_stage_worker

        mock_result = MagicMock(success=True)
        with patch(
            "winml.modelkit.compiler.compile_onnx", return_value=mock_result
        ) as mock_compile:
            _compile_stage_worker("src.onnx", "dst.onnx", "openvino", {})

        config = mock_compile.call_args.args[2]
        assert config.ep_config.provider == "openvino"

    def test_raises_for_non_epcontext_ep(self) -> None:
        """An EP without an EPContext compile step is rejected instead of guessed."""
        from winml.modelkit.session.genai_session import _compile_stage_worker

        with (
            patch("winml.modelkit.compiler.compile_onnx") as mock_compile,
            pytest.raises(RuntimeError, match="does not support EPContext"),
        ):
            _compile_stage_worker("src.onnx", "dst.onnx", "dml", {})
        mock_compile.assert_not_called()

    def test_raises_when_compile_unsuccessful(self) -> None:
        """A failed CompileResult surfaces as a RuntimeError (non-zero subprocess exit)."""
        from winml.modelkit.session.genai_session import _compile_stage_worker

        mock_result = MagicMock(success=False, errors=["ep unavailable"])
        with (
            patch("winml.modelkit.compiler.compile_onnx", return_value=mock_result),
            pytest.raises(RuntimeError, match="Compilation failed"),
        ):
            _compile_stage_worker("src.onnx", "dst.onnx", "qnn", {})


# ---------------------------------------------------------------------------
# Tests: _prepare_compiled_bundle
# ---------------------------------------------------------------------------


class TestPrepareCompiledBundle:
    def test_no_compilable_stages_returns_original_bundle_dir(self, bundle_dir: Path) -> None:
        """When no EPContext-capable stages exist, bundle_dir is returned unchanged."""
        session = GenaiSession(bundle_dir, ep="qnn", compile=True)
        result = session._prepare_compiled_bundle()
        assert result == bundle_dir
        assert not (bundle_dir / "_compiled").exists()

    def test_non_epcontext_stage_is_skipped(self, tmp_path: Path) -> None:
        """A pipeline stage on a non-EPContext EP (DML) stays on JIT, not compiled."""
        cfg = {
            "model": {
                "type": "decoder-pipeline",
                "context_length": 256,
                "decoder": {
                    "pipeline": [
                        {
                            "context": {
                                "filename": "ctx.onnx",
                                "session_options": {"provider_options": [{"dml": {}}]},
                            }
                        }
                    ]
                },
            },
            "search": {"max_length": 256},
        }
        (tmp_path / "genai_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        (tmp_path / "ctx.onnx").write_bytes(b"fake")

        session = GenaiSession(tmp_path, ep="dml", compile=True)
        result = session._prepare_compiled_bundle()

        assert result == tmp_path
        assert not (tmp_path / "_compiled").exists()

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

        with patch("multiprocessing.get_context", return_value=ctx_mock):
            result = session._prepare_compiled_bundle()

        assert result == compiled_dir
        config_out = compiled_dir / "genai_config.json"
        assert config_out.exists()
        written = json.loads(config_out.read_text(encoding="utf-8"))
        assert "model" in written
        # A cache marker is written next to each compiled stage.
        assert (compiled_dir / "context_qnn_ctx.onnx.meta.json").exists()

    def test_override_only_writes_derived_bundle_without_compile(
        self, bundle_dir_with_pipeline: Path
    ) -> None:
        """An override with compile=False still writes a derived _compiled/ bundle.

        The rewritten routing must reach og.Model even when nothing is compiled,
        and the non-compiled ONNX files must be mirrored so they resolve.
        """
        session = GenaiSession(bundle_dir_with_pipeline, ep="cpu")
        cfg = session._read_genai_config()
        effective, overridden = session._apply_ep_override(cfg)
        assert overridden is True

        compiled_dir = bundle_dir_with_pipeline / "_compiled"
        result = session._prepare_compiled_bundle(effective, overridden=True)

        assert result == compiled_dir
        written = json.loads((compiled_dir / "genai_config.json").read_text(encoding="utf-8"))
        assert GenaiSession._bundle_uses_hardware_ep(written) is None
        # ONNX files are mirrored (not compiled) so og.Model finds them by name.
        assert (compiled_dir / "ctx.onnx").exists()
        assert (compiled_dir / "iter.onnx").exists()

    @staticmethod
    def _prime_cache(bundle_dir: Path, marker_opts: dict) -> Path:
        """Pre-create fresh cached EPContext files + markers for both stages."""
        compiled_dir = bundle_dir / "_compiled"
        compiled_dir.mkdir()
        for stage, src_name in (("context", "ctx.onnx"), ("iterator", "iter.onnx")):
            ctx = compiled_dir / f"{stage}_qnn_ctx.onnx"
            ctx.write_bytes(b"ep_ctx")
            GenaiSession._write_compile_marker(ctx, "qnn", marker_opts)
            src_mtime = (bundle_dir / src_name).stat().st_mtime
            os.utime(ctx, (src_mtime + 100, src_mtime + 100))
        return compiled_dir

    def test_reuses_cached_epcontext_when_fresh(
        self, bundle_dir_with_pipeline: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fresh cache (marker matches, ctx newer than sources) is not recompiled."""
        session = GenaiSession(bundle_dir_with_pipeline, ep="qnn", compile=True)
        compiled_dir = self._prime_cache(bundle_dir_with_pipeline, {"backend_path": "QnnHtp.dll"})

        spy = MagicMock(return_value=True)
        monkeypatch.setattr(session, "_compile_stage", spy)
        result = session._prepare_compiled_bundle()

        spy.assert_not_called()
        assert result == compiled_dir
        assert (compiled_dir / "genai_config.json").exists()

    def test_recompiles_when_provider_options_change(
        self, bundle_dir_with_pipeline: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cached EPContext built with different provider options is recompiled."""
        session = GenaiSession(bundle_dir_with_pipeline, ep="qnn", compile=True)
        # Marker records stale options; the bundle uses backend_path=QnnHtp.dll.
        self._prime_cache(bundle_dir_with_pipeline, {"backend_path": "OLD.dll"})

        spy = MagicMock(return_value=True)
        monkeypatch.setattr(session, "_compile_stage", spy)
        session._prepare_compiled_bundle()

        assert spy.call_count == 2

    def test_recompiles_when_data_sidecar_is_newer(
        self, bundle_dir_with_pipeline: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A newer external-weights .data sidecar invalidates the cached EPContext."""
        session = GenaiSession(bundle_dir_with_pipeline, ep="qnn", compile=True)
        compiled_dir = self._prime_cache(bundle_dir_with_pipeline, {"backend_path": "QnnHtp.dll"})

        # ctx.onnx.data is newer than the compiled context stage -> stale.
        ctx_stage = compiled_dir / "context_qnn_ctx.onnx"
        data_sidecar = bundle_dir_with_pipeline / "ctx.onnx.data"
        data_sidecar.write_bytes(b"weights")
        ctx_mtime = ctx_stage.stat().st_mtime
        os.utime(data_sidecar, (ctx_mtime + 100, ctx_mtime + 100))

        spy = MagicMock(return_value=True)
        monkeypatch.setattr(session, "_compile_stage", spy)
        session._prepare_compiled_bundle()

        # Only the context stage (whose .data changed) is recompiled.
        assert spy.call_count == 1
        assert spy.call_args.args[2] == "context"

    def test_compiled_artifact_path_is_keyed_by_ep(
        self, bundle_dir_with_pipeline: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Different EPContext-capable EPs compile to distinct cache artifacts.

        Regression: the derived-bundle cache key must encode the execution
        provider.  Forcing a bundle onto EP-A and later onto EP-B (both leaving
        empty provider options, as happens when a non-native EP is forced) must
        not let EP-A's compiled binary be reused for EP-B, which would load a
        graph built for the wrong accelerator.
        """

        def _context_ctx_path_for(ep: str) -> Path:
            session = GenaiSession(bundle_dir_with_pipeline, ep=ep, compile=True)
            effective, overridden = session._apply_ep_override(session._read_genai_config())
            captured: list[Path] = []

            def _fake_compile(src, ctx, stage_key, ep_alias, ep_opts):
                captured.append(ctx)
                return True

            monkeypatch.setattr(session, "_compile_stage", _fake_compile)
            session._prepare_compiled_bundle(effective, overridden=overridden)
            return next(p for p in captured if p.name.startswith("context"))

        ov_ctx = _context_ctx_path_for("openvino")
        vitis_ctx = _context_ctx_path_for("vitisai")

        assert ov_ctx.name == "context_openvino_ctx.onnx"
        assert vitis_ctx.name == "context_vitisai_ctx.onnx"
        assert ov_ctx != vitis_ctx

    def test_different_ep_does_not_reuse_cached_epcontext(
        self, bundle_dir_with_pipeline: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fresh cache built for one EP is not reused when a different EP is forced.

        Regression companion to :meth:`test_compiled_artifact_path_is_keyed_by_ep`:
        prime a genuinely-fresh OpenVINO cache (matching mtimes + marker with
        empty options), then force VitisAI with identically-empty options and
        assert both stages are recompiled rather than served from the OpenVINO
        artifacts.
        """
        compiled_dir = bundle_dir_with_pipeline / "_compiled"
        compiled_dir.mkdir()
        for stage, src_name in (("context", "ctx.onnx"), ("iterator", "iter.onnx")):
            ov_ctx = compiled_dir / f"{stage}_openvino_ctx.onnx"
            ov_ctx.write_bytes(b"ep_ctx")
            GenaiSession._write_compile_marker(ov_ctx, "openvino", {})
            src_mtime = (bundle_dir_with_pipeline / src_name).stat().st_mtime
            os.utime(ov_ctx, (src_mtime + 100, src_mtime + 100))

        session = GenaiSession(bundle_dir_with_pipeline, ep="vitisai", compile=True)
        effective, overridden = session._apply_ep_override(session._read_genai_config())
        spy = MagicMock(return_value=True)
        monkeypatch.setattr(session, "_compile_stage", spy)
        session._prepare_compiled_bundle(effective, overridden=overridden)

        # Both stages recompiled for VitisAI; the OpenVINO cache is untouched.
        assert spy.call_count == 2
        assert (compiled_dir / "context_openvino_ctx.onnx").exists()


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
