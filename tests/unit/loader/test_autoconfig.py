# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the Transformers config compatibility loader."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.loader import load_hf_config, resolve_task


if TYPE_CHECKING:
    from pathlib import Path


class _FailingAutoConfig:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise ValueError("missing required key: model_type")


class _FakeConfig:
    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        return ("specific", config_dict, kwargs)


class _StrictFallbackConfig:
    dilation: bool = False

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        return {
            "kind": "normalized",
            "config_dict": config_dict,
            "kwargs": kwargs,
        }


class _StrictFallbackConfigWithUnused:
    dilation: bool = False

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        payload = {
            "kind": "normalized",
            "config_dict": config_dict,
            "kwargs": kwargs,
        }
        if kwargs.get("return_unused_kwargs"):
            return payload, {"sentinel": kwargs.get("sentinel")}
        return payload


def test_local_path_uses_saved_model_identifier(tmp_path: Path) -> None:
    local_model_dir = tmp_path / "exports" / "run1"
    local_model_dir.mkdir(parents=True)
    config_dict = {
        "_name_or_path": "owner/specific-model",
        "hidden_size": 128,
    }

    with (
        patch(
            "transformers.PretrainedConfig.get_config_dict",
            return_value=(config_dict, {}),
        ),
        patch(
            "transformers.models.auto.configuration_auto.CONFIG_MAPPING",
            {"specific": _FakeConfig},
        ),
    ):
        config = load_hf_config(_FailingAutoConfig, str(local_model_dir))

    assert config[0] == "specific"


def test_local_parent_directory_does_not_drive_architecture_inference(tmp_path: Path) -> None:
    local_model_dir = tmp_path / "specific-parent" / "run1"
    local_model_dir.mkdir(parents=True)
    config_dict = {"hidden_size": 128}
    generic_config = type("GenericConfig", (), {})()

    with (
        patch(
            "transformers.PretrainedConfig.get_config_dict",
            return_value=(config_dict, {"revision": "main"}),
        ),
        patch(
            "transformers.models.auto.configuration_auto.CONFIG_MAPPING",
            {"specific": _FakeConfig},
        ),
        patch(
            "transformers.PretrainedConfig.from_dict",
            return_value=generic_config,
        ) as mock_generic,
    ):
        config = load_hf_config(_FailingAutoConfig, str(local_model_dir))

    assert config is generic_config
    assert getattr(config, "_winml_generic_fallback", False) is True
    mock_generic.assert_called_once_with(config_dict, revision="main")


def test_hub_model_id_wins_over_stale_saved_identifier() -> None:
    config_dict = {
        "_name_or_path": r"C:\tmp\other-model",
        "hidden_size": 128,
    }

    with (
        patch(
            "transformers.PretrainedConfig.get_config_dict",
            return_value=(config_dict, {}),
        ),
        patch(
            "transformers.models.auto.configuration_auto.CONFIG_MAPPING",
            {"specific": _FakeConfig, "other": object()},
        ),
    ):
        config = load_hf_config(_FailingAutoConfig, "owner/specific-model")

    assert config[0] == "specific"


@pytest.mark.parametrize("transformers_version", ["4.57.1", "5.0.0", "5.14.1"])
def test_model_type_less_config_bypasses_transformers_path_fallback(
    transformers_version: str,
) -> None:
    config_dict = {"hidden_size": 128}
    generic_config = type("GenericConfig", (), {})()
    auto_config = MagicMock()
    auto_config.from_pretrained.return_value = _FakeConfig()

    with (
        patch(
            "transformers.PretrainedConfig.get_config_dict",
            return_value=(config_dict, {}),
        ),
        patch(
            "transformers.models.auto.configuration_auto.CONFIG_MAPPING",
            {"specific": _FakeConfig},
        ),
        patch(
            "transformers.PretrainedConfig.from_dict",
            return_value=generic_config,
        ),
        patch("transformers.__version__", transformers_version),
    ):
        config = load_hf_config(auto_config, r"C:\specific-parent\neutral-model")

    assert config is generic_config
    auto_config.from_pretrained.assert_not_called()


def test_inferred_config_accepts_matching_declared_architectures() -> None:
    from transformers import BertConfig

    config_dict = {
        "architectures": ["BertForMaskedLM", "BertForSequenceClassification"],
        "hidden_size": 128,
    }

    with patch(
        "transformers.PretrainedConfig.get_config_dict",
        return_value=(config_dict, {}),
    ) as raw_config_loader:
        config = load_hf_config(
            _FailingAutoConfig,
            "owner/bert-neutral",
        )

    assert isinstance(config, BertConfig)
    raw_config_loader.assert_called_once()


@pytest.mark.parametrize(
    "architectures",
    [
        ["RobertaForMaskedLM"],
        ["UnknownModel"],
        "BertModel",
        [],
        ["BertModel", 42],
        ["BertModel", "RobertaModel"],
    ],
    ids=[
        "conflicting",
        "unknown",
        "not-a-list",
        "empty",
        "non-string",
        "mixed-family",
    ],
)
def test_inferred_config_rejects_incompatible_declared_architectures(
    architectures: object,
) -> None:
    from transformers import PretrainedConfig

    config_dict = {"architectures": architectures, "hidden_size": 128}

    with patch(
        "transformers.PretrainedConfig.get_config_dict",
        return_value=(config_dict, {}),
    ) as raw_config_loader:
        config = load_hf_config(
            _FailingAutoConfig,
            "owner/bert-neutral",
        )

    assert type(config) is PretrainedConfig
    assert config._winml_generic_fallback is True
    raw_config_loader.assert_called_once()


def test_injected_raw_config_loader_is_used_once_without_global_raw_fetch() -> None:
    auth_token = object()
    raw_config_loader = MagicMock(
        return_value=(
            {"hidden_size": 128},
            {"return_unused_kwargs": True, "sentinel": "unused"},
        )
    )

    with patch(
        "transformers.PretrainedConfig.get_config_dict",
        side_effect=AssertionError("global raw retrieval must not run"),
    ) as global_raw_loader:
        config, unused_kwargs = load_hf_config(
            _FailingAutoConfig,
            "owner/neutral-model",
            raw_config_loader=raw_config_loader,
            return_unused_kwargs=True,
            revision="raw-revision",
            subfolder="export",
            force_download=True,
            token=auth_token,
            code_revision="code-revision",
            sentinel="unused",
        )

    assert config._winml_generic_fallback is True
    assert unused_kwargs == {"sentinel": "unused"}
    raw_config_loader.assert_called_once_with(
        "owner/neutral-model",
        return_unused_kwargs=True,
        revision="raw-revision",
        subfolder="export",
        force_download=True,
        token=auth_token,
        sentinel="unused",
        _from_auto=True,
        name_or_path="owner/neutral-model",
    )
    global_raw_loader.assert_not_called()


@pytest.mark.parametrize(
    ("config_dict", "trust_remote_code", "transformers_version"),
    [
        ({"model_type": "specific"}, False, "4.57.1"),
        ({"auto_map": {"AutoConfig": "config.CustomConfig"}}, True, "4.57.1"),
        ({"auto_map": {"AutoConfig": "config.CustomConfig"}}, False, "4.57.1"),
        ({"model_type": "specific"}, False, "5.14.1"),
    ],
)
def test_concrete_or_trusted_remote_config_uses_auto_config(
    config_dict: dict[str, object],
    trust_remote_code: bool,
    transformers_version: str,
) -> None:
    expected_config = object()
    auto_config = MagicMock()
    auto_config.from_pretrained.return_value = expected_config

    with (
        patch(
            "transformers.PretrainedConfig.get_config_dict",
            return_value=(config_dict, {}),
        ),
        patch("transformers.__version__", transformers_version),
    ):
        config = load_hf_config(
            auto_config,
            "owner/model",
            trust_remote_code=trust_remote_code,
        )

    assert config is expected_config
    auto_config.from_pretrained.assert_called_once_with(
        "owner/model",
        trust_remote_code=trust_remote_code,
    )


def test_concrete_config_strict_none_bool_validation_uses_normalized_from_dict() -> None:
    strict_error = type(
        "StrictDataclassFieldValidationError",
        (Exception,),
        {},
    )(
        "Validation error for field 'dilation': "
        "TypeError: Field 'dilation' expected bool, got NoneType (value: None)"
    )
    auto_config = MagicMock()
    auto_config.from_pretrained.side_effect = strict_error
    source_config_dict = {
        "model_type": "table-transformer",
        "dilation": None,
        "hidden_size": 256,
    }

    with (
        patch(
            "transformers.PretrainedConfig.get_config_dict",
            return_value=(source_config_dict, {"sentinel": "unused"}),
        ),
        patch(
            "transformers.models.auto.configuration_auto.CONFIG_MAPPING",
            {"table-transformer": _StrictFallbackConfig},
        ),
    ):
        config = load_hf_config(auto_config, "owner/model")

    assert config["kind"] == "normalized"
    assert config["config_dict"]["dilation"] is False
    assert source_config_dict["dilation"] is None
    assert config["kwargs"] == {"sentinel": "unused"}


def test_concrete_config_strict_none_bool_validation_preserves_return_unused_kwargs() -> None:
    strict_error = type(
        "StrictDataclassFieldValidationError",
        (Exception,),
        {},
    )(
        "Validation error for field 'dilation': "
        "TypeError: Field 'dilation' expected bool, got NoneType (value: None)"
    )
    auto_config = MagicMock()
    auto_config.from_pretrained.side_effect = strict_error

    with (
        patch(
            "transformers.PretrainedConfig.get_config_dict",
            return_value=(
                {"model_type": "table-transformer", "dilation": None},
                {"sentinel": "unused"},
            ),
        ),
        patch(
            "transformers.models.auto.configuration_auto.CONFIG_MAPPING",
            {"table-transformer": _StrictFallbackConfigWithUnused},
        ),
    ):
        config, unused_kwargs = load_hf_config(
            auto_config,
            "owner/model",
            return_unused_kwargs=True,
            sentinel="unused",
        )

    assert config["kind"] == "normalized"
    assert config["config_dict"]["dilation"] is False
    assert config["kwargs"]["return_unused_kwargs"] is True
    assert unused_kwargs == {"sentinel": "unused"}


def test_concrete_config_non_strict_error_is_raised() -> None:
    auto_config = MagicMock()
    auto_config.from_pretrained.side_effect = RuntimeError("boom")

    with (
        patch(
            "transformers.PretrainedConfig.get_config_dict",
            return_value=({"model_type": "table-transformer"}, {}),
        ),
        pytest.raises(RuntimeError, match="boom"),
    ):
        load_hf_config(auto_config, "owner/model")


def test_generic_fallback_preserves_return_unused_kwargs_shape() -> None:
    with (
        patch(
            "transformers.PretrainedConfig.get_config_dict",
            return_value=(
                {"hidden_size": 128},
                {"return_unused_kwargs": True, "sentinel": "unused"},
            ),
        ),
        patch(
            "transformers.models.auto.configuration_auto.CONFIG_MAPPING",
            {},
        ),
    ):
        config, unused_kwargs = load_hf_config(
            _FailingAutoConfig,
            "owner/neutral-model",
            return_unused_kwargs=True,
            sentinel="unused",
        )

    assert getattr(config, "_winml_generic_fallback", False) is True
    assert unused_kwargs == {"sentinel": "unused"}


def test_fallback_preserves_caller_identity_and_consumes_code_revision(tmp_path: Path) -> None:
    from transformers import AutoConfig

    model_dir = tmp_path / "neutral-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"hidden_size": 128}', encoding="utf-8")

    with patch("transformers.__version__", "4.57.1"):
        config, unused_kwargs = load_hf_config(
            AutoConfig,
            str(model_dir),
            code_revision="revision",
            return_unused_kwargs=True,
            sentinel="unused",
        )

    assert config._name_or_path == str(model_dir)
    assert unused_kwargs == {"sentinel": "unused"}


def test_transformers4_fallback_converts_legacy_auth_token() -> None:
    seen_kwargs: dict[str, object] = {}
    legacy_token = object()

    def _get_config_dict(*_args, **kwargs):
        seen_kwargs.update(kwargs)
        return {"hidden_size": 128}, {}

    with (
        patch("transformers.__version__", "4.57.1"),
        patch(
            "transformers.PretrainedConfig.get_config_dict",
            side_effect=_get_config_dict,
        ),
        patch("transformers.models.auto.configuration_auto.CONFIG_MAPPING", {}),
        pytest.warns(FutureWarning, match="use_auth_token.*deprecated"),
    ):
        load_hf_config(
            _FailingAutoConfig,
            "owner/neutral-model",
            use_auth_token=legacy_token,
        )

    assert seen_kwargs["token"] is legacy_token
    assert "use_auth_token" not in seen_kwargs


def test_transformers4_fallback_ignores_none_legacy_auth_token() -> None:
    raw_config_loader = MagicMock(return_value=({"hidden_size": 128}, {}))

    with (
        patch("transformers.__version__", "4.57.1"),
        patch("transformers.models.auto.configuration_auto.CONFIG_MAPPING", {}),
    ):
        load_hf_config(
            _FailingAutoConfig,
            "owner/neutral-model",
            raw_config_loader=raw_config_loader,
            use_auth_token=None,
        )

    assert "use_auth_token" not in raw_config_loader.call_args.kwargs
    assert "token" not in raw_config_loader.call_args.kwargs


def test_transformers4_fallback_rejects_conflicting_auth_tokens() -> None:
    token = object()
    legacy_token = object()

    with (
        patch("transformers.__version__", "4.57.1"),
        patch("transformers.PretrainedConfig.get_config_dict") as get_config_dict,
        pytest.warns(FutureWarning, match="use_auth_token.*deprecated"),
        pytest.raises(ValueError, match="both specified"),
    ):
        load_hf_config(
            _FailingAutoConfig,
            "owner/neutral-model",
            token=token,
            use_auth_token=legacy_token,
        )

    get_config_dict.assert_not_called()


@pytest.mark.parametrize(
    "saved_model_id",
    [
        r"C:\tmp\specific-export",
        r"C:specific-export",
        "C:models/specific-export",
        "cache/export/specific-export",
        "specific-export/",
        "~alice/specific-export",
    ],
)
def test_explicit_local_saved_identifier_is_not_an_architecture_hint(
    saved_model_id: str,
) -> None:
    config_dict = {
        "_name_or_path": saved_model_id,
        "hidden_size": 128,
    }
    generic_config = type("GenericConfig", (), {})()

    with (
        patch(
            "transformers.PretrainedConfig.get_config_dict",
            return_value=(config_dict, {}),
        ),
        patch(
            "transformers.models.auto.configuration_auto.CONFIG_MAPPING",
            {"specific": _FakeConfig},
        ),
        patch(
            "transformers.PretrainedConfig.from_dict",
            return_value=generic_config,
        ),
    ):
        config = load_hf_config(_FailingAutoConfig, "owner/neutral-model")

    assert config is generic_config


@pytest.mark.parametrize(
    "saved_model_id",
    [
        "specific/run1",
        r"specific\run1",
        r"C:\specific\run1",
    ],
)
def test_local_saved_path_uses_only_its_model_name(
    saved_model_id: str,
    tmp_path: Path,
) -> None:
    local_model_dir = tmp_path / "exports" / "run1"
    local_model_dir.mkdir(parents=True)
    config_dict = {
        "_name_or_path": saved_model_id,
        "hidden_size": 128,
    }
    generic_config = type("GenericConfig", (), {})()

    with (
        patch(
            "transformers.PretrainedConfig.get_config_dict",
            return_value=(config_dict, {}),
        ),
        patch(
            "transformers.models.auto.configuration_auto.CONFIG_MAPPING",
            {"specific": _FakeConfig},
        ),
        patch(
            "transformers.PretrainedConfig.from_dict",
            return_value=generic_config,
        ),
    ):
        config = load_hf_config(_FailingAutoConfig, str(local_model_dir))

    assert config is generic_config


def test_resolve_task_rejects_generic_fallback_even_with_overrides() -> None:
    from transformers import PretrainedConfig

    config = PretrainedConfig()
    config._winml_generic_fallback = True

    with pytest.raises(ValueError, match="overrides are not enough"):
        resolve_task(config, task="fill-mask", model_class="AutoModelForMaskedLM")
