# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""HF config loading with tolerance for model_type-less configs.

:func:`load_hf_config` applies identifier-based inference to a trusted
model-name segment, then returns a tagged generic config when no concrete
architecture can be inferred safely.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast, overload


if TYPE_CHECKING:
    from transformers import PretrainedConfig


logger = logging.getLogger(__name__)


_RawConfigLoader: TypeAlias = Callable[..., tuple[dict[str, Any], dict[str, Any]]]


def _is_strict_none_bool_validation_error(exc: Exception) -> bool:
    """Return whether *exc* matches strict dataclass bool-vs-None validation."""
    message = str(exc)
    return (
        "Validation error for field" in message
        and "expected bool" in message
        and "NoneType" in message
    )


def _declared_bool_fields(config_cls: type[Any]) -> set[str]:
    """Collect fields annotated as ``bool`` across a config class hierarchy."""

    def _is_bool_annotation(annotation: Any) -> bool:
        if annotation is bool:
            return True
        if isinstance(annotation, str):
            return annotation == "bool" or annotation == "builtins.bool"
        return getattr(annotation, "__forward_arg__", None) == "bool"

    fields: set[str] = set()
    for cls in config_cls.__mro__:
        annotations = getattr(cls, "__annotations__", {})
        if not isinstance(annotations, dict):
            continue
        for field_name, field_type in annotations.items():
            if _is_bool_annotation(field_type) and isinstance(field_name, str):
                fields.add(field_name)
    return fields


def _normalize_none_bool_fields(
    config_cls: type[Any],
    config_dict: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Replace ``None`` with ``False`` for bool-annotated fields present in config."""
    normalized = config_dict.copy()
    replaced_fields: list[str] = []
    for field_name in sorted(_declared_bool_fields(config_cls)):
        if normalized.get(field_name) is None:
            normalized[field_name] = False
            replaced_fields.append(field_name)
    return normalized, replaced_fields


def _load_from_dict_with_strict_none_fallback(
    *,
    config_dict: dict[str, Any],
    unused_kwargs: dict[str, Any],
    model_id: str,
    return_unused_kwargs: bool,
) -> PretrainedConfig | tuple[PretrainedConfig, dict[str, Any]] | None:
    """Try rebuilding config after normalizing strict ``None`` bool fields."""
    model_type = config_dict.get("model_type")
    if not isinstance(model_type, str):
        return None

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if model_type not in CONFIG_MAPPING:
        return None

    config_cls = CONFIG_MAPPING[model_type]
    normalized_config_dict, replaced_fields = _normalize_none_bool_fields(config_cls, config_dict)
    if not replaced_fields:
        return None

    logger.warning(
        "Strict dataclass validation failed for '%s'; normalizing None->False for bool fields: %s",
        model_id,
        ", ".join(replaced_fields),
    )
    from_dict_kwargs = unused_kwargs.copy()
    if return_unused_kwargs:
        from_dict_kwargs["return_unused_kwargs"] = True
    return config_cls.from_dict(normalized_config_dict, **from_dict_kwargs)


def _fallback_identifiers(config_dict: dict[str, Any], model_id: str) -> list[str]:
    """Return trusted model-name segments for fallback config inference."""
    from ..utils.hub_utils import _is_local_path, _is_valid_hub_model_id

    def _model_name(value: str) -> str:
        normalized = value.strip().rstrip("\\/").replace("\\", "/")
        return normalized.rsplit("/", 1)[-1]

    model_id_is_local = _is_local_path(model_id)
    saved_model_id = config_dict.get("_name_or_path")
    normalized_saved_model_id = (
        _model_name(saved_model_id)
        if (
            isinstance(saved_model_id, str)
            and saved_model_id.strip()
            and _is_valid_hub_model_id(saved_model_id.strip())
        )
        else None
    )
    normalized_model_id = _model_name(model_id)
    preferred = (
        (normalized_saved_model_id, normalized_model_id)
        if model_id_is_local
        else (normalized_model_id, normalized_saved_model_id)
    )

    identifiers: list[str] = []
    for identifier in preferred:
        if identifier and identifier not in identifiers:
            identifiers.append(identifier)
    return identifiers


def _architectures_match_model_type(config_dict: dict[str, Any], model_type: str) -> bool:
    """Return whether declared architectures all belong to ``model_type``."""
    if "architectures" not in config_dict:
        return True

    architectures = config_dict["architectures"]
    if (
        not isinstance(architectures, list)
        or not architectures
        or any(not isinstance(name, str) or not name for name in architectures)
    ):
        return False

    from transformers.models.auto import modeling_auto

    candidate_architectures: set[str] = set()
    for mapping_name, mapping in vars(modeling_auto).items():
        if (
            not mapping_name.startswith("MODEL")
            or not mapping_name.endswith("_MAPPING_NAMES")
            or not isinstance(mapping, Mapping)
        ):
            continue
        mapped_names = mapping.get(model_type)
        if isinstance(mapped_names, str):
            candidate_architectures.add(mapped_names)
        elif isinstance(mapped_names, (list, tuple)):
            candidate_architectures.update(name for name in mapped_names if isinstance(name, str))

    return all(name in candidate_architectures for name in architectures)


@overload
def load_hf_config(
    auto_config: Any,
    model_id: str,
    *,
    trust_remote_code: bool = False,
    raw_config_loader: _RawConfigLoader | None = None,
    return_unused_kwargs: Literal[True],
    **kwargs: Any,
) -> tuple[PretrainedConfig, dict[str, Any]]: ...


@overload
def load_hf_config(
    auto_config: Any,
    model_id: str,
    *,
    trust_remote_code: bool = False,
    raw_config_loader: _RawConfigLoader | None = None,
    return_unused_kwargs: Literal[False] = False,
    **kwargs: Any,
) -> PretrainedConfig: ...


@overload
def load_hf_config(
    auto_config: Any,
    model_id: str,
    *,
    trust_remote_code: bool = False,
    raw_config_loader: _RawConfigLoader | None = None,
    return_unused_kwargs: bool,
    **kwargs: Any,
) -> PretrainedConfig | tuple[PretrainedConfig, dict[str, Any]]: ...


def load_hf_config(
    auto_config: Any,
    model_id: str,
    *,
    trust_remote_code: bool = False,
    raw_config_loader: _RawConfigLoader | None = None,
    return_unused_kwargs: bool = False,
    **kwargs: Any,
) -> PretrainedConfig | tuple[PretrainedConfig, dict[str, Any]]:
    """Load an HF config, tolerating configs that omit a ``model_type`` key.

    Args:
        auto_config: The caller's own ``AutoConfig`` reference (its module-level
            name). Passing it in — rather than importing ``AutoConfig`` here —
            keeps each call site's ``AutoConfig`` monkeypatchable in tests.
        model_id: HuggingFace model ID or local path.
        trust_remote_code: Forwarded to the transformers loaders.
        raw_config_loader: Optional raw config retrieval callable. Defaults to
            :meth:`PretrainedConfig.get_config_dict`.
        **kwargs: Additional keyword arguments forwarded verbatim (e.g.
            ``revision``).

    Returns:
        The resolved config. Prefers ``auto_config.from_pretrained`` (the
        architecture-specific subclass); when the model omits ``model_type``,
        first tries identifier-based concrete config inference and otherwise
        returns a tagged generic config.
    """
    if trust_remote_code:
        from ..utils._security import _require_remote_code_execution_allowed

        _require_remote_code_execution_allowed()

    from transformers import PretrainedConfig

    load_kwargs = kwargs.copy()
    if return_unused_kwargs:
        load_kwargs["return_unused_kwargs"] = True
    use_auth_token = load_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        import warnings

        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of "
            "Transformers. Please use `token` instead.",
            FutureWarning,
            stacklevel=2,
        )
        if load_kwargs.get("token") is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the "
                "argument `token`."
            )
        load_kwargs["token"] = use_auth_token

    fallback_kwargs = load_kwargs.copy()
    fallback_kwargs["_from_auto"] = True
    fallback_kwargs["name_or_path"] = model_id
    fallback_kwargs.pop("code_revision", None)
    raw_loader = (
        PretrainedConfig.get_config_dict if raw_config_loader is None else raw_config_loader
    )
    config_dict, unused_kwargs = raw_loader(model_id, **fallback_kwargs)
    auto_map = config_dict.get("auto_map")
    has_remote_config = isinstance(auto_map, dict) and isinstance(auto_map.get("AutoConfig"), str)
    if "model_type" in config_dict or has_remote_config:
        try:
            return cast(
                "PretrainedConfig | tuple[PretrainedConfig, dict[str, Any]]",
                auto_config.from_pretrained(
                    model_id,
                    trust_remote_code=trust_remote_code,
                    **load_kwargs,
                ),
            )
        except Exception as exc:
            if not _is_strict_none_bool_validation_error(exc):
                raise
            fallback_result = _load_from_dict_with_strict_none_fallback(
                config_dict=config_dict,
                unused_kwargs=unused_kwargs,
                model_id=model_id,
                return_unused_kwargs=return_unused_kwargs,
            )
            if fallback_result is None:
                raise
            return fallback_result

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    for identifier in _fallback_identifiers(config_dict, model_id):
        identifier_lower = identifier.lower()
        candidates = sorted(
            (name for name in CONFIG_MAPPING if name.lower() in identifier_lower),
            key=lambda name: (-len(name), name),
        )
        if candidates and _architectures_match_model_type(config_dict, candidates[0]):
            return CONFIG_MAPPING[candidates[0]].from_dict(config_dict, **unused_kwargs)

    generic_config_dict = config_dict
    architectures = config_dict.get("architectures")
    if "architectures" in config_dict and (
        not isinstance(architectures, list)
        or any(not isinstance(name, str) for name in architectures)
    ):
        generic_config_dict = config_dict.copy()
        generic_config_dict.pop("architectures")

    generic_result = PretrainedConfig.from_dict(generic_config_dict, **unused_kwargs)
    if isinstance(generic_result, tuple):
        generic_config = cast("PretrainedConfig", generic_result[0])
        returned_unused_kwargs = cast("dict[str, Any]", generic_result[1])
        generic_config._winml_generic_fallback = True
        return generic_config, returned_unused_kwargs
    generic_result._winml_generic_fallback = True
    return generic_result
