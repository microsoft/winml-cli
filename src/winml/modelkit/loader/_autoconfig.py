# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""HF config loading with tolerance for model_type-less configs.

transformers>=5 dropped the lenient fallback that let ``AutoConfig.from_pretrained``
load a config lacking a ``model_type`` key (older Hub models such as
``prajjwal1/bert-tiny``); it now raises ``ValueError: Unrecognized model ...``.
:func:`load_hf_config` first applies the former identifier-based inference to a
trusted model-name segment, then returns a tagged generic config when no
concrete architecture can be inferred safely.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast, overload


if TYPE_CHECKING:
    from transformers import PretrainedConfig


_RawConfigLoader: TypeAlias = Callable[..., tuple[dict[str, Any], dict[str, Any]]]


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
    from transformers import PretrainedConfig, __version__

    load_kwargs = kwargs.copy()
    if return_unused_kwargs:
        load_kwargs["return_unused_kwargs"] = True
    is_transformers4 = __version__.startswith("4.")
    if is_transformers4:
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
        return cast(
            "PretrainedConfig | tuple[PretrainedConfig, dict[str, Any]]",
            auto_config.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                **load_kwargs,
            ),
        )

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
    if return_unused_kwargs:
        generic_config, returned_unused_kwargs = generic_result
        generic_config._winml_generic_fallback = True
        return generic_config, returned_unused_kwargs
    generic_result._winml_generic_fallback = True
    return generic_result
