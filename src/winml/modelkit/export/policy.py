# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any, Literal

from ..utils.constants import normalize_ep_name


if TYPE_CHECKING:
    from collections.abc import Sequence

TransformersAttentionPolicy = Literal["eager"]


@dataclass(frozen=True)
class ExportCompatibilityConfig:
    """Resolved export-time compatibility knobs."""

    transformers_attention: str | None = None

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.transformers_attention is not None

    def to_dict(self) -> dict[str, str]:
        """Serialize resolved compatibility knobs to a dict."""
        result: dict[str, str] = {}
        if self.transformers_attention is not None:
            result["transformers_attention"] = self.transformers_attention
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ExportCompatibilityConfig:
        """Deserialize compatibility config from a dict (or None)."""
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise TypeError(f"export.compatibility must be an object, got {type(data).__name__}")
        unknown = set(data) - {"transformers_attention"}
        if unknown:
            raise ValueError(f"Unknown export.compatibility field(s): {sorted(unknown)}")
        attention = data.get("transformers_attention")
        if attention is not None and attention != "eager":
            raise ValueError(
                "export.compatibility.transformers_attention must be 'eager' or null, "
                f"got {attention!r}"
            )
        return cls(transformers_attention=attention)


@dataclass(frozen=True)
class ExportPolicyTarget:
    """One EP/device target used by export compatibility policy."""

    ep: str
    device: str

    def __post_init__(self) -> None:
        normalized_ep = normalize_ep_name(self.ep)
        object.__setattr__(self, "ep", normalized_ep if normalized_ep is not None else self.ep)
        object.__setattr__(self, "device", self.device.lower())


@dataclass(frozen=True)
class ExportCompatibilityRule:
    """One EP/device compatibility rule."""

    ep: str
    device: str | None
    compatibility: ExportCompatibilityConfig
    reason: str

    def matches(self, target: ExportPolicyTarget) -> bool:
        """Return True if this rule applies to the given target."""
        return target.ep == normalize_ep_name(self.ep) and (
            self.device is None or target.device == self.device
        )


_RULES_RESOURCE = "compatibility_rules.json"


def export_policy_targets_for_request(
    *,
    ep: str | None,
    device: str | None,
    target_was_explicit: bool,
) -> tuple[ExportPolicyTarget, ...] | None:
    """Return explicit policy targets, or None for the portable catalog default."""
    if not target_was_explicit:
        return None

    from ..session import EPDeviceTarget, resolve_device

    resolved = resolve_device(EPDeviceTarget(ep=ep or "auto", device=(device or "auto").lower()))
    return (ExportPolicyTarget(ep=resolved.ep, device=resolved.device),)


def load_export_compatibility_rules() -> tuple[ExportCompatibilityRule, ...]:
    """Load built-in export compatibility rules from package JSON."""
    data = json.loads(resources.files(__package__).joinpath(_RULES_RESOURCE).read_text())
    if data.get("schema_version") != 1:
        raise ValueError(
            f"{_RULES_RESOURCE} schema_version must be 1, got {data.get('schema_version')!r}"
        )
    rules = data.get("rules")
    if not isinstance(rules, list):
        raise TypeError(f"{_RULES_RESOURCE} must contain a 'rules' array")
    return tuple(_rule_from_dict(rule, index=index) for index, rule in enumerate(rules))


def resolve_export_compatibility(
    targets: Sequence[object] | None = None,
    *,
    rules: Sequence[ExportCompatibilityRule] | None = None,
) -> ExportCompatibilityConfig:
    """Resolve export compatibility for explicit targets or the portable catalog."""
    rules = EXPORT_COMPATIBILITY_RULES if rules is None else rules
    resolved_targets = (
        _catalog_targets() if targets is None else tuple(_coerce_target(t) for t in targets)
    )

    transformers_attention: str | None = None
    transformers_attention_source: str | None = None

    for target in resolved_targets:
        for rule in rules:
            if not rule.matches(target):
                continue
            incoming = rule.compatibility.transformers_attention
            if incoming is None:
                continue
            if transformers_attention is None:
                transformers_attention = incoming
                transformers_attention_source = f"{rule.ep}/{rule.device or '*'}"
            elif transformers_attention != incoming:
                raise ValueError(
                    "Conflicting export compatibility for transformers_attention: "
                    f"{transformers_attention!r} from {transformers_attention_source} vs "
                    f"{incoming!r} from {rule.ep}/{rule.device or '*'}"
                )

    return ExportCompatibilityConfig(transformers_attention=transformers_attention)


def _catalog_targets() -> tuple[ExportPolicyTarget, ...]:
    from ..session.ep_device import EP_DEVICE_SPECS

    return tuple(ExportPolicyTarget(ep=spec.ep, device=spec.device) for spec in EP_DEVICE_SPECS)


def _coerce_target(target: object) -> ExportPolicyTarget:
    if isinstance(target, ExportPolicyTarget):
        return target
    ep = getattr(target, "ep", None)
    device = getattr(target, "device", None)
    if not isinstance(ep, str) or not isinstance(device, str):
        raise TypeError(
            "export policy target must expose string 'ep' and 'device' attributes, "
            f"got {type(target).__name__}"
        )
    return ExportPolicyTarget(ep=ep, device=device)


def _rule_from_dict(data: object, *, index: int) -> ExportCompatibilityRule:
    if not isinstance(data, dict):
        raise TypeError(f"{_RULES_RESOURCE} rules[{index}] must be an object")
    unknown = set(data) - {"match", "export", "reason"}
    if unknown:
        raise ValueError(
            f"{_RULES_RESOURCE} rules[{index}] has unknown field(s): {sorted(unknown)}"
        )

    match = data.get("match")
    if not isinstance(match, dict):
        raise TypeError(f"{_RULES_RESOURCE} rules[{index}].match must be an object")
    match_unknown = set(match) - {"ep", "device"}
    if match_unknown:
        raise ValueError(
            f"{_RULES_RESOURCE} rules[{index}].match has unknown field(s): {sorted(match_unknown)}"
        )
    ep = match.get("ep")
    if not isinstance(ep, str) or not ep:
        raise ValueError(f"{_RULES_RESOURCE} rules[{index}].match.ep must be a non-empty string")
    device = match.get("device")
    if device is not None and not isinstance(device, str):
        raise ValueError(f"{_RULES_RESOURCE} rules[{index}].match.device must be a string or null")

    export = data.get("export")
    compatibility = ExportCompatibilityConfig.from_dict(export)
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason:
        raise ValueError(f"{_RULES_RESOURCE} rules[{index}].reason must be a non-empty string")

    return ExportCompatibilityRule(
        ep=ep,
        device=device.lower() if isinstance(device, str) else None,
        compatibility=compatibility,
        reason=reason,
    )


EXPORT_COMPATIBILITY_RULES: tuple[ExportCompatibilityRule, ...] = load_export_compatibility_rules()
