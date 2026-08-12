# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import TYPE_CHECKING, Any

from ..utils.constants import EP_SUPPORTED_DEVICES, normalize_ep_name


if TYPE_CHECKING:
    from collections.abc import Sequence


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
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> ExportCompatibilityConfig:
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

    ep: str | None
    device: str | None
    compatibility: ExportCompatibilityConfig
    reason: str

    def matches(self, target: ExportPolicyTarget) -> bool:
        """Return True if this rule applies to the given target."""
        return (self.ep is None or target.ep == normalize_ep_name(self.ep)) and (
            self.device is None or target.device == self.device
        )


_RULES_RESOURCE = "compatibility_rules.json"
_SUPPORTED_POLICY_DEVICES_BY_EP: dict[str, frozenset[str]] = {
    ep: frozenset(devices) for ep, devices in EP_SUPPORTED_DEVICES.items()
}
_SUPPORTED_POLICY_DEVICES = frozenset(
    device for devices in _SUPPORTED_POLICY_DEVICES_BY_EP.values() for device in devices
)


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
    return _load_export_compatibility_rules()


@cache
def _load_export_compatibility_rules() -> tuple[ExportCompatibilityRule, ...]:
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
    targets: Sequence[ExportPolicyTarget] | None = None,
    *,
    rules: Sequence[ExportCompatibilityRule] | None = None,
) -> ExportCompatibilityConfig:
    """Resolve export compatibility for explicit targets or the portable catalog."""
    rules = load_export_compatibility_rules() if rules is None else rules
    resolved_targets = _catalog_targets() if targets is None else tuple(targets)

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
                transformers_attention_source = f"{rule.ep or '*'}/{rule.device or '*'}"
            elif transformers_attention != incoming:
                raise ValueError(
                    "Conflicting export compatibility for transformers_attention: "
                    f"{transformers_attention!r} from {transformers_attention_source} vs "
                    f"{incoming!r} from {rule.ep or '*'}/{rule.device or '*'}"
                )

    return ExportCompatibilityConfig(transformers_attention=transformers_attention)


def _catalog_targets() -> tuple[ExportPolicyTarget, ...]:
    from ..session.ep_device import EP_DEVICE_SPECS

    return tuple(ExportPolicyTarget(ep=spec.ep, device=spec.device) for spec in EP_DEVICE_SPECS)


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
    normalized_ep: str | None = None
    if ep is not None and (not isinstance(ep, str) or not ep):
        raise ValueError(
            f"{_RULES_RESOURCE} rules[{index}].match.ep must be a non-empty string or null"
        )
    if isinstance(ep, str):
        normalized_ep = normalize_ep_name(ep)
        if normalized_ep not in _SUPPORTED_POLICY_DEVICES_BY_EP:
            raise ValueError(
                f"{_RULES_RESOURCE} rules[{index}].match.ep must be a supported EP "
                f"or alias, got {ep!r}"
            )
    device = match.get("device")
    if device is not None and (not isinstance(device, str) or not device):
        raise ValueError(
            f"{_RULES_RESOURCE} rules[{index}].match.device must be a non-empty string or null"
        )
    normalized_device = device.lower() if isinstance(device, str) else None
    if normalized_device is not None and normalized_device not in _SUPPORTED_POLICY_DEVICES:
        raise ValueError(
            f"{_RULES_RESOURCE} rules[{index}].match.device must be one of "
            f"{sorted(_SUPPORTED_POLICY_DEVICES)}, got {device!r}"
        )
    if (
        normalized_ep is not None
        and normalized_device is not None
        and normalized_device not in _SUPPORTED_POLICY_DEVICES_BY_EP[normalized_ep]
    ):
        raise ValueError(
            f"{_RULES_RESOURCE} rules[{index}].match.ep {normalized_ep!r} "
            f"does not support device {normalized_device!r}"
        )

    export = data.get("export")
    compatibility = ExportCompatibilityConfig.from_dict(export)
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason:
        raise ValueError(f"{_RULES_RESOURCE} rules[{index}].reason must be a non-empty string")

    return ExportCompatibilityRule(
        ep=normalized_ep,
        device=normalized_device,
        compatibility=compatibility,
        reason=reason,
    )
