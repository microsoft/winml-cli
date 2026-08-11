# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import pytest

from winml.modelkit.export.policy import (
    ExportCompatibilityConfig,
    ExportCompatibilityRule,
    ExportPolicyTarget,
    _rule_from_dict,
    export_policy_targets_for_request,
    resolve_export_compatibility,
)


def test_qnn_gpu_target_requires_eager_transformers_attention() -> None:
    cfg = resolve_export_compatibility([ExportPolicyTarget(ep="qnn", device="gpu")])

    assert cfg.transformers_attention == "eager"


def test_default_rules_load_from_json_catalog() -> None:
    from winml.modelkit.export import policy

    rules = policy.load_export_compatibility_rules()

    assert rules == (
        ExportCompatibilityRule(
            ep=None,
            device=None,
            compatibility=ExportCompatibilityConfig(transformers_attention="eager"),
            reason="Transformers SDPA-exported attention guard paths are not broadly portable.",
        ),
    )


def test_global_rule_forces_transformers_attention_for_non_qnn_target() -> None:
    cfg = resolve_export_compatibility(
        [ExportPolicyTarget(ep="DmlExecutionProvider", device="gpu")]
    )

    assert cfg.transformers_attention == "eager"


def test_no_targets_uses_supported_catalog_and_includes_qnn_requirement() -> None:
    cfg = resolve_export_compatibility()

    assert cfg.transformers_attention == "eager"


def test_export_policy_targets_for_request_keeps_portable_default_when_not_explicit() -> None:
    targets = export_policy_targets_for_request(
        ep="QNNExecutionProvider",
        device="gpu",
        target_was_explicit=False,
    )

    assert targets is None


def test_export_policy_targets_for_request_resolves_explicit_alias() -> None:
    targets = export_policy_targets_for_request(
        ep="qnn",
        device="gpu",
        target_was_explicit=True,
    )

    assert targets == (ExportPolicyTarget(ep="QNNExecutionProvider", device="gpu"),)


def test_conflicting_rules_raise_clear_error() -> None:
    rules = (
        ExportCompatibilityRule(
            ep="QNNExecutionProvider",
            device="gpu",
            compatibility=ExportCompatibilityConfig(transformers_attention="eager"),
            reason="first rule",
        ),
        ExportCompatibilityRule(
            ep="QNNExecutionProvider",
            device="gpu",
            compatibility=ExportCompatibilityConfig(transformers_attention="sdpa"),  # type: ignore[arg-type]
            reason="second rule",
        ),
    )

    with pytest.raises(ValueError, match="Conflicting export compatibility"):
        resolve_export_compatibility(
            [ExportPolicyTarget(ep="qnn", device="gpu")],
            rules=rules,
        )


def test_json_rule_rejects_unknown_ep() -> None:
    with pytest.raises(ValueError, match=r"match\.ep"):
        _rule_from_dict(
            {
                "match": {"ep": "TypoExecutionProvider", "device": None},
                "export": {"transformers_attention": "eager"},
                "reason": "typo should not create a dead rule",
            },
            index=0,
        )


def test_json_rule_rejects_unsupported_device_for_ep() -> None:
    with pytest.raises(ValueError, match="does not support device"):
        _rule_from_dict(
            {
                "match": {"ep": "CPUExecutionProvider", "device": "gpu"},
                "export": {"transformers_attention": "eager"},
                "reason": "device mismatch should not create a dead rule",
            },
            index=0,
        )
