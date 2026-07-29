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
    export_policy_targets_for_request,
    resolve_export_compatibility,
)


def test_qnn_gpu_target_requires_eager_transformers_attention() -> None:
    cfg = resolve_export_compatibility([ExportPolicyTarget(ep="qnn", device="gpu")])

    assert cfg.transformers_attention == "eager"


def test_non_qnn_target_does_not_force_transformers_attention() -> None:
    cfg = resolve_export_compatibility(
        [ExportPolicyTarget(ep="DmlExecutionProvider", device="gpu")]
    )

    assert cfg.transformers_attention is None
    assert cfg.to_dict() == {}


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
