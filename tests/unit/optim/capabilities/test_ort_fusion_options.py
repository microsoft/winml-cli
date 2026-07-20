# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Registry tests for ORT ``FusionOptions`` toggles newly exposed as
``BoolCapability`` defs.

Six ORT ``FusionOptions`` flags — ``enable_packed_qkv``, ``enable_packed_kv``,
``enable_group_norm``, ``enable_skip_group_norm``, ``enable_nhwc_conv`` and
``enable_bias_add`` — were previously reachable only via internal
``ORTFusionPipeConfig`` fields. They are now surfaced through the capability
registry so any model config (or CLI user) can opt in via the standard
``--enable-<name>`` / ``**kwargs`` flow.

These tests cover the *registry* wiring:

* Each capability def has the expected metadata (name / python_name /
  default / ``depends_on``).
* Each capability is registered on ``ORTFusionPipe.capabilities``.
* Passing the ``python_name`` kwarg to ``ORTFusionPipe.build_config``
  flips the corresponding ``ORTFusionPipeConfig.enable_*`` field.
* All six defaults are ``False`` (QNN-compatible baseline, parity with
  the other fusion toggles).

They do *not* exercise actual graph rewrites — that is covered by the
capability-isolation tests in the sibling ``test_capability_isolation.py``
against real ONNX fixtures.
"""

from __future__ import annotations

import pytest

from winml.modelkit.optim.capabilities import attention, conv, layernorm, misc
from winml.modelkit.optim.pipes import ORTFusionPipe, ORTFusionPipeConfig


# =============================================================================
# BoolCapability defs — metadata
# =============================================================================


class TestOrtFusionCapabilityDefs:
    """The six newly-exposed capabilities should be defined in the right
    modules with consistent metadata."""

    def test_packed_qkv_fusion_def(self) -> None:
        cap = attention.PACKED_QKV_FUSION
        assert cap.name == "packed-qkv-fusion"
        assert cap.python_name == "packed_qkv_fusion"
        assert cap.default is False
        # Packed-QKV rewrites the input of a fused Attention/MHA node, so
        # the base attention fusion must be enabled first.
        assert "attention-fusion" in cap.depends_on

    def test_packed_kv_fusion_def(self) -> None:
        cap = attention.PACKED_KV_FUSION
        assert cap.name == "packed-kv-fusion"
        assert cap.python_name == "packed_kv_fusion"
        assert cap.default is False
        # Same rationale as packed-QKV: modifies the fused-attention input.
        assert "attention-fusion" in cap.depends_on

    def test_group_norm_fusion_def(self) -> None:
        cap = layernorm.GROUP_NORM_FUSION
        assert cap.name == "group-norm-fusion"
        assert cap.python_name == "group_norm_fusion"
        assert cap.default is False

    def test_skip_group_norm_fusion_def(self) -> None:
        cap = layernorm.SKIP_GROUP_NORM_FUSION
        assert cap.name == "skip-group-norm-fusion"
        assert cap.python_name == "skip_group_norm_fusion"
        assert cap.default is False
        # SkipGroupNorm folds an Add(residual) into GroupNorm; requires
        # the base GroupNorm fusion.
        assert "group-norm-fusion" in cap.depends_on

    def test_nhwc_conv_def(self) -> None:
        cap = conv.NHWC_CONV
        assert cap.name == "nhwc-conv"
        assert cap.python_name == "nhwc_conv"
        assert cap.default is False

    def test_bias_add_fusion_def(self) -> None:
        cap = misc.BIAS_ADD_FUSION
        assert cap.name == "bias-add-fusion"
        assert cap.python_name == "bias_add_fusion"
        assert cap.default is False


# =============================================================================
# Capability registration — the six new caps are wired into ORTFusionPipe
# =============================================================================


@pytest.mark.parametrize(
    "cap_name",
    [
        "packed-qkv-fusion",
        "packed-kv-fusion",
        "group-norm-fusion",
        "skip-group-norm-fusion",
        "nhwc-conv",
        "bias-add-fusion",
    ],
)
def test_capability_registered_on_fusion_pipe(cap_name: str) -> None:
    """Each newly-exposed capability should appear in ``ORTFusionPipe.capabilities``."""
    assert cap_name in ORTFusionPipe.capabilities


# =============================================================================
# build_config kwarg propagation
# =============================================================================


@pytest.mark.parametrize(
    ("kwarg", "config_attr"),
    [
        ("packed_qkv_fusion", "enable_packed_qkv"),
        ("packed_kv_fusion", "enable_packed_kv"),
        ("group_norm_fusion", "enable_group_norm"),
        ("skip_group_norm_fusion", "enable_skip_group_norm"),
        ("nhwc_conv", "enable_nhwc_conv"),
        ("bias_add_fusion", "enable_bias_add"),
    ],
)
def test_build_config_propagates_kwarg(kwarg: str, config_attr: str) -> None:
    """``ORTFusionPipe.build_config(**{kwarg: True})`` should flip the
    corresponding ``ORTFusionPipeConfig`` field.

    Before this change these kwargs were silently ignored because the
    capability defs that drive ``build_config``'s lookup didn't exist.
    """
    cfg = ORTFusionPipe.build_config(**{kwarg: True})
    assert isinstance(cfg, ORTFusionPipeConfig)
    assert getattr(cfg, config_attr) is True


# =============================================================================
# Defaults
# =============================================================================


def test_build_config_defaults_all_new_fusions_off() -> None:
    """Without explicit kwargs every newly-exposed fusion stays off — parity
    with the other fusion toggles (QNN-compatible baseline)."""
    cfg = ORTFusionPipe.build_config()
    assert cfg.enable_packed_qkv is False
    assert cfg.enable_packed_kv is False
    assert cfg.enable_group_norm is False
    assert cfg.enable_skip_group_norm is False
    assert cfg.enable_nhwc_conv is False
    assert cfg.enable_bias_add is False
