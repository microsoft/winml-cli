# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression tests for composite routing under an explicit ``model_type``.

``WinMLAutoModel.from_pretrained`` may be given an explicit ``model_type`` to
select a *variant* composite that shares a task with the model's native type
(e.g. a transformer-only surgical export vs. the full architecture).  The
factory must resolve and call that variant's composite class directly.

Previously it computed the override but then delegated to the base
``WinMLCompositeModel.from_pretrained``, which re-derives the *native*
``model_type`` from the HF config and silently drops the override — building the
wrong (native) composite.  These tests lock the corrected routing using generic
throwaway registry entries so no model-architecture name is baked into the test
logic.
"""

from __future__ import annotations

import pytest

from winml.modelkit.models import WinMLAutoModel
from winml.modelkit.models.winml.composite_model import COMPOSITE_MODEL_REGISTRY


_SHARED_TASK = "text-generation"
_NATIVE_TYPE = "unit-native-type"
_VARIANT_TYPE = "unit-variant-type"


@pytest.fixture
def routing_probes(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Register two composites sharing a task but keyed on different model_types.

    Each records the resolved class name when its ``from_pretrained`` runs, so a
    test can assert *which* composite the factory dispatched to.
    """
    calls: dict[str, list[str]] = {"native": [], "variant": []}

    class _NativeComposite:
        @classmethod
        def from_pretrained(cls, model_id, task, **kwargs):
            calls["native"].append(model_id)
            return "NATIVE"

    class _VariantComposite:
        @classmethod
        def from_pretrained(cls, model_id, task, **kwargs):
            calls["variant"].append(model_id)
            return "VARIANT"

    monkeypatch.setitem(COMPOSITE_MODEL_REGISTRY, (_NATIVE_TYPE, _SHARED_TASK), _NativeComposite)
    monkeypatch.setitem(COMPOSITE_MODEL_REGISTRY, (_VARIANT_TYPE, _SHARED_TASK), _VariantComposite)
    return calls


class TestCompositeModelTypeOverrideRouting:
    def test_explicit_override_selects_variant_composite(
        self, routing_probes: dict[str, list[str]]
    ) -> None:
        # An explicit model_type resolves the composite directly (no HF config
        # lookup), so a dummy id never touches the network.
        result = WinMLAutoModel.from_pretrained(
            "dummy/model", task=_SHARED_TASK, model_type=_VARIANT_TYPE
        )

        assert result == "VARIANT"
        assert routing_probes["variant"] == ["dummy/model"]
        assert routing_probes["native"] == []

    def test_native_model_type_selects_native_composite(
        self, monkeypatch: pytest.MonkeyPatch, routing_probes: dict[str, list[str]]
    ) -> None:
        # Without an override the factory derives the native model_type from the
        # HF config; stub that lookup so the native composite is selected.
        import transformers

        class _Cfg:
            model_type = _NATIVE_TYPE

        monkeypatch.setattr(
            transformers.AutoConfig,
            "from_pretrained",
            classmethod(lambda cls, *a, **k: _Cfg()),
        )

        result = WinMLAutoModel.from_pretrained("dummy/model", task=_SHARED_TASK)

        assert result == "NATIVE"
        assert routing_probes["native"] == ["dummy/model"]
        assert routing_probes["variant"] == []
