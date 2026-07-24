# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression test for native composite routing in ``from_pretrained``.

When ``(model_type, task)`` is a registered composite, ``WinMLAutoModel``
derives the ``model_type`` from the HF config and dispatches directly to that
composite's ``from_pretrained`` (threading the resolved ``ep_device`` through),
rather than delegating to the base ``WinMLCompositeModel.from_pretrained``.  The
test uses a generic throwaway registry entry so no model-architecture name is
baked into the test logic.

Note: this branch resolves the composite from the HF-config ``model_type`` only
— there is no ``model_type=`` override parameter on ``from_pretrained`` (removed
in the ep_device redesign), so the variant-override short-circuit that origin/main
tested no longer exists on this branch.
"""

from __future__ import annotations

from types import SimpleNamespace

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


class TestCompositeNativeRouting:
    def test_native_model_type_selects_native_composite(
        self, monkeypatch: pytest.MonkeyPatch, routing_probes: dict[str, list[str]]
    ) -> None:
        # The factory derives the native model_type from the HF config; stub that
        # lookup so the native composite is selected. A pre-resolved ep_device is
        # threaded in so dispatch does not probe real hardware.
        import transformers

        class _Cfg:
            model_type = _NATIVE_TYPE

        monkeypatch.setattr(
            transformers.AutoConfig,
            "from_pretrained",
            classmethod(lambda cls, *a, **k: _Cfg()),
        )

        ep_device = SimpleNamespace(device=SimpleNamespace(device_type="CPU"))
        result = WinMLAutoModel.from_pretrained(
            "dummy/model", task=_SHARED_TASK, ep_device=ep_device
        )

        assert result == "NATIVE"
        assert routing_probes["native"] == ["dummy/model"]
        assert routing_probes["variant"] == []

    def test_explicit_model_type_routes_without_native_config_probe(
        self, monkeypatch: pytest.MonkeyPatch, routing_probes: dict[str, list[str]]
    ) -> None:
        """An explicit model_type selects its composite before AutoConfig probes."""
        import transformers

        monkeypatch.setattr(
            transformers.AutoConfig,
            "from_pretrained",
            classmethod(
                lambda *_args, **_kwargs: pytest.fail(
                    "AutoConfig should not run when model_type is explicit"
                )
            ),
        )
        ep_device = SimpleNamespace(device=SimpleNamespace(device_type="CPU"))

        result = WinMLAutoModel.from_pretrained(
            "dummy/model",
            task=_SHARED_TASK,
            model_type=_VARIANT_TYPE,
            ep_device=ep_device,
        )

        assert result == "VARIANT"
        assert routing_probes["native"] == []
        assert routing_probes["variant"] == ["dummy/model"]
