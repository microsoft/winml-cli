# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the composite-model registry decorator invariant.

``register_composite_model`` is the single write point for
``COMPOSITE_MODEL_REGISTRY``. Enforcing that every registered class subclasses
``WinMLCompositeModel`` lets all readers (``resolve_composite`` /
``composite_pipeline_tasks`` / ``_composite_components_for_task``) trust the
registry without re-filtering by type.
"""

from __future__ import annotations

import pytest

from winml.modelkit.models.winml import WinMLCompositeModel
from winml.modelkit.models.winml.composite_model import (
    COMPOSITE_MODEL_REGISTRY,
    register_composite_model,
)


_FAKE_KEY = ("zzz-not-a-real-model-type", "zzz-not-a-real-task")


class TestRegisterCompositeModelInvariant:
    def test_rejects_non_composite_class(self):
        # A class that does not subclass WinMLCompositeModel must be refused, and
        # nothing should be added to the registry (the check precedes insertion).
        with pytest.raises(TypeError, match="must subclass WinMLCompositeModel"):

            @register_composite_model(*_FAKE_KEY)
            class _NotComposite:  # plain class — deliberately not a composite
                pass

        assert _FAKE_KEY not in COMPOSITE_MODEL_REGISTRY

    def test_accepts_composite_subclass(self):
        # A genuine WinMLCompositeModel subclass registers as expected.
        try:

            @register_composite_model(*_FAKE_KEY)
            class _TmpComposite(WinMLCompositeModel):
                pass

            assert COMPOSITE_MODEL_REGISTRY[_FAKE_KEY] is _TmpComposite
        finally:
            COMPOSITE_MODEL_REGISTRY.pop(_FAKE_KEY, None)

    def test_all_real_registrations_are_composites(self):
        # The live registry must satisfy the invariant the readers now rely on.
        import winml.modelkit.models.hf  # noqa: F401 — trigger registrations

        assert COMPOSITE_MODEL_REGISTRY  # populated
        for (model_type, task), cls in COMPOSITE_MODEL_REGISTRY.items():
            assert issubclass(cls, WinMLCompositeModel), (model_type, task, cls)
