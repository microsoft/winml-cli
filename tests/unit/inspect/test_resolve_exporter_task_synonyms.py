# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression tests for `inspect.resolver.resolve_exporter` task-synonym handling.

Optimum's ``TasksManager.get_exporter_config_constructor`` only accepts
canonical task names (e.g. ``feature-extraction``); HuggingFace pipeline
aliases (e.g. ``image-feature-extraction``, ``sentence-similarity``) raise
``KeyError``/``ValueError`` and cause the resolver to silently fall through
to ``SupportLevel.UNSUPPORTED``.

These tests pin the contract that ``resolve_exporter`` normalises HF
aliases via ``map_task_synonym`` before the TasksManager lookup.

Regression for https://github.com/microsoft/winml-cli/issues/782.
"""

from __future__ import annotations

from winml.modelkit.inspect.resolver import resolve_exporter
from winml.modelkit.inspect.types import SupportLevel


class TestResolveExporterTaskSynonyms:
    """resolve_exporter must accept HF-alias tasks and normalise them."""

    def test_hf_alias_image_feature_extraction_resolves_for_dinov2(self) -> None:
        """HF alias 'image-feature-extraction' must resolve to a TasksManager config.

        Without normalisation, TasksManager raises and the resolver returns
        ``SupportLevel.UNSUPPORTED`` with ``onnx_config_class=None``.
        """
        info = resolve_exporter("dinov2", "image-feature-extraction", hf_config=None)

        assert info.support_level != SupportLevel.UNSUPPORTED, (
            "dinov2/image-feature-extraction must resolve via TasksManager; "
            "if this is UNSUPPORTED, the HF-alias task likely wasn't "
            "normalised before TasksManager.get_exporter_config_constructor."
        )
        assert info.onnx_config_source == "TasksManager", (
            f"Expected onnx_config_source='TasksManager', got {info.onnx_config_source!r}."
        )
        assert info.onnx_config_class is not None

    def test_canonical_feature_extraction_resolves_for_dinov2(self) -> None:
        """Control: canonical 'feature-extraction' resolves (no normalisation needed)."""
        info = resolve_exporter("dinov2", "feature-extraction", hf_config=None)

        assert info.support_level != SupportLevel.UNSUPPORTED
        assert info.onnx_config_source == "TasksManager"
        assert info.onnx_config_class is not None
