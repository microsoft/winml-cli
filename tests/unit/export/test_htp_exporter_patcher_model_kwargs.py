# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression tests for `HTPExporter._get_optimum_patcher` model_kwargs handling.

Some Optimum model patchers populate a mutable ``model_kwargs`` dict to inject
constant forward arguments at export time. ViTPose's MoE patcher, for example,
sets ``model_kwargs["dataset_index"]`` when ``num_experts > 1``. Optimum's
``patch_model_for_export`` defaults ``model_kwargs`` to ``None``, so such
patchers crash with ``TypeError: 'NoneType' object does not support item
assignment`` unless the caller passes an explicit dict.

This test pins the contract that ``_get_optimum_patcher`` passes an explicit
``model_kwargs={}`` so those patchers can populate it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch.nn as nn

from winml.modelkit.export.htp import HTPExporter


class _FakeConfig:
    """Minimal HF-style config exposing the model_type the patcher checks."""

    model_type = "vitpose"


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeConfig()


class TestGetOptimumPatcherModelKwargs:
    """_get_optimum_patcher must pass an explicit mutable model_kwargs dict."""

    def test_patch_model_for_export_receives_explicit_dict(self) -> None:
        """The patcher call must pass ``model_kwargs={}`` (not the None default).

        We patch the TasksManager lookup to return a fake config constructor
        whose ``patch_model_for_export`` records the ``model_kwargs`` it
        receives. A non-None dict lets MoE patchers populate forward arguments
        without crashing.
        """
        captured: dict[str, object] = {}

        fake_onnx_config = MagicMock()

        def record_patch(model, model_kwargs=None):
            captured["model_kwargs"] = model_kwargs
            return MagicMock()

        fake_onnx_config.patch_model_for_export.side_effect = record_patch

        def fake_ctor(*args: object, **kwargs: object):
            return fake_onnx_config

        with patch(
            "optimum.exporters.tasks.TasksManager.get_exporter_config_constructor",
            return_value=fake_ctor,
        ):
            HTPExporter._get_optimum_patcher(_FakeModel(), task="keypoint-detection")

        assert captured.get("model_kwargs") == {}, (
            "Expected _get_optimum_patcher to pass an explicit model_kwargs={} "
            f"to patch_model_for_export, got {captured.get('model_kwargs')!r}. "
            "MoE patchers (e.g. ViTPose dataset_index) need a mutable dict."
        )
