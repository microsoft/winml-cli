# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for I/O spec structure and new architecture coverage.

Extends export test coverage with:
- OnnxConfig retrieval for architectures NOT in test_io.py
  (albert, distilbert, gpt2, convnext, detr)
- Cross-architecture structural validation of resolve_io_specs output
- Input name verification for new architectures
- Export-specific task synonym mapping via map_task_synonym

See also: tests/export/test_io.py (covers bert, resnet, vit, clip_vision, clip_text)
"""

from __future__ import annotations

from typing import ClassVar

import pytest

# Trigger OnnxConfig registration with TasksManager
import winml.modelkit.models  # noqa: F401
from winml.modelkit.export.io import (
    _get_onnx_config,
    _map_task_synonym,
    resolve_io_specs,
)


# =============================================================================
# Class 1: OnnxConfig retrieval for NEW architectures
# =============================================================================


class TestGetOnnxConfigNewArchitectures:
    """OnnxConfig retrieval for architectures NOT covered by test_io.py.

    test_io.py already covers: bert, resnet, vit, clip_vision_model, clip_text_model.
    This class tests: albert, distilbert, gpt2, convnext, detr.
    """

    @pytest.mark.parametrize(
        "model_type,task,config_fixture",
        [
            ("albert", "fill-mask", "albert_config"),
            ("distilbert", "text-classification", "distilbert_config"),
            ("gpt2", "text-generation", "gpt2_config"),
            ("convnext", "image-classification", "convnext_config"),
            ("detr", "object-detection", "detr_config"),
        ],
        ids=["albert", "distilbert", "gpt2", "convnext", "detr"],
    )
    def test_get_onnx_config_new_architectures(
        self, model_type: str, task: str, config_fixture: str, request: pytest.FixtureRequest
    ) -> None:
        """OnnxConfig retrieval for architectures not in existing test_io.py."""
        hf_config = request.getfixturevalue(config_fixture)
        onnx_config = _get_onnx_config(model_type, task, hf_config)

        assert hasattr(onnx_config, "inputs")
        assert hasattr(onnx_config, "outputs")
        assert len(onnx_config.inputs) > 0
        assert len(onnx_config.outputs) > 0


# =============================================================================
# Class 2: Structural validation across architectures
# =============================================================================

_STRUCTURE_PARAMS = [
    ("bert", "fill-mask", "bert_config"),
    ("gpt2", "text-generation", "gpt2_config"),
    ("resnet", "image-classification", "resnet_config"),
    ("vit", "image-classification", "vit_config"),
    ("detr", "object-detection", "detr_config"),
]

_STRUCTURE_IDS = ["bert", "gpt2", "resnet", "vit", "detr"]


@pytest.mark.parametrize(
    "model_type,task,config_fixture",
    _STRUCTURE_PARAMS,
    ids=_STRUCTURE_IDS,
)
class TestIOSpecsStructure:
    """Validate spec dict structure is consistent across ALL architectures.

    Ensures resolve_io_specs returns a well-formed dict with all
    required keys, consistent lengths, valid types, and correct batch size.
    """

    REQUIRED_KEYS: ClassVar[set[str]] = {
        "inputs",
        "outputs",
        "input_names",
        "output_names",
        "dynamic_axes",
        "input_shapes",
        "input_dtypes",
        "value_ranges",
    }

    def test_has_all_required_keys(
        self, model_type: str, task: str, config_fixture: str, request: pytest.FixtureRequest
    ) -> None:
        """Spec dict contains exactly the required keys."""
        hf_config = request.getfixturevalue(config_fixture)
        specs = resolve_io_specs(model_type, task, hf_config)
        assert set(specs.keys()) == self.REQUIRED_KEYS

    def test_input_output_name_counts(
        self, model_type: str, task: str, config_fixture: str, request: pytest.FixtureRequest
    ) -> None:
        """input_names and output_names are non-empty lists."""
        hf_config = request.getfixturevalue(config_fixture)
        specs = resolve_io_specs(model_type, task, hf_config)
        assert isinstance(specs["input_names"], list)
        assert len(specs["input_names"]) > 0
        assert isinstance(specs["output_names"], list)
        assert len(specs["output_names"]) > 0

    def test_shapes_match_names(
        self, model_type: str, task: str, config_fixture: str, request: pytest.FixtureRequest
    ) -> None:
        """input_shapes and input_dtypes have same length as input_names."""
        hf_config = request.getfixturevalue(config_fixture)
        specs = resolve_io_specs(model_type, task, hf_config)
        assert len(specs["input_shapes"]) == len(specs["input_names"])
        assert len(specs["input_dtypes"]) == len(specs["input_names"])

    def test_shapes_are_valid_tuples(
        self, model_type: str, task: str, config_fixture: str, request: pytest.FixtureRequest
    ) -> None:
        """Each input_shape is a tuple of positive integers."""
        hf_config = request.getfixturevalue(config_fixture)
        specs = resolve_io_specs(model_type, task, hf_config)
        for shape in specs["input_shapes"]:
            assert isinstance(shape, tuple), f"Expected tuple, got {type(shape)}"
            assert all(isinstance(d, int) and d > 0 for d in shape), (
                f"Invalid shape dimensions: {shape}"
            )

    def test_dtypes_are_valid_strings(
        self, model_type: str, task: str, config_fixture: str, request: pytest.FixtureRequest
    ) -> None:
        """Each input_dtype is a recognized dtype string."""
        hf_config = request.getfixturevalue(config_fixture)
        specs = resolve_io_specs(model_type, task, hf_config)
        valid_dtypes = {"float32", "float16", "float64", "int64", "int32", "int8", "bool"}
        for dtype in specs["input_dtypes"]:
            assert dtype in valid_dtypes, f"Unexpected dtype: {dtype}"

    def test_batch_size_respected(
        self, model_type: str, task: str, config_fixture: str, request: pytest.FixtureRequest
    ) -> None:
        """batch_size=2 is reflected in the first dimension of all input shapes."""
        hf_config = request.getfixturevalue(config_fixture)
        specs = resolve_io_specs(model_type, task, hf_config, batch_size=2)
        for shape in specs["input_shapes"]:
            assert shape[0] == 2, f"batch_size=2 not reflected in shape {shape}"

    def test_dynamic_axes_reference_valid_names(
        self, model_type: str, task: str, config_fixture: str, request: pytest.FixtureRequest
    ) -> None:
        """All dynamic_axes keys reference actual input or output names."""
        hf_config = request.getfixturevalue(config_fixture)
        specs = resolve_io_specs(model_type, task, hf_config)
        all_names = set(specs["input_names"] + specs["output_names"])
        for name in specs["dynamic_axes"]:
            assert name in all_names, (
                f"dynamic_axes key '{name}' not in input/output names: {all_names}"
            )

    def test_value_ranges_is_dict(
        self, model_type: str, task: str, config_fixture: str, request: pytest.FixtureRequest
    ) -> None:
        """value_ranges is a dict with keys matching input_names."""
        hf_config = request.getfixturevalue(config_fixture)
        specs = resolve_io_specs(model_type, task, hf_config)
        ranges = specs["value_ranges"]
        assert isinstance(ranges, dict)
        for name in ranges:
            assert name in specs["input_names"], f"Value range for unknown input '{name}'"

    def test_value_ranges_are_numeric_tuples(
        self, model_type: str, task: str, config_fixture: str, request: pytest.FixtureRequest
    ) -> None:
        """Each value range is a (min, max) tuple with min <= max."""
        hf_config = request.getfixturevalue(config_fixture)
        specs = resolve_io_specs(model_type, task, hf_config)
        for name, (lo, hi) in specs["value_ranges"].items():
            assert isinstance(lo, (int, float)), f"{name}: min not numeric: {lo}"
            assert isinstance(hi, (int, float)), f"{name}: max not numeric: {hi}"
            assert lo <= hi, f"{name}: min={lo} > max={hi}"


# =============================================================================
# Class 3: Input name verification for NEW architectures
# =============================================================================


class TestIOSpecsInputNames:
    """Verify expected input_names for architectures not in test_io.py.

    test_io.py already covers bert, resnet, vit, clip_vision, clip_text.
    This validates the additional architectures.
    """

    @pytest.mark.parametrize(
        "model_type,task,config_fixture,expected_input_names",
        [
            (
                "albert",
                "fill-mask",
                "albert_config",
                ["input_ids", "attention_mask", "token_type_ids"],
            ),
            (
                "distilbert",
                "text-classification",
                "distilbert_config",
                ["input_ids", "attention_mask"],
            ),
            (
                "gpt2",
                "text-generation",
                "gpt2_config",
                ["input_ids", "attention_mask", "position_ids"],
            ),
            (
                "convnext",
                "image-classification",
                "convnext_config",
                ["pixel_values"],
            ),
            (
                "detr",
                "object-detection",
                "detr_config",
                ["pixel_values"],
            ),
        ],
        ids=["albert", "distilbert", "gpt2", "convnext", "detr"],
    )
    def test_io_specs_input_names_new_architectures(
        self,
        model_type: str,
        task: str,
        config_fixture: str,
        expected_input_names: list[str],
        request: pytest.FixtureRequest,
    ) -> None:
        """Verify input_names for architectures not in existing test_io.py."""
        hf_config = request.getfixturevalue(config_fixture)
        specs = resolve_io_specs(model_type, task, hf_config)
        assert specs["input_names"] == expected_input_names


# =============================================================================
# Class 4: Export-specific task synonym mapping
# =============================================================================


class TestMapTaskSynonymExport:
    """Verify export-specific task synonym mapping via map_task_synonym.

    Tests our TASK_SYNONYM_EXTENSIONS (next-sentence-prediction),
    Optimum's built-in synonyms (image-feature-extraction),
    identity passthrough for canonical tasks, and unknown tasks.
    """

    @pytest.mark.parametrize(
        "task,expected",
        [
            # Custom extension (TASK_SYNONYM_EXTENSIONS)
            ("next-sentence-prediction", "text-classification"),
            # Optimum built-in synonym (via map_from_synonym)
            ("image-feature-extraction", "feature-extraction"),
            # Identity passthrough - canonical tasks
            ("fill-mask", "fill-mask"),
            ("image-classification", "image-classification"),
            ("text-generation", "text-generation"),
            ("object-detection", "object-detection"),
            # Edge case: unknown task passes through unchanged
            ("custom-task-xyz", "custom-task-xyz"),
        ],
        ids=[
            "nsp-to-text-classification",
            "image-feat-to-feat",
            "fill-mask-passthrough",
            "image-cls-passthrough",
            "text-gen-passthrough",
            "object-det-passthrough",
            "unknown-passthrough",
        ],
    )
    def test_map_task_synonym(self, task: str, expected: str) -> None:
        """map_task_synonym returns the expected canonical task name."""
        assert _map_task_synonym(task) == expected


# =============================================================================
# Class 5: Value range content verification
# =============================================================================


class TestValueRangesContent:
    """Verify intercepted value ranges match expected for known architectures.

    Uses fixture config values (e.g., vocab_size=100) not pretrained values.
    """

    def test_bert_text_ranges(self, bert_config) -> None:
        """BERT text inputs: input_ids uses vocab_size, masks are binary."""
        specs = resolve_io_specs("bert", "fill-mask", bert_config)
        ranges = specs["value_ranges"]

        assert ranges["input_ids"] == (0, bert_config.vocab_size)
        assert ranges["attention_mask"] == (0, 2)
        assert ranges["token_type_ids"] == (0, 2)

    def test_resnet_vision_ranges(self, resnet_config) -> None:
        """ResNet vision input: pixel_values uses default float range."""
        specs = resolve_io_specs("resnet", "image-classification", resnet_config)
        ranges = specs["value_ranges"]

        assert ranges["pixel_values"] == (0, 1)

    def test_gpt2_text_ranges(self, gpt2_config) -> None:
        """GPT-2 text inputs: input_ids uses vocab_size, position_ids uses n_positions."""
        specs = resolve_io_specs("gpt2", "text-generation", gpt2_config)
        ranges = specs["value_ranges"]

        assert ranges["input_ids"] == (0, gpt2_config.vocab_size)
        assert ranges["attention_mask"] == (0, 2)
