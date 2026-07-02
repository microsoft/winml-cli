# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for ``scripts/e2e_eval/utils/recipes.py``.

The script package is not importable as ``winml.*``; load ``recipes.py`` via
``importlib`` (it has no intra-package relative imports, so no stubbing is
needed) and exercise the discovery + filename-parsing helpers against both
synthetic dirs and the real ``examples/recipes`` tree.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_recipes():
    """Load scripts/e2e_eval/utils/recipes.py as a standalone module."""
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "e2e_eval" / "utils" / "recipes.py"
    spec = importlib.util.spec_from_file_location("_e2e_recipes", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_e2e_recipes"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def recipes():
    return _load_recipes()


@pytest.fixture(scope="module")
def recipes_dir():
    return Path(__file__).resolve().parents[3] / "examples" / "recipes"


class TestModelSlug:
    def test_single_slash_replaced(self, recipes):
        assert recipes.model_slug("microsoft/resnet-50") == "microsoft_resnet-50"

    def test_only_first_slash_replaced(self, recipes):
        # org/name is the norm; guard against a stray extra slash mangling the name.
        assert recipes.model_slug("a/b/c") == "a_b/c"

    def test_no_slash_unchanged(self, recipes):
        assert recipes.model_slug("bert-base") == "bert-base"


class TestSplitConfigStem:
    def test_single_config(self, recipes):
        stem, role = recipes.split_config_stem(Path("image-classification_fp16_config.json"))
        assert stem == "image-classification_fp16"
        assert role is None

    def test_composite_config(self, recipes):
        stem, role = recipes.split_config_stem(Path("image-to-text_fp16_config_encoder.json"))
        assert stem == "image-to-text_fp16"
        assert role == "encoder"

    def test_hyphenated_role(self, recipes):
        stem, role = recipes.split_config_stem(
            Path("zero-shot-image-classification_w8a16_config_image-encoder.json")
        )
        assert stem == "zero-shot-image-classification_w8a16"
        assert role == "image-encoder"


class TestSplitTaskPrecision:
    @pytest.mark.parametrize(
        ("group_stem", "task", "precision"),
        [
            ("image-classification_fp16", "image-classification", "fp16"),
            ("image-classification_w8a16", "image-classification", "w8a16"),
            ("question-answering_w8a8", "question-answering", "w8a8"),
            ("zero-shot-image-classification_fp16", "zero-shot-image-classification", "fp16"),
        ],
    )
    def test_known_precisions(self, recipes, group_stem, task, precision):
        assert recipes.split_task_precision(group_stem) == (task, precision)

    def test_unknown_precision_left_intact(self, recipes):
        # A trailing token that is not a known precision must NOT be stripped,
        # so scope never silently changes on a malformed name.
        assert recipes.split_task_precision("image-classification_int4") == (
            "image-classification_int4",
            None,
        )


class TestDiscoverRecipeVariants:
    def test_missing_model_dir_returns_empty(self, recipes, tmp_path):
        assert recipes.discover_recipe_variants(tmp_path, "no/such-model", "x") == []

    def test_single_model_two_precisions(self, recipes, tmp_path):
        model_dir = tmp_path / "microsoft_resnet-50"
        model_dir.mkdir()
        (model_dir / "image-classification_fp16_config.json").write_text("{}")
        (model_dir / "image-classification_w8a16_config.json").write_text("{}")

        variants = recipes.discover_recipe_variants(
            tmp_path, "microsoft/resnet-50", "image-classification"
        )
        assert [v.precision for v in variants] == ["fp16", "w8a16"]
        for v in variants:
            assert len(v.components) == 1
            assert v.components[0].role is None
            assert v.is_composite is False

    def test_composite_components_ordered(self, recipes, tmp_path):
        model_dir = tmp_path / "openai_clip-vit-large-patch14"
        model_dir.mkdir()
        # Write text-encoder first so we can prove ordering is by role, not file order.
        text_cfg = "zero-shot-image-classification_fp16_config_text-encoder.json"
        image_cfg = "zero-shot-image-classification_fp16_config_image-encoder.json"
        (model_dir / text_cfg).write_text("{}")
        (model_dir / image_cfg).write_text("{}")

        variants = recipes.discover_recipe_variants(
            tmp_path, "openai/clip-vit-large-patch14", "zero-shot-image-classification"
        )
        assert len(variants) == 1
        variant = variants[0]
        assert variant.is_composite is True
        assert variant.roles == ["image-encoder", "text-encoder"]

    def test_other_task_in_same_dir_excluded(self, recipes, tmp_path):
        model_dir = tmp_path / "intel_bert-base-uncased-mrpc"
        model_dir.mkdir()
        (model_dir / "feature-extraction_fp16_config.json").write_text("{}")
        (model_dir / "text-classification_fp16_config.json").write_text("{}")

        variants = recipes.discover_recipe_variants(
            tmp_path, "Intel/bert-base-uncased-mrpc", "text-classification"
        )
        assert len(variants) == 1
        assert variants[0].components[0].path.name == "text-classification_fp16_config.json"

    def test_precision_order_is_canonical(self, recipes, tmp_path):
        model_dir = tmp_path / "m"
        model_dir.mkdir()
        # Create in reverse precision order to prove output is canonical (fp16 first).
        (model_dir / "t_w8a16_config.json").write_text("{}")
        (model_dir / "t_fp16_config.json").write_text("{}")
        variants = recipes.discover_recipe_variants(tmp_path, "m", "t")
        assert [v.precision for v in variants] == ["fp16", "w8a16"]


class TestDiscoverAgainstRealRecipes:
    """Smoke-test against the checked-in examples/recipes tree."""

    def test_resnet50_has_two_single_variants(self, recipes, recipes_dir):
        if not recipes_dir.is_dir():
            pytest.skip("examples/recipes not present")
        variants = recipes.discover_recipe_variants(
            recipes_dir, "microsoft/resnet-50", "image-classification"
        )
        precisions = {v.precision for v in variants}
        assert {"fp16", "w8a16"} <= precisions
        for v in variants:
            assert v.is_composite is False

    def test_trocr_is_composite(self, recipes, recipes_dir):
        if not recipes_dir.is_dir():
            pytest.skip("examples/recipes not present")
        variants = recipes.discover_recipe_variants(
            recipes_dir, "microsoft/trocr-base-printed", "image-to-text"
        )
        assert variants, "expected at least one trocr recipe variant"
        for v in variants:
            assert v.is_composite is True
            assert v.roles == ["encoder", "decoder"]
