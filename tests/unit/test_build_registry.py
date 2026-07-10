# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _load_build_registry_module() -> Any:
    module_path = Path(__file__).parents[2] / "scripts" / "e2e_eval" / "build_registry.py"
    spec = importlib.util.spec_from_file_location("build_registry_under_test", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_hf_api_model_id_keeps_regular_model_id() -> None:
    build_registry = _load_build_registry_module()

    assert build_registry.get_hf_api_model_id("org/model") == "org/model"


def test_get_hf_api_model_id_uses_repo_for_nested_onnx_ref() -> None:
    build_registry = _load_build_registry_module()

    assert build_registry.get_hf_api_model_id("org/model/path/to/inference.onnx") == "org/model"


def test_get_model_metadata_uses_repo_id_for_nested_onnx_ref(monkeypatch) -> None:
    build_registry = _load_build_registry_module()
    seen_model_ids: list[str] = []

    def fake_model_info(model_id: str) -> SimpleNamespace:
        seen_model_ids.append(model_id)
        return SimpleNamespace(last_modified=None, downloads=123, pipeline_tag="image-to-text")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(model_info=fake_model_info),
    )

    metadata = build_registry.get_model_metadata("org/model/path/to/inference.onnx")

    assert seen_model_ids == ["org/model"]
    assert metadata["downloads"] == 123


def test_build_registry_recheck_downloads_refreshes_preserved_entries(monkeypatch) -> None:
    build_registry = _load_build_registry_module()
    existing_entries = [
        {
            "hf_id": "org/model-a",
            "task": "image-classification",
            "model_type": "vit",
            "group": "P0",
            "priority": "P0",
            "downloads": 1,
            "last_update_time": None,
        },
        {
            "hf_id": "org/model-b/path/to/model.onnx",
            "task": "image-classification",
            "model_type": "bert",
            "group": "P0",
            "priority": "P0",
            "downloads": 2,
            "last_update_time": None,
        },
    ]

    def fake_get_model_metadata(hf_id: str) -> dict[str, Any]:
        return {
            "last_modified": None,
            "downloads": {
                "org/model-a": 10,
                "org/model-b/path/to/model.onnx": 20,
            }[hf_id],
            "pipeline_tag": "",
        }

    monkeypatch.setattr(build_registry, "get_model_metadata", fake_get_model_metadata)

    entries = build_registry.build_registry(
        tasks=["image-classification"],
        top_n=0,
        existing_entries=existing_entries,
        recheck_downloads=True,
    )

    downloads = {e["hf_id"]: e["downloads"] for e in entries}
    assert downloads == {
        "org/model-a": 10,
        "org/model-b/path/to/model.onnx": 20,
    }
    assert {e["hf_id"]: e["order"] for e in entries} == {
        "org/model-b/path/to/model.onnx": 1,
        "org/model-a": 2,
    }
