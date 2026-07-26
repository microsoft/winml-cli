# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for safe, provenance-preserving release-asset acquisition."""

from __future__ import annotations

import json
import shutil
import zipfile
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from onnx import TensorProto, external_data_helper, helper, load, numpy_helper, save, save_model

from winml.modelkit.loader import release_assets


if TYPE_CHECKING:
    from pathlib import Path


def _identity_model(path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "identity",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    save(helper.make_model(graph), path)


def test_safe_extract_zip_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.onnx", b"not a graph")

    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        release_assets.safe_extract_zip(archive, tmp_path / "out")

    assert not (tmp_path / "escape.onnx").exists()


def test_safe_extract_zip_rejects_duplicate_members(tmp_path: Path) -> None:
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("model.onnx", b"first")
        with pytest.warns(UserWarning):
            bundle.writestr("model.onnx", b"second")

    with pytest.raises(ValueError, match="duplicate member"):
        release_assets.safe_extract_zip(archive, tmp_path / "out")


def _external_model(path: Path) -> Path:
    import numpy as np

    weight = numpy_helper.from_array(np.ones((2, 2), dtype=np.float32), name="weight")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "external",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        [weight],
    )
    model = helper.make_model(graph)
    save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    return path.parent / "weights.data"


def test_external_data_must_be_present_and_colocated(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    sidecar = _external_model(model_path)

    assert release_assets.validate_onnx_external_data(model_path, tmp_path) == (sidecar,)

    sidecar.unlink()
    with pytest.raises(FileNotFoundError, match="missing or out-of-root"):
        release_assets.validate_onnx_external_data(model_path, tmp_path)


def test_external_data_rejects_traversal_location(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    _external_model(model_path)
    model = load(model_path, load_external_data=False)
    location = next(
        field for field in model.graph.initializer[0].external_data if field.key == "location"
    )
    location.value = "../weights.data"
    model_path.write_bytes(model.SerializeToString())

    with pytest.raises(ValueError, match="unsafe external-data location"):
        release_assets.validate_onnx_external_data(model_path, tmp_path)


def _attribute_external_model(path: Path, location: str, *, nested: bool = False) -> Path:
    import numpy as np

    tensor = numpy_helper.from_array(np.array([3.0], dtype=np.float32), name="attribute_value")
    raw_data = tensor.raw_data
    external_data_helper.set_external_data(
        tensor,
        location=location,
        offset=0,
        length=len(raw_data),
    )
    tensor.ClearField("raw_data")
    constant = helper.make_node("Constant", [], ["constant"], value=tensor)
    if nested:
        branch = helper.make_graph(
            [constant],
            "branch",
            [],
            [helper.make_tensor_value_info("constant", TensorProto.FLOAT, [1])],
        )
        other_branch = helper.make_graph(
            [helper.make_node("Identity", ["fallback"], ["constant"])],
            "other_branch",
            [helper.make_tensor_value_info("fallback", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("constant", TensorProto.FLOAT, [1])],
            [helper.make_tensor("fallback", TensorProto.FLOAT, [1], [0.0])],
        )
        nodes = [
            helper.make_node(
                "If",
                ["condition"],
                ["output"],
                then_branch=branch,
                else_branch=other_branch,
            )
        ]
        inputs = [helper.make_tensor_value_info("condition", TensorProto.BOOL, [])]
    else:
        nodes = [constant, helper.make_node("Identity", ["constant"], ["output"])]
        inputs = []
    graph = helper.make_graph(
        nodes,
        "attribute_external",
        inputs,
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    path.write_bytes(helper.make_model(graph).SerializeToString())
    sidecar = path.parent / location
    if ".." not in location:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_bytes(raw_data)
    return sidecar


def test_external_data_recurses_into_nested_graph_tensor_attributes(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    sidecar = _attribute_external_model(model_path, "attribute.data", nested=True)

    assert release_assets.validate_onnx_external_data(model_path, tmp_path) == (sidecar,)

    sidecar.unlink()
    with pytest.raises(FileNotFoundError, match="missing or out-of-root"):
        release_assets.validate_onnx_external_data(model_path, tmp_path)


def test_external_data_rejects_malicious_tensor_attribute_location(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    _attribute_external_model(model_path, "../attribute.data")

    with pytest.raises(ValueError, match="unsafe external-data location"):
        release_assets.validate_onnx_external_data(model_path, tmp_path)


def test_copy_release_contract_files_persists_metadata_and_provenance(tmp_path: Path) -> None:
    release_root = tmp_path / "extracted"
    graph_root = release_root / "bundle"
    graph_root.mkdir(parents=True)
    graph = graph_root / "encoder.onnx"
    graph.touch()
    (release_root / "winml_release_provenance.json").write_text(
        '{"resolved_revision":"abc"}', encoding="utf-8"
    )
    (graph_root / "metadata.json").write_text('{"model_files":{}}', encoding="utf-8")
    output = tmp_path / "built"

    release_assets.copy_release_contract_files(graph, output)

    assert json.loads((output / "winml_release_provenance.json").read_text()) == {
        "resolved_revision": "abc"
    }
    assert json.loads((output / "winml_release_metadata.json").read_text()) == {"model_files": {}}


def test_acquisition_records_pinned_provenance_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    manifest = tmp_path / "release_assets.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "1.2.3",
                "precisions": {
                    "float": {
                        "universal_assets": {
                            "onnx": {
                                "download_url": "https://assets.example/model-onnx-float.zip",
                                "tool_versions": {"onnx_runtime": "1.2.3"},
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source"
    source.mkdir()
    _external_model(source / "encoder.onnx")
    _identity_model(source / "decoder.onnx")
    (source / "metadata.json").write_text("{}", encoding="utf-8")
    archive = tmp_path / "asset.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in source.iterdir():
            bundle.write(path, f"bundle/{path.name}")

    info = SimpleNamespace(sha="a" * 40, pipeline_tag="image-segmentation")
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(model_info=lambda *args, **kwargs: info),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *args, **kwargs: str(manifest),
    )
    downloads: list[str] = []

    def copy_archive(url: str, destination: Path) -> None:
        downloads.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, destination)

    monkeypatch.setattr(release_assets, "_download_archive", copy_archive)

    result = release_assets.acquire_hf_release_asset(
        "org/model", revision="main", precision="fp16", cache_dir=tmp_path / "cache"
    )
    assert result is not None
    assert result.metadata_path is not None
    assert result.provenance["requested_revision"] == "main"
    assert result.provenance["resolved_revision"] == "a" * 40
    assert result.provenance["requested_precision"] == "fp16"
    assert result.provenance["source_precision"] == "float"
    assert result.provenance["archive_sha256"]
    assert result.provenance["manifest_sha256"]
    assert result.provenance["pipeline_tag"] == "image-segmentation"
    assert result.provenance["schema_version"] == 2
    assert result.provenance["external_data"]["bundle/encoder.onnx"] == {
        "bundle/weights.data": release_assets._sha256(result.root / "bundle/weights.data")
    }
    assert downloads == ["https://assets.example/model-onnx-float.zip"]

    cached = release_assets.acquire_hf_release_asset(
        "org/model", revision="main", precision="fp16", cache_dir=tmp_path / "cache"
    )
    assert cached is not None
    assert cached.root == result.root
    assert len(downloads) == 1

    graph = result.root / "bundle/encoder.onnx"
    expected_graph_hash = result.provenance["graphs"]["bundle/encoder.onnx"]
    graph.write_bytes(b"tampered graph")
    repaired_graph = release_assets.acquire_hf_release_asset(
        "org/model", revision="main", precision="fp16", cache_dir=tmp_path / "cache"
    )
    assert repaired_graph is not None
    assert release_assets._sha256(graph) == expected_graph_hash
    assert len(downloads) == 1

    sidecar = result.root / "bundle/weights.data"
    expected_sidecar_hash = result.provenance["external_data"]["bundle/encoder.onnx"][
        "bundle/weights.data"
    ]
    sidecar.write_bytes(b"tampered sidecar")
    repaired_sidecar = release_assets.acquire_hf_release_asset(
        "org/model", revision="main", precision="fp16", cache_dir=tmp_path / "cache"
    )
    assert repaired_sidecar is not None
    assert release_assets._sha256(sidecar) == expected_sidecar_hash
    assert len(downloads) == 1

    unrecorded_graph = result.root / "bundle/unrecorded.onnx"
    _identity_model(unrecorded_graph)
    repaired_inventory = release_assets.acquire_hf_release_asset(
        "org/model", revision="main", precision="fp16", cache_dir=tmp_path / "cache"
    )
    assert repaired_inventory is not None
    assert not unrecorded_graph.exists()
    assert len(downloads) == 1

    cached_archive = next((tmp_path / "cache").rglob("model-onnx-float.zip"))
    cached_archive.write_bytes(b"tampered archive")
    reacquired = release_assets.acquire_hf_release_asset(
        "org/model", revision="main", precision="fp16", cache_dir=tmp_path / "cache"
    )
    assert reacquired is not None
    assert release_assets._sha256(reacquired.root / "bundle/encoder.onnx") == expected_graph_hash
    assert len(downloads) == 2


def test_present_malformed_manifest_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    manifest = tmp_path / "release_assets.json"
    manifest.write_text("{}", encoding="utf-8")
    info = SimpleNamespace(sha="b" * 40, pipeline_tag="image-segmentation")
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(model_info=lambda *args, **kwargs: info),
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: str(manifest))

    with pytest.raises(ValueError, match="has no 'onnx' asset"):
        release_assets.acquire_hf_release_asset("org/model", cache_dir=tmp_path / "cache")
