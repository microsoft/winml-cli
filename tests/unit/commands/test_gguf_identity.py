# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the effective-configuration facts recorded in GGUF evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import time
from pathlib import Path

import psutil
import pytest

import winml.modelkit
from winml.modelkit.commands._gguf_identity import (
    PACKAGE_DISTRIBUTIONS,
    clock_info,
    environment_snapshot,
    gguf_shard_paths,
    hash_modules,
    list_loaded_modules,
    loaded_module_handle,
    model_inventory,
    package_versions,
    process_identity,
    producer_info,
    read_manifests,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _payload(tmp_path: Path) -> Path:
    """Generate a windowsml package with the three manifest kinds of the preview payload.

    The core manifest declares ``WinMLllama.dll`` correctly; the CPU backend
    manifest declares a wrong hash for ``ggml-cpu.dll``.
    """
    package = tmp_path / "windowsml"
    lib = package / "lib"
    backend_dir = lib / "LlamaBackends" / "cpu"
    backend_dir.mkdir(parents=True)
    llama, cpu, core = b"llama-bytes", b"cpu-backend-bytes", b"runtime-core-bytes"
    (lib / "WinMLllama.dll").write_bytes(llama)
    (backend_dir / "ggml-cpu.dll").write_bytes(cpu)
    (lib / "WinMLRuntimeCore.dll").write_bytes(core)
    _write_json(
        lib / "WinMLLlamaCoreManifest.json",
        {"schemaVersion": 1, "files": [{"path": "WinMLllama.dll", "sha256": _sha256(llama)}]},
    )
    _write_json(
        backend_dir / "WinMLLlamaBackendManifest.json",
        {"schemaVersion": 1, "family": "cpu", "backend": "ggml-cpu.dll",
         "sha256": _sha256(b"a different build")},
    )
    _write_json(
        lib / "WinMLLlamaPythonPairing.json",
        {"schemaVersion": 1,
         "files": [{"name": "WinMLRuntimeCore.dll", "sha256": _sha256(core), "size": len(core)}]},
    )
    return package


def test_single_file_inventory_matches_benchmark_digest(tmp_path: Path) -> None:
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"GGUF" + bytes(range(256)))
    shard = {
        "path": "tiny.gguf",
        "size_bytes": model.stat().st_size,
        "sha256": _sha256(model.read_bytes()),
    }
    digest = _sha256(
        json.dumps(sorted([shard], key=lambda s: s["path"]), sort_keys=True,
                   separators=(",", ":")).encode()
    )
    assert model_inventory(model) == {
        "entry_path": str(model.resolve()),
        "shards": [shard],
        "inventory_sha256": digest,
        "hashed_before_initialization": True,
    }


def test_split_model_enumerates_every_gguf_split_shard(tmp_path: Path) -> None:
    names = [f"m-{i:05d}-of-00003.gguf" for i in range(1, 4)]
    for i, name in enumerate(names):
        (tmp_path / name).write_bytes(bytes([i]) * 8)
    assert [p.name for p in gguf_shard_paths(tmp_path / names[1])] == names


def test_split_inventory_uses_shard_names_relative_to_the_entry(tmp_path: Path) -> None:
    names = [f"m-{i:05d}-of-00002.gguf" for i in range(1, 3)]
    for i, name in enumerate(names):
        (tmp_path / name).write_bytes(bytes([i + 1]) * 4)
    inventory = model_inventory(tmp_path / names[0])
    assert [shard["path"] for shard in inventory["shards"]] == names
    assert all(set(shard) == {"path", "size_bytes", "sha256"} for shard in inventory["shards"])


def test_missing_shard_raises(tmp_path: Path) -> None:
    (tmp_path / "m-00001-of-00002.gguf").write_bytes(b"a")
    with pytest.raises(FileNotFoundError, match=re.escape("m-00002-of-00002.gguf")):
        model_inventory(tmp_path / "m-00001-of-00002.gguf")


def test_read_manifests_reports_path_hash_and_fields(tmp_path: Path) -> None:
    manifests = read_manifests(_payload(tmp_path))
    assert [Path(m["path"]).name for m in manifests] == [
        "WinMLLlamaCoreManifest.json",
        "WinMLLlamaBackendManifest.json",
        "WinMLLlamaPythonPairing.json",
    ]
    for manifest in manifests:
        raw = Path(manifest["path"]).read_bytes()
        assert manifest["sha256"] == _sha256(raw)
        assert manifest["fields"] == json.loads(raw)


def test_read_manifests_without_a_package_is_empty() -> None:
    assert read_manifests(None) == []


def test_unreadable_manifest_keeps_hash_without_fields(tmp_path: Path) -> None:
    package = _payload(tmp_path)
    core = package / "lib" / "WinMLLlamaCoreManifest.json"
    core.write_text("{not json", encoding="utf-8")
    (entry,) = [m for m in read_manifests(package) if Path(m["path"]) == core]
    assert entry["fields"] is None
    assert entry["error"]
    assert entry["sha256"] == _sha256(core.read_bytes())


def test_hash_modules_limits_to_package_and_compares_manifests(tmp_path: Path) -> None:
    package = _payload(tmp_path)
    outside = tmp_path / "other.dll"
    outside.write_bytes(b"x")
    lib = package / "lib"
    paths = [
        str(lib / "WinMLllama.dll"),
        str(lib / "LlamaBackends" / "cpu" / "ggml-cpu.dll"),
        str(lib / "WinMLRuntimeCore.dll"),
        str(outside),
    ]
    hashed = hash_modules(paths, package, read_manifests(package))
    assert {Path(h["path"]).name: h["matches_manifest"] for h in hashed} == {
        "WinMLllama.dll": True,
        "ggml-cpu.dll": False,
        "WinMLRuntimeCore.dll": True,
    }
    assert [h["path"] for h in hashed] == sorted(h["path"] for h in hashed)
    for entry in hashed:
        data = Path(entry["path"]).read_bytes()
        assert (entry["size_bytes"], entry["sha256"]) == (len(data), _sha256(data))


def test_hash_modules_reports_undeclared_files_as_unknown(tmp_path: Path) -> None:
    package = _payload(tmp_path)
    extra = package / "lib" / "undeclared.dll"
    extra.write_bytes(b"extra")
    (entry,) = hash_modules([str(extra)], package, read_manifests(package))
    assert entry["matches_manifest"] is None


def test_environment_snapshot_keeps_prefixed_names_and_fixed_keys() -> None:
    env = {"WINML_X": "1", "GGML_Y": "2", "LLAMA_ARG_SPEC_TYPE": "none", "CUDA_PATH": "c",
           "PATH": "p"}
    expected = {key: value for key, value in env.items() if key != "PATH"}
    expected.update({"WINML_LLAMA_USE_MMAP": None, "WINML_LLAMA_SWA_FULL": None})
    snapshot = environment_snapshot(env)
    assert snapshot == dict(sorted(expected.items()))
    assert list(snapshot) == sorted(snapshot)


@pytest.mark.parametrize(
    ("url", "recorded"),
    [
        (
            "https://store.blob.core.windows.net/w/winml_cli-0.4.0-py3-none-any.whl?sv=2024&sig=SECRET",
            "https://store.blob.core.windows.net/w/winml_cli-0.4.0-py3-none-any.whl"
            "?sv=<redacted>&sig=<redacted>",
        ),
        ("https://user:SECRET@feed.example/winml_cli.whl", "https://<redacted>@feed.example/winml_cli.whl"),
        ("https://SECRET@github.com/microsoft/winml-cli", "https://<redacted>@github.com/microsoft/winml-cli"),
        ("git+ssh://git@github.com/microsoft/winml-cli.git", "git+ssh://git@github.com/microsoft/winml-cli.git"),
        ("file:///C:/Projects/WinML-ModelKit", "file:///C:/Projects/WinML-ModelKit"),
        ("https://h.example/w.whl?SECRET", "https://h.example/w.whl?<redacted>"),
        ("https://h.example/w.whl?a=1&SECRET", "https://h.example/w.whl?a=<redacted>&<redacted>"),
        ("https://h.example/w.whl?sig%3DSECRET", "https://h.example/w.whl?<redacted>"),
        ("https://h.example/w.whl?a+b=SECRET", "https://h.example/w.whl?<redacted>"),
        ("https://h.example/w.whl?U0VDUkVUdG9rZW4=", "https://h.example/w.whl?<redacted>"),
        ("https://h.example/w.whl?U0VDUkVUdG9rZW4==", "https://h.example/w.whl?<redacted>"),
        ("https://h.example/w.whl?SECRET=", "https://h.example/w.whl?<redacted>"),
        ("https://h.example/w.whl?SECRET;a=1", "https://h.example/w.whl?<redacted>"),
        ("https://[bad/SECRET.whl", "<redacted>"),
    ],
)
def test_producer_hides_credentials_in_the_install_url(
    monkeypatch: pytest.MonkeyPatch, url: str, recorded: str
) -> None:
    document = {"url": url, "vcs_info": {"vcs": "git", "commit_id": "0123abcd"}}

    class Distribution:
        version = "0.4.0"  # importlib.metadata.version() reads it through distribution()

        @staticmethod
        def read_text(name: str) -> str | None:
            return json.dumps(document) if name == "direct_url.json" else None

    monkeypatch.setattr(importlib.metadata, "distribution", lambda _name: Distribution())
    install = producer_info()["install"]

    assert install["direct_url"] == {**document, "url": recorded}
    assert "SECRET" not in json.dumps(install)


@pytest.mark.parametrize(
    "name",
    [
        "LLAMA_API_KEY",
        "LLAMA_ARG_API_KEY",
        "LLAMA_ARG_HF_TOKEN",
        "WINML_FEED_PASSWORD",
        "CUDA_SECRET",
        "GGML_AUTH_HEADER",
        "WINML_CREDENTIALS",
        "llama_api_key",
    ],
)
def test_environment_snapshot_keeps_secret_names_but_not_values(name: str) -> None:
    snapshot = environment_snapshot({name: "s3cr3t-value", "GGML_CUDA_FORCE_MMQ": "1"})

    assert snapshot[name] == "<redacted>"
    assert snapshot["GGML_CUDA_FORCE_MMQ"] == "1"
    assert "s3cr3t-value" not in json.dumps(snapshot)


def test_loaded_modules_and_handles_use_the_live_process() -> None:
    modules = list_loaded_modules()
    assert modules == sorted(set(modules))
    assert any(Path(path).name.lower() == "kernel32.dll" for path in modules)
    handle = loaded_module_handle("kernel32.dll")
    assert handle is not None
    assert handle[1].lower().endswith("kernel32.dll")
    assert loaded_module_handle("not-a-loaded-module.dll") is None


def test_process_clock_producer_and_packages() -> None:
    assert process_identity() == {"pid": os.getpid(), "create_time": psutil.Process().create_time()}
    info = time.get_clock_info("perf_counter")
    assert clock_info(5) == {
        "domain": "windows-qpc",
        "api": "time.perf_counter_ns",
        "implementation": info.implementation,
        "resolution": info.resolution,
        "monotonic": info.monotonic,
        "adjustable": info.adjustable,
        "started_qpc_ns": 5,
    }
    producer = producer_info()
    assert (producer["name"], producer["command"], producer["version"]) == (
        "winml-cli",
        "perf",
        importlib.metadata.version("winml-cli"),
    )
    assert Path(producer["install"]["location"]) == Path(winml.modelkit.__file__).resolve().parent
    assert set(package_versions()) == set(PACKAGE_DISTRIBUTIONS)
