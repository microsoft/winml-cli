# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E tests for ``winml perf --backend cgc`` text generation on a real GPU.

They need the windowsml 2.8.183.dev0 preview, which provides the model-backed
Task API, and a DXCore GPU. The gate checks the package, then the API in a clean
``python -I`` subprocess, then the GPU, before any model is downloaded, so an
environment with the stable windowsml 2.0.300 skips without downloading. This
process never imports ``windowsml.runtime`` or ``windowsml.tasks``: the root
``tests/conftest.py`` loads onnxruntime, whose ``DirectML.dll`` would clash with
the preview's, so the CLI also runs in a fresh process. The models are
downloaded from Hugging Face into its cache; a failed download fails the test.
"""

from __future__ import annotations

import importlib.util
import json
import mmap
import os
import subprocess
import sys
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download, snapshot_download

from winml.modelkit.sysinfo import enumerate_compute_adapters, open_gpu_adapter


pytestmark = [pytest.mark.e2e, pytest.mark.gpu]

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
# Imports the preview modules in a clean interpreter and prints whether the
# model-backed API is present, or why the import failed.
_PROBE = (
    "import importlib, json\n"
    "try:\n"
    "    importlib.import_module('windowsml.runtime')\n"
    "    t = importlib.import_module('windowsml.tasks')\n"
    "    configuration = getattr(t, 'TextGenerationConfiguration', None)\n"
    "    print(json.dumps({'api': hasattr(configuration, 'set_model')}))\n"
    "except ImportError as exc:\n"
    "    print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}))\n"
)
_NEEDS_PREVIEW = "CGC text generation needs windowsml 2.8.183.dev0 or newer."
_MAX_NEW_TOKENS = 16


@pytest.fixture(scope="session")
def cgc_gpu() -> tuple[str, str]:
    """LUID and selection of the GPU that ``--backend cgc`` picks without ``--device-luid``.

    The package, the model-backed API and a GPU are checked in this order and
    are the only reasons to skip. Every model fixture requests this one, so the
    checks run before any download.
    """
    try:
        installed = importlib.util.find_spec("windowsml.tasks") is not None
    except ModuleNotFoundError:  # windowsml itself is not installed
        installed = False
    if not installed:
        pytest.skip(f"windowsml.tasks is not installed; {_NEEDS_PREVIEW}")
    probe = subprocess.run(  # noqa: S603 -- sys.executable and a constant script
        [sys.executable, "-I", "-c", _PROBE],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    try:
        outcome = json.loads(probe.stdout)
    except json.JSONDecodeError:
        pytest.fail(f"The windowsml.tasks probe exited with {probe.returncode}: {probe.stderr}")
    if "error" in outcome:
        pytest.fail(outcome["error"])
    if not outcome["api"]:
        pytest.skip(
            f"windowsml.tasks lacks TextGenerationConfiguration.set_model; {_NEEDS_PREVIEW}"
        )
    if not any(adapter.device_type == "GPU" for adapter in enumerate_compute_adapters()):
        pytest.skip("No DXCore GPU is available.")
    with open_gpu_adapter() as adapter:
        return adapter.luid, adapter.selection


@pytest.fixture(scope="session")
def gguf_model(cgc_gpu: tuple[str, str]) -> Path:
    """Qwen2.5-0.5B-Instruct as a q4_k_m GGUF file."""
    return Path(
        hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
    )


@pytest.fixture(scope="session")
def checkpoint_folder(cgc_gpu: tuple[str, str]) -> Path:
    """Qwen2.5-0.5B-Instruct as a safetensors checkpoint folder."""
    return Path(
        snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", allow_patterns=["*.json", "*.safetensors"])
    )


@pytest.fixture(scope="session")
def unsupported_folder(cgc_gpu: tuple[str, str]) -> Path:
    """An encoder-only checkpoint folder that the CGC backend cannot run."""
    return Path(
        snapshot_download(
            "sentence-transformers/all-MiniLM-L6-v2",
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        )
    )


def _perf(*args: str) -> subprocess.CompletedProcess[str]:
    """Run ``winml perf`` from this checkout in a fresh Python process."""
    return subprocess.run(  # noqa: S603 -- sys.executable and arguments built by the tests
        [sys.executable, "-m", "winml.modelkit", "perf", *args],
        env={**os.environ, "PYTHONPATH": str(REPO_SRC)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1500,
        check=False,
    )


def _declares_chat_template(model: Path) -> bool:
    """Whether the model files declare a chat template: the expected ``chat_template_applied``."""
    if model.is_dir():
        config = json.loads((model / "tokenizer_config.json").read_text(encoding="utf-8"))
        return "chat_template" in config
    with model.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
        return view.find(b"tokenizer.chat_template") != -1  # the standard GGUF metadata key


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    ("model_fixture", "model_format"),
    [("gguf_model", "gguf"), ("checkpoint_folder", "safetensors")],
)
def test_backend_cgc_benchmarks_text_generation_on_the_gpu(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    cgc_gpu: tuple[str, str],
    model_fixture: str,
    model_format: str,
) -> None:
    model: Path = request.getfixturevalue(model_fixture)
    report = tmp_path / "report.json"

    result = _perf(
        "-m",
        str(model),
        "--backend",
        "cgc",
        "--max-new-tokens",
        str(_MAX_NEW_TOKENS),
        "--warmup",
        "1",
        "--iterations",
        "2",
        "--output",
        str(report),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report.read_text(encoding="utf-8"))
    info = data["benchmark_info"]
    assert (
        data["schema_version"],
        info["backend"],
        info["model_format"],
        info["effective_device"],
    ) == (3, "cgc", model_format, "gpu")
    assert (info["adapter"]["luid"], info["adapter"]["selection"]) == cgc_gpu
    assert (info["apply_template"], info["chat_template_applied"]) == (
        True,
        _declares_chat_template(model),
    )
    assert set(info["packages"]) == {"windowsml", "winml-cli"}
    assert info["generated_tokens"] >= 1
    assert (
        info["context_length"] is None
        or info["context_length"] >= info["prompt_tokens"] + _MAX_NEW_TOKENS
    )
    assert data["load"]["set_model_duration_ms"] > 0
    assert data["load"]["session_create_duration_ms"] > 0
    assert data["aggregate"]["timed_request_count"] == 2
    assert "vram_local_after_compile_mb" in data["memory"]


@pytest.mark.timeout(1800)
def test_protocol_options_hold_on_cgc(tmp_path: Path, gguf_model: Path) -> None:
    prompt_ids = list(range(1000, 1032))
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps(prompt_ids), encoding="utf-8")
    report = tmp_path / "report.json"

    result = _perf(
        "-m",
        str(gguf_model),
        "--backend",
        "cgc",
        "--prompt-ids-file",
        str(ids),
        "--context-length",
        "256",
        "--ignore-eos",
        "--max-new-tokens",
        str(_MAX_NEW_TOKENS),
        "--warmup",
        "1",
        "--iterations",
        "2",
        "--output",
        str(report),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report.read_text(encoding="utf-8"))
    info = data["benchmark_info"]
    assert (info["context_length"], info["ignore_eos"], info["prompt"], info["apply_template"]) == (
        256,
        True,
        None,
        False,
    )
    assert info["prompt_tokens"] == len(prompt_ids)
    # With an empty EOS set, every timed request runs to --max-new-tokens.
    assert data["aggregate"]["stop_reason_counts"] == {"MAX_TOKENS": 2}


@pytest.mark.timeout(1800)
def test_unsupported_checkpoint_fails_without_fallback(
    tmp_path: Path, unsupported_folder: Path
) -> None:
    report = tmp_path / "report.json"

    result = _perf("-m", str(unsupported_folder), "--backend", "cgc", "--output", str(report))

    assert result.returncode == 1, result.stdout + result.stderr
    assert "cannot run this model" in result.stderr
    assert "Traceback" not in result.stderr
    assert not report.exists()
