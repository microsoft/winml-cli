# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Device -> EP mapping coverage for the ``scripts/qwen3.py`` export wrapper.

The thin CLI wrapper must translate ``--device`` into the execution-provider
alias the orchestrator understands.  A bare device token (e.g. ``"gpu"``) would
slip through ``normalize_ep_name`` unresolved and be rejected downstream as an
unknown EP, so every supported device must map to a resolvable EP.  The script
lives outside the importable package tree, so it is loaded by path.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from winml.modelkit.utils.constants import normalize_ep_name


if TYPE_CHECKING:
    from types import ModuleType


_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "qwen3.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("qwen3_export_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("device", "expected_ep"),
    [
        ("cpu", "CPUExecutionProvider"),
        ("gpu", "DmlExecutionProvider"),
        ("npu", "QNNExecutionProvider"),
    ],
)
def test_every_device_maps_to_a_resolvable_ep(device: str, expected_ep: str) -> None:
    module = _load_script()
    ep_alias = module._DEVICE_TO_EP[device]
    assert normalize_ep_name(ep_alias) == expected_ep


def test_export_forwards_mapped_ep_not_device_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--device gpu`` must forward ``ep='dml'``, never the bare ``'gpu'`` token.

    Regression for the refactor that passed the device token straight through as
    the EP, breaking the GPU branch with ``Unknown EP 'gpu'``.
    """
    module = _load_script()
    captured: dict = {}

    def _fake_build_genai_bundle(model_id, output_dir, recipe, **kwargs):
        captured.update(kwargs)
        captured["model_id"] = model_id
        return Path(output_dir) / "genai_config.json"

    monkeypatch.setattr(module, "build_genai_bundle", _fake_build_genai_bundle)

    args = argparse.Namespace(
        model_id="Qwen/Qwen3-0.6B",
        device="gpu",
        precision="w8a16",
        max_cache_len=2048,
        prefill_seq_len=64,
        output=tmp_path / "bundle",
        embeddings=None,
        lm_head=None,
        force_rebuild=False,
    )

    rc = module._cmd_export(args)

    assert rc == 0
    assert captured["ep"] == "dml"
    assert captured["device"] == "gpu"
    assert normalize_ep_name(captured["ep"]) == "DmlExecutionProvider"
