# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the deterministic GenAI multi-prompt benchmark runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from types import ModuleType


_SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts" / "e2e_eval" / "run_genai_multi_prompt_perf.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_genai_multi_prompt_perf", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CharacterTokenizer:
    @staticmethod
    def encode(text: str) -> list[int]:
        return [ord(character) for character in text]

    @staticmethod
    def decode(tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def test_seeded_prompt_is_deterministic_and_exact_length() -> None:
    module = _load_script()
    tokenizer = CharacterTokenizer()

    first = module.build_seeded_prompt(tokenizer, 256, 42)
    repeated = module.build_seeded_prompt(tokenizer, 256, 42)
    different = module.build_seeded_prompt(tokenizer, 256, 43)

    assert first == repeated
    assert first != different
    assert len(tokenizer.encode(first)) == 256


@pytest.mark.parametrize("target_tokens", [0, -1])
def test_seeded_prompt_rejects_non_positive_lengths(target_tokens: int) -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="target_tokens must be positive"):
        module.build_seeded_prompt(CharacterTokenizer(), target_tokens, 42)


def test_stats_reports_distribution() -> None:
    module = _load_script()

    result = module.stats([4.0, 1.0, 3.0, 2.0])

    assert result["mean"] == pytest.approx(2.5)
    assert result["min"] == 1.0
    assert result["max"] == 4.0
    assert result["p50"] == 3.0
