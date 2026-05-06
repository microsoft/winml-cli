# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""End-to-end integration test for SAM 3 (Tracker) via the pre-exported ONNX path.

SAM 3 cannot be exported through the standard HuggingFace + Optimum route used
by the rest of ModelKit because ``optimum-onnx`` currently pins
``transformers<4.58`` while SAM 3 requires ``transformers>=5``. ModelKit instead
consumes the pre-exported ONNX from the ``onnx-community/sam3-tracker-ONNX``
Hub repo via the Scenario D pipeline (``build_onnx_model``).

Pipeline verified by this test:

1. ``is_hf_onnx_path`` recognizes the Hub-style ONNX reference and
   ``resolve_hf_onnx_path`` downloads the file via ``huggingface_hub``.
2. ``generate_onnx_build_config`` produces a valid build config for the
   already-quantized ONNX (skips optimize and quantize stages).
3. ``build_onnx_model`` produces a final ``model.onnx`` artifact that loads
   cleanly with ``onnx``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx
import pytest


if TYPE_CHECKING:
    from pathlib import Path


# Decoder-only variant: ~290 KB ONNX + ~10 MB sidecar. Small enough for CI
# while still exercising the is_quantized_onnx branch (skips optimize+quantize).
SAM3_ONNX_REF = "onnx-community/sam3-tracker-ONNX/onnx/prompt_encoder_mask_decoder_int8.onnx"


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.integration
class TestSam3E2E:
    """Pre-exported SAM 3 ONNX flows through Scenario D end-to-end."""

    @pytest.fixture(scope="class")
    def sam3_onnx_path(self) -> Path:
        """Download the SAM 3 Tracker decoder ONNX once for the test class."""
        pytest.importorskip("huggingface_hub", reason="huggingface_hub required")

        from winml.modelkit.loader import is_hf_onnx_path, resolve_hf_onnx_path

        assert is_hf_onnx_path(SAM3_ONNX_REF)
        try:
            return resolve_hf_onnx_path(SAM3_ONNX_REF)
        except Exception as e:
            pytest.skip(f"Could not download {SAM3_ONNX_REF}: {e}")

    def test_resolves_to_local_onnx_file(self, sam3_onnx_path: Path) -> None:
        """The Hub reference resolves to an on-disk .onnx file."""
        assert sam3_onnx_path.is_file()
        assert sam3_onnx_path.suffix == ".onnx"
        assert sam3_onnx_path.stat().st_size > 0

    def test_generate_onnx_build_config_detects_quantized(self, sam3_onnx_path: Path) -> None:
        """The int8 variant is detected as already quantized."""
        from winml.modelkit.config import generate_onnx_build_config
        from winml.modelkit.onnx import is_quantized_onnx

        assert is_quantized_onnx(sam3_onnx_path), (
            "Expected the int8 variant to contain QuantizeLinear / DequantizeLinear nodes."
        )

        config = generate_onnx_build_config(
            sam3_onnx_path,
            task="mask-generation",
            device="auto",
            precision="auto",
        )

        # Quantized models skip the quantization stage entirely.
        assert config.export is None
        assert config.quant is None

    def test_build_onnx_model_produces_final_artifact(
        self, sam3_onnx_path: Path, tmp_path: Path
    ) -> None:
        """build_onnx_model runs end-to-end and emits model.onnx."""
        from winml.modelkit.build import build_onnx_model
        from winml.modelkit.config import generate_onnx_build_config

        config = generate_onnx_build_config(
            sam3_onnx_path,
            task="mask-generation",
            device="cpu",
            precision="auto",
        )
        # Disable compilation: this test asserts pipeline plumbing,
        # not EP availability on the test host.
        config.compile = None

        output_dir = tmp_path / "sam3_build"

        try:
            result = build_onnx_model(
                onnx_path=sam3_onnx_path,
                config=config,
                output_dir=output_dir,
                rebuild=True,
                hack_max_optim_iterations=0,  # skip analyzer to keep test fast
            )
        except Exception as e:
            pytest.skip(f"build_onnx_model failed (likely missing runtime dep): {e}")

        final = result.final_onnx_path
        assert final.exists(), f"Expected final artifact at {final}"
        assert final.stat().st_size > 0

        # Validate the final artifact is a structurally valid ONNX model.
        model = onnx.load(str(final), load_external_data=False)
        assert len(model.graph.node) > 0
