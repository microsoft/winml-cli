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

1. ``classify_model_input`` recognizes the Hub-style ONNX reference as
   ``kind == "hub_onnx"`` and ``resolve_hf_onnx_path`` downloads the file
   via ``huggingface_hub``.
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

# Vision encoder: ~60 MB QOperator-quantized ViT backbone with ConvInteger
# (no CPU kernel) and 192 MatMulInteger nodes. Exercises the
# is_quantized_onnx (QOperator detection) + skip_optimize fixes that the
# decoder above does NOT cover -- the decoder happens to lack ConvInteger
# so the original "always run optimize" bug went unnoticed for it.
SAM3_ENCODER_ONNX_REF = "onnx-community/sam3-tracker-ONNX/onnx/vision_encoder_int8.onnx"


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.integration
class TestSam3E2E:
    """Pre-exported SAM 3 ONNX flows through Scenario D end-to-end."""

    @pytest.fixture(scope="class")
    def sam3_onnx_path(self) -> Path:
        """Download the SAM 3 Tracker decoder ONNX once for the test class.

        ``huggingface_hub`` is a hard transitive dep (via ``transformers`` /
        ``optimum``) so we do NOT ``importorskip`` it -- a missing import
        is a real packaging bug. Network-related download failures are
        narrowed to the HF Hub error hierarchy + ``OSError`` (DNS, TLS,
        connection reset) and ONLY those become a skip; any other
        exception is allowed to surface as a real test failure.
        """
        from huggingface_hub.utils import HfHubHTTPError

        from winml.modelkit.loader import resolve_hf_onnx_path
        from winml.modelkit.utils.model_input import classify_model_input

        assert classify_model_input(SAM3_ONNX_REF).kind == "hub_onnx"
        try:
            return resolve_hf_onnx_path(SAM3_ONNX_REF)
        except (HfHubHTTPError, OSError) as e:
            pytest.skip(f"Network unavailable to download {SAM3_ONNX_REF}: {e}")
            raise  # unreachable (pytest.skip raises Skipped); satisfies static analyzers

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
            "Expected the int8 variant to be detected as quantized "
            "(QDQ pairs and/or QOperator integer ops such as MatMulInteger)."
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
        """build_onnx_model runs end-to-end and emits model.onnx.

        Build failures are NOT silently skipped. A ``RuntimeError`` from
        ``build_onnx_model`` here means a real regression in the SAM 3
        pipeline (e.g. the ``ConvInteger`` / ``skip_optimize`` bug fixed
        in this PR). Letting that surface as a hard failure is precisely
        the value of an integration test.
        """
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

        result = build_onnx_model(
            onnx_path=sam3_onnx_path,
            config=config,
            output_dir=output_dir,
            rebuild=True,
            hack_max_optim_iterations=0,  # skip analyzer to keep test fast
        )

        final = result.final_onnx_path
        assert final.exists(), f"Expected final artifact at {final}"
        assert final.stat().st_size > 0

        # Validate the final artifact is a structurally valid ONNX model.
        model = onnx.load(str(final), load_external_data=False)
        assert len(model.graph.node) > 0

    def test_analyze_autoconf_runs(self, sam3_onnx_path: Path) -> None:
        """Analyzer autoconf produces an optimization config for SAM 3.

        Issue #324 explicitly requires verifying that the analyzer's
        autoconf loop discovers the correct fusion flags. The build test
        above disables the analyze<->optimize loop with
        ``hack_max_optim_iterations=0`` to keep CI fast, so this test
        exercises the autoconf path directly via ``analyze_onnx``.

        ``winml.modelkit.analyze`` is part of this package, so a missing
        import is a real packaging bug -- not skipped. Analyzer
        ``RuntimeError`` is a real regression and surfaces loudly.
        """
        from winml.modelkit.analyze import analyze_onnx

        result = analyze_onnx(sam3_onnx_path, ep="cpu", autoconf=True)

        # autoconf=True must yield an optimization_config (may be empty
        # if the model needs no further optimization, but must be present).
        assert result.optimization_config is not None, (
            "Expected analyzer to produce an optimization_config when autoconf=True; "
            "got None which signals the autoconf loop did not run."
        )


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.integration
class TestSam3EncoderE2E:
    """SAM 3 vision encoder (QOperator format with ConvInteger) end-to-end.

    Regression test for two bugs found while wiring SAM 3 support:

    1. ``is_quantized_onnx`` only detected QDQ format and missed
       ``QuantFormat.QOperator`` exports (``ConvInteger`` /
       ``MatMulInteger`` / ``QLinear*``). The encoder was therefore
       routed through the optimize + quantize stages.
    2. The pre-quantized branches in ``build_onnx_model`` and
       ``_build_onnx_pipeline`` named themselves "skip optimize" but
       still invoked ``optimize_onnx`` -> ``ort_graph``, which loads the
       model into an ORT session and crashes for QOperator models on
       hosts (e.g. CPU-only) without a ``ConvInteger`` kernel.

    The fix wires a real ``skip_optimize=True`` knob through both
    pipelines. This test downloads the ~60 MB encoder and asserts the
    full pipeline succeeds without invoking the optimizer.
    """

    @pytest.fixture(scope="class")
    def encoder_onnx_path(self) -> Path:
        """Download the SAM 3 Tracker vision encoder ONNX once for the class.

        Network failures are narrowed to the HF Hub error hierarchy +
        ``OSError`` and only those become a skip; any other exception
        surfaces as a real test failure.
        """
        from huggingface_hub.utils import HfHubHTTPError

        from winml.modelkit.loader import resolve_hf_onnx_path
        from winml.modelkit.utils.model_input import classify_model_input

        assert classify_model_input(SAM3_ENCODER_ONNX_REF).kind == "hub_onnx"
        try:
            return resolve_hf_onnx_path(SAM3_ENCODER_ONNX_REF)
        except (HfHubHTTPError, OSError) as e:
            pytest.skip(f"Network unavailable to download {SAM3_ENCODER_ONNX_REF}: {e}")
            raise  # unreachable (pytest.skip raises Skipped); satisfies static analyzers

    def test_encoder_is_detected_as_quantized(self, encoder_onnx_path: Path) -> None:
        """The QOperator-quantized encoder is recognized by is_quantized_onnx."""
        from winml.modelkit.onnx import is_quantized_onnx

        assert is_quantized_onnx(encoder_onnx_path), (
            "Expected QOperator-quantized encoder to be detected by "
            "is_quantized_onnx (regression: previously only QDQ format was checked)."
        )

    def test_build_encoder_skips_optimize_and_succeeds(
        self, encoder_onnx_path: Path, tmp_path: Path
    ) -> None:
        """``build_onnx_model`` runs end-to-end on the encoder without optimize.

        Build failures are NOT silently skipped -- a ``RuntimeError`` here
        means a regression in the QOperator detection / skip_optimize fix
        that this test exists to lock down.
        """
        from winml.modelkit.build import build_onnx_model
        from winml.modelkit.config import generate_onnx_build_config

        config = generate_onnx_build_config(
            encoder_onnx_path,
            task="image-feature-extraction",
            device="cpu",
            precision="auto",
        )
        # Sanity: pre-quantized models must skip the quant stage.
        assert config.quant is None, (
            "Expected pre-quantized encoder to set config.quant=None; "
            "got a quant config which would re-quantize an already-int8 model."
        )
        config.compile = None  # No NPU on the test host.

        output_dir = tmp_path / "sam3_encoder_build"

        result = build_onnx_model(
            onnx_path=encoder_onnx_path,
            config=config,
            output_dir=output_dir,
            rebuild=True,
            hack_max_optim_iterations=0,
        )

        final = result.final_onnx_path
        assert final.exists(), f"Expected final artifact at {final}"
        assert final.stat().st_size > 0

        # Validate the final artifact is structurally a valid ONNX model
        # and still contains the QOperator ops (proof we did not strip them
        # by accidentally running graph optimization).
        model = onnx.load(str(final), load_external_data=False)
        op_types = {n.op_type for n in model.graph.node}
        assert "ConvInteger" in op_types, (
            "Final encoder should still contain ConvInteger nodes -- "
            "presence proves optimize was correctly skipped."
        )
