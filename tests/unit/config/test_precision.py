# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for modelkit.config.precision module.

Tests precision resolution and policy application.
The precision module is pure logic with no I/O -- it receives a concrete
device string and returns a PrecisionPolicy. Device detection tests
belong in tests/sysinfo/test_device.py.
"""

from __future__ import annotations

import logging

import pytest

from winml.modelkit.config.precision import (
    is_quantized_precision,
    resolve_precision,
    resolve_quant_types,
)


# =============================================================================
# TestResolvePrecision - Auto device/precision resolution
# =============================================================================


class TestResolvePrecision:
    """Test resolve_precision() function.

    All tests pass concrete device strings -- no mocking needed.
    """

    # ---- Parametrized matrix: explicit device x precision ----
    @pytest.mark.parametrize(
        "device,precision,exp_device,exp_precision,exp_weight,exp_act,exp_provider",
        [
            # device   precision  exp_device  exp_prec  weight    act      provider
            ("npu", "auto", "npu", "w8a16", "uint8", "uint16", "qnn"),
            ("npu", "int8", "npu", "int8", "uint8", "uint8", "qnn"),
            ("npu", "int16", "npu", "int16", "int16", "uint16", "qnn"),
            ("npu", "fp16", "npu", "fp16", None, None, "qnn"),
            ("npu", "fp32", "npu", "fp32", None, None, "qnn"),
            ("npu", "w8a16", "npu", "w8a16", "uint8", "uint16", "qnn"),
            ("npu", "w8a8", "npu", "w8a8", "uint8", "uint8", "qnn"),
            ("npu", "w16a16", "npu", "w16a16", "int16", "uint16", "qnn"),
            ("gpu", "auto", "gpu", "fp16", None, None, "dml"),
            ("gpu", "w8a16", "gpu", "w8a16", "uint8", "uint16", "dml"),
            ("gpu", "int8", "gpu", "int8", "uint8", "uint8", "dml"),
            ("gpu", "int16", "gpu", "int16", "int16", "uint16", "dml"),
            ("gpu", "fp16", "gpu", "fp16", None, None, "dml"),
            ("gpu", "fp32", "gpu", "fp32", None, None, "dml"),
            ("cpu", "auto", "cpu", "fp16", None, None, None),
            ("cpu", "int8", "cpu", "int8", "uint8", "uint8", None),
            ("cpu", "int16", "cpu", "int16", "int16", "uint16", None),
            ("cpu", "fp16", "cpu", "fp16", None, None, None),
            ("cpu", "fp32", "cpu", "fp32", None, None, None),
        ],
    )
    def test_resolve_precision_matrix(
        self,
        device: str,
        precision: str,
        exp_device: str,
        exp_precision: str,
        exp_weight: str | None,
        exp_act: str | None,
        exp_provider: str | None,
    ) -> None:
        """Full device x precision matrix produces correct PrecisionPolicy."""
        policy = resolve_precision(device=device, precision=precision)
        assert policy.device == exp_device
        assert policy.precision == exp_precision
        assert policy.weight_type == exp_weight
        assert policy.activation_type == exp_act
        assert policy.compile_provider == exp_provider

    # ---- Parametrized: auto device picks best for explicit precision ----
    @pytest.mark.parametrize(
        "precision,available,exp_device",
        [
            ("int8", ["npu", "gpu", "cpu"], "npu"),  # prefers NPU for int8
            ("int8", ["gpu", "cpu"], "gpu"),  # no NPU, falls to first
            ("fp16", ["npu", "gpu", "cpu"], "gpu"),  # prefers GPU for fp16
            ("fp16", ["npu", "cpu"], "npu"),  # no GPU, falls to first
            ("fp32", ["cpu"], "cpu"),  # only CPU
            ("int16", ["npu", "gpu", "cpu"], "npu"),  # prefers NPU for int16
        ],
    )
    def test_auto_device_picks_best(
        self,
        precision: str,
        available: list[str],
        exp_device: str,
    ) -> None:
        """device='auto' + explicit precision picks best from available_devices."""
        policy = resolve_precision(
            device="auto",
            precision=precision,
            available_devices=available,
        )
        assert policy.device == exp_device

    # ---- Non-parametrized edge cases ----

    def test_both_auto_returns_noop(self) -> None:
        """device='auto' + precision='auto' returns no-op policy."""
        policy = resolve_precision(device="auto", precision="auto")
        assert policy.device == "auto"
        assert policy.precision == "auto"
        assert policy.weight_type is None
        assert policy.activation_type is None
        assert policy.compile_provider is None

    def test_unknown_device_raises(self) -> None:
        """Unknown device name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown device"):
            resolve_precision(device="tpu")

    def test_unknown_precision_raises(self) -> None:
        """Unknown precision name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown precision"):
            resolve_precision(device="cpu", precision="bfloat16")


# =============================================================================
# TestGpuLlmWarning - GPU + LLM task warning
# =============================================================================


class TestGpuLlmWarning:
    """Test GPU + LLM task warning about w4a16."""

    def test_gpu_llm_warning(self, caplog) -> None:
        """GPU + text-generation + auto precision logs w4a16 warning."""
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.config.precision"):
            policy = resolve_precision(device="gpu", task="text-generation")

        assert policy.device == "gpu"
        assert policy.precision == "fp16"
        assert any("w4a16" in record.message for record in caplog.records)

    def test_gpu_non_llm_no_warning(self, caplog) -> None:
        """GPU + image-classification does NOT log w4a16 warning."""
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.config.precision"):
            policy = resolve_precision(device="gpu", task="image-classification")

        assert policy.precision == "fp16"
        assert not any("w4a16" in record.message for record in caplog.records)

    def test_gpu_text2text_warning(self, caplog) -> None:
        """GPU + text2text-generation also logs w4a16 warning."""
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.config.precision"):
            resolve_precision(device="gpu", task="text2text-generation")

        assert any("w4a16" in record.message for record in caplog.records)

    def test_npu_llm_no_warning(self, caplog) -> None:
        """NPU + text-generation does NOT log w4a16 warning (not GPU)."""
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.config.precision"):
            policy = resolve_precision(device="npu", task="text-generation")

        assert policy.device == "npu"
        assert not any("w4a16" in record.message for record in caplog.records)


# =============================================================================
# TestEpOverride - --ep flag behavior
# =============================================================================


class TestEpOverride:
    """Test ep parameter in resolve_precision()."""

    def test_ep_overrides_compile_provider(self) -> None:
        """ep='migraphx' should set compile_provider to 'migraphx', not 'dml'."""
        policy = resolve_precision(device="gpu", ep="migraphx")
        assert policy.compile_provider == "migraphx"
        assert policy.device == "gpu"

    def test_ep_overrides_default_dml(self) -> None:
        """Without ep, gpu maps to dml. With ep='nv_tensorrt_rtx', should be nv_tensorrt_rtx."""
        default = resolve_precision(device="gpu")
        assert default.compile_provider == "dml"

        override = resolve_precision(device="gpu", ep="nv_tensorrt_rtx")
        assert override.compile_provider == "nv_tensorrt_rtx"

    def test_ep_infers_device_from_gpu_ep(self) -> None:
        """ep='migraphx' with device='auto' should infer device='gpu'."""
        policy = resolve_precision(ep="migraphx")
        assert policy.device == "gpu"
        assert policy.compile_provider == "migraphx"

    def test_ep_infers_device_from_npu_ep(self) -> None:
        """ep='vitisai' with device='auto' should infer device='npu'."""
        policy = resolve_precision(ep="vitisai")
        assert policy.device == "npu"
        assert policy.compile_provider == "vitisai"

    def test_ep_infers_device_from_qnn(self) -> None:
        """ep='qnn' should infer device='npu'."""
        policy = resolve_precision(ep="qnn")
        assert policy.device == "npu"
        assert policy.compile_provider == "qnn"

    def test_ep_with_explicit_device(self) -> None:
        """ep + explicit device should use the explicit device."""
        policy = resolve_precision(device="gpu", ep="vitisai")
        assert policy.device == "gpu"
        assert policy.compile_provider == "vitisai"

    def test_ep_preserves_precision_logic(self) -> None:
        """ep should not break precision resolution."""
        policy = resolve_precision(device="gpu", precision="int8", ep="migraphx")
        assert policy.precision == "int8"
        assert policy.weight_type == "uint8"
        assert policy.compile_provider == "migraphx"

    def test_unknown_ep_raises(self) -> None:
        """Invalid EP name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown EP"):
            resolve_precision(ep="unknown_ep")

    def test_all_valid_eps(self) -> None:
        """All VALID_EPS should be accepted without error."""
        from winml.modelkit.session import VALID_EPS

        for ep_name in VALID_EPS:
            policy = resolve_precision(ep=ep_name)
            assert policy.compile_provider == ep_name

    def test_ep_none_uses_default_mapping(self) -> None:
        """ep=None should use the default device→provider mapping."""
        policy = resolve_precision(device="npu")
        assert policy.compile_provider == "qnn"

    def test_ep_case_insensitive(self) -> None:
        """EP names should be case-insensitive."""
        policy = resolve_precision(ep="MiGraphX")
        assert policy.compile_provider == "migraphx"


# =============================================================================
# TestResolveQuantTypes - Direct unit tests for resolve_quant_types()
# =============================================================================


class TestResolveQuantTypes:
    """Test resolve_quant_types() function directly.

    This function is the single source of truth for mapping precision strings
    to (weight_type, activation_type) tuples. It handles both named presets
    (int8, int16) and mixed w{x}a{y} format.
    """

    # ---- Named presets: valid quantized ----
    @pytest.mark.parametrize(
        "precision,exp_weight,exp_act",
        [
            ("int8", "uint8", "uint8"),
            ("int16", "int16", "uint16"),
        ],
    )
    def test_named_presets(self, precision: str, exp_weight: str, exp_act: str) -> None:
        """Named quantized presets resolve to correct weight/activation types."""
        w, a = resolve_quant_types(precision)
        assert w == exp_weight
        assert a == exp_act

    # ---- Mixed w{x}a{y} format: valid combinations ----
    @pytest.mark.parametrize(
        "precision,exp_weight,exp_act",
        [
            ("w8a8", "uint8", "uint8"),
            ("w8a16", "uint8", "uint16"),
            ("w16a8", "int16", "uint8"),
            ("w16a16", "int16", "uint16"),
        ],
    )
    def test_mixed_format_valid(self, precision: str, exp_weight: str, exp_act: str) -> None:
        """Valid w{x}a{y} combinations resolve to correct types."""
        w, a = resolve_quant_types(precision)
        assert w == exp_weight
        assert a == exp_act

    # ---- Float types raise ValueError ----
    @pytest.mark.parametrize("precision", ["fp16", "fp32"])
    def test_float_precision_raises(self, precision: str) -> None:
        """Float precisions have no quantization types -- must raise ValueError."""
        with pytest.raises(ValueError, match="float type"):
            resolve_quant_types(precision)

    # ---- "auto" raises ValueError ----
    def test_auto_raises(self) -> None:
        """'auto' is not a quantization precision -- must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown precision"):
            resolve_quant_types("auto")

    # ---- Unsupported bit widths ----
    def test_unsupported_weight_bits_raises(self) -> None:
        """w4a16 has unsupported weight bit-width 4 -- must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported weight bit-width 4"):
            resolve_quant_types("w4a16")

    def test_unsupported_activation_bits_raises(self) -> None:
        """w8a4 has unsupported activation bit-width 4 -- must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported activation bit-width 4"):
            resolve_quant_types("w8a4")

    def test_both_bits_unsupported_raises_weight_first(self) -> None:
        """w4a4 should raise on weight bits first (checked before activation)."""
        with pytest.raises(ValueError, match="Unsupported weight bit-width 4"):
            resolve_quant_types("w4a4")

    # ---- Completely invalid strings ----
    @pytest.mark.parametrize("precision", ["garbage", "w0a0", "bfloat16", ""])
    def test_invalid_strings_raise(self, precision: str) -> None:
        """Completely invalid precision strings must raise ValueError."""
        with pytest.raises(ValueError):
            resolve_quant_types(precision)

    # ---- Non-numeric w{x}a{y} ----
    def test_non_numeric_mixed_raises(self) -> None:
        """wXaY with non-numeric characters must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown precision"):
            resolve_quant_types("wXaY")

    # ---- Case insensitivity ----
    @pytest.mark.parametrize(
        "precision,exp_weight,exp_act",
        [
            ("W8A16", "uint8", "uint16"),
            ("w8A16", "uint8", "uint16"),
            ("INT8", "uint8", "uint8"),
            ("Int16", "int16", "uint16"),
        ],
    )
    def test_case_insensitive(self, precision: str, exp_weight: str, exp_act: str) -> None:
        """resolve_quant_types should be case-insensitive."""
        w, a = resolve_quant_types(precision)
        assert w == exp_weight
        assert a == exp_act

    # ---- Leading zeros ----
    def test_leading_zeros_accepted(self) -> None:
        """w08a16 should be treated as w8a16 (int('08') == 8)."""
        w, a = resolve_quant_types("w08a16")
        assert w == "uint8"
        assert a == "uint16"

    def test_leading_zeros_w016a016(self) -> None:
        """w016a016 should be treated as w16a16."""
        w, a = resolve_quant_types("w016a016")
        assert w == "int16"
        assert a == "uint16"


# =============================================================================
# TestIsQuantizedPrecision - Direct unit tests for is_quantized_precision()
# =============================================================================


class TestIsQuantizedPrecision:
    """Test is_quantized_precision() function directly.

    This function is the gatekeeper that decides whether a precision string
    implies quantization. It must return False for float types AND for
    unsupported w{x}a{y} bit widths (rather than claiming they are quantized).
    """

    # ---- True cases: supported quantized precisions ----
    @pytest.mark.parametrize(
        "precision",
        ["int8", "int16", "w8a8", "w8a16", "w16a8", "w16a16"],
    )
    def test_quantized_returns_true(self, precision: str) -> None:
        """Supported quantized precisions must return True."""
        assert is_quantized_precision(precision) is True

    # ---- False cases: float and auto ----
    @pytest.mark.parametrize("precision", ["fp16", "fp32", "auto"])
    def test_float_and_auto_return_false(self, precision: str) -> None:
        """Float precisions and 'auto' are not quantized."""
        assert is_quantized_precision(precision) is False

    # ---- False cases: unsupported bit widths ----
    @pytest.mark.parametrize("precision", ["w4a16", "w8a4", "w4a4", "w2a8", "w8a2"])
    def test_unsupported_bits_return_false(self, precision: str) -> None:
        """Unsupported w{x}a{y} bit widths must return False, not True."""
        assert is_quantized_precision(precision) is False

    # ---- False cases: completely invalid ----
    @pytest.mark.parametrize("precision", ["garbage", "wXaY", "", "bfloat16", "w0a0"])
    def test_invalid_strings_return_false(self, precision: str) -> None:
        """Completely invalid precision strings must return False."""
        assert is_quantized_precision(precision) is False

    # ---- Case insensitivity ----
    @pytest.mark.parametrize("precision", ["W8A16", "INT8", "Int16", "w8A16"])
    def test_case_insensitive(self, precision: str) -> None:
        """is_quantized_precision should be case-insensitive."""
        assert is_quantized_precision(precision) is True

    # ---- Leading zeros ----
    def test_leading_zeros_recognized(self) -> None:
        """w08a16 should be recognized as quantized (same as w8a16)."""
        assert is_quantized_precision("w08a16") is True


# =============================================================================
# TestMixedPrecisionAutoDevice - w{x}a{y} with device="auto"
# =============================================================================


class TestMixedPrecisionAutoDevice:
    """Test that w{x}a{y} precisions route to NPU when device='auto'.

    The _pick_device_for_precision function uses is_quantized_precision()
    to decide NPU preference. Mixed precisions must behave like int8/int16.
    """

    @pytest.mark.parametrize(
        "precision,available,exp_device",
        [
            ("w8a16", ["npu", "gpu", "cpu"], "npu"),  # prefers NPU
            ("w8a16", ["gpu", "cpu"], "gpu"),  # no NPU, falls to first
            ("w8a8", ["npu", "gpu", "cpu"], "npu"),  # prefers NPU
            ("w16a16", ["npu", "cpu"], "npu"),  # prefers NPU
            ("w8a16", ["cpu"], "cpu"),  # only CPU available
        ],
    )
    def test_mixed_precision_auto_device(
        self,
        precision: str,
        available: list[str],
        exp_device: str,
    ) -> None:
        """device='auto' + w{x}a{y} precision picks best from available_devices."""
        policy = resolve_precision(
            device="auto",
            precision=precision,
            available_devices=available,
        )
        assert policy.device == exp_device

    def test_w8a16_auto_npu_full_policy(self) -> None:
        """w8a16 + device='auto' with NPU available produces complete policy."""
        policy = resolve_precision(
            device="auto",
            precision="w8a16",
            available_devices=["npu", "gpu", "cpu"],
        )
        assert policy.device == "npu"
        assert policy.precision == "w8a16"
        assert policy.weight_type == "uint8"
        assert policy.activation_type == "uint16"
        assert policy.compile_provider == "qnn"


# =============================================================================
# TestMixedPrecisionInvalidInputs - resolve_precision validation
# =============================================================================


class TestMixedPrecisionInvalidInputs:
    """Test that invalid w{x}a{y} inputs are rejected by resolve_precision."""

    @pytest.mark.parametrize(
        "precision",
        ["w4a16", "w4a4", "w2a8"],
    )
    def test_unsupported_mixed_bits_rejected(self, precision: str) -> None:
        """Unsupported w{x}a{y} bit widths should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown precision"):
            resolve_precision(device="npu", precision=precision)

    def test_w0a0_rejected(self) -> None:
        """w0a0 is not a valid precision."""
        with pytest.raises(ValueError, match="Unknown precision"):
            resolve_precision(device="npu", precision="w0a0")

    def test_non_numeric_mixed_rejected(self) -> None:
        """wXaY with letters should be rejected."""
        with pytest.raises(ValueError, match="Unknown precision"):
            resolve_precision(device="npu", precision="wXaY")

    def test_case_insensitive_via_resolve_precision(self) -> None:
        """W8A16 (uppercase) should work through resolve_precision."""
        policy = resolve_precision(device="npu", precision="W8A16")
        assert policy.precision == "w8a16"
        assert policy.weight_type == "uint8"
        assert policy.activation_type == "uint16"

    def test_leading_zeros_via_resolve_precision(self) -> None:
        """w08a16 should be accepted by resolve_precision (leading zeros)."""
        policy = resolve_precision(device="npu", precision="w08a16")
        assert policy.precision == "w08a16"
        assert policy.weight_type == "uint8"
        assert policy.activation_type == "uint16"


# =============================================================================
# TestQuantizeCliResolveQuant - quantize CLI _resolve_quant_types()
# =============================================================================


class TestQuantizeCliResolveQuant:
    """Test _resolve_quant_types from the quantize CLI command.

    This function delegates to config.precision.resolve_quant_types when
    precision is quantized, and falls back to ("uint8", "uint8") otherwise.
    Explicit --weight-type/--activation-type flags override precision defaults.
    """

    @staticmethod
    def _resolve(
        precision: str | None = None,
        weight_type: str | None = None,
        activation_type: str | None = None,
    ) -> tuple[str, str]:
        """Helper to call the quantize CLI internal resolver."""
        from winml.modelkit.commands.quantize import _resolve_quant_types

        return _resolve_quant_types(precision, weight_type, activation_type)

    # ---- w{x}a{y} precision ----
    def test_w8a16_defaults(self) -> None:
        """--precision w8a16 should produce (uint8, uint16)."""
        w, a = self._resolve(precision="w8a16")
        assert w == "uint8"
        assert a == "uint16"

    def test_w8a8_defaults(self) -> None:
        """--precision w8a8 should produce (uint8, uint8)."""
        w, a = self._resolve(precision="w8a8")
        assert w == "uint8"
        assert a == "uint8"

    def test_w16a16_defaults(self) -> None:
        """--precision w16a16 should produce (int16, uint16)."""
        w, a = self._resolve(precision="w16a16")
        assert w == "int16"
        assert a == "uint16"

    # ---- Named presets still work ----
    def test_int8_defaults(self) -> None:
        """--precision int8 should produce (uint8, uint8)."""
        w, a = self._resolve(precision="int8")
        assert w == "uint8"
        assert a == "uint8"

    def test_int16_defaults(self) -> None:
        """--precision int16 should produce (int16, uint16)."""
        w, a = self._resolve(precision="int16")
        assert w == "int16"
        assert a == "uint16"

    # ---- No precision falls back to uint8/uint8 ----
    def test_no_precision_defaults_uint8(self) -> None:
        """No --precision should fall back to (uint8, uint8)."""
        w, a = self._resolve(precision=None)
        assert w == "uint8"
        assert a == "uint8"

    # ---- Unsupported precision falls back to uint8/uint8 ----
    def test_unsupported_precision_falls_back(self) -> None:
        """Unsupported precision (w4a16) is not quantized -> fallback to uint8."""
        w, a = self._resolve(precision="w4a16")
        assert w == "uint8"
        assert a == "uint8"

    # ---- Explicit flags override precision ----
    def test_explicit_weight_overrides_precision(self) -> None:
        """--weight-type int8 should override w8a16 weight default."""
        w, a = self._resolve(precision="w8a16", weight_type="int8")
        assert w == "int8"
        assert a == "uint16"

    def test_explicit_activation_overrides_precision(self) -> None:
        """--activation-type int16 should override w8a16 activation default."""
        w, a = self._resolve(precision="w8a16", activation_type="int16")
        assert w == "uint8"
        assert a == "int16"

    def test_both_explicit_override_precision(self) -> None:
        """Both explicit flags should override w8a16 defaults entirely."""
        w, a = self._resolve(precision="w8a16", weight_type="int8", activation_type="int16")
        assert w == "int8"
        assert a == "int16"

    # ---- Case insensitivity ----
    def test_w8a16_case_insensitive(self) -> None:
        """W8A16 (uppercase) should work through the CLI resolver."""
        w, a = self._resolve(precision="W8A16")
        assert w == "uint8"
        assert a == "uint16"
