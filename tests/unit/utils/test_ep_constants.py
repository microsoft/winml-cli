# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ``normalize_ep_name`` and ``extract_ep_options``.

T-16 migrated these helpers from ``utils.constants`` (deleted) into
``utils.cli``; the previous ``utils.constants`` module also exported
``ALL_EP_NAMES`` / ``EP_ALIASES`` / ``SUPPORTED_EPS`` constants in an
earlier refactor that this test file's pre-existing ``TestSupportedEPs``,
``TestEPAliases``, and ``TestAllEPNames`` classes targeted — those
symbols were already gone at the T-16 baseline (collection-time
ImportError on HEAD), so the broken classes are dropped here together
with the module move.
"""

from __future__ import annotations

import pytest

from winml.modelkit.utils.cli import extract_ep_options, normalize_ep_name


class TestNormalizeEPName:
    """Tests for normalize_ep_name()."""

    def test_none_returns_none(self) -> None:
        assert normalize_ep_name(None) is None

    def test_full_name_unchanged(self) -> None:
        """Full EP names pass through unchanged."""
        assert normalize_ep_name("QNNExecutionProvider") == "QNNExecutionProvider"
        assert normalize_ep_name("CPUExecutionProvider") == "CPUExecutionProvider"
        assert normalize_ep_name("DmlExecutionProvider") == "DmlExecutionProvider"
        ep = "NvTensorRtRtxExecutionProvider"
        assert normalize_ep_name(ep) == ep
        assert normalize_ep_name("MIGraphXExecutionProvider") == "MIGraphXExecutionProvider"

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("qnn", "QNNExecutionProvider"),
            ("openvino", "OpenVINOExecutionProvider"),
            ("ov", "OpenVINOExecutionProvider"),
            ("vitisai", "VitisAIExecutionProvider"),
            ("vitis", "VitisAIExecutionProvider"),
            ("cpu", "CPUExecutionProvider"),
            ("dml", "DmlExecutionProvider"),
            ("nv_tensorrt_rtx", "NvTensorRtRtxExecutionProvider"),
            ("migraphx", "MIGraphXExecutionProvider"),
        ],
    )
    def test_alias_resolves(self, alias: str, expected: str) -> None:
        assert normalize_ep_name(alias) == expected

    def test_alias_case_insensitive(self) -> None:
        """Aliases should resolve regardless of casing."""
        assert normalize_ep_name("QNN") == "QNNExecutionProvider"
        assert normalize_ep_name("Dml") == "DmlExecutionProvider"
        assert normalize_ep_name("NV_TENSORRT_RTX") == "NvTensorRtRtxExecutionProvider"

    def test_unknown_ep_returned_as_is(self) -> None:
        """Unrecognized names are returned unchanged for downstream validation."""
        assert normalize_ep_name("SomeFutureEP") == "SomeFutureEP"


class TestExtractEPOptions:
    """Tests for extract_ep_options()."""

    def test_extracts_single_prefix(self) -> None:
        assert extract_ep_options({"qnn_qairt": "/path"}) == {"qairt": "/path"}

    def test_extracts_multiple_options_same_prefix(self) -> None:
        result = extract_ep_options({"qnn_qairt": "/path", "qnn_backend": "htp"})
        assert result == {"qairt": "/path", "backend": "htp"}

    def test_ignores_non_ep_params(self) -> None:
        result = extract_ep_options({"model": "foo.onnx", "verbose": "1"})
        assert result == {}

    def test_ignores_none_values(self) -> None:
        result = extract_ep_options({"qnn_qairt": None, "qnn_backend": "htp"})
        assert result == {"backend": "htp"}

    def test_mixed_ep_and_non_ep(self) -> None:
        result = extract_ep_options(
            {
                "qnn_qairt": "/sdk",
                "model": "m.onnx",
                "ov_device": "GPU",
                "verbose": "1",
            }
        )
        assert result == {"qairt": "/sdk", "device": "GPU"}

    def test_converts_values_to_str(self) -> None:
        result = extract_ep_options({"qnn_threads": 4})
        assert result == {"threads": "4"}

    def test_empty_input(self) -> None:
        assert extract_ep_options({}) == {}

    def test_param_without_underscore_ignored(self) -> None:
        """Params that match an alias but have no underscore separator are skipped."""
        result = extract_ep_options({"qnn": "value"})
        assert result == {}

    def test_new_aliases_work(self) -> None:
        """Newly added alias migraphx should be recognized as prefix."""
        result = extract_ep_options({"migraphx_target": "gpu"})
        assert result == {"target": "gpu"}

    def test_underscore_alias_not_matched_as_prefix(self) -> None:
        """nv_tensorrt_rtx contains underscores so split('_', 1) won't match it."""
        result = extract_ep_options({"nv_tensorrt_rtx_precision": "fp16"})
        assert result == {}
