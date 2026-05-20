# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for EP constants, normalize_ep_name, and extract_ep_options."""

from __future__ import annotations

import pytest

from winml.modelkit.utils.constants import (
    ALL_EP_NAMES,
    EP_ALIAS_NAMES,
    EP_ALIASES,
    EP_NAME_TO_ALIAS,
    EP_NAMES,
    SUPPORTED_EPS,
    extract_ep_options,
    normalize_ep_name,
)


class TestSupportedEPs:
    """Tests for SUPPORTED_EPS derived from sysinfo EP device map."""

    def test_matches_ep_device_map_keys(self) -> None:
        """SUPPORTED_EPS must exactly match the keys in _EP_DEVICE_MAP."""
        from winml.modelkit.sysinfo.device import get_ep_device_map

        assert set(SUPPORTED_EPS) == set(get_ep_device_map().keys())

    def test_contains_known_eps(self) -> None:
        """Spot-check that well-known EPs are present."""
        for ep in (
            "QNNExecutionProvider",
            "OpenVINOExecutionProvider",
            "VitisAIExecutionProvider",
            "CPUExecutionProvider",
            "DmlExecutionProvider",
            "NvTensorRTRTXExecutionProvider",
            "MIGraphXExecutionProvider",
        ):
            assert ep in SUPPORTED_EPS


class TestEPAliases:
    """Tests for EP_ALIASES mapping."""

    def test_all_alias_values_are_supported_eps(self) -> None:
        """Every alias must resolve to an EP in SUPPORTED_EPS."""
        for alias, full_name in EP_ALIASES.items():
            assert full_name in SUPPORTED_EPS, (
                f"Alias '{alias}' maps to '{full_name}' which is not in SUPPORTED_EPS"
            )

    def test_alias_keys_are_lowercase(self) -> None:
        """Alias keys must be lowercase for case-insensitive lookup."""
        for alias in EP_ALIASES:
            assert alias == alias.lower()

    def test_covers_every_ep_alias_literal(self) -> None:
        """Every value in the `EPAlias` Literal must have a mapping."""
        missing = set(EP_ALIAS_NAMES) - set(EP_ALIASES)
        assert not missing, f"EP_ALIASES is missing entries for: {sorted(missing)}"

    def test_no_extra_keys_outside_literal(self) -> None:
        """EP_ALIASES must not contain keys absent from the `EPAlias` Literal."""
        extra = set(EP_ALIASES) - set(EP_ALIAS_NAMES)
        assert not extra, f"EP_ALIASES has unexpected aliases: {sorted(extra)}"

    def test_values_are_subset_of_ep_names(self) -> None:
        """Every alias value must be a canonical EPName Literal value."""
        invalid = set(EP_ALIASES.values()) - set(EP_NAMES)
        assert not invalid, f"EP_ALIASES maps to non-EPName values: {sorted(invalid)}"


class TestEPNameToAlias:
    """Tests for EP_NAME_TO_ALIAS reverse mapping."""

    def test_covers_every_ep_name_literal(self) -> None:
        """Every value in the `EPName` Literal must have a reverse mapping."""
        missing = set(EP_NAMES) - set(EP_NAME_TO_ALIAS)
        assert not missing, f"EP_NAME_TO_ALIAS is missing entries for: {sorted(missing)}"

    def test_no_extra_keys_outside_literal(self) -> None:
        """EP_NAME_TO_ALIAS must not contain keys absent from the `EPName` Literal."""
        extra = set(EP_NAME_TO_ALIAS) - set(EP_NAMES)
        assert not extra, f"EP_NAME_TO_ALIAS has unexpected canonical names: {sorted(extra)}"

    def test_values_are_subset_of_ep_aliases(self) -> None:
        """Every reverse-mapped value must be a valid EPAlias Literal value."""
        invalid = set(EP_NAME_TO_ALIAS.values()) - set(EP_ALIAS_NAMES)
        assert not invalid, f"EP_NAME_TO_ALIAS maps to non-EPAlias values: {sorted(invalid)}"

    def test_round_trip_through_ep_aliases(self) -> None:
        """For every canonical name, EP_ALIASES[EP_NAME_TO_ALIAS[name]] == name."""
        for name, alias in EP_NAME_TO_ALIAS.items():
            assert EP_ALIASES[alias] == name, (
                f"Round-trip failed: EP_NAME_TO_ALIAS[{name!r}]={alias!r}, "
                f"but EP_ALIASES[{alias!r}]={EP_ALIASES[alias]!r}"
            )


class TestAllEPNames:
    """Tests for ALL_EP_NAMES (full names + aliases)."""

    def test_contains_all_supported_eps(self) -> None:
        """ALL_EP_NAMES must include every full EP name."""
        for ep in SUPPORTED_EPS:
            assert ep in ALL_EP_NAMES

    def test_contains_all_aliases(self) -> None:
        """ALL_EP_NAMES must include every alias key."""
        for alias in EP_ALIASES:
            assert alias in ALL_EP_NAMES

    def test_no_duplicates(self) -> None:
        """No entry should appear more than once."""
        assert len(ALL_EP_NAMES) == len(set(ALL_EP_NAMES))


class TestNormalizeEPName:
    """Tests for normalize_ep_name()."""

    def test_none_returns_none(self) -> None:
        assert normalize_ep_name(None) is None

    def test_full_name_unchanged(self) -> None:
        """Full EP names pass through unchanged."""
        assert normalize_ep_name("QNNExecutionProvider") == "QNNExecutionProvider"
        assert normalize_ep_name("CPUExecutionProvider") == "CPUExecutionProvider"
        assert normalize_ep_name("DmlExecutionProvider") == "DmlExecutionProvider"
        ep = "NvTensorRTRTXExecutionProvider"
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
            ("nv_tensorrt_rtx", "NvTensorRTRTXExecutionProvider"),
            ("migraphx", "MIGraphXExecutionProvider"),
        ],
    )
    def test_alias_resolves(self, alias: str, expected: str) -> None:
        assert normalize_ep_name(alias) == expected

    def test_alias_case_insensitive(self) -> None:
        """Aliases should resolve regardless of casing."""
        assert normalize_ep_name("QNN") == "QNNExecutionProvider"
        assert normalize_ep_name("Dml") == "DmlExecutionProvider"
        assert normalize_ep_name("NV_TENSORRT_RTX") == "NvTensorRTRTXExecutionProvider"

    def test_unknown_ep_raises(self) -> None:
        """Unrecognized names raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported execution provider name: SomeFutureEP"):
            normalize_ep_name("SomeFutureEP")


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
