# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ep_utils: infer_ihv_from_ep_name, has_rule_data_for_ep, get_devices_with_rule_data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from winml.modelkit.analyze.utils import (
    get_devices_with_rule_data,
    has_rule_data_for_ep,
    infer_ihv_from_ep_name,
)


if TYPE_CHECKING:
    from pathlib import Path

_PATCH_TARGET = "winml.modelkit.analyze.utils.rule_loader.get_runtime_rules_search_dirs"


class TestInferIHVFromEPName:
    """Tests for infer_ihv_from_ep_name()."""

    def test_qnn(self) -> None:
        from winml.modelkit.analyze.models.ihv_type import IHVType

        assert infer_ihv_from_ep_name("QNNExecutionProvider") == IHVType.QC

    def test_openvino(self) -> None:
        from winml.modelkit.analyze.models.ihv_type import IHVType

        assert infer_ihv_from_ep_name("OpenVINOExecutionProvider") == IHVType.INTEL

    def test_vitisai(self) -> None:
        from winml.modelkit.analyze.models.ihv_type import IHVType

        assert infer_ihv_from_ep_name("VitisAIExecutionProvider") == IHVType.AMD

    def test_migraphx_maps_to_amd(self) -> None:
        """MIGraphX is an AMD EP — should map to IHVType.AMD."""
        from winml.modelkit.analyze.models.ihv_type import IHVType

        assert infer_ihv_from_ep_name("MIGraphXExecutionProvider") == IHVType.AMD

    def test_case_insensitive(self) -> None:
        from winml.modelkit.analyze.models.ihv_type import IHVType

        assert infer_ihv_from_ep_name("qnnexecutionprovider") == IHVType.QC
        assert infer_ihv_from_ep_name("OPENVINOEXECUTIONPROVIDER") == IHVType.INTEL
        assert infer_ihv_from_ep_name("vitisaiexecutionprovider") == IHVType.AMD

    def test_unknown_ep_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown execution provider"):
            infer_ihv_from_ep_name("TotallyFakeEP")

    def test_cpu_ep_raises(self) -> None:
        """CPUExecutionProvider has no IHV — should raise."""
        with pytest.raises(ValueError, match="Unknown execution provider"):
            infer_ihv_from_ep_name("CPUExecutionProvider")

    def test_dml_ep_raises(self) -> None:
        """DmlExecutionProvider is Microsoft, not an IHV — should raise."""
        with pytest.raises(ValueError, match="Unknown execution provider"):
            infer_ihv_from_ep_name("DmlExecutionProvider")

    def test_nvidia_ep_raises(self) -> None:
        """NvTensorRTRTXExecutionProvider has no IHV mapping — should raise."""
        with pytest.raises(ValueError, match="Unknown execution provider"):
            infer_ihv_from_ep_name("NvTensorRTRTXExecutionProvider")


class TestHasRuleDataForEP:
    """Tests for has_rule_data_for_ep()."""

    def test_returns_false_for_empty_search_dir(self, tmp_path: Path) -> None:
        """No zip files at all — should return False."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert has_rule_data_for_ep("DmlExecutionProvider", "GPU") is False

    def test_returns_false_for_unmatched_ep(self, tmp_path: Path) -> None:
        """Zip files exist but none match the requested EP+device."""
        (tmp_path / "OtherEP_GPU_ai_onnx_opset13.zip").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert has_rule_data_for_ep("FakeEP", "NPU") is False

    def test_returns_false_for_nonexistent_dir(self, tmp_path: Path) -> None:
        """Search dir that does not exist on disk — should return False."""
        missing = tmp_path / "does_not_exist"
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [missing])
            assert has_rule_data_for_ep("QNNExecutionProvider", "NPU") is False

    def test_returns_true_when_zip_exists(self, tmp_path: Path) -> None:
        """Should return True when a matching zip file is present."""
        (tmp_path / "FakeEP_NPU_ai_onnx_opset13.zip").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert has_rule_data_for_ep("FakeEP", "NPU") is True

    def test_device_is_case_insensitive(self, tmp_path: Path) -> None:
        """Device is uppercased internally, so 'npu' should match 'NPU' prefix."""
        (tmp_path / "FakeEP_NPU_ai_onnx_opset13.zip").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert has_rule_data_for_ep("FakeEP", "npu") is True

    def test_searches_multiple_dirs(self, tmp_path: Path) -> None:
        """Should find a match even if it's in the second search directory."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        # Only dir_b has the zip
        (dir_b / "MyEP_GPU_rules.zip").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [dir_a, dir_b])
            assert has_rule_data_for_ep("MyEP", "GPU") is True


class TestGetDevicesWithRuleData:
    """Tests for get_devices_with_rule_data()."""

    def test_returns_devices_from_rule_data(self, tmp_path: Path) -> None:
        """Should return all devices that have matching rule zips."""
        (tmp_path / "TestEP_NPU_opset13.zip").touch()
        (tmp_path / "TestEP_GPU_opset13.zip").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert get_devices_with_rule_data("TestEP") == ["NPU", "GPU"]

    def test_preserves_priority_order(self, tmp_path: Path) -> None:
        """Devices should be returned in NPU > GPU > CPU order."""
        (tmp_path / "TestEP_CPU_opset13.zip").touch()
        (tmp_path / "TestEP_NPU_opset13.zip").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert get_devices_with_rule_data("TestEP") == ["NPU", "CPU"]

    def test_falls_back_to_ep_device_map_for_dml(self, tmp_path: Path) -> None:
        """DML has no rule zips — should fall back to sysinfo EP device map."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert get_devices_with_rule_data("DmlExecutionProvider") == ["GPU"]

    def test_falls_back_to_ep_device_map_for_cpu(self, tmp_path: Path) -> None:
        """CPU EP has no rule zips — should fall back to sysinfo EP device map."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert get_devices_with_rule_data("CPUExecutionProvider") == ["CPU"]

    def test_falls_back_to_ep_device_map_multi_device(self, tmp_path: Path) -> None:
        """OpenVINO supports npu/gpu/cpu — fallback should return all three."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert get_devices_with_rule_data("OpenVINOExecutionProvider") == [
                "NPU",
                "GPU",
                "CPU",
            ]

    def test_returns_empty_for_unknown_ep(self, tmp_path: Path) -> None:
        """Completely unknown EP with no rule data and no EP device map entry."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert get_devices_with_rule_data("TotallyFakeEP") == []

    def test_rule_data_takes_precedence_over_ep_device_map(self, tmp_path: Path) -> None:
        """If rule zips exist, EP device map should NOT be used."""
        # Imagine DML gets rule data for NPU in the future
        (tmp_path / "DmlExecutionProvider_NPU_opset17.zip").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            # Rule data says NPU, not the EP device map ["GPU"]
            assert get_devices_with_rule_data("DmlExecutionProvider") == ["NPU"]
