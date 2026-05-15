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
        assert infer_ihv_from_ep_name("nvtensorrtxexecutionprovider") == IHVType.NVIDIA

    def test_unknown_ep_resolves_to_microsoft(self) -> None:
        from winml.modelkit.analyze.models.ihv_type import IHVType

        assert infer_ihv_from_ep_name("TotallyFakeEP") == IHVType.MICROSOFT

    def test_cpu_ep_resolves_to_microsoft(self) -> None:
        """CPUExecutionProvider is a Microsoft EP — should resolve to MICROSOFT."""
        from winml.modelkit.analyze.models.ihv_type import IHVType

        assert infer_ihv_from_ep_name("CPUExecutionProvider") == IHVType.MICROSOFT

    def test_dml_ep_resolves_to_microsoft(self) -> None:
        """DmlExecutionProvider is a Microsoft EP — should resolve to MICROSOFT."""
        from winml.modelkit.analyze.models.ihv_type import IHVType

        assert infer_ihv_from_ep_name("DmlExecutionProvider") == IHVType.MICROSOFT

    def test_nvidia_ep_maps_to_nvidia(self) -> None:
        """NvTensorRTRTXExecutionProvider should map to IHVType.NVIDIA."""
        from winml.modelkit.analyze.models.ihv_type import IHVType

        assert infer_ihv_from_ep_name("NvTensorRTRTXExecutionProvider") == IHVType.NVIDIA

    def test_trtrtx_ep_maps_to_nvidia(self) -> None:
        """TrtRTXExecutionProvider should map to IHVType.NVIDIA."""
        from winml.modelkit.analyze.models.ihv_type import IHVType

        assert infer_ihv_from_ep_name("TrtRTXExecutionProvider") == IHVType.NVIDIA


class TestHasRuleDataForEP:
    """Tests for has_rule_data_for_ep()."""

    def test_returns_false_for_empty_search_dir(self, tmp_path: Path) -> None:
        """No parquet rule files at all — should return False."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert has_rule_data_for_ep("DmlExecutionProvider", "GPU") is False

    def test_returns_false_for_unmatched_ep(self, tmp_path: Path) -> None:
        """Parquet files exist but none match the requested EP+device."""
        (tmp_path / "Add_OtherEP_GPU_ai.onnx_opset13.parquet").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert has_rule_data_for_ep("FakeEP", "NPU") is False

    def test_returns_false_for_nonexistent_dir(self, tmp_path: Path) -> None:
        """Search dir that does not exist on disk — should return False."""
        missing = tmp_path / "does_not_exist"
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [missing])
            assert has_rule_data_for_ep("QNNExecutionProvider", "NPU") is False

    def test_returns_true_when_parquet_exists_in_flat_layout(self, tmp_path: Path) -> None:
        """Should return True when a matching parquet file is present in root layout."""
        (tmp_path / "Add_FakeEP_NPU_ai.onnx_opset13.parquet").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert has_rule_data_for_ep("FakeEP", "NPU") is True

    def test_returns_true_when_parquet_exists_in_provider_subdir(self, tmp_path: Path) -> None:
        """Should return True for rules_dir/<EP>_<DEVICE>/*.parquet layout."""
        nested = tmp_path / "FakeEP_NPU"
        nested.mkdir()
        (nested / "Add_FakeEP_NPU_ai.onnx_opset13.parquet").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert has_rule_data_for_ep("FakeEP", "NPU") is True

    def test_device_is_case_insensitive(self, tmp_path: Path) -> None:
        """Device is uppercased internally, so 'npu' should match NPU parquet files."""
        (tmp_path / "Add_FakeEP_NPU_ai.onnx_opset13.parquet").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert has_rule_data_for_ep("FakeEP", "npu") is True

    def test_searches_multiple_dirs(self, tmp_path: Path) -> None:
        """Should find a match even if it's in the second search directory."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        # Only dir_b has parquet rules in provider subdir layout
        provider_dir = dir_b / "MyEP_GPU"
        provider_dir.mkdir()
        (provider_dir / "Mul_MyEP_GPU_ai.onnx_opset14.parquet").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [dir_a, dir_b])
            assert has_rule_data_for_ep("MyEP", "GPU") is True


class TestGetDevicesWithRuleData:
    """Tests for get_devices_with_rule_data()."""

    def test_returns_devices_from_rule_data(self, tmp_path: Path) -> None:
        """Should return all devices that have matching parquet rule files."""
        (tmp_path / "Add_TestEP_NPU_ai.onnx_opset13.parquet").touch()
        (tmp_path / "Add_TestEP_GPU_ai.onnx_opset13.parquet").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert get_devices_with_rule_data("TestEP") == ["NPU", "GPU"]

    def test_preserves_priority_order(self, tmp_path: Path) -> None:
        """Devices should be returned in NPU > GPU > CPU order."""
        (tmp_path / "Add_TestEP_CPU_ai.onnx_opset13.parquet").touch()
        (tmp_path / "Add_TestEP_NPU_ai.onnx_opset13.parquet").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert get_devices_with_rule_data("TestEP") == ["NPU", "CPU"]

    def test_falls_back_to_ep_device_map_for_dml(self, tmp_path: Path) -> None:
        """DML has no parquet rule data — should fall back to sysinfo EP device map."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert get_devices_with_rule_data("DmlExecutionProvider") == ["GPU"]

    def test_falls_back_to_ep_device_map_for_cpu(self, tmp_path: Path) -> None:
        """CPU EP has no parquet rule data — should fall back to sysinfo EP device map."""
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
        """If parquet rule data exists, EP device map should NOT be used."""
        # Imagine DML gets rule data for NPU in the future
        (tmp_path / "Add_DmlExecutionProvider_NPU_ai.onnx_opset17.parquet").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            # Rule data says NPU, not the EP device map ["GPU"]
            assert get_devices_with_rule_data("DmlExecutionProvider") == ["NPU"]
