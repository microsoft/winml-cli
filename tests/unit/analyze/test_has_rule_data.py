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
    has_any_rule_data,
    has_rule_data_for_ep,
    infer_ihv_from_ep_name,
)


if TYPE_CHECKING:
    from pathlib import Path

_PATCH_TARGET = "winml.modelkit.analyze.utils.rule_loader.get_runtime_rules_search_dirs"


class TestInferIHVFromEPName:
    """Tests for infer_ihv_from_ep_name()."""

    def test_all_known_eps_resolve(self) -> None:
        """Every canonical EPName maps to a valid IHVType (map covers the Literal)."""
        from winml.modelkit.analyze.models.ihv_type import IHVType
        from winml.modelkit.utils.constants import EP_NAMES

        for ep in EP_NAMES:
            assert isinstance(infer_ihv_from_ep_name(ep), IHVType)

    def test_unknown_ep_raises(self) -> None:
        """Unknown EP names raise rather than silently defaulting."""
        with pytest.raises(ValueError, match="unknown EP name"):
            infer_ihv_from_ep_name("TotallyFakeEP")

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

    def test_alias_resolves(self) -> None:
        """Shorthand aliases are normalized before lookup (EPNameOrAlias)."""
        from winml.modelkit.analyze.models.ihv_type import IHVType

        assert infer_ihv_from_ep_name("openvino") == IHVType.INTEL
        assert infer_ihv_from_ep_name("qnn") == IHVType.QC

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

    def test_cuda_ep_maps_to_nvidia(self) -> None:
        """CUDAExecutionProvider should map to IHVType.NVIDIA."""
        from winml.modelkit.analyze.models.ihv_type import IHVType

        assert infer_ihv_from_ep_name("CUDAExecutionProvider") == IHVType.NVIDIA


class TestHasRuleDataForEP:
    """Tests for has_rule_data_for_ep()."""

    def test_returns_false_for_empty_search_dir(self, tmp_path: Path) -> None:
        """No parquet rule files at all — should return False."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert has_rule_data_for_ep("DmlExecutionProvider", "GPU") is False

    def test_returns_false_for_unmatched_ep(self, tmp_path: Path) -> None:
        """Parquet files exist but none match the requested EP+device."""
        other_dir = tmp_path / "OtherEP_GPU"
        other_dir.mkdir()
        (other_dir / "Add_OtherEP_GPU_ai.onnx_opset13.parquet").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert has_rule_data_for_ep("FakeEP", "NPU") is False

    def test_returns_false_for_nonexistent_dir(self, tmp_path: Path) -> None:
        """Search dir that does not exist on disk — should return False."""
        missing = tmp_path / "does_not_exist"
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [missing])
            assert has_rule_data_for_ep("QNNExecutionProvider", "NPU") is False

    def test_returns_false_when_parquet_exists_in_flat_layout(self, tmp_path: Path) -> None:
        """Flat parquet is ignored; provider subdirectory layout is required."""
        (tmp_path / "Add_FakeEP_NPU_ai.onnx_opset13.parquet").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert has_rule_data_for_ep("FakeEP", "NPU") is False

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
        nested = tmp_path / "FakeEP_NPU"
        nested.mkdir()
        (nested / "Add_FakeEP_NPU_ai.onnx_opset13.parquet").touch()
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


class TestHasAnyRuleData:
    """Tests for has_any_rule_data()."""

    def test_returns_false_for_empty_search_dir(self, tmp_path: Path) -> None:
        """No parquet files in supported layouts should return False."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert has_any_rule_data() is False

    def test_returns_false_for_flat_layout(self, tmp_path: Path) -> None:
        """Flat parquet under search dir should be ignored."""
        (tmp_path / "AnyEP_NPU_dummy.parquet").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert has_any_rule_data() is False

    def test_returns_true_for_provider_subdir_layout(self, tmp_path: Path) -> None:
        """Provider subdirectory parquet should return True."""
        provider_dir = tmp_path / "AnyEP_NPU"
        provider_dir.mkdir()
        (provider_dir / "Add_AnyEP_NPU_ai.onnx_opset13.parquet").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert has_any_rule_data() is True

    def test_returns_true_for_non_ep_device_second_level_layout(self, tmp_path: Path) -> None:
        """Any second-level parquet should count as available data."""
        unrelated_dir = tmp_path / "nested"
        unrelated_dir.mkdir()
        (unrelated_dir / "Add_AnyEP_NPU_ai.onnx_opset13.parquet").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert has_any_rule_data() is True


class TestGetDevicesWithRuleData:
    """Tests for get_devices_with_rule_data()."""

    def test_returns_devices_from_rule_data(self, tmp_path: Path) -> None:
        """Should return all devices that have matching parquet rule files."""
        npu_dir = tmp_path / "TestEP_NPU"
        gpu_dir = tmp_path / "TestEP_GPU"
        npu_dir.mkdir()
        gpu_dir.mkdir()
        (npu_dir / "Add_TestEP_NPU_ai.onnx_opset13.parquet").touch()
        (gpu_dir / "Add_TestEP_GPU_ai.onnx_opset13.parquet").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            assert get_devices_with_rule_data("TestEP") == ["NPU", "GPU"]

    def test_preserves_priority_order(self, tmp_path: Path) -> None:
        """Devices should be returned in NPU > GPU > CPU order."""
        cpu_dir = tmp_path / "TestEP_CPU"
        npu_dir = tmp_path / "TestEP_NPU"
        cpu_dir.mkdir()
        npu_dir.mkdir()
        (cpu_dir / "Add_TestEP_CPU_ai.onnx_opset13.parquet").touch()
        (npu_dir / "Add_TestEP_NPU_ai.onnx_opset13.parquet").touch()
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
        """OpenVINO supports npu/gpu/cpu — fallback should return all three.

        Order matches ``EP_SUPPORTED_DEVICES["OpenVINOExecutionProvider"]``:
        NPU first (the canonical default for OpenVINO), then GPU, then CPU.
        """
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
        npu_dir = tmp_path / "DmlExecutionProvider_NPU"
        npu_dir.mkdir()
        (npu_dir / "Add_DmlExecutionProvider_NPU_ai.onnx_opset17.parquet").touch()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_PATCH_TARGET, lambda: [tmp_path])
            # Rule data says NPU, not the EP device map ["GPU"]
            assert get_devices_with_rule_data("DmlExecutionProvider") == ["NPU"]
