# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for modelkit.sysinfo.pdh_adapters module.

Covers the *pure* logic that doesn't touch the PDH ctypes plumbing:

  * ``AdapterInfo.is_npu`` — the engine-type fingerprint that distinguishes an
    NPU (Compute/Neural engines only) from a GPU.
  * ``_build_adapters`` — grouping PDH GPU-Engine instance strings by LUID.
  * ``_parse_multi_sz`` — null-separated, double-null-terminated buffer parsing.
  * ``_format_pdh_luid`` — decimal LUID -> ``0xHHHHHHHH_0xHHHHHHHH`` hi/lo split.
  * ``discover_npu_luid`` / ``discover_gpu_luid(s)`` — selection over a fake
    adapter map (enumerate_adapters patched out).

The actual ``PdhEnumObjectItemsW`` calls and real-hardware discovery are left to
``tests/e2e/test_perf_e2e.py``, which exercises them on hosts that have the
device. These unit tests run anywhere and hit the NPU branch the e2e suite
skips when no NPU is present.
"""

from __future__ import annotations

import ctypes
import sys

import pytest


if sys.platform != "win32":
    # The module hard-raises ImportError off Windows (it binds pdh.dll at import).
    pytest.skip("pdh_adapters requires Windows (pdh.dll)", allow_module_level=True)

from winml.modelkit.sysinfo import pdh_adapters
from winml.modelkit.sysinfo.pdh_adapters import (
    AdapterInfo,
    _build_adapters,
    _format_pdh_luid,
    _parse_multi_sz,
    discover_gpu_luid,
    discover_gpu_luids,
    discover_npu_luid,
)


def _instance(luid: str, eng: int, engtype: str, pid: int = 1234, phys: int = 0) -> str:
    """Render a PDH GPU-Engine instance name as Windows reports it."""
    return f"pid_{pid}_luid_{luid}_phys_{phys}_eng_{eng}_engtype_{engtype}"


class TestAdapterInfoIsNpu:
    """Tests for the NPU engine-type fingerprint."""

    @pytest.mark.parametrize(
        "engine_types",
        [
            {"Compute"},
            {"Compute_0", "Compute_1"},
            {"Neural"},
            {"Compute", "Neural"},
        ],
    )
    def test_compute_only_is_npu(self, engine_types: set[str]) -> None:
        assert AdapterInfo(luid="0x0_0x1", engine_types=engine_types).is_npu is True

    @pytest.mark.parametrize(
        "engine_types",
        [
            {"3D", "Compute", "Copy", "VideoDecode"},  # discrete GPU
            {"3D"},
            {"Copy"},
            {"Compute", "3D"},  # any non-compute engine disqualifies
        ],
    )
    def test_gpu_engines_not_npu(self, engine_types: set[str]) -> None:
        assert AdapterInfo(luid="0x0_0x1", engine_types=engine_types).is_npu is False

    def test_no_engines_not_npu(self) -> None:
        """An adapter with no engine instances is not an NPU (len == 0 guard)."""
        assert AdapterInfo(luid="0x0_0x1").is_npu is False


class TestBuildAdapters:
    """Tests for grouping instance strings into per-LUID adapters."""

    def test_groups_engines_by_luid(self) -> None:
        instances = [
            _instance("0x00000000_0x0000A111", 0, "3D"),
            _instance("0x00000000_0x0000A111", 1, "Copy"),
            _instance("0x00000000_0x0000B222", 0, "Compute"),
        ]

        adapters = _build_adapters(instances)

        assert set(adapters) == {"0x00000000_0x0000A111", "0x00000000_0x0000B222"}
        assert adapters["0x00000000_0x0000A111"].engine_types == {"3D", "Copy"}
        assert adapters["0x00000000_0x0000B222"].engine_types == {"Compute"}
        assert adapters["0x00000000_0x0000A111"].is_npu is False
        assert adapters["0x00000000_0x0000B222"].is_npu is True

    def test_multiword_engtype_preserved(self) -> None:
        """engtype is everything after the marker, so ``Compute_0`` isn't truncated."""
        adapters = _build_adapters([_instance("0x0_0x1", 2, "Compute_0")])
        assert adapters["0x0_0x1"].engine_types == {"Compute_0"}
        assert adapters["0x0_0x1"].engine_map["Compute_0"] == (2, "Compute_0")

    def test_first_engine_number_wins_per_type(self) -> None:
        """Repeated engtype keeps the first (eng_num, engtype) seen in engine_map."""
        adapters = _build_adapters(
            [
                _instance("0x0_0x1", 3, "Compute"),
                _instance("0x0_0x1", 7, "Compute"),
            ]
        )
        assert adapters["0x0_0x1"].engine_map["Compute"] == (3, "Compute")

    def test_skips_malformed_instances(self) -> None:
        """Instances without luid/engtype markers are ignored, not fatal."""
        adapters = _build_adapters(
            [
                "Total",  # no markers
                "pid_1_luid_0x0_0x1_phys_0_eng_0",  # has luid, missing engtype
                _instance("0x0_0x2", 0, "Compute"),
            ]
        )
        assert set(adapters) == {"0x0_0x2"}

    def test_empty_input(self) -> None:
        assert _build_adapters([]) == {}


class TestParseMultiSz:
    """Tests for the null-separated, double-null-terminated buffer parser."""

    def test_parses_items_and_stops_at_double_null(self) -> None:
        raw = "GPU0\x00GPU1\x00\x00"
        buf = ctypes.create_unicode_buffer(raw)
        assert _parse_multi_sz(buf, len(raw)) == ["GPU0", "GPU1"]

    def test_single_item(self) -> None:
        raw = "OnlyOne\x00\x00"
        buf = ctypes.create_unicode_buffer(raw)
        assert _parse_multi_sz(buf, len(raw)) == ["OnlyOne"]

    def test_empty_buffer(self) -> None:
        raw = "\x00\x00"
        buf = ctypes.create_unicode_buffer(raw)
        assert _parse_multi_sz(buf, len(raw)) == []


class TestFormatPdhLuid:
    """Tests for decimal LUID -> hi/lo hex formatting."""

    @pytest.mark.parametrize(
        ("decimal_luid", "expected"),
        [
            ("0", "0x00000000_0x00000000"),
            # Low 32 bits only: 0x00018393 == 99219.
            ("99219", "0x00000000_0x00018393"),
            # High bit set: (1 << 32) | 0x00018393 == 4295066515.
            ("4295066515", "0x00000001_0x00018393"),
            # Full 64-bit all-ones.
            (str((1 << 64) - 1), "0xFFFFFFFF_0xFFFFFFFF"),
        ],
    )
    def test_format(self, decimal_luid: str, expected: str) -> None:
        assert _format_pdh_luid(decimal_luid) == expected

    def test_non_integer_raises(self) -> None:
        """Callers are documented to pre-validate; a non-int still raises ValueError."""
        with pytest.raises(ValueError):
            _format_pdh_luid("not-a-number")


class TestDiscovery:
    """Tests for discover_npu_luid / discover_gpu_luid(s) over a fake adapter map."""

    def _fake_map(self) -> dict[str, AdapterInfo]:
        return {
            "0x00000000_0x0000A111": AdapterInfo(
                luid="0x00000000_0x0000A111", engine_types={"3D", "Copy", "VideoDecode"}
            ),
            "0x00000000_0x0000B222": AdapterInfo(
                luid="0x00000000_0x0000B222", engine_types={"Compute"}
            ),
        }

    def test_discover_npu_luid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pdh_adapters, "enumerate_adapters", self._fake_map)
        assert discover_npu_luid() == "0x00000000_0x0000B222"

    def test_discover_npu_luid_none_when_no_npu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        only_gpu = {"0x0_0x1": AdapterInfo(luid="0x0_0x1", engine_types={"3D"})}
        monkeypatch.setattr(pdh_adapters, "enumerate_adapters", lambda: only_gpu)
        assert discover_npu_luid() is None

    def test_discover_gpu_luids_excludes_npu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pdh_adapters, "enumerate_adapters", self._fake_map)
        # Only the 3D adapter; the Compute-only one is an NPU and must be excluded.
        assert discover_gpu_luids() == ["0x00000000_0x0000A111"]

    def test_discover_gpu_luid_returns_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pdh_adapters, "enumerate_adapters", self._fake_map)
        assert discover_gpu_luid() == "0x00000000_0x0000A111"

    def test_discovery_swallows_enumeration_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A RuntimeError from enumeration degrades to None/[] rather than propagating."""

        def boom() -> dict[str, AdapterInfo]:
            raise RuntimeError("PDH sizing failed")

        monkeypatch.setattr(pdh_adapters, "enumerate_adapters", boom)
        assert discover_npu_luid() is None
        assert discover_gpu_luids() == []
        assert discover_gpu_luid() is None
