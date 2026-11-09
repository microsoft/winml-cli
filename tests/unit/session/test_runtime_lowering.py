# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from types import SimpleNamespace
from unittest.mock import Mock

import click
import pytest

from winml.modelkit.session import _runtime_lowering as lowering


@pytest.mark.parametrize("target,value", [("disable-d3d12", 1), ("d3d12", 2)])
def test_missing_projection_sets_and_reads_native_target(monkeypatch, target, value):
    seen = []

    def set_target(selected):
        seen.append(selected)
        return 0

    def get_target(pointer):
        pointer._obj.value = value
        return 0

    native = SimpleNamespace(SetCgcLoweringTarget=set_target, GetCgcLoweringTarget=get_target)
    query = Mock(return_value=native)
    monkeypatch.setattr(lowering, "_native_options", query)
    options, runtime = SimpleNamespace(), SimpleNamespace()
    lowering.set_cgc_lowering_target(options, runtime, target)
    query.assert_called_once_with(options)
    assert seen == [value]
    assert not hasattr(runtime, "CgcLoweringTarget")
    assert not hasattr(options, "cgc_lowering_target")


@pytest.mark.parametrize("failure", ["query", "set", "get", "mismatch"])
def test_native_failure_never_falls_back_to_auto(monkeypatch, failure):
    def get_target(pointer):
        pointer._obj.value = 0
        return -1 if failure == "get" else 0

    native = SimpleNamespace(
        SetCgcLoweringTarget=lambda value: -1 if failure == "set" else 0,
        GetCgcLoweringTarget=get_target,
    )
    query = Mock(return_value=native)
    if failure == "query":
        query.side_effect = RuntimeError("No native interface")
    monkeypatch.setattr(lowering, "_native_options", query)
    with pytest.raises(click.ClickException, match="Automatic lowering was not substituted"):
        lowering.set_cgc_lowering_target(SimpleNamespace(), SimpleNamespace(), "disable-d3d12")


def test_auto_does_not_require_native_interface(monkeypatch):
    query = Mock(side_effect=AssertionError("Unexpected native call"))
    monkeypatch.setattr(lowering, "_native_options", query)
    lowering.set_cgc_lowering_target(SimpleNamespace(), SimpleNamespace(), "auto")
    query.assert_not_called()


def test_complete_projection_is_preferred(monkeypatch):
    query = Mock(side_effect=AssertionError("Unexpected native call"))
    monkeypatch.setattr(lowering, "_native_options", query)
    value = object()
    options = SimpleNamespace(cgc_lowering_target=None)
    lowering.set_cgc_lowering_target(
        options, SimpleNamespace(CgcLoweringTarget=SimpleNamespace(DML=value)), "disable-d3d12"
    )
    assert options.cgc_lowering_target is value
    query.assert_not_called()
