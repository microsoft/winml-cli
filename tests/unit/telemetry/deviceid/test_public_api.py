# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


def test_public_api_exposes_get_or_create_device_id():
    """External code imports from winml.modelkit.telemetry.deviceid, never
    reaching into internal submodules."""
    from winml.modelkit.telemetry.deviceid import get_or_create_device_id

    assert callable(get_or_create_device_id)


def test_public_api_exposes_id_status():
    from winml.modelkit.telemetry.deviceid import IdStatus

    assert {m.name for m in IdStatus} == {"EXISTING", "NEW", "FAILED"}


def test_store_is_not_re_exported():
    """Private submodule — should not appear in the package __all__."""
    import winml.modelkit.telemetry.deviceid as pkg

    assert "_store" not in getattr(pkg, "__all__", [])
