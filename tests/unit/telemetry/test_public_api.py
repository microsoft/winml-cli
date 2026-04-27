# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


def test_public_api_exposes_telemetry():
    from winml.modelkit.telemetry import Telemetry

    assert Telemetry is not None


def test_private_submodules_not_in_all():
    import winml.modelkit.telemetry as pkg

    for name in ("_store", "consent", "constants", "utils", "library", "deviceid"):
        assert name not in getattr(pkg, "__all__", []), (
            f"internal submodule {name!r} leaked into public __all__"
        )
