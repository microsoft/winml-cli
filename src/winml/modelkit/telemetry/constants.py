# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Telemetry constants.

INSTRUMENTATION_KEY is intentionally empty in source. The official build
pipeline replaces it with the real iKey before packaging the wheel. Tests
that need to exercise the emission path monkeypatch this constant.
"""

INSTRUMENTATION_KEY = ""
