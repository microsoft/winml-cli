# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Telemetry constants.

INSTRUMENTATION_KEY is intentionally empty in source. The official build
pipeline replaces it with the real iKey before packaging the wheel. Tests
that need to exercise the emission path monkeypatch this constant.

TELEMETRY_ENABLED is the in-source master switch. Flip it to ``False``
to disable the entire telemetry stack regardless of iKey / consent /
stored config — useful for local debugging and for emergency disable
in a hotfix without rebuilding consent state.
"""

INSTRUMENTATION_KEY = ""

TELEMETRY_ENABLED = True
