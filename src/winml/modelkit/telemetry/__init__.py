# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""ModelKit telemetry - OneCollector-backed CLI usage and error reporting.

Public surface consists of :class:`Telemetry` only. The ``ActionGroup``
auto-wrap (Phase 3) will be re-exported here as well when it lands.
"""

from .telemetry import Telemetry


__all__ = ["Telemetry"]
