# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""WinML CLI telemetry - OneCollector-backed CLI usage and error reporting.

Public surface: :class:`Telemetry` (the singleton) and :class:`ActionGroup`
(the Click group subclass that auto-instruments registered subcommands).
"""

from .click_group import ActionGroup
from .telemetry import Telemetry


__all__ = ["ActionGroup", "Telemetry"]
