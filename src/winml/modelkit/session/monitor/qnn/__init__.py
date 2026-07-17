# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Public surface for QNN op-tracing parsers.

The parser implementation lives in the private :mod:`._internal` module.
Tests and downstream consumers should import via this ``__init__.py`` to
preserve the information-hiding boundary documented in spec v2.0.1.
Direct imports from :mod:`._internal` are caught by the architecture
regression test at :mod:`tests.unit.architecture.test_qnn_imports`
unless the imported name is ``_``-prefixed (CLAUDE.md exception for
testing private internals).

Modules:

* :mod:`._internal` — private CSV + QHAS parsers (only ``qnn_monitor``
  is allowed to import non-``_``-prefixed names from here per the v2.4
  information-hiding contract).
* :mod:`.viewer` — QHAS viewer shell-out (locates the QNN SDK and runs
  the ``qnn-profile-viewer`` binary).
"""

from ._internal import parse_qhas, parse_qnn_profiling_csv


__all__ = ["parse_qhas", "parse_qnn_profiling_csv"]
