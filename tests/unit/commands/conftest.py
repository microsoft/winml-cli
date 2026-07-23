# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared fixtures for command tests.

Re-uses the compile-tests autouse fixture (resolve_device stubbed with
EPDeviceTarget semantics + incompatible-pair rejection) so command tests
that invoke the compile/build CLI don't hit the real host resolver.
"""

# Import the fixture from the sibling conftest so it applies here too.
from tests.unit.compiler.conftest import mock_compile_resolution  # noqa: F401
