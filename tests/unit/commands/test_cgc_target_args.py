# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CGC backend and EP selection are mutually exclusive at the CLI boundary."""

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.build import build
from winml.modelkit.commands.config import config


@pytest.mark.parametrize("command", [config, build], ids=["config", "build"])
@pytest.mark.parametrize("ep", ["dml", "winmlcg", "DmlExecutionProvider", "dml@bundled"])
def test_cgc_backend_rejects_ep_before_model_resolution(command, ep):
    result = CliRunner().invoke(command, ["--backend", "cgc", "--ep", ep])

    assert result.exit_code == 2
    assert "--backend cgc cannot be combined with --ep." in result.output
