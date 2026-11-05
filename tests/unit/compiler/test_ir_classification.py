from pathlib import Path
from unittest.mock import Mock

import pytest

from winml.modelkit.export.cgc import ir


@pytest.mark.parametrize('value,expected', [
    ('#cgc.interface<cgc_output>', True),
    (' #cgc.interface<cgc_output> ', True),
    ('#cgc.interface<cgc_input>', False),
    ('', False),
    ('unrecognized', False),
])
def test_output_ir_requires_module_interface(monkeypatch, value, expected):
    parse = Mock(return_value=value)
    monkeypatch.setattr(ir, '_read_mlir', parse)
    path = Path('arbitrary.mlirbc')
    assert ir.is_output_ir(path) is expected
    parse.assert_called_once_with(path, b'interface')
