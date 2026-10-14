"""Native DXCGC boundary validation, without GPU dependencies."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from winml.modelkit.commands._perf_dxcgc import _samples
from winml.modelkit.commands.perf import perf


@pytest.mark.parametrize('value', ['nan', 'inf', '-1', '0'])
def test_reject_invalid_samples(tmp_path, value):
    path = tmp_path / 'samples.csv'
    path.write_text(f'gpu_ms,host_ms\n{value},1\n')
    with pytest.raises(ValueError, match='invalid timings'):
        _samples(path, 1)


def test_reject_incomplete_samples(tmp_path):
    path = tmp_path / 'samples.csv'
    path.write_text('gpu_ms,host_ms\n1,2\n')
    with pytest.raises(ValueError, match='Expected 2'):
        _samples(path, 2)


def test_requires_mlir(tmp_path):
    path = tmp_path / 'model.onnx'
    path.touch()
    result = CliRunner().invoke(perf, ['-m', str(path), '--runtime', 'dxcgc'])
    assert result.exit_code != 0
    assert 'local .mlir' in result.output


def test_reject_runtime_options():
    result = CliRunner().invoke(perf, ['-m', 'x.mlir', '--runtime', 'dxcgc', '--monitor'])
    assert result.exit_code == 2
    assert '--monitor' in result.output


def test_options_require_dxcgc():
    result = CliRunner().invoke(perf, ['-m', 'x.mlir', '--dxcgc-adapter', '0'])
    assert result.exit_code == 2
    assert 'require --runtime dxcgc' in result.output


@pytest.mark.skipif(not os.environ.get('WINML_DXCGC_MODEL'), reason='Requires DXCGC GPU fixtures')
@pytest.mark.parametrize('io_mode', ['resident', 'cpu-roundtrip'])
def test_resnet_native(tmp_path, io_mode):
    args = ['-m', os.environ['WINML_DXCGC_MODEL'], '--runtime', 'dxcgc',
            '--dxcgc-host', os.environ['WINML_DXCGC_HOST'],
            '--dxcgc-dir', os.environ['WINML_DXCGC_DIR'],
            '--dxcgc-sdk-path', os.environ['WINML_DXCGC_SDK'],
            '--input-data', os.environ['WINML_DXCGC_INPUT'],
            '--dxcgc-io-mode', io_mode,
            '--iterations', '100', '--warmup', '10', '-o', str(tmp_path / 'perf.json')]
    result = CliRunner().invoke(perf, args)
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / 'perf.json').read_text())
    assert len(report['samples']['gpu_ms']) == 100
    with np.load(tmp_path / 'perf.outputs.npz') as actual, np.load(
        Path(os.environ['WINML_DXCGC_REFERENCE'])
    ) as expected:
        assert len(actual.files) == len(expected.files)
        for a, b in zip(actual.values(), expected.values(), strict=True):
            assert a.shape == b.shape
            assert a.dtype == b.dtype
            relative = np.linalg.norm(a.astype(float) - b) / np.linalg.norm(b)
            assert relative < 0.01
