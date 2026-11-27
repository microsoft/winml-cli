# Windows wheel build and validation

For native ARM64 setup and available profiles, see the
[installation guide](../../docs/getting-started/installation.md#native-windows-arm64-runtime-preview).

## Release artifacts

The [official build](../../.pipelines/modelkit-official-build.yml) produces
one source distribution from the root before telemetry key injection, then
two wheels from the same source and bundled rules after injection. Native
libraries are supplied by architecture-specific dependencies, not compiled
into the CLI wheels. The version comes from the root `pyproject.toml`.

| Destination | Assets from the same official build ID |
|---|---|
| GitHub release | `winml_cli-<version>-py3-none-win_amd64.whl`, `winml_cli-<version>-py3-none-win_arm64.whl`, `rules-v<version>.zip` |
| PyPI via ESRP | Both wheels and `winml_cli-<version>.tar.gz`; no rules zip |

Before approving publication, download **both** wheels from the pinned official
build. Record their hashes and smoke-test each in a clean environment outside the
checkout: x64 CPython 3.14 and native ARM64 CPython 3.13. Install the matching
archive with its dependencies, run `uv pip check`, `winml sys --list-ep`, and
a CPU `winml perf` inference. Verify bundled rules with `WINMLCLI_RULES_DIR`
unset; ARM64 includes the files, but its default profile cannot read Parquet.
Local builds are useful for development, not substitutes for this release gate.

## Compare local wheels

From the repository root, prepare the x64 control:

```powershell
uv venv --python cpython-3.14-windows-x86_64-none temp\comparison-x64
$env:WINML_X64_PYTHON = (Resolve-Path temp\comparison-x64\Scripts\python.exe).Path
uv build --wheel "-C=--build-option=--plat-name=win_amd64" --out-dir temp\comparison-x64\dist
$wheel = Get-Item temp\comparison-x64\dist\winml_cli-*-win_amd64.whl
uv pip install --python $env:WINML_X64_PYTHON $wheel.FullName
```

Then replace the ARM editable install with the wheel built from the same checkout:

```powershell
cd _envs\arm64
uv sync --extra export
uv build --wheel --out-dir dist
$wheel = Get-Item dist\winml_cli-*-win_arm64.whl
uv pip install --python .venv\Scripts\python.exe --no-index --only-binary :all: --reinstall-package winml-cli "$($wheel.FullName)[export]"
uv pip check --python .venv\Scripts\python.exe
uv run --no-sync python -I -m pytest -q -s tests\test_wheel_comparison.py
```

The [comparison test](tests/test_wheel_comparison.py) checks installed payload
hashes, exercises each CLI outside the checkout, and writes per-workload results
to `dist\wheel-comparison.json`. Set `WINML_CACHED_HF_ID` to a cached Hugging
Face model ID for offline export and build, `WINML_LOCAL_GGUF` to a GGUF file
for generation, and `WINMLCLI_RULES_DIR` to a directory containing
`OpenVINOExecutionProvider_CPU\*.parquet` for rule-backed analysis. QNN may
provision its provider on first use. A passing smoke baseline does not mean
every optional workload passed; inspect the report. Run `uv sync --extra export`
again to restore the editable ARM install.
