# Issues: docs/getting-started/installation.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical
- Python version wrong: doc states `3.10` and claims `requires-python = ">=3.10,<3.11"` (installation.md:3, 11), but `pyproject.toml` at 5e25579 declares `requires-python = ">=3.11,<3.12"`. The install step (`uv python install 3.10`) and the "Verify" expected output (`Python Version 3.10.x`) are also wrong as a result.

## Important
- "No NPU?" callout claims `winml eval` accepts only `cpu|gpu|npu` (no `auto`) (installation.md:16). This is **incorrect**: `eval.py` defines `--device` as `click.Choice(["auto", "cpu", "gpu", "npu"])` with `default="auto"` — `auto` is a valid value for `winml eval`.
- `winml sys --list-device --list-ep` flags: both `--list-device` and `--list-ep` exist in `sys.py` (lines with `@click.option("--list-device", ...)` and `@click.option("--list-ep", ...)`), so this is not an error, but the quickstart.md description (quoted here as context) says these flags "skip SDK versions and Python environment details" — that is not the behavior when both are passed; the full sysinfo is **not** run, only the device/EP lists are printed. Not an issue in installation.md itself.

## Minor
- The `--extra qnn` footnote claims `onnxruntime-qnn` requires Python 3.11+ and is "reserved for future use" (installation.md:70). `pyproject.toml` at 5e25579 already gates the dep on `python_version>='3.11'` and the project itself requires 3.11+, so the "reserved for future use" framing is inaccurate — it is already effective on the required Python version.
