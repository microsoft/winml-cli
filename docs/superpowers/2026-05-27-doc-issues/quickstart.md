# Issues: docs/getting-started/quickstart.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical
- (none)

## Important
- `winml sys --list-device --list-ep` (quickstart.md:14): the doc claims these flags "skip SDK versions and Python environment details that plain `winml sys` would include." This is misleading. In `sys.py`, when `list_device` or `list_ep` is set, the command takes a separate branch that runs *only* the device/EP listing and returns early — it does not run `_gather_system_info()` at all, so "skipping" implies it runs a subset of the normal command, when in fact it is a separate code path. This is a documentation accuracy issue but not a flag-existence issue.
- `winml inspect -m resnet50.onnx` (quickstart.md:40): `inspect.py` explicitly raises `click.ClickException("ONNX file inspection is not yet supported. Use 'winml config -m model.onnx' for ONNX build config.")` when passed a `.onnx` local file. The command as documented will fail rather than produce the shown output.

## Minor
- (none)
