# Issues: docs/commands/sys.md

Source verified against: `src/winml/modelkit/commands/sys.py` @ 5e25579

## Critical (flag/behavior wrong; user gets error)

- **`--verbose` / `-v` flag is missing from the flag table entirely.** Source lines 653–659 define `@click.option("--verbose", "-v", is_flag=True, default=False, ...)`. The doc table lists only `--format`, `--list-device`, `--list-ep`, and `--help` — omitting `--verbose` means users reading the doc have no way to discover a functional, documented flag. The example `winml inspect -m facebook/convnext-tiny-224 -v -H` on `inspect.md` uses the same flag pattern, and `sys.py` line 699 shows `-v`/`--verbose` passed via `docstring`. Using `--verbose` surfaces Backend SDKs and Export Readiness sections (lines 392–433) that are hidden otherwise; presenting the command as having only 3 flags is actively wrong.

## Important (misleading or stale)

- **"How it works" says CUDA details are always probed via PyTorch** — source lines 218–251 show `_get_torch_info(verbose=False)` is the default, which explicitly skips `import torch` and CUDA probing (lines 235–251 are gated on `if not verbose: return info`). CUDA availability (`cuda_available`) only appears in the output when `--verbose` is passed. The doc's "How it works" says "probes PyTorch for CUDA availability and GPU device names" unconditionally, which is misleading — this only happens under `--verbose`.

- **`--format compact` pitfall says it "omits device and EP tables"** (line 106) — but the source (lines 757–774) shows compact *does* support `--list-device` and `--list-ep` and prints device/EP information in a single-line form. The pitfall is only correct for the full default report path (line 812 `elif output_format.lower() == "compact": _output_compact(info)` which skips devices/EPs), but combination of `--format compact --list-device` works and produces output. The pitfall is partially misleading.

- **"Backend SDK detection" described as part of default output** — source lines 392–433 show Backend SDKs and Export Readiness sections are only rendered under `verbose=True` (`if verbose:` guard at line 392). The "How it works" section implies these are always shown.

- **Example output shows "winml-cli System Information"** (line 49) but source line 342 renders `"WinML CLI System Information"`. Minor inconsistency in the example panel title.

## Minor (polish)

- **`--help` short form `-h`** — Click auto-adds `--help` / `-h` for all commands; listing it explicitly in the table is harmless but adds noise.
- **`sys.md` cross-links to `hub.md`** (line 117), but the actual CLI command is `winml catalog` (source: `catalog.py`), not `winml hub`. If `hub.md` documents a `winml hub` alias, verify it exists in `__init__.py`; otherwise the cross-link is confusing.

## Verified correct (key claims checked)

- `--format` flag exists with short `-f`, type `Choice(["text", "json", "compact"])`, default `"text"` → source lines 645–652.
- `--list-device` flag exists as `is_flag=True, default=False`, no short form → source lines 653–658.
- `--list-ep` flag exists as `is_flag=True, default=False`, no short form → source lines 659–664.
- QNN detection uses `QNN_SDK_ROOT` / `QAIRT_SDK_ROOT` env vars → source lines 261–272.
- OpenVINO detection via `import openvino` → source lines 283–290.
- `--format json` emits devices and EPs → source lines 801–812.
- Device enumeration in NPU > GPU > CPU priority order → source lines 495–500.
- EP enumeration merges WinML registry with ORT `get_available_providers()` → source lines 592–623.
