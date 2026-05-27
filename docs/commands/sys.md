# winml sys

> Inspect your machine — devices, EPs, SDKs, runtime versions at a glance.

## When to use this

Run `winml sys` before starting any export or build workflow to confirm that the
required ML libraries are installed and that the target hardware is visible. It is
also the first command to run when diagnosing an unexpected export failure.

## Synopsis

```bash
$ winml sys [options]
```

## Flags

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--format` | `-f` | `text` \| `json` \| `compact` | `text` | Output format. `text` renders rich tables, `json` emits machine-readable JSON, `compact` prints a single-line summary. |
| `--list-device` | — | flag | `false` | List available compute devices (NPU, GPU, CPU) in priority order instead of showing the full system report. |
| `--list-ep` | — | flag | `false` | List available ONNX Runtime execution providers instead of showing the full system report. Can be combined with `--list-device`. |
| `--verbose` | `-v` | flag | `false` | Surface additional diagnostic sections: Backend SDKs and Export Readiness. |
| `--help` | `-h` | flag | — | Show help and exit. |

> `winml sys` takes no `--model`, `--device`, `--ep`, `--task`, or `--precision`
> arguments. It describes the host environment, not a specific model.

## How it works

`winml sys` queries Python's `platform` and `importlib.metadata` modules to report
library versions, then probes PyTorch for CUDA availability and GPU device names.
Backend SDK detection checks for `QNN_SDK_ROOT` / `QAIRT_SDK_ROOT` environment
variables (QNN) and attempts to import `openvino` (OpenVINO). Device enumeration
queries hardware directly in NPU > GPU > CPU priority order, while EP enumeration
merges the WinML EP registry with ONNX Runtime's `get_available_providers()`. When
`--format json` is used the full report — including devices and EPs — is emitted as
a single JSON object, making it easy to capture in CI pipelines.

## Examples

```bash
# Full human-readable system report
$ winml sys
```

```text
╭──────────────────────────────────╮
│   winml-cli System Information    │
╰──────────────────────────────────╯

Environment
  Python Version    3.11.9
  Python Executable C:\...\python.exe
  OS                Windows 11
  Machine           AMD64

ML Libraries
  Library        Version   Status
  torch          2.4.0     OK
  transformers   4.44.0    OK
  onnx           1.16.1    OK
  ...

Available Devices (priority order)
  #1  NPU   Qualcomm(R) AI 100
  #2  GPU   NVIDIA GeForce RTX 4090
  #3  CPU   AMD Ryzen 9 7940HS

Available Execution Providers
  QNNExecutionProvider           -> NPU
  DmlExecutionProvider           -> GPU
  CPUExecutionProvider           -> CPU
```

```bash
# Compact one-liner — useful for CI logs
$ winml sys --format compact
```

```bash
# Machine-readable JSON — pipe to jq or save for later comparison
$ winml sys --format json > env.json
```

```bash
# Only list devices — skip everything else
$ winml sys --list-device
```

```bash
# List EPs as JSON — useful for scripting EP selection
$ winml sys --list-ep --format json
```

## Common pitfalls

- **QNN SDK not found even though it is installed.** The detection relies on the
  `QNN_SDK_ROOT` or `QAIRT_SDK_ROOT` environment variables. If neither is set,
  `winml sys` will report the SDK as absent even if the binaries exist on disk.
  Set the variable and re-run.
- **`--list-device` and `--list-ep` suppress the full report.** When either flag is
  present, only the requested section is printed. Omit both flags to see the
  complete system report.
- **`--format compact` omits device and EP tables.** The compact format is designed
  for single-line log entries and does not include device or EP details. Use `text`
  or `json` when you need the full picture.
- **CUDA shown as unavailable on a machine with a GPU.** PyTorch must be installed
  with CUDA support (`torch+cuXXX`). A CPU-only torch wheel will always report
  `cuda_available: false`.

## See also

- [ONNX & Execution Providers](../concepts/eps-and-devices.md) — background on EPs and
  how `--device` / `--ep` flags interact
- [inspect.md](inspect.md) — inspect a specific HuggingFace model's compatibility
- [hub.md](hub.md) — browse the curated catalog of validated models
- [How winml-cli Works](../concepts/how-it-works.md) — end-to-end pipeline overview
