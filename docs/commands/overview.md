# Commands

winml-cli exposes a CLI named `winml` with 12 subcommands covering the full
journey from model discovery to a deployment-ready artifact. Every subcommand
shares a consistent invocation style — `winml <command> [flags]` — and the
same global flags are available on the root `winml` group.

The commands group by user intent. **Discover** (`sys`, `inspect`, `hub`,
`analyze`) helps you understand your hardware and model before writing any
artifacts. **Configure** (`config`, `optimize`) produces a reusable build
configuration and tunes the ONNX graph. **Build** (`export`, `quantize`,
`compile`, `build`) runs the pipeline stages that produce deployment artifacts.
**Measure** (`perf`, `eval`) benchmarks and validates the result.

The typical workflow follows that order: run `winml sys` to confirm hardware
and EPs, then `winml inspect` or `winml hub` to verify model support. Use
`winml config` to generate a build configuration, then `winml build` to execute
the full pipeline — or chain `export` → `optimize` → `quantize` → `compile`
individually for finer control. Close with `winml perf` and `winml eval` to
measure speed and accuracy.

## Command map

| Command | Group | Purpose |
|---|---|---|
| [`sys`](sys.md) | Discover | Inspect your machine — devices, EPs, SDKs, runtime versions at a glance. |
| [`inspect`](inspect.md) | Discover | Inspect a model's tasks, classes, and hierarchy before committing to an export. |
| [`hub`](hub.md) | Discover | Browse the curated winml-cli catalog of validated models and benchmarks. |
| [`analyze`](analyze.md) | Discover | Verify an ONNX model is compatible with a target execution provider before deployment. |
| [`config`](config.md) | Configure | Generate a reusable build configuration for a Hugging Face model or ONNX file. |
| [`optimize`](optimize.md) | Configure | Apply graph optimizations and fusions to an ONNX model to reduce node count and improve inference speed. |
| [`export`](export.md) | Build | Convert a PyTorch / Hugging Face model to ONNX, preserving module hierarchy. |
| [`quantize`](quantize.md) | Build | Quantize an ONNX model with QDQ insertion and calibration-based scaling. |
| [`compile`](compile.md) | Build | Compile an ONNX model to an EP-specific format for fast runtime loading. |
| [`build`](build.md) | Build | Run the entire winml-cli pipeline (export → optimize → quantize → compile) in one command. |
| [`perf`](perf.md) | Measure | Benchmark an ONNX model's latency and throughput on a target device. |
| [`eval`](eval.md) | Measure | Evaluate ONNX model accuracy on a standard dataset. |

## Choosing a command

- **I want to see what hardware and EPs I have** → `winml sys`
- **I want to know if my model is supported** → `winml inspect`
- **I want to browse validated models with known benchmarks** → `winml hub`
- **I want to verify EP operator compatibility before compiling** → `winml analyze`
- **I want to convert a Hugging Face model to ONNX** → `winml export`
- **I want to run the whole pipeline in one go** → `winml build`
- **I want to benchmark latency and throughput** → `winml perf`
- **I want to measure model accuracy** → `winml eval`

## Global flags

`-v` / `--verbose`, `-q` / `--quiet`, `--debug`, `--version`, and `-h` /
`--help` live on the root `winml` group only. Subcommands access them through
`ctx.obj` and do not redefine them. See
`src/winml/modelkit/commands/_options.py` for the canonical contract.

## Shared flags

Several flags share semantics across the commands that accept them:
`-m` / `--model`, `-d` / `--device`, `--ep`, `-o` / `--output`,
`-t` / `--task`, and `-p` / `--precision`. Defaults and accepted values can
differ per command; check the **Flags** section of each command page rather
than assuming they transfer.

## See also

- [How winml-cli Works](../concepts/how-it-works.md) — end-to-end pipeline overview
- [Config and build](../concepts/config-and-build.md) — structure of `WinMLBuildConfig` and how stages interact
- [ONNX & Execution Providers](../concepts/eps-and-devices.md) — background on EPs and how `--device` / `--ep` interact
- [winml build](build.md) — the single command that runs the entire pipeline
