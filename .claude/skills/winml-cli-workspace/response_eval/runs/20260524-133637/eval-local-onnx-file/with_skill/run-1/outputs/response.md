# Benchmarking a local `resnet50.onnx` on your Intel Core Ultra NPU

Good news: you don't need to re-download anything. `winml` accepts a local `.onnx` path anywhere a model is expected, so you can point it straight at the file on disk. The Intel NPU goes through the **OpenVINO** execution provider, which is Ready on Core Ultra (Meteor Lake / Lunar Lake).

Since you already have an ONNX file, you're entering the pipeline past `export` — the only thing you need is a latency number, which means the right tool is `winml perf` (its published output is a metrics JSON). `perf` will build the artifacts it needs on the fly against your local ONNX; no HF download involved.

## Prereq: install the CLI (skip if you already have `winml` working)

ModelKit pins Python 3.10 exactly. Use `uv` so you land on the right interpreter.

```powershell
uv venv --python 3.10
.venv\Scripts\activate
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"
winml --help
```

## Step 1 — sanity-check the file and your machine

Inspect first, always. It reads the model without re-exporting and tells you whether ModelKit recognizes the architecture and what I/O it expects. Then confirm the OpenVINO EP is actually registered on this box.

```powershell
winml inspect -m .\resnet50.onnx
winml sys --list-device --list-ep
```

You're looking for two things:

- `inspect` should print a sensible loader / WinML inference class and resolved I/O specs for ResNet-50.
- `sys --list-ep` should list the OpenVINO EP. If it isn't there, the NPU run won't work no matter what flags you pass — fix the EP registration first.

## Step 2 — confirm operator coverage on OpenVINO (cheap, optional but worth it)

ResNet-50 is well within scope (classic CNN), but a 30-second analyze pass catches surprises before you pay the compile cost:

```powershell
winml analyze -m .\resnet50.onnx
```

The linter flags any op that's unsupported / partial on the target EP. For a stock ResNet-50 export you'd expect a clean bill.

## Step 3 — benchmark on the NPU

This is the one command that actually produces your number. Point `-m` at the local file and pick the OpenVINO EP / NPU device.

```powershell
winml perf -m .\resnet50.onnx -o .\perf-resnet50-npu
```

A few notes before you run it:

- **Read `winml perf --help` first** to get the current spelling for the EP / device selector flags and confirm whether there's an auto-pick mode. The CLI is the source of truth for flag names; I don't want to invent one (e.g. don't trust a guessed `--ep openvino --device npu` without checking).
- `perf` will internally export, optimize, quantize (unless you pass `--no-quantize`), and compile against the OpenVINO EP, then time it. Subsequent runs reuse the cache; use `--rebuild` or `--ignore-cache` if you want a clean rebuild. Look for `--monitor` in `--help` if you want live NPU utilization during the run.
- The `-o` path receives the **metrics JSON** — not a model. That's the contract for `perf`.

## If you want a deployable artifact *and* a latency number

`perf` alone doesn't give you a model you can keep — its build artifacts live in opaque cache. If you also want a shippable compiled `.onnx` for the NPU, do it in two commands:

```powershell
winml config -m .\resnet50.onnx -o .\resnet50-npu.config.json
winml build -c .\resnet50-npu.config.json -o .\resnet50-npu-build
winml perf  -m .\resnet50-npu-build\<compiled-artifact>.onnx -o .\perf-resnet50-npu
```

`config` auto-detects settings for your hardware (edit the JSON if you want to override anything — quantization, EP, device); `build` runs the full pipeline and writes the artifact you can deploy; then `perf` measures it. This is the standard config + build path and is what I'd default to if you care about reproducibility or want to share the config with a teammate.

## Things that bite people on Intel NPU specifically

- **`compile` writes EP context to an external `.bin` next to the `.onnx` by default.** If you move the compiled artifact, move the `.bin` alongside, or pass `--embed` at compile time to inline it.
- **A compiled artifact is tied to its EP.** Don't try to `perf` the OpenVINO-compiled `.onnx` against the CPU EP for comparison — point CPU runs at the pre-compile *optimized* ONNX instead.
- **CLI flags beat the config file.** If you generated a config and then pass `--no-quantize` (or similar) on the `build` command line, the flag wins.

## TL;DR

For just a latency number against your existing file:

```powershell
winml inspect -m .\resnet50.onnx
winml sys --list-ep                       # confirm OpenVINO EP is registered
winml perf --help                         # read current EP/device flag spelling
winml perf -m .\resnet50.onnx -o .\perf-resnet50-npu   # add the EP/device flags --help showed you
```

Nothing in that flow touches Hugging Face — `-m` accepts your local path and the pipeline runs from there.
