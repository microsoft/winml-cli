Nice — Snapdragon X Elite is a first-class target (Qualcomm QNN EP, Ready), and `microsoft/resnet-50` is squarely in scope (classic CNN). Here's the end-to-end walkthrough.

You'll end up running two commands for the actual work (`config` then `build`), plus a `perf` at the end for the latency number. Everything before that is one-time install + a couple of sanity checks so we don't waste your time chasing a misconfigured environment.

---

## 0. Prereqs — install the `winml` CLI

ModelKit pins **Python 3.10 exactly**. Use `uv` so you get an isolated venv.

```powershell
# 1. Create + activate a Python 3.10 venv
uv venv --python 3.10
.venv\Scripts\activate

# 2. Install the winml-cli wheel from the AITK cache
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"

# 3. Smoke test
winml --help
```

If `winml --help` prints the command list, you're good.

---

## 1. Confirm the NPU + QNN EP are visible

Before quoting any compile/build command, make sure the box actually exposes QNN. On a fresh Snapdragon X Elite dev kit this normally works out of the box, but verify:

```powershell
winml sys --list-device --list-ep
```

You want to see a Qualcomm NPU device and **QNN** in the EP list. If QNN isn't listed, stop here — compiling against an EP that isn't registered will produce an artifact you can't run. (Usually fixed by making sure the QNN runtime that AITK installs is on PATH.)

---

## 2. Inspect the model (golden rule — always do this first)

This reads the HF config **without downloading weights** and confirms ModelKit knows how to handle ResNet-50 (loader, exporter, WinML inference class, I/O specs).

```powershell
winml inspect -m microsoft/resnet-50
```

Note the flag-based form: `-m microsoft/resnet-50`, not a bare positional. If you want JSON for piping into other tools, add `-f json`.

You should see a clean image-classification profile. ResNet-50 is the canonical "happy path" model — if inspect complains here, something's off with the install, not the model.

---

## 3. (Optional but recommended) Analyze operator compatibility for QNN

The linter classifies every op as supported / partial / unsupported on your chosen EP. For ResNet-50 on QNN this should be all-green, but it's a cheap sanity check before you pay for export+quantize+compile:

```powershell
winml analyze -m microsoft/resnet-50 --ep QNN
```

(If the flag spelling differs on your build, `winml analyze --help` is the source of truth — read it instead of guessing.)

---

## 4. Generate a build config

`winml config` auto-detects every setting the pipeline needs (export options, optimization passes, quantization recipe, EP, device) and writes a single JSON file you can version-control. This is the recommended path for a clean, reproducible build:

```powershell
winml config -m microsoft/resnet-50 --ep QNN -o .\resnet50-qnn.config.json
```

Check what got written — `winml config --help` will show you the available knobs (precision, calibration, etc.) if you want to tweak before building. CLI flags override the config file, not the other way around, so the JSON is your stable record.

---

## 5. Build the artifact

One command runs export → optimize → quantize → compile, in that order, using the config:

```powershell
winml build -c .\resnet50-qnn.config.json -o .\out\resnet50-qnn
```

When this finishes, `.\out\resnet50-qnn` contains your deployable, QNN-compiled `.onnx`. **Heads up:** by default, compile writes the EP context to a sidecar `.bin` next to the `.onnx`. If you move the model later, move the `.bin` with it. (If you'd rather have one self-contained file, re-run with `--embed` — check `winml compile --help` for the exact flag plumbing through build.)

---

## 6. Benchmark latency on the NPU

Point `perf` at the built artifact, target QNN, and write the metrics JSON:

```powershell
winml perf -m .\out\resnet50-qnn\model.onnx --ep QNN -o .\out\resnet50-qnn\perf.json
```

(Replace `model.onnx` with whatever filename `build` actually emitted — `ls .\out\resnet50-qnn` to confirm.)

The metrics JSON has per-iteration latency, mean/p50/p95/p99, and throughput. If you want live NPU utilization while it runs, check `winml perf --help` for a `--monitor` flag.

### Bonus: NPU vs CPU comparison

If you want to see what the NPU is actually buying you, run perf a second time against the **pre-compile** optimized ONNX on CPU (the QNN-compiled artifact is tied to QNN — perf'ing it on CPU is meaningless). The optimized intermediate from your build directory is what you want:

```powershell
winml perf -m .\out\resnet50-qnn\<optimized-pre-compile>.onnx --ep CPU -o .\out\resnet50-cpu-perf.json
```

---

## Quick recap of the flow

```
winml --help                     # confirm install
winml sys --list-device --list-ep  # confirm QNN is registered
winml inspect -m microsoft/resnet-50
winml analyze -m microsoft/resnet-50 --ep QNN    # optional
winml config  -m microsoft/resnet-50 --ep QNN -o .\resnet50-qnn.config.json
winml build   -c .\resnet50-qnn.config.json     -o .\out\resnet50-qnn
winml perf    -m .\out\resnet50-qnn\model.onnx  --ep QNN -o .\out\resnet50-qnn\perf.json
```

One thing I'd flag: I quoted likely flag spellings (`--ep QNN`, `-m`, `-o`) based on the toolkit's general shape, but specific flag names can drift between versions. If any line above rejects a flag, run `winml <that-command> --help` and use what it prints — don't try to massage my command. The CLI's `--help` is always the source of truth.

Have fun — ResNet-50 on QNN should land in the low-single-digit-millisecond range on X Elite. Ping back with the perf JSON if anything looks off.
