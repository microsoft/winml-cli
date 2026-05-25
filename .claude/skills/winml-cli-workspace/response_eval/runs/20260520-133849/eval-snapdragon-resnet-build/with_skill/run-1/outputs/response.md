# Running `microsoft/resnet-50` on a Snapdragon X Elite NPU

Nice — ResNet-50 on QNN is the textbook BYOM path for ModelKit. CNN, supported architecture, well-trodden EP. You'll get a compiled NPU artifact and latency numbers in a handful of commands.

Below is the full walkthrough: install → sanity-check → build → benchmark. Copy-paste in order in **PowerShell**.

---

## 0. Prereqs — install `winml`

ModelKit pins **Python 3.10 exactly**. Use `uv` so you get an isolated venv:

```powershell
# 1. Create a Python 3.10 venv
uv venv --python 3.10

# 2. Activate it
.venv\Scripts\activate

# 3. Install the winml-cli wheel from the AITK cache
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"

# 4. Verify the CLI is on PATH and the EP is registered
winml --help
winml sys --list-ep
```

In the `winml sys --list-ep` output you want to see **QNN** listed. That's the Qualcomm execution provider for your Snapdragon X Elite NPU. If it's missing, the rest of this won't work — stop and fix that first.

---

## 1. Inspect — confirm ModelKit knows the model

Always do this before exporting or building. It reads the HF config without downloading weights and tells you what loader/exporter/I-O shape the pipeline will use.

```powershell
winml inspect -m microsoft/resnet-50
```

Expected: it identifies the model as a ResNet (image-classification task) and prints I/O specs (224x224x3 input, 1000-class logits output). If inspect is happy, the pipeline will be too.

Optional — peek at what's actually on the box:

```powershell
winml sys --list-device --list-ep
```

---

## 2. Generate the build config — target NPU / int8

ResNet-50 on QNN NPU is the canonical int8 case. One command produces a JSON config that captures every setting the build needs:

```powershell
winml config -m microsoft/resnet-50 --device npu --precision int8 --compile -o resnet50-qnn.json
```

What this does:
- `--device npu` → picks the NPU lane (QNN on your hardware).
- `--precision int8` → INT8 QDQ quantization, which is what QNN's HTP backend wants.
- `--compile` → include the EP-compile stage in the pipeline (off by default).
- `-o resnet50-qnn.json` → writes the config so the build is reproducible.

Open the JSON if you're curious — that file is the source of truth for everything that happens next. Edit it if you want to override anything.

---

## 3. Build — export, optimize, quantize, compile

One command, runs the whole pipeline end-to-end:

```powershell
winml build -c resnet50-qnn.json -m microsoft/resnet-50 -o .\out\resnet50-qnn --compile
```

What lands in `.\out\resnet50-qnn`:
- Exported ONNX (FP32).
- Optimized ONNX (graph rewrites, fusions).
- Quantized ONNX (INT8 QDQ).
- **Compiled QNN artifact** (`.onnx` + a co-located `.bin` for EP context — that's by design; if you move the `.onnx`, move the `.bin` with it).

If a stage blows up, re-run with `-v` for verbose logs. Most commonly it's an op-pattern issue at quantize/compile — `winml analyze -m <exported.onnx> --ep qnn` will pinpoint the offending operator.

---

## 4. Benchmark — latency on NPU

This is the number you came for:

```powershell
winml perf -m microsoft/resnet-50 --device npu --ep qnn --iterations 500 --warmup 20 -c resnet50-qnn.json --monitor
```

- `--device npu --ep qnn` → run on the Snapdragon NPU via QNN.
- `--iterations 500 --warmup 20` → 20 warmup runs (excluded), 500 measured runs. Default is 100/10; bumping it gives tighter percentile estimates.
- `-c resnet50-qnn.json` → reuse the config so perf uses the same precision/compile settings (and picks up the artifacts you just built).
- `--monitor` → live NPU utilization chart during the run. Worth it the first time so you can visually confirm the NPU is actually doing the work.

You'll get back mean / p50 / p90 / p99 latency and throughput, plus a `*_perf.json` next to where you ran from.

### Bonus — NPU vs CPU comparison

If you also want a CPU baseline (highly recommended — that NPU number means a lot more next to a CPU number), run perf a second time against CPU. Use the **pre-compile optimized ONNX**, not the QNN-compiled one — compiled artifacts are tied to their target EP:

```powershell
winml perf -m .\out\resnet50-qnn\model_optimized.onnx --device cpu --iterations 500 --warmup 20
```

(Adjust the filename to whatever the optimized ONNX actually got named — check `dir .\out\resnet50-qnn` after the build.)

---

## TL;DR — just the commands

```powershell
uv venv --python 3.10
.venv\Scripts\activate
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"

winml sys --list-ep
winml inspect -m microsoft/resnet-50

winml config -m microsoft/resnet-50 --device npu --precision int8 --compile -o resnet50-qnn.json
winml build  -c resnet50-qnn.json -m microsoft/resnet-50 -o .\out\resnet50-qnn --compile
winml perf   -m microsoft/resnet-50 --device npu --ep qnn --iterations 500 --warmup 20 -c resnet50-qnn.json --monitor
```

That's the build + benchmark in five lines once you're past install. Have fun — ResNet-50 on Snapdragon's HTP backend is genuinely impressive the first time you see the latency number.
