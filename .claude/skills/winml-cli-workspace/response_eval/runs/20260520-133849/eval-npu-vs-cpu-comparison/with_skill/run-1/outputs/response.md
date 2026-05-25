# ViT-base NPU vs CPU on Snapdragon X Elite — apples-to-apples latency

You're in a good spot: ViT (`google/vit-base-patch16-224`) is squarely in scope for ModelKit (vision transformer, classic deep learning, not generative), and Snapdragon X Elite's QNN execution provider is ready today. Here's the cleanest path to a fair comparison.

## Prereq: install the `winml` CLI

If you've already got `winml --help` working, skip this. Otherwise:

```powershell
# 1. Python 3.10 venv (ModelKit pins 3.10 exactly)
uv venv --python 3.10
.venv\Scripts\activate

# 2. Install the wheel from the AITK cache
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"

# 3. Verify
winml --help
winml sys --list-ep
```

The `--list-ep` output should show `QNNExecutionProvider` and `CPUExecutionProvider` — if QNN isn't listed, the NPU comparison won't be real, so stop and fix the environment first.

## The key idea: don't compare a QNN-compiled artifact against CPU

EP-compiled artifacts are bound to the EP they were compiled for. If you point `winml perf --device cpu` at a QNN-compiled `.onnx`, you'll get a meaningless number (best case the CPU EP falls back and "runs" it; worst case it errors). The fair comparison is:

- **NPU run** → benchmark the **QNN-compiled** artifact on QNN.
- **CPU run** → benchmark the **pre-compile optimized + quantized** artifact on the CPU EP.

Same source model, same optimization passes, same quantization scheme — only the final EP-compile step differs.

## Step 0 — confirm the model is supported (the golden rule)

```powershell
winml inspect -m google/vit-base-patch16-224
```

This reads the config without downloading weights and shows the loader / exporter / I/O specs and build resolution. ViT should resolve to image-classification with a `(1, 3, 224, 224)` input. If anything looks off there, fix it before going further — every later stage costs more time.

## Step 1 — build once, get both artifacts

Generate a build config targeting NPU, then run the full pipeline. The build emits the optimized + quantized ONNX *and* the QNN-compiled artifact in the same output directory, so you get both halves of the comparison from a single run.

```powershell
# Generate config for NPU/QNN with int8 quantization (matches what the NPU expects)
winml config -m google/vit-base-patch16-224 --device npu --precision int8 --compile -o vit_npu.json

# Build everything: export -> optimize -> quantize -> compile (QNN)
winml build -c vit_npu.json -m google/vit-base-patch16-224 -o vit_out/ --compile
```

After this you'll have something like (exact filenames are whatever the build prints — read them off the build output, don't assume):

- `vit_out/<...>_optimized_quantized.onnx` — the pre-compile artifact. **This is what CPU benchmarks against.**
- `vit_out/<...>_qnn.onnx` (+ a sibling `.bin` with EP context, since `compile` defaults to external context — move them together if you move the artifact) — **this is what NPU benchmarks against.**

Verify flag spelling with `winml build --help` and `winml config --help` if anything errors; flags evolve and the CLI is the source of truth.

## Step 2 — run perf twice, identical settings

Same iterations, same warmup, same batch size. Only `--device` / `--ep` and the input artifact change.

```powershell
# NPU (QNN) — compiled artifact
winml perf -m vit_out/<the_qnn_artifact>.onnx `
  --device npu --ep qnn `
  --iterations 500 --warmup 50 --batch-size 1 `
  -o vit_perf_npu.json

# CPU — pre-compile optimized+quantized artifact (NOT the QNN one)
winml perf -m vit_out/<the_optimized_quantized_artifact>.onnx `
  --device cpu --ep cpu `
  --iterations 500 --warmup 50 --batch-size 1 `
  -o vit_perf_cpu.json
```

Why these settings matter for "apples-to-apples":

- **Same artifact lineage.** Both runs trace back to the same export + optimize + quantize. The CPU side isn't running fp32 while the NPU runs int8 — that would flatter the NPU unfairly. (If you actually want fp32-CPU vs int8-NPU because that's your real shipping question, do *that* comparison too — just label it honestly.)
- **`--iterations 500 --warmup 50`** gives stable percentiles; the default 100/10 is fine for a quick look but noisy. Both runs use the same numbers.
- **`--batch-size 1`** matches typical on-device inference. If your app batches, set it to your real batch size — and use the same value on both runs.
- **Random input data.** `winml perf` generates random tensors matching the I/O spec, which is fine for latency. It's *not* fine for accuracy — for that, use `winml eval`.

Tip: add `--monitor` to the NPU run if you want to see live NPU utilization and confirm the workload is actually landing on the NPU (not silently falling back to CPU). For deeper investigation of NPU op coverage, `--op-tracing basic` gives operator-level profiling on QNN.

## Step 3 — read the numbers

Each run writes a JSON with mean / median / p50 / p90 / p99 latency and throughput. Compare:

- **Median or p50** for typical-case latency.
- **p99** for tail latency — often more important for UX than the mean.
- **Throughput (inf/sec)** if you batch.

Also worth eyeballing from the perf JSON:

- **Active EP** — confirm the NPU run actually used QNN end-to-end. If half the ops fell back to CPU, your "NPU number" is really a hybrid number and the comparison is muddied. `winml analyze -m <the_exported_onnx> --ep qnn` will tell you per-op coverage if you want to dig in.
- **Memory / model size** on disk — sometimes shipping decision factors beyond latency.

## What to do with the result

- **NPU clearly wins on latency AND fully runs on QNN** → ship NPU, keep CPU as fallback for machines without QNN registered.
- **NPU wins but with significant CPU fallback** → either fix the unsupported ops (different quantization config, different optim choices — `winml analyze` will name them) or ship CPU; a partially-offloaded model is fragile.
- **CPU is competitive** → ship CPU. Less surface area, no EP-context file to ship alongside the `.onnx`, works on every Windows machine.

## One thing I'd skip

The `--compare-devices cpu,npu` flag on `winml perf` looks like it would do all of this for you in one command — but per `winml perf --help` it's listed as "Not yet implemented." Don't waste time on it; run perf twice manually as shown above.

## Quick reference — the four commands

```powershell
winml inspect -m google/vit-base-patch16-224
winml config  -m google/vit-base-patch16-224 --device npu --precision int8 --compile -o vit_npu.json
winml build   -c vit_npu.json -m google/vit-base-patch16-224 -o vit_out/ --compile
winml perf    -m vit_out/<qnn>.onnx                 --device npu --ep qnn --iterations 500 --warmup 50 -o vit_perf_npu.json
winml perf    -m vit_out/<optimized_quantized>.onnx --device cpu --ep cpu --iterations 500 --warmup 50 -o vit_perf_cpu.json
```

Fill in the actual artifact filenames from what `winml build` prints into `vit_out/`.
