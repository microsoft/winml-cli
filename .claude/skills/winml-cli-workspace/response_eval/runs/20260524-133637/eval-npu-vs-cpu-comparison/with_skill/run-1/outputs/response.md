Good news: ViT-base is squarely in scope for ModelKit (it's one of the listed vision transformer families), and Snapdragon X Elite means the QNN execution provider is Ready. The trick for a clean apples-to-apples comparison is using the **right artifact for each EP** — you do *not* want to perf the QNN-compiled `.onnx` against CPU, because compiled artifacts are tied to the EP they were compiled for and the number you'd get back would be meaningless.

Here's the cleanest shape.

## Prereq: install winml

If you don't already have the CLI working, install it once. ModelKit pins Python 3.10 exactly, so use `uv` to avoid polluting system Python:

```bash
uv venv --python 3.10
.venv\Scripts\activate
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"
winml --help
winml sys --list-ep
```

`sys --list-ep` should show `QNNExecutionProvider` registered. If it isn't, the NPU path won't work and you'll need to sort that out before benchmarking.

## Step 1: inspect first

Always. It confirms ModelKit knows how to load/export ViT and shows the I/O specs without downloading weights:

```bash
winml inspect -m google/vit-base-patch16-224
```

(Note the `-m` flag — the CLI is flag-based, bare positional model IDs error out.)

## Step 2: optional sanity check with analyze

ViT operator coverage on QNN is generally good, but a quick lint tells you if any op pattern is going to fall back to CPU on the NPU build — that would skew the comparison:

```bash
winml analyze -m google/vit-base-patch16-224 --ep QNN
```

## Step 3: build once, perf twice

This is the key for apples-to-apples. Generate a config, build the artifacts, then perf each one against its intended EP.

```bash
winml config -m google/vit-base-patch16-224 -o vit-config.json
winml build -c vit-config.json -o vit-out\
```

`build` will produce both a pre-compile **optimized** ONNX and a QNN-**compiled** artifact. Use them as follows:

- **NPU number** — perf the QNN-compiled artifact against QNN.
- **CPU number** — perf the pre-compile optimized ONNX against the CPU EP. Do *not* point CPU at the QNN-compiled file.

Concretely (confirm exact flag spelling with `winml perf --help` before pasting — flags evolve):

```bash
# NPU
winml perf -m vit-out\<qnn-compiled>.onnx --ep QNN -o perf-npu.json

# CPU — against the optimized, pre-compile ONNX
winml perf -m vit-out\<optimized>.onnx --ep CPU -o perf-cpu.json
```

For the NPU run, `winml perf --help` also exposes a `--monitor` flag that shows live NPU utilization during the run — useful for confirming the NPU is actually being exercised and not silently falling back.

## What to actually compare

The metrics JSON from each `perf` run is the published output you keep. Compare:

- **Mean / p50 / p95 latency** — p95 matters more than mean for a "should we ship this" call; tail latency is what users feel.
- **Throughput** if you'll batch.
- Quantization parity — if `build` quantized for NPU (typical for QNN), your CPU baseline should be the same precision the user will actually ship. If you're considering shipping fp32 on CPU, also perf the un-quantized optimized ONNX so you're comparing the realistic deployment options, not just isomorphic graphs at different precisions.

## A few gotchas worth flagging

- **Compile writes EP context to a sidecar `.bin` by default.** If you copy the QNN-compiled `.onnx` somewhere, copy the `.bin` next to it, or pass `--embed` at compile time to inline.
- **`perf` alone won't give you a deployable model** — it builds internally and caches that build opaquely. That's why the recipe above is `build` first (so you keep the artifact) and `perf` second (for the number). Don't chain `perf → build → perf`; you'd pay the build cost twice.
- **CLI flags override config-file values**, not the other way around. If you tweak settings between runs, edit `vit-config.json` for reproducibility; use flag overrides only for one-offs.
- **Don't compare across precisions by accident.** If the NPU path quantizes to int8 and your CPU baseline is fp32, you're measuring two different models. Decide what you'd actually ship on CPU and match that.

## TL;DR

1. `winml inspect -m google/vit-base-patch16-224`
2. `winml config` then `winml build` — this produces both the optimized ONNX and the QNN-compiled ONNX in one shot.
3. `winml perf` the QNN-compiled artifact on `--ep QNN` for the NPU number.
4. `winml perf` the pre-compile optimized artifact on `--ep CPU` for the CPU number.
5. Compare p95 latency from the two metrics JSONs.

Run `winml perf --help` and `winml build --help` before pasting commands to confirm current flag spellings — that's the one habit that prevents 90% of "command rejected my flag" round trips.
