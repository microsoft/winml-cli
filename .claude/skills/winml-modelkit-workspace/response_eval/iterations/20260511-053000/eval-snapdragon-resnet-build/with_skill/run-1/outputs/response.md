Nice setup. `microsoft/resnet-50` is squarely in scope (classic CNN, image classification), and Snapdragon X Elite means we're targeting the **QNN** execution provider on the Hexagon NPU. Here's the path end to end.

## 1. Sanity-check the environment

Confirm your dev box exposes the NPU and the QNN EP is registered. If QNN isn't in the list, the rest won't work — stop and fix that first.

```powershell
winml sys --list-device --list-ep
```

## 2. Inspect the model (golden rule — always do this first)

Cheap, weights-free, tells you whether the pipeline supports this architecture before you commit to a build.

```powershell
winml inspect -m microsoft/resnet-50
```

You want the **Overall Support** line to say `Supported`. For ResNet-50 it will.

## 3. Generate a build config targeting the NPU

This auto-detects task (`image-classification`), I/O shapes, and writes a JSON config that's the single source of truth for the build. `--device npu` picks QNN on your hardware; `--precision int8` is the right call for NPU (Hexagon wants quantized integer ops). `--compile` flips on the EP-specific compile stage so you get a QNN-locked artifact at the end.

```powershell
winml config -m microsoft/resnet-50 --device npu --precision int8 --compile -o resnet50-qnn.json
```

Open `resnet50-qnn.json` if you want to peek at or tweak any defaults. Edits there flow through to the build.

## 4. Build the model

Runs export → analyze → optimize → quantize → compile, in order, writing every artifact into `./resnet50-qnn-build/`.

```powershell
winml build -c resnet50-qnn.json -m microsoft/resnet-50 -o resnet50-qnn-build/ --compile
```

When it finishes you'll have a QNN-compiled `.onnx` in that output directory alongside the optimized intermediate. Note the QNN-compiled file is **EP-locked** — only run it on the NPU, not CPU.

## 5. Benchmark on NPU

This is the latency number you came for. `--monitor` gives you a live NPU utilization chart so you can confirm the work is actually landing on Hexagon and not silently falling back.

```powershell
winml perf -m microsoft/resnet-50 -c resnet50-qnn.json --device npu --iterations 500 --warmup 50 --monitor
```

Results print to stdout and also save to `microsoft_resnet-50_perf.json` (mean / p50 / p95 / p99 latency, throughput).

## 6. (Optional) CPU baseline for comparison

Useful to quantify the NPU speedup. This runs perf against the model on CPU EP — ModelKit will reuse cached artifacts where possible.

```powershell
winml perf -m microsoft/resnet-50 --device cpu --iterations 500 --warmup 50 -o resnet50-cpu-perf.json
```

---

A few things worth knowing:

- If any step errors out, run `winml <that-command> --help` — flags evolve, and the CLI is the source of truth.
- `winml analyze` against the exported ONNX is the fastest way to diagnose an EP/operator mismatch if optimize or quantize blows up.
- If you'd rather drive the pipeline one stage at a time (e.g., to swap quantization settings), the primitives are: `export`, `analyze`, `optimize`, `quantize`, `compile`, `perf` — same flag shape, just more commands.
