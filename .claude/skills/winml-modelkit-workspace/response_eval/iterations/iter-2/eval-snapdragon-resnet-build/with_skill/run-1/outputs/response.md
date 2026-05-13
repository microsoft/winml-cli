Nice, Snapdragon X Elite is a great target for this — `microsoft/resnet-50` is a classic CNN and well within the ModelKit happy path. Here's the walkthrough.

The execution provider you want is **QNN** (Qualcomm's), and the device is **NPU**. The workflow is:

1. `winml sys` — confirm the NPU + QNN are visible.
2. `winml inspect` — confirm ResNet-50 is supported (this is the golden rule, don't skip it).
3. `winml config` — auto-generate a build config targeted at NPU/QNN.
4. `winml build` — run the full pipeline (export → optimize → quantize → compile).
5. `winml perf` — benchmark the compiled artifact and print latency stats.

## 0. Sanity-check the environment

```powershell
winml --version
winml sys --list-device --list-ep
```

In the output of `winml sys` you should see your Hexagon NPU listed and `qnn` in the EP list. If `qnn` isn't there, your install is missing the QNN extras — stop and fix that first.

## 1. Inspect the model (non-negotiable)

```powershell
winml inspect -m microsoft/resnet-50
```

You're looking for a "supported" verdict and a sane task (`image-classification`). If this errors, don't bother going further.

## 2. Generate a build config for NPU

```powershell
winml config -m microsoft/resnet-50 --device npu --ep qnn --precision int8 -o resnet50_qnn.json
```

QNN NPUs are int8-native, so `int8` is the right precision here. Open `resnet50_qnn.json` if you want to see/edit what the pipeline will do — that file is the source of truth for the build.

## 3. Build it

```powershell
winml build -c resnet50_qnn.json -m microsoft/resnet-50 -o build\resnet50_qnn --compile -v
```

`--compile` makes sure the QNN compile stage runs (it produces the NPU-locked artifact). This step downloads weights, exports to ONNX, runs the analyzer, optimizes, quantizes to int8, and compiles for QNN. Expect a few minutes the first time.

Artifacts will land under `build\resnet50_qnn\`.

## 4. Benchmark on the NPU

```powershell
winml perf -m microsoft/resnet-50 --device npu --ep qnn --precision int8 --iterations 200 --warmup 20 --monitor -o resnet50_qnn_perf.json
```

This will reuse the cached build from step 3, time 200 inferences after 20 warmup runs, show a live NPU utilization chart, and dump full stats to JSON. The summary table at the end has the latency numbers you're after (mean / p50 / p90 / p99, plus throughput).

## 5. (Optional) Compare against CPU

If you want a CPU baseline to see what the NPU is actually buying you:

```powershell
winml perf -m microsoft/resnet-50 --device cpu --iterations 200 --warmup 20 -o resnet50_cpu_perf.json
```

Don't try to point a CPU run at the QNN-compiled artifact — compiled artifacts are EP-locked and you'll get garbage numbers. The command above will use the optimized (pre-compile) ONNX automatically.

## Notes

- All flags above came from `winml <cmd> --help` — run that yourself if anything looks off in your version; flags evolve.
- If `build` blows up at optimize or quantize, run `winml analyze` against the exported ONNX to see which op pattern QNN doesn't like.
- The `--monitor` flag during perf is the easiest way to confirm you're actually hitting the NPU and not silently falling back to CPU.

Drop the perf summary back here once it runs and I can help sanity-check the numbers.
