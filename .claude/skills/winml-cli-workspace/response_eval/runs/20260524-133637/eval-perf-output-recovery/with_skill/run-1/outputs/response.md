Short answer: it didn't put one anywhere you can ship from. `winml perf` does build an artifact under the hood in order to benchmark it, but the **only published output of `perf` is the metrics JSON** (whatever you passed to `-o`, or whatever it printed). The build it used lives in an internal cache that isn't a supported output — don't try to fish it out, copy it, or hand it to your app. Cache layout is opaque and can change between releases, so anything you grab from there is liable to break.

The way to think about it: every `winml` command has one published output, and that's the only thing you can rely on downstream. For `perf` that's the metrics. For a deployable model, you want the command whose `-o` *is* an `.onnx` artifact — that's `build` (or `compile` if you're doing it stage-by-stage).

## What to do instead

Re-enter the pipeline at `build`, which writes the optimized + quantized + EP-compiled artifact to a directory you choose. Then run `perf` against that artifact if you want to confirm the latency number matches what you saw before.

Rough shape (confirm exact flags with `--help` — I'm not going to invent them):

```powershell
winml config --help
winml build --help
winml perf --help
```

A typical flow for your case (Snapdragon X Elite = QNN EP):

```powershell
# 1. Generate a config that captures the same pipeline perf used
winml config -m microsoft/resnet-50 --device npu -o resnet50-qnn.json

# 2. Build the deployable artifact to a directory you control
winml build -c resnet50-qnn.json -o .\dist\resnet50-qnn

# 3. (Optional) Re-run perf against the built artifact to confirm latency
winml perf -m .\dist\resnet50-qnn\<artifact>.onnx --device npu
```

The `.\dist\resnet50-qnn` directory is what you ship.

## One gotcha for shipping a QNN-compiled model

When `compile` runs (which `build` invokes for you on NPU), it defaults to writing the EP context to a separate `.bin` file co-located with the `.onnx`. If you move the `.onnx` to your app's resources folder, **move the `.bin` alongside it** — they're a pair. If you'd rather ship a single file, look for an embed flag on `winml compile --help` (something like `--embed`) and set it in your config so the EP context is inlined into the `.onnx`. Check `winml compile --help` for the exact spelling before wiring it into the config.

## Why you can't just reuse the perf run

Even if you found the cached build, it's not a contract — the path, naming, and whether the `.bin` sidecar is there are all implementation details. Running `build -o` once costs you the same as the build `perf` already did internally; the difference is you now own the output and can version it, sign it, and drop it into your app installer with confidence.
