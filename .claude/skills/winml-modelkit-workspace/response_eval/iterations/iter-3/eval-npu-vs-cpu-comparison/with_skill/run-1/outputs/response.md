Good news on the model side: `google/vit-base-patch16-224` is a vision transformer, which is squarely in the classic-CV sweet spot for ModelKit. The pipeline is built to handle ViT/Swin/ConvNeXT/ResNet, so QNN (Qualcomm's NPU EP on Snapdragon X Elite) should be a real option, not a maybe.

For apples-to-apples, the rule that matters: **NPU and CPU should benchmark different artifacts from the same build.** The QNN-compiled `.onnx` is EP-locked — you cannot run it on CPU and get meaningful numbers (and vice versa). The clean approach is to build once and then run `perf` twice, pointing each run at the right artifact for that EP.

## Recommended flow

### 1. Confirm the model is supported on your machine

```powershell
uv run winml inspect -m google/vit-base-patch16-224
uv run winml sys --list-ep
```

Inspect should report ViT as `Supported` (it's an in-scope architecture). `sys` confirms QNN is registered on your Snapdragon box.

### 2. Generate a build config targeting the NPU

```powershell
uv run winml config -m google/vit-base-patch16-224 --device npu --precision int8 --compile -o vit_npu.json
```

A couple of notes:
- `--device npu` on a Snapdragon machine maps to the QNN EP.
- `--precision int8` is the typical QNN target — the NPU only really shines on quantized weights/activations. `--precision auto` will pick a sensible default if you'd rather defer.
- `--compile` is required here because the default is `--no-compile`, and you want the QNN-compiled artifact for the NPU run.

### 3. Build once

```powershell
uv run winml build -c vit_npu.json -m google/vit-base-patch16-224 -o build/vit/ --compile
```

This produces a directory containing both the **optimized (pre-compile) ONNX** and the **QNN-compiled ONNX**. The optimized one is your CPU benchmark target; the compiled one is the NPU target.

### 4. Run `perf` twice against the right artifacts

```powershell
# NPU (QNN) — compiled artifact
uv run winml perf -m build/vit/<compiled>.onnx --device npu --ep qnn --iterations 500 --warmup 50 -o vit_npu_perf.json

# CPU — optimized (pre-compile) artifact
uv run winml perf -m build/vit/<optimized>.onnx --device cpu --ep cpu --iterations 500 --warmup 50 -o vit_cpu_perf.json
```

Replace `<compiled>` / `<optimized>` with the actual filenames the build wrote — `winml build` will print them, and they'll also be sitting in the output dir. Match `--iterations`, `--warmup`, and `--batch-size` between the two runs so you're comparing the same statistical sample.

Want a live view of NPU utilization while the NPU run is going? Add `--monitor` to the first `perf` invocation.

## Things to double-check before you trust the numbers

- **Same input shape.** ViT-base is fixed at 224x224, so the default I/O should be fine, but if you ever tweak with `--shape-config` make sure both runs use the same one.
- **Quantization parity.** The CPU run is against the *optimized but not quantized* artifact by design — that's the fair "what CPU can do with this model" number. If you want CPU-with-int8 too, that's a separate build (`--device cpu --precision int8`) and a third `perf` run; just don't conflate it with the unquantized baseline.
- **Compiled-artifact lock.** If your CPU latency comes back nonsensically high or you get garbage outputs, you almost certainly pointed `perf` at the QNN-compiled file with `--ep cpu`. Re-check the path.

## Ship decision heuristic

Once you have the two JSON outputs, look at median (p50) and p95 latency, not just mean. NPU wins are usually most dramatic on p50 with a tighter distribution; if CPU p95 is comparable and your app isn't latency-critical at the tail, the simplicity of shipping CPU-only (no QNN runtime dependency, no per-SoC artifact) is often worth it. If NPU is materially faster *and* you're already shipping Snapdragon-targeted binaries, ship the compiled NPU artifact and keep the optimized ONNX as a CPU fallback for non-Snapdragon machines.

One last thing — verify the exact filenames `winml build` produces in your output directory before pasting the `perf` commands; I don't want to guess at the artifact naming convention.
