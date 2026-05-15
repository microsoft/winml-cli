# eval-npu-vs-cpu-comparison

## Prompt

> i have a ViT model (google/vit-base-patch16-224) and im trying to decide whether to ship the NPU version or just use CPU. running on a snapdragon x elite. whats the cleanest way to get apples-to-apples latency numbers from both?

## with_skill — 7/7

### Response

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

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✓ | Recommends `winml inspect` before any build/benchmark work. | PASS | Step 1: 'uv run winml inspect -m google/vit-base-patch16-224' to confirm Supported. |
| ✓ | Identifies QNN as the EP for Snapdragon X Elite NPU. | PASS | Quote: 'QNN (Qualcomm's NPU EP on Snapdragon X Elite) should be a real option.' |
| ✓ | Recommends building once and running `winml perf` twice. | PASS | Quote: 'The clean approach is to build once and then run perf twice, pointing each run at the right artifact for that EP.' |
| ✓ | Specifies that CPU run uses optimized (pre-compile) artifact, NOT QNN-compiled. | PASS | Quote: 'The optimized one is your CPU benchmark target; the compiled one is the NPU target.' |
| ✓ | Warns EP-compiled artifacts are EP-locked. | PASS | Quote: 'The QNN-compiled .onnx is EP-locked -- you cannot run it on CPU and get meaningful numbers.' |
| ✓ | Directs to `--help` or uses common patterns; no fabricated flags. | PASS | All flags verified via --help. |
| ✓ | All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | All ? winml commands verified runnable against --help. |

---

## without_skill (baseline) — 3/7 ⚠

### Response

For apples-to-apples NPU vs CPU latency on a Snapdragon X Elite, the cleanest path is ONNX Runtime with two different Execution Providers, measured against the *same* ONNX model file. Here's how I'd approach it:

## 1. Export once, run everywhere

Export `google/vit-base-patch16-224` to ONNX a single time so both runs share identical graph + weights:

```bash
optimum-cli export onnx --model google/vit-base-patch16-224 --task image-classification vit-onnx/
```

That gives you `model.onnx` with fixed opset and the same pre/post-processing assumptions.

## 2. Prep two model variants (this is the only "fair" gotcha)

- **CPU**: keep it FP32 (or run ORT's graph optimizations to `optimized_model.onnx`). The CPU EP handles FP32 fine.
- **NPU (QNN EP)**: the Hexagon NPU on X Elite effectively requires **quantized** models — typically static INT8 / QDQ. Running FP32 on QNN will either fall back to CPU silently or refuse to load. Use ORT's quantization tools with a small calibration set of real ImageNet-style images:

```python
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
quantize_static(
    "model.onnx", "model.qdq.onnx",
    calibration_data_reader=my_reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
)
```

Be aware: this means you're comparing FP32-CPU vs INT8-NPU. That is the realistic shipping comparison, but you should also record accuracy delta on a validation set (top-1/top-5 on ImageNet val) — latency alone is misleading if INT8 tanks accuracy.

## 3. Benchmark harness

Use `onnxruntime` directly so both EPs go through the same code path:

```python
import onnxruntime as ort, numpy as np, time

def bench(model_path, providers, runs=200, warmup=20):
    sess = ort.InferenceSession(model_path, providers=providers)
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)  # or uint8 input for QDQ
    name = sess.get_inputs()[0].name
    for _ in range(warmup): sess.run(None, {name: x})
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {"p50": times[runs//2], "p90": times[int(runs*0.9)], "mean": sum(times)/runs}

cpu = bench("model.onnx", [("CPUExecutionProvider", {})])
npu = bench("model.qdq.onnx", [("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"})])
```

Key things to keep identical:
- Batch size, input shape, input dtype semantics
- Warmup runs (NPU especially needs them — first inference includes graph compile/HTP context init)
- Same machine, plugged in, performance power plan, no other GPU/NPU workloads
- Report p50 / p90 / p99, not just mean — NPU tail latency can be spiky
- Verify the NPU run actually ran on NPU: enable `session.set_providers` logging or check `sess.get_providers()` and look at the QNN profiler output to confirm zero CPU fallback nodes. Partial fallback will silently wreck your numbers.

## 4. What to actually decide on

Latency is only half the story. Tabulate:

| Metric | CPU (FP32) | NPU (INT8) |
|---|---|---|
| p50 latency | | |
| p99 latency | | |
| Top-1 accuracy | | |
| Power (if you can measure) | | |
| Cold-start (first inference) | | |

NPU usually wins steady-state latency *and* power by a lot on X Elite, but loses on cold start and accuracy. If your app does occasional inferences, cold start matters; if it streams, steady-state wins.

A couple of side notes:
- ViT-base is attention-heavy; some QNN versions historically had weaker support for certain attention ops and may partition the graph. Check the partition log before trusting numbers.
- `onnxruntime-qnn` is the package you want on ARM64 Windows; make sure you're on the ARM64 Python build, not x64-emulated, or you'll measure emulation overhead on the CPU side.

Want me to sketch the calibration-data reader or the QNN provider options in more detail?

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✗ | Recommends `winml inspect` before any build/benchmark work. | **FAIL** | No winml inspect. Goes straight to `optimum-cli export onnx`. |
| ✓ | Identifies QNN as the EP for Snapdragon X Elite NPU. | PASS | 'NPU (QNN EP): the Hexagon NPU on X Elite effectively requires quantized models'. |
| ✗ | Recommends building once and running `winml perf` twice. | **FAIL** | Exports once but then quantizes specifically for NPU (creating a separate QDQ file), then benchmarks. Doesn't have the 'shared build, two perf runs over different EP artifacts' framing â€” runs CPU o… |
| ✗ | Specifies that CPU run uses optimized (pre-compile) artifact, NOT QNN-compiled. | **FAIL** | Has no notion of a 'compiled artifact' since it doesn't use the winml pipeline. CPU uses FP32 ONNX, NPU uses separately-quantized QDQ file. |
| ✗ | Warns EP-compiled artifacts are EP-locked. | **FAIL** | Never mentions EP-locking. Some QNN gotchas (silent CPU fallback, ARM64 build) but not this specific failure mode. |
| ✓ | Directs to --help or uses common patterns; no fabricated flags. | PASS | Real ORT/QNN APIs. |
| ✓ | All `winml` commands quoted in the response are runnable — every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | Response uses no winml commands (trivially passes). |

---
