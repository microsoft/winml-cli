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
