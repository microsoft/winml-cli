# Fastest way to benchmark `facebook/convnext-tiny-224` on your AMD Ryzen AI NPU

You only care about the latency number, not the artifact — so the goal is: get an ONNX model that can actually load on the VitisAI EP, run it a few times, print the median latency. Nothing more.

Here is the shortest path.

## TL;DR

1. Install the AMD Ryzen AI Software stack (gives you the VitisAI ExecutionProvider for ONNX Runtime plus the NPU driver/XCLBIN).
2. Export `facebook/convnext-tiny-224` to ONNX with Optimum.
3. Quantize it to INT8 (the Ryzen AI NPU only runs quantized models — FP32/FP16 will silently fall back to CPU and your "NPU number" will be meaningless).
4. Run `onnxruntime`'s built-in `onnxruntime_perf_test` (or a 10-line Python loop) with `provider=VitisAIExecutionProvider`.

Total wall-clock: ~15 min once the Ryzen AI SW is installed.

---

## Step 1 — Install Ryzen AI Software

Download from AMD (https://ryzenai.docs.amd.com). The installer sets up:

- NPU driver (check with `xrt-smi examine` — you should see your `Phoenix` / `Hawk Point` / `Strix` NPU).
- A conda env (`ryzen-ai-<ver>`) with `onnxruntime-vitisai`, `vai_q_onnx` (the quantizer), and the matching XCLBIN binaries.
- The env var `XLNX_VART_FIRMWARE` pointing at the right XCLBIN for your silicon (1x4 / 4x4 / Strix Halo configs differ — let the installer pick).

Activate it:

```powershell
conda activate ryzen-ai-1.x
```

## Step 2 — Export the model to ONNX

```powershell
pip install "optimum[exporters]" transformers
optimum-cli export onnx --model facebook/convnext-tiny-224 --task image-classification convnext_onnx/
```

You now have `convnext_onnx/model.onnx` (FP32, opset 14+). Input is `pixel_values` shape `[N,3,224,224]`.

## Step 3 — Quantize to INT8 (this is the step people skip and then complain the NPU is slow)

The Ryzen AI NPU executes INT8 (and on Strix, also BF16) — anything else falls back to CPU. Use `vai_q_onnx` with a tiny calibration set (even 10–50 random-but-realistically-shaped tensors are enough to get a latency number; accuracy doesn't matter for benchmarking).

```python
# quantize.py
import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType
import vai_q_onnx

class RandCalib(CalibrationDataReader):
    def __init__(self, n=32):
        self.data = iter([{"pixel_values": np.random.randn(1,3,224,224).astype(np.float32)}
                          for _ in range(n)])
    def get_next(self):
        return next(self.data, None)

vai_q_onnx.quantize_static(
    model_input="convnext_onnx/model.onnx",
    model_output="convnext_int8.onnx",
    calibration_data_reader=RandCalib(),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    enable_dpu=True,          # VitisAI-friendly QDQ placement
    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
)
```

```powershell
python quantize.py
```

## Step 4 — Run on the NPU and print latency

Easiest option — ONNX Runtime's bundled perf tool:

```powershell
onnxruntime_perf_test.exe -e vitisai -r 200 -t 5 -I convnext_int8.onnx
```

`-r 200` runs 200 iterations, `-t 5` does 5s of warmup, `-I` lets ORT generate random inputs. It prints average + P50/P90/P99 latency and throughput.

Or in Python if you prefer:

```python
import onnxruntime as ort, numpy as np, time
sess = ort.InferenceSession(
    "convnext_int8.onnx",
    providers=[("VitisAIExecutionProvider", {"config_file": "vaip_config.json"}),
               "CPUExecutionProvider"],
)
x = np.random.randn(1,3,224,224).astype(np.float32)
# warmup
for _ in range(20): sess.run(None, {"pixel_values": x})
# measure
ts = []
for _ in range(200):
    t0 = time.perf_counter(); sess.run(None, {"pixel_values": x}); ts.append(time.perf_counter()-t0)
ts.sort()
print(f"median {ts[100]*1000:.2f} ms  p90 {ts[180]*1000:.2f} ms")
```

`vaip_config.json` ships inside the Ryzen AI install (look under `%RYZEN_AI_INSTALLATION_PATH%\voe-*\vaip_config.json`). Point at it once; it tells the VitisAI EP which XCLBIN/overlay to load.

## Sanity check — am I actually on the NPU?

After the first `sess.run`, you should see compilation log lines like `[Vitis AI EP] ... partitions assigned to DPU: N`. If `N == 0`, the model wasn't quantized properly and you're benchmarking CPU. Also watch the NPU column in Task Manager (Win11 24H2+) — it should light up.

## Rough expectation

`convnext-tiny-224` is ~28M params, ~4.5 GFLOPs/image. On Phoenix/Hawk Point (16 TOPS NPU) you should see somewhere in the ~5–15 ms range at batch 1 INT8; on Strix (50 TOPS) more like ~2–5 ms. If your number is >50 ms, you almost certainly fell back to CPU — recheck the DPU partition count.

## Why not just skip quantization?

You can — but then VitisAI EP will partition almost nothing onto the NPU (it needs QDQ ops to map to DPU subgraphs). The session will load, run, return the right answer, and report a "NPU" latency that is actually CPU. This is the #1 trap with Ryzen AI benchmarking.
