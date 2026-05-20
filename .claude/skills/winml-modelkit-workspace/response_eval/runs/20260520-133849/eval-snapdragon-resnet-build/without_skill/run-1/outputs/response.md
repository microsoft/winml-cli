# Running microsoft/resnet-50 on Snapdragon X Elite NPU

Nice — Snapdragon X Elite has a Hexagon NPU exposed through the **QNN (Qualcomm Neural Network)** execution provider in ONNX Runtime. The end-to-end flow looks like:

1. Export the Hugging Face model to ONNX
2. Quantize to INT8 (required — the Hexagon HTP backend runs INT8/INT16, not FP32)
3. Run inference with `onnxruntime-qnn` and benchmark latency

Below is a clean copy-paste path. I'm assuming you're on a fresh ARM64 Windows box with PowerShell open.

---

## 1. Set up a Python environment

Snapdragon X Elite is ARM64, so make sure you grab the **ARM64 build of Python 3.11** (3.12 also works). Once installed:

```powershell
# Create and activate a venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Core dependencies
pip install torch torchvision transformers
pip install onnx onnxruntime
pip install optimum[exporters]
pip install numpy pillow
```

Then add the QNN execution provider package. This is the one that actually targets the Hexagon NPU:

```powershell
pip install onnxruntime-qnn
```

> If `pip install onnxruntime-qnn` complains about a wheel, you need the ARM64 Python build. Confirm with `python -c "import platform; print(platform.machine())"` — it should print `ARM64`.

---

## 2. Export microsoft/resnet-50 to ONNX

Easiest path is `optimum-cli`:

```powershell
optimum-cli export onnx --model microsoft/resnet-50 --task image-classification resnet50_onnx/
```

That gives you `resnet50_onnx/model.onnx` plus the preprocessor config. Verify it opened cleanly:

```powershell
python -c "import onnx; m = onnx.load('resnet50_onnx/model.onnx'); onnx.checker.check_model(m); print('OK, opset:', m.opset_import[0].version)"
```

You want opset 17 or higher for best QNN coverage. If it's lower, upgrade it:

```powershell
python -c "import onnx; from onnx import version_converter; m = onnx.load('resnet50_onnx/model.onnx'); m2 = version_converter.convert_version(m, 17); onnx.save(m2, 'resnet50_onnx/model.onnx')"
```

---

## 3. Quantize to INT8 (static quantization)

The Hexagon HTP backend won't run FP32. You need **static** (calibrated) INT8 quantization — dynamic quantization isn't supported on QNN HTP. Save the following as `quantize.py`:

```python
# quantize.py
import os
import numpy as np
from PIL import Image
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, QuantFormat
from transformers import AutoImageProcessor

MODEL_IN  = "resnet50_onnx/model.onnx"
MODEL_OUT = "resnet50_onnx/model.int8.onnx"
CALIB_DIR = "calib_images"  # drop ~50-200 representative JPEGs here

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

class ImageNetCalib(CalibrationDataReader):
    def __init__(self, folder):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.it = iter(self.files)

    def get_next(self):
        try:
            path = next(self.it)
        except StopIteration:
            return None
        img = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="np")
        # ResNet ONNX input name is usually "pixel_values"
        return {"pixel_values": inputs["pixel_values"].astype(np.float32)}

quantize_static(
    MODEL_IN,
    MODEL_OUT,
    ImageNetCalib(CALIB_DIR),
    quant_format=QuantFormat.QDQ,        # QNN wants QDQ format
    activation_type=QuantType.QUInt8,    # uint8 activations
    weight_type=QuantType.QInt8,         # int8 weights
    per_channel=False,                   # QNN HTP prefers per-tensor
)
print("Wrote", MODEL_OUT)
```

You need a small calibration set — 50 to a few hundred representative images is plenty for ResNet. Drop them in `calib_images/`. ImageNet validation samples are ideal; in a pinch you can use any natural photos.

```powershell
mkdir calib_images
# ...copy some .jpg files into calib_images...
python quantize.py
```

Output: `resnet50_onnx/model.int8.onnx`.

---

## 4. Run a sanity check on the NPU

Save as `run_qnn.py`:

```python
# run_qnn.py
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoImageProcessor

MODEL = "resnet50_onnx/model.int8.onnx"

# QNN provider options — "HTP" is the NPU backend on Snapdragon X Elite.
# "CPU" backend also exists for debugging, but won't hit the NPU.
provider_options = [{
    "backend_path": "QnnHtp.dll",
    # Optional: enable graph caching so the compiled context is reused
    # "qnn_context_cache_enable": "1",
    # "qnn_context_cache_path": "resnet50.qnn.bin",
}]

sess = ort.InferenceSession(
    MODEL,
    providers=[("QNNExecutionProvider", provider_options[0]), "CPUExecutionProvider"],
)
print("Active providers:", sess.get_providers())

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
img = Image.open("calib_images/" + __import__("os").listdir("calib_images")[0]).convert("RGB")
x = processor(images=img, return_tensors="np")["pixel_values"].astype(np.float32)

out = sess.run(None, {"pixel_values": x})
print("Top-5:", np.argsort(out[0][0])[-5:][::-1])
```

```powershell
python run_qnn.py
```

If `Active providers:` lists `QNNExecutionProvider` first, you're on the NPU. If it silently fell back to `CPUExecutionProvider`, an op probably wasn't supported by the quantized model — see troubleshooting at the bottom.

---

## 5. Benchmark latency

Save as `bench.py`:

```python
# bench.py
import time, statistics
import numpy as np
import onnxruntime as ort

MODEL = "resnet50_onnx/model.int8.onnx"
WARMUP = 20
ITERS  = 200

def make_session(provider, opts=None):
    return ort.InferenceSession(
        MODEL,
        providers=[(provider, opts)] if opts else [provider],
    )

def bench(sess, name):
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    feed = {sess.get_inputs()[0].name: x}
    for _ in range(WARMUP):
        sess.run(None, feed)
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    print(f"{name:8s}  mean={statistics.mean(times):6.2f} ms  "
          f"p50={times[ITERS//2]:6.2f}  p90={times[int(ITERS*0.9)]:6.2f}  "
          f"p99={times[int(ITERS*0.99)]:6.2f}  min={times[0]:6.2f}")

# CPU baseline (FP32 model)
bench(make_session("CPUExecutionProvider"), "CPU")

# NPU (INT8 QDQ via QNN HTP)
qnn_opts = {"backend_path": "QnnHtp.dll"}
bench(ort.InferenceSession(MODEL, providers=[("QNNExecutionProvider", qnn_opts)]), "NPU")
```

```powershell
python bench.py
```

You should see something roughly in this ballpark on Snapdragon X Elite for ResNet-50 @ batch=1, 224x224:

- **CPU (FP32)**: ~20–40 ms
- **NPU (INT8, HTP)**: ~1–4 ms

Exact numbers depend on driver version, power mode (plug in the laptop and set the Windows power slider to **Best Performance**), and whether the QNN context is cached. First run after a reboot is slower because the HTP graph has to be compiled — that's why warmup matters.

---

## Troubleshooting

- **"QNNExecutionProvider not in available providers"** → `pip install onnxruntime-qnn` didn't land, or you have both `onnxruntime` and `onnxruntime-qnn` installed and the wrong one is winning. Uninstall plain `onnxruntime` and reinstall `onnxruntime-qnn`.
- **Silently falls back to CPU** → run with `ort.SessionOptions(); so.log_severity_level = 0` to see which nodes were rejected. Common culprits: per-channel quant on conv weights (switch to per-tensor), or an unsupported op like a stray `Cast` to int64. You can also dump `sess.get_providers()` to confirm.
- **Accuracy looks terrible after quantization** → bump the calibration set size, and try `CalibrationMethod.Percentile` or `Entropy` in `quantize_static` instead of the default MinMax.
- **First inference is 200+ ms, rest are fast** → that's QNN compiling the HTP context. Enable `qnn_context_cache_enable=1` and point `qnn_context_cache_path` to a `.bin` file so subsequent runs reuse the compiled graph.

---

## TL;DR command sequence

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch torchvision transformers onnx onnxruntime onnxruntime-qnn optimum[exporters] numpy pillow
optimum-cli export onnx --model microsoft/resnet-50 --task image-classification resnet50_onnx/
# drop ~100 JPEGs into calib_images\
python quantize.py
python run_qnn.py
python bench.py
```

That's the whole pipeline: HF -> ONNX -> static INT8 QDQ -> QNN HTP -> latency numbers.
