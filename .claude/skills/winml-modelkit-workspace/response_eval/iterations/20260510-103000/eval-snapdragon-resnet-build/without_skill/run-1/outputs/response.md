Nice, the X Elite NPU (Hexagon) is fun to play with. The cleanest path is **ONNX Runtime with the QNN execution provider** — that's what targets the Hexagon NPU on Snapdragon X. Below is an end-to-end flow: export ResNet-50 from HF, quantize it (the NPU only runs quantized models efficiently — fp32 will silently fall back to CPU), then benchmark with `onnxruntime_perf_test`.

Open a **PowerShell** terminal on the Snapdragon box. Everything below assumes ARM64 Python (check with `python -c "import platform; print(platform.machine())"` — should say `ARM64`).

### 1. Set up the environment

```powershell
mkdir C:\resnet-npu
cd C:\resnet-npu

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install onnx onnxruntime-qnn optimum[exporters] transformers pillow numpy
```

`onnxruntime-qnn` is the ARM64 build that ships the QNN EP and the Hexagon backend DLLs. Don't install plain `onnxruntime` alongside it — pick one.

### 2. Export microsoft/resnet-50 to ONNX

```powershell
optimum-cli export onnx --model microsoft/resnet-50 --task image-classification .\resnet50-onnx
```

That gives you `.\resnet50-onnx\model.onnx` (fp32).

### 3. Quantize to uint8 (required for NPU)

QNN HTP wants a quantized model. Use ORT's static quantizer with a tiny calibration set. Save this as `quantize.py`:

```python
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image
import numpy as np, glob, os

SRC = r".\resnet50-onnx\model.onnx"
PREP = r".\resnet50-onnx\model.prep.onnx"
DST = r".\resnet50-onnx\model.qdq.onnx"

quant_pre_process(SRC, PREP, skip_symbolic_shape=False)

class Reader(CalibrationDataReader):
    def __init__(self, folder):
        self.files = glob.glob(os.path.join(folder, "*.jp*g")) + glob.glob(os.path.join(folder, "*.png"))
        self.i = 0
    def get_next(self):
        if self.i >= len(self.files): return None
        img = Image.open(self.files[self.i]).convert("RGB").resize((224, 224))
        self.i += 1
        a = np.asarray(img, dtype=np.float32) / 255.0
        a = (a - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        a = a.transpose(2, 0, 1)[None].astype(np.float32)
        return {"pixel_values": a}

# Drop ~20-50 sample images in .\calib\
quantize_static(
    PREP, DST, Reader(r".\calib"),
    quant_format="QDQ", activation_type=QuantType.QUInt8, weight_type=QuantType.QUInt8,
    per_channel=False,
)
print("wrote", DST)
```

Put 20-50 random JPEGs into `.\calib\` (any natural images — ImageNet samples are ideal) and run:

```powershell
mkdir calib
# copy some .jpg files into .\calib\
python quantize.py
```

### 4. Sanity-check it actually loads on the NPU

```powershell
python -c "import onnxruntime as ort; s = ort.InferenceSession(r'.\resnet50-onnx\model.qdq.onnx', providers=[('QNNExecutionProvider', {'backend_path': 'QnnHtp.dll'})]); print(s.get_providers())"
```

You want to see `QNNExecutionProvider` first in the list. If you only see `CPUExecutionProvider`, the model fell back — usually means it didn't quantize cleanly.

### 5. Benchmark with onnxruntime_perf_test

The perf tool ships inside the `onnxruntime-qnn` wheel:

```powershell
$perf = (Get-ChildItem -Recurse .\.venv -Filter onnxruntime_perf_test.exe | Select-Object -First 1).FullName
echo $perf

# NPU run
& $perf -e qnn -i "backend_path|QnnHtp.dll" -r 200 -t 10 -m times .\resnet50-onnx\model.qdq.onnx

# CPU baseline for comparison
& $perf -e cpu -r 200 -t 10 -m times .\resnet50-onnx\model.qdq.onnx
```

Flags: `-r 200` = 200 iterations, `-t 10` = 10s warmup window, `-m times` = report latency. You'll get average/p50/p90/p95/p99 latency in ms.

### What to expect

On X Elite with quantized ResNet-50 at batch 1 / 224x224 you should see roughly **1-3 ms per inference on the NPU** vs. ~10-20 ms on CPU. If your NPU number is suspiciously close to CPU, the model probably fell back — re-run step 4 to confirm the provider.

### Common gotchas

- **`Could not load QnnHtp.dll`** — make sure you installed `onnxruntime-qnn` (not vanilla `onnxruntime`) and that the venv is ARM64.
- **Lots of nodes assigned to CPU** — add `-v` to perf_test or set `ORT_LOGGING_LEVEL=0`; usually means an op wasn't quantized. Re-run quantization with `quant_pre_process` (already in the script above).
- **First inference is slow** — that's HTP graph compilation. `-t 10` warmup handles it. For production you can cache the compiled context with the `qnn_context_cache_enable` provider option.

Report back the perf_test output if anything looks off and I can help dig in.
