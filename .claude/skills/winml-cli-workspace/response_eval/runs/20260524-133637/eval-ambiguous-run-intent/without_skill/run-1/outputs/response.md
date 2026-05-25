# Running ResNet-50 on Snapdragon X Elite with Windows ML

Great choice — Snapdragon X Elite has a Hexagon NPU that's well-suited for vision models like ResNet-50. Here's the end-to-end path using Windows ML (WinML) with ONNX Runtime and the QNN Execution Provider (which is how you actually reach the Snapdragon NPU).

## 1. Prerequisites

On the Snapdragon X Elite device (ARM64 Windows 11):

- Windows 11 (recent build, ideally 24H2 or newer)
- Python for Windows ARM64 (3.10+), or .NET / C++ tooling depending on your stack
- Visual Studio Build Tools (if you'll compile native code)
- Latest Qualcomm NPU drivers (Windows Update usually keeps these current)

For Python:

```powershell
pip install onnxruntime-qnn
pip install onnx
pip install numpy pillow
```

Note: use `onnxruntime-qnn` (the QNN-enabled build), **not** plain `onnxruntime`. The default `onnxruntime` package doesn't include the QNN EP needed to target the Hexagon NPU.

## 2. Get a ResNet-50 ONNX model

You have a few options:

**Option A — ONNX Model Zoo (simplest):**
Download `resnet50-v1-7.onnx` (or v2) from https://github.com/onnx/models/tree/main/vision/classification/resnet.

**Option B — Export from PyTorch:**

```python
import torch
import torchvision.models as models

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, "resnet50.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=17,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
```

## 3. Quantize to INT8 (recommended for NPU)

The Qualcomm NPU runs best on **quantized** models (INT8 / INT16). FP32 either falls back to CPU or runs poorly. Use ONNX Runtime's static quantization with a small calibration set (a few hundred representative images):

```python
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader

class ImageCalibReader(CalibrationDataReader):
    def __init__(self, image_paths):
        # preprocess each image to NCHW float32, 224x224, ImageNet mean/std
        self.data = iter([{"input": preprocess(p)} for p in image_paths])
    def get_next(self):
        return next(self.data, None)

quantize_static(
    "resnet50.onnx",
    "resnet50_int8.onnx",
    ImageCalibReader(calibration_images),
    quant_format=QuantType.QOperator,   # QDQ also works
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8,
)
```

You generally want **QDQ format** for QNN; check the QNN EP docs for the exact recommended settings since they evolve.

## 4. Run with the QNN Execution Provider

```python
import onnxruntime as ort
import numpy as np

so = ort.SessionOptions()

providers = [
    ("QNNExecutionProvider", {
        "backend_path": "QnnHtp.dll",   # HTP = Hexagon Tensor Processor (the NPU)
        # Optional: "htp_performance_mode": "burst",
        # Optional: "enable_htp_fp16_precision": "1",  # for FP16 models
    }),
    "CPUExecutionProvider",  # fallback
]

session = ort.InferenceSession("resnet50_int8.onnx", so, providers=providers)

# Verify it actually picked QNN:
print(session.get_providers())

# Preprocess one image to NCHW float32 [1,3,224,224]
x = preprocess("cat.jpg").astype(np.float32)
out = session.run(None, {"input": x})
print(np.argmax(out[0]))
```

`backend_path` values:
- `QnnHtp.dll` — the NPU (what you want)
- `QnnCpu.dll` — QNN reference CPU backend (for debugging only)
- `QnnGpu.dll` — Adreno GPU

## 5. Verify it's really on the NPU

A few sanity checks:

1. `session.get_providers()` should list `QNNExecutionProvider` first.
2. Open Task Manager → Performance tab → look for **NPU** utilization while inference is running in a loop.
3. Enable ORT verbose logging to see which nodes are assigned to QNN vs CPU:
   ```python
   so.log_severity_level = 0
   ```
   You'll see a partitioning summary. If lots of nodes fall back to CPU, the model isn't fully quantized or uses ops QNN doesn't support — you'll need to revisit quantization.

## 6. Other language stacks

- **C#/.NET**: use `Microsoft.ML.OnnxRuntime.QNN` NuGet package, same provider-options pattern.
- **C++**: link against the QNN-enabled ORT build and pass provider options via `Ort::SessionOptions::AppendExecutionProvider("QNN", {...})`.
- **WinML "native" API** (`Windows.AI.MachineLearning`): this is the older WinRT API and currently does **not** expose QNN device selection well — most people targeting Snapdragon NPU today go through ONNX Runtime + QNN EP directly, which is what Microsoft's current Windows ML guidance recommends.

## Common gotchas

- **Wrong package**: installing plain `onnxruntime` on ARM64 gives you no QNN EP. You must install `onnxruntime-qnn`.
- **Unquantized model**: FP32 ResNet-50 will mostly fall back to CPU. Quantize.
- **Dynamic shapes**: fix the batch dimension to 1 (or whatever you'll actually use) before quantization for best NPU coverage.
- **Mismatched ARM64 vs x64**: make sure your Python interpreter is native ARM64, not x64 running under emulation, or you'll get CPU-only performance.
- **Calibration data quality**: a handful of random images is enough to run, but accuracy will suffer. Use a few hundred representative ImageNet samples for a real deployment.

## Useful links

- ONNX Runtime QNN EP docs: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
- ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- ResNet ONNX Model Zoo: https://github.com/onnx/models/tree/main/vision/classification/resnet
- Windows ML overview: https://learn.microsoft.com/windows/ai/windows-ml/

If you tell me which language (Python / C# / C++) and whether you already have a trained model or are starting from torchvision, I can tighten this into a specific working sample.
