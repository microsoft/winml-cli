# eval-snapdragon-resnet-build

## Prompt

> i just got a snapdragon x elite dev box and i want to run microsoft/resnet-50 on the npu. can you walk me through getting it built and benchmarked? i want to see latency numbers at the end. give me actual commands i can copy paste.

## with_skill — 7/7

### Response

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

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✓ | Recommends `winml inspect` (or equivalent inspection step) before kicking off the rest of the pipeline. | PASS | Section 2 Inspect the model runs winml inspect -m microsoft/resnet-50 before any build step. |
| ✓ | Identifies QNN as the correct EP for Snapdragon X Elite. | PASS | Quote: 'Snapdragon X Elite means we are targeting the QNN execution provider on the Hexagon NPU.' |
| ✓ | Does NOT recommend OpenVINO, VitisAI, or other non-Qualcomm EPs as the primary path. | PASS | Only QNN; no OpenVINO/VitisAI. |
| ✓ | Walks through the build pipeline (export/optimize/quantize/compile or config+build). | PASS | Uses config+build path; lists 'export to analyze to optimize to quantize to compile' as what build runs. |
| ✓ | Includes a benchmark step (`winml perf` or equivalent) that produces latency numbers. | PASS | Section 5 winml perf with --iterations 500 --warmup 50 --monitor, reports mean/p50/p95/p99. |
| ✓ | Either directs the user to `winml --help` for current flags, OR uses only common/plausible flag patterns. | PASS | Quote: 'If any step errors out, run winml <that-command> --help -- flags evolve, and the CLI is the source of truth.' |
| ✓ | All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | All ? winml commands verified runnable against --help. |

---

## without_skill (baseline) — 6/7 ⚠

### Response

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

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✗ | Recommends `winml inspect` before kicking off the rest of the pipeline. | **FAIL** | Never mentions winml CLI at all. Goes straight to optimum-cli + onnxruntime; no support check before downloading and quantizing. |
| ✓ | Identifies QNN as the correct EP for Snapdragon X Elite. | PASS | 'ONNX Runtime with the QNN execution provider â€” that's what targets the Hexagon NPU on Snapdragon X.' |
| ✓ | Does NOT recommend OpenVINO, VitisAI, or other non-Qualcomm EPs as the primary path. | PASS | Only QNN mentioned. |
| ✓ | Walks through the build pipeline (export/optimize/quantize/compile or config+build). | PASS | Walks through env setup -> export (optimum-cli) -> static quantize (QDQ) -> provider sanity check -> perf_test. Multi-stage. |
| ✓ | Includes a benchmark step that produces latency numbers. | PASS | `onnxruntime_perf_test.exe -e qnn -r 200 -t 10 -m times` and CPU comparison. |
| ✓ | Does NOT invent fabricated flags. | PASS | Uses real onnxruntime_perf_test flags; sources Quark/quantize APIs from actual ORT docs. |
| ✓ | All `winml` commands quoted in the response are runnable — every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | Response uses no winml commands (trivially passes). |

---
