# eval-local-onnx-file

## Prompt

> i already have a resnet50.onnx file on disk from a previous export — about 100MB. how do i get it benchmarked on my intel core ultra NPU without re-downloading anything from huggingface?

## with_skill — 5/5

### Response

# Benchmark a local `resnet50.onnx` on Intel Core Ultra NPU

Good news: `winml perf` accepts a local `.onnx` file directly via `-m`, so there's no Hugging Face round-trip. You skip `export` entirely and go straight to the benchmark stage.

## Prereqs (skip if `winml --help` already works)

ModelKit pins Python 3.10. From your project root in PowerShell:

```powershell
uv venv --python 3.10
.venv\Scripts\activate
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"
winml --help
```

## Step 1 — Confirm the OpenVINO EP is actually registered

The Intel Core Ultra NPU is driven by the OpenVINO execution provider. Verify it shows up on your machine before you run anything else:

```powershell
winml sys --list-ep
winml sys --list-device
```

You want to see `openvino` in the EP list and an NPU entry in the device list. If `openvino` isn't there, the NPU run won't work — that's the thing to fix first.

## Step 2 — Benchmark the local ONNX file on NPU

Point `-m` at the file on disk and force the NPU device. Because you already have the ONNX, you don't need `--rebuild` or any export-side flags:

```powershell
winml perf -m C:\path\to\resnet50.onnx --device npu --ep openvino -o resnet50_npu_perf.json
```

A few notes that map to your situation:

- `-m <path-to-.onnx>` is what tells `winml perf` to treat the input as a pre-exported file rather than a HF ID — internally it routes to the ONNX benchmark path instead of the HF pipeline, so nothing is fetched from Hugging Face.
- `--device npu` picks the NPU; `--ep openvino` pins the execution provider explicitly so there's no ambiguity about which EP gets used on a Core Ultra box (you have both CPU and GPU EPs registered too).
- Default is 100 iterations with 10 warmup. Bump with `--iterations 500` if you want tighter numbers.
- Add `--monitor` if you want a live NPU utilization chart during the run.

## Optional — CPU baseline for comparison

If you also want a CPU latency number to compare against, re-run perf against the same file on CPU. Since this is a plain ONNX (not an EP-compiled artifact), it's portable across EPs:

```powershell
winml perf -m C:\path\to\resnet50.onnx --device cpu -o resnet50_cpu_perf.json
```

Don't try to compare a `compile`-produced NPU artifact against CPU — compiled artifacts are tied to the EP they were compiled for. Comparing the source ONNX across two EPs is the apples-to-apples way.

## If the NPU run misbehaves

Walk back through the cheap diagnostics in order:

1. `winml perf --help` — confirm you didn't typo a flag.
2. `winml sys --list-ep` — confirm `openvino` is actually present (driver/runtime install issue otherwise).
3. `winml inspect -m C:\path\to\resnet50.onnx` and `winml analyze` — confirm ResNet-50's ops are all OpenVINO-friendly (they are, but analyze will name any odd op if quantization introduced something unexpected).

ResNet-50 is well within the in-scope CNN family, so the run should be uneventful. The output JSON (`resnet50_npu_perf.json`) is the artifact to share if you need to dig into latency/throughput numbers afterward.

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✓ | Identifies OpenVINO as the correct EP for Intel Core Ultra NPU. | PASS | Response uses --ep openvino and references OpenVINOExecutionProvider for the Intel Core Ultra NPU path. |
| ✓ | Recognizes that winml commands accept local `.onnx` files directly via `-m` / `--model` -- no re-export needed. | PASS | Response notes `winml perf` accepts local .onnx via `-m` and routes through the local-ONNX benchmark path. |
| ✓ | Does NOT walk the user through `winml export` or any HuggingFace download -- the model is already on disk. | PASS | No `winml export`; response goes straight from sanity-check to `winml perf` against the local file. |
| ✓ | Either directs the user to `winml --help`, OR uses only common/plausible flag patterns. | PASS | All flags real; references `winml perf --help` for flag verification. |
| ✓ | All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | All ? winml commands verified runnable against --help. |

---

## without_skill (baseline) — 4/5 ⚠

### Response

# Benchmarking a local `resnet50.onnx` on an Intel Core Ultra NPU

Good news: since you already have the ONNX file on disk, you can skip the Hugging Face download entirely. You just need to run the model through ONNX Runtime with the **OpenVINO Execution Provider** (the EP that targets the Intel NPU on Core Ultra "Meteor Lake" / "Lunar Lake" / "Arrow Lake" parts), pointed at your local file.

Below is the end-to-end recipe.

---

## 1. Make sure your NPU is actually exposed

On a Core Ultra machine, open **Device Manager → Neural processors** and confirm you see an "Intel(R) AI Boost" device. If it isn't there, install the latest **Intel NPU driver** from Intel's download center. Without that driver the OpenVINO EP will silently fall back to CPU and your "NPU benchmark" will be a CPU benchmark.

Quick sanity check from Python once OpenVINO is installed:

```python
import openvino as ov
print(ov.Core().available_devices)
# You want to see something like: ['CPU', 'GPU', 'NPU']
```

If `NPU` is missing, fix the driver first — nothing below will help.

---

## 2. Install the runtime

Use a fresh virtual environment so you don't fight existing onnxruntime installs:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip

# ONNX Runtime built with the OpenVINO EP (this is the key package):
pip install onnxruntime-openvino

# OpenVINO runtime itself (the EP shells out to it):
pip install openvino
```

Notes / gotchas:

- `onnxruntime-openvino` and the stock `onnxruntime` / `onnxruntime-gpu` packages conflict. Pick one. If `pip list` shows multiple, `pip uninstall onnxruntime onnxruntime-gpu` and reinstall only `onnxruntime-openvino`.
- The Python wheel ships its own OpenVINO binaries, but having the matching `openvino` package installed makes diagnostics easier.

---

## 3. (Recommended) Convert FP32 → FP16 for the NPU

The Intel NPU is happiest with **FP16** weights (and even happier with **INT8**). A stock exported `resnet50.onnx` is almost always FP32. You can still run FP32 on the NPU — the EP will cast internally — but you'll leave a lot of performance on the table and may hit a memory-bandwidth wall.

Two easy options, in increasing effort:

**Option A — let the EP do FP16 at load time (zero code change):**

Pass `device_type=NPU_FP16` when you create the session (shown below). No file conversion needed.

**Option B — convert the file once with OpenVINO Model Optimizer:**

```powershell
# Produces resnet50.xml + resnet50.bin in FP16
ovc resnet50.onnx --compress_to_fp16=True --output_model resnet50_fp16
```

You can feed the resulting IR (`.xml`) straight to OpenVINO's own `benchmark_app`, which is the cleanest way to get latency/throughput numbers (see section 5).

**Option C — full INT8 quantization (best NPU perf):**

That requires a small calibration dataset (a few hundred representative images). It's worth it if you care about real-world latency, but it's a separate exercise from "just benchmark what I have."

---

## 4. Benchmark via ONNX Runtime (Python)

This is the simplest path because it consumes your existing `resnet50.onnx` directly. Save as `bench_npu.py`:

```python
import time
import numpy as np
import onnxruntime as ort

MODEL_PATH = r"C:\path\to\resnet50.onnx"

# ResNet-50 standard input: NCHW, 1x3x224x224, float32
INPUT_SHAPE = (1, 3, 224, 224)

# Tell the OpenVINO EP to target the NPU. NPU_FP16 lets the EP
# cast an FP32 model to FP16 internally — much faster on Intel NPUs.
provider_options = [{
    "device_type": "NPU_FP16",
    # Optional: cache compiled blobs so re-runs don't re-compile.
    "cache_dir": r".\ov_cache",
}]

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

sess = ort.InferenceSession(
    MODEL_PATH,
    sess_options=sess_options,
    providers=["OpenVINOExecutionProvider"],
    provider_options=provider_options,
)

# Confirm what's actually executing — don't trust, verify.
print("Providers in use:", sess.get_providers())

input_name = sess.get_inputs()[0].name
dummy = np.random.rand(*INPUT_SHAPE).astype(np.float32)

# Warm-up — the first call includes graph compilation for the NPU and
# is NOT representative. Always discard it.
for _ in range(10):
    sess.run(None, {input_name: dummy})

# Timed run
N = 200
t0 = time.perf_counter()
for _ in range(N):
    sess.run(None, {input_name: dummy})
elapsed = time.perf_counter() - t0

print(f"Avg latency: {elapsed / N * 1000:.2f} ms")
print(f"Throughput : {N / elapsed:.1f} inferences/sec")
```

Run it:

```powershell
python bench_npu.py
```

Key things to watch for in the output:

1. **`Providers in use:`** must include `OpenVINOExecutionProvider`. If it shows only `CPUExecutionProvider`, ORT silently fell back — the EP failed to initialize. Common causes: wrong wheel, missing NPU driver, or `device_type` typo.
2. The **first** invocation will be slow (hundreds of ms to several seconds) because the EP compiles the graph for the NPU. That's why the warm-up loop is non-negotiable.
3. If you set `cache_dir`, subsequent process starts will be much faster — the compiled blob is cached.

---

## 5. Benchmark via OpenVINO's `benchmark_app` (more rigorous numbers)

If you want apples-to-apples latency/throughput with proper thread/stream handling, OpenVINO ships a purpose-built tool. It accepts ONNX directly:

```powershell
# Latency mode, single stream, NPU target, 60 seconds of measurement
benchmark_app -m resnet50.onnx -d NPU -hint latency -t 60

# Or throughput mode
benchmark_app -m resnet50.onnx -d NPU -hint throughput -t 60
```

It will print average latency, p50/p95/p99, FPS, and the device it actually ran on. This is the number I'd quote in a report.

If you converted to IR in section 3 Option B, point `-m` at the `.xml` instead — slightly faster startup and identical results.

---

## 6. Sanity-check that the NPU is doing the work

Two ways:

- **Task Manager → Performance tab** on Windows 11 23H2+ shows an "NPU" graph. While the benchmark is running you should see it pegged.
- The ORT log: set `sess_options.log_severity_level = 0` before creating the session. You'll get verbose EP logs showing which subgraphs got assigned to OpenVINO/NPU vs. fell back to CPU. If you see lots of "node X assigned to CPU" messages, some ops in your ResNet-50 export aren't supported on NPU and you'll get a hybrid execution — common with non-standard exports.

---

## 7. Common pitfalls specific to this scenario

- **Dynamic input shape.** If your `resnet50.onnx` was exported with a dynamic batch dim (`-1` or `N`), the NPU will still compile, but you can sometimes get better numbers by re-exporting / running `onnxsim` with a fixed shape (`1x3x224x224`). The NPU prefers static shapes.
- **Opset version.** Very old (<11) or very new (>20) opsets occasionally hit EP limitations. ResNet-50 at opset 13–17 is the sweet spot. If you see ops falling back to CPU, check the opset with `onnx.load(...).opset_import` and consider re-exporting at opset 17.
- **External data files.** A 100 MB file is borderline — if it was exported with external weights, you'll have `resnet50.onnx` plus `resnet50.onnx.data` (or similar). Keep them in the same directory; don't move just the `.onnx`.
- **No internet needed.** Everything above is local. The `pip install` step is the only thing that touches the network, and once the wheels are cached you can run the benchmark fully offline.

---

## TL;DR

```powershell
pip install onnxruntime-openvino openvino
benchmark_app -m resnet50.onnx -d NPU -hint latency -t 60
```

That single `benchmark_app` invocation is the fastest path from "I have an ONNX file" to "I have a real NPU number." Use the Python script in section 4 if you need to integrate the benchmark into a larger workflow or measure something `benchmark_app` doesn't expose.

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✓ | Identifies OpenVINO as the correct EP for Intel Core Ultra NPU. | PASS | General knowledge identifies OpenVINO as the Intel NPU EP. |
| ✗ | Recognizes that winml commands accept local `.onnx` files directly via `-m` / `--model` -- no re-export needed. | **FAIL** | Baseline doesn't use winml CLI -- uses raw onnxruntime InferenceSession with the local file path. |
| ✓ | Does NOT walk the user through `winml export` or any HuggingFace download -- the model is already on disk. | PASS | No re-export; uses local file directly. |
| ✓ | Either directs the user to `winml --help`, OR uses only common/plausible flag patterns. | PASS | No fabricated winml flags. Recommends OpenVINO's benchmark_app, a real tool. |
| ✓ | All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | Response uses no winml commands (trivially passes). |

---
