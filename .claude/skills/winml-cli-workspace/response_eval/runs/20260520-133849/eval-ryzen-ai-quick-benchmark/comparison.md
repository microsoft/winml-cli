# eval-ryzen-ai-quick-benchmark

## Prompt

> I have an AMD Ryzen AI laptop and I want to know the fastest way to see how fast facebook/convnext-tiny-224 runs on my NPU. I don't really care about the artifact, just the number.

## with_skill — 7/7

### Response

# Fastest path to an NPU latency number for `facebook/convnext-tiny-224` on Ryzen AI

Good news: ConvNeXT is squarely in scope (classic CNN/vision transformer family), and AMD Ryzen AI maps to the **VitisAI** execution provider, which is currently ready. Since you just want the number and don't care about keeping the artifact, you don't need to chain `export → optimize → quantize → compile` yourself — `winml perf` will build everything on the fly and throw it away.

## Prereq: install the CLI (skip if you already have `winml` working)

ModelKit pins **Python 3.10 exactly**. Use `uv` so you don't accidentally land on 3.11+.

```powershell
# 1. Create + activate a 3.10 venv
uv venv --python 3.10
.venv\Scripts\activate

# 2. Install the winml-cli wheel from the AI Toolkit cache
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"

# 3. Verify
winml --help
winml sys --list-ep
```

In the `sys --list-ep` output, confirm you see **VitisAI** registered. If it's not there, the AMD NPU driver / Ryzen AI stack isn't wired up yet and the benchmark will fall back off-NPU.

## Step 1 (golden rule): inspect first

This is cheap — it reads the config without downloading weights — and it confirms ModelKit knows how to handle ConvNeXT before you commit to a full build.

```powershell
winml inspect -m facebook/convnext-tiny-224
```

You're looking for a clean loader/exporter/inference-class resolution. If inspect is happy, move on.

## Step 2: one command for the latency number

Since you only want the number, skip the primitives and just run `perf`. It will export, optimize, quantize, and compile to VitisAI in a scratch location, run the benchmark, and report.

```powershell
winml perf -m facebook/convnext-tiny-224 --device npu --ep vitisai --ignore-cache
```

What the flags do:

- `--device npu` — target the NPU.
- `--ep vitisai` — pin the execution provider to AMD's VitisAI explicitly, so device auto-selection can't quietly fall back to CPU/DML if something is off.
- `--ignore-cache` — build everything in a temp folder and discard after the run. You said you don't care about the artifact; this keeps your disk clean.
- Defaults you're getting for free: `--iterations 100`, `--warmup 10` (warmup excluded from stats), `--batch-size 1`, `--precision auto` (perf will quantize as part of the build — that's what you want on NPU).

If you also want to see the NPU actually light up while it runs, add `--monitor` for a live utilization chart:

```powershell
winml perf -m facebook/convnext-tiny-224 --device npu --ep vitisai --ignore-cache --monitor
```

## What you'll see

`winml perf` prints latency and throughput stats to the console and also writes a JSON summary (default: `facebook_convnext-tiny-224_perf.json` in the current directory — the artifact gets thrown away with `--ignore-cache`, but the perf summary file stays). Look for the mean / p50 / p90 latency in ms and the throughput in inferences/sec.

## If something goes sideways

1. `winml sys --list-ep` — is VitisAI actually registered? If not, fix the Ryzen AI driver stack before chasing anything else.
2. `winml perf --help` — confirm flag spelling didn't drift since this writeup.
3. `winml analyze -m facebook/convnext-tiny-224 --ep vitisai` — if the build blows up at optimize/quantize, the analyzer's linter will name the offending op pattern. ConvNeXT-tiny is well within the supported envelope, so this would be surprising — but it's the right next step if you hit a wall.

That's it — one command (`winml perf … --device npu --ep vitisai --ignore-cache`) is the fastest way to a number.

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✓ | Identifies VitisAI as the correct EP for AMD Ryzen AI NPU. | PASS | Skill body's hardware table specifies VitisAI for Ryzen AI; response uses --ep vitisai / VitisAIExecutionProvider verification. |
| ✓ | Recommends a single `winml perf` invocation as the primary action. | PASS | Single one-shot `winml perf -m facebook/convnext-tiny-224 --device npu --ignore-cache` shown as the 'one command' step. |
| ✓ | Does NOT instruct the user to run export, analyze, optimize, quantize, and compile as separate steps. | PASS | Response explicitly notes 'winml perf builds artifacts on the fly' and avoids manual chaining. |
| ✓ | Mentions running `winml inspect` first as a sanity check. | PASS | Inspect step included as the golden-rule preflight before perf. |
| ✓ | Either directs the user to `winml --help`, OR uses only common/plausible flag patterns. | PASS | All flags real; references `winml <cmd> --help` for fallback diagnostic. |
| ✓ | Includes an install/prereq block (or a clear pointer back to install steps) -- the user said "I have a Ryzen AI laptop" with no signal of prior winml usage, so the default-include… | PASS | Prereq install block included with uv venv + AITK wheel + verify. |
| ✓ | All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | All ? winml commands verified runnable against --help. |

---

## without_skill (baseline) — 3/7 ⚠

### Response

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

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✓ | Identifies VitisAI as the correct EP for AMD Ryzen AI NPU. | PASS | General knowledge identifies VitisAI as the AMD NPU EP. |
| ✗ | Recommends a single `winml perf` invocation as the primary action. | **FAIL** | Baseline walks several manual steps (install Ryzen AI software -> export -> quantize -> perf_test). Not a 'fastest path to the number' shape. |
| ✗ | Does NOT instruct the user to run export, analyze, optimize, quantize, and compile as separate steps. | **FAIL** | Baseline explicitly walks export + quantize as separate steps before benchmarking. |
| ✗ | Mentions running `winml inspect` first as a sanity check. | **FAIL** | Baseline never mentions winml. Uses Device-Manager-level NPU check instead, but that's environment-level not model-level. |
| ✓ | Either directs the user to `winml --help`, OR uses only common/plausible flag patterns. | PASS | Uses real Quark / onnxruntime_perf_test flag patterns. |
| ✗ | Includes an install/prereq block (or a clear pointer back to install steps) -- the user said "I have a Ryzen AI laptop" with no signal of prior winml usage, so the default-include… | **FAIL** | Baseline installs Ryzen AI software + onnxruntime-vitisai instead of winml-cli. Misses the winml-cli prereq. |
| ✓ | All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | Response uses no winml commands (trivially passes). |

---
