# eval-ryzen-ai-quick-benchmark

## Prompt

> I have an AMD Ryzen AI laptop and I want to know the fastest way to see how fast facebook/convnext-tiny-224 runs on my NPU. I don't really care about the artifact, just the number.

## with_skill — 5/6 ⚠

### Response

# Fastest path to an NPU latency number for `facebook/convnext-tiny-224`

ConvNeXT is a CNN — in scope for ModelKit — and you don't care about the artifact, so skip the build pipeline entirely and let `winml perf` do download + export + optimize + quantize + compile in one shot, then time it.

## Two commands

```powershell
# Sanity check: confirm the Ryzen AI NPU is visible and VitisAI is registered
uv run winml sys --list-device --list-ep

# Benchmark on the NPU
uv run winml perf -m facebook/convnext-tiny-224 --device npu
```

That's it. `--device npu` on a Ryzen AI box routes to the VitisAI EP. `perf` will print mean/median/p95 latency and throughput to the console and drop a `convnext-tiny-224_perf.json` next to where you ran it (you can ignore the file — the number is in the table).

## A few things worth knowing

- **Defaults**: 10 warmup iterations, 100 timed iterations, batch size 1. That's enough for a stable mean on a small CNN. Bump with `--iterations 500` if you want tighter p95s.
- **First run is slow**: the first invocation downloads the HF model, exports to ONNX, optimizes, quantizes, and compiles for VitisAI. The *reported latency* excludes all that — it only times inference iterations after warmup — but wall-clock time to first number can be a few minutes. Subsequent runs hit the cache.
- **If you want to discard the cached artifact** after you've got the number: add `--ignore-cache` and `perf` builds in a temp folder and throws it away.
- **Live NPU utilization** while it runs: add `--monitor`. Useful for confirming the NPU is actually being hit and not silently falling back to CPU.
- **Sanity-check the EP**: in the `perf` output, verify the execution provider line says `VitisAIExecutionProvider`. If it says `CPUExecutionProvider`, the VitisAI EP didn't load on your machine and the number is meaningless — `winml sys --list-ep` will tell you why.

If `--device npu` ever fails to resolve, you can pin it explicitly with `--ep vitisai`.

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✓ | Identifies VitisAI as the correct EP for AMD Ryzen AI NPU. | PASS | Quote: '--device npu on a Ryzen AI box routes to the VitisAI EP.' |
| ✓ | Recommends a single `winml perf` invocation (or equivalent one-shot benchmark) as the primary action. | PASS | Two commands (sys + perf). Quote: 'That is it. perf will print mean/median/p95 latency'. |
| ✓ | Does NOT instruct the user to run all of export, analyze, optimize, quantize, and compile as separate explicit steps. | PASS | Quote: 'skip the build pipeline entirely and let winml perf do download + export + optimize + quantize + compile in one shot'. |
| ✗ | Mentions running `winml inspect` first as a sanity check. | **FAIL** | Response uses prior knowledge ('ConvNeXT is a CNN -- in scope for ModelKit') and winml sys as sanity checks, but does NOT run winml inspect. |
| ✓ | Either directs the user to `winml --help`, OR uses only common/plausible flag patterns. | PASS | All flags verified live via winml <cmd> --help. |
| ✓ | All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | All ? winml commands verified runnable against --help. |

---

## without_skill (baseline) — 3/6 ⚠

### Response

For a "just give me the number" NPU benchmark on Ryzen AI, the fastest path is to use ONNX Runtime with the VitisAI EP and run `onnxruntime_perf_test`. Roughly:

### 1. Install prerequisites (one-time)

Install the Ryzen AI Software (gives you the NPU driver + VitisAI EP + the right ORT build):
- Download from AMD: https://ryzenai.docs.amd.com/ (the "Installation Instructions" page).
- It bundles the NPU driver, `onnxruntime-vitisai`, and the XCLBIN files for your NPU (Phoenix / Hawk Point / Strix).
- Confirm the NPU is alive: Device Manager should show "NPU Compute Accelerator Device", and `Get-CimInstance Win32_PnPEntity | ? { $_.Name -match 'NPU' }` should list it.

Then in a fresh conda env (the Ryzen AI installer can create one for you, usually `ryzen-ai-1.x`):

```powershell
conda activate ryzen-ai-1.x   # name varies by version
pip install optimum onnx
```

### 2. Export the model to ONNX

`facebook/convnext-tiny-224` is an image classifier on HF. Easiest export:

```powershell
optimum-cli export onnx --model facebook/convnext-tiny-224 --task image-classification convnext-tiny-224-onnx
```

That produces `convnext-tiny-224-onnx/model.onnx`.

### 3. Quantize to INT8 (required for the NPU)

The Ryzen AI NPU only runs quantized models. The Ryzen AI install ships `quark` (or older installs ship `vai_q_onnx`). The "I don't care about accuracy, just want the number" path is to do a quick static quant with random calibration data:

```powershell
# Using Quark (Ryzen AI 1.2+)
quark quantize --input_model convnext-tiny-224-onnx/model.onnx ^
               --output_model convnext-int8.onnx ^
               --config XINT8
```

If you're on older Ryzen AI, the equivalent is `python -m vai_q_onnx.tools.quantize ...` — the docs page "Model Quantization" has the exact incantation for your version.

### 4. Benchmark on the NPU

Use ORT's built-in perf tool — it prints throughput / latency directly:

```powershell
onnxruntime_perf_test.exe -e vitisai -r 200 -I convnext-int8.onnx
```

- `-e vitisai` runs on the NPU via the VitisAI EP.
- `-r 200` = 200 inference iterations.
- `-I` = ignore missing input shapes / generate random inputs (so you don't need a real image).

First run is slow because VitisAI compiles the model and caches it under `%LOCALAPPDATA%\...\vaip\.cache`. The number you want is from the second run (or just look at the "Average inference time" line in the first run — it excludes compilation).

For an apples-to-apples comparison, run the same command with `-e cpu` against the unquantized FP32 model.

### Gotchas

- If `perf_test` reports that ops fell back to CPU, ConvNeXt has some layernorm/GELU patterns that older VitisAI versions don't fully partition — the printed "Total nodes on NPU vs CPU" line will tell you. Ryzen AI 1.2+ handles ConvNeXt much better than 1.0.
- Make sure the `XLNX_VART_FIRMWARE` env var points at the right XCLBIN for your silicon (the installer sets this, but if you switched envs it may be missing).

If you just want a sanity-check number before doing all this, AMD publishes ConvNeXt-Tiny INT8 results in their model zoo benchmarks — usually ~3–5 ms/inference on Strix NPU, ~6–9 ms on Phoenix.

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✓ | Identifies VitisAI as the correct EP for AMD Ryzen AI NPU. | PASS | 'use ONNX Runtime with the VitisAI EP and run `onnxruntime_perf_test`.' |
| ✗ | Recommends a single one-shot benchmark as the primary action. | **FAIL** | Walks through 4 explicit steps: install Ryzen AI software, export, INT8 quantize with Quark, then run perf_test. Not a 'fastest path to the number' answer. |
| ✗ | Does NOT instruct the user to run export, optimize, quantize, and compile as separate steps. | **FAIL** | Steps 2 and 3 explicitly require `optimum-cli export onnx` and then `quark quantize` before the user can benchmark. |
| ✗ | Mentions running `winml inspect` first as a sanity check. | **FAIL** | Never mentions winml. Performs a Device-Manager NPU check instead, but that's environment-level, not model-level. |
| ✓ | Either directs the user to `--help`, OR uses only common/plausible flag patterns. | PASS | Real Quark and onnxruntime_perf_test flags. |
| ✓ | All `winml` commands quoted in the response are runnable — every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | Response uses no winml commands (trivially passes). |

---
