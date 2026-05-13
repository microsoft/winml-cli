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
