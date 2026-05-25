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
