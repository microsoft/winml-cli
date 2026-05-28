# microsoft/table-transformer-detection

End-to-end build + accuracy walkthrough for `microsoft/table-transformer-detection`
(task: `object-detection`) on the NPU, using the
PubTables-1M detection validation split as the dataset.

Run all commands from the `ModelKit` repo root.

---

## 1. Build the model on NPU

Two steps: `winml config` generates a build config JSON, then `winml build`
consumes it. `--precision w8a16` is the default NPU precision; the build
produces a QDQ-quantized ONNX that executes on the NPU.

```powershell
winml config `
  -m microsoft/table-transformer-detection `
  --task object-detection `
  --device npu `
  --ep openvino `
  --precision w8a16 `
  -o build_config.json
```

```powershell
winml build `
  -c build_config.json `
  -m microsoft/table-transformer-detection `
  --device npu `
  --ep openvino `
  --use-cache
```

Artifacts land under
`~/.cache/winml/artifacts/microsoft_table-transformer-detection/` — the file
to evaluate is `objdet_*_quantized.onnx`.

---

## 2. Evaluate on NPU with `winml eval`

The PubTables-1M dataset must exist on disk first. Build it once:

```powershell
uv run python scripts/e2e_eval/datasets/build_pubtables1m_detection.py `
  --output $HOME/.cache/winml/eval_datasets/build_pubtables1m_detection
```

Then run `winml eval` against the quantized ONNX produced in step 1. Pass the
ONNX file to `-m` and the HuggingFace model ID to `--model-id` (needed for
the preprocessor / postprocessor). `--output` writes a JSON file containing
the parsed metrics:

```powershell
winml eval `
  -m $HOME/.cache/winml/artifacts/microsoft_table-transformer-detection/objdet_<hash>_quantized.onnx `
  --model-id microsoft/table-transformer-detection `
  --task object-detection `
  --device npu `
  --ep openvino `
  --dataset $HOME/.cache/winml/eval_datasets/build_pubtables1m_detection `
  --split validation `
  --samples 1000 `
  --column annotation_column=objects `
  --column bbox_key=bbox `
  --column category_key=category `
  --column box_format=xyxy `
  --output winml_eval_output.json
```

Replace `<hash>` with the actual filename produced by step 1.

The mAP value is `metrics.map` inside `winml_eval_output.json`.

---

## 3. Evaluate the original PyTorch model

`run_pytorch_baseline.py` loads the HuggingFace checkpoint with native PyTorch
on CPU and emits the same metric so the two runs are directly comparable. The
last stdout line is a single JSON object:
`{"metric": "map", "value": <float>, "num_samples": <int>}`.

Pass `--perf-iterations N` (and optionally `--perf-warmup K`, default `10`) to
also measure PyTorch inference latency. When `N > 0`, the script reuses the
HuggingFace pipeline on the first dataset sample, runs `K` untimed warmup
iterations, then `N` timed iterations, and emits a latency JSON line on
stdout immediately before the metric line. The metric line is still the
final stdout line.

```powershell
$columnsMapping = '{"annotation_column":"objects","bbox_key":"bbox","category_key":"category","box_format":"xyxy"}'

uv run python scripts/e2e_eval/run_pytorch_baseline.py `
  --model microsoft/table-transformer-detection `
  --task object-detection `
  --device cpu `
  --num-samples 1000 `
  --dataset $HOME/.cache/winml/eval_datasets/build_pubtables1m_detection `
  --split validation `
  --columns-mapping $columnsMapping `
  --winml-metric-key map `
  --perf-warmup 10 `
  --perf-iterations 100
```

The latency JSON line has the same `mean_ms` / `min_ms` / `max_ms` /
`p50_ms` / `p90_ms` / `p95_ms` / `p99_ms` keys as `winml perf` so the two
runs can be compared directly.

---

## 4. Measure latency with `winml perf`

`winml perf` benchmarks the quantized ONNX directly using random inputs
derived from the model's I/O configuration. Point `-m` at the same
`*_quantized.onnx` produced in step 1. `--warmup` iterations are excluded
from the statistics; `--iterations` is the measured sample count.

```powershell
winml perf `
  -m $HOME/.cache/winml/artifacts/microsoft_table-transformer-detection/objdet_<hash>_quantized.onnx `
  --device npu `
  --ep openvino `
  --warmup 10 `
  --iterations 100 `
  -o winml_perf_output.json
```

The output JSON contains `latency_ms` (`mean`, `min`, `max`, `p50`, `p90`,
`p95`, `p99`, `std`) and `throughput` (`samples_per_sec`, `batches_per_sec`).
Mean and p50 latency are the headline numbers; report them alongside the
device and precision used.

---

## 5. Comparing the results

For WinML, the accuracy value comes from `metrics.map` in
`winml_eval_output.json` while for the PyTorch baseline, it comes from the
last stdout line. Latency comes from `latency_ms` in `winml_perf_output.json`
for WinML and from the latency JSON line on stdout for the PyTorch baseline.

Result on CPU Intel(R) Core(TM) Ultra 7 258V:

| Model | Device | Precision | mAP | mean latency (ms) | p50 latency (ms) | Size (MB) |
|---|---|---|---|---|---|---|
| PyTorch | CPU | fp32 | 0.988714 | 620.859 | 600.336 | 115 |
| WinML (ONNX) | OpenVINO NPU | w8a16 (QDQ) | 0.9822 | 44.09 | 41.60 | 58 |
| WinML (ONNX) | OpenVINO CPU | fp32 | 0.9814 | 33.99 | 30.38 | 110 |
