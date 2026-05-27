# microsoft/table-transformer-detection — QNN EP example

End-to-end build + accuracy walkthrough for `microsoft/table-transformer-detection`
(task: `object-detection`) on the QNN execution provider (NPU) using the
PubTables-1M detection validation split as the dataset.

Registry entry:

```json
{
  "hf_id": "microsoft/table-transformer-detection",
  "task": "object-detection",
  "model_type": "table_transformer",
  "group": "Top200",
  "priority": "P1",
  "dataset_config": {
    "build_script": "scripts/e2e_eval/datasets/build_pubtables1m_detection.py",
    "path": "~/.cache/winml/eval_datasets/build_pubtables1m_detection",
    "split": "validation",
    "metric": "map",
    "winml_metric_key": "map",
    "columns_mapping": {
      "annotation_column": "objects",
      "bbox_key": "bbox",
      "category_key": "category",
      "box_format": "xyxy"
    }
  }
}
```

Run all commands from the `ModelKit` repo root.

---

## 1. Build the model on QNN EP

`winml build` can auto-generate the build config from `-m`, so no separate
`winml config` step is required. `--precision w8a16` matches the default that
`run_eval.py` applies on NPU. `--compile` produces the EPContext-compiled ONNX
that QNN executes on the NPU.

```powershell
uv run winml build `
  -m microsoft/table-transformer-detection `
  --task object-detection `
  --device npu `
  --ep qnn `
  --precision w8a16 `
  --compile `
  --use-cache
```

Artifacts land under `~/.cache/winml/artifacts/microsoft_table-transformer-detection/`
(look for `objdet_*_model.onnx` and the matching `*_qnn_ctx.onnx`).

---

## 2. Evaluate on QNN EP with `winml eval`

The PubTables-1M dataset must exist on disk first. Build it once with the
dataset script referenced by the registry:

```powershell
uv run python scripts/e2e_eval/datasets/build_pubtables1m_detection.py `
  --output $HOME/.cache/winml/eval_datasets/build_pubtables1m_detection
```

Then run `winml eval`. The `--column` flags re-encode the registry's
`columns_mapping`, and `--output` writes the parsed metrics JSON that
`run_eval.py` consumes:

```powershell
uv run winml eval `
  -m microsoft/table-transformer-detection `
  --task object-detection `
  --device npu `
  --ep qnn `
  --dataset $HOME/.cache/winml/eval_datasets/build_pubtables1m_detection `
  --split validation `
  --samples 1000 `
  --column annotation_column=objects `
  --column bbox_key=bbox `
  --column category_key=category `
  --column box_format=xyxy `
  --output winml_eval_output.json
```

The mAP value is `metrics.map` inside `winml_eval_output.json`.

---

## 3. Evaluate the original PyTorch model

`run_pytorch_baseline.py` loads the HuggingFace checkpoint with native PyTorch
on CPU and emits the same metric so the two runs are directly comparable. The
last stdout line is a single JSON object: `{"metric": "map", "value": <float>,
"num_samples": <int>}`.

```powershell
uv run python scripts/e2e_eval/run_pytorch_baseline.py `
  --model microsoft/table-transformer-detection `
  --task object-detection `
  --device cpu `
  --num-samples 1000 `
  --dataset $HOME/.cache/winml/eval_datasets/build_pubtables1m_detection `
  --split validation `
  --columns-mapping '{\"annotation_column\":\"objects\",\"bbox_key\":\"bbox\",\"category_key\":\"category\",\"box_format\":\"xyxy\"}' `
  --winml-metric-key map
```

---

## 4. Comparing the results

| Run | Engine | Device | Precision | Metric (mAP) | Notes |
|---|---|---|---|---|---|
| PyTorch baseline | `transformers` eager | CPU | fp32 | _value from step 3 stdout_ | Ground truth |
| WinML / QNN | ONNX Runtime + QNN EP | NPU | w8a16 (QDQ) | _value from `winml_eval_output.json` → `metrics.map`_ | Production target |

Compute the delta the same way `run_eval.py` does
(`utils.accuracy.compute_delta`):

- **Absolute delta** = `winml_map - baseline_map`
- **Relative delta** = `(winml_map - baseline_map) / baseline_map`

Verdict guide (from `utils.accuracy.derive_verdict`):

- `ACCURACY_PASS` — relative delta within tolerance
- `ACCURACY_AT_RISK` — small regression, watch for trend
- `ACCURACY_REGRESSION` — exceeds tolerance; investigate the quantization /
  EPContext partitioning before promoting the build

To run all three signals (perf + WinML eval + PyTorch baseline) end-to-end in
one shot, the registry-driven runner does it for you:

```powershell
uv run python scripts/e2e_eval/run_eval.py `
  --hf-model microsoft/table-transformer-detection `
  --task object-detection `
  --device npu `
  --ep qnn `
  --eval-type both
```

Per-model results land in
`eval_results/<date>/models/microsoft__table-transformer-detection__object-detection/eval_result.json`,
which contains both the QNN and PyTorch mAP values plus
`delta_absolute` / `delta_relative`.
