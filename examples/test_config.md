# How to Test Generated Configs

Before running tests, follow environment setup in the project README.


## Run All Tests

Run perf + eval for all configs under a given EP and hardware:

```bash
uv run python scripts/run_example_tests.py --ep qnn --hardware npu --device npu
uv run python scripts/run_example_tests.py --ep qnn --hardware gpu --device gpu
uv run python scripts/run_example_tests.py --ep openvino --hardware cpu --device cpu
```

The script:
1. Finds all `*_config.json` files under `examples/<ep>/<hardware>/`
2. For each config, runs **perf** then **eval**:
   - `winml perf -m <hf_id> --device <device> -c <config> -o <perf_output>`
  - `winml eval -m <hf_id> --ep <ep> --device <device> -c <config> -o <eval_output>`
3. Adds `--trust-remote-code` automatically when config has `dataset_script`
4. Skips configs that already have `_perf_result.json`/`_eval_result.json`, `_error.txt`, or `.timeout` results
5. Cleans HF/winml caches between different models to save disk space
6. Safe to re-run — picks up where it left off

Result files intentionally use `_perf_result.json` and `_eval_result.json` because
`*_perf.json` is often covered by gitignore patterns and easy to miss in commits.

Options:
- `--timeout` — Per-model timeout in seconds (default: 1200)
- `--eval-only` — Skip perf, only run eval
- `--models` — Comma-separated model slugs to test a subset (e.g. `--models microsoft_resnet-50,BAAI_bge-base-en-v1.5`)

Results are saved alongside configs. NPU targets have one config per precision; CPU/GPU
targets have a single precision-less config:
```
examples/qnn/npu/microsoft_resnet-50/
├── image-classification_w8a8_config.json
├── image-classification_w8a8_perf_result.json     # perf results
├── image-classification_w8a8_eval_result.json     # eval results
├── image-classification_w8a16_config.json
├── image-classification_w8a16_perf_result.json
├── image-classification_w8a16_eval_result.json
├── image-classification_fp16_config.json
├── image-classification_fp16_eval_result.error.txt  # failure
└── ...

examples/mlas/cpu/microsoft_resnet-50/
├── image-classification_config.json
├── image-classification_perf_result.json
└── image-classification_eval_result.json
```

---

## Reference

### What the script does under the hood

For each config, it runs the equivalent of:

```bash
# 1. Perf (builds model if needed, then measures latency)
winml perf -m microsoft/resnet-50 --device npu \
  -c examples/qnn/npu/microsoft_resnet-50/image-classification_w8a8_config.json \
  -o examples/qnn/npu/microsoft_resnet-50/image-classification_w8a8_perf_result.json

# 2. Eval (uses cached build artifacts from perf)
winml eval -m microsoft/resnet-50 --device npu \
  --ep qnn \
  -c examples/qnn/npu/microsoft_resnet-50/image-classification_w8a8_config.json \
  -o examples/qnn/npu/microsoft_resnet-50/image-classification_w8a8_eval_result.json

# Model with dataset_script (adds --trust-remote-code to eval)
winml eval -m w11wo/indonesian-roberta-base-posp-tagger --device npu \
  --ep qnn \
  -c examples/qnn/npu/w11wo_indonesian-roberta-base-posp-tagger/token-classification_w8a8_config.json \
  -o examples/qnn/npu/w11wo_indonesian-roberta-base-posp-tagger/token-classification_w8a8_eval_result.json \
  --trust-remote-code
```

Both `winml perf` and `winml eval` automatically build the model (export → optimize → quantize → compile) if no artifacts exist.

### Testing individual steps

You can also run build/perf/eval separately:

```bash
# Build only
winml build -m <hf_id> --device npu -c <config_path> -o <output_dir>

# Perf only
winml perf -m <hf_id> --device npu -c <config_path> -o <output_dir>/<task>_<precision>_perf_result.json

# Eval only
winml eval -m <hf_id> --ep <ep> --device npu -c <config_path> -o <output_dir>/<task>_<precision>_eval_result.json
```

### Result status

- **PASS**: `_eval_result.json` exists with metrics
- **FAIL**: `_eval_result.error.txt` exists with error details
- **TIMEOUT**: `_eval_result.timeout` exists (exceeded timeout)

### Eval JSON structure

```json
{
  "config": { "model_id": "...", "task": "...", "device": "..." },
  "metrics": {
    "accuracy": 0.85,
    "f1": 0.82,
    "total_time_in_seconds": 10.5,
    "samples_per_second": 9.5
  }
}
```

### Cleanup

Build artifacts can consume significant disk space. Clean caches:

```powershell
Remove-Item "$env:USERPROFILE\.cache\winml" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:USERPROFILE\.cache\huggingface" -Recurse -Force -ErrorAction SilentlyContinue
```
