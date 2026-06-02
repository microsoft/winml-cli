# How to Test Generated Configs

Before running tests, follow environment setup in the project README.


## Run All Tests

Run build + eval for all configs under a given EP and hardware:

```bash
uv run python scripts/run_example_tests.py --ep qnn --device npu
uv run python scripts/run_example_tests.py --ep qnn --device gpu
uv run python scripts/run_example_tests.py --ep openvino --device cpu
```

The script:
1. Finds all `*_config.json` files under `examples/<ep>/<hardware>/`
2. For each config, runs **build** then **eval built ONNX**:
  - `winml build -m <hf_id> -c <config> -o <build_output_dir>`
  - `winml eval -m <built_onnx> --model-id <hf_id> --ep <ep> --device <device> -c <config> -o <eval_output>`
3. Adds `--trust-remote-code` automatically when config has `dataset.build_script`
4. Skips configs that already have `_eval_result.json`, `_eval_result.error.txt`, or `_eval_result.timeout`
5. If `--retry-failed` is set, existing `*_eval_result.error.txt` / `*_eval_result.timeout` are removed and retried
6. For VitisAI EP, build is executed with `--no-compile`
7. If `--clean-cache` is set, cleans HF/winml caches between different models to save disk space
8. Safe to re-run — picks up where it left off

Options:
- `--timeout` — Per-model timeout in seconds (default: 3600)
- `--clean-cache` — Clean `~/.cache/winml` and `~/.cache/huggingface` between different models (default: disabled)
- `--rebuild` — Pass `--rebuild` to `winml build` to force rebuild instead of reusing existing build artifacts
- `--retry-failed` — Re-run configs previously marked as eval fail/timeout
- `--models` — Comma-separated model slugs to test a subset (e.g. `--models microsoft_resnet-50,BAAI_bge-base-en-v1.5`)

Results are saved alongside configs. NPU targets have one config per precision; CPU/GPU
targets have a single precision-less config:
```
examples/qnn/npu/microsoft_resnet-50/
├── image-classification_w8a8_config.json
├── image-classification_w8a8_eval_result.json     # eval results
├── image-classification_w8a16_config.json
├── image-classification_w8a16_eval_result.json
├── image-classification_fp16_config.json
├── image-classification_fp16_eval_result.error.txt  # failure
├── image-classification_fp16_build_artifacts/        # build output for eval input
└── ...

examples/mlas/cpu/microsoft_resnet-50/
├── image-classification_config.json
└── image-classification_eval_result.json
```

---

## Reference

### What the script does under the hood

For each config, it runs the equivalent of:

```bash
# 1. Build (creates ONNX under build output dir)
winml build -m microsoft/resnet-50 --device npu \
  -c examples/qnn/npu/microsoft_resnet-50/image-classification_w8a8_config.json \
  -o examples/qnn/npu/microsoft_resnet-50/image-classification_w8a8_build_artifacts

# 2. Eval built ONNX
winml eval -m examples/qnn/npu/microsoft_resnet-50/image-classification_w8a8_build_artifacts/model.onnx \
  --model-id microsoft/resnet-50 --device npu \
  --ep qnn \
  -c examples/qnn/npu/microsoft_resnet-50/image-classification_w8a8_config.json \
  -o examples/qnn/npu/microsoft_resnet-50/image-classification_w8a8_eval_result.json

# Model with dataset_script (adds --trust-remote-code to eval)
winml eval -m examples/qnn/npu/w11wo_indonesian-roberta-base-posp-tagger/token-classification_w8a8_build_artifacts/model.onnx \
  --model-id w11wo/indonesian-roberta-base-posp-tagger --device npu \
  --ep qnn \
  -c examples/qnn/npu/w11wo_indonesian-roberta-base-posp-tagger/token-classification_w8a8_config.json \
  -o examples/qnn/npu/w11wo_indonesian-roberta-base-posp-tagger/token-classification_w8a8_eval_result.json \
  --trust-remote-code
```

`run_example_tests.py` performs build explicitly and then evaluates the built ONNX artifact.

### Testing individual steps

You can also run build/eval separately:

```bash
# Build only
winml build -m <hf_id> --device npu -c <config_path> -o <output_dir>

# Eval only
winml eval -m <built_onnx_path> --model-id <hf_id> --ep <ep> --device npu -c <config_path> -o <output_dir>/<task>_<precision>_eval_result.json
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
