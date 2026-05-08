# How to Generate Model Configs

## Generate All Configs

Run the generator:

```bash
uv run python scripts/generate_example_configs.py
```

The script:
1. Calls `winml config -m <hf_id> --task <task> --device <device> --ep <ep> --precision <precision>`
2. Writes `examples/<ep>/<hardware>/<model_slug>/<task>_<precision>_config.json`
3. Injects the new `eval` section from `scripts/e2e_eval/testsets/models_with_acc.json` when dataset metadata exists
4. Skips files that already exist

## EP and Device Matrix Used by Generator

The generator currently writes 9 EP × hardware targets:

| EP | `--ep` flag | Hardware | Output Path Prefix |
|----|-------------|----------|--------------------|
| QNN | `qnn` | `npu` | `examples/qnn/npu/` |
| QNN | `qnn` | `gpu` | `examples/qnn/gpu/` |
| Intel OpenVINO | `openvino` | `npu` | `examples/openvino/npu/` |
| Intel OpenVINO | `openvino` | `cpu` | `examples/openvino/cpu/` |
| Intel OpenVINO | `openvino` | `gpu` | `examples/openvino/gpu/` |
| AMD VitisAI | `vitisai` | `npu` | `examples/vitisai/npu/` |
| NVIDIA TensorRT | `nv_tensorrt_rtx` | `gpu` | `examples/nv_tensorrt_rtx/gpu/` |
| MLAS | `cpu` | `cpu` | `examples/mlas/cpu/` |
| DML | `dml` | `gpu` | `examples/dml/gpu/` |

Reference (`--help`) accepted EP names include:
`qnn`, `dml`, `nv_tensorrt_rtx`, `vitisai`, `openvino`, `cpu`.

## New Eval Schema

Configs now use `eval` (not `eval_option`):

```json
{
  "eval": {
    "task": "zero-shot-image-classification",
    "device": "npu",
    "dataset": {
      "path": "uoft-cs/cifar100",
      "split": "test",
      "samples": 1000,
      "shuffle": true,
      "columns_mapping": {
        "input_column": "img",
        "label_column": "fine_label"
      }
    }
  }
}
```

If a custom dataset builder is needed, use `dataset.build_script`:

```json
{
  "eval": {
    "task": "token-classification",
    "device": "npu",
    "dataset": {
      "columns_mapping": { "label_column": "pos_tags" },
      "build_script": "scripts/e2e_eval/datasets/build_indonlu_posp.py"
    }
  }
}
```

## Adding a New Model

Add `(hf_id, task)` to `MODELS` in `scripts/generate_example_configs.py`, then rerun generation.

## Precision Options

| Precision | Weight | Activation | Description |
|-----------|--------|------------|-------------|
| `w8a8` | uint8 | uint8 | Fully quantized, smallest model |
| `w8a16` | uint8 | uint16 | Mixed precision |
| `fp16` | float16 | float16 | Half precision, no quantization |
