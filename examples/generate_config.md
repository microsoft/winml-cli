# How to Generate Model Configs

## Generate All Configs

Run the generator:

```bash
uv run python scripts/generate_example_configs.py
```

To generate configs for a specific EP and/or hardware only:

```bash
uv run python scripts/generate_example_configs.py --ep qnn --hardware gpu
uv run python scripts/generate_example_configs.py --ep openvino
uv run python scripts/generate_example_configs.py --hardware npu
```

The script:
1. Calls `winml config -m <hf_id> --task <task> --device <device> --ep <ep> --precision fp16` for every EP/hardware target
2. Writes a single `examples/<ep>/<hardware>/<model_slug>/<task>_fp16_config.json` per target
3. Injects the new `eval` section from `scripts/e2e_eval/testsets/models_with_acc.json` when dataset metadata exists for the `(hf_id, task)` pair; otherwise the config is written without an `eval` section
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

## Precision

All EP/hardware targets are generated with `--precision fp16` only. The previous
NPU sweep across `w8a8` / `w8a16` / `fp16` is no longer performed.
