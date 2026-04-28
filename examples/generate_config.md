# How to Generate Model Configs

## Generate All Configs

Run the script to generate configs for all 48 builtin models × 3 EPs × 3 precisions = 432 configs:

```bash
python scripts/generate_example_configs.py
```

The script:
1. Calls `winml config -m <hf_id> --task <task> --device npu --ep <ep> --precision <precision>` for each combination
2. Saves the JSON config to `examples/<ep_folder>/<model_slug>/<task>_<precision>_config.json`
3. Automatically injects `eval_option` from `scripts/e2e_eval/testsets/models_with_acc.json` when available
4. Skips configs that already exist (safe to re-run)

### Adding a New Model

Add the `(hf_id, task)` tuple to the `MODELS` list in the script and re-run.

### Adding eval_option for a Model

Add a `dataset_config` entry in `scripts/e2e_eval/testsets/models_with_acc.json`. The script converts it to the `eval_option` schema automatically.

---

## Reference

### What the script does under the hood

For each model × EP × precision, it runs:

```bash
# AMD (VitisAI)
winml config -m microsoft/resnet-50 --task image-classification --device npu --ep vitisai --precision w8a8

# QNN (Qualcomm)
winml config -m microsoft/resnet-50 --task image-classification --device npu --ep qnn --precision w8a8

# OpenVINO (Intel)
winml config -m microsoft/resnet-50 --task image-classification --device npu --ep openvino --precision w8a8
```

### Precision options

| Precision | Weight | Activation | Description |
|-----------|--------|------------|-------------|
| `w8a8`    | uint8  | uint8      | Fully quantized, smallest model |
| `w8a16`   | uint8  | uint16     | Mixed precision |
| `fp16`    | float16| float16    | Half precision, no quantization |

### Directory structure

```
examples/
├── amd/           # VitisAI EP
│   └── <model_slug>/
│       ├── <task>_w8a8_config.json
│       ├── <task>_w8a16_config.json
│       └── <task>_fp16_config.json
├── qnn/           # QNN EP
│   └── <model_slug>/
│       └── ...
└── ov/            # OpenVINO EP
    └── <model_slug>/
        └── ...
```

Where `<model_slug>` = HuggingFace ID with `/` replaced by `_` (e.g., `microsoft_resnet-50`).

### EP flag reference

| EP Name   | `--ep` flag | Folder | Notes |
|-----------|-------------|--------|-------|
| VitisAI   | `vitisai`   | `amd`  | AMD NPU, `enable_ep_context=false` |
| QNN       | `qnn`       | `qnn`  | Qualcomm NPU |
| OpenVINO  | `openvino`  | `ov`   | Intel NPU/CPU |

### eval_option schema

The `eval_option` section in a config specifies evaluation dataset:

```json
{
  "eval_option": {
    "dataset": {
      "path": "rajpurkar/squad_v2",
      "split": "validation",
      "samples": 100,
      "columns_mapping": {
        "question_column": "question",
        "context_column": "context"
      }
    },
    "dataset_script": null,
    "label_mapping_file": null
  }
}
```

For models requiring a custom dataset build script:

```json
{
  "eval_option": {
    "dataset": {
      "columns_mapping": { "label_column": "pos_tags" }
    },
    "dataset_script": "scripts/e2e_eval/datasets/build_indonlu_posp.py"
  }
}
```
