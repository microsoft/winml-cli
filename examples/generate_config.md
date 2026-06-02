# How to Generate Model Configs

## Generate All Configs

Run the generator:

```bash
uv run python scripts/generate_example_configs.py
```

To generate configs for a specific EP, device, and/or model subset only:

```bash
uv run python scripts/generate_example_configs.py --ep qnn --device gpu
uv run python scripts/generate_example_configs.py --ep openvino
uv run python scripts/generate_example_configs.py --device npu
uv run python scripts/generate_example_configs.py --models laion/CLIP-ViT-B-32-laion2B-s34B-b79K
```

The script:
1. Calls `winml config -m <hf_id> --task <task> --device <device> --ep <ep> -o <out_file>` (with `--precision <precision>` only on NPU targets)
2. Writes:
   - NPU targets: `examples/<ep>/<device>/<model_slug>/<task>_<precision>_config.json` for each of `w8a8`, `w8a16`, `fp16`
   - CPU/GPU targets: a single `examples/<ep>/<device>/<model_slug>/<task>_config.json` (no precision in name, EP default precision)
   - Composite models (e.g. CLIP zero-shot): one file per sub-component, named `<stem>_<role>.json` (e.g. `..._image-encoder.json`, `..._text-encoder.json`). No wrapper file is produced.
3. Injects the new `eval` section from `scripts/e2e_eval/testsets/models_with_acc.json` when dataset metadata exists
4. Uses canonical model list from `scripts/e2e_eval/testsets/models_57.txt` (57 `(model, task)` pairs)
5. Keeps eval config device-agnostic (does **not** write `eval.device`)
6. Skips files that already exist (including composite split files matching `<stem>_*.json`)

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

## Eval Schema

Configs use `eval` (not `eval_option`) and keep device selection in runtime CLI:

```json
{
  "eval": {
    "task": "zero-shot-image-classification",
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
    "dataset": {
      "columns_mapping": { "label_column": "pos_tags" },
      "build_script": "scripts/e2e_eval/datasets/build_indonlu_posp.py"
    }
  }
}
```

## Adding a New Model

Add a line to `scripts/e2e_eval/testsets/models_57.txt` in the format:

```text
<hf_id>|<task>
```

Then rerun generation.

## Precision Options

NPU targets are swept across the precisions below; CPU/GPU targets generate a single config
without `--precision` (using the EP's default), so this table only applies to NPU targets.

| Precision | Weight | Activation | Description |
|-----------|--------|------------|-------------|
| `w8a8` | uint8 | uint8 | Fully quantized, smallest model |
| `w8a16` | uint8 | uint16 | Mixed precision |
| `fp16` | float16 | float16 | Half precision, no quantization |
