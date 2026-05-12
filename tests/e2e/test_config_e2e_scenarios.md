# Test Scenarios — `tests/e2e/test_config_e2e.py`

End-to-end tests for the `config` CLI command. All tests are marked `e2e` and `network` and run with a mocked `resolve_device` fixture.

## `TestConfigBert` — `bert-base-uncased`

| Test | Scenario |
|---|---|
| `test_with_explicit_task[fill-mask]` | Generate config with explicit `--task fill-mask`. |
| `test_with_explicit_task[text-cls]` | Generate config with explicit `--task text-classification`. |
| `test_with_explicit_task[token-cls]` | Generate config with explicit `--task token-classification`. |
| `test_auto_detect` | Without `--task`, pipeline auto-detects a task. |
| `test_device_cpu_precision_fp32` | `--device cpu --precision fp32` produces a config with no `quant` section. |
| `test_output_to_file` | `-o <file>` writes valid JSON to disk. |
| `test_scenario_c_model_type_only` | `--model-type bert` (no `-m`) uses default HF config. |

## `TestConfigVision` — vision models

| Test | Scenario |
|---|---|
| `test_auto_detect[resnet]` | `microsoft/resnet-50` auto-detects `image-classification`. |
| `test_auto_detect[convnext]` | `facebook/convnext-tiny-224` auto-detects `image-classification`. |
| `test_auto_detect[vit]` | `google/vit-base-patch16-224` auto-detects `image-classification`. |

## `TestConfigCLIP` — `openai/clip-vit-base-patch32`

| Test | Scenario |
|---|---|
| `test_feature_extraction` | `--task feature-extraction` produces a valid config. |
| `test_zero_shot_image_classification` | Composite task splits output into per-component `config_*.json` files (image-encoder, text-encoder), each well-formed. |

## `TestConfigDETR` — `facebook/detr-resnet-50`

| Test | Scenario |
|---|---|
| `test_auto_detect` | Auto-detects `object-detection`. |

## `TestConfigONNX` — pre-exported ONNX files

| Test | Scenario |
|---|---|
| `test_onnx_model_path` | `.onnx` file input yields config with `export=None`. |
| `test_onnx_with_no_compile` | `--no-compile` on the ONNX path yields `compile=None`. |
| `test_onnx_with_no_quant` | `--no-quant` on the ONNX path yields `quant=None`. |
| `test_onnx_output_to_file` | `-o <file>` serializes ONNX-path config to disk. |

## `TestConfigBadPath` — argument validation & CLI error handling

| Test | Scenario |
|---|---|
| `test_no_args_is_error` | No args → non-zero exit with usage error (no traceback). |
| `test_missing_entry_point_message` | Error message references `--model`, `--model-type`, or `--model-class`. |
| `test_invalid_device_rejected[tpu/fpga/xpu/DSP]` | Unknown `--device` values rejected by Click `Choice`. |
| `test_invalid_precision_rejected[bf16/fp64/int4/w3a5]` | Unknown `--precision` strings produce `UsageError`, not traceback. |
| `test_invalid_ep_rejected[tflite/coreml/not-a-real-ep]` | Unknown `--ep` values produce `UsageError`. |
| `test_nonexistent_config_file_rejected` | `-c` with missing file rejected by Click. |
| `test_empty_config_file_rejected` | Empty `-c` file produces `UsageError`. |
| `test_invalid_json_config_file_rejected` | Malformed JSON in `-c` produces `UsageError`. |
| `test_non_object_json_config_file_rejected` | JSON array in `-c` rejected (must be object). |
| `test_empty_shape_config_rejected` | Empty `--shape-config` file produces `UsageError`. |
| `test_invalid_json_shape_config_rejected` | Malformed `--shape-config` JSON produces `UsageError`. |
| `test_non_object_shape_config_rejected` | JSON list in `--shape-config` rejected (must be object). |
| `test_module_with_onnx_file_rejected` | `--module` mutually exclusive with `.onnx` input. |

## `TestConfigFlagVariations` — every behavior-bearing flag (`bert-base-uncased` + `fill-mask`)

| Test | Scenario |
|---|---|
| `test_every_device_choice[auto/cpu/gpu/npu]` | Each `--device` choice produces a valid config. |
| `test_every_named_precision[auto/fp32/fp16/int8/int16]` | Each named `--precision` produces a valid config (paired with compatible device). |
| `test_mixed_precision[w8a8/w8a16]` | Mixed `w{x}a{y}` precision accepted on NPU. |
| `test_every_ep_choice[qnn/dml/openvino/vitisai/migraphx/nv_tensorrt_rtx/cpu]` | Every documented `--ep` alias accepted. |
| `test_no_quant_present` | `--no-quant` zeros out the `quant` section. |
| `test_no_quant_absent` | Quantized device (NPU + int8) without `--no-quant` keeps `quant` settings. |
| `test_no_compile_default` | Default behavior excludes `compile`. |
| `test_compile_enabled` | `--compile` produces a non-null `compile` section. |
| `test_shape_config_present` | `--shape-config` file accepted and applied. |
| `test_library_default` | Default `--library transformers` works without explicit flag. |
| `test_library_explicit` | Explicit `--library transformers` accepted. |
| `test_verbose_flag` | `-v` / `--verbose` does not affect JSON output and does not crash. |
| `test_model_type_only` | `--model-type bert` alone auto-picks a supported task. |
| `test_model_type_with_task` | `--model-type bert --task fill-mask` honored. |
| `test_config_file_override` | `-c` override file loaded and merged (e.g. `opset_version: 18`). |
| `test_trust_remote_code_flag` | `--trust-remote-code` accepted on a normal HF model. |
| `test_module_flag_returns_list` | `--module ResNetConvLayer` (with `microsoft/resnet-50`) emits a JSON list of per-submodule configs. |
