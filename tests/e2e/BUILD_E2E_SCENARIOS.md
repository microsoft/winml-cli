# `winml build` E2E Test Scenarios

Companion to `tests/e2e/test_build_e2e.py`. Lists every functional
scenario the suite exercises, grouped by required test category
(happy path / bad path / flag variations).

All tests in `test_build_e2e.py` carry the `e2e` marker and are
auto-skipped unless pytest is invoked with `-m e2e` (see
`tests/e2e/conftest.py`). Heavy HF tests additionally carry
`slow` and `network`.

Run with:

```bash
uv run pytest tests/e2e/ -k build -m e2e
```

## CLI surface inventory

The `winml build` command has the following options (see
`src/winml/modelkit/commands/build.py`):

| Flag                       | Required | Type       | Notes                                                  |
|----------------------------|----------|------------|--------------------------------------------------------|
| `-c` / `--config`          | yes      | path       | `WinMLBuildConfig` JSON (object or module-mode array)  |
| `-m` / `--model`           | no       | str / path | HF model ID, `.onnx` path, or omitted (random-weight)  |
| `-o` / `--output-dir`      | one-of   | path       | Mutually exclusive with `--use-cache`                  |
| `--use-cache`              | one-of   | flag       | Use `~/.cache/winml/`; requires `loader.task`          |
| `--rebuild`                | no       | flag       | Overwrite existing artifacts                           |
| `--no-quant`               | no       | flag       | Sets `config.quant = None`                             |
| `--no-compile` / `--compile` | no     | tri-state  | Override compile section; `None` = inherit             |
| `--ep`                     | no       | str        | EP hint for analyzer (`qnn`, `openvino`, …)            |
| `--device`                 | no       | str        | Device hint for analyzer (`NPU`, `GPU`, …)             |
| `--no-analyze`             | no       | flag       | Forces `hack_max_optim_iterations=0`                   |
| `--no-optimize`            | no       | flag       | Sets `extra_kwargs['skip_optimize']=True`              |
| `--max-optim-iterations N` | no       | int        | Forwarded as `hack_max_optim_iterations`               |
| `--trust-remote-code`      | no       | flag       | Forwarded via `extra_kwargs`                           |
| `-v` / `--verbose`         | no       | flag       | Enable DEBUG logging                                   |

## Bad path (`TestBuildArgValidation`, `TestBuildErrorHandling`)

| Scenario                                              | Test                                                |
|-------------------------------------------------------|-----------------------------------------------------|
| Missing `-c/--config`                                 | `test_missing_config_required`                      |
| Config path does not exist                            | `test_config_file_does_not_exist`                   |
| Neither `--output-dir` nor `--use-cache` provided     | `test_missing_output_and_cache`                     |
| `--output-dir` AND `--use-cache` (mutually exclusive) | `test_output_dir_and_use_cache_mutually_exclusive`  |
| Invalid JSON in config                                | `test_invalid_json_config`                          |
| Empty config file                                     | `test_empty_config_file`                            |
| Config is a JSON scalar                               | `test_config_must_be_object_or_array`               |
| `--compile` on config without a compile section       | `test_compile_flag_without_compile_section`         |
| `--use-cache` without `loader.task`                   | `test_use_cache_requires_loader_task`               |
| Module-mode (array) config with `--use-cache`         | `test_module_mode_requires_output_dir`              |
| Module array entry is not an object                   | `test_module_array_non_object_entry`                |
| `ValueError` from pipeline → UsageError               | `test_value_error_becomes_usage_error`              |
| Generic exception → `Build failed:`                   | `test_generic_failure_is_reported`                  |
| `Quantization failed` hint = `--no-quant`             | `test_quant_failure_hint`                           |
| `Compilation failed` hint = `--no-compile`            | `test_compile_failure_hint`                         |
| `--help` lists every behavior-bearing option         | `test_help_lists_all_options`                       |

## Flag variations (`TestBuildFlagPassthrough`)

Each behavior-bearing flag is exercised both **present** and
**absent**. `--no-compile/--compile` is a tri-state — all three
states are covered.

| Flag                       | Present                                     | Absent / default                          |
|----------------------------|---------------------------------------------|-------------------------------------------|
| `--rebuild`                | `test_rebuild_flag`                         | `test_defaults_no_flags`                  |
| `--no-quant`               | `test_no_quant_clears_quant`                | `test_defaults_no_flags`                  |
| `--no-compile`             | `test_no_compile_clears_compile`            | `test_compile_absent_inherits_from_config`|
| `--compile`                | `test_compile_preserves_compile`            | `test_compile_absent_inherits_from_config`|
| `--no-optimize`            | `test_no_optimize_sets_extra_kwarg`         | `test_defaults_no_flags`                  |
| `--no-analyze`             | `test_no_analyze_zeros_max_iterations`      | `test_max_optim_iterations_explicit`      |
| `--max-optim-iterations`   | `test_max_optim_iterations_explicit`        | `test_defaults_no_flags`                  |
| (precedence)               | `test_no_analyze_wins_over_max_iterations`  | —                                         |
| `--ep`                     | `test_ep_flag_forwarded`                    | `test_defaults_no_flags`                  |
| `--device`                 | `test_device_flag_forwarded`                | `test_defaults_no_flags`                  |
| `--trust-remote-code`      | `test_trust_remote_code_forwarded`          | `test_defaults_no_flags`                  |
| `-v` / `--verbose`         | `test_verbose_flag_accepted`                | `test_defaults_no_flags`                  |
| `-m/--model` omitted       | `test_model_omitted_means_random_weights`   | `test_defaults_no_flags`                  |
| Parent ctx `debug=True`    | `test_debug_inherited_from_parent_ctx`      | `test_defaults_no_flags`                  |

## Happy path

### From HuggingFace (`TestBuildHFHappyPath`, `slow`, `network`)

| Scenario                                                       | Test                                |
|----------------------------------------------------------------|-------------------------------------|
| BERT text-classification, default analyzer                     | `test_bert_text_classification`     |
| ResNet image-classification with `--ep qnn --device NPU`       | `test_resnet_image_classification`  |
| `--rebuild` on a pre-existing output directory                 | `test_rebuild_overwrites`           |

### From ONNX passthrough (`TestBuildONNXHappyPath`)

No HF download — the model is a tiny ONNX file generated in
`tests/e2e/conftest.py::onnx_model_path`.

| Scenario                                          | Test                              |
|---------------------------------------------------|-----------------------------------|
| ONNX passthrough (optimize only)                  | `test_onnx_passthrough`           |
| ONNX passthrough with `--no-optimize` (raw copy)  | `test_onnx_passthrough_no_optimize` |
