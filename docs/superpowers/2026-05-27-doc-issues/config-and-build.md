# Issues: docs/concepts/config-and-build.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical (factually wrong; user would hit error)

- **JSON example `compile` section uses wrong field names** (lines 85-90): The doc shows:
  ```json
  "compile": {
    "ep_config": {
      "provider": "qnn",
      "enable_ep_context": true
    }
  }
  ```
  However, `WinMLCompileConfig.to_dict()` does NOT nest under `ep_config`; it serializes flat with keys `execution_provider`, `provider_options`, `enable_ep_context`, `embed_context`, `compiler`, `qnn_sdk_root`, `device` (source: `src/winml/modelkit/compiler/configs.py` lines 230-245). `WinMLCompileConfig.from_dict()` reads `data.get("execution_provider", "qnn")` (line 253), not `ep_config.provider`. A user who copy-pastes this JSON and passes it to `winml build` will get a config with `provider="qnn"` default (silently ignored nested key), making compilation silent failure or wrong EP.

- **JSON example `optim` section uses non-canonical field names** (lines 75-80): The doc shows:
  ```json
  "optim": {
    "gelu_fusion": false,
    "layer_norm_fusion": false,
    "matmul_add_fusion": false
  }
  ```
  `WinMLOptimizationConfig` is a `dict` subclass that accepts arbitrary kwargs (source: `src/winml/modelkit/optim/config.py` lines 13-31). The field names `gelu_fusion`, `layer_norm_fusion`, `matmul_add_fusion` correspond to capability python_names, which exist in the optimizer (source: `src/winml/modelkit/optim/pipes/graph.py` lines 242-243). These are valid keys but there are no hard-coded defaults for them — the generated JSON would only include keys that were explicitly set. A freshly generated config from `winml config` would likely have `{}` for `optim` unless capabilities are explicitly configured. The presence of all-`false` values is misleading; a real generated config would omit them.

## Important (misleading or stale claim)

- **`WinMLBuildConfig` described as having five nested sub-configs** (lines 48-56, table): The doc lists `loader`, `export`, `optim`, `quant`, `compile`. The actual dataclass also has `eval: WinMLEvaluationConfig | None` and `auto: bool` (source: `src/winml/modelkit/config/build.py` lines 132-138). The table is incomplete; `eval` section is a valid config key that affects `winml eval` behavior when running from a build config.

- **`winml config` `--no-compile` default behavior** (line 33): Doc says "sets the `compile` section to `null`". In the CLI, `--no-compile` is the default (`default=True` for `no_compile`, source: `commands/config.py` lines 162-165), meaning compilation is always excluded unless `--compile` is passed. The doc does not mention that compile is off by default from `winml config`.

- **`WinMLBuildConfig` defined in `src/winml/modelkit/config/build.py`** (line 47): Correct file path. However the description says "one per pipeline stage" — there are actually 6 stages with the `eval` field, not 5 as stated.

## Minor (style, polish, low-impact)

- **`--output-dir` and `--use-cache` enforcement** (line 111): Doc says "enforced at runtime, not parse time". This is accurate — source `commands/build.py` line 377 shows a `click.UsageError` raised in the command body.

- **Cross-links** `[../commands/config.md]` and `[../commands/build.md]` (lines 161-162): Both files exist in `docs/commands/`.

- **Cross-link** `[primitives-and-pipeline.md]` (line 158): File exists in `docs/concepts/`.

## Verified correct (anchored claims you checked)

- `winml config -m microsoft/resnet-50 -o resnet50.json` syntax is valid → `commands/config.py` lines 66-73 (`-m`/`--model`, `-o`/`--output`)
- `--task` flag exists on `winml config` → `commands/config.py` lines 77-80
- `--no-quant` flag exists on `winml config` → `commands/config.py` lines 155-159
- `--trust-remote-code` flag exists on `winml config` → `commands/config.py` line 166
- `-o` omission prints to stdout → `commands/config.py` lines 487-490
- `winml build -c resnet50.json -m microsoft/resnet-50 --output-dir output/` valid → `commands/build.py` lines 233-256
- `--use-cache` writes to `~/.cache/winml/` → `commands/build.py` lines 258-262
- `--no-quant`, `--no-compile`, `--no-optimize` CLI overrides exist on `winml build` → `commands/build.py` lines 273, 275-282, 300-304
- `WinMLBuildConfig.from_dict()` reads `loader`, `export`, `optim`, `quant`, `compile`, `eval` sections → `config/build.py` lines 152-172
- `WinMLLoaderConfig`, `WinMLExportConfig`, `WinMLOptimizationConfig`, `WinMLQuantizationConfig`, `WinMLCompileConfig` all exist → `config/build.py` lines 54-64
- JSON `quant` section fields `weight_type`, `activation_type`, `samples` exist → `quant/config.py` lines 55, 65-66
- No `wmk` or `ModelKit` strings in prose → verified by grep
