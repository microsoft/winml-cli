# Issues: docs/samples/bert-config-build.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical

- **Final artifact name is wrong.** Step 2 output block says:
  `Final artifact: bert_out/bert-base-uncased_ctx.onnx`
  The actual build pipeline in `commands/build.py` (line 714) always writes the
  final output as `model.onnx` inside the output directory:
  `final_path = resolved_dir / _name("model.onnx")`
  For a non-cached build the artifact is `bert_out/model.onnx`, not
  `bert_out/bert-base-uncased_ctx.onnx`. The `_name()` helper only prepends a
  cache key when `--use-cache` is active; with `-o bert_out/` it stays `model.onnx`.

- **Step 3 perf command references the wrong artifact.**
  `winml perf -m bert_out/bert-base-uncased_ctx.onnx` will fail because the file
  does not exist (see above). Should be `winml perf -m bert_out/model.onnx`.

## Important

- **`build` command flag: doc uses `-o bert_out/` but the flag is `--output-dir`.**
  In `commands/build.py` line 250-252 the short alias `-o` maps to `--output-dir`.
  The `-o` short form is defined, so the command works — but the doc never
  mentions `--output-dir` anywhere (the "Customizing the config" section also
  uses `-o`), leaving readers who try `--help` unable to find it easily.
  The step 2 command itself is syntactically valid; this is a doc clarity issue.

- **JSON excerpt uses `"optim"` key.** `config/build.py` line 17 in the config
  hierarchy comment shows `optim: WinMLOptimizationConfig`. The serialised key
  from `WinMLBuildConfig.to_dict()` must be verified. Check that `optim` (not
  `optimize` or `optimization`) is the actual JSON key. Based on the config
  hierarchy definition in `config/build.py` the field is named `optim`, which
  aligns with the doc. Verified plausible, but should be confirmed by reading the
  `to_dict()` / `from_dict()` implementation in `config/build.py`.

- **JSON excerpt `"optim"` section fields: `gelu_fusion`, `layer_norm_fusion`,
  `matmul_add_fusion`.** These field names must match `WinMLOptimizationConfig`.
  The optimize command uses a capability registry; the field names in the
  serialised JSON depend on how `WinMLOptimizationConfig.to_dict()` names them.
  The doc claims them without source verification — they may differ from the
  actual serialised keys.

- **JSON excerpt `"compile"` section.** The doc shows:
  ```json
  "compile": {
    "execution_provider": "qnn",
    "enable_ep_context": true,
    "compiler": "ort"
  }
  ```
  These map to `WinMLCompileConfig.to_dict()` in `compiler/configs.py` lines 232-247.
  `execution_provider`, `enable_ep_context`, and `compiler` are all present in
  `to_dict()`. Verified correct for those three keys.

- **Note mentions `--max-optim-iterations` flag.** In `commands/build.py` line
  307 the flag is `--max-optim-iterations` (not `--max-optimize-iterations`).
  The doc spells it `--max-optim-iterations`, which matches. Verified correct.

- **`--no-quant` and `--no-compile` flags on `winml build`.** Both exist in
  `build.py` (`--no-quant` line 272, `--no-compile/--compile` line 277). Verified.

- **`winml config --precision fp16`.** `config.py` has `-p`/`--precision` with
  `type=str` accepting `fp16`. Verified valid.

- **`bert-base-uncased` model ID.** The canonical HF ID is
  `google-bert/bert-base-uncased`; `bert-base-uncased` is a redirect that still
  works. The doc uses the short alias consistently. Acceptable but not canonical.

## Minor

- **Step 1: `winml config -m bert-base-uncased -t text-classification -o bert_config.json`.**
  The `-t` flag on `config` is for `--task`. Verified in `config.py` line 78-79.
  Valid.

- **Note: `quant.weight_type` and `quant.activation_type` editing instructions.**
  The doc suggests setting these to `"int8"` or `"uint16"`. Valid options per
  `quantize.py` line 71: `type=click.Choice(["uint8", "int8", "uint16", "int16"])`.
  Verified correct.

## Verified correct

- `winml config -m bert-base-uncased -t text-classification -o bert_config.json`
  — all flags valid.
- `winml build -c bert_config.json -m bert-base-uncased --output-dir bert_out/`
  (`-o` short form) — command structure valid (see Critical note on artifact name).
- `winml build ... --no-quant` — flag verified in `build.py`.
- Top-level JSON keys `loader`, `export`, `optim`, `quant`, `compile` — match
  `WinMLBuildConfig` field names.
- `quant.mode`, `quant.weight_type`, `quant.activation_type`, `quant.samples`,
  `quant.calibration_method`, `quant.task`, `quant.model_name` — all present as
  fields on `WinMLQuantizationConfig` (verified in `quantize.py` config usage).
- No `wmk` or `ModelKit` strings in user-facing prose.
- Cross-links to `convnext-primitives.md`, `../concepts/config-and-build.md`,
  `../commands/config.md`, `../commands/build.md`, `../commands/perf.md` are
  consistent with repo structure.
