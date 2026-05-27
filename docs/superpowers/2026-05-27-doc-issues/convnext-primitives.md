# Issues: docs/samples/convnext-primitives.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical

- **Compiled artifact filenames are wrong for CPU and GPU (Step 5 + Step 6).**
  The doc claims `winml compile --device cpu` writes `convnext_int8_cpu_ctx.onnx`
  and `--device gpu` writes `convnext_int8_dml_ctx.onnx`. Both claims are false.
  - `WinMLCompileConfig.for_cpu()` sets `enable_ep_context=False`
    (`compiler/configs.py` line 165). CPUExecutionProvider does not generate an
    EPContext file, so no `_cpu_ctx.onnx` is written at all.
  - `WinMLCompileConfig.for_dml()` also sets `enable_ep_context=False`
    (`compiler/configs.py` line 175). DML does not produce an EPContext either.
  - Additionally, the session filename convention uses the resolved device string,
    so if an EPContext were produced it would be `convnext_int8_gpu_ctx.onnx`
    (device="gpu"), not `convnext_int8_dml_ctx.onnx`.
  - The paragraph at the end of Step 5 restates the incorrect filenames and must
    be corrected alongside the tab blocks.

- **`winml perf --device gpu` line uses the non-existent artifact
  `convnext_int8_dml_ctx.onnx`.** Because DML compile does not produce a ctx file
  (see above), the benchmark command as written will fail with a file-not-found
  error. The entire GPU tab in Step 6 is based on a false premise.

## Important

- **`--output` flag on `winml perf` is described as writing a JSON file.**
  The doc says "Use the JSON output written by `--output`". The actual flag name
  in `perf.py` is `-o` / `--output`, output defaults to a timestamped path under
  `~/.cache/winml/perf/`. This description is essentially correct, but the page
  never shows what the flag looks like in a command, which may confuse readers.
  Minor wording issue only.

- **Step 7 `winml eval` uses `--dataset imagenet-1k`.** HuggingFace's canonical
  dataset ID for ImageNet-1k gated access is `imagenet-1k`, which matches. This
  cannot be independently verified without HF credentials, but the ID is standard
  and consistent with other pages.

- **Note claims `--device auto` is not valid on `winml eval`.**
  `eval.py` line 69: `type=click.Choice(["auto", "cpu", "gpu", "npu"])` — `auto`
  IS listed as a valid choice. The doc's note "Note that `--device` accepts only
  `cpu`, `gpu`, or `npu` — it does not accept `auto`" is incorrect.

## Minor

- **Cross-link to `../getting-started/end-to-end.md` in the admonition.**
  Not verifiable without checking that file, but the link pattern is consistent
  with other pages.

- **Step 2: `winml config -m ... -o convnext_config.json`** — the `-o` flag is
  correct for `config.py` (`cli_utils.output_option`). Verified correct.

- **Step 3 export output text shows `Starting HTP export...` and
  `Success! Model exported to: convnext.onnx`** — matches actual console output
  strings in `export.py` lines 388 and 417. Verified correct.

- **`--method entropy` mentioned in Step 4 note.** `quantize.py` line 65:
  `type=click.Choice(["minmax", "entropy", "percentile"])`. `entropy` is valid.

## Verified correct

- `winml inspect -m facebook/convnext-tiny-224` — `-m` flag exists, model ID is
  a real HF repo.
- `winml config -m facebook/convnext-tiny-224 -o convnext_config.json` — flags
  all exist in `config.py`.
- `winml export -m facebook/convnext-tiny-224 -o convnext.onnx` — `-m` and `-o`
  exist in `export.py`, `-o` is required for export.
- `winml quantize -m convnext.onnx -o convnext_int8.onnx --precision int8 --samples 32`
  — all flags verified in `quantize.py`.
- `winml compile -m convnext_int8.onnx --output-dir . --device npu --qnn-sdk-root`
  — `--output-dir`, `--device`, `--qnn-sdk-root` all exist in `compile.py`.
- `winml compile --device npu` requiring `--qnn-sdk-root` or `QNN_SDK_ROOT` —
  consistent with `compile.py` and source notes.
- `winml perf` flags `--device`, `--iterations` — verified in `perf.py`.
- `winml eval` flags `-m`, `--model-id`, `--dataset`, `--split`, `--samples`,
  `--device` — verified in `eval.py`.
- NPU artifact `convnext_int8_qnn_ctx.onnx` — consistent with session.py naming
  (`{stem}_{device}_ctx.onnx` with device="npu"). Verified plausible.
- "Pick the right ConvNeXt page" admonition links to `../tutorials/npu-convnext.md`
  — resolves correctly; counterpart admonition in npu-convnext.md links back here.
- No `wmk` or `ModelKit` strings found in user-facing prose.
