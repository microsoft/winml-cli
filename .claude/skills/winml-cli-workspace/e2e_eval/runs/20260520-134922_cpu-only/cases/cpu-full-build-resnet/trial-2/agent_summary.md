CPU build of `microsoft/resnet-50` completed successfully in 14.5s (export 5.4s + optimize 5.6s). Skipped quantize/compile per the `--device cpu` config (build's default `--no-compile`, and `cpu` device generates `quant=none`).

Final artifact:
- `C:\repo\WinML-ModelKit\.claude\skills\winml-modelkit-workspace\e2e_eval\scratch\cpu-full-build-resnet\trial-2\output\model.onnx` (97.5 MB)

Other artifacts in the same directory:
- `export.onnx` (97.4 MB) - raw ONNX export
- `optimized.onnx` (97.5 MB) - post-optimization (identical to model.onnx — no optim patterns matched for CPU EP)
- `winml_build_config.json` - resolved config used for the build
- `export_htp_metadata.json` - exporter metadata

Config used: `C:\repo\WinML-ModelKit\.claude\skills\winml-modelkit-workspace\e2e_eval\scratch\cpu-full-build-resnet\trial-2\config.json` (generated via `winml config -m microsoft/resnet-50 --device cpu`).

Notes:
- `winml inspect` printed the model info correctly but crashed at the end on a Windows cp1252 encoding bug rendering a Unicode arrow in the IO Configuration panel — this is a display-only issue and didn't affect support detection (all stages reported "Default" status). ResNet-50 is a CNN, fully in-scope per the skill.
- No analyzer op data exists for CPUExecutionProvider, so analyzer logged "skipping op analysis" warnings — expected and harmless on CPU.
