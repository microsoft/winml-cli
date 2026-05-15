All artifacts produced. Done.

Built a CPU-optimized ResNet-50 via the winml pipeline using the config + build path.

Steps run:
1. Inspected microsoft/resnet-50 — recognized as WinMLModelForImageClassification (classic CNN, in scope).
2. Generated config: uv run winml config -m microsoft/resnet-50 --device cpu -o .../resnet50_cpu_config.json — auto-detected image-classification, input [1,3,224,224] float32, quant=none for CPU.
3. Ran build: uv run winml build -c .../resnet50_cpu_config.json -m microsoft/resnet-50 -o .../build --device CPU — completed in 23.5s (Export 9.3s, Optimize 8.4s). No quantization or compilation since this is CPU EP.

Output artifacts:
- resnet50_cpu_config.json — generated build config
- build/model.onnx (97.5 MB) — final CPU-ready artifact
- build/optimized.onnx (97.5 MB)
- build/export.onnx (97.4 MB) — pre-optimization export
- build/export_htp_metadata.json, build/winml_build_config.json — build metadata

Final .onnx confirmed present at: build/model.onnx — build completed successfully.
