What changed:
- Added QNN recipe JSONs for facebook/dinov2-small and facebook/dinov2-base under their respective qnn/ folders.
- Each recipe is identical to the existing w8a16 recipe but with export.opset_version set to 21 and optim set to {"bias_softmax_fusion": true}.

Validation:
- Created temporary pytest at temp/test_dinov2_qnn_recipes.py which loads each JSON via WinMLBuildConfig.from_dict and asserts:
  - quant.model_id matches expected
  - export.opset_version == 21
  - quant.mode maps to "static" (qdq -> static)
  - quant.weight_type == "uint8" and activation_type == "uint16"
  - optim contains bias_softmax_fusion == True
- Ran tests with: PYTHONPATH=src python -m pytest -q temp\test_dinov2_qnn_recipes.py
- Result: 2 passed in 1.51s

Files changed:
- examples/recipes/facebook_dinov2-small/qnn/image-feature-extraction_w8a16_opset21_bias-softmax_config.json (new)
- examples/recipes/facebook_dinov2-base/qnn/image-feature-extraction_w8a16_opset21_bias-softmax_config.json (new)

Commit:
- 7fe980ed Add DINOv2 QNN recipes: opset21 + bias_softmax_fusion

Self-review findings:
- Recipes preserve loader/quant/eval fields from existing w8a16 configs.
- Only export.opset_version and optim were changed as required.
- WinMLBuildConfig.from_dict accepted both files and all assertions passed.

Concerns:
- The brief suggested using `uv run --no-project ...` but uv failed due to missing CLI plugins; I ran pytest directly with PYTHONPATH=src which validates the config parsing as required.
- No runtime quantization/export performed; this task only adds recipe metadata per instructions.

Controller follow-up validation:
- Ran `PYTHONPATH=src uv run --no-project --with pytest --with pytest-cov --with pytest-timeout --with onnxruntime --with onnx --with pydantic --with rich pytest temp\test_task1_qnn_recipe_validation.py -q -o filterwarnings=`.
- Result: 2 passed in 2.60s.

Report generated at: .superpowers/sdd/task-1-report.md
