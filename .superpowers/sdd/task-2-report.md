Task 2: Add SwinV2-tiny QNN recipe

What changed
- Added QNN recipe for microsoft/swinv2-tiny-patch4-window16-256 at:
  examples/recipes/microsoft_swinv2-tiny-patch4-window16-256/qnn/image-classification_w8a16_opset21_config.json

Validation
- Created temporary pytest test: temp/test_swinv2_qnn.py (removed before commit)
- Ran focused pytest with PYTHONPATH=src using uv run (plugins: pytest-cov, pytest-timeout)
- Test executed: WinMLBuildConfig.from_dict parsed the recipe without error

Pytest output summary
- Collected 1 item
- 1 passed in 0.63s

Files changed (committed)
- examples/recipes/microsoft_swinv2-tiny-patch4-window16-256/qnn/image-classification_w8a16_opset21_config.json
- .superpowers/sdd/task-2-report.md

Self-review findings
- Recipe mirrors existing QNN recipes (dinov2) and uses mode "qdq" which maps to static calibration.
- export.opset_version set to 21 for qnn compatibility; input shape matches existing SwinV2 fp16 recipe (256x256).
- quant uses weight_type uint8 and activation_type uint16 (w8a16) to prioritize accuracy on NPUs.

Concerns
- Accuracy note in brief (+12%) not encoded in recipe schema; accuracy impact should be observed during eval.
- Full e2e test suite (uv run pytest tests/) is blocked on this machine due to missing torch wheel; only focused parsing test was run.

Report file path
C:\Users\qiowu\source\repos\copilot-worktrees\winml-cli\dingmaomaobjtu-cuddly-garbanzo\.superpowers\sdd\task-2-report.md
