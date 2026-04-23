# PR-03: Update Codebase with Latest Changes (#15)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `922b5d3` |
| Date | 2026-03-30 |
| Author | Zhipeng Wang (timenick) |
| PR Number | #15 |
| Files Changed | 202 |
| Insertions | +34,799 |
| Deletions | -4,248 |

## Summary
Large batch sync bringing the ModelKit repository up to date with the latest internal development. The change spans the full codebase: `analyze/` (runtime checker, information engine, output aggregator, doc checkers), `commands/` (build, eval, perf, export), `eval/` (new evaluator framework with base_evaluator, image segmentation, object detection, text/token classification, MAP and IoU metrics), `models/` (depth_pro, segformer, zoedepth, vision_encoder_decoder, winml object detection), `pattern/op_input_gen/` (resize, unary, matmul, slice, variadic generators), `session/` (QAIRT session and compile bin), and `onnx/` (shape, persistence). Also included are updated rule zip files for QNN, OpenVINO, and VitisAI EPs across multiple opsets.

## Files Changed (key areas)
- `src/winml/modelkit/eval/` — new evaluator framework (6 new files: base_evaluator, evaluate, image_segmentation_evaluator, object_detection_evaluator, text_classification_evaluator, token_classification_evaluator)
- `src/winml/modelkit/eval/metrics/` — new MAP and IoU metric files
- `src/winml/modelkit/models/hf/` — depth_pro, segformer, zoedepth, vision_encoder_decoder added
- `src/winml/modelkit/models/winml/object_detection.py` — new WinML object detection model
- `src/winml/modelkit/analyze/runtime_checker/` — case_runner.py added, check_ops.py and result_processor.py heavily updated
- `src/winml/modelkit/commands/eval.py` — new eval command (284 lines)
- `src/winml/modelkit/pattern/op_input_gen/` — resize and shape generators added; existing generators extensively updated
- Rule zip files for QNN NPU opset11–22, OpenVINO NPU, VitisAI NPU updated
