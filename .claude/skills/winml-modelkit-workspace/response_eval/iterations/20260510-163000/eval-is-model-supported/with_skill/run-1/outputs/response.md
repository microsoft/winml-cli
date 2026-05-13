Yes — and this is exactly the right instinct. ModelKit has two cheap pre-flight checks that catch the two different failure modes you're describing, and you should run both before kicking off a full build.

**Step 1: `winml inspect nvidia/segformer-b1-finetuned-ade-512-512`**

This reads only the model config (no weights, no export, no hardware spin-up) and gives a supported / unsupported verdict on the architecture itself. SegFormer is in the in-scope family for ModelKit, so I expect this to come back clean — but run it anyway, because if it says `Unsupported`, nothing else matters and you've saved yourself the half-day. This is the non-negotiable gate.

**Step 2: `winml analyze` against the exported ONNX, targeting the OpenVINO EP (Intel NPU).**

This is the one that would have caught your previous burn. The analyzer runs an operator-level linter and tags every op as supported / partial / unsupported for the EP you're targeting. That's the cheapest way to find out whether optimize/quantize/compile will choke on a specific op pattern — without paying for the full pipeline. Most of the "exported fine, died at optimize" failures show up here as a partial- or unsupported-op warning.

The catch: analyze needs an ONNX file, so you do have to pay the export cost once. Two ways to handle it:

- If you want the absolute cheapest signal first, just run inspect and stop. If it's green, you're past the architecture-level gate.
- If you want full confidence before committing to the build, do inspect → export → analyze (with `--ep` set to whatever the OpenVINO flag is on your build — check `winml analyze --help`). Export is the expensive part but it's still way cheaper than discovering an op problem at compile time.

A couple of practical notes for your run:

- Confirm the exact flag spelling with `winml inspect --help` and `winml analyze --help` before running — I don't want to invent flags here.
- `winml sys --list-ep` will confirm OpenVINO is actually registered on your machine. Worth a sanity check.
- If analyze flags an unsupported op pattern, the fix is usually a different optimize/quantize config rather than a graph edit — but at that point we'd reassess whether SegFormer-b1 is a good fit for your NPU or whether you should fall back to the optimized (pre-compile) artifact on CPU/GPU.

Run inspect, paste me the output, and we'll decide whether to go straight to config+build or do the analyze pass first.
