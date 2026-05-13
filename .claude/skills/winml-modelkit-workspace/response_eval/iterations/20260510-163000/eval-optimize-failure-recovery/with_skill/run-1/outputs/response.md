Swin is in scope for the pipeline, so this is almost certainly a specific op pattern in the optimized graph that QNN doesn't accept — not a "this model can't be built" situation. The right next move is to find out which pattern, then route around it.

Here's what I'd do:

1. **Run `winml analyze` against the exported ONNX, targeting QNN.** That's the linter step — it classifies every op as supported / partial / unsupported for the QNN EP and will usually name the offending pattern (window-shift, roll, layernorm variants, and certain gather/reshape combos are the usual suspects in Swin). Check `winml analyze --help` for the exact flag to scope it to QNN and to point it at your exported model.

2. **Read the optimize error carefully.** ModelKit's errors are normally specific (op name + stage). If it names an op or pattern, cross-reference it with the analyze output — that confirms whether optimize is producing a pattern that didn't exist in the raw export, or just surfacing one that was always there.

3. **Try a different optimize configuration to dodge the pattern.** Run `winml optimize --help` and look at the available knobs (graph-level toggles, fusion options, etc.). The fix is almost always picking a configuration that doesn't emit the problematic fused pattern — don't hand-edit the ONNX. If you're on the config+build path, edit the JSON config rather than stacking flags at build time.

4. **If nothing in optimize sidesteps it,** that's the signal Swin-large at this resolution may not be a clean fit for QNN as-is. At that point the honest answer is to either fall back to a smaller Swin variant, try a different EP (DirectML on the integrated GPU as a sanity check), or accept partial offload.

A couple of things to avoid: don't invent flags like `--preset` or `--mode=fast` — they don't exist and just waste a round-trip. And don't try to run a CPU-EP compiled artifact to "skip" optimize; QNN-bound artifacts have to go through this stage.

If you paste the actual error text and the analyze output, I can point at the specific pattern and the specific knob to flip.
