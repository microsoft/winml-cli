Yes — and the pain you described is exactly what the early stages of the pipeline are designed to prevent. You don't have to commit to a full build to find out if `nvidia/segformer-b1-finetuned-ade-512-512` will survive. There's a cheap two-step pre-flight:

## Step 1: `inspect` — does ModelKit understand this architecture at all?

This reads only the model config (no weights, no GPU). It prints an **Overall Support** verdict (`Supported` / `Default` / `Unsupported`) along with the loader, exporter, and IO the pipeline would pick.

```
uv run winml inspect -m nvidia/segformer-b1-finetuned-ade-512-512
```

Good news up front: SegFormer is in the explicitly-in-scope family for this toolkit (CNNs / vision transformers / segmentation), so you should see a `Supported` verdict. If it comes back `Unsupported`, stop — don't build. If it comes back `Default`, treat it as "might work, verify with analyze."

## Step 2: `analyze` — will every operator survive on your Intel NPU?

This is the step that would have caught your previous half-day burn. `analyze` runs a static lint of every operator in the exported ONNX against the target EP and tells you which ops are supported / partial / unsupported. That's where optimize-stage explosions show up *before* you pay the optimize cost.

The catch is `analyze` needs an ONNX file, so for a HuggingFace ID you do need to export first. But export is by far the cheapest stage that produces a buildable artifact — and once you have the ONNX, analyze runs in seconds.

For your Intel NPU, the EP is **OpenVINO** (`--ep ov`, `--device npu`):

```
uv run winml export   -m nvidia/segformer-b1-finetuned-ade-512-512 -o segformer.onnx
uv run winml analyze  -m segformer.onnx --ep ov --device npu
```

Exit codes do the work for you:
- `0` — fully supported, safe to commit to the full build
- `1` — partial support, some ops unsupported (analyze will name them, and you can often dodge the pattern with a different optimize/quantize config)
- `2` — analysis itself failed

If analyze comes back clean, then you can confidently run `winml config` followed by `winml build` for the end-to-end build. If it flags ops, you've spent maybe 5 minutes instead of 4 hours finding that out.

## Why not just run `winml build` and let it fail fast?

Because build runs export → optimize → quantize → compile in order, and the failure modes you've already seen (export passes, optimize dies) happen *after* the most expensive stage. Inspect + analyze front-load the cheap diagnostics so the expensive stages only run on models that will survive them.

One thing I'd verify before you start: confirm the OpenVINO EP is actually registered on your box with `uv run winml sys --list-ep`. If it isn't, analyze will complain regardless of how good the model is.
