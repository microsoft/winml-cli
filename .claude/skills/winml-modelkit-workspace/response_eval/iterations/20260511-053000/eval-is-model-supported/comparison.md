# eval-is-model-supported

## Prompt

> Before I commit to building `nvidia/segformer-b1-finetuned-ade-512-512` end-to-end on my Intel NPU, is there a way to know if it's going to make it through the full pipeline? Burned half a day last time on a model that exported fine but then died at the optimize stage.

## with_skill — 6/6

### Response

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

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✓ | Recommends `winml inspect` as the first, cheapest check (config-only). | PASS | Step 1 winml inspect; 'reads only the model config (no weights, no GPU).' |
| ✓ | Recommends `winml analyze` after inspect. | PASS | Step 2 winml analyze --ep ov --device npu. |
| ✓ | Identifies OpenVINO as the EP for Intel NPU. | PASS | Quote: 'For your Intel NPU, the EP is OpenVINO (--ep ov, --device npu).' |
| ✓ | Explains analyze's role: linter that classifies operators as supported/partial/unsupported per EP. | PASS | Quote: 'analyze runs a static lint of every operator in the exported ONNX against the target EP and tells you which ops are supported / partial / unsupported.' |
| ✓ | Does NOT just say 'build the full pipeline and see what happens'. | PASS | Explicit section 'Why not just run winml build and let it fail fast?' justifying the pre-flight. |
| ✓ | All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | All ? winml commands verified runnable against --help. |

---

## without_skill (baseline) — 3/6 ⚠

### Response

Yeah, totally fair concern — segformer is one of those models where export usually works but the optimize/quantize stage is where things tend to fall apart (dynamic shapes, unsupported ops for the EP, layout issues, etc.). Burning half a day to find out is rough.

There isn't a single "will this work?" oracle, but here's a fast pre-flight you can run in maybe 15-30 minutes before committing to the full pipeline:

**1. Check the model card / config first (2 min)**
- Look at `config.json` on the HF page. Segformer is a semantic segmentation model with a hierarchical transformer encoder. The ops to watch for: `LayerNorm`, `GELU`, `Gather`, `Reshape` with dynamic dims, and the bilinear upsample at the head.
- ADE20K version has 150 output classes — confirm the output shape matches what you expect.

**2. Do a dry-run export only (5 min)**
Just run the export step and stop. Inspect the resulting ONNX:
- Open it in Netron, or use `onnx.checker.check_model()` + `onnx.shape_inference.infer_shapes()`.
- Look for unresolved dynamic dimensions on intermediate tensors. Segformer often has symbolic shapes from the `Resize`/`Interpolate` at the decoder head — those are the usual optimize-stage killers.
- Run with a fixed input shape (e.g. `1x3x512x512`) during export to eliminate as many dynamic axes as possible up front.

**3. Probe the optimize stage cheaply (5-10 min)**
- Run ONNX Runtime's graph optimizer standalone (`SessionOptions.graph_optimization_level = ORT_ENABLE_ALL`) and save the optimized model. If this fails on CPU EP, it'll definitely fail on the Intel NPU EP.
- Then try loading with the Intel NPU EP (OpenVINO EP or the DirectML/Intel NPU provider, depending on your stack) and check the partition report — most NPU EPs will tell you which nodes fall back to CPU. If a huge chunk falls back, that's your signal to stop.

**4. Check against a known-good baseline**
Has anyone published segformer working on Intel NPU? Quick search of:
- Intel's OpenVINO model zoo
- HF "optimum-intel" examples
- Issues on the OV/NPU EP repos mentioning segformer

If someone's already done it, you'll find quantization configs or notes about which opset / which export flags work. If nobody has, that's also a signal.

**5. The cheapest cheap test**
Export at fp32, skip quantize, and just try to *load* the model with the NPU EP and run one inference. If that works, optimize/quantize is usually tractable. If the EP can't even partition the fp32 graph, no amount of optimization will save you.

What pipeline/tooling are you using — straight ORT, OpenVINO, or something like winml-modelkit? The exact pre-flight commands depend on that, and I can give you something more concrete.

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✗ | Recommends `winml inspect` as the cheapest first check (config-only). | **FAIL** | No winml inspect. Suggests manually reading `config.json` on the HF page. |
| ✗ | Recommends `winml analyze` after inspect. | **FAIL** | No winml analyze. Suggests running ORT's graph optimizer standalone + manually checking partition reports. |
| ✓ | Identifies OpenVINO as EP for Intel NPU. | PASS | 'OpenVINO EP or the DirectML/Intel NPU provider, depending on your stack' â€” names OpenVINO as primary option. |
| ✗ | Explains analyze's role: linter classifies operators per EP. | **FAIL** | No mention of an analyze/linter tool. The pre-flight is much more DIY. |
| ✓ | Does NOT just say 'build and see'. | PASS | Five-step structured pre-flight, not a 'try it' answer. |
| ✓ | All `winml` commands quoted in the response are runnable — every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | Response uses no winml commands (trivially passes). |

---
