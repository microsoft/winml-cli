Good news: ViT is squarely in scope, and "NPU vs CPU on the same machine" is a well-trodden path. The trick to a fair comparison is making sure both runs measure the same model graph, with the only difference being the execution provider. Here's the cleanest way to do it.

## The shape of the workflow

1. **`winml inspect google/vit-base-patch16-224`** — non-negotiable first step. Confirms ViT-base is supported before you spend time on anything else.
2. **`winml sys --list-device --list-ep`** — sanity-check that QNN shows up on your Snapdragon X Elite. If it doesn't register, no amount of benchmarking will help.
3. **Build once, perf twice.** Generate a config targeting QNN and run `winml build`. That produces two artifacts you care about:
   - the **optimized** ONNX (pre-compile, EP-agnostic) — use this for the CPU run
   - the **QNN-compiled** artifact — use this for the NPU run
4. **Run `winml perf` against each artifact**, pointing it at the matching EP (CPU EP for the optimized model, QNN for the compiled one).

The reason for build-once-then-perf-twice rather than two independent pipelines is that you want both runs to share the same export, the same optimization passes, and the same quantization settings up to the point where the NPU path diverges. Otherwise you're measuring pipeline differences, not hardware differences.

## Two things that catch people out here

- **Don't run the QNN-compiled model on CPU EP** (or vice versa). Compiled artifacts are EP-locked — you'll get nonsense numbers or a load failure. Always pair compiled→QNN, optimized→CPU.
- **Quantization matters for apples-to-apples.** The QNN path will quantize (typically to int8/int16); the optimized intermediate is usually fp32 or fp16. If you want a strict latency comparison at the same precision, run perf on the *quantized but not compiled* artifact on CPU as well — that isolates "NPU acceleration" from "quantization speedup." If you want the practical "what would I actually ship" comparison, the optimized fp model on CPU vs the QNN-compiled model on NPU is the more honest answer, because that's what each path actually deploys.

Tell me which comparison you want — strict (same precision, isolate hardware) or practical (each path at its best) — and I can sketch the exact commands.

## Before you run anything

Run `winml --help`, `winml perf --help`, and `winml build --help` to confirm the current flag names (output paths, EP selector, iteration count, warmup, hardware-utilization toggle). The CLI is the source of truth — I'd rather you read the live flags than have me guess names that may have drifted. In particular, look for a hardware-utilization flag on `perf` if you want NPU/CPU/RAM telemetry alongside the latency number; that's worth grabbing for the writeup.

One last thing: run each `perf` invocation a couple of times. First-run latency on QNN includes context-binary load and warmup that you don't want polluting the steady-state number.
