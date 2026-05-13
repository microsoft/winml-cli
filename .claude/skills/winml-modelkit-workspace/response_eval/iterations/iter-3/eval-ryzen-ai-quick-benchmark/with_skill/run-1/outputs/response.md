# Fastest path to an NPU latency number for `facebook/convnext-tiny-224`

ConvNeXT is a CNN — in scope for ModelKit — and you don't care about the artifact, so skip the build pipeline entirely and let `winml perf` do download + export + optimize + quantize + compile in one shot, then time it.

## Two commands

```powershell
# Sanity check: confirm the Ryzen AI NPU is visible and VitisAI is registered
uv run winml sys --list-device --list-ep

# Benchmark on the NPU
uv run winml perf -m facebook/convnext-tiny-224 --device npu
```

That's it. `--device npu` on a Ryzen AI box routes to the VitisAI EP. `perf` will print mean/median/p95 latency and throughput to the console and drop a `convnext-tiny-224_perf.json` next to where you ran it (you can ignore the file — the number is in the table).

## A few things worth knowing

- **Defaults**: 10 warmup iterations, 100 timed iterations, batch size 1. That's enough for a stable mean on a small CNN. Bump with `--iterations 500` if you want tighter p95s.
- **First run is slow**: the first invocation downloads the HF model, exports to ONNX, optimizes, quantizes, and compiles for VitisAI. The *reported latency* excludes all that — it only times inference iterations after warmup — but wall-clock time to first number can be a few minutes. Subsequent runs hit the cache.
- **If you want to discard the cached artifact** after you've got the number: add `--ignore-cache` and `perf` builds in a temp folder and throws it away.
- **Live NPU utilization** while it runs: add `--monitor`. Useful for confirming the NPU is actually being hit and not silently falling back to CPU.
- **Sanity-check the EP**: in the `perf` output, verify the execution provider line says `VitisAIExecutionProvider`. If it says `CPUExecutionProvider`, the VitisAI EP didn't load on your machine and the number is meaningless — `winml sys --list-ep` will tell you why.

If `--device npu` ever fails to resolve, you can pin it explicitly with `--ep vitisai`.

