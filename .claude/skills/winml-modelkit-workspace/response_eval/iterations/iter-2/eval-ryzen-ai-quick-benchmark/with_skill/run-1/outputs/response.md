Short answer: if all you want is the number, use `winml perf` directly on the Hugging Face ID. It will download, export, and optimize as needed before timing, and on a Ryzen AI laptop the AMD NPU is targeted via the **VitisAI** execution provider.

Before you do anything, two sanity checks:

```
winml sys --list-device --list-ep
winml inspect facebook/convnext-tiny-224
```

`sys` confirms VitisAI is actually registered on your box. `inspect` is non-negotiable — it reads only the config and tells you the model is supported. ConvNeXT is in scope, so this should pass cleanly. If it doesn't, stop there.

Then run perf against VitisAI. I don't want to guess flag spellings, so check `winml perf --help` for the exact name of the EP flag and any auto-device option, but the shape is:

```
winml perf facebook/convnext-tiny-224 --ep VitisAI
```

That single invocation is the fastest path to a latency number — no separate export/optimize/compile dance required. If you want live NPU/CPU/RAM utilization alongside the latency, look for a hardware-monitoring flag in `winml perf --help` (there is one) and add it.

A few things worth knowing:

- The compile stage needs the NPU on the local machine — which you have, so you're fine. If you ever want to compare NPU vs CPU later, run perf once on the VitisAI-compiled artifact and once on the optimized (pre-compile) artifact on CPU. Don't try to run the compiled artifact on CPU; EP-compiled models are EP-locked.
- Output paths are explicit on every winml command. Even if you don't care about the artifact, perf will write intermediates somewhere — pass an `--output` (or whatever `winml perf --help` calls it) pointed at a scratch dir you can nuke.
- If perf blows up at the optimize or quantize step, run `winml analyze` against the exported ONNX — its linter will name the offending op pattern per EP.

TL;DR: `winml sys`, `winml inspect`, then `winml perf ... --ep VitisAI`. Check `--help` for the exact flag names before you run.
