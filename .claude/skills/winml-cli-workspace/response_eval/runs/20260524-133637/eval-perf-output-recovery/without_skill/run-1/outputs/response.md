Good question — and the honest answer is: `winml perf` is primarily a **benchmarking** command, not a model-export command, so where (or whether) it leaves a redistributable ONNX on disk depends on how it ran under the hood. A few things to check:

## 1. Check the command's own output / logs

Re-run with verbose logging and look for any line that mentions a path being written. Something like:

```powershell
winml perf -m microsoft/resnet-50 --device npu --verbose
```

Benchmark tools typically print lines like `Saved optimized model to …` or `Cached compiled model at …` when they do persist artifacts. If you don't see such a line, the tool likely did not emit a shippable ONNX at all.

## 2. Look in the usual cache locations

When a `winml`-style CLI downloads a Hugging Face model and runs it on NPU, the artifacts usually land in one of these places on Windows:

- **Hugging Face cache** (the *source* ONNX, unoptimized):
  ```
  %USERPROFILE%\.cache\huggingface\hub\models--microsoft--resnet-50\
  ```
- **Tool-specific cache** for downloaded/converted models, commonly one of:
  ```
  %LOCALAPPDATA%\winml\
  %LOCALAPPDATA%\Microsoft\winml\
  %USERPROFILE%\.winml\
  %USERPROFILE%\.cache\winml\
  ```
- **EP compiled-model cache** (QNN context binary for Snapdragon NPU). On Qualcomm QNN this is often a `.bin` / `.onnx` context-cache file next to the model or in a temp dir. Search for `*qnn*`, `*context*`, or recent `.onnx` files:
  ```powershell
  Get-ChildItem -Path $env:LOCALAPPDATA, $env:USERPROFILE\.cache -Recurse -Include *.onnx,*.bin -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -gt (Get-Date).AddHours(-2) }
  ```

That last command will surface any ONNX or binary written in the last two hours, which is the fastest way to find what `winml perf` actually dropped.

## 3. The important caveat for shipping

Even if you find a file in a cache, **the artifact `winml perf` uses is not necessarily what you want to ship**:

- It may be the **raw downloaded ONNX** (not optimized/quantized — the NPU speedup came from the EP compiling at load time, not from a persistent optimized graph).
- It may be a **QNN context binary** that is **tied to the specific Snapdragon SoC + driver version** on your machine. Those are not portable across devices and sometimes not even across driver updates.
- It may be a temp file that gets cleaned up.

For shipping with your app you almost certainly want an **explicit export** step, not a scavenged cache file. Most `winml`-style CLIs separate these concerns into different verbs, for example:

```powershell
# explicitly produce an optimized/quantized ONNX you own
winml optimize  -m microsoft/resnet-50 --device npu -o .\dist\resnet50.onnx
winml quantize  -m .\dist\resnet50.onnx           -o .\dist\resnet50.int8.onnx
winml compile   -m .\dist\resnet50.int8.onnx --ep QNN -o .\dist\resnet50.qnn.onnx
```

Run `winml --help` and `winml perf --help` to confirm the exact verb names and whether `perf` has a flag like `--save-model`, `--output`, or `--keep-artifacts` that would have written a copy for you. If it does, re-run with that flag set to a known output directory. That's the supported path to a redistributable artifact — much safer than digging a file out of a cache and hoping it's portable.

### TL;DR

- `winml perf` benchmarks; it doesn't owe you a shippable model on disk.
- Check `winml perf --help` for an output/save flag, and re-run with `--verbose` to see what (if anything) it wrote.
- Look in `%LOCALAPPDATA%\winml\`, `%USERPROFILE%\.cache\winml\`, and the HF cache for recently-written `.onnx` / context-binary files.
- To actually ship: use the dedicated `optimize` / `quantize` / `compile` verbs (check `winml --help`) so you get a deterministic, versioned artifact you control — not a cached side-effect of a perf run.
