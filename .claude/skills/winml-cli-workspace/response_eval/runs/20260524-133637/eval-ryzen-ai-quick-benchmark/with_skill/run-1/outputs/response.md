# Fastest path to a latency number on Ryzen AI NPU

Since you only want the number and don't care about keeping the artifact, you want one command: `winml perf`. It builds whatever it needs on the fly (export → optimize → quantize → compile) and writes a metrics JSON. No need to walk the pipeline manually.

For AMD Ryzen AI, the NPU execution provider is **VitisAI** (Ready as of 2026-05-20).

## Prereqs (skip if `winml --help` already works)

ModelKit pins Python 3.10 exactly. Use `uv` to keep it isolated.

```powershell
uv venv --python 3.10
.venv\Scripts\activate
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"
winml --help
```

## Step 1 — Confirm the EP is actually registered

Don't trust a table, trust the machine:

```bash
winml sys --list-device --list-ep
```

You should see `VitisAI` in the EP list. If it isn't there, the NPU driver / VitisAI runtime isn't installed and `perf` won't be able to target the NPU — fix that first.

## Step 2 — Inspect (golden rule, cheap, no download)

```bash
winml inspect -m facebook/convnext-tiny-224
```

ConvNeXT is in the in-scope CNN family, so this should come back clean. If it doesn't, stop here — no point benchmarking something the pipeline can't build.

## Step 3 — Benchmark on the NPU

One shot, no artifact kept (you don't care about the build, only the number):

```bash
winml perf -m facebook/convnext-tiny-224 -o perf-convnext-vitisai.json
```

Before you run it, check the real flag spellings on your installed version:

```bash
winml perf --help
```

Specifically look for:

- the EP/device selection flag — pass the VitisAI EP and the NPU device value it lists
- `--monitor` if you want to watch NPU utilization live during the run
- `--rebuild` / `--ignore-cache` if a previous cached build is making the numbers look weird

The metrics JSON at `-o` is the published output — latency percentiles, throughput, and run config. That's your number.

## A couple of gotchas worth flagging

- **Don't invent flags.** If something like `--mode=fast` or `--preset npu` isn't in `winml perf --help`, it doesn't exist — the command will reject it.
- **CLI is flag-based, not positional.** The model goes through `-m`, never as a bare argument.
- **The artifact `perf` builds internally lives in an opaque cache.** You said you don't want it, which is good — if you ever change your mind and want to keep the compiled `.onnx`, the right shape is `winml config` → `winml build -o <dir>` → `winml perf` against the built artifact. Don't try to fish the build out of cache.

That's it — one `perf` call gets you the number.
