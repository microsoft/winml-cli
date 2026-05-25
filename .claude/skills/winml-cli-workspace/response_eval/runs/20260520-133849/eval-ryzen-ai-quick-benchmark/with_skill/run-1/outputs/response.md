# Fastest path to an NPU latency number for `facebook/convnext-tiny-224` on Ryzen AI

Good news: ConvNeXT is squarely in scope (classic CNN/vision transformer family), and AMD Ryzen AI maps to the **VitisAI** execution provider, which is currently ready. Since you just want the number and don't care about keeping the artifact, you don't need to chain `export → optimize → quantize → compile` yourself — `winml perf` will build everything on the fly and throw it away.

## Prereq: install the CLI (skip if you already have `winml` working)

ModelKit pins **Python 3.10 exactly**. Use `uv` so you don't accidentally land on 3.11+.

```powershell
# 1. Create + activate a 3.10 venv
uv venv --python 3.10
.venv\Scripts\activate

# 2. Install the winml-cli wheel from the AI Toolkit cache
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"

# 3. Verify
winml --help
winml sys --list-ep
```

In the `sys --list-ep` output, confirm you see **VitisAI** registered. If it's not there, the AMD NPU driver / Ryzen AI stack isn't wired up yet and the benchmark will fall back off-NPU.

## Step 1 (golden rule): inspect first

This is cheap — it reads the config without downloading weights — and it confirms ModelKit knows how to handle ConvNeXT before you commit to a full build.

```powershell
winml inspect -m facebook/convnext-tiny-224
```

You're looking for a clean loader/exporter/inference-class resolution. If inspect is happy, move on.

## Step 2: one command for the latency number

Since you only want the number, skip the primitives and just run `perf`. It will export, optimize, quantize, and compile to VitisAI in a scratch location, run the benchmark, and report.

```powershell
winml perf -m facebook/convnext-tiny-224 --device npu --ep vitisai --ignore-cache
```

What the flags do:

- `--device npu` — target the NPU.
- `--ep vitisai` — pin the execution provider to AMD's VitisAI explicitly, so device auto-selection can't quietly fall back to CPU/DML if something is off.
- `--ignore-cache` — build everything in a temp folder and discard after the run. You said you don't care about the artifact; this keeps your disk clean.
- Defaults you're getting for free: `--iterations 100`, `--warmup 10` (warmup excluded from stats), `--batch-size 1`, `--precision auto` (perf will quantize as part of the build — that's what you want on NPU).

If you also want to see the NPU actually light up while it runs, add `--monitor` for a live utilization chart:

```powershell
winml perf -m facebook/convnext-tiny-224 --device npu --ep vitisai --ignore-cache --monitor
```

## What you'll see

`winml perf` prints latency and throughput stats to the console and also writes a JSON summary (default: `facebook_convnext-tiny-224_perf.json` in the current directory — the artifact gets thrown away with `--ignore-cache`, but the perf summary file stays). Look for the mean / p50 / p90 latency in ms and the throughput in inferences/sec.

## If something goes sideways

1. `winml sys --list-ep` — is VitisAI actually registered? If not, fix the Ryzen AI driver stack before chasing anything else.
2. `winml perf --help` — confirm flag spelling didn't drift since this writeup.
3. `winml analyze -m facebook/convnext-tiny-224 --ep vitisai` — if the build blows up at optimize/quantize, the analyzer's linter will name the offending op pattern. ConvNeXT-tiny is well within the supported envelope, so this would be surprising — but it's the right next step if you hit a wall.

That's it — one command (`winml perf … --device npu --ep vitisai --ignore-cache`) is the fastest way to a number.
