Great target — Snapdragon X Elite has a Qualcomm Hexagon NPU, which `winml` drives via the **QNN** execution provider (🟢 Ready). Here's how to get ResNet-50 running on it end-to-end.

## Quick clarifier

"Run ResNet-50" can mean two different things, and the right command differs:

- **You want a deployable `.onnx` artifact** you can ship or load from your app → use `winml config` + `winml build`.
- **You just want a latency / throughput number on the NPU** → use `winml perf` directly against the source model.
- **You want both** → `winml build` first to produce the artifact, then `winml perf` against the built artifact.

I'll cover the most common case (a deployable artifact, then benchmark it) below. If you only want the number, skip to the "Just benchmark" section at the bottom.

## Prereq: install the `winml` CLI

ModelKit pins **Python 3.10 exactly**, so use `uv` to keep this isolated.

```powershell
uv venv --python 3.10
.venv\Scripts\activate
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"
winml --help
winml sys --list-ep
```

The wheel ships locally with AI Toolkit (AITK) today; it'll be on PyPI later. In `winml sys --list-ep` output, confirm **QNN** is listed — that's your NPU EP. If it isn't, the NPU stack isn't registered on this machine and compile won't produce a usable artifact.

## Step 1 — Inspect first (always)

Before anything else, check that ModelKit knows how to handle ResNet-50. This is cheap (no weights downloaded) and tells you the loader, exporter, I/O specs, and the pipeline resolution.

```bash
winml inspect -m microsoft/resnet-50
```

Note the flag is `-m` / `--model` — the CLI is flag-based, not positional. ResNet is squarely in scope (classic CNN), so this should pass cleanly.

## Step 2 — Config + build (recommended path)

For a reproducible end-to-end build targeting QNN on the Snapdragon NPU:

```bash
winml config -m microsoft/resnet-50 -o resnet50-qnn.json
winml build -c resnet50-qnn.json -o ./resnet50-qnn
```

`config` auto-detects every setting the pipeline needs and writes a JSON file. `build` reads it and runs `export → optimize → quantize → compile` in order. You can open `resnet50-qnn.json` and edit it (EP, device, quantization mode, etc.) before running `build` if you want to override defaults — that JSON is the single source of truth for the build.

Two things to check with `--help` before you run, since flag spellings evolve:

```bash
winml config --help
winml build --help
```

Specifically look for how to pin the EP to **QNN** and the device to the NPU (commonly something like `--ep qnn` and a device selector). Don't guess the flag spelling — read it from `--help`. If `config` auto-picks QNN because it sees QNN registered in `winml sys`, you may not need to override anything.

## Step 3 — Benchmark the artifact on the NPU

Once `build` has produced the compiled artifact in `./resnet50-qnn`, measure latency:

```bash
winml perf -m ./resnet50-qnn/<compiled-artifact>.onnx -o resnet50-qnn-perf.json
```

`perf` writes a metrics JSON. Check `winml perf --help` for a `--monitor` flag if you want live NPU utilization during the run.

A few QNN-specific gotchas:

- **EP-compiled models are tied to their EP.** Don't try to `perf` the QNN-compiled artifact against CPU — the result is meaningless. For an NPU-vs-CPU comparison, run CPU `perf` against the pre-compile *optimized* ONNX (also in the build output directory), not the QNN-compiled one.
- **External EP-context `.bin`.** `winml compile` (which `build` runs under the hood) defaults to writing EP context to a `.bin` next to the `.onnx`. If you move the artifact, move the `.bin` with it — or pass `--embed` to inline it. Confirm via `winml compile --help`.

## Just want the latency number?

Skip config/build entirely:

```bash
winml perf -m microsoft/resnet-50 -o resnet50-qnn-perf.json
```

`perf` builds artifacts on the fly internally. Read `winml perf --help` for the flag that pins the EP to QNN, plus `--rebuild`, `--ignore-cache`, `--no-quantize`, and `--monitor` if you need them. Note: the artifact `perf` builds internally lives in opaque cache — you can't ship it. If you later decide you want a deployable model, you'll have to run `build` separately (it won't cost double if cache is warm, but the artifact only becomes "yours" via `build -o`).

## Troubleshooting

- **`winml sys --list-ep` doesn't show QNN** → the Qualcomm NPU runtime isn't registered; compile won't work. Fall back to CPU EP for now while you sort out the driver/runtime install.
- **Optimize or quantize blows up mid-build** → run `winml analyze` against the exported ONNX; the linter will name the offending op pattern per EP. Try a different optim/quant config rather than hand-editing the graph.
- **Any flag I quoted above doesn't exist** → run `winml <command> --help` and use the real one. Flag spellings drift between versions; the CLI is the source of truth.
