# CI job: rebuild optimized ResNet-50 for QNN on config change

Good news — this is exactly what the `config` + `build` pair is designed for. One JSON in the repo is the single source of truth, and `winml build` reads it and replays the full pipeline (export → optimize → quantize → compile) deterministically. Below is the shape I'd set up.

## 1. Generate the config once, locally, then commit it

You only run `winml config` once (per intended configuration) — on a dev machine that has the QNN EP registered. The output JSON is what you check into the repo. The CI job does **not** regenerate the config; it consumes it.

Before generating the config, confirm what the CLI actually expects today:

```powershell
winml config --help
winml build --help
winml sys --list-ep   # confirm QNN is registered on the machine you generate from
```

Then produce the config for ResNet-50 targeting QNN. The exact flag spelling for the model and EP comes from `--help`, but conceptually:

```powershell
winml config -m microsoft/resnet-50 --ep QNN -o configs/resnet50-qnn.json
```

Commit `configs/resnet50-qnn.json`. That file is now your reproducible build spec — version-controlled, diff-able in PRs, and the thing CI watches.

A couple of things to verify in the generated JSON before committing:
- The model identifier and EP are what you expect.
- Quantization settings match what you want shipped (QNN typically wants the model quantized — re-run `winml config --help` to see whether there's a flag to bias toward a specific quant config).
- Paths inside the JSON, if any, are relative or parameterized so they aren't tied to the dev machine that generated it.

## 2. The reproducible build step

In CI, the build is two commands plus the install:

```powershell
# 1. Set up Python 3.10 + winml-cli (same as a fresh dev box)
uv venv --python 3.10
.venv\Scripts\activate
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"
# (swap to `uv pip install winml-cli` once it's on PyPI)

# 2. Sanity-check the runner has QNN registered
winml sys --list-ep

# 3. Build from the committed config
winml build -c configs/resnet50-qnn.json -o artifacts/resnet50-qnn
```

Key points to know:

- **`-c` provides defaults; CLI flags override.** Per every primitive's `--help`, anything you pass on the command line wins over the JSON. For a clean CI build, pass *only* `-c` and `-o` — no extra flags — so the JSON stays the source of truth. Save one-off overrides for local debugging.
- **`-o` is explicit.** There's no implicit "current dir" convention; whatever you give `-o` is where the artifact lands. Make CI upload that directory as a build artifact.
- **Compile writes external EP-context by default.** For a QNN-compiled output, you'll get a `.onnx` plus a co-located `.bin` (the EP context). If you publish/move the artifact, move them together — or pass the embed flag (see `winml compile --help`, surfaced via the build config) to inline the context into the `.onnx`.

## 3. Trigger the job only when the config changes

Use a path filter on the workflow so unrelated PRs don't pay the rebuild cost. On GitHub Actions that's:

```yaml
on:
  push:
    paths:
      - 'configs/resnet50-qnn.json'
  pull_request:
    paths:
      - 'configs/resnet50-qnn.json'
```

ADO Pipelines and other systems have equivalent path filters. Point the filter at the exact config file (or `configs/**` if you'll have more than one).

## 4. The CI runner has to have QNN registered

This is the part people forget. `winml build` with `--ep QNN` (whether explicit or via the config) produces a QNN-compiled artifact, and per the docs `winml compile` requires an NPU device. That means your CI runner has to be a Snapdragon X Elite (or another QNN-capable) Windows host with the QNN EP registered. A generic cloud Windows runner won't have it.

Two options:
- **Self-hosted runner on a Snapdragon X Elite box** — the straightforward path. Run `winml sys --list-ep` once during setup and confirm `QNN` shows up.
- **Build artifact-only on x64 and defer compile.** If you can't get a QNN runner, you can still run the earlier stages (export/optimize/quantize) in CI for fast feedback, and do the QNN compile on a separate self-hosted step or release job. This splits the pipeline, which is less clean — prefer a Snapdragon runner if you can.

The CI job should fail loudly if QNN isn't registered. A simple guard:

```powershell
winml sys --list-ep | Select-String -Pattern '\bQNN\b' -Quiet
if (-not $?) { throw "QNN EP not registered on this runner" }
```

## 5. Recommended repo layout

```
configs/
  resnet50-qnn.json          # the committed source of truth
.github/workflows/
  build-resnet50-qnn.yml     # path-filtered to configs/resnet50-qnn.json
artifacts/                   # gitignored; CI uploads these
```

If you later add a second model or a second EP, drop another JSON next to the first one and either add another path-filtered workflow or parameterize one workflow over a matrix of config files.

## 6. Before you do any of this — inspect once

Even though you know it's ResNet-50, run `winml inspect -m microsoft/resnet-50` on the dev box once and skim the output. It confirms the loader/exporter the pipeline will use and the I/O shapes — cheap insurance that the build resolution is what you expect before you bake it into a config the whole team relies on.

## Summary

- **One JSON in the repo:** `configs/resnet50-qnn.json`, generated once via `winml config -m microsoft/resnet-50 --ep QNN -o ...` and committed.
- **Reproducible build step:** install winml-cli, then `winml build -c configs/resnet50-qnn.json -o artifacts/resnet50-qnn`. No extra flags — JSON is the source of truth.
- **CI trigger:** path filter on the config file.
- **Runner:** must be a QNN-capable Windows host (Snapdragon X Elite, self-hosted) with the QNN EP registered. Guard with `winml sys --list-ep` at the top of the job.
- **Confirm exact flag spelling** with `winml config --help` and `winml build --help` before committing the workflow — the CLI is the source of truth and flags evolve.
