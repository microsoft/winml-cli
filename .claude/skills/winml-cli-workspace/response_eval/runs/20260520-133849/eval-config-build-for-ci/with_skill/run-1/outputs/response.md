# CI: rebuild an optimized ResNet-50 for QNN on config change

The setup you want maps cleanly onto ModelKit's **config + build** path:

- One JSON file in the repo is the single source of truth.
- `winml build -c <that-file>` is the reproducible build step.
- A CI trigger keyed to changes in that file gates the rebuild.

The pipeline ModelKit follows is `inspect → export → analyze → optimize → quantize → compile → perf`. `winml config` produces a `WinMLBuildConfig` JSON that pins every stage's settings; `winml build` replays it deterministically. CLI flags override the config at build time, so we keep the config authoritative and pass only what CI needs (output dir, target model) on the command line.

## Prerequisite: install the CLI on the CI runner

The CI runner (Windows, with a QNN-capable NPU if you want compile + analyze to succeed end-to-end) needs `winml-cli` installed. ModelKit pins **Python 3.10 exactly**.

```powershell
uv venv --python 3.10
.venv\Scripts\activate
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"
winml --help
winml sys --list-ep   # confirm QNNExecutionProvider is registered
```

When `winml-cli` lands on PyPI, the second-to-last line becomes `uv pip install winml-cli` and you can drop the AITK cache dependency.

If your CI runner doesn't have a QNN NPU, you can still produce the optimized + quantized ONNX (export → optimize → quantize) by skipping compile, and run the QNN compile step on a self-hosted Snapdragon X Elite agent. Compile is the only stage that strictly needs the target NPU.

## Step 1 — generate the config once, commit it

Do this locally (or on a one-time CI bootstrap job) and check the result into the repo. Don't regenerate it in CI on every run — that defeats the "config change triggers rebuild" model and lets auto-detection drift silently.

First confirm the model is actually in scope and the pipeline knows how to handle it:

```powershell
winml inspect -m microsoft/resnet-50
```

ResNet is a classic CNN — squarely in scope. Then generate the config:

```powershell
winml config `
  -m microsoft/resnet-50 `
  --device npu `
  --ep qnn `
  --precision int8 `
  --compile `
  -o configs/resnet50-qnn.json
```

Notes on the flags (all verified against `winml config --help`):

- `--device npu --ep qnn` pins Qualcomm NPU as the target. `--ep` overrides device-to-provider mapping and locks the artifact to QNN.
- `--precision int8` is the standard NPU quantization for ResNet-50; auto would pick this on NPU anyway, but pinning it makes the config explicit.
- `--compile` flips `config.compile` on (the default in `winml config` is `--no-compile`, which leaves it out). You want compile baked into CI so each commit produces a ready-to-deploy QNN artifact.
- `-o configs/resnet50-qnn.json` is the file you'll check into the repo.

Inspect the JSON, then commit it:

```powershell
git add configs/resnet50-qnn.json
git commit -m "Add QNN build config for resnet50"
```

This file is the **single source of truth** for the build. Want fp16? Different input shape? Different optim level? Edit the JSON, commit, push — CI replays the build deterministically.

## Step 2 — the reproducible build step

One command. Pass the config; let it drive every stage.

```powershell
winml build `
  -c configs/resnet50-qnn.json `
  -m microsoft/resnet-50 `
  -o artifacts/resnet50-qnn `
  --rebuild
```

- `-c` is the config you committed. It already carries device, precision, and compile settings — you do **not** repeat them on the command line.
- `-m microsoft/resnet-50` tells build where to pull weights from. The config records the *shape* of the build; `-m` supplies the actual model. (You can also point this at a pre-exported `.onnx` and build will skip export.)
- `-o artifacts/resnet50-qnn` is the output dir CI will upload. Don't use `--use-cache` in CI — caches across runners are nondeterministic; you want everything in `-o` so you can archive it.
- `--rebuild` makes the step idempotent on a fresh checkout — if `artifacts/` is left over from a prior build, this overwrites cleanly.

For repeatability, **don't sprinkle one-off overrides on the build command line.** The skill is explicit on this: CLI flags override the config. If you start passing `--no-quant` or `--ep` at build time in CI, the config in the repo no longer describes the actual build. Edit the JSON instead.

## Step 3 — GitHub Actions workflow

Trigger only when the config (or the workflow itself) changes:

```yaml
# .github/workflows/build-resnet50-qnn.yml
name: build-resnet50-qnn

on:
  push:
    paths:
      - "configs/resnet50-qnn.json"
      - ".github/workflows/build-resnet50-qnn.yml"
  pull_request:
    paths:
      - "configs/resnet50-qnn.json"
  workflow_dispatch:

jobs:
  build:
    # Self-hosted Snapdragon X Elite runner so QNN compile + analyze actually work.
    # If you only want export+optimize+quantize, windows-latest works and you'd add --no-compile.
    runs-on: [self-hosted, windows, arm64, qnn]

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        run: irm https://astral.sh/uv/install.ps1 | iex
        shell: pwsh

      - name: Create Python 3.10 venv
        run: uv venv --python 3.10
        shell: pwsh

      - name: Install winml-cli
        run: |
          .venv\Scripts\activate
          uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"
        shell: pwsh

      - name: Verify EP registration
        run: |
          .venv\Scripts\activate
          winml sys --list-ep
        shell: pwsh

      - name: Build optimized ResNet-50 for QNN
        run: |
          .venv\Scripts\activate
          winml build `
            -c configs/resnet50-qnn.json `
            -m microsoft/resnet-50 `
            -o artifacts/resnet50-qnn `
            --rebuild
        shell: pwsh

      - name: Sanity-check latency on NPU
        run: |
          .venv\Scripts\activate
          winml perf -m artifacts/resnet50-qnn --ep qnn
        shell: pwsh

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: resnet50-qnn
          path: artifacts/resnet50-qnn/**
```

Two things worth calling out:

- **`paths:` filter is what makes the job "rebuild on config change."** Pushes that don't touch `configs/resnet50-qnn.json` won't trigger it.
- **Compiled artifacts ship with a `.bin` sidecar by default.** `winml compile` writes EP context to an external `.bin` next to the `.onnx` unless you pass `--embed`. The `path: artifacts/resnet50-qnn/**` glob in upload-artifact handles this; just don't move the `.onnx` without its `.bin`.

## Step 4 — confirm it actually rebuilds

Make a no-op edit to `configs/resnet50-qnn.json` (e.g. flip a non-functional field, or bump a comment-style metadata key if your config has one) and push. The workflow should fire. Make an unrelated commit elsewhere in the repo — the workflow should *not* fire. That's the contract you wanted.

## A few sharp edges worth knowing

- **CLI flags beat the config.** Treat any flag passed at build time as a one-off override; for anything you want CI to reproduce, edit the JSON.
- **`winml sys --list-ep` is the truth about EPs on the runner.** If `QNNExecutionProvider` isn't in that list, `--ep qnn` won't produce a usable compiled artifact regardless of what the config says. Run it on a self-hosted ARM64 Windows runner with Qualcomm NPU + QNN runtime installed.
- **Don't regenerate the config in CI.** `winml config` auto-detects, and auto-detection can drift across CLI versions or HF Hub model updates. Generating it once and committing the JSON is the whole point of having one source of truth.
- **If you need to debug a stage**, switch from `winml build` to the matching primitive (`winml export`, `winml optimize`, `winml quantize`, `winml compile`) locally — they all accept `-c` and let you re-enter the pipeline at any stage with the same config.
