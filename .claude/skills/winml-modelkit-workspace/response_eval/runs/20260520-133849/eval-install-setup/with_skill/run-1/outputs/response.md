# Installing `winml` on your Snapdragon X Elite dev box

Good news: your machine has the right NPU (Qualcomm, QNN execution provider is 🟢 Ready) for the WinML ModelKit pipeline. One important note before you copy/paste anything below — **ModelKit pins Python 3.10 exactly** (`>=3.10,<3.11`). Your system Python 3.11 will not resolve the `winml-cli` wheel. We'll use `uv` to create an isolated 3.10 venv so we don't touch system Python and don't run into that wall.

Run these in order from a PowerShell prompt in whatever folder you want to work out of (e.g. `C:\dev\winml-playground`).

## 1. Install `uv` (if you don't already have it)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then restart your shell so `uv` is on PATH. Verify:

```powershell
uv --version
```

## 2. Create an isolated Python 3.10 venv

`uv` will download a 3.10 toolchain for you if one isn't installed — you do not need to manually install Python 3.10 yourself.

```powershell
uv venv --python 3.10
```

Activate it:

```powershell
.venv\Scripts\activate
```

Your prompt should now be prefixed with `(.venv)`. Confirm you're on 3.10:

```powershell
python --version
```

You should see `Python 3.10.x`. If you see 3.11, the venv isn't activated — re-run the activate command above.

## 3. Install the `winml-cli` wheel

Today the wheel ships locally with the AI Toolkit (AITK) extension, not from PyPI. Install it from the AITK cache:

```powershell
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"
```

If that path doesn't exist on your machine yet, it means AITK hasn't dropped the runtime cache. Install the **AI Toolkit for Visual Studio Code** extension in VS Code and open it once; it will populate `~\.aitk\bin\model_lab_runtime\cache\` with the wheel. Then re-run the `uv pip install` line.

(When `winml-cli` is published to PyPI — planned — this step becomes a plain `uv pip install winml-cli`. Not there yet.)

## 4. Verify the install

```powershell
winml --help
winml sys --list-device --list-ep
```

- `winml --help` should print the top-level command list (`inspect`, `export`, `analyze`, `optimize`, `quantize`, `compile`, `perf`, plus `config` and `build`).
- `winml sys --list-device --list-ep` should show your Qualcomm NPU as a device and list `QNN` among the registered execution providers. **If `QNN` is not in the EP list, stop here and fix that before doing anything else** — without it, you can't compile or perf to the NPU. (CPU EP is always present as a fallback.)

## You're ready

Once `sys --list-ep` confirms QNN is registered, the typical next step on a Snapdragon X Elite is:

1. Pick a model — a Hugging Face ID like `microsoft/resnet-50` or a local `.onnx`.
2. `winml inspect -m <model>` first, every time. It reads the config without downloading weights and tells you whether ModelKit knows how to handle the architecture. Note the CLI is flag-based: the model goes through `-m`, not as a bare positional argument.
3. From there, default to **`winml config` + `winml build`** for a clean end-to-end pipeline, or step through `export → analyze → optimize → quantize → compile → perf` if you want to debug one stage at a time.

One scope warning before you pick a model: ModelKit currently targets **classic deep learning models** — CNNs, vision transformers, encoder NLP models, detection/segmentation (ResNet, ViT, Swin, ConvNeXT, BERT, RoBERTa, SegFormer, etc.). **Generative / decoder-only models are out of scope today** — GPT, LLaMA, Phi, Mistral, Stable Diffusion, Whisper, any seq2seq generator. LLM support is on the roadmap for late 2026 but doesn't work yet, so don't burn time pointing the pipeline at one. If you're not sure where a model lands, `winml inspect` is the source of truth.

Run the four steps above, paste back the output of `winml sys --list-device --list-ep`, and we can pick a first model to push through the pipeline.
