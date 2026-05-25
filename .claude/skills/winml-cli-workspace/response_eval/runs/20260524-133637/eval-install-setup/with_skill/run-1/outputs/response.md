Heads up before you run anything: ModelKit pins **Python 3.10 exactly** (`>=3.10,<3.11`). Your system 3.11 won't resolve the `winml-cli` wheel. The good news is you don't need to touch system Python — `uv` will create an isolated 3.10 venv for you, and (if 3.10 isn't on the machine yet) `uv` will fetch it automatically.

## 1. Install `uv` (skip if you already have it)

From PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell after install so `uv` is on PATH, then sanity-check:

```powershell
uv --version
```

## 2. Create an isolated Python 3.10 venv

In whatever folder you want to work out of:

```powershell
uv venv --python 3.10
```

If 3.10 isn't installed, `uv` will download it for you. Then activate it:

```powershell
.venv\Scripts\activate
```

You should see `(.venv)` in your prompt. Your system 3.11 is untouched.

## 3. Install the `winml-cli` wheel

Today the wheel ships locally with AI Toolkit (AITK), not from PyPI. Install it from the AITK cache:

```powershell
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"
```

If that path doesn't exist on your box, it means AITK hasn't dropped the wheel yet — install AI Toolkit for VS Code first (it ships the runtime cache), then re-run the line above. When `winml-cli` lands on PyPI (planned), the equivalent will just be `uv pip install winml-cli`.

## 4. Verify the install and confirm your NPU is visible

```powershell
winml --help
winml sys --list-device --list-ep
```

- `--help` should print the top-level command list.
- `sys --list-ep` is the one that matters on a fresh Snapdragon X Elite box — you want to see **QNN** in the registered EPs. That's the Qualcomm NPU execution provider, and it's what every `--ep qnn` / `--device npu` command downstream will target. If QNN isn't listed, the NPU compile path won't produce a usable artifact regardless of what flags you pass, so don't move on until it shows up.

## After this you're ready to go

From a new shell, remember to re-activate the venv (`.venv\Scripts\activate`) before calling `winml`. Once the verify step is clean, the natural first command on any model is `winml inspect -m <hf-id-or-path>` — it reads the model config without downloading weights and tells you whether the toolkit knows how to handle it. From there it's the standard pipeline (`inspect → export → analyze → optimize → quantize → compile → perf`), or just `winml config` + `winml build` if you want the shortcut path.

One scope note since you're on a Snapdragon dev box and the obvious thing to try is a chat model: ModelKit today targets **classic DL** (ResNet, ViT, BERT, SegFormer, detection/segmentation, etc.). **LLMs and other generative models — GPT, LLaMA, Phi, Mistral, Stable Diffusion, Whisper, anything decoder-only or seq2seq — are out of scope** and will fail mid-pipeline with errors that don't always point at the real cause. LLM support is on the public roadmap for late 2026. Save yourself the debugging loop and start with a classic model (e.g. `microsoft/resnet-50`) for your first run.
