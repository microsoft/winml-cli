# eval-install-setup

## Prompt

> i just got my snapdragon x elite dev box and i don't have winml installed yet. python 3.11 is already on the system. give me the actual install commands i should run to get going before i can do anything else.

## with_skill — 5/5

### Response

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

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✓ | Recommends Python 3.10 specifically, and calls out that the user's existing 3.11 will NOT work (winml-cli wheel pins >=3.10,<3.11). | PASS | Quote: 'ModelKit pins Python 3.10 exactly (>=3.10,<3.11). Your system Python 3.11 will not resolve the winml-cli wheel.' |
| ✓ | Uses `uv venv --python 3.10` (or an equivalent isolated venv command) rather than installing into system Python or a 3.11 environment. | PASS | Step 2 `uv venv --python 3.10`. |
| ✓ | Points to the AITK cache as the wheel source today (e.g., `$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-*.whl`) and acknowledges PyPI is the planned/future path. D… | PASS | Step 3 uses the exact AITK cache path and notes 'When winml-cli is published to PyPI -- planned -- this step becomes a plain uv pip install winml-cli.' |
| ✓ | Includes a post-install verification step (e.g., `winml --help` or `winml sys --list-ep`) so the user can confirm the install worked before moving on. | PASS | Step 4 'Verify the install' runs both `winml --help` and `winml sys --list-device --list-ep`. |
| ✓ | All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | All ? winml commands verified runnable against --help. |

---

## without_skill (baseline) — 1/5 ⚠

### Response

# Getting Started on Snapdragon X Elite — Install Commands

Congrats on the new dev box! Snapdragon X Elite is an ARM64 device, so a few of the install steps differ from typical x64 Windows setups. Here's a clean order of operations to run before doing anything else. Run all of these in **PowerShell** (the native ARM64 build, not x86 emulation).

> Tip: Open PowerShell and confirm you're on ARM64 first:
> ```powershell
> $env:PROCESSOR_ARCHITECTURE   # should print "ARM64"
> ```

---

## 1. Confirm Python 3.11 is the ARM64 build

You said Python 3.11 is already installed, but on Snapdragon you want the **native ARM64** build, not the x64 emulated one — that's critical for ONNX Runtime / QNN performance.

```powershell
python --version
python -c "import platform; print(platform.machine())"
```

The second command should print `ARM64`. If it prints `AMD64`, you're running the x64 build under emulation — uninstall it and grab the **Windows ARM64** installer from https://www.python.org/downloads/windows/ (look for "Windows installer (ARM64)").

---

## 2. Install Git (if not already there)

```powershell
winget install --id Git.Git -e --source winget
```

Close and reopen PowerShell so `git` is on PATH.

---

## 3. Install Visual Studio Build Tools (ARM64)

Some Python packages (and the QNN tooling) need MSVC + the Windows SDK. Install the Build Tools with ARM64 components:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e
```

Then run the Visual Studio Installer, click **Modify** on Build Tools, and add:
- **Desktop development with C++** workload
- Individual components: **MSVC v143 - VS 2022 C++ ARM64 build tools**, **Windows 11 SDK (latest)**, **C++ CMake tools for Windows**

---

## 4. Install `uv` (recommended Python package manager)

`uv` is fast and handles virtualenvs cleanly on ARM64:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Reopen PowerShell so `uv` is on PATH. Verify:
```powershell
uv --version
```

(If you'd rather stick with stock `pip`, that's fine — just substitute `python -m pip install ...` for the `uv pip install ...` commands below.)

---

## 5. Create a project virtualenv

```powershell
mkdir C:\dev\winml-work
cd C:\dev\winml-work
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
```

You should now see `(.venv)` in your prompt.

---

## 6. Install ONNX Runtime with the QNN execution provider

This is the package that actually lets you target the **Hexagon NPU** on Snapdragon X Elite:

```powershell
uv pip install --upgrade pip
uv pip install onnxruntime-qnn
```

`onnxruntime-qnn` is the Qualcomm-flavored build of ORT and ships the QNN HTP backend DLLs. Don't install vanilla `onnxruntime` and `onnxruntime-qnn` side-by-side — pick one. For NPU work on this device, you want `onnxruntime-qnn`.

Verify the QNN EP is visible:
```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

You should see `QNNExecutionProvider` in the list.

---

## 7. Install the rest of the model-prep toolchain

```powershell
uv pip install onnx onnxruntime-tools numpy
uv pip install optimum[onnxruntime] transformers
uv pip install huggingface_hub
```

This gets you:
- `onnx` — model IR manipulation
- `optimum` + `transformers` — exporting Hugging Face models to ONNX
- `huggingface_hub` — pulling models down

---

## 8. (Optional but useful) Install the WinML / DirectML stack

If you also want to fall back to GPU via DirectML on the Adreno GPU:

```powershell
uv pip install onnxruntime-directml
```

Again — only install this in a **separate** venv from `onnxruntime-qnn` to avoid DLL conflicts. Keep one venv per EP.

---

## 9. Sanity check: pull a small ONNX model and run on NPU

```powershell
python -c @"
import onnxruntime as ort
print('Providers:', ort.get_available_providers())
print('QNN available:', 'QNNExecutionProvider' in ort.get_available_providers())
"@
```

If that prints `QNN available: True`, you're set to start loading models onto the Hexagon NPU.

---

## TL;DR — minimum commands to copy-paste

```powershell
# (Make sure your Python 3.11 is the ARM64 build first!)
winget install --id Git.Git -e --source winget
winget install --id Microsoft.VisualStudio.2022.BuildTools -e
# ...then add "Desktop development with C++" + ARM64 MSVC via the VS Installer UI

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

mkdir C:\dev\winml-work; cd C:\dev\winml-work
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1

uv pip install --upgrade pip
uv pip install onnxruntime-qnn onnx numpy
uv pip install optimum[onnxruntime] transformers huggingface_hub

python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Once `QNNExecutionProvider` shows up in that final print, your environment is ready and you can move on to exporting / running models on the NPU.

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✗ | Recommends Python 3.10 specifically, and calls out that the user's existing 3.11 will NOT work (winml-cli wheel pins >=3.10,<3.11). | **FAIL** | Baseline endorses Python 3.11 / uses py -3.11 -m venv. Doesn't know about the 3.10 pin. |
| ✗ | Uses `uv venv --python 3.10` (or an equivalent isolated venv command) rather than installing into system Python or a 3.11 environment. | **FAIL** | Baseline uses py -3.11 -m venv; uv only mentioned at the end as optional tooling. |
| ✗ | Points to the AITK cache as the wheel source today (e.g., `$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-*.whl`) and acknowledges PyPI is the planned/future path. D… | **FAIL** | Baseline interprets 'winml' as Windows ML runtime + onnxruntime-qnn (pip). No mention of AITK cache or winml-cli. |
| ✗ | Includes a post-install verification step (e.g., `winml --help` or `winml sys --list-ep`) so the user can confirm the install worked before moving on. | **FAIL** | Baseline's smoke test verifies onnxruntime-qnn, not the winml CLI. |
| ✓ | All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | Response uses no winml commands (trivially passes). |

---
